"""Teacher targets: FlashVSR v1.1 (slow streaming diffusion, ported to MPS in bench/) upscales each 720p training
sequence, and the result is saved as the texture target a small fast network learns to imitate.

    python -m detailnet.teacher --data output/detail-data [--limit N] [--force]

For every teacher sequence that `detailnet.build_data` has finished, the margin-expanded source dump (384x244 source px,
burn_in + frames frames) goes through FlashVSR as one shot (x4, 976x1536). Each frame is then area-resized by 0.75 (the
x3 grid the base and the network use) and the margin cropped, leaving the central crop at 960x540:

    <data>/teacher/<id:05d>/teacher/   kind "teacher", 3 x 540 x 960, one frame per source frame, pts copied

Only `teacher/` is ever written; `source/`, `base/` and `flow/` belong to build_data. `teacher/meta.json` is written
after the last frame and a failed or interrupted sequence deletes its partial `teacher/`, so a `meta.json` with the
full frame count means the sequence is done. That is what a rerun skips.

Order is round-robin over the motion buckets (static, slow, fast, local, straddle, dissolve), lowest id first within a
bucket, so a run stopped halfway still covers every bucket evenly. A sequence whose build is not finished yet is left
for later (rerun, or it is picked up when the run finishes the others and looks again).

Guards: refuses to start while the user's batch runs (`--force` overrides); one instance at a time (`<data>/teacher.lock`
holds the pid; a lock of a dead process is taken over); stops with exit 3 when swap grew by more than 8 GB (resumable);
five failures in a row stop the run with exit 1. Progress goes to stdout, one line per sequence, ending in a SUMMARY.
"""
import argparse
import json
import os
import shutil
import signal
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from detailnet.build_data import expected_frames, hms, is_complete
from detailnet.dumps import DumpWriter, read_frame, read_meta
from detailnet.guards import SWAP_EXIT_CODE, SwapGuard, SwapLimitExceeded, refuse_if_batch_running
from detailnet.sampling import BUCKETS

ROOT = Path(__file__).resolve().parents[1]
SCALE = 3  # source px -> teacher px (the base's grid)
MARGIN = 32  # source px of context around the central crop in the source dump
FLASHVSR_SCALE = 4
SWAP_LIMIT_GB = 8.0
MAX_CONSECUTIVE_FAILURES = 5
LOCK_NAME = "teacher.lock"


# ---------------------------------------------------------------- pixels

def area_weights(n_in, n_out):
    """(n_out, n_in) matrix of a box-filter resize: row j averages the input interval [j r, (j + 1) r), r = n_in / n_out,
    weighting each input pixel by how much of it the interval covers."""
    ratio = n_in / n_out
    lo = (np.arange(n_out) * ratio)[:, None]
    hi = (np.arange(1, n_out + 1) * ratio)[:, None]
    pixel = np.arange(n_in)[None, :]
    overlap = np.clip(np.minimum(hi, pixel + 1) - np.maximum(lo, pixel), 0.0, None)
    return (overlap / overlap.sum(axis=1, keepdims=True)).astype(np.float32)


def to_teacher_grid(hr4, margin=MARGIN):
    """FlashVSR's x4 output (3, 4H, 4W) -> area resize x0.75 -> (3, 3H, 3W) -> `3 * margin` px cropped off every side.
    For the 384x244 source dump (margin 32) that is (3, 976, 1536) -> (3, 540, 960): the central 320x180 crop at x3.
    Source pixel (x, y) lands on teacher pixels [3 (x - margin), 3 (x - margin) + 3) horizontally, likewise vertically."""
    hr4 = np.asarray(hr4, dtype=np.float32)
    _, h4, w4 = hr4.shape
    if h4 % FLASHVSR_SCALE or w4 % FLASHVSR_SCALE:
        raise ValueError(f"expected a x{FLASHVSR_SCALE} frame, got {w4}x{h4}")
    h3, w3 = h4 // FLASHVSR_SCALE * SCALE, w4 // FLASHVSR_SCALE * SCALE
    crop = SCALE * margin
    rows = area_weights(h4, h3)[crop:h3 - crop]  # only the rows and columns that survive the crop are computed
    cols = area_weights(w4, w3)[crop:w3 - crop]
    return np.clip(rows @ hr4 @ cols.T, 0.0, 1.0)


def to_uint8_hwc(chw):
    """A source dump frame (float16 CHW in [0, 1]) as the 8-bit HxWx3 image FlashVSR takes."""
    out = np.clip(np.round(np.asarray(chw).astype(np.float32) * 255), 0, 255).astype(np.uint8).transpose(1, 2, 0)
    return np.ascontiguousarray(out)


# ---------------------------------------------------------------- one sequence

def is_done(seq_dir, record):
    """True when seq_dir/teacher/meta.json says the sequence's teacher dump is finished: burn_in + frames frames."""
    try:
        meta = read_meta(Path(seq_dir) / "teacher")
        return meta["kind"] == "teacher" and meta["frames"] == expected_frames(record)
    except (OSError, ValueError, KeyError):
        return False


def make_teacher(seq_dir, upscale):
    """Write seq_dir/teacher from seq_dir/source. `upscale(frames)` takes the sequence's frames (uint8 HxWx3 each) and
    returns one (3, 4H, 4W) float array in [0, 1] per frame (tests inject a fake; `load_flashvsr` is the real one).
    A failure of any kind leaves no teacher/ behind; meta.json is written only after the last frame."""
    seq_dir = Path(seq_dir)
    spec = json.loads((seq_dir / "spec.json").read_text())
    source = seq_dir / "source"
    meta = read_meta(source)
    out = seq_dir / "teacher"
    if out.exists():  # left by a killed run
        shutil.rmtree(out)
    try:
        frames = [to_uint8_hwc(read_frame(source, i)) for i in range(meta["frames"])]
        hr = upscale(frames)
        if len(hr) != len(frames):
            raise ValueError(f"the upscaler returned {len(hr)} frames for {len(frames)}")
        width, height = SCALE * (meta["width"] - 2 * MARGIN), SCALE * (meta["height"] - 2 * MARGIN)
        writer = DumpWriter(out, "teacher", 3, width, height, meta["source_size"], SCALE, spec["crop"])
        for i, frame in enumerate(hr):
            writer.write(to_teacher_grid(frame), meta["pts"][i])
        writer.finish()
    except BaseException:
        shutil.rmtree(out, ignore_errors=True)
        raise


# ---------------------------------------------------------------- order and readiness

def round_robin(entries):
    """`entries` (sequences.json rows) with the buckets taken in turn: static, slow, fast, local, straddle, dissolve
    (any other bucket after those), lowest id first within a bucket."""
    names = list(BUCKETS) + sorted({e["bucket"] for e in entries} - set(BUCKETS))
    queues = [sorted((e for e in entries if e["bucket"] == name), key=lambda e: e["id"]) for name in names]
    ordered = []
    for rank in range(max((len(q) for q in queues), default=0)):
        ordered += [q[rank] for q in queues if rank < len(q)]
    return ordered


def seq_dir(data, entry):
    return Path(data) / "teacher" / f"{entry['id']:05d}"


def is_ready(data, entry):
    """True when build_data has finished this sequence: the same spec.json it was planned with, its timing written
    (the cut and timestamp checks passed) and source, base and flow all holding burn_in + frames frames."""
    return is_complete(seq_dir(data, entry), entry)


def load_entries(data):
    """The teacher rows of <data>/sequences.json."""
    path = Path(data) / "sequences.json"
    try:
        listing = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise SystemExit(f"cannot read {path}: {error}. Run detailnet.build_data first.")
    return [e for e in listing if e["kind"] == "teacher"]


# ---------------------------------------------------------------- the run

def run_all(data, upscale, limit, guard, log=print):
    """Make teacher targets for every finished, undone teacher sequence of `data`, round-robin over the buckets.
    `limit` caps the number of sequences made this run. Returns (exit code, counts)."""
    entries = load_entries(data)
    total = len(entries)
    done = sum(is_done(seq_dir(data, e), e) for e in entries)
    counts = {"made": 0, "failed": 0, "already_done": done, "not_built": 0}
    attempted, seconds_made = set(), 0.0
    started = time.monotonic()
    consecutive_failures, number, code = 0, 0, 0
    while code == 0 and (limit is None or counts["made"] < limit):
        queue = [e for e in round_robin(entries) if e["id"] not in attempted
                 and not is_done(seq_dir(data, e), e) and is_ready(data, e)]
        if not queue:
            break  # a pass over everything finished so far; the builder may have finished more by now, hence the loop
        for entry in queue:
            if limit is not None and counts["made"] >= limit:
                break
            try:
                guard.check()
            except SwapLimitExceeded as error:
                log(f"STOPPED by the swap guard before teacher {entry['id']:05d}: {error}. Rerun to resume.")
                code = SWAP_EXIT_CODE
                break
            attempted.add(entry["id"])
            number += 1
            n = expected_frames(entry)
            t0 = time.monotonic()
            note = ""
            try:
                make_teacher(seq_dir(data, entry), upscale)
                spent = time.monotonic() - t0
                counts["made"] += 1
                done += 1
                seconds_made += spent
                consecutive_failures = 0
                timing = f"{spent:.1f}s {spent / n:.2f} s/frame"
            except Exception as error:
                spent = time.monotonic() - t0
                counts["failed"] += 1
                consecutive_failures += 1
                timing = f"FAILED {spent:.1f}s"
                note = f" | {type(error).__name__}: {str(error).splitlines()[0][:200] if str(error) else ''}"
                for line in traceback.format_exc().rstrip().splitlines()[-12:]:
                    log(f"    {line}")
            eta = ""
            if counts["made"]:
                eta = f" eta {hms((total - done) * seconds_made / counts['made'])}"
            log(f"[{number:4d}] teacher {entry['id']:05d} {entry['bucket']:<8} frames={n} {timing} "
                f"| done {done}/{total}{eta}{note}")
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                log(f"STOPPED after {consecutive_failures} failures in a row")
                code = 1
                break
    counts["not_built"] = sum(not is_done(seq_dir(data, e), e) and not is_ready(data, e) for e in entries)
    code = code or (1 if counts["failed"] else 0)
    log(f"SUMMARY {'ok' if code == 0 else 'incomplete'}: made {counts['made']} already-done {counts['already_done']} "
        f"failed {counts['failed']} not-built {counts['not_built']} | done {done}/{total} | "
        f"{hms(time.monotonic() - started)} exit {code}")
    return code, counts


# ---------------------------------------------------------------- one instance at a time

def pid_alive(pid):
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, just not ours
    return True


def acquire_lock(path, alive=pid_alive):
    """Create the lock file holding our pid; SystemExit if a live process holds it. A lock whose pid is dead (or
    unreadable) is stale and replaced. The file appears atomically with its content (hard link of a private temp)."""
    path = Path(path)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(f"{os.getpid()}\n")
    try:
        for _ in range(3):
            try:
                os.link(tmp, path)
                return
            except FileExistsError:
                pass
            try:
                holder = int(path.read_text().strip())
            except (OSError, ValueError):
                holder = None
            if holder is not None and alive(holder):
                raise SystemExit(f"refusing to start: a teacher run is already running (pid {holder}, lock {path}). "
                                 "If that is wrong, delete the lock file.")
            path.unlink(missing_ok=True)  # stale
        raise SystemExit(f"cannot take the lock {path}")
    finally:
        tmp.unlink(missing_ok=True)


def release_lock(path):
    """Remove the lock file if it is ours."""
    path = Path(path)
    try:
        if int(path.read_text().strip()) == os.getpid():
            path.unlink()
    except (OSError, ValueError):
        pass


# ---------------------------------------------------------------- FlashVSR

def load_flashvsr():
    """The real upscaler: FlashVSR v1.1 (loads ~6 GB of weights; call once per process). Needs `source bench/env.sh`."""
    sys.path.insert(0, str(ROOT / "bench"))
    import torch
    from models import FlashVSR

    model = FlashVSR()

    def upscale(frames):
        out = [None] * len(frames)

        def emit(i, hr):
            out[i] = hr.float().cpu().numpy()

        try:
            model.run_shot(frames, emit)
        finally:
            torch.mps.empty_cache()
        return out

    return upscale


def lazy(load):
    """An upscaler that calls `load()` the first time it is used, so a run with nothing to do never loads weights."""
    loaded = []

    def upscale(frames):
        if not loaded:
            loaded.append(load())
        return loaded[0](frames)

    return upscale


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", required=True, help="the build_data output directory (output/detail-data)")
    parser.add_argument("--limit", type=int, default=None, help="make at most N sequences this run")
    parser.add_argument("--force", action="store_true", help="run even while the batch script is running")
    parser.add_argument("--swap-limit-gb", type=float, default=SWAP_LIMIT_GB)
    args = parser.parse_args(argv)

    refuse_if_batch_running(args.force)
    data = Path(args.data)
    load_entries(data)  # fail early on a missing sequences.json, before taking the lock
    lock = data / LOCK_NAME
    acquire_lock(lock)
    try:
        guard = SwapGuard(args.swap_limit_gb)
        print(f"teacher targets for {data} (pid {os.getpid()}, limit {args.limit}, swap limit "
              f"{args.swap_limit_gb:g} GB)", flush=True)
        code, _ = run_all(data, lazy(load_flashvsr), args.limit, guard,
                          log=lambda line: print(line, flush=True))
        return code
    finally:
        release_lock(lock)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))  # unwind: the partial teacher/ and the lock are removed
    sys.exit(main())
