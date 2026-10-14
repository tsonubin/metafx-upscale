"""Training-data builder: one dump directory per sampled sequence.

    python -m detailnet.build_data --out output/detail-data --teacher-seqs 700 --gt-seqs 400 --seed 0 [--limit N]

Writes `sequences.json` (the sampler's list) and, per sequence,

    <out>/<kind>/<id:05d>/{spec.json, source/, base/, flow/}      plus target/ for gt

where kind is "teacher" (720p episodes, 3x to 4K; FlashVSR targets are added later by teacher.py) or "gt"
(1080p episodes: the real 1080p frames are the target, a 640x360 x264 copy is the input). The dumps come from the
Swift CLI (`--engine ml --dump-source/--dump-base/--dump-flow --dump-crop`); gt sequences also go through ffmpeg.
Every dump holds burn_in + frames frames of the margin-expanded crop (384x244 source px for a 320x180 crop); the gt
target holds the central crop x3 (960x540 real 1080p pixels).

Frame indices are those of the shot maps (`frames.csv`), which decode with ffmpeg's default CFR handling: index i is
the frame shown at i / fps on the container's timeline. The episodes' video starts at 0.08 s, so indices 0 and 1 are
padding copies of the first real frame. `--start i / fps` for Swift (AVAssetReader keeps every frame that overlaps the
range) and `-ss i / fps` for ffmpeg (keeps frames at or after the time) both select that same frame; each is nudged by
half a frame so rounding in either tool cannot move the boundary. `detailnet/check_alignment.py` re-verifies a built
sequence against ffmpeg.

The plan is fixed once written: when `sequences.json` exists and the sampler's plan for these arguments differs (another
--teacher-seqs, --gt-seqs or --seed, or new shot maps; the sampler is not prefix-stable, so asking for more sequences
changes earlier ones too), the run is refused before anything is written or deleted, unless `--replan` is passed.
A replan rebuilds every sequence whose spec changed (deleting its dumps) and leaves those holding teacher/ targets as
STALE-KEPT (see below).

Resumable: a sequence is skipped only when its spec.json matches and every dump's meta.json holds burn_in + frames
frames; anything else is deleted and rebuilt, except a directory that already holds teacher/ targets (FlashVSR costs
seconds per frame): that one is kept, logged STALE-KEPT and counted as failed, unless `--rebuild` is passed. Sequences
run one at a time (the ml engine uses the GPU and Media Engine fully). A sequence that fails is removed and logged; five
failures in a row, or swap growing by more than the guard's limit, stop the run (exit 1, exit 3). Progress goes to
stdout, one line per sequence, ending in a SUMMARY line.
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import NamedTuple

import numpy as np

from detailnet.dumps import read_frame, read_meta
from detailnet.guards import SWAP_EXIT_CODE, SwapGuard, SwapLimitExceeded
from detailnet.sampling import CROP_MARGIN, load_frames_csv, sample

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BINARY = ROOT / ".build-dev" / "release" / "metalfx-upscale"
DEFAULT_EPISODES = Path.home() / "Downloads"
DEFAULT_SHOTS = ROOT / "output" / "sr-bench" / "shots"

# Held out from all training data (never listed here): Carving the Landscape [DO2104H003S00], Ep 1 [DO2303H001S00].
EPISODES = {
    "teacher": ("Series 1 Arteries and Veins [DO2104H001S00]", "Series 1 Following the Rain [DO2104H002S00]"),
    "gt": ("Series 1 Ep 2 [DO2303H002S00]", "Series 1 Ep 3 [DO2303H003S00]"),
}
EPISODE_SIZE = {"teacher": (1280, 720), "gt": (1920, 1080)}
SOURCE_SIZE = {"teacher": (1280, 720), "gt": (640, 360)}  # what the sampler's crops live in
KINDS = tuple(EPISODES)
SCALE = 3
MARGIN = CROP_MARGIN
GT_BPP = 0.1  # bits per pixel of the 640x360 copy, the recipe of bench/make_clips.py
DUMP_KINDS = ("source", "base", "flow")
TMP_VIDEO = "tmp-upscale.mp4"  # the CLI always writes a video; it goes here and is deleted
TMP_PRORES = "tmp-1080p.mov"
TMP_X264 = "tmp-360p.mp4"
SWIFT_BITRATE_MBPS = 5
SWIFT_TIMEOUT_S = 1800
FFMPEG_TIMEOUT_S = 600
MAX_CONSECUTIVE_FAILURES = 5
BT709 = ["-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709"]


class BuildError(Exception):
    """One sequence could not be built."""


class StaleKept(BuildError):
    """A sequence directory out of step with its spec was left alone because it holds teacher/ targets."""


class Episode(NamedTuple):
    path: Path
    fps: float
    first_pts: float  # seconds; the video's first frame on the container timeline
    width: int
    height: int


# ---------------------------------------------------------------- pure helpers

def dump_crop(crop, margin=MARGIN):
    """The window to dump: the sampled crop (x, y, w, h) grown by `margin` px on every side."""
    x, y, w, h = crop
    return (x - margin, y - margin, w + 2 * margin, h + 2 * margin)


def gt_bitrate(fps):
    """x264 bits/s of the 640x360 gt input: 0.1 bits per pixel."""
    return round(640 * 360 * fps * GT_BPP)


def swift_cmd(binary, input, out_dir, start, duration, crop, dumps, size="4k", margin=MARGIN, dir_names=None,
              bitrate=SWIFT_BITRATE_MBPS):
    """argv of one dump run of the Swift CLI. `dumps` is a subset of source/base/flow, written to
    out_dir/<name> (`dir_names` renames, e.g. {"source": "target"}); the crop is grown by `margin`.
    The CLI also writes a video, to out_dir/tmp-upscale.mp4 at a low bitrate; the caller deletes it."""
    if not dumps or set(dumps) - set(DUMP_KINDS):
        raise ValueError(f"dumps must be a non-empty subset of {DUMP_KINDS}, got {sorted(dumps)}")
    out_dir = Path(out_dir)
    names = {**{d: d for d in DUMP_KINDS}, **(dir_names or {})}
    cmd = [str(binary), str(input), str(out_dir / TMP_VIDEO), "--engine", "ml", "--size", size,
           "--bitrate", str(bitrate), "--overwrite"]
    if start is not None:
        cmd += ["--start", f"{start:.6f}"]
    if duration is not None:
        cmd += ["--duration", f"{duration:.6f}"]
    for d in DUMP_KINDS:
        if d in dumps:
            cmd += [f"--dump-{d}", str(out_dir / names[d])]
    cmd += ["--dump-crop", ",".join(str(v) for v in dump_crop(crop, margin))]
    return cmd


def swift_range(first_frame, frames, fps):
    """--start/--duration that make the Swift CLI read exactly shot-map frames first_frame .. first_frame+frames-1.
    AVAssetReader keeps every frame overlapping [start, start+duration): half a frame in from frame
    first_frame's time, and (frames - 1) frame times long, lands in the middle of both boundaries."""
    return (first_frame + 0.5) / fps, (frames - 1) / fps


def ffmpeg_seek(first_frame, fps):
    """-ss before -i that makes ffmpeg start at shot-map frame first_frame: it keeps frames at or after the time, so
    half a frame early."""
    return max(0.0, (first_frame - 0.5) / fps)


def prores_cmd(episode, seek, frames, dst):
    """ffmpeg: `frames` frames of the 1080p episode from `seek` seconds, as 10-bit 4:4:4 ProRes 4444 tagged bt709.
    setpts moves the first frame to time 0: seeking half a frame early otherwise leaves a 0.04 s empty edit in the .mov,
    and AVFoundation then reads one extra frame at its start. Passthrough stops ffmpeg's CFR sync from duplicating
    or dropping frames (it duplicates the first one when the output starts off the frame grid)."""
    return ["ffmpeg", "-v", "error", "-y", "-ss", f"{seek:.6f}", "-i", str(episode), "-frames:v", str(frames), "-an",
            "-fps_mode", "passthrough", "-vf", "setpts=PTS-STARTPTS", "-c:v", "prores_ks", "-profile:v", "4444",
            "-pix_fmt", "yuv444p10le", *BT709, str(dst)]


def x264_cmd(src, dst, fps):
    """ffmpeg: area-downscale to 640x360 and compress like the broadcast episodes (0.1 bits/pixel)."""
    bitrate = gt_bitrate(fps)
    return ["ffmpeg", "-v", "error", "-y", "-i", str(src), "-an", "-vf", "scale=640:360:flags=area",
            "-c:v", "libx264", "-preset", "medium", "-b:v", str(bitrate), "-maxrate", str(2 * bitrate),
            "-bufsize", str(2 * bitrate), "-pix_fmt", "yuv420p", *BT709, str(dst)]


def spec_record(index, spec):
    """spec.json's content: the sampler's SeqSpec plus its id, as plain JSON types."""
    return json.loads(json.dumps({"id": index, **asdict(spec)}))


def expected_frames(record):
    return record["burn_in"] + record["frames"]


def dump_dirs(kind):
    return ("source", "base", "flow") + (("target",) if kind == "gt" else ())


def is_complete(seq_dir, record):
    """True when seq_dir holds a finished build of `record`: the same spec, rewritten with its timing once the checks
    passed, and every dump's meta.json (written last by the CLI, and not at all for a run that recorded no frames)
    has burn_in + frames frames."""
    seq_dir = Path(seq_dir)
    try:
        saved = json.loads((seq_dir / "spec.json").read_text())
        if "build_seconds" not in saved or any(saved.get(k) != v for k, v in record.items()):
            return False
        return all(read_meta(seq_dir / d)["frames"] == expected_frames(record) for d in dump_dirs(record["kind"]))
    except (OSError, ValueError, KeyError):
        return False


def expected_metas(record):
    """{dump dir: (kind, channels, width, height)} each dump of `record` must have."""
    kind = record["kind"]
    x, y, w, h = record["crop"]
    dw, dh = dump_crop((x, y, w, h))[2:]
    metas = {"source": ("source", 3, dw, dh), "base": ("base", 3, dw * SCALE, dh * SCALE),
             "flow": ("flow", 4, dw, dh)}
    if kind == "gt":
        metas["target"] = ("source", 3, w * SCALE, h * SCALE)
    return metas


def check_metas(seq_dir, record):
    """Raises BuildError unless every dump has the frame count, kind and shape `record` calls for. Returns the flow
    frames marked invalid (frame 0 must be one of them)."""
    seq_dir = Path(seq_dir)
    expected = expected_frames(record)
    for name, (kind, channels, width, height) in expected_metas(record).items():
        try:
            meta = read_meta(seq_dir / name)
        except (OSError, ValueError) as error:
            raise BuildError(f"{name}/meta.json is missing or unreadable ({error})")
        got = (meta["kind"], meta["channels"], meta["width"], meta["height"])
        if meta["frames"] != expected:
            raise BuildError(f"{name} has {meta['frames']} frames, expected {expected}")
        if got != (kind, channels, width, height):
            raise BuildError(f"{name} is {got}, expected {(kind, channels, width, height)}")
    flow = read_meta(seq_dir / "flow")
    invalid = [i for i, ok in enumerate(flow["valid"]) if not ok]
    if 0 not in invalid:
        raise BuildError("flow frame 0 should be marked invalid (it has no previous frame)")
    return invalid


def check_pts(source_dir, first_frame, fps, tolerance=0.002):
    """Raises BuildError unless the Swift-written source dump holds shot-map frames first_frame, first_frame + 1, ...:
    frame k of the dump is stamped (first_frame + k) / fps on the container's timeline, which is the shot map's frame
    time. Frame 0 carries the requested range start instead (the reader clamps the first frame to it), half a frame
    later than its own time."""
    pts = read_meta(source_dir)["pts"]
    for k, got in enumerate(pts):
        want = (first_frame + k) / fps
        if k == 0:
            ok = want - tolerance <= got < want + 1 / fps
        else:
            ok = abs(got - want) <= tolerance
        if not ok:
            raise BuildError(f"source frame {k} is stamped {got:.3f} s, expected {want:.3f} s: the excerpt does not "
                             f"start at shot-map frame {first_frame}")


def frame_diffs(source_dir):
    """Mean |frame i - frame i-1| over the source dump, in 8-bit levels; entry 0 is 0."""
    meta = read_meta(source_dir)
    diffs, previous = [0.0], None
    for i in range(meta["frames"]):
        frame = read_frame(source_dir, i).astype(np.float32) * 255.0
        if previous is not None:
            diffs.append(float(np.abs(frame - previous).mean()))
        previous = frame
    return diffs


def check_cuts(source_dir, cuts):
    """Raises BuildError unless each cut offset is where consecutive source frames differ most, which is how the
    shot map's frame numbers and the dump's frames are known to line up (a one-frame slip moves the peak)."""
    if not cuts:
        return
    diffs = frame_diffs(source_dir)
    peak = int(np.argmax(diffs))
    if [peak] != list(cuts):
        raise BuildError(f"frame difference peaks at frame {peak} ({diffs[peak]:.1f}), the cut should be at "
                         f"{list(cuts)}; frames are misaligned with the shot map")


# ---------------------------------------------------------------- running tools

def run_checked(cmd, timeout):
    """Run a tool; BuildError with its output tail when it fails or hangs."""
    name = Path(cmd[0]).name
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise BuildError(f"{name} timed out after {timeout} s")
    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or proc.stdout or "").strip().splitlines()[-8:])
        raise BuildError(f"{name} exited with {proc.returncode}: {tail}")


def probe_episode(path):
    """fps, size and first-frame time of an episode (ffprobe)."""
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,r_frame_rate,start_time", "-of", "json", str(path)], text=True)
    stream = json.loads(out)["streams"][0]
    num, den = stream["r_frame_rate"].split("/")
    return Episode(Path(path), float(num) / float(den), float(stream["start_time"]),
                   int(stream["width"]), int(stream["height"]))


def load_episodes(episodes_dir, kind):
    """{stem: Episode} of the kind's episodes, after checking their frame size."""
    found = {}
    for stem in EPISODES[kind]:
        episode = probe_episode(Path(episodes_dir) / f"{stem}.mp4")
        if (episode.width, episode.height) != EPISODE_SIZE[kind]:
            raise SystemExit(f"{stem} is {episode.width}x{episode.height}, a {kind} episode should be "
                             f"{EPISODE_SIZE[kind][0]}x{EPISODE_SIZE[kind][1]}")
        found[stem] = episode
    return found


def plan(kind, n, seed, shots_dir):
    """The sampler's sequences for one kind, from the episodes' shot maps."""
    rows = {stem: load_frames_csv(Path(shots_dir) / f"{stem}.frames.csv") for stem in EPISODES[kind]}
    return sample(rows, kind, n, SOURCE_SIZE[kind], seed)


# ---------------------------------------------------------------- one sequence

class Builder:
    """Builds sequences into `out`. `run(cmd, timeout)` runs a tool (tests replace it)."""

    def __init__(self, out, binary, episodes, run=run_checked, rebuild=False):
        self.out, self.binary, self.episodes, self.run = Path(out), Path(binary), episodes, run
        self.rebuild = rebuild  # also delete stale sequences that hold teacher/ targets

    def seq_dir(self, kind, index):
        return self.out / kind / f"{index:05d}"

    def build(self, index, spec):
        """Build one sequence (skipped when already complete). Returns 'built' or 'skipped'; BuildError on failure,
        after removing the partial directory. A stale or incomplete directory holding teacher/ targets is not
        touched (StaleKept) unless `rebuild`."""
        record = spec_record(index, spec)
        seq = self.seq_dir(spec.kind, index)
        if is_complete(seq, record):
            return "skipped"
        if seq.exists():
            if (seq / "teacher").exists() and not self.rebuild:
                raise StaleKept(f"{seq} is out of step with its spec but holds teacher targets; kept "
                                "(pass --rebuild to delete and rebuild it)")
            shutil.rmtree(seq)
        seq.mkdir(parents=True)
        started = time.monotonic()
        try:
            (seq / "spec.json").write_text(json.dumps(record, indent=2) + "\n")
            if spec.kind == "teacher":
                self._teacher(seq, spec)
            else:
                self._gt(seq, spec)
            invalid = check_metas(seq, record)
            check_cuts(seq / "source", spec.cuts)
            done = {**record, "flow_invalid": invalid, "build_seconds": round(time.monotonic() - started, 1)}
            (seq / "spec.json").write_text(json.dumps(done, indent=2) + "\n")
        except BaseException:
            shutil.rmtree(seq, ignore_errors=True)
            raise
        finally:
            for tmp in (TMP_VIDEO, TMP_PRORES, TMP_X264):
                (seq / tmp).unlink(missing_ok=True)
        return "built"

    def _first_frame(self, spec, episode):
        first = spec.start - spec.burn_in
        first_real = round(episode.first_pts * episode.fps)
        if first < first_real:
            raise BuildError(f"first frame {first} comes before the episode's first real frame {first_real}")
        return first

    def _swift(self, args, **kwargs):
        self.run(swift_cmd(self.binary, *args, **kwargs), SWIFT_TIMEOUT_S)

    def _teacher(self, seq, spec):
        episode = self.episodes[spec.episode]
        first = self._first_frame(spec, episode)
        n = spec.burn_in + spec.frames
        start, duration = swift_range(first, n, episode.fps)
        self._swift((episode.path, seq, start, duration, spec.crop, set(DUMP_KINDS)), size="4k")
        check_pts(seq / "source", first, episode.fps)

    def _gt(self, seq, spec):
        episode = self.episodes[spec.episode]
        first = self._first_frame(spec, episode)
        n = spec.burn_in + spec.frames
        x, y, w, h = spec.crop
        prores, x264 = seq / TMP_PRORES, seq / TMP_X264
        self.run(prores_cmd(episode.path, ffmpeg_seek(first, episode.fps), n, prores), FFMPEG_TIMEOUT_S)
        # The real 1080p pixels of the central crop; the default 4K output is 2x, and no base is dumped.
        self._swift((prores, seq, None, None, (SCALE * x, SCALE * y, SCALE * w, SCALE * h), {"source"}),
                    size="4k", margin=0, dir_names={"source": "target"})
        self.run(x264_cmd(prores, x264, episode.fps), FFMPEG_TIMEOUT_S)
        self._swift((x264, seq, None, None, spec.crop, set(DUMP_KINDS)), size="1920x1080")


# ---------------------------------------------------------------- the run

def episode_tag(stem):
    """The catalogue number in an episode's name: 'Series 1 Ep 2 [DO2303H002S00]' -> 'DO2303H002S00'."""
    return stem.rpartition("[")[2].rstrip("]")


def hms(seconds):
    seconds = int(seconds)
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def sequences_listing(specs_by_kind):
    """The rows of sequences.json: every kind's list, with the ids the directories use."""
    return [{"id": i, **asdict(s)} for kind in KINDS for i, s in enumerate(specs_by_kind.get(kind, []))]


def write_sequences(out, specs_by_kind):
    """sequences.json: every kind's list, with the ids the directories use."""
    listing = sequences_listing(specs_by_kind)
    tmp = Path(out) / "sequences.json.tmp"
    tmp.write_text(json.dumps(listing, indent=1) + "\n")
    tmp.replace(Path(out) / "sequences.json")


def plan_change(out, specs_by_kind):
    """None when <out>/sequences.json is absent or lists exactly this plan; otherwise what differs, in words."""
    path = Path(out) / "sequences.json"
    if not path.exists():
        return None
    try:
        recorded = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        return f"it cannot be read ({error})"
    planned = json.loads(json.dumps(sequences_listing(specs_by_kind)))
    if recorded == planned:
        return None

    def count(rows, kind):
        return sum(1 for r in rows if isinstance(r, dict) and r.get("kind") == kind)
    sizes = ", ".join(f"{kind} {count(recorded, kind)} -> {count(planned, kind)}" for kind in KINDS)
    changed = sum(1 for a, b in zip(recorded, planned) if a != b) + abs(len(recorded) - len(planned))
    return f"it lists another plan ({sizes}; {changed} rows differ)"


def run_all(builder, specs_by_kind, limit, guard, log=print):
    """Build every kind's sequences in order. Returns (exit code, counts)."""
    jobs = [(kind, i, s) for kind in KINDS for i, s in enumerate(specs_by_kind.get(kind, [])[:limit])]
    counts = {"built": 0, "skipped": 0, "failed": 0}
    started = time.monotonic()
    consecutive_failures = 0
    code = 0
    for number, (kind, i, spec) in enumerate(jobs, 1):
        try:
            guard.check()
        except SwapLimitExceeded as error:
            log(f"STOPPED by the swap guard before {kind} {i:05d}: {error}. Rerun to resume.")
            code = SWAP_EXIT_CODE
            break
        t0 = time.monotonic()
        label = None
        try:
            result = builder.build(i, spec)
            consecutive_failures = 0
            note = ""
        except StaleKept as error:  # a deliberate stop for this sequence, not a tool failure: no streak
            result, label, note = "failed", "STALE-KEPT", f"  {error}"
        except BuildError as error:
            result, note = "failed", f"  {str(error).splitlines()[0][:200]}"
            consecutive_failures += 1
            for line in str(error).splitlines()[1:]:
                log(f"    {line}")
        counts[result] += 1
        elapsed = time.monotonic() - started
        remaining = len(jobs) - number
        eta = f" eta {hms(remaining * elapsed / number)}" if counts["built"] and remaining else ""
        log(f"[{number:4d}/{len(jobs)}] {kind} {i:05d} {episode_tag(spec.episode)} start={spec.start} "
            f"burn_in={spec.burn_in} frames={spec.frames} {spec.bucket:<8} {label or result.upper():<7} "
            f"{time.monotonic() - t0:5.1f}s | built {counts['built']} skipped {counts['skipped']} "
            f"failed {counts['failed']} | elapsed {hms(elapsed)}{eta}{note}")
        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            log(f"STOPPED after {consecutive_failures} failures in a row")
            code = 1
            break
    code = code or (1 if counts["failed"] else 0)
    log(f"SUMMARY {'ok' if code == 0 else 'incomplete'}: built {counts['built']} skipped {counts['skipped']} "
        f"failed {counts['failed']} of {len(jobs)} planned in {hms(time.monotonic() - started)} exit {code}")
    return code, counts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", required=True, help="output directory (output/detail-data)")
    parser.add_argument("--teacher-seqs", type=int, required=True)
    parser.add_argument("--gt-seqs", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None,
                        help="build only the first N sequences of each kind (the lists, and so the ids, are unchanged)")
    parser.add_argument("--binary", default=str(DEFAULT_BINARY), help="the Swift CLI (dev build)")
    parser.add_argument("--episodes", default=str(DEFAULT_EPISODES), help="folder with the episode .mp4 files")
    parser.add_argument("--shots", default=str(DEFAULT_SHOTS), help="folder with the <stem>.frames.csv shot maps")
    parser.add_argument("--swap-limit-gb", type=float, default=8.0)
    parser.add_argument("--rebuild", action="store_true",
                        help="also delete and rebuild stale sequences that already hold teacher/ targets")
    parser.add_argument("--replan", action="store_true",
                        help="replace an existing sequences.json that lists another plan (changed sequences are rebuilt)")
    args = parser.parse_args(argv)

    if not Path(args.binary).exists():
        raise SystemExit(f"{args.binary} not found; run `swift build -c release --scratch-path .build-dev`")
    guard = SwapGuard(args.swap_limit_gb)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    counts = {"teacher": args.teacher_seqs, "gt": args.gt_seqs}
    specs = {kind: plan(kind, counts[kind], args.seed, args.shots) for kind in KINDS}
    for kind in KINDS:
        if len(specs[kind]) < counts[kind]:
            print(f"warning: the sampler found only {len(specs[kind])} of {counts[kind]} {kind} sequences", flush=True)
    change = plan_change(out, specs)
    if change and not args.replan:
        raise SystemExit(f"refusing to start: {out / 'sequences.json'} exists and {change}. These arguments "
                         f"(--teacher-seqs {args.teacher_seqs} --gt-seqs {args.gt_seqs} --seed {args.seed}) plan other "
                         "sequences, and building them would delete the dumps of every sequence whose spec changed "
                         "(and leave its teacher/ targets stale). Rerun with the arguments of the existing plan to "
                         "resume it, or pass --replan to replace the plan on purpose.")
    if change:
        print(f"replanning: {out / 'sequences.json'} {change}", flush=True)
    write_sequences(out, specs)
    episodes = {}
    for kind in KINDS:
        episodes.update(load_episodes(args.episodes, kind))
    print(f"building {len(specs['teacher'])} teacher + {len(specs['gt'])} gt sequences into {out} "
          f"(seed {args.seed}, limit {args.limit}, swap limit {args.swap_limit_gb:g} GB)", flush=True)
    builder = Builder(out, args.binary, episodes, rebuild=args.rebuild)
    code, _ = run_all(builder, specs, args.limit, guard, log=lambda line: print(line, flush=True))
    return code


if __name__ == "__main__":
    sys.exit(main())
