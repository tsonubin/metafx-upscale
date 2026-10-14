"""Run a trained DetailNet over the bench clips and write outputs `bench/metrics.py` can score.

    python -m detailnet.run_bench --ckpt output/detail-runs/r1/latest.pt --name dn-r1 [--strength 1.0] [--ignore-cuts]
                                  [--device mps] output/sr-bench/clips/c-* output/sr-bench/clips/g3-*

Input is `python -m detailnet.bench_dumps CLIP_DIR...`: per shot, the source frames, Apple AI's x3 base and the optical
flow. Output is CLIP_DIR/out/NAME.mkv (lossless FFV1, like every bench output) and NAME.json (the stats format of
`bench/run_model.py`), or NAME-nocut.* with --ignore-cuts.

    4K clips (c-*)   the whole dumped region at once (LR 704x424, base 2112x1272); the composed frame is cut by the
                     context margin (32 source px = 96 output px on every side) to the scored 1920x1080:
                     center_crop true, out_size [1920, 1080];
    g3 clips         the full frame, 1920x1080 directly, center_crop false.

The network is run the way it was trained (`detailnet.train.run_sequence`, rule R23). The state restarts (prev_detail and
hidden zero, flow zero, x_prev = x_t) at the first frame of the run, at every shot start, and at every frame whose flow
the dump marks invalid. Flow is cleaned like training data (non-finite -> 0, clamped to +-64 source px). The recurrence
carries the raw detail; `strength` only scales what is added to the base in the output (compose), so strength 0 gives the
base crop back exactly.

--ignore-cuts is the cut-leak measurement (bench/metrics.py compares NAME-nocut against NAME). At a shot start the state is
carried over instead: the flow is zero there (a shot start has no flow), x_prev is the previous frame, and prev_detail and
hidden keep their values; invalid flow elsewhere still restarts the state. Both runs use the same per-shot dumps, so the
difference between them is DetailNet's own leak over the same base.

`frame_pixels` is the 4x-equivalent pixel count of the processed region (16 x LR pixels), the unit `bench/run_model.py`
uses for its 4x models, so `bench/summarize.py` scales the seconds per frame to a full 720p frame correctly.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from detailnet.bench_dumps import CROP, check_shot, dump_plan  # also puts bench/ on sys.path
from detailnet.build_data import BuildError
from detailnet.data import clean_flow
from detailnet.dumps import read_flow, read_frame, read_meta
from detailnet.model import compose
from clipio import FFV1Writer  # noqa: E402
from run_model import crop_plan  # noqa: E402


class DumpError(RuntimeError):
    """The clip's dumps are missing, unfinished or do not match its clip.json."""


class Shot:
    def __init__(self, index, start, end, directory):
        self.index, self.start, self.end, self.dir = index, start, end, Path(directory)

    def __len__(self):
        return self.end - self.start


def find_shots(clip_dir, clip, plan):
    """The planned shots with their dump directories, checked against the dumps on disk."""
    shots = []
    for k, (a, b) in enumerate(plan.shots):
        d = Path(clip_dir) / "dumps" / f"shot{k}"
        try:
            check_shot(d, f"{clip['name']} shot{k}", b - a, plan)
        except BuildError as error:
            raise DumpError(f"{error}; (re)build the dumps with: python -m detailnet.bench_dumps {clip_dir}")
        shots.append(Shot(k, a, b, d))
    return shots


def read_inputs(shot, i, device):
    """(x_t, base, flow) of frame i of a shot: float32 tensors with a batch dimension on `device`; the flow is
    cleaned (non-finite -> 0, +-64 px) like the training data."""
    x = torch.from_numpy(read_frame(shot.dir / "source", i).astype(np.float32))[None]
    base = torch.from_numpy(read_frame(shot.dir / "base", i).astype(np.float32))[None]
    flow = torch.from_numpy(clean_flow(read_flow(shot.dir / "flow", i)))[None]
    return x.to(device), base.to(device), flow.to(device)


def sync(device):
    device = torch.device(device)
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def margin_of(clip, crop_size):
    """Output px to cut off every side of the composed frame (0 when the dumps are the output)."""
    if clip["target"][0] <= crop_size[0]:
        return 0
    _, _, _, _, ratio, margin = crop_plan(clip, crop_size)
    return round(margin * ratio)


def iter_outputs(net, clip_dir, strength=1.0, ignore_cuts=False, device="cpu", crop_size=CROP):
    """Yield (frame index, uint8 HxWx3 RGB output frame, seconds of compute) for every frame of the clip's dumps, in
    order. A warm-up pass on the first frame (not timed, not kept) comes before the first yield."""
    clip_dir = Path(clip_dir)
    clip = json.loads((clip_dir / "clip.json").read_text())
    plan = dump_plan(clip, crop_size)
    if net.scale != plan.scale:
        raise DumpError(f"the network is x{net.scale} but {clip['name']} is a x{plan.scale} clip")
    shots = find_shots(clip_dir, clip, plan)
    m = margin_of(clip, crop_size)
    h, w = plan.crop[3], plan.crop[2]
    out_size = (plan.scale * w - 2 * m, plan.scale * h - 2 * m)
    if m and out_size != tuple(crop_size):
        raise DumpError(f"{clip['name']}: the dumps give a {out_size} output, expected {tuple(crop_size)}")

    net = net.to(device).eval()
    with torch.inference_mode():
        x, base, flow = read_inputs(shots[0], 0, device)
        net(x, x, base, torch.zeros_like(flow), *net.zero_state(1, h, w, device))  # kernel compilation, untimed
        sync(device)

        prev_detail, hidden = net.zero_state(1, h, w, device)
        prev_x = None
        t = 0
        for shot in shots:
            valid = read_meta(shot.dir / "flow")["valid"]
            for i in range(len(shot)):
                x, base, flow = read_inputs(shot, i, device)
                if t == 0:
                    reset, zero_flow = True, True
                elif i == 0:  # a shot start: its flow is a placeholder
                    reset, zero_flow = not ignore_cuts, True
                else:
                    reset = zero_flow = not valid[i]
                sync(device)
                started = time.perf_counter()
                if reset:
                    prev_detail, hidden = net.zero_state(1, h, w, device)
                    x_prev = x
                else:
                    x_prev = prev_x
                if zero_flow:
                    flow = torch.zeros_like(flow)
                detail, hidden = net(x, x_prev, base, flow, prev_detail, hidden)
                out = compose(base, detail, strength)[0]
                if m:
                    out = out[:, m:-m, m:-m]
                frame = out.mul(255.0).round_().to(torch.uint8).permute(1, 2, 0).contiguous().cpu().numpy()
                seconds = time.perf_counter() - started
                prev_detail, prev_x = detail, x
                yield t, frame, seconds
                t += 1


def run_clip(net, clip_dir, name, strength=1.0, ignore_cuts=False, device="mps", ckpt=None, crop_size=CROP):
    """Run `net` over one clip and write out/NAME[-nocut].{mkv,json}; -> the stats dict."""
    clip_dir = Path(clip_dir)
    clip = json.loads((clip_dir / "clip.json").read_text())
    plan = dump_plan(clip, crop_size)
    center_crop = clip["target"][0] > crop_size[0]
    out_size = list(crop_size) if center_crop else list(clip["target"])
    stem = name + ("-nocut" if ignore_cuts else "")
    out_dir = clip_dir / "out"
    out_dir.mkdir(exist_ok=True)
    stats_path = out_dir / f"{stem}.json"
    stats_path.unlink(missing_ok=True)  # a failed rerun must not leave stats next to a half-written video
    writer = FFV1Writer(out_dir / f"{stem}.mkv", *out_size, clip["fps"])
    seconds = 0.0
    frames = 0
    try:
        for _, frame, s in iter_outputs(net, clip_dir, strength, ignore_cuts, device, crop_size):
            writer.write(frame)
            seconds += s
            frames += 1
    except BaseException:
        writer.proc.kill()
        writer.proc.wait()
        raise
    writer.close()
    assert frames == plan.frames, f"wrote {frames}/{plan.frames} frames"
    peak = torch.mps.driver_allocated_memory() / 2**30 if torch.device(device).type == "mps" else None
    stats = {"model": name, "label": f"DetailNet ({name})", "frames": frames, "seconds": round(seconds, 2),
             "s_per_frame": round(seconds / frames, 4), "peak_gpu_gb": round(peak, 2) if peak is not None else None,
             "note": f"DetailNet compute + compose + crop to 8 bit on {device}, strength {strength}; excl. dump reading, "
                     "video writing and the Apple AI base (Apple's own run); batch job running concurrently on GPU",
             "center_crop": center_crop, "out_size": out_size,
             "frame_pixels": 16 * plan.crop[2] * plan.crop[3], "clip": clip["name"], "ignore_cuts": ignore_cuts,
             "strength": strength, "ckpt": str(ckpt) if ckpt else None}
    stats_path.write_text(json.dumps(stats, indent=1))
    return stats


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run DetailNet over bench clips (see the module docstring).")
    ap.add_argument("clip_dirs", nargs="+", type=Path)
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--name", required=True, help="model name: outputs are out/NAME.mkv and NAME.json")
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--ignore-cuts", action="store_true", help="carry state across shot starts; writes NAME-nocut")
    ap.add_argument("--device", default="mps")
    args = ap.parse_args(argv)
    if "/" in args.name or args.name.endswith("-nocut"):
        ap.error("--name must be a plain model name (no slash, no -nocut suffix)")

    from detailnet.train import load_checkpoint
    net, _ = load_checkpoint(args.ckpt)
    failed = 0
    for clip_dir in args.clip_dirs:
        try:
            stats = run_clip(net, clip_dir, args.name, args.strength, args.ignore_cuts, args.device, args.ckpt)
        except DumpError as error:
            failed += 1
            print(f"FAILED {clip_dir.name}: {error}", flush=True)
            continue
        print(json.dumps({k: stats[k] for k in ("clip", "frames", "seconds", "s_per_frame", "peak_gpu_gb")}), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
