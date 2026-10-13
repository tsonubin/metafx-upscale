"""Run one model on one benchmark clip.

    python bench/run_model.py MODEL CLIP_DIR [--ignore-cuts]

Reads CLIP_DIR/clip.json and CLIP_DIR/lr.mkv, splits at the clip's scene cuts (unless
--ignore-cuts, used to measure how much a model leaks across a cut), upscales by the
model's native factor, resizes to the clip's target size with area averaging (as the
Apple path does), and writes a lossless CLIP_DIR/out/<model>[-nocut].mkv plus a
<model>[-nocut].json with timing and memory.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clipio import read_frames, shots_of, FFV1Writer  # noqa: E402
from models import REGISTRY, sync  # noqa: E402

REPO = Path(__file__).resolve().parent.parent


CROP = (1920, 1080)
CROP_MARGIN = 32  # LR pixels of context around the cropped region


def crop_plan(clip, size=CROP):
    """For --center-crop: the LR box to upscale and how to cut the centre `size` region of
    the target out of the model's 4x output. Returns (x0, y0, w, h, r, m)."""
    (lw, lh), (tw, th) = clip["lr_size"], clip["target"]
    r = tw / lw
    assert abs(th / lh - r) < 1e-6 and r * CROP_MARGIN == int(r * CROP_MARGIN)
    cw, ch = round(size[0] / r), round(size[1] / r)
    x0, y0 = (lw - cw) // 2 - CROP_MARGIN, (lh - ch) // 2 - CROP_MARGIN
    return x0, y0, cw + 2 * CROP_MARGIN, ch + 2 * CROP_MARGIN, r, CROP_MARGIN


def run_python_model(name, clip, frames, shots, out_path, center_crop=False, crop_size=CROP):
    model = REGISTRY[name]()
    tw, th = clip["target"]
    if center_crop:
        x0, y0, cw, ch, r, m = crop_plan(clip, crop_size)
        frames = [f[y0:y0 + ch, x0:x0 + cw] for f in frames]
        rw, rh = round(cw * r), round(ch * r)
        ox, oy = round(m * r), round(m * r)
        tw, th = crop_size

        def post(img):
            img = cv2.resize(img, (rw, rh), interpolation=cv2.INTER_AREA)
            return img[oy:oy + th, ox:ox + tw]
    else:
        def post(img):
            if img.shape[1] != tw or img.shape[0] != th:
                img = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
            return img
    writer = FFV1Writer(out_path, tw, th, clip["fps"])
    pending = {}
    next_out = 0
    peak = 0

    def emit_from(offset):
        def emit(i, hr):
            nonlocal next_out, peak
            img = (hr.permute(1, 2, 0).mul(255.0).round_().to(torch.uint8).cpu().numpy())
            pending[offset + i] = post(img)
            while next_out in pending:
                writer.write(pending.pop(next_out))
                next_out += 1
            peak = max(peak, torch.mps.driver_allocated_memory())
        return emit

    # Warm-up so kernel compilation isn't timed.
    model.run_shot(frames[:getattr(model, "warmup_frames", 2 if model.temporal else 1)], lambda i, hr: None)
    sync()
    t0 = time.perf_counter()
    for a, b in shots:
        model.run_shot(frames[a:b], emit_from(a))
    sync()
    elapsed = time.perf_counter() - t0
    writer.close()
    assert next_out == len(frames), f"wrote {next_out}/{len(frames)} frames"
    return {"model": name, "label": model.label, "frames": len(frames), "seconds": round(elapsed, 2),
            "s_per_frame": round(elapsed / len(frames), 4), "peak_gpu_gb": round(peak / 2**30, 2),
            "note": "model compute + resize to target, excl. decode; batch job running concurrently on GPU",
            "center_crop": center_crop, "out_size": [tw, th], "frame_pixels": int(frames[0].shape[0] * frames[0].shape[1] * model.scale ** 2)}


def run_apple(clip_dir, clip, shots, out_path):
    """Apple VTSuperResolutionScaler through the project's own CLI, one call per shot.
    Output is HEVC 10-bit at a near-lossless bitrate, then decoded into the lossless mkv."""
    binary = REPO / ".build" / "release" / "metalfx-upscale"
    tw, th = clip["target"]
    tmp = clip_dir / "out" / "apple-tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    writer = FFV1Writer(out_path, tw, th, clip["fps"])
    total = 0.0
    bitrate = str(round(tw * th * clip["fps"] * 1.6 / 1e6))  # ~1.6 bits/pixel: far above visible loss
    for k, (a, b) in enumerate(shots):
        src = tmp / f"shot{k}.mov"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(clip_dir / "lr.mkv"),
                        "-vf", f"select='between(n,{a},{b - 1})',setpts=N/FRAME_RATE/TB",
                        "-c:v", "prores_ks", "-profile:v", "4444", "-pix_fmt", "yuv444p10le",
                        "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
                        str(src)], check=True)
        dst = tmp / f"shot{k}.mp4"
        t0 = time.perf_counter()
        subprocess.run([str(binary), str(src), str(dst), "--engine", "ml", "--size", f"{tw}x{th}",
                        "--bitrate", bitrate, "--overwrite"], check=True, capture_output=True)
        total += time.perf_counter() - t0
        n = 0
        for img in read_frames(dst, tw, th):
            writer.write(img)
            n += 1
        assert n == b - a, f"Apple shot {k}: got {n} frames, expected {b - a}"
    writer.close()
    frames = shots[-1][1]
    return {"model": "apple", "label": "Apple AI (VTSuperResolutionScaler)", "frames": frames,
            "seconds": round(total, 2), "s_per_frame": round(total / frames, 4), "peak_gpu_gb": None,
            "note": f"end-to-end CLI incl. decode+HEVC encode at {bitrate} Mbps; batch job running concurrently"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("clip_dir", type=Path)
    ap.add_argument("--ignore-cuts", action="store_true")
    ap.add_argument("--frames", type=int, help="only the first N frames (slow models)")
    ap.add_argument("--center-crop", action="store_true",
                    help="upscale only the LR region behind the target's central 1920x1080 (heavy models)")
    ap.add_argument("--crop", help="like --center-crop with a smaller WxH region; output is tagged -cW")
    args = ap.parse_args()

    clip = json.loads((args.clip_dir / "clip.json").read_text())
    if args.frames:
        clip["frames"] = min(clip["frames"], args.frames)
    shots = shots_of(clip, ignore_cuts=args.ignore_cuts)
    crop_size = tuple(int(v) for v in args.crop.split("x")) if args.crop else CROP
    suffix = (f"-c{crop_size[0]}" if args.crop else "") + ("-nocut" if args.ignore_cuts else "")
    args.center_crop = args.center_crop or bool(args.crop)
    (args.clip_dir / "out").mkdir(exist_ok=True)
    out_path = args.clip_dir / "out" / f"{args.model}{suffix}.mkv"

    if args.model == "apple":
        stats = run_apple(args.clip_dir, clip, shots, out_path)
    else:
        w, h = clip["lr_size"]
        frames = list(read_frames(args.clip_dir / "lr.mkv", w, h))[:clip["frames"]]
        stats = run_python_model(args.model, clip, frames, shots, out_path,
                                 center_crop=args.center_crop and clip["target"][0] > crop_size[0],
                                 crop_size=crop_size)
    stats["model"] = args.model + suffix.replace("-nocut", "")
    stats["clip"] = clip["name"]
    stats["ignore_cuts"] = args.ignore_cuts
    (args.clip_dir / "out" / f"{args.model}{suffix}.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
