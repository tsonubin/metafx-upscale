"""Colour check between a source dump and a base dump (plan M1 Review Focus 1).

Run it on the output of `metalfx-upscale --dump-source SRC --dump-base BASE ...`:

    python detailnet/tests/check_dump_colors.py SRC BASE

The base is area-downscaled by its scale (a mean over each s x s block) and compared with the source frame by
frame, per RGB channel, in 8-bit levels. It fails (exit 1) unless
  * every channel's mean |delta| is <= 3 levels, and
  * the spread (max - min) of the per-channel mean signed delta is < 1 level,
which catches a colour or range mismatch between the Swift-decoded source and Apple's base. Both dumps are read
through detailnet.dumps, so this is also the first check that a Swift-written dump reads correctly in Python.

`--lowpass K` first box-averages both images over K x K source pixels, so only colour and tone are compared and
the detail Apple's model adds or removes drops out. The pass/fail limits are the same either way.

It is a script, not a pytest test: the `check_` prefix keeps pytest from collecting it.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from detailnet.dumps import read_frame, read_meta  # noqa: E402

MAX_MEAN_ABS = 3.0
MAX_BIAS_SPREAD = 1.0
CHANNELS = "RGB"


def area_downscale(frame, s):
    """Mean over each s x s block of a C x H x W array (a ragged edge is dropped)."""
    c, h, w = frame.shape
    h, w = h // s * s, w // s * s
    return frame[:, :h, :w].reshape(c, h // s, s, w // s, s).mean(axis=(2, 4))


def compare(src_dir, base_dir, max_frames=None, lowpass=1):
    """Returns (mean |delta| per channel, mean signed delta per channel, frames compared), in 8-bit levels."""
    src_meta, base_meta = read_meta(src_dir), read_meta(base_dir)
    s = base_meta["scale"]
    if src_meta["kind"] != "source" or base_meta["kind"] != "base":
        raise SystemExit(f"expected a source dump and a base dump, got {src_meta['kind']} and {base_meta['kind']}")
    if src_meta["channels"] != 3 or base_meta["channels"] != 3:
        raise SystemExit("both dumps must have 3 (RGB) channels")
    if src_meta["crop"] != base_meta["crop"] or src_meta["source_size"] != base_meta["source_size"]:
        raise SystemExit("the dumps cover different crops or sources")
    if (base_meta["width"], base_meta["height"]) != (src_meta["width"] * s, src_meta["height"] * s):
        raise SystemExit(f"base is {base_meta['width']}x{base_meta['height']}, not {s}x the source dump")
    frames = min(src_meta["frames"], base_meta["frames"])
    if src_meta["frames"] != base_meta["frames"]:
        print(f"warning: {src_meta['frames']} source frames but {base_meta['frames']} base frames", file=sys.stderr)
    if max_frames:
        frames = min(frames, max_frames)
    if frames == 0:
        raise SystemExit("no frames to compare")

    abs_sum = np.zeros(3)
    signed_sum = np.zeros(3)
    for i in range(frames):
        src = read_frame(src_dir, i).astype(np.float32) * 255.0
        base = area_downscale(read_frame(base_dir, i).astype(np.float32) * 255.0, s)
        if lowpass > 1:
            src, base = area_downscale(src, lowpass), area_downscale(base, lowpass)
        delta = base - src
        abs_sum += np.abs(delta).mean(axis=(1, 2))
        signed_sum += delta.mean(axis=(1, 2))
    return abs_sum / frames, signed_sum / frames, frames


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("src", help="source dump directory")
    parser.add_argument("base", help="base dump directory")
    parser.add_argument("--frames", type=int, default=None, help="compare only the first N frames")
    parser.add_argument("--lowpass", type=int, default=1, metavar="K",
                        help="box-average both images over K x K source pixels first (default 1: off)")
    args = parser.parse_args()

    mean_abs, mean_signed, frames = compare(args.src, args.base, args.frames, args.lowpass)
    blur = f", both box-averaged over {args.lowpass}x{args.lowpass} source px" if args.lowpass > 1 else ""
    print(f"{frames} frames; base area-downscaled to the source grid{blur}; base - source, in 8-bit levels")
    for c, name in enumerate(CHANNELS):
        print(f"  {name}: mean |delta| {mean_abs[c]:.3f}   mean signed delta {mean_signed[c]:+.3f}")
    spread = float(mean_signed.max() - mean_signed.min())
    print(f"  signed-delta spread across channels: {spread:.3f}")

    failures = []
    if mean_abs.max() > MAX_MEAN_ABS:
        failures.append(f"a channel's mean |delta| is {mean_abs.max():.3f} > {MAX_MEAN_ABS}")
    if spread >= MAX_BIAS_SPREAD:
        failures.append(f"the signed-delta spread is {spread:.3f} >= {MAX_BIAS_SPREAD}")
    if failures:
        print("FAIL: " + "; ".join(failures))
        return 1
    print(f"PASS: mean |delta| <= {MAX_MEAN_ABS} per channel and bias spread < {MAX_BIAS_SPREAD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
