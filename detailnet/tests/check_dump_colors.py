"""Colour check between a source dump and a base dump (plan M1 Review Focus 1).

Run it on the output of `metalfx-upscale --dump-source SRC --dump-base BASE ...`:

    python detailnet/tests/check_dump_colors.py SRC BASE

The base is area-downscaled by its scale (a mean over each s x s block) and compared with the source frame by
frame, per RGB channel, in 8-bit levels (base - source). Apple's model changes fine texture by design, so the
check gates on colour, not texture. It fails (exit 1) unless, for every channel,
  * |bias| <= MAX_ABS_BIAS, where bias is the mean signed delta,
  * the spread (max - min) of the three biases is < MAX_BIAS_SPREAD, and
  * the mean |delta| after a Gaussian low-pass (sigma = LOWPASS_SIGMA source px, applied to both images) is
    <= MAX_LOWPASS_ABS.
The raw mean |delta| is printed for information only. Both dumps are read through detailnet.dumps, so this is also
the first check that a Swift-written dump reads correctly in Python.

It is a script, not a pytest test: the `check_` prefix keeps pytest from collecting it
(the gating logic is tested in test_check_dump_colors.py).
"""
import argparse
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from detailnet.dumps import read_frame, read_meta  # noqa: E402

# Gates, in 8-bit levels.
MAX_ABS_BIAS = 3.0       # per channel: |mean signed delta|
MAX_BIAS_SPREAD = 1.0    # max - min of the three channel biases (exclusive)
MAX_LOWPASS_ABS = 3.0    # per channel: mean |delta| after the low-pass
LOWPASS_SIGMA = 8.0      # Gaussian sigma in source pixels, applied to both images
CHANNELS = "RGB"


class Result(NamedTuple):
    frames: int
    sigma: float
    raw_abs: np.ndarray       # per channel, information only
    bias: np.ndarray          # per channel mean signed delta
    lowpass_abs: np.ndarray   # per channel mean |delta| after the Gaussian low-pass


def area_downscale(frame, s):
    """Mean over each s x s block of a C x H x W array (a ragged edge is dropped)."""
    c, h, w = frame.shape
    h, w = h // s * s, w // s * s
    return frame[:, :h, :w].reshape(c, h // s, s, w // s, s).mean(axis=(2, 4))


def gaussian_blur(frame, sigma):
    """Separable Gaussian low-pass of a C x H x W array; the edges are mirrored."""
    radius = int(np.ceil(3 * sigma))
    taps = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    taps /= taps.sum()
    out = frame
    for axis in (1, 2):
        pad = [(0, 0)] * 3
        pad[axis] = (radius, radius)
        padded = np.pad(out, pad, mode="reflect")
        n = out.shape[axis]
        acc = np.zeros_like(out)
        for i, weight in enumerate(taps):
            window = [slice(None)] * 3
            window[axis] = slice(i, i + n)
            acc += weight * padded[tuple(window)]
        out = acc
    return out


def evaluate(src_dir, base_dir, max_frames=None, sigma=LOWPASS_SIGMA):
    """Compares the dumps frame by frame and returns the per-channel statistics, in 8-bit levels."""
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

    raw_sum, bias_sum, lowpass_sum = np.zeros(3), np.zeros(3), np.zeros(3)
    for i in range(frames):
        src = read_frame(src_dir, i).astype(np.float32) * 255.0
        base = area_downscale(read_frame(base_dir, i).astype(np.float32) * 255.0, s)
        delta = base - src
        raw_sum += np.abs(delta).mean(axis=(1, 2))
        bias_sum += delta.mean(axis=(1, 2))
        lowpass_sum += np.abs(gaussian_blur(base, sigma) - gaussian_blur(src, sigma)).mean(axis=(1, 2))
    return Result(frames, sigma, raw_sum / frames, bias_sum / frames, lowpass_sum / frames)


def failures(result):
    """The gates that `result` breaks, as messages; empty when the check passes."""
    failed = []
    if np.abs(result.bias).max() > MAX_ABS_BIAS:
        failed.append(f"a channel's |bias| is {np.abs(result.bias).max():.3f} > {MAX_ABS_BIAS}")
    spread = float(result.bias.max() - result.bias.min())
    if spread >= MAX_BIAS_SPREAD:
        failed.append(f"the channel spread of the biases is {spread:.3f} >= {MAX_BIAS_SPREAD}")
    if result.lowpass_abs.max() > MAX_LOWPASS_ABS:
        failed.append(
            f"a channel's low-pass (sigma {result.sigma:g}) mean |delta| is {result.lowpass_abs.max():.3f} > {MAX_LOWPASS_ABS}")
    return failed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("src", help="source dump directory")
    parser.add_argument("base", help="base dump directory")
    parser.add_argument("--frames", type=int, default=None, help="compare only the first N frames")
    args = parser.parse_args(argv)

    result = evaluate(args.src, args.base, args.frames)
    print(f"{result.frames} frames; base area-downscaled to the source grid; base - source, in 8-bit levels")
    for c, name in enumerate(CHANNELS):
        print(f"  {name}: bias {result.bias[c]:+.3f}   low-pass(sigma {result.sigma:g}) mean |delta| "
              f"{result.lowpass_abs[c]:.3f}   raw mean |delta| {result.raw_abs[c]:.3f} (information only)")
    print(f"  bias spread across channels: {float(result.bias.max() - result.bias.min()):.3f}")

    failed = failures(result)
    if failed:
        print("FAIL: " + "; ".join(failed))
        return 1
    print(f"PASS: |bias| <= {MAX_ABS_BIAS}, bias spread < {MAX_BIAS_SPREAD}, "
          f"low-pass mean |delta| <= {MAX_LOWPASS_ABS} per channel")
    return 0


if __name__ == "__main__":
    sys.exit(main())
