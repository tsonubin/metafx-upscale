"""Sanity-check the jitter metric against synthetic cases with a known answer.

    python bench/validate_jitter.py CLIP_DIR [MODEL ...]

On a ground-truth clip, builds three synthetic "models" from GT itself:
  gt          the real 1080p frames                         → coherent, sharp
  gt-sharp    GT with an unsharp mask (extra detail that      → coherent, should score like gt
              moves with the content)
  gt-reroll   GT + fine random texture re-drawn every frame  → incoherent, must score high
              (same spatial energy as the sharpening adds)
and scores them, plus any real model outputs given, with three motion sources:
  lr    flow from the low-res input, scaled up (what real clips can use)
  gt    flow from the 1080p ground truth (only possible on GT clips)
  self  flow computed on each output itself
A usable metric/flow combination must rank gt ≈ gt-sharp ≪ gt-reroll.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clipio import read_frames, shots_of  # noqa: E402
from metrics import grad_map, luma, warp  # noqa: E402

DIS = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)


def flow_pair(g_t, g_prev):
    """Backward flow (t→t-1) and fb-consistency mask on 8-bit gray frames."""
    b = DIS.calc(g_t, g_prev, None)
    f = DIS.calc(g_prev, g_t, None)
    h, w = g_t.shape
    ly, lx = np.mgrid[0:h, 0:w].astype(np.float32)
    px, py = lx + b[..., 0], ly + b[..., 1]
    f_at = cv2.remap(f, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    err = np.sum((b + f_at) ** 2, axis=2)
    bound = 0.01 * (np.sum(b ** 2, axis=2) + np.sum(f_at ** 2, axis=2)) + 0.5
    inside = (px >= 0) & (px <= w - 1) & (py >= 0) & (py <= h - 1)
    return b, (err < bound) & inside


def to_out(flow, mask, w, h):
    sx, sy = w / flow.shape[1], h / flow.shape[0]
    f = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR)
    gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)
    m = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    return gx + f[..., 0] * sx, gy + f[..., 1] * sy, m


def gray8(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def jitter(frames, bic, flows):
    vals = []
    for t, (mx, my, mask) in flows.items():
        y, yp = luma(frames[t]), luma(frames[t - 1])
        d = (y - warp(yp, mx, my)) - (bic[t] - warp(bic[t - 1], mx, my))
        if mask.mean() > 0.2:
            vals.append(float(np.abs(d)[mask].mean()) / max(float(grad_map(y)[mask].mean()), 1e-6))
    return float(np.mean(vals))


def main(clip_dir, *models):
    clip_dir = Path(clip_dir)
    clip = json.loads((clip_dir / "clip.json").read_text())
    lw, lh = clip["lr_size"]
    tw, th = clip["target"]
    lr = list(read_frames(clip_dir / "lr.mkv", lw, lh))
    gt = list(read_frames(clip_dir / "gt.mkv", tw, th))
    bic = [luma(cv2.resize(f, (tw, th), interpolation=cv2.INTER_CUBIC)) for f in lr]
    shots = shots_of(clip)
    pairs = [t for a, b in shots for t in range(a + 1, b)]

    rng = np.random.default_rng(0)
    blur = [cv2.GaussianBlur(f.astype(np.float32), (0, 0), 1.5) for f in gt]
    sharp = [np.clip(f + 1.0 * (f - b), 0, 255).astype(np.uint8) for f, b in zip(gt, blur)]
    added = float(np.mean([np.abs(s.astype(np.float32) - f).mean() for s, f in zip(sharp, gt)]))
    reroll = []
    for f in gt:
        n = cv2.GaussianBlur(rng.standard_normal((th, tw)).astype(np.float32), (0, 0), 0.7)
        n *= added / max(float(np.abs(n).mean()), 1e-6)
        reroll.append(np.clip(f.astype(np.float32) + n[..., None], 0, 255).astype(np.uint8))
    candidates = {"gt": gt, "gt-sharp": sharp, "gt-reroll": reroll}
    for m in models:
        candidates[m] = list(read_frames(clip_dir / "out" / f"{m}.mkv", tw, th))

    lr_flows = {t: to_out(*flow_pair(gray8(lr[t]), gray8(lr[t - 1])), tw, th) for t in pairs}
    gt_flows = {t: to_out(*flow_pair(gray8(gt[t]), gray8(gt[t - 1])), tw, th) for t in pairs}
    print(f"{clip['name']}  (jitter, output px; lower = steadier)")
    print(f"{'output':20s} {'lr-flow':>8s} {'gt-flow':>8s} {'self-flow':>9s}")
    for name, frames in candidates.items():
        self_flows = {t: to_out(*flow_pair(gray8(frames[t]), gray8(frames[t - 1])), tw, th) for t in pairs}
        print(f"{name:20s} {jitter(frames, bic, lr_flows):8.3f} {jitter(frames, bic, gt_flows):8.3f} "
              f"{jitter(frames, bic, self_flows):9.3f}", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
