"""Quality, temporal-stability and cut-leak metrics for one model's output on one clip.

    python bench/metrics.py CLIP_DIR MODEL

Motion is tracked with DIS optical flow on each output itself at full resolution (forward +
backward, so occluded pixels are masked out) and used to warp the previous output frame
onto the current one; the motion bucket of a pair comes from the low-res source. Pairs
that straddle a scene cut are skipped.

  jitter       temporal instability in output pixels: mean |ΔY_model − ΔY_bicubic| ÷ mean
               |∇Y_model| over non-occluded pixels. ΔY is the current frame minus the
               motion-compensated previous frame; subtracting the bicubic upscale's ΔY
               removes real lighting/content change. Dividing by the spatial gradient
               cancels the first-order effect of flow error on sharp texture, so a sharp
               but temporally coherent output scores like a soft one, while texture that
               re-rolls each frame scores high. GT gives the floor (flow error + real
               change). Reported per motion bucket of the pair (static <0.3, slow <2,
               fast ≥2 px/frame at 720p scale).
  flicker      the same numerator without the normalisation (luma levels, 0–255).
  detail       mean gradient magnitude of the output ÷ that of the bicubic upscale.
  cut_leak     mean |Y| difference between cut-aware and cut-ignoring runs over the 25
               frames after each cut (needs <model>-nocut.mkv).
With a ground-truth clip (gt.mkv):
  psnr_y, lpips, dists   per-frame fidelity to GT (lower lpips/dists = closer)
  tlp          |LPIPS(out_t, out_t-1) − LPIPS(gt_t, gt_t-1)| × 100 (TecoGAN): temporal
               change that doesn't match the real footage; independent of sharpness.
  The same flicker/detail numbers are also computed for GT itself as the floor.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clipio import read_frames, shots_of  # noqa: E402

BUCKETS = [("static", 0.0, 0.3), ("slow", 0.3, 2.0), ("fast", 2.0, 1e9)]
LEAK_FRAMES = 25
REAL_SCORE_FRAMES = 100  # real clips are scored on their first 4 s, the span every model ran on
CROP = (1920, 1080)  # 4K outputs are scored on their central 1920x1080 at 1:1 pixels


def eval_geometry(tw, th, size=CROP):
    """(w, h, ffmpeg crop filter or None) used to read full-size frames for scoring."""
    cw, ch = size
    if tw <= cw and th <= ch:
        return tw, th, None
    return cw, ch, f"crop={cw}:{ch}:{(tw - cw) // 2}:{(th - ch) // 2}"


def source_size(clip_dir, name, target):
    """Frame size of an output file: its centre-crop size if it was run with one, else the target."""
    stats = clip_dir / "out" / f"{name}.json"
    if name != "gt" and stats.exists():
        d = json.loads(stats.read_text())
        if d.get("center_crop"):
            return tuple(d["out_size"])
    return tuple(target)


def region_filter(src_size, w, h):
    """ffmpeg filter cutting the centre w x h out of a frame of src_size (None if equal)."""
    sw, sh = src_size
    return None if (sw, sh) == (w, h) else f"crop={w}:{h}:{(sw - w) // 2}:{(sh - h) // 2}"


def luma(img):
    f = img.astype(np.float32)
    return 0.2126 * f[..., 0] + 0.7152 * f[..., 1] + 0.0722 * f[..., 2]


def grad_map(y):
    gx = cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3) / 8.0  # Sobel gain is 8 on a unit ramp
    gy = cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    return np.sqrt(gx * gx + gy * gy)


def grad_mag(y):
    return float(grad_map(y).mean())


def lr_speed(lr_gray, t, dis):
    """Median motion of pair (t-1, t) on the low-res source, in px/frame at 720p scale."""
    b = dis.calc(lr_gray[t], lr_gray[t - 1], None)
    return float(np.median(np.hypot(b[..., 0], b[..., 1]))) * (1280.0 / lr_gray[t].shape[1])


def self_flow(g_t, g_prev, dis):
    """Backward flow t→t-1 computed on the output itself (validated in validate_jitter.py:
    coherent sharp detail scores like GT, re-rolled texture scores ~2x higher) plus a
    forward-backward consistency mask for occlusions and out-of-frame pixels."""
    b = dis.calc(g_t, g_prev, None)
    f = dis.calc(g_prev, g_t, None)
    h, w = g_t.shape
    ly, lx = np.mgrid[0:h, 0:w].astype(np.float32)
    px, py = lx + b[..., 0], ly + b[..., 1]
    f_at = cv2.remap(f, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    err = np.sum((b + f_at) ** 2, axis=2)
    bound = 0.01 * (np.sum(b ** 2, axis=2) + np.sum(f_at ** 2, axis=2)) + 0.5
    inside = (px >= 0) & (px <= w - 1) & (py >= 0) & (py <= h - 1)
    return px, py, (err < bound) & inside


def warp(y, map_x, map_y):
    return cv2.remap(y, map_x, map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def bucket_of(speed):
    return next(name for name, lo, hi in BUCKETS if lo <= speed < hi)


class LPIPSScorer:
    def __init__(self):
        import lpips
        self.dev = torch.device("mps")
        self.net = lpips.LPIPS(net="alex", verbose=False).to(self.dev).eval()
        try:
            from DISTS_pytorch import DISTS
            self.dists = DISTS().to(self.dev).eval()
        except ImportError:
            self.dists = None

    def t(self, img):
        return torch.from_numpy(img).to(self.dev).permute(2, 0, 1)[None].float().div(255.0)

    @torch.inference_mode()
    def lpips(self, a, b):
        return float(self.net(self.t(a) * 2 - 1, self.t(b) * 2 - 1))

    @torch.inference_mode()
    def dist(self, a, b):
        return float(self.dists(self.t(a), self.t(b))) if self.dists else None


def evaluate(clip_dir, model, max_frames=None, region=None):
    clip = json.loads((clip_dir / "clip.json").read_text())
    lw, lh = clip["lr_size"]
    tw, th = clip["target"]
    lr = list(read_frames(clip_dir / "lr.mkv", lw, lh))
    stats_file = clip_dir / "out" / f"{model}.json"
    if stats_file.exists():  # slow models may have run on the first N frames only
        lr = lr[:json.loads(stats_file.read_text())["frames"]]
    if clip["kind"] == "real":
        lr = lr[:REAL_SCORE_FRAMES]
    if max_frames:
        lr = lr[:max_frames]
    n = len(lr)
    full_w, full_h = tw, th
    own = source_size(clip_dir, model, (full_w, full_h))
    tw, th, crop = eval_geometry(full_w, full_h, region or min(own, CROP))
    out_path = clip_dir / "out" / f"{model}.mkv"
    gt_path = clip_dir / "gt.mkv"
    has_gt = gt_path.exists() and model != "gt"
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    lr_gray = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in lr]
    shot_of = np.zeros(n, int)
    for k, (a, b) in enumerate(shots_of(clip)):
        shot_of[a:b] = k
    scorer = LPIPSScorer() if has_gt else None

    src = out_path if model != "gt" else gt_path
    def out_filter(name):
        return region_filter(source_size(clip_dir, name, (full_w, full_h)), tw, th)

    outs = read_frames(src, tw, th, out_filter(model) if model != "gt" else crop)
    gts = read_frames(gt_path, tw, th, crop) if has_gt else None

    per_bucket = {name: [] for name, _, _ in BUCKETS}
    jitter = {name: [] for name, _, _ in BUCKETS}
    detail, psnr, lp, ds, tlp = [], [], [], [], []
    prev = prev_bic = prev_gt = None
    for t in range(n):
        out = next(outs)
        y = luma(out)
        bic = cv2.resize(lr[t], (full_w, full_h), interpolation=cv2.INTER_CUBIC)
        if crop:
            x0, y0 = (full_w - tw) // 2, (full_h - th) // 2
            bic = bic[y0:y0 + th, x0:x0 + tw]
        bic = luma(bic)
        detail.append(grad_mag(y) / max(grad_mag(bic), 1e-6))
        gt = next(gts) if has_gt else None
        if has_gt:
            mse = float(np.mean((y - luma(gt)) ** 2))
            psnr.append(10 * np.log10(255.0 ** 2 / max(mse, 1e-10)))
            lp.append(scorer.lpips(out, gt))
            ds.append(scorer.dist(out, gt))
        if t > 0 and shot_of[t] == shot_of[t - 1]:
            mx, my, mask = self_flow(cv2.cvtColor(out, cv2.COLOR_RGB2GRAY),
                                     cv2.cvtColor(prev_out_img, cv2.COLOR_RGB2GRAY), dis)
            speed = lr_speed(lr_gray, t, dis)
            if mask.mean() > 0.2:
                d_model = y - warp(prev, mx, my)
                d_bic = bic - warp(prev_bic, mx, my)
                num = float(np.abs(d_model - d_bic)[mask].mean())
                per_bucket[bucket_of(speed)].append(num)
                jitter[bucket_of(speed)].append(num / max(float(grad_map(y)[mask].mean()), 1e-6))
            if has_gt:
                tlp.append(abs(scorer.lpips(out, prev_out_img) - scorer.lpips(gt, prev_gt)) * 100)
        prev, prev_bic, prev_out_img, prev_gt = y, bic, out, gt

    res = {"clip": clip["name"], "model": model, "frames": n, "scored_area": f"{tw}x{th}" + (" centre crop" if crop else ""),
           "detail": round(float(np.mean(detail)), 3),
           "flicker": {k: (round(float(np.mean(v)), 3) if v else None) for k, v in per_bucket.items()},
           "flicker_pairs": {k: len(v) for k, v in per_bucket.items()}}
    all_pairs = [x for v in per_bucket.values() for x in v]
    res["flicker_all"] = round(float(np.mean(all_pairs)), 3) if all_pairs else None
    res["jitter"] = {k: (round(float(np.mean(v)), 3) if v else None) for k, v in jitter.items()}
    all_j = [x for v in jitter.values() for x in v]
    res["jitter_all"] = round(float(np.mean(all_j)), 3) if all_j else None
    if has_gt:
        res.update(psnr_y=round(float(np.mean(psnr)), 2), lpips=round(float(np.mean(lp)), 4),
                   dists=round(float(np.mean(ds)), 4) if ds[0] is not None else None,
                   tlp=round(float(np.mean(tlp)), 3) if tlp else None)

    nocut = clip_dir / "out" / f"{model}-nocut.mkv"
    cuts = [c for c in clip["cuts"] if 0 < c < n]
    if nocut.exists() and cuts:
        a_frames = read_frames(out_path, tw, th, out_filter(model))
        b_frames = read_frames(nocut, tw, th, out_filter(f"{model}-nocut"))
        leak = []
        for t in range(n):
            a, b = next(a_frames), next(b_frames)
            if any(c <= t < c + LEAK_FRAMES for c in cuts):
                leak.append(float(np.abs(luma(a) - luma(b)).mean()))
        res["cut_leak"] = round(float(np.mean(leak)), 3)
        res["cut_leak_first"] = round(leak[0], 3)
    return res


def main():
    clip_dir, model = Path(sys.argv[1]), sys.argv[2]
    # ad-hoc: score only the first N frames [and only the centre WxH], print but don't save
    frames = int(sys.argv[3]) if len(sys.argv) > 3 else None
    region = tuple(int(v) for v in sys.argv[4].split("x")) if len(sys.argv) > 4 else None
    res = evaluate(clip_dir, model, frames, region)
    if frames is None:
        (clip_dir / "metrics").mkdir(exist_ok=True)
        (clip_dir / "metrics" / f"{model}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
