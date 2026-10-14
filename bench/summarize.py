"""Aggregate per-clip metrics into one scoreboard.

    python bench/summarize.py [OUT_JSON]

Per model:
  gt_*        means over ground-truth clips (270p → 1080p, scored against the real 1080p)
  gt3_lpips   mean LPIPS over the x3 ground-truth clips (g3-*: 360p → 1080p); the g3 clips are kept out
              of the x4 gt_* means above
  jitter_vs_gt   model jitter ÷ the real footage's own jitter on the same clips (1.0 = as
                 steady as real video; ~2 = the synthetic re-rolled-texture case)
  real_*      means over real clips (720p/1080p → 4K, scored on the central 1920x1080)
  pulse       mean over the real clips that have it of max ÷ mean of the per-phase jitter (t mod 4);
              ~1 = steady, higher = the jitter breathes with a 4-frame period
  jitter_vs_apple, detail_vs_apple   ratios to Apple AI on the same real clips
  cut_leak    how far the cut-ignoring run differs in the 25 frames after a cut (luma levels)
  hours_per_episode   s/frame on 720p → 4K clips (scaled up when only a crop was run)
                      × 86,877 frames (58 min at 25 fps)
"""
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

CLIPS = Path(__file__).resolve().parent.parent / "output" / "sr-bench" / "clips"
EPISODE_FRAMES = 86877


def load(pattern):
    out = defaultdict(dict)
    for f in glob.glob(str(CLIPS / "*" / pattern)):
        d = json.loads(Path(f).read_text())
        out[d["model"] if "model" in d else Path(f).stem][Path(f).parent.parent.name] = d
    return out


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(float(np.mean(xs)), 3) if xs else None


def main(out_json=None):
    metrics = load("metrics/*.json")
    runs = load("out/*.json")
    clips = {p.name: json.loads((p / "clip.json").read_text()) for p in CLIPS.iterdir() if (p / "clip.json").exists()}
    gt_clips = [c for c, d in clips.items() if d["kind"] == "gt"]
    gt3_clips = [c for c, d in clips.items() if d["kind"] == "gt3"]
    real_clips = [c for c, d in clips.items() if d["kind"] == "real"]
    apple = metrics.get("apple", {})
    gt_self = metrics.get("gt", {})

    board = {}
    for model, per_clip in metrics.items():
        if model == "gt":
            continue
        row = {"clips_scored": len(per_clip)}
        g = [c for c in gt_clips if c in per_clip]
        row["gt_lpips"] = mean(per_clip[c].get("lpips") for c in g)
        row["gt_dists"] = mean(per_clip[c].get("dists") for c in g)
        row["gt_psnr"] = mean(per_clip[c].get("psnr_y") for c in g)
        row["gt_tlp"] = mean(per_clip[c].get("tlp") for c in g)
        row["gt3_lpips"] = mean(per_clip[c].get("lpips") for c in gt3_clips if c in per_clip)
        row["jitter_vs_gt"] = mean(per_clip[c]["jitter_all"] / gt_self[c]["jitter_all"]
                                   for c in g if c in gt_self and gt_self[c]["jitter_all"])
        r = [c for c in real_clips if c in per_clip]
        row["pulse"] = mean(per_clip[c].get("pulse") for c in r)
        row["jitter_vs_apple"] = mean(per_clip[c]["jitter_all"] / apple[c]["jitter_all"]
                                      for c in r if c in apple and apple[c]["jitter_all"])
        row["detail_vs_apple"] = mean(per_clip[c]["detail"] / apple[c]["detail"] for c in r if c in apple)
        row["jitter_by_motion"] = {b: mean(per_clip[c]["jitter"][b] / apple[c]["jitter"][b]
                                           for c in r if c in apple and per_clip[c]["jitter"][b] and apple[c]["jitter"][b])
                                   for b in ("static", "slow", "fast")}
        leaks = [per_clip[c]["cut_leak"] for c in per_clip if "cut_leak" in per_clip[c]]
        row["cut_leak"] = mean(leaks)
        speeds = []
        for c in real_clips:
            run = runs.get(model, {}).get(c)
            if run and clips[c]["lr_size"] == [1280, 720]:
                full = 1280 * 720 * 16  # pixels of a full 4x output frame from 720p
                speeds.append(run["s_per_frame"] * full / run["frame_pixels"] if run.get("center_crop") else run["s_per_frame"])
        row["s_per_frame_720p_to_4k"] = mean(speeds)
        row["hours_per_episode"] = round(row["s_per_frame_720p_to_4k"] * EPISODE_FRAMES / 3600, 1) if speeds else None
        row["peak_gpu_gb"] = max((x.get("peak_gpu_gb") or 0) for x in runs.get(model, {}).values()) or None
        board[model] = row

    order = sorted(board, key=lambda m: (board[m]["gt_lpips"] is None, board[m]["gt_lpips"] or 9,
                                          board[m]["gt3_lpips"] or 9))
    print(f"{'model':20s} {'LPIPS':>6s} {'LPIPS3':>6s} {'DISTS':>6s} {'tLP':>5s} {'jit/GT':>6s} {'jit/Apl':>7s} {'det/Apl':>7s} "
          f"{'pulse':>5s} {'leak':>5s} {'h/ep':>6s}")
    fmt = lambda v, w, p=3: f"{v:{w}.{p}f}" if v is not None else " " * (w - 1) + "-"
    for m in order:
        b = board[m]
        print(f"{m:20s} {fmt(b['gt_lpips'], 6)} {fmt(b['gt3_lpips'], 6)} {fmt(b['gt_dists'], 6)} {fmt(b['gt_tlp'], 5, 2)} "
              f"{fmt(b['jitter_vs_gt'], 6, 2)} {fmt(b['jitter_vs_apple'], 7, 2)} {fmt(b['detail_vs_apple'], 7, 2)} "
              f"{fmt(b['pulse'], 5, 2)} {fmt(b['cut_leak'], 5, 2)} {fmt(b['hours_per_episode'], 6, 1)}")
    if out_json:
        Path(out_json).write_text(json.dumps({"models": board, "clips": clips}, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:2])
