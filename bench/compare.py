"""Side-by-side comparison video and time-slice image for one clip.

    python bench/compare.py CLIP_DIR OUT_DIR MODEL [MODEL ...]

Picks the most detailed 960x540 window of the scored area (the 4K centre crop, or the
whole 1080p frame on ground-truth clips) and tiles every model's output there at 1:1
pixels, labelled, as an HEVC video (plays in QuickTime). The time-slice image stacks one
pixel row of that window over time: steady texture draws clean vertical or diagonal
streaks; texture that re-rolls frame to frame turns the streaks into noise.
"""
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clipio import read_frames  # noqa: E402
from metrics import CROP, grad_map, luma, region_filter, source_size  # noqa: E402

TILE_W, TILE_H = 960, 540
LABELS = {"flashvsr-c960": "FlashVSR v1.1", "gt": "Real 1080p (ground truth)", "bicubic": "Bicubic", "apple": "Apple AI",
          "realesrgan": "Real-ESRGAN x4plus", "realesrgan-general": "Real-ESRGAN general",
          "realbasicvsr": "RealBasicVSR", "realviformer": "RealViformer", "flashvsr": "FlashVSR v1.1"}


def model_frames(clip_dir, model, w, h, target, n):
    src = clip_dir / ("gt.mkv" if model == "gt" else f"out/{model}.mkv")
    filt = region_filter(source_size(clip_dir, model, target), w, h)
    return [f for f, _ in zip(read_frames(src, w, h, filt), range(n))]


def best_window(img):
    g = grad_map(luma(img))
    integral = cv2.integral(g)
    h, w = g.shape
    best, pos = -1.0, (0, 0)
    for y in range(0, h - TILE_H + 1, 30):
        for x in range(0, w - TILE_W + 1, 30):
            s = integral[y + TILE_H, x + TILE_W] - integral[y, x + TILE_W] - integral[y + TILE_H, x] + integral[y, x]
            if s > best:
                best, pos = s, (x, y)
    return pos


def label(tile, text):
    tile = tile.copy()
    cv2.rectangle(tile, (0, 0), (len(text) * 13 + 16, 34), (0, 0, 0), -1)
    cv2.putText(tile, text, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    return tile


def main(clip_dir, out_dir, *models):
    clip_dir, out_dir = Path(clip_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    clip = json.loads((clip_dir / "clip.json").read_text())
    target = tuple(clip["target"])
    w, h = min([source_size(clip_dir, m, target) for m in models] + [CROP])
    w, h = min(w, target[0]), min(h, target[1])
    n = min(json.loads((clip_dir / "out" / f"{m}.json").read_text())["frames"] for m in models if m != "gt")
    seqs = {m: model_frames(clip_dir, m, w, h, target, n) for m in models}
    x0, y0 = best_window(seqs[models[0]][n // 2]) if (w, h) != (TILE_W, TILE_H) else (0, 0)

    cols = 4 if len(models) > 6 else 3 if len(models) > 4 else 2
    rows = -(-len(models) // cols)
    vw, vh = cols * TILE_W, rows * TILE_H
    dst = out_dir / f"{clip['name']}.mp4"
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{vw}x{vh}",
                            "-r", str(clip["fps"]), "-i", "-", "-c:v", "libx265", "-crf", "12", "-preset", "fast",
                            "-tag:v", "hvc1", "-pix_fmt", "yuv420p", "-x265-params", "log-level=error", str(dst)],
                           stdin=subprocess.PIPE)
    blank = np.zeros((TILE_H, TILE_W, 3), np.uint8)
    for t in range(n):
        tiles = [label(seqs[m][t][y0:y0 + TILE_H, x0:x0 + TILE_W], LABELS.get(m, m)) for m in models]
        tiles += [blank] * (rows * cols - len(tiles))
        grid = np.vstack([np.hstack(tiles[r * cols:(r + 1) * cols]) for r in range(rows)])
        enc.stdin.write(grid.tobytes())
    enc.stdin.close()
    enc.wait()

    # Time slice: the middle row of the window over time, 2 px per frame, 480 px wide.
    row, sx = y0 + TILE_H // 2, x0 + (TILE_W - 480) // 2
    strips = []
    for m in models:
        s = np.stack([seqs[m][t][row, sx:sx + 480] for t in range(n)])
        s = np.repeat(s, 2, axis=0)
        strips.append(label(s, LABELS.get(m, m).split(" (")[0]))
        strips.append(np.full((s.shape[0], 8, 3), 255, np.uint8))
    cv2.imwrite(str(out_dir / f"{clip['name']}-timeslice.png"), cv2.cvtColor(np.hstack(strips[:-1]), cv2.COLOR_RGB2BGR))
    print(f"{dst}  window ({x0},{y0}) of the {w}x{h} scored area, {n} frames")


if __name__ == "__main__":
    main(*sys.argv[1:])
