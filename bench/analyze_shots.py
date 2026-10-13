"""Shot map for a video: hard cuts, per-frame motion and texture, per-shot summary.

Decodes a 640x360 proxy with ffmpeg and, for every frame, records
  cut       AdaptiveDetector (PySceneDetect) hard-cut decision
  content   its HSV content score (high on cuts and dissolves)
  pan       |median optical flow|            camera motion, px/frame at 720p scale
  local     mean |flow - median flow|        object motion (water, animals, foliage)
  texture   Laplacian variance at 640x360    how much fine detail there is
  luma      mean luma (0-255)

Writes <out>/<name>.frames.csv, <name>.shots.csv and <name>.sheet.jpg (one
thumbnail per shot, labelled) for picking benchmark clips by eye.

usage: python bench/analyze_shots.py VIDEO OUT_DIR
"""
import csv
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from scenedetect.common import FrameTimecode
from scenedetect.detectors import AdaptiveDetector

W, H = 640, 360
FLOW_W, FLOW_H = 320, 180
TO_720P = 1280 / FLOW_W  # flow is measured at 320 wide; report it at 720p scale


def probe_fps(path):
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=r_frame_rate", "-of", "csv=p=0", path], text=True).strip()
    num, den = out.split("/")
    return float(num) / float(den)


def frames(path):
    cmd = ["ffmpeg", "-v", "error", "-threads", "4", "-i", path, "-an",
           "-vf", f"scale={W}:{H}:flags=area", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=W * H * 3 * 8)
    size = W * H * 3
    while True:
        buf = proc.stdout.read(size)
        if len(buf) < size:
            break
        yield np.frombuffer(buf, np.uint8).reshape(H, W, 3)
    proc.wait()


def main(video, out_dir):
    video, out_dir = Path(video), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = video.stem
    fps = probe_fps(str(video))
    detector = AdaptiveDetector()
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)

    rows, cuts, thumbs = [], {0}, {}
    pending = {}  # frame -> thumbnail since the last cut; shot middles move to `thumbs`
    last_cut = 0

    def close_shot(end):
        nonlocal last_cut, pending
        mid = (last_cut + end) // 2
        if mid in pending:
            thumbs[mid] = pending[mid]
        pending = {k: v for k, v in pending.items() if k >= end}
        last_cut = end

    prev_gray = None
    for i, bgr in enumerate(frames(str(video))):
        small = cv2.resize(bgr, (FLOW_W, FLOW_H), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        pending[i] = small
        for tc in detector.process_frame(FrameTimecode(i, fps=fps), small):
            cuts.add(tc.frame_num)
            close_shot(tc.frame_num)
        pan = local = 0.0
        if prev_gray is not None:
            flow = dis.calc(prev_gray, gray, None)
            med = np.median(flow.reshape(-1, 2), axis=0)
            pan = float(np.hypot(*med)) * TO_720P
            local = float(np.hypot(*(flow - med).reshape(-1, 2).T).mean()) * TO_720P
        prev_gray = gray
        full_gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        texture = float(cv2.Laplacian(full_gray, cv2.CV_32F).var())
        rows.append({"frame": i, "content": round(detector._frame_score or 0.0, 2),
                     "pan": round(pan, 3), "local": round(local, 3),
                     "texture": round(texture, 1), "luma": round(float(full_gray.mean()), 1)})
        if i % 5000 == 0:
            print(f"{name}: frame {i}", flush=True)
    for tc in detector.post_process(FrameTimecode(len(rows), fps=fps)):
        cuts.add(tc.frame_num)
        close_shot(tc.frame_num)
    close_shot(len(rows))

    cuts = sorted(c for c in cuts if c < len(rows))
    for r in rows:
        r["cut"] = int(r["frame"] in cuts and r["frame"] > 0)
    with open(out_dir / f"{name}.frames.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["frame", "cut", "content", "pan", "local", "texture", "luma"])
        w.writeheader()
        w.writerows(rows)

    bounds = cuts + [len(rows)]
    shots = []
    for s, (a, b) in enumerate(zip(bounds, bounds[1:])):
        seg = rows[a + 1:b] if b - a > 1 else rows[a:b]  # skip the cut frame's own flow
        shots.append({
            "shot": s, "start": a, "end": b, "frames": b - a,
            "start_s": round(a / fps, 2), "dur_s": round((b - a) / fps, 2),
            "pan": round(float(np.median([r["pan"] for r in seg])), 2),
            "local": round(float(np.median([r["local"] for r in seg])), 2),
            "texture": round(float(np.median([r["texture"] for r in seg])), 0),
            "luma": round(float(np.mean([r["luma"] for r in seg])), 0),
            "max_content": round(max(r["content"] for r in seg), 1),
        })
    with open(out_dir / f"{name}.shots.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(shots[0]))
        w.writeheader()
        w.writerows(shots)

    # Contact sheet: middle frame of each shot, 12 per row, labelled.
    tiles = []
    for sh in shots:
        mid = (sh["start"] + sh["end"]) // 2
        t = thumbs[mid].copy() if mid in thumbs else np.zeros((FLOW_H, FLOW_W, 3), np.uint8)
        label = f"#{sh['shot']} {sh['start_s']:.0f}s {sh['dur_s']:.0f}s p{sh['pan']:.1f} l{sh['local']:.1f} t{sh['texture']:.0f}"
        cv2.rectangle(t, (0, 0), (FLOW_W, 16), (0, 0, 0), -1)
        cv2.putText(t, label, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(t)
    per_row = 12
    while len(tiles) % per_row:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[i:i + per_row]) for i in range(0, len(tiles), per_row)])
    cv2.imwrite(str(out_dir / f"{name}.sheet.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 80])
    print(f"{name}: {len(rows)} frames, {len(shots)} shots, sheet {sheet.shape[1]}x{sheet.shape[0]}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
