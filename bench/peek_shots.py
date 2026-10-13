"""Tile the middle frame of chosen shots (by shot number) for a quick look.

    python bench/peek_shots.py VIDEO OUT.jpg SHOT [SHOT ...]
"""
import csv
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

SHOTS = Path(__file__).resolve().parent.parent / "output" / "sr-bench" / "shots"


def main(video, out, *ids):
    rows = {r["shot"]: r for r in csv.DictReader(open(SHOTS / f"{Path(video).stem}.shots.csv"))}
    tiles = []
    for s in ids:
        r = rows[s]
        t = (float(r["start_s"]) + float(r["dur_s"]) / 2)
        png = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", video, "-frames:v", "1",
                              "-vf", "scale=640:360", "-f", "image2pipe", "-vcodec", "png", "-"],
                             capture_output=True, check=True).stdout
        img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
        label = f"#{s} @{r['start']} {float(r['dur_s']):.0f}s pan{r['pan']} loc{r['local']} tex{r['texture']}"
        cv2.rectangle(img, (0, 0), (640, 22), (0, 0, 0), -1)
        cv2.putText(img, label, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(img)
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])
    cv2.imwrite(out, sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])


if __name__ == "__main__":
    main(*sys.argv[1:])
