"""Cut benchmark clips out of the episodes, frame-exact, with their scene cuts.

    python bench/make_clips.py bench/clips.json OUT_DIR

clips.json: [{"name", "source", "start" (frame), "frames", "kind": "real"|"gt"|"gt3", "why"}]

real  lr.mkv = the source frames as decoded; target = 4K (3840x2160).
gt    gt.mkv = the 1080p source frames (ground truth); lr.mkv = 480x270, area-downscaled
      and re-compressed with x264 at the same bits/pixel as the 720p broadcast episodes
      (~0.1 bpp), so models see broadcast-like blocking/ringing; target = 1920x1080.
gt3   the same as gt with a 640x360 low-res input, for the 3x models (DetailNet's x3 on top of Apple AI's
      base); target = 1920x1080. Kept apart from gt (4x) in the scoreboard.

Cuts inside each clip come from the episode's shot map (bench/analyze_shots.py) and are
re-checked on the extracted frames.
"""
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clipio import read_frames  # noqa: E402

SHOTS = Path(__file__).resolve().parent.parent / "output" / "sr-bench" / "shots"
GT_LR = (480, 270)
GT3_LR = (640, 360)
GT_LR_SIZES = {"gt": GT_LR, "gt3": GT3_LR}
BPP = 0.1


def gt_lr_size(kind):
    """Low-res size of a ground-truth clip kind."""
    if kind not in GT_LR_SIZES:
        raise ValueError(f"unknown clip kind {kind!r}; expected real or one of {sorted(GT_LR_SIZES)}")
    return GT_LR_SIZES[kind]


def probe(path):
    out = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                   "stream=width,height,r_frame_rate", "-of", "csv=p=0", str(path)], text=True)
    w, h, r = out.strip().split(",")
    num, den = r.split("/")
    return int(w), int(h), float(num) / float(den)


def episode_cuts(source):
    rows = csv.DictReader(open(SHOTS / f"{Path(source).stem}.frames.csv"))
    return [int(r["frame"]) for r in rows if r["cut"] == "1"]


def extract(source, start, frames, fps, dst, vf=None):
    cmd = ["ffmpeg", "-v", "error", "-y", "-ss", f"{start / fps:.6f}", "-i", str(source),
           "-frames:v", str(frames), "-an"]
    if vf:
        cmd += ["-vf", vf]
    cmd += ["-c:v", "ffv1", "-level", "3", "-slices", "16", "-pix_fmt", "bgr0", str(dst)]
    subprocess.run(cmd, check=True)


def check_cuts(path, w, h, cuts):
    """Every listed cut should be the biggest frame difference in its ±2 neighbourhood."""
    prev, diffs = None, []
    for f in read_frames(path, w, h):
        g = f.astype(np.int16)
        diffs.append(0.0 if prev is None else float(np.abs(g - prev).mean()))
        prev = g
    for c in cuts:
        window = diffs[max(c - 2, 1):c + 3]
        assert diffs[c] == max(window), f"{path}: cut {c} not at the diff peak {window}"


def main(spec_path, out_dir):
    out_dir = Path(out_dir)
    for spec in json.loads(Path(spec_path).read_text()):
        if spec["kind"] != "real":
            gt_lr_size(spec["kind"])  # rejects an unknown kind before anything is extracted
        clip_dir = out_dir / spec["name"]
        clip_dir.mkdir(parents=True, exist_ok=True)
        src = Path(spec["source"]).expanduser()
        w, h, fps = probe(src)
        a, n = spec["start"], spec["frames"]
        cuts = [c - a for c in episode_cuts(src) if a < c < a + n]
        clip = {"name": spec["name"], "kind": spec["kind"], "why": spec.get("why", ""),
                "source": str(src), "start": a, "frames": n, "fps": fps, "cuts": cuts}
        if spec["kind"] == "real":
            extract(src, a, n, fps, clip_dir / "lr.mkv")
            clip.update(lr_size=[w, h], target=[3840, 2160])
            check_cuts(clip_dir / "lr.mkv", w, h, cuts)
        else:
            assert (w, h) == (1920, 1080), "gt clips need a 1080p source"
            extract(src, a, n, fps, clip_dir / "gt.mkv")
            lw, lh = gt_lr_size(spec["kind"])
            kbps = round(lw * lh * fps * BPP / 1000)
            tmp = clip_dir / "lr-h264.mp4"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(clip_dir / "gt.mkv"),
                            "-vf", f"scale={lw}:{lh}:flags=area", "-c:v", "libx264", "-preset", "medium",
                            "-b:v", f"{kbps}k", "-maxrate", f"{kbps * 2}k", "-bufsize", f"{kbps * 2}k",
                            "-pix_fmt", "yuv420p", str(tmp)], check=True)
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(tmp), "-c:v", "ffv1", "-level", "3",
                            "-pix_fmt", "bgr0", str(clip_dir / "lr.mkv")], check=True)
            tmp.unlink()
            clip.update(lr_size=[lw, lh], target=[1920, 1080], lr_kbps=kbps)
            check_cuts(clip_dir / "gt.mkv", w, h, cuts)
        (clip_dir / "clip.json").write_text(json.dumps(clip, indent=1))
        print(f"{spec['name']}: {n} frames, cuts {cuts}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
