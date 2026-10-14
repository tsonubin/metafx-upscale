"""Frame-alignment check of a built sequence against ffmpeg (plan M1 Review Focus 2).

    python detailnet/tests/check_alignment.py output/detail-data/teacher/00002
    python detailnet/tests/check_alignment.py output/detail-data/gt/00000

Reads the sequence's spec.json and decodes the same episode frames with ffmpeg (shot-map index i = the frame at
i / 25 s; bt709, limited range -> full-range RGB, same crop; passthrough, so ffmpeg's CFR sync cannot duplicate or
drop a frame). Then it compares dump frame i with reference frame
i + offset for offsets -1, 0, +1 and prints the mean |delta| in 8-bit levels, averaged over the frames. The sequence
passes when offset 0 fits (every frame's mean |delta| < MAX_DELTA) and both neighbouring offsets fit clearly worse
(by at least MIN_GAP levels on average), so pick a sequence with motion; a static shot cannot tell offsets apart.

  teacher  source/ against the episode frames, cropped to the dumped window (decoder differences only: ~1 level)
  gt       target/ against the 1080p episode frames (same), then source/ (the 640x360 x264 copy) against the target
           area-downscaled 3x3 (compression noise raises the absolute level, so only the offset gap is gated there)

It is a script, not a pytest test (the `check_` prefix keeps pytest from collecting it); the gating logic is tested
in test_check_alignment.py.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from detailnet.build_data import DEFAULT_EPISODES, MARGIN, SCALE, ffmpeg_seek, probe_episode  # noqa: E402
from detailnet.dumps import read_frame, read_meta  # noqa: E402

OFFSETS = (-1, 0, 1)
MAX_DELTA = 2.0   # offset 0, every frame: mean |delta| in levels
MIN_GAP = 0.1     # the neighbouring offsets must be at least this much worse on average


def dump_frames(dump_dir):
    """Every frame of a dump as float32 H x W x 3, 0..255."""
    n = read_meta(dump_dir)["frames"]
    return [read_frame(dump_dir, i).astype(np.float32).transpose(1, 2, 0) * 255.0 for i in range(n)]


def reference_frames(episode, first, count, window):
    """`count` frames of the episode from shot-map frame `first`, float32 H x W x 3 cropped to window=(x, y, w, h)."""
    x, y, w, h = window
    ep = probe_episode(episode)
    vf = f"scale=in_range=tv:out_range=pc:in_color_matrix=bt709,format=rgb24,crop={w}:{h}:{x}:{y}"
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{ffmpeg_seek(first, ep.fps):.6f}", "-i", str(episode), "-an",
           "-fps_mode", "passthrough", "-frames:v", str(count), "-vf", vf, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    frames = np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3).astype(np.float32)
    if len(frames) != count:
        raise SystemExit(f"ffmpeg returned {len(frames)} reference frames, expected {count}")
    return list(frames)


def area_down(frame, s):
    """Mean over each s x s block of an H x W x 3 array."""
    h, w, c = frame.shape
    return frame[:h // s * s, :w // s * s].reshape(h // s, s, w // s, s, c).mean(axis=(1, 3))


def offset_table(dump, reference, lead=0, offsets=OFFSETS):
    """{offset: [mean |dump[i] - reference[i + lead + offset]| for each i where both exist]}. `lead` is how many
    frames `reference` holds before the sequence's first frame (the episode references start one frame early)."""
    table = {}
    for off in offsets:
        pairs = [(i, i + lead + off) for i in range(len(dump)) if 0 <= i + lead + off < len(reference)]
        table[off] = [float(np.abs(dump[i] - reference[j]).mean()) for i, j in pairs]
    return table


def verdict(table, max_delta=MAX_DELTA, min_gap=MIN_GAP):
    """Messages for each way `table` (from offset_table) fails; empty when it passes. max_delta=None skips the
    absolute gate."""
    failed = []
    own = table[0]
    if max_delta is not None and max(own) >= max_delta:
        failed.append(f"offset 0 has a frame with mean |delta| {max(own):.2f} >= {max_delta}")
    for off, deltas in table.items():
        if off != 0 and np.mean(deltas) < np.mean(own) + min_gap:
            failed.append(f"offset {off:+d} ({np.mean(deltas):.2f}) is not clearly worse than offset 0 "
                          f"({np.mean(own):.2f}); static content or misaligned frames")
    return failed


def report(name, table, max_delta):
    print(f"{name}:")
    for off, deltas in table.items():
        print(f"  offset {off:+d}: mean |delta| {np.mean(deltas):6.2f}   (per frame max {max(deltas):6.2f}, "
              f"over {len(deltas)} frames)")
    failed = verdict(table, max_delta)
    print("  " + ("PASS" if not failed else "FAIL: " + "; ".join(failed)))
    return not failed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("sequence", help="a sequence directory with spec.json (teacher/00002, gt/00000, ...)")
    parser.add_argument("--episodes", default=str(DEFAULT_EPISODES), help="folder with the episode .mp4 files")
    args = parser.parse_args(argv)

    seq = Path(args.sequence)
    spec = json.loads((seq / "spec.json").read_text())
    n = spec["burn_in"] + spec["frames"]
    first = spec["start"] - spec["burn_in"]
    episode = Path(args.episodes) / f"{spec['episode']}.mp4"
    print(f"{seq}: {spec['kind']} {spec['bucket']} sequence, frames {first}..{first + n - 1} of {spec['episode']}")

    ok = True
    if spec["kind"] == "teacher":
        window = tuple(read_meta(seq / "source")["crop"])
        ref = reference_frames(episode, first - 1, n + 2, window)
        ok &= report("source/ vs episode", offset_table(dump_frames(seq / "source"), ref, lead=1), MAX_DELTA)
    else:
        window = tuple(read_meta(seq / "target")["crop"])
        ref = reference_frames(episode, first - 1, n + 2, window)
        target = dump_frames(seq / "target")
        ok &= report("target/ vs episode (1080p)", offset_table(target, ref, lead=1), MAX_DELTA)
        h, w = window[3] // SCALE, window[2] // SCALE
        source = [f[MARGIN:MARGIN + h, MARGIN:MARGIN + w] for f in dump_frames(seq / "source")]
        small = [area_down(f, SCALE) for f in target]
        ok &= report("source/ (360p x264) vs target/ area-downscaled", offset_table(source, small), None)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
