"""Frame I/O shared by the benchmark scripts: exact RGB frames via ffmpeg pipes,
lossless FFV1 writing, and splitting a clip into shots at its scene cuts."""
import subprocess

import numpy as np


def read_frames(path, w, h, vf=None):
    """Yield HxWx3 uint8 RGB frames of `path` (scaled to w x h only if vf says so)."""
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-an"]
    if vf:
        cmd += ["-vf", vf]
    cmd += ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=w * h * 3 * 4)
    size = w * h * 3
    try:
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    finally:
        proc.stdout.close()
        proc.wait()


class FFV1Writer:
    """Lossless 8-bit RGB video (FFV1 in mkv)."""

    def __init__(self, path, w, h, fps):
        self.proc = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
             "-r", str(fps), "-i", "-", "-c:v", "ffv1", "-level", "3", "-slices", "24", "-threads", "8",
             "-pix_fmt", "bgr0", str(path)], stdin=subprocess.PIPE)

    def write(self, img):
        self.proc.stdin.write(np.ascontiguousarray(img).tobytes())

    def close(self):
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise RuntimeError("ffmpeg FFV1 writer failed")


def shots_of(clip, ignore_cuts=False):
    """[(start, end)] frame ranges of the clip's shots."""
    n = clip["frames"]
    cuts = [] if ignore_cuts else [c for c in clip["cuts"] if 0 < c < n]
    bounds = [0] + cuts + [n]
    return list(zip(bounds, bounds[1:]))
