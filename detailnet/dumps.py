"""Frame dumps shared with the Swift CLI: one raw little-endian float16 CHW file per
frame (`000000.f16`, ...) plus a `meta.json` describing them."""
import json
from pathlib import Path

import numpy as np


def read_meta(d):
    return json.loads((Path(d) / "meta.json").read_text())


def read_frame(d, i):
    """Frame i as float16, C x H x W."""
    meta = read_meta(d)
    shape = (meta["channels"], meta["height"], meta["width"])
    data = np.fromfile(Path(d) / f"{i:06d}.f16", dtype="<f2")
    return data.reshape(shape).astype(np.float16, copy=False)


def _axis_taps(n_in, n_out):
    """Bilinear taps (low index, high index, high weight) with half-pixel centres, edges clamped."""
    src = np.clip((np.arange(n_out) + 0.5) * n_in / n_out - 0.5, 0, n_in - 1)
    lo = np.floor(src).astype(int)
    hi = np.minimum(lo + 1, n_in - 1)
    return lo, hi, (src - lo).astype(np.float32)


def read_flow(d, i, size=None):
    """Flow frame i as float32, 4 x H x W (back_x, back_y, fwd_x, fwd_y).

    With size=(w, h) the flow is bilinearly resized and its values are rescaled to
    pixels of that grid: x by w / width, y by h / height."""
    meta = read_meta(d)
    if meta["kind"] != "flow" or meta["channels"] != 4:
        raise ValueError(f"{d} is not a flow dump")
    flow = read_frame(d, i).astype(np.float32)
    if size is None:
        return flow
    w, h = size
    y0, y1, wy = _axis_taps(meta["height"], h)
    x0, x1, wx = _axis_taps(meta["width"], w)
    rows = flow[:, y0, :] + (flow[:, y1, :] - flow[:, y0, :]) * wy[None, :, None]
    out = rows[:, :, x0] + (rows[:, :, x1] - rows[:, :, x0]) * wx[None, None, :]
    out[0::2] *= w / meta["width"]
    out[1::2] *= h / meta["height"]
    return out


class DumpWriter:
    """Writes the same layout as the Swift FrameDumpWriter. `extra` keys are added to
    meta.json but never replace the standard ones."""

    def __init__(self, d, kind, channels, width, height, source_size, scale, crop, extra=None):
        self.d = Path(d)
        self.d.mkdir(parents=True, exist_ok=True)
        self.shape = (channels, height, width)
        self.header = {
            **(extra or {}),
            "kind": kind, "channels": channels, "width": width, "height": height,
            "layout": "CHW", "dtype": "float16",
            "source_size": list(source_size), "scale": scale, "crop": list(crop),
        }
        self.pts = []
        self.valid = []

    def write(self, chw, pts, valid=True):
        chw = np.asarray(chw)
        if chw.shape != self.shape:
            raise ValueError(f"frame has shape {chw.shape}, expected {self.shape}")
        np.ascontiguousarray(chw, dtype="<f2").tofile(self.d / f"{len(self.pts):06d}.f16")
        self.pts.append(float(pts))
        self.valid.append(bool(valid))

    def finish(self):
        meta = {**self.header, "frames": len(self.pts), "pts": self.pts, "valid": self.valid}
        (self.d / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
