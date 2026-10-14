"""DetailNet training data: fixed-size, fixed-length samples cut from the per-sequence dumps of `build_data` (and
`teacher` for the targets).

    ds = SequenceDataset("output/detail-data", "teacher")        # or "gt"; ping_pong=True for the 39-frame variant
    item = ds[i]

One item is a random sub-crop of one sequence, a stack of T frames the recurrent network runs through in order:

    x        (T, 3, h, w)       source frames (h, w = crop, default 90 x 160)
    base     (T, 3, 3h, 3w)     Apple AI's x3 frames of the same crop
    flow     (T, 4, h, w)       [back_x, back_y, fwd_x, fwd_y] in source px of the crop grid, clamped to +-64
    target   (T, 3, 3h, 3w)     teacher (FlashVSR) or ground-truth frames of the same crop
    reset    (T,) bool          the network's state starts from zero at this frame
    loss_mask(T,) bool          the frame is scored (False for burn-in and padding, and in teacher items for the
                                first frames from each cut, see below)
    kind     "teacher" | "gt";  seq_id  the sequence's id in sequences.json (for debugging)

Time layout. T = 8 burn-in slots + 12 loss frames = 20 (39 with ping-pong), the same for every item so batches
collate. A 12-frame sequence is used whole; a 24-frame one gives a random 12-frame loss window inside its 24, and the
8 dumped frames before the window (burn-in or loss frames of the sequence alike) are the item's burn-in. Where fewer
than 8 real frames precede the window (burn_in < 8, or a window near the start) the front is padded with copies of the
first real frame: pads have loss_mask False, zero flow and reset True.

`reset` is True at: the pads and the first real frame (the previous frame is not part of the item), every cut in the
sequence's spec, every frame whose flow is marked invalid, and, with probability `reset_prob` per item, one random
loss frame (so the network also learns to start cold mid-clip). Flow across a hard cut is *not* marked invalid by the
dumps; the spec's cuts are what makes it reset there.

Teacher frames after a cut (ruling R34). FlashVSR made a straddle sequence's teacher targets as one shot across its cut,
so they lag behind the new shot for a few frames. In teacher items the dumped frames c, c+1, c+2, c+3 of every cut c in
the spec are therefore not scored (loss_mask False) wherever they fall in the item, which may cut the four frames short
at the item's edges. gt items are unaffected (their targets are the real frames). Resets are unchanged.

Ping-pong appends the item's frames in reverse, without repeating the last one: slots 0..19 then 18..0. The reversed
frame that is original frame t-1, following original frame t, takes the flow stored at original index t with its
channels swapped (back <- fwd_t, since fwd_t is the displacement t-1 -> t on frame t-1's grid, which is exactly "back"
for a sequence that visits t first; fwd <- back_t), resets when original t is a cut or has invalid flow, and keeps the
loss_mask of original frame t-1 (so R34's unscored frames are mirrored). Pads keep zero flow and reset. The random reset
is drawn last, among the scored frames of the whole item.

Randomness: every item draws from its own `np.random.default_rng(seed + index)` (crop position, loss window, random
reset), so an item is a pure function of (seed, index) and nothing touches a global generator. Because of that an item
does not change from epoch to epoch; vary `seed` (it is a plain attribute) to get new crops of the same sequences.

Only finished sequences are used, decided once at construction: the sequences.json row must match a finished build
(`build_data.is_complete`) and the target dump must hold burn_in + frames frames (`teacher.is_done` for teacher; the
`target/` dump for gt). Sequences still being built or teacher-labelled are left out; a sequence that completes later
joins the next dataset built.

Reads fetch only the rows a crop needs (a dump frame is CHW float16, one contiguous band per channel), which matters
for `base` at 5 MB a frame.
"""
import json
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch

from detailnet.build_data import is_complete
from detailnet.dumps import read_meta
from detailnet.sampling import CROP_MARGIN
from detailnet.teacher import is_done

KINDS = ("teacher", "gt")
SCALE = 3  # source px -> base / target px
MARGIN = CROP_MARGIN  # source px of context around the central crop in the dumped (margin-expanded) frames
BURN_IN = 8  # burn-in slots in every item
LOSS = 12  # scored frames in every item
FLOW_LIMIT = 64.0  # flow is clamped to +-this many source px
CUT_MASK_FRAMES = 4  # R34: teacher frames not scored from each cut on (the cut frame and the 3 after it)
TARGET_DIR = {"teacher": "teacher", "gt": "target"}


class _Sequence(NamedTuple):
    id: int
    dir: str
    burn_in: int
    frames: int  # loss frames; the dumps hold burn_in + frames
    cuts: frozenset  # dumped frames at which a new shot starts
    valid: tuple  # per dumped frame: the flow of that frame can be trusted
    width: int  # dumped (margin-expanded) source size, px
    height: int


def read_window(path, shape, rows, cols):
    """float16 (C, rows, cols) window of the CHW float16 dump file `path` whose full shape is `shape`; `rows` and
    `cols` are (start, stop) pairs. Only the band of rows is read from disk (the columns are cut in memory)."""
    channels, height, width = shape
    (y0, y1), (x0, x1) = rows, cols
    out = np.empty((channels, y1 - y0, x1 - x0), np.float16)
    count = (y1 - y0) * width
    with open(path, "rb") as f:
        for c in range(channels):
            f.seek((c * height + y0) * width * 2)
            data = f.read(count * 2)
            if len(data) != count * 2:
                raise ValueError(f"{path} is truncated")
            out[c] = np.frombuffer(data, "<f2").reshape(y1 - y0, width)[:, x0:x1]
    return out


def clean_flow(flow):
    """float32 flow with non-finite values replaced by 0 and everything clamped to +-FLOW_LIMIT source px."""
    flow = np.nan_to_num(flow.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(flow, -FLOW_LIMIT, FLOW_LIMIT, out=flow)


class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, root, kind, crop=(160, 90), ping_pong=False, reset_prob=0.15, seed=0):
        """`crop` is (w, h) in source px; it has to fit the central crop of the dumps (320 x 180 by default
        geometry), and its position is drawn from the central crop only."""
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
        self.root, self.kind, self.crop = Path(root), kind, (int(crop[0]), int(crop[1]))
        self.ping_pong, self.reset_prob, self.seed = bool(ping_pong), float(reset_prob), int(seed)
        self.length = BURN_IN + LOSS + ((BURN_IN + LOSS - 1) if ping_pong else 0)
        listing = json.loads((self.root / "sequences.json").read_text())
        self.sequences = [seq for row in listing if row["kind"] == kind
                          if (seq := self._index(row)) is not None]

    def _index(self, row):
        """The usable form of a sequences.json row, or None while its dumps are not all finished."""
        seq_dir = self.root / row["kind"] / f"{row['id']:05d}"
        if not (is_complete(seq_dir, row) and self._has_target(seq_dir, row)):
            return None
        try:
            metas = {name: read_meta(seq_dir / name) for name in ("source", "base", "flow", TARGET_DIR[self.kind])}
        except (OSError, ValueError):  # the builder removed or is rewriting the directory right now
            return None
        if row["frames"] < LOSS:
            raise ValueError(f"sequence {row['id']} has {row['frames']} loss frames, a sample needs {LOSS}")
        width, height = metas["source"]["width"], metas["source"]["height"]
        central = (width - 2 * MARGIN, height - 2 * MARGIN)
        want = {"source": (3, width, height), "flow": (4, width, height),
                "base": (3, SCALE * width, SCALE * height),
                TARGET_DIR[self.kind]: (3, SCALE * central[0], SCALE * central[1])}
        for name, meta in metas.items():
            got = (meta["channels"], meta["width"], meta["height"])
            if got != want[name]:
                raise ValueError(f"sequence {row['id']}: {name} dump is {got} (channels, width, height), "
                                 f"expected {want[name]}")
        if not (0 < self.crop[0] <= central[0] and 0 < self.crop[1] <= central[1]):
            raise ValueError(f"crop {self.crop} (w, h) does not fit the {central} central crop of sequence {row['id']}")
        return _Sequence(row["id"], str(seq_dir), row["burn_in"], row["frames"], frozenset(row["cuts"]),
                         tuple(bool(v) for v in metas["flow"]["valid"]), width, height)

    def _has_target(self, seq_dir, row):
        """The target dump (teacher/ for teacher kind, target/ for gt) is finished: meta.json with all frames."""
        if self.kind == "teacher":
            return is_done(seq_dir, row)
        try:
            return read_meta(seq_dir / "target")["frames"] == row["burn_in"] + row["frames"]
        except (OSError, ValueError, KeyError):
            return False

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        index = int(index)
        if not 0 <= index < len(self.sequences):
            raise IndexError(f"index {index} out of range for {len(self.sequences)} sequences")
        seq = self.sequences[index]
        rng = np.random.default_rng(self.seed + index)
        cw, ch = self.crop
        x0 = int(rng.integers(MARGIN, seq.width - MARGIN - cw + 1))  # dumped px, inside the central crop
        y0 = int(rng.integers(MARGIN, seq.height - MARGIN - ch + 1))
        start = int(rng.integers(seq.burn_in, seq.burn_in + seq.frames - LOSS + 1))  # first loss frame, dumped index
        reset_draw, reset_pick = rng.random(), rng.random()

        first = start - BURN_IN  # dumped index of slot 0; negative when the sequence has fewer frames before the window
        pads = max(0, -first)
        real = list(range(max(first, 0), start + LOSS))
        planes = self._load(seq, real, x0, y0)
        slot_of = np.maximum(np.arange(BURN_IN + LOSS) - pads, 0)  # slot -> position in `real`; pads copy the first
        arrays = {name: plane[slot_of] for name, plane in planes.items()}
        arrays["flow"][:pads] = 0.0

        natural = np.array([i < pads or real[i - pads] in seq.cuts or not seq.valid[real[i - pads]]
                            for i in range(BURN_IN + LOSS)])  # reset without the first-frame and random rules
        reset = natural.copy()
        reset[pads] = True
        loss_mask = np.arange(BURN_IN + LOSS) >= BURN_IN
        if self.kind == "teacher" and seq.cuts:  # R34: the teacher lags behind a cut it ran across
            dumped = [real[i - pads] if i >= pads else None for i in range(BURN_IN + LOSS)]
            lagging = np.array([d is not None and any(c <= d < c + CUT_MASK_FRAMES for c in seq.cuts) for d in dumped])
            loss_mask &= ~lagging

        if self.ping_pong:
            back = np.arange(BURN_IN + LOSS - 2, -1, -1)  # original slots 18..0, in the order they are replayed
            slots = np.concatenate([np.arange(BURN_IN + LOSS), back])
            flow_back = arrays["flow"][back + 1][:, [2, 3, 0, 1]]  # stored at original t = k + 1, channels swapped
            flow_back[back < pads] = 0.0
            reset_back = natural[back + 1] | (back < pads)
            arrays = {name: (a[slots] if name != "flow" else np.concatenate([a, flow_back]))
                      for name, a in arrays.items()}
            reset = np.concatenate([reset, reset_back])
            loss_mask = loss_mask[slots]

        if reset_draw < self.reset_prob:
            scored = np.flatnonzero(loss_mask)
            if len(scored):
                reset[scored[int(reset_pick * len(scored))]] = True

        item = {name: torch.from_numpy(a).to(torch.float32) for name, a in arrays.items()}
        item.update(reset=torch.from_numpy(reset), loss_mask=torch.from_numpy(loss_mask), kind=self.kind,
                    seq_id=seq.id)
        return item

    def _load(self, seq, frames, x0, y0):
        """{"x", "base", "flow", "target"}: stacks over the dumped `frames` of the (x0, y0) sub-crop, float16 (flow
        float32, clean)."""
        cw, ch = self.crop
        s = SCALE
        seq_dir = Path(seq.dir)
        target_dir = TARGET_DIR[self.kind]
        # the target dump holds the central crop only, so its origin is the margin further in
        windows = {
            "x": ("source", (3, seq.height, seq.width), (y0, y0 + ch), (x0, x0 + cw)),
            "base": ("base", (3, s * seq.height, s * seq.width), (s * y0, s * (y0 + ch)), (s * x0, s * (x0 + cw))),
            "flow": ("flow", (4, seq.height, seq.width), (y0, y0 + ch), (x0, x0 + cw)),
            "target": (target_dir, (3, s * (seq.height - 2 * MARGIN), s * (seq.width - 2 * MARGIN)),
                       (s * (y0 - MARGIN), s * (y0 - MARGIN + ch)), (s * (x0 - MARGIN), s * (x0 - MARGIN + cw))),
        }
        planes = {}
        for name, (directory, shape, rows, cols) in windows.items():
            planes[name] = np.stack([read_window(seq_dir / directory / f"{i:06d}.f16", shape, rows, cols)
                                     for i in frames])
        planes["flow"] = clean_flow(planes["flow"])
        return planes
