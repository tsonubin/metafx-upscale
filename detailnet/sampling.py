"""Training-sequence sampler: picks the short frame windows (and crop positions) the
training-data builder extracts from the episodes.

Input is one per-frame table per episode (the rows of `output/sr-bench/shots/<stem>.frames.csv`:
frame, cut, content, pan, local, texture, luma). Windows are chosen so that

* 15% straddle exactly one shot cut, 3..frames-3 frames into the window;
* 5% are dissolves: a cut-free window with a run of >= 3 non-cut frames of content > 15;
* the rest lie inside one shot and are balanced over four motion buckets by the window's
  median pan/local motion (static, slow, fast, local);
* 10% are 24 frames long, the others 12;
* each is preceded by up to 8 burn-in frames of the same shot (none when the window starts at
  a cut);
* windows with mean luma < 20 are rejected;
* picked windows (burn-in included) do not overlap while the tables allow it.

A bucket that cannot supply its share is topped up from the other motion buckets, least filled
first, so the sampler never fails on a thin table; it returns fewer than n only when the tables
hold no usable window at all.
"""
import csv
from collections import Counter
from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

KINDS = ("teacher", "gt")
MOTION_BUCKETS = ("static", "slow", "fast", "local")
BUCKETS = MOTION_BUCKETS + ("straddle", "dissolve")

STRADDLE_FRACTION = 0.15
DISSOLVE_FRACTION = 0.05
LONG_FRACTION = 0.10
SHORT_LEN, LONG_LEN = 12, 24
MAX_BURN_IN = 8
MIN_LUMA = 20.0
STRADDLE_MARGIN = 3
DISSOLVE_CONTENT = 15.0
DISSOLVE_RUN = 3
CROP_W, CROP_H = 320, 180
CROP_MARGIN = 32


@dataclass
class SeqSpec:
    """One sequence to extract. `cuts` are frame offsets relative to `start - burn_in` (the
    first dumped frame); the first frame itself is never listed. `crop` = (x, y, w, h) in the
    sequence's own source pixels."""
    episode: str
    kind: str
    start: int
    frames: int
    burn_in: int
    bucket: str
    cuts: list
    crop: tuple


def load_frames_csv(path):
    """Parse a `*.frames.csv` into the row dicts `sample` takes."""
    with open(path, newline="") as f:
        return [{"frame": int(r["frame"]), "cut": int(float(r["cut"])),
                 **{k: float(r[k]) for k in ("content", "pan", "local", "texture", "luma")}}
                for r in csv.DictReader(f)]


@dataclass
class _Table:
    cut: np.ndarray
    content: np.ndarray
    pan: np.ndarray
    local: np.ndarray
    luma: np.ndarray
    burn_in: np.ndarray  # per start frame: frames available before it in its own shot, capped


def _table(rows):
    if [r["frame"] for r in rows] != list(range(len(rows))):
        raise ValueError("frame table must hold frames 0..N-1 in order")
    column = lambda name: np.array([r[name] for r in rows], dtype=np.float64)
    cut = column("cut") != 0
    index = np.arange(len(rows))
    shot_start = np.maximum.accumulate(np.where(cut, index, 0))
    return _Table(cut, column("content"), column("pan"), column("local"), column("luma"),
                  np.minimum(MAX_BURN_IN, index - shot_start))


def _windows(t, length):
    """Usable window starts of one length in one episode: {bucket: (starts, cut offsets)}.
    The cut offset is the straddle cut's position in the window (0 for the other buckets)."""
    empty = (np.empty(0, np.int64), np.empty(0, np.int64))
    found = {b: empty for b in BUCKETS}
    n = len(t.cut) - length + 1
    if n <= 0:
        return found
    s = np.arange(n)
    cuts = np.concatenate(([0], np.cumsum(t.cut)))  # cuts[k]: cuts among frames < k
    bright = sliding_window_view(t.luma, length).mean(axis=1) >= MIN_LUMA
    cut_free = (cuts[s + length] - cuts[s + 1] == 0) & bright  # only frame s may be a cut

    inside = s[cut_free]
    pan = np.median(sliding_window_view(t.pan, length)[inside], axis=1)
    local = np.median(sliding_window_view(t.local, length)[inside], axis=1)
    static = (pan < 0.3) & (local < 0.5)
    fast = pan >= 2
    local_motion = (local >= 2) & (pan < 2)
    slow = ~(static | fast | local_motion)
    zeros = np.zeros(len(inside), np.int64)
    for name, mask in zip(MOTION_BUCKETS, (static, slow, fast, local_motion)):
        found[name] = (inside[mask], zeros[mask])

    flag = (t.content > DISSOLVE_CONTENT) & ~t.cut
    run = flag[:-2] & flag[1:-1] & flag[2:]  # run[i]: frames i..i+2 all qualify
    runs = np.concatenate(([0], np.cumsum(run)))
    has_run = runs[s + length - 2] - runs[s] > 0
    dissolve = s[cut_free & has_run]
    found["dissolve"] = (dissolve, np.zeros(len(dissolve), np.int64))

    one_cut = s[(cuts[s + length] - cuts[s] == 1) & bright]
    positions = np.flatnonzero(t.cut)
    offset = positions[np.searchsorted(positions, one_cut)] - one_cut
    ok = (offset >= STRADDLE_MARGIN) & (offset <= length - STRADDLE_MARGIN)
    found["straddle"] = (one_cut[ok], offset[ok])
    return found


class _Pool:
    """Candidate windows of one (bucket, length) in random order, handed out one at a time."""

    def __init__(self, length, episode, start, offset, rng):
        order = rng.permutation(len(start))
        self.length = length
        self.episode, self.start, self.offset = episode[order], start[order], offset[order]
        self.cursor = {False: 0, True: 0}  # by allow_overlap; skipped windows never come back

    def take(self, tables, occupied, used, allow_overlap):
        cursor = self.cursor[allow_overlap]
        while cursor < len(self.start):
            ep, s = int(self.episode[cursor]), int(self.start[cursor])
            k, cursor = cursor, cursor + 1
            lo, hi = s - int(tables[ep].burn_in[s]), s + self.length
            if (ep, s, self.length) in used:
                continue
            if not allow_overlap and occupied[ep].find(1, lo, hi) != -1:
                continue
            self.cursor[allow_overlap] = cursor
            occupied[ep][lo:hi] = b"\x01" * (hi - lo)
            used.add((ep, s, self.length))
            return ep, s, int(self.offset[k])
        self.cursor[allow_overlap] = cursor
        return None


def _quotas(n, rng):
    straddle = int(n * STRADDLE_FRACTION + 0.5)
    dissolve = int(n * DISSOLVE_FRACTION + 0.5)
    base, extra = divmod(n - straddle - dissolve, len(MOTION_BUCKETS))
    quotas = {"straddle": straddle, "dissolve": dissolve}
    quotas.update({b: base for b in MOTION_BUCKETS})
    for i in rng.choice(len(MOTION_BUCKETS), extra, replace=False):
        quotas[MOTION_BUCKETS[i]] += 1
    return quotas


def sample(rows_by_episode, kind, n, frame_size, seed):
    """Pick n sequences (`SeqSpec`) from the episodes' frame tables, sorted by (episode, start).

    frame_size = (width, height) of the sequence's source frames: (1280, 720) for teacher,
    (640, 360) for gt. Same inputs and seed give the same list."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    width, height = frame_size
    x_max = width - CROP_W - CROP_MARGIN
    y_max = height - CROP_H - CROP_MARGIN
    if x_max < CROP_MARGIN or y_max < CROP_MARGIN:
        raise ValueError(f"frame size {frame_size} too small for a {CROP_W}x{CROP_H} crop "
                         f"with a {CROP_MARGIN} px margin")
    rng = np.random.default_rng(seed)

    names = [name for name in sorted(rows_by_episode) if rows_by_episode[name]]
    if not names or n <= 0:
        return []
    tables = [_table(rows_by_episode[name]) for name in names]
    occupied = [bytearray(len(t.cut)) for t in tables]
    used = set()

    quotas = _quotas(n, rng)
    slots = [b for b in ("straddle", "dissolve") + MOTION_BUCKETS for _ in range(quotas[b])]
    lengths = [SHORT_LEN] * n
    for i in rng.choice(n, int(n * LONG_FRACTION + 0.5), replace=False):
        lengths[i] = LONG_LEN

    pools = {}
    for length in (SHORT_LEN, LONG_LEN):
        per_episode = [_windows(t, length) for t in tables]
        for bucket in BUCKETS:
            columns = [np.concatenate(c) for c in zip(*[
                (np.full(len(w[bucket][0]), e, np.int64), *w[bucket])
                for e, w in enumerate(per_episode)])]
            pools[bucket, length] = _Pool(length, *columns, rng)

    chosen = [None] * n  # (bucket, episode, start, length, straddle offset)
    filled = Counter()  # motion-bucket sequences picked so far

    def fill(i, bucket_order, allow_overlap):
        for bucket in bucket_order:
            for length in (lengths[i], SHORT_LEN + LONG_LEN - lengths[i]):
                hit = pools[bucket, length].take(tables, occupied, used, allow_overlap)
                if hit:
                    chosen[i] = (bucket, *hit[:2], length, hit[2])
                    filled[bucket] += 1
                    return

    # Each slot tries its own bucket first, then the other motion buckets, least filled first.
    # Windows may overlap only once the non-overlapping ones are used up.
    for allow_overlap in (False, True):
        for cross in (False, True):
            for i, slot in enumerate(slots):
                if chosen[i] is None:
                    order = [slot]
                    if cross:
                        order = sorted((b for b in MOTION_BUCKETS if b != slot),
                                       key=lambda b: (filled[b], MOTION_BUCKETS.index(b)))
                    fill(i, order, allow_overlap)

    picks = sorted((c for c in chosen if c), key=lambda c: (names[c[1]], c[2], c[3]))
    specs = []
    for bucket, ep, start, length, offset in picks:
        burn_in = int(tables[ep].burn_in[start])
        x = int(rng.integers(CROP_MARGIN, x_max + 1))
        y = int(rng.integers(CROP_MARGIN, y_max + 1))
        specs.append(SeqSpec(names[ep], kind, start, length, burn_in, bucket,
                             [burn_in + offset] if bucket == "straddle" else [],
                             (x, y, CROP_W, CROP_H)))
    return specs
