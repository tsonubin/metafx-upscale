import json
import os
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

from detailnet import data as data_module
from detailnet.data import SequenceDataset
from detailnet.dumps import DumpWriter

CROP = (16, 8)  # the dataset crop (w, h) of these tests; the real one is (160, 90)
CENTRAL = (48, 24)  # central crop of the fake dumps, source px
MARGIN = 32
SRC_W, SRC_H = CENTRAL[0] + 2 * MARGIN, CENTRAL[1] + 2 * MARGIN  # 112 x 88 dumped (margin-expanded) source px
X0_RANGE = (MARGIN, MARGIN + CENTRAL[0] - CROP[0])  # inclusive, in dumped source px
Y0_RANGE = (MARGIN, MARGIN + CENTRAL[1] - CROP[1])
N_BURN, N_LOSS = 8, 12


# ---------------------------------------------------------------- fake sequences

def source_frames(n, seed):
    """n source frames (3, SRC_H, SRC_W) float16. Frame k holds (k + u) / 40 with u uniform noise in [0, 0.9), so the
    frame number survives any crop (`frame_ids`) and the noise makes every crop position distinguishable."""
    rng = np.random.default_rng(seed)
    return [((k + rng.random((3, SRC_H, SRC_W)) * 0.9) / 40).astype(np.float16) for k in range(n)]


def frame_ids(t):
    """Dumped frame number of each frame of an (T, 3, h, w) item tensor made by `source_frames`."""
    return np.floor(t.double().mean(dim=(1, 2, 3)).numpy() * 40).astype(int).tolist()


def nearest3(src):
    return np.repeat(np.repeat(src, 3, axis=1), 3, axis=2)


def flow_pattern(k, c, y, x):
    """Flow value of frame k, channel c at grid position (y, x): a small distinct integer (|v| <= 48, so no clamping)."""
    return ((k * 5 + c * 11 + y * 3 + x) % 97 - 48).astype(np.float16)


def flow_frame(k):
    yy, xx = np.meshgrid(np.arange(SRC_H), np.arange(SRC_W), indexing="ij")
    return np.stack([flow_pattern(k, c, yy, xx) for c in range(4)])


def record(index, kind="teacher", burn_in=N_BURN, frames=N_LOSS, cuts=()):
    return {"id": index, "episode": "ep", "kind": kind, "start": 100 + index, "frames": frames, "burn_in": burn_in,
            "bucket": "slow", "cuts": list(cuts), "crop": [40, 40, CENTRAL[0], CENTRAL[1]]}


def make_sequence(root, rec, invalid=(), flow=None, complete=True, target=True, seed=None):
    """Write one sequence dir as build_data (+ teacher.py for the target) leaves it. base = nearest x3 of source and
    target = the central crop of base, so the target is aligned to the source by construction. `invalid` are dumped
    frames whose flow is marked invalid (frame 0 always is); `flow` maps frame -> replacement (4, H, W) array."""
    kind, n = rec["kind"], rec["burn_in"] + rec["frames"]
    seq = Path(root) / kind / f"{rec['id']:05d}"
    seq.mkdir(parents=True, exist_ok=True)
    frames = source_frames(n, rec["id"] if seed is None else seed)
    crop = (8, 8, SRC_W, SRC_H)
    pts = [10 + k / 25 for k in range(n)]
    writers = {"source": DumpWriter(seq / "source", "source", 3, SRC_W, SRC_H, (1280, 720), 1, crop),
               "base": DumpWriter(seq / "base", "base", 3, 3 * SRC_W, 3 * SRC_H, (1280, 720), 3, crop),
               "flow": DumpWriter(seq / "flow", "flow", 4, SRC_W, SRC_H, (1280, 720), 1, crop)}
    tgt_dir = "target" if kind == "gt" else "teacher"
    tgt_w, tgt_h = 3 * CENTRAL[0], 3 * CENTRAL[1]
    if target:
        writers[tgt_dir] = DumpWriter(seq / tgt_dir, "teacher" if kind == "teacher" else "source", 3, tgt_w, tgt_h,
                                      (1280, 720), 3, rec["crop"])
    bad = {0, *invalid}
    for k in range(n):
        base = nearest3(frames[k])
        writers["source"].write(frames[k], pts[k])
        writers["base"].write(base, pts[k])
        writers["flow"].write(flow[k] if flow and k in flow else flow_frame(k), pts[k], valid=k not in bad)
        if target:
            writers[tgt_dir].write(base[:, 3 * MARGIN:3 * MARGIN + tgt_h, 3 * MARGIN:3 * MARGIN + tgt_w], pts[k])
    for name, w in writers.items():
        if complete or name == "source":
            w.finish()
    spec = {**rec, "flow_invalid": sorted(bad)}
    if complete:
        spec["build_seconds"] = 1.5
    (seq / "spec.json").write_text(json.dumps(spec, indent=2))
    return seq


def make_data(root, recs, **kw):
    """sequences.json for `recs` (any mix of kinds) plus a complete dump dir for each."""
    for rec in recs:
        make_sequence(root, rec, **kw)
    (Path(root) / "sequences.json").write_text(json.dumps(recs))
    return recs


def area_down(t):
    """3x3 box average of (..., 3h, 3w) -> (..., h, w), in float64."""
    t = t.double()
    *lead, h, w = t.shape
    return t.reshape(*lead, h // 3, 3, w // 3, 3).mean(dim=(-3, -1))


def locate(item_x, frame):
    """The (x0, y0) at which item_x (3, h, w) is a crop of `frame` (3, SRC_H, SRC_W); fails unless exactly one fits."""
    h, w = item_x.shape[-2:]
    want = item_x.numpy()
    hits = [(x0, y0) for y0, x0 in np.argwhere(frame[0, :SRC_H - h + 1, :SRC_W - w + 1].astype(np.float32) == want[0, 0, 0])
            if np.array_equal(frame[:, y0:y0 + h, x0:x0 + w].astype(np.float32), want)]
    assert len(hits) == 1, hits
    return int(hits[0][0]), int(hits[0][1])


def dataset(tmp_path, recs=None, kind="teacher", **kw):
    make_data(tmp_path, recs if recs is not None else [record(0, kind)], **kw.pop("data", {}))
    kw.setdefault("crop", CROP)
    kw.setdefault("reset_prob", 0.0)
    return SequenceDataset(tmp_path, kind, **kw)


# ---------------------------------------------------------------- shape, alignment

def test_item_shapes_dtypes_and_metadata(tmp_path):
    item = dataset(tmp_path, [record(7)])[0]
    w, h = CROP
    assert item["x"].shape == (20, 3, h, w) and item["x"].dtype == torch.float32
    assert item["base"].shape == (20, 3, 3 * h, 3 * w) and item["base"].dtype == torch.float32
    assert item["flow"].shape == (20, 4, h, w) and item["flow"].dtype == torch.float32
    assert item["target"].shape == (20, 3, 3 * h, 3 * w) and item["target"].dtype == torch.float32
    assert item["reset"].shape == (20,) and item["reset"].dtype == torch.bool
    assert item["loss_mask"].shape == (20,) and item["loss_mask"].dtype == torch.bool
    assert item["kind"] == "teacher" and item["seq_id"] == 7


def test_crop_alignment(tmp_path):
    ds = dataset(tmp_path, [record(0), record(1)])
    for index in (0, 1):
        item = ds[index]
        # base is nearest x3 of the source and the target is the same picture, so both must average back to x exactly
        assert torch.allclose(area_down(item["base"]), item["x"].double(), atol=1e-7, rtol=0)
        assert torch.allclose(area_down(item["target"]), item["x"].double(), atol=1e-7, rtol=0)
        assert not torch.equal(item["x"][0], item["x"][5])  # distinct frames, not one repeated


def test_flow_is_the_same_crop_as_x(tmp_path):
    item = dataset(tmp_path)[0]
    h, w = CROP[1], CROP[0]
    frame = source_frames(20, 0)[-1]
    x0, y0 = locate(item["x"][-1], frame)
    yy, xx = np.meshgrid(np.arange(y0, y0 + h), np.arange(x0, x0 + w), indexing="ij")
    for slot in (8, 19):
        k = slot  # 8 burn-in dumped frames: slot == dumped frame for a sequence with burn_in 8
        want = np.stack([flow_pattern(k, c, yy, xx) for c in range(4)]).astype(np.float32)
        assert np.array_equal(item["flow"][slot].numpy(), want)


def test_subcrop_stays_in_the_central_crop_and_reaches_its_edges(tmp_path):
    ds = dataset(tmp_path)
    frame = source_frames(20, 0)[0]
    positions = set()
    for seed in range(400):
        ds.seed = seed
        positions.add(locate(ds[0]["x"][0], frame))
    xs, ys = {p[0] for p in positions}, {p[1] for p in positions}
    assert min(xs) == X0_RANGE[0] and max(xs) == X0_RANGE[1]  # inclusive at both ends, and never outside
    assert min(ys) == Y0_RANGE[0] and max(ys) == Y0_RANGE[1]
    assert len(xs) > 10 and len(ys) > 10


# ---------------------------------------------------------------- loss mask, windows, padding

def test_burn_in_unmasked(tmp_path):
    item = dataset(tmp_path)[0]
    assert item["loss_mask"].tolist() == [False] * N_BURN + [True] * N_LOSS


def test_frames_come_in_order_from_the_dump(tmp_path):
    item = dataset(tmp_path)[0]
    assert frame_ids(item["x"]) == list(range(20))
    assert frame_ids(item["base"]) == list(range(20))
    assert frame_ids(item["target"]) == list(range(20))


def test_reset_at_invalid_flow_and_cuts(tmp_path):
    rec = record(0, cuts=[11])
    item = dataset(tmp_path, [rec], data={"invalid": [14]})[0]
    # frame 0 (no previous frame; flow invalid), the cut at dumped frame 11, the invalid flow at 14; nothing else
    expected = [i in (0, 11, 14) for i in range(20)]
    assert item["reset"].tolist() == expected


def test_reset_at_a_cut_that_is_not_a_loss_frame_and_at_dumped_offset_zero(tmp_path):
    rec = record(0, cuts=[0, 3])  # offset 0 listed too: harmless
    assert dataset(tmp_path, [rec])[0]["reset"].tolist() == [i in (0, 3) for i in range(20)]


def test_flow_across_a_cut_is_kept_not_zeroed(tmp_path):
    item = dataset(tmp_path, [record(0, cuts=[11])])[0]
    assert item["reset"][11] and item["flow"][11].abs().sum() > 0


def test_random_reset_hits_one_loss_frame(tmp_path):
    ds = dataset(tmp_path, reset_prob=1.0)
    slots = set()
    for seed in range(200):
        ds.seed = seed
        reset = ds[0]["reset"]
        extra = [i for i in range(20) if reset[i] and i != 0]
        assert len(extra) == 1 and extra[0] >= N_BURN  # exactly one, on a loss frame, besides the first frame
        slots.add(extra[0])
    assert slots == set(range(N_BURN, 20))


def test_reset_prob_zero_adds_nothing_and_matches_the_rate(tmp_path):
    ds = dataset(tmp_path)
    assert all(ds[0]["reset"].tolist() == [True] + [False] * 19 for _ in range(3))
    ds = dataset(tmp_path, reset_prob=0.15)
    hits = 0
    for seed in range(400):
        ds.seed = seed
        hits += int(ds[0]["reset"].sum() > 1)
    assert 36 < hits < 84  # 15% of 400 = 60


def test_flow_is_clamped_and_made_finite(tmp_path):
    big = flow_frame(10).astype(np.float32)
    big[0] = 500.0
    big[1] = -500.0
    big[2] = 63.0
    big[3] = -64.0
    bad = flow_frame(12).astype(np.float32)
    bad[:, ::2, ::2] = np.nan
    bad[1, 1::2, 1::2] = np.inf
    rec = record(0)
    item = dataset(tmp_path, [rec], data={"invalid": [12], "flow": {10: big.astype(np.float16), 12: bad.astype(np.float16)}})[0]
    assert torch.isfinite(item["flow"]).all()
    assert item["flow"].abs().max() <= 64
    assert (item["flow"][10, 0] == 64).all() and (item["flow"][10, 1] == -64).all()
    assert (item["flow"][10, 2] == 63).all() and (item["flow"][10, 3] == -64).all()
    assert item["reset"][12]


def test_short_burn_in_is_front_padded(tmp_path):
    item = dataset(tmp_path, [record(0, burn_in=3)])[0]
    pads = N_BURN - 3
    assert frame_ids(item["x"]) == [0] * pads + list(range(15))  # copies of the first real frame, then the dump
    assert frame_ids(item["base"]) == [0] * pads + list(range(15))
    assert frame_ids(item["target"]) == [0] * pads + list(range(15))
    assert item["loss_mask"].tolist() == [False] * N_BURN + [True] * N_LOSS
    assert item["reset"].tolist() == [i <= pads for i in range(20)]  # pads and the first real frame
    assert item["flow"][:pads].abs().sum() == 0 and item["flow"][pads + 1:].abs().sum() > 0
    assert torch.equal(item["x"][0], item["x"][pads]) and torch.equal(item["target"][1], item["target"][pads])


def test_no_burn_in_is_all_padding(tmp_path):
    item = dataset(tmp_path, [record(0, burn_in=0, cuts=[5])])[0]
    assert frame_ids(item["x"]) == [0] * N_BURN + list(range(12))
    # R34: a teacher item does not score dumped frames 5..8 (slots 13..16), the first four from the cut
    assert item["loss_mask"].tolist() == [False] * N_BURN + [True] * 5 + [False] * 4 + [True] * 3
    assert item["reset"].tolist() == [i <= N_BURN or i == N_BURN + 5 for i in range(20)]
    assert item["flow"][:N_BURN].abs().sum() == 0


def test_long_sequence_takes_a_random_loss_window_with_its_own_burn_in(tmp_path):
    ds = dataset(tmp_path, [record(0, frames=24, cuts=[20])])
    starts = set()
    for seed in range(300):
        ds.seed = seed
        item = ds[0]
        ids = frame_ids(item["x"])
        start = ids[N_BURN]  # first loss frame
        assert ids == list(range(start - N_BURN, start + N_LOSS))  # the 8 dumped frames before it, then 12
        # R34: dumped frames 20..23, the first four from the cut, are not scored wherever the window puts them
        assert item["loss_mask"].tolist() == [i >= N_BURN and not 20 <= ids[i] < 24 for i in range(20)]
        assert item["reset"].tolist() == [i == 0 or ids[i] == 20 for i in range(20)]  # window start, and the cut
        starts.add(start)
    assert starts == set(range(N_BURN, 32 - N_LOSS + 1))  # 8..20: every 12-frame window of the 24 loss frames


def test_long_sequence_near_the_start_is_padded(tmp_path):
    ds = dataset(tmp_path, [record(0, burn_in=2, frames=24)])
    seen = {}
    for seed in range(300):
        ds.seed = seed
        item = ds[0]
        ids = frame_ids(item["x"])
        start = ids[N_BURN]
        pads = max(0, N_BURN - start)
        assert ids == [0] * pads + list(range(max(0, start - N_BURN), start + N_LOSS))
        assert item["reset"].tolist() == [i <= pads for i in range(20)]
        seen[start] = pads
    assert seen[2] == 6 and seen[8] == 0 and set(seen) == set(range(2, 15))


# ---------------------------------------------------------------- R34: teacher frames right after a cut

def scored(item):
    return [i for i, v in enumerate(item["loss_mask"].tolist()) if v]


def test_teacher_items_do_not_score_the_four_frames_from_a_cut(tmp_path):
    """FlashVSR made the teacher targets of a straddle sequence as one shot across its cut, so they lag for a few
    frames after it: the cut frame and the three after it are not scored (R34)."""
    assert data_module.CUT_MASK_FRAMES == 4
    item = dataset(tmp_path, [record(0, cuts=[11])])[0]
    assert scored(item) == [8, 9, 10] + list(range(15, 20))
    assert item["reset"].tolist() == [i in (0, 11) for i in range(20)]  # resets are unchanged


@pytest.mark.parametrize("cut,frames", [(18, [18, 19]), (19, [19]), (6, [8, 9]), (2, [])])
def test_the_four_frames_are_cut_short_by_the_item_s_edges(tmp_path, cut, frames):
    item = dataset(tmp_path, [record(0, cuts=[cut])])[0]
    assert scored(item) == [i for i in range(N_BURN, 20) if i not in frames]


def test_two_cuts_each_mask_their_frames(tmp_path):
    item = dataset(tmp_path, [record(0, cuts=[9, 16])])[0]
    assert scored(item) == [8, 13, 14, 15]


def test_gt_items_score_every_loss_frame_across_a_cut(tmp_path):
    item = dataset(tmp_path, [record(0, "gt", cuts=[11])], kind="gt")[0]
    assert item["loss_mask"].tolist() == [False] * N_BURN + [True] * N_LOSS
    assert item["reset"][11]


def test_ping_pong_mirrors_the_frames_after_a_cut(tmp_path):
    item = dataset(tmp_path, [record(0, cuts=[11])], ping_pong=True)[0]
    mask = item["loss_mask"].tolist()
    assert [i for i in range(20) if mask[i]] == [8, 9, 10] + list(range(15, 20))
    for j in range(19):  # the reversed frame is original 18 - j and keeps its mask
        assert mask[20 + j] == mask[18 - j]
    assert not any(mask[20 + (18 - k)] for k in (11, 12, 13, 14))


def test_ping_pong_on_a_padded_teacher_item_with_a_cut(tmp_path):
    item = dataset(tmp_path, [record(0, burn_in=3, cuts=[6])], ping_pong=True)[0]
    pads = N_BURN - 3  # dumped frame d sits in slot d + pads: the cut's frames 6..9 are slots 11..14
    mask = item["loss_mask"].tolist()
    assert [i for i in range(20) if mask[i]] == [8, 9, 10] + list(range(15, 20))
    assert all(mask[20 + j] == mask[18 - j] for j in range(19)) and pads == 5


def test_random_reset_only_lands_on_a_scored_frame(tmp_path):
    ds = dataset(tmp_path, [record(0, cuts=[11])], reset_prob=1.0)
    slots = set()
    for seed in range(200):
        ds.seed = seed
        item = ds[0]
        extra = [i for i in range(20) if item["reset"][i] and i not in (0, 11)]
        assert len(extra) == 1 and item["loss_mask"][extra[0]]
        slots.add(extra[0])
    assert slots == {8, 9, 10, 15, 16, 17, 18, 19}


# ---------------------------------------------------------------- ping-pong

def test_ping_pong_length(tmp_path):
    item = dataset(tmp_path, ping_pong=True)[0]
    assert item["x"].shape[0] == 39 and item["flow"].shape[0] == 39 and item["target"].shape[0] == 39
    assert item["reset"].shape == (39,) and item["loss_mask"].shape == (39,)
    assert dataset(tmp_path)[0]["x"].shape[0] == 20


def test_ping_pong_replays_the_frames_backwards_without_repeating_the_last(tmp_path):
    item = dataset(tmp_path, ping_pong=True)[0]
    assert frame_ids(item["x"]) == list(range(20)) + list(range(18, -1, -1))
    assert frame_ids(item["base"]) == frame_ids(item["x"]) and frame_ids(item["target"]) == frame_ids(item["x"])
    for j in range(19):  # the second half is the forward frames in reverse, pixel for pixel
        assert torch.equal(item["x"][20 + j], item["x"][18 - j])
        assert torch.equal(item["base"][20 + j], item["base"][18 - j])
        assert torch.equal(item["target"][20 + j], item["target"][18 - j])


def test_ping_pong_swaps_flow(tmp_path):
    item = dataset(tmp_path, ping_pong=True)[0]
    flow = item["flow"]
    for j in range(19):
        t = 19 - j  # the reversed frame is original t - 1, following original t
        rev = flow[20 + j]  # == orig t - 1; its predecessor is original t
        assert torch.equal(rev[0:2], flow[t][2:4]), t  # new back = the fwd stored at original t
        assert torch.equal(rev[2:4], flow[t][0:2]), t  # new fwd = the back stored at original t
    # and the stored channels really are the dump's: back of dumped frame 19 is channels 0-1 of the pattern
    assert flow[19][0:2].abs().sum() > 0 and not torch.equal(flow[19][0:2], flow[19][2:4])


def test_ping_pong_resets_follow_original_t(tmp_path):
    item = dataset(tmp_path, [record(0, cuts=[11])], ping_pong=True, data={"invalid": [14]})[0]
    reset = item["reset"].tolist()
    assert reset[:20] == [i in (0, 11, 14) for i in range(20)]
    # the reversed frame original t - 1 follows original t, so it resets when t is a cut or has invalid flow (t = 11, 14)
    assert reset[20:] == [(k + 1) in (11, 14) for k in range(18, -1, -1)]
    assert [k for k, r in zip(range(18, -1, -1), reset[20:]) if r] == [13, 10]


def test_ping_pong_loss_mask_follows_the_original_frame(tmp_path):
    item = dataset(tmp_path, ping_pong=True)[0]
    mask = item["loss_mask"].tolist()
    assert mask[:20] == [False] * N_BURN + [True] * N_LOSS
    for j in range(19):
        assert mask[20 + j] == mask[18 - j]  # the reversed frame is original 18 - j and keeps its mask


def test_ping_pong_with_padding(tmp_path):
    item = dataset(tmp_path, [record(0, burn_in=3)], ping_pong=True)[0]
    pads = N_BURN - 3
    assert frame_ids(item["x"]) == ([0] * pads + list(range(15))) + ([0] * pads + list(range(15)))[-2::-1]
    reset = item["reset"].tolist()
    # reversed half, original slot k = 18 - j: pads (k < pads) reset, with zero flow; the first real frame (k = pads) does not
    for j in range(19):
        k = 18 - j
        if k < pads:
            assert reset[20 + j] and item["flow"][20 + j].abs().sum() == 0
        else:
            assert not reset[20 + j]
    assert item["flow"][20 + (18 - pads)].abs().sum() > 0


# ---------------------------------------------------------------- determinism

def test_items_are_reproducible_for_a_seed_and_index(tmp_path):
    recs = [record(0), record(1, frames=24)]
    make_data(tmp_path, recs)
    a = SequenceDataset(tmp_path, "teacher", crop=CROP, seed=5, reset_prob=0.5)
    b = SequenceDataset(tmp_path, "teacher", crop=CROP, seed=5, reset_prob=0.5)
    for index in (0, 1):
        for key, value in a[index].items():
            same = torch.equal(value, b[index][key]) if torch.is_tensor(value) else value == b[index][key]
            assert same, key
    assert torch.equal(a[1]["x"], a[1]["x"])


def test_items_do_not_depend_on_global_rngs(tmp_path):
    ds = dataset(tmp_path, [record(0, frames=24)], reset_prob=0.5)
    np.random.seed(1)
    torch.manual_seed(1)
    first = ds[0]
    np.random.seed(2)
    torch.manual_seed(2)
    np.random.random(100)
    second = ds[0]
    assert all(torch.equal(first[k], second[k]) for k in ("x", "base", "flow", "target", "reset", "loss_mask"))
    state = np.random.get_state()[1].copy()
    ds[0]
    assert np.array_equal(np.random.get_state()[1], state)  # the global numpy generator was not touched


def test_other_seeds_and_indices_give_other_crops(tmp_path):
    make_data(tmp_path, [record(0), record(1)])
    crops = {(seed, index): SequenceDataset(tmp_path, "teacher", crop=CROP, seed=seed)[index]["x"][0].numpy().tobytes()
             for seed in range(4) for index in (0, 1)}
    assert len(set(crops.values())) == 8


# ---------------------------------------------------------------- which sequences are usable

def test_indexes_only_complete_sequences_of_the_kind(tmp_path):
    recs = [record(i) for i in range(8)] + [record(8, "gt")]
    for rec in recs:
        make_sequence(tmp_path, rec, complete=rec["id"] != 1,  # 1: build still running (no flow meta, no timing in spec)
                      target=rec["id"] != 2)  # 2: FlashVSR has not reached it
    (tmp_path / "sequences.json").write_text(json.dumps(recs))
    teacher = tmp_path / "teacher"
    meta = json.loads((teacher / "00003" / "teacher" / "meta.json").read_text())
    (teacher / "00003" / "teacher" / "meta.json").write_text(json.dumps({**meta, "frames": 5}))  # 3: wrong frame count
    spec = json.loads((teacher / "00004" / "spec.json").read_text())
    (teacher / "00004" / "spec.json").write_text(json.dumps({**spec, "start": 999}))  # 4: re-planned since it was built
    (teacher / "00005" / "teacher" / "meta.json").unlink()  # 5: FlashVSR is mid-sequence
    (teacher / "00006" / "base" / "meta.json").unlink()  # 6: being rebuilt
    ds = SequenceDataset(tmp_path, "teacher", crop=CROP)
    assert [ds[i]["seq_id"] for i in range(len(ds))] == [0, 7]  # the gt row and the six unusable ones are out


def test_gt_needs_its_target_dump(tmp_path):
    recs = [record(0, "gt"), record(1, "gt"), record(2, "teacher")]
    make_sequence(tmp_path, recs[0])
    make_sequence(tmp_path, recs[1], target=False)
    make_sequence(tmp_path, recs[2])
    (tmp_path / "sequences.json").write_text(json.dumps(recs))
    ds = SequenceDataset(tmp_path, "gt", crop=CROP)
    assert len(ds) == 1
    item = ds[0]
    assert item["kind"] == "gt" and item["seq_id"] == 0
    assert torch.allclose(area_down(item["target"]), item["x"].double(), atol=1e-7, rtol=0)


def test_target_is_read_from_the_kind_s_own_dump(tmp_path):
    make_data(tmp_path, [record(0, "gt")])
    target_dir = tmp_path / "gt" / "00000" / "target"
    writer = DumpWriter(target_dir, "source", 3, 3 * CENTRAL[0], 3 * CENTRAL[1], (1920, 1080), 3, (0, 0, 48, 24))
    for k in range(20):
        writer.write(np.full((3, 3 * CENTRAL[1], 3 * CENTRAL[0]), 0.25 + k / 100, np.float16), k)
    writer.finish()
    item = SequenceDataset(tmp_path, "gt", crop=CROP)[0]
    assert torch.allclose(item["target"][4], torch.full((3, 3 * CROP[1], 3 * CROP[0]), 0.29), atol=1e-3)


def test_index_is_fixed_at_construction(tmp_path):
    make_data(tmp_path, [record(0), record(1)])
    ds = SequenceDataset(tmp_path, "teacher", crop=CROP)
    make_data(tmp_path, [record(0), record(1), record(2)])  # a new sequence completes later
    assert len(ds) == 2
    assert len(SequenceDataset(tmp_path, "teacher", crop=CROP)) == 3
    with pytest.raises(IndexError):
        ds[2]


def test_empty_dataset_and_bad_arguments(tmp_path):
    make_data(tmp_path, [record(0)])
    assert len(SequenceDataset(tmp_path, "gt", crop=CROP)) == 0
    with pytest.raises(ValueError):
        SequenceDataset(tmp_path, "other", crop=CROP)
    with pytest.raises(ValueError, match="crop"):
        SequenceDataset(tmp_path, "teacher", crop=(CENTRAL[0] + 1, 8))  # does not fit the central crop
    with pytest.raises(FileNotFoundError):
        SequenceDataset(tmp_path / "nowhere", "teacher", crop=CROP)


def test_geometry_that_is_not_the_dumps_layout_is_refused(tmp_path):
    rec = record(0)
    make_data(tmp_path, [rec])
    meta_path = tmp_path / "teacher" / "00000" / "base" / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["width"] += 3
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="base"):
        SequenceDataset(tmp_path, "teacher", crop=CROP)


# ---------------------------------------------------------------- use with torch

def test_batches_collate_and_the_dataset_pickles(tmp_path):
    ds = dataset(tmp_path, [record(0), record(1, frames=24), record(2, burn_in=2)], ping_pong=False)
    batch = next(iter(torch.utils.data.DataLoader(ds, batch_size=3)))
    assert batch["x"].shape == (3, 20, 3, 8, 16) and batch["reset"].shape == (3, 20)
    assert batch["seq_id"].tolist() == [0, 1, 2] and batch["kind"] == ["teacher"] * 3
    clone = pickle.loads(pickle.dumps(ds))
    assert torch.equal(clone[1]["x"], ds[1]["x"])


def test_reads_do_not_leak_file_descriptors(tmp_path):
    ds = dataset(tmp_path, ping_pong=True)
    ds[0]
    before = len(os.listdir("/dev/fd"))
    for _ in range(5):
        ds[0]
    assert len(os.listdir("/dev/fd")) == before
