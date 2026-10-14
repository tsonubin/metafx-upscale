from collections import Counter

import numpy as np
import pytest

from detailnet.sampling import SeqSpec, load_frames_csv, sample

MOTION = {"static": (0.0, 0.0), "slow": (1.0, 0.2), "fast": (4.0, 0.5), "local": (0.5, 3.0),
          # threshold cases: slow, local, fast, slow
          "edge_a": (0.2, 1.0), "edge_b": (1.5, 2.5), "edge_c": (2.0, 3.0), "edge_d": (0.3, 0.0)}
MOTION_NAMES = ("static", "slow", "fast", "local")
SEEDS = range(6)


def make_episode(shots):
    """Frame table from shots = [(length, motion, extras)]. A shot starts with cut=1 (except the
    first); extras: luma (default 120), dissolve=[(offset, length)] frames with content 20 and
    soft=[(offset, length)] frames with content 10."""
    rows = []
    for index, (length, motion, extras) in enumerate(shots):
        pan, local = MOTION[motion]
        for i in range(length):
            rows.append({"frame": len(rows), "cut": int(index > 0 and i == 0), "content": 0.0,
                         "pan": pan, "local": local, "texture": 5.0,
                         "luma": extras.get("luma", 120.0)})
        for key, content in (("dissolve", 20.0), ("soft", 10.0)):  # soft: below the threshold
            for offset, run in extras.get(key, []):
                for i in range(offset, offset + run):
                    rows[len(rows) - length + i]["content"] = content
        if index > 0:
            rows[len(rows) - length]["content"] = 45.0
    return rows


def shot(length, motion, **extras):
    return (length, motion, extras)


def episode_tables():
    """Two episodes with 27 shots: all four motion types, threshold cases, dissolves, two short
    shots (5 and 11 frames) and one dark shot, so every bucket can be satisfied by non-overlapping windows."""
    def long_shots(offset):
        return [shot(240, MOTION_NAMES[(i + offset) % 4], dissolve=[(100, 6)] if i % 3 == 0 else [],
                     soft=[(100, 6)] if i % 3 == 1 else [])
                for i in range(10)]

    first = long_shots(0)
    first[3:3] = [shot(5, "fast"), shot(11, "static")]
    first.append(shot(240, "static", luma=5.0))
    second = long_shots(1) + [shot(240, m) for m in ("edge_a", "edge_b", "edge_c", "edge_d")]
    return {"ep_a": make_episode(first), "ep_b": make_episode(second)}


def shot_start(rows, frame):
    while frame > 0 and not rows[frame]["cut"]:
        frame -= 1
    return frame


def window(rows, spec):
    return rows[spec.start:spec.start + spec.frames]


def window_bucket(rows, spec):
    pan = np.median([r["pan"] for r in window(rows, spec)])
    local = np.median([r["local"] for r in window(rows, spec)])
    if pan < 0.3 and local < 0.5:
        return "static"
    if pan >= 2:
        return "fast"
    if local >= 2:
        return "local"
    return "slow"


@pytest.fixture(scope="module")
def tables():
    return episode_tables()


@pytest.fixture(scope="module")
def runs(tables):
    return [sample(tables, "teacher", 60, (1280, 720), seed=seed) for seed in SEEDS]


@pytest.fixture(scope="module")
def specs(runs):
    return [s for run in runs for s in run]


def test_proportions_and_lengths(runs):
    for run in runs:
        assert len(run) == 60
        assert all(isinstance(s, SeqSpec) and s.kind == "teacher" for s in run)
        counts = Counter(s.bucket for s in run)
        assert counts["straddle"] == 9 and counts["dissolve"] == 3
        assert Counter(s.frames for s in run) == {12: 54, 24: 6}


def test_buckets_balanced(runs, tables):
    for run in runs:
        counts = Counter(s.bucket for s in run)
        motion = [counts[b] for b in MOTION_NAMES]
        assert sum(motion) == 60 - 9 - 3
        assert max(motion) - min(motion) <= 1
        for s in run:
            if s.bucket in MOTION_NAMES:
                assert window_bucket(tables[s.episode], s) == s.bucket


def test_buckets_balanced_when_count_does_not_divide(tables):
    out = sample(tables, "teacher", 37, (1280, 720), seed=1)
    counts = Counter(s.bucket for s in out)
    motion = [counts[b] for b in MOTION_NAMES]
    assert len(out) == 37 and sum(motion) == 37 - counts["straddle"] - counts["dissolve"]
    assert max(motion) - min(motion) <= 1


def test_within_shot_never_crosses_cut(specs, tables):
    within = [s for s in specs if s.bucket != "straddle"]
    assert len(within) == 51 * len(SEEDS)
    for s in within:
        rows = tables[s.episode]
        assert not any(r["cut"] for r in window(rows, s)[1:])
        assert s.cuts == []


def test_straddle_has_one_cut(specs, tables):
    straddles = [s for s in specs if s.bucket == "straddle"]
    assert len(straddles) == 9 * len(SEEDS)
    for s in straddles:
        rows = tables[s.episode]
        assert len(s.cuts) == 1
        in_window = [i for i, r in enumerate(window(rows, s)) if r["cut"]]
        assert len(in_window) == 1
        assert 3 <= in_window[0] <= s.frames - 3
        assert s.cuts == [s.burn_in + in_window[0]]
        assert rows[s.start - s.burn_in + s.cuts[0]]["cut"] == 1


def test_burn_in_stays_in_shot(specs, tables):
    assert any(s.burn_in == 8 for s in specs) and any(s.burn_in < 8 for s in specs)
    for s in specs:
        rows = tables[s.episode]
        assert s.burn_in == min(8, s.start - shot_start(rows, s.start))
        assert not any(r["cut"] for r in rows[s.start - s.burn_in + 1:s.start + 1])
        if rows[s.start]["cut"]:
            assert s.burn_in == 0


def test_dissolve_has_a_run_of_high_content_without_cuts(specs, tables):
    dissolves = [s for s in specs if s.bucket == "dissolve"]
    assert len(dissolves) == 3 * len(SEEDS)
    for s in dissolves:
        rows = window(tables[s.episode], s)
        assert not any(r["cut"] for r in rows[1:]) and s.cuts == []
        flags = [r["content"] > 15 and not r["cut"] for r in rows]
        assert any(all(flags[i:i + 3]) for i in range(len(flags) - 2))


def test_dark_windows_rejected(specs, tables):
    assert all(np.mean([r["luma"] for r in window(tables[s.episode], s)]) >= 20 for s in specs)


def test_short_shots_skipped():
    short_only = make_episode([shot(11, MOTION_NAMES[i % 4]) for i in range(30)])
    out = sample({"ep": short_only}, "teacher", 40, (1280, 720), seed=0)
    assert not [s for s in out if s.bucket in MOTION_NAMES + ("dissolve",)]

    # shots span frames 0-239, 240-244 (5), 245-255 (11), 256-495: only the long ones can host a window
    mixed = make_episode([shot(240, "slow"), shot(5, "fast"), shot(11, "local"), shot(240, "slow")])
    out = sample({"ep": mixed}, "teacher", 20, (1280, 720), seed=0)
    within = [s for s in out if s.bucket != "straddle"]
    assert within
    for s in within:
        assert s.start + s.frames <= 240 or s.start >= 256
        assert s.bucket not in ("fast", "local")


@pytest.mark.parametrize("kind, size", [("teacher", (1280, 720)), ("gt", (640, 360))])
def test_crop_with_margin_inside(tables, kind, size):
    out = sample(tables, kind, 60, size, seed=5)
    assert all(s.kind == kind for s in out)
    for s in out:
        x, y, w, h = s.crop
        assert (w, h) == (320, 180)
        assert x - 32 >= 0 and y - 32 >= 0
        assert x - 32 + 384 <= size[0] and y - 32 + 244 <= size[1]
    assert len({s.crop for s in out}) > 20
    assert max(s.crop[0] for s in out) > size[0] // 2 - 160 > min(s.crop[0] for s in out)


def test_frame_too_small_for_margin_raises(tables):
    with pytest.raises(ValueError):
        sample(tables, "gt", 4, (383, 360), seed=0)
    with pytest.raises(ValueError):
        sample(tables, "bogus", 4, (640, 360), seed=0)


def test_deterministic_seed(tables):
    a = sample(tables, "teacher", 60, (1280, 720), seed=11)
    assert a == sample(tables, "teacher", 60, (1280, 720), seed=11)
    assert a == sample(dict(reversed(list(tables.items()))), "teacher", 60, (1280, 720), seed=11)
    assert a != sample(tables, "teacher", 60, (1280, 720), seed=12)


def test_windows_do_not_overlap_when_the_table_allows_it(runs):
    for run in runs:
        spans = sorted((s.episode, s.start - s.burn_in, s.start + s.frames) for s in run)
        for (ep, _, end), (ep2, start2, _) in zip(spans, spans[1:]):
            assert ep != ep2 or end <= start2


def test_missing_bucket_is_filled_from_the_others():
    only_static = {"ep": make_episode([shot(1500, "static")])}
    out = sample(only_static, "teacher", 20, (1280, 720), seed=0)
    assert len(out) == 20 and {s.bucket for s in out} == {"static"}
    assert out == sample(only_static, "teacher", 20, (1280, 720), seed=0)


def test_tiny_tables_do_not_crash():
    tiny = {"ep": make_episode([shot(30, "slow")])}
    assert len(sample(tiny, "teacher", 5, (1280, 720), seed=0)) == 5
    too_short = {"ep": make_episode([shot(8, "slow")])}
    assert sample(too_short, "teacher", 5, (1280, 720), seed=0) == []
    assert sample({}, "teacher", 5, (1280, 720), seed=0) == []
    assert sample(tiny, "teacher", 0, (1280, 720), seed=0) == []


def test_load_frames_csv(tmp_path):
    path = tmp_path / "ep.frames.csv"
    path.write_text("frame,cut,content,pan,local,texture,luma\n0,0,0.0,0.5,0.25,6.6,12.5\n"
                    "1,1,33.0,2.0,3.0,6.7,99.0\n")
    assert load_frames_csv(path) == [
        {"frame": 0, "cut": 0, "content": 0.0, "pan": 0.5, "local": 0.25, "texture": 6.6, "luma": 12.5},
        {"frame": 1, "cut": 1, "content": 33.0, "pan": 2.0, "local": 3.0, "texture": 6.7, "luma": 99.0}]
