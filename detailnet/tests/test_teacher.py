import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from detailnet import teacher
from detailnet.dumps import DumpWriter, read_frame, read_meta
from detailnet.guards import BatchGuard, SwapGuard, SwapLimitExceeded
from detailnet.teacher import (acquire_lock, is_done, is_ready, make_teacher, release_lock, round_robin, run_all,
                               to_teacher_grid, to_uint8_hwc)

CROP = (40, 40, 16, 8)  # tiny, so fake dumps stay small; the dumped (margin-expanded) window is 80x72 source px
SRC_W, SRC_H = 80, 72
OUT_W, OUT_H = 48, 24  # the teacher grid of that window: 3 x the central 16x8


def nearest4(frames):
    """A stand-in for FlashVSR: nearest-neighbour x4, (3, 4H, 4W) float32 in [0, 1] for each uint8 HxWx3 frame."""
    return [np.repeat(np.repeat(f.astype(np.float32).transpose(2, 0, 1) / 255.0, 4, axis=1), 4, axis=2)
            for f in frames]


# ---------------------------------------------------------------- the grid

def test_grid_shape():
    assert to_teacher_grid(np.zeros((3, 976, 1536), np.float32)).shape == (3, 540, 960)


def test_grid_alignment():
    lr = np.zeros((3, 244, 384), np.float32)
    lr[:, 32 + 20, 32 + 10] = 1.0  # one bright source pixel in the central crop
    hr = nearest4([np.round(lr * 255).astype(np.uint8).transpose(1, 2, 0)])[0]
    grid = to_teacher_grid(hr)
    ys, xs = np.nonzero(grid[0] > 0.5)
    assert (xs.min(), xs.max(), ys.min(), ys.max()) == (30, 32, 60, 62)  # x in [30, 33), y in [60, 63)
    assert np.allclose(grid[0, 60:63, 30:33], 1.0)
    assert grid.sum() == pytest.approx(3 * 9, abs=1e-3)  # nothing leaked outside the 3x3 block


def test_grid_is_an_exact_area_average_for_fractional_cells():
    hr = np.arange(8 * 8, dtype=np.float32).reshape(1, 8, 8) / 64
    grid = to_teacher_grid(hr, margin=0)  # 8x8 -> 6x6: each output cell covers 4/3 input pixels
    assert grid.shape == (1, 6, 6)
    corner = (hr[0, 0, 0] + hr[0, 0, 1] / 3 + hr[0, 1, 0] / 3 + hr[0, 1, 1] / 9) / (16 / 9)  # cell [0, 4/3)^2
    assert grid[0, 0, 0] == pytest.approx(corner, abs=1e-6)
    assert grid[0, 1, 1] == pytest.approx(hr[0, 1:3, 1:3].mean(), abs=1e-6)  # cell [4/3, 8/3)^2: 2/3 of pixels 1 and 2
    far = (hr[0, 2, 2] / 9 + hr[0, 2, 3] / 3 + hr[0, 3, 2] / 3 + hr[0, 3, 3]) / (16 / 9)  # cell [8/3, 4)^2
    assert grid[0, 2, 2] == pytest.approx(far, abs=1e-6)
    assert grid.mean() == pytest.approx(hr.mean(), abs=1e-6)  # area averaging preserves the mean
    assert to_teacher_grid(np.full((3, 16, 16), 0.4, np.float32), margin=1).std() == pytest.approx(0, abs=1e-6)


def test_grid_rejects_sizes_that_are_not_a_multiple_of_four():
    with pytest.raises(ValueError):
        to_teacher_grid(np.zeros((3, 10, 12), np.float32))


def test_to_uint8_hwc_is_the_documented_conversion():
    chw = np.array([[[0.0, 1.0 / 255, 0.5]], [[1.0, 0.0, 0.25]], [[0.2, 0.3, 0.4]]], np.float16)
    out = to_uint8_hwc(chw)
    assert out.dtype == np.uint8 and out.shape == (1, 3, 3) and out.flags["C_CONTIGUOUS"]
    expected = np.clip(np.round(chw.astype(np.float32) * 255), 0, 255).astype(np.uint8).transpose(1, 2, 0)
    assert np.array_equal(out, expected)


# ---------------------------------------------------------------- fake built sequences

def record(index, bucket="slow", frames=4, burn_in=3):
    return {"id": index, "episode": "ep", "kind": "teacher", "start": 100 + index, "frames": frames,
            "burn_in": burn_in, "bucket": bucket, "cuts": [], "crop": list(CROP)}


def make_built(data, rec, complete=True, seed=0):
    """A teacher sequence dir as build_data leaves it: spec.json, and source/base/flow dumps of burn_in + frames
    frames. With complete=False the build is 'still running': no flow meta, no timing in spec.json."""
    seq = Path(data) / "teacher" / f"{rec['id']:05d}"
    n = rec["burn_in"] + rec["frames"]
    seq.mkdir(parents=True, exist_ok=True)
    (seq / "spec.json").write_text(json.dumps(rec, indent=2))
    rng = np.random.default_rng(seed + rec["id"])
    source = DumpWriter(seq / "source", "source", 3, SRC_W, SRC_H, (1280, 720), 1, (8, 8, SRC_W, SRC_H))
    for k in range(n):
        source.write((rng.integers(0, 256, (3, SRC_H, SRC_W)) / 255).astype(np.float16), 10 + k / 25)
    source.finish()
    for name, ch, scale in (("base", 3, 3), ("flow", 4, 1)):
        writer = DumpWriter(seq / name, name, ch, 6, 4, (1280, 720), scale, (8, 8, SRC_W, SRC_H))
        for k in range(n):
            writer.write(np.zeros((ch, 4, 6), np.float16), 10 + k / 25, valid=k > 0)
        if complete or name != "flow":
            writer.finish()
    if complete:
        (seq / "spec.json").write_text(json.dumps({**rec, "flow_invalid": [0], "build_seconds": 1.5}, indent=2))
    return seq


def make_data(tmp_path, buckets, incomplete=(), **kw):
    """A data dir with one teacher sequence per entry of `buckets` (ids 0..), and its sequences.json (which also
    lists a gt sequence that must be ignored)."""
    recs = [record(i, b, **kw) for i, b in enumerate(buckets)]
    for rec in recs:
        make_built(tmp_path, rec, complete=rec["id"] not in incomplete)
    listing = recs + [{**record(0, "static"), "kind": "gt"}]
    (tmp_path / "sequences.json").write_text(json.dumps(listing))
    return recs


def ids(line):
    return re.search(r"teacher (\d{5})", line).group(1)


class Fake:
    """A fake upscaler that records its calls and can fail on chosen calls."""

    def __init__(self, fail_on=()):
        self.calls, self.fail_on = [], set(fail_on)

    def __call__(self, frames):
        self.calls.append(len(frames))
        if len(self.calls) in self.fail_on:
            raise RuntimeError("MPS out of memory")
        return nearest4(frames)


class Guard:
    def __init__(self, trip_after=None):
        self.calls, self.trip_after = 0, trip_after

    def check(self):
        self.calls += 1
        if self.trip_after is not None and self.calls > self.trip_after:
            raise SwapLimitExceeded("swap grew by 9.0 GB")


# ---------------------------------------------------------------- make_teacher

def test_make_teacher_with_fake_upscaler(tmp_path):
    rec = record(3)
    seq = make_built(tmp_path, rec)
    fake = Fake()
    make_teacher(seq, fake)
    assert fake.calls == [7]  # burn_in + frames frames, one call for the whole sequence
    meta = read_meta(seq / "teacher")
    assert (meta["kind"], meta["channels"], meta["width"], meta["height"]) == ("teacher", 3, OUT_W, OUT_H)
    assert meta["frames"] == 7 and meta["scale"] == 3
    assert meta["source_size"] == [1280, 720]
    assert meta["crop"] == list(CROP)  # the central crop, not the margin-expanded window of the source dump
    assert meta["pts"] == read_meta(seq / "source")["pts"]
    assert meta["valid"] == [True] * 7
    for i in range(7):
        src = read_frame(seq / "source", i).astype(np.float32)[:, 32:-32, 32:-32]
        want = np.repeat(np.repeat(src, 3, axis=1), 3, axis=2)  # nearest x3 of the central source crop
        got = read_frame(seq / "teacher", i).astype(np.float32)
        assert got.shape == want.shape == (3, OUT_H, OUT_W)
        assert np.abs(got - want).max() <= 1 / 255


def test_make_teacher_feeds_the_documented_uint8_frames(tmp_path):
    seq = make_built(tmp_path, record(0))
    seen = []

    def upscale(frames):
        seen.extend(frames)
        return nearest4(frames)

    make_teacher(seq, upscale)
    for i, frame in enumerate(seen):
        assert frame.dtype == np.uint8 and frame.shape == (SRC_H, SRC_W, 3)
        assert np.array_equal(frame, to_uint8_hwc(read_frame(seq / "source", i)))


def test_make_teacher_touches_nothing_but_teacher(tmp_path):
    seq = make_built(tmp_path, record(0))

    def snapshot():
        return {str(p.relative_to(seq)): p.read_bytes() for p in seq.rglob("*") if p.is_file()
                and "teacher" not in p.relative_to(seq).parts}

    before = snapshot()
    make_teacher(seq, Fake())
    assert snapshot() == before


@pytest.mark.parametrize("failure", ["raises", "too_few_frames", "wrong_shape_midway"])
def test_failure_leaves_no_teacher_dir(tmp_path, failure):
    seq = make_built(tmp_path, record(0))

    def upscale(frames):
        if failure == "raises":
            raise RuntimeError("boom")
        out = nearest4(frames)
        if failure == "too_few_frames":
            return out[:-1]
        out[4] = out[4][:, :-4]  # frames 0..3 are written before this one is rejected
        return out

    with pytest.raises((RuntimeError, ValueError)):
        make_teacher(seq, upscale)
    assert not (seq / "teacher").exists()  # so teacher/meta.json existing means complete
    assert not is_done(seq, record(0))


def test_meta_json_appears_only_after_the_last_frame(tmp_path, monkeypatch):
    seq = make_built(tmp_path, record(0))
    seen = []

    class Spy(DumpWriter):
        def write(self, chw, pts, valid=True):
            seen.append((self.d / "meta.json").exists())
            super().write(chw, pts, valid)

    monkeypatch.setattr(teacher, "DumpWriter", Spy)
    make_teacher(seq, Fake())
    assert seen == [False] * 7 and (seq / "teacher" / "meta.json").exists()


def test_interrupt_removes_the_partial_dir_too(tmp_path):
    seq = make_built(tmp_path, record(0))

    def upscale(frames):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        make_teacher(seq, upscale)
    assert not (seq / "teacher").exists()


def test_make_teacher_replaces_a_leftover_partial_dir(tmp_path):
    seq = make_built(tmp_path, record(0))
    (seq / "teacher").mkdir()
    (seq / "teacher" / "000000.f16").write_bytes(b"left by a killed run")
    (seq / "teacher" / "000009.f16").write_bytes(b"x")
    make_teacher(seq, Fake())
    assert is_done(seq, record(0))
    assert not (seq / "teacher" / "000009.f16").exists()


def test_is_done_needs_the_full_frame_count(tmp_path):
    rec = record(0)
    seq = make_built(tmp_path, rec)
    assert not is_done(seq, rec)
    make_teacher(seq, Fake())
    assert is_done(seq, rec)
    meta = json.loads((seq / "teacher" / "meta.json").read_text())
    meta["frames"] -= 1
    (seq / "teacher" / "meta.json").write_text(json.dumps(meta))
    assert not is_done(seq, rec)
    (seq / "teacher" / "meta.json").write_text("{not json")
    assert not is_done(seq, rec)


# ---------------------------------------------------------------- order and readiness

def test_round_robin_cycles_buckets_in_id_order():
    buckets = ["slow", "static", "static", "fast", "slow", "dissolve", "static", "slow"]
    entries = [record(i, b) for i, b in enumerate(buckets)]
    # one of each bucket (static, slow, fast, local, straddle, dissolve) per round, lowest id first within a bucket
    assert [e["id"] for e in round_robin(entries)] == [1, 0, 3, 5, 2, 4, 6, 7]
    assert [e["id"] for e in round_robin(entries[::-1])] == [1, 0, 3, 5, 2, 4, 6, 7]
    assert round_robin([]) == []


def test_round_robin_keeps_a_partial_run_balanced():
    names = ["static", "slow", "fast", "local", "straddle", "dissolve"]
    entries = [record(i, names[(i * 5 + i // 7) % 6]) for i in range(120)]
    first = round_robin(entries)[:30]
    counts = {n: sum(e["bucket"] == n for e in first) for n in names}
    assert set(counts.values()) == {5}


def test_round_robin_keeps_unknown_buckets():
    entries = [record(0, "slow"), record(1, "mystery"), record(2, "static")]
    assert [e["id"] for e in round_robin(entries)] == [2, 0, 1]


def test_is_ready_only_for_a_finished_build(tmp_path):
    recs = make_data(tmp_path, ["slow", "slow", "slow"], incomplete={1})
    assert is_ready(tmp_path, recs[0])
    assert not is_ready(tmp_path, recs[1])  # flow/meta.json not written yet
    assert not is_ready(tmp_path, recs[2] | {"start": 999})  # spec.json is from other sampling
    assert not is_ready(tmp_path, record(7))  # no directory at all


def test_is_ready_needs_full_flow_frames(tmp_path):
    recs = make_data(tmp_path, ["slow"])
    meta_path = tmp_path / "teacher" / "00000" / "flow" / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["frames"] -= 1
    meta_path.write_text(json.dumps(meta))
    assert not is_ready(tmp_path, recs[0])


# ---------------------------------------------------------------- the run

def test_run_all_makes_ready_sequences_in_round_robin_order(tmp_path):
    recs = make_data(tmp_path, ["slow", "static", "slow", "static", "fast"], incomplete={3})
    lines = []
    code, counts = run_all(tmp_path, Fake(), None, Guard(), lines.append)
    assert code == 0 and (counts["made"], counts["failed"], counts["not_built"]) == (4, 0, 1)
    done = [i for i in (1, 0, 4, 2) if is_done(tmp_path / "teacher" / f"{i:05d}", recs[i])]
    assert done == [1, 0, 4, 2]
    assert not (tmp_path / "teacher" / "00003" / "teacher").exists()  # unbuilt: left alone
    assert [ids(line) for line in lines[:-1]] == ["00001", "00000", "00004", "00002"]  # static, slow, fast, slow


def test_run_all_logs_one_line_per_sequence_and_a_summary(tmp_path):
    make_data(tmp_path, ["static", "slow", "fast"])
    lines = []
    code, counts = run_all(tmp_path, Fake(), None, Guard(), lines.append)
    assert code == 0 and len(lines) == 4
    first = lines[0]
    assert first.startswith("[   1] teacher 00000 static   frames=7 ")
    assert " s/frame" in first and "done 1/3" in first and "eta " in first
    assert ids(lines[1]) == "00001" and "done 2/3" in lines[1]
    assert "done 3/3" in lines[2] and lines[2].rstrip().endswith("eta 0:00:00")
    assert lines[-1].startswith("SUMMARY ok: made 3 already-done 0 failed 0 not-built 0 | done 3/3 | ")
    assert lines[-1].endswith("exit 0")


def test_run_all_resumes_and_skips_done_sequences(tmp_path):
    make_data(tmp_path, ["static", "slow", "fast"])
    run_all(tmp_path, Fake(), 1, Guard(), lambda line: None)  # one sequence, then stop
    fake, lines = Fake(), []
    code, counts = run_all(tmp_path, fake, None, Guard(), lines.append)
    assert code == 0 and fake.calls == [7, 7]  # not the first again
    assert counts["made"] == 2 and counts["already_done"] == 1
    assert "done 3/3" in lines[-2] and lines[-1].startswith("SUMMARY ok: made 2 already-done 1")
    again = Fake()
    code, counts = run_all(tmp_path, again, None, Guard(), lines.append)
    assert code == 0 and again.calls == [] and counts["made"] == 0 and counts["already_done"] == 3


def test_run_all_limit_counts_sequences_made(tmp_path):
    make_data(tmp_path, ["static", "slow", "fast", "local"])
    fake = Fake()
    code, counts = run_all(tmp_path, fake, 2, Guard(), lambda line: None)
    assert code == 0 and counts["made"] == 2 and len(fake.calls) == 2
    run_all(tmp_path, fake, 2, Guard(), lambda line: None)  # the next two, not the same two
    assert len(fake.calls) == 4


def test_run_all_ignores_gt_sequences_and_a_missing_listing(tmp_path):
    make_data(tmp_path, ["static"])
    fake = Fake()
    run_all(tmp_path, fake, None, Guard(), lambda line: None)
    assert len(fake.calls) == 1
    (tmp_path / "sequences.json").unlink()
    with pytest.raises(SystemExit, match="sequences.json"):
        run_all(tmp_path, fake, None, Guard(), lambda line: None)


def test_run_all_picks_up_sequences_built_while_it_runs(tmp_path):
    recs = make_data(tmp_path, ["static", "slow", "fast"], incomplete={1, 2})
    fake = Fake()

    def upscale(frames):
        if len(fake.calls) == 0:  # the build finishes sequences 1 and 2 meanwhile
            for i in (1, 2):
                make_built(tmp_path, recs[i])
        return fake(frames)

    code, counts = run_all(tmp_path, upscale, None, Guard(), lambda line: None)
    assert code == 0 and counts["made"] == 3 and counts["not_built"] == 0


def test_swap_guard_stops_the_run_with_exit_3(tmp_path):
    make_data(tmp_path, ["static", "slow", "fast", "local"])
    lines = []
    code, counts = run_all(tmp_path, Fake(), None, Guard(trip_after=2), lines.append)
    assert code == 3 and counts["made"] == 2
    assert any("swap guard" in line for line in lines)
    assert lines[-1].startswith("SUMMARY incomplete") and lines[-1].endswith("exit 3")


def batch_from_look(n):
    """A BatchGuard whose pgrep says the batch runs from the n-th look on; -> (guard, looks)."""
    looked = []

    def pgrep(pattern):
        looked.append(pattern)
        return len(looked) >= n

    return BatchGuard(False, pgrep=pgrep), looked


def test_the_batch_starting_stops_the_run_between_sequences_with_exit_3(tmp_path):
    """The batch is looked for before each sequence, so the one being made is finished first (a partial teacher/ is
    removed on any interruption anyway, R13) and the run stops with exit 3; a rerun resumes."""
    recs = make_data(tmp_path, ["static", "slow", "fast", "local"])
    batch, looked = batch_from_look(3)
    lines = []
    code, counts = run_all(tmp_path, Fake(), None, Guard(), lines.append, batch=batch)
    assert code == 3 and counts["made"] == 2 and len(looked) == 3
    assert sum(is_done(tmp_path / "teacher" / f"{r['id']:05d}", r) for r in recs) == 2
    assert all(not (tmp_path / "teacher" / f"{r['id']:05d}" / "teacher").exists() or
               is_done(tmp_path / "teacher" / f"{r['id']:05d}", r) for r in recs)  # nothing half-made
    assert any("batch" in line and "STOPPED" in line for line in lines)
    assert lines[-1].startswith("SUMMARY incomplete") and lines[-1].endswith("exit 3")
    code, counts = run_all(tmp_path, Fake(), None, Guard(), lambda line: None, batch=batch_from_look(99)[0])
    assert code == 0 and counts["made"] == 2 and counts["already_done"] == 2


def test_with_force_the_batch_is_never_looked_for(tmp_path):
    make_data(tmp_path, ["static", "slow"])
    batch = BatchGuard(True, pgrep=lambda pattern: pytest.fail("force must not even look"))
    code, counts = run_all(tmp_path, Fake(), None, Guard(), lambda line: None, batch=batch)
    assert code == 0 and counts["made"] == 2


def test_failures_are_logged_and_exit_1_but_do_not_stop_the_run(tmp_path):
    recs = make_data(tmp_path, ["static", "slow", "fast"])
    lines = []
    code, counts = run_all(tmp_path, Fake(fail_on={2}), None, Guard(), lines.append)
    assert code == 1 and (counts["made"], counts["failed"]) == (2, 1)
    failed = [line for line in lines if "FAILED" in line]
    assert len(failed) == 1 and "00001" in failed[0] and "MPS out of memory" in failed[0]
    assert not (tmp_path / "teacher" / "00001" / "teacher").exists()
    assert is_done(tmp_path / "teacher" / "00002", recs[2])
    assert lines[-1].startswith("SUMMARY incomplete") and "failed 1" in lines[-1]


def test_a_failed_sequence_is_not_retried_in_the_same_run(tmp_path):
    make_data(tmp_path, ["static"])
    fake = Fake(fail_on={1, 2, 3})
    code, counts = run_all(tmp_path, fake, None, Guard(), lambda line: None)
    assert len(fake.calls) == 1 and counts["failed"] == 1 and code == 1


def test_five_failures_in_a_row_stop_the_run(tmp_path):
    make_data(tmp_path, ["static"] * 8)
    lines = []
    code, counts = run_all(tmp_path, Fake(fail_on=range(1, 20)), None, Guard(), lines.append)
    assert code == 1 and counts["failed"] == 5
    assert any("5 failures in a row" in line for line in lines)


# ---------------------------------------------------------------- the lock

def test_lock_refuses_a_second_instance_and_is_released(tmp_path):
    lock = tmp_path / "teacher.lock"
    acquire_lock(lock)
    assert lock.read_text().strip() == str(os.getpid())
    with pytest.raises(SystemExit, match=str(os.getpid())):
        acquire_lock(lock)
    assert lock.read_text().strip() == str(os.getpid())  # the refused attempt did not disturb it
    release_lock(lock)
    assert not lock.exists()
    acquire_lock(lock)  # free again
    release_lock(lock)
    assert not list(tmp_path.iterdir())  # no temp files left behind


def test_stale_lock_of_a_dead_process_is_taken_over(tmp_path):
    lock = tmp_path / "teacher.lock"
    done = subprocess.run([sys.executable, "-c", "import os; print(os.getpid())"], capture_output=True, text=True)
    dead = int(done.stdout)
    lock.write_text(f"{dead}\n")
    acquire_lock(lock)
    assert lock.read_text().strip() == str(os.getpid())
    release_lock(lock)


@pytest.mark.parametrize("content", ["", "not a pid\n", "-5\n", "0\n"])
def test_garbage_lock_is_stale(tmp_path, content):
    lock = tmp_path / "teacher.lock"
    lock.write_text(content)
    acquire_lock(lock)
    assert lock.read_text().strip() == str(os.getpid())
    release_lock(lock)


def test_release_leaves_a_lock_that_belongs_to_someone_else(tmp_path):
    lock = tmp_path / "teacher.lock"
    lock.write_text(f"{os.getpid() + 1}\n")
    release_lock(lock)
    assert lock.exists()
    release_lock(tmp_path / "missing.lock")  # and a missing lock is fine


# ---------------------------------------------------------------- main

@pytest.fixture
def fake_main(tmp_path, monkeypatch):
    """main() with the batch check recorded, the swap read fixed, and the FlashVSR loader replaced."""
    state = {"forced": [], "loaded": 0, "fake": Fake(), "batch_forced": []}
    monkeypatch.setattr(teacher, "refuse_if_batch_running", lambda force: state["forced"].append(force))
    monkeypatch.setattr(teacher, "BatchGuard", lambda force: state["batch_forced"].append(force) or
                        BatchGuard(force, pgrep=lambda pattern: False), raising=False)
    monkeypatch.setattr(teacher, "SwapGuard", lambda limit: SwapGuard(limit, read=lambda: 1.0))

    def load():
        state["loaded"] += 1
        return state["fake"]

    monkeypatch.setattr(teacher, "load_flashvsr", load)
    return state


def test_main_runs_and_releases_the_lock(tmp_path, fake_main):
    make_data(tmp_path, ["static", "slow", "fast"])
    assert teacher.main(["--data", str(tmp_path), "--limit", "2"]) == 0
    assert fake_main["forced"] == [False]
    assert fake_main["loaded"] == 1 and len(fake_main["fake"].calls) == 2
    assert not (tmp_path / "teacher.lock").exists()


def test_main_does_not_load_the_model_when_there_is_nothing_to_do(tmp_path, fake_main):
    make_data(tmp_path, ["static", "slow"], incomplete={0, 1})
    assert teacher.main(["--data", str(tmp_path)]) == 0
    assert fake_main["loaded"] == 0


def test_main_force_skips_the_batch_check(tmp_path, fake_main):
    make_data(tmp_path, ["static"])
    teacher.main(["--data", str(tmp_path), "--force"])
    assert fake_main["forced"] == [True] and fake_main["batch_forced"] == [True]


def test_main_exits_3_when_the_batch_starts_and_releases_the_lock(tmp_path, fake_main, monkeypatch):
    make_data(tmp_path, ["static", "slow", "fast"])
    looks = []
    monkeypatch.setattr(teacher, "BatchGuard",
                        lambda force: BatchGuard(force, pgrep=lambda pattern: looks.append(1) or len(looks) >= 2))
    assert teacher.main(["--data", str(tmp_path)]) == 3
    assert len(fake_main["fake"].calls) == 1  # the first sequence was finished, the second never started
    assert not (tmp_path / "teacher.lock").exists()


def test_main_refuses_while_the_batch_runs(tmp_path, monkeypatch, fake_main):
    make_data(tmp_path, ["static"])

    def refuse(force):
        raise SystemExit("refusing to start: scripts/batch-upscale.sh is running")

    monkeypatch.setattr(teacher, "refuse_if_batch_running", refuse)
    with pytest.raises(SystemExit, match="batch"):
        teacher.main(["--data", str(tmp_path)])
    assert fake_main["loaded"] == 0 and not (tmp_path / "teacher.lock").exists()


def test_main_refuses_a_second_instance(tmp_path, fake_main):
    make_data(tmp_path, ["static"])
    acquire_lock(tmp_path / "teacher.lock")  # "another instance" (this very process, which is alive)
    with pytest.raises(SystemExit, match="already running"):
        teacher.main(["--data", str(tmp_path)])
    assert fake_main["loaded"] == 0
    assert (tmp_path / "teacher.lock").read_text().strip() == str(os.getpid())  # still the first one's


def test_main_releases_the_lock_when_the_run_dies(tmp_path, fake_main, monkeypatch):
    make_data(tmp_path, ["static"])
    monkeypatch.setattr(teacher, "run_all", lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt))
    with pytest.raises(KeyboardInterrupt):
        teacher.main(["--data", str(tmp_path)])
    assert not (tmp_path / "teacher.lock").exists()


def test_main_exit_code_3_when_swap_grows(tmp_path, fake_main, monkeypatch):
    make_data(tmp_path, ["static", "slow", "fast"])
    used = iter([1.0, 20.0, 20.0, 20.0])
    monkeypatch.setattr(teacher, "SwapGuard", lambda limit: SwapGuard(limit, read=lambda: next(used)))
    assert teacher.main(["--data", str(tmp_path)]) == 3
    assert not (tmp_path / "teacher.lock").exists()


def test_main_uses_the_eight_gb_swap_limit(tmp_path, fake_main, monkeypatch):
    make_data(tmp_path, ["static"])
    limits = []
    monkeypatch.setattr(teacher, "SwapGuard", lambda limit: limits.append(limit) or SwapGuard(limit, read=lambda: 1.0))
    teacher.main(["--data", str(tmp_path)])
    assert limits == [8.0]
