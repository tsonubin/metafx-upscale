import json
from pathlib import Path

import numpy as np
import pytest

from detailnet import build_data
from detailnet.build_data import (EPISODES, BuildError, Builder, Episode, StaleKept, check_cuts, check_metas,
                                  check_pts, dump_crop, ffmpeg_seek, gt_bitrate, is_complete, prores_cmd, run_all,
                                  spec_record, swift_cmd, swift_range, x264_cmd)
from detailnet.dumps import DumpWriter, read_meta
from detailnet.guards import SwapGuard, SwapLimitExceeded
from detailnet.sampling import SeqSpec

CROP = (40, 40, 16, 8)  # tiny, so fake dumps stay small; the dumped window is (8, 8, 80, 72)


def test_dump_crop_adds_margin():
    assert dump_crop((320, 180, 320, 180)) == (288, 148, 384, 244)
    assert dump_crop((320, 180, 320, 180), margin=0) == (320, 180, 320, 180)
    assert dump_crop((32, 32, 320, 180)) == (0, 0, 384, 244)


def test_gt_bitrate():
    assert gt_bitrate(25) == 576000
    assert isinstance(gt_bitrate(25), int)
    assert gt_bitrate(24) == 552960


def test_swift_cmd_flags():
    cmd = swift_cmd("bin/metalfx-upscale", "ep.mp4", "seq", 120.02, 0.44, (320, 180, 320, 180),
                    {"source", "base", "flow"})
    assert cmd == ["bin/metalfx-upscale", "ep.mp4", "seq/tmp-upscale.mp4", "--engine", "ml", "--size", "4k",
                   "--bitrate", "5", "--overwrite", "--start", "120.020000", "--duration", "0.440000",
                   "--dump-source", "seq/source", "--dump-base", "seq/base", "--dump-flow", "seq/flow",
                   "--dump-crop", "288,148,384,244"]


def test_swift_cmd_whole_clip_target_without_margin():
    cmd = swift_cmd("bin", "clip.mov", Path("seq"), None, None, (960, 540, 960, 540), {"source"},
                    margin=0, dir_names={"source": "target"})
    assert cmd == ["bin", "clip.mov", "seq/tmp-upscale.mp4", "--engine", "ml", "--size", "4k", "--bitrate", "5",
                   "--overwrite", "--dump-source", "seq/target", "--dump-crop", "960,540,960,540"]


def test_swift_cmd_size_and_dump_validation():
    cmd = swift_cmd("bin", "in.mp4", "seq", None, None, (32, 32, 320, 180), {"flow", "source"}, size="1920x1080")
    assert cmd[cmd.index("--size") + 1] == "1920x1080"
    assert "--dump-base" not in cmd
    assert cmd[cmd.index("--dump-source") + 1] == "seq/source"  # fixed order, whatever the set order
    with pytest.raises(ValueError):
        swift_cmd("bin", "in.mp4", "seq", None, None, CROP, {"bogus"})
    with pytest.raises(ValueError):
        swift_cmd("bin", "in.mp4", "seq", None, None, CROP, set())


def test_seek_times_sit_half_a_frame_inside_each_boundary():
    # Swift keeps frames overlapping [start, start + duration); ffmpeg keeps frames at or after -ss.
    start, duration = swift_range(3000, 12, 25)
    assert start == pytest.approx(120.02) and duration == pytest.approx(0.44)
    assert start < 3000.5 / 25 + 1e-9 and start + duration == pytest.approx((3000 + 11) / 25 + 0.02)
    assert ffmpeg_seek(3000, 25) == pytest.approx(119.98)
    assert ffmpeg_seek(0, 25) == 0.0


def test_ffmpeg_commands():
    assert prores_cmd("ep.mp4", 119.98, 20, "seq/tmp-1080p.mov") == [
        "ffmpeg", "-v", "error", "-y", "-ss", "119.980000", "-i", "ep.mp4", "-frames:v", "20", "-an",
        "-fps_mode", "passthrough", "-vf", "setpts=PTS-STARTPTS", "-c:v", "prores_ks", "-profile:v", "4444",
        "-pix_fmt", "yuv444p10le", "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
        "seq/tmp-1080p.mov"]
    assert x264_cmd("a.mov", "b.mp4", 25) == [
        "ffmpeg", "-v", "error", "-y", "-i", "a.mov", "-an", "-vf", "scale=640:360:flags=area",
        "-c:v", "libx264", "-preset", "medium", "-b:v", "576000", "-maxrate", "1152000", "-bufsize", "1152000",
        "-pix_fmt", "yuv420p", "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", "b.mp4"]


def test_held_out_episodes_are_never_used():
    used = " ".join(stem for stems in EPISODES.values() for stem in stems)
    assert "DO2104H003S00" not in used and "DO2303H001S00" not in used
    assert len(EPISODES["teacher"]) == len(EPISODES["gt"]) == 2


# ---------------------------------------------------------------- fake tools

FPS = 25.0
EPISODE_STEM = {"teacher": "Series 1 Arteries and Veins [DO2104H001S00]", "gt": "Series 1 Ep 2 [DO2303H002S00]"}


def make_spec(kind="teacher", start=100, frames=4, burn_in=3, bucket="slow", cuts=(), crop=CROP):
    return SeqSpec(EPISODE_STEM[kind], kind, start, frames, burn_in, bucket, list(cuts), crop)


class FakeTools:
    """Stands in for ffmpeg and the Swift CLI: parses the argv and writes dumps of the right shape."""

    def __init__(self, frame_delta=0, cut_at=None, fail_on=None, pts_shift=0):
        self.cmds = []
        self.frame_delta = frame_delta  # added to every dump's frame count (a seek that returns too many/few)
        self.pts_shift = pts_shift  # frames the excerpt's timestamps are off by (a seek that lands early/late)
        self.cut_at = cut_at
        self.fail_on = fail_on
        self.clip_frames = None

    def __call__(self, cmd, timeout):
        self.cmds.append(cmd)
        name = Path(cmd[0]).name
        if self.fail_on == name:
            raise BuildError(f"{name} exited with 1: boom")
        if name == "ffmpeg":
            if "-frames:v" in cmd:
                self.clip_frames = int(cmd[cmd.index("-frames:v") + 1])
            Path(cmd[-1]).write_bytes(b"clip")
            return
        Path(cmd[2]).write_bytes(b"video")  # the CLI always writes a video
        crop = tuple(int(v) for v in cmd[cmd.index("--dump-crop") + 1].split(","))
        if "--duration" in cmd:
            n = round(float(cmd[cmd.index("--duration") + 1]) * FPS) + 1
        else:
            n = self.clip_frames
        n += self.frame_delta
        if "--start" in cmd:  # the reader clamps the first frame to the range start; the rest sit on the frame grid
            start = float(cmd[cmd.index("--start") + 1])
            stamps = [start] + [(int(start * FPS + 1e-6) + k + self.pts_shift) / FPS for k in range(1, n)]
        else:
            stamps = [i / FPS for i in range(n)]
        dumps = [(a, Path(b)) for a, b in zip(cmd, cmd[1:]) if a.startswith("--dump-") and a != "--dump-crop"]
        for flag, dump_dir in dumps:
            kind = flag[len("--dump-"):]
            scale = 3 if kind == "base" or dump_dir.name == "target" else 1
            channels = 4 if kind == "flow" else 3
            w, h = crop[2] * (3 if kind == "base" else 1), crop[3] * (3 if kind == "base" else 1)
            writer = DumpWriter(dump_dir, "source" if dump_dir.name == "target" else kind, channels, w, h,
                                (64, 64), scale, crop)
            for i in range(n):
                level = 0.2 + (0.5 if self.cut_at is not None and i >= self.cut_at else 0.0)
                writer.write(np.full((channels, h, w), level, np.float16), stamps[i], valid=i > 0)
            writer.finish()


def episodes():
    return {EPISODE_STEM[k]: Episode(Path(f"/eps/{k}.mp4"), FPS, 0.08, 1280, 720) for k in EPISODE_STEM}


def make_builder(tmp_path, tools):
    return Builder(tmp_path / "out", "bin/metalfx-upscale", episodes(), run=tools)


def swift_calls(tools):
    return [c for c in tools.cmds if Path(c[0]).name != "ffmpeg"]


# ---------------------------------------------------------------- one sequence

def test_teacher_sequence_is_one_swift_run(tmp_path):
    tools = FakeTools()
    builder = make_builder(tmp_path, tools)
    spec = make_spec("teacher", start=100, frames=4, burn_in=3)
    assert builder.build(7, spec) == "built"
    seq = tmp_path / "out" / "teacher" / "00007"
    assert [Path(c[0]).name for c in tools.cmds] == ["metalfx-upscale"]
    cmd = tools.cmds[0]
    assert cmd[1] == "/eps/teacher.mp4"
    assert cmd[cmd.index("--start") + 1] == "3.900000"            # frame 97 (start - burn_in) at 97/25 + half a frame
    assert cmd[cmd.index("--duration") + 1] == "0.240000"         # 7 frames - 1 frame time
    assert cmd[cmd.index("--dump-crop") + 1] == "8,8,80,72"
    assert cmd[cmd.index("--size") + 1] == "4k"
    assert sorted(p.name for p in seq.iterdir()) == ["base", "flow", "source", "spec.json"]  # temp video deleted
    saved = json.loads((seq / "spec.json").read_text())
    assert saved["id"] == 7 and saved["start"] == 100 and saved["flow_invalid"] == [0]
    assert read_meta(seq / "source")["frames"] == 7
    assert is_complete(seq, spec_record(7, spec))


def test_gt_sequence_chain(tmp_path):
    tools = FakeTools()
    builder = make_builder(tmp_path, tools)
    spec = make_spec("gt", start=100, frames=4, burn_in=3)
    assert builder.build(2, spec) == "built"
    seq = tmp_path / "out" / "gt" / "00002"
    names = [Path(c[0]).name for c in tools.cmds]
    assert names == ["ffmpeg", "metalfx-upscale", "ffmpeg", "metalfx-upscale"]
    prores, target, x264, final = tools.cmds
    assert prores[prores.index("-ss") + 1] == "3.860000"  # 97/25 minus half a frame
    assert prores[prores.index("-frames:v") + 1] == "7"
    assert target[1].endswith("tmp-1080p.mov") and "--start" not in target
    assert target[target.index("--dump-source") + 1] == str(seq / "target")
    assert target[target.index("--dump-crop") + 1] == "120,120,48,24"   # crop x3, no margin
    assert "--dump-base" not in target and "--dump-flow" not in target
    assert x264[x264.index("-i") + 1].endswith("tmp-1080p.mov") and x264[-1].endswith("tmp-360p.mp4")
    assert final[1].endswith("tmp-360p.mp4") and final[final.index("--size") + 1] == "1920x1080"
    assert final[final.index("--dump-crop") + 1] == "8,8,80,72"
    assert sorted(p.name for p in seq.iterdir()) == ["base", "flow", "source", "spec.json", "target"]
    assert read_meta(seq / "target")["width"] == 48 and read_meta(seq / "target")["frames"] == 7
    assert is_complete(seq, spec_record(2, spec))


def test_wrong_frame_count_fails_and_leaves_nothing(tmp_path):
    builder = make_builder(tmp_path, FakeTools(frame_delta=1))
    with pytest.raises(BuildError, match="8 frames, expected 7"):
        builder.build(0, make_spec())
    assert not (tmp_path / "out" / "teacher" / "00000").exists()


def test_tool_failure_removes_the_sequence_and_temp_files(tmp_path):
    builder = make_builder(tmp_path, FakeTools(fail_on="metalfx-upscale"))
    with pytest.raises(BuildError):
        builder.build(1, make_spec("gt"))
    assert not (tmp_path / "out" / "gt" / "00001").exists()


def test_first_frame_before_the_first_real_frame_is_refused(tmp_path):
    tools = FakeTools()
    builder = make_builder(tmp_path, tools)
    with pytest.raises(BuildError, match="first real frame"):
        builder.build(0, make_spec(start=3, burn_in=2))  # first frame 1: the shot map's index 1 is padding
    assert tools.cmds == []


@pytest.mark.parametrize("shift", [-1, 1])
def test_teacher_excerpt_that_lands_off_the_shot_map_frame_is_refused(tmp_path, shift):
    builder = make_builder(tmp_path, FakeTools(pts_shift=shift))
    with pytest.raises(BuildError, match="shot-map frame 97"):
        builder.build(0, make_spec("teacher", start=100, burn_in=3))
    assert not (tmp_path / "out" / "teacher" / "00000").exists()


def test_check_pts_accepts_the_clamped_first_frame_and_rejects_slips(tmp_path):
    def dump(stamps):
        writer = DumpWriter(tmp_path / "s", "source", 3, 2, 2, (8, 8), 1, (0, 0, 2, 2))
        for t in stamps:
            writer.write(np.zeros((3, 2, 2), np.float16), t)
        writer.finish()

    # shot-map frames 97, 98, 99, 100 sit at 3.88, 3.92, 3.96, 4.00 s; the reader stamps the first one with the range
    # start (3.90) instead of its own time
    dump([3.90, 3.92, 3.96, 4.00])
    check_pts(tmp_path / "s", 97, 25.0)
    dump([3.88, 3.92, 3.96, 4.00])
    check_pts(tmp_path / "s", 97, 25.0)
    for slipped in ([3.86, 3.88, 3.92, 3.96], [3.94, 3.96, 4.00, 4.04]):  # one frame early, one frame late
        dump(slipped)
        with pytest.raises(BuildError, match="frame 97"):
            check_pts(tmp_path / "s", 97, 25.0)


def test_cut_must_sit_at_the_frame_difference_peak(tmp_path):
    spec = make_spec(start=100, frames=6, burn_in=2, bucket="straddle", cuts=[5])
    assert make_builder(tmp_path, FakeTools(cut_at=5)).build(0, spec) == "built"
    with pytest.raises(BuildError, match="misaligned"):
        make_builder(tmp_path / "other", FakeTools(cut_at=4)).build(0, spec)  # one frame early
    with pytest.raises(BuildError, match="misaligned"):
        make_builder(tmp_path / "other2", FakeTools(cut_at=6)).build(0, spec)  # one frame late


# ---------------------------------------------------------------- resuming

def test_complete_sequence_is_skipped(tmp_path):
    tools = FakeTools()
    builder = make_builder(tmp_path, tools)
    spec = make_spec()
    assert builder.build(0, spec) == "built"
    calls = len(tools.cmds)
    assert builder.build(0, spec) == "skipped"
    assert len(tools.cmds) == calls


@pytest.mark.parametrize("damage", ["no_flow_meta", "short_flow", "no_spec", "other_spec", "no_target", "unchecked"])
def test_incomplete_or_stale_sequence_is_rebuilt(tmp_path, damage):
    kind = "gt" if damage == "no_target" else "teacher"
    tools = FakeTools()
    builder = make_builder(tmp_path, tools)
    spec = make_spec(kind)
    builder.build(0, spec)
    seq = tmp_path / "out" / kind / "00000"
    (seq / "marker.txt").write_text("from the first build")
    if damage == "no_flow_meta":
        (seq / "flow" / "meta.json").unlink()
    elif damage == "short_flow":
        meta = json.loads((seq / "flow" / "meta.json").read_text())
        meta["frames"] -= 1
        (seq / "flow" / "meta.json").write_text(json.dumps(meta))
    elif damage == "no_spec":
        (seq / "spec.json").unlink()
    elif damage == "unchecked":  # the CLI finished but the run died before the cut/timestamp checks passed
        saved = json.loads((seq / "spec.json").read_text())
        del saved["build_seconds"]
        (seq / "spec.json").write_text(json.dumps(saved))
    elif damage == "other_spec":  # same id, different sampling (another seed)
        saved = json.loads((seq / "spec.json").read_text())
        saved["start"] += 1
        (seq / "spec.json").write_text(json.dumps(saved))
    else:
        (seq / "target" / "meta.json").unlink()
    before = len(tools.cmds)
    assert builder.build(0, spec) == "built"
    assert len(tools.cmds) > before
    assert not (seq / "marker.txt").exists()  # the partial directory was deleted first
    assert is_complete(seq, spec_record(0, spec))


def add_teacher_targets(seq):
    """A teacher/ dump next to the build's dumps, as detailnet/teacher.py leaves it."""
    writer = DumpWriter(seq / "teacher", "teacher", 3, 6, 4, (1280, 720), 3, CROP)
    writer.write(np.zeros((3, 4, 6), np.float16), 0.0)
    writer.finish()


def make_stale_with_teacher(tmp_path, tools, builder=None):
    """Sequence 0 built, given teacher targets, then its spec.json changed (another seed's sampling)."""
    builder = builder or make_builder(tmp_path, tools)
    spec = make_spec()
    builder.build(0, spec)
    seq = tmp_path / "out" / "teacher" / "00000"
    add_teacher_targets(seq)
    saved = json.loads((seq / "spec.json").read_text())
    saved["start"] += 1
    (seq / "spec.json").write_text(json.dumps(saved))
    return builder, spec, seq


def test_stale_sequence_with_teacher_targets_is_kept(tmp_path):
    tools = FakeTools()
    builder, spec, seq = make_stale_with_teacher(tmp_path, tools)
    spec_before = (seq / "spec.json").read_text()
    calls = len(tools.cmds)
    with pytest.raises(StaleKept, match="teacher"):
        builder.build(0, spec)
    assert len(tools.cmds) == calls  # nothing was run
    assert (seq / "teacher" / "meta.json").exists() and (seq / "source" / "meta.json").exists()
    assert (seq / "spec.json").read_text() == spec_before


def test_rebuild_flag_replaces_a_stale_sequence_that_has_teacher_targets(tmp_path):
    tools = FakeTools()
    builder = make_builder(tmp_path, tools)
    builder.rebuild = True
    builder, spec, seq = make_stale_with_teacher(tmp_path, tools, builder)
    calls = len(tools.cmds)
    assert builder.build(0, spec) == "built"
    assert len(tools.cmds) > calls
    assert not (seq / "teacher").exists()  # the old targets belonged to the old sampling
    assert is_complete(seq, spec_record(0, spec))


def test_stale_sequence_without_teacher_targets_is_still_rebuilt_by_default(tmp_path):
    tools = FakeTools()
    builder = make_builder(tmp_path, tools)
    spec = make_spec()
    builder.build(0, spec)
    seq = tmp_path / "out" / "teacher" / "00000"
    saved = json.loads((seq / "spec.json").read_text())
    saved["start"] += 1
    (seq / "spec.json").write_text(json.dumps(saved))
    assert builder.build(0, spec) == "built"
    assert is_complete(seq, spec_record(0, spec))


def test_check_metas_reports_wrong_shapes(tmp_path):
    spec = make_spec()
    FakeTools()(["bin/metalfx-upscale", "in.mp4", str(tmp_path / "tmp-upscale.mp4"), "--start", "1", "--duration",
                 "0.24", "--dump-source", str(tmp_path / "source"), "--dump-base", str(tmp_path / "base"),
                 "--dump-flow", str(tmp_path / "flow"), "--dump-crop", "8,8,80,72"], 1)
    assert check_metas(tmp_path, spec_record(0, spec)) == [0]
    other = spec_record(0, make_spec(crop=(40, 40, 20, 8)))  # a different window than was dumped
    with pytest.raises(BuildError, match="expected"):
        check_metas(tmp_path, other)


def test_check_cuts_ignores_sequences_without_cuts(tmp_path):
    check_cuts(tmp_path / "does-not-exist", [])


def test_check_cuts_on_a_real_dump(tmp_path):
    writer = DumpWriter(tmp_path / "source", "source", 3, 4, 4, (8, 8), 1, (0, 0, 4, 4))
    for level in (0.1, 0.1, 0.1, 0.6, 0.6):
        writer.write(np.full((3, 4, 4), level, np.float16), 0.0)
    writer.finish()
    check_cuts(tmp_path / "source", [3])
    with pytest.raises(BuildError):
        check_cuts(tmp_path / "source", [2])


# ---------------------------------------------------------------- the run

class Guard:
    def __init__(self, trip_after=None):
        self.calls, self.trip_after = 0, trip_after

    def check(self):
        self.calls += 1
        if self.trip_after is not None and self.calls > self.trip_after:
            raise SwapLimitExceeded("swap grew by 9.0 GB")


def specs_by_kind(teacher=3, gt=2):
    return {"teacher": [make_spec("teacher", start=100 + 20 * i) for i in range(teacher)],
            "gt": [make_spec("gt", start=100 + 20 * i) for i in range(gt)]}


def test_run_all_logs_one_line_per_sequence_and_a_summary(tmp_path):
    lines = []
    code, counts = run_all(make_builder(tmp_path, FakeTools()), specs_by_kind(), None, Guard(), lines.append)
    assert code == 0 and counts == {"built": 5, "skipped": 0, "failed": 0}
    assert len(lines) == 6
    assert lines[0].startswith("[   1/5] teacher 00000 DO2104H001S00 start=100 burn_in=3 frames=4")
    assert "BUILT" in lines[0] and "s |" in lines[0]
    assert lines[-1].startswith("SUMMARY ok: built 5 skipped 0 failed 0 of 5 planned in ")
    # a second run skips everything
    lines.clear()
    code, counts = run_all(make_builder(tmp_path, FakeTools()), specs_by_kind(), None, Guard(), lines.append)
    assert counts == {"built": 0, "skipped": 5, "failed": 0} and code == 0
    assert "SKIPPED" in lines[0]


def test_run_all_limit_is_per_kind(tmp_path):
    code, counts = run_all(make_builder(tmp_path, FakeTools()), specs_by_kind(), 1, Guard(), lambda line: None)
    assert counts["built"] == 2
    assert (tmp_path / "out" / "teacher" / "00000").exists() and (tmp_path / "out" / "gt" / "00000").exists()
    assert not (tmp_path / "out" / "teacher" / "00001").exists()


def test_swap_guard_stops_the_run_with_exit_3(tmp_path):
    lines = []
    code, counts = run_all(make_builder(tmp_path, FakeTools()), specs_by_kind(), None, Guard(trip_after=2),
                           lines.append)
    assert code == 3 and counts["built"] == 2
    assert any("swap guard" in line for line in lines)
    assert lines[-1].startswith("SUMMARY incomplete") and lines[-1].endswith("exit 3")


def test_failures_are_logged_and_exit_1_but_do_not_stop_the_run(tmp_path):
    specs = specs_by_kind(teacher=2, gt=0)
    tools = FakeTools(frame_delta=1)  # every sequence has the wrong number of frames
    lines = []
    code, counts = run_all(make_builder(tmp_path, tools), specs, None, Guard(), lines.append)
    assert code == 1 and counts == {"built": 0, "skipped": 0, "failed": 2}
    assert "FAILED" in lines[0] and "expected 7" in lines[0]


def test_five_failures_in_a_row_stop_the_run(tmp_path):
    specs = specs_by_kind(teacher=8, gt=0)
    lines = []
    code, counts = run_all(make_builder(tmp_path, FakeTools(frame_delta=1)), specs, None, Guard(), lines.append)
    assert code == 1 and counts["failed"] == 5
    assert any("5 failures in a row" in line for line in lines)


def test_run_all_logs_stale_kept_counts_it_as_failed_and_continues(tmp_path):
    specs = specs_by_kind(teacher=8, gt=0)  # more than the five-failures-in-a-row limit
    tools = FakeTools()
    builder = make_builder(tmp_path, tools)
    run_all(builder, specs, None, Guard(), lambda line: None)
    for i in range(1, 7):  # sequences 1..6 get teacher targets and a stale spec
        seq = builder.seq_dir("teacher", i)
        add_teacher_targets(seq)
        saved = json.loads((seq / "spec.json").read_text())
        saved["start"] += 1
        (seq / "spec.json").write_text(json.dumps(saved))
    lines = []
    code, counts = run_all(builder, specs, None, Guard(), lines.append)
    assert counts == {"built": 0, "skipped": 2, "failed": 6}  # the run went on past the sixth to the last
    assert code == 1
    assert sum("STALE-KEPT" in line for line in lines) == 6
    assert not any("in a row" in line for line in lines)
    assert all((builder.seq_dir("teacher", i) / "teacher" / "meta.json").exists() for i in range(1, 7))


def test_main_rebuild_flag_reaches_the_builder(tmp_path, monkeypatch):
    binary = tmp_path / "metalfx-upscale"
    binary.write_text("")
    monkeypatch.setattr(build_data, "plan", lambda kind, n, seed, shots: specs_by_kind()[kind])
    monkeypatch.setattr(build_data, "load_episodes", lambda d, kind: {})
    seen = []
    monkeypatch.setattr(build_data, "run_all", lambda builder, *a, **k: seen.append(builder) or (0, {}))
    args = ["--out", str(tmp_path / "out"), "--teacher-seqs", "3", "--gt-seqs", "2", "--binary", str(binary)]
    build_data.main(args)
    build_data.main(args + ["--rebuild"])
    assert [b.rebuild for b in seen] == [False, True]


def test_main_exit_code_3_when_swap_grows(tmp_path, monkeypatch):
    binary = tmp_path / "metalfx-upscale"
    binary.write_text("")
    monkeypatch.setattr(build_data, "plan", lambda kind, n, seed, shots: specs_by_kind()[kind])
    monkeypatch.setattr(build_data, "load_episodes", lambda d, kind: {})
    used = iter([1.0, 20.0, 20.0, 20.0])
    monkeypatch.setattr(build_data, "SwapGuard", lambda limit: SwapGuard(limit, read=lambda: next(used)))
    code = build_data.main(["--out", str(tmp_path / "out"), "--teacher-seqs", "3", "--gt-seqs", "2",
                            "--binary", str(binary)])
    assert code == 3
    listing = json.loads((tmp_path / "out" / "sequences.json").read_text())
    assert [(e["kind"], e["id"]) for e in listing] == [("teacher", 0), ("teacher", 1), ("teacher", 2),
                                                       ("gt", 0), ("gt", 1)]
