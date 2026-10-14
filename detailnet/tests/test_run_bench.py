"""DetailNet in the bench: dump planning (bench_dumps), the runner (run_bench), the pulse metric, the gt3 clip kind.
Everything runs on the CPU with synthetic clip directories and dumps, never on output/."""
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from detailnet import bench_dumps, run_bench
from detailnet.build_data import BuildError
from detailnet.dumps import DumpWriter, read_meta
from detailnet.model import DetailNet

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bench"))
import make_clips  # noqa: E402
import metrics  # noqa: E402
from clipio import read_frames  # noqa: E402

H, W = 12, 16  # synthetic source size (px); the x3 base is 36 x 48


@pytest.fixture(autouse=True)
def seeded():
    torch.manual_seed(0)


def small_net(boost=False):
    net = DetailNet(scale=3, channels=8, blocks=2, hidden=4).eval()
    if boost:  # the untrained texture head is ~0 on purpose; make the detail visible in 8 bits
        with torch.no_grad():
            net.head_tex.weight.mul_(100.0)
    return net


# ---------------------------------------------------------------- synthetic clip + dumps

def synth(n, h=H, w=W, seed=0):
    rng = np.random.default_rng(seed)
    return SimpleNamespace(
        x=rng.random((n, 3, h, w), dtype=np.float32).astype(np.float16),
        base=rng.random((n, 3, 3 * h, 3 * w), dtype=np.float32).astype(np.float16),
        flow=(rng.standard_normal((n, 4, h, w)) * 2).astype(np.float16))


def clip_spec(n, cuts=(), kind="gt3", h=H, w=W, target=None, name="t-clip"):
    return {"name": name, "kind": kind, "frames": n, "fps": 25.0, "cuts": list(cuts), "lr_size": [w, h],
            "target": list(target or (3 * w, 3 * h))}


def write_clip(clip_dir, data, spec, crop_size=bench_dumps.CROP, invalid=()):
    """clip.json plus per-shot dumps the way bench_dumps lays them out. As in the Swift dumps, frame 0 of every shot
    is marked invalid (the stored flow there is junk on purpose: the runner has to ignore it); `invalid` adds more
    invalid frames (global indices) whose flow is NaN."""
    clip_dir = Path(clip_dir)
    clip_dir.mkdir(parents=True, exist_ok=True)
    (clip_dir / "clip.json").write_text(json.dumps(spec))
    plan = bench_dumps.dump_plan(spec, crop_size)
    x0, y0, cw, ch = plan.crop
    for k, (a, b) in enumerate(plan.shots):
        d = clip_dir / "dumps" / f"shot{k}"
        writers = {
            "source": DumpWriter(d / "source", "source", 3, cw, ch, spec["lr_size"], 3, plan.crop),
            "base": DumpWriter(d / "base", "base", 3, 3 * cw, 3 * ch, spec["lr_size"], 3, plan.crop),
            "flow": DumpWriter(d / "flow", "flow", 4, cw, ch, spec["lr_size"], 3, plan.crop,
                               extra={"flow_channels": ["back_x", "back_y", "fwd_x", "fwd_y"]})}
        for t in range(a, b):
            valid = t != a and t not in invalid
            flow = np.full_like(data.flow[t], np.nan) if t in invalid else data.flow[t]
            writers["source"].write(data.x[t], t / 25, valid)
            writers["base"].write(data.base[t], t / 25, valid)
            writers["flow"].write(flow, t / 25, valid)
        for w_ in writers.values():
            w_.finish()
    return plan


def sub_data(data, a, b):
    return SimpleNamespace(x=data.x[a:b], base=data.base[a:b], flow=data.flow[a:b])


def run_frames(net, clip_dir, **kwargs):
    kwargs.setdefault("device", "cpu")
    return {t: img for t, img, _ in run_bench.iter_outputs(net, clip_dir, **kwargs)}


def to_u8(chw):
    return np.rint(np.clip(chw.astype(np.float32), 0, 1) * 255).astype(np.uint8).transpose(1, 2, 0)


def tensor(a):
    return torch.from_numpy(np.asarray(a, np.float32))[None]


def reference_run(net, data, resets=(), zero_flow_only=()):
    """The spec written out as a plain loop: at a reset the state is zero, the flow zero and x_prev = x_t; at a
    `zero_flow_only` frame the state is carried and x_prev is the previous frame, but the flow is zero."""
    n = len(data.x)
    h, w = data.x.shape[-2:]
    prev_detail, hidden = net.zero_state(1, h, w, "cpu")
    outs = []
    with torch.no_grad():
        for t in range(n):
            x_t, base = tensor(data.x[t]), tensor(data.base[t])
            flow = tensor(np.nan_to_num(data.flow[t].astype(np.float32)).clip(-64, 64))
            x_prev = tensor(data.x[t - 1]) if t else x_t
            if t == 0 or t in resets:
                prev_detail, hidden = net.zero_state(1, h, w, "cpu")
                flow, x_prev = torch.zeros_like(flow), x_t
            elif t in zero_flow_only:
                flow = torch.zeros_like(flow)
            detail, hidden = net(x_t, x_prev, base, flow, prev_detail, hidden)
            prev_detail = detail
            outs.append(to_u8((base + detail).clamp(0, 1)[0].numpy()))
    return outs


def max_diff(a, b):
    return int(np.abs(a.astype(np.int16) - b.astype(np.int16)).max())


# ---------------------------------------------------------------- run_bench: strength

def test_strength_zero_equals_base(tmp_path):
    data = synth(4)
    write_clip(tmp_path, data, clip_spec(4))
    out = run_frames(DetailNet(scale=3, channels=8, blocks=2, hidden=4).eval(), tmp_path, strength=0.0)
    assert sorted(out) == [0, 1, 2, 3]
    for t, img in out.items():
        assert img.shape == (3 * H, 3 * W, 3) and img.dtype == np.uint8
        assert max_diff(img, to_u8(data.base[t])) <= 1


def test_strength_zero_equals_base_crop_with_margin_cut(tmp_path):
    spec = clip_spec(3, kind="real", h=82, w=96, target=(3840, 2160))
    spec["lr_size"] = [1280, 720]  # a 720p clip; the dumped LR box is the crop plan's (96 x 82 for a 96 x 54 output)
    data = synth(3, h=82, w=96)
    plan = write_clip(tmp_path, data, spec, crop_size=(96, 54))
    assert plan.crop == (592, 319, 96, 82)
    out = run_frames(small_net(), tmp_path, strength=0.0, crop_size=(96, 54))
    for t, img in out.items():
        assert img.shape == (54, 96, 3)
        want = to_u8(data.base[t][:, 96:-96, 96:-96])
        assert max_diff(img, want) <= 1


def test_untrained_net_nearly_leaves_the_base_alone(tmp_path):
    data = synth(3)
    write_clip(tmp_path, data, clip_spec(3))
    out = run_frames(DetailNet(scale=3, channels=8, blocks=2, hidden=4).eval(), tmp_path, strength=1.0)
    assert all(max_diff(img, to_u8(data.base[t])) <= 2 for t, img in out.items())


def test_strength_scales_the_detail_but_not_the_recurrence(tmp_path):
    data = synth(5)
    write_clip(tmp_path, data, clip_spec(5))
    net = small_net(boost=True)
    full, half, zero = (run_frames(net, tmp_path, strength=s) for s in (1.0, 0.5, 0.0))
    for t in range(5):
        assert max_diff(full[t], zero[t]) > 5  # the boosted net really changes the picture
        mid = (full[t].astype(np.float32) + zero[t]) / 2
        unclamped = (full[t] > 0) & (full[t] < 255)
        assert unclamped.mean() > 0.3
        assert np.abs(half[t].astype(np.float32) - mid)[unclamped].max() <= 1  # linear in strength until clamped
    # the recurrent state is the raw detail: output t at strength 0.5 comes from the strength-1 recurrence
    ref = reference_run(net, data)
    base_t = [to_u8(data.base[t]) for t in range(5)]
    for t in range(5):
        assert max_diff(full[t], ref[t]) <= 1
        assert max_diff(zero[t], base_t[t]) <= 1


# ---------------------------------------------------------------- run_bench: resets

def test_resets_at_cuts(tmp_path):
    data = synth(7)
    write_clip(tmp_path / "a", data, clip_spec(7, cuts=[3]))
    write_clip(tmp_path / "b", sub_data(data, 3, 7), clip_spec(4))  # a fresh run that starts at frame 3
    net = small_net(boost=True)
    cut, fresh = run_frames(net, tmp_path / "a"), run_frames(net, tmp_path / "b")
    for t in range(3, 7):
        assert np.array_equal(cut[t], fresh[t - 3]), t
    # the test has teeth: carrying the state across the cut changes frame 3
    carried = run_frames(net, tmp_path / "a", ignore_cuts=True)
    assert max_diff(carried[3], fresh[0]) > 2
    assert all(np.array_equal(carried[t], cut[t]) for t in range(3))


@pytest.mark.parametrize("bad", [5])
def test_resets_at_invalid_flow(tmp_path, bad):
    data = synth(8)
    write_clip(tmp_path / "a", data, clip_spec(8), invalid={bad})
    write_clip(tmp_path / "b", sub_data(data, bad, 8), clip_spec(8 - bad))
    net = small_net(boost=True)
    full, fresh = run_frames(net, tmp_path / "a"), run_frames(net, tmp_path / "b")
    for t in range(bad, 8):
        assert np.array_equal(full[t], fresh[t - bad]), t
    ref = reference_run(net, data, resets={bad})
    assert all(max_diff(full[t], ref[t]) <= 1 for t in range(8))


def test_ignore_cuts_carries_state_with_zero_flow_across_shot_starts(tmp_path):
    data = synth(8)
    write_clip(tmp_path, data, clip_spec(8, cuts=[3, 6]))
    net = small_net(boost=True)
    out = run_frames(net, tmp_path, ignore_cuts=True)
    ref = reference_run(net, data, zero_flow_only={3, 6})
    assert all(max_diff(out[t], ref[t]) <= 1 for t in range(8))
    plain = reference_run(net, data, resets={3, 6})
    assert max_diff(out[3], plain[3]) > 2


def test_invalid_flow_still_resets_with_ignore_cuts(tmp_path):
    data = synth(8)
    write_clip(tmp_path, data, clip_spec(8, cuts=[3]), invalid={5})
    net = small_net(boost=True)
    out = run_frames(net, tmp_path, ignore_cuts=True)
    ref = reference_run(net, data, resets={5}, zero_flow_only={3})
    assert all(max_diff(out[t], ref[t]) <= 1 for t in range(8))


def test_flow_is_clamped_like_training(tmp_path):
    data = synth(4)
    data.flow[2] = 500.0  # absurd (flat-frame noise); training clamps to +-64
    write_clip(tmp_path, data, clip_spec(4))
    net = small_net(boost=True)
    out = run_frames(net, tmp_path)
    ref = reference_run(net, data)
    assert all(max_diff(out[t], ref[t]) <= 1 for t in range(4))


# ---------------------------------------------------------------- run_bench: files and stats

def save_ckpt(path, net):
    torch.save({"model": net.state_dict(), "cfg": {"channels": 8, "blocks": 2, "hidden": 4}}, path)


def test_main_writes_output_and_stats(tmp_path):
    data = synth(4)
    write_clip(tmp_path / "t-clip", data, clip_spec(4, cuts=[2]))
    ckpt = tmp_path / "net.pt"
    net = small_net(boost=True)
    save_ckpt(ckpt, net)
    run_bench.main(["--ckpt", str(ckpt), "--name", "dn", "--device", "cpu", str(tmp_path / "t-clip")])
    run_bench.main(["--ckpt", str(ckpt), "--name", "dn", "--device", "cpu", "--ignore-cuts", str(tmp_path / "t-clip")])

    out = tmp_path / "t-clip" / "out"
    assert sorted(p.name for p in out.iterdir()) == ["dn-nocut.json", "dn-nocut.mkv", "dn.json", "dn.mkv"]
    stats = json.loads((out / "dn.json").read_text())
    assert stats["model"] == "dn" and stats["clip"] == "t-clip" and stats["ignore_cuts"] is False
    assert stats["frames"] == 4 and stats["center_crop"] is False and stats["out_size"] == [3 * W, 3 * H]
    for key in ("label", "seconds", "s_per_frame", "peak_gpu_gb", "note", "frame_pixels"):
        assert key in stats
    assert stats["frame_pixels"] == 16 * W * H  # 4x-equivalent pixels of the processed region, as run_model counts
    nocut = json.loads((out / "dn-nocut.json").read_text())
    assert nocut["model"] == "dn" and nocut["ignore_cuts"] is True

    frames = list(read_frames(out / "dn.mkv", 3 * W, 3 * H))
    assert len(frames) == 4
    again = run_frames(net, tmp_path / "t-clip")
    assert all(np.array_equal(frames[t], again[t]) for t in range(4))
    carried = list(read_frames(out / "dn-nocut.mkv", 3 * W, 3 * H))
    assert not np.array_equal(carried[2], frames[2]) and np.array_equal(carried[1], frames[1])


def test_center_crop_stats(tmp_path):
    spec = clip_spec(3, kind="real", h=82, w=96, target=(3840, 2160))
    spec["lr_size"] = [1280, 720]
    write_clip(tmp_path, synth(3, h=82, w=96), spec, crop_size=(96, 54))
    stats = run_bench.run_clip(small_net(), tmp_path, "dn", strength=1.0, ignore_cuts=False, device="cpu",
                               crop_size=(96, 54))
    assert stats["center_crop"] is True and stats["out_size"] == [96, 54]
    assert stats["frame_pixels"] == 16 * 96 * 82
    assert len(list(read_frames(tmp_path / "out" / "dn.mkv", 96, 54))) == 3


def test_missing_or_mismatched_dumps_fail_clearly(tmp_path):
    data = synth(4)
    write_clip(tmp_path, data, clip_spec(4, cuts=[2]))
    net = small_net()
    shutil.rmtree(tmp_path / "dumps" / "shot1")
    with pytest.raises(run_bench.DumpError, match="shot1.*bench_dumps"):
        run_frames(net, tmp_path)
    write_clip(tmp_path, data, clip_spec(4, cuts=[2]))
    spec = clip_spec(4, cuts=[2])
    spec["cuts"] = [3]  # clip.json now disagrees with the dumps' shot lengths
    (tmp_path / "clip.json").write_text(json.dumps(spec))
    with pytest.raises(run_bench.DumpError, match="shot0"):
        run_frames(net, tmp_path)


# ---------------------------------------------------------------- run_bench --base-only (the apple-base reference)

def test_base_only_is_the_strength_zero_composite_bit_for_bit(tmp_path):
    data = synth(4)
    write_clip(tmp_path, data, clip_spec(4, cuts=[2]))
    zero = run_frames(small_net(boost=True), tmp_path, strength=0.0)
    base = {t: img for t, img, _ in run_bench.iter_outputs(None, tmp_path, device="cpu")}
    assert sorted(base) == [0, 1, 2, 3]
    assert all(np.array_equal(base[t], zero[t]) for t in range(4))
    assert all(max_diff(base[t], to_u8(data.base[t])) <= 1 for t in range(4))


def test_base_only_cuts_the_same_margin(tmp_path):
    spec = clip_spec(3, kind="real", h=82, w=96, target=(3840, 2160))
    spec["lr_size"] = [1280, 720]
    data = synth(3, h=82, w=96)
    write_clip(tmp_path, data, spec, crop_size=(96, 54))
    zero = run_frames(small_net(), tmp_path, strength=0.0, crop_size=(96, 54))
    base = {t: img for t, img, _ in run_bench.iter_outputs(None, tmp_path, device="cpu", crop_size=(96, 54))}
    assert all(base[t].shape == (54, 96, 3) and np.array_equal(base[t], zero[t]) for t in range(3))


def test_base_only_main_needs_no_checkpoint_and_writes_the_stats_format(tmp_path):
    data = synth(4)
    write_clip(tmp_path / "t-clip", data, clip_spec(4, cuts=[2]))
    clip = str(tmp_path / "t-clip")
    assert run_bench.main(["--base-only", "--name", "apple-base", clip]) == 0
    assert run_bench.main(["--base-only", "--name", "apple-base", "--ignore-cuts", clip]) == 0
    out = tmp_path / "t-clip" / "out"
    assert sorted(p.name for p in out.iterdir()) == ["apple-base-nocut.json", "apple-base-nocut.mkv",
                                                     "apple-base.json", "apple-base.mkv"]
    stats = json.loads((out / "apple-base.json").read_text())
    assert stats["model"] == "apple-base" and stats["clip"] == "t-clip" and stats["strength"] == 0.0
    assert stats["ckpt"] is None and stats["ignore_cuts"] is False and stats["frames"] == 4
    assert stats["center_crop"] is False and stats["out_size"] == [3 * W, 3 * H]
    assert stats["frame_pixels"] == 16 * W * H and stats["peak_gpu_gb"] is None
    for key in ("label", "seconds", "s_per_frame", "note"):
        assert key in stats
    assert json.loads((out / "apple-base-nocut.json").read_text())["ignore_cuts"] is True
    frames = list(read_frames(out / "apple-base.mkv", 3 * W, 3 * H))
    carried = list(read_frames(out / "apple-base-nocut.mkv", 3 * W, 3 * H))  # no state: ignoring cuts changes nothing
    assert len(frames) == 4 and all(np.array_equal(frames[t], carried[t]) for t in range(4))
    assert all(max_diff(frames[t], to_u8(data.base[t])) <= 1 for t in range(4))


def test_checkpoint_is_required_exactly_when_not_base_only(tmp_path, capsys):
    write_clip(tmp_path / "t-clip", synth(2), clip_spec(2))
    clip = str(tmp_path / "t-clip")
    ckpt = tmp_path / "net.pt"
    save_ckpt(ckpt, small_net())
    with pytest.raises(SystemExit):
        run_bench.main(["--name", "dn", clip])
    with pytest.raises(SystemExit):
        run_bench.main(["--base-only", "--ckpt", str(ckpt), "--name", "apple-base", clip])
    err = capsys.readouterr().err
    assert "--ckpt" in err and "--base-only" in err


# ---------------------------------------------------------------- bench_dumps

def real_spec(frames=150, cuts=(75,)):
    return {"name": "c-cut-nest", "kind": "real", "frames": frames, "fps": 25.0, "cuts": list(cuts),
            "lr_size": [1280, 720], "target": [3840, 2160]}


def test_dump_plan_real_clip_uses_crop_plan_and_first_100_frames():
    plan = bench_dumps.dump_plan(real_spec())
    assert plan.crop == (288, 148, 704, 424) and plan.scale == 3 and plan.size == "3840x2160"
    assert plan.frames == 100 and plan.shots == [(0, 75), (75, 100)]
    plan = bench_dumps.dump_plan(real_spec(frames=200, cuts=()))
    assert plan.shots == [(0, 100)]
    assert bench_dumps.dump_plan(real_spec(frames=60, cuts=())).shots == [(0, 60)]
    assert bench_dumps.dump_plan(real_spec(frames=150, cuts=(120,))).shots == [(0, 100)]  # cut outside the scored span


def test_dump_plan_gt3_clip_is_the_full_frame():
    spec = {"name": "g3-cut-gorge", "kind": "gt3", "frames": 100, "fps": 25.0, "cuts": [40],
            "lr_size": [640, 360], "target": [1920, 1080]}
    plan = bench_dumps.dump_plan(spec)
    assert plan.crop == (0, 0, 640, 360) and plan.scale == 3 and plan.size == "1920x1080"
    assert plan.shots == [(0, 40), (40, 100)] and plan.frames == 100


def test_prores_cmd_matches_run_apple():
    cmd = bench_dumps.prores_cmd(Path("clip/lr.mkv"), 75, 100, Path("tmp/shot.mov"))
    assert cmd == ["ffmpeg", "-v", "error", "-y", "-i", "clip/lr.mkv", "-vf",
                   "select='between(n,75,99)',setpts=N/FRAME_RATE/TB", "-c:v", "prores_ks", "-profile:v", "4444",
                   "-pix_fmt", "yuv444p10le", "-color_primaries", "bt709", "-color_trc", "bt709",
                   "-colorspace", "bt709", "tmp/shot.mov"]


class FakeTools:
    """Stands in for ffmpeg and the Swift CLI: records the commands, writes dumps of the requested shape."""

    def __init__(self, fail_swift_on=None):
        self.calls = []
        self.fail_swift_on = fail_swift_on

    def __call__(self, cmd, timeout):
        self.calls.append(cmd)
        if Path(cmd[0]).name == "ffmpeg":
            Path(cmd[-1]).write_bytes(b"prores")
            return
        if self.fail_swift_on is not None and sum(Path(c[0]).name != "ffmpeg" for c in self.calls) == self.fail_swift_on:
            raise BuildError("metalfx-upscale exited with 1: boom")
        crop = [int(v) for v in cmd[cmd.index("--dump-crop") + 1].split(",")]
        frames = self.frames
        for kind, channels, scale in (("source", 3, 1), ("base", 3, 3), ("flow", 4, 1)):
            d = Path(cmd[cmd.index(f"--dump-{kind}") + 1])
            w = DumpWriter(d, kind, channels, crop[2] * scale, crop[3] * scale, (1280, 720), 3, crop)
            for i in range(frames):
                w.write(np.zeros((channels, crop[3] * scale, crop[2] * scale), np.float16), i / 25, valid=i > 0)
            w.finish()
        Path(cmd[2]).write_bytes(b"video")


def build(tmp_path, spec, tools, plan_frames):
    tools.frames = plan_frames
    (tmp_path / "clip.json").write_text(json.dumps(spec))
    return bench_dumps.build_clip(tmp_path, binary=Path("dev/metalfx-upscale"), run=tools)


def test_build_clip_writes_per_shot_dumps_and_is_resumable(tmp_path):
    spec = real_spec(frames=6, cuts=(2,))
    spec_small = {**spec, "frames": 6}

    class Tools(FakeTools):
        def __call__(self, cmd, timeout):
            if Path(cmd[0]).name != "ffmpeg":  # shots are 2 and 4 frames long
                self.frames = 2 if "shot0" in cmd[2] else 4
            super().__call__(cmd, timeout)

    tools = Tools()
    (tmp_path / "clip.json").write_text(json.dumps(spec_small))
    shots = bench_dumps.build_clip(tmp_path, binary=Path("dev/metalfx-upscale"), run=tools)
    assert shots == [(0, 2), (2, 6)]
    dumps = tmp_path / "dumps"
    assert sorted(p.name for p in dumps.iterdir()) == ["shot0", "shot1"]
    for k, frames in ((0, 2), (1, 4)):
        assert sorted(p.name for p in (dumps / f"shot{k}").iterdir()) == ["base", "flow", "source"]
        meta = read_meta(dumps / f"shot{k}" / "base")
        assert meta["frames"] == frames and meta["valid"][0] is False
    ffmpeg = [c for c in tools.calls if c[0] == "ffmpeg"]
    swift = [c for c in tools.calls if c[0] != "ffmpeg"]
    assert len(ffmpeg) == 2 and len(swift) == 2
    assert "select='between(n,2,5)',setpts=N/FRAME_RATE/TB" in ffmpeg[1]
    cmd = swift[1]
    assert cmd[0] == "dev/metalfx-upscale" and cmd[cmd.index("--engine") + 1] == "ml"
    assert cmd[cmd.index("--size") + 1] == "3840x2160" and cmd[cmd.index("--dump-crop") + 1] == "288,148,704,424"
    assert {"--dump-source", "--dump-base", "--dump-flow", "--overwrite"} <= set(cmd)
    assert "--start" not in cmd and "--duration" not in cmd

    before = len(tools.calls)  # finished shots are skipped
    assert bench_dumps.build_clip(tmp_path, binary=Path("dev/metalfx-upscale"), run=tools) == shots
    assert len(tools.calls) == before

    # a shot with the wrong number of frames is rebuilt, and nothing but the three dumps is left in a shot directory
    (dumps / "shot1" / "base" / "meta.json").write_text(json.dumps({**read_meta(dumps / "shot1" / "base"), "frames": 3}))
    bench_dumps.build_clip(tmp_path, binary=Path("dev/metalfx-upscale"), run=tools)
    assert len(tools.calls) == before + 2
    assert sorted(p.name for p in (dumps / "shot1").iterdir()) == ["base", "flow", "source"]


def test_failed_shot_leaves_no_shot_directory(tmp_path):
    spec = real_spec(frames=6, cuts=(2,))
    tools = FakeTools(fail_swift_on=2)
    tools.frames = 2
    (tmp_path / "clip.json").write_text(json.dumps(spec))
    with pytest.raises(BuildError, match="boom"):
        bench_dumps.build_clip(tmp_path, binary=Path("dev/metalfx-upscale"), run=tools)
    assert sorted(p.name for p in (tmp_path / "dumps").iterdir() if not p.name.startswith(".")) == ["shot0"]
    assert not (tmp_path / "dumps" / "shot1").exists()


def test_wrong_frame_count_from_the_cli_is_an_error(tmp_path):
    tools = FakeTools()
    tools.frames = 3  # the clip's first shot is 2 frames
    (tmp_path / "clip.json").write_text(json.dumps(real_spec(frames=6, cuts=(2,))))
    with pytest.raises(BuildError, match="shot0.*3 frames, expected 2"):
        bench_dumps.build_clip(tmp_path, binary=Path("dev/metalfx-upscale"), run=tools)


# ---------------------------------------------------------------- pulse metric

def test_pulse_detects_period4():
    pairs = [(t, 2.0 if t % 4 == 0 else 1.0) for t in range(1, 41)]  # every 4th pair twice as jittery
    phase = metrics.jitter_phase(pairs)
    assert phase == [2.0, 1.0, 1.0, 1.0]
    assert metrics.pulse_of(phase) > 1.5
    flat = metrics.jitter_phase([(t, 1.0) for t in range(1, 41)])
    assert flat == [1.0] * 4 and metrics.pulse_of(flat) < 1.05


def test_pulse_ignores_empty_phases():
    phase = metrics.jitter_phase([(1, 1.0), (2, 3.0), (5, 3.0)])  # no pair has t % 4 in {0, 3}
    assert phase == [None, 2.0, 3.0, None]
    assert metrics.pulse_of(phase) == pytest.approx(3.0 / 2.5)
    assert metrics.jitter_phase([]) == [None] * 4 and metrics.pulse_of([None] * 4) is None


def test_evaluate_reports_jitter_phase_and_pulse(tmp_path):
    """End to end on a tiny synthetic clip: a smooth texture drifting 1 px a frame; the model output either follows
    it or also flashes every 4th frame. Existing keys stay, the new ones appear, and the flashing output pulses."""
    from clipio import FFV1Writer
    n, w, h = 17, 160, 120
    rng = np.random.default_rng(1)
    texture = (rng.random((h, w + n)).astype(np.float32))
    texture = np.stack([np.convolve(r, np.ones(5) / 5, mode="same") for r in texture])
    base = [np.repeat(np.clip(texture[:, t:t + w, None] * 255, 0, 255), 3, axis=2).astype(np.uint8) for t in range(n)]
    clip = tmp_path / "c-synth"
    (clip / "out").mkdir(parents=True)
    (clip / "clip.json").write_text(json.dumps({"name": "c-synth", "kind": "real", "frames": n, "fps": 25.0, "cuts": [],
                                                "lr_size": [w, h], "target": [w, h]}))
    steady = [(f.astype(np.float32) * 0.8 + 5).astype(np.uint8) for f in base]  # differs from the bicubic, coherently
    flash = [np.clip(f.astype(np.int16) + (40 if t % 4 == 0 else 0), 0, 255).astype(np.uint8)
             for t, f in enumerate(steady)]
    for path, frames in ((clip / "lr.mkv", base), (clip / "out" / "steady.mkv", steady),
                         (clip / "out" / "flash.mkv", flash)):
        wr = FFV1Writer(path, w, h, 25)
        for f in frames:
            wr.write(f)
        wr.close()
    results = {m: metrics.evaluate(clip, m) for m in ("steady", "flash")}
    for res in results.values():
        assert {"detail", "flicker", "jitter", "jitter_all", "flicker_pairs"} <= set(res)
        assert len(res["jitter_phase"]) == 4 and "pulse" in res
    assert results["steady"]["pulse"] < 1.15
    assert results["flash"]["pulse"] > results["steady"]["pulse"] + 0.2


# ---------------------------------------------------------------- make_clips: gt3

def test_gt3_clip_kind(tmp_path, monkeypatch):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=1920x1080:rate=25",
                    "-frames:v", "8", "-pix_fmt", "yuv420p", str(tmp_path / "src.mp4")], check=True)
    shots = tmp_path / "shots"
    shots.mkdir()
    (shots / "src.frames.csv").write_text("frame,cut\n0,0\n1,0\n")
    monkeypatch.setattr(make_clips, "SHOTS", shots)
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps([{"name": "g3-x", "source": str(tmp_path / "src.mp4"), "start": 0, "frames": 8,
                                 "kind": "gt3", "why": "test"}]))
    make_clips.main(str(spec), tmp_path / "clips")
    clip = json.loads((tmp_path / "clips" / "g3-x" / "clip.json").read_text())
    assert clip["kind"] == "gt3" and clip["lr_size"] == [640, 360] and clip["target"] == [1920, 1080]
    assert clip["lr_kbps"] == round(640 * 360 * 25 * 0.1 / 1000)
    assert len(list(read_frames(tmp_path / "clips" / "g3-x" / "lr.mkv", 640, 360))) == 8
    assert len(list(read_frames(tmp_path / "clips" / "g3-x" / "gt.mkv", 1920, 1080))) == 8


def test_gt_kinds_and_unknown_kind(tmp_path):
    assert make_clips.gt_lr_size("gt") == (480, 270) and make_clips.gt_lr_size("gt3") == (640, 360)
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps([{"name": "x", "source": str(tmp_path / "nope.mp4"), "start": 0, "frames": 1,
                                 "kind": "gt4"}]))
    with pytest.raises(ValueError, match="gt4"):  # rejected before the source is even probed
        make_clips.main(str(spec), tmp_path / "clips")
