import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from detailnet import train as tr
from detailnet.guards import SwapLimitExceeded
from detailnet.losses import Perceptual, ping_pong, texture_lock
from detailnet.model import DetailNet, compose, upsample_flow
from detailnet.tests.test_data import CROP, make_data, record
from detailnet.train import TrainConfig, load_checkpoint

H = W = 8  # source px of the synthetic batches (the base is 24 x 24)
TILE = torch.tensor([[1.0, -1.0, 0.0], [0.0, 1.0, -1.0], [-1.0, 0.0, 1.0]]) * 0.1  # zero mean per 3x3 block


@pytest.fixture(autouse=True)
def seeded():
    torch.manual_seed(0)


@pytest.fixture(scope="module")
def perceptual():
    return Perceptual()


class QuietGuard:
    """Stands in for SwapGuard so the tests do not depend on the machine's swap."""

    def __init__(self, limit_gb=8.0):
        pass

    def check(self):
        pass


@pytest.fixture(autouse=True)
def quiet_swap_guard(monkeypatch):
    monkeypatch.setattr(tr, "SwapGuard", QuietGuard)


def synthetic_batch(n=1, t=4, loss_from=2, kind="gt", seed=0, resets=(0,), h=H, w=W):
    """A batch in the dataset's format: random source frames, nearest-x3 base, and a target that adds a zero-mean
    3x3 texture to the base (which DetailNet can learn from its bias alone)."""
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, t, 3, h, w, generator=g)
    base = F.interpolate(x.flatten(0, 1), scale_factor=3, mode="nearest").view(n, t, 3, 3 * h, 3 * w)
    target = (base + TILE.repeat(h, w)).clamp(0, 1)
    flow = 0.4 * torch.randn(n, t, 4, h, w, generator=g)
    reset = torch.zeros(n, t, dtype=torch.bool)
    reset[:, list(resets)] = True
    loss_mask = (torch.arange(t) >= loss_from).expand(n, t).clone()
    return {"x": x, "base": base, "flow": flow, "target": target, "reset": reset, "loss_mask": loss_mask,
            "kind": [kind] * n, "seq_id": torch.arange(n)}


def pingpong_batch(kind="teacher", n=1, h=H, w=W):
    """T = 39 like a ping-pong dataset item: slots 0..19 forward (loss on 8..19), then 18..0 reversed."""
    b = synthetic_batch(n=n, t=39, kind=kind, h=h, w=w, resets=(0, 25))
    mask = torch.zeros(39, dtype=torch.bool)
    mask[8:20] = True
    for j in range(20, 39):
        mask[j] = mask[38 - j]
    b["loss_mask"] = mask.expand(n, 39).clone()
    return b


def cfg_for(tmp_path, **kw):
    values = dict(data=str(tmp_path / "data"), out=str(tmp_path / "run"), iters=4, batch=2, channels=8, blocks=1,
                  hidden=4, crop=CROP, device="cpu", num_workers=0, ckpt_every=2, log_every=1, lr=1e-3,
                  perc_frames=2)
    values.update(kw)
    return TrainConfig(**values)


def tiny_model(**kw):
    return DetailNet(channels=kw.pop("channels", 8), blocks=kw.pop("blocks", 1), hidden=kw.pop("hidden", 4))


def fake_data(tmp_path, teacher=2, gt=3):
    recs = [record(i, "teacher") for i in range(teacher)] + [record(100 + i, "gt") for i in range(gt)]
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    make_data(tmp_path / "data", recs)
    return recs


def read_log(cfg):
    return [json.loads(line) for line in (Path(cfg.out) / "train.jsonl").read_text().splitlines()]


# ---------------------------------------------------------------- the pins of the brief

def test_overfits_one_sample(perceptual):
    cfg = TrainConfig(data="-", out="-", channels=16, blocks=2, hidden=8, lr=2e-3, device="cpu")
    model = DetailNet(channels=16, blocks=2, hidden=8)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    batch = synthetic_batch()
    rng = np.random.default_rng(0)
    totals = [tr.train_step(model, perceptual, opt, batch, cfg, rng=rng)["loss"] for _ in range(40)]
    assert all(np.isfinite(totals))
    assert totals[-1] <= 0.7 * totals[0], (totals[0], totals[-1])


def test_checkpoint_roundtrip(tmp_path, perceptual):
    model = DetailNet(channels=16, blocks=2, hidden=8)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    cfg = TrainConfig(data="d", out="o", channels=16, blocks=2, hidden=8, device="cpu")
    tr.train_step(model, perceptual, opt, synthetic_batch(), cfg, rng=np.random.default_rng(0))  # optimizer state
    path = tmp_path / "ckpt.pt"
    tr.save_checkpoint(path, model, opt, 7, cfg)
    loaded, saved_cfg = load_checkpoint(path)
    assert saved_cfg["channels"] == 16 and saved_cfg["blocks"] == 2 and saved_cfg["data"] == "d"
    assert isinstance(loaded, DetailNet) and not loaded.training
    b = synthetic_batch()
    args = (b["x"][:, 1], b["x"][:, 0], b["base"][:, 1], b["flow"][:, 1], *model.zero_state(1, H, W, "cpu"))
    with torch.no_grad():
        want = model.eval()(*args)
        got = loaded(*args)
    assert torch.equal(want[0], got[0]) and torch.equal(want[1], got[1])
    state = torch.load(path, map_location="cpu", weights_only=True)
    assert set(state) == {"model", "opt", "iter", "cfg"} and state["iter"] == 7


def test_trained_checkpoint_roundtrip_and_files(tmp_path):
    fake_data(tmp_path)
    cfg = cfg_for(tmp_path, iters=4, ckpt_every=2)
    last = tr.train(cfg)
    out = Path(cfg.out)
    assert last == out / "ckpt_000004.pt"
    assert sorted(p.name for p in out.glob("*.pt")) == ["ckpt_000002.pt", "ckpt_000004.pt", "latest.pt"]
    latest = torch.load(out / "latest.pt", map_location="cpu", weights_only=True)
    assert set(latest) == {"model", "opt", "iter", "cfg"} and latest["iter"] == 4
    assert latest["cfg"]["channels"] == 8 and latest["cfg"]["blocks"] == 1
    model, saved = load_checkpoint(out / "latest.pt")
    again, _ = load_checkpoint(last)
    for a, b in zip(model.parameters(), again.parameters()):
        assert torch.equal(a, b)
    assert saved["iters"] == 4


def test_resume_continues_iter(tmp_path):
    fake_data(tmp_path)
    cfg = cfg_for(tmp_path, iters=3, ckpt_every=10)
    tr.train(cfg)
    first = torch.load(Path(cfg.out) / "latest.pt", map_location="cpu", weights_only=True)
    assert first["iter"] == 3
    last = tr.train(cfg_for(tmp_path, iters=5, ckpt_every=10, resume=True))
    assert last.name == "ckpt_000005.pt"
    second = torch.load(Path(cfg.out) / "latest.pt", map_location="cpu", weights_only=True)
    assert second["iter"] == 5
    assert second["opt"]["state"][0]["step"] == 5  # the Adam state continued rather than restarted
    assert not torch.equal(second["model"]["head_hid.bias"], first["model"]["head_hid.bias"])
    iters = [row["iter"] for row in read_log(cfg)]
    assert iters == [1, 2, 3, 4, 5]


def test_resume_without_a_checkpoint_starts_fresh(tmp_path):
    fake_data(tmp_path)
    last = tr.train(cfg_for(tmp_path, iters=2, resume=True))
    assert last.name == "ckpt_000002.pt"


def test_resume_refuses_a_different_architecture(tmp_path):
    fake_data(tmp_path)
    tr.train(cfg_for(tmp_path, iters=1))
    with pytest.raises(ValueError, match="channels"):
        tr.train(cfg_for(tmp_path, iters=2, resume=True, channels=16))


# ---------------------------------------------------------------- R23: resets

def test_prepare_inputs_resets_flow_and_x_prev_per_element():
    b = synthetic_batch(n=3, t=5, resets=())
    b["reset"][0, 3] = True
    b["reset"][2, 0] = True
    x_prev, flow = tr.prepare_inputs(b["x"], b["flow"], b["reset"])
    for n in range(3):
        for t in range(5):
            if b["reset"][n, t]:
                assert torch.equal(x_prev[n, t], b["x"][n, t])
                assert not flow[n, t].any()
            else:
                assert torch.equal(x_prev[n, t], b["x"][n, t - 1] if t else b["x"][n, 0])
                assert torch.equal(flow[n, t], b["flow"][n, t])
    assert x_prev.shape == b["x"].shape and flow.shape == b["flow"].shape


class Recorder(nn.Module):
    """A DetailNet stand-in that notes its inputs; its detail and hidden grow by 1 each frame it is not reset."""
    scale, hidden_channels = 3, 2

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.ones(()))
        self.calls = []

    def zero_state(self, n, h, w, device):
        return torch.zeros(n, 3, 3 * h, 3 * w), torch.zeros(n, 2, h, w)

    def forward(self, x_t, x_prev, base, flow4, prev_detail, hidden):
        self.calls.append(dict(x=x_t, x_prev=x_prev, flow=flow4, prev=prev_detail, hidden=hidden))
        return (prev_detail + self.w), hidden + 1


@pytest.mark.parametrize("checkpoint", [False, True])
def test_run_sequence_zeroes_state_flow_and_x_prev_at_resets(checkpoint):
    b = synthetic_batch(n=2, t=5, resets=(0,))
    b["reset"][1, 2] = True
    model = Recorder()
    details = tr.run_sequence(model, b["x"], b["base"], b["flow"], b["reset"], checkpoint=checkpoint)
    assert len(details) == 5 and details[0].shape == (2, 3, 3 * H, 3 * W)
    # element 0 never resets after frame 0: its detail counts 1, 2, 3, 4, 5; element 1 restarts at frame 2
    got0 = [float(d[0].mean()) for d in details]
    got1 = [float(d[1].mean()) for d in details]
    assert got0 == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert got1 == [1.0, 2.0, 1.0, 2.0, 3.0]
    if not checkpoint:  # the recorder sees the inputs once per frame (checkpointing replays them in backward)
        calls = model.calls
        assert not calls[2]["prev"][1].any() and not calls[2]["hidden"][1].any()
        assert calls[2]["prev"][0].mean() == 2.0 and calls[2]["hidden"][0].mean() == 2.0
        assert not calls[2]["flow"][1].any() and calls[2]["flow"][0].any()
        assert torch.equal(calls[2]["x_prev"][1], b["x"][1, 2])
        assert torch.equal(calls[2]["x_prev"][0], b["x"][0, 1])
        assert torch.equal(calls[3]["x_prev"][1], b["x"][1, 2])  # after the reset, "previous" is the reset slot's x


# ---------------------------------------------------------------- texture lock

def lock_inputs(t=4, n=2):
    b = synthetic_batch(n=n, t=t, resets=(0,))
    b["flow"].zero_()
    b["x"][:] = b["x"][:, :1]  # static scene: the luma error is zero and a zero flow is consistent
    details = [torch.rand(n, 3, 3 * H, 3 * W) for _ in range(t)]
    return b, details


def test_lock_mask_is_dropped_at_reset_frames_per_element():
    b, details = lock_inputs()
    b["reset"][:] = False
    x_prev, flow = tr.prepare_inputs(b["x"], b["flow"], b["reset"])
    frames = [1, 2, 3]
    full, coverage = tr.lock_term(details, b["x"], x_prev, flow, b["reset"], frames)
    assert float(full) > 0 and float(coverage) == 1.0
    # one element reset at every frame: only the other element is locked, and the term equals the term on it alone
    b["reset"][0, :] = True
    x_prev, flow = tr.prepare_inputs(b["x"], b["flow"], b["reset"])
    half, coverage = tr.lock_term(details, b["x"], x_prev, flow, b["reset"], frames)
    assert float(coverage) == pytest.approx(0.5)
    alone = [d[1:] for d in details]
    ref = texture_lock(torch.cat([alone[t] for t in frames]), torch.cat([alone[t - 1] for t in frames]),
                       torch.zeros(3, 2, 3 * H, 3 * W), torch.ones(3, 1, 3 * H, 3 * W, dtype=torch.bool))
    assert float(half) == pytest.approx(float(ref), rel=1e-5)
    # all reset: nothing is locked; the term is exactly 0 and still differentiable
    b["reset"][:] = True
    x_prev, flow = tr.prepare_inputs(b["x"], b["flow"], b["reset"])
    leaf = [d.clone().requires_grad_(True) for d in details]
    none, coverage = tr.lock_term(leaf, b["x"], x_prev, flow, b["reset"], frames)
    assert float(none) == 0.0 and float(coverage) == 0.0
    none.backward()


def test_lock_uses_the_hr_mask_and_the_upsampled_back_flow():
    from detailnet.model import occlusion_mask, tracking_maps
    b, details = lock_inputs(t=3, n=1)
    b["reset"][:] = False
    ramp = torch.arange(W, dtype=torch.float32) * 0.01  # smooth, so a 1 px shift changes the luma by 0.01 only
    b["x"][:] = ramp.view(1, 1, 1, 1, W).expand_as(b["x"])
    b["flow"][:, :, 0] = 1.0  # back_x = +1 px: the previous detail is sampled 3 output px to the right
    b["flow"][:, :, 2] = -1.0  # fwd_x = -1 px: consistent with it
    x_prev, flow = tr.prepare_inputs(b["x"], b["flow"], b["reset"])
    value, coverage = tr.lock_term(details, b["x"], x_prev, flow, b["reset"], [1, 2])
    d_t = torch.cat([details[1], details[2]])
    d_p = torch.cat([details[0], details[1]])
    fl = torch.cat([flow[:, 1], flow[:, 2]])
    mask = occlusion_mask(tracking_maps(torch.cat([b["x"][:, 1], b["x"][:, 2]]),
                                        torch.cat([x_prev[:, 1], x_prev[:, 2]]), fl), fl)
    assert float(mask.float().mean()) == pytest.approx((W - 1) / W)  # only the last column looks outside the frame
    mask_hr = mask.repeat_interleave(3, dim=2).repeat_interleave(3, dim=3)
    ref = texture_lock(d_t, d_p, upsample_flow(fl[:, 0:2], 3), mask_hr)
    assert float(ref) > 0 and float(value) == pytest.approx(float(ref), rel=1e-5)
    assert float(coverage) == pytest.approx((W - 1) / W)
    wrong = texture_lock(d_t, d_p, fl[:, 0:2].repeat_interleave(3, 2).repeat_interleave(3, 3), mask_hr)  # flow not x3
    assert float(value) != pytest.approx(float(wrong), rel=1e-3)


# ---------------------------------------------------------------- ping-pong

def test_pingpong_pairs_follow_the_reversed_index_rule():
    mask = pingpong_batch()["loss_mask"][0]
    fwd, bwd = tr.pingpong_pairs(mask)
    assert fwd == list(range(8, 19)) and bwd == [38 - k for k in fwd]
    assert bwd[0] == 30 and bwd[-1] == 20
    mask = mask.clone()
    mask[10] = False  # a forward frame that is not scored drops its pair
    fwd, bwd = tr.pingpong_pairs(mask)
    assert 10 not in fwd and 28 not in bwd
    mask[:] = False
    assert tr.pingpong_pairs(mask) == ([], [])


def test_pingpong_term_pairs_each_frame_with_its_reversed_twin(perceptual):
    b = pingpong_batch()
    model = tiny_model()
    cfg = TrainConfig(data="-", out="-", w_perc=0.0, w_l1=0.0, w_lock=0.0, w_pp=1.0, device="cpu")
    model.zero_grad()
    m = tr.compute_gradients(model, perceptual, b, cfg, pingpong=True, rng=np.random.default_rng(0))
    x_prev, flow = tr.prepare_inputs(b["x"], b["flow"], b["reset"])
    details = tr.run_sequence(model, b["x"], b["base"], b["flow"], b["reset"])
    outs = [compose(b["base"][:, t], details[t], 1.0) for t in range(39)]
    want = ping_pong(torch.stack([outs[k] for k in range(8, 19)]), torch.stack([outs[38 - k] for k in range(8, 19)]))
    assert m["pp"] == pytest.approx(float(want), rel=1e-5)
    assert m["loss"] == pytest.approx(float(want), rel=1e-5)


# ---------------------------------------------------------------- two-stage backward == one backward

def direct_gradients(model, perceptual, batch, cfg, pingpong):
    """The same loss as `compute_gradients`, but with the perceptual term inside the one autograd graph."""
    model.zero_grad()
    kind = batch["kind"][0]
    frames = batch["loss_mask"][0].nonzero().flatten().tolist()
    x_prev, flow = tr.prepare_inputs(batch["x"], batch["flow"], batch["reset"])
    details = tr.run_sequence(model, batch["x"], batch["base"], batch["flow"], batch["reset"])
    outs = {t: compose(batch["base"][:, t], details[t], 1.0) for t in frames}
    total = cfg.w_perc * torch.stack([perceptual(outs[t], batch["target"][:, t]) for t in frames]).mean()
    if kind == "gt":
        total = total + cfg.w_l1 * torch.stack([(outs[t] - batch["target"][:, t]).abs() for t in frames]).mean()
    lock, _ = tr.lock_term(details, batch["x"], x_prev, flow, batch["reset"], frames)
    total = total + cfg.w_lock * lock
    if pingpong:
        fwd, bwd = tr.pingpong_pairs(batch["loss_mask"][0])
        total = total + cfg.w_pp * ping_pong(torch.stack([outs[k] for k in fwd]), torch.stack([outs[k] for k in bwd]))
    total.backward()
    return float(total), {n: p.grad.clone() for n, p in model.named_parameters()}


@pytest.mark.parametrize("case,checkpoint", [("gt", False), ("gt", True), ("teacher-pingpong", False)])
def test_two_stage_backward_matches_one_backward(perceptual, case, checkpoint):
    pingpong = case.endswith("pingpong")
    batch = pingpong_batch(kind="teacher", n=2) if pingpong else synthetic_batch(n=2, t=5, loss_from=2, kind="gt",
                                                                                 resets=(0, 3))
    model = tiny_model()
    for p in model.head_tex.parameters():
        p.data.mul_(100)  # the untrained texture head is near zero: scale it so every gradient is well above float noise
    cfg = TrainConfig(data="-", out="-", device="cpu", grad_checkpoint=checkpoint)
    want_loss, want = direct_gradients(model, perceptual, batch, cfg, pingpong)
    model.zero_grad()
    m = tr.compute_gradients(model, perceptual, batch, cfg, pingpong=pingpong, rng=np.random.default_rng(0))
    assert m["loss"] == pytest.approx(want_loss, rel=1e-4)
    for name, p in model.named_parameters():
        assert torch.allclose(p.grad, want[name], rtol=1e-3, atol=1e-6), name
    assert sum(float(g.abs().sum()) for g in want.values()) > 0


def test_loss_terms_present_per_kind(perceptual):
    model = tiny_model()
    cfg = TrainConfig(data="-", out="-", device="cpu")
    gt = tr.compute_gradients(model, perceptual, synthetic_batch(kind="gt"), cfg, rng=np.random.default_rng(0))
    teacher = tr.compute_gradients(model, perceptual, synthetic_batch(kind="teacher"), cfg,
                                   rng=np.random.default_rng(0))
    assert gt["l1"] is not None and teacher["l1"] is None  # L1 is for ground truth only
    assert gt["pp"] is None and teacher["pp"] is None  # no ping-pong term on a plain batch
    expected = cfg.w_perc * gt["perc"] + cfg.w_l1 * gt["l1"] + cfg.w_lock * gt["lock"]
    assert gt["loss"] == pytest.approx(expected, rel=1e-5)
    assert teacher["loss"] == pytest.approx(cfg.w_perc * teacher["perc"] + cfg.w_lock * teacher["lock"], rel=1e-5)


def test_losses_only_see_loss_mask_frames(perceptual):
    """Changing the target of a burn-in frame changes nothing; changing the target of a loss frame does."""
    cfg = TrainConfig(data="-", out="-", device="cpu")
    model = tiny_model()
    a = synthetic_batch(t=4, loss_from=2)
    b = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in a.items()}
    b["target"][:, 0] = 1.0 - b["target"][:, 0]
    c = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in a.items()}
    c["target"][:, 3] = 1.0 - c["target"][:, 3]
    ma = tr.compute_gradients(model, perceptual, a, cfg, rng=np.random.default_rng(0))
    mb = tr.compute_gradients(model, perceptual, b, cfg, rng=np.random.default_rng(0))
    mc = tr.compute_gradients(model, perceptual, c, cfg, rng=np.random.default_rng(0))
    assert mb["perc"] == pytest.approx(ma["perc"]) and mb["l1"] == pytest.approx(ma["l1"])
    assert mc["perc"] != pytest.approx(ma["perc"])


# ---------------------------------------------------------------- perceptual subset

def test_perceptual_frame_subset():
    rng = np.random.default_rng(0)
    frames = list(range(8, 20))
    assert tr.choose_perceptual_frames(frames, 12, rng) == frames  # the default: every loss frame
    assert tr.choose_perceptual_frames(frames, 99, rng) == frames
    picks = [tr.choose_perceptual_frames(frames, 4, rng) for _ in range(600)]
    assert all(len(p) == 4 and p == sorted(set(p)) and set(p) <= set(frames) for p in picks)
    counts = np.bincount(np.concatenate(picks), minlength=20)[8:20]
    assert counts.min() > 120 and counts.max() < 280  # uniform: each frame is picked 4/12 of the time (200 of 600)
    long = list(range(8, 20)) + list(range(20, 31))  # a ping-pong batch has 23 loss frames: 4 per 12 scales
    assert len(tr.choose_perceptual_frames(long, 4, rng)) == 8
    assert tr.choose_perceptual_frames(long, 12, rng) == long
    assert tr.choose_perceptual_frames([9], 4, rng) == [9]


def test_perceptual_subset_is_an_unbiased_estimate_of_the_full_mean(perceptual):
    cfg_all = TrainConfig(data="-", out="-", device="cpu", w_lock=0.0, w_l1=0.0)
    cfg_sub = TrainConfig(data="-", out="-", device="cpu", w_lock=0.0, w_l1=0.0, perc_frames=6)
    batch = synthetic_batch(t=8, loss_from=2, kind="teacher")  # 6 loss frames: 3 of them per step
    model = tiny_model()
    full = tr.compute_gradients(model, perceptual, batch, cfg_all, rng=np.random.default_rng(0))["perc"]
    rng = np.random.default_rng(1)
    runs = [tr.compute_gradients(model, perceptual, batch, cfg_sub, rng=rng)["perc"] for _ in range(16)]
    assert len(set(round(r, 6) for r in runs)) > 1  # different subsets give different values ...
    assert np.mean(runs) == pytest.approx(full, rel=0.05)  # ... around the full mean


# ---------------------------------------------------------------- the epoch: schedule, seeds, small datasets

def test_schedule_alternates_kinds_and_mixes_pingpong():
    rng = np.random.default_rng(0)
    schedule = tr.epoch_schedule(("teacher", "gt"), 500, 0.2, rng)
    assert len(schedule) == 1000
    assert [k for k, _ in schedule[:6]] == ["teacher", "gt"] * 3
    assert all(k == ("teacher", "gt")[i % 2] for i, (k, _) in enumerate(schedule))
    for kind in ("teacher", "gt"):
        share = np.mean([pp for k, pp in schedule if k == kind])
        assert 0.14 < share < 0.26
    assert [pp for _, pp in tr.epoch_schedule(("gt",), 5, 0.0, rng)] == [False] * 5
    assert [k for k, _ in tr.epoch_schedule(("gt",), 3, 0.2, rng)] == ["gt"] * 3
    assert tr.epoch_schedule(("teacher", "gt"), 50, 0.2, np.random.default_rng(3)) == \
           tr.epoch_schedule(("teacher", "gt"), 50, 0.2, np.random.default_rng(3))


def test_epoch_rebuilds_datasets_with_the_epoch_seed(tmp_path):
    fake_data(tmp_path, teacher=2, gt=12)
    cfg = cfg_for(tmp_path, batch=2, pp_prob=0.5, seed=5)
    for epoch in (0, 3):
        e = tr.build_epoch(cfg, epoch, 1000)
        seeds = {loader.dataset.dataset.seed for loader in e.loaders.values()}
        assert seeds == {5 + 1_000_003 * epoch}
        assert {loader.dataset.dataset.ping_pong for loader in e.loaders.values()} == {False, True}
        assert all(loader.drop_last and loader.num_workers == 0 for loader in e.loaders.values())
        assert e.sizes == {"teacher": 2, "gt": 12}
        assert len(e.schedule) == 2 * (12 // 2)  # one pass over the larger dataset, in batches of 2, both kinds


def test_epoch_loader_settings_match_the_ruling(tmp_path):
    fake_data(tmp_path)
    e = tr.build_epoch(cfg_for(tmp_path, num_workers=2), 0, 10)
    for loader in e.loaders.values():
        assert loader.num_workers == 2 and loader.drop_last and not loader.persistent_workers


def test_epoch_picks_up_sequences_finished_in_the_meantime(tmp_path):
    recs = fake_data(tmp_path, teacher=1, gt=2)
    cfg = cfg_for(tmp_path)
    assert tr.build_epoch(cfg, 0, 100).sizes["teacher"] == 1
    make_data(tmp_path / "data", recs + [record(1, "teacher")])
    assert tr.build_epoch(cfg, 1, 100).sizes["teacher"] == 2


def test_a_short_dataset_is_reiterated_within_the_epoch(tmp_path):
    fake_data(tmp_path, teacher=1, gt=8)
    cfg = cfg_for(tmp_path, batch=2, pp_prob=0.0)
    e = tr.build_epoch(cfg, 0, 1000)
    assert len(e.schedule) == 8  # 4 batches of gt, and 4 teacher batches from one sequence
    teacher = list(e.loaders[("teacher", False)])
    assert len(teacher) == 4 and all(b["x"].shape[0] == 2 for b in teacher)
    crops = {tuple(b["x"][i, 0, 0, 0, :4].tolist()) for b in teacher for i in range(2)}
    assert len(crops) > 2  # the repeats are other crops of the sequence, not the same item again


def served(loaders, kind):
    """Sequence ids (in serving order) that the epoch's loaders of `kind` deliver, plain and ping-pong together."""
    ids = []
    for (k, _), loader in sorted(loaders.items()):
        if k == kind:
            for batch in loader:
                ids += batch["seq_id"].tolist()
    return ids


def test_an_epoch_serves_every_sequence_of_the_larger_kind(tmp_path):
    fake_data(tmp_path, teacher=5, gt=7)  # 7 is not a multiple of the batch: 4 batches, the last one wraps
    cfg = cfg_for(tmp_path, batch=2, pp_prob=0.5)
    both = 0
    for epoch in range(6):
        e = tr.build_epoch(cfg, epoch, 1000)
        gt = served(e.loaders, "gt")
        assert len(gt) == 8 and set(gt) == set(range(100, 107)), (epoch, gt)  # every sequence at least once
        both += ("gt", False) in e.loaders and ("gt", True) in e.loaders
    assert both >= 3  # the plain and the ping-pong loaders were both in play in these epochs


def test_the_tail_of_the_sequence_list_is_not_starved_in_either_loader(tmp_path):
    """The sequence list is sorted and a dataset index is a sequence: a loader that reads the first `count` indices
    serves the head of the list every epoch (here the plain loader would serve the last six sequences about 40% as
    often as the first six, the ping-pong loader never)."""
    fake_data(tmp_path, teacher=2, gt=20)
    cfg = cfg_for(tmp_path, batch=2, pp_prob=0.2)
    plain, pingpong = Counter(), Counter()
    for epoch in range(25):
        e = tr.build_epoch(cfg, epoch, 1000)
        for (kind, pp), loader in e.loaders.items():
            if kind == "gt":
                (pingpong if pp else plain).update(i for batch in loader for i in batch["seq_id"].tolist())
    head, tail = range(100, 106), range(114, 120)
    mean = lambda counts, ids: np.mean([counts[i] for i in ids])
    mean_plain = np.mean(list(plain.values()))
    assert all(plain[i] > 0.6 * mean_plain for i in range(100, 120))  # flat: about 20 serves each over the 25 epochs
    assert mean(plain, tail) > 0.75 * mean(plain, head)
    assert mean(pingpong, tail) > 0.5 * mean(pingpong, head) and mean(pingpong, tail) > 2


def test_a_short_kind_is_repeated_evenly_and_the_extra_serves_rotate(tmp_path):
    fake_data(tmp_path, teacher=5, gt=7)
    cfg = cfg_for(tmp_path, batch=2, pp_prob=0.0)
    twice = set()
    for epoch in range(10):
        counts = Counter(served(tr.build_epoch(cfg, epoch, 1000).loaders, "teacher"))
        assert sum(counts.values()) == 8 and set(counts) == set(range(5))  # 8 items over 5 sequences: 1 or 2 each
        assert set(counts.values()) == {1, 2}
        twice |= {i for i, n in counts.items() if n == 2}
    assert twice == set(range(5))  # not always the same head of the list


def test_the_serving_order_changes_between_epochs_but_not_between_builds(tmp_path):
    fake_data(tmp_path, teacher=2, gt=12)
    cfg = cfg_for(tmp_path, batch=2, pp_prob=0.0)
    first = served(tr.build_epoch(cfg, 0, 1000).loaders, "gt")
    assert first == served(tr.build_epoch(cfg, 0, 1000).loaders, "gt")
    assert first != served(tr.build_epoch(cfg, 1, 1000).loaders, "gt")
    assert sorted(first) == list(range(100, 112))


def test_trains_with_a_dataset_shorter_than_the_batch(tmp_path):
    fake_data(tmp_path, teacher=1, gt=1)
    cfg = cfg_for(tmp_path, iters=3, batch=2, ckpt_every=10)
    assert tr.train(cfg).name == "ckpt_000003.pt"


def test_trains_on_the_other_kind_when_one_has_no_sequences(tmp_path):
    fake_data(tmp_path, teacher=0, gt=3)
    cfg = cfg_for(tmp_path, iters=2, batch=2)
    tr.train(cfg)
    assert all(row["l1"] is not None for row in read_log(cfg))  # gt steps only


def test_no_finished_sequence_at_all_is_an_error_and_leaves_no_checkpoint(tmp_path):
    fake_data(tmp_path, teacher=0, gt=0)
    cfg = cfg_for(tmp_path)
    with pytest.raises(tr.NoDataError):
        tr.train(cfg)
    assert not (Path(cfg.out) / "latest.pt").exists()  # nothing was done, so there is nothing to resume from
    assert tr.main(["--data", cfg.data, "--out", str(tmp_path / "again"), "--device", "cpu"]) == 1


# ---------------------------------------------------------------- learning rate

def want_lr(base, it, iters):
    return 0.5 * base * (1 + math.cos(math.pi * it / iters))


def test_cosine_lr_shape():
    cfg = TrainConfig(data="-", out="-", lr=2e-4, iters=100)
    assert tr.cosine_lr(cfg, 0) == pytest.approx(2e-4)
    assert tr.cosine_lr(cfg, 50) == pytest.approx(1e-4)
    assert tr.cosine_lr(cfg, 100) == pytest.approx(0.0, abs=1e-12)
    assert tr.cosine_lr(cfg, 250) == pytest.approx(0.0, abs=1e-12)  # past the horizon it stays at 0
    lrs = [tr.cosine_lr(cfg, i) for i in range(101)]
    assert all(a >= b for a, b in zip(lrs, lrs[1:]))


def test_lr_follows_the_cosine_and_is_back_on_it_after_a_resume(tmp_path, monkeypatch):
    fake_data(tmp_path)
    base, iters = 1e-3, 6
    cfg = cfg_for(tmp_path, iters=iters, lr=base, ckpt_every=100)

    class StopAfterThree:
        calls = 0

        def __init__(self, limit_gb):
            pass

        def check(self):
            StopAfterThree.calls += 1
            if StopAfterThree.calls == 3:
                raise SwapLimitExceeded("stop")

    monkeypatch.setattr(tr, "SwapGuard", StopAfterThree)
    with pytest.raises(SwapLimitExceeded):
        tr.train(cfg)
    stale = torch.load(Path(cfg.out) / "latest.pt", map_location="cpu", weights_only=True)
    assert stale["iter"] == 3
    assert stale["opt"]["param_groups"][0]["lr"] == pytest.approx(want_lr(base, 2, iters))  # the lr of the last step
    monkeypatch.setattr(tr, "SwapGuard", QuietGuard)
    tr.train(cfg_for(tmp_path, iters=iters, lr=base, ckpt_every=100, resume=True))
    rows = read_log(cfg)
    assert [r["iter"] for r in rows] == [1, 2, 3, 4, 5, 6]
    for row in rows:  # the step that ends at iteration i started at i - 1
        assert row["lr"] == pytest.approx(want_lr(base, row["iter"] - 1, iters), rel=1e-5), row["iter"]
    assert rows[0]["lr"] == pytest.approx(base) and rows[-1]["lr"] == pytest.approx(want_lr(base, 5, iters), rel=1e-5)
    assert rows[3]["lr"] != pytest.approx(stale["opt"]["param_groups"][0]["lr"], rel=1e-3)  # not the restored one
    final = torch.load(Path(cfg.out) / "latest.pt", map_location="cpu", weights_only=True)
    assert final["opt"]["param_groups"][0]["lr"] == pytest.approx(want_lr(base, 5, iters))


def test_train_step_uses_the_optimizers_current_lr(perceptual):
    model = tiny_model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    cfg = TrainConfig(data="-", out="-", device="cpu")
    tr.set_lr(opt, 0.0)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    tr.train_step(model, perceptual, opt, synthetic_batch(), cfg, rng=np.random.default_rng(0))
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())  # lr 0: Adam moves nothing


# ---------------------------------------------------------------- non-finite steps are skipped

def with_flaky_steps(monkeypatch, bad, nan=False):
    """Make the `bad` calls (1-based) of `train_step` fail: with a real NaN target (nan=True) or by raising the
    FloatingPointError directly. Returns the list of calls made."""
    real = tr.train_step
    calls = []

    def step(model, perceptual, opt, batch, *args, **kwargs):
        calls.append(1)
        if len(calls) in bad:
            if not nan:
                raise FloatingPointError("injected")
            batch = {**batch, "target": torch.full_like(batch["target"], float("nan"))}
        return real(model, perceptual, opt, batch, *args, **kwargs)

    monkeypatch.setattr(tr, "train_step", step)
    return calls


def test_a_non_finite_batch_is_skipped_logged_and_does_not_count(tmp_path, monkeypatch):
    fake_data(tmp_path)
    calls = with_flaky_steps(monkeypatch, {2, 3}, nan=True)
    cfg = cfg_for(tmp_path, iters=4, ckpt_every=100)
    assert tr.train(cfg).name == "ckpt_000004.pt"
    assert len(calls) == 6  # two skipped batches on top of the four steps
    rows = read_log(cfg)
    skips = [r for r in rows if "skipped" in r]
    assert [(r["iter"], r["skipped"]) for r in skips] == [(1, 1), (1, 2)] and "non-finite" in skips[0]["error"]
    assert [r["iter"] for r in rows if "skipped" not in r] == [1, 2, 3, 4]
    latest = torch.load(Path(cfg.out) / "latest.pt", map_location="cpu", weights_only=True)
    assert latest["iter"] == 4 and latest["opt"]["state"][0]["step"] == 4  # four optimizer steps, none from the NaN
    assert all(torch.isfinite(v).all() for v in latest["model"].values())


def test_ten_non_finite_batches_in_a_row_end_the_run_after_saving_latest(tmp_path, monkeypatch):
    fake_data(tmp_path)
    calls = with_flaky_steps(monkeypatch, set(range(3, 100)))  # two good steps, then nothing but failures
    cfg = cfg_for(tmp_path, iters=20, ckpt_every=100)
    with pytest.raises(FloatingPointError, match="10 non-finite"):
        tr.train(cfg)
    assert len(calls) == 12
    assert len([r for r in read_log(cfg) if "skipped" in r]) == 10
    assert torch.load(Path(cfg.out) / "latest.pt", map_location="cpu", weights_only=True)["iter"] == 2
    assert tr.MAX_CONSECUTIVE_SKIPS == 10


def test_the_skip_count_is_of_consecutive_batches(tmp_path, monkeypatch):
    fake_data(tmp_path)
    bad = set(range(1, 10)) | set(range(11, 20))  # nine failures, one good step, nine failures, then good steps
    calls = with_flaky_steps(monkeypatch, bad)
    cfg = cfg_for(tmp_path, iters=3, ckpt_every=100)
    tr.train(cfg)
    assert len(calls) == 21


def test_a_run_that_fails_before_its_first_step_leaves_no_latest(tmp_path, monkeypatch):
    fake_data(tmp_path)
    bad = set(range(1, 100))
    with_flaky_steps(monkeypatch, bad)
    cfg = cfg_for(tmp_path, iters=1)
    with pytest.raises(FloatingPointError, match="10 non-finite"):
        tr.train(cfg)
    assert not (Path(cfg.out) / "latest.pt").exists()
    bad.clear()
    assert tr.train(cfg).name == "ckpt_000001.pt"  # a rerun with the same --out is not refused as an existing run


# ---------------------------------------------------------------- logs and guards

def test_log_lines(tmp_path):
    fake_data(tmp_path)
    cfg = cfg_for(tmp_path, iters=4, log_every=2, pp_prob=1.0)
    tr.train(cfg)
    rows = read_log(cfg)
    assert [r["iter"] for r in rows] == [2, 4]
    for r in rows:
        for key in ("loss", "perc", "lock", "it_per_s", "mem_gb", "grad_norm", "epoch"):
            assert key in r and np.isfinite(r[key]), key
        assert r["pp"] is not None and r["l1"] is not None  # every step was ping-pong; the window held gt steps


def test_nonfinite_loss_stops_the_run_before_any_step(perceptual):
    model = tiny_model()
    opt = torch.optim.AdamW(model.parameters())
    cfg = TrainConfig(data="-", out="-", device="cpu")
    batch = synthetic_batch()
    batch["target"][:] = float("nan")
    before = {k: v.clone() for k, v in model.state_dict().items()}
    with pytest.raises(FloatingPointError):
        tr.train_step(model, perceptual, opt, batch, cfg, rng=np.random.default_rng(0))
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())  # no step was taken


def test_swap_guard_trip_saves_latest_and_main_exits_3(tmp_path, monkeypatch, capsys):
    fake_data(tmp_path)

    class Tripping:
        calls = 0

        def __init__(self, limit_gb):
            self.limit = limit_gb

        def check(self):
            Tripping.calls += 1
            if Tripping.calls == 2:
                raise SwapLimitExceeded("swap grew by 9.0 GB")

    monkeypatch.setattr(tr, "SwapGuard", Tripping)
    monkeypatch.setattr(tr, "refuse_if_batch_running", lambda force: None)
    cfg = cfg_for(tmp_path, iters=10, ckpt_every=100)
    with pytest.raises(SwapLimitExceeded):
        tr.train(cfg)
    latest = torch.load(Path(cfg.out) / "latest.pt", map_location="cpu", weights_only=True)
    assert latest["iter"] == 2  # the guard fired after the second step, and that step is in the checkpoint
    Tripping.calls = 0
    code = tr.main(["--data", cfg.data, "--out", str(tmp_path / "run2"), "--iters", "10", "--device", "cpu",
                    "--num-workers", "0", "--channels", "8", "--blocks", "1", "--hidden", "4", "--batch", "2",
                    "--crop", "16", "8"])
    assert code == 3
    assert "swap" in capsys.readouterr().out


def test_main_calls_the_batch_guard_with_force(tmp_path, monkeypatch):
    fake_data(tmp_path)
    seen = []
    monkeypatch.setattr(tr, "refuse_if_batch_running", lambda force: seen.append(force))
    argv = ["--data", str(tmp_path / "data"), "--out", str(tmp_path / "run"), "--iters", "1", "--device", "cpu",
            "--num-workers", "0", "--channels", "8", "--blocks", "1", "--hidden", "4", "--batch", "2",
            "--crop", "16", "8"]
    assert tr.main(argv) == 0
    assert tr.main(argv[:3] + [str(tmp_path / "run-forced")] + argv[4:] + ["--force"]) == 0
    assert seen == [False, True]


def test_a_finished_run_is_not_overwritten_without_resume(tmp_path):
    fake_data(tmp_path)
    cfg = cfg_for(tmp_path, iters=1)
    tr.train(cfg)
    with pytest.raises(FileExistsError, match="--resume"):
        tr.train(cfg)


def test_main_refuses_while_the_batch_runs(tmp_path, monkeypatch):
    from detailnet import guards
    monkeypatch.setattr(tr, "refuse_if_batch_running",
                        lambda force: guards.refuse_if_batch_running(force, pgrep=lambda pattern: True))
    with pytest.raises(SystemExit, match="refusing"):
        tr.main(["--data", "x", "--out", str(tmp_path / "run"), "--device", "cpu"])
    assert not (tmp_path / "run").exists()


def test_config_defaults_are_the_briefs():
    cfg = TrainConfig(data="d", out="o")
    assert (cfg.iters, cfg.batch, cfg.lr, cfg.weight_decay) == (100_000, 4, 2e-4, 1e-4)
    assert (cfg.w_perc, cfg.w_l1, cfg.w_lock, cfg.w_pp) == (1.0, 1.0, 2.0, 0.5)
    assert (cfg.ckpt_every, cfg.seed, cfg.channels, cfg.blocks) == (5000, 0, 64, 12)
    assert cfg.device == "mps" and cfg.perc_frames == 12 and cfg.num_workers == 2
    assert cfg.grad_checkpoint and cfg.pp_prob == 0.2 and cfg.reset_prob == 0.15
