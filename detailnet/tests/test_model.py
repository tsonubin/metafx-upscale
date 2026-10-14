import pytest
import torch
import torch.nn.functional as F

from detailnet.model import (DetailNet, compose, depth_to_space, occlusion_mask, space_to_depth,
                             texture_only, tracking_maps, upsample_flow, warp)

torch.manual_seed(0)


def energy(t):
    return float((t.double() ** 2).sum())


def make_inputs(n, h, w, s, flow=None):
    x_t = torch.rand(n, 3, h, w)
    x_prev = torch.rand(n, 3, h, w)
    base = torch.rand(n, 3, h * s, w * s)
    if flow is None:
        flow = torch.zeros(n, 4, h, w)
    return x_t, x_prev, base, flow


# ---------------------------------------------------------------- shapes, size

@pytest.mark.parametrize("s", [3, 2])
def test_shapes_s3_s2(s):
    net = DetailNet(scale=s).eval()
    n, h, w = 2, 12, 16
    x_t, x_prev, base, flow = make_inputs(n, h, w, s)
    prev_detail, hidden = net.zero_state(n, h, w, "cpu")
    assert prev_detail.shape == (n, 3, s * h, s * w)
    assert hidden.shape == (n, 32, h, w)
    with torch.no_grad():
        detail, new_hidden = net(x_t, x_prev, base, flow, prev_detail, hidden)
    assert detail.shape == (n, 3, s * h, s * w)
    assert new_hidden.shape == (n, 32, h, w)
    assert torch.isfinite(detail).all() and torch.isfinite(new_hidden).all()


def test_param_count():
    n_params = sum(p.numel() for p in DetailNet().parameters())
    assert 0.8e6 <= n_params <= 2.5e6, n_params


# ---------------------------------------------------------------- texture_only

def test_texture_only_kills_smooth():
    h, w = 48, 48
    ys, xs = torch.meshgrid(torch.linspace(0, 1, h), torch.linspace(0, 1, w), indexing="ij")
    ramp = (0.2 + 0.5 * xs + 0.3 * ys).expand(1, 3, h, w).contiguous()
    kept = energy(texture_only(ramp, 3)) / energy(ramp - ramp.mean())
    assert kept < 0.05, kept


def test_texture_only_keeps_hf():
    h, w = 48, 48
    ys, xs = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    checker = (((xs + ys) % 2).float() * 2 - 1).expand(1, 3, h, w).contiguous()
    kept = energy(texture_only(checker, 3)) / energy(checker)
    assert kept > 0.8, kept


@pytest.mark.parametrize("s", [3, 2])
def test_texture_only_zero_mean(s):
    r = torch.randn(2, 3, 12 * s, 16 * s)
    assert abs(float(texture_only(r, s).mean())) < 1e-3


def test_texture_only_shape_and_divisibility():
    assert texture_only(torch.rand(1, 3, 12, 18), 3).shape == (1, 3, 12, 18)
    with pytest.raises(Exception):
        texture_only(torch.rand(1, 3, 13, 18), 3)


# ---------------------------------------------------------------- compose

def test_compose_strength_zero_is_base():
    base = torch.rand(1, 3, 12, 12)
    detail = torch.randn(1, 3, 12, 12)
    assert torch.equal(compose(base, detail, 0.0), base)


def test_compose_adds_scaled_detail_and_clamps():
    base = torch.tensor([[[[0.5, 0.95, 0.05]]]])
    detail = torch.tensor([[[[0.1, 0.2, -0.2]]]])
    out = compose(base, detail, 0.5)
    assert torch.allclose(out, torch.tensor([[[[0.55, 1.0, 0.0]]]]))
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


# ---------------------------------------------------------------- warp and flow

def test_warp_inverts_shift():
    h, w = 20, 24
    prev = torch.rand(1, 3, h, w)
    cur = torch.zeros_like(prev)
    cur[..., 2:] = prev[..., :-2]  # content moves right by 2 px
    back = torch.zeros(1, 2, h, w)
    back[:, 0] = -2.0  # the pixel at p in frame t was at p - 2 in frame t-1
    out = warp(prev, back)
    assert (out[..., 2:] - cur[..., 2:]).abs().max() < 1e-5


def test_warp_zero_flow_is_identity():
    x = torch.rand(2, 3, 9, 11)
    assert (warp(x, torch.zeros(2, 2, 9, 11)) - x).abs().max() < 1e-6


def test_warp_vertical_and_fractional():
    h, w = 16, 16
    ys = torch.arange(h).float().view(1, 1, h, 1).expand(1, 1, h, w)
    flow = torch.zeros(1, 2, h, w)
    flow[:, 1] = 1.5  # sample 1.5 px further down
    out = warp(ys, flow)
    assert (out[..., : h - 2, :] - (ys[..., : h - 2, :] + 1.5)).abs().max() < 1e-4


def test_warp_border_padding():
    x = torch.arange(8.0).view(1, 1, 1, 8)
    flow = torch.zeros(1, 2, 1, 8)
    flow[:, 0] = 5.0
    out = warp(x, flow)
    assert out[0, 0, 0, -1] == 7.0 and out[0, 0, 0, -3] == 7.0 and out[0, 0, 0, 0] == 5.0


def test_upsample_flow_scales_values_and_size():
    flow = torch.full((1, 4, 5, 7), 1.5)
    up = upsample_flow(flow, 3)
    assert up.shape == (1, 4, 15, 21)
    assert torch.allclose(up, torch.full_like(up, 4.5))


def test_space_to_depth_roundtrip():
    x = torch.rand(2, 3, 12, 18)
    for s in (2, 3):
        d = space_to_depth(x, s)
        assert d.shape == (2, 3 * s * s, 12 // s, 18 // s)
        assert torch.equal(depth_to_space(d, s), x)


# ---------------------------------------------------------------- tracking and occlusion

def uniform_flow(n, h, w, back, fwd):
    flow = torch.zeros(n, 4, h, w)
    flow[:, 0], flow[:, 1] = back
    flow[:, 2], flow[:, 3] = fwd
    return flow


def test_tracking_maps_perfect_track():
    h, w = 24, 24
    prev = torch.rand(1, 3, h, w)
    cur = torch.zeros_like(prev)
    cur[..., 2:] = prev[..., :-2]
    flow = uniform_flow(1, h, w, back=(-2.0, 0.0), fwd=(2.0, 0.0))
    maps = tracking_maps(cur, prev, flow)
    assert maps.shape == (1, 2, h, w)
    assert maps[:, 0, :, 4:].abs().max() < 1e-5  # luma error
    assert maps[:, 1, :, 4:-4].abs().max() < 1e-5  # forward-backward inconsistency
    assert occlusion_mask(maps, flow)[:, :, :, 4:-4].all()


def test_tracking_maps_luma_is_rec709():
    h, w = 8, 8
    x_t = torch.zeros(1, 3, h, w)
    x_t[:, 1] = 1.0  # pure green
    maps = tracking_maps(x_t, torch.zeros_like(x_t), torch.zeros(1, 4, h, w))
    assert torch.allclose(maps[:, 0], torch.full((1, h, w), 0.7152), atol=1e-6)


def test_occlusion_mask_flags_luma_mismatch_and_fb_inconsistency():
    h, w = 8, 8
    flow = torch.zeros(1, 4, h, w)
    maps = torch.zeros(1, 2, h, w)
    maps[:, 0, :, :4] = 0.2  # large luma error on the left half
    maps[:, 1, :4] = 3.0  # fb inconsistency of 3 px on the top half, no motion
    mask = occlusion_mask(maps, flow)
    assert mask.dtype == torch.bool and mask.shape == (1, 1, h, w)
    assert mask[0, 0, 4:, 4:].all()  # untouched quadrant
    assert not mask[0, 0, :, :4].any()  # luma mismatch
    assert not mask[0, 0, :4, :].any()  # fb inconsistency


def test_occlusion_mask_fb_tolerance_grows_with_motion():
    h, w = 8, 8
    maps = torch.zeros(1, 2, h, w)
    maps[:, 1] = 1.0  # 1 px inconsistency
    still = uniform_flow(1, h, w, back=(0.0, 0.0), fwd=(0.0, 0.0))
    moving = uniform_flow(1, h, w, back=(-20.0, 0.0), fwd=(20.0, 0.0))
    assert not occlusion_mask(maps, still).any()  # tolerance sqrt(0.5) px at rest
    assert occlusion_mask(maps, moving).all()  # sqrt(0.5 + 0.01 * 800) px when moving fast


# ---------------------------------------------------------------- recurrence

def test_hidden_bounded():
    n, h, w, s = 1, 16, 16, 3
    net = DetailNet(scale=s).eval()
    prev_detail, hidden = net.zero_state(n, h, w, "cpu")
    flow = torch.randn(n, 4, h, w) * 2
    with torch.no_grad():
        for _ in range(50):
            x_t, x_prev, base, _ = make_inputs(n, h, w, s)
            prev_detail, hidden = net(x_t, x_prev, base, flow, prev_detail, hidden)
            assert float(hidden.abs().max()) <= 1.0
            assert torch.isfinite(prev_detail).all()


def test_untrained_net_leaves_base_nearly_unchanged():
    n, h, w, s = 1, 16, 16, 3
    net = DetailNet(scale=s).eval()
    x_t, x_prev, base, flow = make_inputs(n, h, w, s)
    with torch.no_grad():
        detail, _ = net(x_t, x_prev, base, flow, *net.zero_state(n, h, w, "cpu"))
    assert float(detail.abs().max()) < 0.02


def test_gradients_reach_every_layer_despite_near_zero_head():
    n, h, w, s = 1, 12, 12, 3
    net = DetailNet(scale=s, channels=16, blocks=2)
    x_t, x_prev, base, flow = make_inputs(n, h, w, s)
    detail, hidden = net(x_t, x_prev, base, flow, *net.zero_state(n, h, w, "cpu"))
    (detail.abs().mean() + hidden.abs().mean()).backward()
    for name, p in net.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    assert net.conv_in.weight.grad.abs().max() > 0


def test_hidden_and_detail_are_warped_by_back_flow():
    n, h, w, s = 1, 16, 16, 3
    net = DetailNet(scale=s, channels=16, blocks=2).eval()
    x_t, x_prev, base, _ = make_inputs(n, h, w, s)
    hidden = torch.tanh(torch.randn(n, 32, h, w))
    prev_detail = torch.randn(n, 3, s * h, s * w)
    still = torch.zeros(n, 4, h, w)
    moved = uniform_flow(n, h, w, back=(-3.0, 1.0), fwd=(3.0, -1.0))
    with torch.no_grad():
        d0, h0 = net(x_t, x_prev, base, still, prev_detail, hidden)
        d1, h1 = net(x_t, x_prev, base, moved, prev_detail, hidden)
    assert (h0 - h1).abs().max() > 1e-4  # the flow reaches the network
    assert (d0 - d1).abs().max() > 0


def test_zero_state_on_requested_size():
    net = DetailNet(scale=2, hidden=8)
    prev_detail, hidden = net.zero_state(3, 5, 7, torch.device("cpu"))
    assert prev_detail.shape == (3, 3, 10, 14) and hidden.shape == (3, 8, 5, 7)
    assert not prev_detail.any() and not hidden.any()
