import pytest
import torch
import torch.nn.functional as F

from detailnet.model import (DetailNet, compose, depth_to_space, occlusion_mask, space_to_depth,
                             texture_only, tracking_maps, upsample_flow, warp)



@pytest.fixture(autouse=True)
def seeded():
    torch.manual_seed(0)  # per test, so results do not depend on test order or selection


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
    gen = torch.Generator().manual_seed(1234)
    r = torch.randn(2, 3, 12 * s, 16 * s, generator=gen) + 0.5  # a DC offset must not leak into the texture
    assert abs(float(texture_only(r, s).mean())) < 1e-3


def test_texture_only_shape_and_divisibility():
    assert texture_only(torch.rand(1, 3, 12, 18), 3).shape == (1, 3, 12, 18)
    with pytest.raises(ValueError, match="divisible"):
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
    h, w = 8, 48
    maps = torch.zeros(1, 2, h, w)
    maps[:, 1] = 1.0  # 1 px inconsistency
    still = uniform_flow(1, h, w, back=(0.0, 0.0), fwd=(0.0, 0.0))
    moving = uniform_flow(1, h, w, back=(-20.0, 0.0), fwd=(20.0, 0.0))
    assert not occlusion_mask(maps, still).any()  # tolerance sqrt(0.5) px at rest
    assert occlusion_mask(maps, moving)[..., 20:].all()  # sqrt(0.5 + 0.01 * 800) px when moving fast


def test_occlusion_mask_fb_test_is_the_squared_form():
    # The bound 0.5 + 0.01 * (|b|^2 + |f|^2) is for the squared inconsistency; each case here
    # goes the other way if the plain length is compared instead.
    h, w = 8, 48
    maps = torch.zeros(1, 2, h, w)
    still = uniform_flow(1, h, w, back=(0.0, 0.0), fwd=(0.0, 0.0))
    moving = uniform_flow(1, h, w, back=(-20.0, 0.0), fwd=(20.0, 0.0))
    maps[:, 1] = 0.6  # 0.36 < 0.5: kept at rest (0.6 >= 0.5 would be masked)
    assert occlusion_mask(maps, still).all()
    maps[:, 1] = 5.0  # 25 > 0.5 + 0.01 * 800 = 8.5: masked at 20 px motion (5 < 8.5 would be kept)
    assert not occlusion_mask(maps, moving).any()


def test_occlusion_mask_marks_pixels_whose_backward_target_is_outside_the_frame():
    h, w = 6, 16
    maps = torch.zeros(1, 2, h, w)  # a flawless track everywhere, so only the frame edge can mask
    # pan: content moves right, so the pixels in the first 2 columns came from outside the frame
    pan = uniform_flow(1, h, w, back=(-2.0, 0.0), fwd=(2.0, 0.0))
    mask = occlusion_mask(maps, pan)
    assert not mask[..., :2].any() and mask[..., 2:].all()  # x - 2 == 0 is still inside
    # content moving left: the last 3 columns are masked (x + 3 > W - 1)
    mask = occlusion_mask(maps, uniform_flow(1, h, w, back=(3.0, 0.0), fwd=(-3.0, 0.0)))
    assert not mask[..., w - 3:].any() and mask[..., : w - 3].all()
    # vertical, and fractional: x + back = -0.5 is outside, 0.5 inside
    mask = occlusion_mask(maps, uniform_flow(1, h, w, back=(-0.5, 1.0), fwd=(0.5, -1.0)))
    assert not mask[..., 0].any() and not mask[:, :, h - 1].any()
    assert mask[:, :, : h - 1, 1:].all()
    # no motion: nothing is outside
    assert occlusion_mask(maps, torch.zeros(1, 4, h, w)).all()


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


def conv_in_stack(net, *args):
    """The input stack the first conv sees, captured with a forward-pre-hook."""
    captured = []
    handle = net.conv_in.register_forward_pre_hook(lambda module, inputs: captured.append(inputs[0].detach().clone()))
    try:
        with torch.no_grad():
            net(*args)
    finally:
        handle.remove()
    assert len(captured) == 1
    return captured[0]


@pytest.mark.parametrize("s, channels_in", [(3, 91), (2, 61)])
def test_input_stack_layout(s, channels_in):
    n, h, w = 2, 10, 12
    net = DetailNet(scale=s, channels=8, blocks=1).eval()
    x_t, x_prev, base, _ = make_inputs(n, h, w, s)
    flow = torch.randn(n, 4, h, w) * 2  # fractional flow in both directions
    hidden = torch.tanh(torch.randn(n, 32, h, w))
    prev_detail = torch.randn(n, 3, s * h, s * w)
    stack = conv_in_stack(net, x_t, x_prev, base, flow, prev_detail, hidden)

    back = flow[:, 0:2]
    expected = [
        x_t,
        space_to_depth(base, s),
        space_to_depth(warp(prev_detail, upsample_flow(back, s)), s),
        warp(hidden, back),
        tracking_maps(x_t, x_prev, flow),
    ]
    assert [e.shape[1] for e in expected] == [3, 3 * s * s, 3 * s * s, 32, 2]
    assert stack.shape == (n, channels_in, h, w)
    assert net.conv_in.in_channels == channels_in
    start = 0
    for index, e in enumerate(expected):
        part = stack[:, start:start + e.shape[1]]
        assert torch.equal(part, e), f"slice {index} of the input stack is not what the brief lists"
        start += e.shape[1]
    assert start == channels_in
    # the warped slices really differ from the unwarped tensors, so the equalities above are not vacuous
    assert (expected[3] - hidden).abs().max() > 1e-3
    assert (expected[2] - space_to_depth(prev_detail, s)).abs().max() > 1e-3


def test_input_stack_warps_by_back_flow_scaled_to_each_grid():
    s, n, h, w = 3, 1, 14, 16
    net = DetailNet(scale=s, channels=8, blocks=1).eval()
    x_t, x_prev, base, _ = make_inputs(n, h, w, s)
    hidden = torch.tanh(torch.randn(n, 32, h, w))
    prev_detail = torch.randn(n, 3, s * h, s * w)
    dx, dy = -2, 1  # back flow in source pixels: the pixel at (x, y) was at (x + dx, y + dy) in t-1
    flow = uniform_flow(n, h, w, back=(float(dx), float(dy)), fwd=(float(-dx), float(-dy)))
    stack = conv_in_stack(net, x_t, x_prev, base, flow, prev_detail, hidden)

    # memory: shifted by the flow in source pixels
    hidden_slice = stack[:, 3 + 6 * s * s: 3 + 6 * s * s + 32]
    got = hidden_slice[..., 0:h - dy, -dx:]  # interior: y + dy <= h - 1, x + dx >= 0
    assert (got - hidden[..., dy:h, 0:w + dx]).abs().max() < 1e-5
    assert (hidden_slice - hidden).abs().max() > 1e-2  # the memory did move

    # texture: shifted by s x the flow, in output pixels
    hr_slice = depth_to_space(stack[:, 3 + 3 * s * s: 3 + 6 * s * s], s)
    hdx, hdy = s * dx, s * dy
    hh, hw = s * h, s * w
    interior = hr_slice[..., 0:hh - hdy, -hdx:]
    shifted = prev_detail[..., hdy:hh, 0:hw + hdx]
    assert interior.shape == shifted.shape
    assert (interior - shifted).abs().max() < 1e-4  # float32 grid_sample positions, on unit-variance noise
    # and not by the unscaled flow
    wrong = prev_detail[..., dy:hh, 0:hw + dx]
    assert (hr_slice[..., 0:hh - dy, -dx:] - wrong).abs().max() > 1e-2


def test_zero_state_on_requested_size():
    net = DetailNet(scale=2, hidden=8)
    prev_detail, hidden = net.zero_state(3, 5, 7, torch.device("cpu"))
    assert prev_detail.shape == (3, 3, 10, 14) and hidden.shape == (3, 8, 5, 7)
    assert not prev_detail.any() and not hidden.any()
