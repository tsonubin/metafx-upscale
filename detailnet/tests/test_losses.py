import pytest
import torch

from detailnet.losses import Perceptual, ping_pong, texture_lock
from detailnet.model import warp



@pytest.fixture(autouse=True)
def seeded():
    torch.manual_seed(0)  # per test, so results do not depend on test order or selection


# ---------------------------------------------------------------- texture_lock

def shifted_pair(h=24, w=24, dx=2):
    prev = torch.rand(1, 3, h, w)
    cur = torch.zeros_like(prev)
    cur[..., dx:] = prev[..., :-dx]
    back = torch.zeros(1, 2, h, w)
    back[:, 0] = -float(dx)
    return cur, prev, back


def test_texture_lock_zero_when_consistent():
    cur, prev, back = shifted_pair()
    mask = torch.ones(1, 1, 24, 24, dtype=torch.bool)
    mask[..., :2] = False  # the first dx columns have no source
    assert float(texture_lock(cur, prev, back, mask)) < 1e-5


def test_texture_lock_measures_inconsistency_inside_the_mask_only():
    cur, prev, back = shifted_pair()
    mask = torch.zeros(1, 1, 24, 24, dtype=torch.bool)
    mask[..., 4:12, 4:12] = True
    broken = cur.clone()
    broken[..., 4:12, 4:12] += 0.25
    assert abs(float(texture_lock(broken, prev, back, mask)) - 0.25) < 1e-5
    broken2 = cur.clone()
    broken2[..., 16:, 16:] += 1.0  # outside the mask: ignored
    assert float(texture_lock(broken2, prev, back, mask)) < 1e-5


def test_texture_lock_accepts_float_mask_and_matches_manual_mean():
    d_t, d_prev = torch.rand(2, 3, 10, 10), torch.rand(2, 3, 10, 10)
    back = torch.randn(2, 2, 10, 10)
    mask = (torch.rand(2, 1, 10, 10) > 0.5)
    expected = (d_t - warp(d_prev, back)).abs().masked_select(mask.expand(2, 3, 10, 10)).mean()
    assert torch.allclose(texture_lock(d_t, d_prev, back, mask), expected, atol=1e-6)
    assert torch.allclose(texture_lock(d_t, d_prev, back, mask.float()), expected, atol=1e-6)


def test_texture_lock_empty_mask_is_zero_not_nan():
    d_t = torch.rand(1, 3, 12, 12, requires_grad=True)
    d_prev = torch.rand(1, 3, 12, 12, requires_grad=True)
    back = torch.zeros(1, 2, 12, 12)
    mask = torch.zeros(1, 1, 12, 12, dtype=torch.bool)
    loss = texture_lock(d_t, d_prev, back, mask)
    assert loss.shape == () and float(loss) == 0.0 and not torch.isnan(loss)
    loss.backward()  # still differentiable, with zero (not NaN) gradients
    assert torch.isfinite(d_t.grad).all() and float(d_t.grad.abs().sum()) == 0.0


def test_texture_lock_gradient_flows_to_current_and_previous_detail():
    d_t = torch.rand(1, 3, 12, 12, requires_grad=True)
    d_prev = torch.rand(1, 3, 12, 12, requires_grad=True)
    back = torch.full((1, 2, 12, 12), 0.5)
    texture_lock(d_t, d_prev, back, torch.ones(1, 1, 12, 12, dtype=torch.bool)).backward()
    assert d_t.grad.abs().sum() > 0 and d_prev.grad.abs().sum() > 0


# ---------------------------------------------------------------- ping_pong

def test_ping_pong_is_mean_abs_difference():
    a = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    b = torch.tensor([[1.0, 1.0], [0.0, 6.0]])
    assert float(ping_pong(a, b)) == (1 + 0 + 2 + 3) / 4
    assert float(ping_pong(a, a)) == 0.0


# ---------------------------------------------------------------- Perceptual

def test_perceptual_identity_near_zero_and_positive_for_noise():
    perceptual = Perceptual()
    x = torch.rand(2, 3, 64, 64)
    same = perceptual(x, x.clone())
    assert same.shape == () and abs(float(same)) < 1e-3, float(same)
    noise = perceptual(x, torch.rand(2, 3, 64, 64))
    assert float(noise) > 0.1, float(noise)


def test_perceptual_frozen_but_gradient_reaches_inputs():
    perceptual = Perceptual()
    assert not any(p.requires_grad for p in perceptual.parameters())
    assert not perceptual.training
    pred = torch.rand(1, 3, 64, 64, requires_grad=True)
    target = torch.rand(1, 3, 64, 64)
    perceptual(pred, target).backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all() and pred.grad.abs().sum() > 0
    perceptual.train()  # stays frozen in eval mode
    assert not perceptual.training


def test_perceptual_is_lower_for_a_closer_image():
    perceptual = Perceptual()
    target = torch.rand(1, 3, 64, 64)
    near = (target + 0.02 * torch.randn_like(target)).clamp(0, 1)
    far = (target + 0.3 * torch.randn_like(target)).clamp(0, 1)
    assert float(perceptual(near, target)) < float(perceptual(far, target))


class ZeroTerm(torch.nn.Module):
    """Stands in for one of the two metrics and contributes nothing, to isolate the other."""

    def forward(self, a, b, *args, **kwargs):
        return torch.zeros(a.shape[0], 1, 1, 1)


def grad_of(loss_fn, pred):
    pred = pred.detach().clone().requires_grad_(True)
    loss_fn(pred).backward()
    return pred.grad


def test_perceptual_dists_term_alone_gives_the_full_gradient():
    # DISTS' default (require_grad=False) runs its VGG stages under no_grad, yet still hands back a small
    # gradient through the raw-pixel statistics, so "grad is nonzero" would not catch the missing flag.
    # Compare against the real thing instead: the gradient through every stage.
    perceptual = Perceptual()
    perceptual.lpips = ZeroTerm()
    pred, target = torch.rand(2, 3, 64, 64), torch.rand(2, 3, 64, 64)
    got = grad_of(lambda p: perceptual(p, target), pred)
    full = grad_of(lambda p: perceptual.dists(p, target, require_grad=True).mean(), pred)
    pixel_only = grad_of(lambda p: perceptual.dists(p, target, require_grad=False).mean(), pred)
    assert torch.isfinite(got).all() and got.abs().sum() > 0
    assert (pixel_only - full).abs().sum() > 0.5 * full.abs().sum()  # the trap is real: these differ a lot
    assert torch.allclose(got, full, rtol=1e-4, atol=1e-9)


def test_perceptual_lpips_term_alone_gives_the_full_gradient():
    perceptual = Perceptual()
    perceptual.dists = ZeroTerm()
    pred, target = torch.rand(2, 3, 64, 64), torch.rand(2, 3, 64, 64)
    got = grad_of(lambda p: perceptual(p, target), pred)
    full = grad_of(lambda p: perceptual.lpips(2 * p - 1, 2 * target - 1).mean(), pred)
    assert torch.isfinite(got).all() and got.abs().sum() > 0
    assert torch.allclose(got, full, rtol=1e-4, atol=1e-9)


def test_perceptual_is_dists_plus_lpips_vgg_one_to_one():
    perceptual = Perceptual()
    pred, target = torch.rand(2, 3, 64, 64), torch.rand(2, 3, 64, 64)
    with torch.no_grad():
        dists = perceptual.dists(pred, target).mean()
        lp = perceptual.lpips(2 * pred - 1, 2 * target - 1).mean()  # LPIPS takes [-1, 1]
        both = perceptual(pred, target)
    assert float(dists) > 0.05 and float(lp) > 0.05
    assert abs(float(both) - (float(dists) + float(lp))) < 1e-5
