"""Losses for DetailNet: look like FlashVSR's teacher frames, stay temporally locked.

Images are float tensors in [0, 1], shape (N, 3, H, W). Nothing here detaches its inputs;
gradients flow to every tensor argument that requires them.
"""
import lpips
import torch
import torch.nn as nn
from DISTS_pytorch import DISTS

from detailnet.model import warp


class Perceptual(nn.Module):
    """DISTS + LPIPS-VGG (weight 1:1) between `pred` and `target`, both (N, 3, H, W) in [0, 1].
    Returns the batch mean as a scalar. Both metrics are frozen (no parameter gradients, eval mode);
    gradients still flow back to `pred` (and `target`, if it requires them)."""

    def __init__(self):
        super().__init__()
        self.dists = DISTS()
        self.lpips = lpips.LPIPS(net="vgg", verbose=False)
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        return super().train(False)  # always eval: the metrics are fixed measuring sticks

    def per_item(self, pred, target):
        """(N,) DISTS + LPIPS of each item; both are per-image, so an item's value does not depend on the others."""
        dists = self.dists(pred, target, require_grad=True).reshape(-1)  # DISTS squeezes a batch of 1
        lp = self.lpips(2.0 * pred - 1.0, 2.0 * target - 1.0).reshape(-1)  # LPIPS wants [-1, 1]
        return dists + lp

    def forward(self, pred, target):
        return self.per_item(pred, target).mean()


def texture_lock(d_t, d_prev, back_flow_hr, mask_hr):
    """Temporal lock of the invented texture: mean over the masked pixels of |d_t - warp(d_prev, back)|.
    d_t, d_prev (N, 3, H, W) at output resolution; back_flow_hr (N, 2, H, W) in output pixels; mask_hr
    (N, 1, H, W) bool (or 0/1 float), broadcast over the channels. Exactly 0 (not NaN) when nothing is
    masked, and the result stays attached to the graph with zero gradients."""
    diff = (d_t - warp(d_prev, back_flow_hr)).abs()
    keep = (mask_hr != 0).expand_as(diff)
    total = torch.where(keep, diff, torch.zeros_like(diff)).sum()
    return total / keep.sum().clamp_min(1)


def ping_pong(fwd, bwd):
    """Mean |fwd - bwd|: a sequence run forward and the same sequence run backward should agree."""
    return (fwd - bwd).abs().mean()
