"""DetailNet: a small recurrent network that adds fine texture on top of Apple AI's 4K base frame.

It runs at source resolution (720p for x3 to 4K), warps its own previous texture and memory along
optical flow so invented texture stays locked to the scene, and its output is high-pass by
construction (`texture_only`), so it cannot change brightness, colour or shapes.

Conventions (shared with the training data):
* images are float tensors in [0, 1], shape (N, 3, H, W);
* flow is (N, 4, h, w) with channels [back_x, back_y, fwd_x, fwd_y] in pixels of its own grid
  (source resolution). `back` is the displacement from frame t to t-1: the pixel at p in frame t was
  at p + back(p) in frame t-1. `fwd` is the displacement from t-1 to t;
* `warp(x, flow_px)` samples x at p + flow(p), so `warp(prev, back)` aligns the previous frame to
  the current one.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

LUMA = (0.2126, 0.7152, 0.0722)  # Rec.709

# occlusion_mask thresholds (Sundaram et al. forward-backward consistency check)
LUMA_TOLERANCE = 0.05
FB_ABSOLUTE = 0.5
FB_RELATIVE = 0.01


def space_to_depth(x, s):
    """(N, C, s*h, s*w) -> (N, C*s*s, h, w)."""
    return F.pixel_unshuffle(x, s)


def depth_to_space(x, s):
    """(N, C*s*s, h, w) -> (N, C, s*h, s*w); the inverse of `space_to_depth`."""
    return F.pixel_shuffle(x, s)


def upsample_flow(flow_px, s):
    """Flow in pixels of a grid -> flow in pixels of the grid s times finer: bilinear x s on the grid,
    values x s. Works for the 4-channel flow and for 2-channel halves alike."""
    return F.interpolate(flow_px, scale_factor=s, mode="bilinear", align_corners=False) * s


def sample_positions(flow_px):
    """(x, y), each (N, H, W): where p + flow_px(p) lands, in the pixels of flow_px's own grid."""
    h, w = flow_px.shape[-2:]
    ys, xs = torch.meshgrid(torch.arange(h, device=flow_px.device, dtype=flow_px.dtype),
                            torch.arange(w, device=flow_px.device, dtype=flow_px.dtype), indexing="ij")
    return xs + flow_px[:, 0], ys + flow_px[:, 1]


def warp(x, flow_px):
    """Sample x (N, C, H, W) at p + flow_px(p); flow_px is (N, 2, H, W) = [dx, dy] in x's pixels.
    Bilinear, border padding (out-of-frame samples repeat the edge pixel)."""
    if flow_px.shape[-2:] != x.shape[-2:] or flow_px.shape[0] != x.shape[0] or flow_px.shape[1] != 2:
        raise ValueError(f"flow {tuple(flow_px.shape)} does not match image {tuple(x.shape)}")
    _, _, h, w = x.shape
    px, py = sample_positions(flow_px)
    gx = px * (2.0 / max(w - 1, 1)) - 1.0  # align_corners=True: pixel i <-> -1 + 2i/(size-1)
    gy = py * (2.0 / max(h - 1, 1)) - 1.0
    grid = torch.stack((gx, gy), dim=-1)
    return F.grid_sample(x, grid.to(x.dtype), mode="bilinear", padding_mode="border", align_corners=True)


def texture_only(r, s):
    """r - bicubic_up(area_down(r, s), s): what an s x s block average and a smooth re-expansion cannot
    explain, i.e. the part of r finer than the source grid. H and W must be divisible by s."""
    if r.shape[-2] % s or r.shape[-1] % s:
        raise ValueError(f"texture_only needs H and W divisible by {s}, got {tuple(r.shape[-2:])}")
    low = F.interpolate(F.avg_pool2d(r, s), scale_factor=s, mode="bicubic", align_corners=False)
    return r - low


def luma(x):
    return LUMA[0] * x[:, 0:1] + LUMA[1] * x[:, 1:2] + LUMA[2] * x[:, 2:3]


def tracking_maps(x_t, x_prev, flow4):
    """(N, 2, h, w) evidence of how far the flow can be trusted:
    channel 0: luma error |luma(x_t) - luma(warp(x_prev, back))|;
    channel 1: forward-backward inconsistency |back + warp(fwd, back)| in pixels (vector length)."""
    back, fwd = flow4[:, 0:2], flow4[:, 2:4]
    luma_error = (luma(x_t) - luma(warp(x_prev, back))).abs()
    fb = torch.linalg.vector_norm(back + warp(fwd, back), dim=1, keepdim=True)
    return torch.cat((luma_error, fb), dim=1)


def occlusion_mask(maps, flow4):
    """(N, 1, h, w) bool, True where the track can be trusted (visible, flow consistent):
    luma error < 0.05, fb^2 < 0.5 + 0.01 * (|b|^2 + |f|^2) with b = back and f = warp(fwd, back),
    and the backward-flow target p + back(p) inside the frame (x in [0, W-1], y in [0, H-1]).
    `warp` clamps to the border, so without the last test a pan would lock texture toward
    border-replicated detail at the leading edge.
    `maps` is the output of `tracking_maps` for the same `flow4`. The forward-backward test is the
    standard squared form (the tolerance grows with the squared motion), so channel 1 (a length in
    pixels) is squared before it is compared."""
    back = flow4[:, 0:2]
    fwd = warp(flow4[:, 2:4], back)
    motion_sq = back.pow(2).sum(1, keepdim=True) + fwd.pow(2).sum(1, keepdim=True)
    luma_ok = maps[:, 0:1] < LUMA_TOLERANCE
    fb_ok = maps[:, 1:2].pow(2) < FB_ABSOLUTE + FB_RELATIVE * motion_sq
    h, w = back.shape[-2:]
    px, py = sample_positions(back)
    inside = ((px >= 0) & (px <= w - 1) & (py >= 0) & (py <= h - 1)).unsqueeze(1)
    return luma_ok & fb_ok & inside


def compose(base, detail, strength):
    """base + strength * detail, clamped to [0, 1]. strength 0 gives back the base exactly."""
    return (base + strength * detail).clamp(0.0, 1.0)


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        return x + self.conv2(F.relu(self.conv1(x)))


class DetailNet(nn.Module):
    """Recurrent texture generator. Per frame, at source resolution, it sees
    x_t (3), s2d(base) (3 s^2), s2d(warp(prev_detail, up(back))) (3 s^2), warp(hidden, back) (hidden)
    and tracking_maps (2); conv3x3 -> ResBlocks -> two heads: a texture head (3 s^2, depth-to-space
    to full resolution and passed through `texture_only`) and a hidden-state head (tanh, bounded)."""

    def __init__(self, scale=3, channels=64, blocks=12, hidden=32):
        super().__init__()
        self.scale = scale
        self.hidden_channels = hidden
        in_channels = 3 + 2 * 3 * scale * scale + hidden + 2
        self.conv_in = nn.Conv2d(in_channels, channels, 3, padding=1)
        self.blocks = nn.Sequential(*[ResBlock(channels) for _ in range(blocks)])
        self.head_tex = nn.Conv2d(channels, 3 * scale * scale, 3, padding=1)
        self.head_hid = nn.Conv2d(channels, hidden, 3, padding=1)
        # An untrained net should read as "Apple AI unchanged": start the texture head near zero (small
        # weights rather than exactly zero, so gradients still reach the layers before it).
        with torch.no_grad():
            self.head_tex.weight.mul_(0.01)
            self.head_tex.bias.zero_()

    def zero_state(self, n, h, w, device):
        """(prev_detail, hidden) for the first frame of a sequence; h, w are source-resolution."""
        s = self.scale
        return (torch.zeros(n, 3, s * h, s * w, device=device),
                torch.zeros(n, self.hidden_channels, h, w, device=device))

    def forward(self, x_t, x_prev, base, flow4, prev_detail, hidden):
        """-> (detail, hidden): detail (N, 3, s*h, s*w) is the zero-DC texture to add to `base`;
        hidden (N, hidden, h, w) in [-1, 1] is the memory for the next frame."""
        s = self.scale
        back = flow4[:, 0:2]
        prev_detail_warped = warp(prev_detail, upsample_flow(back, s))
        hidden_warped = warp(hidden, back)
        stack = torch.cat((x_t,
                           space_to_depth(base, s),
                           space_to_depth(prev_detail_warped, s),
                           hidden_warped,
                           tracking_maps(x_t, x_prev, flow4)), dim=1)
        features = self.blocks(self.conv_in(stack))
        detail = texture_only(depth_to_space(self.head_tex(features), s), s)
        return detail, torch.tanh(self.head_hid(features))
