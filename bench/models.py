"""Benchmark model wrappers. Every model exposes

    scale: int                          native upscale factor
    temporal: bool                      True if it uses neighbouring frames
    run_shot(frames, emit)              frames: list of HxWx3 uint8 RGB (one shot, in order);
                                        emit(i, hr) with hr a (3,H,W) float tensor in [0,1]

The harness splits clips at scene cuts and calls run_shot once per shot, so recurrent
models never carry state across a cut unless the harness is told to ignore cuts.
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from archs import RealBasicVSRNet, RRDBNet, SRVGGNetCompact

WEIGHTS = Path(__file__).resolve().parent.parent / "third_party" / "weights"
DEVICE = torch.device("mps")
DTYPE = torch.float16


def to_tensor(frame):
    return torch.from_numpy(frame).to(DEVICE).permute(2, 0, 1).to(DTYPE).div_(255.0)


def sync():
    torch.mps.synchronize()


class Bicubic:
    """Floor: plain bicubic resize (what a TV does)."""
    scale, temporal, label = 4, False, "Bicubic"

    def run_shot(self, frames, emit):
        for i, f in enumerate(frames):
            x = to_tensor(f)[None].float()
            emit(i, F.interpolate(x, scale_factor=4, mode="bicubic", align_corners=False)[0].clamp_(0, 1))


class PerFrame:
    """Single-image SR applied to each frame independently, tiled to bound memory."""
    temporal = False
    tile, pad = 512, 16

    def __init__(self, net):
        self.net = net.to(DEVICE, DTYPE).eval()

    @torch.inference_mode()
    def upscale(self, x):
        _, _, h, w = x.shape
        s = self.scale
        out = x.new_empty(1, 3, h * s, w * s)
        for y0 in range(0, h, self.tile):
            for x0 in range(0, w, self.tile):
                y1, x1 = min(y0 + self.tile, h), min(x0 + self.tile, w)
                py0, px0 = max(y0 - self.pad, 0), max(x0 - self.pad, 0)
                py1, px1 = min(y1 + self.pad, h), min(x1 + self.pad, w)
                o = self.net(x[:, :, py0:py1, px0:px1])
                out[:, :, y0 * s:y1 * s, x0 * s:x1 * s] = o[:, :, (y0 - py0) * s:(y0 - py0 + y1 - y0) * s,
                                                            (x0 - px0) * s:(x0 - px0 + x1 - x0) * s]
        return out

    def run_shot(self, frames, emit):
        for i, f in enumerate(frames):
            emit(i, self.upscale(to_tensor(f)[None])[0].float().clamp_(0, 1))


class RealESRGAN(PerFrame):
    scale, label = 4, "Real-ESRGAN x4plus"

    def __init__(self):
        net = RRDBNet()
        net.load_state_dict(torch.load(WEIGHTS / "RealESRGAN_x4plus.pth", map_location="cpu", weights_only=True)["params_ema"])
        super().__init__(net)


class RealESRGANGeneral(PerFrame):
    scale, label = 4, "Real-ESRGAN general-x4v3"

    def __init__(self):
        net = SRVGGNetCompact()
        net.load_state_dict(torch.load(WEIGHTS / "realesr-general-x4v3.pth", map_location="cpu", weights_only=True)["params"])
        super().__init__(net)


class RealBasicVSR:
    """Bidirectional recurrent VSR. Long shots are processed in overlapping windows
    (memory grows with window length); outputs come from each window's centre part."""
    scale, temporal, label = 4, True, "RealBasicVSR"
    window, overlap = 24, 6

    def __init__(self):
        self.net = RealBasicVSRNet()
        ck = torch.load(WEIGHTS / "realbasicvsr_x4.pth", map_location="cpu", weights_only=False)["state_dict"]
        self.net.load_state_dict({k[len("generator_ema."):]: v for k, v in ck.items()
                                  if k.startswith("generator_ema.")})
        # SPyNet's flow pyramid needs fp32 precision; the rest runs in fp16.
        self.net.to(DEVICE, DTYPE).eval()
        self.net.basicvsr.spynet.float()
        spy = self.net.basicvsr.spynet
        spy_forward = spy.forward
        spy.forward = lambda a, b: spy_forward(a.float(), b.float()).to(DTYPE)

    @torch.inference_mode()
    def run_shot(self, frames, emit):
        n = len(frames)
        if n == 1:  # BasicVSR needs at least two frames
            frames = frames * 2
        step = self.window - 2 * self.overlap
        starts = list(range(0, max(n - self.window, 0) + 1, step))
        if starts[-1] + self.window < n:
            starts.append(n - self.window)
        done = set()
        for s in starts:
            e = min(s + self.window, len(frames))
            lqs = torch.stack([to_tensor(f) for f in frames[s:e]])[None]
            keep_from = s + (self.overlap if s > 0 else 0)
            keep_to = e - (self.overlap if e < n else 0)

            def on_frame(j, hr, s=s, keep_from=keep_from, keep_to=keep_to):
                i = s + j
                if keep_from <= i < keep_to and i < n and i not in done:
                    done.add(i)
                    emit(i, hr[0].float().clamp_(0, 1))

            self.net(lqs, on_frame)
        # windows that start late may leave a gap at the overlap boundary; none expected
        assert len(done) == n, f"RealBasicVSR covered {len(done)}/{n} frames"


class FlashVSR:
    """FlashVSR v1.1 tiny: one-step streaming video diffusion (Wan2.1-1.3B based) with a KV cache
    over past frames and a tiny conditional decoder. Runs a whole shot per call."""
    scale, temporal, label = 4, True, "FlashVSR v1.1 (tiny, streaming)"
    warmup_frames = 21

    full = False

    def __init__(self):
        import flashvsr_mps
        self.fv = flashvsr_mps
        self.pipe = flashvsr_mps.load_pipeline(full=self.full)

    @torch.inference_mode()
    def run_shot(self, frames, emit):
        from PIL import Image
        n = len(frames)
        h, w = frames[0].shape[:2]
        ph, pw = (-h) % 32, (-w) % 32  # 4x output must be a multiple of 128
        total = max(n + 4, 25)
        total = (total - 1 + 7) // 8 * 8 + 1  # 8k+1 frames
        ups = []
        for i in list(range(n)) + [n - 1] * (total - n):
            f = np.pad(frames[i], ((0, ph), (0, pw), (0, 0)), mode="reflect")
            up = Image.fromarray(f).resize((f.shape[1] * 4, f.shape[0] * 4), Image.BICUBIC)
            ups.append(torch.from_numpy(np.asarray(up)).to(DEVICE).permute(2, 0, 1)
                       .to(torch.bfloat16).div_(127.5).sub_(1.0))
        lq = torch.stack(ups, 1)[None]
        del ups
        out = self.fv.upscale_clip(self.pipe, lq)
        assert out.shape[1] >= n, f"FlashVSR returned {out.shape[1]} frames for {n}"
        for i in range(n):
            emit(i, out[:, i, :h * 4, :w * 4].float().add(1.0).mul(0.5).clamp(0, 1))


class RealViformer:
    """RealViformer (ECCV 2024): forward-only recurrent VSR with channel-attention transformer
    blocks; SPyNet-aligned hidden state from the previous frame. Run frame by frame so outputs
    stream out instead of being stacked (the upstream forward keeps every 4x frame in memory)."""
    scale, temporal, label = 4, True, "RealViformer"

    def __init__(self, dtype=torch.float32):
        import importlib
        import importlib.util
        import sys
        # Upstream's package is called `archs`, like ours; load it under another name.
        pkg = WEIGHTS.parent / "RealViformer" / "archs"
        spec = importlib.util.spec_from_file_location("rvf_archs", pkg / "__init__.py",
                                                      submodule_search_locations=[str(pkg)])
        sys.modules["rvf_archs"] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sys.modules["rvf_archs"])
        Net = importlib.import_module("rvf_archs.realviformer_arch").RealViformer
        flow_warp = importlib.import_module("rvf_archs.arch_util").flow_warp
        self.flow_warp = flow_warp
        self.dtype = dtype
        self.net = Net(num_feat=48, num_blocks=[2, 3, 4, 1], spynet_path=None, heads=[1, 2, 4],
                       ffn_expansion_factor=2.66, merge_head=2, bias=False, LayerNorm_type="BiasFree",
                       ch_compress=True, squeeze_factor=[4, 4, 4], masked=True)
        sd = torch.load(WEIGHTS / "realviformer" / "RealViformer_model_weights" / "weights.pth",
                        map_location="cpu", weights_only=True)["params"]
        self.net.load_state_dict(sd, strict=False)  # only an unused 'masktemp' buffer is extra
        self.net.to(DEVICE, dtype).eval()

    @torch.inference_mode()
    def run_shot(self, frames, emit):
        n = self.net
        h, w = frames[0].shape[:2]
        ph, pw = (-h) % 4, (-w) % 4
        feat = prev = None
        for i, f in enumerate(frames):
            x = to_tensor(f)[None].to(self.dtype)
            if ph or pw:  # same front padding as upstream inference
                x = F.pad(x, (pw, 0, ph, 0), mode="reflect")
            shallow = n.shallow_extraction(x)
            if feat is None:
                feat = x.new_zeros(1, n.num_feat, x.shape[2], x.shape[3])
            else:
                flow = n.spynet(x, prev)
                feat = self.flow_warp(feat, flow.permute(0, 2, 3, 1))
            feat = n.attn_merge(shallow, feat)
            e1 = n.encoder_level1(feat)
            e2 = n.encoder_level2(n.down1_2(e1))
            lat = n.latent(n.down2_3(e2))
            d2 = n.decoder_level2(n.reduce_chan_level2(torch.cat([n.up3_2(lat), e2], 1)))
            d1 = n.decoder_level1(torch.cat([n.up2_1(d2), e1], 1))
            out = n.refinement(d1)
            feat = n.compress(out)
            out = n.lrelu(n.pixel_shuffle(n.upconv1(out)))
            out = n.lrelu(n.pixel_shuffle(n.upconv2(out)))
            out = n.conv_last(n.lrelu(n.conv_hr(out)))
            out = out + F.interpolate(x, scale_factor=4, mode="bilinear", align_corners=False)
            prev = x
            emit(i, out[0, :, ph * 4:, pw * 4:].float().clamp_(0, 1))


class FlashVSRFull(FlashVSR):
    """Same diffusion model, decoded by the original Wan2.1 VAE instead of the tiny TCDecoder."""
    label, full = "FlashVSR v1.1 (full, Wan VAE)", True


REGISTRY = {
    "bicubic": Bicubic,
    "realesrgan": RealESRGAN,
    "realesrgan-general": RealESRGANGeneral,
    "realbasicvsr": RealBasicVSR,
    "flashvsr": FlashVSR,
    "flashvsr-full": FlashVSRFull,
    "realviformer": RealViformer,
}
