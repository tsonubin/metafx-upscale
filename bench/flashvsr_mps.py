"""FlashVSR v1.1 (tiny streaming pipeline) on Apple-silicon MPS, without CUDA kernels.

Upstream (third_party/FlashVSR @ cf910c6) needs mit-han-lab's CUDA block-sparse
attention, float64 RoPE/timestep embeddings and hard-coded "cuda" devices. This module
patches those at import time instead of editing upstream files:
  * block_sparse_attn_func → dense attention with FlashVSR's own 128-token block mask
    applied by broadcasting (rows with no allowed block output zeros)
  * float64 → float32 / complex64 (MPS has no float64)
  * devices → mps
"""
import math
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent / "third_party"
REPO = ROOT / "FlashVSR"
WEIGHTS = ROOT / "weights" / "FlashVSR-v1.1"
BLOCK = 128
Q_CHUNK_BLOCKS = 2  # query blocks per matmul; scores are H x 256 x Sk fp32 per chunk


def block_sparse_attn_func(q, k, v, cu_seqlens_q, cu_seqlens_k, head_mask_type, streaming_info,
                           base_blockmask, max_seqlen_q, max_seqlen_k, p_dropout,
                           deterministic=False, softmax_scale=None, is_causal=False,
                           exact_streaming=False, return_attn_probs=False):
    """q (Sq,H,D), k/v (Sk,H,D); base_blockmask (1,H,QB,KB) bool over 128-token blocks → (Sq,H,D).

    Dense scores with the block mask applied by broadcasting, so every call has the same
    shapes (MPS caches one compiled graph per shape; data-dependent shapes made it recompile
    per chunk and grow memory without bound)."""
    sq, h, d = q.shape
    sk = k.shape[0]
    qb_n, kb_n = sq // BLOCK, sk // BLOCK
    assert qb_n * BLOCK == sq and kb_n * BLOCK == sk
    mask = base_blockmask[0]  # H, QB, KB
    scale = softmax_scale or d ** -0.5
    qh = q.view(qb_n, BLOCK, h, d).permute(2, 0, 1, 3)  # H, QB, 128, D
    kt = k.permute(1, 2, 0)                              # H, D, Sk
    vh = v.permute(1, 0, 2)                              # H, Sk, D
    out = torch.empty_like(qh)
    for qs in range(0, qb_n, Q_CHUNK_BLOCKS):
        qe = min(qs + Q_CHUNK_BLOCKS, qb_n)
        c = qe - qs
        scores = torch.matmul(qh[:, qs:qe].reshape(h, c * BLOCK, d), kt).float().mul_(scale)
        scores = scores.view(h, c, BLOCK, kb_n, BLOCK)
        scores.masked_fill_(~mask[:, qs:qe, None, :, None], float("-inf"))
        probs = scores.view(h, c * BLOCK, sk).softmax(-1).nan_to_num_(0.0)  # fully masked rows → 0
        out[:, qs:qe] = torch.matmul(probs.to(v.dtype), vh).view(h, c, BLOCK, d)
    return out.permute(1, 2, 0, 3).reshape(sq, h, d)


def _install_patches():
    shim = types.ModuleType("block_sparse_attn")
    shim.block_sparse_attn_func = block_sparse_attn_func
    sys.modules["block_sparse_attn"] = shim
    # Upstream's model downloader imports modelscope; we load local weights only.
    ms = types.ModuleType("modelscope")
    ms.snapshot_download = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("downloads disabled"))
    sys.modules.setdefault("modelscope", ms)
    for p in (REPO, REPO / "examples" / "WanVSR"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    torch.cuda.empty_cache = lambda: None  # called unconditionally by the upstream base pipeline

    import diffsynth.models.wan_video_dit as wdit
    from einops import rearrange

    def sinusoidal_embedding_1d(dim, position):
        sinusoid = torch.outer(position.float(), torch.pow(
            10000, -torch.arange(dim // 2, dtype=torch.float32, device=position.device).div(dim // 2)))
        return torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1).to(position.dtype)

    def rope_apply(x, freqs, num_heads):
        x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
        x_out = torch.view_as_complex(x.float().reshape(x.shape[0], x.shape[1], x.shape[2], -1, 2))
        return torch.view_as_real(x_out * freqs).flatten(2).to(x.dtype)

    wdit.sinusoidal_embedding_1d = sinusoidal_embedding_1d
    wdit.rope_apply = rope_apply
    import diffsynth.pipelines.flashvsr_full as ffull
    import diffsynth.pipelines.flashvsr_tiny as ftiny
    ftiny.sinusoidal_embedding_1d = sinusoidal_embedding_1d
    ffull.sinusoidal_embedding_1d = sinusoidal_embedding_1d
    torch.cuda.synchronize = torch.mps.synchronize  # the full pipeline times each step


def load_pipeline(device="mps", dtype=torch.bfloat16, full=False):
    """full=False: the tiny pipeline (TCDecoder, 4 frames per latent). full=True: decode with
    the original Wan2.1 video VAE instead."""
    _install_patches()
    from diffsynth import FlashVSRFullPipeline, FlashVSRTinyPipeline, ModelManager
    from utils.TCDecoder import build_tcdecoder
    from utils.utils import Causal_LQ4x_Proj

    mm = ModelManager(torch_dtype=dtype, device="cpu")
    files = [str(WEIGHTS / "diffusion_pytorch_model_streaming_dmd.safetensors")]
    if full:
        files.append(str(WEIGHTS / "Wan2.1_VAE.pth"))
    mm.load_models(files)
    pipe = (FlashVSRFullPipeline if full else FlashVSRTinyPipeline).from_model_manager(mm, device=device)
    dit = pipe.denoising_model()
    dit.freqs = tuple(f.to(torch.complex64) for f in dit.freqs)
    dit.LQ_proj_in = Causal_LQ4x_Proj(in_dim=3, out_dim=1536, layer_num=1)
    dit.LQ_proj_in.load_state_dict(torch.load(WEIGHTS / "LQ_proj_in.ckpt", map_location="cpu", weights_only=True),
                                   strict=True)
    dit.LQ_proj_in.to(device, dtype)
    if full:
        pipe.vae.model.encoder = None  # LQ_proj_in replaces the encoder
        pipe.vae.model.conv1 = None
    else:
        pipe.TCDecoder = build_tcdecoder(new_channels=[512, 256, 128, 128], device="cpu", dtype=dtype,
                                         new_latent_channels=16 + 768)
        missing = pipe.TCDecoder.load_state_dict(
            torch.load(WEIGHTS / "TCDecoder.ckpt", map_location="cpu", weights_only=True), strict=False)
        assert not missing.unexpected_keys, missing
        pipe.TCDecoder.to(device, dtype).eval()
    pipe.to(device)
    ctx = torch.load(REPO / "examples" / "WanVSR" / "prompt_tensor" / "posi_prompt.pth",
                     map_location="cpu", weights_only=True)
    pipe.init_cross_kv(context_tensor=ctx)
    return pipe


@torch.inference_mode()
def upscale_clip(pipe, lq, sparse_ratio=2.0, local_range=11, seed=0, tiled=True):
    """lq: (1,3,F,H,W) in [-1,1], already bicubic-upscaled 4x, H and W multiples of 128,
    F = 8n+1. Returns (3,F',H,W) in [-1,1]."""
    _, _, f, h, w = lq.shape
    return pipe(prompt="", negative_prompt="", cfg_scale=1.0, num_inference_steps=1, seed=seed,
                LQ_video=lq, num_frames=f, height=h, width=w, is_full_block=False, if_buffer=True,
                topk_ratio=sparse_ratio * 768 * 1280 / (h * w), kv_ratio=3.0, local_range=local_range,
                color_fix=True, tiled=tiled, progress_bar_cmd=lambda x: x)
