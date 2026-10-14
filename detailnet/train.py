"""Train DetailNet: imitate FlashVSR teacher frames and real 1080p frames, keep the texture locked over time.

    python -m detailnet.train --data output/detail-data --out output/detail-runs/r1 [--iters N] [--resume] [--force]
    python -m detailnet.train --data output/detail-data --out output/detail-runs/r2 --iters 30000
                              --init-from output/detail-runs/r1/latest.pt      (a tuning round: r1's weights, new run)

What a step does. A batch is `batch` sequences of T frames from `detailnet.data.SequenceDataset` (T = 20: 8 burn-in
frames plus 12 loss frames; 39 for a ping-pong batch: the 20 forward, then 18..0 replayed backwards). DetailNet runs
over every frame in order with gradients (truncated BPTT over the whole sequence). Where a sequence has `reset` set, the
network starts cold there: prev_detail and hidden are zero, the flow is zero and x_prev = x_t, per batch element, with
masks rather than branches (`prepare_inputs`, `run_sequence`). The output is compose(base, detail, 1.0). Only frames
with `loss_mask` are scored, per item: the items of a batch may score different frames (a teacher item does not score
the first frames from a cut, R34), and every term is a mean over the (item, frame) pairs scored:

    perceptual (DISTS + LPIPS)  vs the target, teacher and gt batches         w_perc
    L1                          vs the target, gt batches only                w_l1
    texture lock                |d_t - warp(d_t-1, back)| where occlusion_mask trusts the track, at output
                                resolution (mask upsampled nearest x3, flow x3); never at a reset frame     w_lock
    ping-pong                   |out_k - out_(T-1-k)| for the forward frame k and its reversed twin          w_pp

Batches alternate teacher, gt, teacher, ...; each has probability `pp_prob` (0.2) of being a ping-pong batch.

Memory. The perceptual nets are the memory hog, so the perceptual term is backpropagated in two stages: the recurrent
forward keeps the DetailNet graph; then, frame by frame, the output is copied out of the graph, DISTS+LPIPS run on the
copy, and their backward gives dL/d(output) while the VGG graph is freed at once; finally one `autograd.backward`
pushes those gradients (and the cheap terms' graph) through DetailNet. The result equals a single backward through
everything. `grad_checkpoint` (on by default) additionally recomputes each DetailNet frame in the backward instead of
keeping its activations: at batch 4 and T = 39 it takes the peak MPS memory from 23.9 GB (18.7 GB of live tensors) to
8.5 GB (2.4 GB live) for about 10 % more time. `perc_frames` scores only a random subset of the loss frames
perceptually, drawn per item from that item's own loss frames and weighted so the sum is an unbiased estimate of the
mean over all scored (item, frame) pairs (`perceptual_plan`; default: all of them).

Epochs. An epoch is one pass over the larger of the teacher and gt datasets, in whole batches (the last batch wraps
into the next pass). Every epoch rebuilds all datasets and loaders with seed = cfg.seed + 1_000_003 * epoch (new
crops, and sequences finished meanwhile by the FlashVSR job). Each kind is served from a stream of seeded permutations
of all its sequences, re-drawn on every pass, so no sequence is starved: the plain and the ping-pong loaders of a kind
take consecutive slices of that stream, which together cover the dataset, and the smaller dataset is simply repeated
within the epoch, each repeat with another permutation and another crop seed.

Optimizer. AdamW (lr 2e-4, weight decay 1e-4) with cosine decay to 0 over `--iters`: before each step the lr is set to
0.5 * lr * (1 + cos(pi * it / iters)) from the step counter, not from a scheduler object, so a resumed run (whose
restored optimizer state carries a stale lr) is exactly on the curve. A step whose loss or gradient is not finite is
skipped and logged (no weight is touched, it does not count as an iteration); 10 skipped steps in a row end the run.

Outputs in `--out`: train.jsonl (one JSON line per `log_every` steps, and one per skipped step), ckpt_000100.pt, ... every `ckpt_every` steps and at
the end, latest.pt = {"model", "opt", "iter", "cfg"}. `--resume` continues from latest.pt (an error when there is none,
so a mistyped --out never starts from scratch). `--init-from CKPT` starts a new run from CKPT's weights with a fresh
optimizer at iteration 0, so the cosine schedule spans the new --iters (the tuning rounds: --init-from r1/latest.pt
--iters 30000 --out r2); it cannot be combined with --resume, and a resumed run of it uses --resume alone.

Guards: refuses to start while the user's batch runs (`--force` overrides), and looks again every 50 iterations: once the
batch runs it stops with exit 3 and a fresh latest.pt (rerun with --resume); stops the same way when swap grew by more
than 8 GB.
"""
import argparse
import json
import math
import os
import shutil
import signal
import sys
import time
from collections import Counter, defaultdict
from dataclasses import MISSING, asdict, dataclass, fields
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from torch.utils.data import DataLoader, Dataset

from detailnet.data import KINDS, LOSS, SequenceDataset
from detailnet.guards import (BATCH_EXIT_CODE, SWAP_EXIT_CODE, BatchGuard, BatchStarted, SwapGuard, SwapLimitExceeded,
                              refuse_if_batch_running)
from detailnet.losses import Perceptual, texture_lock
from detailnet.model import DetailNet, compose, occlusion_mask, tracking_maps, upsample_flow

SCALE = 3  # source px -> output px
EPOCH_SEED_STRIDE = 1_000_003  # seed step per epoch; larger than any dataset, so epochs never share (seed + index)
PASS_SEED_STRIDE = 10_007  # seed step per repeat of a short dataset inside an epoch (also larger than the datasets)
ARCHITECTURE = ("channels", "blocks", "hidden")
MAX_CONSECUTIVE_SKIPS = 10  # non-finite steps in a row after which the run is aborted
BATCH_CHECK_EVERY = 50  # iterations between looks for the user's batch


class NoDataError(RuntimeError):
    """The data directory has no finished sequence of either kind."""


@dataclass
class TrainConfig:
    data: str
    out: str
    iters: int = 100_000
    batch: int = 4
    lr: float = 2e-4
    weight_decay: float = 1e-4
    w_perc: float = 1.0
    w_l1: float = 1.0
    w_lock: float = 2.0
    w_pp: float = 0.5
    ckpt_every: int = 5000
    seed: int = 0
    channels: int = 64
    blocks: int = 12
    hidden: int = 32
    crop: tuple = (160, 90)  # (w, h) in source px of the training crops
    reset_prob: float = 0.15  # chance per item of one extra random reset at a loss frame
    pp_prob: float = 0.2  # chance per batch of being a ping-pong batch (T = 39)
    perc_frames: int = 12  # perceptual frames per 12 loss frames of an item, each step; 12 or more: all of them
    grad_checkpoint: bool = True  # recompute each DetailNet frame in the backward instead of keeping its activations
    grad_clip: float = 1.0  # max gradient norm; 0 disables clipping
    device: str = "mps"
    num_workers: int = 2  # per loader
    log_every: int = 10
    swap_limit_gb: float = 8.0
    resume: bool = False
    init_from: str = ""  # checkpoint whose weights start a new run (fresh optimizer, iteration 0); "" for none


# ---------------------------------------------------------------- device memory

class MemoryPeak:
    """Highest MPS memory seen at the points `sample()` is called: the driver's allocation (what the machine gives
    up, caches included) and the bytes held by live tensors. Does nothing off MPS."""

    def __init__(self):
        self.active = False
        self.driver_gb = 0.0
        self.live_gb = 0.0

    def start(self, device):
        self.active = torch.device(device).type == "mps"
        self.driver_gb = self.live_gb = 0.0

    def sample(self):
        if self.active:
            self.driver_gb = max(self.driver_gb, torch.mps.driver_allocated_memory() / 1e9)
            self.live_gb = max(self.live_gb, torch.mps.current_allocated_memory() / 1e9)


MEMORY = MemoryPeak()


# ---------------------------------------------------------------- the recurrent forward

def prepare_inputs(x, flow, reset):
    """What DetailNet is given as x_prev and flow, for x (N, T, 3, h, w), flow (N, T, 4, h, w), reset (N, T) bool.
    x_prev is the previous slot's x. At a reset frame of an element the network must start cold: its flow is zero and
    x_prev = x_t (so no motion and no luma error); the state is zeroed in `run_sequence`."""
    x_prev = torch.cat((x[:, :1], x[:, :-1]), dim=1)
    r = reset.view(*reset.shape, 1, 1, 1)
    return torch.where(r, x, x_prev), flow.masked_fill(r, 0.0)


def _frame(model, x_t, x_prev, base_t, flow_t, keep_t, prev_detail, hidden):
    return model(x_t, x_prev, base_t, flow_t, prev_detail * keep_t, hidden * keep_t)


def run_sequence(model, x, base, flow, reset, checkpoint=False):
    """Run DetailNet over all T frames; -> list of T details (N, 3, 3h, 3w), graph attached. `checkpoint` recomputes
    each frame in the backward instead of storing its activations."""
    n, t = x.shape[:2]
    x_prev, flow = prepare_inputs(x, flow, reset)
    keep = (~reset).to(x.dtype).view(n, t, 1, 1, 1)
    prev_detail, hidden = model.zero_state(n, x.shape[-2], x.shape[-1], x.device)
    details = []
    for i in range(t):
        args = (model, x[:, i], x_prev[:, i], base[:, i], flow[:, i], keep[:, i], prev_detail, hidden)
        if checkpoint:
            prev_detail, hidden = torch_checkpoint(_frame, *args, use_reentrant=False, preserve_rng_state=False)
        else:
            prev_detail, hidden = _frame(*args)
        details.append(prev_detail)
    return details


# ---------------------------------------------------------------- loss terms

def lock_term(details, x, x_prev, flow, reset, frames, scored=None):
    """(texture lock, mask coverage) over the scored `frames` (those with a predecessor): the mean over every trusted
    output-resolution pixel of the frames and the batch of |d_t - warp(d_t-1, back)|. The mask is `occlusion_mask` at
    source resolution (made from the inputs the network actually saw: `x_prev` and `flow` from `prepare_inputs`)
    upsampled nearest x3, and is empty at the reset frames of an element and, given `scored` (N, T) bool, at the frames
    an element does not score. `coverage` is the share of source pixels trusted."""
    frames = [t for t in frames if t >= 1]
    if not frames:
        return details[0].sum() * 0.0, torch.zeros((), device=details[0].device)
    d_t = torch.stack([details[t] for t in frames], dim=1).flatten(0, 1)
    d_prev = torch.stack([details[t - 1] for t in frames], dim=1).flatten(0, 1)
    with torch.no_grad():
        xs, xp, fl = (a[:, frames].flatten(0, 1) for a in (x, x_prev, flow))
        trusted = occlusion_mask(tracking_maps(xs, xp, fl), fl)
        trusted &= ~reset[:, frames].flatten().view(-1, 1, 1, 1)
        if scored is not None:
            trusted &= scored[:, frames].flatten().view(-1, 1, 1, 1)
        mask_hr = F.interpolate(trusted.float(), scale_factor=SCALE, mode="nearest") > 0.5
        back_hr = upsample_flow(fl[:, 0:2], SCALE)
        coverage = trusted.float().mean()
    return texture_lock(d_t, d_prev, back_hr, mask_hr), coverage


def pingpong_pairs(loss_mask):
    """(forward slots, reversed slots) of the frames that have a twin. A ping-pong item of T slots replays the forward
    half (slots 0 .. (T-1)/2) backwards without repeating the turning point, so the reversed slot of forward slot k is
    T-1-k (38-k for T = 39), for k = 1 .. 18, where both are scored."""
    mask = [bool(v) for v in loss_mask]
    t = len(mask)
    forward = [k for k in range(1, (t + 1) // 2 - 1) if mask[k] and mask[t - 1 - k]]
    return forward, [t - 1 - k for k in forward]


def masked_mean(values, mask):
    """Mean of `values` (N, F) over the entries where `mask` (N, F, bool) is True; 0 when none is."""
    total = torch.where(mask, values, torch.zeros_like(values)).sum()
    return total / mask.sum().clamp_min(1).to(values.dtype)


def choose_perceptual_frames(frames, perc_frames, rng):
    """The loss `frames` that get the perceptual term this step: `perc_frames` per 12 frames (so 4 of a plain item's 12,
    8 of a ping-pong item's 23), uniformly at random; every frame when perc_frames >= 12. The mean over a uniform
    subset has the full mean as its expectation."""
    n = len(frames)
    if perc_frames <= 0 or perc_frames >= LOSS:
        return list(frames)
    k = min(n, max(1, round(perc_frames * n / LOSS)))
    return sorted(frames[i] for i in rng.choice(n, size=k, replace=False))


def perceptual_plan(scored, perc_frames, rng):
    """{frame: [(item, weight)]}, frames ascending: the (item, frame) pairs that get the perceptual term this step.
    `scored` (N, T) bool is the batch's loss_mask. Each item draws its own subset of its own loss frames
    (`choose_perceptual_frames`: perc_frames per 12, all of them at 12 or more). The weight n_i / (k_i * total), for an
    item with n_i loss frames of which k_i were drawn and `total` scored pairs in the batch, makes the weighted sum an
    unbiased estimate of the mean over all scored pairs, and exactly that mean when every frame is drawn. The weights
    add up to 1."""
    scored = torch.as_tensor(scored).bool().cpu()
    total = int(scored.sum())
    plan = defaultdict(list)
    for i in range(scored.shape[0]):
        own = scored[i].nonzero().flatten().tolist()
        if not own:
            continue
        picked = choose_perceptual_frames(own, perc_frames, rng)
        weight = len(own) / (len(picked) * total)
        for t in picked:
            plan[t].append((i, weight))
    return dict(sorted(plan.items()))


def batch_kind(batch):
    kind = batch["kind"]
    kinds = {kind} if isinstance(kind, str) else set(kind)
    if len(kinds) != 1:
        raise ValueError(f"a batch has to be of one kind, got {sorted(kinds)}")
    return kinds.pop()


def compute_gradients(model, perceptual, batch, cfg, pingpong=False, rng=None):
    """Forward and backward of one batch (tensors already on the model's device); leaves the gradient in the
    parameters (added to what is there) and returns floats: the weighted total `loss`, its terms `perc`, `l1`, `lock`
    and `pp`, and `mask` (the share of source pixels the lock trusts). A term that does not apply is None: perc when
    w_perc is 0, l1 on a teacher batch, pp unless `pingpong` and some frame has a twin. Each term is the mean over the
    (item, frame) pairs that `loss_mask` (N, T) scores, so the items may score different frames. The perceptual term
    is backpropagated in two stages (see the module docstring)."""
    rng = rng if rng is not None else np.random.default_rng()
    x, base, flow, target, reset = (batch[k] for k in ("x", "base", "flow", "target", "reset"))
    scored = batch["loss_mask"].bool()  # (N, T): which frames each item scores
    frames = scored.any(0).nonzero().flatten().tolist()  # the frames some item scores
    if not frames:
        raise ValueError("the batch has no loss frames")
    kind = batch_kind(batch)

    # stage 1: the recurrent forward, with the DetailNet graph
    details = run_sequence(model, x, base, flow, reset, cfg.grad_checkpoint)
    outs = {t: compose(base[:, t], details[t], 1.0) for t in frames}
    MEMORY.sample()

    # the cheap terms stay in this graph
    x_prev, flow_eff = prepare_inputs(x, flow, reset)
    lock, coverage = lock_term(details, x, x_prev, flow_eff, reset, frames, scored)
    other = cfg.w_lock * lock
    l1 = pp = None
    if kind == "gt":
        per_item = torch.stack([(outs[t] - target[:, t]).abs().flatten(1).mean(1) for t in frames], dim=1)
        l1 = masked_mean(per_item, scored[:, frames])
        other = other + cfg.w_l1 * l1
    if pingpong:
        forward, reverse = pingpong_pairs(scored.any(0))
        both = scored[:, forward] & scored[:, reverse]  # the item scores the frame and its twin
        if forward and bool(both.any()):
            per_item = torch.stack([(outs[k] - outs[r]).abs().flatten(1).mean(1) for k, r in zip(forward, reverse)],
                                   dim=1)  # ping_pong's mean |out_k - out_r|, per item
            pp = masked_mean(per_item, both)
            other = other + cfg.w_pp * pp

    # stage 2: perceptual loss on a detached copy of each chosen output; its backward gives dL/d(output) and frees the
    # VGG graph before the next frame
    perc = None
    grads = {}
    if cfg.w_perc > 0:
        perc_sum = 0.0
        for t, picks in perceptual_plan(scored, cfg.perc_frames, rng).items():
            items = [i for i, _ in picks]
            leaf = outs[t].detach().requires_grad_(True)
            if len(items) == leaf.shape[0]:
                pred, tgt = leaf, target[:, t]
            else:
                pred, tgt = leaf[items], target[items, t]  # the other items get no gradient from this frame
            weight = torch.tensor([w for _, w in picks], dtype=leaf.dtype, device=leaf.device)
            value = (perceptual.per_item(pred, tgt) * weight).sum()
            (cfg.w_perc * value).backward()
            grads[t] = leaf.grad
            perc_sum = perc_sum + value.detach()
            MEMORY.sample()
        perc = perc_sum

    # stage 3: one backward through the DetailNet graph with those gradients, plus the cheap terms
    roots = [other] + [outs[t] for t in grads]
    torch.autograd.backward(roots, [None] + [grads[t] for t in grads])
    MEMORY.sample()

    terms = {"perc": perc, "l1": l1, "lock": lock, "pp": pp}
    values = {name: (None if v is None else float(v)) for name, v in terms.items()}
    weights = {"perc": cfg.w_perc, "l1": cfg.w_l1, "lock": cfg.w_lock, "pp": cfg.w_pp}
    values["loss"] = sum(weights[name] * v for name, v in values.items() if v is not None)
    values["mask"] = float(coverage)
    return values


def cosine_lr(cfg, it):
    """Learning rate of the step that starts at iteration `it`: cfg.lr, cosine-decayed to 0 over cfg.iters steps."""
    return 0.5 * cfg.lr * (1.0 + math.cos(math.pi * min(it, cfg.iters) / max(cfg.iters, 1)))


def set_lr(opt, lr):
    for group in opt.param_groups:
        group["lr"] = lr


def train_step(model, perceptual, opt, batch, cfg, pingpong=False, rng=None):
    """One optimizer step on `batch` at the optimizer's current lr; -> the metrics of `compute_gradients` plus
    grad_norm. A non-finite loss or gradient raises FloatingPointError before any weight is touched."""
    opt.zero_grad(set_to_none=True)
    metrics = compute_gradients(model, perceptual, batch, cfg, pingpong, rng)
    if not math.isfinite(metrics["loss"]):
        raise FloatingPointError(f"non-finite loss: {metrics}")
    max_norm = cfg.grad_clip if cfg.grad_clip > 0 else float("inf")
    norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm))
    if not math.isfinite(norm):
        raise FloatingPointError(f"non-finite gradient norm with loss terms {metrics}")
    opt.step()
    metrics["grad_norm"] = norm
    return metrics


# ---------------------------------------------------------------- data: schedule, epoch, loaders

def epoch_schedule(kinds, steps, pp_prob, rng):
    """[(kind, ping_pong)] for `steps` rounds over `kinds`: the kinds alternate, each batch is a ping-pong batch with
    probability `pp_prob`."""
    return [(kind, bool(rng.random() < pp_prob)) for _ in range(steps) for kind in kinds]


class Passes(Dataset):
    """`count` items of one kind, taken from position `offset` of a stream of passes over `dataset`. Every pass is a
    permutation of all the dataset's sequences drawn from (`stream_key`, pass number), so each sequence is served once
    per pass, in an order that differs between passes and between epochs, and the tail of a sorted sequence list is not
    starved. The plain and the ping-pong loaders of a kind take consecutive slices (different `offset`) of one stream.
    Every pass also shifts the dataset's seed, so a dataset shorter than the epoch needs gives other crops of its
    sequences instead of the same items again."""

    def __init__(self, dataset, count, stream_key, offset=0):
        self.dataset, self.count, self.offset = dataset, int(count), int(offset)
        self.stream_key, self.base_seed = tuple(stream_key), dataset.seed
        self._permutation = None  # (pass number, order) of the pass served last

    def __len__(self):
        return self.count

    def _order(self, number):
        if self._permutation is None or self._permutation[0] != number:
            rng = np.random.default_rng([*self.stream_key, number])
            self._permutation = (number, rng.permutation(len(self.dataset)))
        return self._permutation[1]

    def __getitem__(self, index):
        if not 0 <= index < self.count:
            raise IndexError(index)
        number, slot = divmod(self.offset + int(index), len(self.dataset))
        self.dataset.seed = self.base_seed + PASS_SEED_STRIDE * number
        return self.dataset[int(self._order(number)[slot])]


class Epoch(NamedTuple):
    schedule: list  # [(kind, ping_pong)], one entry per step
    loaders: dict  # (kind, ping_pong) -> DataLoader yielding exactly the batches the schedule asks of it
    sizes: dict  # kind -> number of finished sequences


def epoch_seed(cfg, epoch):
    return cfg.seed + EPOCH_SEED_STRIDE * epoch


def _sequence_dataset(cfg, kind, ping_pong, seed):
    return SequenceDataset(cfg.data, kind, crop=tuple(cfg.crop), ping_pong=ping_pong, reset_prob=cfg.reset_prob,
                           seed=seed)


def _steps_per_kind(sizes, batch):
    """Batches of one kind per epoch: one pass over the larger dataset, rounded up to whole batches."""
    return max(1, -(-max(sizes.values()) // batch))


def build_epoch(cfg, epoch, max_steps):
    """Rebuild the datasets (seeded for this epoch) and the loaders of epoch number `epoch`; the schedule is cut to
    `max_steps`. A kind with no finished sequence is left out; no data at all raises NoDataError."""
    seed = epoch_seed(cfg, epoch)
    plain = {kind: _sequence_dataset(cfg, kind, False, seed) for kind in KINDS}
    sizes = {kind: len(ds) for kind, ds in plain.items()}
    kinds = tuple(kind for kind in KINDS if sizes[kind])
    if not kinds:
        raise NoDataError(f"no finished sequences in {cfg.data}")
    rng = np.random.default_rng([cfg.seed, epoch])
    schedule = epoch_schedule(kinds, _steps_per_kind({k: sizes[k] for k in kinds}, cfg.batch), cfg.pp_prob,
                              rng)[:max_steps]
    counts = Counter(schedule)
    loaders = {}
    for (kind, pp), steps in sorted(counts.items()):
        dataset = _sequence_dataset(cfg, kind, True, seed) if pp else plain[kind]
        offset = counts[(kind, False)] * cfg.batch if pp else 0  # the ping-pong slice follows the plain one
        extra = {"prefetch_factor": 1} if cfg.num_workers > 0 else {}
        loaders[(kind, pp)] = DataLoader(
            Passes(dataset, steps * cfg.batch, (seed, KINDS.index(kind)), offset), batch_size=cfg.batch,
            shuffle=False, drop_last=True, num_workers=cfg.num_workers, persistent_workers=False, **extra)
    return Epoch(schedule, loaders, sizes)


def to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


# ---------------------------------------------------------------- checkpoints

def _to_cpu(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_cpu(v) for v in obj)
    return obj


def save_checkpoint(path, model, opt, it, cfg):
    """{"model", "opt", "iter", "cfg"} to `path`, written to a temporary file first so an interruption never leaves a
    half-written checkpoint."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    torch.save({"model": _to_cpu(model.state_dict()), "opt": _to_cpu(opt.state_dict()), "iter": int(it),
                "cfg": asdict(cfg)}, tmp)
    os.replace(tmp, path)


def save_run_checkpoint(out, model, opt, it, cfg, numbered=True):
    """`ckpt_{it:06d}.pt` (when `numbered`) and `latest.pt` in `out`."""
    out = Path(out)
    latest = out / "latest.pt"
    if not numbered:
        save_checkpoint(latest, model, opt, it, cfg)
        return latest
    path = out / f"ckpt_{it:06d}.pt"
    save_checkpoint(path, model, opt, it, cfg)
    tmp = latest.with_name(latest.name + ".tmp")
    shutil.copyfile(path, tmp)
    os.replace(tmp, latest)
    return path


def read_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def build_model(cfg):
    """A DetailNet of the architecture in `cfg`, a TrainConfig or a saved cfg dict."""
    get = cfg.get if isinstance(cfg, dict) else lambda k, d=None: getattr(cfg, k, d)
    return DetailNet(channels=get("channels", 64), blocks=get("blocks", 12), hidden=get("hidden", 32))


def load_checkpoint(path):
    """(model in eval mode on the CPU, the saved cfg dict); the model is rebuilt from the cfg's channels, blocks and
    hidden."""
    state = read_checkpoint(path)
    model = build_model(state["cfg"])
    model.load_state_dict(state["model"])
    return model.eval(), state["cfg"]


def _check_architecture(saved, cfg, action="resume"):
    mismatch = [f"{k} is {saved.get(k, 32 if k == 'hidden' else None)} in the checkpoint but {getattr(cfg, k)} now"
                for k in ARCHITECTURE if saved.get(k, 32 if k == "hidden" else None) != getattr(cfg, k)]
    if mismatch:
        raise ValueError(f"cannot {action}: " + "; ".join(mismatch))


# ---------------------------------------------------------------- the loop

def _round(v):
    return None if v is None else float(f"{v:.6g}")


def _log_row(it, epoch, window, elapsed, opt):
    mean = lambda name: float(np.mean(window[name])) if window[name] else None
    return {"iter": it, "epoch": epoch, "loss": _round(mean("loss")), "perc": _round(mean("perc")),
            "l1": _round(mean("l1")), "lock": _round(mean("lock")), "pp": _round(mean("pp")),
            "mask": _round(mean("mask")), "grad_norm": _round(mean("grad_norm")),
            "gt_steps": len(window["l1"]), "pp_steps": len(window["pp"]),
            "lr": _round(opt.param_groups[0]["lr"]),
            "it_per_s": _round(len(window["loss"]) / max(elapsed, 1e-9)),
            "mem_gb": _round(MEMORY.driver_gb), "live_gb": _round(MEMORY.live_gb)}


def train(cfg, force=False):
    """Train until `cfg.iters` steps are done (counting those of a resumed run); -> the last checkpoint's path.
    Raises SwapLimitExceeded, BatchStarted (the user's batch runs, looked for every BATCH_CHECK_EVERY steps unless
    `force`), KeyboardInterrupt, FloatingPointError (10 non-finite steps in a row), ... after saving latest.pt (unless
    nothing was done yet). Refuses, before writing anything: --resume without latest.pt, a new run over an existing
    latest.pt, and --init-from with --resume or with a checkpoint that is missing or of another architecture."""
    out = Path(cfg.out)
    latest = out / "latest.pt"
    if cfg.resume and cfg.init_from:
        raise ValueError("--init-from starts a new run from a checkpoint's weights and cannot be combined with --resume "
                         "(which continues the run in --out)")
    if latest.exists() and not cfg.resume:
        raise FileExistsError(f"{latest} exists: pass --resume to continue that run, or choose another --out")
    if cfg.resume and not latest.exists():
        raise FileNotFoundError(f"{latest} does not exist, so there is nothing to --resume: check --out, or drop "
                                "--resume to start a new run")
    init = None
    if cfg.init_from:
        if not Path(cfg.init_from).is_file():
            raise FileNotFoundError(f"--init-from {cfg.init_from}: no such checkpoint")
        init = read_checkpoint(cfg.init_from)
        _check_architecture(init["cfg"], cfg, f"initialise from {cfg.init_from}")
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg.device)
    torch.manual_seed(cfg.seed)
    model = build_model(cfg).to(device)
    if init is not None:
        model.load_state_dict(init["model"])
        print(f"starting from the weights of {cfg.init_from} (its iteration {init.get('iter')}): fresh optimizer, "
              f"iteration 0 of {cfg.iters}", flush=True)
        del init
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    it = 0
    resumed = False
    if cfg.resume:
        state = read_checkpoint(latest)
        _check_architecture(state["cfg"], cfg)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        it = int(state["iter"])
        resumed = True
        print(f"resuming from {latest} at iteration {it}", flush=True)

    # epoch numbers continue where the run stopped, so a resumed run does not replay the first epochs' crops
    epoch = 0
    if it:
        probe = {kind: len(_sequence_dataset(cfg, kind, False, 0)) for kind in KINDS}
        kinds = [k for k in KINDS if probe[k]]
        if kinds:
            epoch = -(-it // (len(kinds) * _steps_per_kind({k: probe[k] for k in kinds}, cfg.batch)))

    perceptual = Perceptual().to(device)
    guard = SwapGuard(cfg.swap_limit_gb)
    batch_guard = BatchGuard(force)
    rng = np.random.default_rng([cfg.seed, 7, it])  # which frames get the perceptual term
    MEMORY.start(device)
    window = defaultdict(list)
    window_start = time.perf_counter()
    saved_at = None
    skipped = 0  # non-finite steps in a row
    log_path = out / "train.jsonl"

    try:
        while it < cfg.iters:
            current = build_epoch(cfg, epoch, cfg.iters - it)
            print(f"epoch {epoch}: {current.sizes['teacher']} teacher and {current.sizes['gt']} gt sequences, "
                  f"{len(current.schedule)} steps", flush=True)
            iterators = {}
            for key in current.schedule:
                if key not in iterators:
                    iterators[key] = iter(current.loaders[key])
                batch = to_device(next(iterators[key]), device)
                set_lr(opt, cosine_lr(cfg, it))
                try:
                    metrics = train_step(model, perceptual, opt, batch, cfg, pingpong=key[1], rng=rng)
                except FloatingPointError as err:
                    skipped += 1
                    row = {"iter": it, "epoch": epoch, "skipped": skipped, "error": str(err)[:300]}
                    with open(log_path, "a") as f:
                        f.write(json.dumps(row) + "\n")
                    print(json.dumps(row), flush=True)
                    if skipped >= MAX_CONSECUTIVE_SKIPS:
                        raise FloatingPointError(f"{skipped} non-finite steps in a row, the last: {err}") from err
                    continue
                skipped = 0
                it += 1
                for name, value in metrics.items():
                    if value is not None:
                        window[name].append(value)
                if it % cfg.log_every == 0 or it == cfg.iters:
                    row = _log_row(it, epoch, window, time.perf_counter() - window_start, opt)
                    with open(log_path, "a") as f:
                        f.write(json.dumps(row) + "\n")
                    print(json.dumps(row), flush=True)
                    window.clear()
                    window_start = time.perf_counter()
                if it % cfg.ckpt_every == 0 or it == cfg.iters:
                    save_run_checkpoint(out, model, opt, it, cfg)
                    saved_at = it
                guard.check()
                if it % BATCH_CHECK_EVERY == 0 and it < cfg.iters:  # a finished run just ends
                    batch_guard.check()
            del iterators
            epoch += 1
    except BaseException:
        if it or resumed:  # a run that fails before its first step has nothing worth resuming from
            try:
                save_run_checkpoint(out, model, opt, it, cfg, numbered=False)
            except Exception as err:  # keep the original failure
                print(f"could not save {latest}: {err}", file=sys.stderr, flush=True)
        raise

    if saved_at != it or not (out / f"ckpt_{it:06d}.pt").exists():
        save_run_checkpoint(out, model, opt, it, cfg)
    return out / f"ckpt_{it:06d}.pt"


# ---------------------------------------------------------------- command line

HELP = {
    "data": "the build_data output directory (output/detail-data)",
    "out": "run directory for checkpoints and train.jsonl",
    "iters": "total optimizer steps (a resumed run counts the steps it already did)",
    "resume": "continue from latest.pt in --out (an error when there is none)",
    "init_from": "start a new run from this checkpoint's weights: fresh optimizer, iteration 0 (not with --resume)",
    "crop": "training crop in source px",
    "perc_frames": "perceptual frames per 12 loss frames of each item, drawn per item (12 or more: all)",
    "grad_checkpoint": "recompute each DetailNet frame in the backward to save memory",
    "grad_clip": "max gradient norm (0: no clipping)",
    "num_workers": "data loader workers per loader",
    "log_every": "write a train.jsonl line every N steps",
    "swap_limit_gb": "stop (exit 3) when swap grew by more than this many GB",
}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--force", action="store_true", help="run even while the batch script is running")
    for f in fields(TrainConfig):
        flag = "--" + f.name.replace("_", "-")
        help_text = HELP.get(f.name)
        if f.default is MISSING:
            parser.add_argument(flag, required=True, help=help_text)
        elif f.name == "crop":
            parser.add_argument(flag, type=int, nargs=2, metavar=("W", "H"), default=list(f.default), help=help_text)
        elif f.type is bool:
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=f.default, help=help_text)
        else:
            parser.add_argument(flag, type=f.type, default=f.default, help=help_text)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.resume and args.init_from:
        parser.error("--init-from starts a new run from a checkpoint's weights; it cannot be combined with --resume")
    refuse_if_batch_running(args.force)
    cfg = TrainConfig(**{f.name: getattr(args, f.name) for f in fields(TrainConfig)})
    cfg.crop = tuple(cfg.crop)
    try:
        last = train(cfg, force=args.force)
    except SwapLimitExceeded as err:
        print(f"stopping: {err}. latest.pt is saved; rerun with --resume", flush=True)
        return SWAP_EXIT_CODE
    except BatchStarted as err:
        print(f"stopping: {err}. latest.pt is saved; rerun with --resume once the batch is done (or pass --force)",
              flush=True)
        return BATCH_EXIT_CODE
    except (NoDataError, FileExistsError, FileNotFoundError, FloatingPointError) as err:
        print(f"error: {err}", flush=True)
        return 1
    print(f"done: {last}", flush=True)
    return 0


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))  # unwinds through train(), which saves latest.pt
    sys.exit(main())
