"""Go / no-go gates for a detail-layer model, computed from the per-clip metrics JSON of the in-scope clips.

    python bench/gates.py MODEL [--reference apple-base] [--summary output/sr-bench/summary.json] [--clips output/sr-bench/clips]

Compares MODEL's `metrics/MODEL.json` with the reference's on the same clips and prints one PASS / FAIL line per gate with
the numbers. Exit status 0 only when every gate passes. These gates decide whether the project continues, so they use only
the clips M1 is judged on and never summary.json, which pools in the e-* (1080p) clips that wait for M3. `--summary` is
accepted for the documented command line and ignored.

The reference is `apple-base`: Apple AI's output before the bench's HEVC encode, which is exactly what DetailNet adds to. It
is made by `python -m detailnet.run_bench --base-only --name apple-base CLIP_DIR...` (the strength-0 composite of the dumps) and
scored like any model. The bench's own `apple` output has been through HEVC and an ffmpeg colour conversion (about 2 levels
darker, slightly lower LPIPS), so gating against it would hand DetailNet a handicap before it adds anything. The cut-leak gate
is the one exception, see below. After the gated table the same numbers against `apple` are printed as information only.

Real-clip gates, over the c-* clips (the 720p -> 4K clips), as ratios MODEL / reference per clip, then averaged:

    detail          mean ratio of `detail` (gradient energy vs the bicubic) >= 1.5
    jitter_all      mean ratio of `jitter_all` <= 1.1
    jitter_static, jitter_slow, jitter_fast
                    mean ratio of `jitter[bucket]`, over the clips that have that bucket in the reference, <= 1.1 each
    pulse           the mean of MODEL's per-clip `jitter_phase` vectors (4 phases: the frame's index within its shot,
                    mod 4); its max <= 1.1 x its mean

The jitter gates also check that MODEL was scored on the same pairs as the reference. bench/metrics.py averages a model's
jitter only over the frame pairs its own optical flow tracks well (mask over 20 % of the frame), so a model that shimmers
could be scored on its most stable pairs alone. Per clip and bucket, MODEL's `flicker_pairs[bucket]` must be at least 95 %
of the reference's: a bucket gate checks its bucket, jitter_all every bucket. A value the reference has and MODEL has not
(null: MODEL kept no pair of that bucket in that clip) is a FAIL too, since each bucket rests on two clips; a value only
MODEL has is left out of the ratio (there is nothing to compare it with). A failing check FAILs the gate with the clip and
the counts in the note; the ratio over the comparable clips is still shown. The pulse gate reads MODEL alone; a model
whose pairs shrink fails jitter_all, so the overall verdict covers it.

A ratio needs a positive reference value: a reference value at or below 0 FAILs that gate, naming the clip and the value.

Other gates:

    cut_leak        MODEL's `cut_leak` on c-cut-nest <= the bench `apple` output's. MODEL's own is the difference between its
                    cut-aware run and its cut-ignoring run (out/MODEL-nocut) over the same base (run_bench --ignore-cuts). It is
                    compared with `apple`, not the reference: apple-base carries no state across a cut, so its leak is 0 by
                    construction (a base-only --ignore-cuts run writes identical frames, which scores 0)
    lpips           mean `lpips` over the g3-* clips (360p -> 1080p, vs the real 1080p) <= the reference's mean

Anything the gates need that is missing (a metrics file, a key, a clip) is a FAIL with a message saying what, never a crash and
never a silent pass. Ratios are compared with a 1e-9 tolerance so a value exactly at its threshold passes.

After the table, information only (never gated): the raw flicker ratio, `flicker_all` (the jitter numerator in luma levels,
without the division by the gradient) MODEL / reference averaged over the c-* clips, and then the same numbers against the
bench `apple` output.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import NamedTuple, Optional

import numpy as np

CLIPS = Path(__file__).resolve().parent.parent / "output" / "sr-bench" / "clips"
REFERENCE = "apple-base"  # Apple AI before the bench's HEVC encode: what DetailNet adds to
BENCH_APPLE = "apple"  # Apple AI as the bench encodes it: the cut-leak threshold, and the information block
HINTS = {REFERENCE: "generate it with `python -m detailnet.run_bench --base-only --name apple-base "
                    "output/sr-bench/clips/c-* output/sr-bench/clips/g3-*` and score it with bench/score_all.sh"}
REAL_PREFIX = "c-"  # the real clips M1 is judged on (e-* are 1080p: M3)
GT3_PREFIX = "g3-"  # the x3 ground-truth clips
CUT_CLIP = "c-cut-nest"
BUCKETS = ("static", "slow", "fast")
DETAIL_MIN = 1.5
JITTER_MAX = 1.1
PULSE_MAX = 1.1
PAIRS_MIN = 0.95  # MODEL's scored pairs per clip and bucket, as a share of the reference's
TOLERANCE = 1e-9


class Gate(NamedTuple):
    name: str
    value: Optional[float]  # None when the gate could not be computed
    op: str  # ">=" or "<="
    threshold: Optional[float]
    passed: bool
    note: str


class Missing(Exception):
    """A gate cannot be evaluated: something it needs is not there, or a value makes the comparison meaningless; the
    message says what."""


def scope(root, prefix):
    """Sorted names of the clip directories of one family."""
    root = Path(root)
    return sorted(p.name for p in root.iterdir() if p.is_dir() and p.name.startswith(prefix)) if root.is_dir() else []


def load(root, clip, model):
    path = Path(root) / clip / "metrics" / f"{model}.json"
    if not path.exists():
        hint = f" ({HINTS[model]})" if model in HINTS else ""
        raise Missing(f"missing metrics file {path}{hint}")
    try:
        return json.loads(path.read_text())
    except ValueError as error:
        raise Missing(f"unreadable metrics file {path}: {error}")


def field(metrics, *keys):
    """metrics[k1][k2]...; None if a key is absent or the value null."""
    for key in keys:
        if not isinstance(metrics, dict) or key not in metrics:
            return None
        metrics = metrics[key]
    return metrics


def clips_of(root, prefix):
    clips = scope(root, prefix)
    if not clips:
        raise Missing(f"no {prefix}* clip directory under {root}")
    return clips


def mean_ratio(root, model, reference, prefix, *keys, pairs=()):
    """(mean over the clips of model / reference, number of clips used, problems) for metrics[keys]. A clip whose value
    is null in the reference is skipped. `problems` lists what makes the gate fail although a mean could be computed: a
    value null in the model only, a reference value <= 0 (that clip is left out of the mean), and, for every bucket in
    `pairs`, the model scored on fewer than PAIRS_MIN of the reference's pairs. A missing file, a key absent from a
    file, or no clip to take a mean over raises Missing."""
    name = ".".join(keys)
    ratios, problems = [], []
    for clip in clips_of(root, prefix):
        a, b = load(root, clip, model), load(root, clip, reference)
        for who, m in ((model, a), (reference, b)):
            if not _has(m, keys):
                raise Missing(f"{clip}/metrics/{who}.json has no '{name}' (re-score the clip with the current bench/metrics.py)")
        problems += pair_shortfall(clip, model, a, reference, b, pairs)
        x, y = field(a, *keys), field(b, *keys)
        if y is None:
            continue  # nothing to compare with (null in both, or the model has pairs the reference has not)
        if x is None:
            problems.append(f"{clip}: {name} is null for {model} but {y} for {reference} (the model kept no scored pair "
                            "there, so the clip would drop out of the gate)")
            continue
        if y <= 0:
            problems.append(f"{clip}: {reference} {name} is {y}; a ratio needs a positive reference value")
            continue
        ratios.append(x / y)
    if not ratios:
        detail = f": {'; '.join(problems)}" if problems else ""
        raise Missing(f"no {prefix}* clip has a value of '{name}' in both {model} and {reference} to compare{detail}")
    return float(np.mean(ratios)), len(ratios), problems


def pair_shortfall(clip, model, a, reference, b, buckets):
    """Problems (strings) where `model`'s metrics `a` scored fewer than PAIRS_MIN of the pairs `reference`'s `b` did, per
    bucket in `buckets`. Missing when a file has no flicker_pairs for them."""
    if not buckets:
        return []
    counts = {}
    for who, m in ((model, a), (reference, b)):
        pairs = field(m, "flicker_pairs")
        if not isinstance(pairs, dict) or any(not isinstance(pairs.get(k), (int, float)) for k in buckets):
            raise Missing(f"{clip}/metrics/{who}.json has no 'flicker_pairs' for {', '.join(buckets)} (re-score the clip "
                          "with the current bench/metrics.py)")
        counts[who] = pairs
    return [f"{clip}: {model} was scored on {counts[model][k]} {k} pairs, {reference} on {counts[reference][k]} "
            f"(below {PAIRS_MIN:.0%}: its own flow mask dropped the rest, so it would be judged on a subset)"
            for k in buckets if counts[model][k] < PAIRS_MIN * counts[reference][k] - TOLERANCE]


def _has(metrics, keys):
    for key in keys:
        if not isinstance(metrics, dict) or key not in metrics:
            return False
        metrics = metrics[key]
    return True


def averaged_phase(root, model, prefix):
    """The mean over clips of the per-clip jitter_phase vectors, per phase over the clips that have it."""
    vectors = []
    for clip in clips_of(root, prefix):
        m = load(root, clip, model)
        if not _has(m, ("jitter_phase",)):
            raise Missing(f"{clip}/metrics/{model}.json has no 'jitter_phase' (re-score the clip with the pulse metric)")
        if not isinstance(m["jitter_phase"], list) or len(m["jitter_phase"]) != 4:
            raise Missing(f"{clip}/metrics/{model}.json: jitter_phase should be a list of 4")
        vectors.append(m["jitter_phase"])
    averaged = []
    for k in range(4):
        values = [v[k] for v in vectors if v[k] is not None]
        averaged.append(float(np.mean(values)) if values else None)
    present = [v for v in averaged if v is not None]
    if not present:
        raise Missing(f"no {prefix}* clip has any jitter_phase pair")
    if np.mean(present) <= 0:
        raise Missing(f"the mean of {model}'s averaged jitter_phase is {float(np.mean(present))}; max / mean needs a "
                      "positive mean")
    return averaged, len(vectors)


def judge(name, value, op, threshold, note=""):
    ok = value >= threshold - TOLERANCE if op == ">=" else value <= threshold + TOLERANCE
    return Gate(name, value, op, threshold, bool(ok), note)


def failed(name, op, threshold, message):
    return Gate(name, None, op, threshold, False, message)


def check_gates(model, clips_root=CLIPS, reference=REFERENCE, leak_reference=BENCH_APPLE):
    """The list of Gates for `model`, in the order they are printed. Every gate but cut_leak compares with `reference`;
    cut_leak compares with `leak_reference`."""
    root = Path(clips_root)
    results = []

    def ratio_gate(name, op, threshold, *keys, pairs=()):
        try:
            value, n, problems = mean_ratio(root, model, reference, REAL_PREFIX, *keys, pairs=pairs)
        except Missing as error:
            results.append(failed(name, op, threshold, str(error)))
            return
        note = f"mean over {n} {REAL_PREFIX}* clips of {model} / {reference}"
        if problems:
            results.append(Gate(name, value, op, threshold, False, f"{note}; FAILED: {'; '.join(problems)}"))
        else:
            results.append(judge(name, value, op, threshold, note))

    ratio_gate("detail", ">=", DETAIL_MIN, "detail")
    ratio_gate("jitter_all", "<=", JITTER_MAX, "jitter_all", pairs=BUCKETS)
    for bucket in BUCKETS:
        ratio_gate(f"jitter_{bucket}", "<=", JITTER_MAX, "jitter", bucket, pairs=(bucket,))

    try:
        phase, n = averaged_phase(root, model, REAL_PREFIX)
    except Missing as error:
        results.append(failed("pulse", "<=", PULSE_MAX, str(error)))
    else:
        present = [v for v in phase if v is not None]
        value = max(present) / float(np.mean(present))
        shown = ", ".join("-" if v is None else f"{v:.4f}" for v in phase)
        results.append(judge("pulse", value, "<=", PULSE_MAX,
                             f"max / mean of the {model} jitter_phase averaged over {n} {REAL_PREFIX}* clips: [{shown}]"))

    try:
        mine, theirs = load(root, CUT_CLIP, model), load(root, CUT_CLIP, leak_reference)
        for who, m in ((model, mine), (leak_reference, theirs)):
            if field(m, "cut_leak") is None:
                hint = (" (run run_bench --ignore-cuts for it and re-score)" if who == model else "")
                raise Missing(f"{CUT_CLIP}/metrics/{who}.json has no 'cut_leak'{hint}")
    except Missing as error:
        results.append(failed("cut_leak", "<=", None, str(error)))
    else:
        results.append(judge("cut_leak", float(mine["cut_leak"]), "<=", float(theirs["cut_leak"]),
                             f"{model} on {CUT_CLIP} vs {leak_reference} (threshold = {leak_reference}'s)"))

    try:
        values = {}
        for who in (model, reference):
            per_clip = []
            for clip in clips_of(root, GT3_PREFIX):
                m = load(root, clip, who)
                if field(m, "lpips") is None:
                    raise Missing(f"{clip}/metrics/{who}.json has no 'lpips'")
                per_clip.append(m["lpips"])
            values[who] = (float(np.mean(per_clip)), len(per_clip))
    except Missing as error:
        results.append(failed("lpips", "<=", None, str(error)))
    else:
        results.append(judge("lpips", values[model][0], "<=", values[reference][0],
                             f"mean over {values[model][1]} {GT3_PREFIX}* clips vs {reference} (threshold = {reference}'s)"))
    return results


def format_table(results):
    lines = [f"{'gate':14s} {'result':6s} {'value':>9s}   {'limit':>11s}  note"]
    for g in results:
        value = "n/a" if g.value is None else f"{g.value:.4f}"
        limit = "" if g.threshold is None else f"{g.op} {g.threshold:.4f}"
        lines.append(f"{g.name:14s} {'PASS' if g.passed else 'FAIL':6s} {value:>9s}   {limit:>11s}  {g.note}")
    lines.append("")
    lines.append(f"OVERALL: {'PASS' if all(g.passed for g in results) else 'FAIL'} "
                 f"({sum(g.passed for g in results)}/{len(results)} gates)")
    return "\n".join(lines)


def raw_flicker(model, clips_root, reference):
    """Information row (a Gate whose verdict is never used): the mean over the c-* clips of MODEL / reference
    `flicker_all`, the jitter numerator in luma levels without the division by the gradient."""
    try:
        value, n, problems = mean_ratio(Path(clips_root), model, reference, REAL_PREFIX, "flicker_all")
    except Missing as error:
        return Gate("flicker_all", None, "", None, False, f"raw flicker, not gated: {error}")
    note = f"raw flicker, not gated: mean over {n} {REAL_PREFIX}* clips of {model} / {reference} (luma levels)"
    return Gate("flicker_all", value, "", None, False, note + (f"; {'; '.join(problems)}" if problems else ""))


def format_info(results, title):
    """Rows with no verdict under `title`. A number that cannot be computed shows n/a and why."""
    lines = [title, f"{'gate':14s} {'value':>9s}   {'limit':>11s}  note"]
    for g in results:
        value = "n/a" if g.value is None else f"{g.value:.4f}"
        limit = "" if g.threshold is None else f"{g.op} {g.threshold:.4f}"
        lines.append(f"{g.name:14s} {value:>9s}   {limit:>11s}  {g.note.replace('FAILED: ', '')}")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Go / no-go gates for a detail-layer model (see the module docstring).")
    ap.add_argument("model")
    ap.add_argument("--reference", default=REFERENCE,
                    help=f"what the gates compare with (default {REFERENCE}: Apple AI before the HEVC encode)")
    ap.add_argument("--summary", help="accepted for compatibility and ignored: the gates read the per-clip metrics")
    ap.add_argument("--clips", type=Path, default=CLIPS, help="the clips directory (default: output/sr-bench/clips)")
    args = ap.parse_args(argv)
    results = check_gates(args.model, args.clips, args.reference)
    print(f"model {args.model} vs {args.reference} (cut leak vs {BENCH_APPLE}); clips under {args.clips}")
    print(format_table(results))
    print()
    print(format_info([raw_flicker(args.model, args.clips, args.reference)],
                      f"Information only (not gated): raw flicker against `{args.reference}`"))
    if args.reference != BENCH_APPLE:
        print()
        print(format_info(check_gates(args.model, args.clips, BENCH_APPLE)
                          + [raw_flicker(args.model, args.clips, BENCH_APPLE)],
                          f"Information only (not gated): the same numbers against the bench output `{BENCH_APPLE}` "
                          "(after the HEVC encode)"))
    return 0 if all(g.passed for g in results) else 1


if __name__ == "__main__":
    sys.exit(main())
