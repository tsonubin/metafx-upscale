"""Go / no-go gates for a detail-layer model, computed from the per-clip metrics JSON of the in-scope clips.

    python bench/gates.py MODEL [--summary output/sr-bench/summary.json] [--clips output/sr-bench/clips]

Compares MODEL's `metrics/MODEL.json` with Apple AI's (`metrics/apple.json`) on the same clips and prints one PASS / FAIL line
per gate with the numbers. Exit status 0 only when every gate passes. These gates decide whether the project continues, so
they use only the clips M1 is judged on and never summary.json, which pools in the e-* (1080p) clips that wait for M3.
`--summary` is accepted for the documented command line and ignored.

Real-clip gates, over the c-* clips (the 720p -> 4K clips), as ratios MODEL / apple per clip, then averaged:

    detail          mean ratio of `detail` (gradient energy vs the bicubic) >= 1.5
    jitter_all      mean ratio of `jitter_all` <= 1.1
    jitter_static, jitter_slow, jitter_fast
                    mean ratio of `jitter[bucket]`, over the clips that have that bucket in both models, <= 1.1 each
    pulse           the mean of MODEL's per-clip `jitter_phase` vectors (4 phases, t mod 4); its max <= 1.1 x its mean

Other gates:

    cut_leak        MODEL's `cut_leak` on c-cut-nest <= apple's. MODEL's own is the difference between its cut-aware run and its
                    cut-ignoring run (out/MODEL-nocut) over the same base (run_bench --ignore-cuts)
    lpips           mean `lpips` over the g3-* clips (360p -> 1080p, vs the real 1080p) <= apple's mean

Anything the gates need that is missing (a metrics file, a key, a clip) is a FAIL with a message saying what, never a crash and
never a silent pass. Ratios are compared with a 1e-9 tolerance so a value exactly at its threshold passes.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import NamedTuple, Optional

import numpy as np

CLIPS = Path(__file__).resolve().parent.parent / "output" / "sr-bench" / "clips"
REFERENCE = "apple"
REAL_PREFIX = "c-"  # the real clips M1 is judged on (e-* are 1080p: M3)
GT3_PREFIX = "g3-"  # the x3 ground-truth clips
CUT_CLIP = "c-cut-nest"
BUCKETS = ("static", "slow", "fast")
DETAIL_MIN = 1.5
JITTER_MAX = 1.1
PULSE_MAX = 1.1
TOLERANCE = 1e-9


class Gate(NamedTuple):
    name: str
    value: Optional[float]  # None when the gate could not be computed
    op: str  # ">=" or "<="
    threshold: Optional[float]
    passed: bool
    note: str


class Missing(Exception):
    """Something a gate needs is not there; the message says what."""


def scope(root, prefix):
    """Sorted names of the clip directories of one family."""
    root = Path(root)
    return sorted(p.name for p in root.iterdir() if p.is_dir() and p.name.startswith(prefix)) if root.is_dir() else []


def load(root, clip, model):
    path = Path(root) / clip / "metrics" / f"{model}.json"
    if not path.exists():
        raise Missing(f"missing metrics file {path}")
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


def mean_ratio(root, model, reference, prefix, *keys):
    """(mean over the clips of model / reference, number of clips used) for metrics[keys]. A clip whose value is null in
    either model is skipped; a missing file, or a key absent from a file, is an error."""
    ratios = []
    for clip in clips_of(root, prefix):
        a, b = load(root, clip, model), load(root, clip, reference)
        for who, m in ((model, a), (reference, b)):
            if not _has(m, keys):
                raise Missing(f"{clip}/metrics/{who}.json has no '{'.'.join(keys)}' (re-score the clip with the current bench/metrics.py)")
        x, y = field(a, *keys), field(b, *keys)
        if x is None or y is None:
            continue
        if y <= 0:
            raise Missing(f"{clip}: {reference} {'.'.join(keys)} is {y}, a ratio is undefined")
        ratios.append(x / y)
    if not ratios:
        raise Missing(f"no {prefix}* clip has a value of '{'.'.join(keys)}' in both {model} and {reference}")
    return float(np.mean(ratios)), len(ratios)


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
    if not present or np.mean(present) <= 0:
        raise Missing(f"no {prefix}* clip has any jitter_phase pair")
    return averaged, len(vectors)


def judge(name, value, op, threshold, note=""):
    ok = value >= threshold - TOLERANCE if op == ">=" else value <= threshold + TOLERANCE
    return Gate(name, value, op, threshold, bool(ok), note)


def failed(name, op, threshold, message):
    return Gate(name, None, op, threshold, False, message)


def check_gates(model, clips_root=CLIPS, reference=REFERENCE):
    """The list of Gates for `model`, in the order they are printed."""
    root = Path(clips_root)
    results = []

    def ratio_gate(name, op, threshold, *keys):
        try:
            value, n = mean_ratio(root, model, reference, REAL_PREFIX, *keys)
        except Missing as error:
            results.append(failed(name, op, threshold, str(error)))
        else:
            results.append(judge(name, value, op, threshold, f"mean over {n} {REAL_PREFIX}* clips of {model} / {reference}"))

    ratio_gate("detail", ">=", DETAIL_MIN, "detail")
    ratio_gate("jitter_all", "<=", JITTER_MAX, "jitter_all")
    for bucket in BUCKETS:
        ratio_gate(f"jitter_{bucket}", "<=", JITTER_MAX, "jitter", bucket)

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
        mine, theirs = load(root, CUT_CLIP, model), load(root, CUT_CLIP, reference)
        for who, m in ((model, mine), (reference, theirs)):
            if field(m, "cut_leak") is None:
                hint = (" (run run_bench --ignore-cuts for it and re-score)" if who == model else "")
                raise Missing(f"{CUT_CLIP}/metrics/{who}.json has no 'cut_leak'{hint}")
    except Missing as error:
        results.append(failed("cut_leak", "<=", None, str(error)))
    else:
        results.append(judge("cut_leak", float(mine["cut_leak"]), "<=", float(theirs["cut_leak"]),
                             f"{model} on {CUT_CLIP} vs {reference} (threshold = {reference}'s)"))

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


def main(argv=None):
    ap = argparse.ArgumentParser(description="Go / no-go gates for a detail-layer model (see the module docstring).")
    ap.add_argument("model")
    ap.add_argument("--summary", help="accepted for compatibility and ignored: the gates read the per-clip metrics")
    ap.add_argument("--clips", type=Path, default=CLIPS, help="the clips directory (default: output/sr-bench/clips)")
    args = ap.parse_args(argv)
    results = check_gates(args.model, args.clips)
    print(f"model {args.model} vs {REFERENCE}; clips under {args.clips}")
    print(format_table(results))
    return 0 if all(g.passed for g in results) else 1


if __name__ == "__main__":
    sys.exit(main())
