"""bench/gates.py: the go/no-go check. Synthetic per-clip metrics under a temporary clips directory, never output/.

The gates compare against `apple-base` (Apple AI's pre-encode output, the strength-0 run of run_bench); only the cut-leak
gate uses the `apple` bench output, and `apple` is also printed as information."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bench"))
import gates  # noqa: E402
import summarize  # noqa: E402

C_CLIPS = ["c-static-foliage", "c-slow-gorge", "c-fast-waterfall", "c-water-climbers", "c-cut-nest"]
G3_CLIPS = ["g3-lizard-grass", "g3-waterfall", "g3-wallaby-fast", "g3-frog-rain", "g3-cut-gorge"]

REF = "apple-base"
APPLE = {"detail": 2.0, "jitter_all": 0.2, "jitter": {"static": 0.1, "slow": 0.2, "fast": 0.4},
         "jitter_phase": [0.2, 0.2, 0.2, 0.2]}


def put(root, clip, model, **metrics):
    d = Path(root) / clip / "metrics"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{model}.json").write_text(json.dumps({"clip": clip, "model": model, **metrics}))


def scaled(ratio_detail=1.5, ratio_jitter=1.1, ratio_bucket=1.1, phase=(1.1, 1.1, 0.9, 0.9)):
    """DetailNet metrics for a c-* clip as ratios of the apple ones."""
    return {"detail": APPLE["detail"] * ratio_detail, "jitter_all": APPLE["jitter_all"] * ratio_jitter,
            "jitter": {k: v * ratio_bucket for k, v in APPLE["jitter"].items()},
            "jitter_phase": [0.2 * p for p in phase]}


def build(root, model="dn", lpips=(0.30, 0.30), leak=(5.0, 5.0), ref_over=None, **kw):
    """Metrics of the reference (apple-base), the bench output (apple) and `model` for every in-scope clip. `lpips` is
    (model, reference) and `leak` (model, apple); apple-base has no cut_leak and apple's numbers equal the reference's, so
    only the cut-leak gate and the information block read them."""
    for clip in C_CLIPS:
        put(root, clip, REF, **{**APPLE, **(ref_over or {})})
        put(root, clip, "apple", **{**APPLE, **({"cut_leak": leak[1]} if clip == "c-cut-nest" else {})})
        put(root, clip, model, **{**scaled(**kw), **({"cut_leak": leak[0]} if clip == "c-cut-nest" else {})})
    for clip in G3_CLIPS:
        put(root, clip, REF, lpips=lpips[1])
        put(root, clip, "apple", lpips=lpips[1])
        put(root, clip, model, lpips=lpips[0])


def by_name(results):
    return {g.name: g for g in results}


def statuses(root, model="dn"):
    return {g.name: g.passed for g in gates.check_gates(model, root)}


GATES = ["detail", "jitter_all", "jitter_static", "jitter_slow", "jitter_fast", "pulse", "cut_leak", "lpips"]


def test_gates_logic_exactly_at_the_thresholds_passes(tmp_path):
    build(tmp_path)
    results = gates.check_gates("dn", tmp_path)
    assert [g.name for g in results] == GATES
    assert all(g.passed for g in results), gates.format_table(results)
    assert gates.main(["dn", "--clips", str(tmp_path)]) == 0


def test_float_noise_at_a_threshold_does_not_flip_the_verdict():
    assert gates.judge("jitter_all", 1.1 + 4e-16, "<=", 1.1).passed
    assert gates.judge("detail", 1.5 - 4e-16, ">=", 1.5).passed
    assert not gates.judge("jitter_all", 1.1 + 1e-6, "<=", 1.1).passed
    assert not gates.judge("detail", 1.5 - 1e-6, ">=", 1.5).passed
    assert not gates.judge("detail", 1.49, ">=", 1.5).passed


def test_detail_below_threshold_fails_only_detail(tmp_path):
    build(tmp_path, ratio_detail=1.49)
    ok = statuses(tmp_path)
    assert ok.pop("detail") is False
    assert all(ok.values())
    assert gates.main(["dn", "--clips", str(tmp_path)]) == 1


@pytest.mark.parametrize("kwargs,failing", [
    ({"ratio_jitter": 1.11}, "jitter_all"),
    ({"ratio_bucket": 1.11}, {"jitter_static", "jitter_slow", "jitter_fast"}),
    ({"phase": (1.0, 1.0, 1.0, 1.3)}, "pulse"),
    ({"leak": (5.01, 5.0)}, "cut_leak"),
    ({"lpips": (0.31, 0.30)}, "lpips"),
])
def test_each_gate_fails_on_its_own(tmp_path, kwargs, failing):
    build(tmp_path, **kwargs)
    failing = {failing} if isinstance(failing, str) else failing
    result = statuses(tmp_path)
    assert {name for name, ok in result.items() if not ok} == failing


def test_numbers_are_printed(tmp_path):
    build(tmp_path, ratio_detail=1.49)
    table = gates.format_table(gates.check_gates("dn", tmp_path))
    assert "FAIL" in table and "PASS" in table
    assert "1.49" in table and "1.5" in table
    for name in GATES:
        assert name in table


def test_ratios_are_means_over_clips_not_pooled(tmp_path):
    build(tmp_path)
    # detail ratios 2.0, 1.0, 1.5, 1.5, 1.5: the mean is 1.5 although one clip is below it
    for clip, r in zip(C_CLIPS[:2], (2.0, 1.0)):
        put(tmp_path, clip, "dn", **{**scaled(ratio_detail=r)})
    got = by_name(gates.check_gates("dn", tmp_path))["detail"]
    assert got.value == pytest.approx(1.5) and got.passed


def test_bucket_ratio_uses_only_clips_that_have_the_bucket(tmp_path):
    build(tmp_path)
    for clip in C_CLIPS[:4]:  # no static pairs in four clips on either side (null), the remaining one is at 1.1
        put(tmp_path, clip, REF, **{**APPLE, "jitter": {"static": None, "slow": 0.2, "fast": 0.4}})
        put(tmp_path, clip, "dn", **{**scaled(), "jitter": {"static": None, "slow": 0.22, "fast": 0.44}})
    put(tmp_path, "c-cut-nest", "dn", **{**scaled(), "cut_leak": 5.0, "jitter": {"static": 0.0099, "slow": None, "fast": None}})
    put(tmp_path, "c-cut-nest", REF, **{**APPLE, "jitter": {"static": 0.01, "slow": None, "fast": None}})
    result = by_name(gates.check_gates("dn", tmp_path))
    assert result["jitter_static"].value == pytest.approx(0.99) and result["jitter_static"].passed
    assert result["jitter_slow"].value == pytest.approx(1.1) and result["jitter_slow"].passed


def test_pulse_uses_the_mean_phase_vector_over_clips(tmp_path):
    build(tmp_path, phase=(1, 1, 1, 1))
    # every clip pulses on its own (a peak 1.2x its mean) but at a different phase: the average is flat
    for clip, hot in zip(C_CLIPS, (0, 1, 2, 3, 0)):
        vec = [0.2] * 4
        vec[hot] = 0.2 * 1.6
        put(tmp_path, clip, "dn", **{**scaled(), "jitter_phase": vec, **({"cut_leak": 5.0} if clip == "c-cut-nest" else {})})
    got = by_name(gates.check_gates("dn", tmp_path))["pulse"]
    # averaged vector: phase 0 hot in two clips of five
    avg = [0.2 + 0.12 * 2 / 5, 0.2 + 0.12 / 5, 0.2 + 0.12 / 5, 0.2 + 0.12 / 5]
    assert got.value == pytest.approx(max(avg) / (sum(avg) / 4))
    assert got.passed and got.value < 1.1  # each clip alone would fail: 0.32 / 0.23 = 1.39


def test_pulse_threshold_is_relative_to_the_models_own_mean(tmp_path):
    build(tmp_path, phase=(1.0, 1.0, 1.0, 1.0))
    assert statuses(tmp_path)["pulse"]
    build(tmp_path, phase=(1.0, 1.0, 1.0, 1.2))  # max / mean = 1.2 / 1.05 = 1.143
    assert not statuses(tmp_path)["pulse"]
    build(tmp_path, phase=(1.0, 1.0, 1.0, 1.1))  # 1.1 / 1.025 = 1.073
    assert statuses(tmp_path)["pulse"]


def test_scope_excludes_e_clips_and_other_clips(tmp_path):
    build(tmp_path)
    # a 1080p clip with terrible numbers, a gt clip with a terrible LPIPS, and a g (x4) clip: all out of scope
    put(tmp_path, "e-gorge-river", REF, **APPLE)
    put(tmp_path, "e-gorge-river", "dn", **{**scaled(ratio_detail=0.5, ratio_jitter=9, ratio_bucket=9)})
    put(tmp_path, "g-waterfall", REF, lpips=0.1)
    put(tmp_path, "g-waterfall", "dn", lpips=0.9)
    assert all(statuses(tmp_path).values())


def test_missing_metrics_is_a_fail_with_a_message_not_a_crash(tmp_path):
    build(tmp_path)
    (tmp_path / "c-slow-gorge" / "metrics" / "dn.json").unlink()
    (tmp_path / "g3-waterfall" / "metrics" / "apple-base.json").unlink()
    results = by_name(gates.check_gates("dn", tmp_path))
    for name in ("detail", "jitter_all", "jitter_static", "jitter_slow", "jitter_fast", "pulse"):
        assert not results[name].passed and "c-slow-gorge" in results[name].note and "dn.json" in results[name].note
    assert results["cut_leak"].passed  # c-cut-nest is intact
    assert not results["lpips"].passed and "g3-waterfall" in results["lpips"].note and "apple-base.json" in results["lpips"].note
    assert gates.main(["dn", "--clips", str(tmp_path)]) == 1


def test_no_clips_at_all_fails_every_gate(tmp_path):
    results = gates.check_gates("dn", tmp_path)
    assert [g.name for g in results] == GATES and not any(g.passed for g in results)


def test_cut_leak_missing_from_the_metrics_fails(tmp_path):
    build(tmp_path)
    put(tmp_path, "c-cut-nest", "dn", **scaled())  # scored before dn-nocut existed: no cut_leak key
    got = by_name(gates.check_gates("dn", tmp_path))["cut_leak"]
    assert not got.passed and "cut_leak" in got.note


def test_old_metrics_without_the_pulse_fields_fail_the_pulse_gate(tmp_path):
    build(tmp_path)
    stale = scaled()
    del stale["jitter_phase"]
    put(tmp_path, "c-slow-gorge", "dn", **stale)
    got = by_name(gates.check_gates("dn", tmp_path))["pulse"]
    assert not got.passed and "jitter_phase" in got.note and "c-slow-gorge" in got.note


def test_reference_against_itself_passes_every_ratio_gate_except_detail(tmp_path):
    for clip in C_CLIPS:
        put(tmp_path, clip, REF, **APPLE)  # apple-base has no cut_leak: its run carries no state
        put(tmp_path, clip, "apple", **APPLE, **({"cut_leak": 4.0} if clip == "c-cut-nest" else {}))
    put(tmp_path, "c-cut-nest", REF, **APPLE, cut_leak=0.0)  # run_bench --base-only --ignore-cuts: identical frames
    for clip in G3_CLIPS:
        put(tmp_path, clip, REF, lpips=0.3)
        put(tmp_path, clip, "apple", lpips=0.33)
    result = statuses(tmp_path, REF)
    assert not result.pop("detail")  # ratio 1.0 < 1.5, expected
    assert all(result.values()), result
    got = by_name(gates.check_gates(REF, tmp_path))
    assert got["lpips"].value == pytest.approx(0.3) and got["lpips"].threshold == pytest.approx(0.3)


def test_the_default_reference_is_apple_base(tmp_path):
    build(tmp_path, lpips=(0.31, 0.30))
    put(tmp_path, "g3-waterfall", "apple", lpips=0.9)  # the bench output is not the reference
    got = by_name(gates.check_gates("dn", tmp_path))["lpips"]
    assert got.threshold == pytest.approx(0.30) and not got.passed
    assert gates.REFERENCE == "apple-base"


def test_every_gate_but_cut_leak_uses_the_reference_not_the_bench_output(tmp_path):
    build(tmp_path)
    for clip in C_CLIPS:  # the bench output is worse than the pre-encode base in every number the ratio gates read
        put(tmp_path, clip, "apple", detail=1.0, jitter_all=0.01, jitter={"static": 0.001, "slow": 0.001, "fast": 0.001},
            jitter_phase=[0.2] * 4, **({"cut_leak": 5.0} if clip == "c-cut-nest" else {}))
    assert all(statuses(tmp_path).values())


def test_cut_leak_is_compared_with_the_bench_apple_not_the_reference(tmp_path):
    build(tmp_path, leak=(5.0, 5.0))
    put(tmp_path, "c-cut-nest", REF, **APPLE, cut_leak=100.0)  # would pass if this were the threshold
    got = by_name(gates.check_gates("dn", tmp_path))["cut_leak"]
    assert got.threshold == pytest.approx(5.0) and got.passed
    build(tmp_path, leak=(5.0, 4.0))
    put(tmp_path, "c-cut-nest", REF, **APPLE, cut_leak=100.0)
    assert not by_name(gates.check_gates("dn", tmp_path))["cut_leak"].passed
    (tmp_path / "c-cut-nest" / "metrics" / "apple.json").unlink()
    got = by_name(gates.check_gates("dn", tmp_path))["cut_leak"]
    assert not got.passed and "apple.json" in got.note


def test_missing_reference_metrics_fail_with_the_command_to_make_them(tmp_path):
    build(tmp_path)
    for clip in C_CLIPS + G3_CLIPS:
        (tmp_path / clip / "metrics" / "apple-base.json").unlink()
    results = by_name(gates.check_gates("dn", tmp_path))
    for name in GATES:
        if name in ("cut_leak", "pulse"):
            assert results[name].passed  # these read the model (and apple for the leak) only, not the reference
        else:
            note = results[name].note
            assert not results[name].passed and "apple-base.json" in note and "--base-only" in note, (name, note)
    assert gates.main(["dn", "--clips", str(tmp_path)]) == 1


def test_reference_can_be_chosen_on_the_command_line(tmp_path):
    build(tmp_path)
    for clip in C_CLIPS:  # a bench output that is much better than apple-base
        put(tmp_path, clip, "apple", detail=1.0, jitter_all=0.01, jitter={"static": 0.001, "slow": 0.001, "fast": 0.001},
            jitter_phase=[0.2] * 4, **({"cut_leak": 5.0} if clip == "c-cut-nest" else {}))
    assert gates.main(["dn", "--clips", str(tmp_path)]) == 0
    assert gates.main(["dn", "--clips", str(tmp_path), "--reference", "apple"]) == 1


def test_apple_bench_numbers_are_printed_as_information_only(tmp_path, capsys):
    build(tmp_path)
    for clip in C_CLIPS:  # against the bench output the model would fail every ratio gate
        put(tmp_path, clip, "apple", **{**APPLE, "detail": 4.0}, **({"cut_leak": 5.0} if clip == "c-cut-nest" else {}))
    assert gates.main(["dn", "--clips", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    head, _, info = out.partition("Information only")
    assert "OVERALL: PASS" in head and "apple-base" in head
    assert info and "apple" in info and "0.7500" in info  # detail ratio 3.0 / 4.0 against the bench output
    assert "PASS" not in info.split("\n", 1)[1] and "FAIL" not in info.split("\n", 1)[1]


def test_information_block_does_not_affect_the_verdict_when_apple_is_missing(tmp_path, capsys):
    build(tmp_path)
    (tmp_path / "c-slow-gorge" / "metrics" / "apple.json").unlink()  # the info block cannot use this clip
    assert gates.main(["dn", "--clips", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "OVERALL: PASS" in out and "n/a" in out.partition("Information only")[2]


def test_summary_argument_is_accepted_and_not_read(tmp_path):
    build(tmp_path)
    assert gates.main(["dn", "--clips", str(tmp_path), "--summary", str(tmp_path / "nope.json")]) == 0


# ---------------------------------------------------------------- summarize: pulse and gt3_lpips

def test_summarize_carries_pulse_and_keeps_g3_out_of_the_x4_means(tmp_path, monkeypatch):
    clips = {"c-a": "real", "e-b": "real", "g-x": "gt", "g3-y": "gt3"}
    for name, kind in clips.items():
        d = tmp_path / name
        d.mkdir()
        (d / "clip.json").write_text(json.dumps({"name": name, "kind": kind, "lr_size": [1280, 720]}))
    base = {"detail": 1.0, "jitter_all": 0.2, "jitter": {"static": 0.2, "slow": 0.2, "fast": 0.2}}
    put(tmp_path, "c-a", "apple", **base, pulse=1.0)
    put(tmp_path, "e-b", "apple", **base)
    put(tmp_path, "g-x", "apple", **base, lpips=0.4)
    put(tmp_path, "g3-y", "apple", **base, lpips=0.3)
    put(tmp_path, "c-a", "dn", **base, pulse=1.2)
    put(tmp_path, "e-b", "dn", **base, pulse=1.4)
    put(tmp_path, "g-x", "dn", **base, lpips=0.5)
    put(tmp_path, "g3-y", "dn", **base, lpips=0.2)
    for model in ("apple", "dn"):  # the speed column needs a run record
        (tmp_path / "c-a" / "out").mkdir(exist_ok=True)
        (tmp_path / "c-a" / "out" / f"{model}.json").write_text(json.dumps(
            {"model": model, "clip": "c-a", "s_per_frame": 1.0, "frame_pixels": 1280 * 720 * 16, "peak_gpu_gb": 1.0}))
    monkeypatch.setattr(summarize, "CLIPS", tmp_path)
    out = tmp_path / "summary.json"
    summarize.main(str(out))
    board = json.loads(out.read_text())["models"]
    assert board["dn"]["pulse"] == pytest.approx(1.3) and board["apple"]["pulse"] == pytest.approx(1.0)
    assert board["dn"]["gt3_lpips"] == pytest.approx(0.2) and board["apple"]["gt3_lpips"] == pytest.approx(0.3)
    assert board["dn"]["gt_lpips"] == pytest.approx(0.5) and board["apple"]["gt_lpips"] == pytest.approx(0.4)
