"""The gating logic of check_dump_colors.py: it must gate on colour (bias, low-pass difference), not on texture."""
import numpy as np

from detailnet.dumps import DumpWriter
from detailnet.tests import check_dump_colors as check

W, H, S = 64, 48, 3
LEVEL = 1 / 255


def make_dumps(tmp_path, change=None, frames=2):
    """A blocky source dump and a base dump that is the source upsampled xS, then `change(base)` (C x H*S x W*S)."""
    rng = np.random.default_rng(1)
    src_dir, base_dir = tmp_path / "s", tmp_path / "b"
    kwargs = dict(source_size=(1280, 720), crop=(0, 0, W, H))
    src_w = DumpWriter(src_dir, "source", 3, W, H, scale=1, **kwargs)
    base_w = DumpWriter(base_dir, "base", 3, W * S, H * S, scale=S, **kwargs)
    for i in range(frames):
        src = np.kron(rng.random((3, H // 8, W // 8)) * 0.6 + 0.2, np.ones((1, 8, 8)))
        base = np.kron(src, np.ones((1, S, S)))
        if change is not None:
            base = change(base)
        src_w.write(src.astype(np.float16), pts=i * 0.04)
        base_w.write(base.astype(np.float16), pts=i * 0.04)
    src_w.finish()
    base_w.finish()
    return src_dir, base_dir


def per_channel(offsets):
    return lambda base: base + np.array(offsets)[:, None, None] * LEVEL


def test_unchanged_base_passes(tmp_path):
    result = check.evaluate(*make_dumps(tmp_path))
    assert check.failures(result) == []
    np.testing.assert_allclose(result.bias, 0, atol=0.05)
    np.testing.assert_allclose(result.lowpass_abs, 0, atol=0.05)


def test_global_offset_fails_the_bias_gate(tmp_path):
    result = check.evaluate(*make_dumps(tmp_path, per_channel([4, 4, 4])))
    np.testing.assert_allclose(result.bias, 4, atol=0.1)
    assert any("bias" in f for f in check.failures(result))


def test_channel_spread_fails_even_when_every_bias_is_small(tmp_path):
    result = check.evaluate(*make_dumps(tmp_path, per_channel([0, 0.7, 1.5])))
    failed = check.failures(result)
    assert len(failed) == 1 and "spread" in failed[0]


def test_texture_only_difference_passes(tmp_path):
    def texture(base):
        # +-10 levels on alternating source pixels (constant inside each SxS block): zero-mean fine detail.
        sign = np.kron((np.indices((H, W)).sum(axis=0) % 2) * 2.0 - 1.0, np.ones((S, S)))
        return base + 10 * LEVEL * sign

    result = check.evaluate(*make_dumps(tmp_path, texture))
    np.testing.assert_allclose(result.raw_abs, 10, atol=0.1)  # reported, not gated
    assert result.lowpass_abs.max() < 1
    assert check.failures(result) == []


def test_low_frequency_colour_error_fails_the_lowpass_gate_even_with_zero_bias(tmp_path):
    def halves(base):
        sign = np.where(np.arange(W * S) < W * S // 2, 1.0, -1.0)  # +8 levels left, -8 right: mean bias 0
        return base + 8 * LEVEL * sign[None, None, :]

    result = check.evaluate(*make_dumps(tmp_path, halves))
    np.testing.assert_allclose(result.bias, 0, atol=0.1)
    assert result.lowpass_abs.min() > check.MAX_LOWPASS_ABS
    assert any("low-pass" in f for f in check.failures(result))


def test_main_exit_codes(tmp_path, capsys):
    good = make_dumps(tmp_path / "good")
    bad = make_dumps(tmp_path / "bad", per_channel([4, 4, 4]))
    assert check.main([str(good[0]), str(good[1])]) == 0
    assert "PASS" in capsys.readouterr().out
    assert check.main([str(bad[0]), str(bad[1])]) == 1
    assert "FAIL" in capsys.readouterr().out
