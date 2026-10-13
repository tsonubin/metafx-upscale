import numpy as np
import pytest

from detailnet.dumps import DumpWriter, read_flow, read_frame, read_meta


def make_writer(d, kind="source", channels=3, width=4, height=2, extra=None):
    return DumpWriter(d, kind, channels, width, height, source_size=(1280, 720), scale=3,
                      crop=(10, 20, 320, 180), extra=extra)


def make_flow_dump(d, values, width=8, height=4):
    w = make_writer(d, "flow", 4, width, height, extra={
        "flow_channels": ["back_x", "back_y", "fwd_x", "fwd_y"], "flow_units": "source_px"})
    w.write(np.broadcast_to(np.array(values, np.float16)[:, None, None], (4, height, width)), pts=0.0)
    w.finish()


def test_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    frames = [rng.standard_normal((3, 2, 4)).astype(np.float16) for _ in range(2)]
    w = make_writer(tmp_path)
    w.write(frames[0], pts=0.0)
    w.write(frames[1], pts=0.04, valid=False)
    w.finish()

    for i, frame in enumerate(frames):
        out = read_frame(tmp_path, i)
        assert out.dtype == np.float16 and out.shape == (3, 2, 4)
        np.testing.assert_array_equal(out, frame)
    assert (tmp_path / "000001.f16").stat().st_size == 48

    meta = read_meta(tmp_path)
    assert meta["frames"] == 2 and meta["pts"] == [0.0, 0.04] and meta["valid"] == [True, False]
    assert meta["kind"] == "source" and meta["layout"] == "CHW" and meta["dtype"] == "float16"
    assert (meta["channels"], meta["width"], meta["height"]) == (3, 4, 2)
    assert meta["source_size"] == [1280, 720] and meta["scale"] == 3 and meta["crop"] == [10, 20, 320, 180]


def test_extra_keys_are_merged_into_meta(tmp_path):
    make_flow_dump(tmp_path, (4, 0, -4, 0))
    meta = read_meta(tmp_path)
    assert meta["flow_channels"] == ["back_x", "back_y", "fwd_x", "fwd_y"]
    assert meta["flow_units"] == "source_px"


def test_write_rejects_wrong_shape(tmp_path):
    w = make_writer(tmp_path)
    with pytest.raises(ValueError):
        w.write(np.zeros((3, 4, 2), np.float16), pts=0.0)


def test_read_flow_rescales_units(tmp_path):
    make_flow_dump(tmp_path, (4, 0, -4, 0))

    same = read_flow(tmp_path, 0, size=(8, 4))
    assert same.dtype == np.float32 and same.shape == (4, 4, 8)
    np.testing.assert_allclose(same[0], 4)

    big = read_flow(tmp_path, 0, size=(24, 12))
    assert big.shape == (4, 12, 24)
    np.testing.assert_allclose(big[0], 12)
    np.testing.assert_allclose(big[2], -12)


def test_read_flow_scales_y_by_height_ratio(tmp_path):
    make_flow_dump(tmp_path, (4, 2, -4, -2))
    out = read_flow(tmp_path, 0, size=(24, 8))
    np.testing.assert_allclose(out[0], 12)
    np.testing.assert_allclose(out[1], 4)
    np.testing.assert_allclose(out[2], -12)
    np.testing.assert_allclose(out[3], -4)


def test_read_flow_without_size_keeps_dump_grid(tmp_path):
    make_flow_dump(tmp_path, (4, 2, -4, -2))
    out = read_flow(tmp_path, 0)
    assert out.dtype == np.float32 and out.shape == (4, 4, 8)
    np.testing.assert_allclose(out[:, 0, 0], [4, 2, -4, -2])


def test_read_flow_resize_is_bilinear(tmp_path):
    w = make_writer(tmp_path, "flow", 4, 2, 1)
    w.write(np.array([[[0, 2]], [[0, 0]], [[0, 0]], [[0, 0]]], np.float16), pts=0.0)
    w.finish()
    out = read_flow(tmp_path, 0, size=(4, 1))
    # half-pixel centres sample [0, 2] at -0.25, 0.25, 0.75, 1.25 (clamped), then x is scaled by 2
    np.testing.assert_allclose(out[0, 0], np.array([0.0, 0.5, 1.5, 2.0]) * 2)


def test_read_flow_rejects_non_flow_dump(tmp_path):
    w = make_writer(tmp_path)
    w.write(np.zeros((3, 2, 4), np.float16), pts=0.0)
    w.finish()
    with pytest.raises(ValueError):
        read_flow(tmp_path, 0)
