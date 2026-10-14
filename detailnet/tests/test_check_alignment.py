import numpy as np
import pytest

from detailnet.tests.check_alignment import area_down, offset_table, verdict


def moving_frames(n, shift_per_frame=4):
    """Frames of a bright bar moving across a dark image: a one-frame slip changes every frame a lot."""
    frames = []
    for i in range(n):
        f = np.zeros((6, 64, 3), np.float32)
        f[:, i * shift_per_frame:i * shift_per_frame + 8] = 200.0
        frames.append(f)
    return frames


def test_offset_table_finds_the_right_offset():
    ref = moving_frames(10)
    dump = [f + 0.5 for f in ref[2:8]]  # the sequence starts at reference frame 2
    table = offset_table(dump, ref, lead=2)
    assert np.mean(table[0]) == pytest.approx(0.5)
    assert np.mean(table[-1]) > 5 and np.mean(table[1]) > 5
    assert verdict(table) == []


def test_offset_table_drops_pairs_outside_the_reference():
    ref = moving_frames(4)
    table = offset_table(ref, ref)
    assert len(table[0]) == 4 and len(table[-1]) == 3 and len(table[1]) == 3


def test_verdict_flags_a_slipped_sequence():
    ref = moving_frames(10)
    slipped = [f for f in ref[3:9]]  # claims to start at frame 2 but holds frames 3..8
    table = offset_table(slipped, ref, lead=2)
    failed = verdict(table)
    assert any("offset 0" in m for m in failed)  # too far from the reference
    assert any("offset +1" in m for m in failed)  # and +1 is the better fit


def test_verdict_flags_static_content_and_noisy_decodes():
    static = [np.full((6, 64, 3), 100.0, np.float32)] * 6
    failed = verdict(offset_table(static, static, lead=0))
    assert len(failed) == 2 and all("not clearly worse" in m for m in failed)
    ref = moving_frames(10)
    noisy = [f + 3.0 for f in ref[2:8]]
    assert any("offset 0" in m for m in verdict(offset_table(noisy, ref, lead=2)))
    assert verdict(offset_table(noisy, ref, lead=2), max_delta=None) == []  # absolute gate off (the x264 copy)


def test_area_down_means_blocks():
    f = np.arange(6 * 6 * 3, dtype=np.float32).reshape(6, 6, 3)
    small = area_down(f, 3)
    assert small.shape == (2, 2, 3)
    assert small[0, 0, 0] == pytest.approx(f[:3, :3, 0].mean())
