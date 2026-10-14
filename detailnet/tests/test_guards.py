import pytest

from detailnet.guards import (SwapGuard, SwapLimitExceeded, batch_running, parse_swap_used_gb,
                              refuse_if_batch_running)

GB = 1.0


def test_parse_swap_used_handles_units():
    line = "total = 4096.00M  used = 3252.69M  free = 843.31M  (encrypted)"
    assert parse_swap_used_gb(line) == pytest.approx(3252.69 / 1024)
    assert parse_swap_used_gb("total = 12.00G  used = 2.50G  free = 9.50G  (encrypted)") == 2.5
    assert parse_swap_used_gb("total = 0.00M  used = 0.00M  free = 0.00M  (encrypted)") == 0.0
    assert parse_swap_used_gb("total = 10.00M  used = 512.00K  free = 9.50M") == pytest.approx(0.5 / 1024)


def test_parse_swap_used_rejects_garbage():
    with pytest.raises(ValueError):
        parse_swap_used_gb("vm.swapusage: not available")


def test_batch_running_follows_pgrep():
    seen = []

    def pgrep(pattern):
        seen.append(pattern)
        return True

    assert batch_running(pgrep) is True
    assert seen == ["scripts/batch-upscale.sh"]
    assert batch_running(lambda pattern: False) is False


def test_refuse_if_batch_running_exits_with_a_message():
    with pytest.raises(SystemExit) as stopped:
        refuse_if_batch_running(False, pgrep=lambda pattern: True)
    assert "batch-upscale.sh" in str(stopped.value)
    assert "--force" in str(stopped.value)


def test_refuse_if_batch_running_passes_when_idle_or_forced():
    refuse_if_batch_running(False, pgrep=lambda pattern: False)
    refuse_if_batch_running(True, pgrep=lambda pattern: pytest.fail("force must not even look"))


def test_swap_guard_allows_growth_up_to_the_limit():
    used = [3.0]
    guard = SwapGuard(limit_gb=8.0, read=lambda: used[0])
    guard.check()
    used[0] = 11.0  # exactly 8 GB of growth is still fine
    guard.check()
    used[0] = 1.0  # swap shrinking is fine
    guard.check()


def test_swap_guard_raises_past_the_limit():
    used = [3.0]
    guard = SwapGuard(limit_gb=8.0, read=lambda: used[0])
    used[0] = 11.5
    with pytest.raises(SwapLimitExceeded) as tripped:
        guard.check()
    assert "8.5" in str(tripped.value) and "8" in str(tripped.value)


def test_swap_guard_measures_from_construction():
    used = [5.0]
    guard = SwapGuard(limit_gb=2.0, read=lambda: used[0])
    used[0] = 6.9
    guard.check()
    used[0] = 7.1
    with pytest.raises(SwapLimitExceeded):
        guard.check()


def test_swap_guard_default_limit_is_8_gb():
    used = [0.0]
    guard = SwapGuard(read=lambda: used[0])
    used[0] = 7.9
    guard.check()
    used[0] = 8.1
    with pytest.raises(SwapLimitExceeded):
        guard.check()


def test_swap_limit_exceeded_is_an_exception():
    assert issubclass(SwapLimitExceeded, Exception)


def test_real_readers_run_on_this_machine():
    # Not a check of the numbers, only that the default readers work here.
    assert isinstance(batch_running(), bool)
    SwapGuard().check()
