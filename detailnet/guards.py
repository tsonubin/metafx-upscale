"""Guards for long jobs: do not start while the user's 4K batch runs, and stop if swap balloons.

    refuse_if_batch_running(force)   GPU-heavy CLIs call this first; `--force` skips it.
    guard = SwapGuard()              record swap usage at the start ...
    guard.check()                    ... and call this between work units; it raises
                                     SwapLimitExceeded when swap grew by more than the limit.
    batch = BatchGuard(force)        multi-hour jobs also call batch.check() between work units; it
                                     raises BatchStarted once the batch runs (never with `force`).

CLIs catch SwapLimitExceeded and exit with code SWAP_EXIT_CODE (3), and BatchStarted with
BATCH_EXIT_CODE (also 3), after saving their state; their work is resumable.
The readers (`pgrep`, `sysctl`) are injectable so tests don't depend on the machine.
"""
import re
import subprocess

BATCH_PATTERN = "scripts/batch-upscale.sh"
SWAP_EXIT_CODE = 3
BATCH_EXIT_CODE = 3  # stopped on purpose for the user's batch; resumable, like a swap stop
DEFAULT_SWAP_LIMIT_GB = 8.0

_UNIT_GB = {"K": 1 / 1024 ** 2, "M": 1 / 1024, "G": 1.0, "T": 1024.0}


class SwapLimitExceeded(Exception):
    """Swap usage grew by more than the guard's limit."""


class BatchStarted(Exception):
    """The user's batch script is running while a long job runs; the job saves its state and stops."""


def _pgrep(pattern):
    return subprocess.run(["pgrep", "-f", pattern], capture_output=True).returncode == 0


def batch_running(pgrep=_pgrep):
    """True if the user's batch script is running (`pgrep -f scripts/batch-upscale.sh` finds it)."""
    return bool(pgrep(BATCH_PATTERN))


def refuse_if_batch_running(force, pgrep=_pgrep):
    """Exit with a clear message while the batch runs, unless `force`."""
    if not force and batch_running(pgrep):
        raise SystemExit(f"refusing to start: {BATCH_PATTERN} is running and this job would compete with "
                         "it for the GPU. Wait for the batch to finish, or pass --force.")


class BatchGuard:
    """For jobs that run for hours: `check()` between work units raises BatchStarted while the user's batch runs, so
    a batch started after the job began still gets the GPU. With `force` it never looks."""

    def __init__(self, force=False, pgrep=_pgrep):
        self.force = bool(force)
        self._pgrep = pgrep

    def check(self):
        if not self.force and batch_running(self._pgrep):
            raise BatchStarted(f"{BATCH_PATTERN} is running and this job competes with it for the GPU")


def parse_swap_used_gb(text):
    """The 'used' figure of `sysctl -n vm.swapusage` output, in GB."""
    match = re.search(r"\bused\s*=\s*([\d.]+)\s*([KMGT])", text)
    if not match:
        raise ValueError(f"cannot find 'used = <n><unit>' in swap usage {text!r}")
    return float(match.group(1)) * _UNIT_GB[match.group(2)]


def _swap_used_gb():
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True, check=True).stdout
    return parse_swap_used_gb(out)


class SwapGuard:
    """Remembers the swap in use now; `check()` raises SwapLimitExceeded once it has grown by more than
    `limit_gb` since. `read` returns the swap in use, in GB."""

    def __init__(self, limit_gb=DEFAULT_SWAP_LIMIT_GB, read=_swap_used_gb):
        self.limit_gb = limit_gb
        self._read = read
        self.start_gb = read()

    def growth_gb(self):
        return self._read() - self.start_gb

    def check(self):
        growth = self.growth_gb()
        if growth > self.limit_gb:
            raise SwapLimitExceeded(f"swap grew by {growth:.1f} GB since the job started "
                                    f"(limit {self.limit_gb:g} GB); stopping so the machine stays usable")
