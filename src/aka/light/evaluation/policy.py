"""Light's Runtime-owned measurement defaults.

Keep these local to Light: selecting another AKA composition does not change
its evaluator or acceptance policy. Explicit request/operator overrides still
participate in the measurement identity.
"""

from aka.contracts.evaluation import (
    DEFAULT_CANDIDATE_TIMEOUT,
    DEFAULT_PERFORMANCE_TIMEOUT,
    DEFAULT_SHAPE_BATCH_SIZE,
)


# One base case plus four additional deterministic input cases per Shape.
CORRECTNESS_CASES = 5
BENCH_ITERS = 100
WARMUP_ITERS = 20
# The Runtime-owned step timeout supplies both aliases at request construction.
# A multi-Shape request keeps fixed work single and adds only the sequential
# performance budget for each extra Shape.
CANDIDATE_TIMEOUT = DEFAULT_CANDIDATE_TIMEOUT
PERFORMANCE_TIMEOUT = DEFAULT_PERFORMANCE_TIMEOUT
# Light creates one independent Eval job for each outer batch.  Every job keeps
# the complete reference bundle and uses Agate's native ``shape_ids`` selector.
# The request currently forces ``shape_shards=1`` to use the Ray Data contract
# without requiring heterogeneous workers to produce merge-compatible runtime
# environments.
SHAPE_BATCH_SIZE = DEFAULT_SHAPE_BATCH_SIZE
BATCH_WORKERS = 16
MEASUREMENT_CONTRACT = "runtime-native-eval-v1"
ACCEPTANCE_METRIC = "geomean-latency-reduction-v1"

def validate_abba_repeats(repeats: int) -> None:
    if type(repeats) is not int or not 2 <= repeats <= 16 or repeats % 2:
        raise ValueError("--comparison-repeats must be even, in 2..16 (measurements per side)")


def resolved_gpu_clock_mhz(
    hardware: str,
    configured: int | None = None,
) -> int | None:
    """Preserve the old configuration surface while delegating bins to Agate.

    Light no longer sends an explicit graphics clock.  Existing Campaign
    configurations may still contain ``gpu_clock_mhz`` and must remain
    resumable, so normalize both product defaults and historical explicit
    values to ``None`` instead of rejecting the workspace.
    """
    del hardware, configured
    return None


def runner_overrides(
    step_timeout: int = CANDIDATE_TIMEOUT,
    *,
    gpu_clock_mhz: int | None = None,
) -> dict:
    """Return Runtime-owned Bench limits shared by V0 and every candidate."""
    if type(step_timeout) is not int or step_timeout <= 0:
        raise ValueError("Evaluation step timeout must be a positive integer")
    overrides = {
        "warmup_iters": WARMUP_ITERS,
        "candidate_timeout_s": step_timeout,
        "perf_timeout_s": step_timeout,
        "skip_kernel_attribution": True,
    }
    # Kept in the Python signature for callers restoring older Campaign
    # configurations.  Agate owns frequency selection; never put this value on
    # the wire.
    del gpu_clock_mhz
    return overrides


def measurement_defaults(
    hardware: str = "",
    gpu_clock_mhz: int | None = None,
) -> dict:
    del hardware, gpu_clock_mhz
    return {"contract": MEASUREMENT_CONTRACT, "bench_iters": BENCH_ITERS,
            "correctness_cases": CORRECTNESS_CASES, "lock_clocks": True,
            "runner_overrides": runner_overrides()}
