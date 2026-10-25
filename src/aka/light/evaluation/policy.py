"""Light's measurement defaults, with Legacy-aligned timing and warmup counts.

Keep these local to Light: selecting another AKA composition does not change
its evaluator or acceptance policy. Explicit request/operator overrides still
participate in the measurement identity.
"""

from aka.contracts.evaluation import (
    DEFAULT_CANDIDATE_TIMEOUT,
    DEFAULT_PERFORMANCE_TIMEOUT,
)


# Preserve the legacy candidate gate: one base seed plus five additional seeds.
CORRECTNESS_CASES = 6
BENCH_ITERS = 20
WARMUP_ITERS = 5
# Budget each Shape independently for candidate initialization/correctness and
# timed measurement.  Operators may still override the enclosing allocation.
CANDIDATE_TIMEOUT = DEFAULT_CANDIDATE_TIMEOUT
PERFORMANCE_TIMEOUT = DEFAULT_PERFORMANCE_TIMEOUT
SHAPE_BATCH_SIZE = 1
BATCH_WORKERS = 16
MEASUREMENT_CONTRACT = "runtime-native-eval-v1"
ACCEPTANCE_METRIC = "geomean-latency-reduction-v1"


def validate_abba_repeats(repeats: int) -> None:
    if type(repeats) is not int or not 2 <= repeats <= 16 or repeats % 2:
        raise ValueError("--comparison-repeats must be even, in 2..16 (measurements per side)")


def runner_overrides() -> dict:
    return {"warmup_iters": WARMUP_ITERS, "candidate_timeout_s": CANDIDATE_TIMEOUT,
            "perf_timeout_s": PERFORMANCE_TIMEOUT}


def measurement_defaults() -> dict:
    return {"contract": MEASUREMENT_CONTRACT, "bench_iters": BENCH_ITERS,
            "correctness_cases": CORRECTNESS_CASES, "lock_clocks": True,
            "runner_overrides": runner_overrides()}
