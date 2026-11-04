"""Light's measurement defaults, aligned with Runtime's production policy.

Keep these local to Light: selecting another AKA composition does not change
its evaluator or acceptance policy. Explicit request/operator overrides still
participate in the measurement identity.
"""

# Preserve the legacy candidate gate: one base seed plus five additional seeds.
CORRECTNESS_CASES = 6
SHAPE_BATCH_SIZE = 1
BATCH_WORKERS = 16
MEASUREMENT_CONTRACT = "runtime-native-eval-v1"
ACCEPTANCE_METRIC = "geomean-latency-reduction-v1"


def validate_abba_repeats(repeats: int) -> None:
    if type(repeats) is not int or not 2 <= repeats <= 16 or repeats % 2:
        raise ValueError("--comparison-repeats must be even, in 2..16 (measurements per side)")


def runner_overrides() -> dict:
    # Match Legacy: let the selected evaluator supply its inner timeout defaults.
    return {}


def measurement_defaults() -> dict:
    return {"contract": MEASUREMENT_CONTRACT,
            "correctness_cases": CORRECTNESS_CASES, "lock_clocks": True,
            "runner_overrides": runner_overrides()}
