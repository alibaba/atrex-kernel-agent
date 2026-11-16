"""Resolve Evaluation timing choices before dispatch; Atrex-Bench owns their units."""

import math

# The standalone adapter holds these values so copied workers need no AKA package.
from .drivers.native_harness import (
    DEFAULT_BENCH_ITERS, DEFAULT_WARMUP_ITERS,
    DEFAULT_CANDIDATE_TIMEOUT, DEFAULT_PERFORMANCE_TIMEOUT,
)


def warmup_overrides(environment=None):
    """Resolve explicit warmup, including zero, or Evaluation's shared default."""
    raw = (environment or {}).get("AKA_GATEWAY_WARMUP_ITERS")
    if raw is None or raw == "default":
        raw = DEFAULT_WARMUP_ITERS
    if type(raw) not in (int, str):
        raise ValueError("warmup_iters must be a non-negative integer or default")
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError("warmup_iters must be a non-negative integer or default") from error
    if value < 0:
        raise ValueError("warmup_iters must be a non-negative integer or default")
    return {"warmup_iters": value}


def runner_overrides(command=(), environment=None, defaults=None):
    """Command > warmup environment > configured policy > shared defaults."""
    from aka.execution.command_syntax import option_value
    values = {"candidate_timeout_s": DEFAULT_CANDIDATE_TIMEOUT,
              "perf_timeout_s": DEFAULT_PERFORMANCE_TIMEOUT,
              "warmup_iters": DEFAULT_WARMUP_ITERS, **(defaults or {})}
    raw = (environment or {}).get("AKA_GATEWAY_WARMUP_ITERS")
    if raw is not None and raw != "default":
        values["warmup_iters"] = raw
    for flag, key in (("--warmup", "warmup_iters"),
                      ("--candidate-timeout-s", "candidate_timeout_s"),
                      ("--perf-timeout-s", "perf_timeout_s")):
        raw = option_value(list(command), flag)
        if raw is not None:
            values[key] = raw
    values.update(warmup_overrides({"AKA_GATEWAY_WARMUP_ITERS": values["warmup_iters"]}))
    for key in ("candidate_timeout_s", "perf_timeout_s"):
        raw = values[key]
        try:
            if type(raw) not in (int, float, str):
                raise ValueError()
            value = float(raw)
            if not math.isfinite(value) or value <= 0:
                raise ValueError()
        except ValueError as error:
            raise ValueError(f"{key} must be a positive finite number") from error
        values[key] = value
    return values


def native_command(command, environment=None):
    """Freeze shared defaults for the built-in script; preserve custom launchers."""
    from aka.execution.command_syntax import command_parts, test_kernel_script_index, option_value
    parts = command_parts(list(command))
    index = test_kernel_script_index(parts)
    if index is None or parts[index] not in {"test_kernel.py", "./test_kernel.py"}:
        return tuple(command)
    timing = runner_overrides(parts, environment)
    defaults = (("--warmup", timing["warmup_iters"]), ("--timed-runs", DEFAULT_BENCH_ITERS),
                ("--candidate-timeout-s", timing["candidate_timeout_s"]),
                ("--perf-timeout-s", timing["perf_timeout_s"]))
    for flag, value in defaults:
        if option_value(parts, flag) is None:
            parts.extend((flag, str(value)))
    return tuple(parts)
