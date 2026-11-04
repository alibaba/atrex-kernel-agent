"""Default timing budgets; the selected evaluator defines their units and statistics."""

DEFAULT_BENCH_ITERS = 20
# Legacy leaves warmup to the selected evaluator unless explicitly requested.
DEFAULT_WARMUP_ITERS = None


def warmup_overrides(environment=None):
    """Preserve an explicit warmup choice; otherwise use the evaluator default."""
    raw = (environment or {}).get("AKA_GATEWAY_WARMUP_ITERS")
    if raw is None or raw == "default":
        return {}
    if raw is not None and type(raw) not in (int, str):
        raise ValueError("warmup_iters must be a non-negative integer or default")
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError("warmup_iters must be a non-negative integer or default") from error
    if value < 0:
        raise ValueError("warmup_iters must be a non-negative integer or default")
    return {"warmup_iters": value}


def native_command(command, environment=None):
    """Resolve the native script method's defaults before Execution dispatch.

    Typed requests retain their own Legacy method defaults. Frozen/custom Task
    scripts are not rewritten. Iteration units still belong to Atrex-Bench.
    """
    from aka.execution.command_syntax import command_parts, test_kernel_script_index, option_value
    from .drivers import native_harness
    parts = command_parts(list(command))
    index = test_kernel_script_index(parts)
    if index is None or parts[index] not in {"test_kernel.py", "./test_kernel.py"}:
        return tuple(command)
    explicit_warmup = option_value(parts, "--warmup")
    warmup = (explicit_warmup if explicit_warmup is not None else
              warmup_overrides(environment).get("warmup_iters", native_harness.DEFAULT_WARMUP_ITERS))
    defaults = (("--warmup", warmup), ("--timed-runs", native_harness.DEFAULT_BENCH_ITERS),
                ("--candidate-timeout-s", native_harness.DEFAULT_CANDIDATE_TIMEOUT),
                ("--perf-timeout-s", native_harness.DEFAULT_PERFORMANCE_TIMEOUT))
    for flag, value in defaults:
        if option_value(parts, flag) is None:
            parts.extend((flag, str(value)))
    return tuple(parts)
