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
