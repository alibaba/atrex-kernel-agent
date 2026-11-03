"""Default timing budgets; the selected evaluator defines their units and statistics."""

DEFAULT_BENCH_ITERS = 100
DEFAULT_WARMUP_ITERS = 10


def warmup_overrides(environment=None):
    """Preserve an explicit warmup choice; otherwise freeze the common default."""
    raw = (environment or {}).get("AKA_GATEWAY_WARMUP_ITERS")
    if raw == "default":
        return {}
    if raw is not None and type(raw) not in (int, str):
        raise ValueError("warmup_iters must be a non-negative integer or default")
    value = DEFAULT_WARMUP_ITERS if raw is None else int(raw)
    if isinstance(raw, bool) or value < 0:
        raise ValueError("warmup_iters must be a non-negative integer or default")
    return {"warmup_iters": value}
