"""Latency-only normalized measurements; raw provider evidence is unchanged."""
from __future__ import annotations

SCORE_FIELDS = frozenset({
    "performance_score", "performance_objective", "speedup_vs_ref_mean",
    "speedup_vs_ref_geomean", "speedup", "speedup_factor", "improvement_pct",
    "candidate_performance_score", "incumbent_performance_score",
    "utilization_pct", "sol_utilization_pct",
})


def latency_only_result(result: dict) -> dict:
    """Remove scores from measurement containers, including cached results.

    Raw diagnostics and stdout/stderr remain intact for auditing.
    """
    value = {key: item for key, item in result.items() if key not in SCORE_FIELDS}
    for key in ("result", "candidate", "baseline", "performance"):
        if isinstance(value.get(key), dict):
            value[key] = latency_only_result(value[key])
    for key in ("runs", "measurements"):
        if isinstance(value.get(key), list):
            value[key] = [latency_only_result(item) if isinstance(item, dict) else item
                          for item in value[key]]
    return value
