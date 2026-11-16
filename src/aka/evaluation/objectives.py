"""Aggregate measurements using the Task objective, independently of Workflow."""
from __future__ import annotations

import math

SPEEDUP_ARITHMETIC_MEAN = "shape_speedup_arithmetic_mean"
SPEEDUP_GEOMETRIC_MEAN = "shape_speedup_geometric_mean"
LATENCY_GEOMETRIC_MEAN = "shape_latency_geometric_mean"
SUPPORTED_OBJECTIVES = frozenset({
    SPEEDUP_ARITHMETIC_MEAN, SPEEDUP_GEOMETRIC_MEAN, LATENCY_GEOMETRIC_MEAN,
})


def metadata_performance_objective(metadata):
    value = metadata.get("performance_objective") if isinstance(metadata, dict) else None
    return value.strip() if isinstance(value, str) and value.strip() else SPEEDUP_ARITHMETIC_MEAN


def metadata_performance_metrics(metadata, shape_ids, latency_by_shape, hardware=""):
    """Compute once over the requested Shape union, never average shard scores."""
    objective = metadata_performance_objective(metadata)
    metrics = {"performance_objective": objective, "performance_score": None,
               "speedup_vs_ref_mean": None, "speedup_vs_ref_geomean": None}
    if objective not in SUPPORTED_OBJECTIVES:
        return metrics, ["unsupported metadata.performance_objective: " + objective]
    if not shape_ids or any(key not in latency_by_shape for key in shape_ids):
        return metrics, []
    values = [latency_by_shape[key] for key in shape_ids]
    if any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0 for value in values):
        return metrics, ["performance aggregation received an invalid latency"]
    if objective == LATENCY_GEOMETRIC_MEAN:
        metrics["performance_score"] = math.exp(sum(math.log(value) for value in values) / len(values))

    from aka.task.problem.inputs import validate_performance_baseline, PerformanceBaselineError
    try:
        reference = validate_performance_baseline(dict.fromkeys(shape_ids, {}), metadata, hardware)
    except PerformanceBaselineError as error:
        return metrics, [] if objective == LATENCY_GEOMETRIC_MEAN else [str(error)]
    ratios = [reference[key] / latency_by_shape[key] for key in shape_ids]
    arithmetic = sum(ratios) / len(ratios)
    geometric = math.exp(sum(math.log(value) for value in ratios) / len(ratios))
    metrics.update(speedup_vs_ref_mean=arithmetic, speedup_vs_ref_geomean=geometric)
    if objective == SPEEDUP_ARITHMETIC_MEAN:
        metrics["performance_score"] = arithmetic
    elif objective == SPEEDUP_GEOMETRIC_MEAN:
        metrics["performance_score"] = geometric
    return metrics, []
