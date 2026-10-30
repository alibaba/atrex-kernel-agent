"""Aggregate complete measurement samples in their declared per-Shape order."""
from __future__ import annotations

import math
import statistics

from .correctness_summary import correctness_summary


def _median_side(sides: list[dict]) -> dict:
    maps = [side.get("latency_us_by_shape", {}) for side in sides]
    if not maps[0] or any(set(values) != set(maps[0]) for values in maps):
        raise ValueError("Repeated measurement Shape coverage differs; no aggregate is available")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0
           for values in maps for value in values.values()):
        raise ValueError("Repeated measurement contains an invalid latency")
    result = dict(sides[-1])
    values = {key: statistics.median(row[key] for row in maps) for key in maps[0]}
    result["latency_us_by_shape"] = values
    result["latency_us_geomean"] = math.exp(statistics.mean(math.log(value) for value in values.values()))
    result["latency_us_arith_mean"] = statistics.mean(values.values())
    for key in ("max_abs_err", "max_rel_err"):
        errors = [side[key] for side in sides if isinstance(side.get(key), (int, float))]
        if errors:
            result[key] = max(errors)
    result["correctness"] = correctness_summary(sides, passed=all(
        side.get("correct", side.get("all_pass")) is True for side in sides))
    # Scores computed on an individual run must not be mislabeled as median facts.
    for key in ("performance_score", "speedup_vs_ref_geomean", "speedup_vs_ref_mean", "utilization_pct", "sol_utilization_pct"):
        result.pop(key, None)
    return result



def aggregate(results: list[dict], operation: str) -> dict:
    if len(results) == 1:
        return results[0]
    if operation == "evaluate":
        if any(row.get("all_pass") is not True for row in results):
            raise ValueError("Cannot aggregate incomplete or rejected evaluations")
        return _median_side(results)
    if any(row.get("correct") is not True for row in results):
        raise ValueError("Cannot aggregate incomplete or rejected comparisons")
    value = dict(results[-1])
    value["baseline"] = _median_side([row["baseline"] for row in results])
    value["candidate"] = _median_side([row["candidate"] for row in results])
    value["speedup"] = value["baseline"]["latency_us_geomean"] / value["candidate"]["latency_us_geomean"]
    value["improvement_pct"] = (value["speedup"] - 1.0) * 100
    value.pop("measurements", None)
    return value

