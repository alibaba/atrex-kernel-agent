"""Build exact measurement plans and interpret results without submitting jobs."""
from __future__ import annotations
from dataclasses import dataclass
import json
import hashlib
import math
import statistics
from .objectives import latency_only_result
from .tool_results import PREFIXES, correctness_summary
from .result_json import parse_evaluation_result_line

def result_from_response(response: dict, operation: str) -> dict | None:
    if operation == "evaluate":
        for line in reversed(response["stdout"].splitlines()):
            if value := parse_evaluation_result_line(line):
                return value
        return None
    prefix = PREFIXES.get(operation)
    for line in reversed(response["stdout"].splitlines()):
        if prefix and line.startswith(prefix):
            value = json.loads(line[len(prefix):])
            return value if isinstance(value, dict) else None
    return None


def _median_side(sides: list[dict]) -> dict:
    maps = [side.get("latency_us_by_shape", {}) for side in sides]
    if not maps[0] or any(set(values) != set(maps[0]) for values in maps):
        raise ValueError("Repeated measurement Shape coverage differs; no aggregate is available")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0
           for values in maps for value in values.values()):
        raise ValueError("Repeated measurement contains an invalid latency")
    result = latency_only_result(sides[-1])
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
    return result


def aggregate(results: list[dict], operation: str) -> dict:
    if len(results) == 1:
        return latency_only_result(results[0])
    if operation == "evaluate":
        if any(row.get("all_pass") is not True for row in results):
            raise ValueError("Cannot aggregate incomplete or rejected evaluations")
        return _median_side(results)
    if any(row.get("correct") is not True for row in results):
        raise ValueError("Cannot aggregate incomplete or rejected comparisons")
    value = latency_only_result(results[-1])
    value["baseline"] = _median_side([row["baseline"] for row in results])
    value["candidate"] = _median_side([row["candidate"] for row in results])
    value.pop("measurements", None)
    return value
