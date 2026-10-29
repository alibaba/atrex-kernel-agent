"""Interpret retained episode measurements without a Workflow result projection."""
import json
import math

from aka.contracts.evaluation import EpisodeMeasurementFacts


def _number(value):
    if value is None:
        return None
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("Workflow measurement must be finite numeric or absent")
    return float(value)


def _measurements(receipt, expected):
    result = json.loads(receipt.result_json)
    if not isinstance(result, dict) or type(result.get("all_pass")) is not bool:
        raise ValueError("episode evaluation requires an explicit all_pass boolean")
    by_case = result.get("latency_us_by_shape") or {}
    if not isinstance(by_case, dict):
        raise ValueError("episode evaluation case latencies must be an object")
    latencies = {}
    for key, value in by_case.items():
        number = _number(value)
        if number is not None and number < 0:
            raise ValueError("episode evaluation latency must not be negative")
        if number is not None and number > 0:
            latencies[str(key)] = number
    latency = _number(result.get("latency_us_geomean", result.get("latency_us")))
    if latency is not None and latency < 0:
        raise ValueError("episode evaluation latency must not be negative")
    latency = latency if latency else None
    complete = bool(expected) and set(latencies) == set(expected) and latency is not None
    passed = result["all_pass"]
    metrics = {name: value for name in ("latency_us_arith_mean", "speedup_vs_ref_mean",
        "speedup_vs_ref_geomean", "max_abs_err", "max_rel_err")
        if (value := _number(result.get(name))) is not None}
    score = _number(result.get("performance_score"))
    return EpisodeMeasurementFacts(
        operation=receipt.operation, timestamp=receipt.timestamp,
        source_sha256=receipt.source_sha256, gate="PASS" if passed else "FAIL",
        correctness="PASS" if passed else "FAIL", complete=complete,
        latency_us=latency, case_latencies_us=latencies, expected_case_ids=expected,
        objective=result.get("performance_objective"), score=score, metrics=metrics,
    )


class LegacyEvaluationEpisodeInterpreter:
    """Explain every retained receipt; completeness is separate from correctness."""
    def interpret_episode_receipts(self, receipts, *, expected_case_ids):
        expected = tuple(expected_case_ids)
        return tuple(_measurements(receipt, expected) for receipt in receipts)
