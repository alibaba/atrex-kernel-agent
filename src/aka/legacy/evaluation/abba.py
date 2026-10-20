"""ABBA schedule and scoring; evidence acquisition remains with each caller."""
from __future__ import annotations
import math
from typing import Any
from .models import VerificationRun, VerificationResult


def verification_schedule(repeats: int) -> list[dict[str, int | str]]:
    schedule: list[dict[str, int | str]] = []
    for repeat in range(max(1, repeats)):
        revisions = (
            ("incumbent", "candidate")
            if repeat % 2 == 0
            else ("candidate", "incumbent")
        )
        schedule.extend(
            {"revision": revision, "repeat": repeat} for revision in revisions
        )
    return schedule


def geomean(values: list[float]) -> float | None:
    if not values or any(value <= 0.0 or not math.isfinite(value) for value in values):
        return None
    return math.exp(sum(math.log(value) for value in values) / len(values))


def merge_batch_results(
    results: list[dict[str, Any]], shape_ids: list[str]
) -> dict[str, Any]:
    latency_by_shape = {
        str(shape_id): float(latency)
        for result in results
        for shape_id, latency in (result.get("latency_us_by_shape") or {}).items()
    }
    missing_shape_ids = [
        shape_id for shape_id in shape_ids if shape_id not in latency_by_shape
    ]
    latencies = [
        latency_by_shape[shape_id]
        for shape_id in shape_ids
        if shape_id in latency_by_shape
    ]
    weighted_scores: list[tuple[float, int]] = []
    for result in results:
        raw_score = result.get(
            "performance_score", result.get("speedup_vs_ref_geomean")
        )
        by_shape = result.get("latency_us_by_shape")
        if (
            isinstance(raw_score, (int, float))
            and not isinstance(raw_score, bool)
            and raw_score > 0.0
            and math.isfinite(float(raw_score))
            and isinstance(by_shape, dict)
        ):
            weighted_scores.append((float(raw_score), len(by_shape)))
    measured_score_shapes = sum(count for _score, count in weighted_scores)
    performance_score = (
        sum(score * count for score, count in weighted_scores) / measured_score_shapes
        if measured_score_shapes == len(shape_ids) and measured_score_shapes > 0
        else None
    )
    merged = dict(results[-1])
    failures = [
        str(failure)
        for result in results
        for failure in (result.get("failures") or [])
    ]
    if missing_shape_ids:
        failures.append(
            "shape batches did not return latency for: "
            + ", ".join(missing_shape_ids)
        )
    merged.update(
        {
            "all_pass": not missing_shape_ids
            and performance_score is not None
            and all(result.get("all_pass") for result in results),
            "failures": failures,
            "latency_us_geomean": geomean(latencies),
            "latency_us_arith_mean": (
                sum(latencies) / len(latencies) if latencies else 0.0
            ),
            "latency_us_by_shape": latency_by_shape,
            "speedup_vs_ref_mean": performance_score,
            "speedup_vs_ref_geomean": None,
            "performance_score": performance_score,
            "max_abs_err": max(
                float(result.get("max_abs_err") or 0.0) for result in results
            ),
            "max_rel_err": max(
                float(result.get("max_rel_err") or 0.0) for result in results
            ),
        }
    )
    return merged


def result_latency(result: object) -> float | None:
    if not isinstance(result, dict) or not result.get("all_pass"):
        return None
    value = result.get("latency_us_geomean")
    if not isinstance(value, (int, float)):
        performance = result.get("performance")
        value = performance.get("latency_us") if isinstance(performance, dict) else None
    if (
        not isinstance(value, (int, float))
        or value <= 0
        or not math.isfinite(float(value))
    ):
        return None
    return float(value)


def result_performance_score(result: object) -> float | None:
    if not isinstance(result, dict) or not result.get("all_pass"):
        return None
    value = result.get("performance_score", result.get("speedup_vs_ref_geomean"))
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or value <= 0
        or not math.isfinite(float(value))
    ):
        return None
    return float(value)


def score_verification_payload(
    payload: object,
    *,
    schedule: list[dict[str, int | str]],
    repeats: int,
    min_improvement_pct: float,
    artifact: str = "",
    result_type=VerificationResult,
) -> VerificationResult:
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        return result_type(
            "ERROR", None, None, None, error="unsupported result schema"
        )
    rows = payload.get("runs")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        return result_type(
            "ERROR", None, None, None, error="runs must be a list of objects"
        )
    runs = [
        VerificationRun(
            revision=str(row.get("revision", "")),
            repeat=int(row.get("repeat", -1)),
            exit_code=int(row.get("exit_code", -1)),
            result=row.get("result") if isinstance(row.get("result"), dict) else None,
            stdout_tail=str(row.get("stdout_tail", "")),
            stderr_tail=str(row.get("stderr_tail", "")),
        )
        for row in rows
    ]
    if payload.get("error"):
        return result_type(
            "ERROR",
            None,
            None,
            None,
            runs=runs,
            error=str(payload["error"]),
            artifact=artifact,
        )
    actual = [{"revision": run.revision, "repeat": run.repeat} for run in runs]
    if actual != schedule:
        return result_type(
            "ERROR",
            None,
            None,
            None,
            runs=runs,
            error="remote verifier did not execute the exact ABBA schedule",
            artifact=artifact,
        )
    candidate_latency_values = [
        value
        for run in runs
        if run.revision == "candidate" and run.exit_code == 0
        if (value := result_latency(run.result)) is not None
    ]
    incumbent_latency_values = [
        value
        for run in runs
        if run.revision == "incumbent" and run.exit_code == 0
        if (value := result_latency(run.result)) is not None
    ]
    candidate_score_values = [
        value
        for run in runs
        if run.revision == "candidate" and run.exit_code == 0
        if (value := result_performance_score(run.result)) is not None
    ]
    incumbent_score_values = [
        value
        for run in runs
        if run.revision == "incumbent" and run.exit_code == 0
        if (value := result_performance_score(run.result)) is not None
    ]
    candidate_latency = geomean(candidate_latency_values)
    incumbent_latency = geomean(incumbent_latency_values)
    candidate_score = (
        sum(candidate_score_values) / len(candidate_score_values)
        if candidate_score_values
        else None
    )
    incumbent_score = (
        sum(incumbent_score_values) / len(incumbent_score_values)
        if incumbent_score_values
        else None
    )
    if (
        len(candidate_latency_values) != repeats
        or len(incumbent_latency_values) != repeats
        or len(candidate_score_values) != repeats
        or len(incumbent_score_values) != repeats
    ):
        return result_type(
            "FAIL",
            candidate_latency,
            incumbent_latency,
            None,
            runs=runs,
            error=(
                "not every authoritative ABBA run passed with a valid "
                "performance score"
            ),
            artifact=artifact,
            candidate_performance_score=candidate_score,
            incumbent_performance_score=incumbent_score,
        )
    improvement = (
        ((candidate_score / incumbent_score) - 1.0) * 100.0
        if candidate_score and incumbent_score
        else None
    )
    if improvement is None or improvement <= min_improvement_pct:
        return result_type(
            "FAIL",
            candidate_latency,
            incumbent_latency,
            improvement,
            runs=runs,
            error=(
                f"candidate improvement {improvement if improvement is not None else 'unknown'} "
                f"in performance score did not exceed {min_improvement_pct:.3f}%"
            ),
            artifact=artifact,
            candidate_performance_score=candidate_score,
            incumbent_performance_score=incumbent_score,
        )
    return result_type(
        "PASS",
        candidate_latency,
        incumbent_latency,
        improvement,
        runs=runs,
        artifact=artifact,
        candidate_performance_score=candidate_score,
        incumbent_performance_score=incumbent_score,
    )
