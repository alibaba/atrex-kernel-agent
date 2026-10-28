"""Original ABBA schedule, shape batches and result interpretation."""

from __future__ import annotations
import json
import math
from pathlib import Path
from typing import Any
from aka.contracts.evaluation import VerificationResult, VerificationRun

ABBA_RESULT_PREFIX = "__ATREX_LONG_HORIZON_ABBA_RESULT__="
DEFAULT_SHAPE_BATCH_SIZE = 4
DEFAULT_SHAPE_BATCH_WORKERS = 4


def _payload_from_stdout(stdout: str) -> dict[str, Any]:
    """Extract the long-horizon ABBA payload from ordinary sandbox stdout."""
    for line in reversed(stdout.splitlines()):
        if not line.startswith(ABBA_RESULT_PREFIX):
            continue
        try:
            payload = json.loads(line[len(ABBA_RESULT_PREFIX) :])
        except json.JSONDecodeError as exc:
            raise ValueError("malformed ABBA result sentinel") from exc
        if not isinstance(payload, dict):
            raise ValueError("ABBA result sentinel must contain a JSON object")
        return payload
    raise ValueError("missing ABBA result sentinel")


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


def _geomean(values: list[float]) -> float | None:
    if not values or any(value <= 0.0 or not math.isfinite(value) for value in values):
        return None
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _verification_shape_batches(
    workspace: Path, private_reference_dir: Path | None, batch_size: int
) -> tuple[list[list[str] | None], list[str]]:
    harness = workspace / "test_kernel.py"
    if not harness.is_file() or "--shape-id" not in harness.read_text(encoding="utf-8"):
        return [None], []
    shapes_path = (private_reference_dir or workspace) / "shapes.json"
    if not shapes_path.is_file():
        return [None], []
    shapes = json.loads(shapes_path.read_text(encoding="utf-8"))
    shape_ids = sorted(
        (str(shape_id) for shape_id in shapes),
        key=lambda shape_id: (
            (
                0,
                int(shape_id),
            )
            if shape_id.isdigit()
            else (1, shape_id)
        ),
    )
    return (
        [
            shape_ids[offset : offset + batch_size]
            for offset in range(0, len(shape_ids), batch_size)
        ],
        shape_ids,
    )


def _merge_batch_results(
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
        str(failure) for result in results for failure in (result.get("failures") or [])
    ]
    if missing_shape_ids:
        failures.append(
            "shape batches did not return latency for: " + ", ".join(missing_shape_ids)
        )
    merged.update(
        {
            "all_pass": not missing_shape_ids
            and performance_score is not None
            and all(result.get("all_pass") for result in results),
            "failures": failures,
            "latency_us_geomean": None if missing_shape_ids else _geomean(latencies),
            "latency_us_arith_mean": (
                None if missing_shape_ids else sum(latencies) / len(latencies) if latencies else 0.0
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


def _merge_batch_payloads(
    payloads: list[dict[str, Any]],
    schedule: list[dict[str, int | str]],
    shape_ids: list[str],
) -> dict[str, Any]:
    if len(payloads) == 1:
        return payloads[0]
    for payload in payloads:
        rows = payload.get("runs")
        if (
            payload.get("schema_version") != 1
            or payload.get("error")
            or not isinstance(rows, list)
            or len(rows) != len(schedule)
        ):
            raise ValueError(str(payload.get("error") or "invalid shape batch result"))
        if any(
            not isinstance(row, dict) or not isinstance(row.get("result"), dict)
            for row in rows
        ):
            raise ValueError("shape batch has no valid run result")
        actual = [
            {"revision": row.get("revision"), "repeat": row.get("repeat")}
            for row in rows
        ]
        if actual != schedule:
            raise ValueError("shape batch returned an invalid ABBA schedule")
    runs = []
    for index, step in enumerate(schedule):
        rows = [payload["runs"][index] for payload in payloads]
        result = _merge_batch_results([row["result"] for row in rows], shape_ids)
        runs.append(
            {
                **step,
                "exit_code": (
                    0
                    if result["all_pass"] and all(row["exit_code"] == 0 for row in rows)
                    else 1
                ),
                "result": result,
                "stdout_tail": "\n".join(
                    str(row.get("stdout_tail", "")) for row in rows
                )[-3000:],
                "stderr_tail": "\n".join(
                    str(row.get("stderr_tail", "")) for row in rows
                )[-3000:],
            }
        )
    return {
        "schema_version": 1,
        "runs": runs,
        "error": "; ".join(
            str(payload["error"]) for payload in payloads if payload.get("error")
        )
        or None,
        "shape_batch_count": len(payloads),
    }


def _result_latency(result: object) -> float | None:
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


def _result_performance_score(result: object) -> float | None:
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


def interpret_verification_payload(
    payload: object,
    *,
    schedule: list[dict[str, int | str]],
    repeats: int,
    artifact: str = "",
) -> VerificationResult:
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        return VerificationResult(
            "ERROR", None, None, None, error="unsupported result schema"
        )
    rows = payload.get("runs")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        return VerificationResult(
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
        return VerificationResult(
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
        return VerificationResult(
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
        if (value := _result_latency(run.result)) is not None
    ]
    incumbent_latency_values = [
        value
        for run in runs
        if run.revision == "incumbent" and run.exit_code == 0
        if (value := _result_latency(run.result)) is not None
    ]
    candidate_score_values = [
        value
        for run in runs
        if run.revision == "candidate" and run.exit_code == 0
        if (value := _result_performance_score(run.result)) is not None
    ]
    incumbent_score_values = [
        value
        for run in runs
        if run.revision == "incumbent" and run.exit_code == 0
        if (value := _result_performance_score(run.result)) is not None
    ]
    candidate_latency = _geomean(candidate_latency_values)
    incumbent_latency = _geomean(incumbent_latency_values)
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
        return VerificationResult(
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
    return VerificationResult(
        "PASS",
        candidate_latency,
        incumbent_latency,
        improvement,
        runs=runs,
        artifact=artifact,
        candidate_performance_score=candidate_score,
        incumbent_performance_score=incumbent_score,
    )
