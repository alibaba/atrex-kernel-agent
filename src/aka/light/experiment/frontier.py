"""Observed per-Shape minima from the Campaign's standard full Evaluations."""
from __future__ import annotations

import logging
import math
from datetime import datetime

from aka.light.evaluation.provider import result_from_response
from aka.light.evaluation.public_result import canonical_result, input_scope
from aka.light.experiment.store import RECORD_ID, MeasurementStore, read_record_header


def standard_full_evaluation(record: dict) -> dict | None:
    """Parse one ordinary full Evaluate, including a compact failure reason."""
    if (
        record.get("operation") != "evaluate"
        or record.get("request", {}).get("operation") != "evaluate"
        or record.get("cacheable") is not True
    ):
        return None
    options = record["request"].get("options", {})
    if not isinstance(options, dict):
        return None
    command = options.get("command", [])
    policy = options.get("evaluation_policy", {})
    if not isinstance(policy, dict) or not isinstance(command, list):
        return None
    if (
        (options.get("evaluation_mode") or "full") != "full"
        or policy.get("mode", "full") != "full"
        or options.get("evaluation_input_path")
        or options.get("evaluation_shapes_path")
        or options.get("baseline_path")
        or command[:2] != ["python3", "test_kernel.py"]
        or any(
            isinstance(arg, str)
            and (arg == "--shape-id" or arg.startswith("--shape-id="))
            for arg in command
        )
    ):
        return None

    response = record.get("response", {})
    result = result_from_response(response, "evaluate")
    result_parsed = isinstance(result, dict)
    result = result if result_parsed else {}
    values = result.get("latency_us_by_shape")
    latencies = (
        {
            shape: float(value)
            for shape, value in values.items()
            if isinstance(shape, str)
            and type(value) in {int, float}
            and math.isfinite(float(value))
            and float(value) > 0
        }
        if isinstance(values, dict)
        else {}
    )
    correctness_passed = bool(
        response.get("exit_code") == 0
        and not response.get("truncated")
        and result_parsed
        and result.get("all_pass") is True
        and not result.get("error")
        and result.get("mode", "full") == "full"
    )
    performance_complete = bool(
        isinstance(values, dict)
        and values
        and len(latencies) == len(values)
    )
    passed = correctness_passed and performance_complete
    geomean = result.get("latency_us_geomean")
    if latencies and (
        type(geomean) not in {int, float}
        or not math.isfinite(float(geomean))
        or float(geomean) <= 0
    ):
        geomean = math.exp(
            sum(math.log(value) for value in latencies.values()) / len(latencies)
        )
    if not result_parsed:
        failure_kind = "result_unparseable"
        reason = "full Evaluate did not produce a parseable result"
    elif (
        result.get("all_pass") is not True
        or result.get("error")
        or result.get("mode", "full") != "full"
    ):
        failure_kind = "correctness_failed"
        reason = result.get("error") or result.get("outcome")
    elif response.get("exit_code") != 0 or response.get("truncated"):
        failure_kind = "evaluation_execution_failed"
        reason = result.get("error") or result.get("outcome")
    elif not performance_complete:
        failure_kind = "performance_result_incomplete"
        reason = (
            "full Evaluate correctness passed, but latency_us_by_shape is missing, "
            "empty, or contains non-finite/non-positive values"
        )
    else:
        failure_kind = None
        reason = result.get("error") or result.get("outcome")
    if not isinstance(reason, str) or not reason:
        reason = (
            f"Evaluate exited with status {response.get('exit_code')}"
            if response.get("exit_code") != 0
            else "full Evaluate did not produce a complete passing result"
        )
    return {
        "passed": passed,
        "correctness_passed": correctness_passed,
        "performance_complete": performance_complete,
        "failure_kind": failure_kind,
        "latencies": latencies,
        "latency_us_geomean": (
            float(geomean)
            if type(geomean) in {int, float}
            and math.isfinite(float(geomean))
            and float(geomean) > 0
            else None
        ),
        "reason": reason,
    }


def kernel_full_evaluation_evidence(
    store: MeasurementStore,
    kernel_id: str,
) -> dict:
    """Return saved full-Evaluate evidence for one exact Agent Git commit."""
    passing: list[tuple[float, str, float | None, int]] = []
    failed: list[tuple[float, str, str | None, str]] = []
    for path in (store.root / "records").glob("gateway-*/record.json"):
        record_id = path.parent.name
        try:
            if not RECORD_ID.fullmatch(record_id):
                continue
            header = read_record_header(path)
            if header["operation"] != "evaluate":
                continue
            record = store.read(record_id)
            if record["kernels"]["candidate"]["kernel_id"] != kernel_id:
                continue
            parsed = standard_full_evaluation(record)
            if parsed is None:
                continue
            created_at = datetime.fromisoformat(header["created_at"]).timestamp()
            if parsed["passed"]:
                passing.append(
                    (
                        created_at,
                        record_id,
                        parsed["latency_us_geomean"],
                        len(parsed["latencies"]),
                    )
                )
            else:
                failed.append(
                    (
                        created_at,
                        record_id,
                        parsed["failure_kind"],
                        parsed["reason"],
                    )
                )
        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
            AttributeError,
            RecursionError,
            OverflowError,
        ):
            logging.getLogger(__name__).warning(
                "Skipping invalid full-Evaluate evidence %s", record_id
            )
    if passing:
        _created_at, record_id, latency, shape_count = max(passing)
        evidence = {
            "status": "passed",
            "gateway_record_id": record_id,
        }
        if latency is not None:
            evidence["latency_us_geomean"] = latency
        if shape_count:
            evidence["measured_shape_count"] = shape_count
        return evidence
    if failed:
        _created_at, record_id, reason_code, reason = max(failed)
        return {
            "status": "failed",
            "gateway_record_id": record_id,
            "reason_code": reason_code,
            "reason": reason,
        }
    return {"status": "missing"}


def latest_gateway_result(
    store: MeasurementStore,
    *,
    record_ids: set[str] | None = None,
) -> dict | None:
    """Return a compact, trusted summary of the newest terminal Gateway Record."""
    candidates: list[tuple[float, str]] = []
    for path in (store.root / "records").glob("gateway-*/record.json"):
        record_id = path.parent.name
        try:
            if not RECORD_ID.fullmatch(record_id) or (
                record_ids is not None and record_id not in record_ids
            ):
                continue
            header = read_record_header(path)
            if header["gateway_record_id"] != record_id:
                raise ValueError("Gateway Record identity differs from its path")
            candidates.append(
                (datetime.fromisoformat(header["created_at"]).timestamp(), record_id)
            )
        except (OSError, ValueError, KeyError, TypeError, RecursionError):
            logging.getLogger(__name__).warning(
                "Skipping invalid latest-result header %s", record_id
            )
    for created_at, record_id in sorted(candidates, reverse=True):
        try:
            record = store.read(record_id)
            public = canonical_result(
                record["response"],
                record["request"],
                record_id=record_id,
                kernels=record["kernels"],
            )
            result = public.get("result")
            result = result if isinstance(result, dict) else {}
            options = record["request"].get("options", {})
            options = options if isinstance(options, dict) else {}
            latencies = result.get("latency_us_by_shape")
            shape_count = len(latencies) if isinstance(latencies, dict) else None
            geomean = result.get("latency_us_geomean")
            latency_values = (
                [
                    float(value)
                    for value in latencies.values()
                    if type(value) in {int, float}
                    and math.isfinite(float(value))
                    and float(value) > 0
                ]
                if isinstance(latencies, dict)
                else []
            )
            if (
                (type(geomean) not in {int, float} or not math.isfinite(float(geomean)))
                and latency_values
            ):
                geomean = math.exp(
                    sum(math.log(value) for value in latency_values) / len(latency_values)
                )
            operation = public.get("operation") or record["operation"]
            scope = {
                "mode": result.get("mode"),
                "input_scope": (
                    result.get("input_scope")
                    if isinstance(result.get("input_scope"), str)
                    else input_scope(options)
                    if operation == "evaluate"
                    else None
                ),
                "measured_shape_count": shape_count,
            }
            return {
                "created_at_epoch_s": created_at,
                "gateway_record_id": record_id,
                "operation": operation,
                "status": public.get("status"),
                "kernel_commit_id": (
                    public.get("candidate_kernel_commit_id")
                    or public.get("kernel_commit_id")
                ),
                "scope": {key: value for key, value in scope.items() if value is not None},
                "correct": result.get("correct"),
                "correctness_status": (
                    result.get("correctness", {}).get("status")
                    if isinstance(result.get("correctness"), dict)
                    else None
                ),
                "latency_us_geomean": geomean,
                "speedup": result.get("speedup"),
                "error": result.get("error"),
                "outcome": result.get("outcome"),
            }
        except (OSError, ValueError, KeyError, TypeError, AttributeError, RecursionError):
            logging.getLogger(__name__).warning(
                "Skipping invalid latest-result evidence %s", record_id
            )
    return None


def kernel_pareto_frontier(
    store: MeasurementStore,
    *,
    record_ids: set[str] | None = None,
    kernel_ids: set[str] | None = None,
) -> dict:
    """Read existing facts only; a per-Shape envelope is not a measured composite Kernel.

    Use the same ordinary-full-Evaluate eligibility as nomination, never Profile,
    ABBA, custom-input or subset timings. The store is Campaign-scoped and shared
    across Episodes. Unlike Runtime's Registry, Light does not distinguish Agent
    and controller-origin full Evaluations; both are eligible saved observations.
    """
    winners: dict[str, tuple[float, str, str]] = {}
    best_full: tuple[float, str, str] | None = None
    evaluation_count = 0
    for path in (store.root / "records").glob("gateway-*/record.json"):
        record_id = path.parent.name
        try:
            if not RECORD_ID.fullmatch(record_id) or (
                record_ids is not None and record_id not in record_ids
            ):
                continue
            # Do not load response bodies or source artifacts for other operations.
            header = read_record_header(path)
            if header["operation"] != "evaluate":
                continue
            record = store.read(record_id)
            parsed = standard_full_evaluation(record)
            if parsed is None or not parsed["passed"] or not parsed["latencies"]:
                continue
            latencies = parsed["latencies"]
            kernel_id = record["kernels"]["candidate"]["kernel_id"]
            if kernel_ids is not None and kernel_id not in kernel_ids:
                continue
            geomean = parsed["latency_us_geomean"]
            if geomean is None:
                continue
        except (OSError, ValueError, KeyError, TypeError, AttributeError, RecursionError, OverflowError):
            # An interrupted/corrupt historical record must not break every query;
            # details stay in the Supervisor log, not in the Agent response.
            logging.getLogger(__name__).warning("Skipping invalid frontier evidence %s", record_id)
            continue
        evaluation_count += 1
        full_candidate = (float(geomean), kernel_id, record_id)
        if best_full is None or full_candidate < best_full:
            best_full = full_candidate
        for shape, latency in latencies.items():
            candidate = (latency, kernel_id, record_id)
            if shape not in winners or candidate < winners[shape]:
                winners[shape] = candidate
    return {
        "best_by_shape": {
            shape: {
                "kernel_commit_id": kernel,
                "gateway_record_id": record,
                "latency_us": latency,
            }
            for shape, (latency, kernel, record) in sorted(winners.items())
        },
        "shape_count": len(winners),
        "winner_kernel_commit_ids": sorted({kernel for _, kernel, _ in winners.values()}),
        "evaluation_count": evaluation_count,
        "best_full_evaluation": (
            {
                "latency_us_geomean": best_full[0],
                "kernel_commit_id": best_full[1],
                "gateway_record_id": best_full[2],
            }
            if best_full is not None
            else None
        ),
    }
