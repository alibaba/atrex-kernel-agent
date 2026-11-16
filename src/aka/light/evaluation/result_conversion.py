"""Light's retained Atrex-Bench result conversion and safe diagnostics.

The upstream Workflow no longer exposes these conversion helpers from its
history-oriented ``evaluation.results`` module.  Keep the Light projection
contract local instead of extending the shared module with Light-only APIs.
"""
from __future__ import annotations

from typing import Any
import math
import statistics

PERFORMANCE_OBJECTIVE = "shape_speedup_arithmetic_mean"
SPEEDUP_ARITHMETIC_MEAN = PERFORMANCE_OBJECTIVE
SPEEDUP_GEOMETRIC_MEAN = "shape_speedup_geometric_mean"
LATENCY_GEOMETRIC_MEAN = "shape_latency_geometric_mean"
SUPPORTED_PERFORMANCE_OBJECTIVES = frozenset({
    SPEEDUP_ARITHMETIC_MEAN,
    SPEEDUP_GEOMETRIC_MEAN,
    LATENCY_GEOMETRIC_MEAN,
})


def finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def compile_failures(compile_result: object, shape_ids: list[str]) -> list[str]:
    if not isinstance(compile_result, dict):
        compile_result = {}
    if "status" in compile_result:
        if compile_result.get("status") == "passed":
            return []
        return ["compile did not pass; see actionable_diagnostics"]
    failures = []
    for shape_id in shape_ids:
        status = compile_result.get(shape_id)
        status = status if isinstance(status, dict) else {}
        if status.get("status") != "passed":
            failures.append(f"sid={shape_id}: compile did not pass; see actionable_diagnostics")
    return failures


def actionable_diagnostic(value: object) -> str:
    """Return the complete diagnostic produced by Agate or the compiler.

    Compiler output is commonly repetitive near the front while the decisive
    error is at the end.  Length-based head/tail projection therefore changes
    the meaning of the evidence and is not safe at the Agent boundary.  NULs
    are removed because they cannot be represented by every downstream text
    transport; redaction is applied later by the public projection.
    """
    return str(value or "").replace("\x00", "").strip()


def looks_like_candidate_exception(value: object) -> bool:
    lowered = str(value or "").lower()
    return any(marker in lowered for marker in (
        "compilation error", "compileerror", "cuda_error_", "cudaerror",
        "culaunchkernel", "cumodule", "invalid context", "kernelparams",
        "modulenotfounderror:", "importerror:", "nvrtc", "syntaxerror:",
        "undefined symbol",
    ))


def compile_diagnostics(compile_result: object,
                        shape_ids: list[str]) -> list[dict[str, Any]]:
    if not isinstance(compile_result, dict):
        return []
    if "status" in compile_result:
        if compile_result.get("status") == "passed":
            return []
        message = actionable_diagnostic(
            compile_result.get("reason") or compile_result.get("status")
        )
        return ([{"stage": "compile_import", "shape_id": "", "message": message}]
                if message else [])
    diagnostics = []
    seen: dict[str, dict[str, Any]] = {}
    for shape_id in shape_ids:
        status = compile_result.get(shape_id)
        status = status if isinstance(status, dict) else {}
        if status.get("status") == "passed":
            continue
        message = actionable_diagnostic(
            status.get("reason") or status.get("status") or "missing"
        )
        if not message:
            continue
        if message in seen:
            item = seen[message]
            item.setdefault("shape_ids", [item["shape_id"]]).append(shape_id)
            continue
        item = {
            "stage": "compile_import",
            "shape_id": shape_id,
            "message": message,
        }
        seen[message] = item
        diagnostics.append(item)
    return diagnostics


def candidate_exception_diagnostics(correctness_status: dict[str, Any],
                                    correctness_shapes: dict[str, Any],
                                    shape_ids: list[str]) -> list[dict[str, Any]]:
    diagnostics = []
    seen: dict[str, dict[str, Any]] = {}
    for shape_id in shape_ids:
        status = correctness_status.get(shape_id)
        status = status if isinstance(status, dict) else {}
        reasons = [status.get("reason")]
        shape_result = correctness_shapes.get(shape_id)
        shape_result = shape_result if isinstance(shape_result, dict) else {}
        cases = shape_result.get("cases")
        for case in cases if isinstance(cases, list) else []:
            if isinstance(case, dict):
                reasons.append(case.get("error"))
        for reason in reasons:
            if not looks_like_candidate_exception(reason):
                continue
            message = actionable_diagnostic(reason)
            if not message:
                continue
            if message in seen:
                item = seen[message]
                known_shapes = item.setdefault("shape_ids", [item["shape_id"]])
                if shape_id not in known_shapes:
                    known_shapes.append(shape_id)
                continue
            item = {
                "stage": "candidate_runtime",
                "shape_id": shape_id,
                "message": message,
            }
            seen[message] = item
            diagnostics.append(item)
    return diagnostics


def metadata_shape_latency_us(metadata: object, shape_id: str) -> float | None:
    if not isinstance(metadata, dict):
        return None
    shapes = metadata.get("shapes")
    shape = shapes.get(shape_id) if isinstance(shapes, dict) else None
    if not isinstance(shape, dict):
        return None
    production = shape.get("production_performance")
    if not isinstance(production, dict):
        return None
    direct = finite_number(production.get("performance_us"))
    if direct is not None and direct > 0.0:
        return direct
    nested = [
        value for entry in production.values() if isinstance(entry, dict)
        if (value := finite_number(entry.get("performance_us"))) is not None
        and value > 0.0
    ]
    return nested[0] if len(nested) == 1 else None


def metadata_performance_objective(metadata: object) -> str:
    """Return the task's declared objective, retaining the legacy default.

    Older tasks predate ``metadata.performance_objective`` and used the
    arithmetic mean of per-Shape speedups.  An explicit value is never
    rewritten to that compatibility default.
    """
    if isinstance(metadata, dict):
        objective = metadata.get("performance_objective")
        if isinstance(objective, str) and objective.strip():
            return objective.strip()
    return PERFORMANCE_OBJECTIVE


def metadata_performance_metrics(
    metadata: object,
    shape_ids: list[str],
    latency_by_shape: dict[str, float],
) -> tuple[dict[str, float | str | None], list[str]]:
    """Aggregate a complete Shape set according to the task metadata.

    This function is intentionally a whole-request operation.  Callers must
    first merge all outer Shape shards and pass the complete requested Shape
    set; averaging shard-level scores changes geometric objectives.
    """
    objective = metadata_performance_objective(metadata)
    metrics: dict[str, float | str | None] = {
        "performance_objective": objective,
        "performance_score": None,
        "speedup_vs_ref_mean": None,
        "speedup_vs_ref_geomean": None,
    }
    if objective not in SUPPORTED_PERFORMANCE_OBJECTIVES:
        return metrics, [
            "unsupported metadata.performance_objective: " + objective
        ]
    if any(shape_id not in latency_by_shape for shape_id in shape_ids):
        return metrics, []
    latencies = [latency_by_shape[shape_id] for shape_id in shape_ids]
    if not latencies:
        return metrics, []
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
        for value in latencies
    ):
        return metrics, ["performance aggregation received an invalid latency"]

    latency_geomean = math.exp(
        sum(math.log(float(value)) for value in latencies) / len(latencies)
    )
    if objective == LATENCY_GEOMETRIC_MEAN:
        metrics["performance_score"] = latency_geomean

    speedups: list[float] = []
    failures: list[str] = []
    for shape_id in shape_ids:
        reference_us = metadata_shape_latency_us(metadata, shape_id)
        if reference_us is None:
            failures.append(
                f"sid={shape_id}: metadata has no unambiguous positive "
                "production_performance.performance_us"
            )
        else:
            speedups.append(reference_us / latency_by_shape[shape_id])
    if not failures and len(speedups) == len(shape_ids) and speedups:
        arithmetic = sum(speedups) / len(speedups)
        geometric = math.exp(
            sum(math.log(value) for value in speedups) / len(speedups)
        )
        metrics["speedup_vs_ref_mean"] = arithmetic
        metrics["speedup_vs_ref_geomean"] = geometric
        if objective == SPEEDUP_ARITHMETIC_MEAN:
            metrics["performance_score"] = arithmetic
        elif objective == SPEEDUP_GEOMETRIC_MEAN:
            metrics["performance_score"] = geometric
    elif objective == LATENCY_GEOMETRIC_MEAN:
        # Reference speedups are optional diagnostics for a latency objective.
        failures = []
    return metrics, failures


def metadata_speedup_mean(
    metadata: object,
    shape_ids: list[str],
    latency_by_shape: dict[str, float],
) -> tuple[float | None, list[str]]:
    """Compatibility adapter returning the metadata-selected score."""
    metrics, failures = metadata_performance_metrics(
        metadata, shape_ids, latency_by_shape
    )
    score = metrics.get("performance_score")
    return (float(score) if isinstance(score, (int, float)) else None), failures


def optimizer_result_from_eval(payload: dict[str, Any], shape_ids: list[str],
                               metadata: object, *, require_performance: bool = True,
                               finalize_performance: bool = True) -> dict[str, Any]:
    failures = []
    if payload.get("error"):
        failures.append("evaluation: " + str(payload["error"]))
    passed = payload.get("passed")
    passed = passed if isinstance(passed, dict) else {}
    compile_result = passed.get("compile")
    failures.extend(compile_failures(compile_result, shape_ids))
    diagnostics = compile_diagnostics(compile_result, shape_ids)

    correctness_status = passed.get("correctness")
    correctness_status = correctness_status if isinstance(correctness_status, dict) else {}
    correctness = payload.get("correctness")
    correctness = correctness if isinstance(correctness, dict) else {}
    correctness_shapes = correctness.get("shapes")
    correctness_shapes = correctness_shapes if isinstance(correctness_shapes, dict) else {}
    diagnostics.extend(candidate_exception_diagnostics(
        correctness_status, correctness_shapes, shape_ids
    ))
    max_abs = max_rel = 0.0
    for shape_id in shape_ids:
        status = correctness_status.get(shape_id)
        status = status if isinstance(status, dict) else {}
        if status.get("status") != "passed":
            failures.append(
                f"sid={shape_id}: correctness "
                + str(status.get("reason") or status.get("status") or "missing")
            )
        shape_result = correctness_shapes.get(shape_id)
        shape_result = shape_result if isinstance(shape_result, dict) else {}
        cases = shape_result.get("cases")
        for case in cases if isinstance(cases, list) else []:
            if not isinstance(case, dict):
                continue
            outputs = case.get("outputs")
            for output in outputs if isinstance(outputs, list) else []:
                if not isinstance(output, dict):
                    continue
                abs_diff = finite_number(output.get("max_elementwise_abs_diff"))
                rel_diff = finite_number(output.get("max_elementwise_rel_diff"))
                if abs_diff is not None:
                    max_abs = max(max_abs, abs_diff)
                if rel_diff is not None:
                    max_rel = max(max_rel, rel_diff)

    latency_by_shape = {}
    if require_performance:
        performance = payload.get("performance")
        performance = performance if isinstance(performance, dict) else {}
        performance_shapes = performance.get("shapes")
        performance_shapes = performance_shapes if isinstance(performance_shapes, dict) else {}
        for shape_id in shape_ids:
            shape_result = performance_shapes.get(shape_id)
            shape_result = shape_result if isinstance(shape_result, dict) else {}
            samples = shape_result.get("samples")
            sample_ms = [
                value for sample in samples if isinstance(sample, dict)
                if (value := finite_number(sample.get("end_to_end_time_ms"))) is not None
                and value > 0.0
            ] if isinstance(samples, list) else []
            if shape_result.get("error") is not None or not sample_ms:
                failures.append(
                    f"sid={shape_id}: performance "
                    + str(shape_result.get("error") or "has no valid samples")
                )
            else:
                latency_by_shape[shape_id] = statistics.median(sample_ms) * 1000.0

    latencies = [latency_by_shape[sid] for sid in shape_ids if sid in latency_by_shape]
    complete = len(latencies) == len(shape_ids)
    geomean = (math.exp(sum(math.log(value) for value in latencies) / len(latencies))
               if complete and latencies else 0.0)
    arithmetic = sum(latencies) / len(latencies) if complete and latencies else 0.0
    performance_metrics, _ = (
        metadata_performance_metrics(metadata, shape_ids, latency_by_shape)
        if require_performance and finalize_performance
        else (
            {
                "performance_objective": metadata_performance_objective(metadata),
                "performance_score": None,
                "speedup_vs_ref_mean": None,
                "speedup_vs_ref_geomean": None,
            },
            [],
        )
    )
    return {
        "all_pass": not failures,
        "failures": failures,
        "latency_us_geomean": geomean,
        "latency_us_arith_mean": arithmetic,
        "latency_us_by_shape": latency_by_shape,
        **performance_metrics,
        "max_abs_err": max_abs,
        "max_rel_err": max_rel,
        "evaluator": "atrex-gpu-gateway/run",
        "eval_id": payload.get("eval_id"),
        "actionable_diagnostics": diagnostics,
    }


def merge_optimizer_results(results: list[dict[str, Any]], shape_ids: list[str],
                            metadata: object, *, require_performance: bool = True) -> dict[str, Any]:
    latency_by_shape = {
        str(shape_id): float(latency)
        for result in results
        for shape_id, latency in (result.get("latency_us_by_shape") or {}).items()
    }
    latencies = [latency_by_shape[sid] for sid in shape_ids if sid in latency_by_shape]
    complete = not require_performance or len(latencies) == len(shape_ids)
    diagnostics = []
    seen: dict[tuple[str, str], dict[str, Any]] = {}
    for result in results:
        for diagnostic in result.get("actionable_diagnostics") or []:
            if not isinstance(diagnostic, dict):
                continue
            item = {
                "stage": str(diagnostic.get("stage") or "candidate_runtime"),
                "shape_id": str(diagnostic.get("shape_id") or ""),
                "message": str(diagnostic.get("message") or ""),
            }
            key = (item["stage"], item["message"])
            raw_shape_ids = diagnostic.get("shape_ids")
            shape_ids_for_item = (
                [str(value) for value in raw_shape_ids]
                if isinstance(raw_shape_ids, list)
                else [item["shape_id"]]
            )
            if not item["message"]:
                continue
            if key in seen:
                existing = seen[key]
                existing_shapes = existing.setdefault(
                    "shape_ids", [existing["shape_id"]]
                )
                for shape_id in shape_ids_for_item:
                    if shape_id not in existing_shapes:
                        existing_shapes.append(shape_id)
                continue
            if len(shape_ids_for_item) > 1:
                item["shape_ids"] = shape_ids_for_item
            seen[key] = item
            diagnostics.append(item)
    performance_metrics, _metadata_failures = (
        metadata_performance_metrics(metadata, shape_ids, latency_by_shape)
        if require_performance
        else (
            {
                "performance_objective": metadata_performance_objective(metadata),
                "performance_score": None,
                "speedup_vs_ref_mean": None,
                "speedup_vs_ref_geomean": None,
            },
            [],
        )
    )
    failures = [str(failure) for result in results
                for failure in (result.get("failures") or [])]
    return {
        "all_pass": complete and all(result.get("all_pass") for result in results),
        "failures": failures,
        "latency_us_geomean": (
            math.exp(sum(math.log(value) for value in latencies) / len(latencies))
            if complete and latencies else 0.0
        ),
        "latency_us_arith_mean": (
            sum(latencies) / len(latencies) if complete and latencies else 0.0
        ),
        "latency_us_by_shape": latency_by_shape,
        **performance_metrics,
        "max_abs_err": max(float(result.get("max_abs_err") or 0.0) for result in results),
        "max_rel_err": max(float(result.get("max_rel_err") or 0.0) for result in results),
        "evaluator": "atrex-gpu-gateway/run/batched",
        "eval_id": results[-1].get("eval_id"),
        "shape_batch_count": len(results),
        "actionable_diagnostics": diagnostics,
    }
