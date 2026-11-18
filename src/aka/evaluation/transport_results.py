"""Original native/SOL result conversion, batching, records and public feedback."""

from __future__ import annotations
import hashlib
import json
import math
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aka.contracts.workflow_records import measurement_document
from .objectives import latency_only_result

TEST_RESULT_PREFIX = "[test_kernel] RESULT_JSON="
ABBA_RESULT_PREFIX = "__ATREX_LONG_HORIZON_ABBA_RESULT__="
NUMERICAL_RESULT_PREFIX = "__ATREX_NUMERICAL_RESULT__="
EPISODE_EVALUATIONS_PATH = ".atrex_long_horizon/evaluations.jsonl"


def _json_object(path: Path, *, required: bool = False) -> dict[str, Any] | None:
    if not path.is_file():
        if required:
            raise ValueError(f"required typed-gateway input is missing: {path.name}")
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON in {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _request_shape_ids(request: dict[str, Any]) -> list[str]:
    shapes = request["reference"]["shapes"]

    def sort_key(shape_id: str) -> tuple[int, object]:
        return (0, int(shape_id)) if shape_id.isdigit() else (1, shape_id)

    return sorted((str(shape_id) for shape_id in shapes), key=sort_key)


def _shape_batches(shape_ids: list[str], batch_size: int) -> list[list[str]]:
    return [
        shape_ids[offset : offset + batch_size]
        for offset in range(0, len(shape_ids), batch_size)
    ]


def _shape_batch_request(
    request: dict[str, Any], shape_ids: list[str]
) -> dict[str, Any]:
    batched = dict(request)
    reference = dict(request["reference"])
    reference["shapes"] = {
        shape_id: request["reference"]["shapes"][shape_id] for shape_id in shape_ids
    }
    for field in ("metadata", "roofline"):
        payload = reference.get(field)
        if not isinstance(payload, dict):
            continue
        payload = dict(payload)
        if isinstance(payload.get("shapes"), dict):
            payload["shapes"] = {
                shape_id: payload["shapes"][shape_id]
                for shape_id in shape_ids
                if shape_id in payload["shapes"]
            }
        if field == "metadata" and "num_shapes" in payload:
            payload["num_shapes"] = len(shape_ids)
        reference[field] = payload
    batched["reference"] = reference
    return batched


def _shape_batch_reference(reference):
    """Return frozen reference file content without filesystem effects."""
    return {filename: (value if isinstance(value, str) else json.dumps(value)).encode("utf-8")
            for filename, field in (("reference.py", "reference_py"), ("input.py", "input_py"),
                ("shapes.json", "shapes"), ("metadata.json", "metadata"), ("roofline.json", "roofline"))
            if (value := reference.get(field)) is not None}


def _compile_failures(compile_result: object, shape_ids: list[str]) -> list[str]:
    """Return compile failures for aggregate and per-shape evaluator schemas.

    Older gateway/evaluator versions returned one ``{"status": ...}`` object,
    while current Atrex-Bench returns ``{shape_id: {"status": ...}}``.  Keep
    accepting the aggregate form, but require every expected shape to pass when
    the result is shape-scoped.
    """
    if not isinstance(compile_result, dict):
        compile_result = {}

    if "status" in compile_result:
        if compile_result.get("status") == "passed":
            return []
        return [
            "compile: "
            + str(
                compile_result.get("reason")
                or compile_result.get("status")
                or "did not pass"
            )
        ]

    failures: list[str] = []
    for shape_id in shape_ids:
        status = compile_result.get(shape_id)
        status = status if isinstance(status, dict) else {}
        if status.get("status") != "passed":
            failures.append(
                f"sid={shape_id}: compile "
                + str(status.get("reason") or status.get("status") or "missing")
            )
    return failures


def _bounded_actionable_diagnostic(value: object, *, limit: int = 6000) -> str:
    """Keep useful compiler/exception context without emitting unbounded tracebacks."""
    text = str(value or "").replace("\x00", "").strip()
    if len(text) <= limit:
        return text
    head = min(1000, limit // 3)
    tail = limit - head - len("\n... diagnostic truncated ...\n")
    return text[:head] + "\n... diagnostic truncated ...\n" + text[-tail:]


def _looks_like_candidate_exception(value: object) -> bool:
    """Recognize build/import/driver failures without exposing arbitrary case errors."""
    text = str(value or "")
    lowered = text.lower()
    return any(
        marker in lowered
        for marker in (
            "compilation error",
            "compileerror",
            "cuda_error_",
            "cudaerror",
            "culaunchkernel",
            "cumodule",
            "invalid context",
            "kernelparams",
            "modulenotfounderror:",
            "importerror:",
            "nvrtc",
            "syntaxerror:",
            "undefined symbol",
        )
    )


def _compile_diagnostics(
    compile_result: object, shape_ids: list[str]
) -> list[dict[str, str]]:
    """Extract compile/import failures that are safe and useful in generalized mode."""
    if not isinstance(compile_result, dict):
        return []
    if "status" in compile_result:
        if compile_result.get("status") == "passed":
            return []
        message = _bounded_actionable_diagnostic(
            compile_result.get("reason") or compile_result.get("status")
        )
        return (
            [{"stage": "compile_import", "shape_id": "", "message": message}]
            if message
            else []
        )

    diagnostics: list[dict[str, str]] = []
    seen_messages: set[str] = set()
    for shape_id in shape_ids:
        status = compile_result.get(shape_id)
        status = status if isinstance(status, dict) else {}
        if status.get("status") == "passed":
            continue
        message = _bounded_actionable_diagnostic(
            status.get("reason") or status.get("status") or "missing"
        )
        if message in seen_messages:
            continue
        seen_messages.add(message)
        diagnostics.append(
            {
                "stage": "compile_import",
                "shape_id": shape_id,
                "message": message,
            }
        )
    return diagnostics


def _candidate_exception_diagnostics(
    correctness_status: dict[str, Any],
    correctness_shapes: dict[str, Any],
    shape_ids: list[str],
) -> list[dict[str, str]]:
    """Surface traceback-backed candidate failures without exposing numeric case data."""
    diagnostics: list[dict[str, str]] = []
    seen: set[str] = set()
    for shape_id in shape_ids:
        status = correctness_status.get(shape_id)
        status = status if isinstance(status, dict) else {}
        reasons: list[object] = [status.get("reason")]
        shape_result = correctness_shapes.get(shape_id)
        shape_result = shape_result if isinstance(shape_result, dict) else {}
        cases = shape_result.get("cases")
        for case in cases if isinstance(cases, list) else []:
            if isinstance(case, dict):
                reasons.append(case.get("error"))
        for reason in reasons:
            if not _looks_like_candidate_exception(reason):
                continue
            message = _bounded_actionable_diagnostic(reason)
            if not message or message in seen:
                continue
            seen.add(message)
            diagnostics.append(
                {
                    "stage": "candidate_runtime",
                    "shape_id": shape_id,
                    "message": message,
                }
            )
    return diagnostics


def _evaluation_failure(payload: dict[str, Any]) -> dict[str, str] | None:
    """Classify evaluator startup failures that precede all candidate imports."""
    error = str(payload.get("error") or "")
    if (
        "import torch" in error
        and "typing._ClassVar" in error
        and "module 'typing' has no attribute '_ClassVar'" in error
    ):
        return {
            "category": "infrastructure",
            "code": "worker_bootstrap_failed",
            "message": (
                "GPU worker bootstrap failed before candidate import: the worker "
                "Python typing/dataclasses runtime is incompatible with torch"
            ),
        }
    return None


def _optimizer_result_from_eval(
    payload: dict[str, Any],
    shape_ids: list[str],
    metadata: object,
    *,
    require_performance: bool = True,
    hardware: str = "",
    finalize_performance: bool = True,
) -> dict[str, Any]:
    """Convert the typed gateway's Atrex-Bench result to optimizer RESULT_JSON."""
    failures: list[str] = []
    evaluation_failure = _evaluation_failure(payload)
    if payload.get("error"):
        failures.append(
            evaluation_failure["message"]
            if evaluation_failure is not None
            else "evaluation: " + str(payload["error"])
        )
    passed = payload.get("passed")
    passed = passed if isinstance(passed, dict) else {}
    compile_result = passed.get("compile")
    failures.extend(_compile_failures(compile_result, shape_ids))
    actionable_diagnostics = _compile_diagnostics(compile_result, shape_ids)
    if evaluation_failure is not None:
        actionable_diagnostics.insert(
            0,
            {
                "stage": "worker_bootstrap",
                "shape_id": "",
                "message": evaluation_failure["message"],
            },
        )

    correctness_status = passed.get("correctness")
    correctness_status = (
        correctness_status if isinstance(correctness_status, dict) else {}
    )
    correctness = payload.get("correctness")
    correctness = correctness if isinstance(correctness, dict) else {}
    correctness_shapes = correctness.get("shapes")
    correctness_shapes = (
        correctness_shapes if isinstance(correctness_shapes, dict) else {}
    )
    actionable_diagnostics.extend(
        _candidate_exception_diagnostics(
            correctness_status, correctness_shapes, shape_ids
        )
    )
    max_abs = 0.0
    max_rel = 0.0
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
                abs_diff = _finite_number(output.get("max_elementwise_abs_diff"))
                rel_diff = _finite_number(output.get("max_elementwise_rel_diff"))
                if abs_diff is not None:
                    max_abs = max(max_abs, abs_diff)
                if rel_diff is not None:
                    max_rel = max(max_rel, rel_diff)

    latency_by_shape: dict[str, float] = {}
    if require_performance:
        performance_status = passed.get("performance")
        performance_status = performance_status if isinstance(performance_status, dict) else {}
        performance = payload.get("performance")
        performance = performance if isinstance(performance, dict) else {}
        performance_shapes = performance.get("shapes")
        performance_shapes = (
            performance_shapes if isinstance(performance_shapes, dict) else {}
        )
        for shape_id in shape_ids:
            status = performance_status.get(shape_id)
            status = status if isinstance(status, dict) else {}
            shape_result = performance_shapes.get(shape_id)
            shape_result = shape_result if isinstance(shape_result, dict) else {}
            sample_ms: list[float] = []
            samples = shape_result.get("samples")
            for sample in samples if isinstance(samples, list) else []:
                if not isinstance(sample, dict):
                    continue
                value = _finite_number(sample.get("end_to_end_time_ms"))
                if value is not None and value > 0.0:
                    sample_ms.append(value)
            if status.get("status") == "failed" or shape_result.get("error") is not None or not sample_ms:
                failures.append(
                    f"sid={shape_id}: performance "
                    + str(shape_result.get("error") or status.get("reason")
                          or ("failed" if status.get("status") == "failed" else "has no valid samples"))
                )
                continue
            latency_by_shape[shape_id] = statistics.median(sample_ms) * 1000.0

    latencies = [
        latency_by_shape[shape_id]
        for shape_id in shape_ids
        if shape_id in latency_by_shape
    ]
    complete = len(latencies) == len(shape_ids)
    geomean = (
        math.exp(sum(math.log(value) for value in latencies) / len(latencies))
        if complete and latencies
        else 0.0
    )
    arithmetic = sum(latencies) / len(latencies) if complete and latencies else 0.0
    result = {
        "all_pass": not failures,
        "failures": failures,
        "latency_us_geomean": geomean,
        "latency_us_arith_mean": arithmetic,
        "latency_us_by_shape": latency_by_shape,
        "max_abs_err": max_abs,
        "max_rel_err": max_rel,
        "evaluator": "atrex-gpu-gateway/run",
        "eval_id": payload.get("eval_id"),
        "actionable_diagnostics": actionable_diagnostics[:8],
    }
    if evaluation_failure is not None:
        result["evaluation_failure"] = evaluation_failure
    return result


def _merge_optimizer_results(
    results: list[dict[str, Any]],
    shape_ids: list[str],
    metadata: object,
    *,
    require_performance: bool = True,
    hardware: str = "",
) -> dict[str, Any]:
    evaluation_failure = next(
        (
            result["evaluation_failure"]
            for result in results
            if isinstance(result.get("evaluation_failure"), dict)
        ),
        None,
    )
    latency_by_shape = {
        str(shape_id): float(latency)
        for result in results
        for shape_id, latency in (result.get("latency_us_by_shape") or {}).items()
        if type(latency) in (int, float) and math.isfinite(latency) and latency > 0
    }
    latencies = [
        latency_by_shape[shape_id]
        for shape_id in shape_ids
        if shape_id in latency_by_shape
    ]
    complete = not require_performance or bool(shape_ids) and len(latencies) == len(shape_ids)
    actionable_diagnostics: list[dict[str, str]] = []
    seen_diagnostics: set[tuple[str, str]] = set()
    for result in results:
        if len(actionable_diagnostics) >= 8:
            break
        for diagnostic in result.get("actionable_diagnostics") or []:
            if not isinstance(diagnostic, dict):
                continue
            normalized = {
                "stage": str(diagnostic.get("stage") or "candidate_runtime"),
                "shape_id": str(diagnostic.get("shape_id") or ""),
                "message": str(diagnostic.get("message") or ""),
            }
            key = (
                normalized["stage"],
                normalized["message"],
            )
            if not normalized["message"] or key in seen_diagnostics:
                continue
            seen_diagnostics.add(key)
            actionable_diagnostics.append(normalized)
            if len(actionable_diagnostics) >= 8:
                break

    failures = [
        str(failure) for result in results for failure in (result.get("failures") or [])
    ]

    merged = {
        "all_pass": complete
        and all(result.get("all_pass") for result in results),
        "failures": failures,
        "latency_us_geomean": (
            math.exp(sum(math.log(value) for value in latencies) / len(latencies))
            if complete and latencies
            else 0.0
        ),
        "latency_us_arith_mean": (
            sum(latencies) / len(latencies) if complete and latencies else 0.0
        ),
        "latency_us_by_shape": latency_by_shape,
        "max_abs_err": max(
            float(result.get("max_abs_err") or 0.0) for result in results
        ),
        "max_rel_err": max(
            float(result.get("max_rel_err") or 0.0) for result in results
        ),
        "evaluator": "atrex-gpu-gateway/run/batched",
        "eval_id": results[-1].get("eval_id"),
        "shape_batch_count": len(results),
        "actionable_diagnostics": actionable_diagnostics,
    }
    if evaluation_failure is not None:
        merged["evaluation_failure"] = evaluation_failure
    return merged



def _positive_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if number > 0.0 and math.isfinite(number) else None


def _hydrate_result_lines(workspace: Path, stdout: str, *, baseline=None) -> str:
    """Return ordinary evaluator output with latency-only metrics."""
    normalized: list[str] = []
    for line in stdout.splitlines():
        if not line.startswith(TEST_RESULT_PREFIX):
            normalized.append(line)
            continue
        try:
            result = json.loads(line[len(TEST_RESULT_PREFIX) :])
        except json.JSONDecodeError:
            normalized.append(line)
            continue
        if not isinstance(result, dict):
            normalized.append(line)
            continue
        normalized.append(
            TEST_RESULT_PREFIX
            + json.dumps(
                latency_only_result(result),
                ensure_ascii=False,
                allow_nan=False,
            )
        )
    return "\n".join(normalized)


def _public_numerical_result_line(line: str) -> str:
    """Expose coverage/aggregate errors without private shapes or raw diagnostics."""
    payload = json.loads(line[len(NUMERICAL_RESULT_PREFIX) :])
    public = {key: payload.get(key) for key in ("schema_version", "all_pass")}
    public["runs"] = []
    for row in payload.get("runs", []):
        item = {
            key: row.get(key)
            for key in (
                "case_id",
                "passed",
                "exit_code",
                "expected_probes",
                "observed_probes",
                "failed_probes",
                "shape_count",
                "seeds",
                "world_size",
                "selection_digest",
            )
        }
        # Preserve the driver's failure receipt across the privacy boundary.
        # Raw evaluator/input exceptions can contain hidden workload values.
        item["input_error"] = (
            "input generation or evaluator execution incomplete; private diagnostics withheld"
            if row.get("input_error")
            else ""
        )
        result = row.get("result")
        item["result"] = (
            {
                key: result.get(key)
                for key in ("all_pass", "max_abs_err", "max_rel_err", "evaluator")
            }
            if isinstance(result, dict)
            else None
        )
        if isinstance(result, dict):
            item["result"]["nonfinite_outputs"] = [
                role
                for role in ("reference", "candidate", "unknown")
                if role in result.get("nonfinite_outputs", [])
            ]
            metrics = result.get("numerical_metrics") or {}
            item["result"]["numerical_metrics"] = {
                key: value
                for key, value in metrics.items()
                if key
                in {
                    "relative_l2",
                    "max_row_relative_l2",
                    "max_elementwise_abs_diff",
                    "max_elementwise_rel_diff",
                    "max_abs_err",
                    "max_rel_err",
                }
                and (
                    value is None
                    or type(value) in {int, float}
                    and math.isfinite(value)
                )
            }
        public["runs"].append(item)
    if payload.get("error"):
        public["error"] = "numerical evaluator failed; private diagnostics withheld"
    return NUMERICAL_RESULT_PREFIX + json.dumps(public, allow_nan=False)


def _hydrate_abba_result_lines(workspace: Path, stdout: str, *, baseline=None) -> str:
    """Normalize candidate-only ABBA run results for long-lived supervisors."""
    normalized: list[str] = []
    for line in stdout.splitlines():
        if not line.startswith(ABBA_RESULT_PREFIX):
            normalized.append(line)
            continue
        try:
            payload = json.loads(line[len(ABBA_RESULT_PREFIX) :])
        except json.JSONDecodeError:
            normalized.append(line)
            continue
        if not isinstance(payload, dict) or not isinstance(payload.get("runs"), list):
            normalized.append(line)
            continue
        changed = False
        runs: list[object] = []
        for row in payload["runs"]:
            if not isinstance(row, dict) or not isinstance(row.get("result"), dict):
                runs.append(row)
                continue
            result = latency_only_result(row["result"])
            if result is row["result"]:
                runs.append(row)
                continue
            updated = dict(row)
            updated["result"] = result
            runs.append(updated)
            changed = True
        if not changed:
            normalized.append(line)
            continue
        updated_payload = dict(payload)
        updated_payload["runs"] = runs
        normalized.append(
            ABBA_RESULT_PREFIX
            + json.dumps(updated_payload, ensure_ascii=False, allow_nan=False)
        )
    return "\n".join(normalized)
