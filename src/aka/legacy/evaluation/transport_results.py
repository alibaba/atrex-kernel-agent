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

from aka.legacy.task.problem.inputs import validate_performance_baseline

PERFORMANCE_OBJECTIVE = "shape_speedup_arithmetic_mean"
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


def _shape_batch_reference(reference: dict[str, Any], destination: Path) -> None:
    destination.mkdir()
    for filename, field in (
        ("reference.py", "reference_py"),
        ("input.py", "input_py"),
        ("shapes.json", "shapes"),
        ("metadata.json", "metadata"),
        ("roofline.json", "roofline"),
    ):
        value = reference.get(field)
        if value is not None:
            (destination / filename).write_text(
                value if isinstance(value, str) else json.dumps(value),
                encoding="utf-8",
            )


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


def _metadata_shape_latency_us(
    metadata: object, shape_id: str, hardware: str = ""
) -> float | None:
    """Read one authoritative production latency from Atrex-Bench metadata."""
    try:
        return validate_performance_baseline(
            {shape_id: None}, metadata, hardware
        )[shape_id]
    except ValueError:
        return None


def _metadata_speedup_mean(
    metadata: object,
    shape_ids: list[str],
    latency_by_shape: dict[str, float],
    hardware: str = "",
) -> tuple[float | None, list[str]]:
    if any(shape_id not in latency_by_shape for shape_id in shape_ids):
        return None, []
    try:
        baselines = validate_performance_baseline(
            {shape_id: None for shape_id in shape_ids}, metadata, hardware
        )
    except ValueError as exc:
        return None, [str(exc)]
    speedups: list[float] = []
    for shape_id in shape_ids:
        speedups.append(baselines[shape_id] / latency_by_shape[shape_id])
    if len(speedups) != len(shape_ids) or not speedups:
        return None, []
    return sum(speedups) / len(speedups), []


def _optimizer_result_from_eval(
    payload: dict[str, Any],
    shape_ids: list[str],
    metadata: object,
    *,
    require_performance: bool = True,
    hardware: str = "",
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
        performance = payload.get("performance")
        performance = performance if isinstance(performance, dict) else {}
        performance_shapes = performance.get("shapes")
        performance_shapes = (
            performance_shapes if isinstance(performance_shapes, dict) else {}
        )
        for shape_id in shape_ids:
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
            if shape_result.get("error") is not None or not sample_ms:
                failures.append(
                    f"sid={shape_id}: performance "
                    + str(shape_result.get("error") or "has no valid samples")
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
    speedup_mean, metadata_failures = (
        _metadata_speedup_mean(metadata, shape_ids, latency_by_shape, hardware)
        if require_performance
        else (None, [])
    )
    failures.extend(metadata_failures)
    result = {
        "all_pass": not failures,
        "failures": failures,
        "latency_us_geomean": geomean,
        "latency_us_arith_mean": arithmetic,
        "latency_us_by_shape": latency_by_shape,
        "speedup_vs_ref_mean": speedup_mean,
        "speedup_vs_ref_geomean": None,
        "performance_score": speedup_mean,
        "performance_objective": PERFORMANCE_OBJECTIVE,
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
    }
    latencies = [
        latency_by_shape[shape_id]
        for shape_id in shape_ids
        if shape_id in latency_by_shape
    ]
    complete = not require_performance or len(latencies) == len(shape_ids)
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

    speedup_mean, metadata_failures = (
        _metadata_speedup_mean(metadata, shape_ids, latency_by_shape, hardware)
        if require_performance
        else (None, [])
    )
    failures = [
        str(failure) for result in results for failure in (result.get("failures") or [])
    ]
    for failure in metadata_failures:
        if failure not in failures:
            failures.append(failure)

    merged = {
        "all_pass": complete
        and (not require_performance or speedup_mean is not None)
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
        "speedup_vs_ref_mean": speedup_mean,
        "speedup_vs_ref_geomean": None,
        "performance_score": speedup_mean,
        "performance_objective": PERFORMANCE_OBJECTIVE,
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


def _record_episode_evaluation(
    workspace: Path,
    result: dict[str, Any],
    *,
    gateway_kind: str,
    job_id: object = None,
) -> None:
    """Persist exact optimizer-facing results for terminal memory construction.

    Long-horizon workers use ``--no-memory`` because canonical memory belongs to the
    supervisor.  Keep their complete evaluator results in excluded episode runtime
    state so a pivot or interruption can still record this round's real per-shape
    performance without parsing model-authored journal prose.
    """
    runtime = workspace / ".atrex_long_horizon"
    if not (runtime / "journal.json").is_file():
        return
    try:
        kernel_sha256 = hashlib.sha256(
            (workspace / "kernel.py").read_bytes()
        ).hexdigest()
    except OSError:
        kernel_sha256 = None
    manifest = workspace / "solution.json"
    solution_sha256 = (
        hashlib.sha256(manifest.read_bytes()).hexdigest() if manifest.exists() else None
    )
    payload = {
        "schema_version": 3,
        "solution_sha256": solution_sha256,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gateway_kind": gateway_kind,
        "job_id": str(job_id) if job_id else None,
        "kernel_sha256": kernel_sha256,
        "result": result,
    }
    path = workspace / EPISODE_EVALUATIONS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    line = (json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(descriptor, line)
    finally:
        os.close(descriptor)


def _record_result_lines(workspace: Path, stdout: str, *, gateway_kind: str) -> None:
    """Record the last ordinary RESULT_JSON emitted by a dev evaluator command."""
    for line in reversed(stdout.splitlines()):
        if not line.startswith(TEST_RESULT_PREFIX):
            continue
        try:
            result = json.loads(line[len(TEST_RESULT_PREFIX) :])
        except json.JSONDecodeError:
            return
        if isinstance(result, dict):
            _record_episode_evaluation(workspace, result, gateway_kind=gateway_kind)
        return


def _positive_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if number > 0.0 and math.isfinite(number) else None


def _with_workspace_performance_score(
    workspace: Path, result: dict[str, Any]
) -> dict[str, Any]:
    """Normalize legacy per-shape results to the shared arithmetic-mean score."""
    if result.get("performance_objective") == PERFORMANCE_OBJECTIVE:
        return result
    candidate_by_shape = result.get("latency_us_by_shape")
    if not isinstance(candidate_by_shape, dict) or not candidate_by_shape:
        return result
    try:
        baseline = _json_object(workspace / "memory" / "v0.json") or {}
    except ValueError:
        return result
    performance = baseline.get("performance")
    performance = performance if isinstance(performance, dict) else {}
    baseline_by_shape = performance.get("latency_us_by_shape")
    if not isinstance(baseline_by_shape, dict) or set(candidate_by_shape) != set(
        baseline_by_shape
    ):
        return result
    speedups: list[float] = []
    for shape_id, raw_candidate in candidate_by_shape.items():
        candidate = _positive_number(raw_candidate)
        reference = _positive_number(baseline_by_shape.get(shape_id))
        if candidate is None or reference is None:
            return result
        speedups.append(reference / candidate)
    if not speedups:
        return result
    score = sum(speedups) / len(speedups)
    hydrated = dict(result)
    hydrated["speedup_vs_ref_mean"] = score
    hydrated["speedup_vs_ref_geomean"] = None
    hydrated["performance_score"] = score
    hydrated["performance_objective"] = PERFORMANCE_OBJECTIVE
    return hydrated


def _hydrate_result_lines(workspace: Path, stdout: str) -> str:
    """Return ordinary dev evaluator output with the shared score contract."""
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
                _with_workspace_performance_score(workspace, result),
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


def _hydrate_abba_result_lines(workspace: Path, stdout: str) -> str:
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
            result = _with_workspace_performance_score(workspace, row["result"])
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
