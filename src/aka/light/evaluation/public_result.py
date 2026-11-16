"""Canonical Agent results; executor streams remain private compatibility evidence."""
from __future__ import annotations

from copy import deepcopy
import json
import math
import statistics

from aka.light.evaluation.response_policy import known_gateway_error
from aka.light.evaluation.result_json import (
    LEGACY_TEST_RESULT_PREFIX,
    parse_evaluation_result_line,
)

PREFIXES = {
    "evaluate": LEGACY_TEST_RESULT_PREFIX,
    "same_allocation_abba": "[sandbox] ABBA_JSON=",
    "profile": "[sandbox] PROFILE_JSON=",
    "check": "[sandbox] CHECK_JSON=",
    "disassemble": "[sandbox] DISASSEMBLE_JSON=",
    "env": "[sandbox] ENV_JSON=",
}
ERROR_PREFIXES = ("[sandbox] SOURCE_ERROR_JSON=", "[sandbox] GATEWAY_ERROR_JSON=")
PRIVATE_IDENTITIES = ("gateway_record_id", "kernel_id", "baseline_kernel_id")
PUBLIC_IDENTITIES = (
    "gateway_record_id",
    "kernel_commit_id",
    "candidate_kernel_commit_id",
    "baseline_kernel_commit_id",
)
IDENTITIES = (*PRIVATE_IDENTITIES, *PUBLIC_IDENTITIES)
RECEIPT_TEXT_PREVIEW_BYTES = 2048
RECEIPT_ACTIONABLE_DIAGNOSTIC_LIMIT = 4
RECEIPT_DIAGNOSTIC_SHAPE_LIMIT = 16
RECEIPT_VALIDATION_ERROR_LIMIT = 16
RECEIPT_VALIDATION_LOCATION_LIMIT = 16
_RECEIPT_RESULT_FIELDS = (
    "correct",
    "passed",
    "exit_code",
    "mode",
    "input_scope",
    "latency_us_geomean",
    "latency_us_arith_mean",
    "performance_score",
    "speedup",
    "speedup_vs_ref_geomean",
    "improvement_pct",
    "measured_shape_count",
    "outcome",
    "reason",
    "http_status",
)
_RECEIPT_ERROR_FIELDS = (
    "code",
    "reason",
    "error_class",
    "message",
    "timeout_s",
    "stage",
    "http_status",
    "outcome",
    "candidate_repairable",
)


def _receipt_validation_error(value: object) -> dict | None:
    """Project one bounded request-validation diagnostic without its input."""
    if isinstance(value, str):
        return {"message": _receipt_string(value, limit=1024)}
    if not isinstance(value, dict):
        return None
    summary = {}
    for key in ("type", "code", "msg", "message", "field", "path"):
        item = value.get(key)
        if isinstance(item, str):
            summary[key] = _receipt_string(item, limit=1024)
        elif type(item) in {int, float} and math.isfinite(item):
            summary[key] = item
    location = value.get("loc")
    if isinstance(location, (list, tuple)):
        summary["loc"] = [
            _receipt_string(str(item), limit=256)
            for item in location[:RECEIPT_VALIDATION_LOCATION_LIMIT]
            if isinstance(item, (str, int)) and not isinstance(item, bool)
        ]
        if len(location) > RECEIPT_VALIDATION_LOCATION_LIMIT:
            summary["loc_truncated"] = True
    return summary or None


def _receipt_error_details(value: object) -> dict | None:
    if not isinstance(value, dict):
        return None
    errors = value.get("validation_errors")
    if not isinstance(errors, list):
        return None
    projected = []
    count = 0
    for error in errors:
        item = _receipt_validation_error(error)
        if item is None:
            continue
        count += 1
        if len(projected) < RECEIPT_VALIDATION_ERROR_LIMIT:
            projected.append(item)
    if not projected:
        return None
    return {
        "validation_errors": projected,
        "validation_errors_count": count,
        "validation_errors_truncated": count > len(projected),
    }


def _receipt_text(value: str, *, limit: int = RECEIPT_TEXT_PREVIEW_BYTES) -> dict:
    raw = value.encode("utf-8")
    if len(raw) <= limit:
        preview = value
        truncated = False
    else:
        marker = b"\n... [receipt preview omitted middle bytes] ...\n"
        remaining = max(0, limit - len(marker))
        left = remaining // 2
        right = remaining - left
        preview = (
            raw[:left] + marker + (raw[-right:] if right else b"")
        ).decode("utf-8", errors="replace")
        truncated = True
    return {"bytes": len(raw), "preview": preview, "truncated": truncated}


def _receipt_string(
    value: str, *, limit: int = RECEIPT_TEXT_PREVIEW_BYTES
) -> str:
    preview = _receipt_text(value, limit=limit)
    return preview["preview"]


def _receipt_error(value: object) -> dict | None:
    if isinstance(value, str):
        return {"message": _receipt_string(value)}
    if not isinstance(value, dict):
        return None
    summary = {}
    for key in _RECEIPT_ERROR_FIELDS:
        item = value.get(key)
        if isinstance(item, str):
            summary[key] = _receipt_string(item)
        elif type(item) is bool or item is None:
            if key in value:
                summary[key] = item
        elif type(item) in {int, float} and math.isfinite(item):
            summary[key] = item
    details = _receipt_error_details(value.get("details"))
    if details is not None:
        summary["details"] = details
    return summary or {"message": "Full error details are available in the saved Gateway Record."}


def _receipt_metrics(value: object) -> dict:
    if not isinstance(value, dict):
        return {}
    summary = {}
    for key in _RECEIPT_RESULT_FIELDS:
        item = value.get(key)
        if isinstance(item, str):
            summary[key] = _receipt_string(item)
        elif type(item) is bool or item is None:
            if key in value:
                summary[key] = item
        elif type(item) in {int, float} and math.isfinite(item):
            summary[key] = item
    correctness = value.get("correctness")
    if isinstance(correctness, dict):
        summary["correctness"] = {
            key: item
            for key, item in correctness.items()
            if key in {"status", "rel_err", "max_abs_err", "max_rel_err"}
            and (
                isinstance(item, str)
                or type(item) is bool
                or item is None
                or (type(item) in {int, float} and math.isfinite(item))
            )
        }
    return summary


def _receipt_actionable_diagnostic(value: object) -> dict | None:
    """Return one bounded diagnostic while preserving the actionable tail."""
    if not isinstance(value, dict):
        return None
    summary = {}
    for key in ("stage", "shape_id"):
        item = value.get(key)
        if isinstance(item, str):
            summary[key] = _receipt_text(item, limit=256)["preview"]
    shape_ids = value.get("shape_ids")
    if isinstance(shape_ids, list):
        summary["shape_ids"] = [
            _receipt_text(str(item), limit=256)["preview"]
            for item in shape_ids[:RECEIPT_DIAGNOSTIC_SHAPE_LIMIT]
        ]
        if len(shape_ids) > RECEIPT_DIAGNOSTIC_SHAPE_LIMIT:
            summary["shape_ids_truncated"] = True
    message = value.get("message")
    if isinstance(message, str):
        preview = _receipt_text(message)
        summary["message"] = preview["preview"]
        summary["message_bytes"] = preview["bytes"]
        summary["message_truncated"] = preview["truncated"]
    return summary or None


def compact_receipt(agent_result: dict) -> dict:
    """Return bounded terminal feedback while the full result stays in records/."""
    result = agent_result.get("result")
    result = result if isinstance(result, dict) else {}
    receipt = {
        key: deepcopy(agent_result[key])
        for key in (
            "operation",
            "status",
            "gateway_record_id",
            "kernel_commit_id",
            "candidate_kernel_commit_id",
            "baseline_kernel_commit_id",
        )
        if key in agent_result
    }
    record_id = receipt.get("gateway_record_id")
    if isinstance(record_id, str):
        receipt["record_path"] = f"records/{record_id}.json"
    research_next_action = agent_result.get("research_next_action")
    if isinstance(research_next_action, dict):
        operation = research_next_action.get("operation")
        instruction = research_next_action.get("instruction")
        gateway_record_ids = research_next_action.get("gateway_record_ids")
        direction_id = research_next_action.get("direction_id")
        projected_action: dict[str, object] = {}
        if isinstance(operation, str):
            projected_action["operation"] = _receipt_string(operation, limit=128)
        if isinstance(instruction, str):
            projected_action["instruction"] = _receipt_string(
                instruction, limit=1024
            )
        if isinstance(gateway_record_ids, list):
            projected_action["gateway_record_ids"] = [
                str(value)[:128]
                for value in gateway_record_ids[:8]
                if isinstance(value, str)
            ]
        if isinstance(direction_id, str):
            projected_action["direction_id"] = direction_id[:128]
        if projected_action:
            receipt["research_next_action"] = projected_action
    summary = _receipt_metrics(result)
    for side in ("baseline", "candidate"):
        metrics = _receipt_metrics(result.get(side))
        if metrics:
            summary[side] = metrics
    error = _receipt_error(result.get("error"))
    if error is not None:
        summary["error"] = error
    errors = result.get("errors")
    if isinstance(errors, list) and errors:
        summary["error_count"] = len(errors)
        if "error" not in summary:
            summary["error"] = _receipt_error(errors[0]) or {
                "message": "Full error details are available in the saved Gateway Record."
            }
    for key in ("next_action", "scientific_next_check"):
        value = result.get(key)
        if isinstance(value, str):
            summary[key] = _receipt_string(value)
    for stream in ("stdout", "stderr"):
        value = result.get(stream)
        if isinstance(value, str):
            preview = _receipt_text(value)
            summary[f"{stream}_bytes"] = preview["bytes"]
            summary[f"{stream}_preview"] = preview["preview"]
            summary[f"{stream}_truncated"] = preview["truncated"]
            if "record_path" in receipt:
                receipt[f"{stream}_location"] = {
                    "path": receipt["record_path"],
                    "field": f"result.{stream}",
                }
    diagnostics = result.get("actionable_diagnostics")
    if isinstance(diagnostics, list):
        bounded_diagnostics = []
        diagnostic_count = 0
        for diagnostic in diagnostics:
            projected = _receipt_actionable_diagnostic(diagnostic)
            if projected is None:
                continue
            diagnostic_count += 1
            if len(bounded_diagnostics) < RECEIPT_ACTIONABLE_DIAGNOSTIC_LIMIT:
                bounded_diagnostics.append(projected)
        if bounded_diagnostics:
            receipt["actionable_diagnostics"] = bounded_diagnostics
            receipt["actionable_diagnostics_count"] = diagnostic_count
            receipt["actionable_diagnostics_truncated"] = (
                diagnostic_count > len(bounded_diagnostics)
            )
            if "record_path" in receipt:
                receipt["actionable_diagnostics_location"] = {
                    "path": receipt["record_path"],
                    "field": "result.actionable_diagnostics",
                }
    receipt["result"] = summary
    receipt["result_fields"] = sorted(str(key) for key in result)[:128]
    receipt["details"] = (
        f"Full result: {receipt['record_path']}. Read or query that JSON file directly and "
        "select only the fields or Shape needed for the next decision. Complete stream and "
        "actionable diagnostic locations are listed explicitly when present."
        if "record_path" in receipt
        else "No Agent-visible Gateway Record path is available for this response."
    )
    return receipt


def bound_diagnostic_response(response: dict) -> dict:
    """Preserve complete redacted diagnostics at the Agent boundary.

    Redaction and schema projection happen before this function.  A serialized
    length limit is intentionally not applied: compiler diagnostics often put
    the only actionable root cause after a large repeated template trace.
    """
    return response


def finite_number(value):
    return float(value) if type(value) in (int, float) and math.isfinite(value) else None


def correctness_summary(value, *, passed: bool) -> dict:
    """Runtime's worst-case scalar summary, never case inputs or raw outputs."""
    aliases = {"rel_err": {"rel_err", "relative_l2"},
               "max_abs_err": {"max_abs_err", "max_elementwise_abs_diff"},
               "max_rel_err": {"max_rel_err", "max_elementwise_rel_diff"}}
    metrics = dict.fromkeys(aliases)
    pending = [value.get("correctness", value) if isinstance(value, dict) else value]
    while pending:
        item = pending.pop()
        if isinstance(item, dict):
            if "correctness" in item:
                pending.append(item["correctness"])
                continue
            for key, child in item.items():
                number = finite_number(child)
                if number is not None and number >= 0:
                    for name, names in aliases.items():
                        if key in names:
                            metrics[name] = number if metrics[name] is None else max(metrics[name], number)
                if isinstance(child, (dict, list, tuple)):
                    pending.append(child)
        elif isinstance(item, (list, tuple)):
            pending.extend(item)
    return {"status": "PASS" if passed else "FAIL", **metrics}


def input_scope(options: dict) -> str:
    command = options.get("command") or []
    subset = any(isinstance(item, str) and item.split("=", 1)[0] == "--shape-id" for item in command)
    return "custom" if (options.get("evaluation_input_path") or options.get("evaluation_shapes_path")
                        or options.get("shape_id") or subset) else "contract"


def _metrics(raw: dict, *, verdict, mode="full", known=True) -> dict:
    value = deepcopy(raw)
    for key in (*IDENTITIES, "all_pass", "max_abs_err", "max_rel_err", "rel_err"):
        value.pop(key, None)
    value["correct"] = verdict if known and isinstance(verdict, bool) else None
    value["correctness"] = (correctness_summary(raw, passed=verdict)
                            if known and isinstance(verdict, bool) else None)
    shapes = raw.get("latency_us_by_shape")
    shapes = shapes if isinstance(shapes, dict) and mode != "correctness_only" else {}
    value["latency_us_by_shape"] = shapes
    numbers = [number for item in shapes.values() if (number := finite_number(item)) is not None and number > 0]
    latency = finite_number(raw.get("latency_us_geomean"))
    value["latency_us_geomean"] = latency if latency is not None and latency > 0 and mode != "correctness_only" else None
    value["latency_us_arith_mean"] = statistics.fmean(numbers) if numbers else None
    return value


def canonical_result(response: dict, request: dict, *, record_id=None, kernels=None) -> dict:
    """Normalize only already-projected evidence. Never read private request files.

    Old records are converted in memory; their immutable streams/digests are not
    rewritten. New records seal this object alongside the controller's streams.
    """
    if isinstance(response.get("agent_result"), dict):
        result = deepcopy(response["agent_result"])
        # Historical immutable records used ``kernel_id`` for a value that has
        # always been the full Agent Git commit.  Keep that storage compatible,
        # but expose the precise name at the Agent boundary.
        if "kernel_id" in result:
            result["kernel_commit_id"] = result.pop("kernel_id")
        if "baseline_kernel_id" in result:
            result["baseline_kernel_commit_id"] = result.pop("baseline_kernel_id")
        if "baseline_kernel_commit_id" in result and "kernel_commit_id" in result:
            result["candidate_kernel_commit_id"] = result.pop("kernel_commit_id")
        return result
    operation = request["operation"]
    options = request.get("options", {})
    prefix = PREFIXES.get(operation)
    sample, errors, notes = None, [], []
    for line in response.get("stdout", "").splitlines():
        if operation == "evaluate":
            direct = parse_evaluation_result_line(line)
            if direct is not None:
                sample = direct
                continue
        marker = next((p for p in (*ERROR_PREFIXES, prefix) if p and line.startswith(p)), None)
        if marker:
            try:
                value = json.loads(line[len(marker):])
                if not isinstance(value, dict):
                    raise ValueError("invalid result")
            except (ValueError, TypeError):
                notes.append(line)
                continue
            if marker in ERROR_PREFIXES:
                errors.append(value)
            else:
                sample = value
        elif not line.startswith("[sandbox] RECORD_JSON="):
            notes.append(line)
    has_sample = sample is not None
    sample = sample or {}
    identity = {}
    if "gateway_record_id" in sample:
        identity["gateway_record_id"] = sample["gateway_record_id"]
    candidate_commit = sample.get(
        "candidate_kernel_commit_id",
        sample.get("kernel_commit_id", sample.get("kernel_id")),
    )
    baseline_commit = sample.get(
        "baseline_kernel_commit_id", sample.get("baseline_kernel_id")
    )
    if candidate_commit is not None:
        identity[
            "candidate_kernel_commit_id"
            if operation == "same_allocation_abba"
            else "kernel_commit_id"
        ] = candidate_commit
    if baseline_commit is not None:
        identity["baseline_kernel_commit_id"] = baseline_commit
    if record_id:
        identity["gateway_record_id"] = record_id
    for side, key in (
        (
            "candidate",
            "candidate_kernel_commit_id"
            if operation == "same_allocation_abba"
            else "kernel_commit_id",
        ),
        ("baseline", "baseline_kernel_commit_id"),
    ):
        if kernels and side in kernels:
            identity[key] = kernels[side]["kernel_id"]
    verdict = sample.get("correct", sample.get("all_pass"))
    cancelled = any(item.get("status") in {"cancelled", "canceled"} for item in [*errors, sample])
    status = "completed" if response.get("exit_code") == 0 else "failed"
    # A completed evaluation can reject a candidate without an execution failure.
    if (operation in {"evaluate", "same_allocation_abba"} and verdict is False
            and response.get("exit_code") in {0, 1} and not errors):
        status = "completed"
    if errors or response.get("truncated"):
        status = "failed"
    if isinstance(sample.get("error"), dict) and sample["error"].get("error_class") == "infra":
        status = "failed"
    missing_result = operation in PREFIXES and not has_sample and not errors
    if missing_result:
        status = "failed"
    if cancelled:
        status = "cancelled"
    policy = options.get("evaluation_policy") or {}
    mode = options.get("evaluation_mode") or policy.get("mode") or sample.get("mode") or "full"
    if operation == "evaluate":
        result = _metrics(sample, verdict=verdict, mode=mode, known=status == "completed")
        result.update(mode=mode, input_scope=input_scope(options))
    elif operation == "same_allocation_abba":
        result = deepcopy(sample)
        for key in (*IDENTITIES, "status", "operation", "schedule", "measurements",
                    "measurement_aggregation", "shape_batch_count", "completed_shape_batch_count"):
            result.pop(key, None)
        for side in ("baseline", "candidate"):
            raw = sample.get(side) or {}
            result[side] = _metrics(raw, verdict=raw.get("correct", raw.get("all_pass")), known=status == "completed")
        result.update(mode="full", input_scope=input_scope(options),
                      correct=verdict if status == "completed" and isinstance(verdict, bool) else None)
        a, b = (finite_number(result[side].get("latency_us_geomean")) for side in ("baseline", "candidate"))
        speedup = a / b if verdict is True and status == "completed" and a and b and a > 0 and b > 0 else None
        result.update(speedup=speedup, improvement_pct=(speedup - 1) * 100 if speedup is not None else None)
    else:
        result = {key: value for key, value in sample.items() if key not in IDENTITIES}
    # Probe output is data, not an Evaluate verdict. Preserve it even on failure.
    if notes or operation == "dev":
        result["stdout"] = "\n".join(notes) + ("\n" if notes else "")
    if response.get("stderr") or operation == "dev":
        result["stderr"] = response.get("stderr", "")
    if operation == "dev":
        result["exit_code"] = response.get("exit_code")
    if errors:
        first_error = errors[0]
        if known_gateway_error(first_error):
            for key in (
                "error", "next_action", "recovery", "outcome", "job_id",
                "trace_id", "http_status",
            ):
                if key in first_error:
                    result[key] = first_error[key]
        else:
            # Unknown Agate schemas are already redacted at the projection
            # boundary. Keep the complete envelope so the Agent sees the
            # actual failure instead of a Runtime-authored generic wrapper.
            result.update(deepcopy(first_error))
        if len(errors) > 1:
            result["errors"] = errors
    if response.get("truncated"):
        result["truncated"] = True
    if response.get("diagnostic_truncation"):
        result["diagnostic_truncation"] = response["diagnostic_truncation"]
    if missing_result and not (notes or response.get("stderr")):
        result.update(error={"code": "result_unavailable", "message": "No valid structured result was captured."},
                      outcome="unknown", next_action="Inspect the saved Record and ask the operator to reconcile the outcome; do not resubmit blindly.")
    elif missing_result:
        # Unknown Agate output is itself the diagnostic. Preserve the redacted
        # streams and exit code without obscuring them with a synthetic
        # result_unavailable/outcome_unknown wrapper.
        result["exit_code"] = response.get("exit_code")
    return {"operation": "evaluate" if operation == "same_allocation_abba" else operation,
            "status": status, "result": result, **identity}


def seal_response(response: dict, request: dict, *, record_id=None, kernels=None) -> dict:
    response = bound_diagnostic_response(response)
    return {**response, "agent_result": canonical_result(response, request, record_id=record_id, kernels=kernels)}


def cacheable_evaluation(response: dict, request: dict) -> bool:
    """One verdict contract for measurement publication and subsequent reuse."""
    outcome = canonical_result(response, request)
    return (outcome["status"] == "completed"
            and isinstance(outcome["result"].get("correct"), bool)
            and outcome["result"].get("outcome") not in {"unknown", "unavailable"})
