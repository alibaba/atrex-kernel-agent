"""Canonical Agent results; executor streams remain private compatibility evidence."""
from __future__ import annotations

from copy import deepcopy
import json
import math
import statistics

from .result_json import (
    LEGACY_TEST_RESULT_PREFIX,
    parse_evaluation_result_line,
)
from aka.evaluation.correctness_summary import correctness_summary, finite_number
from .response_policy import INFORMATION_CATEGORIES, known_gateway_error, execution_error
from .objectives import latency_only_result

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


def _omit_gateway_policy(receipt: dict) -> dict:
    """Remove internal recovery observations from an owned Agent projection."""
    for value in (receipt, receipt.get("result")):
        if not isinstance(value, dict) or not isinstance(value.get("diagnostics"), list):
            continue
        diagnostics = [item for item in value["diagnostics"]
                       if not (isinstance(item, dict) and item.get("category") == "recovery"
                               and item.get("code") == "gateway-policy")]
        if diagnostics:
            value["diagnostics"] = diagnostics
        else:
            value.pop("diagnostics", None)
    return receipt


def compact_receipt(agent_result: dict) -> dict:
    """Return the complete canonical Agent result stored in the Gateway Record."""
    receipt = deepcopy(agent_result)
    if receipt.get("operation") in {"evaluate", "same_allocation_abba"}:
        receipt = latency_only_result(receipt)
    record_id = receipt.get("gateway_record_id")
    if isinstance(record_id, str):
        receipt["record_path"] = f"records/{record_id}.json"
    return _omit_gateway_policy(receipt)


def with_execution_error(response: dict, receipt, *, operation=None) -> dict:
    """Project an explicit failed receipt before a consumer looks for numeric output."""
    operation = operation or (receipt.attempts[-1].operation if receipt.attempts else None)
    if operation in {"dev", "command"}:
        # Command output is the result, even on a nonzero exit. Do not invent
        # classifications, repairability or recovery advice from that output.
        response = {**response, "outcome": "timeout" if receipt.transport_status == "timed_out"
                    else receipt.facts.disposition}
        raw = next((item.backend_response for item in reversed(receipt.attempts)
                    if item.backend_response is not None), None)
        if raw is not None and raw.get("error") is not None:
            from aka.evaluation.response_policy import _json_value
            response["error"] = _json_value(raw["error"])
            if "http_status" in raw:
                response["http_status"] = raw["http_status"]
        return response
    information = [{"category": item.category, "code": item.code, "details": dict(item.details)}
                   for item in receipt.diagnostics if item.category in INFORMATION_CATEGORIES
                   and item.code not in {"replay-forbidden", "retained-instruction", "gateway-policy"}]
    response = {**response, **({"diagnostics": information} if information else {})}
    stdout = response.get("stdout", "")
    for line in stdout.splitlines():
        prefix = next((p for p in (*PREFIXES.values(), *ERROR_PREFIXES) if line.startswith(p)), None)
        if prefix:
            try:
                value = json.loads(line[len(prefix):])
            except ValueError:
                continue
            if isinstance(value, dict):
                return response
    error = execution_error(receipt)
    if error is None:
        return response
    prefix = ERROR_PREFIXES[0] if error["error"]["code"] == "candidate_source_rejected" else ERROR_PREFIXES[1]
    return {**response, "stdout": stdout + ("\n" if stdout and not stdout.endswith("\n") else "")
        + prefix + json.dumps(error, ensure_ascii=False) + "\n"}


def bound_diagnostic_response(response: dict) -> dict:
    """Preserve complete redacted diagnostics at the Agent boundary.

    Redaction and schema projection happen before this function.  A serialized
    length limit is intentionally not applied: compiler diagnostics often put
    the only actionable root cause after a large repeated template trace.
    """
    return response


def result_from_response(response, operation):
    prefix = PREFIXES.get(operation)
    for line in reversed(response.get("stdout", "").splitlines()):
        if prefix and line.startswith(prefix):
            value = json.loads(line[len(prefix):])
            return value if isinstance(value, dict) else None
    return None


def input_scope(options: dict) -> str:
    command = options.get("command") or []
    subset = any(isinstance(item, str) and item.split("=", 1)[0] == "--shape-id" for item in command)
    return "custom" if (options.get("evaluation_input_path") or options.get("evaluation_shapes_path")
                        or options.get("shape_id") or subset) else "contract"


def _metrics(raw: dict, *, verdict, mode="full", known=True) -> dict:
    value = latency_only_result(deepcopy(raw))
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
        return _omit_gateway_policy(latency_only_result(result)
            if result.get("operation", request.get("operation")) in {"evaluate", "same_allocation_abba"}
            else result)
    operation = request["operation"]
    if operation in {"dev", "command"}:
        result = {"stdout": response.get("stdout", ""), "stderr": response.get("stderr", ""),
                  "exit_code": response.get("exit_code")}
        for key in ("outcome", "error", "http_status"):
            if key in response:
                result[key] = deepcopy(response[key])
        identity = {"gateway_record_id": record_id} if record_id else {}
        if kernels and "candidate" in kernels:
            identity["kernel_commit_id"] = kernels["candidate"]["kernel_id"]
        return {"operation": operation, "status": "completed" if response.get("exit_code") == 0 else "failed",
                "result": result, **identity}
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
        result = latency_only_result(deepcopy(sample))
        for key in (*IDENTITIES, "status", "operation", "schedule", "measurements",
                    "measurement_aggregation", "shape_batch_count", "completed_shape_batch_count"):
            result.pop(key, None)
        for side in ("baseline", "candidate"):
            raw = sample.get(side) or {}
            result[side] = _metrics(raw, verdict=raw.get("correct", raw.get("all_pass")), known=status == "completed")
        result.update(mode="full", input_scope=input_scope(options),
                      correct=verdict if status == "completed" and isinstance(verdict, bool) else None)
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
    if response.get("diagnostics"):
        result["diagnostics"] = deepcopy(response["diagnostics"])
    if missing_result and not (notes or response.get("stderr") or response.get("diagnostics")):
        result.update(error={"code": "result_unavailable",
            "message": f"No structured result or diagnostic was captured: exit={response.get('exit_code')}",
            "candidate_repairable": False})
    if missing_result:
        # Unknown Agate output is itself the diagnostic. Preserve the redacted
        # streams and exit code without obscuring them with a synthetic
        # result_unavailable/outcome_unknown wrapper.
        result["exit_code"] = response.get("exit_code")
    return _omit_gateway_policy({"operation": "evaluate" if operation == "same_allocation_abba" else operation,
            "status": status, "result": result, **identity})


def seal_response(response: dict, request: dict, *, record_id=None, kernels=None) -> dict:
    response = bound_diagnostic_response(response)
    return {**response, "agent_result": canonical_result(response, request, record_id=record_id, kernels=kernels)}


_RESPONSE_FAILURE_CLASSES = frozenset({
    "infra", "infrastructure", "gateway", "transport", "network", "protocol",
    "authentication", "authorization", "request", "validation", "capability",
    "configuration", "timeout", "timed_out", "cancelled", "interrupted",
})


def _response_failure(value) -> bool:
    """Inspect structured failures only; probe text is not an error classification."""
    if isinstance(value, (list, tuple)):
        return any(_response_failure(item) for item in value)
    from collections.abc import Mapping
    if not isinstance(value, Mapping):
        return False
    if any(isinstance(value.get(key), str) and value[key] in _RESPONSE_FAILURE_CLASSES
           for key in ("error_class", "category", "failure_class")):
        return True
    return any(_response_failure(value.get(key)) for key in ("error", "errors", "causes", "diagnostics"))


def normal_evaluation_execution(receipt) -> bool:
    """A terminal response, independent of the candidate's correctness or metrics."""
    from .response_policy import _cause_value
    # Older failed submissions remain in the audit, but the last attempt owns
    # the returned response once recovery has produced a terminal result.
    latest = receipt.attempts[-1] if receipt.attempts else receipt
    return (receipt.facts.disposition == "terminal"
            and receipt.transport_status in {"succeeded", "failed"}
            and not _response_failure([_cause_value(cause) for cause in latest.causes])
            and not _response_failure([{"category": item.category} for item in latest.diagnostics]))


def cacheable_evaluation(response: dict, request: dict) -> bool:
    """Reuse normal Evaluate responses, not infrastructure/Gateway failures.

    This is a request-cache policy, not qualification for submit-kernel. A
    compilation/correctness failure or missing performance data is reusable.
    """
    outcome = canonical_result(response, request)
    result = outcome["result"]
    if response.get("truncated") or _response_failure(result):
        return False
    from .device_failures import embedded_device_failure
    if embedded_device_failure({"status": "succeeded", "result": result}):
        return False
    for line in response.get("stdout", "").splitlines():
        raw = parse_evaluation_result_line(line)
        if raw is None and line.startswith(PREFIXES["same_allocation_abba"]):
            try:
                raw = json.loads(line[len(PREFIXES["same_allocation_abba"]):])
            except ValueError:
                return False
        if isinstance(raw, dict) and (_response_failure(raw)
                or embedded_device_failure({"status": "succeeded", "result": raw})):
            return False
    # The owner records receipt availability separately from measurement
    # qualification, which can be unknown despite a normal terminal response.
    received = response.get("evaluation_response_received")
    if type(received) is bool:
        return received
    if result.get("outcome") in {"unknown", "unavailable", "timeout", "not_submitted"}:
        return False
    error = result.get("error")
    candidate_failure = (isinstance(error, dict)
                         and error.get("error_class") in {"candidate", "code", "compilation", "correctness"})
    return outcome["status"] == "completed" or outcome["status"] == "failed" and candidate_failure
