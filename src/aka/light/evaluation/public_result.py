"""Canonical Agent results; executor streams remain private compatibility evidence."""
from __future__ import annotations

from copy import deepcopy
import json
import math
import statistics

PREFIXES = {
    "evaluate": "[test_kernel] RESULT_JSON=",
    "same_allocation_abba": "[sandbox] ABBA_JSON=",
    "profile": "[sandbox] PROFILE_JSON=",
    "check": "[sandbox] CHECK_JSON=",
    "disassemble": "[sandbox] DISASSEMBLE_JSON=",
    "env": "[sandbox] ENV_JSON=",
}
ERROR_PREFIXES = ("[sandbox] SOURCE_ERROR_JSON=", "[sandbox] GATEWAY_ERROR_JSON=")
IDENTITIES = ("gateway_record_id", "kernel_id", "baseline_kernel_id")
MAX_RESPONSE_BYTES = 8 * 1024 * 1024


def bound_diagnostic_response(response: dict) -> dict:
    """Bound embedded diagnostics using their ASCII JSON storage size.

    Only oversized responses are compacted. Verdicts and non-diagnostic numeric evidence stay
    intact; the private executor receipt remains the source for omitted text.
    This is diagnostic truncation, not incomplete execution/transport evidence.
    """
    def size(value):
        return len(json.dumps(value, ensure_ascii=True, separators=(",", ":")))

    source = {key: value for key, value in response.items() if key != "agent_result"}
    original_bytes = size(source)
    if original_bytes <= MAX_RESPONSE_BYTES:
        return response
    # agent_result is derived from these streams; do not retain a stale copy.
    text_limit, entry_limit = 1024 * 1024, 4096
    while True:
        omitted_chars, omitted_entries = 0, 0

        def compact(value, field=""):
            nonlocal omitted_chars, omitted_entries
            if isinstance(value, str) and len(value) > text_limit:
                omitted_chars += len(value) - text_limit
                return value[:text_limit] + "\n<diagnostic text truncated>"
            if isinstance(value, dict):
                return {key: compact(item, key) for key, item in value.items()}
            if isinstance(value, list):
                if field in {"diagnostics", "actionable_diagnostics", "failures"}:
                    omitted_entries += max(0, len(value) - entry_limit)
                    value = value[:entry_limit]
                return [compact(item, field) for item in value]
            return value

        lines, note_chars = [], 0
        for line in source.get("stdout", "").splitlines():
            marker = next((p for p in (*PREFIXES.values(), *ERROR_PREFIXES,
                                      "__ATREX_NUMERICAL_RESULT__=") if line.startswith(p)), None)
            if marker:
                try:
                    value = json.loads(line[len(marker):])
                except ValueError:
                    lines.append(compact(line))
                else:
                    lines.append(marker + json.dumps(compact(value), ensure_ascii=False))
            else:
                remaining = max(0, text_limit - note_chars)
                kept = line[:remaining]
                omitted_chars += len(line) - len(kept) + (0 if kept else 1)
                if kept:
                    lines.append(kept)
                if len(line) > remaining and note_chars <= text_limit:
                    lines.append("<diagnostic stdout truncated>")
                note_chars += len(line) + 1
        result = dict(source, stdout="\n".join(lines), stderr=compact(source.get("stderr", "")))
        result["diagnostic_truncation"] = {
            "reason": "Oversized diagnostic response; original executor evidence retained privately",
            "original_response_bytes": original_bytes, "limit_bytes": MAX_RESPONSE_BYTES,
            "omitted_text_chars": omitted_chars, "omitted_diagnostic_entries": omitted_entries,
        }
        if size(result) <= MAX_RESPONSE_BYTES:
            return result
        if text_limit == 256 and entry_limit == 1:
            # Numeric/structural evidence must not be dropped to fit a verdict.
            # The caller retains its running reservation and original job IDs.
            raise ValueError("Non-diagnostic measurement evidence exceeds the response size limit")
        text_limit, entry_limit = max(256, text_limit // 2), max(1, entry_limit // 2)


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
        return deepcopy(response["agent_result"])
    operation = request["operation"]
    options = request.get("options", {})
    prefix = PREFIXES.get(operation)
    sample, errors, notes = None, [], []
    for line in response.get("stdout", "").splitlines():
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
    identity = {key: sample[key] for key in IDENTITIES if key in sample}
    if record_id:
        identity["gateway_record_id"] = record_id
    for side, key in (("candidate", "kernel_id"), ("baseline", "baseline_kernel_id")):
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
        for key in ("error", "next_action", "recovery", "outcome", "job_id", "trace_id", "http_status"):
            if key in errors[0]:
                result[key] = errors[0][key]
        if len(errors) > 1:
            result["errors"] = errors
    if response.get("truncated"):
        result["truncated"] = True
    if response.get("diagnostic_truncation"):
        result["diagnostic_truncation"] = response["diagnostic_truncation"]
    if missing_result:
        result.update(error={"code": "result_unavailable", "message": "No valid structured result was captured."},
                      outcome="unknown", next_action="Inspect the saved Record and ask the operator to reconcile the outcome; do not resubmit blindly.")
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
