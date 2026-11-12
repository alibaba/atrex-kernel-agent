"""Preserve original backend causes and distinguish terminal from cleanup evidence."""
import json

from aka.contracts.execution import ExecutionDiagnostic
from aka.contracts.execution_recovery import ExecutionCause

_TYPED_LIMITATIONS = ("kind_not_supported", "invalid_source",
    "deps_install_failed", "gateway_dependency_install_failed", "http 404", "http 413", "http 501")


def report_typed_limitation(detail):
    """Translate known backend transport limitations into explicit receipt facts.

    This fact is only a support reason. Consumers must also prove the original
    attempt is settled before selecting any previously frozen fallback work.
    """
    if "source validation failed" in str(detail).lower():
        return
    reason = next((value for value in _TYPED_LIMITATIONS if value in str(detail).lower()), None)
    if reason is not None:
        from aka.execution.outcomes import emit_diagnostic
        emit_diagnostic({"failure_class": "capability", "reason": "typed-source-unsupported",
                         "limitation": reason})


def recovery_diagnostics(job):
    """Retain native policy values; Experiment decides whether to execute again."""
    if not isinstance(job, dict):
        return ()
    error = job.get("error") if isinstance(job.get("error"), dict) else {}
    details = error.get("details") if isinstance(error.get("details"), dict) else {}
    values = {name: job[name] for name in ("replay_policy", "retry_advice", "execution_state") if name in job}
    for prefix, source in (("error", error), ("error.details", details)):
        if "retry_advice" in source:
            values[prefix + ".retry_advice"] = source["retry_advice"]
    # Diagnostic details are scalar. JSON preserves unfamiliar values and types.
    return (ExecutionDiagnostic("recovery", "gateway-policy", {
        "payload": json.dumps(values, separators=(",", ":"))}),) if values else ()


def causes_from_job(job):
    """Keep native error codes/messages and their causal structure, including deadlines."""
    def decode(value, depth=0):
        if not isinstance(value, dict):
            return ExecutionCause("backend", "unstructured-error", str(value))
        details = value.get("details")
        details = dict(details) if isinstance(details, dict) else {}
        children = []
        for source in (value, details):
            for name in ("cause", "causes", "error", "original_error", "original_cause"):
                child = source.get(name)
                if child is not None:
                    children.extend(child if isinstance(child, list) else [child])
        # Bound hostile nested payloads while retaining the fact of truncation.
        if depth >= 32:
            details["cause_chain_truncated"] = bool(children)
            children = []
        return ExecutionCause(
            str(value.get("category") or value.get("error_class") or value.get("class") or value.get("failure_class") or "backend"),
            str(value.get("code") or value.get("reason") or value.get("type") or "unknown"),
            str(value.get("message") or value.get("detail") or ""), details,
            tuple(decode(child, depth + 1) for child in children),
        )
    error = job.get("error") if isinstance(job, dict) else None
    return (decode(error),) if error else ()


def remote_termination(job):
    """Terminal bookkeeping alone does not certify remote shutdown."""
    if type(job.get("remote_terminated")) is bool:
        return job["remote_terminated"]
    error = job.get("error") or {}
    details = error.get("details") or {} if isinstance(error, dict) else {}
    if job.get("cleanup_pending") or details.get("cleanup_pending"):
        return False
    if job.get("execution_quarantined") or details.get("execution_quarantined"):
        return None
    result = job.get("result")
    if isinstance(result, dict) and type(result.get("exit_code")) is int:
        return True
    if job.get("status") == "succeeded" and isinstance(result, dict):
        return True
    return None
