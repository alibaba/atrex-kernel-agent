"""Preserve original backend causes and distinguish terminal from cleanup evidence."""
from aka.contracts.execution_recovery import ExecutionCause


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


