"""Stable classifications for Agate responses Runtime intentionally understands."""
from __future__ import annotations

from typing import Any
import re

from aka.contracts.execution import JobResult

SOURCE_VIOLATION = re.compile(
    r"(?:Blocked import|Forbidden attribute access|Forbidden name): [A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*"
    r"|Forbidden string literal matching: [A-Za-z_]\w*"
)


def execution_error(result: JobResult) -> dict[str, Any] | None:
    """Interpret captured execution failure facts without inferring dispatch from text."""
    if result.returncode == 0 and result.transport_status in {"succeeded", "failed", "unknown"}:
        return None
    candidate_rejected = (
        result.facts.disposition == "not_submitted" and not result.job_references
        and not any(item.category == "authentication" or item.details.get("reference_rejected")
                    for item in result.diagnostics)
        and any(item.details.get("http_status") == 400
            and item.details.get("source_validation_failed") is True
            and item.details.get("candidate_rejected") is True
            and not item.details.get("reference_rejected")
            for item in result.diagnostics)
    )
    if candidate_rejected:
        violations = []
        for line in result.stderr.splitlines():
            match = re.match(r"\s*- candidate:\s*(.*)", line)
            if match:
                value = re.sub(r"\s*\(trace_id=[^)]+\)\s*$", "", match[1])
                if SOURCE_VIOLATION.fullmatch(value) and value not in violations:
                    violations.append(value)
        return {"status": "failed", "outcome": "not_submitted", "error": {
            "code": "candidate_source_rejected", "error_class": "code",
            "message": "Agate rejected candidate source before GPU submission.",
            "reason": "source_validation_failed", "job_submitted": False,
            "candidate_repairable": True, "violations": violations},
            "next_action": "Replace the reported forbidden names, imports or attribute accesses in the candidate. "
                "Use the documented APIs for the pinned runtime version, then validate the repaired source. "
                "Do not bypass source validation through Dev."}
    primary = result.primary_diagnostic
    message = result.stderr.strip() or result.stdout.strip()
    if not message:
        message = "; ".join(f"{item.category}/{item.code}: {dict(item.details)}"
            for item in result.diagnostics)
    if not message:
        message = f"Execution failed: status={result.transport_status}; exit={result.returncode}"
    return {"status": "failed", "outcome": result.facts.disposition, "error": {
        "code": primary.code if primary is not None else "execution_failed",
        "error_class": "unknown", "message": message,
        "candidate_repairable": False,
        "execution_status": result.transport_status,
        "diagnostics": [{"category": item.category, "code": item.code,
            "details": dict(item.details)} for item in result.diagnostics]},
        "next_action": "Inspect the returned error and original execution receipt. "
            "An unknown execution outcome requires original-job reconciliation; do not resubmit blindly."}


KNOWN_GATEWAY_ERROR_CLASSES = frozenset({
    "candidate",
    "code",
    "compilation",
    "correctness",
    "infra",
    "validation",
})


def known_gateway_error(value: dict[str, Any]) -> bool:
    """Whether Runtime has an intentional recovery policy for this Agate error."""
    if value.get("outcome") == "unknown" or value.get("status") in {
        "cancelled",
        "canceled",
    }:
        return True
    error = value.get("error")
    return (
        isinstance(error, dict)
        and error.get("error_class") in KNOWN_GATEWAY_ERROR_CLASSES
    )
