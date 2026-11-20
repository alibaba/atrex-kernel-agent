"""Stable classifications for Agate responses Runtime intentionally understands."""
from __future__ import annotations

from typing import Any
from collections.abc import Mapping
import json

from aka.contracts.execution import JobResult

INFORMATION_CATEGORIES = frozenset({"recovery", "observation", "budget"})


def _json_value(value):
    if isinstance(value, Mapping):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


def _cause_value(cause):
    return {"code": cause.code, "error_class": cause.category,
            "message": cause.message, "details": _json_value(cause.details),
            **({"causes": [_cause_value(item) for item in cause.causes]} if cause.causes else {})}


def upstream_error(result: JobResult) -> dict[str, Any] | None:
    """Read the current Agate error verbatim, never a previous retry's error."""
    latest = result.attempts[-1] if result.attempts else None
    raw = latest.backend_response if latest is not None else None
    if raw is not None:
        error = raw.get("error")
        if error is None and isinstance(raw.get("result"), Mapping):
            error = raw["result"].get("error")
        if error is not None:
            return {"error": _json_value(error),
                    **({"http_status": raw["http_status"]} if "http_status" in raw else {})}
        # A later successful/accepted observation supersedes old error causes.
        return None
    causes = latest.causes if latest is not None else result.causes
    for cause in reversed(causes):
        if cause.raw_error is not None:
            return {"error": _json_value(cause.raw_error),
                    **({"http_status": cause.http_status} if cause.http_status is not None else {})}
    # Some transports return the original envelope without an attempt channel.
    # Do not interpret a Dev/Command's arbitrary stdout as a gateway envelope.
    if result.output_scope == "transport":
        try:
            raw = json.loads(result.stdout)
        except (TypeError, ValueError):
            raw = None
        if isinstance(raw, dict) and raw.get("error") is not None and (
                "http_status" in raw or "job_id" in raw):
            return {"error": raw["error"],
                    **({"http_status": raw["http_status"]} if "http_status" in raw else {})}
    return None


def _failure_message(result):
    streams = ["\n".join(line for line in stream.splitlines()
                        if not line.startswith("[sandbox]")).strip()
               for stream in (result.stderr, result.stdout)]
    return "\n".join(stream for stream in streams if stream)


def execution_error(result: JobResult) -> dict[str, Any] | None:
    """Interpret captured execution failure facts without inferring dispatch from text."""
    original = upstream_error(result)
    if original is not None:
        collecting = any(item.code == "original-job-wait-incomplete" for item in result.diagnostics)
        return {"status": "unknown" if collecting else "failed", **original}
    if result.returncode == 0 and result.transport_status == "succeeded":
        return None
    diagnostics = tuple(item for item in result.diagnostics if item.category not in INFORMATION_CATEGORIES)
    if result.returncode == 0 and not diagnostics and not result.causes and result.transport_status in {"succeeded", "failed", "unknown"}:
        return None
    primary = result.primary_diagnostic if result.primary_diagnostic in diagnostics else None
    primary = primary or (diagnostics[0] if diagnostics else None)
    if any(item.code == "original-job-wait-incomplete" for item in diagnostics):
        return {"status": "failed", "outcome": "timeout", "error": {
            "code": "gateway_wait_timeout", "error_class": "timeout",
            "message": "Agate did not return a valid terminal result within the request wait budget. "
                "This does not establish that the remote GPU job failed or stopped.",
            "candidate_repairable": False, "execution_status": "timed_out"},
            "next_action": "The Runtime request has returned. No correctness or performance conclusion "
                "is available; do not change the kernel based on this timeout or blindly resubmit."}
    if result.causes:
        error = _cause_value(result.causes[0])
        error.update(execution_status=result.transport_status)
        if len(result.causes) > 1:
            error["causes"] = [_cause_value(item) for item in result.causes]
        http_status = next((item.details["http_status"] for item in diagnostics
                            if isinstance(item.details.get("http_status"), int)), None)
        if http_status is not None:
            error["http_status"] = http_status
        rejected = result.facts.disposition == "not_submitted" and not result.job_references
        if rejected:
            error["job_submitted"] = False
        return {"status": "failed", "outcome": result.facts.disposition, "error": error}
    message = _failure_message(result)
    if not message:
        message = "; ".join(f"{item.category}/{item.code}: {dict(item.details)}"
            for item in diagnostics)
    if not message:
        message = f"Execution failed: status={result.transport_status}; exit={result.returncode}"
    return {"status": "failed", "outcome": result.facts.disposition, "error": {
        "code": primary.code if primary is not None else "execution_failed",
        "error_class": "unknown", "message": message,
        "candidate_repairable": False,
        "execution_status": result.transport_status,
        "diagnostics": [{"category": item.category, "code": item.code,
            "details": dict(item.details)} for item in diagnostics]},
        "next_action": "Inspect the returned error and original execution receipt. " + (
            "An unknown execution outcome requires original-job reconciliation; do not resubmit blindly."
            if result.facts.disposition == "unknown" else
            "Correct the reported failure before retrying; the execution disposition is already known.")}




KNOWN_GATEWAY_ERROR_CLASSES = frozenset({
    "candidate",
    "code",
    "compilation",
    "correctness",
    "infra",
    "timeout",
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
    if (
        isinstance(error, dict)
        and isinstance(error.get("message"), str)
        and "job deadline exceeded" in error["message"].casefold()
    ):
        return True
    return (
        isinstance(error, dict)
        and error.get("error_class") in KNOWN_GATEWAY_ERROR_CLASSES
    )
