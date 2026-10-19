"""Agent availability implementation and callable interface."""
from __future__ import annotations

from aka.legacy.execution.transport import InfrastructureUnavailable
import json


def check_review_service(result) -> None:
    """Classify execution timeouts separately from explicit service outages."""
    if result.timed_out:
        raise TimeoutError("reviewer exceeded its configured execution timeout")
    if result.exit_status == 0:
        return
    for line in getattr(result, "stdout_tail", "").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict) or event.get("type") != "error":
            continue
        error = event.get("error")
        if isinstance(error, dict) and error.get("type") in {
            "overloaded_error", "rate_limit_error", "service_unavailable_error",
        }:
            raise InfrastructureUnavailable(f"review service unavailable: {error['type']}")
    raise ValueError(f"reviewer failed (exit={result.exit_status}); no service-outage evidence")
