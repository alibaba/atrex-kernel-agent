"""Roll up disjoint token receipts without inventing missing counters."""
from collections.abc import Mapping

TOKEN_KEYS = (
    "input_tokens",
    "output_tokens",
    "reasoning_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "total_tokens",
)

def usage(value: object) -> dict:
    value = value if isinstance(value, Mapping) else {}
    return {
        key: value.get(key) if type(value.get(key)) is int else None
        for key in TOKEN_KEYS
    } | {
        "measurement": (
            value.get("measurement")
            if value.get("measurement") in {"exact", "estimated", "partial"}
            else "unavailable"
        )
    }


def summarize_usage(rows: list[dict], *, missing_expected: bool = False) -> dict:
    available = {
        key: sum(type(row["usage"].get(key)) is int for row in rows)
        for key in TOKEN_KEYS
    }
    usage = {
        key: (
            sum(
                row["usage"][key]
                for row in rows
                if type(row["usage"].get(key)) is int
            )
            if available[key]
            else None
        )
        for key in TOKEN_KEYS
    }
    measurements = [row["usage"].get("measurement") for row in rows]
    totals_complete = bool(rows) and available["total_tokens"] == len(rows)
    usage["measurement"] = (
        "unavailable"
        if not rows or not available["total_tokens"]
        else "partial"
        if missing_expected or not totals_complete
        else "exact"
        if all(value == "exact" for value in measurements)
        else "estimated"
        if all(value in {"exact", "estimated"} for value in measurements)
        else "partial"
    )
    return {
        "invocation_count": len(rows),
        "usage_available_invocations": available,
        "usage": usage,
    }


def claude_usage_fallback_reason(
    provider_total: int | None, main_total: int | None, *, finished: bool
) -> str | None:
    """Explain why main responses replace a Claude provider settlement."""
    if provider_total == 0 and main_total is not None and main_total > 0:
        return "claude_zero_result_conflicts_with_responses"
    if not finished:
        return "claude_live_response_usage"
    if provider_total is None:
        return "claude_result_usage_unavailable"
    return None


def select_session_usage(value: Mapping) -> dict:
    """Select counters and their accounting provenance together.

    Positive provider settlements retain their billing scope. Child counters
    are not added to the main Agent, and a conflicting zero is only a partial
    response-based observation, never an exact provider receipt.
    """
    total = usage(value.get("total"))
    basis = value.get("accounting_basis")
    reason = None
    if value.get("backend") == "claude":
        by_agent = value.get("by_agent")
        main = usage(by_agent.get("main") if isinstance(by_agent, Mapping) else None)
        if main["total_tokens"] is not None:
            reason = claude_usage_fallback_reason(
                total["total_tokens"], main["total_tokens"],
                finished=value.get("state") != "running",
            )
            if reason:
                total = main | {"measurement": "partial"}
                basis = "main_response_fallback"
    return {"usage": total, "accounting_basis": basis, "fallback_reason": reason}


def session_usage(value: Mapping) -> dict:
    """Return selected counters for callers that only need usage totals."""
    return select_session_usage(value)["usage"]
