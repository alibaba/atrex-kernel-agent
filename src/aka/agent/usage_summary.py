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


def select_session_usage(value: Mapping) -> dict:
    """Use the same response accounting while running and after completion.

    Provider result settlements are audit evidence, not a replacement for the
    accumulated main responses. Older captures retain those responses in
    by_agent, so readers can select them without mutating archived receipts.
    """
    total = usage(value.get("total"))
    basis = value.get("accounting_basis")
    reason = None
    if value.get("backend") == "claude" and basis not in {
        "main_response_usage", "conversation_text_estimate",
    }:
        by_agent = value.get("by_agent")
        total = usage(by_agent.get("main") if isinstance(by_agent, Mapping) else None)
        if total["total_tokens"] is not None:
            total = total | {"measurement": "partial"}
        basis = "main_response_usage"
        reason = "claude_response_accounting_selected"
    return {"usage": total, "accounting_basis": basis, "fallback_reason": reason}
