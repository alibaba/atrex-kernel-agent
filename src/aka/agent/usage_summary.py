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
