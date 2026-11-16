"""Stable classifications for Agate responses Runtime intentionally understands."""
from __future__ import annotations

from typing import Any


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
