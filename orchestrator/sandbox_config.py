"""Queue budget shared by the sandbox transport and its supervising process."""
from __future__ import annotations

from collections.abc import Mapping

DEFAULT_QUEUE_WAIT_GRACE = 14_400


def queue_wait_grace(environment: Mapping[str, str]) -> int:
    name = "ATREX_SANDBOX_QUEUE_WAIT_GRACE"
    try:
        value = int(environment.get(name, str(DEFAULT_QUEUE_WAIT_GRACE)))
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value
