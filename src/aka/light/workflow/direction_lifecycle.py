"""Direction transitions shared by request validation and persisted history replay."""

from __future__ import annotations

from aka.light.experiment.identifiers import DIRECTION_ID_RE

DIRECTION_STATUS = {
    "propose": "proposed",
    "start": "in_progress",
    "complete": "completed",
    "abandon": "abandoned",
    "block": "blocked",
    "defer": "deferred",
}
DIRECTION_REVISION_ACTION = "revise"
DIRECTION_ACTIONS = frozenset({*DIRECTION_STATUS, DIRECTION_REVISION_ACTION})
CLOSURES = {"complete", "abandon", "block", "defer"}
PAUSED_STATUSES = {"blocked", "deferred"}
TERMINAL_STATUSES = {"completed", "abandoned"}
# Compatibility export for consumers that accept evidence after execution has
# stopped. Lifecycle transitions below deliberately distinguish paused from
# terminal states.
CLOSED_STATUSES = PAUSED_STATUSES | TERMINAL_STATUSES


def advance_direction(
    statuses: dict[str, str],
    direction_id: str,
    action: str,
) -> None:
    """Apply one legal Direction event."""
    if not isinstance(direction_id, str) or DIRECTION_ID_RE.fullmatch(direction_id) is None:
        raise ValueError("Direction ID has an invalid format")
    if not isinstance(action, str) or action not in DIRECTION_ACTIONS:
        raise ValueError(
            "Direction action must be propose, revise, start, complete, abandon, block, or defer"
        )
    status = statuses.get(direction_id)
    if action == "propose":
        if status is not None:
            raise ValueError(f"Direction {direction_id} was already proposed")
    elif status is None:
        raise ValueError(f"Direction ID is outside visible history: {direction_id}")
    elif action == DIRECTION_REVISION_ACTION:
        if status not in {"proposed", "in_progress"}:
            recovery = (
                "start it before revising"
                if status in PAUSED_STATUSES
                else "it is terminal; propose a derived Direction instead"
            )
            raise ValueError(f"Direction cannot revise from status {status}; {recovery}")
        return
    elif action == "start":
        if status not in {"proposed", *PAUSED_STATUSES}:
            raise ValueError(f"Direction cannot start from status {status}")
    elif status != "in_progress":
        recovery = (
            f"submit action=start for {direction_id} first, then retry action={action}"
            if status in {"proposed", *PAUSED_STATUSES}
            else "the Direction is terminal; propose a derived Direction instead"
        )
        raise ValueError(
            f"Direction cannot {action} from status {status}; {recovery}"
        )
    statuses[direction_id] = DIRECTION_STATUS[action]
