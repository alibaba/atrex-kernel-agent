"""Direction transitions shared by request validation and persisted history replay."""

from __future__ import annotations

from supervisor.errors import AgentRequestError
from supervisor.identifiers import DIRECTION_ID_RE

DIRECTION_STATUS = {
    "propose": "proposed",
    "start": "in_progress",
    "complete": "completed",
    "abandon": "abandoned",
    "block": "blocked",
    "defer": "deferred",
}
CLOSURES = {"complete", "abandon", "block", "defer"}
CLOSED_STATUSES = {"completed", "abandoned", "blocked", "deferred"}


def advance_direction(
    statuses: dict[str, str],
    started: set[str],
    direction_id: str,
    action: str,
) -> None:
    """Apply one legal event; ``started`` is reset at each Episode boundary."""
    if not isinstance(direction_id, str) or DIRECTION_ID_RE.fullmatch(direction_id) is None:
        raise ValueError("Direction ID has an invalid format")
    if not isinstance(action, str) or action not in DIRECTION_STATUS:
        raise ValueError(
            "Direction action must be propose, start, complete, abandon, block, or defer"
        )
    status = statuses.get(direction_id)
    if action == "propose":
        if status is not None:
            raise ValueError(f"Direction {direction_id} was already proposed")
    elif status is None:
        raise ValueError(f"Direction ID is outside visible history: {direction_id}")
    elif action == "start":
        active = [
            key for key, value in statuses.items() if value == "in_progress" and key != direction_id
        ]
        if active:
            raise AgentRequestError(
                "only one Direction may be in progress; close or defer "
                f"{active[0]} before starting another",
                code="direction_in_progress",
                direction_id=active[0],
                next_action=(
                    "Use update-direction to complete, abandon, block, or defer that Direction "
                    "before starting another. Every closure requires supporting_experiment_ids "
                    "and hypothesis_status; record real Gateway evidence first."
                ),
            )
        if direction_id not in started and len(started) >= 3:
            raise AgentRequestError(
                "Direction advancement limit exceeded: at most three Directions "
                "may be started in one Episode",
                code="direction_limit_exceeded",
                started_direction_ids=sorted(started),
                next_action=(
                    "Do not start a fourth Direction. Continue an already-started Direction "
                    "if it is in_progress, or explicitly restart it if closed; "
                    "otherwise close remaining "
                    "in_progress Directions and submit episode-report. You may still propose "
                    "new Directions for later Episodes."
                ),
            )
        if status not in {"proposed", *CLOSED_STATUSES}:
            raise ValueError(f"Direction cannot start from status {status}")
    elif status not in {"in_progress", *CLOSED_STATUSES}:
        raise ValueError(f"Direction cannot {action} from status {status}")
    statuses[direction_id] = DIRECTION_STATUS[action]
    if action == "start":
        started.add(direction_id)
