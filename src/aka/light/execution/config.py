"""Shared mechanics with Light-owned remote queue policy."""
from collections.abc import Mapping

from aka.legacy.execution.config import (
    distributed_evaluation_world_size,
    evaluator_num_gpus,
    queue_wait_grace as _legacy_queue_wait_grace,
)


DEFAULT_QUEUE_WAIT_GRACE = 120


def queue_wait_grace(environment: Mapping[str, str]) -> int:
    """Return Light's queue grace, capped at two minutes."""
    if "ATREX_SANDBOX_QUEUE_WAIT_GRACE" not in environment:
        return DEFAULT_QUEUE_WAIT_GRACE
    return min(_legacy_queue_wait_grace(environment), DEFAULT_QUEUE_WAIT_GRACE)
