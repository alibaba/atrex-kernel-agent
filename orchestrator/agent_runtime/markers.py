from __future__ import annotations

import json
from collections.abc import Iterator, Mapping


PHASE_MARKER_PREFIX = "ATREX_TRACE_EVENT="


def _strings(value: object) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for nested in value.values():
            yield from _strings(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _strings(nested)


def phase_marker_receipts(value: object) -> Iterator[tuple[str, str, str]]:
    """Yield validated explicit marker receipts from successful tool output."""
    for text in _strings(value):
        for line in text.splitlines():
            if not line.startswith(PHASE_MARKER_PREFIX):
                continue
            try:
                receipt = json.loads(line[len(PHASE_MARKER_PREFIX) :])
            except json.JSONDecodeError:
                continue
            if not isinstance(receipt, Mapping):
                continue
            action = receipt.get("action")
            phase = receipt.get("phase")
            marker_id = receipt.get("marker_id")
            if (
                receipt.get("schema") == "atrex.iteration_trace.v1"
                and receipt.get("kind") == "phase_marker"
                and action in {"start", "end"}
                and isinstance(phase, str)
                and phase
                and isinstance(marker_id, str)
                and marker_id
            ):
                yield action, phase, marker_id
