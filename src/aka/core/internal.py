"""Core-internal event tokens and payloads.

These belong to the container itself, not to any AKA capability, so they live beside the
core instead of in the domain event catalog. ``aka/events.py`` re-exports them so the
generated catalog lists every event in one place.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .keys import Event, TokenTable

CORE_EVENTS = TokenTable("core event")


@dataclass(frozen=True)
class ConfigDraft:
    """A row's configuration on its way to validation."""

    entry_id: str
    module: str
    config: Mapping[str, Any]

    def replace(self, config: Mapping[str, Any]) -> "ConfigDraft":
        return ConfigDraft(entry_id=self.entry_id, module=self.module, config=config)


@dataclass(frozen=True)
class FiberTransition:
    """One observed fiber state change."""

    uid: int
    entry_id: str
    module: str
    previous: str
    current: str
    error: str = ""


@dataclass(frozen=True)
class ErrorReport:
    """A contained failure: a throwing listener, a failing disposer, a failed row."""

    source: str
    error: str
    kind: str = "exception"


#: Interpolation and other pre-validation rewrites attach here. Runs before the row's
#: schema is checked, so a listener sees exactly what the composition produced.
INTERNAL_CONFIG = CORE_EVENTS.add(
    Event(
        name="internal/config",
        mode="waterfall",
        payload=ConfigDraft,
        result=ConfigDraft,
        doc="Rewrite a row's raw config before validation.",
    )
)

INTERNAL_FIBER = CORE_EVENTS.add(
    Event(
        name="internal/fiber",
        mode="emit",
        payload=FiberTransition,
        doc="One fiber changed state.",
    )
)

INTERNAL_ERROR = CORE_EVENTS.add(
    Event(
        name="internal/error",
        mode="emit",
        payload=ErrorReport,
        doc="A contained failure that did not stop the tree.",
    )
)


__all__ = [
    "CORE_EVENTS",
    "ConfigDraft",
    "ErrorReport",
    "FiberTransition",
    "INTERNAL_CONFIG",
    "INTERNAL_ERROR",
    "INTERNAL_FIBER",
]
