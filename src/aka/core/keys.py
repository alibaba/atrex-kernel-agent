"""Service and event tokens.

Python has no declaration merging, so a seam's interface and an event's dispatch contract
travel in explicit tokens instead of type-level metadata. The string ``name`` stays the
stable identity used by compositions, locks, and logs; the token carries everything the
core has to enforce.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Literal

from .errors import EventContractError

NAME = re.compile(r"[a-z][a-z0-9_-]*")
EVENT_NAME = re.compile(r"[a-z][a-z0-9-]*/[a-z][a-z0-9-]*")

Cardinality = Literal["single", "registry"]
Mode = Literal["emit", "waterfall"]

MODES: frozenset[str] = frozenset(("emit", "waterfall"))
#: Modes whose dispatch produces a value, and therefore must declare a result type.
RETURNING_MODES: frozenset[str] = frozenset(("waterfall",))

#: ``(outer, inner) -> detail`` for a monotonic waterfall; a detail string means the
#: listener removed part of what it was handed.
ShrinkCheck = Callable[[Any, Any], "str | None"]


@dataclass(frozen=True)
class ServiceKey:
    """One capability seam: the ``ctx`` attribute, its interface, and its cardinality."""

    name: str
    definition: str
    cardinality: Cardinality = "single"
    module: str = ""
    protocol_version: int = 1
    doc: str = ""

    def __post_init__(self) -> None:
        if not NAME.fullmatch(self.name):
            raise ValueError(f"service name is not a stable identifier: {self.name!r}")
        if not self.definition:
            raise ValueError(f"service {self.name} declares no definition class")
        if self.cardinality not in ("single", "registry"):
            raise ValueError(
                f"service {self.name} has unsupported cardinality: {self.cardinality!r}"
            )
        if type(self.protocol_version) is not int or self.protocol_version < 1:
            raise ValueError(f"service {self.name} has invalid protocol version")

    def __str__(self) -> str:
        return f"ctx.{self.name}"


@dataclass(frozen=True)
class Event:
    """One typed event. The dispatch mode is part of the public contract."""

    name: str
    mode: Mode
    payload: type
    result: type | None = None
    monotonic: bool = False
    shrink_check: ShrinkCheck | None = field(default=None, compare=False)
    doc: str = ""

    def __post_init__(self) -> None:
        if not EVENT_NAME.fullmatch(self.name):
            raise EventContractError(
                self.name, "name must be <namespace>/<action> in lowercase"
            )
        if self.mode not in MODES:
            raise EventContractError(self.name, f"unsupported mode: {self.mode!r}")
        if not isinstance(self.payload, type):
            raise EventContractError(self.name, "payload must be a type")
        returning = self.mode in RETURNING_MODES
        if returning and self.result is None:
            raise EventContractError(
                self.name, f"{self.mode} dispatch must declare a result type"
            )
        if not returning and self.result is not None:
            raise EventContractError(
                self.name, f"{self.mode} dispatch must not declare a result type"
            )
        if self.monotonic and self.mode != "waterfall":
            raise EventContractError(self.name, "only a waterfall can be monotonic")
        if self.shrink_check is not None and not self.monotonic:
            raise EventContractError(
                self.name, "shrink_check only applies to a monotonic waterfall"
            )

    def __str__(self) -> str:
        return self.name


class TokenTable:
    """Import-time table of tokens keyed by their stable name.

    Building the table at import time turns a duplicate or a typo into a collection
    error rather than a surprise during a campaign.
    """

    def __init__(self, kind: str):
        self._kind = kind
        self._tokens: dict[str, Any] = {}

    def add(self, token: Any) -> Any:
        name = token.name
        existing = self._tokens.get(name)
        if existing is not None and existing != token:
            raise ValueError(
                f"{self._kind} {name!r} metadata conflicts: "
                f"{existing!r} != {token!r}"
            )
        if existing is not None:
            return existing
        self._tokens[name] = token
        return token

    def __getitem__(self, name: str) -> Any:
        return self._tokens[name]

    def __contains__(self, name: object) -> bool:
        return name in self._tokens

    def __iter__(self) -> Iterator[str]:
        return iter(self._tokens)

    def __len__(self) -> int:
        return len(self._tokens)

    def get(self, name: str) -> Any | None:
        return self._tokens.get(name)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._tokens))

    def values(self) -> tuple[Any, ...]:
        return tuple(self._tokens[name] for name in sorted(self._tokens))


__all__ = [
    "Cardinality",
    "EVENT_NAME",
    "Event",
    "MODES",
    "Mode",
    "NAME",
    "RETURNING_MODES",
    "ServiceKey",
    "ShrinkCheck",
    "TokenTable",
]
