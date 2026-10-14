"""Process-neutral invocation shared by independently selected startup targets."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Mapping, Protocol
from types import MappingProxyType


@dataclass(frozen=True)
class Invocation:
    """Borrow current process state; None keeps the caller's sys.argv semantics.

    The host supplies continuation environment overrides. Targets pass these
    to owned child processes, or persist them for delayed restarts; they contain
    no service locator or live composition ownership.
    """

    argv: tuple[str, ...] | None = None
    environment: Mapping[str, str | None] = field(default_factory=dict)

    bindings: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if any(not isinstance(k, str) or (v is not None and not isinstance(v, str)) for k, v in self.environment.items()):
            raise TypeError("continuation environment must contain strings")
        object.__setattr__(self, "environment", MappingProxyType(dict(self.environment)))
        if any(not isinstance(k, str) or not isinstance(v, str) for k, v in self.bindings.items()):
            raise TypeError("invocation bindings must contain strings")
        object.__setattr__(self, "bindings", MappingProxyType(dict(self.bindings)))
        if self.argv is not None:
            if (isinstance(self.argv, (str, bytes)) or not isinstance(self.argv, Sequence)
                    or any(not isinstance(arg, str) for arg in self.argv)):
                raise TypeError("argv must be a sequence of strings or None")
            object.__setattr__(self, "argv", tuple(self.argv))


class Startup(Protocol):
    def run(self, invocation: Invocation) -> int:
        """Run after assembly; borrow dependencies and return an integer exit code."""
        ...
