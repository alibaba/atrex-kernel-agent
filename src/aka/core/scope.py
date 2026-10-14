"""Two-level scoping for named contributions.

Tools, prompt sections, skills, session-environment variables, and gates all need the same
merge rule: a global layer plus a scope chain, where the nearest layer's entry wins a
duplicate name outright without changing its ancestor's contribution.
That rule is implemented once here and reused, rather than growing five times.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .effects import Disposer
from .errors import DuplicateProvide


@dataclass(frozen=True)
class ScopedEntry:
    """One named contribution filed into a scope layer."""

    name: str
    value: Any
    owner: str = ""
    order: int = 500


class Scope:
    """One layer of the contribution chain."""

    def __init__(self, name: str, parent: "Scope | None" = None):
        self.name = name
        self.parent = parent
        self._entries: dict[str, dict[str, ScopedEntry]] = {}

    def child(self, name: str) -> "Scope":
        return Scope(name, parent=self)

    def chain(self) -> tuple["Scope", ...]:
        scopes: list[Scope] = []
        cursor: Scope | None = self
        while cursor is not None:
            scopes.append(cursor)
            cursor = cursor.parent
        return tuple(reversed(scopes))

    # -- contributions ---------------------------------------------------

    def contribute(self, kind: str, entry: ScopedEntry) -> Disposer:
        layer = self._entries.setdefault(kind, {})
        existing = layer.get(entry.name)
        if existing is not None:
            raise DuplicateProvide(
                f"{kind}.{entry.name}", existing.owner or self.name, entry.owner
            )
        layer[entry.name] = entry

        def disposer() -> None:
            if layer.get(entry.name) is entry:
                del layer[entry.name]

        return disposer

    # -- reads -----------------------------------------------------------

    def merged(self, kind: str) -> Mapping[str, ScopedEntry]:
        accumulated: dict[str, ScopedEntry] = {}
        for scope in self.chain():
            accumulated.update(scope._entries.get(kind, {}))
        return accumulated

    def resolve(self, kind: str) -> tuple[ScopedEntry, ...]:
        entries = self.merged(kind).values()
        return tuple(sorted(entries, key=lambda entry: (entry.order, entry.name)))


__all__ = ["Scope", "ScopedEntry"]
