"""The service store and isolation realms.

A context is a repository of services. A seam claims a stable name such as ``sandbox`` or
``numerics``; consumers find it by name instead of importing a concrete implementation, so
one row swap changes the whole product.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Mapping

from .effects import Disposer
from .errors import DuplicateProvide
from .keys import ServiceKey

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .fiber import Fiber


@dataclass(frozen=True)
class Impl:
    """One live service implementation and the fiber that owns it."""

    key: ServiceKey
    value: Any
    fiber: "Fiber"
    #: Monotonic per registration. Dependency epochs key on this rather than on the providing
    #: fiber's id, so a provider that withdraws and re-publishes reloads its dependents
    #: deterministically instead of depending on whether a settle happened in between.
    serial: int = 0

    @property
    def entry_id(self) -> str:
        return self.fiber.entry_id


class Realm:
    """One isolation scope of the service store.

    The root realm holds every service. ``Context.isolate(names)`` creates a child realm
    that owns those names privately and delegates everything else upward, so a row can
    publish a service only its own subtree sees.
    """

    def __init__(
        self,
        label: str,
        *,
        parent: "Realm | None" = None,
        isolated: frozenset[str] = frozenset(),
    ):
        self.label = label
        self.parent = parent
        self.isolated = isolated
        self._impls: dict[str, Impl] = {}
        self._listeners: list[Callable[[str], None]] = []
        self._serial = 0

    # -- topology --------------------------------------------------------

    def _root(self) -> "Realm":
        root = self
        while root.parent is not None:
            root = root.parent
        return root

    def _owner_of(self, name: str) -> "Realm":
        if self.parent is None or name in self.isolated:
            return self
        return self.parent._owner_of(name)

    def isolate(self, label: str, names: frozenset[str]) -> "Realm":
        return Realm(label, parent=self, isolated=names)

    # -- reads -----------------------------------------------------------

    def resolve(self, name: str) -> Impl | None:
        return self._owner_of(name)._impls.get(name)

    def entries(self) -> Mapping[str, Impl]:
        """Every service visible from this realm.

        Isolated names are dropped from the inherited set: ``resolve`` deliberately hides the
        parent's implementation for those, so listing it here would report an implementation the
        realm cannot reach, attributed to the wrong row.
        """
        merged: dict[str, Impl] = (
            {
                name: impl
                for name, impl in self.parent.entries().items()
                if name not in self.isolated
            }
            if self.parent
            else {}
        )
        merged.update(self._impls)
        return merged

    # -- writes ----------------------------------------------------------

    def provide(self, key: ServiceKey, value: Any, fiber: "Fiber") -> Disposer:
        owner = self._owner_of(key.name)
        existing = owner._impls.get(key.name)
        if existing is not None:
            raise DuplicateProvide(key.name, existing.entry_id, fiber.entry_id)
        if key.cardinality == "registry":
            _require_registry_shape(key, value, fiber.entry_id)
        root = self._root()
        root._serial += 1
        impl = Impl(key=key, value=value, fiber=fiber, serial=root._serial)
        owner._impls[key.name] = impl

        def disposer() -> None:
            if owner._impls.get(key.name) is impl:
                del owner._impls[key.name]
                owner._notify(key.name)

        owner._notify(key.name)
        return disposer

    # -- change notification ---------------------------------------------

    def on_mutate(self, listener: Callable[[str], None]) -> Disposer:
        root = self._root()
        root._listeners.append(listener)

        def disposer() -> None:
            if listener in root._listeners:
                root._listeners.remove(listener)

        return disposer

    def _notify(self, name: str) -> None:
        root = self._root()
        for listener in tuple(root._listeners):
            listener(name)


def _require_registry_shape(key: ServiceKey, value: Any, entry_id: str) -> None:
    missing = [
        attribute
        for attribute in ("register", "ids")
        if not callable(getattr(value, attribute, None))
    ]
    if missing:
        raise TypeError(
            f'entry "{entry_id}" provides registry seam {key.name} with a value missing '
            f"{', '.join(missing)}(); provide a registry implementation or declare the seam single"
        )


__all__ = ["Impl", "Realm"]
