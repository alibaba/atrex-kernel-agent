"""The context handed to every plugin.

``apply(ctx, config)`` is the whole setup hook. Everything a plugin contributes goes through
this object, and every contribution is an effect owned by the calling fiber, so unloading the
plugin removes it again with no bookkeeping in the plugin itself.
"""

from __future__ import annotations

from types import ModuleType
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

from .declaration import PluginDeclaration, declare
from .effects import Disposer
from .errors import UndeclaredInjection, UnknownService
from .events import EventBus
from .fiber import Fiber
from .keys import Event, ServiceKey
from .registry import Realm
from .scope import Scope, ScopedEntry

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .invariants import InvariantFacade


class Context:
    """A repository of services plus the registration surface for one fiber."""

    def __init__(
        self,
        fiber: Fiber,
        *,
        scope: Scope | None = None,
        realm: Realm | None = None,
    ):
        self._fiber = fiber
        self._scope = scope or fiber.scope
        self._realm = realm or fiber.realm

    # -- identity --------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"<Context {self._fiber.entry_id}>"

    @property
    def fiber(self) -> Fiber:
        return self._fiber

    @property
    def entry_id(self) -> str:
        return self._fiber.entry_id

    @property
    def bus(self) -> EventBus:
        return self._fiber.root.bus

    @property
    def invariants(self) -> "InvariantFacade":
        from .invariants import InvariantFacade

        return InvariantFacade(self, self._fiber.root.invariants)

    @property
    def current_scope(self) -> Scope:
        return self._scope

    # -- services --------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        fiber = self.__dict__.get("_fiber")
        if fiber is None:  # pragma: no cover - only during partial construction
            raise AttributeError(name)
        declaration = fiber.declaration
        if name in declaration.inject:
            impl = self.__dict__["_realm"].resolve(name)
            if impl is None:
                raise UnknownService(name)
            return impl.value
        if name in declaration.optional_inject:
            impl = self.__dict__["_realm"].resolve(name)
            return None if impl is None else impl.value
        raise UndeclaredInjection(name, fiber.entry_id)

    def get(self, name: str) -> Any | None:
        """Probe a service without declaring it. Returns ``None`` when unprovided.

        Use this for genuinely optional capabilities; reserve attribute access for declared
        injections so the dependency graph stays readable from the declaration alone.
        """
        impl = self._realm.resolve(name)
        return None if impl is None else impl.value

    def has(self, name: str) -> bool:
        return self._realm.resolve(name) is not None

    def provide(self, key: ServiceKey | str, value: Any) -> Disposer:
        """Publish a service into *this* context's realm and return its disposer.

        The realm matters: a service published through ``ctx.isolate(...)`` must land in that
        subtree's realm, not escape into the shared one.
        """
        return self._fiber.provide(self._key(key), value, realm=self._realm)

    def _key(self, key: ServiceKey | str) -> ServiceKey:
        if isinstance(key, ServiceKey):
            resolved = self._fiber.root.key(key.name)
            if resolved is None:
                raise UnknownService(key.name)
            if resolved != key:
                raise UnknownService(f"{key.name} (conflicting token metadata)")
            return resolved
        resolved = self._fiber.root.key(key)
        if resolved is None:
            raise UnknownService(key)
        return resolved

    # -- events ----------------------------------------------------------

    def on(
        self,
        event: Event,
        listener: Callable[..., Any],
        *,
        order: int = 500,
        label: str = "",
    ) -> Disposer:
        disposer = self.bus.on(
            event,
            listener,
            order=order,
            owner=self.entry_id,
            label=label or f"{self.entry_id}:{event.name}",
        )
        return self.effect(disposer, label=f"on {event.name}")

    def emit(self, event: Event, payload: Any) -> None:
        self.bus.emit(event, payload)

    def waterfall(
        self, event: Event, payload: Any, terminal: Callable[[Any], Any]
    ) -> Any:
        return self.bus.waterfall(event, payload, terminal)

    # -- lifecycle -------------------------------------------------------

    def effect(self, dispose: Disposer, *, label: str = "") -> Disposer:
        return self._fiber.effects.add(dispose, label=label)

    def plugin(
        self,
        module: ModuleType | PluginDeclaration,
        config: Mapping[str, Any] | None = None,
        *,
        entry_id: str = "",
        required: bool = False,
        isolate: Sequence[str] = (),
    ) -> Fiber:
        """Mount a child plugin. It unloads with this fiber."""
        declaration = (
            module
            if isinstance(module, PluginDeclaration)
            else declare(module, known_seams=self._fiber.root.seams)
        )
        child = self._fiber.root.mount(
            declaration,
            config,
            entry_id=entry_id or f"{self.entry_id}/{declaration.name}",
            required=required,
            isolate=isolate,
            parent=self._fiber,
            scope=self._scope,
            realm=self._realm,
        )
        self._fiber.root.settle()
        return child

    def isolate(self, names: Sequence[str]) -> "Context":
        """A context whose subtree owns ``names`` privately."""
        isolated = self._fiber.root._isolation_names(names)
        realm = self._realm.isolate(f"{self.entry_id}:isolate", isolated)
        return Context(self._fiber, scope=self._scope, realm=realm)

    def scope(self, name: str) -> "Context":
        """A context whose named contributions land in a nested layer."""
        return Context(self._fiber, scope=self._scope.child(name), realm=self._realm)

    # -- scoped contributions --------------------------------------------

    def contribute(
        self, kind: str, name: str, value: Any, *, order: int = 500
    ) -> Disposer:
        entry = ScopedEntry(name=name, value=value, owner=self.entry_id, order=order)
        return self.effect(
            self._scope.contribute(kind, entry), label=f"contribute {kind}.{name}"
        )

    def resolve(self, kind: str) -> tuple[ScopedEntry, ...]:
        return self._scope.resolve(kind)

    # -- diagnostics -----------------------------------------------------

    def log(self, message: str) -> None:
        print(f"[aka:{self.entry_id}] {message}", flush=True)


__all__ = ["Context"]
