"""Fibers: one plugin instance, its lifecycle, and its dependency epoch.

A fiber owns exactly one mounted plugin: its validated config, its effects, and its state.
Load order is never declared anywhere -- it falls out of the epoch.

A fiber's **epoch** is the tuple of live *registration serials* behind its declared injections.
When a required service is absent the epoch is ``INACTIVE`` and the fiber waits in ``PENDING``,
which is a legitimate steady state rather than an error. When a provider is replaced -- by
another row, or by the same row re-publishing -- the epoch changes, so the dependent unloads and
reloads against the new implementation. That single mechanism gives waiting, hot provider swap,
and cascading restart without any boot sequencing.

Cordis gets a microtask checkpoint for free, so a registration made during another plugin's
load can invalidate that load. Synchronous Python has no such checkpoint, so :class:`Root`
queues work discovered mid-transition and drains it in :meth:`Root.settle`.
"""

from __future__ import annotations

import enum
import sys
from collections import deque
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from aka.core.plugin_runtime.schema import PluginError, validate_schema

from .composition import apply_interpolation
from .declaration import PluginDeclaration
from .effects import EffectStack, call_sync
from .errors import (
    ConfigError,
    CoreError,
    DisposalErrors,
    SettleDivergence,
    UnknownService,
)
from .events import EventBus
from .internal import (
    INTERNAL_ERROR,
    INTERNAL_FIBER,
    ConfigDraft,
    ErrorReport,
    FiberTransition,
    INTERNAL_CONFIG,
)
from .keys import ServiceKey
from .registry import Realm
from .scope import Scope

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .context import Context
    from .invariants import InvariantRegistry


class FiberState(enum.Enum):
    PENDING = "pending"
    LOADING = "loading"
    ACTIVE = "active"
    UNLOADING = "unloading"
    DISPOSED = "disposed"
    FAILED = "failed"


class _Inactive:
    """Sentinel epoch meaning "a required service is missing"."""

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return "INACTIVE"


INACTIVE = _Inactive()

DEFAULT_SETTLE_ROUNDS = 64


class Fiber:
    """One mounted plugin instance."""

    def __init__(
        self,
        root: "Root",
        declaration: PluginDeclaration,
        raw_config: Mapping[str, Any],
        *,
        entry_id: str,
        realm: Realm,
        scope: Scope,
        required: bool = False,
        config_resolved: bool = False,
        parent: "Fiber | None" = None,
    ):
        self.root = root
        self.declaration = declaration
        self.raw_config = dict(raw_config)
        self.config_resolved = config_resolved
        self.entry_id = entry_id
        self.realm = realm
        self.scope = scope
        self.required = required
        self.parent = parent
        self.uid = root.next_uid()
        self.children: list[Fiber] = []
        self.effects = EffectStack(entry_id)
        self.provided: set[str] = set()
        self.config: Mapping[str, Any] | None = None
        self.state = FiberState.PENDING
        self.error: BaseException | None = None
        self.epoch: Any = INACTIVE
        self._restart_requested = False
        self._ctx: "Context | None" = None
        if parent is not None:
            parent.children.append(self)

    # -- identity --------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"<Fiber {self.entry_id} {self.state.value}>"

    @property
    def ctx(self) -> "Context":
        if self._ctx is None:
            from .context import Context

            self._ctx = Context(self)
        return self._ctx

    @property
    def unsatisfied(self) -> tuple[str, ...]:
        return tuple(
            name
            for name in self.declaration.inject
            if self.realm.resolve(name) is None
        )

    # -- dependency epoch ------------------------------------------------

    def compute_epoch(self) -> Any:
        ids: list[int] = []
        for name in sorted(self.declaration.inject):
            impl = self.realm.resolve(name)
            if impl is None:
                return INACTIVE
            # The registration's serial, not the providing fiber's id: a provider that withdraws
            # and re-publishes must reload its dependents whether or not a settle happened in
            # between, so the reload is a property of the composition rather than of timing.
            ids.append(impl.serial)
        for name in sorted(self.declaration.optional_inject):
            impl = self.realm.resolve(name)
            # An optional provider appearing or vanishing changes the epoch too, so an
            # optional swap reloads cleanly instead of leaving a stale reference behind.
            ids.append(impl.serial if impl is not None else 0)
        return tuple(ids)

    def refresh(self) -> None:
        if self.state is FiberState.DISPOSED:
            return
        if self._restart_requested:
            self._restart_requested = False
            if self.state is FiberState.FAILED:
                self.error = None
                self._transition(FiberState.PENDING)
            elif self.effects or self.children or self.state is FiberState.ACTIVE:
                self._unload()
            self.epoch = INACTIVE
        if self.state is FiberState.FAILED:
            return
        epoch = self.compute_epoch()
        if epoch == self.epoch:
            return
        previous, self.epoch = self.epoch, epoch
        if epoch is INACTIVE:
            if self.state is FiberState.ACTIVE:
                self._unload()
        elif previous is INACTIVE:
            self._load()
        else:
            self._unload()
            self._load()

    # -- lifecycle -------------------------------------------------------

    def _load(self) -> None:
        self._transition(FiberState.LOADING)
        try:
            self.config = self._resolve_config()
        except Exception as exc:  # noqa: BLE001 - a bad config is a contained failure
            self._fail(exc)
            return
        except BaseException:
            # Same rule as an aborted apply: leave the fiber inert and retryable rather than
            # stranded in LOADING with a live epoch that refresh() will never revisit.
            self._discard_effects()
            self.epoch = INACTIVE
            self._transition(FiberState.PENDING)
            raise
        try:
            result = call_sync(self.declaration.apply, self.ctx, self.config)
            if callable(result):
                self.effects.add(result, label="apply() return")
        except Exception as exc:  # noqa: BLE001 - a failing plugin is a contained failure
            self._discard_effects()
            self._fail(exc)
            return
        except BaseException:
            # EnvironmentUnavailable and friends must reach the caller untouched; leave the
            # fiber inert rather than half-loaded.
            self._discard_effects()
            self.epoch = INACTIVE
            self._transition(FiberState.PENDING)
            raise
        missing = sorted(set(self.declaration.provide) - self.provided)
        if missing:
            self._discard_effects()
            self._fail(
                ConfigError(
                    self.entry_id,
                    f"declared provide {', '.join(missing)} but never registered it",
                )
            )
            return
        self._transition(FiberState.ACTIVE)

    def _unload(self) -> None:
        self._transition(FiberState.UNLOADING)
        self._discard_effects()
        self.config = None
        self._ctx = None
        self._transition(FiberState.PENDING)

    def _discard_effects(self) -> None:
        """Undo everything this fiber's current load produced, children included.

        Children are disposed first and unconditionally: a child mounted partway through an
        ``apply`` that then failed or aborted would otherwise stay ACTIVE under a dead parent,
        keep its services published, and collide with its own row id on the next attempt.
        """
        abort = None
        for child in tuple(self.children):
            try:
                child.dispose()
            except BaseException as exc:
                if abort is None:
                    abort = exc
        self.provided.clear()
        try:
            self.effects.unwind()
        except DisposalErrors as exc:
            self.root.report(f"{self.entry_id} teardown", exc)
        except BaseException as exc:
            if abort is None:
                abort = exc
        if abort is not None:
            raise abort

    def dispose(self) -> None:
        if self.state is FiberState.DISPOSED:
            return
        try:
            if self.state is not FiberState.FAILED:
                self._unload()
            else:
                self._discard_effects()
        finally:
            if self.parent is not None and self in self.parent.children:
                self.parent.children.remove(self)
            self._transition(FiberState.DISPOSED)
            self.root.forget(self)

    def restart(self) -> None:
        """Request an unload/load cycle and settle before returning to an external caller.

        Requests made during a settle are deferred until the current transition finishes.
        Missing dependencies leave the fiber pending; failed fibers may retry. Requests
        during tree disposal or after this fiber is disposed have no effect.
        """
        if self.state is FiberState.DISPOSED or self.root._disposing:
            return
        self._restart_requested = True
        self.root.enqueue(self)
        if self.state not in (FiberState.LOADING, FiberState.UNLOADING):
            self.root.settle()

    # -- config ----------------------------------------------------------

    def _resolve_config(self) -> Mapping[str, Any]:
        interpolated = self.raw_config if self.config_resolved else apply_interpolation(
            self.raw_config,
            allowlist=self.declaration.interpolate,
            variables=self.root.variables,
            source=f"entry:{self.entry_id}",
        )
        draft = ConfigDraft(
            entry_id=self.entry_id, module=self.declaration.module, config=interpolated
        )
        draft = self.root.bus.waterfall(INTERNAL_CONFIG, draft, lambda value: value)
        merged = dict(draft.config) if self.config_resolved else {**self.declaration.defaults, **dict(draft.config)}
        schema = self.declaration.config_schema
        if schema is None:
            if merged:
                raise ConfigError(
                    self.entry_id,
                    f"row supplies config but {self.declaration.module} declares no Config "
                    f"schema: {', '.join(sorted(merged))}",
                )
            return {}
        try:
            validate_schema(schema, merged, f"{self.entry_id}.config")
        except PluginError as exc:
            raise ConfigError(self.entry_id, str(exc)) from exc
        if self.declaration.config_validator is not None:
            try:
                call_sync(self.declaration.config_validator, merged)
            except Exception as exc:
                raise ConfigError(self.entry_id, str(exc)) from exc
        return merged

    # -- state -----------------------------------------------------------

    def _transition(self, state: FiberState) -> None:
        previous, self.state = self.state, state
        self.root.observe(
            FiberTransition(
                uid=self.uid,
                entry_id=self.entry_id,
                module=self.declaration.module,
                previous=previous.value,
                current=state.value,
                error="" if self.error is None else str(self.error),
            )
        )

    def _fail(self, exc: BaseException) -> None:
        self.error = exc
        self.epoch = INACTIVE
        self._transition(FiberState.FAILED)
        self.root.report(f"entry {self.entry_id}", exc)

    # -- service registration (called through Context) -------------------

    def provide(self, key: ServiceKey, value: Any, *, realm: Realm | None = None) -> Any:
        """Publish a service owned by this fiber's current load."""
        if self.state not in (FiberState.LOADING, FiberState.ACTIVE):
            raise ConfigError(
                self.entry_id,
                f'cannot provide "{key.name}" from a fiber that is {self.state.value}; the '
                "registration would outlive the load that made it",
            )
        target = realm or self.realm
        disposer = target.provide(key, value, self)
        self.provided.add(key.name)

        def undo() -> None:
            self.provided.discard(key.name)
            disposer()

        return self.effects.add(undo, label=f"provide {key.name}")


class Root:
    """The plugin tree: shared realm, scope, event bus, and the settle loop."""

    def __init__(
        self,
        *,
        seams: Mapping[str, ServiceKey] | None = None,
        variables: Mapping[str, str] | None = None,
        settle_rounds: int = DEFAULT_SETTLE_ROUNDS,
        stderr: Any = None,
    ):
        self.seams: Mapping[str, ServiceKey] = dict(seams or {})
        self.variables: Mapping[str, str] = dict(variables or {})
        self.settle_rounds = settle_rounds
        self.realm = Realm("root")
        self.scope = Scope("global")
        self.bus = EventBus(error_sink=self.report)
        self.fibers: dict[str, Fiber] = {}
        self.transitions: list[FiberTransition] = []
        self.errors: list[ErrorReport] = []
        #: uids in the order they became active; teardown unwinds it in reverse.
        self._activation_order: list[int] = []
        self._uid = 0
        self._pending: deque[Fiber] = deque()
        self._settling = False
        self._disposing = False
        self._reporting = False
        self._stderr = stderr if stderr is not None else sys.stderr
        self._watch = self.realm.on_mutate(self._service_changed)
        from .invariants import InvariantRegistry

        self.invariants: "InvariantRegistry" = InvariantRegistry(self)

    # -- identity --------------------------------------------------------

    def next_uid(self) -> int:
        self._uid += 1
        return self._uid

    def key(self, name: str) -> ServiceKey | None:
        return self.seams.get(name)

    def _isolation_names(self, names: Sequence[str]) -> frozenset[str]:
        if isinstance(names, (str, bytes)) or not isinstance(names, Sequence):
            raise TypeError("isolation names must be a sequence of strings")
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError("isolation names must be non-empty strings")
        isolated = frozenset(names)
        unknown = sorted(isolated - set(self.seams))
        if unknown:
            raise UnknownService(", ".join(unknown))
        return isolated

    # -- mounting --------------------------------------------------------

    def mount(
        self,
        declaration: PluginDeclaration,
        config: Mapping[str, Any] | None = None,
        *,
        entry_id: str = "",
        required: bool = False,
        config_resolved: bool = False,
        isolate: Sequence[str] = (),
        parent: Fiber | None = None,
        scope: Scope | None = None,
        realm: Realm | None = None,
    ) -> Fiber:
        isolated = self._isolation_names(isolate)
        entry_id = entry_id or declaration.name
        if entry_id in self.fibers:
            raise ConfigError(entry_id, "duplicate composition row id")
        base_realm = realm or (parent.realm if parent is not None else self.realm)
        realm = (
            base_realm.isolate(entry_id, isolated) if isolated else base_realm
        )
        base_scope = scope or (parent.scope if parent is not None else self.scope)
        fiber = Fiber(
            self,
            declaration,
            config or {},
            entry_id=entry_id,
            realm=realm,
            scope=base_scope,
            required=required,
            config_resolved=config_resolved,
            parent=parent,
        )
        self.fibers[entry_id] = fiber
        self.enqueue(fiber)
        return fiber

    def forget(self, fiber: Fiber) -> None:
        if self.fibers.get(fiber.entry_id) is fiber:
            del self.fibers[fiber.entry_id]

    # -- the settle loop -------------------------------------------------

    def enqueue(self, fiber: Fiber) -> None:
        self._pending.append(fiber)

    def _service_changed(self, name: str) -> None:
        woken = False
        for fiber in tuple(self.fibers.values()):
            if (
                name in fiber.declaration.inject
                or name in fiber.declaration.optional_inject
            ):
                self.enqueue(fiber)
                woken = True
        # A provide made outside a settle -- from a host caller rather than from an apply -- has
        # nobody to drain the queue for it, and its dependents would wait on a service that
        # already resolves. While the whole tree is being torn down there is nothing to settle
        # toward: draining then could reload a row whose dependencies have not been reached yet.
        if woken and not self._settling and not self._disposing:
            self.settle()

    def settle(self) -> None:
        """Drive every queued fiber until the tree stops changing state.

        A ``provide`` or ``dispose`` performed inside another fiber's ``apply`` enqueues
        instead of recursing, so a plugin never observes a half-loaded neighbour. Exhausting
        the round budget means two rows are fighting -- raise rather than spin.
        """
        if self._settling:
            return
        self._settling = True
        try:
            rounds = 0
            while self._pending:
                rounds += 1
                if rounds > self.settle_rounds:
                    stuck = tuple(
                        dict.fromkeys(fiber.entry_id for fiber in self._pending)
                    )
                    raise SettleDivergence(self.settle_rounds, stuck)
                batch = list(dict.fromkeys(self._pending))
                self._pending.clear()
                while batch:
                    # Pop as we go and put the remainder back if a refresh escapes: an abort must
                    # not silently drop the rows that were still queued behind it, or they stay
                    # PENDING forever with nothing left to wake them.
                    fiber = batch.pop(0)
                    try:
                        fiber.refresh()
                    except BaseException:
                        self._pending.extendleft(reversed(batch))
                        raise
        finally:
            self._settling = False

    # -- diagnostics -----------------------------------------------------

    def observe(self, transition: FiberTransition) -> None:
        self.transitions.append(transition)
        if transition.current == FiberState.ACTIVE.value:
            if transition.uid in self._activation_order:
                self._activation_order.remove(transition.uid)
            self._activation_order.append(transition.uid)
        self.bus.emit(INTERNAL_FIBER, transition)

    def warn(self, message: str) -> None:
        print(f"[aka] {message}", file=self._stderr, flush=True)

    def report(self, source: str, exc: BaseException) -> None:
        report = ErrorReport(source=source, error=f"{type(exc).__name__}: {exc}")
        self.errors.append(report)
        print(f"[aka] {source}: {report.error}", file=self._stderr, flush=True)
        if self._reporting:
            return
        self._reporting = True
        try:
            self.bus.emit(INTERNAL_ERROR, report)
        finally:
            self._reporting = False

    def state_of(self, entry_id: str) -> FiberState | None:
        fiber = self.fibers.get(entry_id)
        return None if fiber is None else fiber.state

    @property
    def active(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                entry_id
                for entry_id, fiber in self.fibers.items()
                if fiber.state is FiberState.ACTIVE
            )
        )

    @property
    def failed(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            sorted(
                (entry_id, f"{type(fiber.error).__name__}: {fiber.error}")
                for entry_id, fiber in self.fibers.items()
                if fiber.state is FiberState.FAILED
            )
        )

    @property
    def pending(self) -> tuple[tuple[str, tuple[str, ...]], ...]:
        return tuple(
            sorted(
                (entry_id, fiber.unsatisfied)
                for entry_id, fiber in self.fibers.items()
                if fiber.state is FiberState.PENDING
            )
        )

    @property
    def live_effects(self) -> int:
        return sum(len(fiber.effects) for fiber in self.fibers.values())

    # -- teardown --------------------------------------------------------

    def dispose(self) -> None:
        """Tear the tree down consumers-first, then close the bus.

        Reverse *activation* order, not reverse mount order: row order carries no load semantics,
        so a consumer can easily be mounted before the provider it injects. Unwinding in the order
        rows became active guarantees a consumer's disposers still see the services it declared.
        """
        self._disposing = True
        abort = None
        try:
            order = {uid: index for index, uid in enumerate(self._activation_order)}
            remaining = sorted(
                self.fibers.values(),
                key=lambda item: (-order.get(item.uid, -1), -item.uid),
            )
            for fiber in remaining:
                try:
                    fiber.dispose()
                except Exception as exc:  # noqa: BLE001 - keep disposing the rest
                    self.report(f"{fiber.entry_id} dispose", exc)
                except BaseException as exc:
                    if abort is None:
                        abort = exc
            self._pending.clear()
            self._watch()
            self.bus.close()
        finally:
            self._disposing = False
        if abort is not None:
            raise abort


__all__ = [
    "DEFAULT_SETTLE_ROUNDS",
    "INACTIVE",
    "Fiber",
    "FiberState",
    "Root",
]
