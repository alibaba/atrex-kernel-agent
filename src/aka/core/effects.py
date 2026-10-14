"""Reversible effects.

Every contribution a plugin makes -- a provided service, a listener, a prompt section, a
registry entry -- is filed as an effect owned by the fiber that made it. Unloading a fiber
unwinds its effects in reverse registration order, so teardown needs no bookkeeping in the
plugin itself.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable

from .errors import DisposalErrors

Disposer = Callable[[], None]


def require_sync(callback: Callable[..., Any], label: str) -> None:
    """Reject unsupported hooks before invoking their bodies."""
    target = callback if inspect.isfunction(callback) else getattr(callback, "__call__", callback)
    if inspect.iscoroutinefunction(target) or inspect.isasyncgenfunction(target):
        raise TypeError(f"{label} must be synchronous; async hooks are unsupported")


def sync_result(result: Any, label: str) -> Any:
    """Catch synchronous wrappers returning awaitables without leaking coroutines."""
    if inspect.isawaitable(result) or inspect.isasyncgen(result):
        if inspect.iscoroutine(result):
            result.close()
        raise TypeError(f"{label} must be synchronous; async results are unsupported")
    return result


def call_sync(callback: Callable[..., Any], *args: Any) -> Any:
    require_sync(callback, "hook")
    return sync_result(callback(*args), "hook")


def _noop() -> None:
    return None


@dataclass(eq=False)
class Effect:
    """One reversible registration.

    ``eq=False`` so list membership and removal compare by identity: two effects can legitimately
    share a dispose callable, a label, and an owner, and field-wise equality would let one remove
    the other.
    """

    dispose: Disposer
    label: str
    owner: str
    disposed: bool = field(default=False)


class EffectStack:
    """LIFO stack of one fiber's reversible contributions."""

    def __init__(self, owner: str):
        self._owner = owner
        self._effects: list[Effect] = []
        self._unwinding = False

    def add(self, dispose: Disposer, *, label: str = "") -> Disposer:
        """File one effect and return a disposer that reverses only that effect.

        Registering during an unwind cannot be tracked -- the stack is going away -- so the
        effect runs immediately and the returned disposer is a no-op.
        """
        require_sync(dispose, "cleanup")
        if self._unwinding:
            call_sync(dispose)
            return _noop
        effect = Effect(dispose=dispose, label=label or "effect", owner=self._owner)
        self._effects.append(effect)

        def disposer() -> None:
            if effect.disposed:
                return
            effect.disposed = True
            if effect in self._effects:
                self._effects.remove(effect)
            call_sync(effect.dispose)

        return disposer

    def unwind(self) -> None:
        """Dispose every live effect in reverse order.

        A disposer raising ``Exception`` is collected so the remaining disposers still run; the
        aggregate is raised at the end.

        A disposer raising ``BaseException`` -- notably ``EnvironmentUnavailable`` -- is never
        converted, but the stack is still drained before it is re-raised. Aborting mid-unwind
        would leave the rest of this fiber's registrations installed with nothing left to remove
        them: a sandbox process group or a listener surviving its plugin is worse than a delayed
        abort.
        """
        self._unwinding = True
        failures: list[tuple[str, BaseException]] = []
        abort: BaseException | None = None
        try:
            while self._effects:
                effect = self._effects.pop()
                if effect.disposed:
                    continue
                effect.disposed = True
                try:
                    call_sync(effect.dispose)
                except Exception as exc:  # noqa: BLE001 - one bad disposer must not starve the rest
                    failures.append((f"{effect.owner}:{effect.label}", exc))
                except BaseException as exc:
                    if abort is None:
                        abort = exc
        finally:
            self._unwinding = False
        if abort is not None:
            raise abort
        if failures:
            raise DisposalErrors(tuple(failures))

    @property
    def live(self) -> tuple[str, ...]:
        return tuple(effect.label for effect in self._effects)

    def __len__(self) -> int:
        return len(self._effects)


__all__ = ["Disposer", "Effect", "EffectStack"]
