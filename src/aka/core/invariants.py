"""Package-owned runtime invariants.

A plugin publishes one installer that asserts the contracts *its own package* owns. The
registry owns selection, name reservation, installer lifecycle, and attribution; it never
imports a product package, so a violation is attributable without a dependency.

What a check may assert: authoritative event streams and mutable data. Never the presence of
a service or a method -- that is what the declaration and the loader already enforce, and a
presence check would pass for a plugin that does nothing.

Installers use the same synchronous setup and registered-effect ownership rules as
ordinary plugins. A returned cleanup callable belongs to the installer's fiber.
"""

from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING, Any, Callable, NoReturn

from .declaration import PluginDeclaration
from .effects import Disposer, call_sync, require_sync
from .errors import DuplicateProvide, InvariantFailed

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .context import Context
    from .fiber import Root

InvariantFailure = Callable[[str], NoReturn]
InvariantInstaller = Callable[["Context", InvariantFailure], Any]

_SLUG = re.compile(r"[^a-z0-9]+")

ENABLE_ENV = "AKA_INVARIANTS"


def _slug(package_name: str) -> str:
    return _SLUG.sub("-", package_name.lower()).strip("-") or "invariant"


class InvariantRegistry:
    """One active registration per package name."""

    def __init__(self, root: "Root"):
        self._root = root
        self._reserved: dict[str, str] = {}
        self._failures: list[str] = []
        self._enabled = self._resolve_enabled()

    @staticmethod
    def _resolve_enabled() -> bool:
        setting = os.environ.get(ENABLE_ENV)
        if setting is not None:
            return setting.strip().lower() not in ("", "0", "false", "no", "off")
        return False

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def reserved(self) -> tuple[str, ...]:
        return tuple(sorted(self._reserved))

    @property
    def failures(self) -> tuple[str, ...]:
        return tuple(self._failures)

    def install(
        self, owner: "Context", package_name: str, installer: InvariantInstaller
    ) -> Disposer:
        """Reserve ``package_name`` and, when selected, run its installer in a child fiber."""
        require_sync(installer, "invariant installer")
        name = package_name.strip()
        if not name or any(character.isspace() for character in name):
            raise ValueError(f"invariant package name is not usable: {package_name!r}")
        holder = self._reserved.get(name)
        if holder is not None:
            raise DuplicateProvide(f"invariant:{name}", holder, owner.entry_id)
        self._reserved[name] = owner.entry_id

        def release() -> None:
            if self._reserved.get(name) == owner.entry_id:
                del self._reserved[name]

        if not self._enabled:
            # The reservation still holds, so two plugins can never silently claim one
            # package name just because checks happen to be off.
            return owner.effect(release, label=f"reserve invariant {name}")

        def fail(message: str) -> NoReturn:
            self._failures.append(f'{name}: {message}')
            raise InvariantFailed(name, message)

        def apply(ctx: "Context", config: Any) -> Any:
            return call_sync(installer, ctx, fail)

        declaration = PluginDeclaration(
            name=_slug(name),
            module=f"invariant:{name}",
            apply=apply,
            config_schema=None,
            defaults={},
            inject=tuple(getattr(installer, "inject", ())),
            optional_inject=tuple(getattr(installer, "optional_inject", ())),
            provide=(),
            interpolate=(),
            doc=f"runtime invariants owned by {name}",
        )
        child = owner.plugin(declaration, entry_id=f"{owner.entry_id}/invariant")

        def disposer() -> None:
            child.dispose()
            release()

        return owner.effect(disposer, label=f"invariant {name}")


class InvariantFacade:
    """The per-context view returned by ``ctx.invariants``."""

    def __init__(self, owner: "Context", registry: InvariantRegistry):
        self._owner = owner
        self._registry = registry

    def register(self, package_name: str, installer: InvariantInstaller) -> Disposer:
        return self._registry.install(self._owner, package_name, installer)

    @property
    def enabled(self) -> bool:
        return self._registry.enabled

    @property
    def failures(self) -> tuple[str, ...]:
        return self._registry.failures


__all__ = [
    "ENABLE_ENV",
    "InvariantFacade",
    "InvariantFailure",
    "InvariantInstaller",
    "InvariantRegistry",
]
