"""Errors raised by the AKA plugin core.

Every error names the composition row (``entry_id``) or the module it belongs to, so a
failure is attributable without the reader already knowing the plugin tree.
"""

from __future__ import annotations


class CoreError(Exception):
    """Base class for every plugin-core failure."""


class DeclarationError(CoreError):
    """A module does not satisfy the plugin declaration contract."""

    def __init__(self, module: str, message: str):
        super().__init__(f'plugin declaration "{module}": {message}')
        self.module = module


class ConfigError(CoreError):
    """A row's configuration failed validation before ``apply`` could run."""

    def __init__(self, entry_id: str, message: str):
        super().__init__(f'config for entry "{entry_id}": {message}')
        self.entry_id = entry_id


class CompositionError(CoreError):
    """A profile, bundle, or patch is not a usable composition."""

    def __init__(self, source: str, message: str):
        super().__init__(f"composition {source}: {message}")
        self.source = source


class DuplicateProvide(CoreError):
    """Two rows provided the same single-implementation service in one realm."""

    def __init__(self, name: str, owner: str, claimant: str):
        super().__init__(
            f'service "{name}" is already provided by entry "{owner}"; '
            f'entry "{claimant}" cannot provide it in the same isolation realm'
        )
        self.name = name
        self.owner = owner
        self.claimant = claimant


class UndeclaredInjection(CoreError, AttributeError):
    """A plugin reached for a service it never declared.

    Also an ``AttributeError`` so ``hasattr`` and three-argument ``getattr`` keep working on
    a context; the message still names the row and the fix.
    """

    def __init__(self, name: str, entry_id: str):
        super().__init__(
            f'entry "{entry_id}" read service "{name}" without declaring it; '
            f'add "{name}" to inject, or probe it with ctx.get("{name}")'
        )
        self.name = name
        self.entry_id = entry_id


class UnknownService(CoreError):
    """A service name is not a registered seam."""

    def __init__(self, name: str):
        super().__init__(f'unknown service "{name}"; it is not a registered seam')
        self.name = name


class EventContractError(CoreError):
    """An event was declared or dispatched against its declared mode."""

    def __init__(self, event: str, message: str):
        super().__init__(f'event "{event}": {message}')
        self.event = event


class WaterfallReentry(EventContractError):
    """A waterfall listener called ``next`` more than once."""

    def __init__(self, event: str, listener: str):
        super().__init__(event, f"listener {listener} called next() more than once")
        self.listener = listener


class MonotonicViolation(EventContractError):
    """A monotonic waterfall listener shrank the result it was handed."""

    def __init__(self, event: str, listener: str, detail: str):
        super().__init__(
            event, f"listener {listener} removed part of the inner result: {detail}"
        )
        self.listener = listener


class SettleDivergence(CoreError):
    """The tree kept changing state and never reached a fixed point."""

    def __init__(self, rounds: int, entry_ids: tuple[str, ...]):
        super().__init__(
            f"plugin tree did not settle within {rounds} rounds; "
            f"rows still changing state: {', '.join(entry_ids) or '<none>'}"
        )
        self.rounds = rounds
        self.entry_ids = entry_ids


class BootFailure(CoreError):
    """A required row could not become active, so the whole tree was disposed."""

    def __init__(self, failures: tuple[tuple[str, str], ...]):
        detail = "; ".join(f"{entry_id}: {reason}" for entry_id, reason in failures)
        super().__init__(f"required plugin rows failed to load: {detail}")
        self.failures = failures


class DisposalErrors(CoreError):
    """One or more disposers failed; the rest still ran."""

    def __init__(self, failures: tuple[tuple[str, BaseException], ...]):
        detail = "; ".join(f"{label}: {error}" for label, error in failures)
        super().__init__(f"effect disposal failed: {detail}")
        self.failures = failures


class InvariantFailed(CoreError):
    """A package-owned runtime invariant was violated."""

    code = "INVARIANT"

    def __init__(self, package_name: str, message: str):
        super().__init__(f'invariant violated by "{package_name}": {message}')
        self.package_name = package_name


__all__ = [
    "BootFailure",
    "CompositionError",
    "ConfigError",
    "CoreError",
    "DeclarationError",
    "DisposalErrors",
    "DuplicateProvide",
    "EventContractError",
    "InvariantFailed",
    "MonotonicViolation",
    "SettleDivergence",
    "UndeclaredInjection",
    "UnknownService",
    "WaterfallReentry",
]
