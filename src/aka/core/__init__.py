"""The AKA plugin core.

A small container with no AKA domain knowledge: a context that is a repository of services, a
fiber per plugin instance whose dependency epoch decides when it loads, reversible effects,
typed notification and waterfall events, and JSON composition. Every capability lives outside
this package, as a plugin.
"""

from __future__ import annotations

from .composition import (
    ResolvedComposition,
    Row,
    dump,
    resolve,
)
from .context import Context
from .declaration import PluginDeclaration, declare, is_plugin
from .effects import Disposer, Effect, EffectStack
from .errors import (
    BootFailure,
    CompositionError,
    ConfigError,
    CoreError,
    DeclarationError,
    DisposalErrors,
    DuplicateProvide,
    EventContractError,
    InvariantFailed,
    MonotonicViolation,
    SettleDivergence,
    UndeclaredInjection,
    UnknownService,
    WaterfallReentry,
)
from .events import EventBus, Listener
from .fiber import INACTIVE, Fiber, FiberState, Root
from .internal import (
    CORE_EVENTS,
    INTERNAL_CONFIG,
    INTERNAL_ERROR,
    INTERNAL_FIBER,
    ConfigDraft,
    ErrorReport,
    FiberTransition,
)
from .invariants import InvariantFailure, InvariantInstaller, InvariantRegistry
from .keys import Event, ServiceKey, TokenTable
from .loader import BootReport, load
from .registry import Impl, Realm
from .scope import Scope, ScopedEntry

__all__ = [
    "BootFailure",
    "BootReport",
    "CORE_EVENTS",
    "CompositionError",
    "ConfigDraft",
    "ConfigError",
    "Context",
    "CoreError",
    "DeclarationError",
    "Disposer",
    "DisposalErrors",
    "DuplicateProvide",
    "Effect",
    "EffectStack",
    "ErrorReport",
    "Event",
    "EventBus",
    "EventContractError",
    "Fiber",
    "FiberState",
    "FiberTransition",
    "INACTIVE",
    "INTERNAL_CONFIG",
    "INTERNAL_ERROR",
    "INTERNAL_FIBER",
    "Impl",
    "InvariantFailed",
    "InvariantFailure",
    "InvariantInstaller",
    "InvariantRegistry",
    "Listener",
    "MonotonicViolation",
    "PluginDeclaration",
    "Realm",
    "ResolvedComposition",
    "Root",
    "Row",
    "Scope",
    "ScopedEntry",
    "ServiceKey",
    "SettleDivergence",
    "TokenTable",
    "UndeclaredInjection",
    "UnknownService",
    "WaterfallReentry",
    "declare",
    "dump",
    "is_plugin",
    "load",
    "resolve",
]
