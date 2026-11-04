"""CLI-specific adapters, independently selected from process execution."""

from collections.abc import Callable
from pathlib import Path
from .base import (
    AgentBackendAdapter,
    ClaudeLikeAdapter,
    _counter,
    _json_events,
    _phase_marker_receipts,
    token_usage_from_mapping,
    token_usage_from_model_usage,
    toml_config_value,
    pi_settings_args,
    codex_settings_args,
)
from .claude import ClaudeAdapter
from .qoder import QoderAdapter
from .pi import PiAdapter
from .codex import CodexAdapter


__all__ = [
    "AgentBackendAdapter",
    "ClaudeLikeAdapter",
    "BackendAdapterRegistry",
    "AdapterFactory",
    "ClaudeAdapter",
    "CodexAdapter",
    "QoderAdapter",
    "PiAdapter",
    "DEFAULT_BACKEND_REGISTRY",
    "token_usage_from_mapping",
    "token_usage_from_model_usage",
    "toml_config_value",
    "pi_settings_args",
    "codex_settings_args",
    "_counter",
    "_json_events",
    "_phase_marker_receipts",
]


AdapterFactory = Callable[[Path | None], AgentBackendAdapter]


class BackendAdapterRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, AdapterFactory] = {}

    def register(self, runtime_id: str, factory: AdapterFactory) -> None:
        if not runtime_id or runtime_id in self._factories:
            raise ValueError(f"agent backend already registered: {runtime_id!r}")
        self._factories[runtime_id] = factory

    def create(
        self, runtime_id: str, runtime_assets_dir: Path | None = None
    ) -> AgentBackendAdapter:
        factory = self._factories.get(runtime_id)
        if factory is None:
            raise ValueError(f"unsupported agent CLI: {runtime_id!r}")
        adapter = factory(runtime_assets_dir)
        if adapter.id != runtime_id:
            raise ValueError(
                "agent backend registry key "
                f"{runtime_id!r} does not match adapter id {adapter.id!r}"
            )
        return adapter

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(self._factories)


DEFAULT_BACKEND_REGISTRY = BackendAdapterRegistry()
DEFAULT_BACKEND_REGISTRY.register("claude", ClaudeAdapter)
DEFAULT_BACKEND_REGISTRY.register("qodercli", QoderAdapter)
DEFAULT_BACKEND_REGISTRY.register("codex", CodexAdapter)
DEFAULT_BACKEND_REGISTRY.register("pi", PiAdapter)
