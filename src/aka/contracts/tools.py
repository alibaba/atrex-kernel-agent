"""Public, serializable contracts for tools executed by the resource host."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, TypedDict, runtime_checkable


def _json_copy(value: Any, label: str) -> Any:
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite JSON value") from exc


@dataclass(frozen=True)
class ToolSpec:
    """A tool declaration. Entrypoints are imported only in an isolated child process."""

    name: str
    description: str
    input_schema: Mapping[str, Any]
    output_schema: Mapping[str, Any]
    entrypoint: str
    config: Mapping[str, Any] = field(default_factory=dict)
    resources: tuple[str, ...] = ()
    timeout_seconds: float = 30.0
    api_version: int = 1
    attribution_provider: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not re.fullmatch(
            r"[a-z][a-z0-9_-]*(?:\.[a-z][a-z0-9_-]*)+", self.name
        ):
            raise ValueError("tool name must be a namespaced lowercase identifier")
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("tool description is required")
        if not isinstance(self.entrypoint, str) or not re.fullmatch(
            r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*", self.entrypoint
        ):
            raise ValueError("tool entrypoint must be module:function")
        if type(self.api_version) is not int or self.api_version != 1:
            raise ValueError("unsupported tool api_version")
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not 0 < self.timeout_seconds <= 3600
        ):
            raise ValueError("timeout_seconds must be between 0 and 3600")
        if not isinstance(self.input_schema, Mapping) or not isinstance(
            self.output_schema, Mapping
        ):
            raise TypeError("tool schemas must be JSON objects")
        if not isinstance(self.config, Mapping):
            raise TypeError("tool config must be a JSON object")
        object.__setattr__(
            self, "input_schema", _json_copy(dict(self.input_schema), "input_schema")
        )
        object.__setattr__(
            self, "output_schema", _json_copy(dict(self.output_schema), "output_schema")
        )
        object.__setattr__(self, "config", _json_copy(dict(self.config), "config"))
        object.__setattr__(self, "resources", tuple(self.resources))
        if any(not isinstance(r, str) or not r for r in self.resources):
            raise ValueError("tool resource references must be non-empty strings")
        if self.attribution_provider is not None and (
            not isinstance(self.attribution_provider, str)
            or not self.attribution_provider
        ):
            raise ValueError("attribution_provider must be a non-empty name")


class ToolContext(TypedDict):
    """JSON context passed to an entrypoint in its child process."""

    call_id: str
    workspace: str
    campaign_id: str
    resource_paths: Mapping[str, str]
    environment: Mapping[str, str]
    config: Mapping[str, Any]


@dataclass(frozen=True)
class ToolDescriptor:
    name: str
    description: str
    input_schema: Mapping[str, Any]
    output_schema: Mapping[str, Any]
    api_version: int
    identity: str
    attribution_provider: str | None = None


@dataclass(frozen=True)
class ToolReceipt:
    call_id: str
    tool: str
    status: str
    implementation_identity: str
    resource_identity: str
    duration_ms: int


@dataclass(frozen=True)
class ToolResult:
    schema: str
    data: Any
    receipt: Mapping[str, Any]
    references: tuple[Mapping[str, Any], ...]
    coverage: Mapping[str, str]
    diagnostics: tuple[Mapping[str, str], ...]


@runtime_checkable
class BoundAgentResources(Protocol):
    """Borrowed, target-bound operations; calls return finite JSON projections."""

    def catalog(self) -> list[Mapping[str, Any]]: ...
    def skill_catalog(self) -> list[Mapping[str, Any]]: ...
    def call_result(
        self,
        name: str,
        request: Any,
        workspace: Path | None = None,
        *,
        cwd: Path | None = None,
    ) -> Mapping[str, Any]: ...
    def instructions(self, phase: str, **values: str) -> str: ...
    def environment(
        self, workspace: Path | None = None, campaign_id: str | None = None
    ) -> Mapping[str, str]: ...
    def install(self, workspace: Path | None = None) -> Any: ...
    def resource_snapshot(self) -> Mapping[str, Any]: ...
    def call_evidence(self, call_id: str) -> Mapping[str, Any]: ...
    def validate_declaration(
        self, provider: str, declaration: Mapping[str, Any]
    ) -> Mapping[str, Any]: ...
    def artifacts(self) -> list[Mapping[str, Any]]: ...


@runtime_checkable
class AgentResources(Protocol):
    """Resource service shared by Agent, Workflow and bootstrap consumers."""

    def freeze(self) -> Mapping[str, Any]: ...
    def bind(
        self,
        workspace: Path,
        campaign_id: str,
        base_selection: Mapping[str, Any] | None = None,
    ) -> BoundAgentResources: ...
    def restore_bound(
        self,
        binding: Mapping[str, Any],
        base_selection: Mapping[str, Any] | None = None,
    ) -> BoundAgentResources: ...


__all__ = [
    "AgentResources",
    "BoundAgentResources",
    "ToolContext",
    "ToolDescriptor",
    "ToolReceipt",
    "ToolResult",
    "ToolSpec",
]
