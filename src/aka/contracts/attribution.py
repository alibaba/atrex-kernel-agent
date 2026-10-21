"""Public attribution declarations and provider contracts."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


def _json_object(value: Mapping[str, Any], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    try:
        return json.loads(json.dumps(dict(value), ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain finite JSON values") from exc


@dataclass(frozen=True)
class AttributionProviderSpec:
    name: str
    api_version: int
    extract_entrypoint: str
    validate_entrypoint: str
    artifacts_entrypoint: str
    config: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not re.fullmatch(
            r"[a-z][a-z0-9_-]*", self.name
        ):
            raise ValueError("attribution provider name must be a lowercase identifier")
        if self.api_version != 1:
            raise ValueError("unsupported attribution api_version")
        for label in (
            "extract_entrypoint",
            "validate_entrypoint",
            "artifacts_entrypoint",
        ):
            value = getattr(self, label)
            if not isinstance(value, str) or not re.fullmatch(
                r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*", value
            ):
                raise ValueError(f"{label} must be module:function")
        object.__setattr__(self, "config", _json_object(self.config, "config"))


@dataclass(frozen=True)
class SourceReference:
    provider: str
    call_id: str
    reference_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for key in ("provider", "call_id", "reference_id"):
            value = getattr(self, key)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{key} is required")
        if (
            len(self.provider) > 128
            or len(self.call_id) > 128
            or len(self.reference_id) > 2048
        ):
            raise ValueError("source reference identifiers exceed the size limit")
        metadata = _json_object(self.metadata, "metadata")
        if (
            len(
                json.dumps(metadata, ensure_ascii=False, allow_nan=False).encode(
                    "utf-8"
                )
            )
            > 16_384
        ):
            raise ValueError("source reference metadata exceeds the size limit")
        object.__setattr__(self, "metadata", metadata)


@dataclass(frozen=True)
class AttributionValidation:
    accepted: bool
    verification: str
    diagnostics: tuple[Mapping[str, str], ...] = ()


@dataclass(frozen=True)
class ArtifactDescriptor:
    path: str
    role: str
    schema: str
    required: bool
    coverage: str
    sha256: str
    provider: str


__all__ = [
    "ArtifactDescriptor",
    "AttributionProviderSpec",
    "AttributionValidation",
    "SourceReference",
]
