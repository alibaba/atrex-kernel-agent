"""Launch manifests reference Core profiles and select registered startup seams."""
from __future__ import annotations

import json
import importlib
import os
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any, Mapping, Sequence

from aka.core.boot import compose, token_table
from aka.core.composition import ResolvedComposition, substitute
from aka.core.errors import CompositionError
from aka.core.keys import ServiceKey


@dataclass(frozen=True)
class Selection:
    composition: ResolvedComposition
    target: str
    tokens: tuple[ServiceKey, ...]
    required_services: tuple[str, ...]
    resources: tuple[tuple[Path, tuple[str, ...]], ...]
    submodules: Mapping[str, Mapping[str, str]] = field(default_factory=dict)
    bindings: Mapping[str, str] = field(default_factory=dict)
    environment: Mapping[str, str | None] = field(default_factory=dict)
    token_definitions: Mapping[str, str] = field(default_factory=dict)
    continuation_version: int = 1


def load_token_definitions(definitions: Mapping[str, str]) -> dict[str, ServiceKey]:
    """Load declared token objects without applying their plugins."""
    if not isinstance(definitions, dict):
        raise ValueError("token_definitions must map service names to module:attribute references")
    references = {}
    for name, reference in definitions.items():
        if not isinstance(name, str) or not isinstance(reference, str):
            raise ValueError("token_definitions must map service names to module:attribute references")
        module, separator, attribute = reference.partition(":")
        if (not separator or not attribute.isidentifier() or not module
                or any(not part.isidentifier() for part in module.split("."))):
            raise ValueError(f"invalid token definition reference: {reference}")
        references[name] = (module, attribute)
    result = {}
    for name, (module, attribute) in references.items():
        try:
            token = getattr(importlib.import_module(module), attribute)
        except (ImportError, AttributeError) as exc:
            raise ValueError(f"token definition {name} is unavailable: {module}:{attribute}") from exc
        if not isinstance(token, ServiceKey) or token.name != name:
            raise ValueError(f"token definition {name} must reference a ServiceKey with the same name")
        result[name] = token
    return result


def registered_tokens(extra: Sequence[ServiceKey] = ()) -> dict[str, ServiceKey]:
    available = entry_points()
    entries = (available.select(group="aka.tokens") if hasattr(available, "select")
               else available.get("aka.tokens", ()))
    values = list(extra)
    for entry in entries:
        token = entry.load()
        if not isinstance(token, ServiceKey):
            raise TypeError(f"registered token {entry.name} must be a ServiceKey")
        values.append(token)
    return token_table(values)


def startup_token(name: str) -> ServiceKey:
    # Names choose implementations; protocol metadata remains defined in code.
    return ServiceKey(name, "Startup", module="aka.contracts.startup")


def resolve_profile(profile: Path, *, patch_files: Sequence[Path] = (),
                    patches: Sequence[Mapping[str, Any]] = (),
                    variables: Mapping[str, str] | None = None,
                    tokens: Sequence[ServiceKey] = ()) -> Selection:
    profile = Path(profile).expanduser().resolve()
    try:
        value = json.loads(profile.read_text())
    except (OSError, ValueError) as exc:
        raise CompositionError(str(profile), f"cannot read launch profile: {exc}") from exc
    fields = {"api_version", "composition", "target", "tokens", "required_services", "resources", "bindings", "environment", "doc"}
    version = value.get("api_version") if isinstance(value, dict) else None
    if version == 2:
        fields.add("token_definitions")
    if (type(version) is not int or version not in (1, 2) or set(value) - fields
            or (version == 2 and "token_definitions" not in value)):
        raise CompositionError(str(profile), "invalid launch profile fields or version")
    composition_name = value.get("composition")
    if (not isinstance(composition_name, str) or not composition_name
            or Path(composition_name).name != composition_name):
        raise CompositionError(str(profile), "composition must name a sibling Core profile")
    target = value.get("target")
    if not isinstance(target, str):
        raise CompositionError(str(profile), "target must name a Startup service")
    startup = startup_token(target)
    def names(key):
        items = value.get(key, [])
        if (not isinstance(items, list) or any(not isinstance(x, str) or not x for x in items)
                or len(items) != len(set(items))):
            raise CompositionError(str(profile), f"{key} must contain unique service names")
        return tuple(items)
    selected = names("tokens")
    required = tuple(dict.fromkeys((target, *names("required_services"))))
    definitions = value.get("token_definitions", {})
    try:
        if version == 2:
            if not isinstance(definitions, dict) or set(definitions) != set(selected) - {target}:
                raise ValueError("token_definitions must cover exactly the selected non-startup tokens")
            registry = load_token_definitions(definitions)
            if set(token_table(tokens)) - set(selected) - {target}:
                raise ValueError("explicit tokens must be selected by a version 2 profile")
            registry = token_table((*registry.values(), *tokens, startup))
        else:
            registry = token_table((*registered_tokens(tokens).values(), startup))
    except ValueError as exc:
        raise CompositionError(str(profile), str(exc)) from exc
    unknown = sorted((set(selected) | set(required)) - set(registry))
    if unknown:
        raise CompositionError(str(profile), f"unregistered tokens: {', '.join(unknown)}")
    seams = token_table((startup, *(registry[name] for name in selected), *tokens))
    if set(required) - set(seams):
        raise CompositionError(str(profile), "required services must have selected tokens")
    composition = compose(composition_name, profiles_dir=profile.parent / "compositions",
                          patch_files=patch_files, patches=patches, variables=variables)
    resources = value.get("resources", [])
    if not isinstance(resources, list):
        raise CompositionError(str(profile), "resources must be an array")
    resolved = []
    pinned = {}
    for resource in resources:
        if (not isinstance(resource, dict) or not {"root", "paths"} <= set(resource) or set(resource) - {"root", "paths", "submodules"}
                or not isinstance(resource["root"], str)
                or not isinstance(resource["paths"], list)
                or not resource["paths"]
                or any(not isinstance(x, str) or not x for x in resource["paths"])):
            raise CompositionError(str(profile), "resources require root and nonempty paths")
        path = Path(substitute(resource["root"], composition.variables, str(profile))).expanduser()
        if not path.is_absolute():
            path = profile.parent / path
        path = path.resolve()
        submodules = resource.get("submodules", {})
        if not isinstance(submodules, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in submodules.items()):
            raise CompositionError(str(profile), "submodules must map relative paths to pinned commits")
        if submodules:
            pinned[str(path)] = dict(submodules)
        resolved.append((path, tuple(resource["paths"])))
    bindings = value.get("bindings", {})
    if not isinstance(bindings, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in bindings.items()):
        raise CompositionError(str(profile), "bindings must map names to strings")
    bindings = {key: substitute(text, composition.variables, str(profile)) for key, text in bindings.items()}
    environment_names = names("environment")
    if any("=" in name or "\0" in name or name.startswith("AKA_LAUNCH_") for name in environment_names):
        raise CompositionError(str(profile), "invalid environment binding name")
    environment = {key: os.environ.get(key) for key in environment_names}
    return Selection(composition, target, tuple(seams.values()), required, tuple(resolved), pinned, bindings, environment,
                     dict(definitions), version)
