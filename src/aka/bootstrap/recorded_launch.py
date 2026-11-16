"""Inspect recorded launch code without assembling services or reading data roots."""
from __future__ import annotations

import json
import math
from copy import deepcopy
from dataclasses import dataclass, fields, replace
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Mapping, Sequence

from aka.core.boot import token_table
from aka.core.composition import ResolvedComposition, Row
from aka.core.declaration import declare
from aka.core.errors import CompositionError
from aka.core.keys import ServiceKey
from aka.core.loader import import_plugin
from aka.core.lock import package_dir, resource_identity, snapshot
from aka.core.plugin_runtime.schema import validate_schema
from .profile import load_token_definitions, registered_tokens, startup_token


@dataclass(frozen=True)
class LaunchResourceRequirement:
    """Original data and Git requirements; their availability is not verified."""

    root: Path
    paths: tuple[str, ...]
    submodules: Mapping[str, str]
    identity: Mapping[str, str]


@dataclass(frozen=True)
class RecordedLaunchFacts:
    """Checked recorded code/config facts, never a runnable selection."""

    payload_digest: str
    target: str
    composition: ResolvedComposition
    required_services: tuple[str, ...]
    token_identity: tuple[ServiceKey, ...]
    resources: tuple[LaunchResourceRequirement, ...]
    python_paths: tuple[Path, ...]


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate recorded field")
        result[key] = value
    return result


def _constant(value):
    raise ValueError("non-finite recorded number")


def _number(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("non-finite recorded number")
    return number


def _fields(value, names):
    if not isinstance(value, dict) or set(value) != set(names):
        raise ValueError("invalid recorded fields")


def _text(value, *, empty=False):
    if not isinstance(value, str) or (not empty and not value) or "\0" in value:
        raise ValueError("invalid recorded text")
    return value


def _strings(value, *, unique=False):
    if not isinstance(value, list):
        raise ValueError("invalid recorded sequence")
    result = tuple(_text(item) for item in value)
    if unique and len(set(result)) != len(result):
        raise ValueError("duplicate recorded sequence member")
    return result


def _relative(value):
    value = _text(value)
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("recorded resource path is outside its root")
    return value


def _absolute(value):
    path = Path(_text(value))
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("invalid recorded absolute path")
    return path


def _hex(value, length):
    if (not isinstance(value, str) or len(value) != length
            or any(char not in "0123456789abcdef" for char in value)):
        raise ValueError("invalid recorded identity")
    return value


def _resource(value):
    _fields(value, ("root", "paths", "submodules", "identity"))
    root = _absolute(value["root"])
    paths = tuple(_relative(path) for path in _strings(value["paths"]))
    pins, identity = value["submodules"], value["identity"]
    if not isinstance(pins, dict) or not isinstance(identity, dict):
        raise ValueError("invalid recorded resource identity")
    for name, commit in pins.items():
        _relative(name)
        if name == ".":
            raise ValueError("invalid pinned submodule path")
        _hex(commit, 40)
    for name, value in identity.items():
        if name.startswith("gitlink:"):
            path = _relative(name[len("gitlink:"):])
            if pins.get(path) != _hex(value, 40):
                raise ValueError("recorded submodule identity mismatch")
        else:
            _relative(name)
            _hex(value, 64)
    if any(identity.get("gitlink:" + name) != commit for name, commit in pins.items()):
        raise ValueError("missing recorded submodule identity")
    return LaunchResourceRequirement(root, paths, MappingProxyType(dict(pins)),
                                     MappingProxyType(dict(identity)))


def _composition(value):
    _fields(value, ("profile", "rows", "layers", "variables"))
    profile = _text(value["profile"])
    layers = _strings(value["layers"])
    variables = value["variables"]
    if (not isinstance(variables, dict)
            or any(not isinstance(key, str) or not isinstance(item, str)
                   for key, item in variables.items())):
        raise ValueError("invalid recorded composition variables")
    for key, item in variables.items():
        _text(key)
        _text(item, empty=True)
    if not isinstance(value["rows"], list):
        raise ValueError("invalid recorded rows")
    rows = []
    for item in value["rows"]:
        _fields(item, (field.name for field in fields(Row)))
        for name in ("id", "name", "group", "doc", "source", "config_source"):
            _text(item[name], empty=name not in ("id", "name"))
        if (any(not part.isidentifier() for part in item["name"].split("."))
                or not isinstance(item["config"], dict)
                or type(item["disabled"]) is not bool or type(item["required"]) is not bool):
            raise ValueError("invalid recorded row")
        rows.append(Row(**{**item, "inject": _strings(item["inject"]),
                          "isolate": _strings(item["isolate"])}))
    if len({row.id for row in rows}) != len(rows):
        raise ValueError("duplicate recorded row")
    return ResolvedComposition(profile, tuple(rows), layers, dict(variables), configs_resolved=True)


def _record(payload):
    if not isinstance(payload, str):
        raise ValueError("recorded payload must be text")
    value = json.loads(payload, object_pairs_hook=_object, parse_constant=_constant, parse_float=_number)
    if not isinstance(value, dict):
        raise ValueError("invalid recorded selection")
    version = value.get("version")
    names = {"version", "target", "required_services", "composition", "identity",
             "resources", "host", "environment"}
    if type(version) is not int or version not in (1, 2):
        raise ValueError("unsupported recorded selection format")
    if version == 2:
        names.update(("token_definitions", "token_definition_identity"))
    if "python_paths" in value:
        names.add("python_paths")
    _fields(value, names)
    environment = value["environment"]
    if (not isinstance(environment, dict) or any(
            not isinstance(key, str) or not key or "=" in key or "\0" in key
            or key.startswith("AKA_LAUNCH_")
            or (item is not None and (not isinstance(item, str) or "\0" in item))
            for key, item in environment.items())):
        raise ValueError("invalid recorded launch environment")
    composition = _composition(value["composition"])
    startup = startup_token(_text(value["target"]))
    required = _strings(value["required_services"], unique=True)
    if not isinstance(value["resources"], list):
        raise ValueError("invalid recorded resources")
    resources = tuple(_resource(item) for item in value["resources"])
    python_paths = tuple(_absolute(item) for item in _strings(value.get("python_paths", []), unique=True))
    # This is lexical binding only. Presence, symlinks and bytes remain requirements.
    for path in python_paths:
        if not any(path == resource.root or any(path.is_relative_to(resource.root / member)
                   for member in resource.paths) for resource in resources):
            raise ValueError("recorded python path is outside declared resources")
    identity = value["identity"]
    if not isinstance(identity, dict) or not isinstance(identity.get("tokens"), list):
        raise ValueError("invalid recorded token identity")
    selected = []
    for item in identity["tokens"]:
        _fields(item, ("name", "definition", "cardinality", "module", "protocol_version"))
        for name in ("name", "definition", "cardinality", "module"):
            _text(item[name], empty=name == "module")
        selected.append(ServiceKey(**item))
    token_table(selected)
    if len({token.name for token in selected}) != len(selected):
        raise ValueError("duplicate recorded token")
    if startup not in selected or set(required) - {token.name for token in selected}:
        raise ValueError("recorded required service is unavailable")
    return value, composition, startup, tuple(selected), required, resources, python_paths


def inspect_recorded_launch(payload: str, *,
                            tokens: Sequence[ServiceKey] = ()) -> RecordedLaunchFacts:
    """Check recorded metadata and available code; do not apply recorded runtime.

    Declared data/Git/import roots are returned as requirements without reading
    them. Normal strict continuation restoration is still required before run.
    Matching declarations and config schemas use the caller's current runtime;
    availability callbacks, providers, tools and plugin apply are not invoked.
    """
    from .continuation import digest, encode, token_definition_identity

    try:
        value, composition, startup, selected, required, resources, python_paths = _record(payload)
        host = {package: resource_identity(package_dir(package), (".",))
                for package in ("aka.core", "aka.bootstrap", "aka.contracts")}
        if encode(host) != encode(value["host"]):
            raise ValueError("recorded host identity mismatch")
        if value["version"] == 2:
            definitions = value["token_definitions"]
            if (not isinstance(definitions, dict)
                    or set(definitions) != {token.name for token in selected} - {startup.name}):
                raise ValueError("recorded token definitions mismatch")
            # Validate reference syntax before resolving any definition module.
            for reference in definitions.values():
                if not isinstance(reference, str):
                    raise ValueError("invalid recorded token reference")
                module, separator, attribute = reference.partition(":")
                if (not separator or not attribute.isidentifier() or not module
                        or any(not part.isidentifier() for part in module.split("."))):
                    raise ValueError("invalid recorded token reference")
            if encode(token_definition_identity(definitions)) != encode(value["token_definition_identity"]):
                raise ValueError("recorded token definition identity mismatch")
            registry = token_table((*load_token_definitions(definitions).values(), *tokens, startup))
        else:
            registry = token_table((*registered_tokens(tokens).values(), startup))
        if any(registry.get(token.name) is None
               or replace(registry[token.name], doc="") != token for token in selected):
            raise ValueError("recorded token is unavailable or changed")
        # Reuse Core's recorded effective-config/code identity. Unlike live
        # freeze, no unavailable enabled row or availability callback is allowed.
        seams = token_table(selected)
        declarations = {row.id: None if row.disabled else
                        declare(import_plugin(row.name), known_seams=seams).with_extra_inject(row.inject)
                        for row in composition.rows}
        recorded_configs = {row.id: row.config for row in composition.enabled}
        current = snapshot(composition, declarations, selected, effective_configs=recorded_configs)
        if encode(current) != encode(value["identity"]):
            raise ValueError("recorded implementation or config identity mismatch")
        for row in composition.enabled:
            declaration = declarations[row.id]
            if declaration.config_schema is None:
                if row.config:
                    raise ValueError("recorded plugin declares no config schema")
            else:
                validate_schema(declaration.config_schema, row.config, row.id + ".config")
        # Custom validate_config may inspect missing data or executable roots;
        # the unchanged strict capture/restore rechecks it before actual use.
        # Core mappings/configs are independent copies; resource mappings are read-only.
        effective = replace(composition, rows=tuple(replace(row, config=deepcopy(dict(row.config)))
                                                    for row in composition.rows),
                            variables=MappingProxyType(dict(composition.variables)))
        return RecordedLaunchFacts(digest(payload), startup.name, effective, required,
                                   selected, resources, python_paths)
    except Exception:
        # Config/import errors can contain runtime values. Do not expose them as
        # an environment projection through this facts-only inspection seam.
        raise CompositionError("recorded-launch", "recorded launch metadata or available code is invalid") from None
