"""Serialize effective selections; validate them before any plugin setup in a child."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Mapping, Sequence

from aka.core.boot import freeze, token_table
from aka.core.composition import ResolvedComposition, Row
from aka.core.errors import CompositionError
from aka.core.keys import ServiceKey
from aka.core.lock import package_dir, resource_identity
from .profile import Selection, load_token_definitions, registered_tokens, startup_token

SELECTION_ENV = "AKA_LAUNCH_SELECTION"
DIGEST_ENV = "AKA_LAUNCH_DIGEST"


def encode(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def token_definition_identity(definitions) -> dict:
    """Pin declared token modules as code, including package resource membership."""
    identities = {}
    for name, reference in definitions.items():
        module = reference.split(":", 1)[0]
        spec = importlib.util.find_spec(module)
        if spec is None:
            raise ValueError(f"token definition {name} has no implementation identity")
        locations = tuple(spec.submodule_search_locations or ())
        if locations:
            root, paths = Path(locations[0]), (".",)
        elif spec.origin and Path(spec.origin).is_file():
            root, paths = Path(spec.origin).parent, (Path(spec.origin).name,)
        else:
            raise ValueError(f"token definition {name} has no implementation identity")
        identities[name] = resource_identity(root, paths)
    return identities


def capture(selection: Selection) -> tuple[Selection, str]:
    from .runtime import selection_runtime
    with selection_runtime(selection.environment, selection.python_paths):
        return _capture(selection)


def _capture(selection: Selection) -> tuple[Selection, str]:
    if type(selection.continuation_version) is not int or selection.continuation_version not in (1, 2):
        raise ValueError("unsupported selection format")
    if selection.continuation_version == 1 and selection.token_definitions:
        raise ValueError("version 1 selections cannot contain token_definitions")
    if selection.continuation_version == 2:
        if set(selection.token_definitions) != {token.name for token in selection.tokens} - {selection.target}:
            raise ValueError("token_definitions must cover exactly the selected non-startup tokens")
        declared = load_token_definitions(dict(selection.token_definitions))
        token_table((*declared.values(), *selection.tokens, startup_token(selection.target)))
    effective, identity = freeze(selection.composition, tokens=selection.tokens)
    resources = [{"root": str(root), "paths": list(paths), "submodules": dict(selection.submodules.get(str(root), {})),
                  "identity": resource_identity(root, paths, submodules=selection.submodules.get(str(root), {}))}
                 for root, paths in selection.resources]
    host = {package: resource_identity(package_dir(package), (".",))
            for package in ("aka.core", "aka.bootstrap", "aka.contracts")}
    value = {"version": selection.continuation_version, "target": selection.target,
             "required_services": list(selection.required_services),
             "bindings": dict(selection.bindings), "environment": dict(selection.environment),
             "composition": {"profile": effective.profile, "rows": [asdict(row) for row in effective.rows],
                             "layers": list(effective.layers), "variables": dict(effective.variables)},
             "identity": identity, "resources": resources, "host": host}
    if selection.continuation_version == 2:
        value["token_definitions"] = dict(selection.token_definitions)
        value["token_definition_identity"] = token_definition_identity(selection.token_definitions)
    if selection.python_paths:
        value["python_paths"] = [str(path) for path in selection.python_paths]
    return replace(selection, composition=effective), encode(value)


def read(environment: Mapping[str, str | None]) -> str | None:
    path, expected = environment.get(SELECTION_ENV), environment.get(DIGEST_ENV)
    if path is None and expected is None:
        return None
    if not path or not expected:
        raise CompositionError("continuation", "selection path and digest must travel together")
    try:
        payload = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise CompositionError("continuation", f"cannot read selection: {exc}") from exc
    if digest(payload) != expected:
        raise CompositionError("continuation", "selection digest mismatch")
    return payload


def restore(payload: str, *, tokens: Sequence[ServiceKey] = ()) -> Selection:
    from .runtime import selection_runtime
    try:
        value = json.loads(payload)
        environment = value["environment"]
        paths = value.get("python_paths", [])
        if (not isinstance(environment, dict) or any(not isinstance(key, str)
                or not key or "=" in key or "\0" in key or key.startswith("AKA_LAUNCH_")
                or (item is not None and (not isinstance(item, str) or "\0" in item))
                for key, item in environment.items()) or not isinstance(paths, list)
                or any(not isinstance(path, str) or not Path(path).is_absolute() for path in paths)):
            raise ValueError("invalid recorded launch runtime")
        for path in paths:
            directory = Path(path).resolve()
            if not directory.is_dir() or not any(directory == Path(item["root"]).resolve()
                    or any(directory.is_relative_to((Path(item["root"]) / name).resolve())
                           for name in item["paths"]) for item in value["resources"]):
                raise ValueError("recorded python path is outside declared resources")
    except (AttributeError, KeyError, OSError, TypeError, ValueError) as exc:
        raise CompositionError("continuation", f"cannot reconstruct selection: {exc}") from exc
    with selection_runtime(environment, paths):
        return _restore(payload, tokens=tokens)


def _restore(payload: str, *, tokens: Sequence[ServiceKey] = ()) -> Selection:
    try:
        value = json.loads(payload)
        version = value.get("version") if isinstance(value, dict) else None
        fields = {"version", "target", "required_services", "composition", "identity", "resources", "host", "bindings", "environment"}
        if version == 2:
            fields.update(("token_definitions", "token_definition_identity"))
        if "python_paths" in value:
            fields.add("python_paths")
        if type(version) is not int or version not in (1, 2) or set(value) != fields:
            raise ValueError("unsupported selection format")
        comp = value["composition"]
        rows = tuple(Row(**{**row, "inject": tuple(row["inject"]), "isolate": tuple(row["isolate"])})
                     for row in comp["rows"])
        composition = ResolvedComposition(comp["profile"], rows, tuple(comp["layers"]), comp["variables"], configs_resolved=True)
        startup = startup_token(value["target"])
        definitions = value.get("token_definitions", {})
        recorded_tokens = tuple(ServiceKey(**recorded) for recorded in value["identity"]["tokens"])
        if version == 2:
            if not isinstance(definitions, dict) or set(definitions) != {token.name for token in recorded_tokens} - {startup.name}:
                raise ValueError("token_definitions must cover exactly the selected non-startup tokens")
            registry = token_table((*load_token_definitions(definitions).values(), *tokens, startup))
        else:
            registry = token_table((*registered_tokens(tokens).values(), startup))
        selected = []
        for token in recorded_tokens:
            if registry.get(token.name) is None or replace(registry[token.name], doc="") != token:
                raise ValueError(f"token {token.name} is unavailable or changed")
            selected.append(token)
        resources = tuple((Path(item["root"]), tuple(item["paths"])) for item in value["resources"])
        selection = Selection(composition, value["target"], tuple(selected),
                              tuple(value["required_services"]), resources,
                              {item["root"]: item["submodules"] for item in value["resources"] if item["submodules"]},
                              value["bindings"], value["environment"], dict(definitions), version,
                              tuple(Path(path) for path in value.get("python_paths", [])))
        _, current = capture(selection)
        if json.loads(current) != value:
            raise ValueError("effective config, implementation or resource identity mismatch")
        return selection
    except (ImportError, AttributeError, KeyError, TypeError, ValueError) as exc:
        raise CompositionError("continuation", f"cannot reconstruct selection: {exc}") from exc

