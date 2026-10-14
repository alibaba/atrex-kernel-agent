"""Composition: profiles stack bundles, patches replace rows.

A plugin tree is composed from ordered layers. A **bundle** is a list
of entry rows. A **profile** names the bundles it stacks plus its own patches. A **patch**
targets a row by id and replaces that row's whole config, or inserts, removes, or disables a
row.

Whole-config replacement is deliberate: a deep merge makes an override's effective value
depend on a bundle it cannot see, so an override must restate what it keeps.

Row order carries no load semantics. Ordering comes from ``inject``.

Interpolation is deliberately not an expression language. Only two literal reference forms
are recognized, and only inside the config fields a plugin lists in ``interpolate``. In
particular, references are data substitution, never executable expressions. Variable
names and values belong to the caller, not to Core.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from .errors import CompositionError

API_VERSION = 1

ROW_FIELDS = frozenset(
    (
        "id",
        "name",
        "config",
        "group",
        "disabled",
        "inject",
        "isolate",
        "required",
        "doc",
    )
)
PATCH_FIELDS = frozenset(
    ("id", "config", "disabled", "insert", "remove", "required", "inject", "isolate", "doc")
)
PROFILE_FIELDS = frozenset(("api_version", "bundles", "patches", "doc"))
BUNDLE_FIELDS = frozenset(("api_version", "entries", "doc"))
PATCH_FILE_FIELDS = frozenset(("api_version", "patches", "doc"))
#: What a group row may carry. A group is a readability and isolation device: it owns no module
#: and no config, so accepting those fields would silently discard them.
GROUP_FIELDS = frozenset(("id", "entries", "isolate", "disabled", "doc"))

_REFERENCE = re.compile(r"\$\{(env|aka):([A-Za-z_][A-Za-z0-9_]*)\}")
_ANY_REFERENCE = "${"


@dataclass(frozen=True)
class Row:
    """One composition entry."""

    id: str
    name: str
    config: Mapping[str, Any] = field(default_factory=dict)
    required: bool = False
    disabled: bool = False
    inject: tuple[str, ...] = ()
    isolate: tuple[str, ...] = ()
    group: str = ""
    doc: str = ""
    source: str = ""
    config_source: str = ""


@dataclass(frozen=True)
class ResolvedComposition:
    profile: str
    rows: tuple[Row, ...]
    layers: tuple[str, ...]
    variables: Mapping[str, str] = field(default_factory=dict)
    configs_resolved: bool = False

    @property
    def enabled(self) -> tuple[Row, ...]:
        return tuple(row for row in self.rows if not row.disabled)

    @property
    def required(self) -> frozenset[str]:
        return frozenset(row.id for row in self.rows if row.required)

    def row(self, entry_id: str) -> Row | None:
        for row in self.rows:
            if row.id == entry_id:
                return row
        return None


# -- reading -------------------------------------------------------------


def _read(path: Path) -> Mapping[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise CompositionError(str(path), f"cannot read: {exc}") from exc
    except ValueError as exc:
        raise CompositionError(str(path), f"invalid JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise CompositionError(str(path), "document must be a JSON object")
    version = document.get("api_version")
    # `type(...) is not int` rather than `!=`: Python makes True == 1 and 1.0 == 1, so a document
    # declaring `"api_version": true` would otherwise load as version 1.
    if type(version) is not int or version != API_VERSION:
        raise CompositionError(
            str(path), f"api_version must be {API_VERSION}, got {version!r}"
        )
    return document


def _check_fields(source: str, document: Mapping[str, Any], allowed: frozenset[str]) -> None:
    unknown = sorted(set(document) - allowed)
    if unknown:
        raise CompositionError(source, f"unsupported field(s): {', '.join(unknown)}")


def _flag(source: str, value: Any, label: str, default: bool = False) -> bool:
    """Read a JSON boolean strictly.

    ``bool("false")`` is ``True``, so coercion would let the string ``"false"`` invert
    ``disabled``, ``remove``, or ``required`` without a word.
    """
    if value is None:
        return default
    if not isinstance(value, bool):
        raise CompositionError(
            source, f"{label} must be true or false, got {value!r}"
        )
    return value


def _names(source: str, value: Any, label: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str) or not isinstance(value, list):
        raise CompositionError(source, f"{label} must be an array of strings")
    if any(not isinstance(item, str) or not item for item in value):
        raise CompositionError(source, f"{label} contains a non-string entry")
    return tuple(value)


def _row(source: str, entry: Any, group: str = "", inherited_isolate: tuple[str, ...] = ()) -> Row:
    if not isinstance(entry, dict):
        raise CompositionError(source, "entry must be a JSON object")
    _check_fields(source, entry, ROW_FIELDS)
    entry_id = entry.get("id")
    if not isinstance(entry_id, str) or not entry_id:
        raise CompositionError(source, f"entry needs a string id, got {entry_id!r}")
    module = entry.get("name")
    if not isinstance(module, str) or not module:
        raise CompositionError(
            source, f'entry "{entry_id}" needs a module path in "name"'
        )
    config = entry.get("config", {})
    if not isinstance(config, dict):
        raise CompositionError(source, f'entry "{entry_id}" config must be an object')
    return Row(
        id=entry_id,
        name=module,
        config=config,
        required=_flag(source, entry.get("required"), f'entry "{entry_id}" required'),
        disabled=_flag(source, entry.get("disabled"), f'entry "{entry_id}" disabled'),
        inject=_names(source, entry.get("inject"), f'entry "{entry_id}" inject'),
        isolate=tuple(
            dict.fromkeys(
                inherited_isolate
                + _names(source, entry.get("isolate"), f'entry "{entry_id}" isolate')
            )
        ),
        group=group or str(entry.get("group", "") or ""),
        doc=str(entry.get("doc", "") or ""),
        source=source,
        config_source=source if config else "",
    )


def _flatten(source: str, entries: Any) -> list[Row]:
    """Read a bundle's entries, flattening groups.

    A group row carries ``entries`` plus an optional ``isolate``; its children are flattened
    with the group's id recorded and its isolation propagated. Grouping is therefore a
    readability and isolation device, not a separate mount level.
    """
    if not isinstance(entries, list):
        raise CompositionError(source, "entries must be an array")
    rows: list[Row] = []
    for entry in entries:
        if isinstance(entry, dict) and "entries" in entry:
            _check_fields(source, entry, GROUP_FIELDS)
            group_id = entry.get("id")
            if not isinstance(group_id, str) or not group_id:
                raise CompositionError(source, "group needs a string id")
            isolate = _names(source, entry.get("isolate"), f'group "{group_id}" isolate')
            disabled = _flag(
                source, entry.get("disabled"), f'group "{group_id}" disabled'
            )
            for child in _flatten(source, entry["entries"]):
                rows.append(
                    replace(
                        child,
                        group=child.group or group_id,
                        isolate=tuple(dict.fromkeys(isolate + child.isolate)),
                        disabled=child.disabled or disabled,
                    )
                )
            continue
        rows.append(_row(source, entry))
    return rows


# -- layering ------------------------------------------------------------


def _apply_patch_document(
    source: str, rows: list[Row], document: Mapping[str, Any]
) -> None:
    _check_fields(source, document, PATCH_FILE_FIELDS)
    patches = document.get("patches", [])
    if not isinstance(patches, list):
        raise CompositionError(source, "patches must be an array")
    updated = list(rows)
    for patch in patches:
        _apply_patch(source, updated, patch)
    rows[:] = updated


def _apply_patch(source: str, rows: list[Row], patch: Any) -> None:
    if not isinstance(patch, dict):
        raise CompositionError(source, "patch must be a JSON object")
    _check_fields(source, patch, PATCH_FIELDS)
    entry_id = patch.get("id")
    if not isinstance(entry_id, str) or not entry_id:
        raise CompositionError(source, f"patch needs a string id, got {entry_id!r}")
    index = next((i for i, row in enumerate(rows) if row.id == entry_id), None)
    insert = patch.get("insert")
    if "insert" in patch and not isinstance(insert, dict):
        raise CompositionError(source, f'patch "{entry_id}" insert must be an object')
    remove = _flag(source, patch.get("remove"), f'patch "{entry_id}" remove')
    if "insert" in patch:
        conflicts = sorted(set(patch) - {"id", "insert"})
        if conflicts:
            raise CompositionError(source, f'patch "{entry_id}" insert conflicts with {", ".join(conflicts)}')
    if remove:
        conflicts = sorted(set(patch) - {"id", "remove"})
        if conflicts:
            raise CompositionError(source, f'patch "{entry_id}" remove conflicts with {", ".join(conflicts)}')
    if isinstance(insert, dict) and "id" in insert:
        # Otherwise the inserted row's id could disagree with the patch target and slip past the
        # duplicate-id check the bundle layer performs.
        raise CompositionError(
            source, f'patch "{entry_id}" insert must not carry its own id'
        )
    if index is None:
        if remove:
            # Removing what is already absent is the state the patch asked for.
            return
        if insert is None:
            raise CompositionError(
                source,
                f'patch targets unknown row "{entry_id}"; add "insert" to create it',
            )
        rows.append(_row(source, {"id": entry_id, **insert}))
        return
    if remove:
        del rows[index]
        return
    row = rows[index]
    updates: dict[str, Any] = {"source": source}
    if "config" in patch:
        config = patch["config"]
        if not isinstance(config, dict):
            raise CompositionError(source, f'patch "{entry_id}" config must be an object')
        # Whole-config replacement, never a deep merge.
        updates["config"] = config
        updates["config_source"] = source
    if "disabled" in patch:
        updates["disabled"] = _flag(
            source, patch["disabled"], f'patch "{entry_id}" disabled'
        )
    if "required" in patch:
        updates["required"] = _flag(
            source, patch["required"], f'patch "{entry_id}" required'
        )
    if "inject" in patch:
        updates["inject"] = _names(source, patch["inject"], f'patch "{entry_id}" inject')
    if "isolate" in patch:
        updates["isolate"] = _names(source, patch["isolate"], f'patch "{entry_id}" isolate')
    if "doc" in patch:
        updates["doc"] = str(patch["doc"] or "")
    if insert is not None:
        raise CompositionError(
            source, f'patch "{entry_id}" cannot insert a row that already exists'
        )
    rows[index] = replace(row, **updates)


def resolve(
    profiles_dir: Path,
    profile: str,
    *,
    patch_files: Sequence[Path] = (),
    patches: Sequence[Mapping[str, Any]] = (),
    variables: Mapping[str, str] | None = None,
) -> ResolvedComposition:
    """Compose one profile into a flat, ordered row list.

    Layers, in order: each bundle the profile lists, the profile's own patches, every
    ``--patch`` file, then programmatic patches (the CLI's flag-derived layer).
    """
    profile_path = profiles_dir / f"{profile}.json"
    document = _read(profile_path)
    _check_fields(str(profile_path), document, PROFILE_FIELDS)
    bundles = _names(str(profile_path), document.get("bundles"), "bundles")
    rows: list[Row] = []
    layers: list[str] = []
    for bundle in bundles:
        bundle_path = profiles_dir / "bundles" / f"{bundle}.json"
        bundle_document = _read(bundle_path)
        _check_fields(str(bundle_path), bundle_document, BUNDLE_FIELDS)
        source = f"bundle:{bundle}"
        layers.append(source)
        for row in _flatten(source, bundle_document.get("entries", [])):
            if any(existing.id == row.id for existing in rows):
                raise CompositionError(source, f'duplicate row id "{row.id}"')
            rows.append(row)

    profile_patches = document.get("patches", [])
    if not isinstance(profile_patches, list):
        raise CompositionError(str(profile_path), "patches must be an array")
    if profile_patches:
        layers.append(f"profile:{profile}")
        for patch in profile_patches:
            _apply_patch(f"profile:{profile}", rows, patch)

    for path in patch_files:
        source = f"patch:{path}"
        layers.append(source)
        _apply_patch_document(source, rows, _read(Path(path)))

    if patches:
        layers.append("cli")
        for patch in patches:
            _apply_patch("cli", rows, patch)

    resolved_variables = _resolve_variables(variables or {})
    return ResolvedComposition(
        profile=profile,
        rows=tuple(rows),
        layers=tuple(layers),
        variables=resolved_variables,
    )


def _resolve_variables(variables: Mapping[str, str]) -> Mapping[str, str]:
    for name in variables:
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise CompositionError("variables", f"invalid composition variable name: {name!r}")
    return {name: str(value) for name, value in variables.items()}


# -- interpolation -------------------------------------------------------


def substitute(text: str, variables: Mapping[str, str], source: str) -> str:
    def replacement(match: re.Match[str]) -> str:
        namespace, key = match.group(1), match.group(2)
        if namespace == "env":
            import os

            return os.environ.get(key, "")
        if key not in variables:
            raise CompositionError(
                source,
                f'unknown composition variable "{key}"; '
                f"supported: {', '.join(sorted(variables))}",
            )
        return str(variables.get(key, ""))

    if _ANY_REFERENCE in _REFERENCE.sub("", text):
        raise CompositionError(source, "unsupported reference syntax; use ${aka:name} or ${env:NAME}")
    return _REFERENCE.sub(replacement, text)


def apply_interpolation(
    config: Mapping[str, Any],
    *,
    allowlist: Sequence[str],
    variables: Mapping[str, str],
    source: str,
) -> Mapping[str, Any]:
    """Substitute references in allowlisted fields; reject them anywhere else.

    A field is identified by its dotted path through nested objects. Paths are compared as
    segment tuples, so a config key that literally contains a dot cannot masquerade as a nested
    path.
    """
    allowed = {tuple(entry.split(".")) for entry in allowlist}
    result = dict(config)
    for path, value in _walk(config):
        if not _contains_reference(value):
            continue
        if path not in allowed:
            raise CompositionError(
                source,
                f'field "{".".join(path)}" contains "{_ANY_REFERENCE}" but is not listed in '
                "the plugin's interpolate allowlist",
            )
        _assign(result, path, _substitute_value(value, variables, source))
    return result


def _contains_reference(value: Any) -> bool:
    if isinstance(value, str):
        return _ANY_REFERENCE in value
    if isinstance(value, (list, tuple)):
        return any(_contains_reference(item) for item in value)
    if isinstance(value, dict):
        # An object nested inside a list is not reachable as its own dotted path, so its
        # references belong to the field that contains it. Missing this is how an
        # uninterpolated expression reaches a plugin as literal text.
        return any(_contains_reference(item) for item in value.values())
    return False


def _substitute_value(value: Any, variables: Mapping[str, str], source: str) -> Any:
    if isinstance(value, str):
        return substitute(value, variables, source)
    if isinstance(value, (list, tuple)):
        return [_substitute_value(item, variables, source) for item in value]
    if isinstance(value, dict):
        return {
            key: _substitute_value(item, variables, source) for key, item in value.items()
        }
    return value


def _walk(
    config: Mapping[str, Any], prefix: tuple[str, ...] = ()
) -> list[tuple[tuple[str, ...], Any]]:
    """Every addressable config field, as a path of segments.

    Nested objects are addressed per key; anything else -- including a list, and any object
    inside that list -- belongs to the field that holds it.
    """
    found: list[tuple[tuple[str, ...], Any]] = []
    for key, value in config.items():
        path = prefix + (key,)
        if isinstance(value, dict):
            found.extend(_walk(value, path))
        else:
            found.append((path, value))
    return found


def _assign(config: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    cursor = config
    for segment in path[:-1]:
        child = cursor.get(segment)
        if not isinstance(child, dict):  # pragma: no cover - _walk only yields real paths
            raise CompositionError(
                "interpolation", f'cannot write "{".".join(path)}": {segment} is not an object'
            )
        child = dict(child)
        cursor[segment] = child
        cursor = child
    cursor[path[-1]] = value


# -- dumping -------------------------------------------------------------


def dump(composition: ResolvedComposition, *, fmt: str = "text") -> str:
    if fmt == "json":
        return json.dumps(
            {
                "api_version": API_VERSION,
                "profile": composition.profile,
                "layers": list(composition.layers),
                "variables": dict(composition.variables),
                "entries": [
                    {
                        "id": row.id,
                        "name": row.name,
                        "config": dict(row.config),
                        "required": row.required,
                        "disabled": row.disabled,
                        "inject": list(row.inject),
                        "isolate": list(row.isolate),
                        "group": row.group,
                        "doc": row.doc,
                        "source": row.source,
                        "config_source": row.config_source,
                    }
                    for row in composition.rows
                ],
            },
            indent=2,
            sort_keys=True,
        )
    if fmt != "text":
        raise CompositionError("dump", f"unsupported format: {fmt}")
    lines = [
        f"profile: {composition.profile}",
        f"layers: {' -> '.join(composition.layers)}",
        "",
    ]
    for row in composition.rows:
        flags = [flag for flag, on in (("required", row.required), ("disabled", row.disabled)) if on]
        header = f"[{row.id}] {row.name}"
        if flags:
            header += f"  ({', '.join(flags)})"
        lines.append(header)
        if row.group:
            lines.append(f"  group: {row.group}")
        if row.doc:
            lines.append(f"  doc: {row.doc}")
        lines.append(f"  from: {row.source}")
        if row.inject:
            lines.append(f"  inject: {', '.join(row.inject)}")
        if row.isolate:
            lines.append(f"  isolate: {', '.join(row.isolate)}")
        if row.config:
            lines.append(f"  config (set by {row.config_source or row.source}):")
            for key in sorted(row.config):
                lines.append(f"    {key} = {json.dumps(row.config[key], sort_keys=True)}")
        else:
            lines.append("  config: module Defaults only")
        lines.append("")
    return "\n".join(lines)


__all__ = [
    "API_VERSION",
    "ResolvedComposition",
    "Row",
    "apply_interpolation",
    "dump",
    "resolve",
    "substitute",
]
