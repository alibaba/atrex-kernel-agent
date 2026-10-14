"""Application-neutral composition and plugin-tree startup."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from dataclasses import replace
from copy import deepcopy

from .composition import ResolvedComposition, apply_interpolation, resolve
from .errors import BootFailure
from .effects import call_sync
from .keys import ServiceKey
from .loader import BootReport, import_plugin, load
from .declaration import declare
from .lock import reconcile, snapshot
from .plugin_runtime.schema import validate_schema

PROFILES_DIR = Path(__file__).with_name("profiles")
DEFAULT_PROFILE = "default"

def validate(composition: ResolvedComposition) -> None:
    """Import declarations and validate effective config without applying plugins."""
    failures = []
    for row in composition.enabled:
        try:
            declaration = declare(import_plugin(row.name)).with_extra_inject(row.inject)
            _validate_plugin_config(declaration, row, composition.variables, resolved=composition.configs_resolved)
        except Exception as exc:
            failures.append((row.id, str(exc)))
    if failures:
        raise BootFailure(tuple(failures))


def token_table(tokens: Iterable[ServiceKey] = ()) -> dict[str, ServiceKey]:
    result: dict[str, ServiceKey] = {}
    for token in tokens:
        previous = result.get(token.name)
        if previous is not None and previous != token:
            raise ValueError(
                f'service token "{token.name}" metadata conflicts: {previous!r} != {token!r}'
            )
        result[token.name] = token
    return result


def compose(
    profile: str = DEFAULT_PROFILE,
    *,
    profiles_dir: Path = PROFILES_DIR,
    patch_files: Sequence[Path] = (),
    patches: Sequence[Mapping[str, Any]] = (),
    variables: Mapping[str, str] | None = None,
) -> ResolvedComposition:
    return resolve(
        profiles_dir,
        profile,
        patch_files=patch_files,
        patches=patches,
        variables=variables or {},
    )


def boot(
    composition: ResolvedComposition,
    *,
    tokens: Iterable[ServiceKey] = (),
    required_rows: Sequence[str] = (),
    required_services: Sequence[str] = (),
    stderr: Any = None,
    workspace: Path | None = None,
    lock_mode: str | None = None,
) -> BootReport:
    seams, declarations, effective_configs = _prepare(composition, tokens, required_rows, required_services)
    required_ids = composition.required | set(required_rows)
    if lock_mode is not None:
        if workspace is None:
            raise ValueError("workspace is required with lock_mode")
        lock_diffs = reconcile(
            workspace,
            snapshot(
                composition,
                declarations,
                seams.values(),
                effective_configs=effective_configs,
            ),
            mode=lock_mode,
        )
    else:
        lock_diffs = ()
    report = load(
        composition,
        seams=seams,
        required=tuple(required_ids),
        stderr=stderr,
    )
    missing = tuple(
        (name, "required service is unavailable in the root realm")
        for name in sorted(set(required_services))
        if report.root.realm.resolve(name) is None
    )
    if missing:
        report.dispose()
        raise BootFailure(missing)
    return replace(report, lock_differences=lock_diffs)


def _prepare(composition, tokens, required_rows=(), required_services=()):
    seams = token_table(tokens)
    unknown = sorted(set(required_services) - set(seams))
    if unknown:
        raise BootFailure(tuple((name, "required service has no registered token") for name in unknown))
    declarations = {}
    preflight_failures = []
    required_ids = {row.id for row in composition.rows if row.required} | set(required_rows)
    for row in composition.rows:
        if row.disabled:
            declarations[row.id] = None
            continue
        try:
            declarations[row.id] = declare(import_plugin(row.name), known_seams=seams).with_extra_inject(row.inject)
        except Exception as exc:
            declarations[row.id] = None
            if row.id in required_ids:
                preflight_failures.append((row.id, str(exc)))
    if preflight_failures:
        raise BootFailure(tuple(preflight_failures))
    config_failures = []
    effective_configs = {}
    for row in composition.enabled:
        declaration = declarations[row.id]
        if declaration is None:
            continue
        try:
            effective_configs[row.id] = _validate_plugin_config(
                declaration, row, composition.variables, resolved=composition.configs_resolved
            )
        except Exception as exc:
            config_failures.append((row.id, str(exc)))
    if config_failures:
        raise BootFailure(tuple(config_failures))
    collisions: dict[str, list[str]] = {}
    for row in composition.enabled:
        declaration = declarations[row.id]
        if declaration is None:
            continue
        for service in declaration.provide:
            if service not in row.isolate:
                collisions.setdefault(service, []).append(row.id)
    duplicates = tuple(
        (service, "single implementation is provided by rows " + ", ".join(rows))
        for service, rows in sorted(collisions.items())
        if len(rows) > 1 and seams[service].cardinality == "single"
    )
    if duplicates:
        raise BootFailure(duplicates)
    return seams, declarations, effective_configs


def freeze(composition: ResolvedComposition, *, tokens: Iterable[ServiceKey] = ()):
    """Capture effective configs and implementation identity without applying plugins.

    The returned composition no longer depends on environment interpolation or
    mutable plugin defaults. Hosts may serialize it for process reconstruction.
    """
    seams, declarations, configs = _prepare(composition, tokens)
    effective = replace(composition, rows=tuple(
        replace(row, config=deepcopy(dict(configs[row.id]))) if row.id in configs else row
        for row in composition.rows
    ), variables=dict(composition.variables), configs_resolved=True)
    identity = snapshot(effective, declarations, seams.values(), effective_configs=configs)
    return effective, identity


def _validate_plugin_config(
    declaration: Any, row: Any, variables: Mapping[str, str], *, resolved: bool = False
) -> Mapping[str, Any]:
    """Validate one row's effective config before lock reconciliation or plugin apply."""
    config = row.config if resolved else apply_interpolation(
        row.config,
        allowlist=declaration.interpolate,
        variables=variables,
        source=f"entry:{row.id}",
    )
    merged = dict(config) if resolved else {**declaration.defaults, **dict(config)}
    if declaration.config_schema is None and merged:
        raise ValueError("plugin declares no Config schema")
    if declaration.config_schema is not None:
        validate_schema(declaration.config_schema, merged, f"{row.id}.config")
    if declaration.config_validator is not None:
        call_sync(declaration.config_validator, merged)
    return merged


__all__ = ["DEFAULT_PROFILE", "PROFILES_DIR", "boot", "compose", "freeze", "token_table", "validate"]
