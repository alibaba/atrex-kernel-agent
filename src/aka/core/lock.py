"""Versioned, declaration-driven composition identity locks."""
from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import tempfile
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .composition import ResolvedComposition, apply_interpolation
from .declaration import PluginDeclaration
from .errors import CompositionError
from .keys import ServiceKey

LOCK_VERSION = 2
LOCK_NAME = "composition.json"
STATE_DIR = ".atrex_plugins"


def package_dir(module: str) -> Path | None:
    try:
        spec = importlib.util.find_spec(module)
    except (ImportError, ValueError):
        return None
    if spec is None:
        return None
    locations = tuple(spec.submodule_search_locations or ())
    return Path(locations[0]) if locations else Path(spec.origin).parent if spec.origin else None


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _files(package: str, names: Iterable[str]) -> dict[str, str]:
    root = package_dir(package)
    if root is None:
        raise CompositionError(package, "identity package is unavailable")
    digests = {}
    for name in names:
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise CompositionError(package, f"identity file is missing: {name}")
        digests[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return digests


def _identity_names(module: str, *, exclude_metadata: bool = False) -> tuple[str, ...]:
    try:
        spec = importlib.util.find_spec(module)
    except (ImportError, ValueError) as exc:
        raise CompositionError(module, "identity implementation is unavailable") from exc
    if spec is None:
        raise CompositionError(module, "identity implementation is unavailable")
    locations = tuple(spec.submodule_search_locations or ())
    if not locations:
        if not spec.origin or not Path(spec.origin).is_file():
            raise CompositionError(module, "identity implementation has no source file")
        return (Path(spec.origin).name,)
    root = Path(locations[0])
    return tuple(
        path.relative_to(root).as_posix()
        for path in sorted(root.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts
        and path.name != ".DS_Store" and path.suffix not in {".pyc", ".pyo"}
        and (not exclude_metadata or not any(
            part.endswith((".dist-info", ".egg-info")) for part in path.parts
        ))
    )


def resource_identity(root: Path, paths: Iterable[str], *, submodules: Mapping[str, str] | None = None) -> dict[str, str]:
    """Hash explicitly selected resources, including tree membership, within a root."""
    root = Path(root).resolve()
    result = {}
    excluded = set()
    gitlinks = None
    if submodules and (root / ".git").exists():
        recorded = subprocess.run(["git", "-C", str(root), "ls-files", "--stage", "-z"],
                                  capture_output=True, text=True)
        if recorded.returncode:
            raise CompositionError(str(root), "cannot read checkout gitlinks")
        gitlinks = {}
        for record in recorded.stdout.split("\0"):
            if not record:
                continue
            metadata, name = record.split("\t", 1)
            mode, commit, stage = metadata.split()
            if mode == "160000" and stage == "0":
                gitlinks[name] = commit
    for name, commit in (submodules or {}).items():
        path = (root / name).absolute()
        if (not path.resolve().is_relative_to(root) or not isinstance(commit, str)
                or len(commit) != 40 or any(x not in "0123456789abcdef" for x in commit)):
            raise CompositionError(str(root), f"invalid pinned submodule: {name}")
        if gitlinks is not None and gitlinks.get(name) != commit:
            raise CompositionError(str(root), f"checkout gitlink differs from pinned implementation: {name}")
        excluded.add(path)
        # Uninitialized gitlinks and clean pinned checkouts describe the same
        # implementation. Automatic initialization does not change selection.
        if path.exists() and any(path.iterdir()):
            if not (path / ".git").exists():
                raise CompositionError(str(root), f"submodule has content without Git identity: {name}")
            def git(*args):
                result = subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True)
                if result.returncode:
                    raise CompositionError(str(root), f"cannot validate submodule: {name}")
                return result.stdout.strip()
            if git("rev-parse", "HEAD") != commit or git("status", "--porcelain", "--untracked-files=all"):
                raise CompositionError(str(root), f"submodule differs from pinned implementation: {name}")
        result[f"gitlink:{name}"] = commit
    for name in paths:
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.exists():
            raise CompositionError(str(root), f"identity resource is missing or outside root: {name}")
        if path.is_file():
            files = (path,)
        else:
            files = []
            for directory, dirs, names in os.walk(path):
                base = Path(directory)
                dirs[:] = sorted(name for name in dirs if name not in {".git", "__pycache__"}
                                 and base / name not in excluded)
                files.extend(base / name for name in sorted(names))
        for item in files:
            if any(item == excluded_path or excluded_path in item.parents for excluded_path in excluded):
                continue
            if (not item.is_file() or "__pycache__" in item.parts or ".git" in item.parts or item.name == ".DS_Store"
                    or item.suffix in {".pyc", ".pyo"}):
                continue
            if not item.resolve().is_relative_to(root):
                raise CompositionError(str(root), f"identity resource escapes root: {item}")
            result[item.relative_to(root).as_posix()] = hashlib.sha256(item.read_bytes()).hexdigest()
    return result


def snapshot(
    composition: ResolvedComposition,
    declarations: Mapping[str, PluginDeclaration | None],
    tokens: Iterable[ServiceKey] = (),
    *,
    effective_configs: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    entries = []
    for row in composition.rows:
        declaration = declarations.get(row.id)
        if declaration is None:
            entries.append({
                "id": row.id,
                "module": row.name,
                "enabled": not row.disabled,
                "required": row.required,
                "group": row.group,
                "isolate": list(row.isolate),
                "inject": list(row.inject),
                "config_digest": _digest(row.config),
                "schema_digest": "unavailable",
                "resume_policy": "strict",
                "identity_files": {},
            })
            continue
        package = (
            declaration.module.rsplit(".", 1)[0]
            if declaration.module.endswith(".plugin")
            else declaration.module
        )
        if declaration.identity_files:
            identity_root, names = package, declaration.identity_files
        else:
            identity_root = declaration.module
            names = _identity_names(identity_root)
        identities = {identity_root: _files(identity_root, names)}
        for dependency in declaration.identity_packages:
            names = _identity_names(dependency, exclude_metadata=True)
            identities[dependency] = _files(dependency, names)
        effective_config = (
            dict(effective_configs[row.id])
            if effective_configs is not None and row.id in effective_configs
            else {
                **declaration.defaults,
                **dict(
                    apply_interpolation(
                        row.config,
                        allowlist=declaration.interpolate,
                        variables=composition.variables,
                        source=f"entry:{row.id}",
                    )
                ),
            }
        )
        entries.append({
            "id": row.id,
            "module": row.name,
            "enabled": not row.disabled,
            "required": row.required,
            "group": row.group,
            "isolate": list(row.isolate),
            "inject": list(row.inject),
            "config_digest": _digest(effective_config),
            "schema_digest": _digest(declaration.config_schema),
            "resume_policy": declaration.resume_policy,
            "identity_files": identities,
            "observational_fields": list(declaration.observational_fields),
        })
    framework = {}
    for package in ("aka.core",):
        root = package_dir(package)
        names = tuple(path.relative_to(root).as_posix() for path in sorted(root.rglob("*.py"))) if root else ()
        framework[package] = _files(package, names) if root else {}
    aka_root = package_dir("aka")
    framework["aka"] = _files("aka", ("__init__.py",)) if aka_root else {}
    try:
        version = importlib.metadata.version("atrex-aka-framework")
    except importlib.metadata.PackageNotFoundError:
        version = "source"
    return {
        "lock_version": LOCK_VERSION,
        "profile": composition.profile,
        "variables_digest": _digest(composition.variables),
        "core_version": version,
        "core_digest": _digest(framework),
        "tokens": [
            {
                "name": token.name,
                "module": token.module,
                "definition": token.definition,
                "cardinality": token.cardinality,
                "protocol_version": token.protocol_version,
            }
            for token in sorted(tokens, key=lambda item: item.name)
        ],
        "entries": entries,
    }


def lock_path(workspace: Path) -> Path:
    return workspace / STATE_DIR / LOCK_NAME


def read(workspace: Path) -> Mapping[str, Any] | None:
    path = lock_path(workspace)
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise CompositionError(str(path), f"unreadable composition lock: {exc}") from exc
    version = value.get("lock_version") if isinstance(value, dict) else None
    if version != LOCK_VERSION:
        raise CompositionError(str(path), f"unsupported composition lock version: {version}")
    entries = value.get("entries")
    if not isinstance(entries, list) or any(
        not isinstance(item, dict) or not isinstance(item.get("id"), str)
        for item in entries
    ):
        raise CompositionError(str(path), "composition lock entries must be objects with string ids")
    return value


@dataclass(frozen=True)
class _Difference:
    entry_id: str | None
    fields: tuple[str, ...]
    kind: str

    def render(self) -> str:
        if self.entry_id is None:
            return f"{self.fields[0]}: changed"
        return f"{self.entry_id}: fields changed {','.join(self.fields)}"


def _difference_records(
    stored: Mapping[str, Any], current: Mapping[str, Any],
) -> tuple[_Difference, ...]:
    records = []
    old = {entry["id"]: entry for entry in stored.get("entries", [])}
    new = {entry["id"]: entry for entry in current.get("entries", [])}
    for entry_id in sorted(set(old) | set(new)):
        if old.get(entry_id) != new.get(entry_id):
            before, after = old.get(entry_id, {}), new.get(entry_id, {})
            fields = tuple(sorted(
                key for key in set(before) | set(after)
                if key not in before or key not in after or before[key] != after[key]
            ))
            kind = "added" if entry_id not in old else "removed" if entry_id not in new else "changed"
            records.append(_Difference(entry_id, fields, kind))
    for field in ("core_version", "core_digest", "tokens", "variables_digest"):
        if stored.get(field) != current.get(field):
            records.append(_Difference(None, (field,), "global"))
    return tuple(records)


def differences(stored: Mapping[str, Any], current: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(record.render() for record in _difference_records(stored, current))


def reconcile(workspace: Path, current: Mapping[str, Any], *, mode: str) -> tuple[str, ...]:
    path, stored = lock_path(workspace), read(workspace)
    if mode == "create":
        if stored is not None:
            raise CompositionError(str(path), "create requires an unlocked workspace")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            descriptor, name = tempfile.mkstemp(prefix=".composition-", dir=path.parent)
            temporary = Path(name)
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(current, stream, indent=2, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary, path)
        except FileExistsError as exc:
            raise CompositionError(str(path), "composition lock appeared during create") from exc
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return ()
    if mode != "resume":
        raise ValueError("lock mode must be create or resume")
    if stored is None:
        raise CompositionError(str(path), "resume requires an existing version 2 lock")
    records = _difference_records(stored, current)
    old = {entry["id"]: entry for entry in stored.get("entries", [])}
    new = {entry["id"]: entry for entry in current.get("entries", [])}
    sensitive = []
    for record in records:
        if record.kind != "changed":
            sensitive.append(record)
            continue
        before, after = old[record.entry_id], new[record.entry_id]
        if (
            before.get("resume_policy", "strict") == "strict"
            or after.get("resume_policy", "strict") == "strict"
        ):
            sensitive.append(record)
            continue
        allowed = set(before.get("observational_fields", ()))
        allowed &= set(after.get("observational_fields", ()))
        if not set(record.fields) <= allowed:
            sensitive.append(record)
    if sensitive:
        raise CompositionError(
            str(path),
            "resume identity mismatch: " + "; ".join(record.render() for record in sensitive),
        )
    return tuple(record.render() for record in records)


__all__ = [
    "LOCK_NAME", "LOCK_VERSION", "differences", "lock_path", "package_dir",
    "read", "reconcile", "snapshot",
]
