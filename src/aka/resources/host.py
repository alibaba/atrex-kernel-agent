"""Scope-aware immutable snapshots, binding, installation and tool execution."""

from __future__ import annotations

import hashlib
import json
import os
import re
import selectors
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from aka.contracts.agent_instructions import (
    InstructionSpec,
    RenderedInstruction,
    instruction_identity,
)
from aka.contracts.agent_resources import AgentSkill, WorkspaceResourceContribution
from aka.contracts.attribution import AttributionProviderSpec, SourceReference
from aka.contracts.tools import ToolSpec


class ResourcesError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class SkillMountSpec:
    """A logical lowercase AgentSkill with a preserved native installation name."""

    skill: AgentSkill
    install_name: str
    optional: bool = False

    def __post_init__(self):
        if not isinstance(self.skill, AgentSkill):
            raise TypeError("skill must be an AgentSkill")
        if not isinstance(self.install_name, str) or not re.fullmatch(
            r"[A-Za-z][A-Za-z0-9_-]*", self.install_name
        ):
            raise ValueError("install_name must be a safe directory name")
        if not isinstance(self.optional, bool):
            raise TypeError("optional must be bool")


@dataclass(frozen=True)
class EnvironmentBinding:
    """Named, deterministic runtime environment values from the selected Scope."""

    name: str
    values: Mapping[str, str]

    def __post_init__(self):
        if not isinstance(self.name, str) or not re.fullmatch(
            r"[a-z][a-z0-9_-]*", self.name
        ):
            raise ValueError("environment binding name must be lowercase")
        if not isinstance(self.values, Mapping) or any(
            not isinstance(k, str)
            or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", k)
            or not isinstance(v, str)
            for k, v in self.values.items()
        ):
            raise ValueError(
                "environment binding values must map environment names to text templates"
            )
        object.__setattr__(self, "values", json.loads(_canonical(dict(self.values))))


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ResourcesError(
            "invalid_config", "contribution is not finite JSON"
        ) from exc


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha_file(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise ResourcesError(
            "resource_changed", f"cannot read resource file {path}: {exc}"
        ) from exc


def _git_common_dir(root: Path) -> Path:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--git-common-dir"],
        text=True,
        capture_output=True,
        timeout=5,
        check=False,
    )
    if result.returncode:
        raise ValueError("not a Git worktree")
    path = Path(result.stdout.strip())
    return (root / path).resolve() if not path.is_absolute() else path.resolve()


def _is_linked_worktree(canonical: Path, target: Path) -> bool:
    """Accept only a registered linked worktree belonging to the canonical repository."""
    try:
        canonical_common = _git_common_dir(canonical)
        target_common = _git_common_dir(target)
        if canonical_common != target_common:
            return False
        target_git = subprocess.run(
            ["git", "-C", str(target), "rev-parse", "--git-dir"],
            text=True,
            capture_output=True,
            timeout=5,
            check=True,
        ).stdout.strip()
        git_dir = Path(target_git)
        if not git_dir.is_absolute():
            git_dir = target / git_dir
        git_dir = git_dir.resolve()
        if git_dir == canonical_common or not git_dir.is_relative_to(
            canonical_common / "worktrees"
        ):
            return False
        listing = subprocess.run(
            ["git", "-C", str(canonical), "worktree", "list", "--porcelain"],
            text=True,
            capture_output=True,
            timeout=5,
            check=True,
        ).stdout
        registered = {
            Path(line[len("worktree ") :]).resolve()
            for line in listing.splitlines()
            if line.startswith("worktree ")
        }
        return target in registered
    except (OSError, subprocess.SubprocessError, ValueError):
        return False


def _tree_identity(path: Path) -> dict[str, Any]:
    path = Path(path)
    if path.is_symlink() or not (path.is_file() or path.is_dir()):
        raise ResourcesError(
            "resource_changed", f"resource is missing or unsafe: {path}"
        )
    rows = []
    candidates = [path] if path.is_file() else sorted(path.rglob("*"))
    for item in candidates:
        if item.is_symlink():
            raise ResourcesError(
                "resource_changed", f"resource contains symlink: {item}"
            )
        rel = "." if item == path else item.relative_to(path).as_posix()
        if any(
            p in {".git", "__pycache__", ".pytest_cache"}
            for p in PurePosixPath(rel).parts
        ) or item.suffix in {".pyc", ".pyo"}:
            continue
        if item.is_file():
            rows.append(
                {
                    "path": rel,
                    "mode": item.stat().st_mode & 0o777,
                    "sha256": _sha_file(item),
                }
            )
        elif item.is_dir():
            rows.append(
                {"path": rel, "mode": item.stat().st_mode & 0o777, "directory": True}
            )
    if not rows:
        raise ResourcesError("resource_changed", f"resource has no files: {path}")
    return {
        "kind": "file" if path.is_file() else "directory",
        "mode": path.stat().st_mode & 0o777,
        "files": rows,
        "digest": _digest(rows),
    }


def _resolve_entrypoint(entrypoint: str) -> tuple[Path, str]:
    import importlib.util

    module = entrypoint.split(":", 1)[0]
    try:
        spec = importlib.util.find_spec(module)
    except (ImportError, ValueError, AttributeError) as exc:
        raise ResourcesError(
            "invalid_config", f"cannot resolve entrypoint {entrypoint}: {exc}"
        ) from exc
    if spec is None:
        raise ResourcesError(
            "invalid_config", f"module not found for entrypoint {entrypoint}"
        )
    if spec.submodule_search_locations:
        origin = Path(next(iter(spec.submodule_search_locations)))
    elif spec.origin and spec.origin not in {"built-in", "frozen"}:
        origin = Path(spec.origin)
        # Pin the entrypoint's own module package, not unrelated sibling
        # distributions sharing a top-level namespace. Dependencies outside
        # this package belong in explicit resource closures or the base launch.
        if (origin.parent / "__init__.py").is_file():
            origin = origin.parent
    else:
        raise ResourcesError(
            "invalid_config", f"entrypoint has no file identity: {entrypoint}"
        )
    return origin, _tree_identity(origin)["digest"]


def _runtime_identity():
    """Pin the standalone executor and public contracts even without Workflow B."""
    import importlib.util

    packages = ("aka.resources", "aka.contracts", "aka.core.plugin_runtime")
    result = {}
    for name in packages:
        spec = importlib.util.find_spec(name)
        if spec is None or not spec.submodule_search_locations:
            raise ResourcesError(
                "selection_changed", f"resource runtime package is missing: {name}"
            )
        root = Path(next(iter(spec.submodule_search_locations)))
        result[name] = {
            "root": str(root.resolve()),
            "digest": _tree_identity(root)["digest"],
        }
    return result


def _schema_error(
    schema: Mapping[str, Any], value: Any, path: str = "$", depth: int = 0
) -> str | None:
    """Small deterministic JSON Schema subset: types, enum, required, properties, items, bounds."""
    if depth > 32:
        return f"{path}: schema nesting limit exceeded"
    if not isinstance(schema, Mapping):
        return f"{path}: schema must be an object"
    typ = schema.get("type")
    checks = {
        "object": lambda v: isinstance(v, dict),
        "array": lambda v: isinstance(v, list),
        "string": lambda v: isinstance(v, str),
        "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
        "number": lambda v: (
            isinstance(v, (int, float))
            and not isinstance(v, bool)
            and (not isinstance(v, float) or __import__("math").isfinite(v))
        ),
        "boolean": lambda v: isinstance(v, bool),
        "null": lambda v: v is None,
    }
    if typ in checks and not checks[typ](value):
        return f"{path}: expected {typ}"
    if "enum" in schema and value not in schema["enum"]:
        return f"{path}: value is not in enum"
    if isinstance(value, dict):
        required = schema.get("required", [])
        if not isinstance(required, list) or any(
            not isinstance(k, str) for k in required
        ):
            return f"{path}: invalid required schema"
        absent = [k for k in required if k not in value]
        if absent:
            return f"{path}: missing required field {absent[0]}"
        properties = schema.get("properties", {})
        if not isinstance(properties, Mapping):
            return f"{path}: invalid properties schema"
        if schema.get("additionalProperties") is False:
            extra = set(value) - set(properties)
            if extra:
                return f"{path}: unexpected field {min(extra)}"
        for key, item in value.items():
            if key in properties:
                error = _schema_error(properties[key], item, f"{path}.{key}", depth + 1)
                if error:
                    return error
    if isinstance(value, list) and "items" in schema:
        for index, item in enumerate(value):
            error = _schema_error(schema["items"], item, f"{path}[{index}]", depth + 1)
            if error:
                return error
    if isinstance(value, (str, list, dict)):
        for key, op in (("minLength", len), ("minItems", len), ("minProperties", len)):
            if key in schema and op(value) < schema[key]:
                return f"{path}: below {key}"
        for key, op in (("maxLength", len), ("maxItems", len), ("maxProperties", len)):
            if key in schema and op(value) > schema[key]:
                return f"{path}: above {key}"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            return f"{path}: below minimum"
        if "maximum" in schema and value > schema["maximum"]:
            return f"{path}: above maximum"
    return None


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ResourcesError("mount_conflict", f"refusing symlink: {path}")
    fd, temp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(
                value,
                stream,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        try:
            os.unlink(temp)
        except FileNotFoundError:
            pass


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ResourcesError("mount_conflict", f"refusing symlink: {path}")
    fd, temp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        try:
            os.unlink(temp)
        except FileNotFoundError:
            pass


def _atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ResourcesError("mount_conflict", f"refusing symlink: {path}")
    fd, temp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        try:
            os.unlink(temp)
        except FileNotFoundError:
            pass


def _durable_text(path: Path, documents, message: str) -> str:
    """Read one durable bundle member, using checkpoint bytes exclusively when supplied."""
    path = Path(path).resolve()
    if documents is not None:
        normalized = {str(Path(k).resolve()): v for k, v in documents.items()}
        raw = normalized.get(str(path))
        if raw is None:
            raise ResourcesError("selection_changed", message)
        if isinstance(raw, bytes):
            try:
                return raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ResourcesError("selection_changed", message) from exc
        if not isinstance(raw, str):
            raise ResourcesError("selection_changed", message)
        return raw
    if path.is_symlink() or not path.is_file():
        raise ResourcesError("selection_changed", message)
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ResourcesError("selection_changed", message) from exc


def read_plan(
    path: str | Path,
    digest: str | None = None,
    *,
    documents: Mapping[str, bytes | str] | None = None,
) -> dict[str, Any]:
    path = Path(path)
    raw = _durable_text(path, documents, "tool plan is missing or unsafe").encode(
        "utf-8"
    )
    actual = hashlib.sha256(raw).hexdigest()
    if digest is not None and actual != digest:
        raise ResourcesError("selection_changed", "tool plan digest mismatch")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ResourcesError("selection_changed", "tool plan is invalid JSON") from exc
    if (
        not isinstance(value, dict)
        or value.get("schema") != "aka.tool-plan.v1"
        or type(value.get("version")) is not int
        or value.get("version") != 1
    ):
        raise ResourcesError("selection_changed", "unsupported tool plan")
    return value


class ResourcesHost:
    def __init__(self, scope, *, known_phases: tuple[str, ...] = ()):
        self.scope = scope
        self.known_phases = tuple(known_phases)
        self._snapshot: dict[str, Any] | None = None

    @property
    def scope_key(self) -> str:
        return "/".join(layer.name for layer in self.scope.chain())

    def _entries(self, kind: str, expected):
        entries = self.scope.resolve(kind)
        for entry in entries:
            if entry.name != getattr(entry.value, "name", entry.name) or not isinstance(
                entry.value, expected
            ):
                raise ResourcesError(
                    "invalid_config", f"invalid {kind} contribution {entry.name}"
                )
        return entries

    def _selected(self):
        tools = self._entries("tools", ToolSpec)
        instructions = self._entries("instructions", InstructionSpec)
        providers = self._entries("attribution_providers", AttributionProviderSpec)
        resources = self._entries("workspace_resources", WorkspaceResourceContribution)
        skills = self._entries("agent_skills", AgentSkill)
        skill_mounts = self.scope.resolve("skill_mounts")
        mounts = []
        for entry in skill_mounts:
            if (
                not isinstance(entry.value, SkillMountSpec)
                or entry.name != entry.value.skill.name
            ):
                raise ResourcesError(
                    "invalid_config", f"invalid skill mount {entry.name}"
                )
            mounts.append(entry.value)
        mounts.extend(SkillMountSpec(entry.value, entry.value.name) for entry in skills)
        environment = self.scope.resolve("environment_bindings")
        env_values = []
        for entry in environment:
            if (
                not isinstance(entry.value, EnvironmentBinding)
                or entry.name != entry.value.name
            ):
                raise ResourcesError(
                    "invalid_config", f"invalid environment binding {entry.name}"
                )
            env_values.append(entry.value)
        return (
            tools,
            instructions,
            providers,
            resources,
            tuple(mounts),
            tuple(env_values),
        )

    def freeze(self) -> dict[str, Any]:
        if self.scope.resolve("tool_catalog_owners"):
            raise ResourcesError(
                "invalid_config",
                "legacy independent tool catalog cannot be combined with agent resources",
            )
        tools, instructions, providers, resources, skills, environment = (
            self._selected()
        )
        from aka.core.plugin_runtime.schema import check_schema

        resource_rows = []
        resource_map = {}
        resource_identities = {}
        for entry in resources:
            resource = entry.value
            identity = _tree_identity(resource.source)
            row = {
                "name": resource.name,
                "source": str(resource.source),
                "destination": resource.destination,
                "visibility": resource.visibility,
                "version": resource.version,
                "identity": identity,
            }
            resource_rows.append(row)
            resource_map[resource.name] = resource.source
            resource_identities[resource.name] = identity["digest"]
        skill_rows = []
        for mount in skills:
            skill = mount.skill
            if not skill.source.is_dir() or not (skill.source / "SKILL.md").is_file():
                if mount.optional:
                    continue
                raise ResourcesError(
                    "resource_changed",
                    f"required Agent Skill is missing: {mount.install_name}",
                )
            identity = _tree_identity(skill.source)
            skill_rows.append(
                {
                    "name": skill.name,
                    "install_name": mount.install_name,
                    "source": str(skill.source),
                    "description": skill.description,
                    "optional": mount.optional,
                    "identity": identity,
                }
            )
        provider_names = {e.name for e in providers}
        tool_rows = []
        for entry in tools:
            spec = entry.value
            try:
                check_schema(dict(spec.input_schema))
                check_schema(dict(spec.output_schema))
            except Exception as exc:
                raise ResourcesError(
                    "invalid_config", f"tool {spec.name} has unsupported schema: {exc}"
                ) from exc
            unknown = set(spec.resources) - set(resource_map)
            if unknown:
                raise ResourcesError(
                    "invalid_config",
                    f"tool {spec.name} references unknown resource {min(unknown)}",
                )
            if (
                spec.attribution_provider is not None
                and spec.attribution_provider not in provider_names
            ):
                raise ResourcesError(
                    "invalid_config",
                    f"tool {spec.name} references unknown attribution provider",
                )
            _, code_digest = _resolve_entrypoint(spec.entrypoint)
            identity_payload = {
                "name": spec.name,
                "api_version": spec.api_version,
                "entrypoint": spec.entrypoint,
                "entrypoint_digest": code_digest,
                "config": spec.config,
                "input_schema": spec.input_schema,
                "output_schema": spec.output_schema,
                "resources": list(spec.resources),
                "resource_identities": {
                    name: resource_identities[name] for name in spec.resources
                },
            }
            tool_rows.append(
                {
                    "name": spec.name,
                    "owner": entry.owner,
                    "scope": self.scope_key,
                    "order": entry.order,
                    "description": spec.description,
                    "input_schema": spec.input_schema,
                    "output_schema": spec.output_schema,
                    "entrypoint": spec.entrypoint,
                    "entrypoint_digest": code_digest,
                    "config": spec.config,
                    "resources": list(spec.resources),
                    "timeout_seconds": spec.timeout_seconds,
                    "api_version": spec.api_version,
                    "attribution_provider": spec.attribution_provider,
                    "identity": _digest(identity_payload),
                }
            )
        instruction_rows = []
        for entry in instructions:
            spec = entry.value
            placeholders = re.findall(r"\{([^{}]*)\}", spec.template)
            if any(
                not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value)
                for value in placeholders
            ):
                raise ResourcesError(
                    "invalid_config",
                    f"instruction {spec.name} contains an unsupported placeholder",
                )
            if set(placeholders) - set(spec.variables):
                raise ResourcesError(
                    "invalid_config",
                    f"instruction {spec.name} uses an undeclared variable",
                )
            instruction_rows.append(
                {
                    "name": spec.name,
                    "owner": entry.owner,
                    "scope": self.scope_key,
                    "order": entry.order if entry.order != 500 else spec.order,
                    "phases": list(spec.phases),
                    "template": spec.template,
                    "variables": list(spec.variables),
                    "identity": instruction_identity(spec),
                }
            )
        provider_rows = []
        for entry in providers:
            spec = entry.value
            identities = {
                name: _resolve_entrypoint(getattr(spec, name))[1]
                for name in (
                    "extract_entrypoint",
                    "validate_entrypoint",
                    "artifacts_entrypoint",
                )
            }
            provider_rows.append(
                {
                    **asdict(spec),
                    "config": dict(spec.config),
                    "owner": entry.owner,
                    "scope": self.scope_key,
                    "identity": _digest(identities),
                    "entrypoint_identities": identities,
                }
            )
        env_rows = [
            {"name": binding.name, "values": dict(binding.values)}
            for binding in environment
        ]
        keys = [key for item in env_rows for key in item["values"]]
        if len(keys) != len(set(keys)) or any(
            key.startswith(("AKA_LAUNCH_", "AKA_AGENT_"))
            or key in {"PATH", "PYTHONPATH", "PYTHONHOME"}
            for key in keys
        ):
            raise ResourcesError(
                "invalid_config",
                "environment binding conflicts or uses a reserved variable",
            )
        scope_identity = _digest(
            {
                "scope": self.scope_key,
                "tools": tool_rows,
                "instructions": instruction_rows,
                "providers": provider_rows,
                "resources": resource_rows,
                "skills": skill_rows,
                "environment": env_rows,
            }
        )
        phase_names = {"common", *self.known_phases}
        for row in instruction_rows:
            if self.known_phases and set(row["phases"]) - phase_names:
                raise ResourcesError(
                    "invalid_config",
                    f"instruction {row['name']} uses an unknown consumer phase",
                )
        # Standalone hosts may infer phases from their explicit contributions;
        # application profiles declare a closed consumer phase vocabulary.
        if not self.known_phases:
            for row in instruction_rows:
                phase_names.update(row["phases"])
        snapshot = {
            "schema": "aka.resource-snapshot.v1",
            "version": 1,
            "runtime_identity": _runtime_identity(),
            "scope_key": self.scope_key,
            "scope_identity": scope_identity,
            "tools": tool_rows,
            "instructions": instruction_rows,
            "attribution_providers": provider_rows,
            "workspace_resources": resource_rows,
            "skills": skill_rows,
            "environment": env_rows,
            "known_phases": sorted(phase_names),
        }
        self._snapshot = json.loads(_canonical(snapshot))
        return json.loads(_canonical(snapshot))

    def bind(
        self,
        workspace: str | Path,
        campaign_id: str,
        base_selection: Mapping[str, Any] | str | None = None,
    ):
        workspace = Path(workspace).resolve()
        if (
            not workspace.is_absolute()
            or not isinstance(campaign_id, str)
            or not campaign_id
        ):
            raise ResourcesError(
                "invalid_input", "workspace and campaign_id are required"
            )
        if (
            any(ord(c) < 32 for c in campaign_id)
            or campaign_id in {".", ".."}
            or "/" in campaign_id
            or "\\" in campaign_id
        ):
            raise ResourcesError(
                "invalid_input", "campaign_id must be a path-safe identifier"
            )
        snapshot = self.freeze()
        return _bind_snapshot(snapshot, workspace, campaign_id, base_selection)

    def restore_bound(
        self,
        binding: Mapping[str, Any],
        base_selection: Mapping[str, Any] | str | None = None,
    ):
        snapshot = self.freeze()
        bound = _restore_binding(binding, base_selection)
        if bound.resource_snapshot() != snapshot:
            raise ResourcesError(
                "selection_changed",
                "current Scope contributions differ from the frozen plan",
            )
        return bound


class BoundAgentResources:
    def __init__(
        self,
        workspace: Path,
        campaign_id: str,
        snapshot: dict[str, Any],
        plan_path: Path,
        plan_digest: str,
        launch_path: Path | None,
        launch_digest: str | None,
        launch_payload: str | None,
    ):
        self.workspace = workspace.resolve()
        self.campaign_id = campaign_id
        self._snapshot = snapshot
        self.plan_path = plan_path.resolve()
        self.plan_digest = plan_digest
        self.launch_path = launch_path.resolve() if launch_path else None
        self.launch_digest = launch_digest
        self.launch_payload = launch_payload
        self.scope_key = snapshot["scope_key"]
        self.scope_identity = snapshot["scope_identity"]
        self._by_name = {row["name"]: row for row in snapshot["tools"]}

    def resource_snapshot(self):
        return json.loads(_canonical(self._snapshot))

    def check_resource_snapshot(self, snapshot):
        if snapshot != self._snapshot:
            raise ResourcesError(
                "resource_changed", "resource snapshot differs from the frozen inputs"
            )
        current = _snapshot_from_plan_sources(snapshot)
        if current != snapshot:
            raise ResourcesError(
                "resource_changed", "selected implementation or resource changed"
            )

    def _current(self):
        self.check_resource_snapshot(self._snapshot)
        plan = read_plan(self.plan_path, self.plan_digest)
        if (
            plan["workspace"] != str(self.workspace)
            or plan["campaign_id"] != self.campaign_id
        ):
            raise ResourcesError(
                "selection_changed",
                "tool plan target differs from bound workspace/campaign",
            )
        if self.launch_path:
            if self.launch_path.is_symlink() or not self.launch_path.is_file():
                raise ResourcesError(
                    "selection_changed", "durable launch selection is missing or unsafe"
                )
            raw = self.launch_path.read_text(encoding="utf-8")
            from aka.bootstrap.continuation import base_payload as neutral_payload
            from aka.bootstrap.continuation import digest as launch_hash
            from aka.bootstrap.continuation import restore as restore_selection

            if launch_hash(raw) != self.launch_digest:
                raise ResourcesError(
                    "selection_changed", "durable launch selection digest mismatch"
                )
            selection = json.loads(raw)
            if selection.get("agent_resources", {}).get("digest") != self.plan_digest:
                raise ResourcesError(
                    "selection_changed",
                    "durable launch selection points to another plan",
                )
            restore_selection(neutral_payload(raw))
        elif plan.get("base_selection_digest") is not None:
            raise ResourcesError(
                "selection_changed", "frozen plan requires its durable launch selection"
            )
        return plan

    def check_lock(self, workspace=None):
        target = self.workspace if workspace is None else Path(workspace).resolve()
        if target != self.workspace and not _is_linked_worktree(self.workspace, target):
            raise ResourcesError(
                "selection_changed",
                "resource installation target is not the canonical workspace or its Git worktree",
            )
        self._current()
        lock = target / ".atrex_plugins" / "agent-resources.json"
        if lock.is_symlink() or (lock.exists() and not lock.is_file()):
            raise ResourcesError(
                "mount_conflict", "resource lock is not a regular file"
            )
        if lock.exists():
            try:
                recorded = json.loads(lock.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ResourcesError(
                    "selection_changed", "resource lock is invalid"
                ) from exc
            expected = {
                "schema": "aka.resource-snapshot.v1",
                "snapshot": self._snapshot,
                "plan": str(self.plan_path),
                "plan_digest": self.plan_digest,
                "workspace": str(self.workspace),
                "campaign_id": self.campaign_id,
            }
            if recorded != expected:
                raise ResourcesError(
                    "selection_changed",
                    "workspace is locked to different resources or campaign",
                )

    def _runtime_values(self):
        base = {}
        if self.launch_payload:
            try:
                base = dict(json.loads(self.launch_payload).get("environment", {}))
            except (TypeError, json.JSONDecodeError):
                raise ResourcesError(
                    "selection_changed", "bound base selection is invalid"
                )
        values = {
            **base,
            "workspace": str(self.workspace),
            "campaign_id": self.campaign_id,
        }
        result = {}
        for row in self._snapshot.get("environment", []):
            for key, template in row["values"].items():
                fields = set(re.findall(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", template))
                missing = fields - values.keys()
                if missing:
                    raise ResourcesError(
                        "invalid_input",
                        f"environment binding {row['name']} needs {min(missing)}",
                    )
                rendered = re.sub(
                    r"\{([A-Za-z_][A-Za-z0-9_]*)\}",
                    lambda m: str(values[m.group(1)]),
                    template,
                )
                result[key] = rendered
        for key, expected in result.items():
            ambient = os.environ.get(key)
            if ambient and ambient != expected:
                raise ResourcesError(
                    "selection_changed",
                    f"{key} conflicts with the frozen environment binding",
                )
        return result

    def environment(self, workspace=None, campaign_id=None):
        if workspace is not None:
            target = Path(workspace).resolve()
            if target != self.workspace and not _is_linked_worktree(
                self.workspace, target
            ):
                raise ResourcesError(
                    "selection_changed",
                    "Agent environment target is not the canonical workspace or its Git worktree",
                )
            self._current()
        if campaign_id is not None and campaign_id != self.campaign_id:
            raise ResourcesError(
                "selection_changed", "campaign differs from the bound resource target"
            )
        env = {
            "AKA_AGENT_TOOL_PLAN": str(self.plan_path),
            "AKA_AGENT_TOOL_PLAN_DIGEST": self.plan_digest,
            "AKA_AGENT_WORKSPACE": str(self.workspace),
            "AKA_AGENT_CAMPAIGN_ID": self.campaign_id,
        }
        env.update(self._runtime_values())
        if self.launch_path:
            env["AKA_LAUNCH_SELECTION"] = str(self.launch_path)
            env["AKA_LAUNCH_DIGEST"] = self.launch_digest or ""
        return env

    def _resource_sources(self):
        return {
            row["name"]: Path(row["source"])
            for row in self._snapshot["workspace_resources"]
        }

    def mounts(self):
        result = {}
        skill_roots = (".claude/skills", ".qoder/skills", ".agents/skills")
        occupied = [
            Path(".git"),
            Path(".gitignore"),
            Path(".atrex_plugins"),
            Path("memory"),
            Path("plans"),
            Path("profiles"),
        ]
        for row in self._snapshot["skills"]:
            for root in skill_roots:
                destination = f"{root}/{row['install_name']}"
                p = Path(destination)
                if any(p.is_relative_to(x) or x.is_relative_to(p) for x in occupied):
                    raise ResourcesError(
                        "mount_conflict", f"conflicting Skill mount: {destination}"
                    )
                occupied.append(p)
                result[destination] = Path(row["source"])
        for row in self._snapshot["workspace_resources"]:
            dest = row["destination"]
            if dest is None:
                continue
            p = Path(dest)
            if (
                not p.parts
                or p.is_absolute()
                or ".." in p.parts
                or p.as_posix() != dest
                or any(p.is_relative_to(x) or x.is_relative_to(p) for x in occupied)
            ):
                raise ResourcesError(
                    "mount_conflict", f"conflicting workspace resource mount: {dest}"
                )
            occupied.append(p)
            result[dest] = Path(row["source"])
        return result

    def install(self, workspace=None):
        workspace = self.workspace if workspace is None else Path(workspace).resolve()
        self.check_lock(workspace)
        mounts = self.mounts()  # Validate all conflicts before touching the workspace.
        ignore = workspace / ".gitignore"
        if ignore.is_symlink() or (ignore.exists() and not ignore.is_file()):
            raise ResourcesError(
                "mount_conflict", "workspace .gitignore is not a regular file"
            )
        original_ignore = ignore.read_text(encoding="utf-8") if ignore.exists() else ""
        ignored = original_ignore.rstrip("\n")
        ignore_paths = [
            "/.aka/",
            "/.atrex_plugins/",
            *("/" + name.rstrip("/") for name in mounts),
        ]
        for key, value in self._runtime_values().items():
            if not value or not any(
                marker in key.upper()
                for marker in ("ROOT", "DIR", "PATH", "PROFILE", "STORE")
            ):
                continue
            candidate = Path(value)
            if candidate.is_absolute():
                try:
                    relative = (
                        candidate.resolve().relative_to(self.workspace).as_posix()
                    )
                except ValueError:
                    continue
            else:
                if ".." in candidate.parts or candidate.as_posix() != value:
                    continue
                relative = candidate.as_posix()
            ignore_paths.append("/" + relative.rstrip("/"))
        for line in ignore_paths:
            if line not in original_ignore.splitlines():
                ignored += ("\n" if ignored else "") + line
        # Preflight every target before making even the first mount.
        planned = []
        for destination, source in mounts.items():
            target = workspace / destination
            if target.is_symlink():
                if target.resolve() == source.resolve():
                    planned.append((target, source, "present"))
                    continue
                raise ResourcesError(
                    "mount_conflict", f"workspace path is a different symlink: {target}"
                )
            if target.exists():
                if _tree_identity(target)["digest"] != _tree_identity(source)["digest"]:
                    raise ResourcesError(
                        "mount_conflict",
                        f"workspace path already contains different content: {target}",
                    )
                planned.append((target, source, "present"))
            else:
                planned.append((target, source, "mount"))
        workspace.mkdir(parents=True, exist_ok=True)
        for target, source, state in planned:
            if state == "present":
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(source.resolve(), target_is_directory=source.is_dir())
        self.record_binding(workspace)
        if ignored != original_ignore.rstrip("\n"):
            _atomic_text(ignore, ignored + "\n")
        return self.resource_result()

    def record_binding(self, workspace=None):
        """Persist the trusted evidence owner without mounting host resources.

        Isolated consumers copy their public projection separately and retain
        this descriptor only in the canonical controller workspace.
        """
        target = Path(workspace).resolve() if workspace is not None else self.workspace
        self.check_lock(target)
        state = target / ".atrex_plugins"
        state.mkdir(parents=True, exist_ok=True)
        _atomic_json(
            state / "agent-resources.json",
            {
                "schema": "aka.resource-snapshot.v1",
                "snapshot": self._snapshot,
                "plan": str(self.plan_path),
                "plan_digest": self.plan_digest,
                "workspace": str(self.workspace),
                "campaign_id": self.campaign_id,
            },
        )

    def resource_result(self, workspace=None):
        if workspace is not None:
            self.check_lock(workspace)
        from aka.contracts.agent_resources import (
            AgentResourceFile,
            AgentResourceInstallation,
            InstalledAgentSkill,
            InstalledWorkspaceResource,
        )

        installed = []
        for row in self._snapshot["skills"]:
            root = Path(row["source"])
            files = tuple(
                AgentResourceFile(path["path"], path["mode"], path["sha256"])
                for path in row["identity"]["files"]
                if "sha256" in path
            )
            installed.append(
                InstalledAgentSkill(
                    row["name"],
                    root,
                    row["description"],
                    row["identity"]["digest"],
                    files,
                )
            )
        resources = tuple(
            InstalledWorkspaceResource(
                row["name"],
                Path(row["source"]),
                row["destination"],
                row["visibility"],
                row["version"],
                row["identity"]["digest"],
                tuple(
                    AgentResourceFile(path["path"], path["mode"], path["sha256"])
                    for path in row["identity"]["files"]
                    if "sha256" in path
                ),
            )
            for row in self._snapshot["workspace_resources"]
        )
        module_instructions = self.render("common")
        instruction_text = "\n\n".join(item.text for item in module_instructions)
        return AgentResourceInstallation(
            Path(workspace).resolve() if workspace is not None else None,
            tuple(self.catalog()),
            tuple(self.skill_catalog()),
            instruction_text,
            tuple(installed),
            resources,
        )

    def catalog(self):
        self._current()
        return [
            {
                "name": row["name"],
                "description": row["description"],
                "input_schema": row["input_schema"],
                "output_schema": row["output_schema"],
                "api_version": row["api_version"],
                "version": str(row["api_version"]),
                "identity": row["identity"],
                "attribution_provider": row["attribution_provider"],
            }
            for row in self._current()["tools"]
        ]

    def skill_catalog(self):
        self._current()
        return [
            {
                "id": f"module.{row['name']}",
                "plugin": "selected-modules",
                "name": row["install_name"],
                "logical_name": row["name"],
                "version": "1",
                "install_name": row["install_name"],
                "path": row["source"],
                "description": row["description"],
                "optional": row["optional"],
                "identity": row["identity"]["digest"],
            }
            for row in self._snapshot["skills"]
        ]

    def render(
        self, phase: str, variables: Mapping[str, str] | None = None, **values: str
    ) -> tuple[RenderedInstruction, ...]:
        if variables is not None:
            if values:
                raise ResourcesError(
                    "invalid_input",
                    "pass variables mapping or keyword values, not both",
                )
            values = dict(variables)
        if not isinstance(phase, str) or not phase:
            raise ResourcesError("invalid_input", "instruction phase is required")
        selected = [
            row
            for row in self._snapshot["instructions"]
            if "common" in row["phases"] or phase in row["phases"]
        ]
        known_phases = set(self._snapshot.get("known_phases", ()))
        if phase not in known_phases:
            raise ResourcesError("invalid_input", f"unknown instruction phase: {phase}")
        rows = []
        for row in selected:
            expected = set(row["variables"])
            found = set(re.findall(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", row["template"]))
            if found - expected:
                raise ResourcesError(
                    "invalid_config",
                    f"instruction {row['name']} uses undeclared variable {min(found - expected)}",
                )
            if expected - values.keys():
                raise ResourcesError(
                    "invalid_input",
                    f"instruction {row['name']} missing variable {min(expected - values.keys())}",
                )
            text = re.sub(
                r"\{([A-Za-z_][A-Za-z0-9_]*)\}",
                lambda match: str(values[match.group(1)]),
                row["template"],
            )
            rows.append(
                RenderedInstruction(
                    row["name"],
                    row["owner"],
                    row["scope"],
                    row["order"],
                    text,
                    row["identity"],
                )
            )
        return tuple(rows)

    def instructions(self, phase: str, **values: str) -> str:
        return "\n\n".join(row.text for row in self.render(phase, values))

    def _context(self, call_id: str, row: Mapping[str, Any]) -> dict[str, Any]:
        sources = self._resource_sources()
        env = self._runtime_values()
        if self.launch_payload:
            env = {**json.loads(self.launch_payload).get("environment", {}), **env}
        return {
            "call_id": call_id,
            "workspace": str(self.workspace),
            "campaign_id": self.campaign_id,
            "resource_paths": {name: str(sources[name]) for name in row["resources"]},
            "environment": env,
        }

    def _run_entry(
        self,
        entrypoint: str,
        value: Any,
        context: dict[str, Any],
        timeout: float,
        *,
        env_extra=None,
        cwd=None,
    ):
        body = json.dumps(
            {"entrypoint": entrypoint, "input": value, "context": context},
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
        python_paths = (
            json.loads(self.launch_payload).get("python_paths", [])
            if self.launch_payload
            else []
        )
        env = {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONPATH": os.pathsep.join([*python_paths, *sys.path]),
            "PYTHONIOENCODING": "utf-8",
        }
        env.update(env_extra or {})
        proc = subprocess.Popen(
            [sys.executable, "-m", "aka.resources.runner"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            cwd=cwd,
            start_new_session=True,
        )

        def send_stdin():
            try:
                proc.stdin.write(body)
                proc.stdin.close()
            except (BrokenPipeError, OSError):
                pass

        writer = threading.Thread(target=send_stdin, daemon=True)
        writer.start()
        selector = selectors.DefaultSelector()
        selector.register(proc.stdout, selectors.EVENT_READ, "stdout")
        selector.register(proc.stderr, selectors.EVENT_READ, "stderr")
        output = {"stdout": bytearray(), "stderr": bytearray()}
        limits = {"stdout": 2_000_000, "stderr": 64_000}
        deadline = time.monotonic() + timeout
        failure = None
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                failure = ResourcesError(
                    "tool_timeout", f"entrypoint exceeded {timeout:g}s"
                )
                break
            for key, _ in selector.select(min(remaining, 0.1)):
                chunk = os.read(key.fileobj.fileno(), 65536)
                if not chunk:
                    selector.unregister(key.fileobj)
                    key.fileobj.close()
                    continue
                stream = key.data
                if len(output[stream]) + len(chunk) > limits[stream]:
                    failure = ResourcesError(
                        "tool_output_limit",
                        f"entrypoint {stream} exceeded output limit",
                    )
                    break
                output[stream].extend(chunk)
            if failure:
                break
        if failure:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
            writer.join(timeout=1)
            raise failure
        try:
            proc.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
            raise ResourcesError("tool_timeout", f"entrypoint exceeded {timeout:g}s")
        writer.join(timeout=1)
        stdout, stderr = bytes(output["stdout"]), bytes(output["stderr"])
        if proc.returncode:
            raise ResourcesError(
                "tool_failed",
                stderr.decode("utf-8", "replace")[:2000]
                or f"entrypoint exited {proc.returncode}",
            )
        try:
            return json.loads(stdout), stderr.decode("utf-8", "replace")[:2000]
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ResourcesError(
                "invalid_output", "entrypoint stdout is not one JSON value"
            ) from exc

    def call_result(
        self, name: str, request: Any, workspace=None, *, cwd=None, executor=None
    ) -> dict[str, Any]:
        if executor is not None and not callable(executor):
            raise TypeError("tool executor must be a trusted callable")

        def execute(entrypoint, value, context, timeout, *, env_extra=None, cwd=None):
            if executor is None:
                return self._run_entry(
                    entrypoint, value, context, timeout, env_extra=env_extra, cwd=cwd
                )
            selected = json.loads(self.launch_payload).get("python_paths", []) if self.launch_payload else []
            paths = tuple(dict.fromkeys(str(Path(p).resolve()) for p in [*selected, *sys.path] if p))
            return executor(
                entrypoint, value, context, timeout, env_extra=env_extra,
                cwd=cwd, python_paths=paths,
            )

        if workspace is not None and Path(workspace).resolve() != self.workspace:
            return {
                "schema": "aka.tool-result.v1",
                "data": None,
                "receipt": {
                    "call_id": "",
                    "tool": name,
                    "status": "selection_changed",
                    "implementation_identity": "",
                    "resource_identity": self.plan_digest,
                    "duration_ms": 0,
                },
                "references": [],
                "coverage": {"receipt": "not_applicable", "references": "unavailable"},
                "diagnostics": [
                    {
                        "code": "selection_changed",
                        "stage": "bind",
                        "message": "call workspace differs from canonical binding",
                    }
                ],
            }
        started = time.monotonic()
        call_id = "tool-call-" + uuid.uuid4().hex
        receipt_path = (
            self.workspace / ".atrex_plugins" / "receipts" / f"{call_id}.json"
        )
        row = None
        receipt_coverage = "unknown"
        try:
            plan = self._current()
            row = next((x for x in plan["tools"] if x["name"] == name), None)
            if row is None:
                raise ResourcesError("tool_not_found", f"tool is not selected: {name}")
            pending = {
                "call_id": call_id,
                "tool": name,
                "status": "pending",
                "plan_digest": self.plan_digest,
                "workspace": str(self.workspace),
                "campaign_id": self.campaign_id,
                "created_at": time.time(),
            }
            try:
                _atomic_json(receipt_path, pending)
                receipt_coverage = "persisted"
            except OSError:
                receipt_coverage = "unavailable"
            from aka.core.plugin_runtime.schema import PluginError, validate_schema

            try:
                validate_schema(dict(row["input_schema"]), request)
            except PluginError as exc:
                raise ResourcesError("schema_validation", str(exc)) from exc
            context = self._context(call_id, row)
            context["config"] = row["config"]
            data, stderr = execute(
                row["entrypoint"], request, context, row["timeout_seconds"], cwd=cwd
            )
            try:
                validate_schema(dict(row["output_schema"]), data)
            except PluginError as exc:
                raise ResourcesError("invalid_output", str(exc)) from exc
            references, diagnostics = (), []
            ref_coverage = "not_applicable"
            provider = row.get("attribution_provider")
            if provider:
                try:
                    spec = next(
                        x
                        for x in plan["attribution_providers"]
                        if x["name"] == provider
                    )
                    extracted, _ = execute(
                        spec["extract_entrypoint"],
                        {"result": data, "call_id": call_id, "config": spec["config"]},
                        context,
                        5,
                    )
                    if not isinstance(extracted, list) or len(extracted) > 1000:
                        raise ValueError("provider references must be a bounded array")
                    refs = []
                    for item in extracted:
                        if not isinstance(item, Mapping):
                            raise TypeError("reference must be an object")
                        ref = SourceReference(**item)
                        if ref.provider != provider or ref.call_id != call_id:
                            raise ValueError(
                                "provider returned a reference for another provider or call"
                            )
                        refs.append(asdict(ref))
                    references, ref_coverage = tuple(refs), "complete"
                except Exception as exc:
                    if (executor is not None and isinstance(exc, ResourcesError)
                            and exc.code in {"tool_cancelled", "tool_timeout"}):
                        raise
                    diagnostics.append(
                        {
                            "code": "attribution_failed",
                            "stage": "extract",
                            "message": str(exc)[:500],
                        }
                    )
                    ref_coverage = "unavailable"
            receipt = {
                "call_id": call_id,
                "tool": name,
                "status": "ok",
                "implementation_identity": row["identity"],
                "resource_identity": plan["snapshot_digest"],
                "duration_ms": int((time.monotonic() - started) * 1000),
            }
            sources_path = (
                self.workspace / ".atrex_plugins" / "sources" / f"{call_id}.json"
            )
            try:
                _atomic_json(
                    sources_path,
                    {
                        "schema": "aka.tool-sources.v1",
                        "call_id": call_id,
                        "references": list(references),
                        "coverage": {
                            "receipt": receipt_coverage,
                            "references": ref_coverage,
                        },
                    },
                )
            except OSError as exc:
                ref_coverage = "unavailable"
                diagnostics.append(
                    {
                        "code": "source_index_write_failed",
                        "stage": "persist",
                        "message": str(exc)[:500],
                    }
                )
            try:
                _atomic_json(
                    receipt_path,
                    {
                        **receipt,
                        "coverage": {
                            "receipt": receipt_coverage,
                            "references": ref_coverage,
                        },
                        "source_index": str(sources_path.relative_to(self.workspace)),
                    },
                )
            except OSError as exc:
                receipt_coverage = "unavailable"
                diagnostics.append(
                    {
                        "code": "receipt_write_failed",
                        "stage": "persist",
                        "message": str(exc)[:500],
                    }
                )
            if stderr:
                diagnostics.append(
                    {"code": "tool_stderr", "stage": "execute", "message": stderr[:500]}
                )
            receipt = {
                **receipt,
                "coverage": {"receipt": receipt_coverage, "references": ref_coverage},
                "source_index": str(sources_path.relative_to(self.workspace)),
            }
            return {
                "schema": "aka.tool-result.v1",
                "data": data,
                "receipt": receipt,
                "references": list(references),
                "coverage": {"receipt": receipt_coverage, "references": ref_coverage},
                "diagnostics": diagnostics,
            }
        except ResourcesError as exc:
            receipt = {
                "call_id": call_id,
                "tool": name,
                "status": exc.code,
                "implementation_identity": row["identity"] if row else "",
                "resource_identity": self.plan_digest,
                "duration_ms": int((time.monotonic() - started) * 1000),
            }
            diagnostics = [
                {"code": exc.code, "stage": "call", "message": str(exc)[:500]}
            ]
            try:
                old = (
                    json.loads(receipt_path.read_text())
                    if receipt_path.exists()
                    else {}
                )
                if old.get("status") == "pending":
                    _atomic_json(receipt_path, receipt)
                receipt_coverage = (
                    "persisted" if old.get("status") == "pending" else "unknown"
                )
            except OSError:
                receipt_coverage = "unavailable"
            return {
                "schema": "aka.tool-result.v1",
                "data": None,
                "receipt": receipt,
                "references": [],
                "coverage": {"receipt": receipt_coverage, "references": "unavailable"},
                "diagnostics": diagnostics,
            }

    def call(self, name: str, request: Any, workspace=None, *, cwd=None):
        result = self.call_result(name, request, workspace, cwd=cwd)
        if result["receipt"]["status"] != "ok":
            error = result["diagnostics"][0]
            raise ResourcesError(error["code"], error["message"])
        return result["data"]

    def call_evidence(self, call_id: str):
        if not isinstance(call_id, str) or not re.fullmatch(
            r"tool-call-[a-f0-9]{32}", call_id
        ):
            raise ResourcesError("invalid_input", "invalid tool call id")
        path = self.workspace / ".atrex_plugins" / "receipts" / f"{call_id}.json"
        source_path = self.workspace / ".atrex_plugins" / "sources" / f"{call_id}.json"
        if path.is_symlink() or not path.is_file():
            raise ResourcesError("evidence_unavailable", "tool call receipt is missing")
        try:
            receipt = json.loads(path.read_text(encoding="utf-8"))
            if receipt.get("call_id") != call_id:
                raise ValueError("receipt call id mismatch")
            sources = (
                json.loads(source_path.read_text(encoding="utf-8"))
                if source_path.is_file() and not source_path.is_symlink()
                else {}
            )
            if sources and sources.get("call_id") != call_id:
                raise ValueError("source index call id mismatch")
            return {
                "receipt": receipt,
                "references": sources.get("references", []),
                "coverage": receipt.get(
                    "coverage",
                    sources.get(
                        "coverage", {"receipt": "persisted", "references": "unknown"}
                    ),
                ),
            }
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise ResourcesError(
                "evidence_unavailable", f"invalid call evidence: {exc}"
            ) from exc

    def _provider_call(self, provider: str, method: str, payload: Any):
        try:
            plan = self._current()
            spec = next(
                (
                    item
                    for item in plan["attribution_providers"]
                    if item["name"] == provider
                ),
                None,
            )
            if spec is None:
                raise ResourcesError(
                    "provider_not_found",
                    f"attribution provider is not selected: {provider}",
                )
            entrypoint = spec[method]
            context = {
                "call_id": "",
                "workspace": str(self.workspace),
                "campaign_id": self.campaign_id,
                "resource_paths": {},
                "environment": self._runtime_values(),
                "config": spec["config"],
            }
            return self._run_entry(
                entrypoint, {**payload, "config": spec["config"]}, context, 5
            )[0]
        except Exception as exc:
            raise ResourcesError("attribution_provider_failed", str(exc)[:500]) from exc

    def validate_declaration(self, provider: str, payload: Mapping[str, Any]):
        from aka.contracts.attribution import AttributionValidation

        try:
            value = self._provider_call(provider, "validate_entrypoint", dict(payload))
            if isinstance(value, Mapping):
                value = AttributionValidation(**value)
            if not isinstance(value, AttributionValidation):
                raise TypeError("provider validation result has wrong type")
            return asdict(value)
        except Exception as exc:  # noqa: BLE001 - provider validation is fail-open.
            return {
                "accepted": False,
                "verification": "unknown",
                "diagnostics": [
                    {
                        "code": "attribution_validation_failed",
                        "stage": "validate",
                        "message": str(exc)[:500],
                    }
                ],
            }

    def artifacts(self):
        from aka.contracts.attribution import ArtifactDescriptor

        result = []
        for row in self._current()["attribution_providers"]:
            try:
                values = self._provider_call(
                    row["name"],
                    "artifacts_entrypoint",
                    {"workspace": str(self.workspace)},
                )
                if not isinstance(values, list) or len(values) > 1000:
                    raise ValueError("artifact provider must return a bounded list")
                for raw in values:
                    item = (
                        ArtifactDescriptor(**raw) if isinstance(raw, Mapping) else raw
                    )
                    if not isinstance(item, ArtifactDescriptor):
                        raise TypeError("invalid ArtifactDescriptor")
                    rel = PurePosixPath(item.path)
                    if (
                        rel.is_absolute()
                        or ".." in rel.parts
                        or not rel.parts
                        or "\\" in item.path
                    ):
                        raise ValueError("artifact path must be workspace relative")
                    target = self.workspace.joinpath(*rel.parts)
                    if (
                        target.is_symlink()
                        or not target.is_file()
                        or not target.resolve().is_relative_to(self.workspace)
                    ):
                        raise ValueError(
                            "artifact path is missing or escapes workspace"
                        )
                    actual = _sha_file(target)
                    if item.sha256 and item.sha256.removeprefix("sha256:") != actual:
                        raise ValueError("artifact digest mismatch")
                    result.append(
                        asdict(
                            ArtifactDescriptor(
                                item.path,
                                item.role,
                                item.schema,
                                item.required,
                                item.coverage,
                                actual,
                                row["name"],
                            )
                        )
                    )
            except Exception as exc:  # noqa: BLE001 - artifact failures must remain fail-open.
                result.append(
                    {
                        "provider": row["name"],
                        "diagnostic": {
                            "code": "artifact_provider_failed",
                            "stage": "artifacts",
                            "message": str(exc)[:500],
                        },
                    }
                )
        return result


def _snapshot_from_plan_sources(snapshot):
    refreshed = json.loads(_canonical(snapshot))
    if snapshot.get("runtime_identity") != _runtime_identity():
        raise ResourcesError(
            "selection_changed", "resource host or public contract identity changed"
        )
    for group in ("workspace_resources", "skills"):
        for row in refreshed[group]:
            current = _tree_identity(Path(row["source"]))
            if current != row["identity"]:
                raise ResourcesError(
                    "resource_changed", f"resource changed: {row['name']}"
                )
    for row in refreshed["tools"]:
        if _resolve_entrypoint(row["entrypoint"])[1] != row["entrypoint_digest"]:
            raise ResourcesError(
                "selection_changed", f"tool implementation changed: {row['name']}"
            )
    for row in refreshed["attribution_providers"]:
        identities = {
            name: _resolve_entrypoint(row[name])[1]
            for name in (
                "extract_entrypoint",
                "validate_entrypoint",
                "artifacts_entrypoint",
            )
        }
        if identities != row["entrypoint_identities"]:
            raise ResourcesError(
                "selection_changed", f"attribution provider changed: {row['name']}"
            )
    return refreshed


def _base_payload(base_selection):
    if base_selection is None:
        return None
    if isinstance(base_selection, str):
        try:
            value = json.loads(base_selection)
        except json.JSONDecodeError as exc:
            raise ResourcesError(
                "selection_changed", "base selection is invalid JSON"
            ) from exc
    elif isinstance(base_selection, Mapping):
        value = dict(base_selection)
    else:
        raise ResourcesError("selection_changed", "base selection must be JSON object")
    if (
        not isinstance(value, dict)
        or type(value.get("version")) is not int
        or value["version"] not in (1, 2)
    ):
        raise ResourcesError(
            "selection_changed", "base selection must be continuation version 1 or 2"
        )
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _bind_snapshot(snapshot, workspace, campaign_id, base_selection):
    base = _base_payload(base_selection)
    directory = workspace / ".aka" / "agent-resources" / campaign_id
    if directory.is_symlink():
        raise ResourcesError("mount_conflict", "resource bundle directory is a symlink")
    snapshot_path = directory / "resource-snapshot.json"
    tools = []
    for row in snapshot["tools"]:
        tools.append(
            {
                key: row[key]
                for key in (
                    "name",
                    "description",
                    "input_schema",
                    "output_schema",
                    "entrypoint",
                    "entrypoint_digest",
                    "config",
                    "resources",
                    "timeout_seconds",
                    "api_version",
                    "attribution_provider",
                    "identity",
                )
            }
        )
    plan = {
        "schema": "aka.tool-plan.v1",
        "version": 1,
        "workspace": str(workspace),
        "campaign_id": campaign_id,
        "scope_key": snapshot["scope_key"],
        "scope_identity": snapshot["scope_identity"],
        "snapshot": str(snapshot_path),
        "snapshot_digest": _digest(snapshot),
        "base_selection_digest": _digest(json.loads(base))
        if base is not None
        else None,
        "tools": tools,
        "attribution_providers": snapshot["attribution_providers"],
    }
    plan_path = directory / "tool-plan.json"
    pretty = lambda value: (
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
        ).encode("utf-8")
        + b"\n"
    )
    plan_bytes = pretty(plan)
    plan_digest = hashlib.sha256(plan_bytes).hexdigest()
    launch_payload = base
    launch_path = launch_digest = None
    if launch_payload is not None:
        from aka.bootstrap.continuation import bind_resources
        from aka.bootstrap.continuation import digest as launch_hash

        binding = {
            "schema": "aka.bound-agent-resources.v1",
            "workspace": str(workspace),
            "campaign_id": campaign_id,
            "plan": str(plan_path),
            "digest": plan_digest,
        }
        launch_payload = bind_resources(launch_payload, binding)
        launch_path = directory / "launch-selection.json"
        launch_digest = launch_hash(launch_payload)
    proposed = {snapshot_path: pretty(snapshot), plan_path: plan_bytes}
    if launch_path is not None:
        proposed[launch_path] = launch_payload.encode("utf-8")
    for path, content in proposed.items():
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise ResourcesError(
                "mount_conflict", f"unsafe durable resource member: {path}"
            )
        if path.exists() and path.read_bytes() != content:
            raise ResourcesError(
                "selection_changed", f"durable resource member already differs: {path}"
            )
    directory.mkdir(parents=True, exist_ok=True)
    for path, content in proposed.items():
        if not path.exists():
            if path == snapshot_path or path == plan_path:
                _atomic_json(path, json.loads(content))
            else:
                _atomic_text(path, content.decode("utf-8"))
    return BoundAgentResources(
        workspace,
        campaign_id,
        snapshot,
        plan_path,
        plan_digest,
        launch_path,
        launch_digest,
        launch_payload,
    )


def _restore_binding(binding, base_selection=None, documents=None):
    required = {"schema", "workspace", "campaign_id", "plan", "digest"}
    if (
        not isinstance(binding, Mapping)
        or set(binding) != required
        or binding.get("schema") != "aka.bound-agent-resources.v1"
    ):
        raise ResourcesError("selection_changed", "invalid bound resource binding")
    if (
        not all(
            isinstance(binding.get(k), str) and binding[k]
            for k in ("workspace", "campaign_id", "plan", "digest")
        )
        or not Path(binding["workspace"]).is_absolute()
        or not Path(binding["plan"]).is_absolute()
        or not re.fullmatch(r"[0-9a-f]{64}", binding["digest"])
    ):
        raise ResourcesError(
            "selection_changed", "invalid bound resource target or digest"
        )
    workspace, campaign_id = (
        Path(binding["workspace"]).resolve(),
        binding["campaign_id"],
    )
    plan_path = Path(binding["plan"]).resolve()
    plan = read_plan(plan_path, binding["digest"], documents=documents)
    if (
        plan.get("workspace") != str(workspace)
        or plan.get("campaign_id") != campaign_id
    ):
        raise ResourcesError("selection_changed", "bound plan target mismatch")
    expected_dir = workspace / ".aka" / "agent-resources" / campaign_id
    if (
        plan_path != (expected_dir / "tool-plan.json").resolve()
        or Path(plan.get("snapshot", "")).resolve()
        != (expected_dir / "resource-snapshot.json").resolve()
    ):
        raise ResourcesError(
            "selection_changed", "bound plan is outside the expected durable bundle"
        )
    snapshot_path = Path(plan["snapshot"])
    raw_snapshot = _durable_text(
        snapshot_path, documents, "resource snapshot is missing or unsafe"
    )
    try:
        snapshot = json.loads(raw_snapshot)
    except json.JSONDecodeError as exc:
        raise ResourcesError(
            "selection_changed", "resource snapshot is invalid"
        ) from exc
    if _digest(snapshot) != plan.get("snapshot_digest") or snapshot.get(
        "scope_identity"
    ) != plan.get("scope_identity"):
        raise ResourcesError("selection_changed", "resource snapshot digest mismatch")
    _snapshot_from_plan_sources(snapshot)
    base = _base_payload(base_selection)
    base_digest = _digest(json.loads(base)) if base is not None else None
    if plan.get("base_selection_digest") != base_digest:
        raise ResourcesError(
            "selection_changed", "base selection digest differs from frozen tool plan"
        )
    launch_path = (
        workspace / ".aka" / "agent-resources" / campaign_id / "launch-selection.json"
    )
    launch_payload = launch_digest = None
    if base is not None:
        launch_payload = _durable_text(
            launch_path, documents, "durable launch selection is missing"
        )
        from aka.bootstrap.continuation import bind_resources
        from aka.bootstrap.continuation import digest as launch_hash

        expected = bind_resources(base, dict(binding))
        if json.loads(expected) != json.loads(launch_payload):
            raise ResourcesError(
                "selection_changed",
                "durable launch selection differs from base selection",
            )
        from aka.bootstrap.continuation import restore as restore_selection

        restore_selection(base)
        launch_digest = launch_hash(launch_payload)
    # The plan is a projection of the immutable resource snapshot; reject mismatched members.
    expected_tools = [
        {
            key: row[key]
            for key in (
                "name",
                "description",
                "input_schema",
                "output_schema",
                "entrypoint",
                "entrypoint_digest",
                "config",
                "resources",
                "timeout_seconds",
                "api_version",
                "attribution_provider",
                "identity",
            )
        }
        for row in snapshot["tools"]
    ]
    if (
        plan.get("tools") != expected_tools
        or plan.get("attribution_providers") != snapshot["attribution_providers"]
    ):
        raise ResourcesError(
            "selection_changed", "tool plan does not match its resource snapshot"
        )
    return BoundAgentResources(
        workspace,
        campaign_id,
        snapshot,
        plan_path,
        binding["digest"],
        launch_path if launch_path.exists() else None,
        launch_digest,
        launch_payload,
    )


def restore_bound(
    binding, base_selection=None, *, host: ResourcesHost | None = None, documents=None
):
    """Validate and reconstruct a bound plan; pass host to also compare current Scope declarations."""
    if host is not None:
        restored = restore_bound(binding, base_selection, documents=documents)
        if host.freeze() != restored.resource_snapshot():
            raise ResourcesError(
                "selection_changed",
                "current Scope contributions differ from checkpoint snapshot",
            )
        return restored
    return _restore_binding(binding, base_selection, documents=documents)


__all__ = [
    "BoundAgentResources",
    "EnvironmentBinding",
    "ResourcesError",
    "ResourcesHost",
    "SkillMountSpec",
    "read_plan",
    "restore_bound",
]
