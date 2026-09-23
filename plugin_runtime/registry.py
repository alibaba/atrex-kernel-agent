"""Supervisor plugin discovery, dependency pinning, and tool invocation."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from .execution import execute_json, resolve_command
from .files import FileBudget, IGNORED_DIRECTORIES, MAX_DOCUMENT_BYTES, tree_digest
from .schema import PluginError, check_schema, read_json, validate_schema

NAME = re.compile(r"[a-z][a-z0-9-]*")
STATE_DIR = ".atrex_plugins"
RUNTIME_TOOLS = {"gpu-wiki.query": "wiki-query"}
ENV_PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_.-]*)\}")
SKILL_ROOTS = (".claude/skills", ".qoder/skills", ".agents/skills")
RESERVED_MOUNTS = frozenset(
    {
        "tools",
        "reference",
        "skills",
        "reference-projects",
        "atrex-bench",
        "memory",
        "plans",
        "profiles",
    }
)


def local_file(root: Path, relative: str) -> Path:
    if not isinstance(relative, str):
        raise PluginError("invalid_manifest", "file path must be a string")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise PluginError(
            "invalid_manifest", f"missing or out-of-plugin file: {relative}"
        )
    relative_path = path.relative_to(root.resolve())
    if any(part in IGNORED_DIRECTORIES or part.endswith((".pyc", ".pyo")) for part in relative_path.parts):
        raise PluginError("invalid_manifest", f"metadata cannot live in a fingerprint-excluded path: {relative}")
    if path.stat().st_size > MAX_DOCUMENT_BYTES:
        raise PluginError("invalid_manifest", f"plugin metadata exceeds {MAX_DOCUMENT_BYTES} bytes: {relative}")
    return path


@dataclass(frozen=True)
class Plugin:
    root: Path
    manifest: dict
    tools: dict
    resources: dict[str, tuple[Path, bool]]
    fingerprint: str

    @property
    def id(self) -> str:
        return self.manifest["id"]


class PluginRegistry:
    def __init__(self, plugin_dir: Path | str, *, plugin_root: Path | None = None):
        self.plugin_dir = Path(plugin_dir).resolve()
        if not self.plugin_dir.is_dir():
            raise PluginError(
                "invalid_config", f"missing plugin directory: {self.plugin_dir}"
            )
        self.plugins: list[Plugin] = []
        self.file_budget = FileBudget()
        seen = set()
        if plugin_root is None:
            manifests = []
            with os.scandir(self.plugin_dir) as entries:
                for entry in entries:
                    try:
                        self.file_budget.visit()
                    except ValueError as exc:
                        raise PluginError("invalid_config", str(exc)) from exc
                    if entry.is_dir() and (Path(entry.path) / "plugin.json").exists():
                        manifests.append(Path(entry.path) / "plugin.json")
            manifests.sort()
        else:
            # A Supervisor-pinned root avoids scanning/hashing unrelated plugins.
            # Keep its lexical path, as full discovery does for symlinked roots.
            selected = Path(plugin_root).absolute()
            if selected.parent != self.plugin_dir:
                raise PluginError("invalid_config", "selected plugin must be a direct child of plugin_dir")
            manifests = [selected / "plugin.json"]
        for manifest in manifests:
            try:
                plugin = self._load(manifest.parent)
            except PluginError:
                raise
            except (OSError, ValueError) as exc:
                raise PluginError(
                    "invalid_manifest",
                    f"cannot read local plugin {manifest.parent}: {exc}",
                ) from exc
            if plugin.id in seen:
                raise PluginError("invalid_config", f"duplicate plugin id: {plugin.id}")
            seen.add(plugin.id)
            self.plugins.append(plugin)
        self.mounts()  # Reject conflicting resource/Skill declarations at discovery.
        # Check conflicts now; caller-provided placeholder values exist only
        # when environment() renders a concrete invocation.
        tuple(self._environment_declarations())

    def _load(self, root: Path) -> Plugin:
        manifest_path = root / "plugin.json"
        manifest = read_json(manifest_path.resolve(), budget=self.file_budget)
        required = {"id", "version", "api_version"}
        optional = {
            "instructions",
            "resources",
            "skills",
            "environment",
            "tools",
        }
        if (
            not isinstance(manifest, dict)
            or not required <= manifest.keys()
            or set(manifest) - required - optional
        ):
            raise PluginError("invalid_manifest", f"invalid manifest: {manifest_path}")
        if (
            type(manifest["api_version"]) is not int
            or manifest["api_version"] != 1
            or not isinstance(manifest["id"], str)
            or not NAME.fullmatch(manifest["id"])
            or not isinstance(manifest["version"], str)
            or not manifest["version"]
        ):
            raise PluginError(
                "invalid_manifest", f"invalid plugin identity/API: {manifest_path}"
            )
        for key in optional:
            if key in manifest and not isinstance(manifest[key], dict):
                raise PluginError("invalid_manifest", f"{key} must be an object")
        for phase, relative in manifest.get("instructions", {}).items():
            if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]*", phase):
                raise PluginError(
                    "invalid_manifest", f"invalid instruction scope: {phase}"
                )
            local_file(root, relative)
        if not manifest.get("tools") and not manifest.get("skills"):
            raise PluginError(
                "invalid_manifest", "plugin must contribute tools or skills"
            )
        tools = {}
        for name, tool in manifest.get("tools", {}).items():
            if (
                not NAME.fullmatch(name)
                or not isinstance(tool, dict)
                or not {
                    "description",
                    "input_schema",
                    "output_schema",
                    "timeout_seconds",
                }
                <= tool.keys()
                or set(tool)
                - {
                    "description",
                    "input_schema",
                    "output_schema",
                    "timeout_seconds",
                    "command",
                    "runtime_tool",
                }
                or ("command" in tool) == ("runtime_tool" in tool)
            ):
                raise PluginError("invalid_manifest", f"invalid tool: {name}")
            if (
                not isinstance(tool["description"], str)
                or type(tool["timeout_seconds"]) is not int
                or not 1 <= tool["timeout_seconds"] <= 3600
            ):
                raise PluginError(
                    "invalid_manifest", f"invalid description/timeout: {name}"
                )
            loaded = dict(tool)
            if "runtime_tool" in tool:
                expected = RUNTIME_TOOLS.get(f"{manifest['id']}.{name}")
                if expected is None or tool["runtime_tool"] != expected:
                    raise PluginError("invalid_manifest", f"unsupported Runtime tool: {name}")
            else:
                loaded["resolved_command"] = resolve_command(root, tool)
            for field in ("input_schema", "output_schema"):
                path = local_file(root, tool[field])
                loaded[field] = read_json(path, budget=self.file_budget)
                check_schema(loaded[field])
            tools[name] = loaded
        resources = {}
        for name, resource in manifest.get("resources", {}).items():
            if isinstance(resource, str):
                resource = {"path": resource}
            if (
                not NAME.fullmatch(name)
                or not isinstance(resource, dict)
                or set(resource) - {"path", "optional", "mount"}
                or not isinstance(resource.get("path"), str)
                or not isinstance(resource.get("optional", False), bool)
                or not isinstance(resource.get("mount", True), bool)
            ):
                raise PluginError("invalid_manifest", f"invalid resource: {name}")
            path = (root / resource["path"]).resolve()
            if not path.exists() and not resource.get("optional", False):
                raise PluginError("invalid_manifest", f"missing resource: {name}")
            resources[name] = (path, resource.get("mount", True))
        for name, skill in manifest.get("skills", {}).items():
            if (
                not re.fullmatch(r"[A-Za-z][A-Za-z0-9-]*", name)
                or not isinstance(skill, dict)
                or set(skill) - {"path", "optional", "description"}
                or not isinstance(skill.get("optional", False), bool)
                or not isinstance(skill.get("path"), str)
                or not isinstance(skill.get("description", ""), str)
            ):
                raise PluginError("invalid_manifest", f"invalid skill: {name}")
            if not (root / skill["path"] / "SKILL.md").is_file() and not skill.get(
                "optional", False
            ):
                raise PluginError("invalid_manifest", f"missing skill: {name}")
        for key, item in manifest.get("environment", {}).items():
            if not re.fullmatch(r"[A-Z][A-Z0-9_]*", key) or not isinstance(item, str):
                raise PluginError("invalid_manifest", "invalid environment declaration")
        digest = hashlib.sha256()
        # Source, manifest, schemas and instructions are all covered by this tree.
        # Metadata parsing above is bounded separately; do not hash it twice.
        digest.update(tree_digest(root, budget=self.file_budget).encode())
        for name, (path, _mount) in sorted(resources.items()):
            digest.update(name.encode())
            digest.update(str(path).encode())
            digest.update(tree_digest(path, budget=self.file_budget).encode())
        for name, skill in sorted(manifest.get("skills", {}).items()):
            digest.update(name.encode())
            digest.update(tree_digest((root / skill["path"]).resolve(), budget=self.file_budget).encode())
        return Plugin(root, manifest, tools, resources, digest.hexdigest())

    def snapshot(self) -> dict:
        return {
            "api_version": 1,
            "plugin_dir": str(self.plugin_dir),
            "plugins": [
                {
                    "id": p.id,
                    "version": p.manifest["version"],
                    "root": str(p.root),
                    "fingerprint": p.fingerprint,
                    "commands": {
                        name: list(tool["resolved_command"])
                        for name, tool in p.tools.items()
                        if "resolved_command" in tool
                    },
                    "runtime_tools": {
                        name: tool["runtime_tool"]
                        for name, tool in p.tools.items()
                        if "runtime_tool" in tool
                    },
                }
                for p in self.plugins
            ],
        }

    def mounts(self) -> dict[str, Path]:
        mounts = {}
        for plugin in self.plugins:
            entries = {
                name: path
                for name, (path, mount) in plugin.resources.items()
                if mount and path.exists()
            }
            for name, skill in plugin.manifest.get("skills", {}).items():
                source = (plugin.root / skill["path"]).resolve()
                if (source / "SKILL.md").is_file():
                    for skill_root in SKILL_ROOTS:
                        entries[f"{skill_root}/{name}"] = source
            for name, source in entries.items():
                candidate = Path(name)
                occupied = [Path(STATE_DIR), *(Path(key) for key in mounts)]
                if name in RESERVED_MOUNTS or any(
                    candidate.is_relative_to(path) or path.is_relative_to(candidate)
                    for path in occupied
                ):
                    raise PluginError("invalid_manifest", f"conflicting mount: {name}")
                mounts[name] = source
        return mounts

    def check_lock(self, workspace: Path) -> None:
        lock = workspace / STATE_DIR / "lock.json"
        if lock.exists() and read_json(lock) != self.snapshot():
            raise PluginError(
                "plugin_changed",
                "discovered plugins changed; restore the locked inputs or use a new workspace",
            )

    def catalog(self) -> list[dict]:
        return [
            {
                "name": f"{p.id}.{name}",
                "version": p.manifest["version"],
                "description": tool["description"],
                "input_schema": tool["input_schema"],
                "output_schema": tool["output_schema"],
            }
            for p in self.plugins
            for name, tool in p.tools.items()
        ]

    def skill_catalog(self) -> list[dict]:
        return [
            {
                "id": f"{plugin.id}.{name}",
                "plugin": plugin.id,
                "name": name,
                "version": plugin.manifest["version"],
                "path": str((plugin.root / skill["path"]).resolve()),
                "description": skill.get(
                    "description", "Follow the supplied Skill instructions."
                ),
            }
            for plugin in self.plugins
            for name, skill in plugin.manifest.get("skills", {}).items()
            if (plugin.root / skill["path"] / "SKILL.md").is_file()
        ]

    def _environment_declarations(self):
        seen = set()
        for plugin in self.plugins:
            for key, value in plugin.manifest.get("environment", {}).items():
                if key in seen or key == "PLUGIN_ROOT":
                    raise PluginError(
                        "invalid_manifest", f"conflicting environment variable: {key}"
                    )
                seen.add(key)
                yield plugin.id, key, value

    def environment(
        self, workspace: Path, context: dict[str, str] | None = None
    ) -> dict[str, str]:
        values = dict(context or {}, workspace=str(workspace.resolve()))
        environment = {}
        for plugin_id, key, value in self._environment_declarations():
            for name, replacement in values.items():
                value = value.replace("{" + name + "}", replacement)
            if unresolved := sorted(set(ENV_PLACEHOLDER.findall(value))):
                raise PluginError(
                    "invalid_manifest",
                    f"plugin {plugin_id} environment {key} has unresolved placeholders: "
                    f"{', '.join(unresolved)}; provide their context values or fix the declaration",
                )
            environment[key] = value
        return environment

    def _tool_environment(self, plugin: Plugin) -> dict[str, str]:
        return dict(os.environ, PLUGIN_ROOT=str(plugin.root))

    def call(
        self, name: str, request: object, workspace: Path, *, cwd: Path | None = None,
        supervised: bool = False,
    ) -> object:
        self.check_lock(workspace)
        plugin_id, _, tool_name = name.partition(".")
        plugin = next((p for p in self.plugins if p.id == plugin_id), None)
        if plugin is None or tool_name not in plugin.tools:
            raise PluginError("tool_not_found", f"tool is not enabled: {name}")
        tool = plugin.tools[tool_name]
        if "runtime_tool" in tool:
            raise PluginError("runtime_required", f"{name} requires the authenticated Supervisor route")
        started = time.monotonic()
        call_id = "plugin-call-" + uuid.uuid4().hex
        status = "ok"
        try:
            try:
                request_json = json.dumps(request, allow_nan=False)
            except (TypeError, ValueError) as exc:
                raise PluginError(
                    "schema_validation", f"input is not JSON: {exc}"
                ) from exc
            validate_schema(tool["input_schema"], request)
            environment = self._tool_environment(plugin)
            result = execute_json(
                tool["resolved_command"],
                request_json,
                cwd=cwd or workspace,
                environment=environment,
                timeout=tool["timeout_seconds"],
                tool_name=name,
                supervised=supervised,
            )
            try:
                validate_schema(tool["output_schema"], result, "output")
            except PluginError as exc:
                raise PluginError("invalid_output", str(exc)) from exc
            return result
        except PluginError as exc:
            status = exc.code
            raise
        except OSError as exc:
            status = "tool_failed"
            raise PluginError(status, f"cannot execute {name}: {exc}") from exc
        except (KeyboardInterrupt, SystemExit):
            status = "interrupted"
            raise
        finally:
            event_dir = workspace / STATE_DIR / "calls"
            if (workspace / STATE_DIR / "lock.json").exists():
                event = {
                    "call_id": call_id,
                    "tool": name,
                    "plugin_version": plugin.manifest["version"],
                    "status": status,
                    "duration_ms": round((time.monotonic() - started) * 1000, 3),
                }
                # One file per invocation avoids concurrent JSONL append races. No payloads.
                try:
                    event_dir.mkdir(parents=True, exist_ok=True)
                    with (event_dir / f"{call_id}.json").open("x") as stream:
                        json.dump(event, stream)
                except OSError as exc:
                    print(
                        f"plugin telemetry could not be written: {exc}", file=sys.stderr
                    )
