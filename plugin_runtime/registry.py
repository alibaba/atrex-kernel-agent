"""Reusable local plugin registry for tools and Skills; host policy is injected."""

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
from .schema import PluginError, check_schema, read_json, validate_schema

NAME = re.compile(r"[a-z][a-z0-9-]*")


@dataclass(frozen=True)
class HostLayout:
    """Host-owned installation paths; the runtime knows no Agent or workflow names."""

    state_dir: str = ".plugins"
    skill_roots: tuple[str, ...] = ()
    reserved_mounts: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        for name in (self.state_dir, *self.skill_roots):
            path = Path(name)
            if (
                not name
                or path.is_absolute()
                or ".." in path.parts
                or path == Path(".")
            ):
                raise PluginError(
                    "invalid_config", f"host path must be workspace-relative: {name}"
                )


def local_file(root: Path, relative: str) -> Path:
    if not isinstance(relative, str):
        raise PluginError("invalid_manifest", "file path must be a string")
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise PluginError(
            "invalid_manifest", f"missing or out-of-plugin file: {relative}"
        )
    return path


def tree_digest(root: Path) -> str:
    """Fingerprint local code/data, excluding Git metadata and generated Python caches."""
    digest = hashlib.sha256()
    if not root.exists():
        return "missing"
    if root.is_file():
        return hashlib.sha256(root.read_bytes()).hexdigest()
    for directory, dirs, names in os.walk(root):
        dirs[:] = sorted(
            d for d in dirs if d not in {".git", "__pycache__", ".pytest_cache"}
        )
        for name in sorted(names):
            if name == ".git" or name.endswith((".pyc", ".pyo")):
                continue
            path = Path(directory) / name
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class Plugin:
    root: Path
    manifest: dict
    tools: dict
    resources: dict[str, tuple[Path, bool]]
    settings: dict
    fingerprint: str

    @property
    def id(self) -> str:
        return self.manifest["id"]


class PluginRegistry:
    def __init__(self, config: Path | str, *, layout: HostLayout | None = None):
        self.layout = layout or HostLayout()
        self.config = Path(config).resolve()
        value = read_json(self.config)
        if (
            not isinstance(value, dict)
            or type(value.get("schema_version")) is not int
            or value["schema_version"] != 1
            or set(value) != {"schema_version", "plugins"}
            or not isinstance(value["plugins"], list)
        ):
            raise PluginError(
                "invalid_config", "expected schema_version=1 and plugins array"
            )
        self.plugins: list[Plugin] = []
        seen = set()
        for entry in value["plugins"]:
            if (
                not isinstance(entry, dict)
                or set(entry) - {"path", "enabled", "settings"}
                or not isinstance(entry.get("path"), str)
                or not isinstance(entry.get("enabled", True), bool)
                or not isinstance(entry.get("settings", {}), dict)
            ):
                raise PluginError(
                    "invalid_config",
                    "plugin entry requires path and optional enabled boolean",
                )
            if not entry.get("enabled", True):
                continue
            try:
                plugin = self._load(
                    (self.config.parent / entry["path"]).resolve(),
                    entry.get("settings", {}),
                )
            except OSError as exc:
                raise PluginError(
                    "invalid_manifest",
                    f"cannot read local plugin {entry['path']}: {exc}",
                ) from exc
            if plugin.id in seen:
                raise PluginError("invalid_config", f"duplicate plugin id: {plugin.id}")
            seen.add(plugin.id)
            self.plugins.append(plugin)
        self.mounts()  # Fail before installing anything on conflicting resources.
        self.environment(Path("/workspace"))

    def _load(self, root: Path, settings: dict) -> Plugin:
        manifest_path = root / "plugin.json"
        manifest = read_json(manifest_path)
        required = {"id", "version", "api_version"}
        optional = {
            "instructions",
            "resources",
            "skills",
            "environment",
            "tools",
            "settings_schema",
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
        for key in optional - {"settings_schema"}:
            if key in manifest and not isinstance(manifest[key], dict):
                raise PluginError("invalid_manifest", f"{key} must be an object")
        files = {manifest_path}
        if "settings_schema" in manifest:
            path = local_file(root, manifest["settings_schema"])
            files.add(path)
            schema = read_json(path)
            check_schema(schema)
            try:
                validate_schema(schema, settings, "settings")
            except PluginError as exc:
                raise PluginError("invalid_config", str(exc)) from exc
        for phase, relative in manifest.get("instructions", {}).items():
            if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]*", phase):
                raise PluginError(
                    "invalid_manifest", f"invalid instruction scope: {phase}"
                )
            files.add(local_file(root, relative))
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
                    "entrypoint",
                    "command",
                }
                or len({"entrypoint", "command"} & tool.keys()) != 1
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
            loaded["resolved_command"] = resolve_command(root, tool)
            for field in ("input_schema", "output_schema"):
                path = local_file(root, tool[field])
                files.add(path)
                loaded[field] = read_json(path)
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
        for path in sorted(files):
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
        digest.update(tree_digest(root).encode())
        for name, (path, _mount) in sorted(resources.items()):
            digest.update(name.encode())
            digest.update(str(path).encode())
            digest.update(tree_digest(path).encode())
        for name, skill in sorted(manifest.get("skills", {}).items()):
            digest.update(name.encode())
            digest.update(tree_digest((root / skill["path"]).resolve()).encode())
        return Plugin(root, manifest, tools, resources, settings, digest.hexdigest())

    def snapshot(self) -> dict:
        return {
            "api_version": 1,
            "config": str(self.config),
            "layout": {
                "state_dir": self.layout.state_dir,
                "skill_roots": list(self.layout.skill_roots),
                "reserved_mounts": sorted(self.layout.reserved_mounts),
            },
            "plugins": [
                {
                    "id": p.id,
                    "version": p.manifest["version"],
                    "root": str(p.root),
                    "fingerprint": p.fingerprint,
                    "settings_digest": hashlib.sha256(
                        json.dumps(p.settings, sort_keys=True).encode()
                    ).hexdigest(),
                    "commands": {
                        name: list(tool["resolved_command"])
                        for name, tool in p.tools.items()
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
                    for skill_root in self.layout.skill_roots:
                        entries[f"{skill_root}/{name}"] = source
            for name, source in entries.items():
                candidate = Path(name)
                occupied = [Path(self.layout.state_dir), *(Path(key) for key in mounts)]
                if name in self.layout.reserved_mounts or any(
                    candidate.is_relative_to(path) or path.is_relative_to(candidate)
                    for path in occupied
                ):
                    raise PluginError("invalid_manifest", f"conflicting mount: {name}")
                mounts[name] = source
        return mounts

    def check_lock(self, workspace: Path) -> None:
        lock = workspace / self.layout.state_dir / "lock.json"
        if lock.exists():
            current = PluginRegistry(self.config, layout=self.layout).snapshot()
            if read_json(lock) != current or current != self.snapshot():
                raise PluginError(
                    "plugin_changed",
                    "plugin configuration, code, or host layout changed; restore the locked inputs or use a new workspace",
                )

    def install(self, workspace: Path) -> None:
        self.check_lock(workspace)
        mounts = self.mounts()
        for name, source in mounts.items():
            destination = workspace / name
            if (destination.exists() or destination.is_symlink()) and not (
                destination.is_symlink() and destination.resolve() == source
            ):
                raise PluginError(
                    "mount_conflict", f"plugin mount would replace {destination}"
                )
        state = workspace / self.layout.state_dir
        state.mkdir(parents=True, exist_ok=True)
        for name, source in mounts.items():
            destination = workspace / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.is_symlink():
                destination.symlink_to(source)
        lock = state / "lock.json"
        if not lock.exists():
            lock.write_text(json.dumps(self.snapshot(), indent=2) + "\n")
        (state / "instructions.md").write_text(self.instructions("common"))
        ignore = workspace / ".gitignore"
        existing = ignore.read_text() if ignore.exists() else ""
        additions = [f"/{self.layout.state_dir}/", *(f"/{name}" for name in mounts)]
        missing = [line for line in additions if line not in existing.splitlines()]
        if missing:
            with ignore.open("a") as stream:
                stream.write("\n# Plugin runtime\n" + "\n".join(missing) + "\n")

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

    def instructions(self, phase: str, **values: str) -> str:
        parts = ["## Enabled plugins"]
        if not self.plugins:
            parts.append("No plugins enabled.")
        for plugin in self.plugins:
            parts.append(f"### {plugin.id} ({plugin.manifest['version']})")
            parts.extend(
                f"- `{plugin.id}.{name}`: {tool['description']}"
                for name, tool in plugin.tools.items()
            )
            parts.extend(
                f"- Skill `{skill['id']}` (name: `{skill['name']}`): {skill['description']} Read `{skill['path']}/SKILL.md`."
                for skill in self.skill_catalog()
                if skill["plugin"] == plugin.id
            )
            for key in dict.fromkeys(("common", phase)):
                relative = plugin.manifest.get("instructions", {}).get(key)
                if relative:
                    text = local_file(plugin.root, relative).read_text()
                    for name, value in values.items():
                        text = text.replace("{{" + name + "}}", str(value))
                    parts.append(text)
        return "\n\n".join(parts)

    def environment(
        self, workspace: Path, context: dict[str, str] | None = None
    ) -> dict[str, str]:
        values = dict(context or {}, workspace=str(workspace.resolve()))
        environment = {}
        for plugin in self.plugins:
            for key, value in plugin.manifest.get("environment", {}).items():
                if key in environment or key in {"PLUGIN_ROOT", "PLUGIN_SETTINGS_JSON"}:
                    raise PluginError(
                        "invalid_manifest", f"conflicting environment variable: {key}"
                    )
                for name, replacement in values.items():
                    value = value.replace("{" + name + "}", replacement)
                environment[key] = value
        return environment

    def _tool_environment(self, plugin: Plugin) -> dict[str, str]:
        return dict(
            os.environ,
            PLUGIN_ROOT=str(plugin.root),
            PLUGIN_SETTINGS_JSON=json.dumps(plugin.settings),
        )

    def call(
        self, name: str, request: object, workspace: Path, *, cwd: Path | None = None
    ) -> object:
        self.check_lock(workspace)
        plugin_id, _, tool_name = name.partition(".")
        plugin = next((p for p in self.plugins if p.id == plugin_id), None)
        if plugin is None or tool_name not in plugin.tools:
            raise PluginError("tool_not_found", f"tool is not enabled: {name}")
        tool = plugin.tools[tool_name]
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
            event_dir = workspace / self.layout.state_dir / "calls"
            if (workspace / self.layout.state_dir / "lock.json").exists():
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
