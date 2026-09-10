"""Local AKA plugins: declarative discovery, workspace wiring and JSON subprocess tools.

Plugins are trusted local code, not a sandbox. The v1 schema dialect is deliberately
small and rejects unsupported keywords rather than silently ignoring constraints.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from .constants import REPO_ROOT

DEFAULT_CONFIG = REPO_ROOT / "plugins/default.json"
STATE_DIR = ".atrex_plugins"
NAME = re.compile(r"[a-z][a-z0-9-]*")


class PluginError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def decode_json(text: str) -> object:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-JSON numeric constant: {value}")

    return json.loads(text, parse_constant=reject_constant)


def read_json(path: Path) -> object:
    try:
        return decode_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PluginError("invalid_config", f"cannot read JSON {path}: {exc}") from exc


def validate_schema(schema: dict, value: object, path: str = "input") -> None:
    kind = schema["type"]
    types = {
        "object": dict,
        "array": list,
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "null": type(None),
    }
    if not isinstance(value, types[kind]) or (
        kind in {"integer", "number"} and isinstance(value, bool)
    ):
        raise PluginError("schema_validation", f"{path}: expected {kind}")
    if "enum" in schema and value not in schema["enum"]:
        raise PluginError("schema_validation", f"{path}: unsupported value")
    if kind == "object":
        properties = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in value:
                raise PluginError("schema_validation", f"{path}: missing {key}")
        for key, item in value.items():
            child = properties.get(key, schema.get("additionalProperties", True))
            if child is False:
                raise PluginError("schema_validation", f"{path}: unknown field {key}")
            if isinstance(child, dict):
                validate_schema(child, item, f"{path}.{key}")
    elif kind == "array":
        for index, item in enumerate(value):
            validate_schema(schema["items"], item, f"{path}[{index}]")
    elif kind == "string":
        if len(value.strip()) < schema.get("minLength", 0):
            raise PluginError(
                "schema_validation", f"{path}: string is empty or too short"
            )
    elif kind in {"number", "integer"}:
        if (isinstance(value, float) and not math.isfinite(value)) or not schema.get(
            "minimum", -float("inf")
        ) <= value <= schema.get("maximum", float("inf")):
            raise PluginError(
                "schema_validation", f"{path}: number outside allowed range"
            )


def check_schema(schema: object) -> None:
    allowed = {
        "type",
        "description",
        "properties",
        "required",
        "additionalProperties",
        "items",
        "enum",
        "minLength",
        "minimum",
        "maximum",
    }
    if not isinstance(schema, dict) or set(schema) - allowed:
        raise PluginError("invalid_manifest", "unsupported schema keywords")
    if not isinstance(schema.get("type"), str) or schema["type"] not in {
        "object",
        "array",
        "string",
        "integer",
        "number",
        "boolean",
        "null",
    }:
        raise PluginError("invalid_manifest", "schema requires a supported type")
    typed_keywords = {
        "object": {"properties", "required", "additionalProperties"},
        "array": {"items"},
        "string": {"minLength"},
        "integer": {"minimum", "maximum"},
        "number": {"minimum", "maximum"},
        "boolean": set(),
        "null": set(),
    }
    if set(schema) - {"type", "description", "enum"} - typed_keywords[schema["type"]]:
        raise PluginError(
            "invalid_manifest", "schema constraint does not apply to its type"
        )
    if "enum" in schema and (
        not isinstance(schema["enum"], list) or not schema["enum"]
    ):
        raise PluginError("invalid_manifest", "enum must be a nonempty array")
    if "minLength" in schema and (
        type(schema["minLength"]) is not int or schema["minLength"] < 0
    ):
        raise PluginError("invalid_manifest", "minLength must be a nonnegative integer")
    for bound in ("minimum", "maximum"):
        if bound in schema and (
            type(schema[bound]) not in (int, float)
            or (isinstance(schema[bound], float) and not math.isfinite(schema[bound]))
        ):
            raise PluginError("invalid_manifest", "numeric bounds must be numbers")
    if schema.get("minimum", -float("inf")) > schema.get("maximum", float("inf")):
        raise PluginError("invalid_manifest", "minimum exceeds maximum")
    properties = schema.get("properties", {})
    if not isinstance(properties, dict) or not isinstance(
        schema.get("required", []), list
    ):
        raise PluginError("invalid_manifest", "invalid schema properties/required")
    if any(
        not isinstance(key, str) or key not in properties
        for key in schema.get("required", [])
    ):
        raise PluginError("invalid_manifest", "required key has no property schema")
    for child in properties.values():
        check_schema(child)
    extra = schema.get("additionalProperties", True)
    if isinstance(extra, dict):
        check_schema(extra)
    elif not isinstance(extra, bool):
        raise PluginError("invalid_manifest", "invalid additionalProperties")
    if schema["type"] == "array":
        check_schema(schema.get("items"))


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
    fingerprint: str

    @property
    def id(self) -> str:
        return self.manifest["id"]


class PluginRegistry:
    def __init__(self, config: Path | str = DEFAULT_CONFIG):
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
                or set(entry) - {"path", "enabled"}
                or not isinstance(entry.get("path"), str)
                or not isinstance(entry.get("enabled", True), bool)
            ):
                raise PluginError(
                    "invalid_config",
                    "plugin entry requires path and optional enabled boolean",
                )
            if not entry.get("enabled", True):
                continue
            try:
                plugin = self._load((self.config.parent / entry["path"]).resolve())
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
        self.environment(Path("/workspace"), "campaign")

    def _load(self, root: Path) -> Plugin:
        manifest_path = root / "plugin.json"
        manifest = read_json(manifest_path)
        required = {"id", "version", "api_version", "tools"}
        optional = {"instructions", "resources", "skills", "environment"}
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
        files = {manifest_path}
        for phase, relative in manifest.get("instructions", {}).items():
            if phase not in {
                "common",
                "setup",
                "episode",
                "fast_episode",
                "framework_baseline",
            }:
                raise PluginError(
                    "invalid_manifest", f"unknown instruction phase: {phase}"
                )
            files.add(local_file(root, relative))
        if not isinstance(manifest["tools"], dict) or not manifest["tools"]:
            raise PluginError("invalid_manifest", "plugin requires tools")
        tools = {}
        for name, tool in manifest["tools"].items():
            if (
                not NAME.fullmatch(name)
                or not isinstance(tool, dict)
                or set(tool)
                != {
                    "description",
                    "entrypoint",
                    "input_schema",
                    "output_schema",
                    "timeout_seconds",
                }
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
            files.add(local_file(root, tool["entrypoint"]))
            loaded = dict(tool)
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
                or set(skill) != {"path", "optional"}
                or not isinstance(skill["optional"], bool)
                or not isinstance(skill["path"], str)
            ):
                raise PluginError("invalid_manifest", f"invalid skill: {name}")
            if (
                not (root / skill["path"] / "SKILL.md").is_file()
                and not skill["optional"]
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
        return Plugin(root, manifest, tools, resources, digest.hexdigest())

    def snapshot(self) -> dict:
        return {
            "api_version": 1,
            "config": str(self.config),
            "plugins": [
                {
                    "id": p.id,
                    "version": p.manifest["version"],
                    "root": str(p.root),
                    "fingerprint": p.fingerprint,
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
                    for backend in (".claude", ".qoder", ".agents"):
                        entries[f"{backend}/skills/{name}"] = source
            for name, source in entries.items():
                if (
                    name
                    in {
                        "tools",
                        "reference",
                        "skills",
                        "reference-projects",
                        "atrex-bench",
                        "memory",
                        "plans",
                        "profiles",
                    }
                    or name in mounts
                ):
                    raise PluginError("invalid_manifest", f"conflicting mount: {name}")
                mounts[name] = source
        return mounts

    def check_lock(self, workspace: Path) -> None:
        lock = workspace / STATE_DIR / "lock.json"
        if lock.exists():
            current = PluginRegistry(self.config).snapshot()
            if read_json(lock) != current or current != self.snapshot():
                raise PluginError(
                    "plugin_changed",
                    "campaign plugin set/version/code changed; use the original plugin configuration/code or a new campaign workspace",
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
        state = workspace / STATE_DIR
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
        additions = [f"/{STATE_DIR}/", *(f"/{name}" for name in mounts)]
        missing = [line for line in additions if line not in existing.splitlines()]
        if missing:
            with ignore.open("a") as stream:
                stream.write("\n# AKA plugin runtime\n" + "\n".join(missing) + "\n")

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

    def instructions(self, phase: str, **values: str) -> str:
        parts = [
            "## Enabled plugin tools",
            (
                "Run `python3 tools/plugin.py list` to inspect tool schemas.\n"
                "Call `python3 tools/plugin.py call <plugin.tool> --input <request.json>`.\n"
                "Only use enabled tools. Hardware claims require auditable sources; missing facts remain unknown."
            ),
        ]
        if not self.plugins:
            parts.append(
                "No plugins enabled. Use available reference sources; do not run disabled tools."
            )
        for plugin in self.plugins:
            parts.append(f"### {plugin.id} ({plugin.manifest['version']})")
            parts.extend(
                f"- `{plugin.id}.{name}`: {tool['description']}"
                for name, tool in plugin.tools.items()
            )
            for key in dict.fromkeys(("common", phase)):
                relative = plugin.manifest.get("instructions", {}).get(key)
                if relative:
                    text = local_file(plugin.root, relative).read_text()
                    for name, value in values.items():
                        text = text.replace("{{" + name + "}}", str(value))
                    parts.append(text)
        parts.append(
            "For experiments without Wiki queries or reconsidered Wiki responses, use "
            "`wiki_usage_status: not_queried` and omit `wiki_query_ids` and `wiki_usage`."
        )
        return "\n\n".join(parts)

    def environment(self, workspace: Path, campaign_name: str) -> dict[str, str]:
        environment = {
            "ATREX_PLUGIN_CONFIG": str(self.config),
            "ATREX_PLUGIN_WORKSPACE": str(workspace.resolve()),
        }
        for plugin in self.plugins:
            for key, value in plugin.manifest.get("environment", {}).items():
                if key in environment:
                    raise PluginError(
                        "invalid_manifest", f"conflicting environment variable: {key}"
                    )
                environment[key] = value.replace(
                    "{workspace}", str(workspace.resolve())
                ).replace("{campaign_name}", campaign_name)
        return environment

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
            environment = dict(os.environ, ATREX_PLUGIN_ROOT=str(plugin.root))
            process = subprocess.Popen(
                [sys.executable, str(local_file(plugin.root, tool["entrypoint"]))],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=cwd or workspace,
                env=environment,
                start_new_session=True,
            )
            try:
                stdout, stderr = process.communicate(
                    request_json, timeout=tool["timeout_seconds"]
                )
            except subprocess.TimeoutExpired as exc:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.communicate()
                raise PluginError(
                    "tool_timeout", f"{name} exceeded its timeout"
                ) from exc
            except BaseException:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.communicate()
                raise
            if process.returncode:
                raise PluginError(
                    "tool_failed",
                    f"{name} exited {process.returncode}: {stderr[-2000:]}",
                )
            try:
                result = decode_json(stdout)
            except ValueError as exc:
                raise PluginError(
                    "invalid_output", f"{name} did not return JSON"
                ) from exc
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
