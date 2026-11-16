"""Validate retained Resources selections without original roots or registries."""
import hashlib
import json
import os
from pathlib import Path
import re


def _fields(value, required, optional=()):
    if not isinstance(value, dict) or not set(required) <= value.keys() or set(value) - set(required) - set(optional):
        raise ValueError("invalid retained Resources selection fields")


def _text(value):
    if not isinstance(value, str) or not value.strip() or "\0" in value:
        raise ValueError("invalid retained Resources selection text")
    return value


def _name(value, pattern=r"[a-z][a-z0-9-]*"):
    if not isinstance(value, str) or not re.fullmatch(pattern, value):
        raise ValueError("invalid retained Resources binding name")
    return value


def _path(value):
    if not isinstance(value, str):
        raise ValueError("invalid retained Resources source path")
    path = Path(value)
    if (not path.is_absolute() or path.as_posix() != value or ".." in path.parts
            or "\\" in value or any(ord(char) < 32 for char in value)):
        raise ValueError("invalid retained Resources source binding")
    return path


def _api(value, field):
    _fields(value, ("api_version", field))
    if type(value["api_version"]) is not int or value["api_version"] != 1 or not isinstance(value[field], list):
        raise ValueError("invalid retained Resources selection API")
    return value[field]


def _digest(payload):
    return hashlib.sha256(payload).hexdigest()


def _identity(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("invalid retained Resources selection fingerprint")
    return value


class _Closure:
    def __init__(self, manifest, read):
        self.roots = {Path(row["source"]): row for row in manifest["roots"]}
        self.read = read
        self.selected = set()
        self.absent = set()
        self.mounts = [Path(".atrex_plugins")]

    def mount(self, name):
        path = Path(name)
        if any(path.is_relative_to(other) or other.is_relative_to(path) for other in self.mounts):
            raise ValueError("conflicting retained Resources mount authorization")
        self.mounts.append(path)

    def tree(self, path, *, optional=False):
        for root, row in self.roots.items():
            if path == root:
                return row
            if path.is_relative_to(root) and row["kind"] == "directory":
                relative = path.relative_to(root).as_posix()
                entries = {entry["path"]: entry for entry in row["entries"]}
                member = entries.get(relative)
                if member is not None:
                    if member["kind"] == "file":
                        return {"source": str(path), "kind": "file", "mode": member["mode"],
                                "entries": [{**member, "path": "."}]}
                    prefix = relative + "/"
                    return {"source": str(path), "kind": "directory", "mode": member["mode"],
                            "entries": [{**entry, "path": entry["path"][len(prefix):]}
                                        for entry in row["entries"] if entry["path"].startswith(prefix)]}
        if optional:
            return None
        raise ValueError(f"retained Resources closure omits a declared root: {path}")

    def select(self, path, *, kind=None, optional=False):
        row = self.tree(path, optional=optional)
        if row is not None:
            if kind is not None and row["kind"] != kind:
                raise ValueError("retained Resources root has the wrong kind")
            self.selected.add(path)
        elif optional:
            self.absent.add(path)
        return row

    def file(self, path):
        row = self.tree(path)
        if row["kind"] != "file":
            raise ValueError("retained Resources declaration requires a file")
        return self.read(row["entries"][0]["sha256"])

    def tree_digest(self, path, *, optional=False):
        row = self.tree(path, optional=optional)
        if row is None:
            return "missing"
        if row["kind"] == "file":
            return row["entries"][0]["sha256"]
        files, directories = {}, {".": []}
        for entry in row["entries"]:
            member = Path(entry["path"])
            parent = member.parent.as_posix()
            if entry["kind"] == "file":
                files.setdefault(parent, []).append(entry)
            else:
                directories.setdefault(parent, []).append(entry["path"])
                directories.setdefault(entry["path"], [])
        digest = hashlib.sha256()
        def visit(directory):
            for entry in sorted(files.get(directory, ()), key=lambda item: item["path"]):
                digest.update(entry["path"].encode())
                digest.update(self.read(entry["sha256"]))
            for child in sorted(directories[directory]):
                visit(child)
        visit(".")
        return digest.hexdigest()

    def check_roots(self):
        coalesced = {path for path in self.selected if not any(other != path and path.is_relative_to(other)
                                                              for other in self.selected)}
        if coalesced != set(self.roots):
            raise ValueError("retained Resources roots differ from the declared selection")


def _relative(root, value, *, inside=False):
    _text(value)
    if "\\" in value or any(ord(char) < 32 for char in value):
        raise ValueError("invalid retained Resources declared path")
    result = Path(os.path.normpath(root / value))
    if inside and not result.is_relative_to(root):
        raise ValueError("retained plugin local file escapes its root")
    return result


def _plugin(closure, value):
    _fields(value, ("id", "version", "root", "fingerprint", "commands"))
    _name(value["id"]); _text(value["version"]); _identity(value["fingerprint"])
    root = _path(value["root"])
    closure.select(root, kind="directory")
    manifest_path = root / "plugin.json"
    manifest = json.loads(closure.file(manifest_path))
    _fields(manifest, ("id", "version", "api_version"),
            ("instructions", "resources", "skills", "environment", "tools"))
    if type(manifest["api_version"]) is not int or manifest["api_version"] != 1:
        raise ValueError("invalid retained plugin API")
    if manifest["id"] != value["id"] or manifest["version"] != value["version"]:
        raise ValueError("retained plugin declaration differs from its selection")
    for field in ("instructions", "resources", "skills", "environment", "tools"):
        if not isinstance(manifest.get(field, {}), dict):
            raise ValueError("invalid retained plugin declaration")
    if not manifest.get("tools") and not manifest.get("skills"):
        raise ValueError("retained plugin contributes no tools or Skills")
    files = {manifest_path}
    for phase, relative in manifest.get("instructions", {}).items():
        _name(phase, r"[A-Za-z][A-Za-z0-9_.-]*")
        files.add(_relative(root, relative, inside=True))
    if not isinstance(value["commands"], dict) or set(value["commands"]) != set(manifest.get("tools", {})):
        raise ValueError("retained plugin command bindings differ")
    for name, tool in manifest.get("tools", {}).items():
        _name(name)
        _fields(tool, ("description", "input_schema", "output_schema", "timeout_seconds", "command"))
        if not isinstance(tool["description"], str) or type(tool["timeout_seconds"]) is not int or not 1 <= tool["timeout_seconds"] <= 3600:
            raise ValueError("invalid retained plugin tool declaration")
        for argv in (tool["command"], value["commands"][name]):
            if not isinstance(argv, list) or not argv or any(not isinstance(arg, str) or "\0" in arg for arg in argv):
                raise ValueError("invalid retained plugin argv binding")
        for field in ("input_schema", "output_schema"):
            path = _relative(root, tool[field], inside=True)
            from aka.core.plugin_runtime.schema import check_schema
            check_schema(json.loads(closure.file(path)))
            files.add(path)
    resources = {}
    from .plugin_runtime.registry import RESERVED_MOUNTS, SKILL_ROOTS
    for name, resource in manifest.get("resources", {}).items():
        _name(name)
        resource = {"path": resource} if isinstance(resource, str) else resource
        _fields(resource, ("path",), ("optional", "mount"))
        if type(resource.get("optional", False)) is not bool or type(resource.get("mount", True)) is not bool:
            raise ValueError("invalid retained plugin resource authorization")
        path = _relative(root, resource["path"])
        row = closure.select(path, optional=resource.get("optional", False))
        if row is not None and resource.get("mount", True):
            if name in RESERVED_MOUNTS: raise ValueError("retained plugin mount name is reserved")
            closure.mount(name)
        resources[name] = path
    skills = {}
    for name, skill in manifest.get("skills", {}).items():
        _name(name, r"[A-Za-z][A-Za-z0-9-]*")
        _fields(skill, ("path",), ("optional", "description"))
        if type(skill.get("optional", False)) is not bool or not isinstance(skill.get("description", ""), str):
            raise ValueError("invalid retained plugin Skill declaration")
        path = _relative(root, skill["path"])
        closure.select(path, optional=skill.get("optional", False))
        if not skill.get("optional", False): closure.file(path / "SKILL.md")
        if closure.tree(path / "SKILL.md", optional=True) is not None:
            for prefix in SKILL_ROOTS: closure.mount(f"{prefix}/{name}")
        skills[name] = path
    for name, setting in manifest.get("environment", {}).items():
        _name(name, r"[A-Z][A-Z0-9_]*")
        if not isinstance(setting, str): raise ValueError("invalid retained plugin environment")
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(closure.file(path))
    digest.update(closure.tree_digest(root).encode())
    for name, path in sorted(resources.items()):
        digest.update(name.encode()); digest.update(str(path).encode())
        digest.update(closure.tree_digest(path, optional=True).encode())
    for name, path in sorted(skills.items()):
        digest.update(name.encode()); digest.update(closure.tree_digest(path, optional=True).encode())
    if digest.hexdigest() != value["fingerprint"]:
        raise ValueError("retained plugin fingerprint differs from its selected bytes")
    return root


def validate_selection(manifest, read):
    """Validate roles, identities and the exact retained closure using bytes only."""
    selection = manifest["selection"]
    _fields(selection, ("plugins", "agent_skills", "workspace_resources"))
    closure = _Closure(manifest, read)
    plugins = selection["plugins"]
    _fields(plugins, ("api_version", "plugin_dir", "plugins"), ("plugin_paths",))
    if type(plugins["api_version"]) is not int or plugins["api_version"] != 1 or not isinstance(plugins["plugins"], list):
        raise ValueError("invalid retained plugin selection API")
    catalog = _path(plugins["plugin_dir"]) if plugins["plugin_dir"] is not None else None
    if catalog is not None: closure.select(catalog, kind="directory")
    paths = plugins.get("plugin_paths", [])
    if not isinstance(paths, list): raise ValueError("invalid retained explicit plugins")
    explicit = tuple(_path(path) for path in paths)
    if len(set(explicit)) != len(explicit): raise ValueError("duplicate retained explicit plugin")
    roots, ids = [], set()
    for value in plugins["plugins"]:
        if not isinstance(value, dict): raise ValueError("invalid retained plugin identity")
        identifier = _name(value.get("id"))
        if identifier in ids:
            raise ValueError("duplicate retained plugin identity")
        ids.add(identifier)
        roots.append(_plugin(closure, value))
    expected = set(explicit)
    if catalog is not None:
        expected.update(catalog / Path(entry["path"]).parent for entry in closure.tree(catalog)["entries"]
                        if entry["kind"] == "file" and len(Path(entry["path"]).parts) == 2
                        and Path(entry["path"]).name == "plugin.json")
    if len(set(roots)) != len(roots) or set(roots) != expected:
        raise ValueError("retained catalog or explicit plugin bindings differ")
    names = set()
    from .plugin_runtime.registry import SKILL_ROOTS
    for skill in _api(selection["agent_skills"], "skills"):
        _fields(skill, ("name", "source", "description", "fingerprint"))
        name = _name(skill["name"]); _text(skill["description"]); _identity(skill["fingerprint"])
        if name in names: raise ValueError("duplicate retained module Skill")
        names.add(name)
        source = _path(skill["source"])
        row = closure.select(source, kind="directory")
        closure.file(source / "SKILL.md")
        for prefix in SKILL_ROOTS: closure.mount(f"{prefix}/{name}")
        digest = hashlib.sha256()
        for entry in row["entries"]:
            if entry["kind"] == "file":
                digest.update(entry["path"].encode() + b"\0")
                digest.update(str(entry["mode"]).encode() + b"\0")
                digest.update(read(entry["sha256"]) + b"\0")
        if digest.hexdigest() != skill["fingerprint"]: raise ValueError("retained module Skill fingerprint differs")
    names = set()
    for resource in _api(selection["workspace_resources"], "resources"):
        _fields(resource, ("name", "source", "destination", "visibility", "version", "fingerprint", "files"))
        name = _name(resource["name"]); _text(resource["version"]); _identity(resource["fingerprint"])
        if name in names: raise ValueError("duplicate retained workspace resource")
        names.add(name)
        if resource["visibility"] == "runtime":
            if resource["destination"] is not None: raise ValueError("retained runtime resource cannot mount")
        elif resource["visibility"] == "agent":
            destination = _text(resource["destination"])
            path = Path(destination)
            if path.is_absolute() or ".." in path.parts or path.as_posix() != destination or "\\" in destination or any(ord(c) < 32 for c in destination):
                raise ValueError("invalid retained workspace resource destination")
            if path.parts[0] in {".git", ".gitignore", ".atrex_plugins", "memory", "plans", "profiles", "atrex-bench"}:
                raise ValueError("retained workspace resource destination is reserved")
            closure.mount(destination)
        else: raise ValueError("invalid retained workspace resource visibility")
        row = closure.select(_path(resource["source"]))
        files = [{key: entry[key] for key in ("path", "mode", "sha256")} for entry in row["entries"] if entry["kind"] == "file"]
        if not files or resource["files"] != files: raise ValueError("retained workspace resource file identities differ")
        identity = {"files": files, "directories": [(entry["path"], entry["mode"]) for entry in row["entries"] if entry["kind"] == "directory"],
                    "kind": row["kind"], "mode": row["mode"]}
        if _digest(json.dumps(identity, sort_keys=True).encode()) != resource["fingerprint"]:
            raise ValueError("retained workspace resource fingerprint differs")
    closure.check_roots()
    return tuple(sorted(closure.absent))
