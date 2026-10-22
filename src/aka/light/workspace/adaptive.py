"""Campaign-private snapshots of untrusted, reusable Agent files (never code to import).

Each Episode pins its seed once. Session checkpoints are immutable and an atomic
per-Episode pointer selects the latest complete snapshot, independent of promotion.
"""
from __future__ import annotations

import hashlib
import fcntl
import json
import os
import re
import stat
from pathlib import Path

from aka.light.orchestrator.agent_home import open_private_directory
from aka.light.orchestrator.durable_state import durable_write_json, fsync_directory
from aka.light.orchestrator.session_tail import read_regular_bytes
from aka.light.supervisor.workspace import publish

TREES = ("skills", "tools")
PROTECTED_PATHS_ENV = "ATREX_ADAPTIVE_PROTECTED_PATHS"
MAX_FILE = 16 * 1024 * 1024
MAX_TOTAL = 64 * 1024 * 1024
MAX_ENTRIES = 4096
MAX_DEPTH = 32
IGNORED = {"__pycache__", ".git", ".pytest_cache", ".mypy_cache", ".ruff_cache"}

READMES = {
    "skills": """# Reusable Skills

Store reusable methods, workflows and lessons in `<name>/SKILL.md`, with supporting
files alongside it. Maintain an index below: name, purpose, when to use it and its
entry path. Update this README whenever you add, change or remove a custom Skill.
Read new Skills directly; discovery links are refreshed before the next Session.

Supervisor-installed Skills are reserved and read-only in sandboxed Sessions.
Do not replace them. Your own Skill directories are writable. Put one-off probes,
requests and outputs in `scratch/`, not here. The Session contract states whether
custom Skills carry forward to the next Episode; current files are your starting state.

## Custom Skill index

No custom Skills yet.
""",
    "tools": """# Reusable tools

Store reusable scripts here. Maintain an index below: path, purpose, invocation,
dependencies and important limitations. Update this README whenever you add,
change or remove a custom tool. Use `scratch/` for temporary inputs and outputs.

`sandbox.py` and `plugin.py` are Supervisor-installed HTTP clients: do not modify,
replace or duplicate them. Your other scripts are writable. To execute a custom
GPU probe, use the GPU measurement Skill, for example:
`python3 tools/sandbox.py --kind dev --no-sync -- python3 tools/probe.py`.
Declare extra script dependencies with `--input`; local helpers may run directly.
The Session contract states whether custom tools carry forward to the next Episode.

## Custom tool index

No custom tools yet.
""",
}


def reserved_paths(plugin_registry) -> tuple[str, ...]:
    from aka.light.orchestrator.agent_skill_manifest import SKILL_MANIFEST
    from aka.light.orchestrator.workspace_runtime import RETIRED_SKILL_NAMES
    names = set(SKILL_MANIFEST) | set(RETIRED_SKILL_NAMES)
    for plugin in getattr(plugin_registry, "plugins", ()):
        names.update(plugin.manifest.get("skills", {}))
    skill_catalog = getattr(plugin_registry, "skill_catalog", None)
    if callable(skill_catalog):
        for row in skill_catalog():
            name = row.get("install_name")
            if isinstance(name, str) and name and "/" not in name and name not in {".", ".."}:
                names.add(name)
    return ("tools/sandbox.py", "tools/plugin.py", *(f"skills/{name}" for name in sorted(names)))


def is_reserved(relative: str, reserved: tuple[str, ...]) -> bool:
    return any(relative == name or relative.startswith(name + "/") for name in reserved)


def custom_entries(workspace: Path, reserved: tuple[str, ...]):
    """Bound directory enumeration and reads; pin every traversed directory FD."""
    count = total = 0

    def walk(directory, prefix, depth):
        nonlocal count, total
        if depth > MAX_DEPTH:
            raise ValueError("Reusable files exceed 32 directory levels")
        with os.scandir(directory) as scan:
            for entry in scan:
                count += 1
                if count > MAX_ENTRIES:
                    raise ValueError("Reusable files exceed 4096 entries; remove generated files")
                relative = f"{prefix}/{entry.name}"
                if (entry.name in IGNORED or entry.name.endswith((".pyc", ".pyo"))
                        or is_reserved(relative, reserved)):
                    continue
                mode = entry.stat(follow_symlinks=False).st_mode
                if stat.S_ISDIR(mode):
                    child = os.open(entry.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                    dir_fd=directory)
                    try:
                        yield relative, None, False
                        yield from walk(child, relative, depth + 1)
                    finally:
                        os.close(child)
                elif stat.S_ISREG(mode):
                    descriptor = os.open(entry.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                                         dir_fd=directory)
                    with os.fdopen(descriptor, "rb") as stream:
                        before = os.fstat(stream.fileno())
                        if not stat.S_ISREG(before.st_mode):
                            raise ValueError("Reusable files must be regular files")
                        limit = min(MAX_FILE, MAX_TOTAL - total)
                        if before.st_size > limit:
                            raise ValueError("Reusable files exceed 16 MiB/file or 64 MiB total")
                        data = stream.read(limit + 1)
                        after = os.fstat(stream.fileno())
                        if len(data) > limit:
                            raise ValueError("Reusable files grew beyond the snapshot limit")
                        if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                                after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                            raise ValueError("Reusable file changed while saving; retry after writes finish")
                    total += len(data)
                    yield relative, data, bool(before.st_mode & 0o111)
                else:
                    raise ValueError(f"Reusable file is a symlink or special file: {relative}")

    for tree in TREES:
        with open_private_directory(workspace / tree) as directory:
            yield from walk(directory, tree, 0)


def ensure_readmes(workspace: Path) -> None:
    for tree, content in READMES.items():
        try:
            read_regular_bytes(workspace / tree / "README.md", limit=MAX_FILE + 1)
        except FileNotFoundError:
            publish(workspace, f"{tree}/README.md", content.encode())


def write_entry(root: Path, relative: str, data: bytes | None, executable=False) -> None:
    if data is None:
        with open_private_directory(root / relative):
            pass
    else:
        publish(root, relative, data)
        if executable:
            with open_private_directory((root / relative).parent) as parent:
                fd = os.open(Path(relative).name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent)
                try:
                    os.fchmod(fd, 0o700)
                finally:
                    os.close(fd)


def read_manifest(path: Path) -> dict:
    raw = read_regular_bytes(path, limit=2 * 1024 * 1024 + 1)
    if len(raw) > 2 * 1024 * 1024:
        raise ValueError("Reusable-state manifest exceeds 2 MiB")
    value = json.loads(raw)
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("Invalid reusable-state manifest")
    return value


class AdaptiveState:
    def __init__(self, root: Path, episode: int, reserved: tuple[str, ...]):
        if type(episode) is not int or episode < 1:
            raise ValueError("Reusable state needs a controller Episode number")
        self.root, self.episode, self.reserved = root, episode, reserved

    @property
    def pointer(self):
        return self.root / "episodes" / f"e{self.episode:08d}.json"

    def previous(self) -> dict | None:
        # A single active Episode is normal; never let a late older Session
        # overwrite the starting point for an already initialized newer draft.
        directory = self.root / "episodes"
        with open_private_directory(directory):
            pass
        for path in sorted(directory.glob("e*.json"), reverse=True):
            if re.fullmatch(r"e[0-9]{8,}\.json", path.name) and int(path.stem[1:]) < self.episode:
                return read_manifest(path)
        return None

    def restore(self, workspace: Path, pointer: dict) -> None:
        key = pointer.get("snapshot")
        if not isinstance(key, str) or re.fullmatch(r"[0-9a-f]{64}", key) is None:
            raise ValueError("Invalid reusable-state snapshot identity")
        root = self.root / "snapshots" / key
        manifest = read_manifest(root / "manifest.json")
        rows = manifest.get("entries")
        if not isinstance(rows, list) or len(rows) > MAX_ENTRIES:
            raise ValueError("Invalid reusable-state entries")
        encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
        if hashlib.sha256(encoded).hexdigest() != key:
            raise ValueError("Reusable-state manifest digest mismatch")
        total = 0
        for row in rows:
            relative = row["path"]
            path = Path(relative)
            if (path.is_absolute() or ".." in path.parts or len(path.parts) < 2
                    or path.parts[0] not in TREES or path.as_posix() != relative):
                raise ValueError("Invalid reusable-state file path")
            if is_reserved(relative, self.reserved):
                continue  # A newly installed plugin may reserve an old custom name.
            data = None
            if row["sha256"] is not None:
                data = read_regular_bytes(root / relative, limit=min(MAX_FILE, MAX_TOTAL - total) + 1)
                total += len(data)
                if len(data) > MAX_FILE or total > MAX_TOTAL or hashlib.sha256(data).hexdigest() != row["sha256"]:
                    raise ValueError("Reusable-state content exceeds limits or digest mismatch")
            write_entry(workspace, relative, data, row["executable"])

    def save(self, workspace: Path) -> dict:
        # Serialize immutable publication/dedup across resumed controllers too.
        with open_private_directory(self.root) as directory:
            descriptor = os.open(".lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW,
                                 0o600, dir_fd=directory)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX)
                return self._save(workspace)
            finally:
                os.close(descriptor)

    def _save(self, workspace: Path) -> dict:
        import tempfile
        with open_private_directory(self.root / "snapshots"):
            pass
        rows = []
        with tempfile.TemporaryDirectory(prefix=".saving-", dir=self.root / "snapshots") as name:
            staging = Path(name)
            for relative, data, executable in custom_entries(workspace, self.reserved):
                write_entry(staging, relative, data, executable)
                if data is not None:
                    with (staging / relative).open("rb") as stream:
                        os.fsync(stream.fileno())
                rows.append({"path": relative, "sha256": hashlib.sha256(data).hexdigest() if data is not None else None,
                             "executable": executable})
            rows.sort(key=lambda row: row["path"])
            key = hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            durable_write_json(staging / "manifest.json", {"schema_version": 1, "entries": rows})
            for current, _, _ in os.walk(staging, topdown=False):
                fsync_directory(Path(current))
            target = self.root / "snapshots" / key
            if not target.exists():
                os.rename(staging, target)
                fsync_directory(target.parent)
            value = {"schema_version": 1, "episode": self.episode, "snapshot": key}
            durable_write_json(self.pointer, value)
            return value
