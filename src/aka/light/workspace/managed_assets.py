"""Persistent Agent-owned tools/Skills and Runtime-managed asset boundaries."""

from __future__ import annotations

import os
import stat
from pathlib import Path

from aka.light.agent.home import open_private_directory
from aka.light.workspace.files import publish

TREES = ("skills", "tools")
PROTECTED_PATHS_ENV = "ATREX_MANAGED_PROTECTED_PATHS"
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
requests and outputs in `scratch/`, not here. The files remain in this trajectory's
persistent workspace across Episodes.

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
The files remain in this trajectory's persistent workspace across Episodes.

## Custom tool index

No custom tools yet.
""",
}


def reserved_paths(plugin_registry) -> tuple[str, ...]:
    from aka.light.workspace.assets import RETIRED_SKILL_NAMES
    from aka.light.workspace.skill_manifest import SKILL_MANIFEST

    names = set(SKILL_MANIFEST) | set(RETIRED_SKILL_NAMES)
    for plugin in plugin_registry.plugins:
        names.update(plugin.manifest.get("skills", {}))
    return (
        "tools/sandbox.py",
        "tools/plugin.py",
        *(f"skills/{name}" for name in sorted(names)),
    )


def is_reserved(relative: str, reserved: tuple[str, ...]) -> bool:
    return any(relative == name or relative.startswith(name + "/") for name in reserved)


def custom_entries(workspace: Path, reserved: tuple[str, ...]):
    """Bound directory enumeration and reads without following mutable links."""
    count = total = 0

    def walk(directory, prefix, depth):
        nonlocal count, total
        if depth > MAX_DEPTH:
            raise ValueError("Reusable files exceed 32 directory levels")
        with os.scandir(directory) as scan:
            for entry in scan:
                count += 1
                if count > MAX_ENTRIES:
                    raise ValueError(
                        "Reusable files exceed 4096 entries; remove generated files"
                    )
                relative = f"{prefix}/{entry.name}"
                if (
                    entry.name in IGNORED
                    or entry.name.endswith((".pyc", ".pyo"))
                    or is_reserved(relative, reserved)
                ):
                    continue
                mode = entry.stat(follow_symlinks=False).st_mode
                if stat.S_ISDIR(mode):
                    child = os.open(
                        entry.name,
                        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                        dir_fd=directory,
                    )
                    try:
                        yield relative, None, False
                        yield from walk(child, relative, depth + 1)
                    finally:
                        os.close(child)
                elif stat.S_ISREG(mode):
                    descriptor = os.open(
                        entry.name,
                        os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                        dir_fd=directory,
                    )
                    with os.fdopen(descriptor, "rb") as stream:
                        before = os.fstat(stream.fileno())
                        if not stat.S_ISREG(before.st_mode):
                            raise ValueError("Reusable files must be regular files")
                        limit = min(MAX_FILE, MAX_TOTAL - total)
                        if before.st_size > limit:
                            raise ValueError(
                                "Reusable files exceed 16 MiB/file or 64 MiB total"
                            )
                        data = stream.read(limit + 1)
                        after = os.fstat(stream.fileno())
                        if len(data) > limit:
                            raise ValueError(
                                "Reusable files grew beyond the validation limit"
                            )
                        if (
                            before.st_size,
                            before.st_mtime_ns,
                            before.st_ctime_ns,
                        ) != (
                            after.st_size,
                            after.st_mtime_ns,
                            after.st_ctime_ns,
                        ):
                            raise ValueError(
                                "Reusable file changed while validating; retry after writes finish"
                            )
                    total += len(data)
                    yield relative, data, bool(before.st_mode & 0o111)
                else:
                    raise ValueError(
                        f"Reusable file is a symlink or special file: {relative}"
                    )

    for tree in TREES:
        with open_private_directory(workspace / tree) as directory:
            yield from walk(directory, tree, 0)


def ensure_readmes(workspace: Path) -> None:
    for tree, content in READMES.items():
        path = workspace / tree / "README.md"
        if not path.is_file():
            publish(workspace, f"{tree}/README.md", content.encode())
