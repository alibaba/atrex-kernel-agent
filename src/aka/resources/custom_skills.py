"""Refresh selected workspace-owned Skill discovery without loading custom code."""
import os
from pathlib import Path
import stat

from aka.filesystem import open_private_directory, read_regular_bytes
from .plugin_runtime.registry import SKILL_ROOTS

MAX_ENTRIES = 4096
MAX_FILE = 16 * 1024 * 1024
MAX_TOTAL = 64 * 1024 * 1024


def refresh_custom_skills(registry, workspace, *, directory="skills"):
    workspace = Path(workspace).resolve(strict=True)
    path = Path(directory)
    if (not path.parts or path.is_absolute() or ".." in path.parts or path.as_posix() != directory
            or any(part.casefold() == ".git" for part in path.parts) or "\\" in directory
            or any(ord(char) < 32 for char in directory)):
        raise ValueError("custom Skill directory must be a strict workspace-relative path")
    reserved = registry.installed_paths(workspace)
    count = total = 0
    names = set()

    def walk(parent, prefix, depth):
        nonlocal count, total
        if depth > 32:
            raise ValueError("custom Skills exceed 32 directory levels")
        with os.scandir(parent) as entries:
            for entry in entries:
                count += 1
                if count > MAX_ENTRIES:
                    raise ValueError("custom Skills exceed 4096 entries")
                relative = f"{prefix}/{entry.name}"
                if any(relative == name or relative.startswith(name + "/") for name in reserved):
                    continue
                if entry.name in {"__pycache__", ".git", ".pytest_cache", ".mypy_cache", ".ruff_cache"} or entry.name.endswith((".pyc", ".pyo")):
                    continue
                mode = entry.stat(follow_symlinks=False).st_mode
                if stat.S_ISDIR(mode):
                    child = os.open(entry.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
                    try:
                        walk(child, relative, depth + 1)
                    finally:
                        os.close(child)
                elif stat.S_ISREG(mode):
                    content = read_regular_bytes(workspace / relative, limit=MAX_FILE + 1)
                    total += len(content)
                    if len(content) > MAX_FILE or total > MAX_TOTAL:
                        raise ValueError("custom Skills exceed bounded file content limits")
                    candidate = Path(relative)
                    if len(candidate.parts) == len(path.parts) + 2 and candidate.name == "SKILL.md":
                        names.add(candidate.parts[len(path.parts)])
                else:
                    raise ValueError(f"custom Skill contains a link or special file: {relative}")

    with open_private_directory(workspace / path) as parent:
        walk(parent, directory, 0)
    for destination in SKILL_ROOTS:
        root = workspace / destination
        with open_private_directory(root) as parent:
            with os.scandir(parent) as entries:
                for index, entry in enumerate(entries):
                    if index >= MAX_ENTRIES:
                        raise ValueError("Skill discovery exceeds 4096 entries")
                    if any(f"{destination}/{entry.name}" == name or f"{destination}/{entry.name}".startswith(name + "/") for name in reserved):
                        continue
                    try:
                        target = os.readlink(entry.name, dir_fd=parent)
                    except OSError:
                        continue
                    expected = workspace / path / entry.name
                    if Path(os.path.abspath(root / target)) == expected and entry.name not in names:
                        os.unlink(entry.name, dir_fd=parent)
            for name in sorted(names):
                try:
                    os.stat(name, dir_fd=parent, follow_symlinks=False)
                except FileNotFoundError:
                    os.symlink(workspace / path / name, name, dir_fd=parent)
