"""Narrow, read-only installation mounts, separate from Provider state and credentials."""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import sys
from collections.abc import Iterator, Mapping
from pathlib import Path

_CLI_PACKAGES = frozenset({
    "@anthropic-ai/claude-code", "@openai/codex", "@qoder-ai/qodercli",
    "@mariozechner/pi-coding-agent",
})
_PACKAGE_NAME = re.compile(r"(?:@[a-zA-Z0-9_.-]+/)?[a-zA-Z0-9_-][a-zA-Z0-9_.-]*\Z")
_PROVIDER_ROOTS = (".claude", ".codex", ".qoder", ".qodersec", ".pi")


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(path))


def _package_manifest(root: Path) -> dict:
    path = root / "package.json"
    if not path.is_file():
        return {}
    if path.stat().st_size > 1024 * 1024:
        raise RuntimeError(f"Agent installation package manifest is too large: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Invalid Agent installation package manifest: {path}") from error
    return value if isinstance(value, dict) else {}


def _package_root(executable: Path) -> Path | None:
    for root in executable.parents:
        # Never infer a package by a generic package.json in a project or Home.
        in_modules = root.parent.name == "node_modules" or (
            root.parent.name.startswith("@") and root.parent.parent.name == "node_modules"
        )
        legacy_claude = root.name == "local" and root.parent.name == ".claude"
        if in_modules or legacy_claude:
            manifest = _package_manifest(root)
            if manifest and (in_modules or manifest.get("name") in _CLI_PACKAGES):
                return root
    return None


def installation_mounts(
    command: list[str],
    backends: tuple[str, ...],
    environment: Mapping[str, str],
    host_home: Path,
    *,
    hidden_paths: tuple[Path, ...],
    forbidden_paths: tuple[Path, ...],
) -> tuple[tuple[Path, Path], ...]:
    """Restore executables, selected npm packages and Python libraries, never their parents.

    Sources are canonical; aliases are recreated as symlinks to the mounted target.
    Dependency roots must have a recognized installation layout. Unknown wrappers get
    only their exact file and shebang interpreter, not an inferred directory tree.
    """
    home = host_home.resolve()
    hidden = tuple(_absolute(path) for path in hidden_paths)
    forbidden = tuple(path.resolve() for path in forbidden_paths)
    provider_roots = tuple(home / name for name in _PROVIDER_ROOTS)
    mounts: dict[Path, Path] = {}
    packages: set[Path] = set()
    visited: set[Path] = set()

    def masked(path: Path) -> bool:
        return any(path.is_relative_to(root) for root in hidden)

    def add(source: Path, destination: Path | None = None) -> None:
        destination = _absolute(destination or source)
        source = source.resolve(strict=True)
        directory = source.is_dir()
        # Fail closed on both sides: an installation symlink must not bypass a
        # private-source mask or shadow the isolated workspace/provider Home.
        for path in (source, destination):
            if any(path.is_relative_to(root) or (directory and root.is_relative_to(path)) for root in forbidden):
                raise RuntimeError(f"Agent installation mount overlaps private Runtime paths: {path}")
            if path == home or (directory and home.is_relative_to(path)):
                raise RuntimeError(f"Agent installation mount would expose host Home: {path}")
            for root in provider_roots:
                if path == root or (directory and root.is_relative_to(path)):
                    raise RuntimeError(f"Agent installation mount would expose Provider Home: {path}")
                if path.is_relative_to(root):
                    relative = path.relative_to(root)
                    # Only code layouts are installation candidates inside a
                    # Provider Home. Auth/history files never enter through here.
                    allowed = {"bin", "local"} if root.name == ".claude" else {"bin"}
                    if relative.parts[0] not in allowed:
                        raise RuntimeError(f"Agent installation path points into Provider state: {path}")
        if source != destination and directory and masked(destination) and masked(source):
            # A declared package symlink may use another installation directory,
            # but not silently turn an arbitrary project into a code mount.
            if _package_root(source / "__entry__") != source:
                raise RuntimeError(f"Unrecognized Agent package symlink target: {source}")
        if masked(destination):
            previous = mounts.get(destination)
            if previous is not None and previous != source:
                raise RuntimeError(f"Conflicting Agent installation mounts: {destination}")
            mounts[destination] = source

    def package_dependencies(canonical: Path) -> Iterator[Path]:
        manifest = _package_manifest(canonical)
        dependencies: dict[str, object] = {}
        for field in ("dependencies", "optionalDependencies", "peerDependencies"):
            value = manifest.get(field, {})
            if isinstance(value, dict):
                dependencies.update(value)
        for name in sorted(dependencies):
            if not _PACKAGE_NAME.fullmatch(name):
                raise RuntimeError(f"Invalid Agent installation dependency name: {name!r}")
            for parent in (canonical, *canonical.parents):
                dependency = parent / "node_modules" / name
                if (dependency / "package.json").is_file():
                    yield dependency
                    break  # Node's nearest installed dependency wins.

    def package(root: Path) -> None:
        # Explicit DFS preserves dependency/alias validation order without using
        # Python's call stack for a potentially deep installed-package graph.
        pending = [iter((root,))]
        while pending:
            selected = next(pending[-1], None)
            if selected is None:
                pending.pop()
                continue
            canonical = selected.resolve(strict=True)
            add(selected)
            add(canonical)
            if canonical in packages:
                continue
            packages.add(canonical)
            if len(packages) > 512:
                raise RuntimeError("Agent installation dependency graph exceeds 512 packages")
            pending.append(package_dependencies(canonical))

    def executable(value: str) -> None:
        if not value:
            return
        selected = value if Path(value).is_absolute() else shutil.which(value, path=environment.get("PATH"))
        if not selected:
            return  # Existing CLI discovery reports absent backends.
        path = _absolute(Path(selected))
        if path in visited:
            return
        visited.add(path)
        if len(visited) > 64:
            raise RuntimeError("Agent installation interpreter chain exceeds 64 files")
        target = path.resolve(strict=True)
        if not target.is_file():
            raise RuntimeError(f"Agent executable is not a regular file: {path}")
        add(target, path)
        add(target)
        for location in dict.fromkeys((path, target)):
            root = _package_root(location)
            if root is not None:
                package(root)
            # A venv beneath a repository needs its libraries/config, not the
            # repository or all sibling virtualenvs. Also handles pyenv/Conda.
            if location.parent.name == "bin" and re.fullmatch(r"python(?:\d+(?:\.\d+)*)?", location.name):
                prefix = location.parent.parent
                config = prefix / "pyvenv.cfg"
                library = prefix / "lib"
                if config.is_file() or any(library.glob("python[0-9]*")):
                    if config.is_file():
                        add(config)
                    if library.is_dir():
                        add(library)
                        add(library.resolve())
        # Shell/Python/npm launchers may depend on a user-installed interpreter.
        with target.open("rb") as stream:
            first = stream.readline(4096)
        if first.startswith(b"#!"):
            try:
                words = shlex.split(first[2:].decode("utf-8").strip())
            except (UnicodeError, ValueError) as error:
                raise RuntimeError(f"Invalid Agent executable shebang: {path}") from error
            if words:
                executable(words[0])
                if Path(words[0]).name == "env":
                    for word in words[1:]:
                        if not word.startswith("-") and "=" not in word:
                            executable(word)
                            break

    for value in dict.fromkeys((command[0] if command else "", *backends, sys.executable, "python3", "node")):
        executable(value)
    # Parent package/library mounts precede entrypoint aliases nested inside them.
    # A package mount already preserves its internal symlinks. Only recreate
    # aliases outside restored trees (for example ~/.local/bin/codex).
    result: list[tuple[Path, Path]] = []
    for destination, source in sorted(mounts.items(), key=lambda item: (len(item[0].parts), str(item[0]))):
        if any(
            destination.is_relative_to(parent)
            and source == (base / destination.relative_to(parent)).resolve()
            for base, parent in result if base.is_dir()
        ):
            continue
        result.append((source, destination))
    return tuple(result)
