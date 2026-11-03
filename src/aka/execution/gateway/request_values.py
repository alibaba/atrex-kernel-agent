"""Pure Gateway request syntax; policy and physical submission stay with the caller."""
from __future__ import annotations
import ast
import re
from pathlib import Path, PurePosixPath
from typing import Iterable


def safe_relative(value: str) -> str:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"path must be relative to the workspace: {value!r}")
    normalized = path.as_posix()
    if normalized in ("", "."):
        raise ValueError(f"path must not resolve to the workspace root: {value!r}")
    return normalized


def python_inline_imports(parts: list[str]) -> set[str]:
    """Return top-level modules imported by a direct ``python -c`` command."""
    if not parts or not re.fullmatch(r"python(?:[0-9.]+)?", Path(parts[0]).name):
        return set()
    try:
        code_index = parts.index("-c") + 1
        tree = ast.parse(parts[code_index])
    except (ValueError, IndexError, SyntaxError):
        return set()
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".", 1)[0])
    return imported


def parse_env_items(items: Iterable[str]) -> dict[str, str]:
    env_vars: dict[str, str] = {}
    for item in items:
        if "=" not in item or item.startswith("="):
            raise ValueError(f"invalid --env {item!r}; expected KEY=VALUE")
        key, value = item.split("=", 1)
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError(f"invalid --env key {key!r}")
        env_vars[key] = value
    return env_vars

