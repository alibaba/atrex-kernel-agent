"""Shared syntactic argv helpers, without Gateway policy or submission state."""

from __future__ import annotations
import ast
import re
import shlex
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


def safe_relative(value: str) -> str:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"path must be relative to the workspace: {value!r}")
    normalized = path.as_posix()
    if normalized in ("", "."):
        raise ValueError(f"path must not resolve to the workspace root: {value!r}")
    return normalized


def parsed_command_parts(parts: list[str]) -> tuple[list[str], bool]:
    """Return command words and whether a single shell string stayed opaque."""
    command = parts[1:] if parts and parts[0] == "--" else list(parts)
    if len(command) != 1:
        return command, False
    try:
        parsed = shlex.split(command[0])
    except ValueError:
        return command, True
    if shlex.join(parsed) == command[0]:
        return parsed, False
    return command, True


def command_parts(parts: list[str]) -> list[str]:
    return parsed_command_parts(parts)[0]


def option_value(parts: list[str], name: str, default: Any = None) -> Any:
    """Read a simple ``--flag value``/``--flag=value`` option from command argv."""
    command = command_parts(parts)
    for index, token in enumerate(command):
        if token == name:
            return command[index + 1] if index + 1 < len(command) else default
        if token.startswith(name + "="):
            return token.split("=", 1)[1]
    return default


def option_values(parts: list[str], name: str) -> list[str]:
    """Read every repeated ``--flag value``/``--flag=value`` option."""
    command = command_parts(parts)
    values: list[str] = []
    for index, token in enumerate(command):
        if token == name:
            if index + 1 >= len(command):
                raise ValueError(f"{name} requires a value")
            values.append(command[index + 1])
        elif token.startswith(name + "="):
            values.append(token.split("=", 1)[1])
    return values


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


def is_fp4_dtype(value: object) -> bool:
    if not isinstance(value, str):
        return False
    normalized = value.casefold().replace("-", "").replace("_", "")
    return "fp4" in normalized or "float4" in normalized


def metadata_has_fp4_dtype(metadata: object) -> bool:
    if isinstance(metadata, dict):
        if any(is_fp4_dtype(metadata.get(field)) for field in ("dtype", "dtype_compute")):
            return True
        return any(metadata_has_fp4_dtype(value) for value in metadata.values())
    if isinstance(metadata, list):
        return any(metadata_has_fp4_dtype(value) for value in metadata)
    return False
