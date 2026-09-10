"""Language-neutral argv execution using JSON stdin/stdout, with bounded lifetime."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

from .schema import PluginError, decode_json


def resolve_command(root: Path, declaration: dict) -> tuple[str, ...]:
    if "entrypoint" in declaration:
        entry = declaration["entrypoint"]
        if not isinstance(entry, str):
            raise PluginError(
                "invalid_manifest", "entrypoint must be a relative Python file"
            )
        path = (root / entry).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise PluginError("invalid_manifest", f"missing entrypoint: {entry}")
        return (sys.executable, str(path))
    command = declaration["command"]
    if (
        not isinstance(command, list)
        or not command
        or any(not isinstance(arg, str) or "\0" in arg for arg in command)
    ):
        raise PluginError("invalid_manifest", "command must be a nonempty argv array")
    argv = [
        arg.replace("{plugin_root}", str(root)).replace("{python}", sys.executable)
        for arg in command
    ]
    executable = argv[0]
    if "/" in executable:
        path = (root / executable).resolve()
        executable = str(path) if path.is_file() and os.access(path, os.X_OK) else ""
    else:
        executable = shutil.which(executable) or ""
    if not executable:
        raise PluginError(
            "invalid_manifest", f"tool executable is unavailable: {argv[0]}"
        )
    return (executable, *argv[1:])


def execute_json(
    command: tuple[str, ...],
    request_json: str,
    *,
    cwd: Path,
    environment: dict[str, str],
    timeout: int,
    tool_name: str,
) -> object:
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=cwd,
        env=environment,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(request_json, timeout=timeout)
    except BaseException as exc:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.communicate()
        if isinstance(exc, subprocess.TimeoutExpired):
            raise PluginError(
                "tool_timeout", f"{tool_name} exceeded its timeout"
            ) from exc
        raise
    if process.returncode:
        raise PluginError(
            "tool_failed", f"{tool_name} exited {process.returncode}: {stderr[-2000:]}"
        )
    try:
        return decode_json(stdout)
    except ValueError as exc:
        raise PluginError("invalid_output", f"{tool_name} did not return JSON") from exc
