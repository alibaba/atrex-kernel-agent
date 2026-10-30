"""Private per-session Home; seed only the selected backend's declared files."""

from __future__ import annotations
import hashlib
import os
import stat
from pathlib import Path
from aka.filesystem import open_private_directory

PREPARED_ENV = "ATREX_AGENT_PREPARED_HOME"


def read_regular_file(path, limit=8 * 1024 * 1024):
    path = Path(os.path.abspath(path))
    directory = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for name in path.parts[1:-1]:
            child = os.open(
                name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory
            )
            os.close(directory)
            directory = child
        fd = os.open(
            path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory
        )
    finally:
        os.close(directory)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("Backend configuration must be a regular file")
        data = stream.read(limit + 1)
        if len(data) > limit:
            raise ValueError("Backend configuration exceeds its size limit")
        return data, info


def read_regular_bytes(path, limit=8 * 1024 * 1024):
    return read_regular_file(path, limit)[0]


def prepare_home(workspace, session_id, backend, environment, *, state_root=None):
    host = Path(environment.get("HOME") or Path.home()).expanduser().resolve()
    state = (
        Path(state_root)
        if state_root is not None
        else workspace.parent / ".aka-agent-homes"
    )
    key = hashlib.sha256(
        (str(workspace) + "\0" + backend.id + "\0" + session_id).encode()
    ).hexdigest()
    state = Path(os.path.abspath(state))
    if state.resolve() != state or state in {
        Path("/"),
        Path("/tmp"),
        Path("/run"),
        host,
    }:
        raise ValueError(
            "Agent Home storage must be a dedicated directory without symlinks"
        )
    home = state / key
    if state.absolute().is_relative_to(workspace) or workspace.is_relative_to(
        state.absolute()
    ):
        raise ValueError("Agent Home storage must be outside its workspace")
    if state.exists():
        info = state.stat()
        if info.st_uid != os.geteuid() or info.st_mode & 0o077:
            raise ValueError("Agent Home storage must be privately owned")
    with open_private_directory(home) as directory:
        info = os.fstat(directory)
        if info.st_uid != os.geteuid() or info.st_mode & 0o077:
            raise ValueError("Agent session Home must be privately owned")
    for relative in backend.files.config_files:
        rel = Path(relative)
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError("Backend configuration paths must be relative")
        source = host / rel
        for variable, prefix in backend.files.environment_roots:
            if environment.get(variable) and relative.startswith(prefix + "/"):
                source = (
                    Path(environment[variable]).expanduser()
                    / relative[len(prefix) + 1 :]
                )
        try:
            content = read_regular_bytes(source)
        except FileNotFoundError:
            continue
        destination = home / rel
        with open_private_directory(destination.parent) as directory:
            try:
                fd = os.open(
                    destination.name,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                    0o600,
                    dir_fd=directory,
                )
            except FileExistsError:
                # Resume keeps existing credentials/settings; validate rather than follow links.
                read_regular_bytes(destination)
                continue
            with os.fdopen(fd, "wb") as output:
                output.write(content)
    values = dict(environment)
    for variable, relative in backend.files.environment_roots:
        with open_private_directory(home / relative):
            pass
        values[variable] = str(home / relative)
    for name in (".cache", ".config", ".local/share", ".local/state"):
        with open_private_directory(home / name):
            pass
    values.update(
        HOME=str(home),
        XDG_CONFIG_HOME=str(home / ".config"),
        XDG_CACHE_HOME=str(home / ".cache"),
        XDG_DATA_HOME=str(home / ".local/share"),
        XDG_STATE_HOME=str(home / ".local/state"),
    )
    values[PREPARED_ENV] = str(home)
    return values, host, home
