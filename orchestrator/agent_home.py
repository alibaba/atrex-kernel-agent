"""Session-local Provider state; never mount an operator's whole Provider Home."""

from __future__ import annotations

import hashlib
import os
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path

from .session_tail import read_regular_bytes

STATE_FILES = {
    "claude": (".claude.json", ".claude/settings.json", ".claude/.credentials.json"),
    "codex": (".codex/auth.json", ".codex/config.toml", ".codex/models_cache.json"),
    "qodercli": (
        ".qoder/settings.json",
        ".qoder/state.json",
        ".qoder/auth.json",
        ".qoder/credentials.json",
        ".qoder/installation_id",
        ".qoder/.auth/user",
        ".qoder/.auth/machine_id",
        ".qodersec/config.yaml",
        ".qodersec/runtime.json",
        ".qodersec/.config-version",
    ),
    "pi": (".pi/agent/auth.json", ".pi/agent/settings.json", ".pi/agent/models.json"),
}
CONFIG_ROOTS = {
    "claude": ("CLAUDE_CONFIG_DIR", ".claude"),
    "codex": ("CODEX_HOME", ".codex"),
    "pi": ("PI_CODING_AGENT_DIR", ".pi/agent"),
}
HOST_HOME_ENV = "ATREX_AGENT_HOST_HOME"
PREPARED_ENV = "ATREX_AGENT_PREPARED_HOME"


def projected_backends(environment: dict[str, str]) -> tuple[str, ...]:
    values = [
        environment.get("ATREX_AGENT_CLI", ""),
        environment.get("ATREX_AGENT_SANDBOX_BACKEND", ""),
    ]
    for flag, backend in (("CODEX", "codex"), ("QODER", "qodercli")):
        if environment.get(f"ATREX_PLAN_REVIEW_{flag}_ENABLED") == "1":
            values.append(backend)
    return tuple(dict.fromkeys(value for value in values if value in STATE_FILES))


@contextmanager
def open_private_directory(path: Path) -> Iterator[int]:
    """Yield a pinned directory FD; create missing components with mode 0700.

    No component may be a symlink. This context owns and closes the returned FD.
    """
    path = Path(os.path.abspath(path))
    descriptor = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for name in path.parts[1:]:
            with suppress(FileExistsError):
                os.mkdir(name, mode=0o700, dir_fd=descriptor)
            child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        yield descriptor
    finally:
        os.close(descriptor)


def _seed(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    # Reject links/devices and bound each configuration file, including auth.
    content = read_regular_bytes(source, limit=8 * 1024 * 1024 + 1)
    if len(content) > 8 * 1024 * 1024:
        raise ValueError(f"Provider configuration file exceeds 8 MiB: {source.name}")
    with open_private_directory(destination.parent) as parent:
        try:
            descriptor = os.open(
                destination.name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=parent,
            )
        except FileExistsError:
            # Preserve this Session's existing login/settings on resume. The
            # Agent's local symlink cannot make the Supervisor write elsewhere.
            return
        with os.fdopen(descriptor, "wb") as output:
            output.write(content)


def _git_identity(workspace: Path, environment: dict[str, str]) -> dict[str, str]:
    """Preserve legacy commits without copying host Git hooks/includes/credentials."""
    identity = {}
    for field in ("name", "email"):
        result = subprocess.run(
            ["git", "config", "--get", f"user.{field}"],
            cwd=workspace,
            env=environment,
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
        value = result.stdout.strip() if result.returncode == 0 else ""
        for kind in ("AUTHOR", "COMMITTER"):
            key = f"GIT_{kind}_{field.upper()}"
            if environment.get(key) or value:
                identity[key] = environment.get(key) or value
    return identity


def _seed_provider(backend: str, host_home: Path, home: Path, environment: dict[str, str]) -> None:
    for relative in STATE_FILES[backend]:
        source = host_home / relative
        variable, prefix = CONFIG_ROOTS.get(backend, ("", ""))
        if variable and environment.get(variable) and relative.startswith(prefix + "/"):
            source = (
                Path(environment[variable]).expanduser().resolve() / relative[len(prefix) + 1 :]
            )
        _seed(source, home / relative)


def prepare_agent_environment(
    workspace: Path,
    environment: dict[str, str],
    session_key: str,
) -> dict[str, str]:
    """Prepare before constructing a Codex ledger observer; resume reuses this Home."""
    from .agent_sandbox import sandbox_executable

    if sandbox_executable(environment) is None:
        return environment
    if environment.get(PREPARED_ENV):
        return environment
    values = dict(environment)
    host_home = Path(values.get("HOME") or str(Path.home())).expanduser().resolve()
    role = values.get("ATREX_AGENT_WORKSPACE_ROLE", "optimizer")
    identity = "\0".join(
        (str(workspace.resolve()), values.get("ATREX_AGENT_CLI", ""), role, session_key)
    )
    key = hashlib.sha256(identity.encode()).hexdigest()[:32]
    state_root = workspace.resolve().parent / ".atrex-agent-homes"
    if state_root.is_symlink():
        raise ValueError("Agent Home root cannot be a symlink")
    home = state_root / key
    if home.is_symlink():
        raise ValueError("Agent Home cannot be a symlink")
    with open_private_directory(home):
        pass
    for backend in projected_backends(values):
        _seed_provider(backend, host_home, home, values)
    # Optional campaign-persistent consultations must not resume a thread in a
    # newly empty Home every Episode. Keep each reviewer's state separate from
    # both the operator Home and the primary Agent's per-Session Home.
    if role == "optimizer":
        for name, backend in (("CODEX", "codex"), ("QODER", "qodercli")):
            session_file = values.get(f"ATREX_{name}_REVIEW_SESSION_FILE")
            if not session_file:
                continue
            reviewer_home = Path(session_file).parent / f"{backend}-home"
            with open_private_directory(reviewer_home):
                pass
            _seed_provider(backend, host_home, reviewer_home, values)
            if backend == "codex":
                reviewer_home = reviewer_home / ".codex"
                with open_private_directory(reviewer_home):
                    pass
            values[f"ATREX_{name}_REVIEW_HOME"] = str(reviewer_home)
    # GPU configuration stays in the Supervisor; only legacy Git identity is
    # still needed by the optimizer until the separate Git-ownership migration.
    if role == "optimizer":
        values.update(_git_identity(workspace, values))
    for variable, relative in CONFIG_ROOTS.values():
        with open_private_directory(home / relative):
            pass
        values[variable] = str(home / relative)
    for name in (".cache", ".config", ".local/share", ".local/state", ".qoder", ".qodersec"):
        with open_private_directory(home / name):
            pass
    values.update(
        HOME=str(home),
        XDG_CONFIG_HOME=str(home / ".config"),
        XDG_CACHE_HOME=str(home / ".cache"),
        XDG_DATA_HOME=str(home / ".local/share"),
        XDG_STATE_HOME=str(home / ".local/state"),
    )
    values[HOST_HOME_ENV] = str(host_home)
    values[PREPARED_ENV] = str(home)
    return values
