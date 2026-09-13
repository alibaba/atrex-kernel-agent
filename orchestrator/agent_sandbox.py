"""Bubblewrap boundary for AKA coding-agent processes.

The Agent's HOME and working directory are the same Git-free workspace.
Selected Skills are read-only; tools are seeded into an Episode-local writable directory.
Supervisor implementations, private evaluator inputs, and Agate/Wiki
credentials remain outside the namespace.
"""

from __future__ import annotations

import hashlib
import os
import platform
import re
import shutil
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

from .agent_assets import DEFAULT_AGENT_SKILLS, initialize_writable_tools, materialize_agent_assets
from .agent_workspace import AgentWorkspace, project_reference_tree

VISIBLE_WORKSPACE = Path("/home/agent/workspace")
VISIBLE_HOME = VISIBLE_WORKSPACE
_PROTECTED_WORKSPACE_PATHS = (
    "CLAUDE.md",
    "README.md",
    ".gitignore",
    "agent_problem.json",
    "definition.json",
    "input.py",
    "metadata.json",
    "profile_driver.py",
    "reference.py",
    "roofline.json",
    "shapes.json",
    "test_kernel.py",
    "valid.py",
    "workload.jsonl",
)
_CLAUDE_READ_ONLY = (Path(".claude/.credentials.json"), Path(".claude/plugins"))
_CLAUDE_STATE_FILES = (Path(".claude.json"), Path(".claude/settings.json"))
_CODEX_READ_ONLY = (Path(".codex/auth.json"), Path(".codex/skills"))
_CODEX_STATE_FILES = tuple(
    Path(".codex") / name
    for name in (".sandbox_migration", "config.toml", "installation_id", "models_cache.json")
)
_QODER_STATE_FILES = tuple(
    Path(".qoder") / name
    for name in ("settings.json", "state.json", "installation_id", ".last-cleanup")
)
_CODEX_CA_BUNDLE_CANDIDATES = (
    Path("/etc/ssl/certs/ca-certificates.crt"),
    Path("/etc/pki/tls/certs/ca-bundle.crt"),
    Path("/etc/ssl/ca-bundle.pem"),
)


def _bwrap_path(value: str) -> str | None:
    candidate = Path(value).expanduser()
    if candidate.is_absolute() and candidate.is_file() and os.access(candidate, os.X_OK):
        return str(candidate)
    return shutil.which(value)


def _enabled(mode: str, executable: str) -> str | None:
    if mode == "none":
        return None
    resolved = _bwrap_path(executable)
    if mode == "bwrap":
        if platform.system() != "Linux":
            raise RuntimeError("--agent-sandbox=bwrap requires Linux")
        if resolved is None:
            raise RuntimeError(f"Bubblewrap executable not found: {executable}")
        return resolved
    if mode != "auto":
        raise ValueError("Agent sandbox mode must be auto, bwrap, or none")
    return resolved if platform.system() == "Linux" else None


def _translate(value: str, workspace: Path) -> str:
    source = str(workspace)
    if value == source:
        return str(VISIBLE_WORKSPACE)
    return value.replace(source + os.sep, str(VISIBLE_WORKSPACE) + "/")


def _session_key(
    command: list[str],
    backend: str,
    environment: Mapping[str, str],
) -> str:
    for option in ("--session-id", "--resume"):
        try:
            index = command.index(option)
        except ValueError:
            continue
        if index + 1 < len(command):
            value = re.sub(r"[^A-Za-z0-9_.-]", "_", command[index + 1])[:120]
            if value:
                return value
    # Codex assigns its persistent thread id after the first process starts, so
    # the fresh and ``exec resume`` argv cannot share a command-derived key.
    # Long Horizon keeps the invocation prefix stable across resume turns.
    telemetry = environment.get("ATREX_TELEMETRY_ATTEMPT_ID", "")
    if backend == "codex" and telemetry:
        stable = re.sub(r"-[0-9]+$", "", telemetry)
        role = environment.get("ATREX_AGENT_SANDBOX_BACKEND", "main")
        value = re.sub(r"[^A-Za-z0-9_.-]", "_", f"{role}-{stable}")[:120]
        if value:
            return f"codex-{value}"
    digest = hashlib.sha256("\0".join(command).encode()).hexdigest()[:24]
    return f"{backend or 'agent'}-{digest}"


def _projected_backends(
    backend: str, environment: Mapping[str, str]
) -> tuple[str, ...]:
    values = [backend] if backend else []
    explicit = environment.get("ATREX_AGENT_SANDBOX_BACKEND", "")
    if explicit:
        values.append(explicit)
    if environment.get("ATREX_PLAN_REVIEW_CODEX_ENABLED") == "1":
        values.append("codex")
    if environment.get("ATREX_PLAN_REVIEW_QODER_ENABLED") == "1":
        values.append("qodercli")
    return tuple(
        dict.fromkeys(
            value for value in values if value in {"claude", "codex", "qodercli", "pi"}
        )
    )


def _copy_state_file(source: Path, destination: Path) -> None:
    if not source.is_file() or source.is_symlink() or source.stat().st_size > 8 * 1024 * 1024:
        return
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not destination.exists():
        shutil.copyfile(source, destination)
        destination.chmod(0o600)


def _prepare_provider_home(
    root: Path,
    key: str,
    backends: tuple[str, ...],
    host_home: Path,
) -> Path:
    home = root / key
    home.mkdir(parents=True, exist_ok=True, mode=0o700)
    if "claude" in backends:
        for relative in _CLAUDE_STATE_FILES:
            _copy_state_file(host_home / relative, home / relative)
        for name in ("backups", "cache", "projects", "sessions"):
            (home / ".claude" / name).mkdir(parents=True, exist_ok=True, mode=0o700)
    if "codex" in backends:
        for relative in _CODEX_STATE_FILES:
            _copy_state_file(host_home / relative, home / relative)
        for name in ("cache", "sessions", "shell_snapshots", "thread-writer-locks", "tmp"):
            (home / ".codex" / name).mkdir(parents=True, exist_ok=True, mode=0o700)
    if "qodercli" in backends:
        for relative in _QODER_STATE_FILES:
            _copy_state_file(host_home / relative, home / ".qoder-state" / relative.name)
        for relative in (
            "tasks", "projects", "logs", "tmp", "cache", ".cache",
            ".codebase-status", "session-env", "shell-snapshots",
            "external-commands/locks",
        ):
            (home / ".qoder-writable" / relative).mkdir(
                parents=True, exist_ok=True, mode=0o700
            )
        (home / ".qodersec-logs").mkdir(exist_ok=True, mode=0o700)
    if "pi" in backends:
        (home / ".pi" / "agent").mkdir(parents=True, exist_ok=True, mode=0o700)
        configured = os.environ.get("PI_CODING_AGENT_DIR")
        source = Path(configured).expanduser() if configured else host_home / ".pi/agent"
        for name in ("auth.json", "settings.json", "models.json"):
            _copy_state_file(source / name, home / ".pi/agent" / name)
    return home


def _parents(destination: Path) -> list[str]:
    values: list[str] = []
    parent = destination.parent
    while parent.as_posix() not in {"/", "."}:
        values.append(parent.as_posix())
        parent = parent.parent
    return list(reversed(values))


def _minimal_hidden_paths(paths: tuple[Path, ...]) -> tuple[Path, ...]:
    """Collapse host paths to the smallest safe set of bwrap masks."""
    resolved = sorted({path.resolve() for path in paths}, key=lambda path: len(path.parts))
    selected: list[Path] = []
    for path in resolved:
        if path == Path("/"):
            raise ValueError("Agent sandbox cannot hide the host root")
        if path.as_posix() in {"/home", "/root", "/run", "/tmp"}:
            continue
        if path.is_relative_to("/home") or path.is_relative_to("/root"):
            continue
        if not any(path.is_relative_to(parent) for parent in selected):
            selected.append(path)
    return tuple(selected)


def _mount(
    argv: list[str],
    source: Path,
    destination: Path,
    *,
    writable: bool,
    created: set[str],
) -> None:
    for parent in _parents(destination):
        if parent in created:
            continue
        argv += ["--dir", parent]
        created.add(parent)
    argv += ["--bind" if writable else "--ro-bind", str(source), str(destination)]
    if source.is_dir():
        created.add(destination.as_posix())


def _git_common_directory(workspace: Path) -> Path | None:
    result = subprocess.run(
        ["git", "rev-parse", "--git-common-dir"],
        cwd=workspace,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        return None
    value = Path(result.stdout.strip())
    return (workspace / value).resolve() if not value.is_absolute() else value.resolve()


def _installation_roots(
    command: list[str],
    backends: tuple[str, ...],
    environment: Mapping[str, str],
    host_home: Path,
) -> tuple[Path, ...]:
    paths: list[Path] = []
    for executable in (
        command[0] if command else "",
        *backends,
        sys.executable,
    ):
        resolved = (
            Path(executable)
            if Path(executable).is_absolute()
            else Path(shutil.which(executable, path=environment.get("PATH")) or "")
        )
        if not resolved.is_absolute():
            continue
        try:
            relative = resolved.absolute().relative_to(host_home)
        except ValueError:
            continue
        if relative.parts:
            root = host_home / relative.parts[0]
            if root.is_dir() and not root.is_symlink():
                paths.append(root)
    return tuple(dict.fromkeys(paths))


def _credential_mounts(
    backends: tuple[str, ...],
    host_home: Path,
    provider_home: Path,
) -> tuple[list[tuple[Path, Path, bool]], list[tuple[Path, Path]]]:
    mounts: list[tuple[Path, Path, bool]] = []
    overlays: list[tuple[Path, Path]] = []
    if "claude" in backends:
        for relative in _CLAUDE_READ_ONLY:
            source = host_home / relative
            if source.exists() and not source.is_symlink():
                mounts.append((source.resolve(), VISIBLE_HOME / relative, False))
    if "codex" in backends:
        for relative in _CODEX_READ_ONLY:
            source = host_home / relative
            if source.exists() and not source.is_symlink():
                mounts.append((source.resolve(), VISIBLE_HOME / relative, False))
    if "qodercli" in backends:
        for name in (".qoder", ".qodersec"):
            source = host_home / name
            if source.is_dir() and not source.is_symlink():
                mounts.append((source.resolve(), VISIBLE_HOME / name, False))
        for relative in _QODER_STATE_FILES:
            staged = provider_home / ".qoder-state" / relative.name
            if staged.is_file():
                overlays.append((staged, VISIBLE_HOME / relative))
    return mounts, overlays


def wrap_agent_command(
    command: list[str],
    *,
    workspace: Path,
    environment: Mapping[str, str],
    repository_root: Path,
    provider_homes: Path,
    hidden_host_paths: tuple[Path, ...],
    mode: str,
    bwrap_executable: str,
    agent_skills: tuple[str, ...] = DEFAULT_AGENT_SKILLS,
    agent_reference_projects: bool = False,
    agent_workspace: Path | None = None,
    provider_home: Path | None = None,
    read_only_paths: tuple[str, ...] | None = None,
) -> tuple[list[str], dict[str, str]]:
    """Return the bwrap command and the exact environment visible to the Agent."""
    bwrap = _enabled(mode, bwrap_executable)
    if bwrap is None:
        if _git_common_directory(workspace) is not None:
            raise RuntimeError(
                "Git-backed Agent workspaces require --agent-sandbox=bwrap on Linux; "
                "an unsandboxed process can access Supervisor Git metadata"
            )
        return list(command), dict(environment)

    workspace = workspace.resolve()
    repository_root = repository_root.resolve()
    if (workspace / ".git").is_symlink():
        raise RuntimeError("Agent workspace .git cannot be a symlink")
    assets = materialize_agent_assets(workspace, repository_root, agent_skills)
    view = agent_workspace or AgentWorkspace(workspace, provider_homes.parent).prepare()
    from .workspace_runtime import _install_link

    initialize_writable_tools(view, assets)
    _install_link(view / "skills", assets / "skills")
    for backend_root in (".claude", ".qoder", ".agents"):
        _install_link(view / backend_root / "skills", assets / "skills")
    raw_home = environment.get("HOME") or os.environ.get("HOME")
    if not raw_home:
        raise RuntimeError("Bubblewrap Agent sandbox requires a host HOME")
    host_home = Path(raw_home).expanduser().resolve()
    backend = str(environment.get("ATREX_AGENT_CLI") or "")
    backends = _projected_backends(backend, environment)
    key = _session_key(command, backend, environment)
    provider_home = provider_home or _prepare_provider_home(provider_homes, key, backends, host_home)
    for backend_root in (".claude", ".qoder", ".agents"):
        _install_link(provider_home / backend_root / "skills", assets / "skills")

    mapped = {
        key: _translate(str(value), workspace)
        for key, value in environment.items()
        if key not in {"PWD", "OLDPWD", "SHLVL", "_", "BASH_ENV"} and not key.startswith("GIT_")
    }
    mapped["HOME"] = str(VISIBLE_HOME)
    mapped["ATREX_WORKSPACE"] = str(VISIBLE_WORKSPACE)
    mapped["ATREX_AGENT_SANDBOX"] = "bwrap"
    if "claude" in backends:
        mapped["CLAUDE_CONFIG_DIR"] = str(VISIBLE_HOME / ".claude")
    if "codex" in backends:
        mapped["CODEX_HOME"] = str(VISIBLE_HOME / ".codex")
    if "pi" in backends:
        mapped["PI_CODING_AGENT_DIR"] = str(VISIBLE_HOME / ".pi/agent")
    mapped["XDG_CONFIG_HOME"] = str(VISIBLE_HOME / ".config")
    mapped["XDG_CACHE_HOME"] = str(VISIBLE_HOME / ".cache")
    mapped["XDG_DATA_HOME"] = str(VISIBLE_HOME / ".local/share")
    mapped["XDG_STATE_HOME"] = str(VISIBLE_HOME / ".local/state")
    if "codex" in backends and not mapped.get("SSL_CERT_FILE"):
        for candidate in _CODEX_CA_BUNDLE_CANDIDATES:
            if candidate.is_file():
                mapped["SSL_CERT_FILE"] = str(candidate)
                break

    translated_command = [_translate(value, workspace) for value in command]
    assignments = [f"{key}={value}" for key, value in sorted(mapped.items())]
    argv = [
        bwrap,
        "--die-with-parent",
        "--new-session",
        "--unshare-user",
        "--unshare-pid",
        "--unshare-ipc",
        "--unshare-uts",
        "--unshare-cgroup-try",
        "--hostname",
        "aka-agent",
        "--cap-drop",
        "ALL",
        "--ro-bind",
        "/",
        "/",
        "--dev",
        "/dev",
        "--proc",
        "/proc",
        "--tmpfs",
        "/run",
        "--dir",
        "/run/systemd",
        "--dir",
        "/run/systemd/resolve",
        "--tmpfs",
        "/tmp",
        "--tmpfs",
        "/home",
        "--tmpfs",
        "/root",
    ]
    created: set[str] = {
        "/",
        "/run",
        "/run/systemd",
        "/run/systemd/resolve",
        "/tmp",
        "/home",
        "/root",
    }
    # The read-only root is not sufficient isolation: without masks an Agent
    # could still inspect sibling Campaign worktrees and unrelated files.  As
    # in Atrex Runtime's BwrapProcessLauncher, hide the host Home, the complete
    # AKA checkout, and the parent containing this worktree before restoring
    # only the explicitly scoped paths below.
    for hidden in _minimal_hidden_paths(
        (
            host_home,
            repository_root,
            workspace.parent,
            *hidden_host_paths,
        )
    ):
        argv += ["--tmpfs", str(hidden)]
        created.add(hidden.as_posix())
    resolver = Path("/etc/resolv.conf").resolve()
    if resolver.is_file():
        for destination in (
            Path("/run/systemd/resolve/resolv.conf"),
            Path("/run/systemd/resolve/stub-resolv.conf"),
        ):
            _mount(
                argv,
                resolver,
                destination,
                writable=False,
                created=created,
            )
    _mount(argv, view, VISIBLE_WORKSPACE, writable=True, created=created)
    # CLI state is visible below HOME, but physically scoped to this session.
    # Never mount the whole provider home over the task files.
    _copy_state_file(provider_home / ".claude.json", view / ".claude.json")
    for name in (".claude", ".codex", ".qoder", ".pi"):
        source = provider_home / name
        if source.exists() and not source.is_symlink():
            _mount(argv, source, VISIBLE_HOME / name, writable=True, created=created)

    # Restore the exact selected view, never the Supervisor/source trees or the
    # parent private directory. Workspace/backend links resolve into this view.
    _mount(argv, assets, assets, writable=False, created=created)
    for name in ("tools", "skills"):
        _mount(argv, assets / name, repository_root / name, writable=False, created=created)
    if agent_reference_projects:
        source = repository_root / "reference-projects"
        if source.is_dir() and not source.is_symlink():
            references = project_reference_tree(source, provider_homes.parent / "reference-assets")
            _install_link(view / "reference-projects", references)
            _mount(argv, references, references, writable=False, created=created)

    for root in _installation_roots(command, backends, environment, host_home):
        _mount(argv, root, root, writable=False, created=created)

    credential_mounts, state_overlays = _credential_mounts(
        backends, host_home, provider_home
    )
    for source, destination, writable in credential_mounts:
        _mount(argv, source, destination, writable=writable, created=created)
    if "qodercli" in backends and (host_home / ".qoder").is_dir():
        for relative in (
            "tasks", "projects", "logs", "tmp", "cache", ".cache",
            ".codebase-status", "session-env", "shell-snapshots",
            "external-commands/locks",
        ):
            if not (host_home / ".qoder" / relative).is_dir():
                continue
            source = provider_home / ".qoder-writable" / relative
            _mount(
                argv,
                source,
                VISIBLE_HOME / ".qoder" / relative,
                writable=True,
                created=created,
            )
        if (host_home / ".qodersec/logs").is_dir():
            qodersec = provider_home / ".qodersec-logs"
            _mount(
                argv,
                qodersec,
                VISIBLE_HOME / ".qodersec/logs",
                writable=True,
                created=created,
            )
    for source, destination in state_overlays:
        _mount(argv, source, destination, writable=True, created=created)
    if "qodercli" in backends:
        _mount(argv, assets / "skills", VISIBLE_HOME / ".qoder/skills", writable=False, created=created)

    # Overlay evaluator-owned inputs read-only after mounting the workspace.
    for relative in (_PROTECTED_WORKSPACE_PATHS if read_only_paths is None else read_only_paths):
        source = view / relative
        if not source.exists() or source.is_symlink():
            continue
        _mount(
            argv,
            source,
            VISIBLE_WORKSPACE / relative,
            writable=False,
            created=created,
        )

    # Canonical reports are the only Agent-facing memory. Legacy audit files may
    # remain tracked in old commits, but must not reappear in a resumed sandbox.
    memory = view / "memory"
    if memory.is_symlink():
        raise RuntimeError("Agent memory directory cannot be a symlink")
    if memory.is_dir():
        destination = VISIBLE_WORKSPACE / "memory"
        argv += ["--tmpfs", str(destination)]
        created.add(destination.as_posix())
        for source in sorted(memory.iterdir()):
            if re.fullmatch(r"v[0-9]+\.json", source.name) and source.is_file() and not source.is_symlink():
                _mount(argv, source, destination / source.name, writable=False, created=created)
        argv += ["--remount-ro", str(destination)]

    argv += [
        "--chdir",
        str(VISIBLE_WORKSPACE),
        "--",
        "/usr/bin/env",
        "-i",
        *assignments,
        *translated_command,
    ]
    return argv, mapped


__all__ = ["VISIBLE_HOME", "VISIBLE_WORKSPACE", "wrap_agent_command"]
