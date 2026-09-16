"""Opt-in Bubblewrap launch boundary, without migrating the legacy Agent workflow.

Only system runtime paths, selected installations, this workspace and explicit
grants enter the namespace. Campaign Git/Gateway access remains an intentional
compatibility grant until their Supervisor ownership is introduced separately.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from pathlib import Path

from .agent_home import HOST_HOME_ENV, PREPARED_ENV, _directory, projected_backends
from .agent_installations import installation_mounts
from .agent_workspace import WORKSPACE_LAYOUTS, WORKSPACE_ROLE_ENV, AuxiliaryWorkspace
from .recovery_processes import HANDOFF_ID_ENV
from .sandbox_launch import SandboxLaunch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_ASSETS = (
    "tools",
    "skills",
    "agents",
    "reference",
    "reference-projects",
    "gpu-wiki",
    "3rdparty",
    "orchestrator",
    "long_horizon",
    "plugin_runtime",
    "plugins",
)
SYSTEM_PATHS = (
    "/usr",
    "/bin",
    "/sbin",
    "/lib",
    "/lib64",
    "/etc/ssl",
    "/etc/pki",
    "/etc/fonts",
    "/etc/alternatives",
    "/etc/ld.so.cache",
    "/etc/ld.so.conf",
    "/etc/ld.so.conf.d",
    "/etc/nsswitch.conf",
    "/etc/passwd",
    "/etc/group",
    "/etc/hosts",
    "/etc/resolv.conf",
    "/etc/localtime",
    "/etc/os-release",
    "/etc/services",
    "/etc/protocols",
    "/etc/gai.conf",
)


def sandbox_executable(environment: dict[str, str]) -> str | None:
    mode = environment.get("ATREX_AGENT_SANDBOX", "none")
    if mode == "none":
        return None
    if mode != "bwrap":
        raise ValueError("ATREX_AGENT_SANDBOX must be none or bwrap")
    if platform.system() != "Linux":
        raise RuntimeError(
            "Agent Bubblewrap requires a Linux coordinator; use Lima/Linux, "
            "or --agent-sandbox none for the existing native path"
        )
    executable = environment.get("ATREX_BWRAP_EXECUTABLE", "bwrap")
    resolved = shutil.which(executable, path=environment.get("PATH"))
    if not resolved:
        raise RuntimeError(f"Agent Bubblewrap executable not found: {executable}")
    return resolved


def read_only_grants(environment: dict[str, str]) -> tuple[Path, ...]:
    raw = json.loads(environment.get("ATREX_AGENT_READ_ONLY_PATHS", "[]"))
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        raise ValueError("ATREX_AGENT_READ_ONLY_PATHS must be a JSON array of paths")
    return tuple(Path(value).expanduser().resolve(strict=True) for value in raw)


def _mount(argv: list[str], source: Path, destination: Path, *, writable=False) -> None:
    for parent in reversed(destination.parents):
        if parent != Path("/"):
            argv.extend(("--dir", str(parent)))
    argv.extend(("--bind" if writable else "--ro-bind", str(source), str(destination)))


def _git_directory(workspace: Path) -> Path | None:
    result = subprocess.run(
        ["git", "rev-parse", "--git-common-dir"],
        cwd=workspace,
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )
    if result.returncode:
        return None
    text = result.stdout.strip()
    if not text or len(text) > 4096 or "\n" in text:
        raise RuntimeError("Invalid campaign Git directory")
    common = (workspace / text).resolve(strict=True)
    if not common.is_dir():
        raise RuntimeError("Campaign Git metadata must be a directory")
    return common


def _grant_environment_paths(
    argv: list[str],
    environment: dict[str, str],
    workspace: Path,
    *,
    host_home: Path,
    session_home_root: Path,
) -> None:
    # These grants are transitional: legacy sandbox.py still packages private
    # evaluator inputs, writes Wiki feedback, and records phase markers itself.
    for variable, writable, directory in (
        ("ATREX_PRIVATE_REFERENCE_DIR", False, True),
        ("ATREX_WIKI_PROFILE_ROOT", True, True),
        ("ATREX_TELEMETRY_TRACE", True, False),
        ("ATREX_ENVIRONMENT_STATE_FILE", True, False),
        ("ATREX_JOURNAL_LIVE_FILE", True, False),
        ("ATREX_CODEX_REVIEW_SESSION_FILE", True, False),
        ("ATREX_QODER_REVIEW_SESSION_FILE", True, False),
    ):
        value = environment.get(variable)
        if not value:
            continue
        source = Path(os.path.abspath(Path(value).expanduser()))
        if source.is_relative_to(workspace):
            # Already visible. Never resolve an Agent-created workspace symlink
            # into a fresh grant outside the workspace.
            continue
        if writable:
            # Helpers use atomic rename/lock files: allow the explicitly scoped
            # containing directory, rather than a non-replaceable file mount.
            source = source if directory else source.parent
        if source in {Path("/"), Path("/home"), Path("/root"), Path("/tmp")}:
            raise ValueError(f"Refusing broad legacy path grant: {variable}")
        if source.resolve() != source:
            raise ValueError(f"Legacy path grant must not traverse symlinks: {variable}")
        # Validate the actual directory grant, not just the requested file.
        # HOME already points into the sandbox; host_home is the operator Home.
        if host_home.is_relative_to(source):
            raise ValueError(f"Legacy path grant would expose host Home: {variable}")
        if source.is_relative_to(session_home_root) or session_home_root.is_relative_to(source):
            raise ValueError(f"Legacy path grant overlaps Agent Session Homes: {variable}")
        if writable:
            with _directory(source):
                pass
        if source.exists():
            _mount(argv, source, source, writable=writable)


def wrap_agent_command(
    command: list[str],
    workspace: Path,
    environment: dict[str, str],
    *, auxiliary_input_files: dict[str, Path] | None = None,
) -> tuple[SandboxLaunch, AuxiliaryWorkspace | None]:
    executable = sandbox_executable(environment)
    if executable is None:
        if auxiliary_input_files:
            raise ValueError("Explicit auxiliary inputs require a Bubblewrap auxiliary workspace")
        return SandboxLaunch(list(command), dict(environment)), None
    workspace = workspace.resolve(strict=True)
    if REPOSITORY_ROOT.is_relative_to(workspace):
        raise ValueError(
            "Run optimization in a Campaign workspace, not the AKA source checkout or its parent"
        )
    home = Path(environment[PREPARED_ENV]).resolve(strict=True)
    host_home = Path(environment[HOST_HOME_ENV]).resolve()
    role = environment.get(WORKSPACE_ROLE_ENV, "optimizer")
    if role != "optimizer" and role not in WORKSPACE_LAYOUTS:
        raise ValueError(f"Unknown Agent workspace role: {role}")
    if role == "optimizer" and auxiliary_input_files:
        raise ValueError("Explicit auxiliary inputs require an auxiliary workspace role")
    view = (
        AuxiliaryWorkspace(workspace, home, role, input_files=auxiliary_input_files)
        if role != "optimizer" else None
    )
    try:
        argv = [executable]
        # Only a durable handoff owner controls the sandbox lifetime. Direct
        # launches preserve native behavior on Supervisor/spawning-thread death.
        # spawn_owned_session validates the handoff before starting its wrapper.
        if environment.get(HANDOFF_ID_ENV):
            argv.append("--die-with-parent")
        argv.extend((
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
        ))
        # Empty-root allowlist, not a read-only bind of the host root. In
        # particular, /home, /root, /opt and Supervisor storage are absent.
        for name in SYSTEM_PATHS:
            path = Path(name)
            if path.exists():
                _mount(argv, path.resolve(), path)
        argv.extend(("--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp", "--tmpfs", "/run"))
        _mount(argv, view.root if view else workspace, workspace, writable=True)
        _mount(argv, home, home, writable=True)
        assets = PUBLIC_ASSETS if view is None else ("tools/session_shell_guard.sh",)
        if role == "plan-review-probe":
            assets += ("skills/gen-plan/scripts",)
        for name in assets:
            source = REPOSITORY_ROOT / name
            if source.exists():
                _mount(argv, source.resolve(), source)
        if view:
            for name in view.inputs:
                source = view.root / name
                if source.exists():
                    _mount(argv, source, workspace / name)
        else:
            common = _git_directory(workspace)
            if common:
                if common == _git_directory(REPOSITORY_ROOT):
                    raise ValueError(
                        "Run optimization in a Campaign workspace, not the AKA source checkout"
                    )
                if not common.is_relative_to(workspace):
                    _mount(argv, common, common, writable=True)
            # Preserve the existing symlink-based tools/skills layout, but never
            # bind the entire AKA checkout, its .git, or its workspaces.
            bench = workspace / "atrex-bench"
            if bench.is_dir():
                if bench.is_symlink():
                    raise ValueError("Legacy Atrex-Bench must be the code-only workspace copy")
                _mount(argv, bench, bench)
            _grant_environment_paths(
                argv, environment, workspace,
                host_home=host_home, session_home_root=home.parent,
            )
        forbidden = (workspace, home.parent, REPOSITORY_ROOT / ".git")
        installations = projected_backends(environment) + (("agate",) if view is None else ())
        for source, destination in installation_mounts(
            command,
            installations,
            environment,
            host_home,
            hidden_paths=tuple(
                path
                for path in Path("/").iterdir()
                if path.name not in {"usr", "bin", "sbin", "lib", "lib64", "etc"} and path.is_dir()
            ),
            forbidden_paths=forbidden,
        ):
            if source == destination:
                _mount(argv, source, destination)
            else:
                for parent in reversed(destination.parents):
                    if parent != Path("/"):
                        argv.extend(("--dir", str(parent)))
                argv.extend(("--symlink", str(source), str(destination)))
        for path in read_only_grants(environment):
            if any(
                root.is_relative_to(path) or path.is_relative_to(root)
                for root in (workspace, home.parent)
            ):
                raise ValueError(f"Read-only grant overlaps Agent workspace/state: {path}")
            if path in {Path("/"), host_home, REPOSITORY_ROOT}:
                raise ValueError(f"Read-only grant is too broad: {path}")
            _mount(argv, path, path)
        # Preserve the pre-PR CLI command and cwd; no prompt rewriting is needed.
        # Secrets are set by bwrap from an anonymous FD, never KEY=value argv.
        values = {
            key: value
            for key, value in environment.items()
            if key not in {HOST_HOME_ENV, PREPARED_ENV, "PWD", "OLDPWD"}
            and (
                not key.startswith("GIT_")
                or key
                in {
                    "GIT_AUTHOR_NAME",
                    "GIT_AUTHOR_EMAIL",
                    "GIT_COMMITTER_NAME",
                    "GIT_COMMITTER_EMAIL",
                }
            )
        }
        values.update(PWD=str(workspace), TMPDIR="/tmp")
        # An auxiliary reviewer has no Gateway/Wiki or recovery-control grants.
        if view:
            values = {
                key: value
                for key, value in values.items()
                if not key.startswith(
                    (
                        "AGATE_",
                        "ATREX_PRIVATE_",
                        "ATREX_WIKI_",
                        "GPU_WIKI_",
                        "ATREX_TELEMETRY_",
                        "ATREX_RECOVERY_",
                        "ATREX_ENVIRONMENT_",
                        "ATREX_JOURNAL_",
                    )
                )
            }
        argv.extend(("--chdir", str(workspace)))
        return SandboxLaunch.with_bwrap_environment(argv, command, values), view
    except BaseException:
        if view:
            view.close()
        raise
