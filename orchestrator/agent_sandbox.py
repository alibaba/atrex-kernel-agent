"""Default Bubblewrap boundary for managed Episode drafts and legacy roles.

Only system runtime paths, selected installations, this workspace and explicit
grants enter the namespace. Git is granted only to legacy, non-managed roles; GPU/Wiki
operations use the Supervisor HTTP service rather than Agent-side credentials.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from pathlib import Path

from .agent_home import HOST_HOME_ENV, PREPARED_ENV, open_private_directory, projected_backends
from .agent_installations import installation_mounts
from .agent_workspace import WORKSPACE_LAYOUTS, WORKSPACE_ROLE_ENV, AuxiliaryWorkspace
from .recovery_processes import HANDOFF_ID_ENV
from .episode_workspace import EPISODE_WORKSPACE_ENV, PUBLIC_FILES
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
    mode = environment.get("ATREX_AGENT_SANDBOX", "bwrap")
    if mode == "none":
        return None
    if mode != "bwrap":
        raise ValueError("ATREX_AGENT_SANDBOX must be none or bwrap")
    if platform.system() != "Linux":
        raise RuntimeError(
            "Agent Bubblewrap requires a Linux coordinator; use Lima/Linux, "
            "or explicitly disable isolation with --agent-sandbox none "
            "(ATREX_AGENT_SANDBOX=none)"
        )
    executable = environment.get("ATREX_BWRAP_EXECUTABLE", "bwrap")
    resolved = shutil.which(executable, path=environment.get("PATH"))
    if not resolved:
        raise RuntimeError(
            f"Agent Bubblewrap executable not found: {executable}. "
            "Install bwrap or set --bwrap-executable; to explicitly disable isolation, "
            "use --agent-sandbox none (ATREX_AGENT_SANDBOX=none)"
        )
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


def git_directory(workspace: Path) -> Path | None:
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
    private_paths: tuple[Path, ...] = (),
) -> None:
    # Phase markers/reviewer helpers retain explicitly scoped legacy grants.
    for variable, writable, directory in (
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
        if any(source.is_relative_to(path) or path.is_relative_to(source) for path in private_paths):
            raise ValueError(f"Path grant overlaps Supervisor private storage: {variable}")
        # Validate the actual directory grant, not just the requested file.
        # HOME already points into the sandbox; host_home is the operator Home.
        if host_home.is_relative_to(source):
            raise ValueError(f"Legacy path grant would expose host Home: {variable}")
        if source.is_relative_to(session_home_root) or session_home_root.is_relative_to(source):
            raise ValueError(f"Legacy path grant overlaps Agent Session Homes: {variable}")
        if writable:
            with open_private_directory(source):
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
    managed = environment.get(EPISODE_WORKSPACE_ENV) == str(workspace)
    private_paths = tuple(Path(value) for value in json.loads(
        environment.get("ATREX_EPISODE_PRIVATE_PATHS", "[]")
    )) if managed else ()
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
        # HTTP sessions carry their reviewed tool/Skill copies in the workspace.
        # Do not re-expose the repository's retired workflows or evaluator code.
        assets = PUBLIC_ASSETS if view is None and not environment.get("ATREX_AKA_RUNTIME_URL") else ("tools/session_shell_guard.sh",)
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
            common = None if managed else git_directory(workspace)
            if common:
                if common == git_directory(REPOSITORY_ROOT):
                    raise ValueError(
                        "Run optimization in a Campaign workspace, not the AKA source checkout"
                    )
                if not common.is_relative_to(workspace):
                    _mount(argv, common, common, writable=True)
            if managed:
                # This is already a separate persistent draft, not the Git
                # worktree. Public inputs and canonical memory stay read-only.
                for name in (*PUBLIC_FILES, "memory"):
                    source = workspace / name
                    if source.exists():
                        _mount(argv, source, source)
            for name in ("tools", "skills"):
                source = workspace / name
                if source.exists():
                    _mount(argv, source, source)
            # Keep Supervisor's legacy evaluator copy available for independent
            # acceptance, but mask it from Agent sessions using the HTTP service.
            bench = workspace / "atrex-bench"
            if bench.is_dir():
                if bench.is_symlink():
                    raise ValueError("Legacy Atrex-Bench must be the code-only workspace copy")
                if environment.get("ATREX_AKA_RUNTIME_URL"):
                    argv.extend(("--tmpfs", str(bench)))
                else:
                    _mount(argv, bench, bench)
            if environment.get("ATREX_AKA_RUNTIME_URL"):
                argv.extend(("--tmpfs", str(workspace / ".gpu_wiki_profile")))
                # A resumed PR2 Home may still contain old Gateway credentials.
                # Mask those copies without changing the operator's files.
                for relative in (".atrex", ".agate", ".config/agate"):
                    argv.extend(("--tmpfs", str(home / relative)))
            _grant_environment_paths(
                argv, environment, workspace,
                host_home=host_home, session_home_root=home.parent,
                private_paths=private_paths,
            )
        forbidden = (workspace, home.parent, REPOSITORY_ROOT / ".git", *private_paths)
        installations = projected_backends(environment)
        if view is None and not environment.get("ATREX_AKA_RUNTIME_URL"):
            installations += ("agate",)
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
                for root in (workspace, home.parent, *private_paths)
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
            if key not in {HOST_HOME_ENV, PREPARED_ENV, "PWD", "OLDPWD", "ATREX_EPISODE_PRIVATE_PATHS"}
            and (
                not key.startswith("GIT_")
                or key
                in {
                    "GIT_AUTHOR_NAME",
                    "GIT_AUTHOR_EMAIL",
                    "GIT_COMMITTER_NAME",
                    "GIT_COMMITTER_EMAIL",
                    "GIT_CEILING_DIRECTORIES",
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
