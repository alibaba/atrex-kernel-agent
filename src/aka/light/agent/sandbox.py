"""Optional Bubblewrap boundary for managed Episode drafts and legacy roles.

Only system runtime paths, selected installations, this workspace and explicit
grants enter the namespace. Git is granted only to legacy, non-managed roles; GPU/Wiki
operations use the Supervisor HTTP service rather than Agent-side credentials.
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import stat
import subprocess
from pathlib import Path

from aka.light.agent.home import HOST_HOME_ENV, PREPARED_ENV, open_private_directory, projected_backends
from aka.light.agent.installations import installation_mounts
from aka.light.workspace.auxiliary import WORKSPACE_LAYOUTS, WORKSPACE_ROLE_ENV, AuxiliaryWorkspace
from aka.light.agent.recovery_processes import HANDOFF_ID_ENV
from aka.light.workspace.episode import EPISODE_WORKSPACE_ENV, PUBLIC_FILES
from aka.light.agent.sandbox_launch import SandboxLaunch

from aka.light.runtime.paths import SESSION_SHELL_GUARD, legacy_resource_root

REPOSITORY_ROOT = legacy_resource_root()
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
    # Legacy command shims import the extracted modules from this checkout.
    # Only code/assets are mounted; managed HTTP sessions do not use this list.
    "src/aka/__init__.py",
    "src/aka/bootstrap",
    "src/aka/contracts",
    "src/aka/core",
    "src/aka/legacy",
    "src/aka/observation",
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
    if environment.get("ATREX_AGENT_PROC_MODE", "private") not in {"private", "host"}:
        raise ValueError("ATREX_AGENT_PROC_MODE must be private or host")
    mode = environment.get("ATREX_AGENT_SANDBOX", "none")
    if mode not in {"none", "bwrap"}:
        raise ValueError("ATREX_AGENT_SANDBOX must be none or bwrap")
    claim_review = (
        environment.get("ATREX_AGENT_WORKSPACE_ROLE")
        == "research-claim-review"
    )
    if claim_review and mode != "bwrap":
        raise RuntimeError(
            "Research claim review requires Bubblewrap; native execution is not permitted"
        )
    if mode == "none":
        return None
    if platform.system() != "Linux":
        if claim_review:
            raise RuntimeError(
                "Research claim review requires Bubblewrap on a Linux coordinator; "
                "native execution is not permitted"
            )
        raise RuntimeError(
            "Agent Bubblewrap requires a Linux coordinator; use Lima/Linux, "
            "or set sandbox=none on the selected aka.light.agent.plugin module"
        )
    executable = environment.get("ATREX_BWRAP_EXECUTABLE", "bwrap")
    resolved = shutil.which(executable, path=environment.get("PATH"))
    if not resolved:
        if claim_review:
            raise RuntimeError(
                f"Research claim review requires Bubblewrap, but the executable "
                f"was not found: {executable}. Install bwrap or configure "
                "bwrap_executable; native execution is not permitted"
            )
        raise RuntimeError(
            f"Agent Bubblewrap executable not found: {executable}. "
            "Install bwrap or configure the Agent module's bwrap_executable; "
            "set its sandbox=none to explicitly disable isolation"
        )
    return resolved


def read_only_grants(environment: dict[str, str]) -> tuple[Path, ...]:
    raw = json.loads(environment.get("ATREX_AGENT_READ_ONLY_PATHS", "[]"))
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        raise ValueError("ATREX_AGENT_READ_ONLY_PATHS must be a JSON array of paths")
    return tuple(Path(value).expanduser().resolve(strict=True) for value in raw)


REFERENCE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")


def reference_sources(environment: dict[str, str]) -> tuple[tuple[str, Path], ...]:
    """Resolve and validate named operator knowledge roots."""
    raw = json.loads(environment.get("ATREX_AGENT_REFERENCE_PATHS", "{}"))
    if not isinstance(raw, dict) or any(
        not isinstance(name, str)
        or not REFERENCE_NAME.fullmatch(name)
        or not isinstance(value, str)
        for name, value in raw.items()
    ):
        raise ValueError("ATREX_AGENT_REFERENCE_PATHS must map safe names to paths")
    sources = []
    for name, value in sorted(raw.items()):
        source = Path(value).expanduser().resolve(strict=True)
        if not source.is_dir():
            raise ValueError(f"Agent reference must be a directory: {source}")
        sources.append((name, source))
    return tuple(sources)


def _reference_root(workspace: Path) -> Path:
    root = workspace / "references"
    root.mkdir(exist_ok=True)
    if root.is_symlink() or root.resolve(strict=True) != root:
        raise ValueError("Agent references root must be a real workspace directory")
    return root


def reference_grants(
    environment: dict[str, str], workspace: Path
) -> tuple[tuple[str, Path, Path], ...]:
    """Resolve named knowledge roots to read-only Bubblewrap mountpoints."""
    sources = reference_sources(environment)
    if not sources:
        return ()
    root = _reference_root(workspace)
    grants = []
    for name, source in sources:
        destination = root / name
        destination.mkdir(exist_ok=True)
        if destination.is_symlink() or destination.resolve(strict=True) != destination:
            raise ValueError(f"Agent reference mountpoint must be a real directory: {destination}")
        if any(destination.iterdir()):
            raise ValueError(f"Agent reference mountpoint must be empty: {destination}")
        grants.append((name, source, destination))
    return tuple(grants)


def project_native_references(environment: dict[str, str], workspace: Path) -> None:
    """Link knowledge roots into an unsandboxed Agent workspace.

    This is a discoverability projection, not a protection boundary. The Agent
    runs as the coordinator OS user and can modify the referenced source tree.
    """
    sources = reference_sources(environment)
    if not sources:
        return
    workspace = workspace.resolve(strict=True)
    root = _reference_root(workspace)
    for name, source in sources:
        if source.is_relative_to(workspace) or workspace.is_relative_to(source):
            raise ValueError(f"Agent reference overlaps Agent workspace: {source}")
        destination = root / name
        if destination.is_symlink():
            if destination.resolve(strict=True) != source:
                raise ValueError(
                    f"Agent reference link targets a different directory: {destination}"
                )
            continue
        if destination.exists():
            raise ValueError(
                f"Native Agent reference projection must be absent or a managed link: {destination}"
            )
        destination.symlink_to(source, target_is_directory=True)


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
        if environment.get(WORKSPACE_ROLE_ENV, "optimizer") == "optimizer":
            project_native_references(environment, workspace)
        native_environment = {
            key: value for key, value in environment.items()
            if key not in {
                "ATREX_EPISODE_PRIVATE_PATHS",
                "ATREX_AKA_RUNTIME_OWNER",
                "ATREX_AGENT_REFERENCE_PATHS",
                "GIT_CEILING_DIRECTORIES",
            }
        }
        return SandboxLaunch(list(command), native_environment), None
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
    from aka.light.agent.claude_tools import CONTROL_ROOT_ENV
    raw_control_root = environment.get(CONTROL_ROOT_ENV, "")
    control_root = None
    if raw_control_root:
        control_root = Path(raw_control_root)
        if not control_root.is_absolute():
            raise ValueError("Agent control root must be absolute")
        control_root = control_root.resolve(strict=True)
        if not control_root.is_dir() or control_root.is_symlink():
            raise ValueError("Agent control root must be a real directory")
        private_paths = (*private_paths, control_root)
    host_home = Path(environment[HOST_HOME_ENV]).resolve()
    role = environment.get(WORKSPACE_ROLE_ENV, "optimizer")
    if role != "optimizer" and role not in WORKSPACE_LAYOUTS:
        raise ValueError(f"Unknown Agent workspace role: {role}")
    # Operator-provided knowledge is available only to the optimization Agent.
    # Independent reviewers receive their own bounded evidence view and must
    # not be influenced by the optimizer's optional reference library.
    references = reference_grants(environment, workspace) if role == "optimizer" else ()
    if role == "optimizer" and auxiliary_input_files:
        raise ValueError("Explicit auxiliary inputs require an auxiliary workspace role")
    view = (
        AuxiliaryWorkspace(workspace, home, role, input_files=auxiliary_input_files)
        if role != "optimizer" else None
    )
    try:
        host_proc = environment.get("ATREX_AGENT_PROC_MODE", "private") == "host"
        argv = [executable]
        # Only a durable handoff owner controls the sandbox lifetime. Direct
        # launches preserve native behavior on Supervisor/spawning-thread death.
        # spawn_owned_session validates the handoff before starting its wrapper.
        if environment.get(HANDOFF_ID_ENV):
            argv.append("--die-with-parent")
        argv.extend(("--new-session", "--unshare-user"))
        if not host_proc:
            argv.append("--unshare-pid")
        argv.extend((
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
        argv.extend(("--dev", "/dev", "--tmpfs", "/tmp", "--tmpfs", "/run"))
        # Aone Sandbox already provides a container-scoped procfs but rejects
        # mounting another procfs in a nested Bubblewrap namespace. Host mode
        # deliberately shares that complete procfs and PID namespace. Private
        # mode retains an isolated PID namespace and a newly mounted procfs.
        argv.extend(("--bind", "/proc", "/proc") if host_proc else ("--proc", "/proc"))
        _mount(argv, view.root if view else workspace, workspace, writable=True)
        if managed:
            for name, directory in (
                ("research-library", True),
                ("records", True),
                ("PROGRESS.md", False),
            ):
                projection = workspace / name
                if not projection.exists():
                    if name == "research-library":
                        # Research-disabled Campaigns do not project this tree.
                        continue
                    raise ValueError(
                        f"Managed Runtime projection is unavailable before sandbox launch: {name}"
                    )
                mode = projection.lstat().st_mode
                if (
                    projection.is_symlink()
                    or (not stat.S_ISDIR(mode) if directory else not stat.S_ISREG(mode))
                    or projection.resolve(strict=True) != projection
                ):
                    kind = "directory" if directory else "file"
                    raise ValueError(
                        f"Managed {name} mountpoint must be a real {kind}"
                    )
                # Overlay Runtime projections read-only. Their backing paths
                # remain writable to the trusted Supervisor outside this
                # namespace. Directories reflect atomic child replacement;
                # PROGRESS.md keeps a stable inode for live refreshes.
                _mount(argv, projection, projection)
        _mount(argv, home, home, writable=True)
        if control_root is not None:
            # Hooks receive only their narrow rendezvous directory, never the
            # containing Supervisor campaign tree.
            _mount(argv, control_root, control_root, writable=True)
        # HTTP sessions carry their reviewed tool/Skill copies in the workspace.
        # Do not re-expose the repository's retired workflows or evaluator code.
        assets = PUBLIC_ASSETS if view is None and not environment.get("ATREX_AKA_RUNTIME_URL") else ()
        for name in assets:
            source = REPOSITORY_ROOT / name
            if source.exists():
                _mount(argv, source.resolve(), source)
        # BASH_ENV and its mount must name the same packaged Light asset.
        # The checkout's old tools/ location is not an installed resource.
        if not SESSION_SHELL_GUARD.is_file():
            raise ValueError("selected Agent shell guard resource is unavailable")
        _mount(argv, SESSION_SHELL_GUARD.resolve(), SESSION_SHELL_GUARD)
        if environment.get("ATREX_AKA_RUNTIME_URL"):
            from aka.light.agent import claude_tools
            hook = Path(claude_tools.__file__).resolve()
            _mount(argv, hook, hook)
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
                # worktree. Public inputs stay read-only.
                for name in (*PUBLIC_FILES, "input"):
                    source = workspace / name
                    if source.exists():
                        if source.is_symlink() or source.resolve(strict=True) != source:
                            raise ValueError("Managed public inputs must not traverse symlinks")
                        _mount(argv, source, source)
            from aka.light.workspace.managed_assets import PROTECTED_PATHS_ENV
            protected = ("tools", "skills")
            if managed and PROTECTED_PATHS_ENV in environment:
                protected = json.loads(environment[PROTECTED_PATHS_ENV])
                from aka.light.workspace.sources import POLICY, SELECTION, source_name
                if not isinstance(protected, list) or any(
                    not isinstance(name, str) or not name or name == "." or "\\" in name
                    or Path(name).is_absolute() or ".." in Path(name).parts
                    or not (name in {"tools/sandbox.py", "tools/plugin.py", "WORKSPACE.md"}
                            or len(Path(name).parts) == 2 and Path(name).parts[0] == "skills"
                            or name in {POLICY, SELECTION} or source_name(name))
                    for name in protected
                ):
                    raise ValueError("Invalid managed asset protection list")
                # Writable mountpoints prevent renaming an entire parent tree
                # around its read-only installed children.
                for tree in ("tools", "skills"):
                    source = workspace / tree
                    if source.is_symlink() or source.resolve(strict=True) != source:
                        raise ValueError("Reusable workspace directories must not be symlinks")
                    _mount(argv, source, source, writable=True)
            for name in protected:
                source = workspace / name
                if managed and PROTECTED_PATHS_ENV in environment and not source.exists():
                    raise ValueError("Managed asset disappeared before sandbox launch")
                if source.exists():
                    # The upstream Workspace provider installs these exact
                    # repository links. Their destinations are already mounted
                    # read-only by PUBLIC_ASSETS above. Managed drafts still
                    # require real directories and reject every symlink.
                    if (not managed and not environment.get("ATREX_AKA_RUNTIME_URL")
                            and source.is_symlink()
                            and source.resolve(strict=True) == REPOSITORY_ROOT / name):
                        continue
                    if source.is_symlink() or source.resolve() != source:
                        raise ValueError("Managed assets must not traverse symlinks")
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
        forbidden = (
            workspace,
            home.parent,
            REPOSITORY_ROOT / ".git",
            *private_paths,
        )
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
        for _, source, destination in references:
            if any(
                root.is_relative_to(source) or source.is_relative_to(root)
                for root in (workspace, home.parent, *private_paths)
            ):
                raise ValueError(f"Agent reference overlaps Agent workspace/state: {source}")
            if source in {Path("/"), host_home, REPOSITORY_ROOT}:
                raise ValueError(f"Agent reference is too broad: {source}")
            _mount(argv, source, destination)
        # Preserve the pre-PR CLI command and cwd; no prompt rewriting is needed.
        # Secrets are set by bwrap from an anonymous FD, never KEY=value argv.
        values = {
            key: value
            for key, value in environment.items()
            if key not in {HOST_HOME_ENV, PREPARED_ENV, "PWD", "OLDPWD", "ATREX_EPISODE_PRIVATE_PATHS", "ATREX_AGENT_REFERENCE_PATHS"}
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
