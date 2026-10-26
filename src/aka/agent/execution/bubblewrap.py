"""Bubblewrap implementation of an explicit Agent filesystem view."""

from __future__ import annotations
import os
import platform
import shutil
from pathlib import Path
from aka.contracts.agent import AgentAccess, AgentMount
from .home import PREPARED_ENV
from .installations import installation_mounts
from .launch import SandboxLaunch

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


def executable(value="bwrap"):
    if platform.system() != "Linux":
        raise RuntimeError(
            "Agent Bubblewrap requires Linux; select sandbox=none for native execution"
        )
    resolved = shutil.which(value)
    if not resolved:
        raise RuntimeError("Agent Bubblewrap executable is unavailable")
    return resolved


def system_visible(path):
    return any(Path(path).is_relative_to(Path(root)) for root in SYSTEM_PATHS)


def local_path(path, *, must_exist=True):
    path = Path(path).expanduser()
    if not path.is_absolute():
        raise ValueError("Agent grants require absolute paths")
    original = Path(os.path.abspath(path))
    resolved = original.resolve(strict=must_exist)
    if resolved != original:
        raise ValueError("Agent grants must not traverse symlinks")
    if original in {
        Path("/"),
        Path("/home"),
        Path("/root"),
        Path("/Users"),
        Path("/tmp"),
        Path("/run"),
        Path("/proc"),
        Path("/dev"),
        Path("/etc"),
        Path("/var"),
        Path("/opt"),
        Path("/srv"),
        Path("/mnt"),
        Path("/media"),
    }:
        raise ValueError("Agent grant is too broad")
    return original


def overlap(left, right):
    return left.is_relative_to(right) or right.is_relative_to(left)


def validate_access(workspace, home, host_home, access, *, home_exists=True):
    workspace = local_path(workspace)
    home = local_path(home, must_exist=home_exists)
    if (
        overlap(workspace, home)
        or workspace == host_home
        or host_home.is_relative_to(workspace)
    ):
        raise ValueError("Agent workspace overlaps private Home")
    if any(
        overlap(workspace, Path(root)) or overlap(home, Path(root))
        for root in SYSTEM_PATHS
    ):
        raise ValueError("Agent private storage overlaps system runtime mounts")
    if system_visible(host_home):
        raise ValueError("Operator Home overlaps a system runtime mount")
    hidden = tuple(local_path(path, must_exist=False) for path in access.hidden_paths)
    read_only = tuple(local_path(path) for path in access.read_only_paths)
    for path in (*hidden, *read_only):
        files = path.rglob("*") if path.is_dir() else (path,)
        if any(member.is_file() and member.stat().st_nlink != 1 for member in files):
            raise ValueError("Restricted Agent paths must not have hardlink aliases")
    for path in hidden:
        if not path.is_relative_to(workspace) or path == workspace:
            raise ValueError("Hidden Agent paths must be inside the workspace")
    if any(
        overlap(left, right)
        for index, left in enumerate(hidden)
        for right in hidden[index + 1 :]
    ):
        raise ValueError("Hidden Agent paths overlap")
    if any(
        overlap(left, right)
        for index, left in enumerate(read_only)
        for right in read_only[index + 1 :]
    ):
        raise ValueError("Read-only Agent paths overlap")
    for path in read_only:
        if path.is_file() and path.stat().st_nlink != 1:
            raise ValueError(
                "Read-only Agent files must not have writable hardlink aliases"
            )
        if (
            not path.is_relative_to(workspace)
            or path == workspace
            or any(overlap(path, h) for h in hidden)
        ):
            raise ValueError("Read-only Agent paths must be visible workspace children")
    mounts = []
    targets = []
    for mount in access.mounts:
        if type(mount.writable) is not bool:
            raise TypeError("Agent mount writable flag must be boolean")
        source = local_path(mount.source)
        if any(source.is_relative_to(Path(root)) for root in ("/proc", "/dev", "/run")):
            raise ValueError(
                "Agent grants cannot expose host process or device namespaces"
            )
        if not source.is_dir() and not source.is_file():
            raise ValueError("Agent grants require regular files or directories")
        target = local_path(mount.target, must_exist=False)
        if system_visible(source) or system_visible(target):
            raise ValueError("Agent mount cannot alias system runtime paths")
        if (
            overlap(source, home.parent)
            or source == host_home
            or host_home.is_relative_to(source)
            or any(overlap(source, h) for h in hidden)
        ):
            raise ValueError("Agent mount exposes private storage")
        if overlap(source, workspace) and source != target:
            raise ValueError("Workspace aliases could bypass access restrictions")
        if any(overlap(target, previous) for previous in targets):
            raise ValueError("Agent mount targets overlap")
        if (
            overlap(target, workspace)
            or overlap(target, home)
            or any(overlap(target, Path(p)) for p in SYSTEM_PATHS)
        ):
            raise ValueError("Agent mount shadows an existing sandbox view")
        if (
            target.is_relative_to(Path("/proc"))
            or target.is_relative_to(Path("/dev"))
            or target.is_relative_to(Path("/run"))
        ):
            raise ValueError("Agent mount shadows a sandbox runtime path")
        mounts.append(AgentMount(source, target, mount.writable))
        targets.append(target)
    return workspace, hidden, read_only, tuple(mounts)


def mount_path(argv, source, target, *, writable=False):
    for parent in reversed(target.parents):
        if parent != Path("/"):
            argv.extend(("--dir", str(parent)))
    argv.extend(("--bind" if writable else "--ro-bind", str(source), str(target)))


class BubblewrapExecution:
    def __init__(self, bwrap="bwrap", *, proc_mode="private"):
        if proc_mode not in {"private", "projected"}:
            raise ValueError("Agent proc_mode must be private or projected")
        self.proc_mode = proc_mode
        self.executable = executable(bwrap)
        if proc_mode == "projected":
            from .proc_view import validate_runtime
            validate_runtime()

    def launch(self, command, workspace, environment, backend, host_home, access):
        if not command:
            raise ValueError("Agent command cannot be empty")
        access = access or AgentAccess()
        home = Path(environment[PREPARED_ENV])
        workspace, hidden, read_only, mounts = validate_access(
            workspace, home, host_home, access
        )
        argv = [
            self.executable,
            "--new-session",
            "--die-with-parent",
            "--unshare-user",
            "--unshare-pid",
            "--unshare-ipc",
            "--unshare-uts",
            "--unshare-cgroup-try",
            "--hostname",
            "aka-agent",
            "--cap-drop",
            "ALL",
        ]
        for name in SYSTEM_PATHS:
            path = Path(name)
            if path.exists():
                mount_path(argv, path.resolve(), path)
        argv.extend(("--dev", "/dev", "--tmpfs", "/tmp", "--tmpfs", "/run"))
        if self.proc_mode == "projected":
            from .proc_view import projected_mounts, SHIM_SOURCE, SHIM_TARGET
            argv.extend(projected_mounts())
            argv.extend(("--dir", "/run/aka-proc", "--ro-bind", SHIM_SOURCE, SHIM_TARGET))
        else:
            argv.extend(("--proc", "/proc"))
        mount_path(argv, home, home, writable=True)
        mount_path(argv, workspace, workspace, writable=True)
        for path in read_only:
            mount_path(argv, path, path)
        for path in sorted(hidden, key=lambda p: len(p.parts)):
            if path.is_dir() or not path.exists():
                argv.extend(("--tmpfs", str(path)))
            else:
                argv.extend(("--ro-bind", "/dev/null", str(path)))
        for grant in mounts:
            mount_path(argv, grant.source, grant.target, writable=grant.writable)
        forbidden = (workspace, home.parent, *hidden)
        provider_roots = tuple(host_home / p for p in backend.files.state_roots)
        code_roots = tuple(host_home / p for p in backend.files.code_roots)
        for source, target in installation_mounts(
            command,
            (),
            environment,
            host_home,
            hidden_paths=(host_home, Path("/opt")),
            forbidden_paths=forbidden,
            provider_roots=provider_roots,
            code_roots=code_roots,
            visible_paths=tuple(Path(p) for p in SYSTEM_PATHS),
        ):
            if any(overlap(target, grant.target) for grant in mounts):
                raise ValueError("Agent installation overlaps an explicit mount")
            mount_path(argv, source, target)
        values = {
            key: str(value)
            for key, value in environment.items()
            if key not in {PREPARED_ENV, "PWD", "OLDPWD"}
        }
        values.update(PWD=str(workspace), TMPDIR="/tmp")
        argv.extend(("--chdir", str(workspace)))
        # Preserve Light's current networking behavior. This backend does not promise network isolation.
        if self.proc_mode == "projected":
            from .proc_view import prepare_launch
            provider = shutil.which(command[0], path=values.get("PATH"))
            return prepare_launch(argv, command, values, workspace, provider_executable=provider)
        return SandboxLaunch.with_bwrap_environment(argv, command, values)
