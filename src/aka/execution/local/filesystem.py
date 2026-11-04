"""Pin declared Linux mount sources before entering a restricted namespace."""
from contextlib import contextmanager
from pathlib import Path
import fcntl
import json
import os
import platform
import shutil
import stat
import subprocess
import sys

from aka.contracts.execution import UnsupportedExecution
from aka.contracts.filesystem_access import require_path_access

RUNTIME_PATHS = ("/usr", "/bin", "/sbin", "/lib", "/lib64", "/etc/ld.so.cache")
PRIVATE_NAMESPACES = (Path("/proc"), Path("/dev"), Path("/run"))
ROOTS = {Path(name) for name in ("/", "/root", "/home", "/Users", "/tmp", "/var", "/etc", "/opt", "/srv", "/mnt", "/media")}
CLOSE_AND_EXEC = """import os,sys,json
with open('/run/aka-command-environment') as stream:
 environment = json.load(stream)
os.unlink('/run/aka-command-environment')
for value in sys.argv[1].split(','):
 try: os.close(int(value))
 except OSError: pass
os.execvpe(sys.argv[2], sys.argv[2:], environment)
"""


def absolute(path):
    path = Path(path)
    if not path.is_absolute() or ".." in path.parts:
        raise UnsupportedExecution("filesystem grants require absolute paths without parent traversal")
    return path


def overlaps(left, right):
    return left.is_relative_to(right) or right.is_relative_to(left)


class Sources:
    """Own source descriptors; every untrusted component uses O_NOFOLLOW."""
    def __init__(self):
        self.fds = []

    def close(self):
        for fd in reversed(self.fds):
            os.close(fd)
        self.fds.clear()

    def data(self, content=b""):
        fd = os.memfd_create("aka-command-data", os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
        self.fds.append(fd)
        pending = memoryview(content)
        while pending:
            pending = pending[os.write(fd, pending):]
        os.lseek(fd, 0, os.SEEK_SET)
        fcntl.fcntl(fd, fcntl.F_ADD_SEALS,
            fcntl.F_SEAL_SEAL | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_GROW | fcntl.F_SEAL_WRITE)
        return fd

    def open(self, path, *, parent=None):
        path = Path(path)
        flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK
        if parent is None:
            path = absolute(path)
            current = os.open("/", flags | os.O_DIRECTORY)
            parts = path.parts[1:]
        else:
            if path.is_absolute() or ".." in path.parts:
                raise UnsupportedExecution("source traversal escapes its pinned parent")
            current = os.dup(parent)
            parts = path.parts
        try:
            for index, name in enumerate(parts):
                opened = os.open(name, flags | (os.O_DIRECTORY if index < len(parts) - 1 else 0), dir_fd=current)
                os.close(current)
                current = opened
            mode = os.fstat(current).st_mode
            if not stat.S_ISREG(mode) and not stat.S_ISDIR(mode):
                raise UnsupportedExecution("filesystem grants require regular files or directories")
            self.fds.append(current)
            return current
        except BaseException:
            os.close(current)
            raise


def check_tree(fd):
    info = os.fstat(fd)
    if stat.S_ISREG(info.st_mode):
        if info.st_nlink != 1:
            raise UnsupportedExecution("filesystem grants contain hardlink aliases")
        return
    for name in os.listdir(fd):
        info = os.stat(name, dir_fd=fd, follow_symlinks=False)
        if stat.S_ISLNK(info.st_mode):
            continue  # A symlink resolves within the completed namespace view.
        if not stat.S_ISREG(info.st_mode) and not stat.S_ISDIR(info.st_mode):
            raise UnsupportedExecution("filesystem grants contain nonregular files")
        child = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK, dir_fd=fd)
        try:
            if os.fstat(child).st_ino != info.st_ino or os.fstat(child).st_dev != info.st_dev:
                raise UnsupportedExecution("filesystem source changed during validation")
            check_tree(child)
        finally:
            os.close(child)


def mount(argv, fd, target, writable=False):
    for parent in reversed(target.parents):
        if parent != Path("/"):
            argv.extend(("--dir", str(parent)))
    argv.extend(("--bind" if writable else "--ro-bind", "/proc/self/fd/" + str(fd), str(target)))


def prefix(binary, sources):
    argv = [binary, "--new-session", "--die-with-parent", "--unshare-user", "--unshare-pid",
        "--unshare-ipc", "--unshare-uts", "--unshare-cgroup-try", "--cap-drop", "ALL"]
    runtime = []
    for name in RUNTIME_PATHS:
        path = Path(name)
        if path.exists():
            # Only the immutable system-runtime aliases may be resolved here.
            source = path.resolve()
            fd = sources.open(source)
            mount(argv, fd, path)
            runtime.append((source, path, False, fd))
    argv.extend(("--dev", "/dev", "--tmpfs", "/tmp", "--tmpfs", "/run", "--dir", "/proc", "--dir", "/tmp/aka-home"))
    return argv, runtime


def interpreter_path():
    for candidate in (Path(sys.executable).resolve(), Path("/usr/bin/python3").resolve(), Path("/usr/local/bin/python3").resolve()):
        if candidate.is_file() and any(candidate.is_relative_to(Path(root)) for root in RUNTIME_PATHS):
            return candidate
    raise UnsupportedExecution("isolated command closure requires Python in the immutable system runtime")


def exec_command(argv, sources, command, environment=None):
    # Isolated Python startup cannot execute caller PYTHONPATH/sitecustomize
    # before closing the mount descriptors. The payload receives only stdio.
    interpreter = interpreter_path()
    data = json.dumps(environment or {"PATH": "/usr/bin:/bin"}).encode("utf-8")
    fd = sources.data(data)
    # bwrap copies sealed input into its fresh /run; the trusted closure reads
    # and unlinks that copy before any workload or caller loader hook executes.
    argv.extend(("--file", str(fd), "/run/aka-command-environment"))
    argv.extend(("--", str(interpreter), "-I", "-S", "-c", CLOSE_AND_EXEC,
        ",".join(map(str, sources.fds)), *command))


def availability(value):
    if platform.system() != "Linux":
        return None, "Linux mount namespaces are unavailable"
    binary = shutil.which(value)
    if binary is None:
        return None, "Bubblewrap executable is unavailable"
    sources = Sources()
    try:
        argv, _ = prefix(binary, sources)
        argv.extend(("--chdir", "/tmp"))
        exec_command(argv, sources, ("/bin/true",))
        probe = subprocess.run(argv, env={"PATH": "/usr/bin:/bin"}, capture_output=True,
            timeout=10, close_fds=True, pass_fds=tuple(sources.fds), cwd="/")
        if probe.returncode:
            return None, "Bubblewrap descriptor mount namespace probe failed"
        return binary, "available"
    except (OSError, subprocess.TimeoutExpired, UnsupportedExecution) as error:
        return None, "Bubblewrap probe failed: " + type(error).__name__
    finally:
        sources.close()


def restrictions(paths, grants):
    targets = set()
    for path in map(absolute, paths):
        protected = []
        for source, target, _, _ in grants:
            if path.is_relative_to(target):
                protected.append(source / path.relative_to(target))
            elif target.is_relative_to(path):
                protected.append(source)
        for source_path in protected:
            for source, target, _, _ in grants:
                if source_path.is_relative_to(source):
                    targets.add(target / source_path.relative_to(source))
                elif source.is_relative_to(source_path):
                    targets.add(target)
    return sorted(targets, key=lambda value: (len(value.parts), str(value)))


@contextmanager
def launch(binary, command):
    sources = Sources()
    try:
        yield _launch(binary, command, sources)
    except OSError as error:
        raise UnsupportedExecution("filesystem source could not be pinned: " + type(error).__name__) from error
    finally:
        sources.close()


def _launch(binary, command, sources):
    access = command.filesystem_access
    grants = []
    for grant in sorted(access.mounts, key=lambda value: len(value.source.parts)):
        source, target = absolute(grant.source), absolute(grant.target)
        if source in ROOTS or target in ROOTS or any(source.is_relative_to(path) or target.is_relative_to(path) for path in PRIVATE_NAMESPACES):
            raise UnsupportedExecution("filesystem grant exposes a private or broad host namespace")
        if any(overlaps(target, Path(name)) for name in RUNTIME_PATHS):
            if not grant.writable and source == target and any(source.is_relative_to(Path(name)) for name in RUNTIME_PATHS):
                continue
            raise UnsupportedExecution("filesystem grant shadows the system runtime")
        if any(overlaps(target, previous[1]) for previous in grants):
            raise UnsupportedExecution("filesystem mount targets overlap")
        if any(overlaps(source, previous[0]) and grant.writable != previous[2] for previous in grants):
            raise UnsupportedExecution("filesystem mount sources contain writable aliases")
        parents = [item for item in grants if source.is_relative_to(item[0])]
        parent = max(parents, key=lambda value: len(value[0].parts)) if parents else None
        fd = sources.open(source.relative_to(parent[0]), parent=parent[3]) if parent else sources.open(source)
        check_tree(fd)
        grants.append((source, target, grant.writable, fd))
    workspace = absolute(command.workspace)
    try:
        require_path_access(access, workspace, write=True)
    except PermissionError as error:
        raise UnsupportedExecution(str(error)) from error
    writable = [grant for grant in grants if grant[2] and workspace.is_relative_to(grant[1])]
    if len(writable) != 1:
        raise UnsupportedExecution("command workspace is outside the bound writable view")
    working = sources.open(workspace.relative_to(writable[0][1]), parent=writable[0][3])
    if not stat.S_ISDIR(os.fstat(working).st_mode):
        raise UnsupportedExecution("command workspace is not an existing granted directory")
    argv, runtime = prefix(binary, sources)
    for source, target, writable, fd in grants:
        mount(argv, fd, target, writable)
    visible = [*grants, *runtime]

    def physical(path):
        candidates = [grant for grant in visible if path.is_relative_to(grant[1])]
        grant = max(candidates, key=lambda value: len(value[1].parts))
        return sources.open(path.relative_to(grant[1]), parent=grant[3])

    for path in restrictions(access.read_only_paths, visible):
        fd = physical(path)
        check_tree(fd)
        mount(argv, fd, path)
    for path in restrictions(access.hidden_paths, visible):
        try:
            fd = physical(path)
        except FileNotFoundError:
            argv.extend(("--tmpfs", str(path)))
            continue
        if stat.S_ISREG(os.fstat(fd).st_mode):
            mount(argv, sources.data(), path)
        else:
            argv.extend(("--tmpfs", str(path)))
    environment = dict(command.environment)
    if any(not isinstance(key, str) or not key or "=" in key or "\0" in key
            or not isinstance(value, str) or "\0" in value for key, value in environment.items()):
        raise UnsupportedExecution("execution environment contains invalid names or values")
    environment.setdefault("PATH", "/usr/local/bin:/usr/bin:/bin")
    environment.setdefault("HOME", "/tmp/aka-home")
    environment.setdefault("TMPDIR", "/tmp")
    environment["PWD"] = str(workspace)
    argv.extend(("--chdir", str(workspace)))
    exec_command(argv, sources, command.command, environment)
    # Dynamic loader/Python startup values belong only to the isolated payload.
    return argv, {"PATH": "/usr/bin:/bin"}, Path("/"), tuple(sources.fds)
