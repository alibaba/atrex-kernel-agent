"""Runtime's container proc view, adapted to Light's FD-based process launcher.

OCI may reject mounting a new procfs. The trusted parent mirrors only stat,
status, cmdline and task/children of its own descendants into a private directory.
There are no host fd, root, cwd, mem or environ links. The launcher may add a
fixed exe alias to the allowlisted Provider binary for standalone Bun. Core can still identify,
suspend and clean up detached Agent subprocesses using namespace-local PIDs.
"""

from __future__ import annotations

import array
import argparse
import hashlib
import os
import shutil
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

PROC_SOURCE = "/__aka_private_proc_view__"
SHIM_SOURCE = "/__aka_private_proc_shim__"
SHIM_TARGET = "/run/aka-proc/libproc-maps.so"
PUBLIC_STATISTICS = ("cpuinfo", "meminfo", "stat", "uptime", "version", "loadavg")


def projected_mounts() -> list[str]:
    """Sources are substituted by the trusted wrapper before starting bwrap."""
    return [
        "--ro-bind", PROC_SOURCE, "/proc",
        *[part for name in PUBLIC_STATISTICS if (Path("/proc") / name).is_file()
          for part in ("--ro-bind", f"/proc/{name}", f"/proc/{name}")],
    ]


def _write(path: Path, value: bytes) -> None:
    if path.is_file() and path.read_bytes() == value:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".next")
    temporary.write_bytes(value)
    temporary.replace(path)


def snapshot(root_pid: int, destination: Path, *, proc: Path = Path("/proc")) -> bool:
    """Publish a namespace-local census; never enumerate unrelated host processes."""
    pending = [root_pid]
    records: dict[int, tuple[list[str], bytes, str]] = {}
    seen: set[int] = set()
    namespace_depth = None
    while pending:
        pid = pending.pop()
        if pid in seen:
            continue
        seen.add(pid)
        path = proc / str(pid)
        try:
            children: set[int] = set()
            for task in (path / "task").iterdir():
                children.update(int(v) for v in (task / "children").read_text().split())
            pending.extend(children)
            status = (path / "status").read_text()
            ns = {
                parts[0].rstrip(":"): parts[1:]
                for line in status.splitlines()
                if (parts := line.split()) and parts[0] in {"NSpid:", "NSpgid:", "NSsid:"}
            }
            if pid == root_pid:
                namespace_depth = len(ns.get("NSpid", []))
            if namespace_depth is None or len(ns.get("NSpid", [])) <= namespace_depth:
                continue
            raw = (path / "stat").read_text()
            name, rest = raw.rsplit(")", 1)
            fields = rest.split()
            local = ns["NSpid"][namespace_depth]
            fields[2] = ns["NSpgid"][namespace_depth]
            fields[3] = ns["NSsid"][namespace_depth]
            records[pid] = (
                [local, name[name.index("(") :] + ")", *fields],
                (path / "cmdline").read_bytes(),
                status,
            )
        except (OSError, ValueError, IndexError, KeyError):
            continue  # Exit races are normal; never substitute unrelated PID data.
    if not records:
        return False
    ids = {outer: fields[0] for outer, (fields, _, _) in records.items()}
    if "1" not in ids.values():
        return False
    children_by_parent: dict[str, list[str]] = {}
    for fields, command, status in records.values():
        fields[3] = ids.get(int(fields[3]), "0")  # stat ppid
        children_by_parent.setdefault(fields[3], []).append(fields[0])
        directory = destination / fields[0]
        _write(directory / "stat", (" ".join(fields) + "\n").encode())
        # Core sees local PIDs; don't expose host identities in NSpid mappings.
        statuses = []
        for line in status.splitlines():
            key = line.split(":", 1)[0]
            if key in {"Pid", "Tgid", "PPid", "NSpid", "NStgid", "NSpgid", "NSsid"}:
                continue
            # Name, State and other public statistics describe only this Agent.
            statuses.append(line)
        statuses.extend(
            (
                f"Pid:\t{fields[0]}",
                f"PPid:\t{fields[3]}",
                f"NSpid:\t{fields[0]}",
                f"NSpgid:\t{fields[4]}",
                f"NSsid:\t{fields[5]}",
            )
        )
        _write(directory / "status", ("\n".join(statuses) + "\n").encode())
        _write(directory / "cmdline", command)
    for local in ids.values():
        _write(
            destination / local / "task" / local / "children",
            " ".join(children_by_parent.get(local, [])).encode(),
        )
    for old in destination.iterdir():
        if old.name.isdecimal() and old.name not in ids.values():
            shutil.rmtree(old)
    # A synthetic self supplies the namespace identity to Core, not host files.
    if not (destination / "self").is_symlink():
        (destination / "self").symlink_to("1")
    return True


def owns_process(root: int, pid: int, *, proc: Path = Path("/proc")) -> bool:
    """Verify host ancestry; a client cannot request another PID's mappings."""
    visited: set[int] = set()
    while pid > 1 and pid not in visited:
        if pid == root:
            return True
        visited.add(pid)
        try:
            pid = int((proc / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            return False
    return False


def _serve_maps(
    listener: socket.socket,
    root: int,
    stopped: threading.Event,
    workspace: Path,
    visible_workspace: str,
) -> None:
    while not stopped.is_set():
        try:
            client, _ = listener.accept()
        except TimeoutError:
            continue
        except OSError:
            return
        with client:
            client.settimeout(0.5)
            pid, _uid, _gid = struct.unpack(
                "3i",
                client.getsockopt(socket.SOL_SOCKET, int(getattr(socket, "SO_PEERCRED", 17)), 12),
            )
            if not owns_process(root, pid):
                continue
            try:
                request = b""
                while b"\n" not in request and len(request) < 32:
                    part = client.recv(32 - len(request))
                    if not part:
                        break
                    request += part
                if request == b"M\n":
                    fd = os.open(f"/proc/{pid}/maps", os.O_RDONLY | os.O_CLOEXEC)
                    try:
                        client.sendmsg(
                            [b"M"], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", [fd]))]
                        )
                    finally:
                        os.close(fd)
                elif request.startswith(b"R") and request.endswith(b"\n"):
                    number = request[1:-1]
                    if not number.isdigit() or len(number) > 10:
                        continue
                    # Readlink metadata only. Never send a descriptor for a requested
                    # FD, follow its target, or accept a target PID/path from the client.
                    target = os.readlink(f"/proc/{pid}/fd/{int(number)}")
                    host = str(workspace)
                    if target == host or target.startswith(host + "/"):
                        target = visible_workspace + target[len(host) :]
                    client.sendall(target.encode())
            except OSError:
                continue


def _install_maps_shim(destination: Path) -> Path:
    source = Path(__file__).with_name("proc_maps.c")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    cache = Path(tempfile.gettempdir()) / f"aka-proc-maps-{os.getuid()}-{digest}"
    cache.mkdir(mode=0o700, exist_ok=True)
    if (cache.is_symlink() or cache.stat().st_uid != os.getuid()
            or cache.stat().st_mode & 0o077):
        raise RuntimeError("Unsafe process-view compiler cache")
    compiled = cache / "libproc-maps.so"
    if compiled.is_symlink():
        raise RuntimeError("Unsafe process-view compiler output")
    if not compiled.exists():
        with tempfile.TemporaryDirectory(dir=cache) as temporary:
            output = Path(temporary) / compiled.name
            subprocess.run(
                ["cc", "-shared", "-fPIC", "-O2", str(source), "-ldl", "-o", str(output)],
                check=True,
                capture_output=True,
                timeout=60,
            )
            output.chmod(0o500)
            output.replace(compiled)
    if (not compiled.is_file() or compiled.stat().st_uid != os.getuid()
            or compiled.stat().st_mode & 0o022):
        raise RuntimeError("Unsafe process-view compiler output")
    target = destination / compiled.name
    shutil.copy2(compiled, target)
    return target


@contextmanager
def local_process_view(argv: list[str], executable: Path | None = None) -> Iterator[tuple[Path, list[str]]]:
    """Keep high-frequency process metadata off the persistent workspace filesystem."""
    source = next(
        i + 1 for i in range(len(argv) - 2)
        if argv[i:i + 3] == ["--ro-bind", PROC_SOURCE, "/proc"]
    )
    # Linux containers normally provide a small local tmpfs here. No shared host
    # /dev/shm tree is exposed: bwrap mounts only this private directory read-only.
    memory = Path("/dev/shm")
    parent = str(memory) if memory.is_dir() and os.access(memory, os.W_OK) else None
    with tempfile.TemporaryDirectory(prefix="aka-proc-view-", dir=parent) as name:
        destination = Path(name)
        (destination / "1").mkdir()
        (destination / "1/cgroup").write_text("0::/\n")
        if executable is not None:
            (destination / "1/exe").symlink_to(executable)
        for relative in PUBLIC_STATISTICS:
            (destination / relative).touch()
        rewritten = list(argv)
        rewritten[source] = str(destination)
        yield destination, rewritten


def _run(destination: Path, argv: list[str], workspace: Path, argument_fd: int) -> int:
    endpoint = destination / ".maps.sock"
    endpoint.unlink(missing_ok=True)
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    # Attempt paths exceed AF_UNIX's short sockaddr pathname limit. Bind relative
    # to the private directory before starting any thread, then restore cwd.
    previous = os.open(".", os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.chdir(destination)
        listener.bind(endpoint.name)
    finally:
        os.fchdir(previous)
        os.close(previous)
    listener.listen(32)
    listener.settimeout(0.2)
    try:
        # The anonymous environment FD must cross BOTH the durable wrapper and
        # this wrapper. Never turn its contents into process-visible argv.
        process = subprocess.Popen(argv, pass_fds=(argument_fd,))
    except BaseException:
        listener.close()
        raise
    stopped = threading.Event()
    # Light preserves the workspace path in both namespaces; auxiliary views
    # may bind a different source but still use this visible destination.
    visible_workspace = str(workspace)
    broker = threading.Thread(
        target=_serve_maps,
        args=(listener, process.pid, stopped, workspace, visible_workspace),
        daemon=True,
    )
    broker.start()
    received_signal = 0

    def terminate(signum, _frame):
        nonlocal received_signal
        received_signal = signum
        # Killing bwrap also kills its PID namespace; --die-with-parent covers
        # an uncatchable wrapper death. Do not signal any unrelated process.
        if process.poll() is None:
            process.kill()

    handlers = {sig: signal.signal(sig, terminate) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        while process.poll() is None:
            snapshot(process.pid, destination)
            time.sleep(0.025)
        code = process.wait()
        return 128 + received_signal if received_signal else (128 - code if code < 0 else code)
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        stopped.set()
        listener.close()
        broker.join(timeout=1)
        if process.poll() is None:
            process.kill()
            process.wait()


def main() -> int:
    """Own transient view, shim and bwrap for precisely one Agent invocation."""
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--argument-fd", required=True, type=int)
    parser.add_argument("--provider-executable", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    argv = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not argv or args.argument_fd < 3:
        parser.error("proc view requires bwrap argv and an inherited argument FD")
    os.fstat(args.argument_fd)
    try:
        # /dev/shm is frequently noexec. Keep the library separately on /tmp;
        # only the read-only library file, never its cache directory, is mounted.
        with tempfile.TemporaryDirectory(prefix="aka-proc-library-") as name:
            shim = _install_maps_shim(Path(name))
            index = next(i + 1 for i in range(len(argv) - 2)
                         if argv[i:i + 3] == ["--ro-bind", SHIM_SOURCE, SHIM_TARGET])
            argv[index] = str(shim)
            with local_process_view(argv, args.provider_executable) as (destination, rewritten):
                return _run(destination, rewritten, args.workspace, args.argument_fd)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        # No CLI/provider environment, stdout, or private path is echoed.
        print(f"Agent projected /proc setup failed ({type(exc).__name__}); "
              "check cc, writable temporary storage and Bubblewrap namespace permissions", file=sys.stderr)
        return 125
    finally:
        try:
            os.close(args.argument_fd)
        except OSError:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
