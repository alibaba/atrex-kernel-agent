from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Protocol


DEPENDENCY_GUARD_POLL_SECONDS = 0.25
AGENT_HEARTBEAT_SECONDS = 60


class ProcessRunner(Protocol):
    def __call__(
        self,
        command: list[str],
        cwd: Path,
        timeout: int | None,
        env: dict | None = None,
    ) -> tuple[str, str, int, bool]: ...


def descendant_process_commands(root_pid: int) -> list[tuple[int, list[str]]]:
    """Return live descendants and argv, tolerating process exit races."""
    if sys.platform == "darwin":
        result = subprocess.run(
            ["ps", "-axo", "pid=,ppid=,command="],
            capture_output=True,
            text=True,
            check=True,
        )
        rows = {}
        for line in result.stdout.splitlines():
            fields = line.strip().split(None, 2)
            if len(fields) == 3:
                rows[int(fields[0])] = (int(fields[1]), fields[2].split())
        seen = {root_pid}
        descendants = []
        pending = [root_pid]
        while pending:
            parent = pending.pop()
            for pid, (ppid, argv) in rows.items():
                if ppid == parent and pid not in seen:
                    seen.add(pid)
                    pending.append(pid)
                    descendants.append((pid, argv))
        return descendants
    pending = [root_pid]
    seen = {root_pid}
    descendants: list[tuple[int, list[str]]] = []
    while pending:
        parent = pending.pop()
        task_dir = Path(f"/proc/{parent}/task")
        try:
            thread_dirs = list(task_dir.iterdir())
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        children: set[int] = set()
        for thread_dir in thread_dirs:
            try:
                children.update(
                    int(value)
                    for value in (thread_dir / "children").read_text().split()
                )
            except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
                continue
        for pid in children:
            if pid in seen:
                continue
            seen.add(pid)
            pending.append(pid)
            try:
                raw = Path(f"/proc/{pid}/cmdline").read_bytes()
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                continue
            argv = [part.decode(errors="replace") for part in raw.split(b"\0") if part]
            descendants.append((pid, argv))
    return descendants


def descendant_process_groups(root_pid: int) -> set[int]:
    """Capture every process group in a coding session's live process tree."""
    process_groups: set[int] = set()
    for pid in [
        root_pid,
        *[pid for pid, _argv in descendant_process_commands(root_pid)],
    ]:
        try:
            process_groups.add(os.getpgid(pid))
        except ProcessLookupError:
            pass
    return process_groups


def signal_process_groups(process_groups: set[int], sig: signal.Signals) -> None:
    for process_group in tuple(process_groups):
        try:
            os.killpg(process_group, sig)
        except ProcessLookupError:
            pass


def _monitor(
    proc, stop, failures, environment, groups, violation=None, unavailable=None
):
    """Observe a local process tree; application policy is supplied by its host."""
    while not stop.wait(DEPENDENCY_GUARD_POLL_SECONDS):
        groups.update(descendant_process_groups(proc.pid))
        if proc.poll() is not None:
            return
        failure = unavailable(environment) if unavailable is not None else None
        if failure is None and violation is not None:
            for pid, argv in descendant_process_commands(proc.pid):
                reason = violation(argv, environment)
                if reason:
                    failure = (126, f"Agent host policy violation: pid={pid}: {reason}")
                    break
        if failure is not None:
            failures.append(failure)
            signal_process_groups(groups, signal.SIGKILL)
            return


def dependency_guard(
    proc, stop, failures, environment, groups, violation=None, unavailable=None
):
    try:
        _monitor(proc, stop, failures, environment, groups, violation, unavailable)
    except Exception as error:
        failures.append((126, f"Agent process monitor failed: {type(error).__name__}"))
        signal_process_groups(groups, signal.SIGKILL)


def run_bounded(
    command: list[str],
    cwd: Path,
    timeout: int | None,
    env: dict | None = None,
    *,
    spawn=None,
    inherited_descriptors=(),
    unavailable=None,
    violation=None,
) -> tuple[str, str, int, bool]:
    """Run a guarded command, optionally without a wall-clock deadline."""
    from ..provider import validate_timeout

    validate_timeout(timeout)
    spawn = spawn or spawn_session
    proc = spawn(
        command,
        role="coding-agent",
        inherited_descriptors=inherited_descriptors,
        environment=env,
        cwd=str(cwd),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    guard_stop = threading.Event()
    failures = []
    groups = {proc.pid}
    guard = threading.Thread(
        target=dependency_guard,
        args=(proc, guard_stop, failures, env or {}, groups, violation, unavailable),
        name=f"agent-monitor-{proc.pid}",
        daemon=True,
    )
    guard.start()
    timed_out = False
    started = time.monotonic()
    deadline = started + timeout if timeout is not None else None
    last_output_change = started
    output_sizes = (0, 0)
    try:
        while True:
            remaining = (
                None if deadline is None else max(0, deadline - time.monotonic())
            )
            try:
                stdout, stderr = proc.communicate(
                    timeout=AGENT_HEARTBEAT_SECONDS
                    if remaining is None
                    else min(AGENT_HEARTBEAT_SECONDS, remaining)
                )
                break
            except subprocess.TimeoutExpired as exc:
                now = time.monotonic()
                if deadline is not None and now >= deadline:
                    raise
                # communicate() keeps draining both pipes across timeout retries.
                # TimeoutExpired carries cumulative bytes, even with text=True.
                sizes = (len(exc.output or b""), len(exc.stderr or b""))
                if sizes != output_sizes:
                    last_output_change = now
                    output_sizes = sizes
                print(
                    f"[agent-heartbeat] elapsed={round(now - started)}s "
                    f"process_running={proc.poll() is None} "
                    f"stdout_bytes={sizes[0]} stderr_bytes={sizes[1]} "
                    f"output_unchanged_seconds={round(now - last_output_change)}",
                    flush=True,
                )
    except subprocess.TimeoutExpired:
        timed_out = True
        process_groups = groups | descendant_process_groups(proc.pid)
        signal_process_groups(process_groups, signal.SIGKILL)
        stdout, stderr = proc.communicate()
    except BaseException:
        process_groups = groups | descendant_process_groups(proc.pid)
        signal_process_groups(process_groups, signal.SIGTERM)
        try:
            proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            signal_process_groups(process_groups, signal.SIGKILL)
            proc.communicate()
        raise
    finally:
        guard_stop.set()
        guard.join(timeout=1)
    # Reap surviving children even when the CLI leader exited successfully.
    signal_process_groups(groups, signal.SIGKILL)
    returncode = proc.returncode
    if failures:
        returncode, message = failures[0]
        stderr = (stderr or "") + "\n" + message + "\n"
    return stdout or "", stderr or "", returncode, timed_out


def spawn_session(
    command, *, role, environment=None, inherited_descriptors=(), **options
):
    """Own a local process group without constructing a recovery host."""
    del role
    return subprocess.Popen(
        command,
        env=environment,
        start_new_session=True,
        close_fds=True,
        pass_fds=inherited_descriptors,
        **options,
    )
