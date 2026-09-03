#!/usr/bin/env python3
"""Poll a blocked SSH GPU environment and restart its AKA optimization command."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from orchestrator.recovery_processes import (  # noqa: E402
    HANDOFF_ID_ENV,
    HANDOFF_LOCK_FD_ENV,
    clear_process_registry,
    matching_processes,
    process_identity_matches,
    record_process_tree,
    register_process,
    terminate_processes,
)

RESTART_HANDOFF_TIMEOUT = 10_800
STOP_WAIT_TIMEOUT = 120
PROCESS_TREE_REFRESH_INTERVAL = 1.0
HANDOFF_MARKER_KEY = "restart_handoff"
_STOP_SIGNALLED = False


class RecoveryStopRequested(BaseException):
    """A verified rollback request interrupted monitoring or restart."""


def _load_restart(state_dir: Path) -> dict[str, Any]:
    path = state_dir / "restart.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read restart metadata {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError("restart metadata must be a JSON object")
    required_strings = (
        "cwd",
        "environment_state_file",
        "sandbox_hardware",
        "ssh_target",
        "health_command",
    )
    for key in required_strings:
        if not isinstance(value.get(key), str) or not value[key]:
            raise RuntimeError(f"restart metadata has invalid {key}")
    command = value.get("command")
    if not isinstance(command, list) or not command or not all(
        isinstance(item, str) and item for item in command
    ):
        raise RuntimeError("restart metadata has invalid command")
    interval = value.get("poll_interval")
    if not isinstance(interval, int) or isinstance(interval, bool) or interval <= 0:
        raise RuntimeError("restart metadata has invalid poll_interval")
    runtime_binds = value.get("ssh_runtime_binds", [])
    if not isinstance(runtime_binds, list) or not all(
        isinstance(item, str) for item in runtime_binds
    ):
        raise RuntimeError("restart metadata has invalid ssh_runtime_binds")
    ssh_gpu = value.get("ssh_gpu")
    if (
        not isinstance(ssh_gpu, int)
        or isinstance(ssh_gpu, bool)
        or not 0 <= ssh_gpu <= 31
    ):
        raise RuntimeError("restart metadata has invalid ssh_gpu")
    return value


def _acquire_lock(state_dir: Path) -> TextIO | None:
    """Hold an OS-owned advisory lock; file contents are diagnostic only."""
    path = state_dir / "monitor.lock"
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    os.fchmod(descriptor, 0o600)
    handle = os.fdopen(descriptor, "r+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    handle.seek(0)
    handle.truncate()
    handle.write(str(os.getpid()) + "\n")
    handle.flush()
    os.fsync(handle.fileno())
    return handle


def _handoff_lock_active(state_dir: Path) -> bool:
    """Return whether the exact optimizer spawned for a restart still owns its lock."""
    descriptor = os.open(
        state_dir / "restart-child.lock", os.O_RDWR | os.O_CREAT, 0o600
    )
    os.fchmod(descriptor, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        return False
    finally:
        os.close(descriptor)


def _acquire_handoff_lock(state_dir: Path) -> int | None:
    descriptor = os.open(
        state_dir / "restart-child.lock", os.O_RDWR | os.O_CREAT, 0o600
    )
    os.fchmod(descriptor, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(descriptor)
        return None
    os.set_inheritable(descriptor, True)
    return descriptor


def _read_pid(path: Path) -> int | None:
    try:
        pid = int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None
    return pid if pid > 0 else None


def _remove_matching_pid(path: Path, pid: int) -> None:
    if _read_pid(path) != pid:
        return
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _write_private_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.chmod(0o600)
    temporary.replace(path)


def _handoff_metadata(path: Path) -> tuple[str, float] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        handoff = value[HANDOFF_MARKER_KEY]
        handoff_id = handoff["id"]
        started_at = handoff["started_at"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return None
    if (
        not isinstance(handoff_id, str)
        or len(handoff_id) != 32
        or any(character not in "0123456789abcdef" for character in handoff_id)
        or not isinstance(started_at, (int, float))
        or isinstance(started_at, bool)
        or started_at <= 0
    ):
        return None
    return handoff_id, float(started_at)


def _write_handoff_metadata(
    path: Path, handoff_id: str, started_at: float
) -> None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot update handoff marker {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"handoff marker must contain a JSON object: {path}")
    value[HANDOFF_MARKER_KEY] = {
        "id": handoff_id,
        "started_at": started_at,
        "started_at_iso": datetime.fromtimestamp(
            started_at, timezone.utc
        ).isoformat(),
    }
    _write_private_json(path, value)


def _begin_handoff(
    failure: Path,
    restarting: Path,
    handoff_id: str,
    started_at: float,
) -> None:
    # Refresh the durable failure payload before the atomic name transition. Its
    # original mtime may predate recovery by hours and is not a handoff clock.
    _write_handoff_metadata(failure, handoff_id, started_at)
    failure.replace(restarting)


def _ensure_handoff_metadata(
    state_dir: Path, restarting: Path, pid: int | None
) -> tuple[str, float]:
    metadata = _handoff_metadata(restarting)
    if metadata is None:
        # Migrate a handoff created by the previous release. Starting the timeout
        # at migration is conservative and avoids treating an old outage as an old
        # optimizer initialization.
        metadata = uuid.uuid4().hex, time.time()
        _write_handoff_metadata(restarting, *metadata)
    handoff_id, started_at = metadata
    if pid is not None and process_identity_matches(state_dir, handoff_id, pid):
        record_process_tree(state_dir, handoff_id, pid)
    return handoff_id, started_at


def _stop_requested(state_dir: Path) -> bool:
    return _STOP_SIGNALLED or (state_dir / "stop.request").is_file()


def _raise_if_stop_requested(state_dir: Path) -> None:
    if _stop_requested(state_dir):
        raise RecoveryStopRequested


def _interruptible_sleep(state_dir: Path, seconds: float) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        _raise_if_stop_requested(state_dir)
        time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))


def _restore_restarting_marker(state_dir: Path) -> None:
    restarting = state_dir / "restarting.json"
    failure = state_dir / "failure.json"
    if restarting.is_file():
        if failure.is_file():
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            restarting.replace(state_dir / f"superseded-restart-{stamp}.json")
        else:
            restarting.replace(failure)
    for name in ("restart.ready", "restart.pid"):
        try:
            (state_dir / name).unlink()
        except FileNotFoundError:
            pass


def _archive_restarting_marker(state_dir: Path, handoff_id: str) -> Path:
    restarting = state_dir / "restarting.json"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    archived = state_dir / f"recovered-{stamp}.json"
    restarting.replace(archived)
    clear_process_registry(state_dir, handoff_id)
    return archived


def _terminate_handoff_processes(
    state_dir: Path,
    handoff_id: str,
    pid: int | None,
) -> None:
    """Terminate every identity-owned session before permitting another restart."""
    result = terminate_processes(
        state_dir,
        handoff_id,
        root_pid=pid,
        require_registered_owner=_handoff_lock_active(state_dir),
    )
    lock_still_active = _handoff_lock_active(state_dir)
    if not result.complete:
        detail = "; ".join(result.errors) or (
            "owned processes remain: "
            + ", ".join(str(value) for value in result.remaining_pids)
        )
        raise RuntimeError(f"cannot terminate recovery process tree: {detail}")
    if lock_still_active:
        raise RuntimeError(
            "cannot terminate recovery process tree: handoff lock remains owned by "
            "an unregistered descendant"
        )


def _reconcile_interrupted_handoff(state_dir: Path) -> int | None:
    """Adopt a live restart child or restore a marker left by a dead monitor."""
    restarting = state_dir / "restarting.json"
    if not restarting.is_file():
        return None
    pid = _read_pid(state_dir / "restart.pid")
    handoff_id, started_at = _ensure_handoff_metadata(state_dir, restarting, pid)
    active = _handoff_lock_active(state_dir)
    root_alive = pid is not None and process_identity_matches(
        state_dir, handoff_id, pid
    )
    if not active or not root_alive:
        owned = matching_processes(state_dir, handoff_id)
        if owned or active:
            _terminate_handoff_processes(state_dir, handoff_id, pid)
        _restore_restarting_marker(state_dir)
        clear_process_registry(state_dir, handoff_id)
        print(
            "[environment-monitor] restored interrupted restart marker after "
            "clearing its process tree",
            flush=True,
        )
        return None
    print(
        f"[environment-monitor] adopting interrupted restart handoff "
        f"id={handoff_id} pid={pid}",
        flush=True,
    )
    ready = state_dir / "restart.ready"
    failure = state_dir / "failure.json"
    while True:
        _raise_if_stop_requested(state_dir)
        record_process_tree(state_dir, handoff_id, pid)
        active = _handoff_lock_active(state_dir)
        root_alive = process_identity_matches(state_dir, handoff_id, pid)
        if ready.is_file() and not failure.is_file():
            _archive_restarting_marker(state_dir, handoff_id)
            ready.unlink(missing_ok=True)
            _remove_matching_pid(state_dir / "restart.pid", pid)
            return pid
        if not active or not root_alive:
            _terminate_handoff_processes(state_dir, handoff_id, pid)
            _restore_restarting_marker(state_dir)
            clear_process_registry(state_dir, handoff_id)
            return None
        if failure.is_file() or time.time() - started_at >= RESTART_HANDOFF_TIMEOUT:
            _terminate_handoff_processes(state_dir, handoff_id, pid)
            _restore_restarting_marker(state_dir)
            clear_process_registry(state_dir, handoff_id)
            return None
        _interruptible_sleep(state_dir, PROCESS_TREE_REFRESH_INTERVAL)


def _health_command(metadata: dict[str, Any]) -> list[str]:
    sandbox = Path(__file__).resolve().parent / "sandbox.py"
    command = [
        str(Path(sys.executable).resolve()),
        str(sandbox),
        "--hardware",
        metadata["sandbox_hardware"],
        "--ssh",
        metadata["ssh_target"],
        "--ssh-gpu",
        str(metadata["ssh_gpu"]),
        "--health-command",
        metadata["health_command"],
        "--check-health",
    ]
    ssh_init = metadata.get("ssh_init")
    if isinstance(ssh_init, str) and ssh_init:
        command += ["--ssh-init", ssh_init]
    for runtime_bind in metadata.get("ssh_runtime_binds", []):
        command += ["--ssh-runtime-bind", runtime_bind]
    return command


def _archive_failure(state_dir: Path) -> Path | None:
    failure = state_dir / "failure.json"
    if not failure.is_file():
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    archived = state_dir / f"recovered-{stamp}.json"
    failure.replace(archived)
    return archived


def _restart(metadata: dict[str, Any], state_dir: Path) -> int:
    """Supervise optimizer initialization and restore the marker on early failure."""
    command = metadata["command"]
    if len(command) < 2 or not Path(command[1]).is_file():
        missing = command[1] if len(command) > 1 else ""
        raise FileNotFoundError(f"optimizer script is missing: {missing}")
    failure = Path(metadata["environment_state_file"]).expanduser().resolve()
    expected_failure = (state_dir / "failure.json").resolve()
    restarting = state_dir / "restarting.json"
    if failure != expected_failure:
        raise RuntimeError("restart metadata state path does not match its state directory")
    if restarting.exists():
        adopted_pid = _reconcile_interrupted_handoff(state_dir)
        if adopted_pid is not None:
            return adopted_pid
        raise RuntimeError(
            "interrupted restart handoff was restored; health must be checked again"
        )
    if not failure.is_file():
        raise RuntimeError("blocked marker disappeared before restart")
    handoff_lock = _acquire_handoff_lock(state_dir)
    if handoff_lock is None:
        raise RuntimeError("another optimizer restart child is still active")
    handoff_id = uuid.uuid4().hex
    started_at = time.time()

    environment = os.environ.copy()
    environment["ATREX_ENVIRONMENT_STATE_FILE"] = metadata[
        "environment_state_file"
    ]
    environment["ATREX_ENVIRONMENT_RECOVERY_OWNER"] = "1"
    environment["ATREX_SANDBOX_SSH"] = metadata["ssh_target"]
    environment["ATREX_SANDBOX_SSH_INIT"] = str(metadata.get("ssh_init") or "")
    environment["ATREX_SANDBOX_SSH_RUNTIME_BINDS"] = json.dumps(
        metadata.get("ssh_runtime_binds", []), separators=(",", ":")
    )
    environment["ATREX_SANDBOX_SSH_GPU"] = str(metadata["ssh_gpu"])
    environment["ATREX_SANDBOX_HEALTH_COMMAND"] = metadata["health_command"]
    environment["ATREX_ENVIRONMENT_POLL_INTERVAL"] = str(metadata["poll_interval"])
    environment.pop("ATREX_SANDBOX_URL", None)
    environment.pop("ATREX_SANDBOX_PROFILE", None)
    environment["ATREX_ENVIRONMENT_RESTART_HANDOFF"] = "1"
    environment["ATREX_ENVIRONMENT_RESTART_SUPERVISED"] = "1"
    environment[HANDOFF_ID_ENV] = handoff_id
    environment[HANDOFF_LOCK_FD_ENV] = str(handoff_lock)
    ready = state_dir / "restart.ready"
    try:
        ready.unlink()
    except FileNotFoundError:
        pass
    environment["ATREX_ENVIRONMENT_RESTART_READY_FILE"] = str(ready)
    log_path = state_dir / "restart.log"
    try:
        with log_path.open("a", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                cwd=metadata["cwd"],
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
                pass_fds=(handoff_lock,),
            )
    finally:
        # Popen duplicated this locked descriptor into the optimizer. Closing the
        # monitor's copy makes lock ownership exactly track the child lifetime.
        os.close(handoff_lock)

    try:
        register_process(state_dir, handoff_id, process.pid, "optimizer-root")
        record_process_tree(state_dir, handoff_id, process.pid)
    except (OSError, ProcessLookupError, ValueError):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        clear_process_registry(state_dir, handoff_id)
        raise

    def stop_process() -> None:
        _terminate_handoff_processes(state_dir, handoff_id, process.pid)
        try:
            process.wait(timeout=0)
        except subprocess.TimeoutExpired:
            pass

    try:
        (state_dir / "restart.pid").write_text(
            str(process.pid) + "\n", encoding="utf-8"
        )
    except OSError:
        stop_process()
        clear_process_registry(state_dir, handoff_id)
        raise
    try:
        _begin_handoff(failure, restarting, handoff_id, started_at)
    except (OSError, RuntimeError):
        stop_process()
        clear_process_registry(state_dir, handoff_id)
        _remove_matching_pid(state_dir / "restart.pid", process.pid)
        raise

    # Baseline setup can legitimately use the configured two-hour session budget.
    # The monitor keeps ownership throughout and restores failure.json on any early exit.
    deadline = time.monotonic() + RESTART_HANDOFF_TIMEOUT
    try:
        while not ready.is_file():
            _raise_if_stop_requested(state_dir)
            record_process_tree(state_dir, handoff_id, process.pid)
            if process.poll() is not None:
                raise RuntimeError(
                    "optimizer exited before durable campaign resume "
                    f"with status {process.returncode}"
                )
            if failure.is_file():
                raise RuntimeError("remote environment failed again during restart")
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "optimizer did not reach durable campaign resume within 10800 seconds"
                )
            _interruptible_sleep(state_dir, PROCESS_TREE_REFRESH_INTERVAL)
        if failure.is_file():
            raise RuntimeError("remote environment failed again during restart")
    except BaseException:
        # Marker restoration is allowed only after every registered identity and
        # process group has been verified gone. A cleanup failure therefore leaves
        # restarting.json in place and recovery fail-closed.
        stop_process()
        _restore_restarting_marker(state_dir)
        clear_process_registry(state_dir, handoff_id)
        raise
    _archive_restarting_marker(state_dir, handoff_id)
    ready.unlink(missing_ok=True)
    _remove_matching_pid(state_dir / "restart.pid", process.pid)
    return process.pid


def _retry_remote_cleanups(metadata: dict[str, Any], state_dir: Path) -> bool:
    ssh = shutil.which("ssh")
    if ssh is None:
        print("[environment-monitor] ssh executable not found for cleanup", flush=True)
        return False
    target = metadata["ssh_target"]
    for path in sorted(state_dir.glob("cleanup-*.json")):
        _raise_if_stop_requested(state_dir)
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            remote_dir = value["remote_dir"]
            if value.get("target") != target or not isinstance(remote_dir, str):
                raise ValueError("target or remote_dir mismatch")
            if not re.fullmatch(
                r"/tmp/atrex-sandbox\.[A-Za-z0-9._-]+", remote_dir
            ):
                raise ValueError("unsafe remote_dir")
        except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
            print(
                f"[environment-monitor] invalid cleanup marker {path}: {exc}",
                flush=True,
            )
            return False
        try:
            result = subprocess.run(
                [
                    ssh,
                    "-o",
                    "ConnectTimeout=15",
                    target,
                    "rm -rf -- " + shlex.quote(remote_dir),
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            print(
                f"[environment-monitor] deferred cleanup failed: {exc}", flush=True
            )
            return False
        if result.returncode != 0:
            detail = " ".join((result.stderr or result.stdout).split())[-1000:]
            print(f"[environment-monitor] deferred cleanup failed: {detail}", flush=True)
            return False
        path.unlink()
    return True


def _registry_handoff_ids(state_dir: Path) -> list[str]:
    root = state_dir / "restart-processes"
    if not root.is_dir():
        return []
    return sorted(
        path.name
        for path in root.iterdir()
        if path.is_dir()
        and len(path.name) == 32
        and all(character in "0123456789abcdef" for character in path.name)
    )


def _cancel_active_handoff(state_dir: Path) -> str | None:
    restarting = state_dir / "restarting.json"
    failure = state_dir / "failure.json"
    pid = _read_pid(state_dir / "restart.pid")
    marker = restarting if restarting.is_file() else failure
    metadata = _handoff_metadata(marker) if marker.is_file() else None
    if restarting.is_file() and metadata is None:
        metadata = _ensure_handoff_metadata(state_dir, restarting, pid)
    if metadata is None:
        registered_ids = _registry_handoff_ids(state_dir)
        if len(registered_ids) == 1:
            metadata = registered_ids[0], time.time()
        elif len(registered_ids) > 1:
            return "multiple recovery handoff registries require manual diagnosis"

    active = _handoff_lock_active(state_dir)
    if metadata is not None:
        handoff_id, _started_at = metadata
        if pid is not None and process_identity_matches(state_dir, handoff_id, pid):
            record_process_tree(state_dir, handoff_id, pid)
        if active or matching_processes(state_dir, handoff_id):
            try:
                _terminate_handoff_processes(state_dir, handoff_id, pid)
            except RuntimeError as exc:
                return str(exc)
        clear_process_registry(state_dir, handoff_id)
    elif active:
        return "handoff lock is active but no durable process identity is available"

    if restarting.is_file():
        _restore_restarting_marker(state_dir)
    else:
        for name in ("restart.ready", "restart.pid"):
            (state_dir / name).unlink(missing_ok=True)
    return None


def stop_recovery(state_dir: Path) -> int:
    """Request verified rollback and wait until the monitor and process tree stop."""
    state_dir = state_dir.expanduser().resolve()
    state_dir.mkdir(parents=True, exist_ok=True)
    request = state_dir / "stop.request"
    _write_private_json(
        request,
        {
            "schema_version": 1,
            "requested_at": datetime.now(timezone.utc).isoformat(),
            "requester_pid": os.getpid(),
        },
    )
    deadline = time.monotonic() + STOP_WAIT_TIMEOUT
    signalled_pid: int | None = None
    while time.monotonic() < deadline:
        lock = _acquire_lock(state_dir)
        if lock is not None:
            try:
                error = _cancel_active_handoff(state_dir)
                if error is not None:
                    print(
                        f"[environment-monitor] rollback incomplete: {error}",
                        file=sys.stderr,
                        flush=True,
                    )
                    return 1
                (state_dir / "monitor.pid").unlink(missing_ok=True)
                request.unlink(missing_ok=True)
                print(
                    "[environment-monitor] rollback stop completed; no owned "
                    "process remains",
                    flush=True,
                )
                return 0
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
                lock.close()

        monitor_pid = _read_pid(state_dir / "monitor.pid")
        lock_pid = _read_pid(state_dir / "monitor.lock")
        if (
            monitor_pid is not None
            and monitor_pid == lock_pid
            and monitor_pid != signalled_pid
        ):
            try:
                os.kill(monitor_pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            else:
                signalled_pid = monitor_pid
        time.sleep(0.1)
    print(
        "[environment-monitor] rollback timed out before monitor lock release",
        file=sys.stderr,
        flush=True,
    )
    return 1


def run_monitor(
    state_dir: Path, *, once: bool = False, no_restart: bool = False
) -> int:
    global _STOP_SIGNALLED

    state_dir = state_dir.expanduser().resolve()
    metadata = _load_restart(state_dir)
    lock = _acquire_lock(state_dir)
    if lock is None:
        print("[environment-monitor] another monitor is already active", flush=True)
        return 0
    monitor_pid = state_dir / "monitor.pid"
    _STOP_SIGNALLED = False
    handled_signals = (signal.SIGTERM, signal.SIGHUP)
    previous_handlers: dict[signal.Signals, Any] = {}

    def request_stop(_signum: int, _frame: object) -> None:
        global _STOP_SIGNALLED

        _STOP_SIGNALLED = True

    for handled_signal in handled_signals:
        try:
            previous_handlers[handled_signal] = signal.getsignal(handled_signal)
            signal.signal(handled_signal, request_stop)
        except ValueError:
            previous_handlers.pop(handled_signal, None)
    try:
        monitor_pid.write_text(str(os.getpid()) + "\n", encoding="utf-8")
        _raise_if_stop_requested(state_dir)
        adopted_pid = _reconcile_interrupted_handoff(state_dir)
        if adopted_pid is not None:
            print(
                f"[environment-monitor] optimization restart handoff adopted pid={adopted_pid}",
                flush=True,
            )
            return 0
        while True:
            _raise_if_stop_requested(state_dir)
            checked_at = datetime.now(timezone.utc).isoformat()
            result = subprocess.run(
                _health_command(metadata),
                cwd=metadata["cwd"],
                capture_output=True,
                text=True,
            )
            _raise_if_stop_requested(state_dir)
            if result.returncode == 0:
                print(
                    f"[environment-monitor] environment recovered at {checked_at}",
                    flush=True,
                )
                if not _retry_remote_cleanups(metadata, state_dir):
                    detail = "deferred remote workspace cleanup is still pending"
                elif no_restart:
                    _archive_failure(state_dir)
                    return 0
                else:
                    try:
                        pid = _restart(metadata, state_dir)
                    except (
                        OSError,
                        RuntimeError,
                        subprocess.SubprocessError,
                    ) as exc:
                        detail = f"optimizer restart failed: {exc}"
                    else:
                        print(
                            f"[environment-monitor] optimization restarted pid={pid}",
                            flush=True,
                        )
                        return 0
                print(
                    f"[environment-monitor] recovery incomplete at {checked_at}: {detail}",
                    flush=True,
                )
                if once:
                    return 1
                _interruptible_sleep(state_dir, metadata["poll_interval"])
                continue
            detail = " ".join((result.stderr or result.stdout).split())[-1000:]
            print(
                f"[environment-monitor] still unavailable at {checked_at}: {detail}",
                flush=True,
            )
            if once:
                return 1
            _interruptible_sleep(state_dir, metadata["poll_interval"])
    except RecoveryStopRequested:
        error = _cancel_active_handoff(state_dir)
        if error is not None:
            print(
                f"[environment-monitor] rollback incomplete: {error}",
                file=sys.stderr,
                flush=True,
            )
            return 1
        print(
            "[environment-monitor] rollback request completed by active monitor",
            flush=True,
        )
        return 0
    finally:
        _remove_matching_pid(monitor_pid, os.getpid())
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()
        for handled_signal, previous_handler in previous_handlers.items():
            signal.signal(handled_signal, previous_handler)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--once", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no-restart", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--stop",
        action="store_true",
        help="stop recovery and its verified process tree before transport rollback",
    )
    args = parser.parse_args(argv)
    if args.stop:
        if args.once or args.no_restart:
            parser.error("--stop cannot be combined with --once or --no-restart")
        return stop_recovery(Path(args.state_dir))
    return run_monitor(
        Path(args.state_dir), once=args.once, no_restart=args.no_restart
    )


if __name__ == "__main__":
    raise SystemExit(main())
