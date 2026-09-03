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
    ACTIVE_MARKER,
    HANDOFF_ID_ENV,
    HANDOFF_LOCK_FD_ENV,
    STOP_MARKER,
    clear_process_registry,
    matching_processes,
    process_identity_matches,
    spawn_owned_session,
    terminate_processes,
)

RESTART_HANDOFF_TIMEOUT = 10_800
STOP_WAIT_TIMEOUT = 120
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


def _ensure_handoff_metadata(restarting: Path) -> tuple[str, float]:
    metadata = _handoff_metadata(restarting)
    if metadata is None:
        # Migrate a handoff created by the previous release. Starting the timeout
        # at migration is conservative and avoids treating an old outage as an old
        # optimizer initialization.
        metadata = uuid.uuid4().hex, time.time()
        _write_handoff_metadata(restarting, *metadata)
    return metadata


def _stop_requested(state_dir: Path) -> bool:
    return (
        _STOP_SIGNALLED
        or (state_dir / STOP_MARKER).is_file()
        or (state_dir / "stop.request").is_file()
    )


def _raise_if_stop_requested(state_dir: Path) -> None:
    if _stop_requested(state_dir):
        raise RecoveryStopRequested


def _interruptible_sleep(state_dir: Path, seconds: float) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        _raise_if_stop_requested(state_dir)
        time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))


def _restore_handoff_marker(state_dir: Path, marker: Path) -> None:
    failure = state_dir / "failure.json"
    if marker.is_file():
        if failure.is_file():
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            marker.replace(state_dir / f"superseded-{marker.stem}-{stamp}.json")
        else:
            marker.replace(failure)
    for name in ("restart.ready", "restart.pid"):
        try:
            (state_dir / name).unlink()
        except FileNotFoundError:
            pass


def _restore_restarting_marker(state_dir: Path) -> None:
    _restore_handoff_marker(state_dir, state_dir / "restarting.json")


def _activate_restarting_marker(state_dir: Path) -> Path:
    restarting = state_dir / "restarting.json"
    active = state_dir / ACTIVE_MARKER
    if active.exists():
        raise RuntimeError(f"active recovery marker already exists: {active}")
    restarting.replace(active)
    return active


def _archive_marker(marker: Path, prefix: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    archived = marker.with_name(f"{prefix}-{stamp}.json")
    marker.replace(archived)
    return archived


def _terminate_handoff_processes(
    state_dir: Path,
    handoff_id: str,
) -> None:
    """Terminate every identity-owned session before permitting another restart."""
    result = terminate_processes(
        state_dir,
        handoff_id,
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
    handoff_id, started_at = _ensure_handoff_metadata(restarting)
    active = _handoff_lock_active(state_dir)
    root_alive = pid is not None and process_identity_matches(
        state_dir, handoff_id, pid
    )
    if not active or not root_alive:
        owned = matching_processes(state_dir, handoff_id)
        if owned or active:
            _terminate_handoff_processes(state_dir, handoff_id)
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
        active = _handoff_lock_active(state_dir)
        root_alive = process_identity_matches(state_dir, handoff_id, pid)
        if ready.is_file() and not failure.is_file():
            _activate_restarting_marker(state_dir)
            ready.unlink(missing_ok=True)
            return pid
        if not active or not root_alive:
            _terminate_handoff_processes(state_dir, handoff_id)
            _restore_restarting_marker(state_dir)
            clear_process_registry(state_dir, handoff_id)
            return None
        if failure.is_file() or time.time() - started_at >= RESTART_HANDOFF_TIMEOUT:
            _terminate_handoff_processes(state_dir, handoff_id)
            _restore_restarting_marker(state_dir)
            clear_process_registry(state_dir, handoff_id)
            return None
        _interruptible_sleep(state_dir, 0.1)


def _reconcile_active_run(state_dir: Path) -> tuple[bool, int | None]:
    """Reconcile a ready optimizer before allowing another recovery cycle."""
    active_marker = state_dir / ACTIVE_MARKER
    if not active_marker.is_file():
        return True, None
    metadata = _handoff_metadata(active_marker)
    if metadata is None:
        raise RuntimeError(f"active recovery marker is invalid: {active_marker}")
    handoff_id, _started_at = metadata
    pid = _read_pid(state_dir / "restart.pid")
    lock_active = _handoff_lock_active(state_dir)
    owned = matching_processes(state_dir, handoff_id)
    root_alive = pid is not None and any(record.pid == pid for record in owned)
    failure = state_dir / "failure.json"

    if failure.is_file():
        if lock_active or owned:
            _terminate_handoff_processes(state_dir, handoff_id)
        if active_marker.is_file():
            _archive_marker(active_marker, "superseded-active")
        clear_process_registry(state_dir, handoff_id)
        if pid is not None:
            _remove_matching_pid(state_dir / "restart.pid", pid)
        return True, None

    if lock_active and root_alive:
        return False, pid
    if lock_active or owned:
        _terminate_handoff_processes(state_dir, handoff_id)
    if active_marker.is_file():
        _archive_marker(active_marker, "recovered")
    clear_process_registry(state_dir, handoff_id)
    if pid is not None:
        _remove_matching_pid(state_dir / "restart.pid", pid)
    return False, None


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

    def persist_handoff(process: subprocess.Popen[Any]) -> None:
        (state_dir / "restart.pid").write_text(
            str(process.pid) + "\n", encoding="utf-8"
        )
        _begin_handoff(failure, restarting, handoff_id, started_at)

    try:
        with log_path.open("a", encoding="utf-8") as log:
            process = spawn_owned_session(
                command,
                role="optimizer-root",
                environment=environment,
                finalize_handoff=True,
                registered_callback=persist_handoff,
                cwd=metadata["cwd"],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
    except BaseException:
        _restore_restarting_marker(state_dir)
        (state_dir / "restart.pid").unlink(missing_ok=True)
        clear_process_registry(state_dir, handoff_id)
        raise
    finally:
        # Popen duplicated this locked descriptor into the optimizer. Closing the
        # monitor's copy makes lock ownership exactly track the child lifetime.
        os.close(handoff_lock)

    def stop_process() -> None:
        _terminate_handoff_processes(state_dir, handoff_id)
        try:
            process.wait(timeout=0)
        except subprocess.TimeoutExpired:
            pass

    # Baseline setup can legitimately use the configured two-hour session budget.
    # The monitor keeps ownership throughout and restores failure.json on any early exit.
    deadline = time.monotonic() + RESTART_HANDOFF_TIMEOUT
    try:
        while not ready.is_file():
            _raise_if_stop_requested(state_dir)
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
            _interruptible_sleep(state_dir, 0.1)
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
    _activate_restarting_marker(state_dir)
    ready.unlink(missing_ok=True)
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
    active_marker = state_dir / ACTIVE_MARKER
    failure = state_dir / "failure.json"
    marker = (
        active_marker
        if active_marker.is_file()
        else restarting
        if restarting.is_file()
        else failure
    )
    metadata = _handoff_metadata(marker) if marker.is_file() else None
    if restarting.is_file() and metadata is None:
        metadata = _ensure_handoff_metadata(restarting)
    if metadata is None:
        registered_ids = _registry_handoff_ids(state_dir)
        if len(registered_ids) == 1:
            metadata = registered_ids[0], time.time()
        elif len(registered_ids) > 1:
            return "multiple recovery handoff registries require manual diagnosis"

    active = _handoff_lock_active(state_dir)
    if metadata is not None:
        handoff_id, _started_at = metadata
        if active or matching_processes(state_dir, handoff_id):
            try:
                _terminate_handoff_processes(state_dir, handoff_id)
            except RuntimeError as exc:
                return str(exc)
        clear_process_registry(state_dir, handoff_id)
    elif active:
        return "handoff lock is active but no durable process identity is available"

    if active_marker.is_file():
        _restore_handoff_marker(state_dir, active_marker)
    elif restarting.is_file():
        _restore_handoff_marker(state_dir, restarting)
    else:
        for name in ("restart.ready", "restart.pid"):
            (state_dir / name).unlink(missing_ok=True)
    return None


def stop_recovery(state_dir: Path) -> int:
    """Request verified rollback and wait until the monitor and process tree stop."""
    state_dir = state_dir.expanduser().resolve()
    state_dir.mkdir(parents=True, exist_ok=True)
    try:
        state_dir.chmod(0o700)
    except OSError:
        pass
    stopped = state_dir / STOP_MARKER
    _write_private_json(
        stopped,
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
                    if error == (
                        "handoff lock is active but no durable process identity "
                        "is available"
                    ):
                        time.sleep(0.1)
                        continue
                    print(
                        f"[environment-monitor] rollback incomplete: {error}",
                        file=sys.stderr,
                        flush=True,
                    )
                    return 1
                (state_dir / "monitor.pid").unlink(missing_ok=True)
                print(
                    "[environment-monitor] rollback stop completed; recovery "
                    "remains disabled by stopped.json",
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
    state_dir: Path,
    *,
    once: bool = False,
    no_restart: bool = False,
    resume: bool = False,
) -> int:
    global _STOP_SIGNALLED

    state_dir = state_dir.expanduser().resolve()
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
        if resume:
            (state_dir / STOP_MARKER).unlink(missing_ok=True)
            (state_dir / "stop.request").unlink(missing_ok=True)
        elif _stop_requested(state_dir):
            print(
                "[environment-monitor] recovery is disabled by stopped.json; "
                "run recover.sh or pass --resume to re-enable it",
                flush=True,
            )
            return 0

        metadata = _load_restart(state_dir)
        monitor_pid.write_text(str(os.getpid()) + "\n", encoding="utf-8")
        _raise_if_stop_requested(state_dir)
        continue_recovery, active_pid = _reconcile_active_run(state_dir)
        if not continue_recovery:
            if active_pid is not None:
                print(
                    f"[environment-monitor] recovered optimizer is already active "
                    f"pid={active_pid}",
                    flush=True,
                )
            return 0
        adopted_pid = _reconcile_interrupted_handoff(state_dir)
        if adopted_pid is not None:
            print(
                f"[environment-monitor] optimization restart handoff adopted pid={adopted_pid}",
                flush=True,
            )
            return 0
        if not (state_dir / "failure.json").is_file():
            print(
                "[environment-monitor] no blocked environment remains to recover",
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
        "--resume",
        action="store_true",
        help="clear a persistent recovery stop while holding the monitor lock",
    )
    parser.add_argument(
        "--stop",
        action="store_true",
        help="stop recovery and its verified process tree before transport rollback",
    )
    args = parser.parse_args(argv)
    if args.stop:
        if args.once or args.no_restart or args.resume:
            parser.error(
                "--stop cannot be combined with --once, --no-restart, or --resume"
            )
        return stop_recovery(Path(args.state_dir))
    return run_monitor(
        Path(args.state_dir),
        once=args.once,
        no_restart=args.no_restart,
        resume=args.resume,
    )


if __name__ == "__main__":
    raise SystemExit(main())
