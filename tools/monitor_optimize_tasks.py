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
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from orchestrator.durable_state import (  # noqa: E402
    durable_replace,
    durable_rmdir,
    durable_unlink,
    durable_write_json,
    durable_write_text,
    ensure_private_directory,
)
from orchestrator.recovery_processes import (  # noqa: E402
    ACTIVE_MARKER,
    HANDOFF_ID_ENV,
    HANDOFF_LOCK_FD_ENV,
    RESTART_COMPLETE,
    RESTART_EXIT,
    RESTART_PRIMARY_PID,
    STOP_MARKER,
    clear_process_registry,
    matching_processes,
    process_identity_matches,
    spawn_owned_session,
    terminate_processes,
)

RESTART_HANDOFF_TIMEOUT = 10_800
RESTART_CLEANUP_TIMEOUT = 30
STOP_WAIT_TIMEOUT = 120
HANDOFF_MARKER_KEY = "restart_handoff"
STOP_REQUEST_DIRECTORY = "stop-requests"
_STOP_SIGNALLED = False


class RecoveryStopRequested(BaseException):
    """A verified rollback request interrupted monitoring or restart."""


class RecoveryCleanupUnverified(RuntimeError):
    """Cleanup ownership disappeared before completion was committed."""


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
    if not isinstance(value.get("runtime_health_command", ""), str):
        raise RuntimeError("restart metadata has invalid runtime_health_command")
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
    durable_unlink(path, missing_ok=True)


def _write_private_json(path: Path, value: object) -> None:
    durable_write_json(path, value, indent=2, ensure_ascii=False)


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
    durable_replace(failure, restarting)


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
        or bool(_stop_request_paths(state_dir))
    )


def _stop_request_paths(state_dir: Path) -> tuple[Path, ...]:
    directory = state_dir / STOP_REQUEST_DIRECTORY
    if not directory.is_dir():
        return ()
    return tuple(sorted(directory.glob("*.json")))


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
            durable_replace(
                marker, state_dir / f"superseded-{marker.stem}-{stamp}.json"
            )
        else:
            durable_replace(marker, failure)
    for name in (
        "restart.ready",
        "restart.ack",
        "restart.pid",
        RESTART_PRIMARY_PID,
        RESTART_EXIT,
        RESTART_COMPLETE,
    ):
        durable_unlink(state_dir / name, missing_ok=True)


def _restore_restarting_marker(state_dir: Path) -> None:
    _restore_handoff_marker(state_dir, state_dir / "restarting.json")


def _activate_restarting_marker(state_dir: Path) -> Path:
    restarting = state_dir / "restarting.json"
    active = state_dir / ACTIVE_MARKER
    if active.exists():
        raise RuntimeError(f"active recovery marker already exists: {active}")
    durable_replace(restarting, active)
    return active


def _archive_marker(marker: Path, prefix: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    archived = marker.with_name(f"{prefix}-{stamp}.json")
    durable_replace(marker, archived)
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
    if not result.complete:
        detail = "; ".join(result.errors) or (
            "owned processes remain: "
            + ", ".join(str(value) for value in result.remaining_pids)
        )
        raise RuntimeError(f"cannot terminate recovery process tree: {detail}")
    lock_deadline = time.monotonic() + 5.0
    while _handoff_lock_active(state_dir) and time.monotonic() < lock_deadline:
        time.sleep(0.05)
    if _handoff_lock_active(state_dir):
        raise RuntimeError(
            "cannot terminate recovery process tree: handoff lock remains owned by "
            "an unregistered descendant"
        )


def _protocol_event(
    path: Path,
    handoff_id: str,
    *,
    primary_pid: int | None = None,
    require_returncode: bool = False,
) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"cannot read recovery protocol event {path}: {exc}"
        ) from exc
    event_primary = value.get("primary_pid") if isinstance(value, dict) else None
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != 1
        or value.get("handoff_id") != handoff_id
        or not isinstance(event_primary, int)
        or isinstance(event_primary, bool)
        or event_primary <= 0
        or (primary_pid is not None and event_primary != primary_pid)
    ):
        raise RuntimeError(f"invalid recovery protocol event: {path}")
    if require_returncode:
        returncode = value.get("returncode")
        if not isinstance(returncode, int) or isinstance(returncode, bool):
            raise RuntimeError(f"invalid recovery return code event: {path}")
    return value


def _root_processes(
    state_dir: Path, handoff_id: str
) -> tuple[int | None, int | None, bool, bool, tuple[Any, ...]]:
    wrapper_pid = _read_pid(state_dir / "restart.pid")
    primary_pid = _read_pid(state_dir / RESTART_PRIMARY_PID)
    owned = matching_processes(state_dir, handoff_id)
    wrapper_alive = wrapper_pid is not None and any(
        record.pid == wrapper_pid and record.owner_kind == "session-wrapper"
        for record in owned
    )
    primary_alive = (
        wrapper_pid is not None
        and primary_pid is not None
        and any(
            record.pid == primary_pid
            and record.owner_kind == "session-primary"
            and record.owner_pid == wrapper_pid
            for record in owned
        )
    )
    return wrapper_pid, primary_pid, wrapper_alive, primary_alive, owned


def _transition_handoff_to_failure(
    state_dir: Path,
    marker: Path,
    handoff_id: str,
    *,
    preserve_protocol: bool = False,
) -> None:
    failure = state_dir / "failure.json"
    if marker.is_file():
        if failure.is_file():
            _archive_marker(marker, f"superseded-{marker.stem}")
        else:
            durable_replace(marker, failure)
    if preserve_protocol:
        return
    clear_process_registry(state_dir, handoff_id)
    for name in (
        "restart.ready",
        "restart.ack",
        "restart.pid",
        RESTART_PRIMARY_PID,
        RESTART_EXIT,
        RESTART_COMPLETE,
    ):
        durable_unlink(state_dir / name, missing_ok=True)


def _event_age_seconds(event: dict[str, Any]) -> float:
    recorded_at = event.get("recorded_at")
    if not isinstance(recorded_at, str):
        raise RuntimeError("recovery protocol event has no recorded_at timestamp")
    try:
        recorded = datetime.fromisoformat(recorded_at)
    except ValueError as exc:
        raise RuntimeError("recovery protocol event has invalid recorded_at") from exc
    if recorded.tzinfo is None:
        raise RuntimeError("recovery protocol event timestamp has no timezone")
    return max(0.0, time.time() - recorded.timestamp())


def _raise_for_preserved_incomplete_cleanup(state_dir: Path) -> None:
    failure = state_dir / "failure.json"
    metadata = _handoff_metadata(failure) if failure.is_file() else None
    if metadata is None:
        return
    handoff_id, _started_at = metadata
    exit_event = _protocol_event(
        state_dir / RESTART_EXIT, handoff_id, require_returncode=True
    )
    complete_event = _protocol_event(
        state_dir / RESTART_COMPLETE, handoff_id, require_returncode=True
    )
    if exit_event is not None and complete_event is None:
        raise RecoveryCleanupUnverified(
            "optimizer exit is durable but cleanup completion is missing; "
            "manual process verification is required"
        )
    if complete_event is not None and (
        exit_event is None
        or complete_event["returncode"] != exit_event["returncode"]
    ):
        raise RecoveryCleanupUnverified(
            "optimizer cleanup protocol is inconsistent; manual verification "
            "is required"
        )


def _await_restart_ack(
    state_dir: Path,
    marker: Path,
    handoff_id: str,
    started_at: float,
) -> int:
    ready = state_dir / "restart.ready"
    ack = state_dir / "restart.ack"
    failure = state_dir / "failure.json"
    activated = marker.name == ACTIVE_MARKER
    while True:
        _raise_if_stop_requested(state_dir)
        wrapper_pid, primary_pid, wrapper_alive, primary_alive, owned = _root_processes(
            state_dir, handoff_id
        )
        exit_event = _protocol_event(
            state_dir / RESTART_EXIT,
            handoff_id,
            primary_pid=primary_pid,
            require_returncode=True,
        )
        ready_event = _protocol_event(ready, handoff_id, primary_pid=primary_pid)
        if ready_event is not None and primary_alive:
            if not activated:
                marker = _activate_restarting_marker(state_dir)
                activated = True
        ack_event = (
            _protocol_event(ack, handoff_id, primary_pid=primary_pid)
            if activated and ready_event is not None
            else None
        )
        if ack_event is not None and wrapper_pid is not None:
            # Revalidate after observing the acknowledgement. This prevents a
            # ready/exit race from being committed without the child's second
            # phase. Once ack is durable, a simultaneous exit belongs to active
            # supervision and its real return code/completion protocol.
            if exit_event is not None or (
                primary_alive
                and process_identity_matches(state_dir, handoff_id, primary_pid)
            ):
                return wrapper_pid
        if exit_event is not None:
            raise RuntimeError(
                "optimizer exited before acknowledging active recovery "
                f"with status {exit_event['returncode']}"
            )
        if failure.is_file():
            raise RuntimeError("remote environment failed again during restart")
        if not _handoff_lock_active(state_dir) or not wrapper_alive:
            raise RuntimeError(
                "optimizer ownership ended before restart acknowledgement"
            )
        # The wrapper persists restart.exit.json immediately after wait() reports
        # the primary's death. Let that durable event win over a transient liveness
        # observation so callers receive the real status instead of a generic race.
        if time.time() - started_at >= RESTART_HANDOFF_TIMEOUT:
            raise RuntimeError(
                "optimizer did not acknowledge active recovery within 10800 seconds"
            )
        if not owned:
            raise RuntimeError("optimizer ownership disappeared during restart")
        _interruptible_sleep(state_dir, 0.1)


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
    try:
        return _await_restart_ack(state_dir, restarting, handoff_id, started_at)
    except (OSError, RuntimeError):
        if _handoff_lock_active(state_dir) or matching_processes(state_dir, handoff_id):
            _terminate_handoff_processes(state_dir, handoff_id)
        marker = (
            state_dir / ACTIVE_MARKER
            if (state_dir / ACTIVE_MARKER).is_file()
            else restarting
        )
        _transition_handoff_to_failure(state_dir, marker, handoff_id)
        return None


def _reconcile_active_run(state_dir: Path) -> tuple[bool, int | None]:
    """Supervise an active optimizer until durable completion or retry."""
    active_marker = state_dir / ACTIVE_MARKER
    if not active_marker.is_file():
        _raise_for_preserved_incomplete_cleanup(state_dir)
        return True, None
    metadata = _handoff_metadata(active_marker)
    if metadata is None:
        raise RuntimeError(f"active recovery marker is invalid: {active_marker}")
    handoff_id, started_at = metadata
    failure = state_dir / "failure.json"
    cleanup_deadline: float | None = None
    while True:
        _raise_if_stop_requested(state_dir)
        _wrapper_pid, primary_pid, wrapper_alive, primary_alive, owned = _root_processes(
            state_dir, handoff_id
        )
        lock_active = _handoff_lock_active(state_dir)
        if failure.is_file():
            if lock_active or owned:
                _terminate_handoff_processes(state_dir, handoff_id)
            _transition_handoff_to_failure(state_dir, active_marker, handoff_id)
            return True, None

        exit_event = _protocol_event(
            state_dir / RESTART_EXIT,
            handoff_id,
            primary_pid=primary_pid,
            require_returncode=True,
        )
        complete_event = _protocol_event(
            state_dir / RESTART_COMPLETE,
            handoff_id,
            primary_pid=primary_pid,
            require_returncode=True,
        )
        if complete_event is not None:
            if exit_event is None:
                raise RuntimeError(
                    "cleanup completion exists without a matching optimizer exit"
                )
            if complete_event["returncode"] != exit_event["returncode"]:
                raise RuntimeError(
                    "optimizer exit and cleanup completion return codes differ"
                )
            if lock_active and not owned:
                raise RuntimeError(
                    "completed recovery still has an unregistered handoff-lock owner"
                )
            if lock_active or owned:
                if cleanup_deadline is None:
                    remaining = max(
                        0.0,
                        RESTART_CLEANUP_TIMEOUT - _event_age_seconds(complete_event),
                    )
                    cleanup_deadline = time.monotonic() + remaining
                if time.monotonic() >= cleanup_deadline:
                    _terminate_handoff_processes(state_dir, handoff_id)
                _interruptible_sleep(state_dir, 0.05)
                continue
            returncode = complete_event["returncode"]
            if returncode == 0:
                if active_marker.is_file():
                    _archive_marker(active_marker, "recovered")
                clear_process_registry(state_dir, handoff_id)
                for name in (
                    "restart.ready",
                    "restart.ack",
                    "restart.pid",
                    RESTART_PRIMARY_PID,
                    RESTART_EXIT,
                    RESTART_COMPLETE,
                ):
                    durable_unlink(state_dir / name, missing_ok=True)
                return False, None
            _transition_handoff_to_failure(state_dir, active_marker, handoff_id)
            return True, None

        if exit_event is not None:
            # An exit code, including a non-zero code, says nothing about cleanup.
            # Keep a bounded window for a live wrapper/guardian to commit proof.
            if not lock_active and not owned:
                _transition_handoff_to_failure(
                    state_dir,
                    active_marker,
                    handoff_id,
                    preserve_protocol=True,
                )
                raise RecoveryCleanupUnverified(
                    "optimizer ownership disappeared before cleanup completion"
                )
            if cleanup_deadline is None:
                remaining = max(
                    0.0,
                    RESTART_CLEANUP_TIMEOUT - _event_age_seconds(exit_event),
                )
                cleanup_deadline = time.monotonic() + remaining
            if time.monotonic() >= cleanup_deadline:
                _terminate_handoff_processes(state_dir, handoff_id)
            _interruptible_sleep(state_dir, 0.05)
            continue

        if not wrapper_alive and primary_alive:
            _terminate_handoff_processes(state_dir, handoff_id)
            _transition_handoff_to_failure(state_dir, active_marker, handoff_id)
            return True, None
        if not owned or not lock_active or not wrapper_alive or not primary_alive:
            if lock_active or owned:
                _terminate_handoff_processes(state_dir, handoff_id)
            _transition_handoff_to_failure(state_dir, active_marker, handoff_id)
            return True, None

        ack_event = _protocol_event(
            state_dir / "restart.ack", handoff_id, primary_pid=primary_pid
        )
        if ack_event is None and time.time() - started_at >= RESTART_HANDOFF_TIMEOUT:
            _terminate_handoff_processes(state_dir, handoff_id)
            _transition_handoff_to_failure(state_dir, active_marker, handoff_id)
            return True, None
        _interruptible_sleep(state_dir, 0.1)


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
        "--runtime-health-command",
        metadata.get("runtime_health_command", ""),
        "--check-health",
    ]
    ssh_init = metadata.get("ssh_init")
    if isinstance(ssh_init, str) and ssh_init:
        command += ["--ssh-init", ssh_init]
    for runtime_bind in metadata.get("ssh_runtime_binds", []):
        command += ["--ssh-runtime-bind", runtime_bind]
    return command


def _terminate_pollable_process(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + 2.0
    while process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.05)
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait()


def _run_health_probe(
    metadata: dict[str, Any], state_dir: Path
) -> subprocess.CompletedProcess[str]:
    command = _health_command(metadata)
    with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as capture:
        process = subprocess.Popen(
            command,
            cwd=metadata["cwd"],
            stdin=subprocess.DEVNULL,
            stdout=capture,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
            close_fds=True,
        )
        try:
            while process.poll() is None:
                _raise_if_stop_requested(state_dir)
                time.sleep(0.05)
        except BaseException:
            _terminate_pollable_process(process)
            raise
        capture.flush()
        capture.seek(0)
        output = capture.read()
    return subprocess.CompletedProcess(command, process.returncode, output, "")


def _archive_failure(state_dir: Path) -> Path | None:
    failure = state_dir / "failure.json"
    if not failure.is_file():
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    archived = state_dir / f"recovered-{stamp}.json"
    durable_replace(failure, archived)
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
    environment["ATREX_SANDBOX_RUNTIME_HEALTH_COMMAND"] = metadata.get(
        "runtime_health_command", ""
    )
    environment["ATREX_ENVIRONMENT_POLL_INTERVAL"] = str(metadata["poll_interval"])
    environment.pop("ATREX_SANDBOX_URL", None)
    environment.pop("ATREX_SANDBOX_PROFILE", None)
    environment["ATREX_ENVIRONMENT_RESTART_HANDOFF"] = "1"
    environment["ATREX_ENVIRONMENT_RESTART_SUPERVISED"] = "1"
    environment[HANDOFF_ID_ENV] = handoff_id
    environment[HANDOFF_LOCK_FD_ENV] = str(handoff_lock)
    ready = state_dir / "restart.ready"
    for name in (
        "restart.ready",
        "restart.ack",
        RESTART_PRIMARY_PID,
        RESTART_EXIT,
        RESTART_COMPLETE,
    ):
        durable_unlink(state_dir / name, missing_ok=True)
    environment["ATREX_ENVIRONMENT_RESTART_READY_FILE"] = str(ready)
    log_path = state_dir / "restart.log"

    def persist_handoff(process: subprocess.Popen[Any]) -> None:
        durable_write_text(state_dir / "restart.pid", str(process.pid) + "\n")
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
        durable_unlink(state_dir / "restart.pid", missing_ok=True)
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

    try:
        return _await_restart_ack(state_dir, restarting, handoff_id, started_at)
    except BaseException:
        # Marker restoration is allowed only after every registered identity and
        # process group has been verified gone. A cleanup failure therefore leaves
        # restarting.json in place and recovery fail-closed.
        stop_process()
        marker = (
            state_dir / ACTIVE_MARKER
            if (state_dir / ACTIVE_MARKER).is_file()
            else restarting
        )
        _transition_handoff_to_failure(state_dir, marker, handoff_id)
        raise


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
        durable_unlink(path)
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
        exit_event = _protocol_event(
            state_dir / RESTART_EXIT, handoff_id, require_returncode=True
        )
        complete_event = _protocol_event(
            state_dir / RESTART_COMPLETE, handoff_id, require_returncode=True
        )
        if exit_event is not None and complete_event is None:
            return (
                "optimizer exit is durable but cleanup completion is missing; "
                "refusing to report rollback complete"
            )
        if complete_event is not None and (
            exit_event is None
            or complete_event["returncode"] != exit_event["returncode"]
        ):
            return "optimizer cleanup protocol is inconsistent"
        clear_process_registry(state_dir, handoff_id)
    elif active:
        return "handoff lock is active but no durable process identity is available"

    if active_marker.is_file():
        _restore_handoff_marker(state_dir, active_marker)
    elif restarting.is_file():
        _restore_handoff_marker(state_dir, restarting)
    else:
        for name in (
            "restart.ready",
            "restart.ack",
            "restart.pid",
            RESTART_PRIMARY_PID,
            RESTART_EXIT,
            RESTART_COMPLETE,
        ):
            durable_unlink(state_dir / name, missing_ok=True)
    return None


def _ensure_stop_request(path: Path, request_id: str) -> None:
    if path.is_file():
        return
    ensure_private_directory(path.parent)
    _write_private_json(
        path,
        {
            "schema_version": 1,
            "request_id": request_id,
            "requested_at": datetime.now(timezone.utc).isoformat(),
            "requester_pid": os.getpid(),
        },
    )


def _acquire_resume_lock(state_dir: Path) -> TextIO | None:
    lock = _acquire_lock(state_dir)
    if lock is not None or not _stop_requested(state_dir):
        return lock
    deadline = time.monotonic() + STOP_WAIT_TIMEOUT
    while time.monotonic() < deadline:
        time.sleep(0.1)
        lock = _acquire_lock(state_dir)
        if lock is not None:
            return lock
    return None


def stop_recovery(state_dir: Path) -> int:
    """Request verified rollback and wait until the monitor and process tree stop."""
    state_dir = state_dir.expanduser().resolve()
    ensure_private_directory(state_dir)
    stopped = state_dir / STOP_MARKER
    request_id = uuid.uuid4().hex
    request_path = state_dir / STOP_REQUEST_DIRECTORY / f"{request_id}.json"
    _ensure_stop_request(request_path, request_id)
    deadline = time.monotonic() + STOP_WAIT_TIMEOUT
    while time.monotonic() < deadline:
        # A concurrent resume only clears the immutable request names in its lock
        # snapshot. Recreate our unique request while this stopper is still live,
        # so it cannot be lost before we commit stopped.json under the lock.
        _ensure_stop_request(request_path, request_id)
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
                _write_private_json(
                    stopped,
                    {
                        "schema_version": 2,
                        "request_id": request_id,
                        "requested_at": datetime.now(timezone.utc).isoformat(),
                        "requester_pid": os.getpid(),
                    },
                )
                durable_unlink(state_dir / "monitor.pid", missing_ok=True)
                print(
                    "[environment-monitor] rollback stop completed; recovery "
                    "remains disabled by stopped.json",
                    flush=True,
                )
                return 0
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
                lock.close()
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
    lock = _acquire_resume_lock(state_dir) if resume else _acquire_lock(state_dir)
    if lock is None:
        if resume and _stop_requested(state_dir):
            print(
                "[environment-monitor] resume timed out before stop state could "
                "be cleared under the monitor lock",
                file=sys.stderr,
                flush=True,
            )
            return 1
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
            request_snapshot = _stop_request_paths(state_dir)
            durable_unlink(state_dir / STOP_MARKER, missing_ok=True)
            durable_unlink(state_dir / "stop.request", missing_ok=True)
            for request_path in request_snapshot:
                durable_unlink(request_path, missing_ok=True)
            try:
                durable_rmdir(state_dir / STOP_REQUEST_DIRECTORY)
            except OSError:
                pass
            # A stop created after the snapshot wins and is observed before any
            # health check or optimizer process is launched.
            _raise_if_stop_requested(state_dir)
        elif _stop_requested(state_dir):
            print(
                "[environment-monitor] recovery is disabled by stopped.json; "
                "run recover.sh or pass --resume to re-enable it",
                flush=True,
            )
            return 0

        metadata = _load_restart(state_dir)
        durable_write_text(monitor_pid, str(os.getpid()) + "\n")
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
            continue_recovery, _active_pid = _reconcile_active_run(state_dir)
            if not continue_recovery:
                return 0
        if not (state_dir / "failure.json").is_file():
            print(
                "[environment-monitor] no blocked environment remains to recover",
                flush=True,
            )
            return 0
        while True:
            _raise_if_stop_requested(state_dir)
            # A restarted older campaign may have upgraded its preflight
            # metadata before encountering an incompatible evaluator runtime.
            metadata = _load_restart(state_dir)
            checked_at = datetime.now(timezone.utc).isoformat()
            result = _run_health_probe(metadata, state_dir)
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
                        continue_recovery, _active_pid = _reconcile_active_run(
                            state_dir
                        )
                        if not continue_recovery:
                            return 0
                        detail = (
                            "recovered optimizer ended unexpectedly; returning "
                            "to environment health polling"
                        )
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
    except RecoveryCleanupUnverified as exc:
        print(
            f"[environment-monitor] recovery stopped fail-closed: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return 1
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
