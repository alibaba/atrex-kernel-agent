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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

RESTART_HANDOFF_TIMEOUT = 10_800


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


def _archive_restarting_marker(state_dir: Path) -> Path:
    restarting = state_dir / "restarting.json"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    archived = state_dir / f"recovered-{stamp}.json"
    restarting.replace(archived)
    return archived


def _terminate_handoff_process(state_dir: Path, pid: int) -> None:
    """Terminate only while the handoff lock proves the recorded child still exists."""
    if not _handoff_lock_active(state_dir):
        return
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 5
    while _handoff_lock_active(state_dir) and time.monotonic() < deadline:
        time.sleep(0.05)
    if not _handoff_lock_active(state_dir):
        return
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _reconcile_interrupted_handoff(state_dir: Path) -> int | None:
    """Adopt a live restart child or restore a marker left by a dead monitor."""
    restarting = state_dir / "restarting.json"
    if not restarting.is_file():
        return None
    if not _handoff_lock_active(state_dir):
        _restore_restarting_marker(state_dir)
        print(
            "[environment-monitor] restored interrupted restart marker with no live child",
            flush=True,
        )
        return None
    pid = _read_pid(state_dir / "restart.pid")
    if pid is None:
        raise RuntimeError("live restart handoff has no valid restart.pid")
    print(
        f"[environment-monitor] adopting interrupted restart handoff pid={pid}",
        flush=True,
    )
    ready = state_dir / "restart.ready"
    failure = state_dir / "failure.json"
    started_at = restarting.stat().st_mtime
    while True:
        active = _handoff_lock_active(state_dir)
        if ready.is_file() and active and not failure.is_file():
            _archive_restarting_marker(state_dir)
            ready.unlink(missing_ok=True)
            _remove_matching_pid(state_dir / "restart.pid", pid)
            return pid
        if not active:
            _restore_restarting_marker(state_dir)
            return None
        if failure.is_file() or time.time() - started_at >= RESTART_HANDOFF_TIMEOUT:
            _terminate_handoff_process(state_dir, pid)
            _restore_restarting_marker(state_dir)
            return None
        time.sleep(0.1)


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

    def stop_process() -> None:
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()

    try:
        (state_dir / "restart.pid").write_text(
            str(process.pid) + "\n", encoding="utf-8"
        )
    except OSError:
        stop_process()
        raise
    try:
        failure.replace(restarting)
    except OSError:
        stop_process()
        _remove_matching_pid(state_dir / "restart.pid", process.pid)
        raise

    # Baseline setup can legitimately use the configured two-hour session budget.
    # The monitor keeps ownership throughout and restores failure.json on any early exit.
    deadline = time.monotonic() + RESTART_HANDOFF_TIMEOUT
    try:
        while not ready.is_file():
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
            time.sleep(0.1)
        if failure.is_file():
            raise RuntimeError("remote environment failed again during restart")
    except BaseException:
        stop_process()
        _restore_restarting_marker(state_dir)
        raise
    _archive_restarting_marker(state_dir)
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


def run_monitor(
    state_dir: Path, *, once: bool = False, no_restart: bool = False
) -> int:
    state_dir = state_dir.expanduser().resolve()
    metadata = _load_restart(state_dir)
    lock = _acquire_lock(state_dir)
    if lock is None:
        print("[environment-monitor] another monitor is already active", flush=True)
        return 0
    monitor_pid = state_dir / "monitor.pid"
    try:
        monitor_pid.write_text(str(os.getpid()) + "\n", encoding="utf-8")
        adopted_pid = _reconcile_interrupted_handoff(state_dir)
        if adopted_pid is not None:
            print(
                f"[environment-monitor] optimization restart handoff adopted pid={adopted_pid}",
                flush=True,
            )
            return 0
        while True:
            checked_at = datetime.now(timezone.utc).isoformat()
            result = subprocess.run(
                _health_command(metadata),
                cwd=metadata["cwd"],
                capture_output=True,
                text=True,
            )
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
                time.sleep(metadata["poll_interval"])
                continue
            detail = " ".join((result.stderr or result.stdout).split())[-1000:]
            print(
                f"[environment-monitor] still unavailable at {checked_at}: {detail}",
                flush=True,
            )
            if once:
                return 1
            time.sleep(metadata["poll_interval"])
    finally:
        _remove_matching_pid(monitor_pid, os.getpid())
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--once", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no-restart", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    return run_monitor(
        Path(args.state_dir), once=args.once, no_restart=args.no_restart
    )


if __name__ == "__main__":
    raise SystemExit(main())
