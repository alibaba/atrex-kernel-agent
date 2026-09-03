#!/usr/bin/env python3
"""Poll a blocked SSH GPU environment and restart its AKA optimization command."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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
    return value


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _acquire_lock(state_dir: Path) -> Path | None:
    lock = state_dir / "monitor.lock"
    for _attempt in range(2):
        try:
            descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            try:
                pid = int(lock.read_text(encoding="utf-8").strip())
            except (OSError, ValueError):
                pid = -1
            if pid > 0 and _pid_is_alive(pid):
                return None
            try:
                lock.unlink()
            except FileNotFoundError:
                pass
            continue
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(str(os.getpid()) + "\n")
        return lock
    return None


def _health_command(metadata: dict[str, Any]) -> list[str]:
    sandbox = Path(__file__).resolve().parent / "sandbox.py"
    command = [
        str(Path(sys.executable).resolve()),
        str(sandbox),
        "--hardware",
        metadata["sandbox_hardware"],
        "--ssh",
        metadata["ssh_target"],
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
    """Spawn the real optimizer while preserving the marker through Popen failures."""
    command = metadata["command"]
    if len(command) < 2 or not Path(command[1]).is_file():
        missing = command[1] if len(command) > 1 else ""
        raise FileNotFoundError(f"optimizer script is missing: {missing}")
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
    environment["ATREX_SANDBOX_HEALTH_COMMAND"] = metadata["health_command"]
    environment.pop("ATREX_SANDBOX_URL", None)
    environment.pop("ATREX_SANDBOX_PROFILE", None)
    environment["ATREX_ENVIRONMENT_RESTART_HANDOFF"] = "1"
    ready = state_dir / "restart.ready"
    try:
        ready.unlink()
    except FileNotFoundError:
        pass
    environment["ATREX_ENVIRONMENT_RESTART_READY_FILE"] = str(ready)
    log_path = state_dir / "restart.log"
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
        )
    try:
        (state_dir / "restart.pid").write_text(
            str(process.pid) + "\n", encoding="utf-8"
        )
    except OSError:
        process.terminate()
        raise
    deadline = time.monotonic() + 30
    while not ready.is_file():
        if process.poll() is not None:
            try:
                (state_dir / "restart.pid").unlink()
            except FileNotFoundError:
                pass
            raise RuntimeError(
                f"optimizer exited during restart handoff with status {process.returncode}"
            )
        if time.monotonic() >= deadline:
            process.terminate()
            try:
                (state_dir / "restart.pid").unlink()
            except FileNotFoundError:
                pass
            raise RuntimeError("optimizer did not initialize recovery within 30 seconds")
        time.sleep(0.05)
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
    try:
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
                    pid: int | None = None
                    try:
                        pid = _restart(metadata, state_dir)
                        if _archive_failure(state_dir) is None:
                            raise RuntimeError(
                                "blocked marker disappeared before restart"
                            )
                        try:
                            (state_dir / "restart.ready").unlink()
                        except FileNotFoundError:
                            pass
                    except (
                        OSError,
                        RuntimeError,
                        subprocess.SubprocessError,
                    ) as exc:
                        if pid is not None:
                            try:
                                os.kill(pid, 15)
                            except ProcessLookupError:
                                pass
                            try:
                                (state_dir / "restart.pid").unlink()
                            except FileNotFoundError:
                                pass
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
        try:
            lock.unlink()
        except FileNotFoundError:
            pass


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
