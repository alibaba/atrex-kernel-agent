#!/usr/bin/env python3
"""Poll a blocked SSH GPU environment and restart its AKA optimization command."""

from __future__ import annotations

import argparse
import json
import os
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
    return command


def _archive_failure(state_dir: Path) -> None:
    failure = state_dir / "failure.json"
    if not failure.is_file():
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    failure.replace(state_dir / f"recovered-{stamp}.json")


def _restart(metadata: dict[str, Any], state_dir: Path) -> int:
    environment = os.environ.copy()
    environment["ATREX_ENVIRONMENT_STATE_FILE"] = metadata[
        "environment_state_file"
    ]
    environment["ATREX_ENVIRONMENT_RECOVERY_OWNER"] = "1"
    environment["ATREX_SANDBOX_SSH"] = metadata["ssh_target"]
    environment["ATREX_SANDBOX_SSH_INIT"] = str(metadata.get("ssh_init") or "")
    environment["ATREX_SANDBOX_HEALTH_COMMAND"] = metadata["health_command"]
    log_path = state_dir / "restart.log"
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            metadata["command"],
            cwd=metadata["cwd"],
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    (state_dir / "restart.pid").write_text(
        str(process.pid) + "\n", encoding="utf-8"
    )
    return process.pid


def run_monitor(state_dir: Path, *, once: bool = False, no_restart: bool = False) -> int:
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
                _archive_failure(state_dir)
                if not no_restart:
                    pid = _restart(metadata, state_dir)
                    print(
                        f"[environment-monitor] optimization restarted pid={pid}",
                        flush=True,
                    )
                return 0
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
