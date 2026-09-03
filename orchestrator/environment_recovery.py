from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


ENVIRONMENT_STATE_ENV = "ATREX_ENVIRONMENT_STATE_FILE"
RECOVERY_OWNER_ENV = "ATREX_ENVIRONMENT_RECOVERY_OWNER"
ENVIRONMENT_TEMPFAIL = 75
DEFAULT_SSH_HEALTH_COMMAND = (
    "python -c \"import torch; "
    "assert torch.cuda.is_available(); "
    "p=torch.cuda.get_device_properties(0); "
    "print(getattr(p, 'gcnArchName', '') or torch.cuda.get_device_capability(0))\""
)


class EnvironmentUnavailable(BaseException):
    """Stop orchestration immediately after a confirmed remote environment failure."""


@dataclass(frozen=True)
class RecoveryContext:
    directory: Path
    state_file: Path
    owner: bool


def environment_state_file() -> Path | None:
    value = os.environ.get(ENVIRONMENT_STATE_ENV, "").strip()
    return Path(value).expanduser().resolve() if value else None


def environment_is_blocked() -> bool:
    path = environment_state_file()
    if path is None:
        return False
    if path.is_file():
        return True
    return (
        os.environ.get("ATREX_ENVIRONMENT_RESTART_SUPERVISED") != "1"
        and path.with_name("restarting.json").is_file()
    )


def current_recovery_context() -> RecoveryContext | None:
    path = environment_state_file()
    if path is None:
        return None
    return RecoveryContext(
        directory=path.parent,
        state_file=path,
        owner=os.environ.get(RECOVERY_OWNER_ENV, "1") != "0",
    )


def raise_if_environment_blocked() -> None:
    if os.environ.pop("ATREX_ENVIRONMENT_RESTART_HANDOFF", "") == "1":
        deadline = time.monotonic() + 30
        while environment_is_blocked() and time.monotonic() < deadline:
            time.sleep(0.05)
    if environment_is_blocked():
        raise EnvironmentUnavailable("remote GPU environment is unavailable")


def _write_private_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.chmod(0o600)
    temporary.replace(path)


def signal_restart_ready() -> None:
    """Tell the monitor that campaign state is durable and execution can resume."""
    value = os.environ.pop("ATREX_ENVIRONMENT_RESTART_READY_FILE", "").strip()
    if not value:
        os.environ.pop("ATREX_ENVIRONMENT_RESTART_SUPERVISED", None)
        return
    path = Path(value).expanduser().resolve()
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(str(os.getpid()) + "\n", encoding="utf-8")
    temporary.chmod(0o600)
    temporary.replace(path)
    deadline = time.monotonic() + 30
    restarting = path.with_name("restarting.json")
    while restarting.is_file() and time.monotonic() < deadline:
        time.sleep(0.05)
    if restarting.is_file():
        raise EnvironmentUnavailable("recovery monitor did not complete restart handoff")
    os.environ.pop("ATREX_ENVIRONMENT_RESTART_SUPERVISED", None)


def configure_recovery(
    *,
    workspace_base: Path,
    raw_argv: Sequence[str],
    optimize_script: Path,
    sandbox_hardware: str,
    ssh_target: str,
    ssh_init: str,
    ssh_runtime_binds: Sequence[str],
    ssh_gpu: int,
    health_command: str,
    poll_interval: int,
) -> RecoveryContext:
    """Create durable restart metadata and export the shared failure marker path."""
    inherited = environment_state_file()
    owner = os.environ.get(RECOVERY_OWNER_ENV, "1") != "0"
    command = [
        str(Path(sys.executable).resolve()),
        str(optimize_script.resolve()),
        *raw_argv,
    ]
    stable_metadata = {
        "cwd": str(Path.cwd().resolve()),
        "command": command,
        "sandbox_hardware": sandbox_hardware,
        "ssh_target": ssh_target,
        "ssh_init": ssh_init,
        "ssh_runtime_binds": list(ssh_runtime_binds),
        "ssh_gpu": ssh_gpu,
        "health_command": health_command,
        "poll_interval": poll_interval,
    }
    if inherited is not None:
        directory = inherited.parent
    else:
        identity = json.dumps(
            {
                "cwd": stable_metadata["cwd"],
                "argv": list(raw_argv),
                "ssh": {
                    key: stable_metadata[key]
                    for key in (
                        "sandbox_hardware",
                        "ssh_target",
                        "ssh_init",
                        "ssh_runtime_binds",
                        "ssh_gpu",
                        "health_command",
                        "poll_interval",
                    )
                },
                # Separate simultaneous identical commands while a restarted child keeps
                # using the inherited state path from its supervising monitor.
                "invocation": uuid.uuid4().hex,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
        directory = workspace_base / ".atrex_environment" / digest
        inherited = directory / "failure.json"
        os.environ[ENVIRONMENT_STATE_ENV] = str(inherited)
    directory.mkdir(parents=True, exist_ok=True)
    try:
        directory.chmod(0o700)
    except OSError:
        pass

    os.environ["ATREX_SANDBOX_SSH"] = ssh_target
    os.environ["ATREX_SANDBOX_SSH_INIT"] = ssh_init
    os.environ["ATREX_SANDBOX_SSH_RUNTIME_BINDS"] = json.dumps(
        list(ssh_runtime_binds), separators=(",", ":")
    )
    os.environ["ATREX_SANDBOX_SSH_GPU"] = str(ssh_gpu)
    os.environ["ATREX_SANDBOX_HEALTH_COMMAND"] = health_command
    os.environ.pop("ATREX_SANDBOX_URL", None)
    os.environ.pop("ATREX_SANDBOX_PROFILE", None)
    os.environ[RECOVERY_OWNER_ENV] = "1" if owner else "0"

    restart_path = directory / "restart.json"
    if restart_path.is_file():
        try:
            existing = json.loads(restart_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"cannot validate active recovery metadata: {exc}") from exc
        if not isinstance(existing, dict):
            raise RuntimeError("cannot validate active recovery metadata: expected an object")
        mismatches = [
            key
            for key, expected in stable_metadata.items()
            if existing.get(key) != expected
        ]
        if existing.get("environment_state_file") != str(inherited):
            mismatches.append("environment_state_file")
        if mismatches:
            raise RuntimeError(
                "refusing to reuse recovery state with different resolved configuration: "
                + ", ".join(sorted(set(mismatches)))
            )
    if owner or not restart_path.is_file():
        _write_private_json(
            restart_path,
            {
                "schema_version": 2,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "environment_state_file": str(inherited),
                **stable_metadata,
            },
        )
        monitor = optimize_script.resolve().parent.parent / "tools" / "monitor_optimize_tasks.py"
        recover = directory / "recover.sh"
        recover.write_text(
            "#!/usr/bin/env bash\nset -euo pipefail\nexec "
            + " ".join(
                [
                    _shell_quote(str(Path(sys.executable).resolve())),
                    _shell_quote(str(monitor)),
                    "--state-dir",
                    _shell_quote(str(directory)),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        recover.chmod(0o700)
    return RecoveryContext(directory=directory, state_file=inherited, owner=owner)


def _shell_quote(value: str) -> str:
    import shlex

    return shlex.quote(value)


def launch_recovery_monitor(context: RecoveryContext) -> int | None:
    """Start one detached monitor. Its own lock resolves concurrent launch races."""
    if not context.owner or not context.state_file.is_file():
        return None
    monitor = Path(__file__).resolve().parents[1] / "tools" / "monitor_optimize_tasks.py"
    log_path = context.directory / "monitor.log"
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            [
                str(Path(sys.executable).resolve()),
                str(monitor),
                "--state-dir",
                str(context.directory),
            ],
            cwd=str(context.directory),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    (context.directory / "monitor.pid").write_text(
        str(process.pid) + "\n", encoding="utf-8"
    )
    return process.pid
