"""Execution-owned environment blocking facts consumed by host recovery."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

ENVIRONMENT_STATE_ENV = "ATREX_ENVIRONMENT_STATE_FILE"
RECOVERY_OWNER_ENV = "ATREX_ENVIRONMENT_RECOVERY_OWNER"
ENVIRONMENT_TEMPFAIL = 75


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
