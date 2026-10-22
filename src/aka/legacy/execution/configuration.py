"""Queue budget shared by the sandbox transport and its supervising process."""
from __future__ import annotations

from collections.abc import Mapping
import json

DEFAULT_QUEUE_WAIT_GRACE = 14_400


def queue_wait_grace(environment: Mapping[str, str]) -> int:
    name = "ATREX_SANDBOX_QUEUE_WAIT_GRACE"
    try:
        value = int(environment.get(name, str(DEFAULT_QUEUE_WAIT_GRACE)))
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def legacy_settings(environment=None):
    """Read connection and supervision settings inside Execution."""
    import os
    from types import SimpleNamespace
    e = os.environ if environment is None else environment
    raw_gpu = e.get("ATREX_SANDBOX_SSH_GPU", "")
    values = dict(profile=e.get("ATREX_SANDBOX_PROFILE", ""), url=e.get("ATREX_SANDBOX_URL", ""),
                  ssh=e.get("ATREX_SANDBOX_SSH", ""), ssh_init=e.get("ATREX_SANDBOX_SSH_INIT", ""),
                  ssh_gpu=int(raw_gpu) if raw_gpu else None,
                  health_command=e.get("ATREX_SANDBOX_HEALTH_COMMAND", ""))
    if sum(bool(values[k]) for k in ("profile", "url", "ssh")) > 1:
        raise ValueError("execution endpoints are mutually exclusive")
    return SimpleNamespace(**values)


def configure_legacy_arguments(values):
    """Translate retained CLI knobs once, before selection is frozen."""
    import os
    mapping = {"sandbox_profile": "ATREX_SANDBOX_PROFILE", "sandbox_url": "ATREX_SANDBOX_URL",
               "sandbox_ssh": "ATREX_SANDBOX_SSH", "sandbox_ssh_init": "ATREX_SANDBOX_SSH_INIT",
               "sandbox_ssh_gpu": "ATREX_SANDBOX_SSH_GPU", "sandbox_health_command": "ATREX_SANDBOX_HEALTH_COMMAND",
               "environment_poll_interval": "ATREX_ENVIRONMENT_POLL_INTERVAL"}
    for key, env in mapping.items():
        if key in values:
            value = values.pop(key)
            if value is not None:
                os.environ[env] = str(value)
    if "sandbox_ssh_runtime_bind" in values:
        os.environ["ATREX_SANDBOX_SSH_RUNTIME_BINDS"] = json.dumps(values.pop("sandbox_ssh_runtime_bind") or [])


def selected_ssh():
    """Expose the execution mode needed by immutable policy and recovery coordination."""
    from .selection import selected_execution
    return selected_execution().get("profile", "legacy") == "legacy" and bool(legacy_settings().ssh)
