"""Queue budget shared by the sandbox transport and its supervising process."""
from __future__ import annotations

from collections.abc import Mapping
import json

DEFAULT_QUEUE_WAIT_GRACE = 14_400

_WORKLOAD_RESERVED_KEYS = frozenset(("ATREX_PRIVATE_REFERENCE_DIR", "ATREX_EVAL_SHAPE_BATCH_SIZE"))
_WORKLOAD_RESERVED_PREFIXES = ("AGATE_", "ATREX_SANDBOX_", "AKA_GATEWAY_", "AKA_EXECUTION_", "AKA_EXPERIMENT_COMMAND_", "AKA_AGENT_TOOL_")


def transport_environment_key(key):
    """Reserve host transport/configuration names from explicit workload values."""
    return key in _WORKLOAD_RESERVED_KEYS or key.startswith(_WORKLOAD_RESERVED_PREFIXES)


def validate_workload_environment(environment):
    import re
    if (not isinstance(environment, Mapping) or any(not isinstance(key, str) or not isinstance(value, str)
            for key, value in environment.items())):
        raise TypeError("command environment requires string keys and values")
    for key in environment:
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) is None:
            raise ValueError(f"invalid --env key {key!r}")
    if any(transport_environment_key(key) for key in environment):
        raise ValueError("transport configuration cannot be supplied as workload environment")


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


def selected_ssh():
    """Expose the execution mode needed by immutable policy and recovery coordination."""
    from .selection import selected_execution
    return selected_execution().get("profile", "legacy") == "legacy" and bool(legacy_settings().ssh)
