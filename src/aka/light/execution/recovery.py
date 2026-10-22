"""Shared campaign recovery, including durable module-selection continuation.

Legacy owns restart metadata, selection/digest validation and the recovery monitor.
Re-export its exception too so WorkflowStartup handles Light outages directly.
Light's physical Job checkpoints/idempotency and validation retry policy remain in
gateway_jobs.py and retry.py; Agent process ownership remains in agent/.
"""
from aka.legacy.execution.recovery import (
    DEFAULT_SSH_HEALTH_COMMAND,
    ENVIRONMENT_STATE_ENV,
    ENVIRONMENT_TEMPFAIL,
    RECOVERY_OWNER_ENV,
    EnvironmentUnavailable,
    RecoveryContext,
    configure_recovery,
    current_recovery_context,
    environment_is_blocked,
    environment_state_file,
    launch_recovery_monitor,
    raise_if_environment_blocked,
    signal_restart_ready,
)

__all__ = [
    "DEFAULT_SSH_HEALTH_COMMAND", "ENVIRONMENT_STATE_ENV", "ENVIRONMENT_TEMPFAIL",
    "RECOVERY_OWNER_ENV", "EnvironmentUnavailable", "RecoveryContext",
    "configure_recovery", "current_recovery_context", "environment_is_blocked",
    "environment_state_file", "launch_recovery_monitor", "raise_if_environment_blocked",
    "signal_restart_ready",
]
