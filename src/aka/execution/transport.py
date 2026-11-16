"""Execution transport implementation and callable interface."""
from __future__ import annotations

import subprocess


INFRASTRUCTURE_MARKER = "__ATREX_INFRASTRUCTURE_UNAVAILABLE__"


INFRASTRUCTURE_EXIT_CODE = 75


from aka.contracts.execution import ExecutionUnavailable

InfrastructureUnavailable = ExecutionUnavailable


def check_transport(process: subprocess.CompletedProcess) -> None:
    """Only classify transport failures; never infer an outage from a kernel failure."""
    if (process.returncode == INFRASTRUCTURE_EXIT_CODE
            and INFRASTRUCTURE_MARKER in (process.stderr or "").splitlines()):
        raise InfrastructureUnavailable("GPU validation transport unavailable")
