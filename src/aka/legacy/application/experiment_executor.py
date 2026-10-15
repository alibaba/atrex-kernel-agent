"""Borrow a selected Executor with the original per-command sandbox host binding."""
from dataclasses import replace
from contextvars import ContextVar

from aka.contracts.execution import CommandSpec, JobResult, UnsupportedExecution
from aka.legacy.execution.legacy.executor import LegacyExecutor
from aka.legacy.execution.legacy.runtime import sandbox_invocation


class SandboxExecutor:
    def __init__(self, executor: LegacyExecutor, request, hooks):
        self.executor = executor
        self.request = request
        self.hooks = hooks
        self._completed_process = ContextVar("aka_experiment_completed_process", default=None)

    @property
    def completed_process(self):
        return self._completed_process.get()

    @property
    def capabilities(self):
        return self.executor.capabilities

    @property
    def resource_identity(self):
        return self.executor.resource_identity

    def run(self, command: CommandSpec) -> JobResult:
        self._completed_process.set(None)
        request = replace(
            self.request, workspace=command.workspace, command=command.command,
            timeout=command.timeout,
            environment={**self.request.environment, **command.environment},
            hardware=command.hardware or self.request.hardware,
            sync=command.artifacts,
            gateway_kind=self.request.gateway_kind if command.operation == "preflight" else command.operation,
            preflight=command.operation == "preflight", cancel_event=command.cancel_event,
        )
        if not isinstance(self.executor, LegacyExecutor):
            raise UnsupportedExecution("sandbox host binding requires the legacy execution provider")
        bound = replace(command, environment=request.environment, hardware=request.hardware)
        with sandbox_invocation(request, self.hooks) as invocation:
            result = self.executor.run(bound)
            original = invocation.result
            if original is not None and isinstance(result, JobResult) and (
                original.returncode, original.stdout, original.stderr
            ) == (result.transport_returncode, result.stdout, result.stderr):
                self._completed_process.set(original)
        if not isinstance(result, JobResult):
            raise TypeError("Executor.run must return JobResult")
        return result


def sandbox_executor(executor: LegacyExecutor, *, hardware="", profile="", url="",
                     ssh="", ssh_init="", health_command="", wall_timeout=None,
                     private_reference_dir=None) -> SandboxExecutor:
    """Construct the host dependency; Experiment only sees Executor.run."""
    import os
    import sys
    from pathlib import Path

    from aka.legacy.resources.paths import SANDBOX_TOOL
    from aka.legacy.execution.recovery import raise_if_environment_blocked
    from aka.legacy.execution.processes import spawn_owned_session
    from aka.legacy.execution.configuration import queue_wait_grace
    from aka.legacy.execution.legacy.runtime import ExecutionHooks, SandboxRequest
    from .execution import execution_environment

    environment = os.environ.copy()
    environment.pop("ATREX_PRIVATE_REFERENCE_DIR", None)
    if private_reference_dir is not None:
        environment["ATREX_PRIVATE_REFERENCE_DIR"] = str(private_reference_dir)
    request = SandboxRequest(
        workspace=Path("."), launcher=(sys.executable, str(SANDBOX_TOOL)),
        hardware=hardware, profile=profile, url=url, timeout=600, command=(),
        environment=execution_environment(environment), ssh=ssh, ssh_init=ssh_init,
        health_command=health_command, wall_timeout=wall_timeout,
    )
    hooks = ExecutionHooks(spawn_owned_session, raise_if_environment_blocked, queue_wait_grace)
    return SandboxExecutor(executor, request, hooks)
