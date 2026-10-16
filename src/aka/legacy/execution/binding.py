"""Borrow a selected Executor with the original per-command sandbox host binding."""
from dataclasses import replace
from contextvars import ContextVar

from aka.contracts.execution import CommandSpec, JobResult, Executor, UnsupportedExecution
from aka.legacy.execution.legacy.executor import LegacyExecutor
from aka.legacy.execution.legacy.runtime import sandbox_invocation


class SandboxExecutor:
    def __init__(self, executor: Executor, request, hooks):
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
            environment={**self.request.environment, **command.environment,
                         **{key: self.request.environment[key] for key in
                            ("ATREX_SANDBOX_REQUIREMENTS_FILE", "ATREX_PRIVATE_REFERENCE_DIR")
                            if key in self.request.environment}},
            hardware=command.hardware or self.request.hardware,
            sync=command.artifacts,
            gateway_kind=self.request.gateway_kind if command.operation == "preflight" else command.operation,
            preflight=command.operation == "preflight", cancel_event=command.cancel_event,
        )
        if not isinstance(self.executor, LegacyExecutor):
            if any((request.profile, request.url, request.ssh, request.ssh_init, request.health_command,
                    request.wall_timeout is not None)):
                raise UnsupportedExecution("selected executor owns endpoint configuration; omit legacy endpoint options")
            bound = replace(command, environment=request.environment, hardware=request.hardware)
            self.executor.capabilities.require(bound)
            result = self.executor.run(bound)
            if not isinstance(result, JobResult):
                raise TypeError("Executor.run must return JobResult")
            return result
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


def task_environment(worker_requirements):
    """Translate an explicit Task dependency file into the transport environment."""
    if worker_requirements is None:
        raise ValueError("Task worker requirements are missing")
    if worker_requirements.source is None:
        raise ValueError("Task worker requirements must be frozen before execution")
    path = worker_requirements.source
    if not path.is_file():
        raise FileNotFoundError(f"missing Task worker requirements: {path}")
    return {"ATREX_SANDBOX_REQUIREMENTS_FILE": str(path), "AKA_GATEWAY_DEV_DEPENDENCIES": "[]"}


def sandbox_executor(executor: Executor, *, hardware="", timeout=600, formal=False,
                     private_reference_dir=None, worker_requirements=None) -> SandboxExecutor:
    """Construct the host dependency; Experiment only sees Executor.run."""
    import os
    import sys
    from pathlib import Path

    from aka.legacy.resources.paths import SANDBOX_TOOL
    from aka.legacy.execution.recovery import raise_if_environment_blocked
    from aka.legacy.execution.processes import spawn_owned_session
    from aka.legacy.execution.configuration import queue_wait_grace
    from aka.legacy.execution.legacy.runtime import ExecutionHooks, SandboxRequest
    from .selection import execution_environment

    environment = os.environ.copy()
    environment.pop("ATREX_PRIVATE_REFERENCE_DIR", None)
    environment.pop("ATREX_SANDBOX_REQUIREMENTS_FILE", None)
    environment.pop("AKA_GATEWAY_DEV_DEPENDENCIES", None)
    if worker_requirements is not None:
        environment.update(task_environment(worker_requirements))
    if private_reference_dir is not None:
        environment["ATREX_PRIVATE_REFERENCE_DIR"] = str(private_reference_dir)
    from .configuration import legacy_settings
    settings = legacy_settings() if isinstance(executor, LegacyExecutor) else None
    request = SandboxRequest(
        workspace=Path("."), launcher=(sys.executable, str(SANDBOX_TOOL)),
        hardware=hardware, profile=settings.profile if settings else "",
        url=settings.url if settings else "", timeout=timeout, command=(),
        environment=execution_environment(environment), ssh=settings.ssh if settings else "",
        ssh_init=settings.ssh_init if settings else "", health_command=settings.health_command if settings else "",
        wall_timeout=timeout + 14_400 + 120 if formal and settings else None,
    )
    hooks = ExecutionHooks(spawn_owned_session, raise_if_environment_blocked, queue_wait_grace)
    return SandboxExecutor(executor, request, hooks)
