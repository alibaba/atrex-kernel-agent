"""Borrow an explicitly bound selected Executor for a host operation."""
from dataclasses import replace

from aka.contracts.execution import CommandSpec, JobResult, Executor, ExecutionBinder, ExecutionContext


class SandboxExecutor:
    def __init__(self, executor: Executor, context: ExecutionContext):
        self.context = context
        self.executor = executor.bind(context) if isinstance(executor, ExecutionBinder) else executor

    @property
    def capabilities(self):
        return self.executor.capabilities

    @property
    def resource_identity(self):
        return self.executor.resource_identity

    def run(self, command: CommandSpec) -> JobResult:
        environment = {**self.context.environment, **command.environment}
        for key in ("ATREX_SANDBOX_REQUIREMENTS_FILE", "ATREX_PRIVATE_REFERENCE_DIR"):
            if key in self.context.environment:
                environment[key] = self.context.environment[key]
        bound = replace(command, environment=environment, hardware=command.hardware or self.context.hardware)
        self.executor.capabilities.require(bound)
        result = self.executor.run(bound)
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
    import os
    from .selection import execution_environment
    environment = os.environ.copy()
    for key in ("ATREX_PRIVATE_REFERENCE_DIR", "ATREX_SANDBOX_REQUIREMENTS_FILE", "AKA_GATEWAY_DEV_DEPENDENCIES"):
        environment.pop(key, None)
    if worker_requirements is not None:
        environment.update(task_environment(worker_requirements))
    if private_reference_dir is not None:
        environment["ATREX_PRIVATE_REFERENCE_DIR"] = str(private_reference_dir)
    return SandboxExecutor(executor, ExecutionContext(
        execution_environment(environment), hardware, timeout, formal))
