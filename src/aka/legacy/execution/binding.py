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

    @property
    def recovery_capabilities(self):
        from aka.contracts.execution_recovery import JobRecoveryCapabilities
        return getattr(self.executor, "recovery_capabilities", JobRecoveryCapabilities())

    def _recovery(self, method, *args, **kwargs):
        from aka.contracts.execution import UnsupportedExecution
        operation = getattr(self.executor, method, None)
        if operation is None:
            raise UnsupportedExecution("selected executor does not implement job recovery")
        return operation(*args, **kwargs)

    def query_job(self, reference, *, timeout, retry=None):
        arguments = {"timeout": timeout}
        if retry is not None:
            arguments["retry"] = retry
        return self._recovery("query_job", reference, **arguments)

    def cancel_job(self, reference, *, timeout):
        return self._recovery("cancel_job", reference, timeout=timeout)

    def replay_job(self, command, *, idempotency_key):
        bound = replace(command, environment={**self.context.environment, **command.environment},
                        hardware=command.hardware or self.context.hardware)
        return self._recovery("replay_job", bound, idempotency_key=idempotency_key)

    def query_environment(self, query):
        method = getattr(self.executor, "query_environment", None)
        if method is not None:
            return method(replace(query, hardware=query.hardware or self.context.hardware))
        from aka.contracts.execution_environment import EnvironmentSnapshot
        resource = self.resource_identity
        return EnvironmentSnapshot(resource.backend if resource else "custom", None,
            query.hardware or self.context.hardware, None,
            {"operations": sorted(self.capabilities.operations)}, diagnostics=("environment-query-not-supported",))

    def check_environment(self, query):
        method = getattr(self.executor, "check_environment", None)
        if method is not None:
            return method(replace(query, hardware=query.hardware or self.context.hardware))
        from .environment_query import match_environment
        return match_environment(query, self.query_environment(query))

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
