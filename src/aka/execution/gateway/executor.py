"""Own the shared gateway launcher and its synchronous cancellation lifetime."""
import os
from pathlib import Path
import sys
from dataclasses import replace
from aka.contracts.execution import BackendCapabilities, JobResult, ResourceIdentity
from aka.execution.configuration import queue_wait_grace
from aka.execution.job_recovery import GatewayRecoveryMixin, recovery_policy


class GatewayExecutor(GatewayRecoveryMixin):
    sandbox_cli_options = True
    sandbox_cli_endpoint_environment = True
    def __init__(self, configuration=None):
        self.configuration = dict(configuration or {})
        self.environment = os.environ.copy()
        recovery_policy(self.configuration)
        if "requirements_file" in self.configuration:
            raise ValueError("worker requirements belong to Task, not gateway provider configuration")
    capabilities = BackendCapabilities(frozenset(("auto", "run", "dev", "profile", "check", "disassemble")), True, True,
                                       input_files=True, output_options=True, prepared_inputs=True)
    resource_identity = ResourceIdentity("gateway")

    def bind(self, context):
        from aka.execution.filesystem_scope import require_scope_backend
        require_scope_backend(context, self.capabilities)
        executor = type(self)(self.configuration)
        executor.environment = dict(context.environment)
        executor._bound_context = True
        return executor

    def _environment_backend(self, environment=None):
        from aka.execution.environment_query import BackendEnvironment
        env = {**self.environment, **(environment or {})}
        url = self.configuration.get("url") or env.get("ATREX_SANDBOX_URL", "")
        profile = self.configuration.get("profile") or env.get("ATREX_SANDBOX_PROFILE", "")
        if self.configuration.get("url"):
            profile = ""
        elif self.configuration.get("profile"):
            url = ""
        return BackendEnvironment(backend=self.resource_identity.backend, operations=self.capabilities.operations,
            environment=env, url=url, profile=profile)

    def query_environment(self, query):
        return self._environment_backend().query(query)

    def check_environment(self, query):
        return self._environment_backend().check(query)

    def run(self, command):
        from aka.execution.command_transport import prepared_input_workspace
        self.capabilities.require(command)
        with prepared_input_workspace(command) as physical:
            return self._run(physical)

    def _run(self, command):
        self.capabilities.require(command)
        unsupported = ("ATREX_SANDBOX_SSH", "ATREX_SANDBOX_SSH_INIT", "ATREX_SANDBOX_HEALTH_COMMAND",
                       "ATREX_SANDBOX_SSH_GPU")
        if any(command.environment.get(key) for key in unsupported):
            from aka.contracts.execution import UnsupportedExecution
            raise UnsupportedExecution("gateway profile does not support SSH or health configuration")
        if not getattr(self, "_bound_context", False):
            from aka.contracts.execution import ExecutionContext
            bound = self.bind(ExecutionContext(command.environment, command.hardware, command.timeout,
                filesystem_access=command.filesystem_access))
            try:
                return bound._run(command)
            finally:
                if hasattr(bound, "_selected_job_transport"):
                    self._selected_job_transport = bound._selected_job_transport
        from aka.execution.environment_query import command_query, require_supported
        support = self._environment_backend(command.environment).check(command_query(command), discover=False)
        require_supported(support)
        if type(command.timeout) is not int or command.timeout <= 0:
            raise ValueError("execution timeout must be a positive integer")
        argv = [sys.executable, "-m", "aka.execution.gateway.entrypoint",
                "--workspace", str(command.workspace), "--hardware", command.hardware,
                "--kind", command.operation, "--timeout", str(command.timeout)]
        for artifact in command.artifacts:
            argv += ["--sync", artifact]
        if not command.artifacts:
            argv += ["--no-sync"]
        from aka.execution.legacy.runtime import current_transport_arguments
        from aka.execution.command_transport import sandbox_options
        argv += ["--num-gpus", str(command.num_gpus), *current_transport_arguments(), *sandbox_options(command)]
        argv += ["--", *command.command]
        environment = dict(command.environment)
        transport = self._job_transport()
        environment.update({key: value for key, value in transport.environment.items() if key.startswith("AGATE_")})
        import json
        environment["AKA_EXECUTION_GATEWAY_ROUTE"] = json.dumps({"endpoint": transport.endpoint, "profile": transport.profile})
        if transport.endpoint:
            environment["AGATE_URL"] = transport.endpoint
        from .requests import bound_limits, effective_budget
        environment = bound_limits(environment, support)
        if self.configuration.get("profile") or self.configuration.get("url"):
            for key in ("ATREX_SANDBOX_PROFILE", "ATREX_SANDBOX_URL"):
                environment.pop(key, None)
        # Source checkout launches inherit the loaded source root; installed launches
        # use normal module resolution. Do not require callers to mirror package paths.
        package_root = str(Path(__file__).resolve().parents[3])
        environment["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(
            [package_root, *environment.get("PYTHONPATH", "").split(os.pathsep)]))
        mapping = {"url":"ATREX_SANDBOX_URL", "profile":"ATREX_SANDBOX_PROFILE",
                   "warmup_iters":"AKA_GATEWAY_WARMUP_ITERS", "retry_delays":"AKA_GATEWAY_RETRY_DELAYS",
                   "dev_dependencies":"AKA_GATEWAY_DEV_DEPENDENCIES"}
        mapping.update({"recovery_retry_delays":"AKA_GATEWAY_RECOVERY_RETRY_DELAYS",
                        "recovery_retry_budget_s":"AKA_GATEWAY_RECOVERY_RETRY_BUDGET_S"})
        for key, env in mapping.items():
            if key in self.configuration:
                value = self.configuration[key]
                import json
                environment[env] = json.dumps(value) if isinstance(value, list) else str(value)
        if command.requirements:
            import json
            environment["AKA_GATEWAY_DEV_DEPENDENCIES"] = json.dumps(command.requirements)

        grace = queue_wait_grace(environment)
        from aka.execution.outcomes import run_owned
        from aka.execution.prepared_inputs import payload as prepared_payload
        def decorate(result):
            return replace(result, environment_snapshot=support.environment,
                environment_identity=support.environment.identity, environment_support=support,
                budgets=(*result.budgets, effective_budget(command.timeout + grace + 240,
                    scope="local-wait", emit=False)))
        def observed(result):
            if command.on_observation is not None:
                command.on_observation(decorate(result))
        try:
            result = run_owned(argv, environment=environment, workspace=command.workspace,
                timeout=command.timeout + grace + 240, resource=self.resource_identity,
                hardware=command.hardware, cancel_event=command.cancel_event, input_payload=command.submission,
                prepared_input_payload=prepared_payload(command.prepared_inputs, command.input_files),
                on_observation=observed if command.on_observation is not None else None)
        except BaseException as error:
            if isinstance(getattr(error, "execution_outcome", None), JobResult):
                error.execution_outcome = decorate(error.execution_outcome)
            raise
        return decorate(result)
