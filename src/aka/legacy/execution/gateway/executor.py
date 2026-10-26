"""Own the shared gateway launcher and its synchronous cancellation lifetime."""
import os
from pathlib import Path
import sys
from dataclasses import replace
from aka.contracts.execution import BackendCapabilities, JobResult, ResourceIdentity
from aka.legacy.execution.configuration import queue_wait_grace


class GatewayExecutor:
    sandbox_cli_options = True
    sandbox_cli_endpoint_environment = True
    def __init__(self, configuration=None):
        self.configuration = dict(configuration or {})
        self.environment = os.environ.copy()
        if "requirements_file" in self.configuration:
            raise ValueError("worker requirements belong to Task, not gateway provider configuration")
    capabilities = BackendCapabilities(frozenset(("auto", "run", "dev", "profile", "check", "disassemble")), True, True)
    resource_identity = ResourceIdentity("gateway")

    def _environment_backend(self, environment=None):
        from aka.legacy.execution.environment_query import BackendEnvironment
        env = {**self.environment, **(environment or {})}
        return BackendEnvironment(backend=self.resource_identity.backend, operations=self.capabilities.operations,
            environment=env, url=self.configuration.get("url") or env.get("ATREX_SANDBOX_URL", ""),
            profile=self.configuration.get("profile") or env.get("ATREX_SANDBOX_PROFILE", ""))

    def query_environment(self, query):
        return self._environment_backend().query(query)

    def check_environment(self, query):
        return self._environment_backend().check(query)

    def run(self, command):
        self.capabilities.require(command)
        from aka.legacy.execution.environment_query import command_query, require_supported
        support = self._environment_backend(command.environment).check(command_query(command), discover=False)
        require_supported(support)
        if type(command.timeout) is not int or command.timeout <= 0:
            raise ValueError("execution timeout must be a positive integer")
        argv = [sys.executable, "-m", "aka.legacy.execution.gateway.entrypoint",
                "--workspace", str(command.workspace), "--hardware", command.hardware,
                "--kind", command.operation, "--timeout", str(command.timeout)]
        for artifact in command.artifacts:
            argv += ["--sync", artifact]
        if not command.artifacts:
            argv += ["--no-sync"]
        from aka.legacy.execution.legacy.runtime import current_transport_arguments
        argv += ["--num-gpus", str(command.num_gpus), *current_transport_arguments()]
        argv += ["--", *command.command]
        environment = dict(command.environment)
        from .requests import bound_limits, effective_budget
        environment = bound_limits(environment, support)
        if self.configuration.get("profile") or self.configuration.get("url"):
            for key in ("ATREX_SANDBOX_PROFILE", "ATREX_SANDBOX_URL"):
                environment.pop(key, None)
        unsupported = ("ATREX_SANDBOX_SSH", "ATREX_SANDBOX_SSH_INIT", "ATREX_SANDBOX_HEALTH_COMMAND",
                       "ATREX_SANDBOX_SSH_GPU")
        if any(environment.get(key) for key in unsupported):
            from aka.contracts.execution import UnsupportedExecution
            raise UnsupportedExecution("gateway profile does not support SSH or health configuration")
        # Source checkout launches inherit the loaded source root; installed launches
        # use normal module resolution. Do not require callers to mirror package paths.
        package_root = str(Path(__file__).resolve().parents[4])
        environment["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(
            [package_root, *environment.get("PYTHONPATH", "").split(os.pathsep)]))
        mapping = {"url":"ATREX_SANDBOX_URL", "profile":"ATREX_SANDBOX_PROFILE",
                   "warmup_iters":"AKA_GATEWAY_WARMUP_ITERS", "retry_delays":"AKA_GATEWAY_RETRY_DELAYS",
                   "dev_dependencies":"AKA_GATEWAY_DEV_DEPENDENCIES"}
        for key, env in mapping.items():
            if key in self.configuration:
                value = self.configuration[key]
                import json
                environment[env] = json.dumps(value) if isinstance(value, list) else str(value)

        grace = queue_wait_grace(environment)
        from aka.legacy.execution.outcomes import run_owned
        result = run_owned(argv, environment=environment, workspace=command.workspace,
            timeout=command.timeout + grace + 240, resource=self.resource_identity,
            hardware=command.hardware, cancel_event=command.cancel_event, input_payload=command.submission)
        return replace(result, environment_snapshot=support.environment,
                       environment_identity=support.environment.identity, environment_support=support,
                       budgets=(*result.budgets, effective_budget(command.timeout + grace + 240,
                           scope="local-wait", emit=False)))
