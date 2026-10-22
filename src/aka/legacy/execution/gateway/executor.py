"""Own the shared gateway launcher and its synchronous cancellation lifetime."""
import os
from pathlib import Path
import sys
from aka.contracts.execution import BackendCapabilities, JobResult, ResourceIdentity
from aka.legacy.execution.configuration import queue_wait_grace


class GatewayExecutor:
    def __init__(self, configuration=None):
        self.configuration = dict(configuration or {})
        if "requirements_file" in self.configuration:
            raise ValueError("worker requirements belong to Task, not gateway provider configuration")
    capabilities = BackendCapabilities(frozenset(("auto", "run", "dev", "profile")), True)
    resource_identity = ResourceIdentity("gateway")

    def run(self, command):
        self.capabilities.require(command)
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
        argv += current_transport_arguments()
        argv += ["--", *command.command]
        environment = dict(command.environment)
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
        return run_owned(argv, environment=environment, workspace=command.workspace,
            timeout=command.timeout + grace + 240, resource=self.resource_identity,
            hardware=command.hardware, cancel_event=command.cancel_event)
