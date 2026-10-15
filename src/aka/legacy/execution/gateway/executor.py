"""Own the shared gateway launcher and its synchronous cancellation lifetime."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from aka.contracts.execution import BackendCapabilities, JobResult, ResourceIdentity, ExecutionUnavailable
from aka.legacy.execution.configuration import queue_wait_grace


class GatewayExecutor:
    def __init__(self, configuration=None):
        self.configuration = dict(configuration or {})
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
        mapping = {"url":"ATREX_SANDBOX_URL", "profile":"ATREX_SANDBOX_PROFILE", "requirements_file":"ATREX_SANDBOX_REQUIREMENTS_FILE",
                   "warmup_iters":"AKA_GATEWAY_WARMUP_ITERS", "retry_delays":"AKA_GATEWAY_RETRY_DELAYS",
                   "dev_dependencies":"AKA_GATEWAY_DEV_DEPENDENCIES"}
        for key, env in mapping.items():
            if key in self.configuration:
                value = self.configuration[key]
                import json
                environment[env] = json.dumps(value) if isinstance(value, list) else str(value)

        grace = queue_wait_grace(environment)
        # Transport owns endpoint configuration in the selected profile/environment.
        process = subprocess.Popen(argv, cwd=command.workspace, env=environment,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
        deadline = time.monotonic() + command.timeout + grace + 240

        def stop():
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            try:
                return process.communicate(timeout=20)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                return process.communicate()

        try:
            while True:
                if command.cancel_event is not None and command.cancel_event.is_set():
                    raise subprocess.SubprocessError("gateway command cancelled")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    stdout, stderr = stop()
                    raise subprocess.TimeoutExpired(argv, command.timeout, output=stdout, stderr=stderr)
                try:
                    stdout, stderr = process.communicate(timeout=min(remaining, .25))
                    break
                except subprocess.TimeoutExpired:
                    continue
        except BaseException:
            stop()
            raise
        if process.returncode == 75 and "__ATREX_INFRASTRUCTURE_UNAVAILABLE__" in stderr.splitlines():
            raise ExecutionUnavailable("GPU validation transport unavailable")
        return JobResult(process.returncode, stdout, stderr, resource=self.resource_identity)
