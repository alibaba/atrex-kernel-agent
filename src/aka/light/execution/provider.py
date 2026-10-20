"""Use the hardened eval route and checkpoint/retry policy at the Execution seam."""
from dataclasses import replace
from pathlib import Path
import os
import sys

from aka.contracts.execution import BackendCapabilities, ResourceIdentity
from aka.legacy.execution.legacy.executor import LegacyExecutor


class ReliableExecutor(LegacyExecutor):
    def run_gateway(self, command, cwd, environment, control):
        from .process import run_gateway
        return run_gateway(command, cwd, environment, control)

    resource_identity = ResourceIdentity("agate-recorded-transport")
    capabilities = BackendCapabilities(
        operations=frozenset({"auto", "run", "profile", "dev", "preflight", "check", "disassemble", "env"}),
        cancellation=True,
    )

    def _run_sandbox(self, request, *, hooks):
        from aka.light.runtime.paths import supervisor_campaign_root
        from aka.light.supervisor.measurement_records import JOB_ROOT_ENV
        from aka.light.orchestrator.agent_home import open_private_directory
        from aka.legacy.application.paths import checkout_root
        evidence = supervisor_campaign_root(request.workspace) / "execution"
        with open_private_directory(evidence):
            pass
        environment = dict(request.environment)
        environment["AKA_LEGACY_REPO_ROOT"] = str(checkout_root())
        environment[JOB_ROOT_ENV] = str(evidence)
        source_root = str(Path(__file__).resolve().parents[3])
        environment["PYTHONPATH"] = os.pathsep.join(filter(None, (source_root, environment.get("PYTHONPATH"))))
        return super()._run_sandbox(replace(request,
            launcher=(sys.executable, "-m", "aka.light.supervisor.gateway"),
            environment=environment), hooks=hooks)
