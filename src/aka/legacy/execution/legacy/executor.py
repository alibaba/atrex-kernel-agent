"""Main sandbox command construction and owned-process execution.

The launcher retains SSH/gateway transport, staging and remote recovery. This
module owns argv construction, queue/wall budgets and the host process lifetime;
it does not interpret evaluator output or classify candidate correctness.
"""
from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

from aka.contracts.execution import BackendCapabilities, CommandSpec, JobResult, ResourceIdentity
from .runtime import ExecutionHooks, SandboxRequest, current_transport_arguments


class LegacyExecutor:
    sandbox_cli_options = True
    sandbox_cli_endpoint_environment = False
    capabilities = BackendCapabilities(
        operations=frozenset(("auto", "run", "profile", "dev", "preflight")), cancellation=True,
    )
    resource_identity = ResourceIdentity("legacy")

    def __init__(self, request=None, hooks=None):
        self.request = request
        self.hooks = hooks

    def bind(self, context):
        from .runtime import default_binding
        request, hooks = default_binding(context)
        return self.bind_sandbox(request, hooks)

    def bind_sandbox(self, request, hooks):
        return type(self)(request, hooks)

    def run(self, command: CommandSpec) -> JobResult:
        self.capabilities.require(command)
        if self.request is None:
            from aka.contracts.execution import ExecutionContext
            return self.bind(ExecutionContext(command.environment, command.hardware, command.timeout)).run(command)
        request = replace(self.request, workspace=command.workspace, command=command.command,
            timeout=command.timeout, environment=command.environment, hardware=command.hardware,
            sync=command.artifacts,
            gateway_kind=self.request.gateway_kind if command.operation == "preflight" else command.operation,
            preflight=command.operation == "preflight", cancel_event=command.cancel_event)
        return self._run_sandbox(request, hooks=self.hooks)

    def _run_sandbox(self, request: SandboxRequest, *, hooks: ExecutionHooks) -> JobResult:
        workspace = request.workspace
        hardware = request.hardware
        profile = request.profile
        url = request.url
        timeout = request.timeout
        command = request.command
        ssh = request.ssh
        ssh_init = request.ssh_init
        health_command = request.health_command
        sync = request.sync
        wall_timeout = request.wall_timeout
        gateway_kind = request.gateway_kind
        preflight = request.preflight
        cancel_event = request.cancel_event
        if sum(bool(value) for value in (ssh, url, profile)) > 1:
            raise ValueError("ssh, url, and profile sandbox endpoints are mutually exclusive")
        if not request.launcher or any(not isinstance(item, str) or not item for item in request.launcher):
            raise ValueError("sandbox launcher must be a non-empty argument vector")
        cmd = [
            *request.launcher,
            "--kind",
            gateway_kind,
            "--hardware",
            hardware,
            "--workspace",
            str(workspace),
            "--timeout",
            str(timeout),
        ]
        if url:
            cmd += ["--url", url]
        elif profile:
            cmd += ["--gateway-profile", profile]
        elif ssh:
            cmd += ["--ssh", ssh]
        if ssh_init:
            cmd += ["--ssh-init", ssh_init]
        if health_command:
            cmd += ["--health-command", health_command]
        if sync:
            for path in sync:
                cmd += ["--sync", path]
        else:
            cmd.append("--no-sync")
        cmd += current_transport_arguments()
        if preflight:
            cmd.append("--preflight")
        else:
            cmd += ["--", *command]
        environment = dict(request.environment)
        package_root = str(Path(__file__).resolve().parents[4])
        environment["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(
            [package_root, *environment.get("PYTHONPATH", "").split(os.pathsep)]))
        if wall_timeout is not None:
            effective_timeout = wall_timeout
        else:
            uses_ssh = bool(ssh or (not profile and not url and environment.get("ATREX_SANDBOX_SSH")))
            grace = 0 if uses_ssh else hooks.queue_wait_grace(environment)
            effective_timeout = timeout + grace + 240
        from aka.legacy.execution.outcomes import run_owned
        return run_owned(cmd, environment=environment, workspace=workspace,
            timeout=effective_timeout, resource=self.resource_identity, hardware=hardware,
            cancel_event=cancel_event, spawn=hooks.spawn, check_environment=hooks.check_environment)
