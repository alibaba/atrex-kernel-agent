"""Main sandbox command construction and owned-process execution.

The launcher retains SSH/gateway transport, staging and remote recovery. This
module owns argv construction, queue/wall budgets and the host process lifetime;
it does not interpret evaluator output or classify candidate correctness.
"""
from __future__ import annotations

import os
import signal
import subprocess
import time
from dataclasses import replace
from pathlib import Path

from aka.contracts.execution import BackendCapabilities, CommandSpec, JobResult, ResourceIdentity
from .runtime import ExecutionHooks, SandboxRequest, current_invocation, current_transport_arguments


class LegacyExecutor:
    capabilities = BackendCapabilities(
        operations=frozenset(("auto", "run", "profile", "dev", "preflight")), cancellation=True,
    )
    resource_identity = ResourceIdentity("legacy")

    def run(self, command: CommandSpec) -> JobResult:
        self.capabilities.require(command)
        invocation = current_invocation()
        request = replace(invocation.request, workspace=command.workspace, command=command.command,
            timeout=command.timeout, environment=command.environment, hardware=command.hardware,
            sync=command.artifacts,
            gateway_kind=invocation.request.gateway_kind if command.operation == "preflight" else command.operation,
            preflight=command.operation == "preflight", cancel_event=command.cancel_event)
        result = self._run_sandbox(request, hooks=invocation.hooks)
        invocation.result = result
        return JobResult(result.returncode, result.stdout, result.stderr, resource=self.resource_identity)

    def _run_sandbox(self, request: SandboxRequest, *, hooks: ExecutionHooks) -> subprocess.CompletedProcess[str]:
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
        cmd = [
            request.launcher[0], "-m", "aka.legacy.execution.gateway.sandbox",
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
        hooks.check_environment()
        process = hooks.spawn(
            cmd,
            role="sandbox",
            environment=environment,
            cwd=str(workspace),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        def stop_process_group() -> tuple[str, str]:
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

        deadline = time.monotonic() + effective_timeout
        try:
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    raise subprocess.SubprocessError("sandbox command cancelled")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    stdout, stderr = stop_process_group()
                    raise subprocess.TimeoutExpired(
                        cmd, effective_timeout, output=stdout, stderr=stderr
                    )
                try:
                    stdout, stderr = process.communicate(timeout=min(0.25, remaining))
                except subprocess.TimeoutExpired:
                    hooks.check_environment()
                    continue
                break
        except BaseException:
            try:
                stop_process_group()
            except BaseException:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            raise
        result = subprocess.CompletedProcess(
            args=cmd,
            returncode=process.returncode,
            stdout=stdout,
            stderr=stderr,
        )
        hooks.check_environment()
        return result
