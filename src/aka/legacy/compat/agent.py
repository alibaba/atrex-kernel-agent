"""Translate historical session flags and integer budgets to the Agent Interface."""

from dataclasses import dataclass
from collections.abc import Mapping
from pathlib import Path
from aka.contracts.agent import *  # retained historical exports
from aka.contracts.agent import (
    AgentRunRequest as InvocationRequest,
    AgentRunResult as InvocationResult,
)


@dataclass(frozen=True)
class AgentRunRequest:
    workspace: Path
    prompt: str
    timeout_s: int
    reasoning_effort: str = "max"
    sandbox_hardware: str = ""
    sandbox_profile: str = ""
    sandbox_url: str = ""
    sandbox_timeout_s: int = 600
    session_id: str | None = None
    extra_environment: Mapping[str, str] | None = None
    sandbox_ssh: str = ""
    sandbox_ssh_init: str = ""
    sandbox_health_command: str = ""
    environment_state_file: str = ""


def invocation_request(request):
    if isinstance(request, InvocationRequest):
        return request
    from aka.legacy.execution.environment import session_environment
    environment = session_environment(
        hardware=request.sandbox_hardware, profile=request.sandbox_profile,
        url=request.sandbox_url, timeout=request.sandbox_timeout_s,
        ssh=request.sandbox_ssh, ssh_init=request.sandbox_ssh_init,
        health_command=request.sandbox_health_command,
        state_file=request.environment_state_file, extra=request.extra_environment,
    )
    return InvocationRequest(
        request.workspace,
        request.prompt,
        request.timeout_s,
        request.reasoning_effort,
        request.session_id,
        environment,
    )


class RuntimeAdapter:
    def __init__(self, runtime):
        self._runtime = runtime

    @property
    def id(self):
        return self._runtime.id

    def run(self, request):
        return ResultAdapter(self._runtime.run(invocation_request(request)))

    def __getattr__(self, name):
        return getattr(self._runtime, name)


@dataclass(frozen=True)
class AgentRunResult(InvocationResult):
    @property
    def tokens(self):
        return self.terminal_usage.total_tokens or 0


def ResultAdapter(result):
    from dataclasses import fields

    return AgentRunResult(
        **{f.name: getattr(result, f.name) for f in fields(InvocationResult)}
    )


def build_agent_runtime(*args, **kwargs):
    from aka.legacy.agent.runtime import build_agent_runtime as build

    return RuntimeAdapter(build(*args, **kwargs))
