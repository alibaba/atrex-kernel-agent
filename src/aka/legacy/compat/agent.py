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
    environment = {str(k): str(v) for k, v in (request.extra_environment or {}).items()}
    remove = []
    if request.sandbox_hardware:
        environment["ATREX_SANDBOX_GPU"] = request.sandbox_hardware
    for value, key, others in (
        (
            request.sandbox_ssh,
            "ATREX_SANDBOX_SSH",
            ("ATREX_SANDBOX_URL", "ATREX_SANDBOX_PROFILE"),
        ),
        (
            request.sandbox_url,
            "ATREX_SANDBOX_URL",
            ("ATREX_SANDBOX_SSH", "ATREX_SANDBOX_PROFILE"),
        ),
        (
            request.sandbox_profile,
            "ATREX_SANDBOX_PROFILE",
            ("ATREX_SANDBOX_SSH", "ATREX_SANDBOX_URL"),
        ),
    ):
        if value:
            environment[key] = value
            for other in others:
                environment.pop(other, None)
            remove.extend(others)
            break
    for value, key in (
        (request.sandbox_ssh_init, "ATREX_SANDBOX_SSH_INIT"),
        (request.sandbox_health_command, "ATREX_SANDBOX_HEALTH_COMMAND"),
        (request.environment_state_file, "ATREX_ENVIRONMENT_STATE_FILE"),
    ):
        if value:
            environment[key] = value
    environment["ATREX_SANDBOX_TIMEOUT"] = str(request.sandbox_timeout_s)
    for key in remove:
        environment[key] = None
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
