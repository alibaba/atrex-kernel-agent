"""Agent session implementation and callable interface."""

from __future__ import annotations

from dataclasses import dataclass
from aka import agent as _agent_runtime
import uuid
from pathlib import Path
from typing import Optional
from aka.execution.recovery import (
    environment_state_file,
    raise_if_environment_blocked,
)


@dataclass
class SessionResult:
    exit_status: int
    timed_out: bool
    tokens: int
    stdout_tail: str
    stderr_tail: str
    session_id: str = ""
    terminal_usage: _agent_runtime.TokenUsage | None = None
    events: tuple[_agent_runtime.NormalizedAgentEvent, ...] = ()
    capabilities: _agent_runtime.AgentRuntimeCapabilities | None = None
    observation_errors: tuple[str, ...] = ()


def run_session(
    workspace: Path,
    prompt: str,
    timeout: int,
    agent_cli: str = "claude",
    reasoning_effort: str = "max",
    extra_environment: Optional[dict[str, str]] = None,
    agent_plugins: bool = True,
    runtime: _agent_runtime.AgentRuntime | None = None,
    observation_directory: Path | None = None,
    access: _agent_runtime.AgentAccess | None = None,
) -> SessionResult:
    """Run one clean coding-agent session with no conversational memory from prior iterations."""
    # Kept for the dependency-review call contract. Runtime plan generation is now a
    # workspace-local skill rather than a process-level plugin.
    del agent_plugins
    session_id = str(uuid.uuid4())
    owner = None
    if runtime is None:
        from .agent_host import create_agent

        owner = create_agent()
        runtime = owner.runtime(agent_cli)
    from aka.contracts.agent import AgentRunRequest
    from aka.execution.environment import session_environment

    environment = session_environment(
        state_file=str(environment_state_file() or ""),
        extra=extra_environment,
    )
    request = AgentRunRequest(
        workspace,
        prompt,
        timeout,
        reasoning_effort,
        session_id,
        environment,
        observation_directory,
        access,
    )
    try:
        result = runtime.run(request)
    finally:
        if owner is not None:
            owner.close()
    raise_if_environment_blocked()
    return SessionResult(
        exit_status=result.exit_status,
        timed_out=result.timed_out,
        tokens=result.terminal_usage.total_tokens or 0,
        stdout_tail=result.stdout_tail,
        stderr_tail=result.stderr_tail,
        session_id=result.session_id,
        terminal_usage=result.terminal_usage,
        events=result.events,
        capabilities=result.capabilities,
        observation_errors=result.observation_errors,
    )
