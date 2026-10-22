"""Internal legacy request adapter; shared usage types have one public owner."""
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from aka.contracts.agent import (
    TokenUsage, NormalizedAgentEvent, AgentRuntimeCapabilities,
    sum_token_usages, token_usage_exceeds, subtract_token_usage, resequence_agent_events,
)

def merge_token_usage_evidence(left: TokenUsage, right: TokenUsage) -> TokenUsage:
    """Preserve known counters without adding possibly overlapping observations.

    Component maxima are a conservative reconciliation, not a proven total for
    unseen child requests. The result must remain partial.
    """
    if left.total_tokens is None and right.total_tokens is None:
        return TokenUsage.unavailable()
    values = {}
    for key in ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens"):
        known = [getattr(row, key) for row in (left, right) if getattr(row, key) is not None]
        values[key] = max(known) if known else None
    total = max(
        left.total_tokens or 0, right.total_tokens or 0, sum(v or 0 for v in values.values())
    )
    return TokenUsage(**values, total_tokens=total, measurement="partial")


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
    observation_directory: Path | None = None


@dataclass(frozen=True)
class AgentRunResult:
    runtime_id: str
    exit_status: int
    timed_out: bool
    terminal_usage: TokenUsage
    events: tuple[NormalizedAgentEvent, ...]
    capabilities: AgentRuntimeCapabilities
    observation_errors: tuple[str, ...]
    stdout_tail: str
    stderr_tail: str
    session_id: str = ""

    @property
    def tokens(self) -> int:
        """Compatibility total used by existing campaign token budgets."""
        return self.terminal_usage.total_tokens or 0


class AgentRuntime(Protocol):
    @property
    def id(self) -> str:
        ...

    def run(self, request: AgentRunRequest) -> AgentRunResult:
        ...
