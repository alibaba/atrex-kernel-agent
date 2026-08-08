from __future__ import annotations

import json
import signal
import time
import uuid
from pathlib import Path
from collections.abc import Mapping
from typing import Callable

from orchestrator.agent_runtime.codex_ledger import (
    CodexSessionLedgerObserver,
    codex_home,
)
from orchestrator.agent_runtime.model import (
    NormalizedAgentEvent,
    TokenUsage,
    resequence_agent_events,
    subtract_token_usage,
    token_usage_exceeds,
)

from . import main_adapter
from .models import EpisodeHandoff, InvocationObservation, SessionResult
from .protocol import handoff_diagnosis, read_handoff


CompletionCheck = Callable[[EpisodeHandoff], str]
CommandExecutor = Callable[
    [list[str], Path, int, dict[str, str]], tuple[str, str, int, bool]
]


_CLAUDE_TRANSIENT_API_ERRORS = {"api error: terminated"}
_MAX_HANDOFF_GRACE_SECONDS = 600


def _codex_invocation_usage(
    session_usage: TokenUsage, previous_session_usage: TokenUsage | None
) -> TokenUsage:
    if session_usage.total_tokens is None:
        return TokenUsage.unavailable()
    previous = previous_session_usage or TokenUsage.zero()
    if token_usage_exceeds(previous, session_usage):
        return TokenUsage.unavailable()
    return subtract_token_usage(session_usage, previous)


def _replace_terminal_usage(
    events: tuple[NormalizedAgentEvent, ...], usage: TokenUsage
) -> tuple[NormalizedAgentEvent, ...]:
    retained = [event for event in events if event.kind != "terminal_usage"]
    if usage.total_tokens is not None:
        retained.append(
            NormalizedAgentEvent(
                sequence=len(retained), kind="terminal_usage", usage=usage
            )
        )
    return resequence_agent_events(retained)


def _claude_transient_api_error(stdout: str) -> str:
    """Return a whitelisted transient error from Claude's structured stream."""
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict) or event.get("isApiErrorMessage") is not True:
            continue
        message = event.get("message")
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "text":
                continue
            value = block.get("text")
            if not isinstance(value, str):
                continue
            normalized = value.strip().casefold()
            if normalized in _CLAUDE_TRANSIENT_API_ERRORS:
                return value.strip()
    return ""


class LongSessionRunner:
    def __init__(self, executor: CommandExecutor | None = None, agent_cli: str = "claude"):
        self.executor = executor or main_adapter.run_bounded
        self.agent_cli = agent_cli

    def run(
        self,
        workspace: Path,
        prompt: str,
        *,
        timeout: int,
        handoff_path: Path,
        handoff_resumes: int,
        completion_check: CompletionCheck,
        reasoning_effort: str = "max",
        session_id: str = "",
        telemetry_environment: Mapping[str, str] | None = None,
    ) -> SessionResult:
        session_id = session_id or str(uuid.uuid4())
        is_codex = self.agent_cli == "codex"
        active_session_id = "" if is_codex else session_id
        handoff_path.parent.mkdir(parents=True, exist_ok=True)
        handoff_path.unlink(missing_ok=True)
        deadline = time.monotonic() + timeout
        # Keep a small part of the episode budget for a same-session terminal
        # handoff.  Otherwise the first coding turn can consume the complete
        # deadline and get SIGKILLed with no opportunity to commit or publish
        # the work it already finished.
        handoff_grace = (
            min(_MAX_HANDOFF_GRACE_SECONDS, max(1, timeout // 10))
            if timeout >= 10
            and handoff_resumes > 0
            and main_adapter.supports_same_session_resume(self.agent_cli)
            else 0
        )
        environment = (
            main_adapter.session_environment()
            if self.agent_cli == "claude"
            else main_adapter.session_environment(self.agent_cli)
        )
        environment["IS_SANDBOX"] = "1"
        if telemetry_environment:
            environment.update(
                {str(key): str(value) for key, value in telemetry_environment.items()}
            )
        telemetry_attempt_prefix = environment.get("ATREX_TELEMETRY_ATTEMPT_ID")
        codex_observer = None
        codex_setup_errors: tuple[str, ...] = ()
        if is_codex:
            try:
                codex_observer = CodexSessionLedgerObserver(
                    codex_home(environment)
                )
            except Exception as exc:
                codex_setup_errors = (
                    f"codex_ledger_setup_failed:{type(exc).__name__}",
                )
        codex_stdout_session_usage: TokenUsage | None = None
        codex_ledger_usable = codex_observer is not None
        stdout_parts: list[str] = []
        stderr_parts: list[str] = []
        total_tokens = 0
        completion_diagnosis = ""
        handoff: EpisodeHandoff | None = None
        exit_status = 0
        timed_out = False
        resume_count = 0
        invocations: list[InvocationObservation] = []

        for attempt in range(max(0, handoff_resumes) + 1):
            remaining = int(deadline - time.monotonic())
            if remaining <= 0:
                timed_out = True
                exit_status = -1
                break
            turn_timeout = remaining
            if attempt == 0 and handoff_grace:
                turn_timeout = max(1, remaining - handoff_grace)
            if attempt == 0:
                turn_prompt = prompt
                command = (
                    main_adapter.fresh_session_command(
                        turn_prompt, session_id, reasoning_effort
                    )
                    if self.agent_cli == "claude"
                    else main_adapter.fresh_session_command(
                        turn_prompt, session_id, reasoning_effort, self.agent_cli
                    )
                )
            else:
                if not main_adapter.supports_same_session_resume(self.agent_cli):
                    completion_diagnosis = (
                        completion_diagnosis
                        or f"{self.agent_cli} ended without a valid terminal handoff"
                    )
                    break
                if not active_session_id:
                    completion_diagnosis = (
                        completion_diagnosis
                        or f"{self.agent_cli} ended without exposing a resumable session id"
                    )
                    break
                resume_count += 1
                diagnosis = completion_diagnosis or handoff_diagnosis(handoff_path)
                if completion_diagnosis.startswith("engineering budget expired"):
                    turn_prompt = (
                        "The engineering budget for this long-horizon episode has expired. Stop "
                        "further exploration, profiling, and evaluation. Preserve the completed "
                        "work now: inspect the current Git worktree, commit a coherent candidate "
                        "when one is ready, finalize the episode journal, and atomically publish a "
                        "valid handoff. If there is no coherent candidate, publish an honest pivot "
                        "or blocked handoff instead of continuing to optimize."
                    )
                else:
                    turn_prompt = (
                        "Continue the same long-horizon optimization episode. The previous turn did "
                        f"not satisfy the terminal contract: {diagnosis}. Resume concrete engineering "
                        "work from the current Git worktree. Do not merely explain the problem. Before "
                        "stopping, finalize the episode journal and atomically publish a valid handoff."
                    )
                command = (
                    main_adapter.resume_session_command(
                        turn_prompt, active_session_id, reasoning_effort
                    )
                    if self.agent_cli == "claude"
                    else main_adapter.resume_session_command(
                        turn_prompt, active_session_id, reasoning_effort, self.agent_cli
                    )
                )
            if telemetry_attempt_prefix:
                environment["ATREX_TELEMETRY_ATTEMPT_ID"] = (
                    f"{telemetry_attempt_prefix}-{attempt + 1}"
                )
            stdout, stderr, exit_status, turn_timed_out = self.executor(
                command, workspace, turn_timeout, environment
            )
            stdout_parts.append(stdout)
            stderr_parts.append(stderr)
            observed_session_id = main_adapter.session_id_from_stream(
                self.agent_cli, stdout, session_id
            )
            if observed_session_id:
                active_session_id = observed_session_id
            stream_session_usage = (
                main_adapter.terminal_usage_from_stream(stdout)
                if is_codex
                else None
            )
            events, terminal_usage, capabilities, observation_errors = (
                main_adapter.normalize_stream(
                    self.agent_cli,
                    stdout,
                    session_id=active_session_id,
                    codex_observer=(
                        codex_observer if codex_ledger_usable else None
                    ),
                )
            )
            if attempt == 0 and codex_setup_errors:
                observation_errors = codex_setup_errors + observation_errors
            ledger_failed = any(
                value.startswith("codex_ledger_unavailable:")
                for value in observation_errors
            )
            if ledger_failed:
                codex_ledger_usable = False
            ledger_usage_observed = bool(
                is_codex
                and capabilities.usage_delta_observed
                and not ledger_failed
            )
            resume_usage_qualified = bool(
                ledger_usage_observed
                and not any(
                    value.startswith("codex_")
                    for value in observation_errors
                )
            )
            if is_codex and not ledger_usage_observed:
                fallback_usage = _codex_invocation_usage(
                    stream_session_usage or TokenUsage.unavailable(),
                    codex_stdout_session_usage,
                )
                if fallback_usage.total_tokens is not None:
                    terminal_usage = fallback_usage
                    events = _replace_terminal_usage(events, fallback_usage)
                else:
                    terminal_usage = TokenUsage.unavailable()
                    events = _replace_terminal_usage(events, terminal_usage)
                    observation_errors += (
                        "codex_cumulative_fallback_unavailable",
                    )
            if (
                stream_session_usage is not None
                and stream_session_usage.total_tokens is not None
                and (
                    codex_stdout_session_usage is None
                    or not token_usage_exceeds(
                        codex_stdout_session_usage, stream_session_usage
                    )
                )
            ):
                codex_stdout_session_usage = stream_session_usage
            total_tokens += (
                terminal_usage.total_tokens
                if terminal_usage.total_tokens is not None
                else (
                    0
                    if is_codex
                    else main_adapter.tokens_from_stream(stdout)
                )
            )
            invocations.append(
                InvocationObservation(
                    terminal_usage=terminal_usage,
                    events=events,
                    capabilities=capabilities,
                    observation_errors=observation_errors,
                    resume_usage_qualified=resume_usage_qualified,
                )
            )
            timed_out = turn_timed_out
            observed = read_handoff(handoff_path)
            if observed is not None:
                completion_diagnosis = completion_check(observed)
                if not completion_diagnosis:
                    handoff = observed
                    # A valid, committed terminal handoff is authoritative even
                    # when the CLI lingered until its turn timeout afterward.
                    exit_status = 0
                    timed_out = False
                    break
            else:
                completion_diagnosis = handoff_diagnosis(handoff_path)
            if timed_out:
                can_resume = (
                    bool(active_session_id)
                    and main_adapter.supports_same_session_resume(self.agent_cli)
                    and attempt < max(0, handoff_resumes)
                    and int(deadline - time.monotonic()) > 0
                )
                if can_resume:
                    completion_diagnosis = (
                        "engineering budget expired before a valid terminal handoff"
                    )
                    continue
                break
            if exit_status != 0:
                externally_terminated = exit_status in {
                    -signal.SIGTERM,
                    128 + signal.SIGTERM,
                }
                dependency_terminated = "dependency policy violation" in stderr.lower()
                transient_api_error = (
                    _claude_transient_api_error(stdout)
                    if self.agent_cli == "claude"
                    else ""
                )
                can_resume = (
                    (externally_terminated or bool(transient_api_error))
                    and not dependency_terminated
                    and active_session_id
                    and main_adapter.supports_same_session_resume(self.agent_cli)
                    and attempt < max(0, handoff_resumes)
                )
                if can_resume:
                    completion_diagnosis = (
                        f"Claude coding session hit transient {transient_api_error} before "
                        "publishing a valid handoff"
                        if transient_api_error
                        else "coding session received SIGTERM before publishing a valid handoff"
                    )
                    continue
                break

        stdout_all = "\n".join(stdout_parts)
        stderr_all = "\n".join(stderr_parts)
        return SessionResult(
            exit_status=exit_status,
            timed_out=timed_out,
            tokens=total_tokens,
            session_id=active_session_id or session_id,
            resume_count=resume_count,
            handoff=handoff,
            stdout_tail=stdout_all[-4000:],
            stderr_tail=stderr_all[-4000:],
            completion_diagnosis=completion_diagnosis,
            invocations=tuple(invocations),
        )
