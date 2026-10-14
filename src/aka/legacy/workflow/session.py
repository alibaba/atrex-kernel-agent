from __future__ import annotations

import json
import signal
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Callable

from orchestrator.environment_recovery import raise_if_environment_blocked

from aka.contracts.agent import Agent, NativeInvocation
from .models import EpisodeHandoff, InvocationObservation, SessionResult
from .protocol import handoff_diagnosis, read_handoff

CompletionCheck = Callable[[EpisodeHandoff], str]
_CLAUDE_TRANSIENT_API_ERRORS = {"api error: terminated"}


def _claude_retryable_api_error(message: str) -> bool:
    """Return whether a bounded same-session retry can preserve useful work."""
    normalized = message.strip().casefold()
    if normalized in _CLAUDE_TRANSIENT_API_ERRORS:
        return True
    return (
        "request rejected (429)" in normalized
        and "throttling" in normalized
        and "exceeded your current quota" in normalized
    )


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
            if _claude_retryable_api_error(value):
                return value.strip()
    return ""


class LongSessionRunner:
    def __init__(
        self,
        *,
        agent: Agent,
        usage_policy: Callable[[NativeInvocation], int],
        agent_cli: str = "claude",
    ):
        self.agent_cli = agent_cli
        self.agent = agent
        self.usage_policy = usage_policy

    def run(
        self,
        workspace: Path,
        prompt: str,
        *,
        handoff_path: Path,
        handoff_resumes: int,
        completion_check: CompletionCheck,
        reasoning_effort: str = "max",
        session_id: str = "",
        telemetry_environment: Mapping[str, str] | None = None,
        record_usage: Callable[[int], None] | None = None,
    ) -> SessionResult:
        session_id = session_id or str(uuid.uuid4())
        is_codex = self.agent_cli == "codex"
        active_session_id = "" if is_codex else session_id
        handoff_path.parent.mkdir(parents=True, exist_ok=True)
        handoff_path.unlink(missing_ok=True)
        native = self.agent.open_session(
            self.agent_cli,
            workspace,
            session_id=session_id,
            environment=telemetry_environment,
        )
        try:
            telemetry_attempt_prefix = native.environment.get(
                "ATREX_TELEMETRY_ATTEMPT_ID"
            )
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
                if attempt == 0:
                    turn_prompt = prompt
                else:
                    if not native.supports_resume:
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
                    turn_prompt = (
                        "Continue the same long-horizon optimization episode. The previous turn did "
                        f"not satisfy the terminal contract: {diagnosis}. Resume concrete engineering "
                        "work from the current Git worktree. Candidate commits may contain only "
                        "kernel.py; leave all evidence uncommitted. Do not merely explain the problem. "
                        "Before stopping, finalize the episode journal and atomically publish a valid "
                        "handoff."
                    )
                environment = (
                    {
                        "ATREX_TELEMETRY_ATTEMPT_ID": f"{telemetry_attempt_prefix}-{attempt + 1}"
                    }
                    if telemetry_attempt_prefix
                    else None
                )
                turn = native.invoke(
                    turn_prompt,
                    resume=attempt > 0,
                    reasoning_effort=reasoning_effort,
                    environment=environment,
                )
                stdout, stderr = turn.stdout, turn.stderr
                exit_status, turn_timed_out = turn.exit_status, turn.timed_out
                stdout_parts.append(stdout)
                stderr_parts.append(stderr)
                active_session_id = turn.session_id or active_session_id
                total_tokens += self.usage_policy(turn)
                invocations.append(
                    InvocationObservation(
                        terminal_usage=turn.terminal_usage,
                        events=turn.events,
                        capabilities=turn.capabilities,
                        observation_errors=turn.observation_errors,
                        resume_usage_qualified=turn.resume_usage_qualified,
                    )
                )
                if record_usage is not None:
                    record_usage(total_tokens)
                # Even an interrupted candidate has consumed real model tokens.
                # Account first, then stop without consuming an episode outcome.
                raise_if_environment_blocked()
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
                        and native.supports_resume
                        and attempt < max(0, handoff_resumes)
                    )
                    if can_resume:
                        completion_diagnosis = (
                            "coding session timed out before a valid terminal handoff"
                        )
                        continue
                    break
                if exit_status != 0:
                    externally_terminated = exit_status in {
                        -signal.SIGTERM,
                        128 + signal.SIGTERM,
                    }
                    dependency_terminated = (
                        "dependency policy violation" in stderr.lower()
                    )
                    transient_api_error = (
                        _claude_transient_api_error(stdout)
                        if self.agent_cli == "claude"
                        else ""
                    )
                    can_resume = (
                        (externally_terminated or bool(transient_api_error))
                        and not dependency_terminated
                        and active_session_id
                        and native.supports_resume
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
        finally:
            native.close()
