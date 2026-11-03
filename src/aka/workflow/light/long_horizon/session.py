from __future__ import annotations

import json
import signal
import time
import uuid
from threading import Event
from collections.abc import Mapping
from contextlib import ExitStack
from pathlib import Path
from typing import Callable

from aka.execution.recovery import raise_if_environment_blocked

from aka.contracts.agent import Agent, NativeInvocation
from .models import EpisodeHandoff, InvocationObservation, SessionResult
from .protocol import handoff_diagnosis, read_handoff
from .output_limit import recovery_prompt
from aka.workflow.light.orchestration.infrastructure_retry import InfrastructureUnavailable

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


def _claude_terminal_failure(stdout, session_id):
    """Workflow recovery policy for the root session's terminal provider error."""
    from aka.agent.provider_errors import rate_limit_code
    if not session_id:
        return None, None
    last_assistant = None
    failure = (None, None)
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except (ValueError, TypeError):
            continue
        if (not isinstance(event, dict) or event.get("session_id") != session_id
                or event.get("parent_tool_use_id") is not None):
            continue
        if event.get("type") == "assistant":
            last_assistant, failure = event, (None, None)
            if event.get('isApiErrorMessage') is True:
                message = event.get('message')
                blocks = message.get('content', []) if isinstance(message, dict) else []
                for block in blocks:
                    if isinstance(block, dict) and block.get('type') == 'text':
                        code = rate_limit_code(str(block.get('text', '')))
                        if code:
                            failure = ('rate_limit', code)
        elif event.get("type") == "result":
            if event.get('is_error') is True and failure[0] == 'rate_limit':
                continue
            failure = (None, None)
            assistant, last_assistant = last_assistant, None
            if (event.get("is_error") is not True or event.get("terminal_reason") != "api_error"
                    or assistant is None or assistant.get("is_api_error_message") is not True):
                continue
            codes = {assistant.get(key) for key in ("error", "api_error")
                     if isinstance(assistant.get(key), str)}
            if codes == {"max_output_tokens"}:
                failure = ("output_limit", "max_output_tokens")
    return failure


def _pi_terminal_failure(stdout):
    from aka.agent.provider_errors import rate_limit_code
    failure = (None, None)
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if not isinstance(event, dict):
            continue
        message = event.get('message')
        if event.get('type') != 'message_end' or not isinstance(message, dict) or message.get('role') != 'assistant':
            continue
        failure = (None, None)
        if message.get('stopReason') in {'error', 'aborted'}:
            code = rate_limit_code(str(message.get('errorMessage', ''))) if message['stopReason'] == 'error' else None
            failure = ('rate_limit', code) if code else ('provider_error', 'pi_assistant_' + message['stopReason'])
    return failure


class LongSessionRunner:
    def __init__(
        self,
        *,
        agent: Agent,
        usage_policy: Callable[[NativeInvocation], int],
        agent_cli: str = "claude",
        source_inventory=None,
    ):
        self.agent_cli = agent_cli
        self.agent = agent
        self.usage_policy = usage_policy
        self.source_inventory = source_inventory

    def run(self, *args, episode_timeout_seconds=14400, resumed=False, **kwargs) -> SessionResult:
        from aka.workflow.light.deadline import episode_deadline
        deadline = episode_deadline(episode_timeout_seconds, kwargs.get("output_directory"), resumed=resumed)
        if deadline <= time.monotonic():
            return SessionResult(-signal.SIGKILL, True, 0, "", 0, None,
                completion_diagnosis="Episode deadline exhausted; no session started")
        return self._run(*args, deadline=deadline, **kwargs)

    def _run(
        self,
        workspace: Path,
        prompt: str,
        *,
        handoff_path: Path,
        handoff_resumes: int,
        completion_check: CompletionCheck,
        deadline: float,
        reasoning_effort: str = "max",
        session_id: str = "",
        output_directory: Path | None = None,
        telemetry_environment: Mapping[str, str] | None = None,
        access=None,
        tool_session_factory=None,
        record_usage: Callable[[int], None] | None = None,
        output_limit_recovery_retries: int = 2,
        remaining_token_budget: int | None = None,
    ) -> SessionResult:
        if (type(output_limit_recovery_retries) is not int
                or not 0 <= output_limit_recovery_retries <= 10):
            raise ValueError("output_limit_recovery_retries must be an integer from 0 to 10")
        session_id = session_id or str(uuid.uuid4())
        is_codex = self.agent_cli == "codex"
        active_session_id = "" if is_codex else session_id
        handoff_path.parent.mkdir(parents=True, exist_ok=True)
        handoff_path.unlink(missing_ok=True)
        owned = ExitStack()
        current_session = ExitStack()
        owned.callback(current_session.close)

        def open_native(identity):
            current_session.close()
            environment = dict(telemetry_environment or {})
            selected_access = access
            if tool_session_factory is not None:
                grant = current_session.enter_context(
                    tool_session_factory(workspace, access=access, deadline=deadline)
                )
                environment.update(grant.environment)
                selected_access = grant.filesystem_access
            native = self.agent.open_session(
                self.agent_cli,
                workspace,
                session_id=identity,
                environment=environment,
                access=selected_access,
                output_directory=output_directory,
            )
            current_session.callback(native.close)
            return native

        try:
            native = open_native(session_id)
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
            recoveries = 0
            next_prompt = prompt
            purpose = "optimization"
            attempt = 0
            rate_limit_retries = 0
            retry_resume = False

            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    if purpose == "rate_limit_recovery":
                        raise InfrastructureUnavailable("Rate-limited Episode exhausted its deadline; draft preserved")
                    timed_out = True
                    exit_status = exit_status or -signal.SIGKILL
                    completion_diagnosis = "Episode deadline exhausted; no continuation started"
                    break
                cancel = getattr(self, "cancel_event", None)
                if cancel is not None and cancel.is_set():
                    from concurrent.futures import CancelledError
                    raise CancelledError("Search coordinator stopped this Episode")
                if remaining_token_budget is not None and total_tokens >= remaining_token_budget:
                    if purpose == "rate_limit_recovery":
                        raise InfrastructureUnavailable("Rate-limited Episode exhausted its token budget; draft preserved")
                    completion_diagnosis = "Episode token budget exhausted; no continuation started"
                    exit_status = exit_status or 1
                    break
                if next_prompt is not None:
                    turn_prompt = next_prompt
                    resume = retry_resume
                    retry_resume = False
                else:
                    if resume_count >= max(0, handoff_resumes):
                        break
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
                    resume = True
                    purpose = "handoff_resume"
                    diagnosis = completion_diagnosis or handoff_diagnosis(handoff_path)
                    turn_prompt = (
                        "Continue the same long-horizon optimization episode. The previous turn did "
                        f"not satisfy the terminal contract: {diagnosis}. Resume concrete engineering "
                        "work from the current draft. Submit tools/sandbox.py --kind episode-report; "
                        "the Supervisor seals the candidate. Do not merely explain the problem. "
                        "Correct the report fields indicated by Runtime feedback and resubmit "
                        "episode-report. Do not create Git commits or hand-write a handoff."
                    )
                environment = {"ATREX_AGENT_TIMEOUT_SECONDS": str(remaining)}
                if telemetry_attempt_prefix:
                    environment["ATREX_TELEMETRY_ATTEMPT_ID"] = f"{telemetry_attempt_prefix}-{attempt + 1}"
                turn = native.invoke(
                    turn_prompt,
                    resume=resume,
                    reasoning_effort=reasoning_effort,
                    environment=environment,
                )
                attempt += 1
                next_prompt = None
                stdout, stderr = turn.stdout, turn.stderr
                exit_status, turn_timed_out = turn.exit_status, turn.timed_out
                stdout_parts.append(stdout)
                stderr_parts.append(stderr)
                active_session_id = turn.session_id or active_session_id
                failure_kind, provider_error_code = (
                    _claude_terminal_failure(stdout, active_session_id)
                    if self.agent_cli == "claude" else _pi_terminal_failure(stdout)
                    if self.agent_cli == "pi" else (None, None)
                )
                if failure_kind:
                    exit_status = exit_status or 1
                total_tokens += self.usage_policy(turn)
                invocations.append(
                    InvocationObservation(
                        terminal_usage=turn.terminal_usage,
                        events=turn.events,
                        capabilities=turn.capabilities,
                        observation_errors=turn.observation_errors,
                        resume_usage_qualified=turn.resume_usage_qualified,
                        session_id=active_session_id,
                        purpose=purpose,
                        failure_kind=failure_kind,
                        provider_error_code=provider_error_code,
                    )
                )
                # Keep the invocation's recovery label, but let subsequent
                # budget exhaustion record normal work once throttling ends.
                if purpose == "rate_limit_recovery" and failure_kind != "rate_limit":
                    purpose = "optimization"
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
                if failure_kind == "rate_limit" and not timed_out:
                    rate_limit_retries += 1
                    purpose = "rate_limit_recovery"
                    delay = min(60, 5 * 2 ** min(rate_limit_retries - 1, 4))
                    print(f"[long-horizon] provider rate limit ({provider_error_code}); "
                          f"retry {rate_limit_retries} in {delay}s; same Episode and draft; "
                          "no optimization round consumed", flush=True)
                    if (deadline - time.monotonic() <= 0 or
                            (remaining_token_budget is not None and total_tokens >= remaining_token_budget)):
                        raise InfrastructureUnavailable("Rate-limited Episode reached its budget; draft preserved")
                    if (cancel if cancel is not None else Event()).wait(min(delay, max(0, deadline - time.monotonic()))):
                        from concurrent.futures import CancelledError
                        raise CancelledError("Search coordinator stopped rate-limit recovery")
                    continuation = (
                        "Continue the SAME optimization Episode after provider throttling. "
                        "Preserve the existing kernel/source tree and scratch. Inspect saved progress "
                        "and reconcile existing Runtime/Gateway results before repeating work. "
                        "Finish the existing episode-report contract; do not reset the candidate."
                    )
                    if native.supports_resume and active_session_id:
                        next_prompt = continuation
                        retry_resume = True
                    else:
                        active_session_id = str(uuid.uuid4())
                        native = open_native(active_session_id)
                        next_prompt = prompt.rstrip() + "\n\n" + continuation
                    continue
                if failure_kind == "provider_error":
                    completion_diagnosis = f"Provider session failed: {provider_error_code}"
                if timed_out:
                    if time.monotonic() >= deadline:
                        completion_diagnosis = "Episode deadline exhausted; no continuation started"
                        break
                    can_resume = (
                        bool(active_session_id)
                        and native.supports_resume
                        and resume_count < max(0, handoff_resumes)
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
                    if failure_kind == "output_limit":
                        completion_diagnosis = (
                            "Claude max_output_tokens: response output limit exceeded; "
                            f"{recoveries}/{output_limit_recovery_retries} fresh-session recoveries used"
                        )
                        if (exit_status != 1
                                or recoveries >= output_limit_recovery_retries):
                            break
                        if turn.terminal_usage.total_tokens is None:
                            completion_diagnosis += "; usage unavailable, cannot safely extend execution"
                            break
                        if (remaining_token_budget is not None
                                and total_tokens >= remaining_token_budget):
                            completion_diagnosis += "; Episode token budget exhausted"
                            break
                        recoveries += 1
                        source_paths = ("kernel.py",)
                        if self.source_inventory is not None:
                            try:
                                source_paths = self.source_inventory(workspace)
                            except (OSError, ValueError, RuntimeError, TypeError):
                                pass  # Inventory is an explicit partial hint, never a gate.
                        next_prompt = recovery_prompt(
                            workspace, prompt, stdout, active_session_id,
                            recoveries, output_limit_recovery_retries, source_paths=source_paths,
                        )
                        # Unlike a report-format repair, this starts a fresh
                        # model context without resetting the Episode/draft.
                        active_session_id = str(uuid.uuid4())
                        native = open_native(active_session_id)
                        purpose = "output_limit_recovery"
                        print(f"[long-horizon] max_output_tokens; preserving Episode draft; "
                              f"fresh-session recovery {recoveries}/{output_limit_recovery_retries}",
                              flush=True)
                        continue
                    transient_api_error = (
                        _claude_transient_api_error(stdout)
                        if self.agent_cli == "claude"
                        else ""
                    )
                    can_resume = (
                        (externally_terminated or bool(transient_api_error))
                        and active_session_id
                        and native.supports_resume
                        and resume_count < max(0, handoff_resumes)
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
                output_limit_recoveries=recoveries,
            )
        finally:
            owned.close()
