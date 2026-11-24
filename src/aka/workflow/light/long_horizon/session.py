from __future__ import annotations

import json
import signal
import uuid
from threading import Event
from collections.abc import Mapping
from pathlib import Path
from typing import Callable

from aka.execution.recovery import raise_if_environment_blocked

from aka.contracts.agent import Agent, NativeInvocation
from aka.workflow.light.long_horizon.models import InvocationObservation, SessionResult
from aka.workflow.light.long_horizon.output_limit import (
    EPISODE_COMPLETION_INSTRUCTION,
    dependency_policy_recovery_prompt,
    output_limit_continuation_prompt,
)
from aka.agent.adapter import DEFAULT_BACKEND_REGISTRY
from aka.agent.process import (
    OUTPUT_LIMIT_EXIT_STATUS,
    OUTPUT_LIMIT_INTERRUPT_MARKER,
)
from aka.agent.claude_tools import (
    claim_outer_output_limit_recovery,
    output_limit_recovery_count,
)
from aka.workflow.light.orchestration.infrastructure_retry import InfrastructureUnavailable
from aka.workflow.light.orchestration.constants import DEFAULT_OUTPUT_LIMIT_RECOVERY_RETRIES

_CLAUDE_TRANSIENT_API_ERRORS = {"api error: terminated"}
_CLAUDE_COMPACTION_FAILURE = "automatic compaction failed"
_CLAUDE_COMPACTION_CONNECTION_LOSS = "connection lost mid-response"
_DEPENDENCY_POLICY_MARKER = (
    "Agent host policy violation:"
)


def _dependency_policy_violation(exit_status: int, stderr: str) -> str:
    """Return bounded guard evidence without conflating ordinary exit 126."""
    if exit_status != 126 or _DEPENDENCY_POLICY_MARKER not in stderr:
        return ""
    detail = stderr.rsplit(_DEPENDENCY_POLICY_MARKER, 1)[1].strip()
    return (detail or "dependency guard terminated the coding session")[:2000]


def _claude_retryable_api_error(message: str) -> bool:
    """Return whether a bounded same-session retry can preserve useful work."""
    normalized = message.strip().casefold()
    if normalized in _CLAUDE_TRANSIENT_API_ERRORS:
        return True
    if _claude_retryable_compaction_error(normalized):
        return True
    return (
        "request rejected (429)" in normalized
        and "throttling" in normalized
        and "exceeded your current quota" in normalized
    )


def _claude_retryable_compaction_error(message: str) -> bool:
    """Recognize a failed provider compaction without treating hard limits as transient."""
    normalized = message.strip().casefold()
    return (
        _CLAUDE_COMPACTION_FAILURE in normalized
        and _CLAUDE_COMPACTION_CONNECTION_LOSS in normalized
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

    def run(self, *args, **kwargs) -> SessionResult:
        """Episode active time is intentionally unbounded."""
        from aka.core.active_time import ActiveBudget, budget_scope
        with budget_scope(ActiveBudget(None)) as budget:
            return self._run(*args, **kwargs, active_budget=budget)

    def _run(
        self,
        workspace: Path,
        prompt: str,
        *,
        reasoning_effort: str = "max",
        session_id: str = "",
        resume_session: bool = False,
        output_directory: Path | None = None,
        telemetry_environment: Mapping[str, str] | None = None,
        access=None,
        active_budget=None,
        record_usage: Callable[[int], None] | None = None,
        record_session: Callable[[str], None] | None = None,
        output_limit_recovery_retries: int = DEFAULT_OUTPUT_LIMIT_RECOVERY_RETRIES,
        dependency_policy_recovery_retries: int = 2,
        remaining_token_budget: int | None = None,
        stage_name: str = "Episode",
        completion_instruction: str = EPISODE_COMPLETION_INSTRUCTION,
        initial_purpose: str = "optimization",
    ) -> SessionResult:
        if (type(output_limit_recovery_retries) is not int
                or not 0 <= output_limit_recovery_retries <= 10):
            raise ValueError("output_limit_recovery_retries must be an integer from 0 to 10")
        if (type(dependency_policy_recovery_retries) is not int
                or not 0 <= dependency_policy_recovery_retries <= 10):
            raise ValueError(
                "dependency_policy_recovery_retries must be an integer from 0 to 10"
            )
        if resume_session and not session_id:
            raise ValueError("resume_session requires an existing session_id")
        session_id = session_id or str(uuid.uuid4())
        assigns_session_id = self.agent_cli not in {"codex", "dsh"}
        active_session_id = (
            session_id
            if resume_session
            else (session_id if assigns_session_id else "")
        )
        # Declare Workflow-owned per-turn fields before the shared Agent freezes
        # the scoped Session environment allowlist (also used by recovery Sessions).
        session_environment = {
            **(telemetry_environment or {}),
            "ATREX_AGENT_STAGE_NAME": stage_name,
            "ATREX_OUTPUT_LIMIT_RECOVERY_RETRIES": str(output_limit_recovery_retries),
            "ATREX_OUTPUT_LIMIT_RECOVERY_SCOPE": session_id,
        }
        native = self.agent.open_session(
            self.agent_cli,
            workspace,
            session_id=session_id,
            environment=session_environment,
            output_directory=output_directory,
            access=access,
        )
        try:
            if resume_session and not native.supports_resume:
                raise ValueError(
                    f"{self.agent_cli} does not support cross-Episode Session resume"
                )
            telemetry_attempt_prefix = native.environment.get(
                "ATREX_TELEMETRY_ATTEMPT_ID"
            )
            stdout_tail = ""
            stderr_tail = ""
            total_tokens = 0
            completion_diagnosis = ""
            exit_status = 0
            timed_out = False
            resume_count = 0
            invocations: list[InvocationObservation] = []
            output_limit_recoveries = 0
            dependency_policy_recoveries = 0
            terminal_failure_kind: str | None = None
            next_prompt = prompt
            purpose = initial_purpose
            attempt = 0
            rate_limit_retries = 0
            compaction_fresh_recovery_used = False
            # A trajectory continuation must use the provider's real resume
            # command on its first invocation. Supplying the old id to a fresh
            # command would silently create a disconnected conversation.
            retry_resume = resume_session
            recorded_session_id = ""

            while True:
                if active_budget.remaining() <= 0:
                    if purpose == "rate_limit_recovery":
                        raise InfrastructureUnavailable(
                            f"Rate-limited {stage_name} exhausted its active-time budget; draft preserved"
                        )
                    timed_out = True
                    exit_status = exit_status or -signal.SIGKILL
                    terminal_failure_kind = "episode_budget_exhausted"
                    completion_diagnosis = (
                        f"{stage_name} active-time budget exhausted; no continuation started"
                    )
                    break
                cancel = getattr(self, "cancel_event", None)
                if cancel is not None and cancel.is_set():
                    from concurrent.futures import CancelledError
                    raise CancelledError(f"Search coordinator stopped this {stage_name}")
                if remaining_token_budget is not None and total_tokens >= remaining_token_budget:
                    if purpose == "rate_limit_recovery":
                        raise InfrastructureUnavailable(
                            f"Rate-limited {stage_name} exhausted its token budget; draft preserved"
                        )
                    completion_diagnosis = (
                        f"{stage_name} token budget exhausted; no continuation started"
                    )
                    exit_status = exit_status or 1
                    break
                if next_prompt is not None:
                    turn_prompt = next_prompt
                    resume = retry_resume
                    retry_resume = False
                else:
                    break
                environment = {
                    "ATREX_AGENT_STAGE_NAME": stage_name,
                    "ATREX_OUTPUT_LIMIT_RECOVERY_RETRIES": str(
                        output_limit_recovery_retries
                    ),
                    "ATREX_OUTPUT_LIMIT_RECOVERY_SCOPE": session_id,
                }
                if telemetry_attempt_prefix:
                    environment["ATREX_TELEMETRY_ATTEMPT_ID"] = (
                        f"{telemetry_attempt_prefix}-{attempt + 1}"
                    )
                # Claude/Pi/Qoder identities are assigned before the process starts.
                # Persist here on EVERY invocation: recovery can replace the Session
                # inside an Episode too. Never persist a placeholder for provider-
                # assigned Codex/DSH identities.
                identity = getattr(native, "session_id", "") or (
                    active_session_id if resume or assigns_session_id else ""
                )
                if record_session is not None and identity and identity != recorded_session_id:
                    record_session(identity)
                    recorded_session_id = identity
                # Release the previous invocation before waiting for its successor.
                turn = None
                stdout = stderr = ""
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
                stdout_tail = (stdout_tail + ("\n" if attempt > 1 else "") + stdout[-4000:])[-4000:]
                stderr_tail = (stderr_tail + ("\n" if attempt > 1 else "") + stderr[-4000:])[-4000:]
                active_session_id = turn.session_id or active_session_id
                if (record_session is not None and turn.session_id
                        and turn.session_id != recorded_session_id):
                    record_session(turn.session_id)
                    recorded_session_id = turn.session_id
                output_limit_recoveries = max(
                    output_limit_recoveries,
                    output_limit_recovery_count(workspace, session_id),
                )
                failure_kind, provider_error_code = (
                    DEFAULT_BACKEND_REGISTRY.create(
                        self.agent_cli
                    ).classify_terminal_failure(stdout, active_session_id)
                    if self.agent_cli in DEFAULT_BACKEND_REGISTRY.ids
                    else (None, None)
                )
                if (
                    self.agent_cli == "claude"
                    and exit_status == OUTPUT_LIMIT_EXIT_STATUS
                    and OUTPUT_LIMIT_INTERRUPT_MARKER in stderr
                ):
                    failure_kind, provider_error_code = (
                        "output_limit", "max_output_tokens"
                    )
                dependency_policy_violation = _dependency_policy_violation(
                    exit_status, stderr
                )
                if dependency_policy_violation:
                    # A confirmed local guard decision is more specific than
                    # any provider-stream symptom emitted during termination.
                    failure_kind, provider_error_code = (
                        "dependency_policy_violation", None
                    )
                    terminal_failure_kind = failure_kind
                else:
                    terminal_failure_kind = None
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
                    purpose = initial_purpose
                if record_usage is not None:
                    record_usage(total_tokens)
                # Even an interrupted candidate has consumed real model tokens.
                # Account first, then stop without consuming an episode outcome.
                raise_if_environment_blocked()
                timed_out = turn_timed_out
                completion_diagnosis = ""
                if failure_kind == "rate_limit" and not timed_out:
                    rate_limit_retries += 1
                    purpose = "rate_limit_recovery"
                    delay = min(60, 5 * 2 ** min(rate_limit_retries - 1, 4))
                    print(f"[long-horizon] provider rate limit ({provider_error_code}); "
                          f"retry {rate_limit_retries} in {delay}s; same {stage_name} and draft; "
                          "no optimization round consumed", flush=True)
                    if (active_budget.remaining() <= 0 or
                            (remaining_token_budget is not None and total_tokens >= remaining_token_budget)):
                        raise InfrastructureUnavailable(
                            f"Rate-limited {stage_name} reached its budget; draft preserved"
                        )
                    if (cancel if cancel is not None else Event()).wait(delay):
                        from concurrent.futures import CancelledError
                        raise CancelledError("Search coordinator stopped rate-limit recovery")
                    continuation = (
                        f"Continue the SAME {stage_name} after provider throttling. "
                        "Preserve the existing kernel/source tree and scratch. Inspect saved progress "
                        "and reconcile existing Runtime/Gateway results before repeating work. "
                        f"Do not reset the candidate. {completion_instruction.strip()}"
                    )
                    if native.supports_resume and active_session_id:
                        next_prompt = continuation
                        retry_resume = True
                    else:
                        native.close()
                        active_session_id = str(uuid.uuid4())
                        native = self.agent.open_session(
                            self.agent_cli, workspace, session_id=active_session_id,
                            environment=session_environment,
                            output_directory=output_directory,
            access=access,
                        )
                        next_prompt = prompt.rstrip() + "\n\n" + continuation
                    continue
                if failure_kind == "provider_error":
                    completion_diagnosis = f"Provider session failed: {provider_error_code}"
                if timed_out:
                    if active_budget.remaining() <= 0:
                        terminal_failure_kind = "episode_budget_exhausted"
                        completion_diagnosis = (
                            f"{stage_name} active-time budget exhausted; no continuation started"
                        )
                        break
                    break
                if exit_status != 0:
                    externally_terminated = exit_status in {
                        -signal.SIGTERM,
                        128 + signal.SIGTERM,
                    }
                    if failure_kind == "dependency_policy_violation":
                        completion_diagnosis = (
                            "Dependency policy guard terminated the coding session: "
                            f"{dependency_policy_violation}; "
                            f"{dependency_policy_recoveries}/"
                            f"{dependency_policy_recovery_retries} fresh-session "
                            "recoveries used"
                        )
                        if (
                            dependency_policy_recoveries
                            >= dependency_policy_recovery_retries
                        ):
                            break
                        if (
                            remaining_token_budget is not None
                            and total_tokens >= remaining_token_budget
                        ):
                            completion_diagnosis += f"; {stage_name} token budget exhausted"
                            break
                        if active_budget.remaining() <= 0:
                            completion_diagnosis += (
                                f"; {stage_name} active-time budget exhausted"
                            )
                            timed_out = True
                            break
                        dependency_policy_recoveries += 1
                        next_prompt = dependency_policy_recovery_prompt(
                            workspace,
                            prompt,
                            stdout,
                            active_session_id,
                            dependency_policy_violation,
                            dependency_policy_recoveries,
                            dependency_policy_recovery_retries,
                            stage_name=stage_name,
                            completion_instruction=completion_instruction,
                        )
                        # The guarded process group no longer exists and must
                        # never be resumed. Preserve files, but require a fresh
                        # replacement provider Session over the preserved files.
                        native.close()
                        active_session_id = str(uuid.uuid4())
                        native = self.agent.open_session(
                            self.agent_cli,
                            workspace,
                            session_id=active_session_id,
                            environment=session_environment,
                            output_directory=output_directory,
            access=access,
                        )
                        purpose = "dependency_policy_recovery"
                        print(
                            "[long-horizon] dependency policy violation; preserving "
                            f"{stage_name} draft; fresh-session recovery "
                            f"{dependency_policy_recoveries}/"
                            f"{dependency_policy_recovery_retries}",
                            flush=True,
                        )
                        continue
                    if failure_kind == "output_limit":
                        completion_diagnosis = (
                            "Claude max_output_tokens: response output limit exceeded; "
                            f"{output_limit_recoveries}/{output_limit_recovery_retries} "
                            "same-session recoveries used"
                        )
                        if (exit_status not in {1, OUTPUT_LIMIT_EXIT_STATUS}
                                or output_limit_recoveries >= output_limit_recovery_retries):
                            break
                        if not native.supports_resume or not active_session_id:
                            completion_diagnosis += (
                                "; existing provider session cannot be resumed safely"
                            )
                            break
                        if turn.terminal_usage.total_tokens is None:
                            completion_diagnosis += "; usage unavailable, cannot safely extend execution"
                            break
                        if (remaining_token_budget is not None
                                and total_tokens >= remaining_token_budget):
                            completion_diagnosis += f"; {stage_name} token budget exhausted"
                            break
                        claimed = claim_outer_output_limit_recovery(
                            workspace,
                            session_id,
                            f"{active_session_id}:{attempt}",
                            output_limit_recovery_retries,
                        )
                        # A state-file failure must not turn a recoverable
                        # terminal API error into a dead end. The in-memory
                        # count still bounds this controller invocation.
                        output_limit_recoveries = (
                            claimed
                            if claimed is not None
                            else output_limit_recoveries + 1
                        )
                        next_prompt = output_limit_continuation_prompt(
                            output_limit_recoveries, output_limit_recovery_retries,
                            stage_name=stage_name,
                            completion_instruction=completion_instruction,
                        )
                        # Keep the complete native conversation. The short
                        # continuation prompt changes response granularity; it
                        # does not reconstruct context from files or receipts.
                        retry_resume = True
                        purpose = "output_limit_recovery"
                        print(f"[long-horizon] max_output_tokens; preserving {stage_name} draft; "
                              "same-session continuation "
                              f"{output_limit_recoveries}/{output_limit_recovery_retries}",
                              flush=True)
                        continue
                    transient_api_error = (
                        _claude_transient_api_error(stdout)
                        if self.agent_cli == "claude"
                        else ""
                    )
                    can_resume = (
                        (externally_terminated or bool(transient_api_error))
                        and bool(active_session_id)
                        and native.supports_resume
                        and resume_count < 1
                    )
                    if can_resume:
                        resume_count += 1
                        retry_resume = True
                        purpose = "session_recovery"
                        next_prompt = (
                            f"Continue the SAME {stage_name} after a transient provider "
                            "interruption. Preserve the current source tree, Git checkpoints, "
                            "measurements, and recorded research. Inspect durable state before "
                            "repeating work; continue with one small evidence-backed step at a "
                            f"time. {completion_instruction.strip()}"
                        )
                        completion_diagnosis = (
                            f"Provider session hit transient {transient_api_error}; "
                            "resuming the same Session"
                            if transient_api_error
                            else "Provider session received SIGTERM; resuming the same Session"
                        )
                        continue
                    if (
                        transient_api_error
                        and _claude_retryable_compaction_error(transient_api_error)
                        and not compaction_fresh_recovery_used
                    ):
                        # A second failed compaction can leave the provider Session
                        # unable to accept even a short continuation. Preserve the
                        # draft and Runtime records, but rebuild conversational
                        # context once from the ordinary Episode prompt instead of
                        # discarding the whole audit segment.
                        native.close()
                        active_session_id = str(uuid.uuid4())
                        native = self.agent.open_session(
                            self.agent_cli,
                            workspace,
                            session_id=active_session_id,
                            environment=session_environment,
                            output_directory=output_directory,
            access=access,
                        )
                        next_prompt = (
                            prompt.rstrip()
                            + "\n\n"
                            + f"Recover the SAME {stage_name} from the existing workspace and "
                            "Runtime records. Claude's automatic context compaction failed after "
                            "the provider connection was lost. Preserve the current source tree, "
                            "Git checkpoints, measurements, and recorded research. Inspect the "
                            "durable state before doing new work; do not reset or repeat completed "
                            "measurements. Continue with one small evidence-backed step at a time; "
                            "do not treat the recovery itself as a terminal condition. "
                            f"{completion_instruction.strip()}"
                        )
                        retry_resume = False
                        purpose = "compaction_recovery"
                        compaction_fresh_recovery_used = True
                        completion_diagnosis = (
                            "Claude automatic compaction repeatedly lost the provider connection; "
                            "started one fresh recovery Session over the preserved draft"
                        )
                        print(
                            "[long-horizon] automatic compaction connection loss; preserving "
                            f"{stage_name} draft; fresh-session recovery 1/1",
                            flush=True,
                        )
                        continue
                    break

            return SessionResult(
                exit_status=exit_status,
                timed_out=timed_out,
                tokens=total_tokens,
                session_id=active_session_id or session_id,
                resume_count=resume_count,
                stdout_tail=stdout_tail,
                stderr_tail=stderr_tail,
                completion_diagnosis=completion_diagnosis,
                invocations=tuple(invocations),
                output_limit_recoveries=output_limit_recoveries,
                dependency_policy_recoveries=dependency_policy_recoveries,
                terminal_failure_kind=terminal_failure_kind,
            )
        finally:
            native.close()
