from __future__ import annotations

import json
import signal
import time
import uuid
from pathlib import Path
from typing import Callable

from . import main_adapter
from .models import EpisodeHandoff, SessionResult
from .protocol import handoff_diagnosis, read_handoff


CompletionCheck = Callable[[EpisodeHandoff], str]
CommandExecutor = Callable[
    [list[str], Path, int, dict[str, str]], tuple[str, str, int, bool]
]


_CLAUDE_TRANSIENT_API_ERRORS = {"api error: terminated"}


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
    ) -> SessionResult:
        session_id = session_id or str(uuid.uuid4())
        active_session_id = session_id if self.agent_cli != "codex" else ""
        handoff_path.parent.mkdir(parents=True, exist_ok=True)
        handoff_path.unlink(missing_ok=True)
        deadline = time.monotonic() + timeout
        environment = (
            main_adapter.session_environment()
            if self.agent_cli == "claude"
            else main_adapter.session_environment(self.agent_cli)
        )
        environment["IS_SANDBOX"] = "1"
        stdout_parts: list[str] = []
        stderr_parts: list[str] = []
        total_tokens = 0
        completion_diagnosis = ""
        handoff: EpisodeHandoff | None = None
        exit_status = 0
        timed_out = False
        resume_count = 0

        for attempt in range(max(0, handoff_resumes) + 1):
            remaining = int(deadline - time.monotonic())
            if remaining <= 0:
                timed_out = True
                exit_status = -1
                break
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
            stdout, stderr, exit_status, turn_timed_out = self.executor(
                command, workspace, remaining, environment
            )
            stdout_parts.append(stdout)
            stderr_parts.append(stderr)
            total_tokens += main_adapter.tokens_from_stream(stdout)
            observed_session_id = main_adapter.session_id_from_stream(
                self.agent_cli, stdout, session_id
            )
            if observed_session_id:
                active_session_id = observed_session_id
            timed_out = timed_out or turn_timed_out
            observed = read_handoff(handoff_path)
            if observed is not None:
                completion_diagnosis = completion_check(observed)
                if not completion_diagnosis:
                    handoff = observed
                    break
            else:
                completion_diagnosis = handoff_diagnosis(handoff_path)
            if timed_out:
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
        )
