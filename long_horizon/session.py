from __future__ import annotations

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


class LongSessionRunner:
    def __init__(self, executor: CommandExecutor | None = None):
        self.executor = executor or main_adapter.run_bounded

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
        handoff_path.parent.mkdir(parents=True, exist_ok=True)
        handoff_path.unlink(missing_ok=True)
        deadline = time.monotonic() + timeout
        environment = main_adapter.session_environment()
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
                command = main_adapter.fresh_session_command(
                    turn_prompt, session_id, reasoning_effort
                )
            else:
                resume_count += 1
                diagnosis = completion_diagnosis or handoff_diagnosis(handoff_path)
                turn_prompt = (
                    "Continue the same long-horizon optimization episode. The previous turn did "
                    f"not satisfy the terminal contract: {diagnosis}. Resume concrete engineering "
                    "work from the current Git worktree. Do not merely explain the problem. Before "
                    "stopping, finalize the episode journal and atomically publish a valid handoff."
                )
                command = main_adapter.resume_session_command(
                    turn_prompt, session_id, reasoning_effort
                )
            stdout, stderr, exit_status, turn_timed_out = self.executor(
                command, workspace, remaining, environment
            )
            stdout_parts.append(stdout)
            stderr_parts.append(stderr)
            total_tokens += main_adapter.tokens_from_stream(stdout)
            timed_out = timed_out or turn_timed_out
            observed = read_handoff(handoff_path)
            if observed is not None:
                completion_diagnosis = completion_check(observed)
                if not completion_diagnosis:
                    handoff = observed
                    break
            else:
                completion_diagnosis = handoff_diagnosis(handoff_path)
            if exit_status != 0 or timed_out:
                break

        stdout_all = "\n".join(stdout_parts)
        stderr_all = "\n".join(stderr_parts)
        return SessionResult(
            exit_status=exit_status,
            timed_out=timed_out,
            tokens=total_tokens,
            session_id=session_id,
            resume_count=resume_count,
            handoff=handoff,
            stdout_tail=stdout_all[-4000:],
            stderr_tail=stderr_all[-4000:],
            completion_diagnosis=completion_diagnosis,
        )
