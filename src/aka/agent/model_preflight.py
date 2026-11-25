"""Verify a model override through the selected Agent owner before launch."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from aka.contracts.agent import AgentRunRequest
from aka.agent.public_output import conversation_events


def check_model(agent, backend: str, model: str, *, timeout_s: int = 120, reasoning_effort: str = "max") -> None:
    """Require a real assistant reply, not CLI startup or an echoed prompt."""
    marker = "AKA_MODEL_READY_" + uuid4().hex
    try:
        with TemporaryDirectory(prefix="aka-model-check-") as directory:
            # Codex requires a trusted Git workspace, as in a real Campaign.
            subprocess.run(["git", "init", "--quiet", directory], check=True, capture_output=True)
            result = agent.runtime(backend).run(AgentRunRequest(
                workspace=Path(directory),
                prompt=f"Reply with exactly {marker}. Do not use tools or read files.",
                timeout_s=timeout_s,
                reasoning_effort=reasoning_effort,
                environment={"AKA_AGENT_MODEL": model},
            ))
    except Exception as error:
        details = str(error)
        if isinstance(error, subprocess.CalledProcessError):
            details = (error.stderr or error.stdout or details)
            if isinstance(details, bytes):
                details = details.decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Model preflight failed: backend={backend}, model={model}, "
            f"setup/invocation error ({type(error).__name__}): {details}"
        ) from error
    replies = [event["content"] for event in
               conversation_events(backend, result.stdout_tail, 0)
               if event["kind"] == "message" and event.get("role") not in {"user", "system"}]
    # Claude-compatible backends may expose only their final result payload.
    for line in result.stdout_tail.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict) and event.get("type") == "result" and not event.get("is_error"):
            if isinstance(event.get("result"), str):
                replies.append(event["result"])
    if not result.timed_out and result.exit_status == 0 and any(marker in text for text in replies):
        return
    reason = (f"timed out after {timeout_s}s" if result.timed_out else
              f"exit={result.exit_status}" if result.exit_status else "no valid assistant reply")
    details = result.stderr_tail.strip() or result.stdout_tail.strip()
    raise RuntimeError(f"Model preflight failed: backend={backend}, model={model}, {reason}\n{details}")
