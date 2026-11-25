"""Verify a model override through the selected Agent owner before launch."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from aka.contracts.agent import AgentRunRequest


def _has_reply(backend, event, marker):
    """Match only assistant output, never an echoed user prompt or tool result."""
    if not isinstance(event, dict):
        return False
    kind = event.get("type")
    if backend == "codex":
        item = event.get("item")
        return (kind == "item.completed" and isinstance(item, dict)
                and item.get("type") == "agent_message"
                and marker in str(item.get("text", "")))
    if backend == "dsh":
        return kind == "text" and marker in str(event.get("text", ""))
    if backend in {"claude", "qodercli"} and kind == "result":
        return not event.get("is_error") and marker in str(event.get("result", ""))
    message = event.get("message")
    if not isinstance(message, dict):
        return False
    if backend == "pi":
        if kind != "message_end" or message.get("role") != "assistant":
            return False
    elif kind != "assistant":
        return False
    content = message.get("content", [])
    if isinstance(content, str):
        return marker in content
    return isinstance(content, list) and any(
        isinstance(block, dict) and block.get("type") == "text"
        and marker in str(block.get("text", "")) for block in content)


def _reply_in_archive(path, backend, marker):
    # Session capture spools the entire provider stream to disk. Scan one event
    # at a time so trailing usage/debug events cannot evict the assistant reply.
    if not path.is_file():
        return False
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if (isinstance(record, dict) and record.get("type") == "provider_event"
                    and _has_reply(backend, record.get("event"), marker)):
                return True
    return False


def check_model(agent, backend: str, model: str, *, timeout_s: int = 120, reasoning_effort: str = "max") -> None:
    """Require a real assistant reply, not CLI startup or an echoed prompt."""
    marker = "AKA_MODEL_READY_" + uuid4().hex
    try:
        with TemporaryDirectory(prefix="aka-model-check-") as directory:
            # Codex requires a trusted Git workspace, as in a real Campaign.
            subprocess.run(["git", "init", "--quiet", directory], check=True, capture_output=True)
            capture = Path(directory) / "capture"
            result = agent.runtime(backend).run(AgentRunRequest(
                workspace=Path(directory),
                prompt=f"Reply with exactly {marker}. Do not use tools or read files.",
                timeout_s=timeout_s,
                reasoning_effort=reasoning_effort,
                environment={"AKA_AGENT_MODEL": model, "ATREX_SESSION_CAPTURE": "1",
                             "ATREX_SESSION_CAPTURE_DIR": str(capture)},
            ))
            replied = _reply_in_archive(capture / "conversation.jsonl", backend, marker)
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
    # Keep a tail fallback for runtimes without capture, including final-result
    # payloads; successful normal execution uses the full archive above.
    for line in result.stdout_tail.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        replied = replied or _has_reply(backend, event, marker)
    if not result.timed_out and result.exit_status == 0 and replied:
        return
    reason = (f"timed out after {timeout_s}s" if result.timed_out else
              f"exit={result.exit_status}" if result.exit_status else "no valid assistant reply")
    details = result.stderr_tail.strip() or result.stdout_tail.strip()
    raise RuntimeError(f"Model preflight failed: backend={backend}, model={model}, {reason}\n{details}")
