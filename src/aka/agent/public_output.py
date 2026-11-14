"""Agent-owned durable normalized invocation facts, independent of CLI layout."""

from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import uuid

from aka.contracts.agent import AgentObservation


def record_invocation(
    workspace,
    runtime,
    session_id,
    usage,
    events,
    errors=(),
    *,
    resume_supported=False,
    stdout="",
):
    root = Path(workspace) / ".aka-agent-observations"
    value = {
        "schema": 1,
        "runtime": runtime,
        "session_id": session_id,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "usage": asdict(usage),
        "events": [asdict(event) for event in events],
        "errors": list(errors),
        "resume_supported": resume_supported,
    }
    value["events"].extend(_conversation_events(runtime, stdout, len(value["events"])))
    path = root / (str(uuid.uuid4()) + ".json")
    temporary = path.with_suffix(".tmp-" + str(os.getpid()))
    from aka.core.filesystem import open_private_directory
    with open_private_directory(root) as directory:
        fd = os.open(temporary.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                     0o600, dir_fd=directory)
        try:
            with os.fdopen(fd, "w") as stream:
                stream.write(json.dumps(value, allow_nan=False) + "\n")
            os.replace(temporary.name, path.name, src_dir_fd=directory, dst_dir_fd=directory)
        finally:
            try:
                os.unlink(temporary.name, dir_fd=directory)
            except FileNotFoundError:
                pass


def _observation(content):
    try:
        value = json.loads(content)
        if type(value["schema"]) is not int or value["schema"] != 1:
            raise ValueError("unsupported Agent observation schema")
        if (not all(isinstance(value[key], str) for key in ("session_id", "runtime", "updated_at"))
                or not isinstance(value["usage"], dict) or not isinstance(value["events"], list)
                or any(not isinstance(event, dict) for event in value["events"])
                or type(value["resume_supported"]) is not bool):
            raise ValueError("invalid Agent observation fields")
        return AgentObservation(value["session_id"], value["runtime"],
            "observation_error" if value.get("errors") else "observed",
            value["updated_at"], value["usage"], tuple(value["events"]), value["resume_supported"])
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError("Agent observation is corrupt") from exc


def _records(root):
    from .execution.home import read_regular_file
    if root.is_symlink() or root.resolve() != root.absolute():
        raise ValueError("Agent observation storage must not traverse symlinks")
    result = []
    for path in sorted(root.glob("*.json")):
        content, info = read_regular_file(path)
        if info.st_nlink != 1:
            raise ValueError("Agent observations must not have hardlink aliases")
        result.append((path.name, content, _observation(content)))
    return result


def read_sessions(workspace, *, direct=False):
    rows = []
    root = Path(workspace) if direct else Path(workspace) / ".aka-agent-observations"
    rows.extend(row for _, _, row in _records(root))
    return tuple(sorted(rows, key=lambda row: (row.updated_at or "", row.session_id)))


def export_observations(workspace, destination):
    """Export validated records without exposing the live storage layout."""
    from aka.core.filesystem import open_private_directory

    destination = Path(os.path.abspath(destination))
    if destination.exists() or destination.is_symlink():
        raise ValueError("Agent observation destination must be new")
    root = Path(os.path.abspath(workspace)) / ".aka-agent-observations"
    if destination.absolute().is_relative_to(root.absolute()):
        raise ValueError("Agent observation export overlaps live storage")
    contents = _records(root)
    with open_private_directory(destination.parent) as parent:
        os.mkdir(destination.name, mode=0o700, dir_fd=parent)
        directory = os.open(destination.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
        try:
            for name, content, _ in contents:
                fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                             0o600, dir_fd=directory)
                with os.fdopen(fd, "wb") as stream:
                    stream.write(content)
        finally:
            os.close(directory)
    return destination


def _conversation_events(runtime, stdout, sequence):
    """Normalize Agent-visible conversation content; never export native records."""
    result = []
    for line in stdout.splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if not isinstance(record, dict):
            continue
        if runtime == 'dsh':
            event_type = record.get('type')
            if event_type in {'text', 'thinking', 'tool_call', 'tool_result'}:
                content = record.get('text', record.get('input', record.get('result', '')))
                result.append({
                    'sequence': sequence + len(result),
                    'kind': {'text': 'message', 'thinking': 'message'}.get(event_type, event_type),
                    'role': 'assistant' if event_type in {'text', 'thinking'} else 'tool',
                    'tool_name': record.get('tool'), 'tool_id': record.get('callId'),
                    'content': (content if isinstance(content, str)
                                else json.dumps(content, ensure_ascii=False))[:16000],
                })
            continue
        message = record.get("message")
        blocks = message.get("content", []) if isinstance(message, dict) else []
        if isinstance(blocks, str):
            blocks = [{"type": "text", "text": blocks}]
        if runtime == "codex":
            item = record.get("item") or {}
            if isinstance(item, dict):
                if item.get("type") == "agent_message":
                    blocks = [{"type": "text", "text": item.get("text", "")}]
                elif item.get("type") == "command_execution":
                    blocks = [
                        {
                            "type": "tool_use",
                            "name": "command",
                            "id": item.get("id"),
                            "input": item.get("command"),
                        }
                    ]
        for block in blocks if isinstance(blocks, list) else []:
            if not isinstance(block, dict) or block.get("type") not in {
                "text",
                "tool_use",
                "tool_result",
            }:
                continue
            kind = {
                "text": "message",
                "tool_use": "tool_call",
                "tool_result": "tool_result",
            }[block["type"]]
            content = block.get("text", block.get("input", block.get("content", "")))
            rendered = (
                content
                if isinstance(content, str)
                else json.dumps(content, ensure_ascii=False)
            )
            result.append(
                {
                    "sequence": sequence + len(result),
                    "kind": kind,
                    "role": record.get("type"),
                    "tool_name": block.get("name"),
                    "tool_id": block.get("id", block.get("tool_use_id")),
                    "content": rendered[:16000],
                }
            )
    return result
