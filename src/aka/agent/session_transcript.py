"""Build one unredacted, backend-neutral conversation transcript."""

from __future__ import annotations

import base64
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from typing import Any

def record_provider_line(line: str) -> bool:
    """Drop only high-frequency Claude token-estimate telemetry from Session evidence."""
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        return True
    return not (
        isinstance(event, dict)
        and event.get("type") == "system"
        and event.get("subtype") == "thinking_tokens"
    )


def _record(sequence: int, record_type: str, source: str, **data: Any) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "sequence": sequence,
        "type": record_type,
        "source": source,
        **data,
    }


def initial_records(
    *,
    backend: str,
    session_id: str,
    prompt: str,
    system_prompt: str = "",
) -> list[dict[str, Any]]:
    """Describe the capture boundary and the exact Runtime-supplied initial message."""
    records = [
        _record(
            0,
            "session_start",
            "runtime",
            backend=backend,
            session_id=session_id,
            provider_system_prompt={
                "captured": False,
                "reason": "provider-managed system prompt is not exported by the CLI",
            },
            runtime_system_prompt={"supplied": bool(system_prompt)},
        ),
    ]
    if system_prompt:
        records.append(
            _record(
                len(records),
                "message",
                "runtime_input",
                role="system",
                content=[{"type": "text", "text": system_prompt}],
            )
        )
    records.append(
        _record(
            len(records),
            "message",
            "runtime_input",
            role="user",
            content=[{"type": "text", "text": prompt}],
        )
    )
    return records


def provider_line_record(
    sequence: int,
    *,
    path: str,
    line: str,
) -> dict[str, Any]:
    """Retain one Provider line as parsed JSON or exact text when it is not JSON."""
    try:
        event = json.loads(line)
        json.dumps(event, ensure_ascii=False, allow_nan=False)
    except (json.JSONDecodeError, TypeError, ValueError):
        return _record(
            sequence,
            "provider_text",
            "provider",
            path=path,
            text=line,
        )
    return _record(
        sequence,
        "provider_event",
        "provider",
        path=path,
        event=event,
    )


def provider_bytes_records(
    sequence: int,
    *,
    path: str,
    payload: bytes,
) -> tuple[list[dict[str, Any]], int]:
    """Project a Provider-owned transcript file without dropping non-UTF-8 data."""
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError:
        return (
            [
                _record(
                    sequence,
                    "provider_binary",
                    "provider",
                    path=path,
                    encoding="base64",
                    content=base64.b64encode(payload).decode("ascii"),
                )
            ],
            sequence + 1,
        )
    records: list[dict[str, Any]] = []
    for line in text.splitlines():
        records.append(provider_line_record(sequence, path=path, line=line))
        sequence += 1
    return records, sequence


def terminal_record(
    sequence: int,
    *,
    state: str,
    exit_status: int | None,
    timed_out: bool | None,
    raw_provider_capture_complete: bool,
    error_type: str | None = None,
) -> dict[str, Any]:
    value = _record(
        sequence,
        "session_end",
        "runtime",
        state=state,
        exit_status=exit_status,
        timed_out=timed_out,
        raw_provider_capture_complete=raw_provider_capture_complete,
    )
    if error_type is not None:
        value["error_type"] = error_type
    return value


def encode_records(records: Iterable[dict[str, Any]]) -> str:
    return "".join(
        json.dumps(record, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n"
        for record in records
    )


def _fingerprint(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def _content_blocks(event: dict[str, Any]) -> Counter[str] | None:
    message = event.get("message")
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]
    if not isinstance(content, list) or not content:
        return None
    return Counter(_fingerprint(block) for block in content)


def _compact_claude_records(
    stream: list[dict[str, Any]],
    native: list[dict[str, Any]],
    prompt: str,
    *,
    factory=list,
):
    """Use native content once, falling back to stdout when coverage is uncertain.

    Native content blocks are never collapsed by message ID. Only the duplicate
    stdout representation is removed; counters and the audit files are untouched.
    """
    main_path = "provider/claude-session.raw-jsonl"
    housekeeping = {"queue-operation", "last-prompt", "ai-title", "file-history-snapshot"}
    main = factory()
    other = factory()
    for row in native:
        path = str(row.get("path", ""))
        event = row.get("event")
        is_claude = path == main_path or path.startswith("provider/claude-subagents/")
        if (
            is_claude
            and isinstance(event, dict)
            and (
                event.get("type") in housekeeping
                or (event.get("type") == "system" and event.get("subtype") == "thinking_tokens")
            )
        ):
            continue
        (main if path == main_path else other).append(row)

    # Match only the main session. Subagent message IDs are a separate conversation.
    assistants: dict[str, tuple[int, Counter[str]]] = {}
    fingerprints: dict[str, list[int]] = defaultdict(list)
    user_content: dict[str, list[int]] = defaultdict(list)
    initial_prompt_index = None
    first_user_seen = False
    for index, row in enumerate(main):
        event = row.get("event")
        if not isinstance(event, dict):
            continue
        fingerprints[_fingerprint(event)].append(index)
        blocks = _content_blocks(event)
        message = event.get("message")
        if event.get("type") == "assistant" and isinstance(message, dict) and blocks:
            first_user_seen = True
            message_id = message.get("id")
            if isinstance(message_id, str):
                previous = assistants.get(message_id, (index, Counter()))[1]
                assistants[message_id] = (index, previous | blocks)
        if event.get("type") == "user" and blocks:
            user_content[json.dumps(sorted(blocks.items()))].append(index)
            if not first_user_seen:
                initial_prompt_index = (
                    index if blocks == _content_blocks({"message": {"content": prompt}}) else None
                )
                first_user_seen = True

    # Place stdout-only diagnostics near the last matched native event, keeping
    # unmatched content on partial/crashed sessions instead of assuming a full ledger.
    extras = defaultdict(factory)
    terminal = factory()
    matched_users: set[int] = set()
    terminal_fingerprints: set[str] = set()
    anchor = -1
    for row in stream:
        event = row.get("event")
        match = None
        if isinstance(event, dict):
            if event.get("type") == "result":
                terminal.append(row)
                terminal_fingerprints.add(_fingerprint(event))
                continue
            message = event.get("message")
            blocks = _content_blocks(event)
            if not event.get("parent_tool_use_id") and not any(
                event.get(key) for key in ("error", "errors", "is_error")
            ):
                if event.get("type") == "assistant" and isinstance(message, dict) and blocks:
                    message_id = message.get("id")
                    entry = assistants.get(message_id) if isinstance(message_id, str) else None
                    if entry is not None and all(entry[1][key] >= n for key, n in blocks.items()):
                        match = entry[0]
                elif event.get("type") == "user" and blocks:
                    match = next(
                        (
                            i
                            for i in user_content.get(json.dumps(sorted(blocks.items())), [])
                            if i not in matched_users and i >= anchor
                        ),
                        None,
                    )
                    if match is not None:
                        matched_users.add(match)
            if match is None:
                match = next(iter(fingerprints.get(_fingerprint(event), [])), None)
        if match is None:
            extras[anchor].append(row)
        else:
            anchor = max(anchor, match)
    rows = factory()
    rows.extend(extras[-1])
    for index, row in enumerate(main):
        event = row.get("event")
        duplicate_terminal = (
            isinstance(event, dict)
            and event.get("type") == "result"
            and _fingerprint(event) in terminal_fingerprints
        )
        if index != initial_prompt_index and not duplicate_terminal:
            rows.append(row)
        rows.extend(extras[index])
    from itertools import chain
    return list(chain(rows, other, terminal)) if factory is list else chain(rows, other, terminal)


def render_conversation(
    *,
    backend: str,
    session_id: str,
    prompt: str,
    stdout: str,
    raw_provider_files: Iterable[tuple[str, bytes]],
    state: str,
    exit_status: int | None,
    timed_out: bool | None,
    raw_provider_capture_complete: bool,
    error_type: str | None = None,
    system_prompt: str = "",
    stderr: str = "",
    started_at: str | None = None,
    context: dict[str, Any] | None = None,
    capture_errors: list[str] | None = None,
) -> str:
    """Render a reading view; original Provider streams remain separate audit files."""
    records = initial_records(
        backend=backend,
        session_id=session_id,
        prompt=prompt,
        system_prompt=system_prompt,
    )
    if started_at is not None:
        records[0]["started_at"] = started_at
    if context is not None:
        records[0]["context"] = context
    initial_count = len(records)
    sequence = initial_count
    for line in stdout.splitlines():
        if not record_provider_line(line):
            continue
        records.append(
            provider_line_record(
                sequence,
                path="provider/stdout.stream-json",
                line=line,
            )
        )
        sequence += 1
    native: list[dict[str, Any]] = []
    for path, payload in raw_provider_files:
        projected, sequence = provider_bytes_records(
            sequence,
            path=path,
            payload=payload,
        )
        native.extend(projected)
    if backend == "claude":
        records = records[:initial_count] + _compact_claude_records(
            records[initial_count:],
            native,
            prompt,
        )
    else:
        records.extend(native)
    for line in stderr.splitlines():
        records.append(provider_line_record(0, path="provider/stderr.log", line=line))
    for index, record in enumerate(records):
        record["sequence"] = index
    sequence = len(records)
    terminal = terminal_record(
        sequence,
        state=state,
        exit_status=exit_status,
        timed_out=timed_out,
        raw_provider_capture_complete=raw_provider_capture_complete,
        error_type=error_type,
    )
    terminal.update(capture_errors=list(capture_errors or []))
    records.append(terminal)
    return encode_records(records)


class _DiskRecords:
    """A sequence of JSON records backed by a shared temporary SQLite database."""

    def __init__(self, db, bucket):
        self.db, self.bucket, self.size = db, bucket, 0

    def append(self, value):
        self.db.execute("INSERT INTO records VALUES (?, ?, ?)",
                        (self.bucket, self.size, json.dumps(value, ensure_ascii=False)))
        self.size += 1

    def extend(self, rows):
        for row in rows:
            self.append(row)

    def __iter__(self):
        for (text,) in self.db.execute(
            "SELECT value FROM records WHERE bucket=? ORDER BY ordinal", (self.bucket,)
        ):
            yield json.loads(text)


def write_conversation(*, destination, backend, session_id, prompt, stdout,
                       raw_provider_files, state, exit_status, timed_out,
                       raw_provider_capture_complete, stderr=(), started_at=None,
                       context=None, capture_errors=None):
    """Finalize the reading projection without retaining transcript bodies in RAM.

    Keep the existing Claude matching/order rules. Only compact fingerprints
    and message identities stay resident; event bodies and extras live on disk.
    """
    import sqlite3
    import tempfile
    from pathlib import Path
    from itertools import count, chain

    destination = Path(destination)
    temporary = destination.with_name(destination.name + ".tmp")
    with tempfile.TemporaryDirectory(prefix="aka-transcript-", dir=destination.parent) as directory:
        db = sqlite3.connect(str(Path(directory) / "records.sqlite"))
        try:
            db.execute("PRAGMA journal_mode=OFF")
            db.execute("PRAGMA cache_size=-2048")
            db.execute("CREATE TABLE records (bucket INTEGER, ordinal INTEGER, value TEXT, PRIMARY KEY(bucket,ordinal))")
            buckets = count()
            def factory():
                return _DiskRecords(db, next(buckets))
            stream, native = factory(), factory()
            for line in stdout:
                if record_provider_line(line):
                    stream.append(provider_line_record(0, path="provider/stdout.stream-json", line=line.rstrip("\r\n")))
            for name, path in raw_provider_files:
                # Match provider_bytes_records' binary fallback, but never read
                # an ordinary JSONL transcript in one allocation.
                start = native.size
                try:
                    with path.open(encoding="utf-8") as source:
                        for line in source:
                            native.append(provider_line_record(0, path=name, line=line.rstrip("\r\n")))
                except UnicodeDecodeError:
                    db.execute("DELETE FROM records WHERE bucket=? AND ordinal>=?", (native.bucket, start))
                    native.size = start
                    projected, _ = provider_bytes_records(0, path=name, payload=path.read_bytes())
                    native.extend(projected)
            initial = initial_records(backend=backend, session_id=session_id, prompt=prompt)
            if started_at is not None:
                initial[0]["started_at"] = started_at
            if context is not None:
                initial[0]["context"] = context
            body = (_compact_claude_records(stream, native, prompt, factory=factory)
                    if backend == "claude" else chain(stream, native))
            terminal = terminal_record(0, state=state, exit_status=exit_status,
                timed_out=timed_out, raw_provider_capture_complete=raw_provider_capture_complete)
            terminal["capture_errors"] = list(capture_errors or [])
            errors = (provider_line_record(0, path="provider/stderr.log", line=line.rstrip("\r\n")) for line in stderr)
            with temporary.open("w", encoding="utf-8") as output:
                for index, row in enumerate(chain(initial, body, errors, [terminal])):
                    row["sequence"] = index
                    output.write(encode_records([row]))
            temporary.replace(destination)
        finally:
            db.close()
            temporary.unlink(missing_ok=True)
