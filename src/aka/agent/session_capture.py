"""Live conversation/usage copies, independent of CLI Home and workflow.

Native files are data, not Runtime session envelopes. High resource ceilings
bound capture, not Agent execution: exceeding them makes evidence partial.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import threading
import tempfile
import struct
import time
import uuid
from contextvars import ContextVar
from collections import OrderedDict
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from aka.agent.session_tail import CaptureBudget, CaptureLimits, TranscriptTail
from aka.agent.session_transcript import (
    encode_records,
    initial_records,
    provider_line_record,
    record_provider_line,
)
from aka.agent.session_usage import (
    UsageAccumulator,
    estimate_usage_from_conversation,
)

if TYPE_CHECKING:
    from aka.agent.model import AgentRuntimeCapabilities, NormalizedAgentEvent, TokenUsage


class CapturedObservation(NamedTuple):
    """Accounting observation plus capture health, independent of warning wording."""

    events: tuple[NormalizedAgentEvent, ...]
    terminal_usage: TokenUsage
    capabilities: AgentRuntimeCapabilities
    errors: tuple[str, ...]


class CompleteSessionArchive(NamedTuple):
    """One finalized, Supervisor-owned Session capture."""

    root: Path
    conversation: Path
    token_usage: Path


def latest_complete_session_archive(root: Path) -> CompleteSessionArchive | None:
    """Return the newest finalized capture below ``root``.

    A Review is auditable only when both projections exist.  The final token
    receipt is the commit marker written after the finalized conversation, so
    live/partial captures are deliberately ignored.
    """
    root = Path(root)
    if not root.is_dir() or root.is_symlink():
        return None
    candidates: list[tuple[int, str, CompleteSessionArchive]] = []
    for directory in root.glob("run-*"):
        if not directory.is_dir() or directory.is_symlink():
            continue
        conversation = directory / "conversation.jsonl"
        token_usage = directory / "token-usage.json"
        if any(
            not path.is_file() or path.is_symlink()
            for path in (conversation, token_usage)
        ):
            continue
        try:
            with conversation.open("rb") as stream:
                stream.seek(0, os.SEEK_END)
                size = stream.tell()
                stream.seek(max(0, size - 1024 * 1024))
                terminal_lines = stream.read().splitlines()
            terminal = json.loads(terminal_lines[-1]) if terminal_lines else None
            if not isinstance(terminal, dict) or terminal.get("type") != "session_end":
                continue
            usage = json.loads(token_usage.read_text(encoding="utf-8"))
            if (
                not isinstance(usage, dict)
                or usage.get("state")
                not in {"completed", "failed", "timed_out", "interrupted"}
                or not isinstance(usage.get("finished_at"), str)
                or not usage["finished_at"]
            ):
                continue
            modified = max(
                conversation.stat().st_mtime_ns,
                token_usage.stat().st_mtime_ns,
            )
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        candidates.append(
            (
                modified,
                directory.name,
                CompleteSessionArchive(directory, conversation, token_usage),
            )
        )
    return max(candidates, key=lambda row: row[:2])[2] if candidates else None


_LAST_CAPTURE: ContextVar[tuple[str, dict] | None] = ContextVar("aka_session_capture", default=None)

# Repeated invocations of a persistent provider Session seed usage from compact
# counters/IDs. Only newly appended transcript bytes are parsed again. A process
# restart or replaced transcript performs one bounded rebuild.
_HISTORY_INDEX = OrderedDict()
_HISTORY_INDEX_LOCK = threading.Lock()
_HISTORY_INDEX_LIMIT = 32


def _claude_root_output_limit(text: str) -> bool:
    """Recognize the native root response that precedes StopFailure."""
    for line in text.splitlines():
        try:
            event = json.loads(line)
        except (TypeError, ValueError):
            continue
        if (
            not isinstance(event, dict)
            or event.get("type") != "assistant"
            or event.get("parent_tool_use_id") is not None
        ):
            continue
        message = event.get("message")
        if isinstance(message, dict) and message.get("stop_reason") == "max_tokens":
            return True
        if event.get("isApiErrorMessage") is True and {
            event.get("error"), event.get("api_error")
        } & {"max_output_tokens"}:
            return True
    return False


def clear_capture() -> None:
    _LAST_CAPTURE.set(None)


def captured_observation(stdout: str, events, capabilities) -> CapturedObservation | None:
    """Use the current invocation's native accounting, never an unrelated run."""
    from aka.agent.model import NormalizedAgentEvent, TokenUsage, resequence_agent_events

    captured = _LAST_CAPTURE.get()
    if not captured or captured[0] != hashlib.sha256(stdout.encode()).hexdigest():
        return None
    report = captured[1]
    total = report["total"]
    if total["total_tokens"] is None:
        return None
    # Legacy phases still depend on stream order. Native-only responses have no
    # reliable position among those markers; do not fabricate a phase. Claude's
    # total accumulates main responses and excludes separately observed child counters.
    # Full per-response counters remain in token-usage.json for every backend.
    if report["backend"] == "claude":
        from aka.agent.adapter import ClaudeAdapter
        settled = {
            row["message_id"]: TokenUsage(**row["usage"])
            for row in report["responses"]
            if row["agent"] == "main" and row["path"] != "provider/stdout.stream-json"
        }
        events, _ = ClaudeAdapter().normalize_stream(stdout, settled_usage=settled)
    normalized = [event for event in events if event.kind != "terminal_usage"]
    if not normalized:
        normalized.extend(
            NormalizedAgentEvent(0, "usage_delta", TokenUsage(**row["usage"]))
            for row in report["responses"]
            if report["backend"] != "claude" or row["agent"] == "main"
        )
    usage = TokenUsage(**total)
    normalized.append(NormalizedAgentEvent(0, "terminal_usage", usage))
    capabilities = replace(
        capabilities,
        usage_delta_observed=any(row.kind == "usage_delta" for row in normalized),
    )
    return CapturedObservation(
        resequence_agent_events(normalized),
        usage,
        capabilities,
        tuple([*report["warnings"], *report.get("capture_errors", [])]),
    )


def _atomic(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        output.write(text)
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


def start_session_capture(command: list[str], cwd: Path, environment: dict[str, str]):
    """Observe an Agent invocation without changing its command or environment."""
    clear_capture()
    # Check before path resolution, native discovery, or capture construction.
    # Clearing the prior observation also keeps disabled invocations on the
    # existing stream/ledger path, even if their stdout matches an earlier run.
    if environment.get("ATREX_SESSION_CAPTURE", "1").strip() == "0":
        return None
    backend = environment.get("ATREX_AGENT_CLI") or (Path(command[0]).name if command else "")
    if backend not in {"claude", "codex", "qodercli", "pi", "dsh"}:
        return None
    # Outside the Git workspace: traces must not enter baseline/candidate commits.
    configured = environment.get("ATREX_SESSION_CAPTURE_DIR")
    root = (
        Path(configured).expanduser()
        if configured
        else (cwd.resolve().parent / ".atrex-session-traces" / cwd.resolve().name)
    )
    if not root.is_absolute():
        root = cwd / root
    context = {"workspace": str(cwd.resolve())}
    for label in ("CAMPAIGN_ID", "ITERATION_ID", "ATTEMPT_ID"):
        value = environment.get("ATREX_TELEMETRY_" + label)
        if value:
            context[label.lower()] = value
    try:
        return SessionCapture(
            root,
            backend=backend,
            command=command,
            context=context,
            native_environment=environment,
        )
    except Exception as error:
        print(f"[orchestrator] session capture setup failed at {root}: {error}", flush=True)
        return None


def finish_session_capture(capture, **status) -> None:
    """Persistence failure cannot replace the process result or kill the CLI."""
    if capture is None:
        return
    try:
        capture.finish(**status)
    except Exception as error:
        capture._capture_error("final_capture", error)
        # Keep structured partial accounting even if final disk writes failed.
        try:
            with capture._lock:
                capture._sync_stdout_usage()
                report = capture._usage.report(finished=False)
                report.update(capture_errors=list(capture.errors))
                _LAST_CAPTURE.set((capture._chunks["stdout"].digest.hexdigest(), report))
        except Exception:
            # Accounting can fail too; retain the original CLI outcome and
            # leave the existing stream/ledger parser available as a fallback.
            clear_capture()
        finally:
            with capture._lock:
                capture.finished = True
                for spool in (*capture._chunks.values(), *capture._retained_chunks.values()):
                    spool.close()
        print(f"[orchestrator] session capture failed at {capture.root}: {error}", flush=True)


class _OutputSpool:
    """Append-only UTF-8 stream on disk; cursors never retain consumed text."""

    def __init__(self, directory):
        self.file = tempfile.TemporaryFile(mode="w+b", dir=directory)
        self.digest = hashlib.sha256()
        self.cursor = 0

    def append(self, text):
        data = text.encode("utf-8")
        self.file.seek(0, os.SEEK_END)
        self.file.write(struct.pack("!Q", len(data)))
        self.file.write(data)
        self.digest.update(data)

    def unread(self):
        self.file.seek(self.cursor)
        while header := self.file.read(8):
            data = self.file.read(struct.unpack("!Q", header)[0])
            self.cursor = self.file.tell()
            yield data.decode("utf-8")

    def __iter__(self):
        self.file.seek(0)
        while header := self.file.read(8):
            yield self.file.read(struct.unpack("!Q", header)[0]).decode("utf-8")

    def lines(self):
        # Pipe reads may split oversized records. Reconstruct lines without
        # keeping an entire invocation resident.
        pending = ""
        for chunk in self:
            lines = (pending + chunk).splitlines(keepends=True)
            pending = ""
            if lines and not lines[-1].endswith(("\n", "\r")):
                pending = lines.pop()
            yield from lines
        if pending:
            yield pending

    def text(self):
        # Compatibility boundary: process runners still return complete strings.
        import io
        output = io.StringIO()
        for chunk in self:
            output.write(chunk)
        return output.getvalue()

    def close(self):
        self.file.close()


class SessionCapture:
    def __init__(
        self,
        root: Path,
        *,
        backend: str,
        command: list[str],
        context: dict[str, str],
        native_environment: dict[str, str] | None = None,
        limits: CaptureLimits | None = None,
    ) -> None:
        self.root = root.resolve() / ("run-" + uuid.uuid4().hex)
        self.root.mkdir(parents=True, mode=0o700)
        (self.root / "provider").mkdir()
        self.backend = backend
        self.session_id = self.root.name
        session_options = ("--session-id", "--session") if backend == "pi" else ("--session-id", "--resume")
        for flag in session_options:
            if flag in command and command.index(flag) + 1 < len(command):
                self.session_id = command[command.index(flag) + 1]
                break
        # All supported adapters pass their exact initial/resume prompt last.
        self.prompt = command[-1] if backend and command else ""
        self.context = context
        self.started_at = datetime.now(UTC).isoformat()
        self._lock = threading.RLock()
        self._stop = threading.Event()
        # This is a functional fallback for Claude versions or compatible
        # endpoints that persist max_tokens but fail to emit StopFailure.
        self.output_limit_detected = threading.Event()
        self._readers: list[threading.Thread] = []
        self._monitor: threading.Thread | None = None
        # Functional output has the same full-stream contract as Popen.communicate.
        # Neither diagnostic limits nor native/history reads may truncate it.
        self._chunks = {name: _OutputSpool(self.root) for name in ("stdout", "stderr")}
        self._retained_chunks = {name: _OutputSpool(self.root) for name in ("stdout", "stderr")}
        self._pipe_bytes = {"stdout": 0, "stderr": 0}
        self._raw_pipe_bytes = {"stdout": 0, "stderr": 0}
        self.errors: list[str] = []
        self._output_error: OSError | None = None
        self._budget = CaptureBudget(limits or CaptureLimits(), self._capture_error)
        # History scanning is bounded independently. Old bytes/records never
        # consume the invocation's diagnostic retention allowance.
        self._history_budget = CaptureBudget(
            self._budget.limits,
            lambda label, error: self._capture_error("resume_history_" + label, error),
        )
        self._usage = UsageAccumulator(backend)
        self._usage_dirty = True
        self._tails: dict[str, TranscriptTail] = {}
        self._history_seeds = {}
        self._initial_sizes: dict[str, int] = {}
        self._host_transcripts = None
        if native_environment is not None:
            from aka.agent.session_native import HostSessionTranscripts

            try:
                self._host_transcripts = HostSessionTranscripts(
                    backend,
                    native_environment,
                    command,
                    max_files=self._budget.limits.files,
                    on_limit=lambda: self._budget.warning("native_files_exceeded"),
                )
            except (OSError, ValueError) as error:
                self._capture_error("native_capture_setup", error)
        for name, (path, previous_size) in self._native_paths().items():
            try:
                size = previous_size if self._host_transcripts else path.stat().st_size
                self._initial_sizes[name] = size
                self._tail(name, path, size)
            except (OSError, ValueError) as error:
                self._capture_error("native_capture_setup", error)
        self.native: dict[str, Path] = {}
        self._sequence = 0
        self.finished = False
        initial = initial_records(backend=backend, session_id=self.session_id, prompt=self.prompt)
        initial[0].update(started_at=self.started_at, context=context)
        (self.root / "conversation.jsonl").write_text(encode_records(initial), encoding="utf-8")
        self._sequence = len(initial)
        self._write_usage(False)

    def _native_paths(self, *, force: bool = False) -> dict[str, tuple[Path, int]]:
        if self._host_transcripts is not None:
            try:
                return self._host_transcripts.selected(force=force)
            except (OSError, ValueError) as error:
                self._capture_error("native_capture", error)
                return {}
        return {}

    def _tail(self, name: str, path: Path, previous_size: int) -> TranscriptTail | None:
        if name not in self._tails:
            if len(self._tails) >= self._budget.limits.files:
                self._budget.warning("native_files_exceeded")
                return None
            tail = TranscriptTail()
            key = (self.backend, str(path.absolute()))
            status = path.stat()
            identity = (status.st_dev, status.st_ino)
            with _HISTORY_INDEX_LOCK:
                cached = _HISTORY_INDEX.get(key)
            seed = UsageAccumulator(self.backend)
            start = 0
            if cached and cached[0] == identity and cached[1] <= previous_size:
                start = cached[1]
                seed.old_ids.update(cached[2])
                seed.prior_codex.update(cached[3])
                if self.backend == "dsh":
                    seed.dsh_sessions.update(cached[4])
                    seed.dsh_seeded.update(cached[5])
            try:
                for line in tail.resume_history(path, self._history_budget, previous_size,
                                                start=start, identity=identity):
                    seed.feed(name, line.decode("utf-8", errors="replace"), previous=True)
                self._usage.old_ids.update(seed.old_ids)
                self._usage.prior_codex.update(seed.prior_codex)
                if self.backend == "dsh":
                    self._usage.dsh_sessions.update(seed.dsh_sessions)
                    self._usage.dsh_seeded.update(seed.dsh_seeded)
            finally:
                if seed.missing:
                    tail.history_incomplete = True
                if tail.history_incomplete:
                    self._usage.mark_history_incomplete(name)
            # Publish only after stat and history initialization succeed. A
            # transient IO error must leave the next capture pass free to retry.
            self._history_seeds[name] = (key, seed)
            self._tails[name] = tail
        return self._tails[name]

    def _checkpoint_history(self, name, tail):
        if tail.invalidated or tail.history_incomplete or tail.stopped or tail.dropping_line:
            return
        key, seed = self._history_seeds[name]
        # Incomplete trailing bytes are rescanned once, never cached as a record.
        offset = tail.offset - len(tail.pending)
        value = (tail.identity, offset, frozenset(seed.old_ids), dict(seed.prior_codex),
                 dict(getattr(seed, "dsh_sessions", {})), dict(getattr(seed, "dsh_seeded", {})))
        with _HISTORY_INDEX_LOCK:
            old = _HISTORY_INDEX.get(key)
            if old is None or old[0] != value[0] or old[1] <= offset:
                _HISTORY_INDEX[key] = value
                _HISTORY_INDEX.move_to_end(key)
            while len(_HISTORY_INDEX) > _HISTORY_INDEX_LIMIT:
                _HISTORY_INDEX.popitem(last=False)

    def _feed_usage(self, path: str, text: str, *, previous: bool = False) -> None:
        if (
            self.backend == "claude"
            and not previous
            and _claude_root_output_limit(text)
        ):
            self.output_limit_detected.set()
        try:
            self._usage.feed(path, text, previous=previous)
            self._usage_dirty = True
        except Exception as error:
            # Provider/native JSON is untrusted data. A malformed counter cannot
            # unwind a pipe reader or stop the other transcripts being captured.
            self._capture_error("usage_capture", error)

    def _sync_stdout_usage(self) -> None:
        for text in self._chunks["stdout"].unread():
            self._feed_usage("provider/stdout.stream-json", text)
            if self.backend == "dsh":
                from aka.agent.dsh import session_id_from_stream
                identity = session_id_from_stream(text)
                if identity:
                    self.session_id = identity
                    if self._host_transcripts:
                        self._host_transcripts.session_id = identity
            if self.backend == "codex":
                from aka.agent.codex_ledger import codex_thread_id_from_stream

                try:
                    identity = codex_thread_id_from_stream(text)
                except Exception as error:
                    self._capture_error("codex_session_identity", error)
                    continue
                if identity:
                    self.session_id = identity
                    if self._host_transcripts:
                        self._host_transcripts.session_id = identity

    def _append(self, path: str, line: str) -> None:
        row = provider_line_record(self._sequence, path=path, line=line.rstrip("\n"))
        self._sequence += 1
        with (self.root / "conversation.jsonl").open("a", encoding="utf-8") as output:
            output.write(encode_records([row]))

    def sync_native(self, *, final: bool = False) -> None:
        with self._lock:
            if self.finished:
                return
            self._sync_stdout_usage()
            for name, (path, previous_size) in self._native_paths(force=final).items():
                try:
                    tail = self._tail(name, path, previous_size)
                    if tail is None:
                        continue
                    for addition in tail.read(path, self._budget, final=final):
                        text = addition.decode("utf-8", errors="replace")
                        self._feed_usage(name, text)
                        self._history_seeds[name][1].feed(name, text, previous=True)
                        destination = self.root / name
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        with destination.open("ab") as output:
                            output.write(addition)
                        self.native[name] = destination
                        if record_provider_line(text):
                            self._append(name, text)
                    self._checkpoint_history(name, tail)
                except (OSError, ValueError) as error:
                    self._capture_error("native_capture", error)
            self._write_usage(False)

    def _write_usage(self, finished: bool) -> dict:
        self._sync_stdout_usage()
        if not finished and not self._usage_dirty:
            return {}
        report = self._usage.report(finished=finished and not self.errors)
        report.update(
            session_id=self.session_id,
            started_at=self.started_at,
            state="finished" if finished else "running",
            context=self.context,
            capture_errors=list(self.errors),
        )
        persisted = report if finished else dict(report)
        if not finished and len(report.get("responses", [])) > 100:
            persisted["responses"] = report["responses"][-100:]
            persisted["responses_truncated"] = True
        _atomic(self.root / "token-usage.json", json.dumps(persisted, ensure_ascii=False, indent=2) + "\n")
        self._usage_dirty = False
        return report

    def _capture_error(self, label: str, error: Exception) -> None:
        warning = f"{label}:{type(error).__name__}:{error}"
        with self._lock:
            if warning not in self.errors and len(self.errors) < 100:
                self.errors.append(warning)

    def output_sizes(self) -> tuple[int, int]:
        """Raw pipe bytes, including data beyond the capture retention budget."""
        with self._lock:
            return self._raw_pipe_bytes["stdout"], self._raw_pipe_bytes["stderr"]

    def _read_pipe(self, name: str, pipe) -> None:
        path = "provider/" + ("stdout.stream-json" if name == "stdout" else "stderr.log")
        output = None
        conversation_enabled = True

        def close_output() -> None:
            nonlocal output
            sink, output = output, None
            if sink is not None:
                try:
                    sink.close()
                except Exception as error:
                    # Closing a buffered sink can itself flush and fail. It must
                    # not unwind the reader or prevent the other sink from working.
                    self._capture_error(f"{name}_capture_close", error)

        try:
            try:
                output = (self.root / path).open("a", encoding="utf-8")
            except Exception as error:
                self._capture_error(f"{name}_capture_open", error)
            # Drain to EOF even if *all* persistence sinks have failed. The CLI
            # owns its lifetime; a full disk must not inject EPIPE/SIGPIPE into it.
            dropping_line = False
            drain_only = False
            for line in iter(lambda: pipe.readline(self._budget.limits.line_bytes + 1), ""):
                # Preserve even oversized-line fragments and output received
                # after a capture failure; joining reconstructs the original stream.
                with self._lock:
                    try:
                        self._chunks[name].append(line)
                    except OSError as error:
                        # A failed spool must not close the pipe on a live Agent.
                        # Drain to EOF, then report loss explicitly instead of
                        # returning an apparently complete functional stream.
                        self._output_error = error
                        self._capture_error(f"{name}_output_spool", error)
                    self._raw_pipe_bytes[name] += len(line.encode("utf-8", errors="replace"))
                if drain_only:
                    continue
                try:
                    with self._lock:
                        if dropping_line:
                            dropping_line = not line.endswith("\n")
                            continue
                        size = len(line.encode("utf-8", errors="replace"))
                        if size > self._budget.limits.line_bytes:
                            dropping_line = not line.endswith("\n")
                            self._budget.warning(f"{name}_line_bytes_exceeded")
                            continue
                        if name == "stdout":
                            try:
                                if not record_provider_line(line):
                                    continue
                            except Exception as error:
                                self._capture_error(f"{name}_capture_filter", error)
                                continue
                        if (
                            not self._budget.retain(size, file_used=self._pipe_bytes[name])
                            or not self._budget.record()
                        ):
                            continue
                        self._pipe_bytes[name] += size
                        # Only this bounded copy may enter persisted diagnostics.
                        # Keep it even if a disk sink fails, for final projection.
                        self._retained_chunks[name].append(line)
                        if output is not None:
                            try:
                                output.write(line)
                                output.flush()
                            except Exception as error:
                                self._capture_error(f"{name}_capture_write", error)
                                close_output()
                        if conversation_enabled:
                            try:
                                self._append(path, line)
                            except Exception as error:
                                self._capture_error(f"{name}_conversation_capture", error)
                                conversation_enabled = False
                except Exception as error:
                    # Last-resort capture boundary: even an unexpected parser or
                    # accounting failure must not close a live child pipe. Stop
                    # diagnostic processing, but preserve functional output to EOF.
                    self._capture_error(f"{name}_capture_processing", error)
                    drain_only = True
                    close_output()
        except Exception as error:
            self._capture_error(f"{name}_pipe_read", error)
        finally:
            close_output()
            try:
                pipe.close()
            except Exception as error:
                self._capture_error(f"{name}_pipe_close", error)

    def _poll(self) -> None:
        while not self._stop.wait(1):
            try:
                self.sync_native()
            except Exception as error:
                self._capture_error("native_capture", error)

    def communicate(self, process, timeout: float | None = None) -> tuple[str, str]:
        deadline = None if timeout is None else time.monotonic() + timeout
        if not self._readers:
            for name in ("stdout", "stderr"):
                reader = threading.Thread(
                    target=self._read_pipe, args=(name, getattr(process, name)), daemon=True
                )
                reader.start()
                self._readers.append(reader)
            self._monitor = threading.Thread(target=self._poll, daemon=True)
            self._monitor.start()
        process.wait(timeout=timeout)  # TimeoutExpired handled by the existing process guard.
        for reader in self._readers:
            # A descendant may still hold the pipe after the CLI exits. Normal
            # return requires EOF; a deadline covers both process and pipe wait.
            remaining = None if deadline is None else max(0, deadline - time.monotonic())
            reader.join(timeout=remaining)
            if reader.is_alive():
                raise subprocess.TimeoutExpired(process.args, timeout)
        with self._lock:
            if self._output_error is not None:
                raise OSError("Agent output spool failed; complete output is unavailable") from self._output_error
            return self._chunks["stdout"].text(), self._chunks["stderr"].text()

    def finish(
        self, *, exit_status: int | None = None, timed_out: bool = False, interrupted: bool = False
    ) -> None:
        if self.finished:
            return
        self._stop.set()
        if self._monitor:
            self._monitor.join(timeout=5)
        with self._lock:
            self.sync_native(final=True)
            stdout_digest = self._chunks["stdout"].digest.hexdigest()
            # sync_native already folded the functional stream and its identity.
            # Re-parsing it here would replay failed records without that guard.
            state = (
                "timed_out"
                if timed_out
                else "interrupted"
                if interrupted
                else "completed"
                if exit_status == 0
                else "failed"
            )
            # Reuse Runtime's reading projection; native subagents are not required
            # to have a synthetic session header. Original provider files remain.
            files = []
            for path, value in self.native.items():
                display = path
                if ".claude/projects/" in path:
                    display = (
                        "provider/claude-session.raw-jsonl"
                        if path.endswith(f"/{self.session_id}.jsonl")
                        else "provider/claude-subagents/" + path.split(".claude/projects/", 1)[1]
                    )
                files.append((display, value))
            from aka.agent.session_transcript import write_conversation
            write_conversation(
                destination=self.root / "conversation.jsonl",
                backend=self.backend,
                session_id=self.session_id,
                prompt=self.prompt,
                stdout=self._retained_chunks["stdout"].lines(),
                raw_provider_files=files,
                state=state,
                exit_status=exit_status,
                timed_out=timed_out,
                raw_provider_capture_complete=bool(self.native) and not self.errors,
                stderr=self._retained_chunks["stderr"].lines(),
                started_at=self.started_at,
                context=self.context,
                capture_errors=list(self.errors),
            )
            report = self._write_usage(True)
            total = report.get("total") if isinstance(report, dict) else None
            if not isinstance(total, dict) or total.get("total_tokens") is None:
                with (self.root / "conversation.jsonl").open(encoding="utf-8") as conversation:
                    estimated, estimate_detail = estimate_usage_from_conversation(conversation)
                if estimated.total_tokens is not None:
                    report["total"] = asdict(estimated)
                    report["accounting_basis"] = "conversation_text_estimate"
                    report["conversation_estimate"] = estimate_detail
                    warnings = report.setdefault("warnings", [])
                    warning = "provider_usage_unavailable; estimated_from_conversation"
                    if warning not in warnings:
                        warnings.append(warning)
            report.update(
                state=state,
                exit_status=exit_status,
                timed_out=timed_out,
                finished_at=datetime.now(UTC).isoformat(),
            )
            _atomic(
                self.root / "token-usage.json",
                json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            )
            _LAST_CAPTURE.set((stdout_digest, report))
            self.finished = True
            for spool in (*self._chunks.values(), *self._retained_chunks.values()):
                spool.close()
