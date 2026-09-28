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
import time
import uuid
from contextvars import ContextVar
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from .session_tail import CaptureBudget, CaptureLimits, TranscriptTail
from .session_transcript import (
    encode_records,
    initial_records,
    provider_line_record,
    record_provider_line,
    render_conversation,
)
from .session_usage import UsageAccumulator

if TYPE_CHECKING:
    from .agent_runtime.model import AgentRuntimeCapabilities, NormalizedAgentEvent, TokenUsage


class CapturedObservation(NamedTuple):
    """Accounting observation plus capture health, independent of warning wording."""

    events: tuple[NormalizedAgentEvent, ...]
    terminal_usage: TokenUsage
    capabilities: AgentRuntimeCapabilities
    errors: tuple[str, ...]
    capture_complete: bool


_LAST_CAPTURE: ContextVar[tuple[str, dict] | None] = ContextVar("aka_session_capture", default=None)


def clear_capture() -> None:
    _LAST_CAPTURE.set(None)


def captured_observation(stdout: str, events, capabilities) -> CapturedObservation | None:
    """Use the current invocation's native accounting, never an unrelated run."""
    from .agent_runtime.model import NormalizedAgentEvent, TokenUsage, resequence_agent_events

    captured = _LAST_CAPTURE.get()
    if not captured or captured[0] != hashlib.sha256(stdout.encode()).hexdigest():
        return None
    report = captured[1]
    total = report["total"]
    if total["total_tokens"] is None:
        return None
    # Legacy phases still depend on stream order. Native-only responses have no
    # reliable position among those markers; do not fabricate a phase. Claude's
    # total follows result.usage and excludes separately observed child counters.
    # Full per-response counters remain in token-usage.json for every backend.
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
        report["capture_complete"],
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
    if backend not in {"claude", "codex", "qodercli", "pi"}:
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
            provider_home=None,
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
                report.update(capture_complete=False, capture_errors=list(capture.errors))
                stdout = "".join(capture._chunks["stdout"])
                _LAST_CAPTURE.set((hashlib.sha256(stdout.encode()).hexdigest(), report))
        except Exception:
            # Accounting can fail too; retain the original CLI outcome and
            # leave the existing stream/ledger parser available as a fallback.
            clear_capture()
        print(f"[orchestrator] session capture failed at {capture.root}: {error}", flush=True)


class SessionCapture:
    def __init__(
        self,
        root: Path,
        *,
        backend: str,
        command: list[str],
        provider_home: Path | None,
        context: dict[str, str],
        native_environment: dict[str, str] | None = None,
        limits: CaptureLimits | None = None,
    ) -> None:
        self.root = root.resolve() / ("run-" + uuid.uuid4().hex)
        self.root.mkdir(parents=True, mode=0o700)
        (self.root / "provider").mkdir()
        self.backend = backend
        self.home = provider_home.resolve() if provider_home is not None else None
        self.session_id = self.root.name
        for flag in ("--session-id", "--resume"):
            if flag in command and command.index(flag) + 1 < len(command):
                self.session_id = command[command.index(flag) + 1]
                break
        # All supported adapters pass their exact initial/resume prompt last.
        self.prompt = command[-1] if backend and command else ""
        self.context = context
        self.started_at = datetime.now(UTC).isoformat()
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._readers: list[threading.Thread] = []
        self._monitor: threading.Thread | None = None
        # Functional output has the same full-stream contract as Popen.communicate.
        # Neither diagnostic limits nor native/history reads may truncate it.
        self._chunks: dict[str, list[str]] = {"stdout": [], "stderr": []}
        self._retained_chunks: dict[str, list[str]] = {"stdout": [], "stderr": []}
        self._pipe_bytes = {"stdout": 0, "stderr": 0}
        self._stdout_cursor = 0
        self.errors: list[str] = []
        self._budget = CaptureBudget(limits or CaptureLimits(), self._capture_error)
        # History scanning is bounded independently. Old bytes/records never
        # consume the invocation's diagnostic retention allowance.
        self._history_budget = CaptureBudget(
            self._budget.limits,
            lambda label, error: self._capture_error("resume_history_" + label, error),
        )
        self._usage = UsageAccumulator(backend)
        self._tails: dict[str, TranscriptTail] = {}
        self._initial_sizes: dict[str, int] = {}
        self._host_transcripts = None
        if self.home is None and native_environment is not None:
            from .session_native import HostSessionTranscripts

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
        self.native: dict[str, bytearray] = {}
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
        if self.home is None:
            return {}
        # No broad Home scan: credentials/settings are never session artifacts.
        patterns = (
            ".claude/projects/**/*.jsonl",
            ".codex/sessions/**/*.jsonl",
            ".qoder-writable/projects/**/*.jsonl",
            ".qoder-writable/tasks/**/*.jsonl",
            ".qoder/projects/**/*.jsonl",
            ".qoder/tasks/**/*.jsonl",
            ".pi/agent/sessions/**/*.jsonl",
        )
        result = {}
        count = 0
        for pattern in patterns:
            for path in self.home.glob(pattern):
                count += 1
                if count > self._budget.limits.files:
                    self._budget.warning("native_files_exceeded")
                    return result
                relative = "provider/native/" + path.relative_to(self.home).as_posix()
                result[relative] = (path, self._initial_sizes.get(relative, 0))
        return result

    def _tail(self, name: str, path: Path, previous_size: int) -> TranscriptTail | None:
        if name not in self._tails:
            if len(self._tails) >= self._budget.limits.files:
                self._budget.warning("native_files_exceeded")
                return None
            tail = self._tails[name] = TranscriptTail()
            try:
                for line in tail.resume_history(path, self._history_budget, previous_size):
                    self._feed_usage(name, line.decode("utf-8", errors="replace"), previous=True)
            finally:
                if tail.history_incomplete:
                    self._usage.mark_history_incomplete(name)
        return self._tails[name]

    def _feed_usage(self, path: str, text: str, *, previous: bool = False) -> None:
        try:
            self._usage.feed(path, text, previous=previous)
        except Exception as error:
            # Provider/native JSON is untrusted data. A malformed counter cannot
            # unwind a pipe reader or stop the other transcripts being captured.
            self._capture_error("usage_capture", error)

    def _sync_stdout_usage(self) -> None:
        while self._stdout_cursor < len(self._chunks["stdout"]):
            text = self._chunks["stdout"][self._stdout_cursor]
            self._stdout_cursor += 1
            self._feed_usage("provider/stdout.stream-json", text)
            if self.backend == "codex":
                from .agent_runtime.codex_ledger import codex_thread_id_from_stream

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
            self._sync_stdout_usage()
            for name, (path, previous_size) in self._native_paths(force=final).items():
                try:
                    tail = self._tail(name, path, previous_size)
                    if tail is None:
                        continue
                    for addition in tail.read(path, self._budget, final=final):
                        text = addition.decode("utf-8", errors="replace")
                        self._feed_usage(name, text)
                        self.native.setdefault(name, bytearray()).extend(addition)
                        destination = self.root / name
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        with destination.open("ab") as output:
                            output.write(addition)
                        if record_provider_line(text):
                            self._append(name, text)
                except (OSError, ValueError) as error:
                    self._capture_error("native_capture", error)
            self._write_usage(False)

    def _write_usage(self, finished: bool) -> dict:
        self._sync_stdout_usage()
        report = self._usage.report(finished=finished and not self.errors)
        report.update(
            session_id=self.session_id,
            started_at=self.started_at,
            state="finished" if finished else "running",
            context=self.context,
            capture_errors=list(self.errors),
            capture_complete=bool(finished and not self.errors),
        )
        _atomic(
            self.root / "token-usage.json", json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        )
        return report

    def _capture_error(self, label: str, error: Exception) -> None:
        warning = f"{label}:{type(error).__name__}:{error}"
        with self._lock:
            if warning not in self.errors and len(self.errors) < 100:
                self.errors.append(warning)

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
                    self._chunks[name].append(line)
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
                        if (
                            not self._budget.retain(size, file_used=self._pipe_bytes[name])
                            or not self._budget.record()
                        ):
                            continue
                        self._pipe_bytes[name] += size
                        # Only this bounded copy may enter persisted diagnostics.
                        # Keep it even if a disk sink fails, for final projection.
                        self._retained_chunks[name].append(line)
                        try:
                            if name == "stdout" and not record_provider_line(line):
                                continue
                        except Exception as error:
                            self._capture_error(f"{name}_capture_filter", error)
                            continue
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
        return "".join(self._chunks["stdout"]), "".join(self._chunks["stderr"])

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
            stdout = "".join(self._chunks["stdout"])
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
                files.append((display, bytes(value)))
            conversation = render_conversation(
                backend=self.backend,
                session_id=self.session_id,
                prompt=self.prompt,
                stdout="".join(self._retained_chunks["stdout"]),
                raw_provider_files=files,
                state=state,
                exit_status=exit_status,
                timed_out=timed_out,
                raw_provider_capture_complete=bool(self.native) and not self.errors,
            )
            # stderr can contain launch/auth/failure diagnostics absent from stdout.
            rows = [json.loads(line) for line in conversation.splitlines()]
            rows[0].update(started_at=self.started_at, context=self.context)
            rows[-1].update(capture_complete=not self.errors, capture_errors=list(self.errors))
            for line in "".join(self._retained_chunks["stderr"]).splitlines():
                rows.insert(-1, provider_line_record(0, path="provider/stderr.log", line=line))
            for index, row in enumerate(rows):
                row["sequence"] = index
            _atomic(self.root / "conversation.jsonl", encode_records(rows))
            report = self._write_usage(True)
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
            _LAST_CAPTURE.set((hashlib.sha256(stdout.encode()).hexdigest(), report))
            self.finished = True
