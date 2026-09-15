"""Live Supervisor-owned conversation/usage copies, independent of CLI Home.

Native files are data, not Runtime session envelopes. There is deliberately no
small trace-file count limit: a session can contain arbitrarily many subagents.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import uuid
from contextvars import ContextVar
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from .agent_workspace import _regular_bytes
from .session_transcript import (
    encode_records,
    initial_records,
    provider_line_record,
    record_provider_line,
    render_conversation,
)
from .session_usage import summarize_usage

_LAST_CAPTURE: ContextVar[tuple[str, dict] | None] = ContextVar("aka_session_capture", default=None)


def clear_capture() -> None:
    _LAST_CAPTURE.set(None)


def captured_observation(stdout: str, events, capabilities):
    """Use the current invocation's native accounting, never an unrelated run."""
    from .agent_runtime.model import NormalizedAgentEvent, TokenUsage, resequence_agent_events

    captured = _LAST_CAPTURE.get()
    if not captured or captured[0] != hashlib.sha256(stdout.encode()).hexdigest():
        return None
    report = captured[1]
    total = report["total"]
    if total["total_tokens"] is None:
        return None
    # Keep phase receipts, replace overlapping stream counters with unique responses.
    normalized = [event for event in events if event.kind == "phase_marker"]
    normalized.extend(
        NormalizedAgentEvent(0, "usage_delta", TokenUsage(**row["usage"]))
        for row in report["responses"]
    )
    usage = TokenUsage(**total)
    normalized.append(NormalizedAgentEvent(0, "terminal_usage", usage))
    capabilities = replace(capabilities, usage_delta_observed=bool(report["responses"]))
    return resequence_agent_events(normalized), usage, capabilities, tuple(
        [*report["warnings"], *report.get("capture_errors", [])]
    )


def _atomic(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        output.write(text)
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


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
        self._chunks: dict[str, list[str]] = {"stdout": [], "stderr": []}
        self.errors: list[str] = []
        self.previous: dict[str, bytes] = {}
        self._offsets: dict[str, int] = {}
        self._host_transcripts = None
        if self.home is None and native_environment is not None:
            from .session_native import HostSessionTranscripts

            try:
                self._host_transcripts = HostSessionTranscripts(backend, native_environment, command)
            except (OSError, ValueError) as error:
                self._capture_error("native_capture_setup", error)
        self.previous.update(self._native_snapshot())
        self._offsets.update({path: len(text) for path, text in self.previous.items()})
        self.native: dict[str, bytes] = {}
        self._sequence = 0
        self.finished = False
        initial = initial_records(backend=backend, session_id=self.session_id, prompt=self.prompt)
        initial[0].update(started_at=self.started_at, context=context)
        (self.root / "conversation.jsonl").write_text(encode_records(initial), encoding="utf-8")
        self._sequence = len(initial)
        self._write_usage(False)

    def _native_snapshot(self) -> dict[str, bytes]:
        if self._host_transcripts is not None:
            try:
                snapshot, previous = self._host_transcripts.snapshot("".join(self._chunks["stdout"]))
                for path, payload in previous.items():
                    if path not in self.previous:
                        self.previous[path] = payload
                        self._offsets[path] = len(payload)
                return snapshot
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
        for pattern in patterns:
            for path in self.home.glob(pattern):
                relative = "provider/native/" + path.relative_to(self.home).as_posix()
                try:
                    payload = _regular_bytes(path)
                    if payload is not None:
                        result[relative] = payload
                except (OSError, ValueError) as error:
                    warning = f"native_capture:{relative}:{type(error).__name__}"
                    if warning not in self.errors:
                        self.errors.append(warning)
        return result

    def _append(self, path: str, line: str) -> None:
        row = provider_line_record(self._sequence, path=path, line=line.rstrip("\n"))
        self._sequence += 1
        with (self.root / "conversation.jsonl").open("a", encoding="utf-8") as output:
            output.write(encode_records([row]))

    def sync_native(self, *, final: bool = False) -> None:
        with self._lock:
            for path, payload in self._native_snapshot().items():
                offset = self._offsets.get(path, 0)
                if len(payload) < offset:
                    warning = f"native_transcript_truncated:{path}"
                    if warning not in self.errors:
                        self.errors.append(warning)
                    continue
                end = len(payload) if final else payload.rfind(b"\n") + 1
                if end <= offset:
                    continue
                addition = payload[offset:end]
                self._offsets[path] = end
                self.native[path] = self.native.get(path, b"") + addition
                destination = self.root / path
                destination.parent.mkdir(parents=True, exist_ok=True)
                with destination.open("ab") as output:
                    output.write(addition)
                for line in addition.decode("utf-8", errors="replace").splitlines():
                    if record_provider_line(line):
                        self._append(path, line)
            self._write_usage(False)

    def _write_usage(self, finished: bool) -> dict:
        report = summarize_usage(
            self.backend,
            "".join(self._chunks["stdout"]),
            {path: value.decode("utf-8", errors="replace") for path, value in self.native.items()},
            previous={
                path: value.decode("utf-8", errors="replace")
                for path, value in self.previous.items()
            },
            finished=finished and not self.errors,
        )
        report.update(
            session_id=self.session_id,
            started_at=self.started_at,
            state="finished" if finished else "running",
            context=self.context,
            capture_errors=list(self.errors),
        )
        _atomic(
            self.root / "token-usage.json", json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        )
        return report

    def _capture_error(self, label: str, error: Exception) -> None:
        warning = f"{label}:{type(error).__name__}:{error}"
        with self._lock:
            if warning not in self.errors:
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
            for line in iter(pipe.readline, ""):
                with self._lock:
                    # Keep the adapter's original, unfiltered stream in memory.
                    self._chunks[name].append(line)
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
            reader.join(timeout=5)
            if reader.is_alive():
                self.errors.append("provider_pipe_did_not_close")
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
            if self.backend == "codex":
                from .agent_runtime.codex_ledger import codex_thread_id_from_stream

                self.session_id = codex_thread_id_from_stream(stdout) or self.session_id
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
            conversation = render_conversation(
                backend=self.backend,
                session_id=self.session_id,
                prompt=self.prompt,
                stdout=stdout,
                raw_provider_files=files,
                state=state,
                exit_status=exit_status,
                timed_out=timed_out,
                raw_provider_capture_complete=bool(self.native) and not self.errors,
            )
            # stderr can contain launch/auth/failure diagnostics absent from stdout.
            rows = [json.loads(line) for line in conversation.splitlines()]
            rows[0].update(started_at=self.started_at, context=self.context)
            for line in "".join(self._chunks["stderr"]).splitlines():
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
