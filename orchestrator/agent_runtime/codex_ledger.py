from __future__ import annotations

import json
import os
import re
import tempfile
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .markers import phase_marker_receipts
from .model import (
    AgentRuntimeCapabilities,
    NormalizedAgentEvent,
    TokenUsage,
    sum_token_usages,
)


_THREAD_ID = re.compile(r"^[0-9a-fA-F-]{32,64}$")


class CodexLedgerError(ValueError):
    pass


@dataclass(frozen=True)
class CodexLedgerObservation:
    events: tuple[NormalizedAgentEvent, ...]
    terminal_usage: TokenUsage
    session_usage: TokenUsage


def codex_home(environment: Mapping[str, str] | None = None) -> Path:
    source = environment if environment is not None else os.environ
    configured = source.get("CODEX_HOME")
    return Path(configured).expanduser() if configured else Path.home() / ".codex"


def codex_thread_id_from_stream(stdout: str) -> str:
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, Mapping) or event.get("type") != "thread.started":
            continue
        value = event.get("thread_id") or event.get("threadId")
        if isinstance(value, str) and _THREAD_ID.fullmatch(value.strip()):
            return value.strip()
    return ""


def _counter(value: Mapping[str, Any], name: str) -> int | None:
    item = value.get(name)
    return item if isinstance(item, int) and not isinstance(item, bool) and item >= 0 else None


def usage_matches(observed: TokenUsage, terminal: TokenUsage) -> bool:
    for name in (
        "input_tokens",
        "output_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
        "total_tokens",
    ):
        left = getattr(observed, name)
        right = getattr(terminal, name)
        if left is not None and right is not None and left != right:
            return False
    return observed.total_tokens is not None and terminal.total_tokens is not None


def _usage(value: object) -> TokenUsage:
    if not isinstance(value, Mapping):
        return TokenUsage.unavailable()
    components = (
        _counter(value, "input_tokens"),
        _counter(value, "output_tokens"),
        _counter(value, "cached_input_tokens"),
        _counter(value, "cache_write_input_tokens"),
        _counter(value, "total_tokens"),
    )
    if any(item is None for item in components):
        return TokenUsage.unavailable()
    return TokenUsage(*components, measurement="exact")


CODEX_LEDGER_CAPABILITIES = AgentRuntimeCapabilities(
    terminal_usage=True,
    usage_delta=True,
    phase_marker_receipt=True,
    usage_delta_observed=True,
)


def observe_codex_usage(
    observer: "CodexSessionLedgerObserver",
    thread_id: str,
    stream_terminal: TokenUsage,
) -> tuple[
    tuple[NormalizedAgentEvent, ...],
    TokenUsage,
    AgentRuntimeCapabilities,
    tuple[str, ...],
]:
    if stream_terminal.total_tokens is not None:
        observation = observer.observe_reconciled(thread_id, stream_terminal)
        return (
            observation.events,
            observation.terminal_usage,
            CODEX_LEDGER_CAPABILITIES,
            (),
        )
    observation = observer.observe(thread_id)
    terminal = replace(observation.terminal_usage, measurement="partial")
    events = tuple(
        replace(
            event,
            usage=(
                replace(event.usage, measurement="partial")
                if event.usage is not None
                else None
            ),
        )
        for event in observation.events
    )
    return (
        events,
        terminal,
        CODEX_LEDGER_CAPABILITIES,
        ("codex_stdout_terminal_unavailable",),
    )


class CodexTemporaryHome:
    """Isolate ordinary Codex rollout state while reusing read-only configuration."""

    _SHARED_ENTRIES = (
        "auth.json",
        "config.toml",
        "skills",
        "plugins",
        "hooks.json",
        "models_cache.json",
        "installation_id",
        "vendor_imports",
        "mcp-oauth-locks",
    )

    def __init__(self, source: Path):
        self.source = source.resolve()
        self._temporary: tempfile.TemporaryDirectory[str] | None = None
        self.path: Path | None = None

    def open(self) -> Path:
        self._temporary = tempfile.TemporaryDirectory(prefix="atrex-codex-home-")
        self.path = Path(self._temporary.name).resolve()
        for name in self._SHARED_ENTRIES:
            source = self.source / name
            if source.exists():
                (self.path / name).symlink_to(
                    source, target_is_directory=source.is_dir()
                )
        return self.path

    def close(self) -> str | None:
        if self._temporary is None:
            return None
        try:
            self._temporary.cleanup()
        except Exception as exc:
            return f"codex_temporary_home_cleanup_failed:{type(exc).__name__}"
        self._temporary = None
        return None


class CodexSessionLedgerObserver:
    """Incrementally normalize one Codex rollout without persisting its raw content."""

    def __init__(self, home: Path | None = None):
        self.home = (home or codex_home()).resolve()
        self._thread_id = ""
        self._path: Path | None = None
        self._offset = 0
        self._session_usage: TokenUsage | None = None

    def _rollout_paths(self) -> Iterator[Path]:
        root = self.home / "sessions"
        if root.is_dir():
            yield from root.rglob("rollout-*.jsonl")

    def _find_rollout_path(self, thread_id: str) -> Path:
        if not _THREAD_ID.fullmatch(thread_id):
            raise CodexLedgerError("invalid Codex thread id")
        if self._path is not None:
            return self._path
        root = self.home / "sessions"
        for _ in range(20):
            matches = sorted(
                root.rglob(f"*{thread_id}.jsonl") if root.is_dir() else (),
                key=lambda path: path.stat().st_mtime_ns,
                reverse=True,
            )
            if matches:
                self._thread_id = thread_id
                self._path = matches[0].resolve()
                return self._path
            time.sleep(0.05)
        raise CodexLedgerError("Codex session ledger not found")

    @staticmethod
    def _session_meta(path: Path) -> tuple[str, Path] | None:
        try:
            with path.open(encoding="utf-8") as handle:
                for _ in range(32):
                    line = handle.readline()
                    if not line:
                        break
                    record = json.loads(line)
                    body = record.get("payload") if isinstance(record, Mapping) else None
                    if (
                        isinstance(body, Mapping)
                        and record.get("type") == "session_meta"
                        and isinstance(body.get("id"), str)
                        and isinstance(body.get("cwd"), str)
                    ):
                        return body["id"], Path(body["cwd"]).resolve()
        except (OSError, UnicodeError, json.JSONDecodeError):
            return None
        return None

    def identify_new_thread(self, workspace: Path) -> str:
        expected_workspace = workspace.resolve()
        candidates: list[tuple[str, Path]] = []
        for resolved in (path.resolve() for path in self._rollout_paths()):
            metadata = self._session_meta(resolved)
            if metadata is not None and metadata[1] == expected_workspace:
                candidates.append((metadata[0], resolved))
        if len(candidates) != 1:
            raise CodexLedgerError(
                "cannot uniquely identify new Codex rollout for workspace"
            )
        thread_id, path = candidates[0]
        if not _THREAD_ID.fullmatch(thread_id):
            raise CodexLedgerError("new Codex rollout has invalid thread id")
        self._thread_id = thread_id
        self._path = path
        return thread_id

    def observe(self, thread_id: str) -> CodexLedgerObservation:
        path = self._find_rollout_path(thread_id)
        if self._thread_id and thread_id != self._thread_id:
            raise CodexLedgerError("Codex observer cannot switch thread ids")
        with path.open("rb") as handle:
            handle.seek(self._offset)
            payload = handle.read()
            next_offset = handle.tell()
        if not payload:
            raise CodexLedgerError("Codex session ledger has no new events")

        events: list[NormalizedAgentEvent] = []
        pending_markers: list[tuple[str, str, str]] = []
        deltas: list[TokenUsage] = []
        final_session_usage: TokenUsage | None = None
        previous_cumulative = self._session_usage.total_tokens if self._session_usage else 0
        observed_cumulative = previous_cumulative
        for raw_line in payload.splitlines():
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise CodexLedgerError("malformed Codex session ledger JSON") from exc
            if not isinstance(record, Mapping):
                continue
            body = record.get("payload")
            body = body if isinstance(body, Mapping) else {}
            if (
                record.get("type") == "response_item"
                and body.get("type") == "custom_tool_call_output"
            ):
                pending_markers.extend(phase_marker_receipts(body.get("output")))
                continue
            if record.get("type") != "event_msg" or body.get("type") != "token_count":
                continue
            info = body.get("info")
            info = info if isinstance(info, Mapping) else {}
            last_usage = _usage(info.get("last_token_usage"))
            session_usage = _usage(info.get("total_token_usage"))
            if last_usage.total_tokens is None or session_usage.total_tokens is None:
                raise CodexLedgerError("Codex token_count is missing usage totals")
            if last_usage.total_tokens <= 0:
                raise CodexLedgerError("Codex token_count reported non-positive usage")
            if session_usage.total_tokens < observed_cumulative:
                raise CodexLedgerError("Codex cumulative usage moved backwards")
            duplicate = session_usage.total_tokens == observed_cumulative
            if not duplicate:
                deltas.append(last_usage)
                events.append(
                    NormalizedAgentEvent(
                        sequence=len(events), kind="usage_delta", usage=last_usage
                    )
                )
                observed_cumulative = session_usage.total_tokens
                final_session_usage = session_usage
            for action, phase, marker_id in pending_markers:
                events.append(
                    NormalizedAgentEvent(
                        sequence=len(events),
                        kind="phase_marker",
                        phase=phase,
                        action=action,
                        marker_id=marker_id,
                    )
                )
            pending_markers = []

        if not deltas or final_session_usage is None:
            raise CodexLedgerError("Codex ledger exposed no new token_count delta")
        invocation_usage = sum_token_usages(deltas)
        previous_usage = self._session_usage or TokenUsage.zero()
        for name in (
            "input_tokens",
            "output_tokens",
            "cache_read_tokens",
            "cache_write_tokens",
            "total_tokens",
        ):
            previous_value = getattr(previous_usage, name)
            invocation_value = getattr(invocation_usage, name)
            final_value = getattr(final_session_usage, name)
            if (
                previous_value is not None
                and invocation_value is not None
                and final_value is not None
                and previous_value + invocation_value != final_value
            ):
                raise CodexLedgerError(
                    f"Codex {name} component does not reconcile to cumulative usage"
                )
        for action, phase, marker_id in pending_markers:
            events.append(
                NormalizedAgentEvent(
                    sequence=len(events),
                    kind="phase_marker",
                    phase=phase,
                    action=action,
                    marker_id=marker_id,
                )
            )
        events.append(
            NormalizedAgentEvent(
                sequence=len(events), kind="terminal_usage", usage=invocation_usage
            )
        )
        self._offset = next_offset
        self._session_usage = final_session_usage
        return CodexLedgerObservation(
            events=tuple(events),
            terminal_usage=invocation_usage,
            session_usage=final_session_usage,
        )

    def observe_reconciled(
        self, thread_id: str, stream_terminal: TokenUsage
    ) -> CodexLedgerObservation:
        previous_offset = self._offset
        previous_session_usage = self._session_usage
        try:
            observation = self.observe(thread_id)
            if not usage_matches(observation.session_usage, stream_terminal):
                raise CodexLedgerError(
                    "Codex ledger usage does not match stdout terminal usage"
                )
            return observation
        except Exception:
            self._offset = previous_offset
            self._session_usage = previous_session_usage
            raise
