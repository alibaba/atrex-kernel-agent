"""Shared CLI adapters with Light's accounting and transcript policies.

Command syntax and Pi/Codex normalization belong to the legacy Agent module.
Light only overrides Claude accounting and Qoder transcript persistence.
"""
from collections.abc import Mapping
from dataclasses import replace
from aka.legacy.agent.adapter import (
    AgentBackendAdapter, AdapterFactory, BackendAdapterRegistry,
    ClaudeLikeAdapter as _ClaudeLikeAdapter, ClaudeAdapter as _ClaudeAdapter,
    QoderAdapter as _QoderAdapter, PiAdapter, CodexAdapter,
    token_usage_from_mapping, token_usage_from_model_usage, pi_event_usage,
    toml_config_value, pi_settings_args, codex_settings_args,
    usage_counter as _counter, json_events as _json_events,
    event_phase_marker_receipts as _phase_marker_receipts,
)
from .model import NormalizedAgentEvent, TokenUsage, sum_token_usages


class ClaudeLikeAdapter(_ClaudeLikeAdapter):
    def normalize_stream(
        self, stdout: str, *, settled_usage: Mapping[str, TokenUsage] | None = None
    ) -> tuple[tuple[NormalizedAgentEvent, ...], TokenUsage]:
        normalized: list[NormalizedAgentEvent] = []
        terminal = TokenUsage.unavailable()
        sequence = 0
        usage_message_positions: dict[str, int] = {}
        for event in _json_events(stdout):
            event_type = event.get("type")
            if event_type == "result":
                terminal = token_usage_from_mapping(event.get("usage"))
                if terminal.total_tokens is None and self.id != "claude":
                    terminal = token_usage_from_model_usage(event.get("modelUsage"))
                if terminal.total_tokens is not None:
                    normalized.append(
                        NormalizedAgentEvent(
                            sequence=sequence,
                            kind="terminal_usage",
                            usage=terminal,
                        )
                    )
                    sequence += 1
                continue
            usage: object = event.get("usage")
            message = event.get("message")
            if usage is None and isinstance(message, Mapping):
                usage = message.get("usage")
            # Task progress counters are cumulative, not new model responses.
            is_main_response = self.id != "claude" or not event.get("parent_tool_use_id")
            parsed = token_usage_from_mapping(
                usage if event_type == "assistant" and is_main_response else None
            )
            message_id = (
                message.get("id")
                if isinstance(message, Mapping)
                and isinstance(message.get("id"), str)
                else None
            )
            if self.id == "claude" and parsed.total_tokens is not None:
                # Like Runtime's Claude ledger, prefer the settled native row
                # for this exact response ID. Keep the original marker order.
                parsed = (settled_usage or {}).get(message_id) or replace(
                    parsed, measurement="partial"
                )
            duplicate_usage = bool(
                message_id and message_id in usage_message_positions
            )
            if parsed.total_tokens is not None and duplicate_usage:
                # Streaming repeats update the same response, not a second bill.
                position = usage_message_positions[message_id]
                normalized[position] = replace(normalized[position], usage=parsed)
            if parsed.total_tokens is not None and not duplicate_usage:
                normalized.append(
                    NormalizedAgentEvent(
                        sequence=sequence,
                        kind="usage_delta",
                        usage=parsed,
                    )
                )
                sequence += 1
                if message_id:
                    usage_message_positions[message_id] = len(normalized) - 1
            for action, phase, marker_id in _phase_marker_receipts(event):
                normalized.append(
                    NormalizedAgentEvent(
                        sequence=sequence,
                        kind="phase_marker",
                        phase=phase,
                        action=action,
                        marker_id=marker_id,
                    )
                )
                sequence += 1
        if terminal.total_tokens is None:
            deltas = [
                event.usage
                for event in normalized
                if event.kind == "usage_delta" and event.usage is not None
            ]
            terminal = sum_token_usages(deltas)
            if terminal.total_tokens is not None:
                terminal = replace(terminal, measurement="partial")
        return tuple(normalized), terminal


class ClaudeAdapter(_ClaudeAdapter, ClaudeLikeAdapter):
    """Use shared command construction with result.usage accounting."""

    def classify_terminal_failure(self, stdout: str, session_id: str):
        """Runtime's root-session terminal-error guard, not a text heuristic."""
        if not session_id:
            return None, None
        last_assistant = None
        failure = (None, None)
        for event in _json_events(stdout):
            if (event.get("session_id") != session_id
                    or event.get("parent_tool_use_id") is not None):
                continue
            if event.get("type") == "assistant":
                last_assistant = event
                failure = (None, None)
            elif event.get("type") == "result":
                failure = (None, None)
                assistant, last_assistant = last_assistant, None
                if (event.get("is_error") is not True
                        or event.get("terminal_reason") != "api_error"
                        or assistant is None
                        or assistant.get("is_api_error_message") is not True):
                    continue
                codes = {assistant.get(key) for key in ("error", "api_error")
                         if isinstance(assistant.get(key), str)}
                if codes == {"max_output_tokens"}:
                    failure = ("output_limit", "max_output_tokens")
        return failure


class QoderAdapter(_QoderAdapter, ClaudeLikeAdapter):
    def build_command(self, *args, **options):
        command = super().build_command(*args, **options)
        command.remove("--no-session-persistence")
        return command


DEFAULT_BACKEND_REGISTRY = BackendAdapterRegistry()
DEFAULT_BACKEND_REGISTRY.register("claude", ClaudeAdapter)
DEFAULT_BACKEND_REGISTRY.register("qodercli", QoderAdapter)
DEFAULT_BACKEND_REGISTRY.register("codex", CodexAdapter)
DEFAULT_BACKEND_REGISTRY.register("pi", PiAdapter)
