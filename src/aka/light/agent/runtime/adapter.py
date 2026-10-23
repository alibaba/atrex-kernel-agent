"""Shared CLI adapters with Light's accounting and transcript policies.

Command syntax and Pi/Codex normalization belong to the legacy Agent module.
Light also classifies terminal provider failures independently of CLI exit codes.
"""
from collections.abc import Mapping
from dataclasses import replace
from aka.legacy.agent.adapter import (
    AgentBackendAdapter, AdapterFactory, BackendAdapterRegistry,
    ClaudeLikeAdapter as _ClaudeLikeAdapter, ClaudeAdapter as _ClaudeAdapter,
    QoderAdapter as _QoderAdapter, PiAdapter as _PiAdapter, CodexAdapter,
    token_usage_from_mapping, token_usage_from_model_usage,
    toml_config_value, pi_settings_args, codex_settings_args,
    _counter, _json_events, _phase_marker_receipts,
)
from aka.light.agent.runtime.model import NormalizedAgentEvent, TokenUsage, sum_token_usages
from aka.light.agent.runtime.provider_errors import rate_limit_code


def pi_event_usage(event: Mapping[str, object]) -> TokenUsage:
    """Return billable Pi terminal-event usage for live capture reconciliation."""
    if event.get("type") == "message_end":
        message = event.get("message")
        if isinstance(message, Mapping) and message.get("role") in {
            "assistant", "toolResult",
        }:
            return token_usage_from_mapping(message.get("usage"))
    elif event.get("type") == "compaction_end":
        result = event.get("result")
        if isinstance(result, Mapping):
            return token_usage_from_mapping(result.get("usage"))
    return TokenUsage.unavailable()


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


class PiAdapter(_PiAdapter):
    def classify_terminal_failure(self, stdout: str, session_id: str):
        # A retry can produce an error followed by a successful assistant turn.
        # Only the final assistant message determines the invocation outcome.
        failure = (None, None)
        for event in _json_events(stdout):
            message = event.get("message")
            if (event.get("type") != "message_end" or not isinstance(message, Mapping)
                    or message.get("role") != "assistant"):
                continue
            failure = (None, None)
            if message.get("stopReason") in {"error", "aborted"}:
                code = rate_limit_code(str(message.get("errorMessage", "")))
                failure = (("rate_limit", code) if code else
                           ("provider_error", "pi_assistant_" + message["stopReason"]))
        return failure


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
                if (event.get("is_api_error_message") is True
                        or event.get("isApiErrorMessage") is True):
                    message = event.get("message")
                    content = message.get("content", []) if isinstance(message, Mapping) else []
                    for block in content:
                        if isinstance(block, Mapping) and block.get("type") == "text":
                            code = rate_limit_code(str(block.get("text", "")))
                            if code:
                                failure = ("rate_limit", code)
            elif event.get("type") == "result":
                if event.get("is_error") is True and failure[0] == "rate_limit":
                    continue
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
