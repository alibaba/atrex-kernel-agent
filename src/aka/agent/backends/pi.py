from __future__ import annotations
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from .base import (
    BackendFiles,
    AgentBackendAdapter,
    _json_events,
    _phase_marker_receipts,
    token_usage_from_mapping,
    pi_settings_args,
)
from ..model import (
    NormalizedAgentEvent,
    TokenUsage,
    AgentRuntimeCapabilities,
    sum_token_usages,
)


class PiAdapter(AgentBackendAdapter):
    id = "pi"
    settings_variable = "ATREX_PI_SESSION_SETTINGS"
    environment_variables = (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_BASE_URL",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "PI_CODING_AGENT_DIR",
        "PI_SKIP_VERSION_CHECK",
        "PI_TELEMETRY",
    )
    files = BackendFiles(
        (".pi/agent/auth.json", ".pi/agent/settings.json", ".pi/agent/models.json"),
        (".pi",),
        (".pi/bin",),
        (("PI_CODING_AGENT_DIR", ".pi/agent"),),
    )

    capabilities = AgentRuntimeCapabilities(
        terminal_usage=True,
        usage_delta=True,
        phase_marker_receipt=True,
    )

    def __init__(self, runtime_assets_dir: Path | None = None) -> None:
        del runtime_assets_dir

    def build_command(
        self,
        prompt: str,
        session_id: str,
        reasoning_effort: str,
        settings: str,
    ) -> list[str]:
        command = [
            "pi",
            "--mode",
            "json",
            "--session-id",
            session_id,
            "--approve",
            "--thinking",
            reasoning_effort,
        ]
        command += pi_settings_args(settings)
        command.append(prompt)
        return command

    def normalize_stream(
        self, stdout: str
    ) -> tuple[tuple[NormalizedAgentEvent, ...], TokenUsage]:
        normalized: list[NormalizedAgentEvent] = []
        deltas: list[TokenUsage] = []
        settled = False
        for event in _json_events(stdout):
            if event.get("type") == "message_end":
                message = event.get("message")
                if isinstance(message, Mapping) and message.get("role") in {
                    "assistant",
                    "toolResult",
                }:
                    usage = token_usage_from_mapping(message.get("usage"))
                    if usage.total_tokens is not None:
                        deltas.append(usage)
                        normalized.append(
                            NormalizedAgentEvent(
                                sequence=len(normalized),
                                kind="usage_delta",
                                usage=usage,
                            )
                        )
                for action, phase, marker_id in _phase_marker_receipts(event):
                    normalized.append(
                        NormalizedAgentEvent(
                            sequence=len(normalized),
                            kind="phase_marker",
                            phase=phase,
                            action=action,
                            marker_id=marker_id,
                        )
                    )
            elif event.get("type") == "compaction_end":
                result = event.get("result")
                usage = token_usage_from_mapping(
                    result.get("usage") if isinstance(result, Mapping) else None
                )
                if usage.total_tokens is not None:
                    deltas.append(usage)
                    normalized.append(
                        NormalizedAgentEvent(
                            sequence=len(normalized),
                            kind="usage_delta",
                            usage=usage,
                        )
                    )
            elif event.get("type") == "agent_settled":
                settled = True

        terminal = sum_token_usages(deltas)
        if terminal.total_tokens is not None:
            if not settled:
                terminal = replace(terminal, measurement="partial")
            else:
                normalized.append(
                    NormalizedAgentEvent(
                        sequence=len(normalized),
                        kind="terminal_usage",
                        usage=terminal,
                    )
                )
        return tuple(normalized), terminal

    def auth_hint(self) -> str:
        return 'run `pi --list-models` and `pi -p "reply ok"` to diagnose'

    def configure_environment(self, environment):
        return dict(environment, PI_SKIP_VERSION_CHECK="1", PI_TELEMETRY="0")
