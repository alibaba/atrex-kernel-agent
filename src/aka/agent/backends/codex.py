from __future__ import annotations
from collections.abc import Mapping
from pathlib import Path
from .base import (
    BackendFiles,
    AgentBackendAdapter,
    _counter,
    _json_events,
    _phase_marker_receipts,
    token_usage_from_mapping,
    codex_settings_args,
)
from ..model import NormalizedAgentEvent, TokenUsage, AgentRuntimeCapabilities


class CodexAdapter(AgentBackendAdapter):
    id = "codex"
    settings_variable = "ATREX_CODEX_SESSION_SETTINGS"
    environment_variables = (
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "CODEX_API_KEY",
        "CODEX_HOME",
    )
    files = BackendFiles(
        (".codex/auth.json", ".codex/config.toml", ".codex/models_cache.json"),
        (".codex",),
        (".codex/bin",),
        (("CODEX_HOME", ".codex"),),
    )

    capabilities = AgentRuntimeCapabilities(
        terminal_usage=True,
        usage_delta=False,
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
        del session_id
        command = [
            "codex",
            "exec",
            "--json",
            "--color",
            "never",
            "--dangerously-bypass-approvals-and-sandbox",
            "-c",
            f'model_reasoning_effort="{reasoning_effort}"',
        ]
        command += codex_settings_args(settings)
        command.append(prompt)
        return command

    def normalize_stream(
        self, stdout: str
    ) -> tuple[tuple[NormalizedAgentEvent, ...], TokenUsage]:
        normalized: list[NormalizedAgentEvent] = []
        terminal = TokenUsage.unavailable()
        for event in _json_events(stdout):
            if event.get("type") in {"turn.completed", "result"}:
                raw_usage = event.get("usage")
                terminal = token_usage_from_mapping(raw_usage)
                if isinstance(raw_usage, Mapping):
                    cached = _counter(
                        raw_usage, "cached_input_tokens", "cachedInputTokens"
                    )
                    if cached is not None:
                        terminal = TokenUsage(
                            input_tokens=terminal.input_tokens,
                            output_tokens=terminal.output_tokens,
                            cache_read_tokens=cached,
                            cache_write_tokens=terminal.cache_write_tokens,
                            total_tokens=terminal.total_tokens,
                            measurement=terminal.measurement,
                        )
                if terminal.total_tokens is not None:
                    normalized.append(
                        NormalizedAgentEvent(
                            sequence=len(normalized),
                            kind="terminal_usage",
                            usage=terminal,
                        )
                    )
                continue
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
        return tuple(normalized), terminal

    def auth_hint(self) -> str:
        return 'run `codex login status` and `codex exec "reply ok"` to diagnose'

    supports_resume = True

    def fresh_command(self, command):
        command = list(command)
        if command[:2] != ["codex", "exec"]:
            raise RuntimeError("Codex command has no compatible exec seam")
        if "--ephemeral" in command:
            command.remove("--ephemeral")
        return command

    def resume_command(self, command, session_id):
        command = self.fresh_command(command)
        prompt = command.pop()
        if "--color" in command:
            index = command.index("--color")
            del command[index : index + 2]
        command.insert(2, "resume")
        return [*command, session_id, prompt]
