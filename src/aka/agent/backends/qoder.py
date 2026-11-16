from __future__ import annotations
from dataclasses import replace
from pathlib import Path
from .base import BackendFiles, ClaudeLikeAdapter
from ..model import NormalizedAgentEvent, TokenUsage


class QoderAdapter(ClaudeLikeAdapter):
    id = "qodercli"
    settings_variable = "ATREX_QODER_SESSION_SETTINGS"
    environment_variables = ("QODER_API_KEY", "QODER_BASE_URL", "QODER_CONFIG_DIR")
    files = BackendFiles(
        (
            ".qoder/settings.json",
            ".qoder/state.json",
            ".qoder/auth.json",
            ".qoder/credentials.json",
            ".qoder/installation_id",
            ".qoder/.auth/user",
            ".qoder/.auth/machine_id",
            ".qodersec/config.yaml",
            ".qodersec/runtime.json",
            ".qodersec/.config-version",
        ),
        (".qoder", ".qodersec"),
        (".qoder/bin", ".qodersec/bin"),
        (),
    )

    def __init__(self, runtime_assets_dir: Path | None = None) -> None:
        del runtime_assets_dir

    def normalize_stream(
        self, stdout: str
    ) -> tuple[tuple[NormalizedAgentEvent, ...], TokenUsage]:
        events, terminal = super().normalize_stream(stdout)
        usage_events = [event for event in events if event.usage is not None]
        if (
            terminal.total_tokens == 0
            and usage_events
            and all(
                event.usage.total_tokens == 0 for event in usage_events if event.usage
            )
        ):
            markers = [event for event in events if event.kind == "phase_marker"]
            return (
                tuple(
                    replace(event, sequence=index)
                    for index, event in enumerate(markers)
                ),
                TokenUsage.unavailable(),
            )
        return events, terminal

    def build_command(
        self,
        prompt: str,
        session_id: str,
        reasoning_effort: str,
        settings: str,
    ) -> list[str]:
        command = [
            "qodercli",
            "--print",
            "--dangerously-skip-permissions",
            "--output-format",
            "stream-json",
            "--session-id",
            session_id,
            "--no-session-persistence",
            "--reasoning-effort",
            reasoning_effort,
        ]
        if settings:
            command += ["--settings", settings]
        command.append(prompt)
        return command

    def auth_hint(self) -> str:
        return 'run `qodercli status` and `qodercli --print "test"` to diagnose'
