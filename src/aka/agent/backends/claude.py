from __future__ import annotations
from pathlib import Path
from .base import BackendFiles, ClaudeLikeAdapter


class ClaudeAdapter(ClaudeLikeAdapter):
    id = "claude"
    settings_variable = "ATREX_CLAUDE_SESSION_SETTINGS"
    environment_variables = (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_MODEL",
        "ANTHROPIC_DEFAULT_SONNET_MODEL",
        "ANTHROPIC_DEFAULT_OPUS_MODEL",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC",
        "CLAUDE_CODE_DISABLE_AUTOUPDATER",
        "CLAUDE_CONFIG_DIR",
    )
    files = BackendFiles(
        (".claude.json", ".claude/settings.json", ".claude/.credentials.json"),
        (".claude",),
        (".claude/bin", ".claude/local"),
        (("CLAUDE_CONFIG_DIR", ".claude"),),
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
            "claude",
            "--print",
            "--verbose",
            "--dangerously-skip-permissions",
            "--output-format",
            "stream-json",
            "--session-id",
            session_id,
            "--effort",
            reasoning_effort,
        ]
        if settings:
            command += ["--settings", settings]
        command.append(prompt)
        return command

    def normalize_stream(self, stdout, *, settled_usage=None):
        from aka.agent.model import NormalizedAgentEvent, TokenUsage, resequence_agent_events
        from aka.agent.session_usage import UsageAccumulator

        events, _ = super().normalize_stream(stdout, settled_usage=settled_usage)
        accumulator = UsageAccumulator(self.id)
        accumulator.feed("provider/stdout.stream-json", stdout)
        if settled_usage:
            for message_id, usage in settled_usage.items():
                if message_id in accumulator.responses:
                    accumulator.responses[message_id]["usage"] = usage
        total = TokenUsage(**accumulator.report(finished=True)["total"])
        events = [event for event in events if event.kind != "terminal_usage"]
        events.append(NormalizedAgentEvent(0, "terminal_usage", total))
        return resequence_agent_events(events), total

    def auth_hint(self) -> str:
        return 'run `claude auth status` and `claude --print "test"` to diagnose'

    supports_resume = True

    def configure_environment(self, environment):
        values = dict(environment)
        if values.get("ANTHROPIC_AUTH_TOKEN"):
            values.pop("ANTHROPIC_API_KEY", None)
        return values

    def resume_command(self, command, session_id):
        command = list(command)
        try:
            index = command.index("--session-id")
        except ValueError as exc:
            raise RuntimeError("Claude command has no session identity") from exc
        command[index : index + 2] = ["--resume", session_id]
        return command

    def name_session(self, command, name, session_id):
        if name and "--name" not in command:
            command[1:1] = ["--name", name.format(session_id=session_id)]
        return command
