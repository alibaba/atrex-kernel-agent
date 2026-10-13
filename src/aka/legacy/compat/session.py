"""Retain old executor/default construction outside Workflow."""

from aka.legacy.workflow.session import *
from aka.legacy.workflow.session import LongSessionRunner as SessionRunner
from . import main_adapter


class LongSessionRunner(SessionRunner):
    def __init__(self, executor=None, agent_cli="claude", *, agent=None):
        from aka.legacy.agent.selection import LegacyAgent

        self._executor = executor
        self._owns_agent = agent is None
        if agent is None:
            execute = executor or main_adapter.run_bounded
            agent = LegacyAgent(
                process_runner=lambda command, *, cwd, timeout, env: execute(
                    command, cwd, timeout, env
                )
            )
        super().__init__(
            agent=agent,
            agent_cli=agent_cli,
            usage_policy=lambda turn: legacy_budget_tokens(turn, agent_cli),
        )

    def run(self, *args, **kwargs):
        if not self._owns_agent:
            return super().run(*args, **kwargs)
        from aka.legacy.agent.selection import LegacyAgent

        execute = self._executor or main_adapter.run_bounded
        self.agent = LegacyAgent(
            process_runner=lambda command, *, cwd, timeout, env: execute(
                command, cwd, timeout, env
            )
        )
        try:
            return super().run(*args, **kwargs)
        finally:
            self.agent.close()


def legacy_budget_tokens(turn, agent_cli):
    if turn.terminal_usage.total_tokens is not None:
        return turn.terminal_usage.total_tokens
    if agent_cli == "codex":
        return 0
    from aka.legacy.agent.runtime import terminal_usage_from_stream

    return terminal_usage_from_stream(turn.stdout).total_tokens or 0
