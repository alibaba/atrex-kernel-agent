"""Configure central research/submission Runtime operations.

The historical plugin key remains ``light_journal`` for launch-profile
compatibility, but this factory no longer creates or binds Episode Journals.
"""
from dataclasses import dataclass
from aka.light.workflow.journal_service import SupervisorJournalService


@dataclass(frozen=True)
class JournalFactory:
    tool_modules: tuple[str, ...] | list[str] = ("directions", "experiments")

    def __post_init__(self):
        modules = tuple(self.tool_modules)
        if len(set(modules)) != len(modules) or set(modules) - {"directions", "experiments"}:
            raise ValueError("tool_modules may contain only unique directions and experiments")
        object.__setattr__(self, "tool_modules", modules)

    @property
    def research_records(self) -> bool:
        return bool(self.tool_modules)

    @property
    def directions_enabled(self) -> bool:
        return "directions" in self.tool_modules

    @property
    def experiments_enabled(self) -> bool:
        return "experiments" in self.tool_modules

    def open(self, **options):
        return SupervisorJournalService(**options)
