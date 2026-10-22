"""Bind the existing durable Journal protocol to one controller-owned Episode."""
from dataclasses import dataclass
from aka.light.workflow.journal_service import (
    SupervisorJournalService,
    initialize_journal,
    load_journal,
    journal_lock,
)


@dataclass(frozen=True)
class JournalFactory:
    research_records: bool = False

    def __post_init__(self):
        if type(self.research_records) is not bool:
            raise ValueError("research_records must be a boolean")

    def initialize(self, path, *, episode, base_commit, branch, memory_version):
        expected = dict(episode=episode, base_commit=base_commit, episode_branch=branch,
                        memory_version=memory_version)
        with journal_lock(path):
            if path.exists():
                value = load_journal(path)
                if any(value.get(key) != item for key, item in expected.items()):
                    raise RuntimeError("Private Runtime Journal does not match the controller Episode")
                if value.get("research_records", True) != self.research_records:
                    raise RuntimeError(
                        "Cannot change research_records for an existing Episode; resume with its "
                        "original configuration and change the option at an Episode boundary"
                    )
            else:
                initialize_journal(path, episode=episode, base_commit=base_commit,
                                   branch=branch, memory_version=memory_version,
                                   research_records=self.research_records)

    def open(self, **options):
        return SupervisorJournalService(**options)
