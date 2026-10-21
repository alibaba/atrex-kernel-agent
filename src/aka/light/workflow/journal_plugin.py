"""Journal is a Workflow component, not a GPU execution service."""
from aka.core.keys import ServiceKey
JOURNAL = ServiceKey("light_journal", "JournalFactory", module="aka.light.contracts")
name = "journal-plugin"
provide = ("light_journal",)
identity_packages = ("aka.light.workflow.journal", "aka.light.workflow.journal_service")
Config = {"type": "object", "properties": {
    "research_records": {"type": "boolean"},
}, "additionalProperties": False}
Defaults = {"research_records": False}

def apply(ctx, config):
    from .journal import JournalFactory
    ctx.provide(JOURNAL, JournalFactory(**config))
