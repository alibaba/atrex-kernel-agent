"""Journal is a Workflow component, not a GPU execution service."""
from aka.core.keys import ServiceKey
JOURNAL = ServiceKey("light_journal", "JournalFactory", module="aka.light.contracts")
name = "journal-plugin"
provide = ("light_journal",)
identity_packages = ("aka.light.workflow.journal", "aka.light.workflow.journal_service")
Config = {"type": "object", "properties": {
    "tool_modules": {
        "type": "array",
        "items": {"type": "string", "enum": ["directions", "experiments"]},
    },
}, "additionalProperties": False}
Defaults = {"tool_modules": []}

def apply(ctx, config):
    from .journal import JournalFactory
    ctx.provide(JOURNAL, JournalFactory(**config))
