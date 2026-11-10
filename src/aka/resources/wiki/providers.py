"""Select knowledge providers as Resources contributions, without Workflow keys."""
from aka.contracts.wiki import WikiProvider
from aka.resources.contributions import _selected_scope


def contribute(ctx, provider):
    if not isinstance(provider, WikiProvider):
        raise TypeError("Wiki resource requires the public WikiProvider contract")
    ctx.contribute("knowledge_providers", "wiki", provider)


def selected_provider():
    scope = _selected_scope.get()
    entries = scope.resolve("knowledge_providers") if scope is not None else ()
    if len(entries) > 1 or any(entry.name != "wiki" or not isinstance(entry.value, WikiProvider) for entry in entries):
        raise ValueError("Resources require at most one selected Wiki provider")
    return entries[0].value if entries else None
