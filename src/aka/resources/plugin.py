"""Core plugin that publishes the generic resource host service."""

from aka.core import ServiceKey

from .host import ResourcesHost

AGENT_RESOURCES = ServiceKey(
    "agent_resources",
    "AgentResources",
    module="aka.contracts.tools",
    protocol_version=1,
    doc="Settled Core contributions and bound Agent resource plans.",
)


def apply(ctx, config):
    """Publish a scope-bound host; contribution collection happens after settle."""
    ctx.provide(
        AGENT_RESOURCES,
        ResourcesHost(
            ctx.current_scope, known_phases=tuple(config.get("known_phases", ()))
        ),
    )


name = "resources"
provide = ("agent_resources",)
identity_packages = (
    "aka.resources",
    "aka.contracts.tools",
    "aka.contracts.agent_instructions",
    "aka.contracts.attribution",
    "aka.contracts.agent_resources",
)
Config = {
    "type": "object",
    "properties": {
        "known_phases": {"type": "array", "items": {"type": "string"}},
    },
    "additionalProperties": False,
}
Defaults = {}


__all__ = [
    "AGENT_RESOURCES",
    "Config",
    "Defaults",
    "apply",
    "identity_packages",
    "name",
    "provide",
]
