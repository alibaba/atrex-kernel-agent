"""Assemble the shared Agent and its independently selected execution adapter."""

from aka.core.keys import ServiceKey

AGENT = ServiceKey("agent", "Agent", module="aka.contracts.agent")
name = "agent"
provide = ("agent",)
identity_packages = (
    "aka.agent",
    "aka.contracts.agent",
    "aka.contracts.agent_resources",
)
Config = {
    "type": "object",
    "properties": {
        "session_name": {"type": "string"},
        "sandbox": {"type": "string", "enum": ["none", "bwrap"]},
        "bwrap_executable": {"type": "string", "minLength": 1},
        "timeout_seconds": {"type": "number", "minimum": 0},
        "state_directory": {"type": "string", "minLength": 1},
    },
    "additionalProperties": False,
}
Defaults = {"sandbox": "none", "bwrap_executable": "bwrap", "session_name": ""}


def options(config):
    from pathlib import Path
    from .provider import AgentOptions

    return AgentOptions(
        config.get("sandbox", "none"),
        config.get("bwrap_executable", "bwrap"),
        config.get("timeout_seconds"),
        Path(config["state_directory"]) if config.get("state_directory") else None,
    )


def validate_config(config):
    value = options(config)
    if value.sandbox == "bwrap":
        from .execution.bubblewrap import executable

        executable(value.bwrap_executable)


def selected_guard(ctx):
    # Resolve lazily after the other module contributions have been mounted.
    cached = {}

    def get():
        from aka.contracts.agent_resources import WorkspaceResourceContribution
        from .resources import selected_runtime_file

        entries = [
            entry.value
            for entry in ctx.resolve("workspace_resources")
            if entry.name == "execution-shell-guard"
        ]
        if len(entries) != 1 or not isinstance(
            entries[0], WorkspaceResourceContribution
        ):
            raise ValueError(
                "Agent requires exactly one selected execution-shell-guard"
            )
        contribution = entries[0]
        if contribution not in cached:
            cached[contribution] = selected_runtime_file(contribution)
        return cached[contribution]

    return get


def apply(ctx, config):
    from .provider import Agent

    agent = Agent(
        options(config),
        shell_guard=selected_guard(ctx),
        session_name=config.get("session_name", ""),
    )
    ctx.effect(agent.close, label="close Agent")
    ctx.provide(AGENT, agent)
