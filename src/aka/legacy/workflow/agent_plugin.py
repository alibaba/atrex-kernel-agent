"""Select the shared Agent with Workflow-owned recovery host adapters."""

from aka.agent.plugin import (
    AGENT,
    Config as Config,
    Defaults as Defaults,
    options,
    selected_guard,
    validate_config as validate_config,
)

name = "agent-plugin"
provide = ("agent",)
identity_packages = (
    "aka.agent",
    "aka.legacy.workflow",
    "aka.execution",
    "aka.contracts.agent",
)


def apply(ctx, config):
    from .agent_host import create_agent

    agent = create_agent(
        options(config),
        shell_guard=selected_guard(ctx),
        session_name=config.get("session_name", ""),
    )
    ctx.effect(agent.close, label="close Workflow Agent")
    ctx.provide(AGENT, agent)
