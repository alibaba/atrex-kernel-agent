"""Provide one Agent module with the four existing backend adapters."""

from aka.core.keys import ServiceKey

AGENT = ServiceKey("agent", "Agent", module="aka.contracts.agent")
name = "agent"
provide = ("agent",)
identity_packages = ("aka.legacy.agent", "aka.contracts.agent")
Config = {"type": "object", "properties": {"session_name":{"type":"string"}}, "additionalProperties": False}


def apply(ctx, config):
    from .selection import LegacyAgent

    agent = LegacyAgent(session_name=config.get("session_name", ""))
    ctx.effect(agent.close, label="close Agent selection")
    ctx.provide(AGENT, agent)
