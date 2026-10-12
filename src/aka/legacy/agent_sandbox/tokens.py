"""Registered Core seam, separate from Agent-callable tool plugins."""
from aka.core.keys import ServiceKey

AGENT_SANDBOX = ServiceKey(
    "agent_sandbox", "AgentSandbox", module="aka.contracts.agent_sandbox",
    doc="Coordinator-side preparation and wrapping of coding/auxiliary sessions",
)
