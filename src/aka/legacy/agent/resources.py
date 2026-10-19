"""Project installer results into the existing Agent tool discovery response."""
from aka.contracts.agent_resources import AgentResourceInstallation


def discover_resources(installation: AgentResourceInstallation) -> dict:
    return {"tools": installation.catalog(), "skills": installation.skill_catalog()}
