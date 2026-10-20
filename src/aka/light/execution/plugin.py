from pathlib import Path
from aka.contracts.agent_resources import AgentSkill

name = "execution"
provide = ("executor",)
identity_packages = ("aka.light.execution", "aka.light.supervisor", "aka.light.orchestrator")
Config = {"type": "object", "properties": {}, "additionalProperties": False}


def apply(ctx, config):
    from .provider import ReliableExecutor
    ctx.provide("executor", ReliableExecutor())
    ctx.contribute("agent_skills", "aka-execution", AgentSkill(
        "aka-execution", Path(__file__).resolve().parent / "skills/aka-execution",
        "Reliable GPU transport: recover infra jobs; replay uncertain submissions only with persisted idempotency."))
