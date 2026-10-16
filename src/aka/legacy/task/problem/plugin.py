"""Select the existing problem loader as a Core plugin."""
from aka.core.keys import ServiceKey

PROBLEM = ServiceKey("problem", "ProblemProvider", module="aka.contracts.problem_provider")
name = "problem"
provide = ("problem",)
identity_files = ("plugin.py", "legacy.py", "layout.py", "inputs.py", "snapshot.py", "evaluation.py", "resources.py", "native_harness.py", "sol_harness.py", "prompts/generalize_agent_problem.md")
identity_packages = ("aka.contracts.problem", "aka.contracts.problem_provider", "aka.contracts.agent_resources", "aka.contracts.content")
Config = {
    "type": "object",
    "properties": {"optimization_mode": {"type": "string", "enum": ["leaderboard", "production"]}},
    "required": ["optimization_mode"],
    "additionalProperties": False,
}
Defaults = {"optimization_mode": "leaderboard"}


def apply(ctx, config):
    from aka.legacy.task.problem.resources import contribute_resources
    contribute_resources(ctx)
    from .legacy import LegacyProblemProvider
    ctx.provide(PROBLEM, LegacyProblemProvider(config["optimization_mode"]))

