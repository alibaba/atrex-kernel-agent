"""Select the existing problem loader as a Core plugin."""
from aka.core.keys import ServiceKey

PROBLEM = ServiceKey("problem", "ProblemProvider", module="aka.contracts.problem_provider")
name = "problem"
provide = ("problem",)
identity_files = ("plugin.py", "legacy.py", "layout.py", "inputs.py", "snapshot.py", "evaluation.py", "resources.py")
identity_packages = ("aka.contracts.problem", "aka.contracts.problem_provider", "aka.contracts.content")
Config = {
    "type": "object",
    "properties": {"optimization_mode": {"type": "string", "enum": ["leaderboard", "production"]}},
    "required": ["optimization_mode"],
    "additionalProperties": False,
}
Defaults = {"optimization_mode": "leaderboard"}


def apply(ctx, config):
    from .legacy import LegacyProblemProvider
    ctx.provide(PROBLEM, LegacyProblemProvider(config["optimization_mode"]))

