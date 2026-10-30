"""Select the existing problem loader as a Core plugin."""
from pathlib import Path

from aka.core.keys import ServiceKey

PROBLEM = ServiceKey("problem", "ProblemProvider", module="aka.contracts.problem_provider")
name = "problem"
provide = ("problem",)
identity_files = ("plugin.py", "legacy.py", "layout.py", "inputs.py", "snapshot.py", "evaluation.py", "resources.py", "native_harness.py", "sol_harness.py", "prompts/generalize_agent_problem.md")
identity_packages = ("aka.contracts.problem", "aka.contracts.problem_provider", "aka.contracts.agent_resources", "aka.contracts.content")
Config = {
    "type": "object",
    "properties": {
        "optimization_mode": {"type": "string", "enum": ["leaderboard", "production"]},
        "worker_requirements": {"type": "string"},
        "source_tree": {"type": "object", "properties": {
            "manifest": {"type": "string", "minLength": 1},
            "repository": {"type": "string", "minLength": 1},
            "directory": {"type": "string", "minLength": 1},
        }, "required": ["manifest"], "additionalProperties": False},
    },
    "required": ["optimization_mode"],
    "additionalProperties": False,
}
Defaults = {"optimization_mode": "leaderboard", "worker_requirements": ""}
interpolate = ("worker_requirements", "source_tree.manifest", "source_tree.repository", "source_tree.directory")


def apply(ctx, config):
    from aka.task.problem.resources import contribute_resources
    contribute_resources(ctx)
    from .legacy import LegacyProblemProvider
    requirements = config["worker_requirements"].strip()
    ctx.provide(PROBLEM, LegacyProblemProvider(
        config["optimization_mode"],
        worker_requirements=Path(requirements).expanduser().resolve() if requirements else None,
        source_tree=config.get("source_tree"),
    ))

