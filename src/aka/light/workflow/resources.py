"""Declare the selected Light Workflow's tool syntax and report instructions."""
from importlib.resources import files
from pathlib import Path
from aka.contracts.agent_resources import AgentSkill, WorkspaceResourceContribution


def contribute(ctx, config):
    root = Path(str(files("aka.light")))
    for name, description in (("gpu-measurement", "Request measurements through the selected Evaluation and Experiment."),
            ("runtime-records", "Inspect retained evidence and submit this Workflow's Episode report.")):
        ctx.contribute("agent_skills", name, AgentSkill(name, root / "skills" / name, description))
    selected = (
        WorkspaceResourceContribution("light-gateway-client", root / "tools/sandbox.py", "tools/sandbox.py"),
        WorkspaceResourceContribution("light-plugin-client", root / "tools/plugin.py", "tools/plugin.py"),
        WorkspaceResourceContribution("light-workflow-instructions", root / "reference/CLAUDE.md", visibility="runtime"),
    )
    for resource in selected:
        ctx.contribute("workspace_resources", resource.name, resource)
