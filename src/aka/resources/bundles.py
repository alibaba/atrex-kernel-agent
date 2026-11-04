"""Explicit installed resource sets, independent of a Workflow implementation."""
from importlib.resources import files
from pathlib import Path

from aka.contracts.agent_resources import AgentSkill, WorkspaceResourceContribution


def resource_path(relative: str = "") -> Path:
    """Resolve one packaged content member without consulting a checkout root."""
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Resource content requires a package-relative path")
    root = Path(str(files("aka.resources"))).resolve()
    selected = (root / path).resolve()
    if not selected.is_relative_to(root) or not selected.exists():
        raise FileNotFoundError(f"Missing installed resource content: {relative}")
    return selected


_SKILLS = {
    "gen-plan": "Generate a reviewed implementation plan from a concrete draft.",
    "gpu-kernel-baseline": "Implement and validate the initial GPU kernel baseline.",
    "gpu-kernel-episode-loop": "Run the profile, research, implementation and verification episode loop.",
    "gpu-measurement": "Request measurements through the selected Evaluation and Experiment.",
    "runtime-records": "Inspect retained evidence and submit the selected Workflow's Episode report.",
    "kernel-wiki": "Query selected GPU kernel knowledge and reference material.",
    "ppu-acu-joint-profile": "Inspect PPU and ACU profiling evidence.",
    "autonomous-gpu-kernel-timeline": "Inspect an explicit GPU kernel timeline.",
    "runtime-report": "Produce an Episode report from retained research evidence.",
}
_BUNDLES = {
    "kernel-development": (
        ("gen-plan", "gpu-kernel-baseline", "gpu-kernel-episode-loop"),
        (("workflow-baseline-agent", "agents/gpu-kernel-baseline.md", ".claude/agents/gpu-kernel-baseline.md", "agent"),
         ("workflow-instructions", "reference/kernel-development/CLAUDE.md", None, "runtime"),
         ("workflow-iteration-schema", "reference/kernel-development/v_iteration.schema.json", "reference/v_iteration.schema.json", "agent"),
         ("workflow-workspace-reference", "reference/kernel-development/README.md", "reference/README.md", "agent")),
    ),
    "measurement-session": (
        ("gpu-measurement",),
        (("measurement-session-client", "tools/sandbox.py", "tools/sandbox.py", "agent"),
         ("plugin-session-client", "tools/plugin.py", "tools/plugin.py", "agent"),
         ("measurement-session-instructions", "reference/CLAUDE.md", None, "runtime")),
    ),
    "research-records": (("runtime-records",), ()),
    "kernel-wiki": (("kernel-wiki",), ()),
    "profiling-analysis": (("ppu-acu-joint-profile", "autonomous-gpu-kernel-timeline"), ()),
    "episode-report": (("runtime-report",), ()),
}


def bundle_contributions(names):
    """Validate the entire explicit selection before returning contributions."""
    if not isinstance(names, (tuple, list)) or any(not isinstance(name, str) for name in names):
        raise ValueError("Resource bundles must be an explicit list of names")
    if len(names) != len(set(names)) or any(name not in _BUNDLES for name in names):
        raise ValueError("Resource bundles contain an unknown or duplicate name")
    resources, skills = [], []
    for name in names:
        selected_skills, selected_resources = _BUNDLES[name]
        skills.extend(AgentSkill(skill, resource_path("skills/KernelWiki" if skill == "kernel-wiki" else f"skills/{skill}"), _SKILLS[skill])
                      for skill in selected_skills)
        resources.extend(WorkspaceResourceContribution(key, resource_path(source), destination,
                                                       visibility=visibility)
                         for key, source, destination, visibility in selected_resources)
    return tuple(resources), tuple(skills)


def contribute_bundles(ctx, names):
    resources, skills = bundle_contributions(names)
    for resource in resources:
        ctx.contribute("workspace_resources", resource.name, resource)
    for skill in skills:
        ctx.contribute("agent_skills", skill.name, skill)


def contribute_reference_projects(ctx, directories):
    """Declare only explicitly selected project roots through existing contracts."""
    selected = []
    for index, source in enumerate(directories):
        source = Path(source).expanduser().resolve()
        if not source.is_dir():
            raise ValueError("Selected reference project must be a directory")
        selected.append(WorkspaceResourceContribution(f"workflow-reference-{index}", source,
                                                     "reference-projects/" + source.name))
    for resource in selected:
        ctx.contribute("workspace_resources", resource.name, resource)
