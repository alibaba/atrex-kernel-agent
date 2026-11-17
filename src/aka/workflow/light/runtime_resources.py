"""Install selected resources without changing a recorded candidate boundary."""
from pathlib import Path

from aka.contracts.content import WorkspaceFile
from aka.contracts.workspace import WorkspaceExclusion


def runtime_resource_registry(campaign_root: Path, *, agate_tools, wiki: bool):
    """Adapt shared contributions to Light's Agent-facing Runtime interface."""
    from aka.resources.contributions import selected_resources
    from aka.resources.plugins import PluginRegistry
    from aka.resources.runtime_measurement import scoped_measurement_skill

    resources, skills = selected_resources()
    # These are host/operator manuals, not the Session's callable interface.
    excluded = ("aka-execution", "atrex-gpu-gateway") + (() if wiki else ("kernel-wiki",))
    selected = []
    for skill in skills:
        if skill.name in excluded:
            continue
        if skill.name == "gpu-measurement":
            skill = scoped_measurement_skill(skill, campaign_root / "resources", agate_tools)
        selected.append(skill)
    return PluginRegistry(resources=resources, agent_skills=tuple(selected),
        excluded_plugins=() if wiki else ("gpu-wiki",), excluded_skills=excluded)


def install_runtime_resources(campaign, workspace: Path) -> None:
    """Refresh non-source inputs without changing the Agent's index or ignore file."""
    owner = campaign.candidate_workspace
    owner.install_files(workspace, campaign.problem_provider.runtime_files(campaign.problem), immutable=True)
    campaign._module_resources.install(workspace)


def agent_runtime_directive(agent_cli: str, *, is_ppu: bool = False, wiki: bool = False) -> str:
    root = ".agents" if agent_cli in {"codex", "pi"} else ".qoder" if agent_cli == "qodercli" else ".claude"
    wiki_policy = "" if wiki else "GPU Wiki queries are disabled for this Campaign; use the supplied reference directories instead. "
    return (f"- `{root}/skills/` contains the selected module Skills. "
            "Use `python -m aka.resources.cli list` to discover installed tools, "
            "measurement methods and optional knowledge/profiling capabilities. "
            "Read their supplied Skills for request schemas and examples. "
            + wiki_policy +
            "Before choosing DSL imports, run `python3 tools/sandbox.py --kind env`. "
            "Its `task_worker_requirements` field lists the selected Task's per-Job dependency pins "
            "and API compatibility notes; these may differ from the base image or local Python. "
            "For FlyDSL 0.3.4, read the selected shared reference "
            "`reference/dsl/flydsl/0.3.4/guide.md` before implementing or repairing imports, "
            "JIT/Context handling or memory operations. Match the Task's dependency version; "
            "if the reference is absent or mismatched, record that fact and inspect the matching "
            "version through the selected sandbox or pinned sources.")
