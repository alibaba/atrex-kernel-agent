"""Install selected resources without changing a recorded candidate boundary."""
from pathlib import Path

from aka.contracts.content import WorkspaceFile
from aka.contracts.workspace import WorkspaceExclusion


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
