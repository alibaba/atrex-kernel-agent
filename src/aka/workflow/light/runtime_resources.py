"""Install selected resources without changing a recorded candidate boundary."""
from pathlib import Path

from aka.contracts.content import WorkspaceFile
from aka.contracts.workspace import WorkspaceExclusion


def install_runtime_resources(campaign, workspace: Path) -> None:
    owner = campaign.candidate_workspace
    revision = owner.current_revision(workspace)
    original = owner.read_working(workspace, (".gitignore",)).files[".gitignore"]
    if revision:
        recorded = owner.read_revision(workspace, revision, [".gitignore"]).files[".gitignore"]
        if recorded is None or original != recorded:
            raise ValueError("recorded workspace .gitignore is missing or changed")
    text = original.decode("utf-8") if original is not None else ""
    runtime = campaign.problem_provider.runtime_files(campaign.problem)
    owner.install_files(workspace, runtime, immutable=True)
    exclusions = (
        WorkspaceExclusion("tools", subtree=True),
        WorkspaceExclusion("skills", subtree=True),
        WorkspaceExclusion("reference", subtree=True),
        WorkspaceExclusion("reference-projects", subtree=True),
        WorkspaceExclusion("gpu-wiki", subtree=True),
        *(WorkspaceExclusion(name, subtree=True) for name in (".claude", ".qoder", ".agents")),
        WorkspaceExclusion(".orchestrator_state.json"),
        WorkspaceExclusion(".orchestrator_mode.json"),
        *((WorkspaceExclusion("atrex-bench", subtree=True),) if runtime else ()),
    )
    resources = campaign._module_resources
    try:
        resources.install(workspace)
        if revision:
            installed = resources.installed_paths(workspace)
            owner.exclude_runtime_paths(workspace, (*exclusions, *(
                WorkspaceExclusion(path, subtree=(workspace / path).is_dir()) for path in installed)))
        else:
            # New V0 records the ignore choices with its first source commit.
            selected = owner.read_working(workspace, (".gitignore",)).files[".gitignore"]
            text = selected.decode("utf-8") if selected is not None else text
            lines = text.splitlines()
            missing = ["/" + value.path for value in exclusions if "/" + value.path not in lines]
            if missing:
                owner.install_files(workspace, (WorkspaceFile(".gitignore",
                    text=text.rstrip() + "\n" + "\n".join(missing) + "\n"),))
    finally:
        if revision:
            # Resources owns discovery and locks; Workflow keeps recorded Git
            # rules stable and supplies local additions through Workspace.
            owner.install_files(workspace, (WorkspaceFile(".gitignore", text=original.decode("utf-8")),))


def agent_runtime_directive(agent_cli: str, *, is_ppu: bool = False, wiki: bool = False) -> str:
    root = ".agents" if agent_cli in {"codex", "pi"} else ".qoder" if agent_cli == "qodercli" else ".claude"
    return (f"- `{root}/skills/` contains the selected module Skills. "
            "Use `python -m aka.resources.cli list` to discover installed tools, "
            "measurement methods and optional knowledge/profiling capabilities. "
            "Read their supplied Skills for request schemas and examples. "
            "Before choosing DSL imports, run `python3 tools/sandbox.py --kind env`. "
            "Its `task_worker_requirements` field lists the selected Task's per-Job dependency pins "
            "and API compatibility notes; these may differ from the base image or local Python. "
            "For FlyDSL 0.3.4, read the selected shared reference "
            "`reference/dsl/flydsl/0.3.4/guide.md` before implementing or repairing imports, "
            "JIT/Context handling or memory operations. Match the Task's dependency version; "
            "if the reference is absent or mismatched, record that fact and inspect the matching "
            "version through the selected sandbox or pinned sources.")
