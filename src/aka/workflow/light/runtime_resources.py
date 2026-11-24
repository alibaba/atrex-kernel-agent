"""Install selected resources without changing a recorded candidate boundary."""
from pathlib import Path

from aka.contracts.content import WorkspaceFile
from aka.contracts.workspace import WorkspaceExclusion


def runtime_resource_registry(campaign_root: Path, *, agate_tools, wiki: bool, research_modules=()):
    """Adapt shared contributions to Light's Agent-facing Runtime interface."""
    from aka.resources.contributions import selected_resources
    from aka.resources.plugins import PluginRegistry
    from aka.resources.runtime_measurement import scoped_measurement_skill
    from aka.resources.runtime_research import scoped_research_skill

    resources, skills = selected_resources()
    # These are host/operator manuals, not the Session's callable interface.
    excluded = ("aka-execution", "atrex-gpu-gateway") + (() if wiki else ("kernel-wiki",))
    selected = []
    for skill in skills:
        if skill.name in excluded:
            continue
        if skill.name == "gpu-measurement":
            skill = scoped_measurement_skill(skill, campaign_root / "resources", agate_tools)
        if skill.name == "runtime-records":
            skill = scoped_research_skill(skill, campaign_root / "resources", research_modules)
        selected.append(skill)
    return PluginRegistry(resources=resources, agent_skills=tuple(selected),
        excluded_plugins=() if wiki else ("gpu-wiki",), excluded_skills=excluded)


def install_runtime_resources(campaign, workspace: Path) -> None:
    """Refresh non-source inputs without changing the Agent's index or ignore file."""
    owner = campaign.candidate_workspace
    # Peer Git contains only candidate source. Every lineage also needs the
    # public Task contract, including on resume; private evaluator files are
    # deliberately absent from problem.agent.files.
    owner.install_files(workspace, campaign.problem.agent.files, immutable=True)
    owner.install_files(workspace, campaign.problem_provider.runtime_files(campaign.problem), immutable=True)
    campaign._module_resources.install(workspace)
    install_reference_knowledge(workspace, campaign.agent_reference_paths)


def install_reference_knowledge(workspace: Path, references) -> None:
    """Link configured knowledge without replacing Agent files or following links."""
    import os
    from aka.core.filesystem import open_private_directory

    if not references:
        return
    # Keep the old path as an alias; the Wiki-facing mount is the discoverable entry.
    wiki = workspace.resolve(strict=True) / "gpu-wiki"
    with open_private_directory(wiki):
        pass
    with open_private_directory(wiki / "references") as directory:
        for name, configured in references:
            source = Path(configured).resolve(strict=True)
            if not source.is_dir() or Path(name).name != name or name in {".", ".."}:
                raise ValueError("Invalid reference knowledge directory")
            try:
                os.symlink(str(source), name, dir_fd=directory, target_is_directory=True)
            except FileExistsError:
                if os.readlink(name, dir_fd=directory) != str(source):
                    raise ValueError(f"Reference knowledge link changed: gpu-wiki/references/{name}")
    with open_private_directory(workspace.resolve(strict=True) / "references") as directory:
        for name, configured in references:
            source = Path(configured).resolve(strict=True)
            if not source.is_dir() or Path(name).name != name or name in {".", ".."}:
                raise ValueError("Invalid reference knowledge directory")
            try:
                os.symlink(str(source), name, dir_fd=directory, target_is_directory=True)
            except FileExistsError:
                if os.readlink(name, dir_fd=directory) != str(source):
                    raise ValueError(f"Reference knowledge link changed: references/{name}")


def reference_knowledge_access(workspace: Path, references):
    """Use the public Agent access contract for both native and sandbox Sessions."""
    from aka.contracts.filesystem import FilesystemAccess, FilesystemMount

    mounts = []
    for name, configured in references:
        source = Path(configured).resolve(strict=True)
        for base in (workspace / "references", workspace / "gpu-wiki" / "references"):
            link = base / name
            if not source.is_dir() or not link.is_symlink() or link.resolve(strict=True) != source:
                raise ValueError(f"Reference knowledge link changed: {base.name}/{name}")
        mounts.append(FilesystemMount(source, source, origin=f"Reference knowledge:{name}"))
    return FilesystemAccess(tuple(mounts), read_only_paths=(workspace / "references", workspace / "gpu-wiki") if references else ())


def agent_runtime_directive(agent_cli: str, *, is_ppu: bool = False, wiki: bool = True) -> str:
    root = ".agents" if agent_cli in {"codex", "pi"} else ".qoder" if agent_cli == "qodercli" else ".claude"
    wiki_policy = "" if wiki else "GPU Wiki queries are disabled for this Campaign; use the supplied reference directories instead. "
    return (f"- `{root}/skills/` contains the selected module Skills. "
            "Use `python3 tools/sandbox.py --help` to discover enabled Runtime tools. "
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
