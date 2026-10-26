"""Translate selected Workflow inputs into an explicit Agent filesystem grant."""

from pathlib import Path

from aka.contracts.agent import AgentAccess, AgentMount
from aka.legacy.task.candidate_workspace.excludes import RUNTIME_DIR, VERIFY_DIR
from aka.legacy.task.candidate_workspace.policy import IMMUTABLE_BASELINE_PATHS
from aka.legacy.resources.plugin_runtime.registry import STATE_DIR


def _git_mounts(workspace):
    association = workspace / ".git"
    if not association.is_file():
        return ()
    text = association.read_text().strip()
    if not text.startswith("gitdir: "):
        raise ValueError("Invalid candidate Git association")
    directory = Path(text[8:])
    if not directory.is_absolute():
        directory = workspace / directory
    directory = directory.resolve(strict=True)
    common = (directory / (directory / "commondir").read_text().strip()).resolve(strict=True)
    head = (directory / "HEAD").read_text().strip()
    prefix = "ref: refs/heads/atrex/"
    # Each episode has its own ref directory so Git's adjacent .lock and rename
    # transaction cannot change the incumbent or another episode's branch.
    branch = head.removeprefix("ref: ")
    if (not head.startswith(prefix) or not branch.endswith("/candidate")
            or len(Path(branch).parts) != 5 or ".." in Path(branch).parts):
        raise ValueError("Sandboxed linked candidates require an isolated atrex/<episode>/candidate branch; create a new candidate worktree")
    selected = [(directory, True), (common / "objects", True),
                ((common / branch).parent, True)]
    for name in ("config", "packed-refs", "shallow", "info/exclude"):
        if (common / name).is_file():
            selected.append((common / name, False))
    # Keep reflog updates in the same isolated episode namespace.
    log = common / "logs" / branch
    if log.parent.is_dir():
        selected.append((log.parent, True))
    if (common / "objects/info/alternates").exists():
        raise ValueError("Sandboxed candidates cannot use external Git object alternates")
    return tuple(AgentMount(path, path, writable) for path, writable in selected)


def for_workspace(campaign, workspace, *, resources=True, read_only=()):
    from importlib.resources import files
    import sys
    from aka.agent.execution.bubblewrap import system_visible

    workspace = Path(workspace).resolve(strict=True)
    hidden = [workspace / name for name in (RUNTIME_DIR, VERIFY_DIR, "private-review-observations")]
    private = getattr(campaign, "private_reference_dir", None)
    if private is not None:
        private = Path(private).resolve(strict=True)
        if private == workspace:
            raise ValueError("Agent workspace cannot be the private evaluator corpus")
        if private.is_relative_to(workspace):
            hidden.append(private)
    hidden = tuple(p for p in hidden if not any(p != q and p.is_relative_to(q) for q in hidden))
    mounts = list(_git_mounts(workspace))
    runtime = Path(str(files("aka"))).resolve(strict=True)
    interpreter_libraries = (Path(sys.prefix) / "lib", Path(sys.base_prefix) / "lib")
    if not system_visible(runtime) and not any(runtime.is_relative_to(path) for path in interpreter_libraries):
        mounts.append(AgentMount(runtime, runtime))
    protected = [workspace / name for name in (*IMMUTABLE_BASELINE_PATHS, "CLAUDE.md", STATE_DIR, "framework_baseline_resume.json")]
    protected.extend(Path(path) for path in read_only)
    if resources:
        registry = campaign.plugin_registry
        registry.check_lock(workspace)
        for name, source in registry.mounts().items():
            source = Path(source).resolve(strict=True)
            link = workspace / name
            if not link.is_symlink() or link.resolve(strict=True) != source:
                raise ValueError("Selected Agent resource was not installed in the workspace")
            if (private is not None and (source.is_relative_to(private) or private.is_relative_to(source))):
                raise ValueError("Selected Agent resource exposes private evaluator inputs")
            if source.is_relative_to(workspace):
                protected.append(source)
            elif not system_visible(source):
                mounts.append(AgentMount(source, source))
    # Several Skill aliases can point to one source; grant its canonical tree once.
    normalized = []
    for mount in sorted(set(mounts), key=lambda m: len(m.source.parts)):
        if any(mount.source.is_relative_to(m.source) and mount.writable == m.writable for m in normalized):
            continue
        if any(mount.source.is_relative_to(p) or p.is_relative_to(mount.source) for p in hidden):
            raise ValueError("Agent resource exposes private Workflow state")
        normalized.append(mount)
    protected = sorted(set(p for p in protected if p.exists() and not p.is_symlink()
                           and not any(p.is_relative_to(h) for h in hidden)), key=lambda p: len(p.parts))
    read_only = []
    for path in protected:
        if not any(path.is_relative_to(p) for p in read_only):
            read_only.append(path)
    return AgentAccess(tuple(normalized), hidden, tuple(read_only))
