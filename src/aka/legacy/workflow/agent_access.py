"""Compose Workflow visibility policy from selected modules' public descriptions."""
from pathlib import Path
from importlib.resources import files
from aka.contracts.agent import AgentAccess, AgentMount
from .store import RUNTIME_DIR, VERIFY_DIR


def for_workspace(campaign, workspace, *, resources=True, read_only=()):
    workspace = Path(workspace).resolve(strict=True)
    # Direct compatibility calls select the Legacy workspace explicitly. Campaigns
    # always use their injected CandidateWorkspace.
    candidate = getattr(campaign, "candidate_workspace", None)
    if candidate is None:
        from aka.legacy.task.candidate_workspace.provider import GitCandidateWorkspace
        candidate = GitCandidateWorkspace()
    view = candidate.filesystem_access(workspace, include_git=True)
    mounts = list(view.mounts)
    hidden = [workspace / name for name in (RUNTIME_DIR, VERIFY_DIR, "private-review-observations")]
    hidden.extend(view.hidden_paths)
    protected = list(view.read_only_paths)
    protected.extend(workspace / name for name in ("CLAUDE.md", "framework_baseline_resume.json"))
    protected.extend(Path(p) for p in read_only)
    private = getattr(campaign, "private_reference_dir", None)
    if private is not None:
        private = Path(private).resolve(strict=True)
        if private == workspace:
            raise ValueError("Agent workspace cannot be the private evaluator corpus")
        if private.is_relative_to(workspace):
            hidden.append(private)
    # Package identity is explicit; interpreting system runtime paths belongs to Agent.
    runtime = Path(str(files("aka"))).resolve(strict=True)
    mounts.append(AgentMount(runtime, runtime))
    if resources:
        selected = campaign.plugin_registry.filesystem_access(workspace)
        mounts.extend(selected.mounts)
        protected.extend(selected.read_only_paths)
        hidden.extend(selected.hidden_paths)
    hidden = tuple(p for p in set(hidden) if not any(p != q and p.is_relative_to(q) for q in hidden))
    normalized = []
    for mount in sorted(set(mounts), key=lambda m: len(m.source.parts)):
        if private is not None and (mount.source.is_relative_to(private) or private.is_relative_to(mount.source)):
            raise ValueError("Selected Agent resource exposes private evaluator inputs")
        if any(mount.source.is_relative_to(p) or p.is_relative_to(mount.source) for p in hidden):
            raise ValueError("Agent resource exposes private Workflow state")
        if any(mount.source.is_relative_to(m.source) and mount.writable == m.writable for m in normalized):
            continue
        normalized.append(mount)
    protected = sorted(set(p for p in protected if p.exists() and not p.is_symlink()
                           and not any(p.is_relative_to(h) for h in hidden)), key=lambda p: len(p.parts))
    readonly = []
    for path in protected:
        if not any(path.is_relative_to(p) for p in readonly):
            readonly.append(path)
    return AgentAccess(tuple(normalized), hidden, tuple(readonly))
