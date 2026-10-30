"""Read and restore complete declared candidate source through public facts."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
import stat
import tempfile

from aka.contracts.content import WorkspaceFile
from aka.contracts.workspace import RevisionContent
from aka.task.source.tree import POLICY, _directory_sources, editable, source_name, validate_files, validate_policy
from .files import MAX_FILE, read_regular_bytes
from .history import resolve_revision
from .records import _git, current_revision, read_working


def _policy(workspace, revision=None):
    if revision is not None:
        result = _git(workspace, "show", f"{revision}:{POLICY}", check=False)
        raw = result.stdout if not result.returncode else None
    else:
        path = workspace / POLICY
        if not path.exists() and not path.is_symlink():
            raw = None
        else:
            raw = read_regular_bytes(path, limit=MAX_FILE)
    return (validate_policy(json.loads(raw)), raw) if raw is not None else (None, None)


def _committed_files(workspace, revision, policy):
    if policy is None:
        result = _git(workspace, "show", f"{revision}:kernel.py", check=False)
        return {"kernel.py": result.stdout if not result.returncode else None}
    files = {}
    listing = _git(workspace, "ls-tree", "-r", "-z", revision).stdout
    for entry in listing.split(b"\0"):
        if not entry:
            continue
        metadata, encoded = entry.split(b"\t", 1)
        name = encoded.decode("utf-8")
        if not source_name(name):
            continue
        mode, kind, identity = metadata.split()
        if kind != b"blob" or mode not in {b"100644", b"100755"}:
            raise ValueError("committed source contains a link or special object")
        files[name] = _git(workspace, "cat-file", "blob", identity.decode()).stdout
    return validate_files(files, policy)


def _declared_files(workspace, policy, *, validate=True):
    names = tuple(dict.fromkeys((*policy["immutable_files"], *policy["editable_roots"])))
    selected = tuple(name for name in names if (workspace / name).exists() or (workspace / name).is_symlink())
    files = _directory_sources(workspace, selected)
    return validate_files(files, policy) if validate else files


def read_source(workspace, *, revision=None):
    workspace = Path(workspace).absolute()
    if revision is not None:
        revision = resolve_revision(workspace, revision)
    policy, raw = _policy(workspace, revision)
    if revision is not None:
        files = _committed_files(workspace, revision, policy)
    elif policy is not None:
        head = current_revision(workspace)
        if head:
            committed, recorded = _policy(workspace, head)
            if committed is None or recorded != raw:
                raise ValueError("working source declaration differs from its committed identity")
        files = _declared_files(workspace, policy)
    else:
        files = dict(read_working(workspace, ("kernel.py",)).files)
        if files["kernel.py"] is not None and len(files["kernel.py"]) > MAX_FILE:
            raise ValueError("source file exceeds the existing bounded limit")
    if raw is not None:
        files[POLICY] = raw  # Opaque contribution; only this owner interprets it.
    return RevisionContent(files)


def source_identity(source):
    from aka.task.source.tree import from_inputs
    data = dict(source.files)
    if data.get("kernel.py") is None:
        raise ValueError("source identity requires kernel.py")
    if data.get(POLICY) is not None:
        data = {name: value for name, value in data.items() if value is not None}
    return hashlib.sha256(from_inputs(data)).hexdigest()


def validate_source_deletions(workspace, names):
    from .policy import PROTECTED_PATHS
    from .context import load_context, uses_context
    protected = {*PROTECTED_PATHS, POLICY}
    if uses_context(workspace):
        value = load_context(workspace)
        protected.update(value["context"])
        protected.update(value.get("namespaces", {}))
    revision = current_revision(workspace)
    policy, _ = _policy(workspace, revision or None)
    for name in names:
        if (name in protected or any(name.startswith(root + "/") for root in protected)
                or name.startswith(".atrex_")
                or (policy is not None and source_name(name)
                    and (name in policy["immutable_files"] or not editable(policy, name)))):
            raise ValueError(f"cannot delete source constraints, immutable inputs or control: {name}")


def validate_source_record(workspace, selected):
    revision = current_revision(workspace)
    policy, raw = _policy(workspace, revision or None)
    if policy is None or not revision:
        return
    if POLICY in selected and selected[POLICY] != raw:
        raise ValueError("source declaration cannot change within a candidate")
    for name, data in selected.items():
        if name == POLICY or not source_name(name):
            continue
        expected = policy["immutable_files"].get(name)
        if expected is not None:
            import hashlib
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError(f"fixed source content cannot change: {name}")
        elif not editable(policy, name):
            raise ValueError(f"source record is outside declared editable roots: {name}")


def install_source(workspace, source):
    """Publish a complete declared source contribution in a Git-free draft too."""
    from .content import _install_files, selected_file_bytes
    from .records import _restore_files
    workspace = Path(workspace)
    if source.files.get("kernel.py") is None:
        raise ValueError("source publication requires kernel.py")
    files = tuple(WorkspaceFile(name, text=data.decode("utf-8"))
                  for name, data in source.files.items() if data is not None)
    selected = selected_file_bytes(workspace, files)
    policy = validate_policy(json.loads(selected[POLICY])) if POLICY in selected else None
    content = {name: data for name, data in selected.items() if name != POLICY}
    if policy is not None:
        validate_files(content, policy)
    elif set(selected) != {"kernel.py"}:
        raise ValueError("multiple source files require a complete Task source declaration")
    previous, _ = _policy(workspace)
    if previous != policy:
        if previous is not None:
            raise ValueError("source declaration cannot change within a draft")
        if (workspace / "kernel.py").exists() and policy is not None:
            raise ValueError("a single-file draft cannot acquire a new source declaration")
    current = _declared_files(workspace, previous, validate=False) if previous is not None else {}
    deleted = tuple(name for name in current.keys() - content.keys() if policy is not None and editable(policy, name))
    selected_file_bytes(workspace, (*files, *(WorkspaceFile(name, text="") for name in deleted)))
    original, directories = {}, set()
    for name in (*selected, *deleted):
        target = workspace / name
        if target.exists() and target.stat().st_nlink != 1:
            raise ValueError("source publication must not replace a hard-linked file")
        original[name] = (target.read_bytes(), stat.S_IMODE(target.stat().st_mode)) if target.exists() else None
        parent = target.parent
        while parent != workspace and not parent.exists():
            directories.add(parent)
            parent = parent.parent
    try:
        for name in deleted:
            (workspace / name).unlink()
        _install_files(workspace, files)
    except BaseException:
        _restore_files(workspace, original, directories)
        raise


def _candidate(handle, candidate_commit, *, require_worktree_match):
    from .binding import inspect_candidate
    from .context import context_changes, uses_context
    from .legacy import EpisodeWorktree
    from .policy import TIMELINE_PROBE_MARKERS
    facts = inspect_candidate(handle.path, handle)
    if not facts.registered or facts.branch != handle.branch:
        return "candidate has no matching registered isolated branch", []
    try:
        revision = resolve_revision(handle.path, candidate_commit)
        base = resolve_revision(handle.path, handle.base_commit)
    except (ValueError, RuntimeError):
        return "candidate_commit does not resolve", []
    if revision != facts.revision:
        return "candidate_commit must equal episode HEAD", []
    if _git(handle.path, "merge-base", "--is-ancestor", base, revision, check=False).returncode:
        return "candidate_commit is not descended from incumbent", []
    if uses_context(handle.path):
        changed = context_changes(handle.path, base)
        if changed:
            return "candidate modified frozen workspace context: " + ", ".join(changed), []
    policy, baseline_raw = _policy(handle.path, base)
    if policy is None:
        if require_worktree_match:
            return EpisodeWorktree(handle.episode, handle.base_commit, handle.branch,
                                   handle.path).validate_candidate(revision)
        paths = [name for name in _git(handle.path, "diff", "--name-only", "--no-renames",
                                      "-z", base, revision).stdout.decode().split("\0") if name]
        from .policy import CANDIDATE_PATHS
        if not paths or "kernel.py" not in paths or not set(paths).issubset(CANDIDATE_PATHS):
            return "candidate commit must change kernel.py and may also update solution.json", paths
        original = _git(handle.path, "show", f"{base}:solution.json", check=False)
        candidate = _git(handle.path, "show", f"{revision}:solution.json", check=False)
        if original.returncode == 0 and candidate.returncode:
            return "candidate must preserve the incumbent solution.json manifest", paths
    else:
        policy_after, candidate_raw = _policy(handle.path, revision)
        if policy_after != policy or candidate_raw != baseline_raw:
            return "candidate changed its source declaration", []
        paths = [name for name in _git(handle.path, "diff", "--name-only", "--no-renames",
                                      "-z", base, revision).stdout.decode().split("\0") if name]
        if any(not editable(policy, name) or name in policy["immutable_files"] for name in paths):
            return "candidate changed a fixed or out-of-scope source file", paths
    try:
        committed = read_source(handle.path, revision=revision)
        if require_worktree_match and read_source(handle.path).files != committed.files:
            return "working source must match the sealed candidate", paths
        if any(marker.encode() in data for data in committed.files.values() if data is not None
               for marker in TIMELINE_PROBE_MARKERS):
            return "candidate source contains timeline profiling probes", paths
    except (ValueError, OSError, RuntimeError) as error:
        return str(error), paths
    return "", paths


def validate_candidate(handle, candidate_commit):
    return _candidate(handle, candidate_commit, require_worktree_match=True)


def restore_sealed_candidate(handle, candidate_commit, source):
    violation, paths = _candidate(handle, candidate_commit, require_worktree_match=False)
    if violation:
        return violation, paths
    committed = read_source(handle.path, revision=candidate_commit)
    if committed.files != source.files:
        return "sealed source does not match candidate_commit", paths
    policy, _ = _policy(handle.path, candidate_commit)
    if policy is None:
        deleted = ()
    else:
        working = _declared_files(handle.path, policy, validate=False)
        deleted = tuple(name for name in working.keys() - source.files.keys() if editable(policy, name))
    from .records import record
    with tempfile.TemporaryDirectory(prefix="aka-sealed-source-") as temporary:
        files = []
        for number, (name, data) in enumerate(source.files.items()):
            if data is None:
                continue
            retained = Path(temporary) / str(number)
            retained.write_bytes(data)
            listing = _git(handle.path, "ls-tree", "-z", candidate_commit, "--", name).stdout
            mode = listing.split(b" ", 1)[0]
            retained.chmod(0o755 if mode == b"100755" else 0o644)
            files.append(WorkspaceFile(name, source=retained))
        record(handle.path, expected_revision=candidate_commit,
               files=tuple(files), deleted_paths=deleted, message="Restore sealed candidate source")
    return "", paths


def _candidate_source(handle, source):
    """Read-only preflight of exact measured bytes against an Episode binding."""
    from .binding import inspect_candidate
    from .context import context_changes, uses_context
    from .history import working_facts
    from .policy import CANDIDATE_PATHS, TIMELINE_PROBE_MARKERS
    from aka.task.git import protected_violation
    from .records import record
    facts = inspect_candidate(handle.path, handle)
    if not facts.registered or facts.branch != handle.branch or not facts.revision:
        raise ValueError("candidate has no matching registered isolated branch")
    head, base = facts.revision, resolve_revision(handle.path, handle.base_commit)
    if _git(handle.path, "merge-base", "--is-ancestor", base, head, check=False).returncode:
        raise ValueError("candidate is not descended from its Episode baseline")
    if uses_context(handle.path):
        changed = context_changes(handle.path, base)
        if changed:
            raise ValueError("candidate modified frozen workspace context: " + ", ".join(changed))
    violation = protected_violation(list(working_facts(handle.path).changed_paths))
    if violation:
        raise ValueError(violation)
    selected = dict(source.files)
    if selected.get("kernel.py") is None:
        raise ValueError("candidate source requires kernel.py")
    if any(marker.encode() in data for data in selected.values() if data is not None
           for marker in TIMELINE_PROBE_MARKERS):
        raise ValueError("candidate source contains timeline profiling probes")
    baseline = read_source(handle.path, revision=base)
    previous = read_source(handle.path, revision=head)
    policy, raw = _policy(handle.path, base)
    current_policy, current_raw = _policy(handle.path, head)
    changed = tuple(name for name in _git(handle.path, "diff", "--name-only", "--no-renames", "-z",
                                        base, head).stdout.decode().split("\0") if name)
    if policy is None:
        if set(selected) != {"kernel.py"}:
            raise ValueError("a single-file candidate may seal only kernel.py")
        if selected == dict(baseline.files):
            raise ValueError("candidate has no Kernel change; report pivot or select another")
        if not set(changed).issubset(CANDIDATE_PATHS):
            raise ValueError("candidate changed source outside its existing edit scope")
        deleted = ()
    else:
        if current_policy != policy or current_raw != raw or selected.get(POLICY) != raw:
            raise ValueError("candidate source declaration differs from its Episode baseline")
        validate_files({name: data for name, data in selected.items() if name != POLICY and data is not None}, policy)
        if any(not editable(policy, name) or name in policy["immutable_files"] for name in changed):
            raise ValueError("candidate changed fixed or out-of-scope source")
        # Delete only previously declared editable sources. Scratch and discovery
        # state are unrelated to the measured source and retain their staging.
        deleted = tuple(name for name in previous.files.keys() - selected.keys()
                        if editable(policy, name) and name != POLICY)
    return head, selected, deleted


def validate_candidate_source(handle, source):
    """Validate a frozen source before backend work or Git publication."""
    _candidate_source(handle, source)


def seal_candidate(handle, source):
    """Commit an exact measured contribution after all candidate preflight checks."""
    from .records import record
    head, selected, deleted = _candidate_source(handle, source)
    with tempfile.TemporaryDirectory(prefix="aka-seal-candidate-") as temporary:
        files = []
        for number, (name, data) in enumerate(selected.items()):
            if data is None:
                raise ValueError("sealed source cannot contain absent members")
            retained = Path(temporary) / str(number)
            retained.write_bytes(data)
            listing = _git(handle.path, "ls-tree", "-z", head, "--", name).stdout
            retained.chmod(0o755 if listing.startswith(b"100755 ") else 0o644)
            files.append(WorkspaceFile(name, source=retained))
        revision = record(handle.path, expected_revision=head, files=tuple(files),
                          deleted_paths=deleted, message=f"Episode {handle.episode}: selected Kernel")
    violation, _ = validate_candidate(handle, revision)
    if violation:
        raise RuntimeError(violation)
    return revision
