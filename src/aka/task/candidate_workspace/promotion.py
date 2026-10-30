"""Promote validated candidate bytes and opaque owner evidence in one transaction."""
from pathlib import Path
import tempfile

from aka.contracts.content import WorkspaceFile
from .content import selected_file_bytes
from .context import MANIFEST, context_changes, load_context, uses_context
from .history import resolve_revision, working_facts
from .policy import CANDIDATE_PATHS
from .records import _git, current_revision, record
from .source import POLICY, _policy, read_source
from aka.task.source.tree import editable


def promote_candidate(workspace, *, base_commit, candidate_commit, episode,
                      record_files=(), commit_message=None):
    workspace = Path(workspace).absolute()
    if current_revision(workspace) != base_commit:
        raise RuntimeError("incumbent advanced during episode; refusing promotion")
    candidate_commit = resolve_revision(workspace, candidate_commit)
    if _git(workspace, "merge-base", "--is-ancestor", base_commit, candidate_commit, check=False).returncode:
        raise ValueError("promotion candidate is not descended from its incumbent")
    if uses_context(workspace):
        changed = context_changes(workspace, base_commit)
        if changed:
            raise ValueError("incumbent frozen context changed: " + ", ".join(changed))
    paths = tuple(name for name in _git(workspace, "diff", "--name-only", "--no-renames", "-z",
                                       base_commit, candidate_commit).stdout.decode().split("\0") if name)
    policy, raw = _policy(workspace, base_commit)
    if policy is None:
        if "kernel.py" not in paths or not set(paths).issubset(CANDIDATE_PATHS):
            raise ValueError("promotion requires kernel.py with only an optional solution.json update")
        previous = _git(workspace, "show", f"{base_commit}:solution.json", check=False)
        updated = _git(workspace, "show", f"{candidate_commit}:solution.json", check=False)
        if previous.returncode == 0 and updated.returncode:
            raise ValueError("candidate must preserve the incumbent solution.json manifest")
        reserved = tuple(CANDIDATE_PATHS)
    else:
        candidate_policy, candidate_raw = _policy(workspace, candidate_commit)
        if candidate_policy != policy or candidate_raw != raw:
            raise ValueError("promotion candidate changed its frozen source declaration")
        if not paths or any(not editable(policy, name) or name in policy["immutable_files"] for name in paths):
            raise ValueError("promotion may change only declared editable source")
        reserved = (POLICY, *policy["immutable_files"], *policy["editable_roots"])
    source = read_source(workspace, revision=candidate_commit)
    if source.files.get("kernel.py") is None:
        raise ValueError("promotion candidate has no kernel.py")
    if commit_message is not None and (not isinstance(commit_message, str) or not commit_message.strip()):
        raise ValueError("promotion commit_message must be a nonempty string")
    if uses_context(workspace):
        value = load_context(workspace)
        reserved = (*reserved, MANIFEST, *value["context"], *value.get("namespaces", {}))
    selected_file_bytes(workspace, record_files, reserved=reserved)
    if set(paths).intersection(working_facts(workspace).changed_paths):
        raise ValueError("incumbent candidate source has uncommitted changes")
    with tempfile.TemporaryDirectory(prefix="aka-promote-source-") as temporary:
        files, deleted = list(record_files), []
        for number, name in enumerate(paths):
            listing = _git(workspace, "ls-tree", "-z", candidate_commit, "--", name).stdout
            if not listing:
                deleted.append(name)
                continue
            metadata, encoded = listing.split(b"\t", 1)
            mode, kind, identity = metadata.split()
            if kind != b"blob" or mode not in {b"100644", b"100755"}:
                raise ValueError("promotion candidate source is not an ordinary file")
            retained = Path(temporary) / str(number)
            retained.write_bytes(_git(workspace, "cat-file", "blob", identity.decode()).stdout)
            retained.chmod(0o755 if mode == b"100755" else 0o644)
            files.append(WorkspaceFile(name, source=retained))
        return record(workspace, expected_revision=base_commit, files=tuple(files),
                      deleted_paths=tuple(deleted),
                      message=(commit_message if commit_message is not None else
                               f"episode {episode}: promote verified long-horizon candidate"))
