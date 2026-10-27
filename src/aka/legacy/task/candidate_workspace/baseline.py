"""Check and restore original baseline files without deciding task acceptance."""
from pathlib import Path, PurePosixPath
import hashlib
import subprocess
from typing import Iterator

from aka.legacy.task.git import git_path_blob, git_worktree_blob
from .policy import IMMUTABLE_BASELINE_PATHS
from .context import MANIFEST, context_changes, restore_context_projection, uses_context


def _changed_baseline_paths(workspace: Path, baseline_commit: str) -> Iterator[str]:
    for path in IMMUTABLE_BASELINE_PATHS:
        original = git_path_blob(workspace, baseline_commit, path)
        if original and original != git_worktree_blob(workspace, path):
            yield path


def _required_path_names(paths):
    for name in paths:
        relative = PurePosixPath(name)
        if (not relative.parts or relative.is_absolute() or ".." in relative.parts
                or relative.as_posix() != name or "\\" in name
                or any(ord(char) < 32 for char in name)
                or any(part.casefold() == ".git" for part in relative.parts)):
            raise ValueError(f"unsafe required baseline path: {name}")
    return tuple(dict.fromkeys(paths))


def _required_path_changes(workspace, revision, paths):
    from .context import _decode, _directory_mode, _projection_mode, _revision_manifest, load_context

    resolved = subprocess.run(["git", "rev-parse", "--verify", f"{revision}^{{commit}}"],
                              cwd=workspace, capture_output=True)
    if resolved.returncode:
        raise ValueError("required baseline revision does not resolve")
    revision = resolved.stdout.decode().strip()
    context = {}
    directories = {}
    if uses_context(workspace):
        load_context(workspace)  # A corrupt current management identity is an error.
        raw = _revision_manifest(workspace, revision)
        if raw is None:
            raise ValueError("required baseline is missing its context management identity")
        original = _decode(raw)
        context = original["context"]
        for snapshot in original.get("namespaces", {}).values():
            if not snapshot["directories"]:
                raise ValueError("required baseline context namespace is not sealed")
            directories.update(snapshot["directories"])
    changed = []
    for name in paths:
        try:
            if name in context:
                expected = context[name]
                if (_projection_mode(workspace, name) != expected["mode"]
                        or hashlib.sha256((workspace / name).read_bytes()).hexdigest() != expected["sha256"]):
                    changed.append(name)
            elif name in directories:
                if _directory_mode(workspace / name) != directories[name]:
                    changed.append(name)
            else:
                result = subprocess.run(["git", "ls-tree", "-z", revision, "--", name],
                                        cwd=workspace, capture_output=True, check=True)
                rows = [row for row in result.stdout.split(b"\0") if row]
                row = next((row for row in rows if row.split(b"\t", 1)[1].decode() == name), None)
                if row is None:
                    changed.append(name)
                    continue
                mode, kind, identity = row.split(b"\t", 1)[0].split()
                if kind != b"blob" or mode not in {b"100644", b"100755"}:
                    changed.append(name)
                    continue
                working_mode = _projection_mode(workspace, name)
                # Git source identity preserves the executable bit; context keeps full modes.
                if (bool(working_mode & 0o111) != (mode == b"100755")
                        or git_worktree_blob(workspace, name) != identity.decode()):
                    changed.append(name)
        except (OSError, ValueError):
            changed.append(name)
    return changed


def baseline_changes(workspace: Path, baseline_commit: str, *, required_paths=()) -> list[str]:
    """List baseline-owned files whose on-disk blobs differ, in policy order.

    Tracked legacy workspaces keep the original content-only, absent-path behavior.
    New workspaces validate frozen context identity, bytes, modes and membership.
    """
    workspace = Path(workspace)
    required = _required_path_names(tuple(required_paths))
    required_changed = _required_path_changes(workspace, baseline_commit, required) if required else []
    if uses_context(workspace):
        changed = context_changes(workspace, baseline_commit)
        # Agent-authored tracked baseline assets retain their original policy.
        changed.extend(_changed_baseline_paths(workspace, baseline_commit))
        return list(dict.fromkeys([*changed, *required_changed]))
    return list(dict.fromkeys([*_changed_baseline_paths(workspace, baseline_commit), *required_changed]))


def restore_baseline(workspace: Path, baseline_commit: str) -> list[str]:
    """Explicitly restore changed baseline files and return successful paths.

    Legacy workspaces preserve individual Git checkout behavior. New workspaces
    restore only retained projections and management identity, plus V0 memory.
    """
    restored = []
    if uses_context(workspace):
        changed = context_changes(workspace, baseline_commit)
        restored = restore_context_projection(workspace, baseline_commit)
        if MANIFEST in changed:
            subprocess.run(["git", "checkout", baseline_commit, "--", MANIFEST],
                           cwd=workspace, check=True, capture_output=True)
            restored.insert(0, MANIFEST)
        for name in _changed_baseline_paths(workspace, baseline_commit):
            checkout = subprocess.run(["git", "checkout", baseline_commit, "--", name],
                                      cwd=workspace, check=False, capture_output=True)
            if checkout.returncode == 0:
                restored.append(name)
        return list(dict.fromkeys(restored))
    for path in _changed_baseline_paths(workspace, baseline_commit):
        checkout = subprocess.run(
            ["git", "checkout", baseline_commit, "--", path],
            cwd=str(workspace),
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if checkout.returncode == 0:
            restored.append(path)
    return restored
