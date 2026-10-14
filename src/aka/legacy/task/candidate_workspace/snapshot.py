"""Committed-file snapshot staging independent of evaluation or execution."""
from pathlib import Path, PurePosixPath
from aka.legacy.task.git import _git

def _git_blob(workspace: Path, revision: str, relative: str) -> bytes | None:
    result = _git(workspace, "show", f"{revision}:{relative}", check=False, binary=True)
    return result.stdout if result.returncode == 0 else None


def _snapshot_revision(
    workspace: Path,
    revision: str,
    changed_paths: list[str],
    destination: Path,
    label: str,
) -> dict[str, str | None]:
    manifest: dict[str, str | None] = {}
    for index, relative in enumerate(changed_paths):
        path = PurePosixPath(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"unsafe candidate path: {relative}")
        content = _git_blob(workspace, revision, relative)
        if content is None:
            manifest[relative] = None
            continue
        snapshot_relative = f"snapshots/{label}/{index:04d}.bin"
        target = destination / snapshot_relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        manifest[relative] = snapshot_relative
    return manifest


class GitSnapshots:
    def write_revision(self, workspace, revision, changed_paths, destination, label):
        return _snapshot_revision(workspace, revision, changed_paths, destination, label)
