"""Read committed candidate content without staging experiment files."""
from pathlib import Path, PurePosixPath
from aka.contracts.workspace import RevisionContent
from aka.legacy.task.git import _git


class GitSnapshots:
    def read_revision(self, workspace: Path, revision: str,
                      changed_paths: list[str]) -> RevisionContent:
        # Invalid revisions must not masquerade as deleted paths.
        commit = _git(workspace, "rev-parse", "--verify", f"{revision}^{{commit}}").stdout.strip()
        files = {}
        for relative in changed_paths:
            path = PurePosixPath(relative)
            if not relative or path.is_absolute() or ".." in path.parts or path == PurePosixPath("."):
                raise ValueError(f"unsafe candidate path: {relative}")
            result = _git(workspace, "show", f"{commit}:{relative}", check=False, binary=True)
            files[relative] = result.stdout if result.returncode == 0 else None
        return RevisionContent(files)
