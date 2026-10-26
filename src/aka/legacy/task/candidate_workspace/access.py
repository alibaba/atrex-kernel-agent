"""Own the filesystem and Git scope of a selected candidate."""
from pathlib import Path
from aka.contracts.filesystem import FilesystemAccess, FilesystemMount
from .policy import IMMUTABLE_BASELINE_PATHS


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
    return tuple(FilesystemMount(path, path, writable) for path, writable in selected)


def filesystem_access(workspace, *, include_git=False):
    workspace = Path(workspace).resolve(strict=True)
    protected = tuple(workspace / name for name in IMMUTABLE_BASELINE_PATHS
                      if (workspace / name).exists())
    return FilesystemAccess(_git_mounts(workspace) if include_git else (), (), protected)
