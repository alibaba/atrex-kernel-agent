"""Read mechanical Git revision and working facts without interpreting records."""
from pathlib import Path, PurePosixPath
import re

from aka.contracts.content import WorkspaceFile
from aka.contracts.workspace import RevisionFacts, WorkingFacts
from .records import _git, _selected, current_revision


def validate_paths(paths):
    paths = tuple(paths)
    for name in paths:
        path = PurePosixPath(name)
        if (not name or path.is_absolute() or path.as_posix() != name or ".." in path.parts
                or "\\" in name or any(ord(char) < 32 for char in name)
                or any(part.casefold() == ".git" for part in path.parts)):
            raise ValueError(f"unsafe workspace fact path: {name}")
    if len(set(paths)) != len(paths):
        raise ValueError("duplicate workspace fact paths")
    return paths


def resolve_revision(workspace, revision):
    # Git identities are object IDs (including historical unique abbreviations),
    # never user-provided Git expressions, refs or command options.
    current_revision(workspace)
    if not (Path(workspace) / ".git").exists():
        raise ValueError("revision facts require an initialized Workspace root")
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-fA-F]{4,64}", revision) is None:
        raise ValueError("unknown opaque workspace revision identity")
    return _git(workspace, "rev-parse", "--verify", "--end-of-options",
                revision + "^{commit}").stdout.decode().strip()


def content_identities(workspace, revision, paths):
    identities = dict.fromkeys(paths)
    if not paths:
        return identities
    result = _git(workspace, "--literal-pathspecs", "ls-tree", "-z", revision, "--", *paths)
    for entry in result.stdout.split(b"\0"):
        if not entry:
            continue
        metadata, encoded_path = entry.split(b"\t", 1)
        mode, kind, identity = metadata.decode().split()
        name = encoded_path.decode("utf-8", errors="surrogateescape")
        if name not in identities:
            continue
        if kind != "blob" or mode == "120000":
            raise ValueError(f"revision content identity is not a regular file: {name}")
        identities[name] = identity
    return identities


def revision_history(workspace: Path, *, revision=None, paths=(), changed_path=None):
    workspace = Path(workspace).absolute()
    paths = validate_paths(paths)
    if changed_path is not None:
        validate_paths((changed_path,))
    if revision is None:
        revision = current_revision(workspace)
        if not revision:
            return ()
    resolved = resolve_revision(workspace, revision)
    arguments = ("--", changed_path) if changed_path is not None else ()
    commits = _git(workspace, "--literal-pathspecs", "rev-list", resolved,
                   *arguments).stdout.decode().splitlines()
    facts = []
    for commit in commits:
        result = _git(workspace, "show", "-s", "--format=%H%x00%P%x00%ct%x00%B", commit)
        identity, parents, committed_at, message = result.stdout.split(b"\0", 3)
        facts.append(RevisionFacts(identity.decode(), tuple(parents.decode().split()),
            message.decode("utf-8", errors="replace").rstrip("\n"), int(committed_at),
            content_identities(workspace, commit, paths)))
    return tuple(facts)


def working_facts(workspace: Path, *, paths=()):
    workspace = Path(workspace).absolute()
    current_revision(workspace)
    if not (workspace / ".git").exists():
        raise ValueError("working facts require an initialized Workspace root")
    paths = validate_paths(paths)
    _selected(workspace, tuple(WorkspaceFile(name, text="") for name in paths))
    output = _git(workspace, "--no-optional-locks", "status", "--porcelain=v1", "-z",
                  "--untracked-files=all").stdout.split(b"\0")
    changed = []
    position = 0
    while position < len(output):
        row = output[position]
        position += 1
        if not row:
            continue
        status = row[:2]
        name = row[3:].decode("utf-8", errors="surrogateescape")
        target = workspace / name
        if status != b"??" or (target.is_file() and not target.is_symlink()):
            changed.append(name)
        if b"R" in status or b"C" in status:
            changed.append(output[position].decode("utf-8", errors="surrogateescape"))
            position += 1
    identities = {}
    for name in paths:
        if not (workspace / name).exists():
            identities[name] = None
        else:
            identities[name] = _git(workspace, "hash-object", "--", name).stdout.decode().strip()
    return WorkingFacts(tuple(dict.fromkeys(changed)), identities)
