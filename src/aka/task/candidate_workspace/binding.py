"""Read actual candidate registration and Git identity without accepting source."""
from pathlib import Path
import os
import stat
import subprocess

from aka.contracts.workspace import CandidateBindingFacts


def _git(workspace, *arguments, check=True):
    environment = os.environ.copy()
    for name in ("GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR", "GIT_INDEX_FILE",
                 "GIT_OBJECT_DIRECTORY", "GIT_ALTERNATE_OBJECT_DIRECTORIES"):
        environment.pop(name, None)
    result = subprocess.run(["git", "--no-optional-locks", *arguments],
                            cwd=workspace, env=environment, capture_output=True)
    if check and result.returncode:
        raise RuntimeError("cannot inspect candidate Git binding: "
                           + result.stderr.decode(errors="replace").strip())
    return result


def _directory(workspace, argument):
    value = _git(workspace, "rev-parse", argument).stdout.decode().strip()
    if not value:
        raise ValueError("candidate Git directory identity is missing")
    path = Path(value)
    if not path.is_absolute():
        path = workspace / path
    path = path.resolve(strict=True)
    if not path.is_dir():
        raise ValueError("candidate Git directory is not a directory")
    return path


def _registration(workspace, path):
    rows = []
    output = _git(workspace, "worktree", "list", "--porcelain", "-z").stdout
    for block in output.split(b"\0\0"):
        if not block:
            continue
        row = {}
        for entry in block.split(b"\0"):
            if not entry:
                continue
            name, _, value = entry.partition(b" ")
            key = name.decode()
            if key in row:
                raise ValueError("duplicate candidate Git registration field")
            row[key] = os.fsdecode(value)
        if "worktree" not in row:
            raise ValueError("candidate Git registration has no worktree path")
        if Path(row["worktree"]).resolve() == path:
            rows.append(row)
    if len(rows) > 1:
        raise ValueError("candidate has duplicate Git registration")
    return rows[0] if rows else None


def _ordinary_file(path):
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError("candidate Git association must be a regular file")
    return path.read_text().strip()


def inspect_candidate(workspace, handle):
    """Absent registration is distinct from damaged or unreadable registration."""
    workspace = Path(workspace).resolve(strict=True)
    path = Path(handle.path).resolve()
    common = _directory(workspace, "--git-common-dir")
    row = _registration(workspace, path)
    if row is None:
        # A damaged backpointer can make Git omit/relocate a real registration.
        # The candidate's own association still identifies its owned metadata.
        association = path / ".git"
        if association.is_file() and not association.is_symlink():
            declared = _ordinary_file(association)
            if declared.startswith("gitdir: "):
                declared_path = Path(declared[len("gitdir: "):])
                if not declared_path.is_absolute():
                    declared_path = path / declared_path
                declared_path = declared_path.resolve()
                if declared_path.parent == common / "worktrees" and declared_path.exists():
                    raise ValueError("candidate owned Git metadata has no matching registration")
        return CandidateBindingFacts(False, None, None)
    if not path.is_dir():
        raise ValueError("registered candidate worktree is missing")
    association = path / ".git"
    if association.is_symlink():
        raise ValueError("registered candidate Git association is a symlink")
    actual_common = _directory(path, "--git-common-dir")
    directory = _directory(path, "--absolute-git-dir")
    if actual_common != common:
        raise ValueError("registered candidate belongs to another common Git directory")
    if association.is_file():
        declared = _ordinary_file(association)
        if not declared.startswith("gitdir: "):
            raise ValueError("registered candidate Git association is invalid")
        declared_path = Path(declared[len("gitdir: "):])
        if not declared_path.is_absolute():
            declared_path = path / declared_path
        if declared_path.resolve(strict=True) != directory or directory.parent != common / "worktrees":
            raise ValueError("registered candidate Git metadata association differs")
        backpointer = Path(_ordinary_file(directory / "gitdir"))
        if backpointer.resolve() != association:
            raise ValueError("registered candidate Git metadata points to another worktree")
        _ordinary_file(directory / "HEAD")
        _ordinary_file(directory / "commondir")
    elif not association.is_dir() or directory != common:
        raise ValueError("registered candidate Git association is missing")
    symbolic = _git(path, "symbolic-ref", "--quiet", "--short", "HEAD", check=False)
    if symbolic.returncode == 0:
        branch = symbolic.stdout.decode().strip()
        if not branch:
            raise ValueError("registered candidate has an empty branch identity")
    elif symbolic.returncode == 1 and not symbolic.stderr:
        branch = None
    else:
        raise RuntimeError("cannot read registered candidate branch: "
                           + symbolic.stderr.decode(errors="replace").strip())
    revision = _git(path, "rev-parse", "--verify", "HEAD^{commit}").stdout.decode().strip()
    registered_branch = row.get("branch")
    if registered_branch is not None:
        if not registered_branch.startswith("refs/heads/"):
            raise ValueError("registered candidate branch identity is invalid")
        registered_branch = registered_branch[len("refs/heads/"):]
    if row.get("HEAD") != revision or registered_branch != branch:
        raise ValueError("registered candidate Git facts changed during inspection")
    if _registration(workspace, path) != row:
        raise ValueError("candidate Git registration changed during inspection")
    return CandidateBindingFacts(True, branch, revision)
