"""Opaque snapshots and transactions for explicitly selected mutable file trees."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import tempfile

from aka.contracts.workspace import WorkspaceFilesSnapshot
from .history import validate_paths

FORMAT = "aka.workspace-files.v1"
MAX_FILE = 16 * 1024 * 1024
MAX_TOTAL = 64 * 1024 * 1024
MAX_ENTRIES = 4096
MAX_DEPTH = 32
IGNORED = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _within(name, root):
    return name == root or name.startswith(root + "/")


def _absolute(path):
    path = Path(path).absolute()
    if path.resolve() != path or path.is_symlink():
        raise ValueError("Workspace file paths must not traverse links or aliases")
    for parent in path.parents:
        if parent.is_symlink() or (parent.exists() and not parent.is_dir()):
            raise ValueError("Workspace file parent must be an ordinary directory")
    return path


@contextmanager
def _directory(path, *, create=False):
    """Pin every component; all subsequent operations use fixed directory FDs."""
    path = _absolute(path)
    descriptor = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in path.parts[1:]:
            if create:
                try:
                    os.mkdir(part, 0o700, dir_fd=descriptor)
                except FileExistsError:
                    pass
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        yield descriptor
    finally:
        os.close(descriptor)


def _read_descriptor(descriptor, limit):
    before = os.fstat(descriptor)
    if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
            or before.st_mode & (stat.S_ISUID | stat.S_ISGID | stat.S_ISVTX)
            or before.st_size > limit):
        raise ValueError("Workspace content must be an ordinary, unlinked, bounded file")
    with os.fdopen(os.dup(descriptor), "rb") as stream:
        data = stream.read(limit + 1)
    after = os.fstat(descriptor)
    identity = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)
    if len(data) > limit or identity(before) != identity(after) or len(data) != after.st_size:
        raise ValueError("Workspace file changed or grew while being read")
    return data, stat.S_IMODE(before.st_mode)


def read_regular_bytes(path, *, limit):
    path = _absolute(path)
    with _directory(path.parent) as parent:
        descriptor = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        try:
            return _read_descriptor(descriptor, limit)[0]
        finally:
            os.close(descriptor)


def _restrictions(workspace):
    from .policy import PROTECTED_PATHS, CANDIDATE_PATHS
    protected = {*PROTECTED_PATHS, *CANDIDATE_PATHS, ".git", "memory", "eval", "input",
                 ".claude", ".codex", ".qoder", ".agents", ".pi", ".atrex_source.json"}
    from .context import load_context, uses_context
    if uses_context(workspace):
        value = load_context(workspace)
        protected.update(value["source_paths"])
        protected.update(value["context"])
        protected.update(value.get("namespaces", {}))
    from aka.task.source.formats import policy_at
    policy = policy_at(workspace)
    if policy is not None:
        protected.update(policy["editable_roots"])
        protected.update(policy["immutable_files"])
        if policy.get("kind") == "agate_cuda":
            from aka.task.source.formats.registry import for_policy
            protected.update(for_policy(policy).evidence_paths(workspace, policy))
    return protected


def _roots(workspace, directories):
    directories = validate_paths(tuple(directories))
    protected = _restrictions(workspace)
    for position, root in enumerate(directories):
        if PurePosixPath(root).parts[0].startswith(".atrex_"):
            raise ValueError("Workspace files cannot include controller paths")
        if any(_within(root, name) or _within(name, root) for name in (*protected, *directories[:position])):
            raise ValueError(f"Workspace file scope overlaps source, immutable inputs or control: {root}")
    return directories


def _excludes(paths):
    return validate_paths(tuple(dict.fromkeys(paths)))


def _scan(workspace, roots, exclude_paths):
    entries, contents = {}, {}
    total = 0
    excluded = lambda name: any(_within(name, root) for root in exclude_paths)
    def inspect(parent, leaf, name, depth):
        nonlocal total
        if excluded(name) or any(part in IGNORED for part in PurePosixPath(name).parts) or name.endswith((".pyc", ".pyo")):
            return
        if depth > MAX_DEPTH or len(entries) >= MAX_ENTRIES:
            raise ValueError("Workspace files exceed directory depth or entry limits")
        try:
            info = os.stat(leaf, dir_fd=parent, follow_symlinks=False)
        except FileNotFoundError:
            return
        mode = stat.S_IMODE(info.st_mode)
        if info.st_mode & (stat.S_ISUID | stat.S_ISGID | stat.S_ISVTX):
            raise ValueError(f"Workspace files have unsupported modes: {name}")
        if stat.S_ISDIR(info.st_mode):
            descriptor = os.open(leaf, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
            try:
                entries[name] = {"kind": "directory", "mode": mode}
                with os.scandir(descriptor) as children:
                    for child in sorted(children, key=lambda value: value.name):
                        validate_paths((name + "/" + child.name,))
                        inspect(descriptor, child.name, name + "/" + child.name, depth + 1)
            finally:
                os.close(descriptor)
        elif stat.S_ISREG(info.st_mode):
            descriptor = os.open(leaf, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
            try:
                payload, mode = _read_descriptor(descriptor, min(MAX_FILE, MAX_TOTAL - total))
            finally:
                os.close(descriptor)
            total += len(payload)
            entries[name] = {"kind": "file", "mode": mode, "sha256": hashlib.sha256(payload).hexdigest()}
            contents[name] = payload
        else:
            raise ValueError(f"Workspace files contain a link or special object: {name}")
    for root in roots:
        path = workspace / root
        try:
            with _directory(path.parent) as parent:
                inspect(parent, path.name, root, 0)
        except FileNotFoundError:
            pass
    return entries, contents


def _write_tree(directory, entries, contents, *, writable_directories=False):
    directory.mkdir(parents=True, exist_ok=True)
    for name, row in sorted(entries.items(), key=lambda item: (len(PurePosixPath(item[0]).parts), item[0])):
        target = directory / name
        if row["kind"] == "directory":
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(contents[name])
    for name, row in sorted(entries.items(), key=lambda item: len(PurePosixPath(item[0]).parts), reverse=True):
        (directory / name).chmod(row["mode"] | (0o700 if writable_directories and row["kind"] == "directory" else 0))


def _validate(snapshot):
    root = _absolute(snapshot.directory)
    raw = read_regular_bytes(root / "manifest.json", limit=2 * 1024 * 1024)
    manifest = json.loads(raw)
    if (not isinstance(manifest, dict) or set(manifest) != {"format", "roots", "excludes", "entries"}
            or manifest["format"] != FORMAT or not isinstance(manifest["entries"], dict)
            or not isinstance(manifest["roots"], list) or not isinstance(manifest["excludes"], list)
            or len(manifest["entries"]) > MAX_ENTRIES
            or hashlib.sha256(_canonical(manifest)).hexdigest() != snapshot.identity):
        raise ValueError("invalid Workspace files snapshot identity")
    roots = validate_paths(tuple(manifest["roots"]))
    excludes = _excludes(manifest["excludes"])
    contents, total = {}, 0
    for name, row in manifest["entries"].items():
        validate_paths((name,))
        if not isinstance(row, dict):
            raise ValueError("invalid Workspace snapshot member")
        expected = {"kind", "mode", "sha256"} if row.get("kind") == "file" else {"kind", "mode"}
        if (set(row) != expected or row["kind"] not in {"file", "directory"}
                or type(row["mode"]) is not int or not 0 <= row["mode"] <= 0o777
                or not any(_within(name, member) for member in roots)
                or any(_within(name, member) for member in excludes)):
            raise ValueError("invalid Workspace snapshot member")
        if row["kind"] == "file":
            data = read_regular_bytes(root / "content" / name, limit=min(MAX_FILE, MAX_TOTAL - total))
            total += len(data)
            if hashlib.sha256(data).hexdigest() != row["sha256"]:
                raise ValueError("Workspace snapshot content changed")
            contents[name] = data
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    expected = {"manifest.json", *("content/" + name for name in contents)}
    expected_directories = {"content", *("content/" + name for name, row in manifest["entries"].items()
                                         if row["kind"] == "directory")}
    for name in contents:
        expected_directories.update((PurePosixPath("content") / name).parent.parents)
        expected_directories.add((PurePosixPath("content") / name).parent)
    expected_directories = {str(name) for name in expected_directories if str(name) != "."}
    actual_directories = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_dir()}
    if (actual != expected or actual_directories != expected_directories
            or any(path.is_symlink() for path in root.rglob("*"))
            or any(manifest["entries"].get(name, {}).get("kind", "directory") != "directory" for name in roots)):
        raise ValueError("Workspace snapshot has unexpected members")
    return manifest, contents


def snapshot_files(workspace, destination, *, directories, exclude_paths=()):
    workspace, destination = _absolute(workspace), _absolute(destination)
    if destination == workspace or destination.is_relative_to(workspace) or workspace.is_relative_to(destination):
        raise ValueError("Workspace snapshot storage overlaps the workspace")
    roots = _roots(workspace, directories)
    excludes = _excludes(exclude_paths)
    entries, contents = _scan(workspace, roots, excludes)
    if any(entries.get(name, {}).get("kind", "directory") != "directory" for name in roots):
        raise ValueError("Workspace snapshot roots must be directories")
    manifest = {"format": FORMAT, "roots": list(roots), "excludes": list(excludes), "entries": entries}
    identity = hashlib.sha256(_canonical(manifest)).hexdigest()
    snapshot = WorkspaceFilesSnapshot(destination, identity)
    if destination.exists():
        _validate(snapshot)
        return snapshot
    with _directory(destination.parent, create=True):
        pass
    with tempfile.TemporaryDirectory(prefix=".aka-files-", dir=destination.parent) as temporary:
        staged = Path(temporary) / "snapshot"
        staged.mkdir(mode=0o700)
        _write_tree(staged / "content", entries, contents)
        # Storage permissions are private; original file/empty-directory modes
        # remain in the sealed manifest and are applied only to destinations.
        for path in sorted((staged / "content").rglob("*"), key=lambda value: len(value.parts), reverse=True):
            path.chmod(0o700 if path.is_dir() else 0o400)
        (staged / "content").chmod(0o700)
        (staged / "manifest.json").write_bytes(_canonical(manifest))
        (staged / "manifest.json").chmod(0o400)
        _validate(WorkspaceFilesSnapshot(staged, identity))
        os.rename(staged, destination)
    return snapshot


def _move_exclusions(source, target, root, excludes, *, moved=None):
    moved = [] if moved is None else moved
    for name in excludes:
        if not _within(name, root) or name == root:
            continue
        relative = PurePosixPath(name).relative_to(root).as_posix()
        origin, destination = source / relative, target / relative
        try:
            with _directory(origin.parent) as parent:
                os.stat(origin.name, dir_fd=parent, follow_symlinks=False)
                with _directory(destination.parent, create=True) as output:
                    original_mode, target_mode = stat.S_IMODE(os.fstat(parent).st_mode), stat.S_IMODE(os.fstat(output).st_mode)
                    try:
                        os.fchmod(parent, original_mode | 0o700)
                        os.fchmod(output, target_mode | 0o700)
                        moved.append(name)
                        os.rename(origin.name, destination.name, src_dir_fd=parent, dst_dir_fd=output)
                    finally:
                        os.fchmod(parent, original_mode)
                        os.fchmod(output, target_mode)
        except FileNotFoundError:
            pass
    return moved


def restore_files(snapshot, workspace, *, exclude_paths=()):
    manifest, contents = _validate(snapshot)
    workspace = _absolute(workspace)
    roots = _roots(workspace, manifest["roots"])
    excludes = _excludes((*manifest["excludes"], *exclude_paths))
    entries = {name: row for name, row in manifest["entries"].items()
               if not any(_within(name, excluded) for excluded in excludes)}
    contents = {name: value for name, value in contents.items() if name in entries}
    # Verify every current mutable member before any root is moved. Excluded
    # selected resources are moved as directory entries without being read.
    _scan(workspace, roots, excludes)
    with _directory(workspace, create=True):
        pass
    with tempfile.TemporaryDirectory(prefix=".aka-files-transaction-", dir=workspace.parent) as temporary:
        transaction = Path(temporary)
        replacement, backup = transaction / "new", transaction / "prior"
        _write_tree(replacement, entries, contents, writable_directories=True)
        changes = []
        try:
            for root in roots:
                if any(_within(root, excluded) for excluded in excludes):
                    continue
                target, saved, prepared = workspace / root, backup / root, replacement / root
                with _directory(target.parent, create=True) as parent, _directory(saved.parent, create=True) as retained:
                    try:
                        os.stat(target.name, dir_fd=parent, follow_symlinks=False)
                    except FileNotFoundError:
                        existed = False
                    else:
                        existed = True
                    change = {"root": root, "existed": existed, "published": False, "moved": []}
                    changes.append(change)
                    if existed:
                        os.rename(target.name, saved.name, src_dir_fd=parent, dst_dir_fd=retained)
                    if existed and saved.is_dir() and any(_within(name, root) and name != root for name in excludes):
                        prepared.mkdir(parents=True, exist_ok=True)
                        _move_exclusions(saved, prepared, root, excludes, moved=change["moved"])
                        if root not in entries and not change["moved"]:
                            prepared.rmdir()
                    if prepared.exists():
                        with _directory(prepared.parent) as incoming:
                            os.rename(prepared.name, target.name, src_dir_fd=incoming, dst_dir_fd=parent)
                        change["published"] = True
            for name, row in sorted(entries.items(), key=lambda item: len(PurePosixPath(item[0]).parts), reverse=True):
                if row["kind"] == "directory":
                    (workspace / name).chmod(row["mode"])
        except BaseException:
            for change in reversed(changes):
                root = change["root"]
                target, saved, prepared = workspace / root, backup / root, replacement / root
                current = target if change["published"] or (target.exists() and not prepared.exists()) else prepared
                if change["existed"] and saved.exists():
                    if current.exists() and current.is_dir():
                        _move_exclusions(current, saved, root, change["moved"])
                    if target.exists():
                        discarded = transaction / "discarded" / root
                        discarded.parent.mkdir(parents=True, exist_ok=True)
                        os.rename(target, discarded)
                    with _directory(saved.parent) as retained, _directory(target.parent) as parent:
                        os.rename(saved.name, target.name, src_dir_fd=retained, dst_dir_fd=parent)
                elif not change["existed"] and target.exists():
                    discarded = transaction / "discarded" / root
                    discarded.parent.mkdir(parents=True, exist_ok=True)
                    os.rename(target, discarded)
            raise


def clear_paths(workspace, paths):
    workspace = _absolute(workspace)
    roots = _roots(workspace, paths)
    if any(PurePosixPath(name).parts[0] not in {"scratch", "profiles", "plans"} for name in roots):
        raise ValueError("clear_paths requires explicit mutable diagnostic paths")
    _scan(workspace, roots, ())
    with tempfile.TemporaryDirectory(prefix=".aka-clear-", dir=workspace.parent) as temporary:
        retained, moved = Path(temporary), []
        try:
            for name in roots:
                target, saved = workspace / name, retained / name
                try:
                    with _directory(target.parent) as parent, _directory(saved.parent, create=True) as output:
                        os.stat(target.name, dir_fd=parent, follow_symlinks=False)
                        moved.append(name)
                        os.rename(target.name, saved.name, src_dir_fd=parent, dst_dir_fd=output)
                except FileNotFoundError:
                    pass
        except BaseException:
            for name in reversed(moved):
                target, saved = workspace / name, retained / name
                if not saved.exists():
                    continue
                with _directory(saved.parent) as output, _directory(target.parent) as parent:
                    os.rename(saved.name, target.name, src_dir_fd=output, dst_dir_fd=parent)
            raise
