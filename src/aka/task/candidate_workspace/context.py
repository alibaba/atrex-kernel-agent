"""Retain explicitly supplied context separately from candidate source paths."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
import tempfile

from aka.contracts.workspace import SourceSnapshot

MANIFEST = ".atrex_context.json"
FORMAT = "aka-workspace-context-v1"
DIRECTORY_FORMAT = "aka-workspace-context-v2"


def _git(workspace, *args):
    return subprocess.run(["git", *args], cwd=workspace, capture_output=True)


def _revision_manifest(workspace, revision):
    result = _git(workspace, "show", f"{revision}:{MANIFEST}")
    return result.stdout if not result.returncode else None


def uses_context(workspace):
    target = Path(workspace) / MANIFEST
    if not (Path(workspace) / ".git").exists():
        return target.exists() or target.is_symlink()
    history = _git(workspace, "rev-list", "-1", "HEAD", "--", MANIFEST)
    path = context_store(workspace)
    store = path.exists()
    return target.exists() or target.is_symlink() or bool(history.stdout.strip()) or store


def _ordinary_file(path):
    value = path.lstat()
    if (not stat.S_ISREG(value.st_mode) or value.st_nlink != 1
            or value.st_mode & (stat.S_ISUID | stat.S_ISGID | stat.S_ISVTX)):
        raise ValueError(f"context must be a regular, unlinked file with ordinary modes: {path}")
    return stat.S_IMODE(value.st_mode)


def _safe_name(name):
    path = PurePosixPath(name)
    if (not path.parts or path.is_absolute() or ".." in path.parts
            or path.as_posix() != name or "\\" in name
            or any(ord(char) < 32 for char in name)
            or any(part.casefold() == ".git" for part in path.parts)
            or name == MANIFEST):
        raise ValueError(f"unsafe frozen context path: {name}")


def _decode(raw):
    value = json.loads(raw)
    fields = {"format", "source_paths", "context"}
    if isinstance(value, dict) and value.get("format") == DIRECTORY_FORMAT:
        fields.add("namespaces")
    if (not isinstance(value, dict) or set(value) != fields
            or value["format"] not in {FORMAT, DIRECTORY_FORMAT} or not isinstance(value["source_paths"], list)
            or not isinstance(value["context"], dict)):
        raise ValueError("invalid workspace context manifest")
    sources = value["source_paths"]
    if len(sources) != len(set(sources)):
        raise ValueError("duplicate source paths in workspace context manifest")
    names = [*sources, *value["context"]]
    if len(names) != len(set(names)):
        raise ValueError("source and frozen context paths overlap")
    names_set = set(names)
    for name in names:
        _safe_name(name)
        path = PurePosixPath(name)
        if any(parent.as_posix() in names_set for parent in path.parents):
            raise ValueError("source and frozen context paths overlap")
    for item in value["context"].values():
        if (not isinstance(item, dict) or set(item) != {"role", "sha256", "mode"}
                or item["role"] not in {"problem", "harness", "runtime", "readme", "configuration", "contribution"}
                or type(item["mode"]) is not int or not 0 <= item["mode"] <= 0o777):
            raise ValueError("invalid frozen context entry")
        digest = item["sha256"]
        if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("invalid frozen context digest")
    if value["format"] == DIRECTORY_FORMAT:
        namespaces = value["namespaces"]
        if not isinstance(namespaces, dict):
            raise ValueError("invalid workspace context namespaces")
        _validate_namespace_names(tuple(namespaces), sources)
        for root, snapshot in namespaces.items():
            if (not isinstance(snapshot, dict) or set(snapshot) != {"directories", "files"}
                    or not isinstance(snapshot["directories"], dict) or not isinstance(snapshot["files"], list)):
                raise ValueError("invalid frozen context directory identity")
            files = snapshot["files"]
            if len(files) != len(set(files)):
                raise ValueError("duplicate frozen context directory members")
            for name, mode in snapshot["directories"].items():
                _safe_name(name)
                if not _within(name, root) or type(mode) is not int or not 0 <= mode <= 0o777:
                    raise ValueError("invalid frozen context directory mode or path")
            if snapshot["directories"] or files:
                expected = {name for name in value["context"] if _within(name, root)}
                if set(files) != expected or set(snapshot["directories"]) != _necessary_directories(root, expected):
                    raise ValueError("frozen context directory membership differs from supplied files")
    return value


def _within(name, root):
    return name == root or name.startswith(root + "/")


def _validate_namespace_names(roots, sources):
    reserved = (MANIFEST, ".gitignore", "memory", "plans", "profiles", "eval", ".claude", ".qoder", ".agents")
    if len(roots) != len(set(roots)):
        raise ValueError("duplicate workspace context directory declarations")
    for index, root in enumerate(roots):
        _safe_name(root)
        if PurePosixPath(root).parts[0].startswith(".atrex_"):
            raise ValueError("context directory overlaps Workspace management")
        for name in (*reserved, *sources, *roots[:index]):
            if _within(root, name) or _within(name, root):
                raise ValueError(f"context directory overlaps source, management or another namespace: {root}")


def _necessary_directories(root, files):
    directories = {root}
    for name in files:
        for parent in PurePosixPath(name).parents:
            path = parent.as_posix()
            if _within(path, root):
                directories.add(path)
    return directories


def _directory_mode(path):
    value = path.lstat()
    if not stat.S_ISDIR(value.st_mode) or value.st_mode & (stat.S_ISUID | stat.S_ISGID | stat.S_ISVTX):
        raise ValueError(f"context directory must have ordinary directory modes: {path}")
    return stat.S_IMODE(value.st_mode)


def _namespace_tree(workspace, root):
    """Inspect the declared namespace without following any replacement link."""
    members = {}
    base = workspace / root
    for parent in base.parents:
        if parent == workspace.parent:
            break
        if parent.is_symlink() or (parent.exists() and not parent.is_dir()):
            raise ValueError(f"frozen context directory parent changed: {root}")

    def inspect(path):
        try:
            mode = path.lstat().st_mode
        except FileNotFoundError:
            return
        name = path.relative_to(workspace).as_posix()
        _safe_name(name)
        if stat.S_ISDIR(mode):
            members[name] = ("directory", stat.S_IMODE(mode))
            for child in sorted(path.iterdir()):
                inspect(child)
        elif stat.S_ISREG(mode):
            members[name] = ("file", stat.S_IMODE(mode))
        else:
            members[name] = ("replacement", stat.S_IMODE(mode))
    inspect(base)
    return members


def _seal_namespaces(workspace, value):
    for root, snapshot in value.get("namespaces", {}).items():
        files = {name for name in value["context"] if _within(name, root)}
        expected_directories = _necessary_directories(root, files)
        actual = _namespace_tree(workspace, root)
        if set(actual) != files | expected_directories:
            raise ValueError(f"context directory has missing or unregistered members: {root}")
        directories = {name: _directory_mode(workspace / name) for name in expected_directories}
        if any(actual[name][0] != "file" for name in files):
            raise ValueError(f"context directory file was replaced: {root}")
        snapshot["directories"] = dict(sorted(directories.items()))
        snapshot["files"] = sorted(files)
    _decode(_encoded(value))
    return value


def _namespace_changes(workspace, value):
    changed = []
    for root, snapshot in value.get("namespaces", {}).items():
        if not snapshot["directories"]:
            continue  # Registration is not anchored until the first source transaction.
        try:
            actual = _namespace_tree(workspace, root)
        except (OSError, ValueError):
            changed.append(root)
            continue
        expected = {*snapshot["directories"], *snapshot["files"]}
        changed.extend(sorted(set(actual) ^ expected))
        for name, mode in snapshot["directories"].items():
            if actual.get(name) != ("directory", mode):
                changed.append(name)
    return list(dict.fromkeys(changed))


def _encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()


def _write_manifest(workspace, value):
    path = workspace / MANIFEST
    temporary = workspace / (MANIFEST + ".tmp")
    if temporary.exists() or temporary.is_symlink():
        raise ValueError("workspace context manifest transaction already exists")
    try:
        with temporary.open("xb") as stream:
            stream.write(_encoded(value))
            stream.flush()
            os.fchmod(stream.fileno(), 0o644)
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _sync_directory(workspace)
    finally:
        temporary.unlink(missing_ok=True)


def _sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def context_store(workspace):
    result = _git(workspace, "rev-parse", "--git-common-dir")
    if result.returncode:
        raise RuntimeError("cannot locate Workspace context store")
    path = Path(result.stdout.decode().strip())
    if not path.is_absolute():
        path = Path(workspace) / path
    path = path.resolve(strict=True) / "aka-context"
    if path.is_symlink() or (path.exists() and not path.is_dir()):
        raise ValueError("invalid Workspace context store")
    return path


def _retained(workspace, digest, *, store=None):
    target = (context_store(workspace) if store is None else store) / digest
    if _ordinary_file(target) != 0o444:
        raise ValueError("retained Workspace context mode changed")
    payload = target.read_bytes()
    if hashlib.sha256(payload).hexdigest() != digest:
        raise ValueError("retained Workspace context bytes changed")
    return payload


def _retain(workspace, digest, payload, *, store=None):
    directory = context_store(workspace) if store is None else store
    created = not directory.exists()
    directory.mkdir(exist_ok=True)
    if created:
        _sync_directory(directory.parent)
    path = directory / digest
    if path.exists() or path.is_symlink():
        if _retained(workspace, digest, store=directory) != payload:
            raise ValueError("retained Workspace context collision")
        return
    with tempfile.NamedTemporaryFile(prefix=".retain-", dir=directory, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(payload)
            stream.flush()
            os.fchmod(stream.fileno(), 0o444)
            os.fsync(stream.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                # The canonical retained identity is never overwritten.
                if path.read_bytes() != payload:
                    raise ValueError("retained Workspace context collision")
            temporary.unlink()
            _sync_directory(directory)
            _retained(workspace, digest, store=directory)
        finally:
            temporary.unlink(missing_ok=True)


def load_context(workspace):
    workspace = Path(workspace)
    path = workspace / MANIFEST
    if _ordinary_file(path) != 0o644:
        raise ValueError("workspace context manifest mode changed")
    value = _decode(path.read_bytes())
    committed = _revision_manifest(workspace, "HEAD")
    if committed is not None:
        if _decode(committed) != value:
            raise ValueError("workspace context management manifest changed")
        if any(not item["directories"] for item in value.get("namespaces", {}).values()):
            raise ValueError("committed context directory identity is not sealed")
    elif _git(workspace, "rev-list", "-1", "HEAD", "--", MANIFEST).stdout.strip():
        raise ValueError("workspace context management anchor is missing")
    return value


def begin_context(workspace, source_paths, *, context_directories=()):
    workspace = Path(workspace)
    if uses_context(workspace):
        raise ValueError("workspace context is already initialized")
    value = {"format": FORMAT, "source_paths": list(source_paths), "context": {}}
    roots = tuple(context_directories)
    if roots:
        _validate_namespace_names(roots, value["source_paths"])
        value["format"] = DIRECTORY_FORMAT
        value["namespaces"] = {root: {"directories": {}, "files": []} for root in roots}
        _create_namespace_roots(workspace, roots)
    _decode(_encoded(value))
    _write_manifest(workspace, value)


def _create_namespace_roots(workspace, roots):
    for root in roots:
        _namespace_tree(workspace, root)
        target = workspace / root
        if target.exists() or target.is_symlink():
            _directory_mode(target)
    for root in roots:
        (workspace / root).mkdir(parents=True, exist_ok=True)


def register_context_directories(workspace, roots):
    """Permit an unanchored file-format scaffold to receive explicit namespaces."""
    roots = tuple(roots)
    value = load_context(workspace)
    if not roots:
        return  # The default never clears an existing namespace or migrates a format.
    previous = tuple(value.get("namespaces", {}))
    if set(roots) == set(previous):
        return
    if _revision_manifest(workspace, "HEAD") is not None:
        raise ValueError("cannot change context directory declarations in an existing source history")
    if previous:
        raise ValueError("context directory declarations already registered")
    _validate_namespace_names(roots, value["source_paths"])
    for root in roots:
        if any(_within(root, name) for name in value["context"]):
            raise ValueError("context directory overlaps an existing file contribution")
    _create_namespace_roots(workspace, roots)
    if roots:
        value["format"] = DIRECTORY_FORMAT
        value["namespaces"] = {root: {"directories": {}, "files": []} for root in roots}
        _write_manifest(workspace, value)


def _projection_mode(workspace, name):
    target = workspace / name
    for parent in target.parents:
        if parent == workspace.parent:
            break
        if parent.is_symlink() or not parent.is_dir():
            raise ValueError(f"frozen context parent changed: {name}")
    return _ordinary_file(target)


def _projection_changes(workspace, value):
    changed = _namespace_changes(workspace, value)
    store = context_store(workspace)
    for name, item in value["context"].items():
        try:
            _retained(workspace, item["sha256"], store=store)
            mode = _projection_mode(workspace, name)
            target = workspace / name
            if (mode != item["mode"] or hashlib.sha256(target.read_bytes()).hexdigest() != item["sha256"]):
                changed.append(name)
        except (OSError, ValueError):
            changed.append(name)
    return changed


def ensure_context_excluded(workspace, value):
    result = _git(workspace, "rev-parse", "--git-path", "info/exclude")
    if result.returncode:
        raise RuntimeError("cannot locate context exclude file")
    path = Path(result.stdout.decode().strip())
    if not path.is_absolute():
        path = workspace / path
    if path.is_symlink():
        raise ValueError("workspace context exclude file must not be a symlink")
    path.parent.mkdir(parents=True, exist_ok=True)
    original = path.read_text() if path.exists() else ""
    # Git wildmatch escapes retain exact supplied membership, including punctuation.
    rules = ["/" + "".join("\\" + char if char in "\\*?[]!# " else char for char in name)
             for name in value["context"]]
    rules.extend("/" + "".join("\\" + char if char in "\\*?[]!# " else char for char in name) + "/"
                 for name in value.get("namespaces", {}))
    missing = [rule for rule in rules if rule not in original.splitlines()]
    if missing:
        path.write_text(original + ("\n" if original and not original.endswith("\n") else "")
                        + "\n".join(missing) + "\n")


def install_context(workspace, files, *, role="contribution"):
    from .content import selected_file_bytes, _install_files

    workspace = Path(workspace)
    files = tuple(files)
    value = load_context(workspace)
    changed = _projection_changes(workspace, value)
    if changed:
        raise ValueError("frozen workspace context changed: " + ", ".join(changed))
    selected = selected_file_bytes(workspace, files, reserved=(MANIFEST, *value["source_paths"]))
    committed = _revision_manifest(workspace, "HEAD")
    if committed is not None and _decode(committed) != value:
        raise ValueError("workspace context management manifest changed")
    additions = {}
    modes = {}
    for item in files:
        name = item.path
        if item.source is not None:
            source_mode = _ordinary_file(item.source)
        else:
            source_mode = None
        modes[name] = source_mode
        expected = value["context"].get(name)
        if expected is not None:
            if (hashlib.sha256(selected[name]).hexdigest() != expected["sha256"]
                    or (source_mode is not None and source_mode != expected["mode"])):
                raise ValueError(f"frozen workspace input changed: {name}")
            continue
        if committed is not None:
            raise ValueError(f"cannot add context after the source record: {name}")
        _safe_name(name)
        for existing in value["context"]:
            path, other = PurePosixPath(name), PurePosixPath(existing)
            if path in other.parents or other in path.parents:
                raise ValueError(f"conflicting frozen context contribution: {name}")
        target = workspace / name
        if target.exists():
            existing_mode = _ordinary_file(target)
            if target.read_bytes() != selected[name] or (source_mode is not None and existing_mode != source_mode):
                raise ValueError(f"immutable workspace contribution changed: {name}")
        additions[name] = item
    # Everything is validated before any projection is installed.
    _install_files(workspace, tuple(additions.values()))
    store = context_store(workspace)
    for name in additions:
        target = workspace / name
        payload = target.read_bytes()
        if payload != selected[name] or (modes[name] is not None and _ordinary_file(target) != modes[name]):
            raise ValueError(f"workspace context source changed during installation: {name}")
        digest = hashlib.sha256(payload).hexdigest()
        _retain(workspace, digest, payload, store=store)
        value["context"][name] = {"role": role, "sha256": digest,
                                   "mode": _ordinary_file(target)}
    _decode(_encoded(value))
    if additions:
        _write_manifest(workspace, value)
    ensure_context_excluded(workspace, value)
    digest = hashlib.sha256()
    for name in selected:
        payload = selected[name]
        digest.update(name.encode() + b"\0" + len(payload).to_bytes(8, "big") + payload)
    return SourceSnapshot(tuple(selected), digest.hexdigest())


def context_changes(workspace, revision):
    workspace = Path(workspace)
    raw = _revision_manifest(workspace, revision)
    if raw is None:
        return [MANIFEST]
    try:
        original = _decode(raw)
    except (ValueError, TypeError, KeyError):
        return [MANIFEST]
    changed = _projection_changes(workspace, original)
    try:
        current = load_context(workspace)
        if current != original:
            changed.insert(0, MANIFEST)
    except (OSError, ValueError, TypeError, KeyError):
        changed.insert(0, MANIFEST)
    # Context and management metadata must never have index-only changes.
    result = _git(workspace, "diff", "--cached", "--name-only", "-z", revision,
                  "--", MANIFEST, *original["context"])
    if result.returncode:
        return list(dict.fromkeys([MANIFEST, *changed]))
    changed.extend(name for name in result.stdout.decode().split("\0") if name)
    return list(dict.fromkeys(changed))


def context_record_file(workspace, paths):
    """Validate an explicit record and attach the first management anchor."""
    from aka.contracts.content import WorkspaceFile

    value = load_context(workspace)
    if (MANIFEST in paths or set(value["context"]).intersection(paths)
            or any(_within(name, root) for name in paths for root in value.get("namespaces", {}))):
        raise ValueError("record cannot stage frozen context or its management manifest")
    staged = _git(workspace, "diff", "--cached", "--name-only", "--", MANIFEST, *value["context"])
    if staged.returncode or staged.stdout.strip():
        raise ValueError("record cannot retain staged context or management changes")
    changed = _projection_changes(workspace, value)
    if changed:
        raise ValueError("frozen workspace context changed: " + ", ".join(changed))
    raw = _revision_manifest(workspace, "HEAD")
    if raw is not None:
        if _decode(raw) != value:
            raise ValueError("workspace context management manifest changed")
        return None
    if not set(value["source_paths"]).issubset(paths):
        raise ValueError("first workspace record must include all selected source files")
    if value.get("namespaces"):
        value = _seal_namespaces(workspace, value)
        return WorkspaceFile(MANIFEST, text=_encoded(value).decode())
    return WorkspaceFile(MANIFEST, source=workspace / MANIFEST)


def restore_context_projection(workspace, revision):
    """Restore only retained context projections; preserve unrelated Agent output."""
    workspace = Path(workspace)
    raw = _revision_manifest(workspace, revision)
    if raw is None:
        raise ValueError("frozen context management manifest is missing from the revision")
    value = _decode(raw)
    changed = _projection_changes(workspace, value)
    store = context_store(workspace)
    files = [name for name in changed if name in value["context"]]
    retained = {name: _retained(workspace, value["context"][name]["sha256"], store=store) for name in files}
    directories = {}
    for root, snapshot in value.get("namespaces", {}).items():
        if not snapshot["directories"]:
            raise ValueError("cannot restore an unsealed context namespace")
        actual = _namespace_tree(workspace, root)
        extra = set(actual) - {*snapshot["directories"], *snapshot["files"]}
        if extra:
            raise ValueError("context namespace has unknown extra members; refusing deletion: " + ", ".join(sorted(extra)))
        directories.update(snapshot["directories"])
    # Validate every declared tree and retained file before changing any projection.
    restored_directories = []
    try:
        for name in sorted(directories, key=lambda path: (len(PurePosixPath(path).parts), path)):
            target = workspace / name
            if target.is_symlink() or (target.exists() and not target.is_dir()):
                target.unlink()
            target.mkdir(parents=True, exist_ok=True)
            target.chmod(directories[name] | stat.S_IWUSR | stat.S_IXUSR)
            if name in changed:
                restored_directories.append(name)
        for name in files:
            target = workspace / name
            for parent in target.parents:
                if parent == workspace.parent:
                    break
                if parent.is_symlink() or (parent.exists() and not parent.is_dir()):
                    raise ValueError(f"cannot restore replaced frozen context parent: {name}")
            if target.is_symlink() or target.is_file():
                target.unlink()
            elif target.exists():
                if value.get("namespaces") and any(_within(name, root) for root in value["namespaces"]):
                    target.rmdir()  # An unknown member was rejected above; only an empty replacement remains.
                else:
                    raise ValueError(f"cannot restore replaced frozen context directory: {name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(retained[name])
            target.chmod(value["context"][name]["mode"])
    finally:
        for name in sorted(directories, key=lambda path: len(PurePosixPath(path).parts), reverse=True):
            target = workspace / name
            if target.is_dir() and not target.is_symlink():
                target.chmod(directories[name])
    path = workspace / MANIFEST
    if path.is_symlink() or path.is_file():
        if path.is_symlink() or path.read_bytes() != raw:
            path.unlink()
            path.write_bytes(raw)
    elif path.exists():
        raise ValueError("cannot restore replaced context management directory")
    else:
        path.write_bytes(raw)
    path.chmod(0o644)
    ensure_context_excluded(workspace, value)
    indexed = []
    if value["context"]:
        result = _git(workspace, "diff", "--cached", "--name-only", "-z", revision,
                      "--", *value["context"])
        if result.returncode:
            raise RuntimeError("cannot inspect frozen context index membership")
        indexed = [name for name in result.stdout.decode().split("\0") if name]
        result = _git(workspace, "update-index", "--force-remove", "--", *value["context"])
        if result.returncode:
            raise RuntimeError("cannot restore frozen context index membership")
    return list(dict.fromkeys([*files, *restored_directories, *indexed]))
