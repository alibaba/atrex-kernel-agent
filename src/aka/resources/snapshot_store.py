"""Immutable Resources content versions, independent of callable registries."""
from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import sys
import uuid
from contextlib import contextmanager

from aka.contracts.agent_resources import ResourcesSnapshot

FORMAT = "aka.resources.snapshot-content.v1"
STORE_FORMAT = "aka.resources.content-store.v1"
DESCRIPTOR_FORMAT = "aka.resources.snapshot-descriptor.v1"
IGNORED = frozenset({".git", "__pycache__", ".pytest_cache"})


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _identity(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("invalid Resources content identity")
    return value


def _absolute(path):
    path = Path(path).absolute()
    if "\\" in str(path) or any(ord(c) < 32 for c in str(path)) or path.resolve(strict=False) != path:
        raise ValueError(f"Resources paths must not traverse links or aliases: {path}")
    for parent in path.parents:
        if parent.is_symlink() or (parent.exists() and not parent.is_dir()):
            raise ValueError(f"unsafe Resources parent path: {path}")
    return path


def _member(name, *, root_file=False):
    if not isinstance(name, str):
        raise ValueError("invalid Resources member name")
    if root_file and name == ".":
        return
    path = PurePosixPath(name)
    if (not path.parts or path.is_absolute() or ".." in path.parts
            or path.as_posix() != name or "\\" in name
            or any(ord(char) < 32 for char in name)):
        raise ValueError(f"unsafe Resources member: {name}")


def _mode(value):
    if type(value) is not int or not 0 <= value <= 0o777:
        raise ValueError("invalid Resources mode")
    return value


def _open_directory(path, *, create=False):
    path = _absolute(path)
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open("/", flags)
    try:
        for part in path.parts[1:]:
            try:
                child = os.open(part, flags, dir_fd=descriptor)
            except FileNotFoundError:
                if not create:
                    raise
                try:
                    os.mkdir(part, 0o700, dir_fd=descriptor)
                    os.fsync(descriptor)
                except FileExistsError:
                    pass
                child = os.open(part, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        _assert_directory(path, descriptor)
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _make_directory(path):
    descriptor = _open_directory(path, create=True)
    os.close(descriptor)


def _remove_tree(parent, name, expected):
    current = os.stat(name, dir_fd=parent, follow_symlinks=False)
    if (current.st_dev, current.st_ino) != expected or not stat.S_ISDIR(current.st_mode):
        raise ValueError("Resources staging directory was replaced")
    directory = os.open(name, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0), dir_fd=parent)
    try:
        opened = os.fstat(directory)
        if (opened.st_dev, opened.st_ino) != expected:
            raise ValueError("Resources staging directory was replaced")
        for child in os.listdir(directory):
            value = os.stat(child, dir_fd=directory, follow_symlinks=False)
            if stat.S_ISDIR(value.st_mode):
                _remove_tree(directory, child, (value.st_dev, value.st_ino))
            else:
                os.unlink(child, dir_fd=directory)
    finally:
        os.close(directory)
    os.rmdir(name, dir_fd=parent)


@contextmanager
def _staging_directory(parent_path):
    parent_path = _absolute(parent_path)
    parent = _open_directory(parent_path, create=True)
    name = ".resources-snapshot-" + uuid.uuid4().hex
    expected = None
    directory = None
    try:
        _assert_directory(parent_path, parent)
        os.mkdir(name, 0o700, dir_fd=parent)
        expected = os.stat(name, dir_fd=parent, follow_symlinks=False)
        directory = os.open(name, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0), dir_fd=parent)
        _assert_directory(parent_path, parent)
        os.mkdir("snapshot", 0o700, dir_fd=directory)
        yield parent_path / name / "snapshot"
    finally:
        if directory is not None:
            os.close(directory)
        try:
            if expected is not None:
                _remove_tree(parent, name, (expected.st_dev, expected.st_ino))
        finally:
            os.close(parent)


def _assert_directory(path, descriptor):
    _absolute(path)
    actual = Path(path).lstat()
    opened = os.fstat(descriptor)
    if (actual.st_dev, actual.st_ino) != (opened.st_dev, opened.st_ino) or not stat.S_ISDIR(actual.st_mode):
        raise ValueError("Resources parent directory changed during IO")


def _read_at(parent, name, *, immutable=False):
    before = os.stat(name, dir_fd=parent, follow_symlinks=False)
    if (not stat.S_ISREG(before.st_mode) or (immutable and before.st_nlink != 1)
            or before.st_mode & (stat.S_ISUID | stat.S_ISGID | stat.S_ISVTX)):
        raise ValueError("Resources content must be an ordinary file without retained aliases")
    mode = stat.S_IMODE(before.st_mode)
    if immutable and mode != 0o444:
        raise ValueError("immutable Resources content mode changed")
    descriptor = os.open(name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=parent)
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise ValueError("Resources content replaced during read")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            payload = stream.read()
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    final = os.stat(name, dir_fd=parent, follow_symlinks=False)
    identity = lambda value: (value.st_dev, value.st_ino, value.st_mode, value.st_nlink,
                              value.st_size, value.st_mtime_ns, value.st_ctime_ns)
    if identity(before) != identity(after) or identity(before) != identity(final):
        raise ValueError("Resources content changed during read")
    return payload, mode


def _read_file(path, *, immutable=False):
    path = _absolute(path)
    parent = _open_directory(path.parent)
    try:
        result = _read_at(parent, path.name, immutable=immutable)
        _assert_directory(path.parent, parent)
        return result
    finally:
        os.close(parent)


def _scan_root(source):
    source = _absolute(source)
    root_stat = source.lstat()
    if (not (stat.S_ISDIR(root_stat.st_mode) or stat.S_ISREG(root_stat.st_mode))
            or root_stat.st_mode & (stat.S_ISUID | stat.S_ISGID | stat.S_ISVTX)):
        raise ValueError(f"Resources source must be an ordinary file or directory: {source}")
    objects = {}
    entries = []

    def file_entry(path, name, expected):
        payload, mode = _read_file(path)
        current = path.lstat()
        if (current.st_dev, current.st_ino, current.st_mode) != (expected.st_dev, expected.st_ino, expected.st_mode) or mode != stat.S_IMODE(expected.st_mode):
            raise ValueError("Resources source file changed during scan")
        digest = _digest(payload)
        if digest in objects and objects[digest] != payload:
            raise ValueError("Resources content identity collision")
        objects[digest] = payload
        entries.append({"path": name, "kind": "file", "mode": mode, "sha256": digest})

    def walk(directory, expected):
        descriptor = _open_directory(directory)
        signature = lambda value: (value.st_dev, value.st_ino, value.st_mode)
        members = lambda: sorted(name for name in os.listdir(descriptor)
                                 if name not in IGNORED and Path(name).suffix not in {".pyc", ".pyo"})
        try:
            before = os.fstat(descriptor)
            if signature(before) != signature(expected):
                raise ValueError("Resources source directory changed during scan")
            names = members()
            for member in names:
                path = directory / member
                name = path.relative_to(source).as_posix()
                _member(name)
                value = os.stat(member, dir_fd=descriptor, follow_symlinks=False)
                mode = value.st_mode
                if stat.S_ISDIR(mode) and not mode & (stat.S_ISUID | stat.S_ISGID | stat.S_ISVTX):
                    entries.append({"path": name, "kind": "directory", "mode": stat.S_IMODE(mode)})
                    walk(path, value)
                elif stat.S_ISREG(mode):
                    file_entry(path, name, value)
                else:
                    raise ValueError(f"unsafe Resources source member: {name}")
            if signature(os.fstat(descriptor)) != signature(before) or members() != names:
                raise ValueError("Resources source directory changed during scan")
            _assert_directory(directory, descriptor)
        finally:
            os.close(descriptor)
    kind = "directory" if stat.S_ISDIR(root_stat.st_mode) else "file"
    if kind == "file":
        file_entry(source, ".", root_stat)
    else:
        walk(source, root_stat)
    row = {"source": str(source), "kind": kind, "mode": stat.S_IMODE(root_stat.st_mode),
           "entries": sorted(entries, key=lambda entry: entry["path"])}
    return row, objects


def inspect_root(source):
    """Read the exact Resources tree twice without writing or invoking a registry."""
    first, payloads = _scan_root(source)
    second, repeated = _scan_root(source)
    if first != second or payloads != repeated:
        raise ValueError("Resources source changed during inspection")
    return first


def _roots(roots):
    values = sorted({_absolute(root) for root in roots}, key=lambda path: (len(path.parts), str(path)))
    selected = []
    for path in values:
        if not any(path.is_relative_to(parent) for parent in selected):
            selected.append(path)
    return tuple(sorted(selected))


def _logical(value):
    if (not isinstance(value, dict) or set(value) != {"format", "selection", "roots"}
            or value["format"] != FORMAT or not isinstance(value["roots"], list)):
        raise ValueError("invalid Resources logical manifest")
    _canonical(value["selection"])
    sources = []
    for root in value["roots"]:
        if (not isinstance(root, dict) or set(root) != {"source", "kind", "mode", "entries"}
                or not isinstance(root["kind"], str) or root["kind"] not in {"file", "directory"}
                or not isinstance(root["entries"], list) or not isinstance(root["source"], str)):
            raise ValueError("invalid Resources retained root")
        source = Path(root["source"])
        if str(source) != root["source"] or not source.is_absolute() or ".." in source.parts or "\\" in str(source) or any(ord(c) < 32 for c in str(source)):
            raise ValueError("invalid Resources original root binding")
        if any(source.is_relative_to(other) or other.is_relative_to(source) for other in sources):
            raise ValueError("overlapping Resources retained roots")
        sources.append(source)
        _mode(root["mode"])
        members = {}
        for entry in root["entries"]:
            fields = {"path", "kind", "mode"}
            if isinstance(entry, dict) and entry.get("kind") == "file":
                fields.add("sha256")
            if (not isinstance(entry, dict) or set(entry) != fields
                    or not isinstance(entry["kind"], str) or entry["kind"] not in {"file", "directory"}):
                raise ValueError("invalid Resources retained member")
            _member(entry["path"], root_file=root["kind"] == "file")
            if entry["path"] != "." and (IGNORED.intersection(PurePosixPath(entry["path"]).parts) or PurePosixPath(entry["path"]).suffix in {".pyc", ".pyo"}):
                raise ValueError("Resources retained members include excluded cache or Git content")
            _mode(entry["mode"])
            if entry["path"] in members:
                raise ValueError("duplicate Resources retained member")
            if entry["kind"] == "file":
                _identity(entry["sha256"])
            members[entry["path"]] = entry
        if root["kind"] == "file":
            if (set(members) != {"."} or members["."]["kind"] != "file"
                    or members["."]["mode"] != root["mode"]):
                raise ValueError("invalid Resources file-root identity")
        else:
            for name in members:
                for parent in PurePosixPath(name).parents:
                    if parent.as_posix() != "." and members.get(parent.as_posix(), {}).get("kind") != "directory":
                        raise ValueError("Resources retained member has no directory parent")
        if [entry["path"] for entry in root["entries"]] != sorted(members):
            raise ValueError("Resources retained members are not in canonical order")
    if sources != sorted(sources):
        raise ValueError("Resources retained roots are not in canonical order")
    return value


def _references(manifest):
    return {entry["sha256"] for root in manifest["roots"] for entry in root["entries"] if entry["kind"] == "file"}


def _sync_directory(path):
    descriptor = _open_directory(path)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_immutable(path, payload):
    """Publish complete ordinary content without replacing any existing identity."""
    path = _absolute(path)
    parent = _open_directory(path.parent, create=True)
    temporary = ".resource-write-" + uuid.uuid4().hex
    created = False
    try:
        try:
            existing = _read_at(parent, path.name, immutable=True)[0]
        except FileNotFoundError:
            existing = None
        if existing is not None:
            if existing != payload:
                raise ValueError("existing Resources content object is corrupt")
            _assert_directory(path.parent, parent)
            return
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600, dir_fd=parent)
        created = True
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fchmod(stream.fileno(), 0o444)
            os.fsync(stream.fileno())
        _assert_directory(path.parent, parent)
        try:
            os.link(temporary, path.name, src_dir_fd=parent, dst_dir_fd=parent, follow_symlinks=False)
        except FileExistsError:
            if _read_at(parent, path.name, immutable=True)[0] != payload:
                raise ValueError("existing Resources content object is corrupt")
        os.unlink(temporary, dir_fd=parent)
        created = False
        os.fsync(parent)
        if _read_at(parent, path.name, immutable=True)[0] != payload:
            raise ValueError("published Resources object differs from complete content")
        _assert_directory(path.parent, parent)
    finally:
        if created:
            os.unlink(temporary, dir_fd=parent)
        os.close(parent)


def _publish_directory(source, destination):
    """Publish using verified directory FDs, without replacement or parent-link following."""
    source, destination = _absolute(source), _absolute(destination)
    source_parent = _open_directory(source.parent)
    target_parent = None
    try:
        target_parent = _open_directory(destination.parent)
        _assert_directory(source.parent, source_parent)
        _assert_directory(destination.parent, target_parent)
        library = ctypes.CDLL(None, use_errno=True)
        if sys.platform == "darwin" and hasattr(library, "renameatx_np"):
            operation = library.renameatx_np
            operation.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
            operation.restype = ctypes.c_int
            result = operation(source_parent, os.fsencode(source.name), target_parent, os.fsencode(destination.name), 4)
        elif sys.platform.startswith("linux") and hasattr(library, "renameat2"):
            operation = library.renameat2
            operation.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
            operation.restype = ctypes.c_int
            result = operation(source_parent, os.fsencode(source.name), target_parent, os.fsencode(destination.name), 1)
        else:
            raise RuntimeError("Resources snapshots require atomic directory publication without replacement")
        if result:
            code = ctypes.get_errno()
            raise OSError(code, os.strerror(code), str(destination))
        os.fsync(target_parent)
        _assert_directory(destination.parent, target_parent)
    finally:
        os.close(source_parent)
        if target_parent is not None:
            os.close(target_parent)


def _store_paths(store, *, create=False):
    store = _absolute(store)
    if create:
        _make_directory(store)
        for name in ("objects", "versions"):
            _make_directory(store / name)
        _write_immutable(store / "format.json", _canonical({"format": STORE_FORMAT}))
        _sync_directory(store)
    for path in (store, store / "objects", store / "versions"):
        if not path.is_dir() or path.is_symlink():
            raise ValueError("missing or unsafe Resources content directory")
    if _read_file(store / "format.json", immutable=True)[0] != _canonical({"format": STORE_FORMAT}):
        raise ValueError("Resources content directory format changed")
    return store


def _storage(snapshot):
    directory = _absolute(snapshot.directory)
    _identity(snapshot.identity)
    raw, _ = _read_file(directory / "descriptor.json", immutable=True)
    descriptor = json.loads(raw)
    if (not isinstance(descriptor, dict) or set(descriptor) != {"format", "identity", "storage", "content_directory"}
            or descriptor["format"] != DESCRIPTOR_FORMAT or descriptor["identity"] != snapshot.identity):
        raise ValueError("invalid Resources snapshot descriptor")
    if snapshot.content_directory is None:
        if descriptor["storage"] != "embedded" or descriptor["content_directory"] is not None:
            raise ValueError("shared Resources snapshots require their explicit content locator")
        store = directory / "content"
    else:
        if descriptor["storage"] != "shared" or not isinstance(descriptor["content_directory"], str):
            raise ValueError("Resources snapshot content mode does not match its explicit locator")
        store = snapshot.content_directory
    return _store_paths(store)


def _read_logical(snapshot):
    store = _storage(snapshot)
    raw, _ = _read_file(store / "versions" / (snapshot.identity + ".json"), immutable=True)
    if _digest(raw) != snapshot.identity:
        raise ValueError("Resources logical manifest identity changed")
    value = _logical(json.loads(raw))
    _nonoverlap(value, _absolute(snapshot.directory), store)
    if snapshot.content_directory is not None and store.is_relative_to(_absolute(snapshot.directory)):
        raise ValueError("shared Resources content and descriptor directories must not overlap")
    if raw != _canonical(value):
        raise ValueError("Resources logical manifest is not canonical")
    return store, value


def _object(store, digest):
    _identity(digest)
    payload, _ = _read_file(store / "objects" / digest, immutable=True)
    if _digest(payload) != digest:
        raise ValueError("Resources retained object identity changed")
    return payload


def load_snapshot(snapshot):
    """Verify only this retained closure; never inspect or fall back to original roots."""
    store, value = _read_logical(snapshot)
    for digest in _references(value):
        _object(store, digest)
    return value


def object_reader(snapshot, manifest):
    """Bind one verified version, then validate each requested object without reparsing."""
    store, value = _read_logical(snapshot)
    if _canonical(manifest) != _canonical(value):
        raise ValueError("Resources object reader manifest differs from its snapshot")
    references = _references(value)
    def read(digest):
        if digest not in references:
            raise ValueError("Resources object is not referenced by this snapshot")
        return _object(store, digest)
    return read


def read_object(snapshot, digest):
    store, manifest = _read_logical(snapshot)
    if digest not in _references(manifest):
        raise ValueError("Resources object is not referenced by this snapshot")
    return _object(store, digest)


def _nonoverlap(manifest, destination, store):
    for row in manifest["roots"]:
        root = Path(row["source"])
        for path in (destination, store):
            if path.is_relative_to(root) or root.is_relative_to(path):
                raise ValueError("Resources snapshot/content directories overlap selected source roots")
    if store == destination or destination.is_relative_to(store):
        raise ValueError("Resources descriptor directory overlaps its content store")


def _publish_descriptor(staging, destination, identity, content_directory):
    descriptor = {"format": DESCRIPTOR_FORMAT, "identity": identity,
                  "storage": "embedded" if content_directory is None else "shared",
                  "content_directory": str(content_directory) if content_directory is not None else None}
    _write_immutable(staging / "descriptor.json", _canonical(descriptor))
    _sync_directory(staging)
    _publish_directory(staging, destination)


def _publish_snapshot(manifest, read, destination, *, content_directory=None, before_publish=None):
    manifest = _logical(manifest)
    destination = _absolute(destination)
    locator = _absolute(content_directory) if content_directory is not None else None
    actual_store = locator if locator is not None else destination / "content"
    _nonoverlap(manifest, destination, actual_store)
    if locator is not None and (locator.is_relative_to(destination) or destination.is_relative_to(locator)):
        raise ValueError("shared Resources content and descriptor directories must not overlap")
    identity = _digest(_canonical(manifest))
    expected = ResourcesSnapshot(destination, identity, locator)
    if destination.exists() or destination.is_symlink():
        if load_snapshot(expected) != manifest:
            raise ValueError("existing Resources snapshot differs from requested version")
        if before_publish is not None:
            before_publish(json.loads(_canonical(manifest)), object_reader(expected, manifest))
        return expected
    with _staging_directory(destination.parent) as staging:
        store = _store_paths(locator if locator is not None else staging / "content", create=True)
        for digest in sorted(_references(manifest)):
            payload = read(digest)
            if _digest(payload) != digest:
                raise ValueError("Resources captured bytes differ from their logical identity")
            _write_immutable(store / "objects" / digest, payload)
        _write_immutable(store / "versions" / (identity + ".json"), _canonical(manifest))
        # Validate the full retained version before its descriptor can become visible.
        for digest in _references(manifest):
            _object(store, digest)
        if before_publish is not None:
            references = _references(manifest)
            def retained_read(digest):
                if digest not in references:
                    raise ValueError("Resources object is not referenced by this snapshot")
                return _object(store, digest)
            before_publish(json.loads(_canonical(manifest)), retained_read)
        try:
            _publish_descriptor(staging, destination, identity, locator)
        except OSError as error:
            if error.errno not in {errno.EEXIST, errno.ENOTEMPTY}:
                raise
            if load_snapshot(expected) != manifest:
                raise ValueError("racing Resources snapshot has another identity") from error
            if before_publish is not None:
                before_publish(json.loads(_canonical(manifest)), object_reader(expected, manifest))
    return expected


def capture_snapshot(selection, roots, destination, *, content_directory=None, before_publish=None):
    """Read a stable declared closure twice, then publish retained content and descriptor."""
    selection = json.loads(_canonical(selection))
    roots = _roots(tuple(roots))
    destination = _absolute(destination)
    target_store = _absolute(content_directory) if content_directory is not None else destination / "content"
    _nonoverlap({"roots": [{"source": str(root)} for root in roots]}, destination, target_store)
    def capture():
        rows, objects = [], {}
        for source in roots:
            row, payloads = _scan_root(source)
            rows.append(row)
            for digest, payload in payloads.items():
                if digest in objects and objects[digest] != payload:
                    raise ValueError("Resources content identity collision")
                objects[digest] = payload
        return {"format": FORMAT, "selection": selection, "roots": rows}, objects
    manifest, objects = capture()
    repeated, repeated_objects = capture()
    if manifest != repeated or objects != repeated_objects:
        raise ValueError("Resources source changed during snapshot capture")
    return _publish_snapshot(manifest, objects.__getitem__, destination, content_directory=content_directory, before_publish=before_publish)


def copy_snapshot(snapshot, destination, *, content_directory=None):
    """Copy a retained version; source roots, registry construction and tools are irrelevant."""
    manifest = load_snapshot(snapshot)
    read = object_reader(snapshot, manifest)
    return _publish_snapshot(manifest, read, destination, content_directory=content_directory)
