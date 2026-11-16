"""Filesystem contract checks, including real process death at publication edges.

Run with PYTHONPATH=src python -m unittest discover -s tests -p test_publication.py.
Set ATREX_TEST_NFS_ROOT to an existing NFS mount to repeat the same public
Workspace and publication contracts on that filesystem. No GPU/model is used.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


PAYLOAD = b"immutable resource\n"


def _initialize(root):
    from aka.contracts.content import SourceContent, WorkspaceFile
    from aka.task.candidate_workspace.provider import GitCandidateWorkspace

    root.mkdir(parents=True, exist_ok=True)
    GitCandidateWorkspace().initialize_kernel(
        root,
        SourceContent((WorkspaceFile("kernel.py", text="def kernel():\n    return 1\n"),)),
        runtime_files=(WorkspaceFile("CLAUDE.md", text="constraints\n"),),
    )


def _child(action, root, edge):
    """Fault hooks live only in a disposable subprocess, never production code."""
    from aka.core import publication

    original_link = os.link
    original_unlink = os.unlink
    original_replace = os.replace
    original_fchmod = os.fchmod
    original_fsync = os.fsync
    original_fdopen = os.fdopen
    manifest_replaces = 0

    def die():
        os.kill(os.getpid(), signal.SIGKILL)

    def link(source, destination, *args, **kwargs):
        staged = str(source).endswith(".aka-publish")
        if staged and edge == "before-link":
            die()
        result = original_link(source, destination, *args, **kwargs)
        if staged and edge == "after-link":
            die()
        return result

    def unlink(path, *args, **kwargs):
        staged = str(path).endswith(".aka-publish")
        if staged and edge == "before-unlink":
            die()
        result = original_unlink(path, *args, **kwargs)
        if staged and edge == "after-unlink":
            die()
        return result

    def replace(source, destination, *args, **kwargs):
        nonlocal manifest_replaces
        manifest = Path(destination).name == ".atrex_context.json"
        if manifest:
            manifest_replaces += 1
        selected = (manifest_replaces == 1 and "runtime-manifest" not in edge
                    or manifest_replaces == 2 and "runtime-manifest" in edge)
        if manifest and selected and edge in ("before-manifest-replace", "before-runtime-manifest-replace"):
            die()
        result = original_replace(source, destination, *args, **kwargs)
        if manifest and selected and edge in ("after-manifest-replace", "after-runtime-manifest-replace"):
            die()
        return result

    def fchmod(descriptor, mode):
        if mode == 0o444 and edge == "before-readonly":
            die()
        return original_fchmod(descriptor, mode)

    def fsync(descriptor):
        result = original_fsync(descriptor)
        value = os.fstat(descriptor)
        if stat.S_ISREG(value.st_mode) and stat.S_IMODE(value.st_mode) == 0o444 and edge == "after-data-fsync":
            die()
        return result

    def fdopen(descriptor, mode, *args, **kwargs):
        stream = original_fdopen(descriptor, mode, *args, **kwargs)
        if mode != "wb" or edge != "partial-write":
            return stream

        class PartialWrite:
            def __enter__(self):
                return self

            def __exit__(self, *error):
                stream.close()

            def fileno(self):
                return stream.fileno()

            def write(self, payload):
                stream.write(payload[:5])
                stream.flush()
                die()

        return PartialWrite()

    with patch.object(publication.os, "link", link), patch.object(publication.os, "unlink", unlink), \
            patch.object(publication.os, "replace", replace), patch.object(publication.os, "fchmod", fchmod), \
            patch.object(publication.os, "fsync", fsync), patch.object(publication.os, "fdopen", fdopen):
        if action == "initialize":
            _initialize(root)
        elif action in ("immutable", "immutable-umask"):
            root.mkdir(parents=True, exist_ok=True)
            previous_umask = os.umask(0o777) if action == "immutable-umask" else None
            try:
                publication.publish_immutable(root / "object", PAYLOAD)
            finally:
                if previous_umask is not None:
                    os.umask(previous_umask)
        elif action == "resource":
            from aka.resources.snapshot_store import _write_immutable
            _write_immutable(root / "object", PAYLOAD)
        else:
            raise ValueError(action)


class PublicationContract(unittest.TestCase):
    def setUp(self):
        directory = os.environ.get("ATREX_TEST_NFS_ROOT")
        self.temporary = tempfile.TemporaryDirectory(prefix="aka-publication-test-", dir=directory)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()

    def child(self, action, root, edge=""):
        environment = os.environ.copy()
        return subprocess.run(
            [sys.executable, *(["-I"] if sys.flags.isolated else []),
             str(Path(__file__).resolve()), "--child", action, str(root), edge],
            env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30,
        )

    def assert_success(self, result):
        self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))

    def assert_immutable(self, path, payload=PAYLOAD):
        value = path.lstat()
        self.assertTrue(stat.S_ISREG(value.st_mode))
        self.assertEqual(value.st_nlink, 1)
        self.assertEqual(stat.S_IMODE(value.st_mode), 0o444)
        self.assertEqual(path.read_bytes(), payload)

    def test_sigkill_immutable_publication_edges(self):
        for edge in ("partial-write", "before-readonly", "after-data-fsync", "before-link", "after-link", "before-unlink", "after-unlink"):
            with self.subTest(edge=edge):
                root = self.root / edge
                killed = self.child("immutable", root, edge)
                self.assertEqual(killed.returncode, -signal.SIGKILL, killed.stderr.decode(errors="replace"))
                self.assert_success(self.child("immutable", root))
                self.assert_immutable(root / "object")
                self.assertFalse((root / ".object.aka-publish").exists())

    def test_public_initialize_recovers_sigkill(self):
        for edge in ("after-link", "before-unlink", "after-unlink", "before-manifest-replace",
                     "after-manifest-replace", "before-runtime-manifest-replace", "after-runtime-manifest-replace"):
            with self.subTest(edge=edge):
                workspace = self.root / edge
                killed = self.child("initialize", workspace, edge)
                self.assertEqual(killed.returncode, -signal.SIGKILL, killed.stderr.decode(errors="replace"))
                if "runtime-manifest" in edge:
                    # The second replacement happens only after retained bytes
                    # and the complete runtime projection are already installed.
                    import hashlib
                    self.assertEqual((workspace / "CLAUDE.md").read_bytes(), b"constraints\n")
                    digest = hashlib.sha256(b"constraints\n").hexdigest()
                    self.assert_immutable(workspace / ".git" / "aka-context" / digest, b"constraints\n")
                    pending = json.loads((workspace / ".atrex_context.json").read_bytes())
                    self.assertEqual("CLAUDE.md" in pending["context"], edge.startswith("after-"))
                self.assert_success(self.child("initialize", workspace))
                self.assert_success(self.child("initialize", workspace))
                manifest = json.loads((workspace / ".atrex_context.json").read_bytes())
                item = manifest["context"]["CLAUDE.md"]
                self.assert_immutable(workspace / ".git" / "aka-context" / item["sha256"], b"constraints\n")
                self.assertEqual((workspace / "kernel.py").read_text(), "def kernel():\n    return 1\n")

    def test_resources_use_recoverable_publication(self):
        root = self.root / "resources"
        killed = self.child("resource", root, "after-link")
        self.assertEqual(killed.returncode, -signal.SIGKILL, killed.stderr.decode(errors="replace"))
        self.assert_success(self.child("resource", root))
        self.assert_immutable(root / "object")

    def test_same_content_is_idempotent_collision_is_preserved(self):
        from aka.core.publication import publish_immutable
        path = self.root / "object"
        publish_immutable(path, PAYLOAD)
        inode = path.stat().st_ino
        publish_immutable(path, PAYLOAD)
        self.assertEqual(path.stat().st_ino, inode)
        with self.assertRaises((ValueError, OSError)):
            publish_immutable(path, b"different\n")
        self.assert_immutable(path)

    def test_unknown_alias_is_never_repaired(self):
        from aka.core.publication import publish_immutable
        path = self.root / "object"
        publish_immutable(path, PAYLOAD)
        alias = self.root / ".retain-legacy-or-unknown"
        os.link(path, alias)
        with self.assertRaises((ValueError, OSError)):
            publish_immutable(path, PAYLOAD)
        self.assertTrue(alias.exists())
        self.assertEqual(path.stat().st_nlink, 2)

    def test_extra_alias_even_with_known_stage_is_rejected(self):
        from aka.core.publication import publish_immutable
        path = self.root / "object"
        publish_immutable(path, PAYLOAD)
        stage = self.root / ".object.aka-publish"
        unknown = self.root / "unknown"
        os.link(path, stage)
        os.link(path, unknown)
        with self.assertRaises((ValueError, OSError)):
            publish_immutable(path, PAYLOAD)
        self.assertTrue(stage.exists())
        self.assertTrue(unknown.exists())
        self.assertEqual(path.stat().st_nlink, 3)

    def test_symlinks_and_unsafe_modes_are_rejected(self):
        from aka.core.publication import publish_immutable
        target = self.root / "outside"
        target.write_bytes(PAYLOAD)
        for name, mode in (("symlink", None), ("writable", 0o644), ("special", 0o4644)):
            with self.subTest(name=name):
                path = self.root / name
                if mode is None:
                    path.symlink_to(target)
                else:
                    path.write_bytes(PAYLOAD)
                    path.chmod(mode)
                with self.assertRaises((ValueError, OSError)):
                    publish_immutable(path, PAYLOAD)
        parent_link = self.root / "parent-link"
        parent_link.symlink_to(self.root, target_is_directory=True)
        with self.assertRaises((ValueError, OSError)):
            publish_immutable(parent_link / "nested", PAYLOAD)
        self.assertFalse((self.root / "nested").exists())

    def test_partial_known_stage_is_reconciled_only_for_expected_prefix(self):
        from aka.core.publication import publish_immutable
        for name, payload, should_pass in (("prefix", PAYLOAD[:5], True), ("foreign", b"unrelated", False)):
            with self.subTest(name=name):
                path = self.root / name
                stage = self.root / f".{name}.aka-publish"
                stage.write_bytes(payload)
                stage.chmod(0o600)
                if should_pass:
                    publish_immutable(path, PAYLOAD)
                    self.assert_immutable(path)
                else:
                    with self.assertRaises((ValueError, OSError)):
                        publish_immutable(path, PAYLOAD)
                    self.assertEqual(stage.read_bytes(), payload)
                    self.assertFalse(path.exists())

    def test_manifest_replacement_reconciles_only_expected_partial_transaction(self):
        from aka.core.publication import replace_bytes
        path = self.root / ".atrex_context.json"
        temporary = self.root / ".atrex_context.json.tmp"
        previous, expected = b'{"previous":true}\n', b'{"next":true}\n'
        path.write_bytes(previous)
        temporary.write_bytes(expected[:5])
        temporary.chmod(0o600)
        replace_bytes(path, expected, mode=0o644, temporary_name=temporary.name)
        self.assertEqual(path.read_bytes(), expected)
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o644)
        self.assertEqual(path.stat().st_nlink, 1)
        temporary.write_bytes(b"unknown transaction")
        temporary.chmod(0o600)
        with self.assertRaises((ValueError, OSError)):
            replace_bytes(path, b"another request", mode=0o644, temporary_name=temporary.name)
        self.assertEqual(path.read_bytes(), expected)
        self.assertEqual(temporary.read_bytes(), b"unknown transaction")

    def test_stage_is_closed_before_link_and_unlink(self):
        """A live-stage FD would let NFS silly rename retain the second link."""
        from aka.core import publication
        opened = []
        original_open, original_link, original_unlink = os.open, os.link, os.unlink

        def open_file(*args, **kwargs):
            descriptor = original_open(*args, **kwargs)
            opened.append(descriptor)
            return descriptor

        def assert_closed(name, directory=None):
            value = os.stat(name, dir_fd=directory, follow_symlinks=False)
            for descriptor in opened:
                try:
                    current = os.fstat(descriptor)
                except OSError:
                    continue
                self.assertNotEqual((value.st_dev, value.st_ino), (current.st_dev, current.st_ino),
                                    "staging inode still has an open descriptor at NFS publication")

        def link(source, destination, *args, **kwargs):
            if str(source).endswith(".aka-publish"):
                assert_closed(source, kwargs.get("src_dir_fd"))
            return original_link(source, destination, *args, **kwargs)

        def unlink(path, *args, **kwargs):
            if str(path).endswith(".aka-publish"):
                assert_closed(path, kwargs.get("dir_fd"))
            return original_unlink(path, *args, **kwargs)

        with patch.object(publication.os, "open", open_file), patch.object(publication.os, "link", link), \
                patch.object(publication.os, "unlink", unlink):
            publication.publish_immutable(self.root / "object", PAYLOAD)
        self.assert_immutable(self.root / "object")

    def test_reserved_lock_cannot_be_a_publication_destination(self):
        from aka.core.publication import publish_immutable, replace_bytes
        path = self.root / ".aka-publication.lock"
        for operation in (publish_immutable, replace_bytes):
            with self.subTest(operation=operation.__name__):
                with self.assertRaises(ValueError):
                    operation(path, PAYLOAD)
                self.assertFalse(path.exists())

    def test_existing_lock_objects_are_not_normalized(self):
        from aka.core.publication import publish_immutable
        for kind in ("symlink", "hardlink", "wrong-mode", "directory", "nonempty"):
            with self.subTest(kind=kind):
                root = self.root / kind
                root.mkdir()
                lock = root / ".aka-publication.lock"
                other = root / "other"
                if kind == "directory":
                    lock.mkdir()
                elif kind == "symlink":
                    other.write_bytes(b"private data")
                    lock.symlink_to(other)
                else:
                    lock.write_bytes(b"unknown owner" if kind == "nonempty" else b"")
                    lock.chmod(0o644 if kind == "wrong-mode" else 0o600)
                    if kind == "hardlink":
                        os.link(lock, other)
                before = lock.lstat()
                with self.assertRaises((ValueError, OSError)):
                    publish_immutable(root / "object", PAYLOAD)
                after = lock.lstat()
                self.assertEqual((before.st_dev, before.st_ino, before.st_mode, before.st_nlink),
                                 (after.st_dev, after.st_ino, after.st_mode, after.st_nlink))
                self.assertFalse((root / "object").exists())
                if kind == "nonempty":
                    self.assertEqual(lock.read_bytes(), b"unknown owner")

    def test_restrictive_umask_creates_ordinary_lock_and_immutable_object(self):
        from aka.core.publication import publish_immutable
        previous = os.umask(0o777)
        try:
            publish_immutable(self.root / "object", PAYLOAD)
        finally:
            os.umask(previous)
        self.assert_immutable(self.root / "object")
        lock = self.root / ".aka-publication.lock"
        self.assertEqual(stat.S_IMODE(lock.stat().st_mode), 0o600)
        self.assertEqual(lock.stat().st_nlink, 1)
        self.assertEqual(lock.read_bytes(), b"")

    def test_partial_sigkill_under_restrictive_umask_is_recoverable(self):
        # The directory exists before the child changes its umask, so this
        # isolates staging permissions from directory-creation policy.
        root = self.root / "restrictive-partial"
        root.mkdir(mode=0o700)
        killed = self.child("immutable-umask", root, "partial-write")
        self.assertEqual(killed.returncode, -signal.SIGKILL, killed.stderr.decode(errors="replace"))
        stage = root / ".object.aka-publish"
        self.assertEqual(stage.read_bytes(), PAYLOAD[:5])
        self.assertEqual(stat.S_IMODE(stage.stat().st_mode), 0o600)
        self.assert_success(self.child("immutable", root))
        self.assert_immutable(root / "object")
        self.assertFalse(stage.exists())

    def test_same_content_fileexists_race_preserves_the_other_inode(self):
        from aka.core import publication
        original_link = os.link
        raced = []

        def competing_creator(source, destination, *args, **kwargs):
            if not str(source).endswith(".aka-publish"):
                return original_link(source, destination, *args, **kwargs)
            descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                                 0o600, dir_fd=kwargs.get("dst_dir_fd"))
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(PAYLOAD)
                stream.flush()
                os.fchmod(stream.fileno(), 0o444)
                os.fsync(stream.fileno())
                raced.append(os.fstat(stream.fileno()).st_ino)
            raise FileExistsError("concurrent identical creator")

        with patch.object(publication.os, "link", competing_creator):
            publication.publish_immutable(self.root / "object", PAYLOAD)
        self.assertEqual((self.root / "object").stat().st_ino, raced[0])
        self.assert_immutable(self.root / "object")
        self.assertFalse((self.root / ".object.aka-publish").exists())

    def test_concurrent_same_content_writers(self):
        environment = os.environ.copy()
        root = self.root / "parallel"
        root.mkdir()
        processes = [subprocess.Popen(
            [sys.executable, *(["-I"] if sys.flags.isolated else []),
             str(Path(__file__).resolve()), "--child", "immutable", str(root), ""],
            env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        ) for _ in range(4)]
        try:
            for process in processes:
                _, error = process.communicate(timeout=30)
                self.assertEqual(process.returncode, 0, error.decode(errors="replace"))
        finally:
            for process in processes:
                if process.poll() is None:
                    process.kill()
                process.communicate(timeout=30)
        self.assert_immutable(root / "object")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        _child(sys.argv[2], Path(sys.argv[3]), sys.argv[4])
    else:
        unittest.main()
