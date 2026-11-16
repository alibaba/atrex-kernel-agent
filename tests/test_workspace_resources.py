"""Workspace keeps strict identities while installed wheel inputs may share cache inodes."""
import base64
from contextlib import contextmanager
import hashlib
from importlib.metadata import FileHash, PackageNotFoundError, PackagePath
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from aka.contracts.content import SourceContent, WorkspaceFile
from aka.resources import workspace as resources
from aka.task.candidate_workspace.provider import GitCandidateWorkspace


def record(name, content):
    item = PackagePath(name)
    digest = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(b"=").decode()
    item.hash = FileHash("sha256=" + digest)
    item.size = len(content)
    return item


class IndependentResources(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.source = self.root / "CLAUDE.md"
        self.source.write_bytes(b"constraints\n")
        self.source.chmod(0o640)
        self.alias = self.root / "cache-alias"
        os.link(self.source, self.alias)
        self.entry = record("plugin/assets/CLAUDE.md", self.source.read_bytes())
        self.item = WorkspaceFile("CLAUDE.md", source=self.source)

    def owned(self):
        return patch.object(resources, "_owned_assets", return_value={self.source: self.entry})

    def test_snapshot_preserves_bytes_mode_and_cache_and_is_cleaned(self):
        before = self.source.stat()
        with self.owned(), resources.independent_resources((self.item,)) as ((item,),):
            target = item.source
            self.assertNotEqual(target, self.source)
            self.assertEqual(target.read_bytes(), self.source.read_bytes())
            self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o640)
            self.assertEqual(target.stat().st_nlink, 1)
            self.assertFalse(os.path.samefile(target, self.source))
        self.assertFalse(target.exists())
        self.assertEqual(self.source.stat(), before)
        self.assertEqual(self.alias.read_bytes(), b"constraints\n")

    def test_undeclared_alias_is_unchanged(self):
        with patch.object(resources, "_owned_assets", return_value={}), resources.independent_resources((self.item,)) as groups:
            self.assertIs(groups[0][0], self.item)
        with patch.object(resources, "_owned_assets", return_value={}), self.assertRaisesRegex(ValueError, "unlinked"):
            self.initialize()

    def test_text_and_unlinked_files_do_not_allocate_staging(self):
        ordinary = self.root / "ordinary"
        ordinary.write_text("ordinary")
        items = (WorkspaceFile("text", text="hello"), WorkspaceFile("ordinary", source=ordinary))
        with patch.object(resources.tempfile, "TemporaryDirectory", side_effect=AssertionError("unexpected staging")):
            with resources.independent_resources(items) as groups:
                self.assertEqual(groups, (items,))

    def test_cleanup_on_consumer_exception(self):
        with self.assertRaisesRegex(RuntimeError, "consumer"):
            with self.owned(), resources.independent_resources((self.item,)) as ((item,),):
                target = item.source
                raise RuntimeError("consumer")
        self.assertFalse(target.exists())
        self.assertTrue(self.source.exists())

    def test_hash_and_size_mismatch_rejected(self):
        self.entry.hash = FileHash("sha256=wrong")
        with self.owned(), self.assertRaisesRegex(ValueError, "wheel RECORD"):
            with resources.independent_resources((self.item,)):
                pass
        self.entry = record("plugin/assets/CLAUDE.md", self.source.read_bytes())
        self.entry.size += 1
        with self.owned(), self.assertRaisesRegex(ValueError, "wheel RECORD"):
            with resources.independent_resources((self.item,)):
                pass

    def test_missing_hash_special_mode_and_oversize_rejected(self):
        self.entry.hash = None
        with self.owned(), self.assertRaisesRegex(ValueError, "wheel RECORD"):
            with resources.independent_resources((self.item,)):
                pass
        self.source.chmod(0o4644)
        with self.owned(), self.assertRaisesRegex(ValueError, "invalid installed"):
            with resources.independent_resources((self.item,)):
                pass
        self.source.chmod(0o644)
        with self.owned(), patch.object(resources, "_MAX_BYTES", 2), self.assertRaisesRegex(ValueError, "invalid installed"):
            with resources.independent_resources((self.item,)):
                pass

    def test_symlink_and_symlink_parent_rejected(self):
        link = self.root / "link"
        link.symlink_to(self.source)
        with self.assertRaises(OSError):
            resources._read_asset(link, self.entry)
        directory = self.root / "directory"
        directory.mkdir()
        (directory / "asset").write_bytes(self.source.read_bytes())
        parent = self.root / "parent-link"
        parent.symlink_to(directory, target_is_directory=True)
        with self.assertRaises(OSError):
            resources._read_asset(parent / "asset", self.entry)

    def test_replacement_during_open_rejected(self):
        original = resources._open_regular
        @contextmanager
        def swapped(path):
            replacement = self.root / "replacement"
            replacement.write_bytes(self.source.read_bytes())
            os.replace(replacement, self.source)
            with original(path) as stream:
                yield stream
        with patch.object(resources, "_open_regular", swapped), self.assertRaisesRegex(ValueError, "invalid installed"):
            resources._read_asset(self.source, self.entry)

    def test_explicit_distribution_prefix_extends_first_party_policy(self):
        package = self.root / "plugin" / "assets"
        package.mkdir(parents=True)
        source = package / "CLAUDE.md"
        source.write_bytes(self.source.read_bytes())
        os.link(source, self.root / "plugin-cache-alias")
        other = record("plugin/private.txt", b"private")
        owner = SimpleNamespace(files=(self.entry, other), locate_file=lambda entry: self.root / entry)
        def distribution(name):
            if name != "my-plugin":
                raise PackageNotFoundError(name)
            return owner
        with patch.object(resources, "distribution", distribution):
            selected = resources._owned_assets({"my-plugin": ("plugin/assets/",)})
            self.assertEqual(set(selected), {source})
            with resources.independent_resources((WorkspaceFile("CLAUDE.md", source=source),), owners={"my-plugin": ("plugin/assets/",)}) as ((item,),):
                self.assertEqual(item.source.stat().st_nlink, 1)
                self.assertFalse(os.path.samefile(source, item.source))

    def test_invalid_owner_prefixes_rejected(self):
        for prefix in ("/absolute/", "../escape/", "plugin/../assets/", "plugin//assets/", "plugin/assets", "./", "plugin\\assets/"):
            with self.subTest(prefix=prefix), self.assertRaises(ValueError):
                resources._owned_assets({"my-plugin": (prefix,)})

    def initialize(self, workspace=None):
        workspace = self.root / "workspace" if workspace is None else workspace
        workspace.mkdir(exist_ok=True)
        content = SourceContent((WorkspaceFile("kernel.py", text="def kernel(): return 1\n"),))
        return GitCandidateWorkspace().initialize_kernel(workspace, content, runtime_files=(self.item,))

    def test_public_native_initialization_and_retry(self):
        with self.owned():
            self.initialize()
            self.initialize()
        target = self.root / "workspace" / "CLAUDE.md"
        self.assertEqual(target.read_bytes(), self.source.read_bytes())
        self.assertEqual(target.stat().st_nlink, 1)
        self.assertFalse(os.path.samefile(target, self.source))

    def test_existing_workspace_alias_remains_rejected(self):
        workspace = self.root / "workspace"
        workspace.mkdir()
        os.link(self.source, workspace / "CLAUDE.md")
        with self.owned(), self.assertRaisesRegex(ValueError, "unlinked"):
            self.initialize(workspace)
        self.assertEqual((workspace / "CLAUDE.md").stat().st_nlink, 3)

    def test_public_sol_initialization_and_retry(self):
        content = SourceContent((WorkspaceFile("kernel.py", text="def kernel(): return 1\n"),), readme="SOL")
        workspace = self.root / "sol"
        workspace.mkdir()
        provider = GitCandidateWorkspace()
        options = dict(problem_files=(), harness=WorkspaceFile("test_kernel.py", source=self.source),
                       runtime_files=(self.item, WorkspaceFile("profile_driver.py", source=self.source)))
        with self.owned():
            revision = provider.prepare_sol(workspace, content, **options)
            self.assertEqual(provider.prepare_sol(workspace, content, **options), revision)
        for name in ("CLAUDE.md", "profile_driver.py", "test_kernel.py"):
            self.assertEqual((workspace / name).stat().st_nlink, 1)
            self.assertEqual((workspace / name).read_bytes(), self.source.read_bytes())


if __name__ == "__main__":
    unittest.main()
