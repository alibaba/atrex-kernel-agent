from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from orchestrator.workspace_state import (
    head_kernel_is_initial_baseline,
    v0_baseline_commit,
)


def _git(workspace: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=str(workspace), check=True, stdout=subprocess.DEVNULL)


def _commit(workspace: Path, name: str, content: str, message: str) -> str:
    (workspace / name).write_text(content, encoding="utf-8")
    _git(workspace, "add", name)
    _git(workspace, "commit", "-m", message)
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=str(workspace), capture_output=True, text=True, check=True
    ).stdout.strip()


class V0BaselineCommitTests(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory(prefix="atrex-v0-")
        self.workspace = Path(self._directory.name)
        _git(self.workspace, "init")
        _git(self.workspace, "config", "user.email", "test@local")
        _git(self.workspace, "config", "user.name", "test")

    def tearDown(self) -> None:
        self._directory.cleanup()

    def test_root_commit_carrying_the_kernel_is_v0(self) -> None:
        root = _commit(self.workspace, "kernel.py", "# v0\n", "V0: baseline kernel")
        self.assertEqual(v0_baseline_commit(self.workspace), root)
        self.assertTrue(head_kernel_is_initial_baseline(self.workspace))

    def test_setup_commit_before_v0_does_not_shadow_the_kernel(self) -> None:
        _commit(self.workspace, "README.md", "# campaign\n", "Campaign setup: README")
        v0 = _commit(self.workspace, "kernel.py", "# v0\n", "V0: baseline kernel")
        self.assertEqual(v0_baseline_commit(self.workspace), v0)
        self.assertTrue(head_kernel_is_initial_baseline(self.workspace))

    def test_changed_kernel_is_no_longer_the_initial_baseline(self) -> None:
        _commit(self.workspace, "README.md", "# campaign\n", "Campaign setup: README")
        _commit(self.workspace, "kernel.py", "# v0\n", "V0: baseline kernel")
        _commit(self.workspace, "kernel.py", "# optimized\n", "v1: optimized kernel")
        self.assertFalse(head_kernel_is_initial_baseline(self.workspace))

    def test_workspace_without_a_committed_kernel_has_no_v0(self) -> None:
        _commit(self.workspace, "README.md", "# campaign\n", "Campaign setup: README")
        self.assertEqual(v0_baseline_commit(self.workspace), "")
        self.assertFalse(head_kernel_is_initial_baseline(self.workspace))


if __name__ == "__main__":
    unittest.main()
