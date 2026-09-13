"""Agent projection preserves edits/recovery without exposing control files."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from .agent_workspace import AgentWorkspace


class AgentWorkspaceTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.worktree = self.root / "worktree"
        self.worktree.mkdir()
        (self.worktree / "kernel.py").write_text("# seed\n")
        (self.worktree / "README.md").write_text("task")
        (self.worktree / "scratch").mkdir()
        (self.worktree / "scratch/old.txt").write_text("diagnostics")
        for name in (".git", ".orchestrator_mode.json", "gpu-wiki", ".atrex_long_horizon"):
            (self.worktree / name).write_text("private")
        memory = self.worktree / "memory"
        memory.mkdir()
        (memory / "v0.json").write_text("{}")
        (memory / "long_horizon_e0001.json").write_text("private audit")
        self.view = AgentWorkspace(self.worktree, self.root / "private")
        self.view.prepare()

    def test_projection_has_no_private_names_and_preserves_atomic_edits(self) -> None:
        for name in (".git", ".orchestrator_mode.json", "gpu-wiki", ".atrex_long_horizon"):
            self.assertFalse((self.view.root / name).exists())
        self.assertEqual([p.name for p in (self.view.root / "memory").iterdir()], ["v0.json"])
        temporary = self.view.root / "new.py"
        temporary.write_text("# candidate\n")
        temporary.replace(self.view.root / "kernel.py")
        (self.view.root / "scratch/request.json").write_text('{"query":"hello"}')
        self.view.publish()
        self.assertEqual((self.worktree / "kernel.py").read_text(), "# candidate\n")
        self.assertEqual(json.loads((self.worktree / "scratch/request.json").read_text()), {"query": "hello"})
        self.assertEqual((self.worktree / ".git").read_text(), "private")

    def test_crash_resume_retains_draft_but_trusted_reset_takes_precedence(self) -> None:
        kernel = self.view.root / "kernel.py"
        kernel.write_text("# unfinished\n")
        self.view.prepare()  # Process died without publishing.
        self.assertEqual(kernel.read_text(), "# unfinished\n")
        (self.worktree / "kernel.py").write_text("# supervisor reset\n")
        self.view.prepare()
        self.assertEqual(kernel.read_text(), "# supervisor reset\n")
        self.assertEqual((self.view.root / "scratch/old.txt").read_text(), "diagnostics")

    def test_publication_never_imports_forged_control_files(self) -> None:
        for name in (".git", ".orchestrator_mode.json", "gpu-wiki", ".atrex_long_horizon"):
            (self.view.root / name).write_text("forged")
        (self.view.root / "README.md").write_text("forged task")
        self.view.publish()
        for name in (".git", ".orchestrator_mode.json", "gpu-wiki", ".atrex_long_horizon"):
            self.assertEqual((self.worktree / name).read_text(), "private")
        self.assertEqual((self.worktree / "README.md").read_text(), "task")

    def test_symlink_outputs_cannot_read_or_overwrite_private_files(self) -> None:
        kernel = self.view.root / "kernel.py"
        kernel.unlink()
        kernel.symlink_to(self.worktree / ".git")
        with self.assertRaises((OSError, ValueError)):
            self.view.publish()
        self.assertEqual((self.worktree / "kernel.py").read_text(), "# seed\n")
        kernel.unlink()
        kernel.write_text("# candidate\n")
        (self.view.root / "scratch/leak").symlink_to(self.worktree)
        with self.assertRaisesRegex(ValueError, "symlinks"):
            self.view.publish()
        self.assertEqual((self.worktree / ".git").read_text(), "private")

    def test_review_projects_bounded_inputs_and_publishes_only_report(self) -> None:
        (self.worktree / "review_request.json").write_text('{"framework":"Triton"}')
        candidate = self.worktree / "candidate"
        candidate.mkdir()
        (candidate / "kernel.py").write_text("# review evidence\n")
        view = AgentWorkspace(self.worktree, self.root / "review", role="production-review")
        view.prepare()
        self.assertEqual(view.read_only_paths, ("review_request.json", "candidate"))
        self.assertEqual((view.root / "candidate/kernel.py").read_text(), "# review evidence\n")
        self.assertFalse((view.root / "kernel.py").exists())
        (view.root / "dependency_review.json").write_text('{"verdict":"allow"}')
        (view.root / "candidate/kernel.py").write_text("# forged evidence\n")
        view.publish()
        self.assertEqual((self.worktree / "dependency_review.json").read_text(), '{"verdict":"allow"}')
        self.assertEqual((candidate / "kernel.py").read_text(), "# review evidence\n")

    def test_problem_generation_can_repair_existing_output(self) -> None:
        (self.worktree / "agent_problem.json").write_text('{"draft":1}')
        view = AgentWorkspace(self.worktree, self.root / "generation", role="problem-generation")
        view.prepare()
        self.assertNotIn("agent_problem.json", view.read_only_paths)
        (view.root / "agent_problem.json").write_text('{"draft":2}')
        view.publish()
        view.prepare()
        self.assertEqual((view.root / "agent_problem.json").read_text(), '{"draft":2}')
        self.assertEqual((self.worktree / "agent_problem.json").read_text(), '{"draft":2}')

    def test_workspace_role_is_explicit_and_stable(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown"):
            AgentWorkspace(self.worktree, self.root / "other", role="anything")
        with self.assertRaisesRegex(ValueError, "Cannot change"):
            AgentWorkspace(self.worktree, self.root / "private", role="production-review").prepare()


if __name__ == "__main__":
    unittest.main()
