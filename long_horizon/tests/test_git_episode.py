from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from long_horizon.git_episode import EpisodeWorktree, git_head, promote_candidate
from long_horizon.tests.helpers import init_repo, run_git


class EpisodeGitTests(unittest.TestCase):
    def test_valid_candidate_and_squash_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repo = root / "repo"
            base = init_repo(repo)
            episode = EpisodeWorktree.create(repo, 1, base, root=root / "worktrees")
            (episode.path / "kernel.py").write_text("VALUE = 5\n", encoding="utf-8")
            run_git(episode.path, "add", "kernel.py")
            run_git(episode.path, "commit", "-m", "candidate")
            candidate = git_head(episode.path)
            violation, paths = episode.validate_candidate(candidate)
            self.assertEqual(violation, "")
            self.assertEqual(paths, ["kernel.py"])
            promoted = promote_candidate(
                repo,
                base_commit=base,
                candidate_commit=candidate,
                episode=1,
                evidence={"accepted": True},
                memory_version=1,
                memory_record={"version": "v1", "quality_gate": {"result": "PASS"}},
            )
            self.assertNotEqual(promoted, base)
            self.assertEqual((repo / "kernel.py").read_text(encoding="utf-8"), "VALUE = 5\n")
            self.assertEqual(run_git(repo, "rev-list", "--count", f"{base}..HEAD"), "1")
            episode.remove(repo)

    def test_dirty_candidate_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repo = root / "repo"
            base = init_repo(repo)
            episode = EpisodeWorktree.create(repo, 1, base, root=root / "worktrees")
            (episode.path / "kernel.py").write_text("VALUE = 5\n", encoding="utf-8")
            violation, _ = episode.validate_candidate(base)
            self.assertIn("clean worktree", violation)
            episode.remove(repo)

    def test_protected_evaluator_change_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repo = root / "repo"
            base = init_repo(repo)
            episode = EpisodeWorktree.create(repo, 1, base, root=root / "worktrees")
            (episode.path / "test_kernel.py").write_text("# hacked\n", encoding="utf-8")
            run_git(episode.path, "add", "test_kernel.py")
            run_git(episode.path, "commit", "-m", "bad")
            violation, _ = episode.validate_candidate(git_head(episode.path))
            self.assertIn("protected path", violation)
            episode.remove(repo)

    def test_main_plan_and_profile_evidence_are_allowed_with_kernel_change(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repo = root / "repo"
            base = init_repo(repo)
            episode = EpisodeWorktree.create(repo, 1, base, root=root / "worktrees")
            (episode.path / "kernel.py").write_text("VALUE = 5\n", encoding="utf-8")
            (episode.path / "plans").mkdir()
            (episode.path / "plans" / "v1_plan.md").write_text("plan\n", encoding="utf-8")
            (episode.path / "profiles").mkdir()
            (episode.path / "profiles" / "REPORT.md").write_text("profile\n", encoding="utf-8")
            run_git(episode.path, "add", "kernel.py", "plans", "profiles")
            run_git(episode.path, "commit", "-m", "candidate with evidence")
            violation, paths = episode.validate_candidate(git_head(episode.path))
            self.assertEqual(violation, "")
            self.assertEqual(
                paths, ["kernel.py", "plans/v1_plan.md", "profiles/REPORT.md"]
            )
            episode.remove(repo)

    def test_archive_preserves_uncommitted_and_untracked_work(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repo = root / "repo"
            base = init_repo(repo)
            episode = EpisodeWorktree.create(repo, 1, base, root=root / "worktrees")
            tracked = episode.path / "kernel.py"
            tracked.write_text("VALUE = 5\n", encoding="utf-8")
            run_git(episode.path, "add", "kernel.py")
            run_git(episode.path, "commit", "-m", "candidate")
            tracked.write_text("VALUE = 7\n", encoding="utf-8")
            untracked = episode.path / "notes" / "experiment.txt"
            untracked.parent.mkdir()
            untracked.write_text("promising partial result\n", encoding="utf-8")

            archive = episode.archive(root / "archive")

            worktree_patch = (archive / "worktree.patch").read_text(encoding="utf-8")
            self.assertIn("VALUE = 7", worktree_patch)
            self.assertEqual(
                (archive / "worktree_files" / "kernel.py").read_text(encoding="utf-8"),
                "VALUE = 7\n",
            )
            self.assertEqual(
                (archive / "worktree_files" / "notes" / "experiment.txt").read_text(
                    encoding="utf-8"
                ),
                "promising partial result\n",
            )
            metadata = json.loads((archive / "git.json").read_text(encoding="utf-8"))
            self.assertEqual(
                sorted(metadata["dirty_paths"]), ["kernel.py", "notes/experiment.txt"]
            )
            episode.remove(repo)


if __name__ == "__main__":
    unittest.main()
