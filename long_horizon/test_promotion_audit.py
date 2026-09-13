"""Promotion receipts are private and recoverable across the Git commit boundary."""

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from orchestrator.workspace_runtime import link_runtime

from .campaign import LongHorizonCampaign
from .git_episode import EpisodeWorktree, promote_candidate
from .models import SupervisorState
from .promotion_audit import (
    AUDIT_TRAILER,
    committed_promotion_audit,
    promotion_audit_path,
)
from .store import CampaignStore


class SimulatedCrash(BaseException):
    pass


class PromotionAuditTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.workspace = self.root / "campaign"
        self.workspace.mkdir()
        self.git("init", "-qb", "main")
        self.git("config", "user.name", "Test")
        self.git("config", "user.email", "test@example.test")
        (self.workspace / "kernel.py").write_text("# baseline\n")
        (self.workspace / "memory").mkdir()
        (self.workspace / "memory/v0.json").write_text('{"version":"v0"}')
        self.git("add", ".")
        self.git("commit", "-qm", "baseline")
        self.base = self.git("rev-parse", "HEAD")
        self.worktree = EpisodeWorktree.create(self.workspace, 1, self.base, self.root / "episodes")
        source = b"# candidate\n"
        (self.worktree.path / "kernel.py").write_bytes(source)
        self.candidate = self.worktree.commit_candidate(source)
        self.evidence = {
            "episode": 1,
            "version": 1,
            "accepted": True,
            "base_commit": self.base,
            "candidate_commit": self.candidate,
            "episode_branch": self.worktree.branch,
            "journal": {"experiments": [{"name": "layout change"}]},
        }
        self.memory = {"version": "v1", "performance": {"latency_us": 8.0}}
        self.store = CampaignStore(self.workspace)
        self.store.save_active(
            {
                "episode": 1,
                "memory_version": 1,
                "phase": "promoting",
                "base_commit": self.base,
                "episode_branch": self.worktree.branch,
                "worktree": str(self.worktree.path),
                "mode": "full",
            }
        )

    def git(self, *args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=self.workspace, check=True, capture_output=True, text=True
        ).stdout.strip()

    def promote(self) -> str:
        return promote_candidate(
            self.workspace,
            base_commit=self.base,
            candidate_commit=self.candidate,
            episode=1,
            evidence=self.evidence,
            memory_version=1,
            memory_record=self.memory,
        )

    def load(self):
        return committed_promotion_audit(
            self.workspace,
            episode=1,
            base_commit=self.base,
            branch=self.worktree.branch,
            version=1,
        )

    def test_promotion_commits_only_kernel_and_canonical_report(self) -> None:
        commit = self.promote()
        path = promotion_audit_path(self.workspace, 1)
        self.assertFalse(path.is_relative_to(self.workspace))
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        self.assertEqual(json.loads(path.read_text()), self.evidence)
        self.assertEqual(self.load(), self.evidence)
        self.assertIn(AUDIT_TRAILER, self.git("log", "-1", "--format=%B"))
        self.assertEqual(
            set(self.git("diff", "--name-only", self.base, commit).splitlines()),
            {"kernel.py", "memory/v1.json"},
        )
        self.assertFalse((self.workspace / "memory/long_horizon_e0001.json").exists())
        later = EpisodeWorktree.create(self.workspace, 2, commit, self.root / "episodes")
        self.assertFalse((later.path / "memory/long_horizon_e0001.json").exists())
        self.assertEqual(promotion_audit_path(later.path, 1), path)

    def test_post_commit_crash_recovers_once_without_another_evaluation(self) -> None:
        real_run = subprocess.run

        def crash(command, *args, **kwargs):
            result = real_run(command, *args, **kwargs)
            if "--only" in command and "commit" in command:
                raise SimulatedCrash()
            return result

        with patch("long_horizon.git_episode.subprocess.run", side_effect=crash):
            with self.assertRaises(SimulatedCrash):
                self.promote()
        state = SupervisorState()
        runner = LongHorizonCampaign(SimpleNamespace(workspace=self.workspace))
        verifier = Mock()
        self.assertIsNone(runner._recover_interrupted(self.store, state, verifier=verifier))
        self.assertEqual(state.accepted, 1)
        self.assertEqual(len(state.attempts), 1)
        self.assertTrue(state.attempts[0]["accepted"])
        self.assertIsNone(self.store.load_active())
        self.assertIsNone(runner._recover_interrupted(self.store, state, verifier=verifier))
        self.assertEqual(state.accepted, 1)
        verifier.verify.assert_not_called()

    def test_pre_commit_crash_does_not_turn_private_receipt_into_promotion(self) -> None:
        real_run = subprocess.run

        def crash(command, *args, **kwargs):
            if "--only" in command and "commit" in command:
                raise SimulatedCrash()
            return real_run(command, *args, **kwargs)

        with patch("long_horizon.git_episode.subprocess.run", side_effect=crash):
            with self.assertRaises(SimulatedCrash):
                self.promote()
        self.assertTrue(promotion_audit_path(self.workspace, 1).exists())
        self.assertIsNone(self.load())
        state = SupervisorState()
        runner = LongHorizonCampaign(SimpleNamespace(workspace=self.workspace))
        with patch.object(runner, "_recover_completed_handoff", return_value=False):
            resumed = runner._recover_interrupted(self.store, state, verifier=Mock())
        self.assertEqual(resumed[0].path, self.worktree.path)
        self.assertEqual(self.git("rev-parse", "HEAD"), self.base)
        self.assertEqual((self.workspace / "kernel.py").read_text(), "# baseline\n")
        self.assertEqual(state.accepted, 0)

    def test_missing_or_corrupt_audit_is_not_silently_treated_as_rejection(self) -> None:
        self.promote()
        path = promotion_audit_path(self.workspace, 1)
        original = path.read_text()
        for text in (None, "{}", "not json"):
            with self.subTest(text=text):
                if text is None:
                    path.unlink()
                else:
                    path.write_text(text)
                state = SupervisorState()
                runner = LongHorizonCampaign(SimpleNamespace(workspace=self.workspace))
                with self.assertRaisesRegex(RuntimeError, "private promotion audit"):
                    runner._recover_interrupted(self.store, state, verifier=Mock())
                self.assertEqual(state.accepted, 0)
                self.assertIsNotNone(self.store.load_active())
                path.write_text(original)

    def test_old_fast_checkpoint_resumes_ordinary_episode_without_losing_files(self) -> None:
        active = self.store.load_active()
        active.update(mode="fast", fast_trials=5, phase="checking_fast_evaluator")
        self.store.save_active(active)
        scratch = self.worktree.path / "scratch"
        scratch.mkdir(exist_ok=True)
        (scratch / "probe.py").write_text("# retained diagnostic\n")
        state = SupervisorState(attempts=[{"episode": 0, "mode": "fast"}])
        runner = LongHorizonCampaign(SimpleNamespace(workspace=self.workspace))
        with patch.object(runner, "_recover_completed_handoff", return_value=False):
            resumed = runner._recover_interrupted(self.store, state, verifier=Mock())
        self.assertEqual(resumed[0].path, self.worktree.path)
        self.assertEqual(resumed[1]["mode"], "full")
        self.assertNotIn("fast_trials", resumed[1])
        self.assertEqual(resumed[1]["resumed_from_phase"], "verifying")
        self.assertEqual((scratch / "probe.py").read_text(), "# retained diagnostic\n")
        self.assertEqual((self.worktree.path / "kernel.py").read_text(), "# candidate\n")
        self.assertEqual(state.attempts, [{"episode": 0, "mode": "fast"}])

    def test_already_committed_old_fast_promotion_keeps_historical_mode(self) -> None:
        active = self.store.load_active()
        active.update(mode="fast", fast_trials=5)
        self.store.save_active(active)
        self.promote()
        state = SupervisorState()
        runner = LongHorizonCampaign(SimpleNamespace(workspace=self.workspace))
        verifier = Mock()
        self.assertIsNone(runner._recover_interrupted(self.store, state, verifier=verifier))
        self.assertTrue(state.attempts[0]["accepted"])
        self.assertEqual(state.attempts[0]["mode"], "fast")
        verifier.verify.assert_not_called()

    def test_committed_legacy_audit_is_copied_privately_and_recovers(self) -> None:
        self.git("merge", "--squash", "--no-commit", self.candidate)
        legacy = self.workspace / "memory/long_horizon_e0001.json"
        legacy.write_text(json.dumps(self.evidence))
        (self.workspace / "memory/v1.json").write_text(json.dumps(self.memory))
        self.git("add", "memory")
        self.git("commit", "-qm", "episode 1: promote verified long-horizon candidate")
        original_head = self.git("rev-parse", "HEAD")
        legacy.write_text("untrusted checkout edit")
        link_runtime(self.workspace)
        self.assertEqual(
            json.loads(promotion_audit_path(self.workspace, 1).read_text()), self.evidence
        )
        self.assertEqual(self.load(), self.evidence)
        self.assertEqual(self.git("rev-parse", "HEAD"), original_head)
        self.assertEqual(legacy.read_text(), "untrusted checkout edit")


if __name__ == "__main__":
    unittest.main()
