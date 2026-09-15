"""Promotion receipts are private and recoverable across the Git commit boundary."""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from orchestrator.workspace_runtime import link_runtime

from .audit_recovery import PromotionAuditRecoveryRequired, repair_promotion_audit
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
        commit = self.promote()
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
                verifier = Mock()
                for _ in range(2):
                    with self.assertRaisesRegex(PromotionAuditRecoveryRequired, "audit_unverifiable") as caught:
                        runner._recover_interrupted(self.store, state, verifier=verifier)
                    self.assertIn("restore --file", str(caught.exception))
                    self.assertEqual("acknowledge-missing" in str(caught.exception), text is None)
                self.assertEqual(state.accepted, 0)
                self.assertEqual(state.rejected, 0)
                self.assertEqual(state.attempts, [])
                active = self.store.load_active()
                self.assertEqual(active["phase"], "promoting")
                self.assertEqual(active["promotion_audit"]["status"], "audit_unverifiable")
                self.assertEqual(active["promotion_audit"]["binding"]["promotion_commit"], commit)
                self.assertEqual(self.git("rev-parse", "HEAD"), commit)
                self.assertTrue(self.worktree.path.is_dir())
                verifier.verify.assert_not_called()
                path.write_text(original)

    def test_restore_exact_backup_recovers_without_new_measurement_or_commit(self) -> None:
        commit = self.promote()
        path = promotion_audit_path(self.workspace, 1)
        backup = self.root / "backup.json"
        backup.write_bytes(path.read_bytes())
        path.unlink()
        runner = LongHorizonCampaign(SimpleNamespace(workspace=self.workspace))
        verifier = Mock()
        state = SupervisorState()
        with self.assertRaises(PromotionAuditRecoveryRequired):
            runner._recover_interrupted(self.store, state, verifier=verifier)
        result = repair_promotion_audit(self.workspace, promotion_commit=commit, source=backup)
        self.assertEqual(result["status"], "restored")
        self.assertEqual(path.read_bytes(), backup.read_bytes())
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        for _ in range(2):
            runner._recover_interrupted(self.store, state, verifier=verifier)
        self.assertEqual(state.accepted, 1)
        self.assertEqual(len(state.attempts), 1)
        self.assertEqual(state.attempts[0]["promotion_audit"], {"status": "verified"})
        self.assertIsNone(self.store.load_active())
        self.assertEqual(self.git("rev-parse", "HEAD"), commit)
        verifier.verify.assert_not_called()

    def test_missing_audit_acknowledgement_is_explicit_durable_and_idempotent(self) -> None:
        commit = self.promote()
        path = promotion_audit_path(self.workspace, 1)
        path.unlink()
        receipt = repair_promotion_audit(
            self.workspace, promotion_commit=commit, reason="Backup unavailable; retaining committed Kernel",
        )
        self.assertFalse(path.exists())  # Never fabricate a replacement audit.
        receipt_path = path.with_suffix(".recovery.json")
        self.assertEqual(receipt_path.stat().st_mode & 0o777, 0o600)
        self.assertEqual(receipt["status"], "audit_unverifiable")
        self.assertEqual(receipt["resolution"], "operator_acknowledged_missing")
        self.assertEqual(json.loads(receipt_path.read_text()), receipt)
        self.assertEqual(repair_promotion_audit(
            self.workspace, promotion_commit=commit, reason="Repeated command",
        ), receipt)
        runner = LongHorizonCampaign(SimpleNamespace(workspace=self.workspace))
        verifier = Mock()
        state = SupervisorState()
        runner._recover_interrupted(self.store, state, verifier=verifier)
        # Simulate a crash after save_state and before clear_active using the old checkpoint.
        state = self.store.load_state()
        self.store.save_active({
            "episode": 1, "memory_version": 1, "phase": "promoted", "base_commit": self.base,
            "episode_branch": self.worktree.branch, "mode": "full",
        })
        runner._recover_interrupted(self.store, state, verifier=verifier)
        self.assertEqual(state.accepted, 1)
        self.assertEqual(state.rejected, 0)
        self.assertEqual(len(state.attempts), 1)
        self.assertEqual(state.attempts[0]["promotion_audit"], receipt)
        self.assertEqual(json.loads((self.store.episode_dir(1) / "attempt.json").read_text())["promotion_audit"], receipt)
        self.assertIsNone(self.store.load_active())
        self.assertEqual(self.git("rev-parse", "HEAD"), commit)
        verifier.verify.assert_not_called()

    def test_repair_requires_exact_commit_and_a_reason_for_missing_evidence(self) -> None:
        commit = self.promote()
        path = promotion_audit_path(self.workspace, 1)
        path.unlink()
        for supplied_commit, reason in ((self.base, "reason"), (commit[:12], "reason"), (commit, "  ")):
            with self.subTest(commit=supplied_commit, reason=reason), self.assertRaises(ValueError):
                repair_promotion_audit(self.workspace, promotion_commit=supplied_commit, reason=reason)
        self.assertFalse(path.with_suffix(".recovery.json").exists())
        self.assertFalse(path.exists())
        self.assertEqual(self.git("rev-parse", "HEAD"), commit)

    def test_corrupt_audit_requires_restore_not_acknowledgement(self) -> None:
        commit = self.promote()
        path = promotion_audit_path(self.workspace, 1)
        backup = self.root / "backup.json"
        backup.write_bytes(path.read_bytes())
        path.write_text("corrupt")
        wrong = self.root / "wrong.json"
        # Equivalent JSON with different bytes is not the committed audit.
        wrong.write_text(json.dumps(self.evidence))
        with self.assertRaisesRegex(ValueError, "Only missing audits"):
            repair_promotion_audit(self.workspace, promotion_commit=commit, reason="No backup")
        with self.assertRaisesRegex(RuntimeError, "digest"):
            repair_promotion_audit(self.workspace, promotion_commit=commit, source=wrong)
        self.assertEqual(path.read_text(), "corrupt")
        self.assertFalse(path.with_suffix(".recovery.json").exists())
        repair_promotion_audit(self.workspace, promotion_commit=commit, source=backup)
        self.assertEqual(self.load(), self.evidence)

    def test_acknowledgement_cannot_bypass_missing_or_changed_canonical_files(self) -> None:
        commit = self.promote()
        audit = promotion_audit_path(self.workspace, 1)
        audit.unlink()
        for relative in ("kernel.py", "memory/v1.json"):
            path = self.workspace / relative
            original = path.read_bytes()
            for contents in (None, b"edited"):
                with self.subTest(path=relative, contents=contents):
                    if contents is None:
                        path.unlink()
                    else:
                        path.write_bytes(contents)
                    with self.assertRaises((RuntimeError, ValueError)):
                        repair_promotion_audit(self.workspace, promotion_commit=commit, reason="Missing audit")
                    self.assertFalse(audit.with_suffix(".recovery.json").exists())
                    path.write_bytes(original)

    def test_acknowledgement_cannot_be_reused_for_another_commit(self) -> None:
        commit = self.promote()
        promotion_audit_path(self.workspace, 1).unlink()
        repair_promotion_audit(self.workspace, promotion_commit=commit, reason="Audit lost")
        message = self.git("log", "-1", "--format=%B")
        self.git("commit", "--amend", "-qm", message + "\n\nDifferent commit")
        runner = LongHorizonCampaign(SimpleNamespace(workspace=self.workspace))
        state = SupervisorState()
        with self.assertRaisesRegex(PromotionAuditRecoveryRequired, "receipt does not match"):
            runner._recover_interrupted(self.store, state, verifier=Mock())
        self.assertEqual(state.accepted, 0)
        self.assertIsNotNone(self.store.load_active())

    def test_acknowledged_recovery_rechecks_checkout_and_receipt(self) -> None:
        commit = self.promote()
        path = promotion_audit_path(self.workspace, 1)
        path.unlink()
        repair_promotion_audit(self.workspace, promotion_commit=commit, reason="Audit lost")
        runner = LongHorizonCampaign(SimpleNamespace(workspace=self.workspace))
        state = SupervisorState()
        kernel = self.workspace / "kernel.py"
        original = kernel.read_bytes()
        kernel.write_bytes(b"unexpected edits")
        with self.assertRaisesRegex(PromotionAuditRecoveryRequired, "kernel.py differs"):
            runner._recover_interrupted(self.store, state, verifier=Mock())
        kernel.write_bytes(original)
        path.with_suffix(".recovery.json").write_text("{}")
        with self.assertRaisesRegex(PromotionAuditRecoveryRequired, "receipt does not match"):
            runner._recover_interrupted(self.store, state, verifier=Mock())
        self.assertEqual(state.accepted, 0)

    def test_restore_checks_identity_even_with_matching_digest(self) -> None:
        self.promote()
        path = promotion_audit_path(self.workspace, 1)
        wrong = self.root / "wrong-episode.json"
        wrong.write_text(json.dumps({**self.evidence, "episode": 2}))
        digest = "sha256:" + hashlib.sha256(wrong.read_bytes()).hexdigest()
        # Simulate a malformed promotion whose trailer was itself bound to the wrong audit.
        self.git("commit", "--amend", "-qm", f"episode 1: promote verified long-horizon candidate\n\n{AUDIT_TRAILER}{digest}")
        commit = self.git("rev-parse", "HEAD")
        original = path.read_bytes()
        with self.assertRaisesRegex(RuntimeError, "does not match the recovering Episode"):
            repair_promotion_audit(self.workspace, promotion_commit=commit, source=wrong)
        self.assertEqual(path.read_bytes(), original)

    def test_symlink_audit_is_not_followed_or_overwritten_by_repair(self) -> None:
        commit = self.promote()
        path = promotion_audit_path(self.workspace, 1)
        backup = self.root / "backup.json"
        backup.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(backup)
        for source in (None, backup):
            with self.subTest(source=source), self.assertRaisesRegex(ValueError, "symlink"):
                repair_promotion_audit(self.workspace, promotion_commit=commit, source=source, reason="Reason")
        self.assertTrue(path.is_symlink())
        self.assertEqual(json.loads(backup.read_bytes()), self.evidence)

    def test_cli_repair_and_clean_campaign_stop(self) -> None:
        from orchestrator import optimize

        commit = self.promote()
        path = promotion_audit_path(self.workspace, 1)
        backup = self.root / "backup.json"
        backup.write_bytes(path.read_bytes())
        path.unlink()
        runner = LongHorizonCampaign(SimpleNamespace(workspace=self.workspace))
        with self.assertRaises(PromotionAuditRecoveryRequired) as caught:
            runner._recover_interrupted(self.store, SupervisorState(), verifier=Mock())
        with patch.object(optimize, "_run_main", side_effect=caught.exception), patch("sys.stderr", new_callable=io.StringIO) as stderr:
            self.assertEqual(optimize.main([]), 2)
        self.assertIn("audit_unverifiable", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())
        command = [sys.executable, "-m", "long_horizon.audit_recovery", "--workspace", str(self.workspace),
                   "--promotion-commit", commit, "restore", "--file", str(backup)]
        result = subprocess.run(command, cwd=Path(__file__).resolve().parent.parent, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('"status": "restored"', result.stdout)
        self.assertEqual(path.read_bytes(), backup.read_bytes())

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
