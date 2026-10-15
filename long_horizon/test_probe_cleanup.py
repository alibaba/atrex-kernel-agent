"""Offline tests for diagnostic cleanup at an incumbent episode boundary."""
from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from long_horizon import campaign as campaign_api
from long_horizon.git_episode import clean_probe_artifacts, working_changes


class ProbeCleanupTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.workspace = self.root / "campaign"
        self.workspace.mkdir()
        self.git("init", "-q")
        self.git("config", "user.name", "Offline Test")
        self.git("config", "user.email", "test@example.invalid")
        self.git("config", "commit.gpgsign", "false")
        for name in ("kernel.py", "test_kernel.py", "input.py", "solution.json"):
            (self.workspace / name).write_text("committed baseline\n")
        self.git("add", "--all")
        self.git("commit", "-qm", "baseline")
        self.head = self.git("rev-parse", "HEAD")

    def git(self, *args):
        return subprocess.run(["git", *args], cwd=self.workspace, check=True,
                              text=True, capture_output=True).stdout

    def write(self, name, content="diagnostic\n"):
        path = self.workspace / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return path

    def test_removes_nested_probes_and_is_repeatable(self):
        self.write(".probe/probe_env.py")
        self.write("probes/v1/recovered/kernel_backup.py")
        self.write("probes/v1/README_evidence.md")
        self.assertTrue(clean_probe_artifacts(self.workspace))
        self.assertEqual(working_changes(self.workspace), [])
        self.assertEqual(self.git("rev-parse", "HEAD"), self.head)
        self.assertEqual(clean_probe_artifacts(self.workspace), "")
        self.write("probes/v2/another-diagnostic.log")
        self.assertTrue(clean_probe_artifacts(self.workspace))
        self.assertEqual(working_changes(self.workspace), [])

    def test_preserves_tracked_and_staged_files_and_other_untracked_files(self):
        tracked = self.write("probes/tracked.py", "tracked original\n")
        self.git("add", "probes/tracked.py")
        self.git("commit", "-qm", "tracked helper")
        tracked.write_text("tracked modification\n")
        staged = self.write(".probe/staged.py")
        self.git("add", ".probe/staged.py")
        self.write("probes/temporary.log")
        candidate = self.write("candidate.py", "untracked candidate\n")
        harness = self.write("test_kernel.py", "modified harness\n")
        clean_probe_artifacts(self.workspace)
        self.assertEqual(tracked.read_text(), "tracked modification\n")
        self.assertTrue(staged.exists())
        self.assertEqual(candidate.read_text(), "untracked candidate\n")
        self.assertEqual(harness.read_text(), "modified harness\n")
        self.assertEqual(set(working_changes(self.workspace)),
                         {".probe/staged.py", "probes/tracked.py", "candidate.py", "test_kernel.py"})

    def test_symlinks_do_not_delete_their_targets(self):
        outside = self.root / "outside"
        outside.mkdir()
        keep = outside / "important.py"
        keep.write_text("keep\n")
        (self.workspace / ".probe").symlink_to(outside, target_is_directory=True)
        probes = self.workspace / "probes"
        probes.mkdir()
        (probes / "linked").symlink_to(outside, target_is_directory=True)
        clean_probe_artifacts(self.workspace)
        self.assertEqual(keep.read_text(), "keep\n")
        self.assertEqual(working_changes(self.workspace), [])

    def test_preserves_ignored_runtime_files_and_nested_repositories(self):
        (self.workspace / ".git/info/exclude").write_text("*.pyc\n.atrex_long_horizon/\n")
        ignored = self.write("probes/v1/__pycache__/helper.pyc")
        receipt = self.write(".atrex_long_horizon/state.json", '{"attempts": []}')
        self.write("probes/v1/diagnostic.py")
        nested = self.workspace / ".probe/nested"
        nested.mkdir(parents=True)
        subprocess.run(["git", "init", "-q", str(nested)], check=True, capture_output=True)
        clean_probe_artifacts(self.workspace)
        self.assertTrue(ignored.exists())
        self.assertEqual(receipt.read_text(), '{"attempts": []}')
        self.assertTrue((nested / ".git").is_dir())
        self.assertFalse((self.workspace / "probes/v1/diagnostic.py").exists())

    def test_resume_cleans_after_baseline_preparation_before_round_checks(self):
        base = SimpleNamespace(workspace=self.workspace)
        campaign = campaign_api.LongHorizonCampaign(base, max_episodes=0,
                                                   verifier=object(), session_runner=object())
        def prepare(_):
            self.write("probes/v1/leftover.log")
        def conversion(*args):
            self.assertFalse((self.workspace / "probes").exists())
            return False
        with mock.patch.object(campaign_api.main_adapter, "prepare_campaign", side_effect=prepare), \
                mock.patch.object(campaign_api.main_adapter, "restored_stall", return_value=0), \
                mock.patch.object(campaign, "_recover_interrupted", return_value=None), \
                mock.patch.object(campaign_api.main_adapter, "conversion_required", side_effect=conversion):
            self.assertEqual(campaign.run(), "max-episodes")
        self.assertEqual(self.git("rev-parse", "HEAD"), self.head)


if __name__ == "__main__":
    unittest.main()
