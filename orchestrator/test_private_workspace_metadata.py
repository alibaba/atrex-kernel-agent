"""Private baseline pins and Git excludes survive Episode worktrees and recovery."""

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from long_horizon.store import CampaignStore

from .campaign import Campaign
from .constants import FRAMEWORK_BASELINE_FILE, STALL_STATE_FILE
from .durable_state import durable_write_json
from .git_metadata import WORKSPACE_EXCLUDES, install_git_excludes
from .optimize import _recorded_workspace_arch
from .optimization_policy import (
    MODE_STATE_FILE, install_workspace_policy, read_workspace_policy, workspace_policy_path,
)
from .workspace_state import (
    framework_baseline_path,
    read_framework_baseline,
    resolve_framework_baseline_commit,
)


class PrivateWorkspaceMetadataTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.workspace = self.root / "campaign"
        self.workspace.mkdir()
        self.git("init", "-q")
        self.git("config", "user.name", "Test")
        self.git("config", "user.email", "test@example.test")
        (self.workspace / "kernel.py").write_text("# baseline\n")
        self.git("add", "kernel.py")
        self.git("commit", "-qm", "baseline")
        self.commit = self.git("rev-parse", "HEAD")
        self.marker = {
            "commit": self.commit,
            "kernel_blob": self.git("rev-parse", "HEAD:kernel.py"),
            "version": "v1",
            "arch": "sm_100",
        }

    def git(self, *args: str, cwd: Path | None = None) -> str:
        return subprocess.run(
            ["git", *args], cwd=cwd or self.workspace,
            check=True, capture_output=True, text=True,
        ).stdout.strip()

    def pin(self) -> None:
        campaign = SimpleNamespace(
            workspace=self.workspace, framework="Triton", platform="B200", arch="sm_100",
        )
        Campaign._pin_framework_baseline(campaign, self.commit, version=1)

    def test_mode_policy_is_private_shared_and_still_enforces_resume_identity(self) -> None:
        install_workspace_policy(self.workspace, "production", "Triton", agent_runtime="claude")
        path = workspace_policy_path(self.workspace)
        self.assertFalse(path.is_relative_to(self.workspace))
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        self.assertFalse((self.workspace / MODE_STATE_FILE).exists())
        episode = self.root / "episode"
        self.git("worktree", "add", "-qb", "episode", str(episode), "HEAD")
        install_workspace_policy(episode, "production", "Triton")
        self.assertEqual(workspace_policy_path(episode), path)
        self.assertEqual(read_workspace_policy(episode)["agent_runtime"], "claude")
        for mode, framework, backend in (
            ("leaderboard", "Triton", "claude"),
            ("production", "CUDA", "claude"),
            ("production", "Triton", "codex"),
        ):
            with self.assertRaisesRegex(RuntimeError, "mismatch"):
                install_workspace_policy(episode, mode, framework, agent_runtime=backend)

    def test_legacy_mode_migrates_without_trusting_later_workspace_edits(self) -> None:
        legacy = self.workspace / MODE_STATE_FILE
        original = {"mode": "production", "framework": "CUDA", "agent_runtime": "claude"}
        legacy.write_text(json.dumps(original))
        install_workspace_policy(self.workspace, "production", "CUDA", agent_runtime="claude")
        self.assertFalse(legacy.exists())
        self.assertEqual(read_workspace_policy(self.workspace), original)
        legacy.write_text('{"mode":"leaderboard"}')
        self.assertEqual(read_workspace_policy(self.workspace), original)
        workspace_policy_path(self.workspace).write_text("broken")
        with self.assertRaisesRegex(RuntimeError, "invalid optimization-mode"):
            read_workspace_policy(self.workspace)

    def test_pin_is_private_shared_and_stable_after_kernel_changes(self) -> None:
        self.pin()
        path = framework_baseline_path(self.workspace)
        self.assertFalse(path.is_relative_to(self.workspace))
        self.assertFalse((self.workspace / FRAMEWORK_BASELINE_FILE).exists())
        self.assertNotIn(FRAMEWORK_BASELINE_FILE, self.git("ls-tree", "--name-only", "HEAD"))
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        (self.workspace / "kernel.py").write_text("# optimized\n")
        self.git("commit", "-qam", "next")
        episode = self.root / "episode"
        self.git("worktree", "add", "-qb", "episode", str(episode), "HEAD")
        self.assertEqual(framework_baseline_path(episode), path)
        for workspace in (self.workspace, episode):
            self.assertEqual(resolve_framework_baseline_commit(workspace), (self.commit, 1))
            self.assertEqual(_recorded_workspace_arch(workspace), "sm_100")

    def test_invalid_pin_fails_closed(self) -> None:
        path = framework_baseline_path(self.workspace)
        for changes, message in (
            ({"commit": "f" * 40}, "missing commit"),
            ({"kernel_blob": "wrong"}, "kernel blob does not match"),
            ({"version": "invalid"}, "unusable version"),
        ):
            with self.subTest(changes=changes):
                durable_write_json(path, {**self.marker, **changes})
                with self.assertRaisesRegex(RuntimeError, message):
                    resolve_framework_baseline_commit(self.workspace)
        path.write_text("{")
        with self.assertRaisesRegex(RuntimeError, "Invalid private framework-baseline JSON"):
            resolve_framework_baseline_commit(self.workspace)

    def test_pin_must_be_an_ancestor(self) -> None:
        future = self.git("commit-tree", "HEAD^{tree}", "-p", "HEAD", "-m", "future")
        durable_write_json(framework_baseline_path(self.workspace), {
            **self.marker, "commit": future,
        })
        with self.assertRaisesRegex(RuntimeError, "not an ancestor"):
            resolve_framework_baseline_commit(self.workspace)

    def test_uncommitted_legacy_marker_is_ignored(self) -> None:
        legacy = self.workspace / FRAMEWORK_BASELINE_FILE
        legacy.write_text(json.dumps(self.marker))
        self.assertIsNone(read_framework_baseline(self.workspace))
        self.git("add", FRAMEWORK_BASELINE_FILE)
        self.git("commit", "-qm", "legacy pin")
        legacy.write_text("invalid workspace data")
        self.assertEqual(resolve_framework_baseline_commit(self.workspace), (self.commit, 1))

    def test_failed_memory_commit_does_not_publish_pin(self) -> None:
        memory = self.workspace / "memory"
        memory.mkdir()
        (memory / "v1.json").write_text("{}")
        real_run = subprocess.run

        def run(command, *args, **kwargs):
            if command[:2] == ["git", "commit"]:
                raise subprocess.CalledProcessError(1, command)
            return real_run(command, *args, **kwargs)

        with patch("orchestrator.campaign.subprocess.run", side_effect=run):
            with self.assertRaises(subprocess.CalledProcessError):
                self.pin()
        self.assertFalse(framework_baseline_path(self.workspace).exists())

    def test_excludes_are_private_idempotent_and_shared(self) -> None:
        exclude = self.workspace / ".git/info/exclude"
        exclude.write_text("# custom\n/local-only")
        install_git_excludes(self.workspace, required=True)
        rules = exclude.read_text()
        self.assertTrue(rules.startswith("# custom\n/local-only\n"))
        for rule in WORKSPACE_EXCLUDES:
            self.assertEqual(rules.splitlines().count(rule), 1)
        CampaignStore.ensure_excluded(self.workspace)
        episode = self.root / "episode"
        self.git("worktree", "add", "-qb", "episode", str(episode), "HEAD")
        CampaignStore.ensure_excluded(episode)
        self.assertEqual(exclude.read_text(), rules)
        for workspace in (self.workspace, episode):
            self.assertFalse((workspace / ".gitignore").exists())
            for name in ("scratch/probe.py", "memory/live.json", STALL_STATE_FILE, "tools"):
                self.assertEqual(self.git("check-ignore", name, cwd=workspace), name)
            result = subprocess.run(
                ["git", "check-ignore", "memory/v1.json", "kernel.py"], cwd=workspace,
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 1)
            self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    unittest.main()
