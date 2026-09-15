"""Git is private Supervisor state; scratch belongs to one Episode only."""

from __future__ import annotations

import io
import shutil
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from long_horizon.campaign import LongHorizonCampaign
from long_horizon.git_episode import EpisodeWorktree, git_head, git_text
from long_horizon.main_adapter import fresh_session_command, resume_session_command
from long_horizon.store import CampaignStore
from orchestrator.agent_assets import REQUIRED_AGENT_SKILLS, SKILL_PATHS
from orchestrator.agent_sandbox import VISIBLE_WORKSPACE, wrap_agent_command


class GitOwnershipTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.repo = self.root / "incumbent"
        self.repo.mkdir()
        self.git("init", "-q", "-b", "main")
        self.git("config", "user.name", "Test")
        self.git("config", "user.email", "test@example.test")
        (self.repo / "kernel.py").write_text("def run(x): return x + 0\n")
        (self.repo / "README.md").write_text("protected\n")
        self.git("add", ".")
        self.git("commit", "-qm", "baseline")
        self.base = git_head(self.repo)

    def git(self, *args: str) -> None:
        subprocess.run(["git", *args], cwd=self.repo, check=True, capture_output=True)

    def episode(self, number: int = 1) -> EpisodeWorktree:
        return EpisodeWorktree.create(self.repo, number, self.base, self.root / "episodes")

    def test_supervisor_commits_only_measured_kernel_and_reuses_commit(self) -> None:
        worktree = self.episode()
        source = b"def run(x): return x\n"
        (worktree.path / "kernel.py").write_bytes(source)
        (worktree.path / "scratch/probe.py").write_text("diagnostic")
        commit = worktree.commit_candidate(source)
        self.assertNotEqual(commit, self.base)
        self.assertEqual(worktree.commit_candidate(source), commit)
        self.assertEqual(worktree.validate_candidate(commit), ("", ["kernel.py"]))
        self.assertEqual(git_head(self.repo), self.base)
        self.assertEqual(
            git_text(worktree.path, "diff", "--name-only", self.base, commit), "kernel.py"
        )
        self.assertEqual(git_text(worktree.path, "log", "-1", "--format=%an"), "AKA Supervisor")

    def test_invalid_candidate_does_not_commit(self) -> None:
        worktree = self.episode()
        original = (worktree.path / "kernel.py").read_bytes()
        with self.assertRaisesRegex(ValueError, "no Kernel change"):
            worktree.commit_candidate(original)
        source = b"def run(x): return x\n"
        (worktree.path / "kernel.py").write_bytes(source)
        with self.assertRaisesRegex(ValueError, "changed during report"):
            worktree.commit_candidate(b"not the current source")
        (worktree.path / "README.md").write_text("unexpected mutation")
        with self.assertRaisesRegex(ValueError, "protected path"):
            worktree.commit_candidate(source)
        self.assertEqual(git_head(worktree.path), self.base)

    def test_new_episode_scratch_is_empty_without_erasing_previous_work(self) -> None:
        (self.repo / "scratch").mkdir()
        (self.repo / "scratch/old.py").write_text("tracked legacy scratch")
        self.git("add", "scratch/old.py")
        self.git("commit", "-qm", "old workspace layout")
        self.base = git_head(self.repo)
        first = self.episode(1)
        self.assertEqual(list((first.path / "scratch").iterdir()), [])
        (first.path / "scratch/probe.py").write_text("episode one")
        second = self.episode(2)
        self.assertEqual(list((second.path / "scratch").iterdir()), [])
        self.assertTrue((self.repo / "scratch/old.py").exists())
        self.assertTrue((first.path / "scratch/probe.py").exists())
        source = b"def run(x): return x\n"
        (first.path / "kernel.py").write_bytes(source)
        self.assertEqual(
            first.validate_candidate(first.commit_candidate(source)), ("", ["kernel.py"])
        )

    def test_scratch_reset_unlinks_symlink_without_touching_target(self) -> None:
        worktree = self.episode()
        outside = self.root / "outside"
        outside.mkdir()
        (outside / "keep.txt").write_text("keep")
        (worktree.path / "scratch").rmdir()
        (worktree.path / "scratch").symlink_to(outside, target_is_directory=True)
        worktree.reset_scratch()
        self.assertFalse((worktree.path / "scratch").is_symlink())
        self.assertEqual(list((worktree.path / "scratch").iterdir()), [])
        self.assertEqual((outside / "keep.txt").read_text(), "keep")

    def test_resume_preserves_scratch_but_interrupted_preparation_resets_it(self) -> None:
        worktree = self.episode()
        store = CampaignStore(self.repo)
        runner = LongHorizonCampaign(SimpleNamespace(workspace=self.repo))
        for phase in ("preparing", "exploring", "verifying", "recording"):
            with self.subTest(phase=phase):
                probe = worktree.path / "scratch/probe.py"
                probe.write_text("unfinished diagnostic")
                store.save_active(
                    {
                        "episode": 1,
                        "base_commit": self.base,
                        "episode_branch": worktree.branch,
                        "worktree": str(worktree.path),
                        "phase": phase,
                        "memory_version": 2,
                    }
                )
                with (
                    patch.object(runner, "_recover_completed_handoff", return_value=False),
                    redirect_stdout(io.StringIO()),
                ):
                    resumed = runner._recover_interrupted(
                        store,
                        SimpleNamespace(attempts=[]),
                        verifier=Mock(),
                    )
                self.assertIsNotNone(resumed)
                self.assertEqual(probe.exists(), phase != "preparing")

    def sandbox(self, worktree: EpisodeWorktree, mode: str = "bwrap"):
        assets = self.root / "assets"
        assets.mkdir(exist_ok=True)
        (assets / "tools").mkdir(exist_ok=True)
        (assets / "tools/sandbox.py").write_text("# HTTP client fixture\n")
        for name in REQUIRED_AGENT_SKILLS:
            skill = assets / SKILL_PATHS[name]
            skill.mkdir(parents=True, exist_ok=True)
            (skill / "SKILL.md").write_text(f"# {name}\nTest fixture.\n")
        submodule = assets / "reference-projects/example"
        submodule.mkdir(parents=True, exist_ok=True)
        (submodule / ".git").write_text("gitdir: /private/metadata")
        host_home = self.root / "home"
        host_home.mkdir(exist_ok=True)
        with patch("orchestrator.agent_sandbox.platform.system", return_value="Linux"):
            return wrap_agent_command(
                ["true"],
                workspace=worktree.path,
                environment={"HOME": str(host_home), "PATH": "/usr/bin:/bin", "GIT_DIR": "/secret"},
                repository_root=assets,
                provider_homes=self.root / "private/providers",
                hidden_host_paths=(self.repo,),
                mode=mode,
                bwrap_executable=shutil.which("true") or "/usr/bin/true",
                agent_skills=(),
                agent_reference_projects=True,
            )

    def test_bwrap_omits_worktree_and_reference_git_metadata(self) -> None:
        worktree = self.episode()
        launch = self.sandbox(worktree)
        self.addCleanup(launch.close)
        command, environment = launch.command, launch.environment
        mounts = {
            command[i + 2]: (arg, Path(command[i + 1]))
            for i, arg in enumerate(command)
            if arg in {"--bind", "--ro-bind"}
        }
        for target in (
            VISIBLE_WORKSPACE / ".git",
            self.repo / ".git",
            self.root / "assets/reference-projects/example/.git",
        ):
            self.assertNotIn(str(target), mounts)
        self.assertEqual(mounts[str(VISIBLE_WORKSPACE)][0], "--bind")
        view = mounts[str(VISIBLE_WORKSPACE)][1]
        self.assertNotEqual(view, worktree.path)
        self.assertFalse((view / ".git").exists())
        self.assertFalse((view / "reference-projects/example/.git").exists())
        self.assertTrue((view / "reference-projects/example").is_dir())
        self.assertNotIn("GIT_DIR", environment)
        self.assertIn("--proc", command)
        self.assertNotIn(str(self.repo / "memory"), mounts)
        self.assertEqual(git_head(worktree.path), self.base)

    def test_git_workspaces_cannot_fall_back_to_unsandboxed_agents(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "Git-backed Agent workspaces require"):
            self.sandbox(self.episode(), mode="none")

    def test_codex_fresh_and_resume_do_not_require_git(self) -> None:
        for build in (fresh_session_command, resume_session_command):
            self.assertIn("--skip-git-repo-check", build("test", "session", "high", "codex"))


if __name__ == "__main__":
    unittest.main()
