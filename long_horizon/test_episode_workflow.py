"""All new and unfinished Episodes use one workflow and the ABBA promotion gate."""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from orchestrator.campaign import Campaign
from orchestrator.constants import REPO_ROOT
from orchestrator.optimize import _run_main

from .campaign import LongHorizonCampaign
from .git_episode import EpisodeWorktree
from .models import EpisodeHandoff, SupervisorState, VerificationResult


class EpisodeWorkflowTest(unittest.TestCase):
    def setUp(self) -> None:
        self.base = SimpleNamespace(
            workspace=Path("/campaign"), platform="L20N", arch="sm_120",
            framework="Triton", name="identity", notes="none", agent_skills=(),
        )
        self.runner = LongHorizonCampaign(self.base)
        self.worktree = Mock(
            path=Path("/episode"), episode=1, base_commit="base", branch="episode-1",
        )
        self.worktree.validate_candidate.return_value = ("", ["kernel.py"])
        self.handoff = EpisodeHandoff("candidate_ready", "candidate")
        self.verifier = Mock()
        self.verifier.verify.return_value = VerificationResult("PASS", 8.0, 10.0, 20.0)

    def test_early_and_late_episodes_use_the_same_prompt(self) -> None:
        directives = dict.fromkeys(
            ("mode_policy", "hardware", "sandbox", "evaluator", "agent_runtime"), "",
        )
        with patch("long_horizon.main_adapter.episode_directives", return_value=directives):
            for episode in (1, 2, 3, 10):
                with self.subTest(episode=episode):
                    prompt = self.runner._prompt(
                        episode=episode, version=episode + 1,
                        worktree=EpisodeWorktree(episode, "base", "branch", Path("/episode")),
                        conversion_pending=False,
                    )
                    self.assertIn(f"# Kernel optimization episode {episode}", prompt)
                    self.assertIn("## Engineering loop", prompt)
                    self.assertIn("required ABBA comparison", prompt)
                    self.assertNotIn("Fast", prompt)
                    self.assertNotIn("{{", prompt)

    def test_new_and_old_mode_markers_cannot_bypass_abba(self) -> None:
        for episode in (1, 2, 3):
            for old_mode in (None, "fast", "full"):
                with self.subTest(episode=episode, old_mode=old_mode):
                    active = {"episode": episode, "mode": old_mode, "fast_trials": 5}
                    self.verifier.reset_mock()
                    with (
                        patch(
                            "long_horizon.main_adapter.candidate_policy_violations",
                            return_value=[],
                        ) as policy,
                        patch("long_horizon.main_adapter.candidate_is_gluon", return_value=False),
                    ):
                        _, _, result, accepted = self.runner._assess_terminal_handoff(
                            Mock(), active, self.worktree, self.handoff,
                            conversion_pending=False,
                            verifier=self.verifier,
                        )
                    self.verifier.verify.assert_called_once_with(
                        self.worktree.path, base_commit="base", candidate_commit="candidate",
                        changed_paths=["kernel.py"],
                    )
                    policy.assert_called_once_with(
                        self.base, self.worktree.path, require_gluon=False,
                    )
                    self.assertIs(result, self.verifier.verify.return_value)
                    self.assertTrue(accepted)
                    self.assertEqual(active["phase"], "verifying")

    def test_one_valid_experiment_is_not_subject_to_five_trial_minimum(self) -> None:
        journal = {
            "experiments": [{"name": "one valid change"}],
            "finalized_at": "2026-09-01T01:00:00+00:00",
        }
        with (
            patch("long_horizon.campaign.validate_terminal", return_value=""),
            patch("long_horizon.campaign.load_journal", return_value=journal),
            patch("long_horizon.campaign.git_text", return_value="1"),
        ):
            self.assertEqual(
                self.runner._completion_check(self.worktree, Path("/journal"), self.handoff), "",
            )

    def test_recovered_terminal_handoff_runs_abba_without_rerunning_agent(self) -> None:
        active = {"mode": "fast", "fast_trials": 5, "memory_version": 2}
        with (
            patch("long_horizon.campaign.read_handoff", return_value=self.handoff),
            patch.object(self.runner, "_completion_check", return_value=""),
            patch("long_horizon.main_adapter.conversion_required", return_value=False),
            patch("long_horizon.main_adapter.candidate_policy_violations", return_value=[]),
            patch("long_horizon.main_adapter.candidate_is_gluon", return_value=False),
            patch.object(self.runner, "_record_terminal_episode") as record,
        ):
            self.assertTrue(self.runner._recover_completed_handoff(
                Mock(), SupervisorState(), active, self.worktree, verifier=self.verifier,
            ))
        self.verifier.verify.assert_called_once()
        self.assertTrue(record.call_args.kwargs["accepted"])
        self.assertTrue(record.call_args.kwargs["recovered_after_supervisor_interruption"])

    def test_retired_options_and_templates_are_removed(self) -> None:
        for cls in (Campaign, LongHorizonCampaign):
            self.assertFalse(any(field.name.startswith("fast_") for field in fields(cls)))
        for name in ("fast_episode.md", "fast_episode_conversion.md"):
            self.assertFalse((REPO_ROOT / "orchestrator/prompts" / name).exists())
        output = io.StringIO()
        with redirect_stdout(output), self.assertRaises(SystemExit) as exited:
            _run_main(["--help"])
        self.assertEqual(exited.exception.code, 0)
        self.assertNotIn("--fast-", output.getvalue())
        for option in ("--fast-episodes", "--fast-trials", "--fast-episode-ask-codex"):
            error = io.StringIO()
            args = ["--op-dir", "/unused", "--platform", "L20N", option]
            if option in ("--fast-episodes", "--fast-trials"):
                args.append("1")
            with redirect_stderr(error), self.assertRaises(SystemExit) as exited:
                _run_main(args)
            self.assertEqual(exited.exception.code, 2)
            self.assertIn(f"unrecognized arguments: {option}", error.getvalue())


if __name__ == "__main__":
    unittest.main()
