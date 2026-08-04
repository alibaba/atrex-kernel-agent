from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from long_horizon.campaign import LongHorizonCampaign, _conversion_parity_passes
from long_horizon.git_episode import EpisodeWorktree, git_head
from long_horizon.journal import append_experiment, finalize
from long_horizon.models import (
    EpisodeHandoff,
    SessionResult,
    VerificationResult,
    VerificationRun,
)
from long_horizon.protocol import atomic_write_json
from long_horizon.store import CampaignStore
from long_horizon.tests.helpers import init_repo, run_git


class CandidateRunner:
    def __init__(self, value: int = 5):
        self.value = value

    def run(
        self,
        workspace,
        prompt,
        *,
        timeout,
        handoff_path,
        handoff_resumes,
        completion_check,
        **kwargs,
    ):
        (workspace / "kernel.py").write_text(f"VALUE = {self.value}\n", encoding="utf-8")
        run_git(workspace, "add", "kernel.py")
        run_git(workspace, "commit", "-m", "candidate")
        candidate = git_head(workspace)
        journal_path = workspace / ".atrex_long_horizon" / "journal.json"
        append_experiment(
            journal_path,
            {
                "name": "rewrite",
                "hypothesis": "less work",
                "evidence": "development benchmark",
                "result": "faster",
                "decision": "continue",
            },
        )
        finalize(
            journal_path,
            state="candidate_ready",
            candidate_commit=candidate,
            outcome={"summary": "candidate is faster", "next_directions": ["more tuning"]},
        )
        handoff = EpisodeHandoff("candidate_ready", candidate_commit=candidate)
        atomic_write_json(handoff_path, handoff.as_dict())
        diagnosis = completion_check(handoff)
        if diagnosis:
            raise AssertionError(diagnosis)
        return SessionResult(
            exit_status=0,
            timed_out=False,
            tokens=123,
            session_id="session-1",
            resume_count=0,
            handoff=handoff,
        )


class FixedVerifier:
    def __init__(self, passed: bool):
        self.passed = passed

    def verify(self, workspace, *, base_commit, candidate_commit, changed_paths):
        candidate_latency = 8.0 if self.passed else 11.0
        improvement = 20.0 if self.passed else -10.0
        runs = [
            VerificationRun(
                "incumbent", 0, 0,
                {"all_pass": True, "latency_us_geomean": 10.0, "latency_us_by_shape": {"0": 10.0}},
            ),
            VerificationRun(
                "candidate", 0, 0,
                {"all_pass": True, "latency_us_geomean": candidate_latency, "latency_us_by_shape": {"0": candidate_latency}},
            ),
        ]
        return VerificationResult(
            "PASS" if self.passed else "FAIL",
            candidate_latency,
            10.0,
            improvement,
            runs=runs,
            error="" if self.passed else "regression",
        )


def fake_base(workspace: Path):
    return SimpleNamespace(
        workspace=workspace,
        platform="B200",
        framework="CuteDSL",
        notes="test",
        arch="sm_100",
        sandbox_hardware="REMOTE_GPU",
        sandbox_profile="",
        sandbox_url="",
        sandbox_timeout=600,
        atrex_bench_root="",
        optimization_mode="leaderboard",
        agent_cli="claude",
        target_util=90.0,
        _notify_improvement=mock.Mock(),
        _notify_iteration=mock.Mock(),
    )


class CampaignIntegrationTests(unittest.TestCase):
    def test_conversion_parity_matches_main_tolerance(self) -> None:
        runs = [
            VerificationRun("incumbent", 0, 0, {"all_pass": True}),
            VerificationRun("candidate", 0, 0, {"all_pass": True}),
        ]
        self.assertTrue(
            _conversion_parity_passes(
                VerificationResult("FAIL", 10.4, 10.0, -4.0, runs=runs)
            )
        )
        self.assertFalse(
            _conversion_parity_passes(
                VerificationResult("FAIL", 10.6, 10.0, -6.0, runs=runs)
            )
        )

    def test_main_max_iters_caps_canonical_versions_not_episode_count(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            repo = Path(temp) / "campaign"
            init_repo(repo)
            (repo / "memory").mkdir()
            (repo / "memory" / "v5.json").write_text(
                json.dumps({"version": "v5", "quality_gate": {"result": "FAIL"}}),
                encoding="utf-8",
            )
            run_git(repo, "add", "memory/v5.json")
            run_git(repo, "commit", "-m", "v5 record")
            runner = mock.Mock()
            with mock.patch("long_horizon.main_adapter.prepare_campaign", return_value=None):
                reason = LongHorizonCampaign(
                    base_campaign=fake_base(repo),
                    max_version=5,
                    session_runner=runner,
                ).run()
            self.assertEqual(reason, "budget: max-iters")
            runner.run.assert_not_called()

    def _patches(self):
        return (
            mock.patch("long_horizon.main_adapter.prepare_campaign", return_value=None),
            mock.patch("long_horizon.main_adapter.link_episode_runtime", return_value=None),
            mock.patch(
                "long_horizon.main_adapter.episode_directives",
                return_value={
                    "hardware": "hardware",
                    "sandbox": "sandbox",
                    "evaluator": "evaluator",
                    "mode_policy": "policy",
                },
            ),
            mock.patch(
                "long_horizon.main_adapter.iteration_playbook",
                return_value="current main playbook",
            ),
            mock.patch("long_horizon.main_adapter.latest_version", return_value=0),
        )

    def test_verified_candidate_is_squash_promoted(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repo = root / "campaign"
            base = init_repo(repo)
            patches = self._patches()
            with patches[0], patches[1], patches[2], patches[3], patches[4]:
                base_campaign = fake_base(repo)
                reason = LongHorizonCampaign(
                    base_campaign=base_campaign,
                    max_episodes=1,
                    verifier=FixedVerifier(True),
                    session_runner=CandidateRunner(5),
                    worktree_root=root / "worktrees",
                ).run()
            self.assertEqual(reason, "max-episodes")
            self.assertNotEqual(git_head(repo), base)
            self.assertEqual((repo / "kernel.py").read_text(encoding="utf-8"), "VALUE = 5\n")
            state = json.loads((repo / ".atrex_long_horizon/state.json").read_text())
            self.assertEqual(state["accepted"], 1)
            self.assertTrue((repo / "memory/v1.json").is_file())
            base_campaign._notify_improvement.assert_called_once()
            base_campaign._notify_iteration.assert_called_once()
            self.assertTrue(base_campaign._notify_iteration.call_args.args[2])

    def test_regressing_candidate_never_moves_incumbent(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repo = root / "campaign"
            base = init_repo(repo)
            patches = self._patches()
            with patches[0], patches[1], patches[2], patches[3], patches[4]:
                base_campaign = fake_base(repo)
                LongHorizonCampaign(
                    base_campaign=base_campaign,
                    max_episodes=1,
                    verifier=FixedVerifier(False),
                    session_runner=CandidateRunner(20),
                    worktree_root=root / "worktrees",
                ).run()
            self.assertNotEqual(git_head(repo), base)
            self.assertEqual(run_git(repo, "diff", base, "HEAD", "--", "kernel.py"), "")
            self.assertEqual((repo / "kernel.py").read_text(encoding="utf-8"), "VALUE = 10\n")
            rejected_memory = json.loads((repo / "memory/v1.json").read_text())
            self.assertEqual(rejected_memory["quality_gate"]["result"], "FAIL")
            state = json.loads((repo / ".atrex_long_horizon/state.json").read_text())
            self.assertEqual(state["rejected"], 1)
            base_campaign._notify_improvement.assert_not_called()
            base_campaign._notify_iteration.assert_called_once()
            self.assertFalse(base_campaign._notify_iteration.call_args.args[2])

    def test_interrupted_episode_is_archived_and_removed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repo = root / "campaign"
            base = init_repo(repo)
            store = CampaignStore(repo)
            episode = EpisodeWorktree.create(repo, 1, base, root=root / "worktrees")
            (episode.path / "kernel.py").write_text("VALUE = 7\n", encoding="utf-8")
            store.save_active(
                {
                    "episode": 1,
                    "base_commit": base,
                    "episode_branch": episode.branch,
                    "worktree": str(episode.path),
                    "phase": "exploring",
                }
            )
            campaign = LongHorizonCampaign(base_campaign=fake_base(repo))
            state = store.load_state()
            campaign._recover_interrupted(store, state)
            self.assertFalse(episode.path.exists())
            recovered = store.load_state()
            self.assertEqual(recovered.interrupted, 1)
            self.assertFalse(store.active_path.exists())


if __name__ == "__main__":
    unittest.main()
