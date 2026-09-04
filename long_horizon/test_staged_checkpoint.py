from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from long_horizon import journal
from long_horizon.campaign import LongHorizonCampaign
from long_horizon.git_episode import EpisodeWorktree, git_head
from long_horizon.models import SupervisorState
from long_horizon.protocol import normalize_handoff
from long_horizon.store import CampaignStore


class StagedActivationTest(unittest.TestCase):
    def test_defaults_activate_after_forty_completed_episodes(self) -> None:
        engine = LongHorizonCampaign(base_campaign=object())  # type: ignore[arg-type]
        self.assertEqual(engine.max_staged_episodes, 4)
        self.assertEqual(engine.staged_after_episodes, 40)
        self.assertEqual(engine.staged_after_stall, 8)
        self.assertFalse(
            engine._staged_allowed(SupervisorState(episodes=39), fast_mode=False)
        )
        self.assertTrue(
            engine._staged_allowed(SupervisorState(episodes=40), fast_mode=False)
        )
        self.assertFalse(
            engine._staged_allowed(SupervisorState(episodes=40), fast_mode=True)
        )

    def test_promotion_drought_starts_architectural_escape_early(self) -> None:
        engine = LongHorizonCampaign(base_campaign=object())  # type: ignore[arg-type]
        almost_stalled = SupervisorState(
            episodes=20,
            consecutive_without_promotion=7,
        )
        stalled = SupervisorState(
            episodes=20,
            consecutive_without_promotion=8,
        )
        self.assertEqual(engine._staged_trigger(almost_stalled, fast_mode=False), "")
        self.assertEqual(
            engine._staged_trigger(stalled, fast_mode=False),
            "promotion_drought",
        )
        self.assertEqual(
            engine._staged_trigger(
                SupervisorState(
                    episodes=92,
                    consecutive_without_promotion=78,
                ),
                fast_mode=False,
            ),
            "promotion_drought",
        )

    def test_active_initiative_continues_until_fourth_checkpoint(self) -> None:
        engine = LongHorizonCampaign(base_campaign=object())  # type: ignore[arg-type]
        self.assertTrue(
            engine._staged_allowed(
                SupervisorState(episodes=12, consecutive_staged=1),
                fast_mode=False,
            )
        )
        self.assertFalse(
            engine._staged_allowed(
                SupervisorState(episodes=54, consecutive_staged=4),
                fast_mode=False,
            )
        )


class StagedJournalTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / "journal.json"
        journal.initialize(
            self.path,
            episode=3,
            base_commit="base",
            branch="episode-3",
        )
        journal.append_experiment(
            self.path,
            {
                "name": "bring up async loader",
                "evaluation": {
                    "correctness": "unknown",
                    "performance": "not_improved",
                    "latency_us": None,
                    "kernel_hash": "checkpoint",
                },
                "wiki_usage_status": "not_queried",
            },
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def outcome() -> dict[str, object]:
        return {
            "summary": "async loader compiles and preserves its tile invariant",
            "next_directions": ["overlap producer and consumer warps"],
            "initiative_id": "warp-specialized-nvfp4",
            "stage": 1,
            "next_stage": "add producer-consumer overlap",
            "escape_hypothesis": "serial tile movement caps tensor-core issue rate",
            "architectural_delta": "dedicated producer and consumer warps",
            "final_success_criterion": "pass correctness and beat the incumbent",
            "abort_criterion": "producer overlap cannot hide measured load latency",
            "stage_gate": {
                "compile": "pass",
                "advancement": "pass",
                "scope": "one tile reaches shared memory",
                "evidence": "official sandbox compile log job-123",
            },
        }

    def test_staged_ready_round_trip(self) -> None:
        value = journal.finalize(
            self.path,
            state="staged_ready",
            outcome=self.outcome(),
            checkpoint_commit="abc123",
        )
        self.assertEqual(value["checkpoint_commit"], "abc123")
        handoff = normalize_handoff(
            {"status": "staged_ready", "checkpoint_commit": "abc123"}
        )
        self.assertIsNotNone(handoff)
        assert handoff is not None
        self.assertEqual(handoff.checkpoint_commit, "abc123")
        self.assertEqual(
            journal.validate_terminal(
                self.path,
                expected_episode=3,
                base_commit="base",
                branch="episode-3",
                state="staged_ready",
                checkpoint_commit="abc123",
            ),
            "",
        )

    def test_staged_ready_requires_compile_gate(self) -> None:
        outcome = self.outcome()
        outcome["stage_gate"] = {
            "compile": "fail",
            "advancement": "pass",
            "scope": "loader",
            "evidence": "compiler error",
        }
        with self.assertRaisesRegex(ValueError, "compile=pass"):
            journal.finalize(
                self.path,
                state="staged_ready",
                outcome=outcome,
                checkpoint_commit="abc123",
            )

    def test_staged_ready_requires_architectural_advancement(self) -> None:
        outcome = self.outcome()
        outcome["stage_gate"] = {
            "compile": "pass",
            "advancement": "fail",
            "scope": "loader",
            "evidence": "only the incumbent schedule compiled",
        }
        with self.assertRaisesRegex(ValueError, "advancement=pass"):
            journal.finalize(
                self.path,
                state="staged_ready",
                outcome=outcome,
                checkpoint_commit="abc123",
            )


class StagedCheckpointStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        subprocess.run(["git", "init", "-q"], cwd=self.workspace, check=True)
        (self.workspace / "kernel.py").write_text("VALUE = 0\n", encoding="utf-8")
        subprocess.run(["git", "add", "kernel.py"], cwd=self.workspace, check=True)
        subprocess.run(
            [
                "git",
                "-c",
                "user.name=test",
                "-c",
                "user.email=test@example.com",
                "commit",
                "-qm",
                "base",
            ],
            cwd=self.workspace,
            check=True,
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_snapshot_bootstraps_next_episode_without_changing_incumbent(self) -> None:
        store = CampaignStore(self.workspace)
        base = git_head(self.workspace)
        kernel = b"VALUE = 1\n"
        saved = store.save_staged_checkpoint(
            {
                "initiative_id": "async-pipeline",
                "stage": 1,
                "next_stage": "add consumer warp",
                "escape_hypothesis": "serialized movement limits utilization",
                "architectural_delta": "split producer and consumer warps",
                "final_success_criterion": "beat incumbent after full overlap",
                "abort_criterion": "official profile shows no load-bound region",
                "checkpoint_commit": "source-checkpoint",
                "source_episode": 1,
            },
            kernel,
        )
        self.assertNotIn("kernel_b64", saved)
        loaded = store.load_staged_checkpoint()
        self.assertIsNotNone(loaded)
        assert loaded is not None
        metadata, restored_kernel = loaded
        self.assertEqual(restored_kernel, kernel)
        self.assertEqual(metadata["initiative_id"], "async-pipeline")
        self.assertEqual(metadata["schema_version"], 2)

        worktree = EpisodeWorktree.create(
            self.workspace,
            episode=2,
            base_commit=base,
            root=self.root / "episode-worktrees",
        )
        try:
            checkpoint = worktree.bootstrap_staged_kernel(
                restored_kernel,
                initiative_id=str(metadata["initiative_id"]),
                stage=int(metadata["stage"]),
            )
            violation, paths = worktree.validate_candidate(checkpoint)
            self.assertEqual(violation, "")
            self.assertEqual(paths, ["kernel.py"])
            self.assertEqual((worktree.path / "kernel.py").read_bytes(), kernel)
            self.assertEqual(
                (self.workspace / "kernel.py").read_text(encoding="utf-8"),
                "VALUE = 0\n",
            )
            self.assertEqual(git_head(self.workspace), base)
        finally:
            worktree.remove(self.workspace)

        store.clear_staged_checkpoint()
        self.assertIsNone(store.load_staged_checkpoint())


if __name__ == "__main__":
    unittest.main()
