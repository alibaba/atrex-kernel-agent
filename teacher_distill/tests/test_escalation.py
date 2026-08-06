from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from orchestrator import optimize
from orchestrator.stop_policy import StopDecision, StopDecisionStatus
from teacher_distill.escalation import (
    LongHorizonEpisodeRunner,
    TeacherEscalationManager,
    mask_half_for_partial_restart,
)
from teacher_distill.session_policy import TeacherSessionPolicy


class FakeEpisodeRunner:
    def __init__(self, promoted: bool) -> None:
        self.promoted = promoted
        self.calls = 0

    def run(self, _candidate: optimize.Campaign) -> bool:
        self.calls += 1
        return self.promoted


class FakeCandidate:
    def __init__(self, workspace: Path, reasons: list[str]) -> None:
        self.workspace = workspace
        self.reasons = list(reasons)
        self.max_stall = 3
        self.run_calls = 0

    def run(self) -> str:
        self.run_calls += 1
        return self.reasons.pop(0)


def _git(workspace: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=workspace, check=True, capture_output=True, text=True
    ).stdout.strip()


def _workspace(root: Path, versions: int = 7) -> Path:
    workspace = root / "candidate"
    (workspace / "memory").mkdir(parents=True)
    (workspace / "kernel.py").write_text("# best\n", encoding="utf-8")
    for version in range(versions):
        (workspace / "memory" / f"v{version}.json").write_text(
            json.dumps({"version": f"v{version}", "masked": False}), encoding="utf-8"
        )
    _git(workspace, "init", "-q")
    _git(workspace, "config", "user.email", "test@local")
    _git(workspace, "config", "user.name", "test")
    _git(workspace, "add", "-A")
    _git(workspace, "commit", "-q", "-m", "best")
    optimize.write_stall(workspace, 3)
    return workspace


class PartialRestartTest(unittest.TestCase):
    def test_mask_preserves_v0_v1_and_latest_and_masks_half_of_middle_history(self) -> None:
        with tempfile.TemporaryDirectory(prefix="teacher-partial-mask-") as temp_dir:
            workspace = _workspace(Path(temp_dir), versions=8)
            masked = mask_half_for_partial_restart(workspace)
            records = {
                path.stem: json.loads(path.read_text(encoding="utf-8"))
                for path in (workspace / "memory").glob("v*.json")
            }

        self.assertEqual(masked, ("v2", "v4", "v6"))
        self.assertFalse(records["v0"]["masked"])
        self.assertFalse(records["v1"]["masked"])
        self.assertFalse(records["v7"]["masked"])
        self.assertTrue(all(records[version]["masked"] for version in masked))


class LongHorizonEpisodeRunnerTest(unittest.TestCase):
    def test_runner_executes_exactly_one_hidden_policy_episode_and_detects_promotion(self) -> None:
        with tempfile.TemporaryDirectory(prefix="teacher-long-episode-") as temp_dir:
            root = Path(temp_dir)
            workspace = _workspace(root)
            candidate = optimize.Campaign(
                name="demo",
                kernel_demo="/tmp/reference.py",
                platform="H20",
                framework="CuteDSL",
                work_dir=str(root),
            )
            # Reuse the initialized fixture repository at the Campaign's canonical path.
            canonical = candidate.workspace
            workspace.rename(canonical)
            sanitized = root / "sanitized"
            teacher = root / "private" / "teacher"
            source_wiki = root / "source-wiki"
            references = root / "reference-projects"
            for path in (sanitized, teacher, source_wiki, references):
                path.mkdir(parents=True, exist_ok=True)
            policy = TeacherSessionPolicy(
                knowledge_view=sanitized,
                teacher_solution=teacher,
                private_root=root / "private",
                source_wiki=source_wiki,
                reference_projects=references,
            )
            captured: dict[str, object] = {}

            class FakeLongCampaign:
                def __init__(self, **kwargs):
                    captured.update(kwargs)

                def run(self):
                    (canonical / "kernel.py").write_text("# promoted\n", encoding="utf-8")
                    _git(canonical, "add", "kernel.py")
                    _git(canonical, "commit", "-q", "-m", "promoted")
                    return "episode-limit"

            with mock.patch(
                "teacher_distill.escalation.LongHorizonCampaign",
                FakeLongCampaign,
            ):
                promoted = LongHorizonEpisodeRunner(
                    session_policy=policy,
                    private_dir=root / "private",
                ).run(candidate)

        self.assertTrue(promoted)
        self.assertEqual(captured["episode_limit"], 1)
        self.assertEqual(captured["max_episodes"], 1)
        self.assertEqual(captured["blocked_retry_limit"], 0)
        self.assertEqual(captured["store_root"], (root / "private/long_horizon").resolve())
        self.assertIsNotNone(captured["session_runner"])
        self.assertIsNotNone(captured["episode_runtime_linker"])


class TeacherEscalationManagerTest(unittest.TestCase):
    def test_failed_episode_performs_one_partial_restart_then_continues(self) -> None:
        with tempfile.TemporaryDirectory(prefix="teacher-escalation-fail-") as temp_dir:
            root = Path(temp_dir)
            workspace = _workspace(root)
            runner = FakeEpisodeRunner(promoted=False)
            candidate = FakeCandidate(workspace, ["budget: max-iters"])
            manager = TeacherEscalationManager(
                private_dir=root / "private",
                episode_runner=runner,
                partial_restart_limit=1,
                final_max_stall=5,
            )

            reason = manager.continue_after_stall(
                candidate, "stall: 3 iterations with no commit"
            )
            state = json.loads((root / "private/escalation_state.json").read_text())

        self.assertEqual(reason, "budget: max-iters")
        self.assertEqual(runner.calls, 1)
        self.assertEqual(candidate.run_calls, 1)
        self.assertEqual(candidate.max_stall, 5)
        self.assertEqual(state["episodes_used"], 1)
        self.assertEqual(state["partial_restarts_used"], 1)
        self.assertTrue(state["masked_versions"])

    def test_promoted_episode_resets_stall_and_does_not_mask_memory(self) -> None:
        with tempfile.TemporaryDirectory(prefix="teacher-escalation-win-") as temp_dir:
            root = Path(temp_dir)
            workspace = _workspace(root)
            runner = FakeEpisodeRunner(promoted=True)
            candidate = FakeCandidate(workspace, ["success: teacher ABBA passed (candidate/teacher 1.0)"])
            manager = TeacherEscalationManager(
                private_dir=root / "private",
                episode_runner=runner,
                partial_restart_limit=1,
                final_max_stall=5,
            )

            reason = manager.continue_after_stall(
                candidate, "stall: 3 iterations with no commit"
            )
            state = json.loads((root / "private/escalation_state.json").read_text())
            restored_stall = optimize.read_stall(workspace)

        self.assertTrue(reason.startswith("success:"))
        self.assertEqual(restored_stall, 0)
        self.assertEqual(state["episodes_used"], 1)
        self.assertEqual(state["partial_restarts_used"], 0)
        self.assertEqual(state["masked_versions"], [])

    def test_promoted_episode_checks_teacher_target_before_starting_another_iteration(self) -> None:
        class TargetReachedCandidate(FakeCandidate):
            def _accepted_stop_decision(self, _version, _memory):
                return StopDecision(
                    StopDecisionStatus.SUCCESS,
                    "success: teacher ABBA passed after long episode",
                )

            @staticmethod
            def _report_stop_policy_infra_error(_decision):
                return None

        with tempfile.TemporaryDirectory(prefix="teacher-escalation-target-") as temp_dir:
            root = Path(temp_dir)
            workspace = _workspace(root)
            runner = FakeEpisodeRunner(promoted=True)
            candidate = TargetReachedCandidate(workspace, ["should not run"])
            manager = TeacherEscalationManager(
                private_dir=root / "private",
                episode_runner=runner,
                partial_restart_limit=1,
                final_max_stall=5,
            )
            reason = manager.continue_after_stall(
                candidate, "stall: 3 iterations with no commit"
            )

        self.assertEqual(reason, "success: teacher ABBA passed after long episode")
        self.assertEqual(candidate.run_calls, 0)

    def test_resume_continues_post_episode_phase_without_second_episode(self) -> None:
        with tempfile.TemporaryDirectory(prefix="teacher-escalation-phase-resume-") as temp_dir:
            root = Path(temp_dir)
            workspace = _workspace(root)
            private = root / "private"
            private.mkdir()
            (private / "escalation_state.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "episodes_used": 1,
                        "partial_restarts_used": 0,
                        "masked_versions": [],
                        "last_episode_promoted": False,
                        "phase": "episode_complete",
                        "last_reason": None,
                    }
                ),
                encoding="utf-8",
            )
            runner = FakeEpisodeRunner(promoted=False)
            candidate = FakeCandidate(workspace, ["budget: max-iters"])
            manager = TeacherEscalationManager(
                private_dir=private,
                episode_runner=runner,
                partial_restart_limit=1,
                final_max_stall=5,
            )

            reason = manager.continue_after_stall(
                candidate, "stall: 3 iterations with no commit"
            )
            state = json.loads((private / "escalation_state.json").read_text())

        self.assertEqual(reason, "budget: max-iters")
        self.assertEqual(runner.calls, 0)
        self.assertEqual(candidate.run_calls, 1)
        self.assertEqual(candidate.max_stall, 5)
        self.assertEqual(state["phase"], "complete")

    def test_resume_never_runs_more_than_one_episode_or_restart(self) -> None:
        with tempfile.TemporaryDirectory(prefix="teacher-escalation-resume-") as temp_dir:
            root = Path(temp_dir)
            workspace = _workspace(root)
            first_runner = FakeEpisodeRunner(promoted=False)
            first = TeacherEscalationManager(
                private_dir=root / "private",
                episode_runner=first_runner,
                partial_restart_limit=1,
                final_max_stall=5,
            )
            first.continue_after_stall(
                FakeCandidate(workspace, ["stall: 5 iterations with no commit"]),
                "stall: 3 iterations with no commit",
            )

            second_runner = FakeEpisodeRunner(promoted=False)
            second = TeacherEscalationManager(
                private_dir=root / "private",
                episode_runner=second_runner,
                partial_restart_limit=1,
                final_max_stall=5,
            )
            candidate = FakeCandidate(workspace, [])
            reason = second.continue_after_stall(
                candidate, "stall: 5 iterations with no commit"
            )

        self.assertEqual(reason, "stall: 5 iterations with no commit")
        self.assertEqual(second_runner.calls, 0)
        self.assertEqual(candidate.run_calls, 0)

    def test_non_stall_reason_is_returned_without_escalation(self) -> None:
        with tempfile.TemporaryDirectory(prefix="teacher-escalation-skip-") as temp_dir:
            runner = FakeEpisodeRunner(promoted=False)
            manager = TeacherEscalationManager(
                private_dir=Path(temp_dir) / "private",
                episode_runner=runner,
                partial_restart_limit=1,
                final_max_stall=5,
            )
            candidate = FakeCandidate(Path(temp_dir) / "candidate", [])
            reason = manager.continue_after_stall(candidate, "budget: max-iters")

        self.assertEqual(reason, "budget: max-iters")
        self.assertEqual(runner.calls, 0)


if __name__ == "__main__":
    unittest.main()
