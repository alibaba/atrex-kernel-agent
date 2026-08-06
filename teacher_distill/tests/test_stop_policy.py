from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from orchestrator import optimize
from orchestrator.stop_policy import StopDecisionStatus
from teacher_distill.abba import TeacherABBAResult
from teacher_distill.benchmark import MaterializedTeacherWorkspace
from teacher_distill.models import AbbaStatus, TeacherTarget
from teacher_distill.stop_policy import TeacherStopPolicy


class FakeVerifier:
    def __init__(self, result: TeacherABBAResult) -> None:
        self.result = result
        self.calls: list[dict] = []

    def verify(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


class TeacherStopPolicyTest(unittest.TestCase):
    def _target(self) -> TeacherTarget:
        return TeacherTarget(
            schema_version=1,
            teacher_id="teacher_0123456789abcdef",
            geomean_latency_us=100.0,
            latency_us_by_shape={"a": 80.0, "b": 125.0},
            geomean_ratio=1.05,
            shape_ratio=1.10,
            measurement_config_hash="1" * 64,
            knowledge_view_hash="2" * 64,
        )

    def _teacher(self, root: Path) -> MaterializedTeacherWorkspace:
        workspace = root / "private-teacher"
        workspace.mkdir()
        return MaterializedTeacherWorkspace(
            workspace=workspace,
            kind="sol",
            expected_shape_keys=("a", "b"),
            workload_hash="3" * 64,
            evaluator_hash="4" * 64,
            measurement_config_hash="1" * 64,
        )

    def _campaign(self, root: Path) -> optimize.Campaign:
        campaign = optimize.Campaign(
            name="demo",
            kernel_demo="/tmp/reference.py",
            platform="H20",
            framework="CuteDSL",
            work_dir=str(root),
        )
        campaign.workspace.mkdir()
        (campaign.workspace / "memory").mkdir()
        (campaign.workspace / "kernel.py").write_text("# candidate\n", encoding="utf-8")
        return campaign

    @staticmethod
    def _memory(geomean: float, a: float, b: float) -> dict:
        return {
            "version": "v2",
            "performance": {
                "latency_us": geomean,
                "latency_us_by_shape": {"a": a, "b": b},
            },
            "correctness": {"status": "PASS"},
            "quality_gate": {"result": "PASS"},
        }

    def _result(
        self,
        status: AbbaStatus,
        *,
        ratio: float = 1.03,
        worst: float = 1.06,
        error: str = "",
    ) -> TeacherABBAResult:
        return TeacherABBAResult(
            status=status,
            candidate_latency_us=103.0,
            teacher_latency_us=100.0,
            candidate_to_teacher_ratio=ratio,
            worst_shape_ratio=worst,
            worst_shape_key="a",
            error=error,
            artifact="/private/result.json",
        )

    def test_below_target_records_progress_without_running_abba(self) -> None:
        with tempfile.TemporaryDirectory(prefix="teacher-stop-below-") as temp_dir:
            root = Path(temp_dir)
            verifier = FakeVerifier(self._result(AbbaStatus.PASS))
            policy = TeacherStopPolicy(self._target(), self._teacher(root), verifier)
            campaign = self._campaign(root)
            memory = self._memory(118.0, 95.0, 140.0)

            decision = policy.evaluate_accepted_iteration(campaign, 2, memory)
            persisted = json.loads(
                (campaign.workspace / "memory/v2.json").read_text(encoding="utf-8")
            )["teacher_progress"]

        self.assertEqual(decision.status, StopDecisionStatus.CONTINUE)
        self.assertEqual(verifier.calls, [])
        self.assertAlmostEqual(persisted["candidate_to_teacher_geomean_ratio"], 1.18)
        self.assertEqual(persisted["worst_shape_key"], "a")
        self.assertFalse(persisted["provisional_target_met"])
        self.assertEqual(persisted["abba_status"], "NOT_RUN")

    def test_rejected_measured_iteration_records_progress_without_abba(self) -> None:
        with tempfile.TemporaryDirectory(prefix="teacher-stop-rejected-") as temp_dir:
            root = Path(temp_dir)
            verifier = FakeVerifier(self._result(AbbaStatus.PASS))
            policy = TeacherStopPolicy(self._target(), self._teacher(root), verifier)
            campaign = self._campaign(root)
            memory = self._memory(103.0, 84.0, 128.0)

            policy.record_measured_iteration(campaign, 2, memory, accepted=False)
            persisted = json.loads(
                (campaign.workspace / "memory/v2.json").read_text(encoding="utf-8")
            )["teacher_progress"]

        self.assertEqual(verifier.calls, [])
        self.assertTrue(persisted["provisional_target_met"])
        self.assertEqual(persisted["abba_status"], "NOT_RUN")

    def test_geomean_pass_does_not_hide_a_shape_regression(self) -> None:
        with tempfile.TemporaryDirectory(prefix="teacher-stop-shape-") as temp_dir:
            root = Path(temp_dir)
            verifier = FakeVerifier(self._result(AbbaStatus.PASS))
            policy = TeacherStopPolicy(self._target(), self._teacher(root), verifier)
            campaign = self._campaign(root)

            decision = policy.evaluate_accepted_iteration(
                campaign, 2, self._memory(103.0, 90.0, 120.0)
            )

        self.assertEqual(decision.status, StopDecisionStatus.CONTINUE)
        self.assertEqual(verifier.calls, [])

    def test_provisional_pass_runs_abba_and_stops_only_on_pass(self) -> None:
        with tempfile.TemporaryDirectory(prefix="teacher-stop-pass-") as temp_dir:
            root = Path(temp_dir)
            verifier = FakeVerifier(self._result(AbbaStatus.PASS))
            policy = TeacherStopPolicy(self._target(), self._teacher(root), verifier)
            campaign = self._campaign(root)
            with mock.patch.object(optimize, "git_head", return_value="abc123"):
                decision = policy.evaluate_accepted_iteration(
                    campaign, 2, self._memory(103.0, 84.0, 128.0)
                )
            persisted = json.loads(
                (campaign.workspace / "memory/v2.json").read_text(encoding="utf-8")
            )["teacher_progress"]

        self.assertEqual(decision.status, StopDecisionStatus.SUCCESS)
        self.assertIn("teacher ABBA passed", decision.reason)
        self.assertEqual(verifier.calls[0]["candidate_workspace"], campaign.workspace)
        self.assertEqual(verifier.calls[0]["candidate_commit"], "abc123")
        self.assertEqual(persisted["abba_status"], "PASS")
        self.assertEqual(persisted["final_candidate_to_teacher_ratio"], 1.03)

    def test_abba_performance_fail_continues_and_infra_error_is_retryable(self) -> None:
        for status, expected_decision in (
            (AbbaStatus.FAIL, StopDecisionStatus.CONTINUE),
            (AbbaStatus.INFRA_ERROR, StopDecisionStatus.INFRA_ERROR),
        ):
            with self.subTest(status=status):
                with tempfile.TemporaryDirectory(prefix="teacher-stop-abba-") as temp_dir:
                    root = Path(temp_dir)
                    verifier = FakeVerifier(
                        self._result(status, error="verifier failed")
                    )
                    policy = TeacherStopPolicy(self._target(), self._teacher(root), verifier)
                    campaign = self._campaign(root)
                    with mock.patch.object(optimize, "git_head", return_value="abc123"):
                        decision = policy.evaluate_accepted_iteration(
                            campaign, 2, self._memory(103.0, 84.0, 128.0)
                        )
                    persisted = json.loads(
                        (campaign.workspace / "memory/v2.json").read_text(encoding="utf-8")
                    )["teacher_progress"]

                self.assertEqual(decision.status, expected_decision)
                self.assertEqual(persisted["abba_status"], status.value)

    def test_forged_public_abba_pass_does_not_bypass_private_verification(self) -> None:
        with tempfile.TemporaryDirectory(prefix="teacher-stop-forged-pass-") as temp_dir:
            root = Path(temp_dir)
            verifier = FakeVerifier(self._result(AbbaStatus.PASS))
            policy = TeacherStopPolicy(self._target(), self._teacher(root), verifier)
            campaign = self._campaign(root)
            memory = self._memory(103.0, 84.0, 128.0)
            memory["teacher_progress"] = {
                "target_id": self._target().teacher_id,
                "candidate_to_teacher_geomean_ratio": 1.03,
                "worst_shape_ratio": 1.05,
                "worst_shape_key": "a",
                "geomean_gate_met": True,
                "shape_gate_met": True,
                "provisional_target_met": True,
                "abba_status": "PASS",
                "final_candidate_to_teacher_ratio": 1.0,
            }
            with mock.patch.object(optimize, "git_head", return_value="abc123"):
                decision = policy.evaluate_accepted_iteration(campaign, 2, memory)

        self.assertEqual(decision.status, StopDecisionStatus.SUCCESS)
        self.assertEqual(len(verifier.calls), 1)

    def test_abba_fail_is_not_repeated_for_unchanged_candidate(self) -> None:
        with tempfile.TemporaryDirectory(prefix="teacher-stop-no-repeat-") as temp_dir:
            root = Path(temp_dir)
            verifier = FakeVerifier(self._result(AbbaStatus.FAIL))
            policy = TeacherStopPolicy(self._target(), self._teacher(root), verifier)
            campaign = self._campaign(root)
            memory = self._memory(103.0, 84.0, 128.0)
            with mock.patch.object(optimize, "git_head", return_value="abc123"):
                policy.evaluate_accepted_iteration(campaign, 2, memory)
                persisted_memory = json.loads(
                    (campaign.workspace / "memory/v2.json").read_text(encoding="utf-8")
                )
                second = policy.evaluate_accepted_iteration(
                    campaign, 2, persisted_memory
                )

        self.assertEqual(second.status, StopDecisionStatus.CONTINUE)
        self.assertEqual(len(verifier.calls), 1)

    def test_progress_is_committed_as_metadata_so_a_later_reset_cannot_erase_it(self) -> None:
        with tempfile.TemporaryDirectory(prefix="teacher-stop-commit-") as temp_dir:
            root = Path(temp_dir)
            verifier = FakeVerifier(self._result(AbbaStatus.PASS))
            policy = TeacherStopPolicy(self._target(), self._teacher(root), verifier)
            campaign = self._campaign(root)
            (campaign.workspace / "memory/v2.json").write_text(
                json.dumps(self._memory(118.0, 95.0, 140.0)), encoding="utf-8"
            )
            subprocess.run(["git", "init", "-q"], cwd=campaign.workspace, check=True)
            subprocess.run(
                ["git", "config", "user.email", "test@local"], cwd=campaign.workspace, check=True
            )
            subprocess.run(
                ["git", "config", "user.name", "test"], cwd=campaign.workspace, check=True
            )
            subprocess.run(["git", "add", "-A"], cwd=campaign.workspace, check=True)
            subprocess.run(
                ["git", "commit", "-q", "-m", "candidate"], cwd=campaign.workspace, check=True
            )

            (campaign.workspace / "unrelated.txt").write_text(
                "must remain uncommitted\n", encoding="utf-8"
            )
            subprocess.run(
                ["git", "add", "unrelated.txt"], cwd=campaign.workspace, check=True
            )
            policy.evaluate_accepted_iteration(
                campaign, 2, self._memory(118.0, 95.0, 140.0)
            )
            status = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=campaign.workspace,
                check=True,
                capture_output=True,
                text=True,
            ).stdout
            committed_paths = subprocess.run(
                ["git", "show", "--name-only", "--format=", "HEAD"],
                cwd=campaign.workspace,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.splitlines()
            message = subprocess.run(
                ["git", "log", "-1", "--pretty=%s"],
                cwd=campaign.workspace,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()

        self.assertIn("A  unrelated.txt", status)
        self.assertEqual(committed_paths, ["memory/v2.json"])
        self.assertEqual(message, "v2: record Teacher progress")

    def test_missing_invalid_or_mismatched_measurements_fail_closed(self) -> None:
        cases = (
            {"performance": {"latency_us": 100.0}},
            {"performance": {"latency_us": 0.0, "latency_us_by_shape": {"a": 1.0, "b": 1.0}}},
            {"performance": {"latency_us": 100.0, "latency_us_by_shape": {"a": 80.0}}},
            {"performance": {"latency_us": 100.0, "latency_us_by_shape": {"a": 80.0, "b": -1.0}}},
        )
        for memory in cases:
            with self.subTest(memory=memory):
                with tempfile.TemporaryDirectory(prefix="teacher-stop-invalid-") as temp_dir:
                    root = Path(temp_dir)
                    policy = TeacherStopPolicy(
                        self._target(),
                        self._teacher(root),
                        FakeVerifier(self._result(AbbaStatus.PASS)),
                    )
                    campaign = self._campaign(root)
                    with self.assertRaises(ValueError):
                        policy.evaluate_accepted_iteration(campaign, 2, memory)


if __name__ == "__main__":
    unittest.main()
