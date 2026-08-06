from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from orchestrator import optimize
from orchestrator.stop_policy import (
    DefaultStopPolicy,
    StopDecision,
    StopDecisionStatus,
)


class RecordingStopPolicy:
    def __init__(self, decision: StopDecision) -> None:
        self.decision = decision
        self.calls: list[tuple[int, dict]] = []

    def evaluate_accepted_iteration(
        self,
        campaign: optimize.Campaign,
        version: int,
        memory: dict,
    ) -> StopDecision:
        self.calls.append((version, memory))
        return self.decision


def _git(workspace: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=workspace, check=True, capture_output=True, text=True
    ).stdout.strip()


def _campaign(root: Path, policy: RecordingStopPolicy | None = None) -> optimize.Campaign:
    campaign = optimize.Campaign(
        name="demo",
        kernel_demo=str(root / "reference.py"),
        platform="H20",
        framework="Triton",
        work_dir=str(root),
        framework_baseline="never",
        max_iters=1,
        stop_policy=policy,
    )
    workspace = campaign.workspace
    (workspace / "memory").mkdir(parents=True)
    (workspace / "kernel.py").write_text("# v0\n", encoding="utf-8")
    (workspace / "README.md").write_text("# test\n", encoding="utf-8")
    (workspace / "memory/v0.json").write_text(
        json.dumps(
            {
                "version": "v0",
                "performance": {"latency_us": 10.0},
                "correctness": {"status": "PASS"},
                "quality_gate": {"result": "PASS"},
            }
        ),
        encoding="utf-8",
    )
    _git(workspace, "init", "-q")
    _git(workspace, "config", "user.email", "test@local")
    _git(workspace, "config", "user.name", "test")
    _git(workspace, "add", "-A")
    _git(workspace, "commit", "-q", "-m", "v0")
    return campaign


class DefaultStopPolicyTest(unittest.TestCase):
    def test_default_policy_preserves_peak_utilization_semantics_and_reason(self) -> None:
        campaign = optimize.Campaign(
            name="demo",
            kernel_demo="/tmp/reference.py",
            platform="H20",
            framework="Triton",
            target_util=90.0,
        )
        policy = DefaultStopPolicy()

        below = policy.evaluate_accepted_iteration(
            campaign,
            1,
            {
                "performance": {
                    "tflops_peak_utilization_pct": 89.9,
                    "bandwidth_peak_utilization_pct": 80.0,
                }
            },
        )
        reached = policy.evaluate_accepted_iteration(
            campaign,
            2,
            {
                "performance": {
                    "tflops_peak_utilization_pct": 70.0,
                    "bandwidth_peak_utilization_pct": 91.25,
                }
            },
        )

        self.assertEqual(below.status, StopDecisionStatus.CONTINUE)
        self.assertEqual(reached.status, StopDecisionStatus.SUCCESS)
        self.assertEqual(reached.reason, "success: peak_util 91.2% >= 90%")


class CampaignStopPolicyIntegrationTest(unittest.TestCase):
    def test_initial_candidate_retries_infra_before_budget_check(self) -> None:
        class RetryPolicy:
            def __init__(self) -> None:
                self.calls = 0

            def evaluate_accepted_iteration(self, _campaign, _version, _memory):
                self.calls += 1
                if self.calls == 1:
                    return StopDecision(
                        StopDecisionStatus.INFRA_ERROR,
                        "temporary verifier failure",
                    )
                return StopDecision(
                    StopDecisionStatus.SUCCESS,
                    "success: initial candidate verified",
                )

        with tempfile.TemporaryDirectory(prefix="initial-stop-policy-") as temp_dir:
            policy = RetryPolicy()
            campaign = _campaign(Path(temp_dir), policy)
            campaign.max_iters = 0
            campaign.evaluate_initial_stop = True
            campaign.stop_policy_infra_retries = 1
            with (
                mock.patch.object(campaign, "_link_runtime"),
                mock.patch.object(optimize, "mask_half_memory"),
                mock.patch.object(campaign, "_finish", side_effect=lambda reason: reason),
            ):
                reason = campaign.run()

        self.assertEqual(reason, "success: initial candidate verified")
        self.assertEqual(policy.calls, 2)

    def test_exhausted_verifier_retries_end_as_infrastructure_error(self) -> None:
        policy = RecordingStopPolicy(
            StopDecision(StopDecisionStatus.INFRA_ERROR, "verifier unavailable")
        )
        with tempfile.TemporaryDirectory(prefix="stop-policy-infra-terminal-") as temp_dir:
            campaign = _campaign(Path(temp_dir), policy)
            campaign.max_iters = 0
            campaign.evaluate_initial_stop = True
            campaign.stop_policy_infra_retries = 1
            campaign.abort_on_stop_policy_infra = True
            with (
                mock.patch.object(campaign, "_link_runtime"),
                mock.patch.object(optimize, "mask_half_memory"),
                mock.patch.object(campaign, "_finish", side_effect=lambda reason: reason),
            ):
                reason = campaign.run()

        self.assertTrue(reason.startswith("infra: stop-policy"))
        self.assertEqual(len(policy.calls), 2)

    def test_iteration_acceptance_reverts_regressing_commit(self) -> None:
        with tempfile.TemporaryDirectory(prefix="iteration-acceptance-") as temp_dir:
            campaign = _campaign(Path(temp_dir))
            campaign.iteration_acceptance = (
                lambda _campaign, _version, memory, previous: (
                    "latency did not improve"
                    if memory["performance"]["latency_us"] >= float(previous or 10.0)
                    else None
                )
            )

            def regressing_iteration(workspace: Path, _prompt: str, **_kwargs: object):
                (workspace / "kernel.py").write_text("# regressed\n", encoding="utf-8")
                (workspace / "memory/v1.json").write_text(
                    json.dumps(
                        {
                            "version": "v1",
                            "performance": {"latency_us": 11.0},
                            "correctness": {"status": "PASS"},
                            "quality_gate": {"result": "PASS"},
                        }
                    ),
                    encoding="utf-8",
                )
                _git(workspace, "add", "kernel.py", "memory/v1.json")
                _git(workspace, "commit", "-q", "-m", "regression")
                return optimize.SessionResult(0, False, 100, "", "", "sid")

            with (
                mock.patch.object(campaign, "_link_runtime"),
                mock.patch.object(optimize, "run_session", side_effect=regressing_iteration),
                mock.patch.object(optimize, "mask_half_memory"),
                mock.patch.object(campaign, "_finish", side_effect=lambda reason: reason),
            ):
                reason = campaign.run()
            kernel = (campaign.workspace / "kernel.py").read_text(encoding="utf-8")
            memory = json.loads(
                (campaign.workspace / "memory/v1.json").read_text(encoding="utf-8")
            )

        self.assertEqual(reason, "budget: max-iters")
        self.assertEqual(kernel, "# v0\n")
        self.assertEqual(memory["quality_gate"]["result"], "FAIL")
        self.assertIn("latency did not improve", memory["quality_gate"]["failure_reason"])

    def _run_with(self, decision: StopDecision) -> tuple[str, RecordingStopPolicy, int]:
        with tempfile.TemporaryDirectory(prefix="stop-policy-") as temp_dir:
            policy = RecordingStopPolicy(decision)
            campaign = _campaign(Path(temp_dir), policy)

            def accepted_iteration(workspace: Path, _prompt: str, **_kwargs: object):
                (workspace / "kernel.py").write_text("# v1 accepted\n", encoding="utf-8")
                (workspace / "memory/v1.json").write_text(
                    json.dumps(
                        {
                            "version": "v1",
                            "performance": {
                                "latency_us": 9.0,
                                "tflops_peak_utilization_pct": 50.0,
                            },
                            "correctness": {"status": "PASS"},
                            "quality_gate": {"result": "PASS"},
                        }
                    ),
                    encoding="utf-8",
                )
                _git(workspace, "add", "kernel.py", "memory/v1.json")
                _git(workspace, "commit", "-q", "-m", "v1")
                return optimize.SessionResult(0, False, 100, "", "", "sid")

            with (
                mock.patch.object(campaign, "_link_runtime"),
                mock.patch.object(optimize, "run_session", side_effect=accepted_iteration),
                mock.patch.object(optimize, "mask_half_memory"),
                mock.patch.object(campaign, "_finish", side_effect=lambda reason: reason),
            ):
                reason = campaign.run()
            stall = optimize.read_stall(campaign.workspace) or 0
            calls = list(policy.calls)

        expected_calls = 2 if decision.status == StopDecisionStatus.INFRA_ERROR else 1
        self.assertEqual(len(calls), expected_calls)
        self.assertEqual(calls[0][0], 1)
        self.assertEqual(calls[0][1]["performance"]["latency_us"], 9.0)
        return reason, policy, stall

    def test_custom_success_stops_with_policy_reason(self) -> None:
        reason, _policy, stall = self._run_with(
            StopDecision(StopDecisionStatus.SUCCESS, "success: teacher ABBA passed")
        )
        self.assertEqual(reason, "success: teacher ABBA passed")
        self.assertEqual(stall, 0)

    def test_custom_continue_keeps_the_accepted_iteration_and_hits_budget(self) -> None:
        reason, _policy, stall = self._run_with(StopDecision.continue_())
        self.assertEqual(reason, "budget: max-iters")
        self.assertEqual(stall, 0)

    def test_policy_infrastructure_error_does_not_turn_a_win_into_a_stall(self) -> None:
        reason, _policy, stall = self._run_with(
            StopDecision(StopDecisionStatus.INFRA_ERROR, "teacher verifier unavailable")
        )
        self.assertEqual(reason, "budget: max-iters")
        self.assertEqual(stall, 0)

    def test_invalid_policy_result_fails_closed(self) -> None:
        class InvalidPolicy:
            def evaluate_accepted_iteration(self, _campaign, _version, _memory):
                return "stop"

        campaign = optimize.Campaign(
            name="demo",
            kernel_demo="/tmp/reference.py",
            platform="H20",
            framework="Triton",
            stop_policy=InvalidPolicy(),
        )
        with self.assertRaisesRegex(TypeError, "StopDecision"):
            campaign._accepted_stop_decision(1, {})


if __name__ == "__main__":
    unittest.main()
