from __future__ import annotations

import argparse
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import PropertyMock, patch

from long_horizon import main_adapter
from orchestrator.campaign import Campaign
from orchestrator.optimization_policy import install_workspace_policy
from orchestrator.sandbox_config import distributed_evaluation_world_size
from tools import sandbox


def metadata(count: object) -> dict:
    return {"benchmark_contract": {"distributed_evaluation": {
        "launcher": "torchrun", "backend": "nccl", "world_size": count,
    }}}


class MultiGpuSourcePolicyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.campaign = Campaign(
            name="communication", kernel_demo=str(self.root / "operator/kernel.py"),
            platform="test", framework="Cuda", optimization_mode="production",
            work_dir=str(self.root),
        )
        self.workspace = self.campaign.workspace
        self.workspace.mkdir()
        for name in ("kernel.py", "reference.py", "input.py"):
            (self.workspace / name).write_text("import torch.distributed as dist\n")
        (self.workspace / "shapes.json").write_text('{"0": {"input_kwargs": {}}}')
        (self.workspace / "solution.json").write_text('{"languages": ["python"]}')

    def set_count(self, count: object) -> None:
        (self.workspace / "metadata.json").write_text(json.dumps(metadata(count)))

    def test_same_count_controls_gate_and_actual_agate_request(self) -> None:
        args = argparse.Namespace(url="", gateway_profile=None, timeout=60, env=[], hardware="test")
        for count in (1, 2, 4, 8):
            with self.subTest(count=count):
                if count == 1:
                    (self.workspace / "metadata.json").write_text("{}")
                else:
                    self.set_count(count)
                request = sandbox._typed_request(
                    self.workspace, "test", 60, [], ["python", "test_kernel.py"], "run"
                )
                command = sandbox._typed_agate_command(
                    "agate", args, self.workspace, "run", request, 0
                )
                self.assertEqual(request["spec"].get("num_gpus", 1), count)
                self.assertEqual(self.campaign.source_checks_enabled, count == 1)
                if count > 1:
                    self.assertEqual(command[command.index("--num-gpus") + 1], str(count))
                else:
                    self.assertNotIn("--num-gpus", command)

    def test_single_gpu_still_inspects_source(self) -> None:
        with patch("orchestrator.campaign.production_kernel_violations",
                   return_value=["source rejected"]) as inspect_source:
            self.assertEqual(self.campaign._production_kernel_violations(), ["source rejected"])
            inspect_source.assert_called_once()

    def test_required_gpu_count_must_match_launcher(self) -> None:
        contract = metadata(4)
        contract["hardware"] = {"required_gpus": 2}
        with self.assertRaises(ValueError):
            distributed_evaluation_world_size(contract)
        with self.assertRaises(ValueError):
            distributed_evaluation_world_size({"hardware": {"required_gpus": 4}})

    def test_pinned_runtime_reaches_api_and_cli(self) -> None:
        self.set_count(4)
        requirements = self.root / "runtime.txt"
        requirements.write_text("atrex-bench==0.1.1.dev1\n")
        args = argparse.Namespace(url="", gateway_profile=None, timeout=60, env=[], hardware="test")
        with patch.dict(os.environ, {"ATREX_SANDBOX_ATREX_BENCH_VERSION": "0.1.1.dev1",
                                     "ATREX_SANDBOX_REQUIREMENTS_FILE": str(requirements)}):
            request = sandbox._typed_request(self.workspace, "test", 60, [],
                                             ["python", "test_kernel.py"], "run")
            command = sandbox._typed_agate_command("agate", args, self.workspace, "run", request, 0)
        self.assertEqual(request["atrex_bench_version"], "0.1.1.dev1")
        self.assertEqual(request["requirements"], ["atrex-bench==0.1.1.dev1"])
        self.assertEqual(command[command.index("--atrex-bench-version") + 1], "0.1.1.dev1")
        self.assertEqual(command[command.index("--requirements") + 1], str(requirements))

    def test_multi_gpu_skips_static_and_independent_source_review(self) -> None:
        self.set_count(4)
        with patch("orchestrator.campaign.production_kernel_violations",
                   side_effect=AssertionError("source inspected")):
            self.assertEqual(self.campaign._production_kernel_violations(), [])
            self.assertEqual(main_adapter.candidate_policy_violations(
                self.campaign, self.workspace, require_gluon=True), [])

    def test_ssh_does_not_skip_source_checks(self) -> None:
        self.set_count(4)
        self.campaign.sandbox_ssh = "gpu-host"
        self.assertTrue(self.campaign.source_checks_enabled)

    def test_private_evaluator_metadata_overrides_candidate_metadata(self) -> None:
        reference = self.root / "private"
        reference.mkdir()
        (reference / "metadata.json").write_text("{}")
        self.set_count(4)
        with patch.object(Campaign, "private_reference_dir", new_callable=PropertyMock,
                          return_value=reference):
            self.assertTrue(self.campaign.source_checks_enabled)
            (reference / "metadata.json").write_text(json.dumps(metadata(2)))
            (self.workspace / "metadata.json").unlink()
            self.assertFalse(self.campaign.source_checks_enabled)

    def test_operator_name_and_solution_do_not_enable_bypass(self) -> None:
        (self.workspace / "solution.json").write_text(
            '{"num_gpus": 8, "dependencies": ["torch.distributed"]}'
        )
        self.assertTrue(self.campaign.source_checks_enabled)

    def test_invalid_contracts_fail_in_both_policy_and_transport(self) -> None:
        for count in (True, "4", 0, -1, 1, 2.5):
            with self.subTest(count=count):
                self.set_count(count)
                with self.assertRaises(ValueError):
                    _ = self.campaign.source_checks_enabled
                with self.assertRaises(ValueError):
                    sandbox._workspace_num_gpus(self.workspace)
        for key, value in (("launcher", "python"), ("backend", "gloo")):
            contract = metadata(4)
            contract["benchmark_contract"]["distributed_evaluation"][key] = value
            with self.assertRaises(ValueError):
                distributed_evaluation_world_size(contract)

    def test_multi_gpu_reuses_validated_v0(self) -> None:
        self.set_count(4)
        with patch("orchestrator.campaign.latest_version", return_value=0), \
             patch("orchestrator.campaign.git_head", return_value="base"), \
             patch("orchestrator.campaign.resolve_framework_baseline_commit", return_value=(None, None)), \
             patch("orchestrator.campaign.head_kernel_is_initial_baseline", return_value=True), \
             patch("orchestrator.campaign.git_worktree_blob", return_value="blob"), \
             patch("orchestrator.campaign.git_path_blob", return_value="blob"), \
             patch("orchestrator.campaign.production_structure_violations",
                   side_effect=AssertionError("source inspected")):
            action, _ = self.campaign._framework_baseline_decision()
            self.assertEqual(action, "pin")

    def test_baseline_recovery_still_checks_immutable_inputs(self) -> None:
        self.set_count(4)
        with patch("orchestrator.campaign.git_worktree_blob", return_value="changed"), \
             patch("orchestrator.campaign.git_path_blob", return_value="original"), \
             patch("orchestrator.campaign.IMMUTABLE_BASELINE_PATHS", ("input.py",)), \
             patch("orchestrator.campaign.production_structure_violations",
                   side_effect=AssertionError("source inspected")):
            problem = self.campaign._framework_baseline_problem(
                "v0", "base", include_policy_review=False
            )
            self.assertIn("modified immutable ground truth: input.py", problem)

    def test_multi_gpu_correctness_failure_still_blocks_baseline(self) -> None:
        self.set_count(4)
        with patch.object(Campaign, "_validate_framework_baseline",
                          return_value=(None, "correctness failed")) as validate, \
             patch.object(Campaign, "_review_production_candidate",
                          side_effect=AssertionError("source reviewed")):
            result, error = self.campaign._framework_baseline_external_gates(1)
            self.assertIsNone(result)
            self.assertEqual(error, "correctness failed")
            validate.assert_called_once_with(1)

    def test_multi_gpu_numerical_probes_still_run(self) -> None:
        self.set_count(2)
        with patch.object(Campaign, "_validate_framework_baseline", return_value=({"all_pass": True}, "")), \
             patch.object(Campaign, "_supplemental_numerical_feedback",
                          return_value="numerical probe failed") as probes:
            _, error = self.campaign._framework_baseline_external_gates(1)
            self.assertEqual(error, "numerical probe failed")
            probes.assert_called_once_with(standard_correctness_passed=True)

    def test_prompt_and_resumed_workspace_match_source_check_mode(self) -> None:
        for count in (2, 1):
            if count == 2:
                self.set_count(2)
            else:
                (self.workspace / "metadata.json").unlink()
            enabled = self.campaign.source_checks_enabled
            install_workspace_policy(self.workspace, "production", "Cuda", source_checks_enabled=enabled)
            state = json.loads((self.workspace / ".orchestrator_mode.json").read_text())
            self.assertEqual(state["source_checks_enabled"], enabled)
            prompt = self.campaign._mode_directive()
            self.assertIn(prompt.strip(), (self.workspace / "CLAUDE.md").read_text())
            if not enabled:
                self.assertIn("source-policy review are disabled", prompt)
                self.assertFalse(main_adapter.conversion_required(self.campaign, 100, self.workspace))


if __name__ == "__main__":
    unittest.main()
