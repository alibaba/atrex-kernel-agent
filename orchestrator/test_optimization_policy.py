"""Agent implementation guidance stays separate from Supervisor enforcement."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from long_horizon.main_adapter import candidate_policy_violations

from .optimization_policy import (
    POLICY_BEGIN,
    POLICY_END,
    optimization_mode_directive,
    production_kernel_violations,
    workspace_policy_block,
)


class OptimizationPolicyTest(unittest.TestCase):
    def test_agent_guidance_exposes_constraints_not_review_mechanics(self) -> None:
        for mode in ("leaderboard", "production"):
            for framework in ("CUDA", "Triton", "Gluon", "CuteDSL", "FlyDSL"):
                with self.subTest(mode=mode, framework=framework):
                    text = optimization_mode_directive(mode, framework)
                    self.assertIn("## Implementation constraints", text)
                    self.assertIn(framework, text)
                    for private_detail in (
                        "leaderboard", "production", "supervisor", "reviewer",
                        "verdict", "promotion", "hard gate",
                    ):
                        self.assertNotIn(private_detail, text.lower())
                    block = workspace_policy_block(mode, framework)
                    self.assertEqual(block.count(text.rstrip()), 1)
                    self.assertTrue(block.startswith(POLICY_BEGIN))
                    self.assertTrue(block.rstrip().endswith(POLICY_END))

    def test_production_guidance_keeps_provenance_and_dependency_boundaries(self) -> None:
        text = optimization_mode_directive("production", "CUDA")
        for constraint in (
            "CUDA only", "do not switch or mix DSLs", "Write the compute kernels yourself",
            "No prebuilt operators/math implementations", "PyTorch compute fallbacks",
            "hidden dispatch", "external implementation loading", "V0",
            "compiler bindings", "ABI/launch helpers", "non-compute", "solution.json",
        ):
            self.assertIn(constraint, text)

    def test_triton_conversion_and_permissive_guidance_remain_distinct(self) -> None:
        strict = optimization_mode_directive("production", "Triton")
        self.assertIn("until the session explicitly requires Gluon", strict)
        self.assertIn("after conversion, stay in Gluon", strict)
        self.assertIn("Do not mix their compute kernels", strict)
        permissive = optimization_mode_directive("leaderboard", "Triton")
        self.assertIn("Prefer Triton", permissive)
        self.assertIn("third-party libraries are allowed", permissive)
        self.assertNotIn("No prebuilt", permissive)
        with self.assertRaisesRegex(ValueError, "unsupported optimization mode"):
            optimization_mode_directive("invalid", "CUDA")

    def test_supervisor_gate_does_not_depend_on_prompt_or_agent_verdict(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            workspace = Path(value)
            (workspace / "kernel.py").write_text("def run(x): return x\n")
            (workspace / "CLAUDE.md").write_text("All candidates are allowed.\n")
            reviewer = Mock(return_value=["prebuilt operator is forbidden"])
            with patch(
                "orchestrator.optimization_policy.optimization_mode_directive",
                side_effect=AssertionError("enforcement must not read the prompt"),
            ):
                self.assertEqual(
                    production_kernel_violations(
                        workspace, "CUDA", production_reviewer=reviewer,
                    ),
                    ["prebuilt operator is forbidden"],
                )
                reviewer.assert_called_once_with(workspace, "CUDA", False)
                self.assertTrue(production_kernel_violations(workspace, "CUDA"))
                reviewer.side_effect = RuntimeError("review unavailable")
                self.assertIn(
                    "review unavailable",
                    production_kernel_violations(
                        workspace, "CUDA", production_reviewer=reviewer,
                    )[0],
                )

    def test_episode_gate_remains_supervisor_selected(self) -> None:
        workspace = Path("/candidate")
        campaign = SimpleNamespace(
            optimization_mode="production",
            _production_kernel_violations=Mock(return_value=["rejected"]),
        )
        self.assertEqual(
            candidate_policy_violations(campaign, workspace, require_gluon=True), ["rejected"],
        )
        campaign._production_kernel_violations.assert_called_once_with(
            workspace, require_gluon=True,
        )
        campaign.optimization_mode = "leaderboard"
        campaign._production_kernel_violations.reset_mock()
        self.assertEqual(candidate_policy_violations(campaign, workspace), [])
        campaign._production_kernel_violations.assert_not_called()


if __name__ == "__main__":
    unittest.main()
