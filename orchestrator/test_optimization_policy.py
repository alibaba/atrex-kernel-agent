from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from orchestrator.optimization_policy import (
    DependencyReviewSignal,
    production_kernel_violations,
)
from orchestrator.campaign import Campaign
from orchestrator.session_io import SessionResult, _validate_dependency_review


_CUDA_KERNEL = '''
from cuda.bindings import nvrtc

CUDA_SOURCE = r"""
extern "C" __global__ void generalized_kernel(const half* x, half* y, int n) {
    int index = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (index < n) y[index] = x[index];
}
"""

def run(x, y):
    return nvrtc.nvrtcVersion()
'''


class ProductionOptimizationPolicyTests(unittest.TestCase):
    def _workspace(self) -> tempfile.TemporaryDirectory[str]:
        return tempfile.TemporaryDirectory(prefix="atrex-policy-")

    def test_cuda_kernel_py_with_nvrtc_loader_is_accepted(self) -> None:
        with self._workspace() as directory:
            workspace = Path(directory)
            (workspace / "kernel.py").write_text(_CUDA_KERNEL, encoding="utf-8")

            self.assertEqual(production_kernel_violations(workspace, "Cuda"), [])

    def test_schema_v3_embedded_dispatch_is_accepted(self) -> None:
        with self._workspace() as directory:
            workspace = Path(directory)
            (workspace / "kernel.py").write_text(_CUDA_KERNEL, encoding="utf-8")
            (workspace / "aggregate_dispatch.json").write_text(
                json.dumps(
                    {
                        "schema_version": 3,
                        "mode": "deterministic_dispatch",
                        "source_layout": "embedded_single_file",
                        "modules": {"range_0": {"embedded": True}},
                    }
                ),
                encoding="utf-8",
            )

            self.assertEqual(production_kernel_violations(workspace, "Cuda"), [])

    def test_native_cuda_entry_reports_bucket_composition_problem(self) -> None:
        with self._workspace() as directory:
            workspace = Path(directory)
            (workspace / "kernel.py").write_text(
                "def run(x, y):\n    raise RuntimeError('native entry')\n",
                encoding="utf-8",
            )
            (workspace / "kernel.cu").write_text(
                'extern "C" __global__ void kernel() {}\n', encoding="utf-8"
            )
            (workspace / "solution.json").write_text(
                json.dumps(
                    {
                        "spec": {
                            "languages": ["cuda_cpp"],
                            "entry_point": "kernel.cu::run",
                            "dependencies": [],
                        },
                        "sources": [{"path": "kernel.cu"}],
                    }
                ),
                encoding="utf-8",
            )

            violations = production_kernel_violations(workspace, "Cuda")
            self.assertTrue(
                any("cannot be versioned or embedded" in item for item in violations),
                violations,
            )

    def test_unlisted_import_requires_independent_review(self) -> None:
        with self._workspace() as directory:
            workspace = Path(directory)
            (workspace / "kernel.py").write_text(
                "import vendor_headers\n" + _CUDA_KERNEL,
                encoding="utf-8",
            )

            violations = production_kernel_violations(workspace, "Cuda")

            self.assertEqual(len(violations), 1)
            self.assertIn("requires independent agent review", violations[0])
            self.assertIn("import:vendor_headers", violations[0])
            self.assertNotIn("is not allowed", violations[0])

    def test_independent_reviewer_can_allow_toolchain_import(self) -> None:
        with self._workspace() as directory:
            workspace = Path(directory)
            (workspace / "kernel.py").write_text(
                "import vendor_headers\n" + _CUDA_KERNEL,
                encoding="utf-8",
            )
            observed = []

            def reviewer(path, framework, signals):
                observed.extend(signals)
                self.assertEqual(path, workspace)
                self.assertEqual(framework, "Cuda")
                return []

            violations = production_kernel_violations(
                workspace,
                "Cuda",
                dependency_reviewer=reviewer,
            )

            self.assertEqual(violations, [])
            self.assertEqual([signal.id for signal in observed], ["import:vendor_headers"])

    def test_independent_reviewer_rejection_is_enforced(self) -> None:
        with self._workspace() as directory:
            workspace = Path(directory)
            (workspace / "kernel.py").write_text(
                "import vendor_operator\n" + _CUDA_KERNEL,
                encoding="utf-8",
            )

            violations = production_kernel_violations(
                workspace,
                "Cuda",
                dependency_reviewer=lambda _path, _framework, signals: [
                    "review rejected " + signals[0].id
                ],
            )

            self.assertIn("review rejected import:vendor_operator", violations)

    def test_solution_dependency_is_delegated_to_reviewer(self) -> None:
        with self._workspace() as directory:
            workspace = Path(directory)
            (workspace / "kernel.py").write_text(_CUDA_KERNEL, encoding="utf-8")
            (workspace / "solution.json").write_text(
                json.dumps(
                    {
                        "spec": {
                            "dependencies": ["torch", "vendor-headers>=1.0"],
                        }
                    }
                ),
                encoding="utf-8",
            )
            observed = []

            violations = production_kernel_violations(
                workspace,
                "Cuda",
                dependency_reviewer=lambda _path, _framework, signals: (
                    observed.extend(signals) or []
                ),
            )

            self.assertEqual(violations, [])
            self.assertEqual(
                [signal.id for signal in observed],
                ["solution_dependency:vendor-headers>=1.0"],
            )

    def test_dynamic_loading_remains_a_mechanical_violation(self) -> None:
        with self._workspace() as directory:
            workspace = Path(directory)
            (workspace / "kernel.py").write_text(
                "import ctypes\n" + _CUDA_KERNEL,
                encoding="utf-8",
            )

            violations = production_kernel_violations(
                workspace,
                "Cuda",
                dependency_reviewer=lambda _path, _framework, _signals: [],
            )

            self.assertTrue(
                any("dynamic external-code loading" in item for item in violations),
                violations,
            )


class IndependentDependencyReviewTests(unittest.TestCase):
    def test_structured_allow_verdict_is_accepted(self) -> None:
        signals = (
            DependencyReviewSignal("import:vendor_headers", "import", "vendor_headers"),
        )
        errors, summary = _validate_dependency_review(
            {
                "schema_version": 1,
                "verdict": "allow",
                "items": [
                    {
                        "id": "import:vendor_headers",
                        "decision": "allow",
                        "category": "toolchain_plumbing",
                        "reason": "Only locates headers for the embedded CUDA source.",
                        "evidence": ["candidate/kernel.py:1"],
                    }
                ],
                "summary": "The dependency supplies build-time headers only.",
            },
            signals,
        )

        self.assertEqual(errors, [])
        self.assertIn("headers only", summary)

    def test_structured_reject_verdict_becomes_policy_error(self) -> None:
        signals = (
            DependencyReviewSignal("import:vendor_op", "import", "vendor_op"),
        )
        errors, _summary = _validate_dependency_review(
            {
                "schema_version": 1,
                "verdict": "reject",
                "items": [
                    {
                        "id": "import:vendor_op",
                        "decision": "reject",
                        "category": "prebuilt_compute",
                        "reason": "run() delegates GEMM to the package operator.",
                        "evidence": ["candidate/kernel.py:19"],
                    }
                ],
                "summary": "The candidate delegates its computation.",
            },
            signals,
        )

        self.assertEqual(len(errors), 1)
        self.assertIn("delegates GEMM", errors[0])

    def test_campaign_uses_fresh_agent_and_caches_the_verdict(self) -> None:
        with tempfile.TemporaryDirectory(prefix="atrex-policy-agent-") as directory:
            workspace = Path(directory) / "candidate"
            workspace.mkdir()
            (workspace / "kernel.py").write_text(
                "import vendor_headers\n" + _CUDA_KERNEL,
                encoding="utf-8",
            )
            campaign = Campaign(
                name="policy-test",
                kernel_demo="unused.py",
                platform="test",
                framework="Cuda",
                work_dir=directory,
                optimization_mode="production",
                agent_cli="claude",
            )

            def fake_session(review_workspace, _prompt, **_kwargs):
                self.assertEqual(_kwargs["reasoning_effort"], "low")
                self.assertFalse(_kwargs["agent_plugins"])
                request = json.loads(
                    (review_workspace / "review_request.json").read_text(encoding="utf-8")
                )
                signal = request["signals"][0]
                (review_workspace / "dependency_review.json").write_text(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "verdict": "allow",
                            "items": [
                                {
                                    "id": signal["id"],
                                    "decision": "allow",
                                    "category": "toolchain_plumbing",
                                    "reason": "Only locates headers for self-authored CUDA.",
                                    "evidence": ["candidate/kernel.py:1"],
                                }
                            ],
                            "summary": "Toolchain-only dependency.",
                        }
                    ),
                    encoding="utf-8",
                )
                return SessionResult(0, False, 42, "", "")

            with patch("orchestrator.campaign.run_session", side_effect=fake_session) as runner:
                self.assertEqual(
                    campaign._production_kernel_violations(workspace),
                    [],
                )
                self.assertEqual(
                    campaign._production_kernel_violations(workspace),
                    [],
                )

            self.assertEqual(runner.call_count, 1)
            self.assertEqual(campaign.tokens_spent, 42)

    def test_campaign_fails_closed_when_review_agent_fails(self) -> None:
        with tempfile.TemporaryDirectory(prefix="atrex-policy-agent-") as directory:
            workspace = Path(directory) / "candidate"
            workspace.mkdir()
            (workspace / "kernel.py").write_text(
                "import vendor_headers\n" + _CUDA_KERNEL,
                encoding="utf-8",
            )
            campaign = Campaign(
                name="policy-test",
                kernel_demo="unused.py",
                platform="test",
                framework="Cuda",
                work_dir=directory,
                optimization_mode="production",
                agent_cli="claude",
            )

            with patch(
                "orchestrator.campaign.run_session",
                return_value=SessionResult(1, False, 7, "", "review failed"),
            ):
                violations = campaign._production_kernel_violations(workspace)

            self.assertTrue(
                any("review agent failed" in violation for violation in violations),
                violations,
            )


if __name__ == "__main__":
    unittest.main()
