from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from orchestrator.optimization_policy import (
    install_workspace_policy,
    optimization_mode_directive,
    production_kernel_violations,
    reject_production_commit,
)


class OptimizationPolicyTest(unittest.TestCase):
    def _workspace(self, root: Path, kernel: str, dependencies: list[str]) -> Path:
        workspace = root / "workspace"
        workspace.mkdir()
        (workspace / "kernel.py").write_text(kernel, encoding="utf-8")
        (workspace / "solution.json").write_text(
            json.dumps({"spec": {"dependencies": dependencies}}),
            encoding="utf-8",
        )
        return workspace

    def test_leaderboard_preserves_permissive_guidance(self) -> None:
        directive = optimization_mode_directive("leaderboard", "Triton")
        self.assertIn("third-party helper/kernel libraries may be used", directive)
        self.assertIn("mixed/alternate implementations are allowed", directive)

    def test_workspace_policy_is_idempotent_and_rejects_mode_switch(self) -> None:
        with tempfile.TemporaryDirectory(prefix="optimization-policy-") as temp_dir:
            workspace = Path(temp_dir)
            (workspace / "CLAUDE.md").write_text("# Base policy\n", encoding="utf-8")
            install_workspace_policy(workspace, "production", "Triton")
            first = (workspace / "CLAUDE.md").read_text(encoding="utf-8")
            install_workspace_policy(workspace, "production", "Triton")
            self.assertEqual(first, (workspace / "CLAUDE.md").read_text(encoding="utf-8"))
            self.assertIn("production (hard gate)", first)
            self.assertIn("exactly **Triton**", first)
            self.assertIn("/.orchestrator_mode.json", (workspace / ".gitignore").read_text())
            with self.assertRaisesRegex(RuntimeError, "workspace policy mismatch"):
                install_workspace_policy(workspace, "leaderboard", "Triton")

    def test_exact_framework_candidates_pass(self) -> None:
        candidates = {
            "Triton": (
                "import torch\nimport triton\nimport triton.language as tl\n"
                "@triton.jit\ndef kernel(x):\n    return\n",
                ["torch", "triton"],
            ),
            "CuteDSL": (
                "import torch\nimport cutlass\nimport cutlass.cute as cute\n"
                "@cute.kernel\ndef kernel(x):\n    return\n",
                ["torch", "nvidia-cutlass-dsl"],
            ),
            "Cuda": (
                "import torch\nfrom torch.utils.cpp_extension import load_inline\n"
                "SOURCE = r'''__global__ void kernel(float* x) {}'''\n",
                ["torch"],
            ),
            "FlyDSL": (
                "import torch\nimport flydsl\nimport flydsl.compiler as flyc\n"
                "def kernel(x):\n    return\n",
                ["torch", "flydsl"],
            ),
        }
        for framework, (kernel, dependencies) in candidates.items():
            with self.subTest(framework=framework):
                with tempfile.TemporaryDirectory(prefix="production-candidate-") as temp_dir:
                    workspace = self._workspace(Path(temp_dir), kernel, dependencies)
                    self.assertEqual(production_kernel_violations(workspace, framework), [])

    def test_third_party_and_wrong_framework_are_rejected(self) -> None:
        kernel = (
            "import torch\nimport triton\nimport triton.language as tl\n"
            "from flash_attn import flash_attn_func\n"
            "@triton.jit\ndef kernel(x):\n    return\n"
        )
        with tempfile.TemporaryDirectory(prefix="production-reject-") as temp_dir:
            workspace = self._workspace(Path(temp_dir), kernel, ["torch", "triton", "flash-attn"])
            violations = production_kernel_violations(workspace, "Triton")
            self.assertTrue(any("flash_attn" in error for error in violations))
            self.assertTrue(any("flash-attn" in error for error in violations))
            wrong_framework = production_kernel_violations(workspace, "CuteDSL")
            self.assertIn("missing CuteDSL implementation", wrong_framework)
            self.assertIn("mixed/alternate framework marker is forbidden: triton", wrong_framework)

    def test_prose_naming_a_third_party_library_is_not_a_dependency(self) -> None:
        described = (
            "from __future__ import annotations\n\n"
            '"""vLLM-style paged causal GQA attention implemented in CuteDSL."""\n\n'
            "import torch\nimport cutlass\nimport cutlass.cute as cute  # not a flash_attn port\n\n"
            "@cute.kernel\ndef kernel(x):\n    return\n"
        )
        with tempfile.TemporaryDirectory(prefix="production-prose-") as temp_dir:
            workspace = self._workspace(Path(temp_dir), described, ["torch", "nvidia-cutlass-dsl"])
            self.assertEqual(production_kernel_violations(workspace, "CuteDSL"), [])

    def test_real_third_party_import_is_still_rejected(self) -> None:
        used = (
            "import torch\nimport vllm\nimport cutlass\nimport cutlass.cute as cute\n"
            "@cute.kernel\ndef kernel(x):\n    return\n"
            "def run(x):\n    return vllm.attention(x)\n"
        )
        with tempfile.TemporaryDirectory(prefix="production-import-") as temp_dir:
            workspace = self._workspace(Path(temp_dir), used, ["torch", "nvidia-cutlass-dsl"])
            violations = production_kernel_violations(workspace, "CuteDSL")
            self.assertIn("third-party import is not allowed in production mode: vllm", violations)
            self.assertIn("third-party kernel/operator library reference", violations)

    def test_prose_cannot_satisfy_a_framework_marker(self) -> None:
        commented = "import torch\n\n\n# implemented with triton under the hood\ndef run(x):\n    return x\n"
        with tempfile.TemporaryDirectory(prefix="production-marker-") as temp_dir:
            workspace = self._workspace(Path(temp_dir), commented, ["torch"])
            self.assertIn(
                "missing Triton implementation/import",
                production_kernel_violations(workspace, "Triton"),
            )

    def test_unused_framework_marker_does_not_hide_pytorch_compute(self) -> None:
        kernel = (
            "import torch\nimport triton\nimport triton.language as tl\n"
            "@triton.jit\ndef kernel(x):\n    return\n"
            "def run(x, y):\n    y[:] = torch.matmul(x, x)\n"
        )
        with tempfile.TemporaryDirectory(prefix="production-pytorch-") as temp_dir:
            workspace = self._workspace(Path(temp_dir), kernel, ["torch", "triton"])
            violations = production_kernel_violations(workspace, "Triton")
            self.assertIn(
                "PyTorch compute call is forbidden in production candidate: torch.matmul",
                violations,
            )

    def test_policy_rejection_reverts_kernel_commit_and_keeps_memory(self) -> None:
        with tempfile.TemporaryDirectory(prefix="production-revert-") as temp_dir:
            workspace = Path(temp_dir)
            (workspace / "memory").mkdir()
            (workspace / "kernel.py").write_text(
                "import torch\nimport triton\nimport triton.language as tl\n"
                "@triton.jit\ndef kernel(x):\n    return\n",
                encoding="utf-8",
            )
            subprocess.run(["git", "init"], cwd=workspace, check=True, stdout=subprocess.DEVNULL)
            subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=workspace, check=True)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=workspace, check=True)
            subprocess.run(["git", "add", "kernel.py"], cwd=workspace, check=True)
            subprocess.run(["git", "commit", "-m", "baseline"], cwd=workspace, check=True, stdout=subprocess.DEVNULL)
            baseline = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=workspace, check=True, capture_output=True, text=True
            ).stdout.strip()

            (workspace / "kernel.py").write_text("from flash_attn import flash_attn_func\n", encoding="utf-8")
            subprocess.run(["git", "add", "kernel.py"], cwd=workspace, check=True)
            subprocess.run(["git", "commit", "-m", "bad production kernel"], cwd=workspace, check=True, stdout=subprocess.DEVNULL)
            reject_production_commit(workspace, 1, baseline, ["third-party import: flash_attn"])

            head = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=workspace, check=True, capture_output=True, text=True
            ).stdout.strip()
            self.assertEqual(head, baseline)
            memory = json.loads((workspace / "memory" / "v1.json").read_text(encoding="utf-8"))
            self.assertEqual(memory["quality_gate"]["result"], "FAIL")
            self.assertEqual(memory["optimization"]["action_category"], "production_policy_rejection")


if __name__ == "__main__":
    unittest.main()
