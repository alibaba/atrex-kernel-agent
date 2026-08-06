from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from orchestrator import optimize
from teacher_distill.benchmark import (
    benchmark_teacher,
    materialize_teacher_workspace,
)
from teacher_distill.bundle import validate_teacher_bundle


class TeacherBenchmarkTest(unittest.TestCase):
    def _bundle(
        self,
        root: Path,
        *,
        compliant: bool = True,
        entry_point: str = "kernel.py::run",
    ):
        bundle = root / "bundle"
        (bundle / "helpers").mkdir(parents=True)
        if compliant and entry_point == "kernel.py::Model":
            kernel = (
                "import torch\nimport cutlass\nimport cutlass.cute as cute\n"
                "@cute.kernel\ndef _kernel(x, out):\n    return\n"
                "class Model(torch.nn.Module):\n"
                "    def forward(self, x):\n        return x\n"
            )
        elif compliant:
            kernel = (
                "import torch\nimport cutlass\nimport cutlass.cute as cute\n"
                "@cute.kernel\ndef _kernel(x, out):\n    return\n"
                "def run(x, out):\n    _kernel(x, out)\n"
            )
        else:
            kernel = "import torch\ndef run(x, out):\n    out[:] = torch.softmax(x, -1)\n"
        (bundle / "kernel.py").write_text(kernel, encoding="utf-8")
        (bundle / "helpers/layout.py").write_text("TILE = 128\n", encoding="utf-8")
        (bundle / "solution.json").write_text(
            json.dumps(
                {
                    "spec": {
                        "languages": ["pytorch", "cutedsl"],
                        "target_hardware": ["H20"],
                        "entry_point": entry_point,
                        "dependencies": ["torch", "nvidia-cutlass-dsl"],
                        "destination_passing_style": True,
                    },
                    "sources": [
                        {"path": "kernel.py"},
                        {"path": "helpers/layout.py"},
                    ],
                }
            ),
            encoding="utf-8",
        )
        (bundle / "provenance.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "operator": {
                        "canonical_id": "gdn_decode",
                        "aliases": ["gdn", "gated_delta_rule"],
                    },
                    "source": {
                        "project": "flashinfer",
                        "revision": "abc123",
                        "license": "Apache-2.0",
                    },
                    "target": {"framework": "CuteDSL", "architecture": "sm90"},
                    "knowledge_deny": {
                        "sources": ["flashinfer"],
                        "paths": [],
                        "tags": ["gdn", "gdn_decode"],
                    },
                }
            ),
            encoding="utf-8",
        )
        return validate_teacher_bundle(bundle, "CuteDSL", "sm90")

    def _sol_op(self, root: Path) -> Path:
        op = root / "op"
        op.mkdir()
        (op / "definition.json").write_text(
            json.dumps(
                {
                    "name": "demo",
                    "inputs": {"x": {}},
                    "outputs": {"out": {}},
                }
            ),
            encoding="utf-8",
        )
        (op / "reference.py").write_text("def run(x): return x\n", encoding="utf-8")
        (op / "workload.jsonl").write_text(
            '\n'.join(
                [
                    json.dumps({"uuid": "shape-a", "axes": {"n": 1}}),
                    json.dumps({"uuid": "shape-b", "axes": {"n": 2}}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        return op

    def _native_op(self, root: Path) -> tuple[Path, Path]:
        op = root / "native-op"
        op.mkdir()
        (op / "reference.py").write_text("class Model: pass\n", encoding="utf-8")
        (op / "input.py").write_text("def make_inputs(): return ()\n", encoding="utf-8")
        (op / "shapes.json").write_text('{"0": {}, "1": {}}\n', encoding="utf-8")
        (op / "metadata.json").write_text('{}\n', encoding="utf-8")
        (op / "roofline.json").write_text('{}\n', encoding="utf-8")
        atrex = root / "atrex-bench"
        (atrex / "scripts").mkdir(parents=True)
        (atrex / "scripts/run_eval.py").write_text("# evaluator\n", encoding="utf-8")
        (atrex / "src/atrex_bench").mkdir(parents=True)
        return op, atrex

    @staticmethod
    def _result(*, all_pass: bool = True, by_shape: dict[str, float] | None = None):
        return {
            "all_pass": all_pass,
            "latency_us_geomean": 100.0,
            "latency_us_arith_mean": 105.0,
            "latency_us_by_shape": by_shape or {"shape-a": 80.0, "shape-b": 125.0},
            "max_abs_err": 0.0,
            "max_rel_err": 0.0,
            "failures": [] if all_pass else ["shape-b=FAILED"],
        }

    @staticmethod
    def _completed(result: dict, returncode: int = 0):
        return subprocess.CompletedProcess(
            args=["sandbox"],
            returncode=returncode,
            stdout=optimize.TEST_RESULT_PREFIX + json.dumps(result) + "\n",
            stderr="",
        )

    def test_materializes_sol_workspace_without_modifying_ground_truth(self) -> None:
        with tempfile.TemporaryDirectory(prefix="teacher-benchmark-sol-") as temp_dir:
            root = Path(temp_dir)
            bundle = self._bundle(root)
            op = self._sol_op(root)
            original = (op / "reference.py").read_bytes()
            workspace = root / "private" / "teacher-workspace"

            materialized = materialize_teacher_workspace(
                bundle,
                op,
                workspace,
                framework="CuteDSL",
            )

            self.assertEqual(materialized.kind, "sol")
            self.assertEqual(materialized.expected_shape_keys, ("shape-a", "shape-b"))
            self.assertEqual((workspace / "reference.py").read_bytes(), original)
            self.assertEqual((workspace / "kernel.py").read_bytes(), (bundle.root / "kernel.py").read_bytes())
            self.assertTrue((workspace / "test_kernel.py").is_file())
            self.assertTrue((workspace / "config.json").is_file())
            self.assertRegex(materialized.workload_hash, r"^[0-9a-f]{64}$")
            self.assertRegex(materialized.evaluator_hash, r"^[0-9a-f]{64}$")
            self.assertRegex(materialized.measurement_config_hash, r"^[0-9a-f]{64}$")

    def test_materializes_native_workspace_with_canonical_adapter(self) -> None:
        with tempfile.TemporaryDirectory(prefix="teacher-benchmark-native-") as temp_dir:
            root = Path(temp_dir)
            bundle = self._bundle(root, entry_point="kernel.py::Model")
            op, atrex = self._native_op(root)
            workspace = root / "private" / "teacher-workspace"

            materialized = materialize_teacher_workspace(
                bundle,
                op,
                workspace,
                framework="CuteDSL",
                atrex_bench_root=atrex,
            )

            self.assertEqual(materialized.kind, "native")
            self.assertEqual(materialized.expected_shape_keys, ("0", "1"))
            self.assertEqual(
                (workspace / "test_kernel.py").read_bytes(),
                optimize.ATREX_BENCH_HARNESS.read_bytes(),
            )
            self.assertTrue((workspace / "atrex-bench").is_symlink())
            self.assertEqual((workspace / "atrex-bench").resolve(), atrex.resolve())

    def test_measurement_context_changes_the_config_hash(self) -> None:
        with tempfile.TemporaryDirectory(prefix="teacher-benchmark-context-") as temp_dir:
            root = Path(temp_dir)
            bundle = self._bundle(root)
            op = self._sol_op(root)
            first = materialize_teacher_workspace(
                bundle,
                op,
                root / "first",
                framework="CuteDSL",
                measurement_context={"sandbox_hardware": "gpu-a", "architecture": "sm90"},
            )
            second = materialize_teacher_workspace(
                bundle,
                op,
                root / "second",
                framework="CuteDSL",
                measurement_context={"sandbox_hardware": "gpu-b", "architecture": "sm90"},
            )

        self.assertNotEqual(
            first.measurement_config_hash,
            second.measurement_config_hash,
        )

    def test_native_evaluator_hash_covers_imported_package_sources(self) -> None:
        with tempfile.TemporaryDirectory(prefix="teacher-benchmark-evaluator-hash-") as temp_dir:
            root = Path(temp_dir)
            bundle = self._bundle(root, entry_point="kernel.py::Model")
            op, atrex = self._native_op(root)
            package_source = atrex / "src/atrex_bench/core.py"
            package_source.write_text("VALUE = 1\n", encoding="utf-8")
            first = materialize_teacher_workspace(
                bundle,
                op,
                root / "first",
                framework="CuteDSL",
                atrex_bench_root=atrex,
            )
            package_source.write_text("VALUE = 2\n", encoding="utf-8")
            second = materialize_teacher_workspace(
                bundle,
                op,
                root / "second",
                framework="CuteDSL",
                atrex_bench_root=atrex,
            )

        self.assertNotEqual(first.evaluator_hash, second.evaluator_hash)

    def test_evaluator_kind_rejects_the_wrong_entry_point_contract(self) -> None:
        with tempfile.TemporaryDirectory(prefix="teacher-benchmark-entry-") as temp_dir:
            root = Path(temp_dir)
            sol_bundle = self._bundle(root, entry_point="kernel.py::Model")
            sol_op = self._sol_op(root)
            with self.assertRaisesRegex(ValueError, "entry_point"):
                materialize_teacher_workspace(
                    sol_bundle,
                    sol_op,
                    root / "sol-workspace",
                    framework="CuteDSL",
                )

        with tempfile.TemporaryDirectory(prefix="teacher-benchmark-native-entry-") as temp_dir:
            root = Path(temp_dir)
            native_bundle = self._bundle(root)
            native_op, atrex = self._native_op(root)
            with self.assertRaisesRegex(ValueError, "entry_point"):
                materialize_teacher_workspace(
                    native_bundle,
                    native_op,
                    root / "native-workspace",
                    framework="CuteDSL",
                    atrex_bench_root=atrex,
                )

    def test_benchmark_runs_single_multi_seed_and_recorded_measurement(self) -> None:
        with tempfile.TemporaryDirectory(prefix="teacher-benchmark-run-") as temp_dir:
            root = Path(temp_dir)
            bundle = self._bundle(root)
            op = self._sol_op(root)
            workspace = root / "private" / "teacher-workspace"
            materialized = materialize_teacher_workspace(bundle, op, workspace, framework="CuteDSL")
            commands: list[list[str]] = []

            def sandbox(_workspace: Path, command: list[str]):
                commands.append(command)
                return self._completed(self._result())

            result = benchmark_teacher(
                materialized,
                framework="CuteDSL",
                sandbox_runner=sandbox,
            )

            self.assertEqual(len(commands), 3)
            self.assertNotIn("--multi-seed", commands[0])
            self.assertIn("--multi-seed", commands[1])
            self.assertNotIn("--multi-seed", commands[2])
            self.assertEqual(result.geomean_latency_us, 100.0)
            self.assertEqual(result.latency_us_by_shape, {"shape-a": 80.0, "shape-b": 125.0})
            for stage in ("single-seed", "multi-seed", "benchmark"):
                self.assertTrue((workspace / "benchmark_runs" / f"{stage}.json").is_file())

    def test_correctness_or_shape_coverage_failure_aborts(self) -> None:
        with tempfile.TemporaryDirectory(prefix="teacher-benchmark-fail-") as temp_dir:
            root = Path(temp_dir)
            bundle = self._bundle(root)
            op = self._sol_op(root)
            materialized = materialize_teacher_workspace(
                bundle, op, root / "private/workspace", framework="CuteDSL"
            )

            with self.assertRaisesRegex(RuntimeError, "correctness"):
                benchmark_teacher(
                    materialized,
                    framework="CuteDSL",
                    sandbox_runner=lambda _w, _c: self._completed(self._result(all_pass=False)),
                )

            with self.assertRaisesRegex(RuntimeError, "shape coverage"):
                benchmark_teacher(
                    materialized,
                    framework="CuteDSL",
                    sandbox_runner=lambda _w, _c: self._completed(
                        self._result(by_shape={"shape-a": 80.0})
                    ),
                )

    def test_noncompliant_teacher_is_rejected_before_gpu_execution(self) -> None:
        with tempfile.TemporaryDirectory(prefix="teacher-benchmark-policy-") as temp_dir:
            root = Path(temp_dir)
            bundle = self._bundle(root, compliant=False)
            op = self._sol_op(root)
            materialized = materialize_teacher_workspace(
                bundle, op, root / "private/workspace", framework="CuteDSL"
            )
            runner = mock.Mock()

            with self.assertRaisesRegex(RuntimeError, "production policy"):
                benchmark_teacher(materialized, framework="CuteDSL", sandbox_runner=runner)
            runner.assert_not_called()

    def test_nonempty_destination_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="teacher-benchmark-existing-") as temp_dir:
            root = Path(temp_dir)
            bundle = self._bundle(root)
            op = self._sol_op(root)
            workspace = root / "private/workspace"
            workspace.mkdir(parents=True)
            (workspace / "unexpected.txt").write_text("state\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "not empty"):
                materialize_teacher_workspace(bundle, op, workspace, framework="CuteDSL")


if __name__ == "__main__":
    unittest.main()
