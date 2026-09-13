from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from orchestrator.constants import ATREX_BENCH_RUNTIME_ENV
from supervisor import gateway as sandbox
from tools import local_gateway

from long_horizon import journal, report, session


class GatewayRecordTest(unittest.TestCase):
    def test_atrex_bench_bundle_uses_only_supervisor_private_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            workspace.mkdir()
            runtime = root / "private" / "atrex-bench"
            (runtime / "scripts").mkdir(parents=True)
            (runtime / "scripts" / "run_eval.py").write_text("print('eval')\n")
            package = runtime / "src" / "atrex_bench"
            (package / "eval").mkdir(parents=True)
            (package / "__init__.py").write_text("")
            (package / "utils.py").write_text("VALUE = 1\n")
            (package / "eval" / "runner.py").write_text("VALUE = 2\n")
            with patch.dict(
                os.environ,
                {ATREX_BENCH_RUNTIME_ENV: str(runtime)},
            ):
                private = sandbox._private_atrex_bench_runtime()
                bundle = sandbox._make_atrex_bench_runtime_bundle(
                    private,
                    evaluator_only=True,
                )
            self.assertIsNotNone(bundle)
            self.assertFalse((workspace / "atrex-bench").exists())

    def test_record_preserves_exact_kernel_and_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            kernel = "def run(x):\n    return x\n"
            (workspace / "kernel.py").write_text(kernel, encoding="utf-8")
            result = {
                "all_pass": True,
                "latency_us_geomean": 12.5,
                "latency_us_by_shape": {"0": 12.5},
            }

            record = sandbox._record_episode_evaluation(
                workspace,
                result,
                gateway_kind="run",
                job_id="ev_1",
                private_result={"job_id": "ev_1", "request": {"private": True}},
            )

            self.assertIsNotNone(record)
            assert record is not None
            self.assertEqual(record["kernel_sha256"], hashlib.sha256(kernel.encode()).hexdigest())
            self.assertEqual(
                record["kernel_artifact_digest"],
                "sha256:" + hashlib.sha256(kernel.encode()).hexdigest(),
            )
            self.assertRegex(record["kernel_id"], r"^kernel-[0-9]+-[0-9a-f]{12}$")
            record_dir = workspace / sandbox.GATEWAY_RECORDS_PATH / record["record_id"]
            self.assertEqual((record_dir / "kernel.py").read_text(encoding="utf-8"), kernel)
            measurement = json.loads((record_dir / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(measurement["result"], result)
            raw = json.loads((record_dir / "raw-result.json").read_text(encoding="utf-8"))
            self.assertTrue(raw["request"]["private"])
            index = json.loads(
                (workspace / sandbox.EPISODE_EVALUATIONS_PATH).read_text(encoding="utf-8").strip()
            )
            self.assertEqual(index["record_id"], record["record_id"])
            self.assertTrue(index["kernel_artifact"].endswith("/kernel.py"))

    def test_gateway_record_read_returns_canonical_evaluate_with_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "kernel.py").write_text("def run(x): return x\n", encoding="utf-8")
            record = sandbox._record_episode_evaluation(
                workspace,
                {
                    "all_pass": True,
                    "latency_us_geomean": 12.0,
                    "latency_us_arith_mean": 13.0,
                    "latency_us_by_shape": {"0": 8.0, "1": 18.0},
                    "failures": [],
                    "actionable_diagnostics": [],
                    "evaluator": "private-evaluator",
                    "measurement_aggregation": {
                        "repetitions": 3, "method": "per_shape_median",
                    },
                },
                gateway_kind="run",
            )
            assert record is not None

            output = io.StringIO()
            with patch("sys.stdout", output):
                status = sandbox._read_gateway_record(workspace, record["record_id"])

            self.assertEqual(status, 0)
            payload = json.loads(
                output.getvalue().strip().removeprefix(sandbox.RECORD_RESULT_PREFIX)
            )
            self.assertEqual(payload["record_type"], "gateway_result")
            self.assertEqual(payload["operation"], "evaluate")
            self.assertEqual(payload["status"], "completed")
            self.assertEqual(payload["kernel_id"], record["kernel_id"])
            self.assertEqual(
                payload["result"]["latency_us_by_shape"], {"0": 8.0, "1": 18.0}
            )
            self.assertNotIn("evaluator", json.dumps(payload))
            self.assertNotIn("measurement_aggregation", payload["result"])
            stored = sandbox._load_gateway_record(workspace, record["record_id"])
            self.assertIn("measurement_aggregation", stored["result"])
            self.assertNotIn(
                "measurement_aggregation",
                sandbox._agent_evaluation_result(stored["result"], record),
            )

    def test_gateway_record_read_abba_identifies_both_kernels_and_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            candidate_source = "def run(x): return x + 1\n"
            (workspace / "kernel.py").write_text(candidate_source, encoding="utf-8")
            incumbent_source = "def run(x): return x\n"
            incumbent_sha256 = hashlib.sha256(incumbent_source.encode()).hexdigest()
            candidate_sha256 = hashlib.sha256(candidate_source.encode()).hexdigest()
            incumbent = sandbox._store_kernel_artifact(
                workspace, incumbent_source.encode()
            )
            record = sandbox._record_episode_evaluation(
                workspace,
                {
                    "status": "succeeded",
                    "operation": "same_allocation_abba",
                    "correct": True,
                    "baseline": {
                        "correct": True,
                        "latency_us_geomean": 12.0,
                        "latency_us_by_shape": {"0": 10.0, "1": 14.4},
                    },
                    "candidate": {
                        "correct": True,
                        "latency_us_geomean": 10.0,
                        "latency_us_by_shape": {"0": 8.0, "1": 12.5},
                    },
                    "speedup": 1.2,
                    "measurements": [{"repetition": 1}],
                    "shape_batch_count": 2,
                    "measurement_aggregation": {
                        "repetitions": 3,
                        "method": "per_shape_median",
                    },
                },
                gateway_kind="same_allocation_abba",
                kernel_subjects={
                    "incumbent": incumbent_sha256,
                    "candidate": candidate_sha256,
                },
            )
            assert record is not None

            payload = sandbox._gateway_record_public_result(
                record["record_id"], sandbox._load_gateway_record(workspace, record["record_id"])
            )

            self.assertEqual(payload["operation"], "same_allocation_abba")
            self.assertNotIn("kernel_sha256", payload)
            self.assertEqual(
                payload["kernels"],
                {
                    "incumbent": {"kernel_id": incumbent["kernel_id"]},
                    "candidate": {"kernel_id": record["kernel_id"]},
                },
            )
            for hidden in ("measurements", "shape_batch_count", "measurement_aggregation"):
                self.assertNotIn(hidden, payload["result"])
            self.assertEqual(
                payload["result"]["baseline"]["latency_us_by_shape"],
                {"0": 10.0, "1": 14.4},
            )
            self.assertEqual(
                payload["result"]["candidate"]["latency_us_by_shape"],
                {"0": 8.0, "1": 12.5},
            )
            self.assertEqual(
                sandbox._kernel_gateway_records(
                    workspace, incumbent["kernel_id"]
                )[0]["role"],
                "incumbent",
            )
            self.assertEqual(
                sandbox._kernel_gateway_records(
                    workspace, record["kernel_id"]
                )[0]["role"],
                "candidate",
            )

    def test_gateway_record_read_supports_profile_dev_check_and_disassemble(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "kernel.py").write_text("def run(x): return x\n", encoding="utf-8")
            examples = {
                "profile": {
                    "status": "succeeded",
                    "shape_id": "7",
                    "kernels": [
                        {
                            "kernel_name": "probe_kernel",
                            "duration": 2000,
                            "duration_unit": "ns",
                            "compute_sol_pct": 40.0,
                            "mem_sol_pct": 70.0,
                        }
                    ],
                },
                "dev": {
                    "status": "succeeded",
                    "command": "python3 scratch/probe.py",
                    "exit_code": 0,
                    "stdout": "shape 7: 2.0 us\n",
                    "stderr": "",
                    "synced_paths": ["scratch/probe.json"],
                },
                "check": {
                    "status": "succeeded",
                    "passed": True,
                    "shape_id": "7",
                    "diagnostics": [],
                },
                "disassemble": {
                    "status": "succeeded",
                    "passed": True,
                    "shape_id": "7",
                    "format": "sass",
                    "assembly": {
                        "format": "sass",
                        "text": "/*0000*/ MOV R0, R0;",
                        "size_bytes": 99,
                        "truncated": True,
                        "bytes_omitted": 79,
                    },
                },
            }
            for kind, result in examples.items():
                with self.subTest(kind=kind):
                    record = sandbox._record_episode_evaluation(
                        workspace, result, gateway_kind=kind
                    )
                    assert record is not None
                    payload = sandbox._gateway_record_public_result(
                        record["record_id"],
                        sandbox._load_gateway_record(workspace, record["record_id"]),
                    )
                    self.assertEqual(payload["operation"], kind)
                    self.assertEqual(payload["status"], "completed")
                    self.assertEqual(payload["kernel_id"], record["kernel_id"])
            self.assertEqual(payload["result"]["assembly"]["text"], "/*0000*/ MOV R0, R0;")
            self.assertEqual(payload["result"]["assembly"]["size_bytes"], 99)
            self.assertEqual(payload["result"]["assembly"]["bytes_omitted"], 79)

    def test_gateway_record_read_rejects_path_like_ids(self) -> None:
        with (
            tempfile.TemporaryDirectory() as directory,
            self.assertRaisesRegex(ValueError, "invalid format"),
        ):
            sandbox._load_gateway_record(Path(directory), "../../private")

    def test_gateway_record_read_cli_does_not_require_a_gateway_endpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "kernel.py").write_text("def run(x): return x\n", encoding="utf-8")
            record = sandbox._record_episode_evaluation(
                workspace,
                {"all_pass": True, "latency_us_by_shape": {"0": 1.0}},
                gateway_kind="run",
            )
            assert record is not None
            output = io.StringIO()
            with patch("sys.stdout", output):
                status = sandbox._main(
                    [
                        "--workspace",
                        str(workspace),
                        "--kind",
                        "record-read",
                        "--record-id",
                        record["record_id"],
                    ]
                )
            self.assertEqual(status, 0)
            self.assertIn(sandbox.RECORD_RESULT_PREFIX, output.getvalue())

    def test_kernel_gateway_record_view_indexes_all_operations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            source = "def run(x): return x\n"
            (workspace / "kernel.py").write_text(source, encoding="utf-8")
            evaluated = sandbox._record_episode_evaluation(
                workspace,
                {"all_pass": True, "latency_us_by_shape": {"0": 1.0}},
                gateway_kind="run",
            )
            profiled = sandbox._record_episode_evaluation(
                workspace,
                {"status": "succeeded", "kernels": []},
                gateway_kind="profile",
            )
            assert evaluated is not None and profiled is not None
            self.assertEqual(evaluated["kernel_id"], profiled["kernel_id"])
            output = io.StringIO()
            with patch("sys.stdout", output):
                status = sandbox._main(
                    [
                        "--workspace",
                        str(workspace),
                        "--kind",
                        "record-read",
                        "--record-id",
                        evaluated["kernel_id"],
                        "--view",
                        "gateway-records",
                    ]
                )

            self.assertEqual(status, 0)
            payload = json.loads(
                output.getvalue().strip().removeprefix(sandbox.RECORD_RESULT_PREFIX)
            )
            self.assertEqual(payload["record_type"], "kernel_gateway_records")
            self.assertEqual(payload["kernel_id"], evaluated["kernel_id"])
            self.assertEqual(
                [value["operation"] for value in payload["gateway_records"]],
                ["evaluate", "profile"],
            )
            self.assertNotIn("source", payload)

            with self.assertRaisesRegex(SystemExit, "requires --view"):
                sandbox._main(
                    [
                        "--workspace",
                        str(workspace),
                        "--kind",
                        "record-read",
                        "--record-id",
                        evaluated["kernel_id"],
                    ]
                )

    def test_kernel_artifact_read_verifies_and_writes_only_under_scratch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            source = "def run(x): return x + 1\n"
            (workspace / "kernel.py").write_text(source, encoding="utf-8")
            record = sandbox._record_episode_evaluation(
                workspace,
                {"all_pass": True, "latency_us_by_shape": {"0": 1.0}},
                gateway_kind="run",
            )
            assert record is not None
            output = io.StringIO()
            with patch("sys.stdout", output):
                status = sandbox._main(
                    [
                        "--workspace",
                        str(workspace),
                        "--kind",
                        "record-read",
                        "--record-id",
                        record["kernel_id"],
                        "--view",
                        "source",
                        "--output-path",
                        "scratch/history/kernel.py",
                    ]
                )
            self.assertEqual(status, 0)
            self.assertEqual(
                (workspace / "scratch/history/kernel.py").read_text(encoding="utf-8"),
                source,
            )
            payload = json.loads(
                output.getvalue().strip().removeprefix(sandbox.RECORD_RESULT_PREFIX)
            )
            self.assertEqual(payload["status"], "written")
            self.assertEqual(payload["record_type"], "kernel_source")
            self.assertEqual(payload["kernel_id"], record["kernel_id"])
            self.assertNotIn("gateway_records", payload)
            with self.assertRaisesRegex(SystemExit, "under scratch"):
                sandbox._read_kernel_source(
                    workspace,
                    record["kernel_id"],
                    "kernel.py",
                )

    def test_supervisor_record_is_outside_agent_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            workspace.mkdir()
            (workspace / "kernel.py").write_text("def run(x): return x\n")
            evidence_root = root / "private" / "evidence"
            with patch.dict(
                os.environ,
                {sandbox.SUPERVISOR_EVIDENCE_ROOT_ENV: str(evidence_root)},
            ):
                record = sandbox._record_episode_evaluation(
                    workspace,
                    {"all_pass": True, "latency_us_geomean": 1.0},
                    gateway_kind="run",
                )
                injected = sandbox._supervisor_runtime_inputs(
                    [
                        "python3",
                        "timeline.py",
                        "--correctness-evidence",
                        sandbox.EPISODE_EVALUATIONS_PATH,
                    ]
                )
            assert record is not None
            self.assertFalse((workspace / ".atrex_long_horizon").exists())
            self.assertTrue((evidence_root / "evaluations.jsonl").is_file())
            self.assertTrue(
                (evidence_root / "gateway-records" / record["record_id"] / "kernel.py").is_file()
            )
            self.assertEqual(
                injected,
                {sandbox.EPISODE_EVALUATIONS_PATH: (evidence_root / "evaluations.jsonl")},
            )

    def test_agent_projection_omits_gateway_transport_details(self) -> None:
        projected = sandbox._agent_evaluation_result(
            {
                "all_pass": False,
                "eval_id": "private-eval-id",
                "evaluator": "internal/evaluator/name",
                "latency_us_geomean": 0.0,
                "failures": ["x" * 2000],
                "actionable_diagnostics": [],
            },
            {"record_id": "gateway-1", "kernel_sha256": "a" * 64},
        )
        self.assertNotIn("eval_id", projected)
        self.assertNotIn("evaluator", projected)
        self.assertLessEqual(len(projected["failures"][0]), 1000)
        self.assertEqual(projected["gateway_record_id"], "gateway-1")

    def test_measurement_aggregation_uses_three_per_shape_medians(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "kernel.py").write_text("class Model: pass\n")
            first = {
                "all_pass": True,
                "latency_us_geomean": (10.0 * 100.0) ** 0.5,
                "latency_us_arith_mean": 55.0,
                "latency_us_by_shape": {"0": 10.0, "1": 100.0},
            }
            second = {
                "all_pass": True,
                "latency_us_geomean": (11.0 * 140.0) ** 0.5,
                "latency_us_arith_mean": 75.5,
                "latency_us_by_shape": {"0": 11.0, "1": 140.0},
            }
            third = {
                "all_pass": True,
                "latency_us_geomean": (9.0 * 105.0) ** 0.5,
                "latency_us_arith_mean": 57.0,
                "latency_us_by_shape": {"0": 9.0, "1": 105.0},
            }
            result, private = sandbox._aggregate_typed_runs(
                workspace,
                {"reference": {"metadata": None}},
                ["0", "1"],
                [
                    (first, [{"job_id": "first"}]),
                    (second, [{"job_id": "second"}]),
                    (third, [{"job_id": "third"}]),
                ],
            )

            self.assertEqual(result["latency_us_by_shape"], {"0": 10.0, "1": 105.0})
            self.assertEqual(
                result["measurement_aggregation"],
                {"repetitions": 3, "method": "per_shape_median"},
            )
            self.assertEqual(len(private["repetitions"]), 3)

    def test_profile_projection_is_an_explicit_bounded_allowlist(self) -> None:
        projected = sandbox._agent_profile_result(
            {
                "status": "succeeded",
                "passed": True,
                "shape_id": "0",
                "clock_lock": {
                    "requested": True,
                    "applied": True,
                    "private_host": "must-not-leak",
                },
                "request": {"private_shape": [1, 2, 3]},
                "stdout": "raw profiler output",
                "artifacts": [{"url": "https://signed.example/secret"}],
                "kernels": [
                    {
                        "kernel_name": "candidate_kernel",
                        "duration": 2_000,
                        "duration_unit": "ns",
                        "compute_sol_pct": 35.0,
                        "mem_sol_pct": 70.0,
                        "registers": 64,
                        "metrics": {
                            "dram__bytes.sum": {"value": 4096, "unit": "byte"},
                        },
                        "source": "private source correlation",
                    }
                ],
            },
            {"record_id": "gateway-2", "kernel_sha256": "b" * 64},
        )

        self.assertEqual(projected["shape_id"], "0")
        self.assertEqual(projected["kernel_count"], 1)
        self.assertEqual(projected["total_duration_us"], 2.0)
        self.assertEqual(projected["weighted_sol_pct"], 70.0)
        self.assertEqual(projected["dominant_bound"], "memory")
        self.assertEqual(projected["kernels"][0]["duration_us"], 2.0)
        self.assertEqual(projected["kernels"][0]["memory_sol_pct"], 70.0)
        self.assertEqual(projected["kernels"][0]["registers_per_thread"], 64.0)
        serialized = json.dumps(projected)
        for private in ("private_shape", "private_host", "signed.example", "source"):
            self.assertNotIn(private, serialized)

    def test_profile_workspace_file_uses_the_same_public_projection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            public = {"status": "succeeded", "kernels": [{"name": "safe"}]}
            sandbox._record_profile_job(
                {
                    "status": "succeeded",
                    "request": {"private_shape": [1, 2, 3]},
                    "result": {
                        "artifacts": [],
                        "kernels": [{"source": "private source correlation"}],
                    },
                },
                workspace,
                ["scratch/profile-v1"],
                public,
            )
            stored = json.loads(
                (workspace / "scratch/profile-v1/gateway_profile.json").read_text(encoding="utf-8")
            )
            self.assertEqual(stored, {"status": "succeeded", "result": public})

    def test_profile_check_and_disassemble_requests_support_dependencies(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "kernel.py").write_text("class Model: pass\n")
            (workspace / "reference.py").write_text("def run(x): return x\n")
            (workspace / "input.py").write_text("def _make_inputs(n=1): return {}\n")
            (workspace / "shapes.json").write_text(
                json.dumps(
                    {
                        "0": {"init_kwargs": None, "input_kwargs": {"n": 1}},
                        "1": {"init_kwargs": {"width": 128}, "input_kwargs": {"n": 2}},
                    }
                )
            )
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop(sandbox.PRIVATE_REFERENCE_ENV, None)
                check = sandbox._typed_request(
                    workspace,
                    "L20N",
                    120,
                    [],
                    [],
                    "check",
                    arch="sm_120",
                    sanitize="memcheck",
                    requirements=("custom-kernel-package==1",),
                    deps_mode="no_deps",
                )
                disassemble = sandbox._typed_request(
                    workspace,
                    "L20N",
                    120,
                    [],
                    [],
                    "disassemble",
                    disassembly_format="isa",
                    requirements=("custom-kernel-package==1",),
                    deps_mode="freeze_installed",
                )
                profile = sandbox._typed_request(
                    workspace,
                    "L20N",
                    120,
                    [],
                    [],
                    "profile",
                    requirements=("custom-kernel-package==1",),
                    deps_mode="no_deps",
                )

            self.assertEqual(check["init_kwargs"], {})
            self.assertEqual(check["shape_id"], "0")
            self.assertIn("diagnostic_reference", check)
            self.assertEqual(check["arch"], "sm_120")
            self.assertEqual(check["sanitize"], "memcheck")
            self.assertEqual(check["requirements"], ["custom-kernel-package==1"])
            self.assertEqual(check["deps_mode"], "no_deps")
            self.assertNotIn("reference", check)
            self.assertEqual(disassemble["init_kwargs"], {})
            self.assertEqual(disassemble["fmt"], "isa")
            self.assertEqual(disassemble["requirements"], ["custom-kernel-package==1"])
            self.assertEqual(disassemble["deps_mode"], "freeze_installed")
            self.assertEqual(profile["requirements"], ["custom-kernel-package==1"])
            self.assertEqual(profile["deps_mode"], "no_deps")

    def test_local_gateway_accepts_compile_and_disassemble_contracts(self) -> None:
        candidate = "class Model:\n    def __init__(self, width=1): self.width = width\n"
        common = {
            "spec": {"target_hardware": ["local"]},
            "candidate": candidate,
            "init_kwargs": {"width": 128},
            "requirements": ["custom-kernel-package==1"],
            "deps_mode": "no_deps",
        }
        checked = local_gateway._validate_diagnostic_request(
            {**common, "sanitize": "memcheck", "arch": "sm_120"},
            "compile",
        )
        disassembled = local_gateway._validate_diagnostic_request(
            {**common, "fmt": "isa"},
            "disassemble",
        )
        self.assertEqual(checked["requirements"], common["requirements"])
        self.assertEqual(checked["deps_mode"], "no_deps")
        self.assertEqual(disassembled["fmt"], "isa")
        with tempfile.TemporaryDirectory() as directory:
            gateway = local_gateway.LocalGateway(Path(directory), workers=1)
            try:
                compile_job, _ = gateway.submit("compile", checked, "req-compile")
                disassemble_job, _ = gateway.submit("disassemble", disassembled, "req-disassemble")
            finally:
                gateway.close()
        self.assertEqual(compile_job["kind"], "compile")
        self.assertTrue(compile_job["job_id"].startswith("ck_"))
        self.assertEqual(disassemble_job["kind"], "disassemble")
        self.assertTrue(disassemble_job["job_id"].startswith("ds_"))

    def test_evaluate_accepts_custom_inputs_and_correctness_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "kernel.py").write_text("class Model: pass\n")
            (workspace / "input.py").write_text("def _make_inputs(): return {}\n")
            (workspace / "reference.py").write_text("def run(x): return x\n")
            (workspace / "input.py").write_text("def _make_inputs(n=1): return {'x': n}\n")
            (workspace / "shapes.json").write_text(json.dumps({"0": {"input_kwargs": {"n": 1}}}))
            (workspace / "metadata.json").write_text(json.dumps({"private": True}))
            (workspace / "roofline.json").write_text(json.dumps({"private": True}))
            scratch = workspace / "scratch"
            scratch.mkdir()
            (scratch / "custom-input.py").write_text("def _make_inputs(n=1): return {'x': n + 1}\n")
            (scratch / "custom-shapes.json").write_text(
                json.dumps({"7": {"input_kwargs": {"n": 32}, "init_kwargs": {}}})
            )

            request = sandbox._typed_request(
                workspace,
                "L20N",
                120,
                [],
                [],
                "run",
                evaluation_input_path="scratch/custom-input.py",
                evaluation_shapes_path="scratch/custom-shapes.json",
                evaluation_mode="correctness_only",
            )

        self.assertEqual(request["mode"], "correctness_only")
        self.assertIn("n + 1", request["reference"]["input_py"])
        self.assertEqual(
            request["reference"]["shapes"],
            {"7": {"input_kwargs": {"n": 32}, "init_kwargs": {}}},
        )
        self.assertNotIn("metadata", request["reference"])
        self.assertNotIn("roofline", request["reference"])

    def test_correctness_only_projection_omits_performance_values(self) -> None:
        result = {
            "all_pass": True,
            "latency_us_geomean": None,
            "latency_us_arith_mean": None,
            "latency_us_by_shape": {},
            "max_abs_err": 0.0,
            "max_rel_err": 0.0,
            "mode": "correctness_only",
            "input_scope": "contract",
        }
        projected = sandbox._agent_evaluation_result(result, None)
        self.assertEqual(projected["mode"], "correctness_only")
        self.assertEqual(projected["input_scope"], "contract")
        self.assertIsNone(projected["latency_us_geomean"])
        self.assertEqual(projected["latency_us_by_shape"], {})

    def test_diagnostic_agate_cli_receives_private_dependency_sidecars(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            workspace.mkdir()
            (workspace / "kernel.py").write_text("class Model: pass\n")
            args = sandbox.build_parser().parse_args(
                [
                    "--hardware",
                    "L20N",
                    "--kind",
                    "disassemble",
                    "--format",
                    "isa",
                    "--requirement",
                    "custom-kernel-package==1",
                    "--deps-mode",
                    "no_deps",
                ]
            )
            command = sandbox._typed_agate_command(
                "agate",
                args,
                workspace,
                "disassemble",
                {
                    "init_kwargs": {"width": 128},
                    "requirements": ["custom-kernel-package==1"],
                    "deps_mode": "no_deps",
                },
                300,
                request_sidecar_dir=root / "private",
            )
            self.assertIn("isa", command)
            self.assertIn("--init-kwargs", command)
            self.assertIn("--requirements", command)
            self.assertIn("--deps-mode", command)
            self.assertEqual(
                (root / "private" / "requirements.txt").read_text(),
                "custom-kernel-package==1\n",
            )
            profile_args = sandbox.build_parser().parse_args(
                [
                    "--hardware",
                    "L20N",
                    "--kind",
                    "profile",
                    "--requirement",
                    "custom-kernel-package==1",
                    "--deps-mode",
                    "freeze_installed",
                ]
            )
            profile_command = sandbox._typed_agate_command(
                "agate",
                profile_args,
                workspace,
                "profile",
                {
                    "spec": {"target_hardware": ["L20N"]},
                    "reference": {"operator": "example"},
                    "options": {
                        "num_correctness_cases": 1,
                        "bench_iters": 1,
                    },
                    "requirements": ["custom-kernel-package==1"],
                    "deps_mode": "freeze_installed",
                },
                300,
                reference_dir=workspace,
                request_sidecar_dir=root / "profile-private",
            )
            self.assertIn("--requirements", profile_command)
            self.assertIn("--deps-mode", profile_command)
            self.assertEqual(
                (root / "profile-private" / "requirements.txt").read_text(),
                "custom-kernel-package==1\n",
            )

    def test_profile_contract_exposes_all_runtime_selectors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "kernel.py").write_text("class Model: pass\n")
            (workspace / "reference.py").write_text("def run(): return None\n")
            (workspace / "input.py").write_text("def _make_inputs(): return {}\n")
            (workspace / "shapes.json").write_text(
                json.dumps({"0": {"input_kwargs": {}}, "1": {"input_kwargs": {}}})
            )
            request = sandbox._typed_request(
                workspace,
                "L20N",
                120,
                [],
                [],
                "profile",
                kernel_name="candidate_kernel",
                profile_source=True,
                launch_skip=2,
                launch_count=7,
                profile_shape_id="1",
            )
        self.assertEqual(request["kernel_name"], "candidate_kernel")
        self.assertTrue(request["source"])
        self.assertEqual(request["launch_skip"], 2)
        self.assertEqual(request["launch_count"], 7)
        self.assertEqual(request["shape_id"], "1")
        self.assertEqual(list(request["reference"]["shapes"]), ["1"])

    def test_environment_query_uses_capabilities_endpoint(self) -> None:
        args = sandbox.build_parser().parse_args(
            [
                "--kind",
                "env",
                "--url",
                "http://127.0.0.1:8000",
                "--env-gpu",
                "local",
                "--env-capabilities",
                "--env-force",
            ]
        )
        output = io.StringIO()
        with (
            patch.object(sandbox, "_find_agate", return_value=None),
            patch.object(
                sandbox,
                "_gateway_json",
                return_value={"gpu": "local", "frameworks": {}},
            ) as request,
            patch("sys.stdout", output),
        ):
            self.assertEqual(sandbox._run_environment_query(args), 0)
        self.assertEqual(request.call_args.args[2], "/v1/env/local/capabilities?force=true")
        self.assertIn(sandbox.ENV_RESULT_PREFIX, output.getvalue())

    def test_local_diagnostics_launch_and_then_collect_generated_ptx(self) -> None:
        candidate = """
import os
from pathlib import Path

class Model:
    def __call__(self, **kwargs):
        Path(os.environ['TRITON_CACHE_DIR'], 'generated.ptx').write_text(
            '.version 8.0\\n.target sm_90\\n.entry generated() {}\\n', encoding='utf-8'
        )
"""
        common = {
            "spec": {"target_hardware": ["local"]},
            "candidate": candidate,
            "init_kwargs": {},
            "shape_id": "0",
            "diagnostic_reference": {
                "input_py": "def _make_inputs(): return {}\n",
                "shapes": {"0": {"init_kwargs": {}, "input_kwargs": {}}},
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            gateway = local_gateway.LocalGateway(Path(directory), workers=1)
            gateway.start()
            try:
                checked, _ = gateway.submit("compile", dict(common), "req-check")
                check_job = gateway.scheduler.wait_for_job(checked["job_id"], 30)
                disassembled, _ = gateway.submit(
                    "disassemble", {**common, "fmt": "ptx"}, "req-disassemble-real"
                )
                disassembly_job = gateway.scheduler.wait_for_job(disassembled["job_id"], 30)
            finally:
                gateway.close()
        assert check_job is not None and disassembly_job is not None
        self.assertEqual(check_job["status"], "succeeded")
        self.assertTrue(check_job["result"]["launch_ok"])
        self.assertEqual(check_job["result"]["shape_id"], "0")
        self.assertEqual(disassembly_job["status"], "succeeded")
        self.assertTrue(disassembly_job["result"]["passed"])
        self.assertIn(".version 8.0", disassembly_job["result"]["exports"]["ptx.txt"]["text"])

    def test_agent_abba_projection_aggregates_both_sides(self) -> None:
        schedule = [
            {"revision": "incumbent", "repeat": 0},
            {"revision": "candidate", "repeat": 0},
            {"revision": "candidate", "repeat": 1},
            {"revision": "incumbent", "repeat": 1},
        ]
        values = {
            ("incumbent", 0): 10.0,
            ("incumbent", 1): 10.0,
            ("candidate", 0): 8.0,
            ("candidate", 1): 8.0,
        }
        payload = {
            "runs": [
                {
                    **step,
                    "exit_code": 0,
                    "result": {
                        "all_pass": True,
                        "latency_us_geomean": values[(step["revision"], step["repeat"])],
                        "latency_us_by_shape": {"0": values[(step["revision"], step["repeat"])]},
                    },
                }
                for step in schedule
            ],
            "error": None,
        }
        result = sandbox._agent_abba_public_result(payload, schedule, ["0"], 2)
        self.assertTrue(result["correct"])
        self.assertAlmostEqual(result["baseline"]["latency_us_geomean"], 10.0)
        self.assertAlmostEqual(result["candidate"]["latency_us_geomean"], 8.0)
        self.assertAlmostEqual(result["speedup"], 1.25)

    def test_agent_abba_uses_private_dev_schedule_and_records_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            for name, source in {
                "kernel.py": "class Model: pass\n",
                "reference.py": "def run(): return None\n",
                "input.py": "def _make_inputs(): return {}\n",
            }.items():
                (workspace / name).write_text(source)
            (workspace / "shapes.json").write_text(
                json.dumps({"0": {"init_kwargs": {}, "input_kwargs": {}}})
            )
            scratch = workspace / "scratch"
            scratch.mkdir()
            (scratch / "baseline.py").write_text("class Model: pass\n")
            args = sandbox.build_parser().parse_args(
                [
                    "--hardware",
                    "L20N",
                    "--url",
                    "https://gateway.example.test",
                    "--workspace",
                    str(workspace),
                    "--timeout",
                    "600",
                    "--kind",
                    "run",
                    "--mode",
                    "full",
                    "--baseline-path",
                    "scratch/baseline.py",
                    "--comparison-repeats",
                    "2",
                    "--no-sync",
                ]
            )
            schedule = [
                {"revision": "incumbent", "repeat": 0},
                {"revision": "candidate", "repeat": 0},
                {"revision": "candidate", "repeat": 1},
                {"revision": "incumbent", "repeat": 1},
            ]
            payload = {
                "schema_version": 1,
                "runs": [
                    {
                        **step,
                        "exit_code": 0,
                        "result": {
                            "all_pass": True,
                            "latency_us_geomean": 10.0 if step["revision"] == "incumbent" else 8.0,
                            "latency_us_by_shape": {
                                "0": 10.0 if step["revision"] == "incumbent" else 8.0
                            },
                        },
                    }
                    for step in schedule
                ],
                "error": None,
            }
            completed = subprocess.CompletedProcess(
                args=["supervisor/gateway.py"],
                returncode=0,
                stdout=sandbox.ABBA_RESULT_PREFIX + json.dumps(payload) + "\n",
                stderr="",
            )
            output = io.StringIO()
            with (
                patch.object(sandbox.subprocess, "run", return_value=completed) as run,
                patch("sys.stdout", output),
            ):
                status = sandbox._run_agent_abba(args, workspace, [], 0)
            self.assertEqual(status, 0)
            self.assertIn(sandbox.ABBA_RESULT_PUBLIC_PREFIX, output.getvalue())

            def gpu_submissions(mock: Mock) -> list:
                return [
                    call
                    for call in mock.call_args_list
                    if len(call.args[0]) > 1
                    and str(call.args[0][1]).endswith("supervisor/gateway.py")
                ]

            # Workspace discovery may run Git; count only actual measurement subprocesses.
            self.assertEqual(len(gpu_submissions(run)), 3)
            rendered = next(
                line.removeprefix(sandbox.ABBA_RESULT_PUBLIC_PREFIX)
                for line in output.getvalue().splitlines()
                if line.startswith(sandbox.ABBA_RESULT_PUBLIC_PREFIX)
            )
            public = json.loads(rendered)
            for hidden in ("measurements", "shape_batch_count", "measurement_aggregation"):
                self.assertNotIn(hidden, public)
            self.assertNotIn("kernel_sha256", public)
            self.assertIn("kernel_id", public)
            self.assertNotIn("kernel_artifact_digest", public)
            self.assertNotIn("kernel_trial_id", public)
            self.assertEqual(
                set(public["kernels"]),
                {"incumbent", "candidate"},
            )
            with (
                patch.object(sandbox.subprocess, "run", return_value=completed) as duplicate_run,
                self.assertRaisesRegex(SystemExit, "duplicate Gateway task") as duplicate,
            ):
                sandbox._run_agent_abba(args, workspace, [], 0)
            self.assertEqual(gpu_submissions(duplicate_run), [])
            self.assertIn(
                "--kind record-read",
                json.loads(duplicate.exception.code)["error"]["next_action"],
            )
            self.assertTrue((workspace / sandbox.EPISODE_EVALUATIONS_PATH).is_file())

    def test_check_projection_is_bounded_and_omits_request_echoes(self) -> None:
        projected = sandbox._agent_check_result(
            {
                "passed": True,
                "compile_ok": True,
                "launch_ok": True,
                "scope": "one_shape_launch_probe",
                "shape_id": "0",
                "correctness_checked": False,
                "diagnostics": [
                    {
                        "severity": "warning",
                        "message": "register pressure",
                        "registers_per_thread": 96,
                        "private_worker": "must-not-leak",
                    }
                ],
                "request": {"init_kwargs": {"secret_shape": 1}},
                "logs": ["raw compiler log"],
            },
            {"record_id": "gateway-3", "kernel_sha256": "c" * 64},
        )
        self.assertTrue(projected["compile_ok"])
        self.assertEqual(projected["scope"], "one_shape_launch_probe")
        self.assertEqual(projected["shape_id"], "0")
        self.assertFalse(projected["correctness_checked"])
        self.assertEqual(projected["diagnostics"][0]["registers_per_thread"], 96)
        serialized = json.dumps(projected)
        self.assertNotIn("secret_shape", serialized)
        self.assertNotIn("private_worker", serialized)
        self.assertNotIn("raw compiler log", serialized)

    def test_disassembly_projection_returns_bounded_assembly(self) -> None:
        assembly = "HEADER\n" + "MOV R0, R1;\n" * 40_000 + "TAIL\n"
        projected = sandbox._agent_disassembly_result(
            {
                "passed": True,
                "compile_ok": True,
                "launch_ok": True,
                "scope": "one_shape_launch_probe",
                "shape_id": "0",
                "format": "sass",
                "exports": {
                    "sass.txt": {"text": assembly},
                    "private.log": {"text": "must-not-leak"},
                },
                "request": {"init_kwargs": {"hidden": 1}},
            },
            {"record_id": "gateway-4", "kernel_sha256": "d" * 64},
        )
        self.assertTrue(projected["launch_ok"])
        self.assertEqual(projected["shape_id"], "0")
        self.assertTrue(projected["assembly"]["truncated"])
        self.assertIn("HEADER", projected["assembly"]["text"])
        self.assertIn("TAIL", projected["assembly"]["text"])
        self.assertNotIn("must-not-leak", json.dumps(projected))

    def test_typed_diagnostics_need_no_shell_command(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "kernel.py").write_text("class Model: pass\n")
            (workspace / "input.py").write_text("def _make_inputs(): return {}\n")
            (workspace / "shapes.json").write_text(
                json.dumps({"0": {"init_kwargs": None, "input_kwargs": {}}})
            )
            jobs = {
                "check": {
                    "job_id": "ck_1",
                    "status": "succeeded",
                    "result": {"passed": True, "compile_ok": True},
                },
                "disassemble": {
                    "job_id": "da_1",
                    "status": "succeeded",
                    "result": {
                        "passed": True,
                        "format": "ptx",
                        "exports": {"ptx.txt": {"text": ".version 8.0\n"}},
                    },
                },
            }
            for kind, job in jobs.items():
                with self.subTest(kind=kind):
                    completed = subprocess.CompletedProcess(
                        args=["direct-gateway", kind],
                        returncode=0,
                        stdout=json.dumps(job),
                        stderr="",
                    )
                    output = io.StringIO()
                    with (
                        patch.object(sandbox, "_run_direct_job", return_value=completed),
                        patch("sys.stdout", output),
                        patch.dict(os.environ, {}, clear=False),
                    ):
                        os.environ.pop(sandbox.PRIVATE_REFERENCE_ENV, None)
                        status = sandbox._main(
                            [
                                "--hardware",
                                "L20N",
                                "--url",
                                "https://gateway.example.test",
                                "--workspace",
                                str(workspace),
                                "--kind",
                                kind,
                                "--no-sync",
                            ]
                        )
                    self.assertEqual(status, 0)
                    prefix = (
                        sandbox.CHECK_RESULT_PREFIX
                        if kind == "check"
                        else sandbox.DISASSEMBLY_RESULT_PREFIX
                    )
                    self.assertIn(prefix, output.getvalue())


class GatewayRetryTest(unittest.TestCase):
    def test_terminal_infrastructure_failure_is_resubmitted(self) -> None:
        failed = subprocess.CompletedProcess(
            args=["agate"],
            returncode=1,
            stdout=json.dumps(
                {
                    "job_id": "ev_old",
                    "status": "failed",
                    "error": {
                        "error_class": "infra",
                        "reason": "logs_unavailable",
                        "details": {"backend_state": "succeeded"},
                    },
                }
            ),
            stderr="",
        )
        succeeded = subprocess.CompletedProcess(
            args=["agate"],
            returncode=0,
            stdout=json.dumps({"job_id": "ev_new", "status": "succeeded", "result": {}}),
            stderr="",
        )
        with (
            patch.object(sandbox, "_run_agate_once", side_effect=[failed, succeeded]) as run,
            patch.object(sandbox.time, "sleep"),
        ):
            result = sandbox._run_agate_with_cancel_retry(
                agate=["agate", "eval"],
                executable="agate",
                url="",
                gateway_profile=None,
                command_timeout=60,
                wait_budget=120,
            )
        self.assertEqual(run.call_count, 2)
        self.assertIn("ev_new", result.stdout)
        self.assertIn("resubmitting a fresh job", result.stderr)

    def test_candidate_failure_is_not_retried(self) -> None:
        job = {
            "job_id": "ev_bad_kernel",
            "status": "failed",
            "error": {"error_class": "candidate", "reason": "compile_failed"},
        }
        self.assertFalse(sandbox._infrastructure_failure(job))

    def test_http_503_cli_failure_is_retryable(self) -> None:
        completed = subprocess.CompletedProcess(
            args=["agate"], returncode=1, stdout="", stderr="agate: HTTP 503"
        )
        self.assertTrue(sandbox._agate_transport_failure(completed, None))


class ClaudeRetryTest(unittest.TestCase):
    def test_connection_loss_is_retryable(self) -> None:
        self.assertTrue(
            session._claude_retryable_api_error(
                "API Error: Connection lost mid-response. The response above may be incomplete."
            )
        )

    def test_dns_failure_is_retryable(self) -> None:
        self.assertTrue(
            session._claude_retryable_api_error(
                "API Error: Can't reach the API server — check your internet or DNS (ENOTFOUND)"
            )
        )

    def test_structured_dns_failure_is_detected(self) -> None:
        stdout = json.dumps(
            {
                "type": "assistant",
                "isApiErrorMessage": True,
                "message": {
                    "content": [
                        {
                            "type": "text",
                            "text": "API Error: Can't reach the API server (ENOTFOUND)",
                        }
                    ]
                },
            }
        )
        self.assertIn("ENOTFOUND", session._claude_transient_api_error(stdout))


class RepairableReportTest(unittest.TestCase):
    def test_bad_report_can_be_fixed_and_resubmitted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            journal_path = root / "journal.json"
            handoff_path = root / "handoff.json"
            journal.initialize(
                journal_path,
                episode=3,
                base_commit="base",
                branch="episode-3",
            )
            journal.append_experiment(journal_path, {"name": "trial"})

            with self.assertRaisesRegex(ValueError, "outcome.summary"):
                report.submit_report(
                    journal_path=journal_path,
                    handoff_path=handoff_path,
                    expected_episode=3,
                    base_commit="base",
                    branch="episode-3",
                    state="pivot",
                    outcome={"summary": "", "next_directions": []},
                )
            self.assertFalse(handoff_path.exists())

            result = report.submit_report(
                journal_path=journal_path,
                handoff_path=handoff_path,
                expected_episode=3,
                base_commit="base",
                branch="episode-3",
                state="pivot",
                outcome={"summary": "direction exhausted", "next_directions": []},
            )
            self.assertTrue(result["ok"])
            self.assertEqual(
                json.loads(handoff_path.read_text(encoding="utf-8")),
                {"status": "pivot"},
            )

    def test_candidate_report_identifies_a_passing_selected_experiment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            journal_path = root / "journal.json"
            handoff_path = root / "handoff.json"
            journal.initialize(
                journal_path,
                episode=1,
                base_commit="base",
                branch="episode-1",
            )
            journal.append_experiment(
                journal_path,
                {
                    "name": "candidate",
                    "decision": "keep_as_best",
                    "evaluation": {
                        "correctness": "pass",
                        "performance": "improved",
                        "latency_us": 10.0,
                        "kernel_hash": "abc",
                    },
                },
            )
            with self.assertRaisesRegex(ValueError, "selected_experiment_index"):
                report.submit_report(
                    journal_path=journal_path,
                    handoff_path=handoff_path,
                    expected_episode=1,
                    base_commit="base",
                    branch="episode-1",
                    state="candidate_ready",
                    candidate_commit="candidate",
                    outcome={"summary": "faster", "next_directions": []},
                )
            result = report.submit_report(
                journal_path=journal_path,
                handoff_path=handoff_path,
                expected_episode=1,
                base_commit="base",
                branch="episode-1",
                state="candidate_ready",
                candidate_commit="candidate",
                outcome={
                    "summary": "faster",
                    "next_directions": [],
                    "selected_experiment_index": 1,
                },
            )
            self.assertTrue(result["ok"])


if __name__ == "__main__":
    unittest.main()
