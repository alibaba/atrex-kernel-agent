from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from long_horizon import verifier
from orchestrator.session_io import _test_result_from_stdout

from supervisor import gateway


class MeasurementReuseTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        environment = patch.dict(
            os.environ,
            {
                gateway.SUPERVISOR_EVIDENCE_ROOT_ENV: str(self.root / "private"),
                gateway.SUPERVISOR_HISTORY_ROOT_ENV: "",
                gateway.REUSE_GATEWAY_RESULTS_ENV: "",
                gateway.COMPARISON_RUN_TIMEOUT_ENV: "120",
            },
        )
        environment.start()
        self.addCleanup(environment.stop)
        self.incumbent = b"class Model: pass\n"
        self.candidate = b"class Model: value = 1\n"
        for name, content in {
            "kernel.py": self.incumbent,
            "reference.py": b"def run(x): return x\n",
            "input.py": b"def _make_inputs(): return {}\n",
            "shapes.json": json.dumps({"0": {"init_kwargs": {}, "input_kwargs": {}}}).encode(),
            "metadata.json": json.dumps(
                {
                    "shapes": {
                        "0": {
                            "production_performance": {"performance_us": 40.0},
                        }
                    }
                }
            ).encode(),
        }.items():
            (self.workspace / name).write_bytes(content)
        self.git("init", "-q")
        self.git("add", ".")
        self.git("commit", "-qm", "baseline")
        self.base_commit = self.git("rev-parse", "HEAD")
        (self.workspace / "kernel.py").write_bytes(self.candidate)
        self.git("add", "kernel.py")
        self.git("commit", "-qm", "candidate")
        self.candidate_commit = self.git("rev-parse", "HEAD")
        (self.workspace / "scratch").mkdir()
        (self.workspace / "scratch" / "baseline.py").write_bytes(self.incumbent)
        self.args = gateway.build_parser().parse_args(
            [
                "--workspace",
                str(self.workspace),
                "--hardware",
                "L20N",
                "--url",
                "https://gateway.example.test",
                "--timeout",
                "600",
                "--kind",
                "run",
                "--mode",
                "full",
                "--no-sync",
                "--baseline-path",
                "scratch/baseline.py",
                "--comparison-repeats",
                "2",
            ]
        )
        self.calls = 0
        self.real_run = subprocess.run
        self.gpu = patch.object(gateway.subprocess, "run", side_effect=self.remote)

    def git(self, *args: str) -> str:
        return subprocess.run(
            [
                "git",
                "-c",
                "user.name=Test",
                "-c",
                "user.email=test@example.test",
                *args,
            ],
            cwd=self.workspace,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    def remote(self, command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        if len(command) < 2 or not str(command[1]).endswith("supervisor/gateway.py"):
            return self.real_run(command, **kwargs)
        index = self.calls % 3
        self.calls += 1
        self.assertEqual(kwargs["env"][gateway.INTERNAL_MEASUREMENT_ENV], "1")
        schedule = verifier.verification_schedule(2)
        runs = []
        for step in schedule:
            latency = (
                [10.0, 12.0, 11.0][index]
                if step["revision"] == "incumbent"
                else [8.0, 100.0, 7.0][index]
            )
            runs.append(
                {
                    **step,
                    "exit_code": 0,
                    "result": {
                        "all_pass": True,
                        "latency_us_geomean": latency,
                        "latency_us_by_shape": {"0": latency},
                        "performance_score": 40.0 / latency,
                    },
                }
            )
        return subprocess.CompletedProcess(
            command,
            0,
            gateway.ABBA_RESULT_PREFIX
            + json.dumps(
                {
                    "schema_version": 1,
                    "runs": runs,
                    "error": None,
                }
            ),
            "",
        )

    def measure(self, *, reuse: bool = False, command: list[str] | None = None) -> tuple[int, str]:
        output = io.StringIO()
        with (
            self.gpu,
            redirect_stdout(output),
            patch.dict(
                os.environ,
                {
                    gateway.REUSE_GATEWAY_RESULTS_ENV: "1" if reuse else "",
                },
            ),
        ):
            status = gateway._run_agent_abba(self.args, self.workspace, command or [], 0)
        return status, output.getvalue()

    def bridge(
        self,
        workspace: Path,
        hardware: str,
        profile: str,
        url: str,
        timeout: int,
        command: list[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess:
        self.assertEqual(kwargs["gateway_kind"], "run")
        self.assertTrue(kwargs["reuse_completed"])
        self.assertEqual(kwargs["comparison_run_timeout"], 120)
        options = list(kwargs["gateway_options"])
        self.args.baseline_path = options[options.index("--baseline-path") + 1]
        try:
            status, output = self.measure(reuse=True, command=command)
        except SystemExit as exc:
            # Match the real Gateway subprocess boundary for execution failures.
            return subprocess.CompletedProcess([], 1, "", str(exc))
        return subprocess.CompletedProcess([], status, output, "")

    def verify(self) -> verifier.VerificationResult:
        with patch.object(verifier.main_adapter, "run_sandbox", side_effect=self.bridge):
            return verifier.GatewayABBAValidator(
                hardware="L20N",
                url=self.args.url,
                min_improvement_pct=1.0,
            ).verify(
                self.workspace,
                base_commit=self.base_commit,
                candidate_commit=self.candidate_commit,
                changed_paths=["kernel.py"],
            )

    def test_supervisor_reuses_agent_abba_with_exact_median_and_record(self) -> None:
        status, output = self.measure()
        self.assertEqual(status, 0)
        self.assertNotIn(gateway.SUPERVISOR_MEASUREMENT_PREFIX, output)
        result = self.verify()
        self.assertTrue(result.passed, result.error)
        self.assertTrue(result.reused)
        self.assertEqual(self.calls, 3)
        self.assertAlmostEqual(result.candidate_latency_us, 8.0)
        self.assertAlmostEqual(result.incumbent_latency_us, 11.0)
        self.assertAlmostEqual(result.improvement_pct, 37.5)
        stored = json.loads(Path(result.artifact).read_text())
        self.assertEqual(Path(result.artifact).parent.name, result.gateway_record_id)
        self.assertAlmostEqual(stored["result"]["candidate"]["latency_us_geomean"], 8.0)
        self.assertEqual(result.as_dict()["gateway_record_id"], result.gateway_record_id)

    def test_missing_measurement_is_created_by_shared_runtime(self) -> None:
        result = self.verify()
        self.assertTrue(result.passed, result.error)
        self.assertFalse(result.reused)
        self.assertEqual(self.calls, 3)
        self.assertTrue(self.verify().reused)
        self.assertEqual(self.calls, 3)

    def test_agent_duplicate_still_rejected(self) -> None:
        self.measure()
        with self.assertRaisesRegex(SystemExit, "duplicate_gateway_task"):
            self.measure()
        self.assertEqual(self.calls, 3)

    def test_labels_and_paths_do_not_change_measurement(self) -> None:
        self.measure(command=["python3", "test_kernel.py", "--version", "v99", "--no-memory"])
        (self.workspace / "scratch" / "another.py").write_bytes(self.incumbent)
        self.args.baseline_path = "scratch/another.py"
        self.measure(reuse=True, command=["python3", "test_kernel.py", "--timed-runs", "100"])
        self.assertEqual(self.calls, 3)

    def test_changed_input_kernel_hardware_or_iterations_misses_cache(self) -> None:
        self.measure()
        (self.workspace / "input.py").write_text("def _make_inputs(): return {'x': 1}\n")
        self.measure(reuse=True)
        (self.workspace / "scratch" / "baseline.py").write_text("class Model: value = 2\n")
        self.measure(reuse=True)
        (self.workspace / "kernel.py").write_text("class Model: value = 3\n")
        self.measure(reuse=True)
        self.args.hardware = "H100"
        self.measure(reuse=True)
        self.measure(reuse=True, command=["python3", "test_kernel.py", "--timed-runs", "5"])
        self.assertEqual(self.calls, 18)

    def test_post_commit_edits_are_not_verified(self) -> None:
        (self.workspace / "kernel.py").write_text("class Model: value = 9\n")
        result = self.verify()
        self.assertEqual(result.gate, "ERROR")
        self.assertIn("differs from the committed", result.error)
        self.assertEqual(self.calls, 0)

    def test_runtime_reuses_only_completed_historical_records(self) -> None:
        history = self.root / "episodes"
        old_evidence = history / "e0001" / "supervisor_runtime"
        with patch.dict(os.environ, {gateway.SUPERVISOR_EVIDENCE_ROOT_ENV: str(old_evidence)}):
            _, original = self.measure()
        public = json.loads(next(
            line.removeprefix(gateway.ABBA_RESULT_PUBLIC_PREFIX)
            for line in original.splitlines()
            if line.startswith(gateway.ABBA_RESULT_PUBLIC_PREFIX)
        ))
        with patch.dict(os.environ, {gateway.SUPERVISOR_HISTORY_ROOT_ENV: str(history)}):
            with self.assertRaises(SystemExit) as duplicate:
                self.measure()
            response = json.loads(duplicate.exception.code)
            self.assertEqual(response["error"]["code"], "duplicate_gateway_task")
            self.assertEqual(response["error"]["gateway_record_id"], public["gateway_record_id"])
            self.assertIn("--kind record-read", response["error"]["next_action"])
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(gateway._read_gateway_record(
                    self.workspace, response["error"]["gateway_record_id"],
                ), 0)
            self.assertIn(public["gateway_record_id"], output.getvalue())
            result = self.verify()
        self.assertTrue(result.passed, result.error)
        self.assertTrue(result.reused)
        self.assertEqual(result.gateway_record_id, public["gateway_record_id"])
        self.assertEqual(self.calls, 3)

    def test_agent_historical_evaluate_duplicate_returns_original_record_without_submission(self) -> None:
        args = copy.copy(self.args)
        args.baseline_path = None
        request = {
            "candidate": self.candidate.decode(),
            "mode": "full",
            "reference": {"shapes": {"0": {}}},
        }
        digest = gateway._gateway_task_digest(
            "run", hashlib.sha256(self.candidate).hexdigest(),
            {"request": request, "execution": gateway._gateway_execution_identity(args)},
        )
        history = self.root / "episodes"
        old_evidence = history / "e0001" / "supervisor_runtime"
        with patch.dict(os.environ, {gateway.SUPERVISOR_EVIDENCE_ROOT_ENV: str(old_evidence)}):
            owner, _ = gateway._reserve_gateway_task(self.workspace, digest)
            record = gateway._record_episode_evaluation(
                self.workspace, {"all_pass": True, "latency_us_by_shape": {"0": 5.0}},
                gateway_kind="run", gateway_task_digest=digest,
            )
            gateway._complete_gateway_task(self.workspace, digest, owner, record["record_id"])
        with (
            patch.dict(os.environ, {gateway.SUPERVISOR_HISTORY_ROOT_ENV: str(history)}),
            patch.object(gateway, "_find_agate", return_value="/unused/agate"),
            patch.object(gateway, "_typed_request", return_value=request),
            patch.object(gateway, "_execute_typed_processes") as execute,
        ):
            with self.assertRaises(SystemExit) as duplicate:
                gateway._run_typed_gateway(args, self.workspace, [], "run", [], 0)
            response = json.loads(duplicate.exception.code)
            self.assertEqual(response["error"]["code"], "duplicate_gateway_task")
            self.assertEqual(response["error"]["gateway_record_id"], record["record_id"])
            execute.assert_not_called()
            self.assertEqual(
                gateway._load_gateway_record(self.workspace, record["record_id"])["result"],
                {"all_pass": True, "latency_us_by_shape": {"0": 5.0}},
            )
        self.assertFalse((self.root / "private" / "gateway-tasks" / f"{digest}.json").exists())

    def test_historical_abba_does_not_block_changed_task_or_unrelated_campaign(self) -> None:
        history = self.root / "other-campaign" / "episodes"
        old_evidence = history / "e0001" / "supervisor_runtime"
        with patch.dict(os.environ, {gateway.SUPERVISOR_EVIDENCE_ROOT_ENV: str(old_evidence)}):
            self.measure()
        # Historical records are consulted only through the Supervisor-selected history root.
        self.assertEqual(self.measure()[0], 0)
        self.assertEqual(self.calls, 6)
        with patch.dict(os.environ, {gateway.SUPERVISOR_HISTORY_ROOT_ENV: str(history)}):
            (self.workspace / "kernel.py").write_text("class Model: value = 2\n")
            self.assertEqual(self.measure()[0], 0)
        self.assertEqual(self.calls, 9)

    def test_incomplete_historical_tasks_do_not_block_new_measurement(self) -> None:
        history = self.root / "episodes"
        old_tasks = history / "e0001" / "supervisor_runtime" / "gateway-tasks"
        old_tasks.mkdir(parents=True)
        with patch.dict(os.environ, {gateway.SUPERVISOR_HISTORY_ROOT_ENV: str(history)}):
            for index, status in enumerate(("running", "failed", "completed")):
                with self.subTest(status=status):
                    digest = f"{index:064x}"
                    # A terminal marker without a recorded result is not a completed measurement.
                    (old_tasks / f"{digest}.json").write_text(json.dumps({"status": status}))
                    owner, record_id = gateway._reserve_gateway_task(self.workspace, digest)
                    self.assertIsNotNone(owner)
                    self.assertIsNone(record_id)

    def test_inflight_is_not_submitted_again(self) -> None:
        digest = "a" * 64
        owner, _ = gateway._reserve_gateway_task(self.workspace, digest)
        self.assertIsNotNone(owner)
        with patch.dict(os.environ, {gateway.REUSE_GATEWAY_RESULTS_ENV: "1"}):
            self.assertEqual(gateway._reserve_gateway_task(self.workspace, digest), (None, None))
            with self.assertRaisesRegex(SystemExit, "currently running"):
                gateway._reusable_gateway_record(self.workspace, digest, None)
        gateway._abandon_gateway_task(self.workspace, digest, owner)
        self.assertIsNotNone(gateway._reserve_gateway_task(self.workspace, digest)[0])

    def test_corrupt_record_is_rebuilt_before_authorizing_promotion(self) -> None:
        _, output = self.measure(reuse=True)
        receipt = json.loads(
            next(
                line.removeprefix(gateway.SUPERVISOR_MEASUREMENT_PREFIX)
                for line in output.splitlines()
                if line.startswith(gateway.SUPERVISOR_MEASUREMENT_PREFIX)
            )
        )
        artifact = Path(receipt["artifact"])
        original = artifact.read_bytes()
        artifact.with_name("kernel.py").write_text("tampered")
        task_digest = json.loads(original)["gateway_task_digest"]
        with self.assertRaisesRegex(RuntimeError, "recorded digest"):
            gateway._reusable_gateway_record(
                self.workspace, task_digest, receipt["gateway_record_id"],
            )
        result = self.verify()
        self.assertTrue(result.passed, result.error)
        self.assertFalse(result.reused)
        self.assertNotEqual(result.gateway_record_id, receipt["gateway_record_id"])
        self.assertIsNotNone(gateway._validated_cached_gateway_record(
            self.workspace, task_digest, result.gateway_record_id,
        ))
        self.assertEqual(Path(result.artifact).with_name("kernel.py").read_bytes(), self.candidate)
        self.assertEqual(artifact.read_bytes(), original)
        self.assertEqual(artifact.with_name("kernel.py").read_text(), "tampered")
        # Independent, validated physical-batch checkpoints can rebuild the record.
        self.assertEqual(self.calls, 3)
        self.assertEqual(self.verify().gateway_record_id, result.gateway_record_id)
        self.assertEqual(self.calls, 3)

    def test_corrupt_record_cannot_authorize_promotion_if_rebuild_fails(self) -> None:
        self.measure()
        record = gateway._visible_gateway_records(self.workspace)[-1]
        (record["record_dir"] / "kernel.py").write_text("tampered")
        with patch(
            "supervisor.abba_checkpoints.AbbaBatchStore.load",
            side_effect=RuntimeError("checkpoint unavailable"),
        ):
            result = self.verify()
        self.assertEqual(result.gate, "ERROR")
        self.assertIn("checkpoint unavailable", result.error)
        self.assertFalse(result.gateway_record_id)
        self.assertFalse(result.artifact)
        self.assertEqual(self.calls, 3)
        marker = self.root / "private/gateway-tasks" / f"{record['gateway_task_digest']}.json"
        self.assertFalse(marker.exists())
        # Recovery failure released the reservation, so a later request can retry.
        self.assertTrue(self.verify().passed)

    def test_evaluate_record_is_reused_without_gpu_calls(self) -> None:
        args = copy.copy(self.args)
        args.baseline_path = None
        request = {
            "candidate": self.candidate.decode(),
            "mode": "full",
            "reference": {
                "shapes": {"0": {}},
            },
        }
        digest = gateway._gateway_task_digest(
            "run",
            hashlib.sha256(self.candidate).hexdigest(),
            {
                "request": request,
                "execution": gateway._gateway_execution_identity(args),
            },
        )
        owner, _ = gateway._reserve_gateway_task(self.workspace, digest)
        # Use more Shapes than the public projection limit to verify trusted consumers
        # read the full persisted result rather than a truncated Agent response.
        expected = {"all_pass": True, "latency_us_by_shape": {str(i): 5.0 for i in range(1000)}}
        record = gateway._record_episode_evaluation(
            self.workspace,
            expected,
            gateway_kind="run",
            gateway_task_digest=digest,
        )
        gateway._complete_gateway_task(self.workspace, digest, owner, record["record_id"])
        output = io.StringIO()
        with (
            patch.dict(os.environ, {gateway.REUSE_GATEWAY_RESULTS_ENV: "1"}),
            patch.object(gateway, "_typed_request", return_value=request),
            patch.object(gateway, "_execute_typed_processes") as execute,
            redirect_stdout(output),
        ):
            status = gateway._run_typed_gateway(args, self.workspace, [], "run", [], 0)
        self.assertEqual(status, 0)
        execute.assert_not_called()
        self.assertEqual(_test_result_from_stdout(output.getvalue()), expected)

    def test_sol_command_does_not_invent_native_options(self) -> None:
        command = gateway._comparison_command(["python3", "test_kernel.py"], sol=True)
        self.assertNotIn("--timed-runs", command)
        self.assertIn("--multi-seed", command)

    def test_sol_workload_can_use_shared_abba(self) -> None:
        (self.workspace / "workload.jsonl").write_text('{"uuid":"0"}\n')
        (self.workspace / "shapes.json").unlink()
        result = self.verify()
        self.assertTrue(result.passed, result.error)
        self.assertEqual(self.calls, 3)

    def test_infrastructure_failure_abandons_reservation(self) -> None:
        def fail(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
            if len(command) > 1 and str(command[1]).endswith("supervisor/gateway.py"):
                raise RuntimeError("remote runtime environment unavailable")
            return self.real_run(command, **kwargs)

        with (
            patch.object(gateway.subprocess, "run", side_effect=fail),
            self.assertRaisesRegex(SystemExit, "abba_comparison_unavailable"),
        ):
            gateway._run_agent_abba(self.args, self.workspace, [], 0)
        self.assertEqual(list((self.root / "private" / "gateway-tasks").glob("*.json")), [])
        self.assertTrue(self.verify().passed)

    def test_failed_measurement_is_reused_but_not_promoted(self) -> None:
        _, output = self.measure(reuse=True)
        receipt = json.loads(
            next(
                line.removeprefix(gateway.SUPERVISOR_MEASUREMENT_PREFIX)
                for line in output.splitlines()
                if line.startswith(gateway.SUPERVISOR_MEASUREMENT_PREFIX)
            )
        )
        path = Path(receipt["artifact"])
        record = json.loads(path.read_text())
        record["result"]["correct"] = False
        record["result"]["candidate"]["correct"] = False
        record["result"]["status"] = "failed"
        path.write_text(json.dumps(record))
        result = self.verify()
        self.assertEqual(result.gate, "FAIL")
        self.assertTrue(result.reused)
        self.assertEqual(self.calls, 3)


if __name__ == "__main__":
    unittest.main()
