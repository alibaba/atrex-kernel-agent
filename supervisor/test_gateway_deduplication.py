"""Task deduplication tests; no model, GPU, or live Gateway is contacted."""

from __future__ import annotations

import io
import json
import os
import subprocess
import tempfile
import unittest
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from supervisor import gateway


class GatewayDeduplicationTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        for name, content in {
            "kernel.py": "class Model: pass\n",
            "reference.py": "class Model: pass\n",
            "input.py": "def _make_inputs(): return {}\n",
            "shapes.json": '{"0":{"init_kwargs":{},"input_kwargs":{}}}',
            "tools/probe.py": "print('probe')\n",
        }.items():
            path = self.workspace / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        self.history = self.root / "episodes"
        self.evidence = self.root / "current-evidence"
        stack = self.enterContext(ExitStack())
        stack.enter_context(patch.dict(os.environ, {
            gateway.SUPERVISOR_EVIDENCE_ROOT_ENV: str(self.evidence),
            gateway.SUPERVISOR_HISTORY_ROOT_ENV: "",
            gateway.REUSE_GATEWAY_RESULTS_ENV: "",
            gateway.INTERNAL_MEASUREMENT_ENV: "",
            gateway.PRIVATE_REFERENCE_ENV: "",
            gateway.ATREX_BENCH_RUNTIME_ENV: "",
        }))
        stack.enter_context(patch.object(gateway, "_find_agate", return_value="/unused/agate"))
        stack.enter_context(patch.object(gateway, "_resolved_gateway_url", return_value="https://gateway.invalid"))
        stack.enter_context(patch.object(gateway, "_uses_standard_oss_gateway", return_value=False))
        self.typed = stack.enter_context(patch.object(
            gateway, "_execute_typed_processes", side_effect=self.typed_response,
        ))
        self.dev_run = stack.enter_context(patch.object(
            gateway, "_run_agate_with_cancel_retry", return_value=self.dev_response(),
        ))
        stack.enter_context(patch.object(gateway, "_optimizer_result_from_eval", return_value={
            "all_pass": True, "latency_us_by_shape": {"0": 5.0},
            "latency_us_geomean": 5.0, "latency_us_arith_mean": 5.0,
        }))

    @staticmethod
    def typed_response(*args, **kwargs) -> list[subprocess.CompletedProcess]:
        return [subprocess.CompletedProcess([], 0, json.dumps({
            "status": "succeeded", "job_id": "job-test",
            "result": {"passed": True, "compile_ok": True, "kernels": [], "format": "ptx"},
        }), "")]

    @staticmethod
    def dev_response() -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess([], 0, json.dumps({
            "status": "succeeded", "job_id": "dev-test",
            "result": {"stdout": "probe result", "stderr": "", "exit_code": 0},
        }), "")

    def run_task(self, *arguments: str) -> tuple[int, str]:
        output = io.StringIO()
        with redirect_stdout(output), redirect_stderr(io.StringIO()):
            code = gateway._main([
                "--workspace", str(self.workspace), "--hardware", "L20N",
                "--url", "https://gateway.invalid", "--no-sync", *arguments,
            ])
        return code, output.getvalue()

    def dev(self, *arguments: str) -> tuple[int, str]:
        return self.run_task("--kind", "dev", *arguments, "--", "python3", "tools/probe.py")

    def assert_duplicate(self, call, record_id: str) -> None:
        with self.assertRaises(SystemExit) as failure:
            call()
        response = json.loads(failure.exception.code)
        self.assertEqual(response["error"]["code"], "duplicate_gateway_task")
        self.assertEqual(response["error"]["gateway_record_id"], record_id)
        self.assertIn("--kind record-read", response["error"]["next_action"])

    def test_all_typed_operations_reject_current_and_historical_duplicates(self) -> None:
        cases = [
            (("--kind", "run", "--mode", "full"), 3),
            (("--kind", "run", "--mode", "correctness_only"), 1),
            (("--kind", "profile"), 1),
            (("--kind", "check"), 1),
            (("--kind", "disassemble"), 1),
        ]
        old_evidence = self.history / "e0001" / "supervisor_runtime"
        for arguments, repetitions in cases:
            with self.subTest(arguments=arguments):
                self.typed.reset_mock()
                with patch.dict(os.environ, {gateway.SUPERVISOR_EVIDENCE_ROOT_ENV: str(old_evidence)}):
                    code, output = self.run_task(*arguments)
                    self.assertEqual(code, 0)
                    record_id = gateway._visible_gateway_records(self.workspace)[-1]["record_id"]
                    self.assertIn(record_id, output)
                    self.assert_duplicate(lambda: self.run_task(*arguments), record_id)
                with patch.dict(os.environ, {gateway.SUPERVISOR_HISTORY_ROOT_ENV: str(self.history)}):
                    self.assert_duplicate(lambda: self.run_task(*arguments), record_id)
                    self.assertIn(record_id, self.run_task("--kind", "record-read", "--record-id", record_id)[1])
                self.assertEqual(self.typed.call_count, repetitions)

    def test_typed_selectors_inputs_dependencies_and_environment_identify_new_tasks(self) -> None:
        cases = [
            ("--kind", "profile"),
            ("--kind", "profile", "--kernel-name", "other"),
            ("--kind", "profile", "--launch-count", "2"),
            ("--kind", "profile", "--requirement", "sample-package==1"),
            ("--kind", "profile", "--env", "MODE=other"),
            ("--kind", "check"),
            ("--kind", "check", "--sanitize", "memcheck"),
            ("--kind", "check", "--arch", "sm_90"),
            ("--kind", "check", "--timeout", "120"),
            ("--kind", "disassemble"),
            ("--kind", "disassemble", "--format", "ptx"),
            ("--kind", "disassemble", "--timeout", "120"),
        ]
        for arguments in cases:
            with self.subTest(arguments=arguments):
                self.assertEqual(self.run_task(*arguments)[0], 0)
        (self.workspace / "input.py").write_text("def _make_inputs(): return {'changed': 1}\n")
        self.assertEqual(self.run_task("--kind", "check")[0], 0)
        self.assertEqual(self.typed.call_count, len(cases) + 1)

    def test_typed_exceptions_and_infra_failures_release_reservations(self) -> None:
        for kind in ("run", "profile", "check", "disassemble"):
            with self.subTest(kind=kind):
                self.typed.side_effect = RuntimeError("transport failure")
                with self.assertRaisesRegex(RuntimeError, "transport failure"):
                    self.run_task("--kind", kind)
                self.assertEqual(list((self.evidence / "gateway-tasks").glob("*.json")), [])
                self.typed.side_effect = lambda *a, **kw: [subprocess.CompletedProcess([], 1, json.dumps({
                    "status": "failed", "error": {"error_class": "infra", "reason": "exec_failed"},
                }), "")]
                self.assertNotEqual(self.run_task("--kind", kind)[0], 0)
                self.assertEqual(list((self.evidence / "gateway-tasks").glob("*.json")), [])
        self.typed.side_effect = self.typed_response
        self.assertEqual(self.run_task("--kind", "check")[0], 0)

    def test_dev_hashes_uploaded_contents_not_timestamps_or_unselected_scratch(self) -> None:
        old_evidence = self.history / "e0001" / "supervisor_runtime"
        with patch.dict(os.environ, {gateway.SUPERVISOR_EVIDENCE_ROOT_ENV: str(old_evidence)}):
            self.assertEqual(self.dev()[0], 0)
            record_id = gateway._visible_gateway_records(self.workspace)[-1]["record_id"]
        probe = self.workspace / "tools/probe.py"
        os.utime(probe, (100, 100))
        scratch = self.workspace / "scratch"
        scratch.mkdir()
        (scratch / "unrelated.txt").write_text("not uploaded")
        with patch.dict(os.environ, {gateway.SUPERVISOR_HISTORY_ROOT_ENV: str(self.history)}):
            self.assert_duplicate(self.dev, record_id)
            self.assertIn(record_id, self.run_task("--kind", "record-read", "--record-id", record_id)[1])
            self.assertEqual(self.dev_run.call_count, 1)
            probe.write_text("print('changed probe')\n")
            self.assertEqual(self.dev()[0], 0)
            self.assertEqual(self.dev("--env", "MODE=other")[0], 0)
            self.assertEqual(self.dev("--input", "scratch/unrelated.txt")[0], 0)
            self.assertEqual(self.dev_run.call_count, 4)

    def test_dev_failure_does_not_lock_out_retries_but_candidate_failure_is_cached(self) -> None:
        self.dev_run.side_effect = RuntimeError("transport failure")
        with self.assertRaisesRegex(RuntimeError, "transport failure"):
            self.dev()
        self.dev_run.side_effect = None
        self.dev_run.return_value = subprocess.CompletedProcess([], 1, json.dumps({
            "status": "failed", "error": {"error_class": "infra", "reason": "exec_failed"},
        }), "")
        self.assertEqual(self.dev()[0], 1)
        self.assertEqual(list((self.evidence / "gateway-tasks").glob("*.json")), [])
        self.dev_run.return_value = subprocess.CompletedProcess([], 1, json.dumps({
            "status": "succeeded", "result": {"exit_code": 1, "stderr": "bad probe"},
        }), "")
        self.assertEqual(self.dev()[0], 1)
        record_id = gateway._visible_gateway_records(self.workspace)[-1]["record_id"]
        self.assert_duplicate(self.dev, record_id)
        self.assertEqual(self.dev_run.call_count, 3)

    def test_dev_record_binds_the_uploaded_kernel_even_if_workspace_changes(self) -> None:
        original = (self.workspace / "kernel.py").read_bytes()
        def complete(**kwargs):
            (self.workspace / "kernel.py").write_text("class Model: changed = True\n")
            return self.dev_response()
        self.dev_run.side_effect = complete
        self.assertEqual(self.dev()[0], 0)
        record = gateway._visible_gateway_records(self.workspace)[-1]
        self.assertEqual((record["record_dir"] / "kernel.py").read_bytes(), original)

    def test_internal_abba_measurements_are_not_deduplicated_as_agent_dev_tasks(self) -> None:
        with patch.dict(os.environ, {gateway.INTERNAL_MEASUREMENT_ENV: "1"}):
            self.assertEqual(self.dev()[0], 0)
            self.assertEqual(self.dev()[0], 0)
        self.assertEqual(self.dev_run.call_count, 2)
        self.assertFalse((self.evidence / "gateway-tasks").exists())

    def test_direct_http_dev_and_profile_fallback_are_deduplicated(self) -> None:
        with (
            patch.object(gateway, "_find_agate", return_value=None),
            patch.object(gateway, "_run_direct_gateway", return_value=self.dev_response()) as execute,
        ):
            self.assertEqual(self.dev()[0], 0)
            record_id = gateway._visible_gateway_records(self.workspace)[-1]["record_id"]
            self.assert_duplicate(self.dev, record_id)
            self.assertEqual(execute.call_count, 1)
        arguments = ("--kind", "profile", "--include-raw-profile")
        self.assertEqual(self.run_task(*arguments)[0], 0)
        record = gateway._visible_gateway_records(self.workspace)[-1]
        self.assertEqual(record["gateway_kind"], "profile")
        self.assert_duplicate(lambda: self.run_task(*arguments), record["record_id"])
        self.assertEqual(self.dev_run.call_count, 1)

    def test_dev_evaluator_record_is_bound_to_task(self) -> None:
        self.dev_run.return_value = subprocess.CompletedProcess([], 0, json.dumps({
            "status": "succeeded", "result": {"exit_code": 0, "stdout":
                gateway.TEST_RESULT_PREFIX + json.dumps({
                    "all_pass": True, "latency_us_geomean": 5.0,
                    "latency_us_by_shape": {"0": 5.0},
                })},
        }), "")
        with patch.object(gateway, "_run_typed_gateway", return_value=None):
            self.assertEqual(self.run_task("--kind", "run")[0], 0)
            record = gateway._visible_gateway_records(self.workspace)[-1]
            self.assertTrue(record["gateway_task_digest"])
            self.assertEqual(record["result"]["latency_us_geomean"], 5.0)
            self.assert_duplicate(lambda: self.run_task("--kind", "run"), record["record_id"])
        self.assertEqual(self.dev_run.call_count, 1)

    def test_environment_queries_remain_repeatable(self) -> None:
        with patch.object(gateway, "_run_environment_query", return_value=0) as query:
            self.assertEqual(self.run_task("--kind", "env")[0], 0)
            self.assertEqual(self.run_task("--kind", "env")[0], 0)
        self.assertEqual(query.call_count, 2)
        self.assertFalse((self.evidence / "gateway-tasks").exists())

    def test_cli_reads_private_candidate_snapshot_for_every_typed_operation(self) -> None:
        for kind in ("run", "profile", "check", "disassemble"):
            with self.subTest(kind=kind):
                args = gateway.build_parser().parse_args([
                    "--kind", kind, "--hardware", "L20N", "--url", "https://gateway.invalid",
                ])
                request = gateway._typed_request(self.workspace, "L20N", args.timeout, [], [], kind)
                expected = request["candidate"]
                (self.workspace / "kernel.py").write_text("class Model: later_edit = True\n")
                sidecar = self.root / "sidecar" / kind
                command = gateway._typed_agate_command(
                    "/unused/agate", args, self.workspace, kind, request, 0,
                    request_sidecar_dir=sidecar,
                )
                self.assertIn(str(sidecar / "kernel.py"), command)
                self.assertNotIn(str(self.workspace / "kernel.py"), command)
                self.assertEqual((sidecar / "kernel.py").read_text(), expected)


if __name__ == "__main__":
    unittest.main()
