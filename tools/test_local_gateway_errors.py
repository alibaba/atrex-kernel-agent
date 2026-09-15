"""Local error contracts and scheduler shutdown; no GPU or HTTP server required."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from supervisor import gateway
from supervisor.gateway_errors import LOCAL_INFRASTRUCTURE_REASONS
from tools import local_gateway as local


class LocalGatewayErrorTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.store = local.JobStore(self.root / "jobs.db")
        self.addCleanup(lambda: self.store.close())
        self.scheduler = local.LocalScheduler(self.store, self.root / "jobs", 1, 4096)

    def claim(self, kind="dev"):
        job, _ = self.store.create(kind, {}, "test-trace")
        self.assertEqual(self.store.claim_next()[0], job["job_id"])
        return job["job_id"]

    def assert_infrastructure(self, job, reason):
        self.assertEqual(job["status"], "failed")
        self.assertEqual(job["error"]["reason"], reason)
        self.assertEqual(job["error"]["error_class"], "infra")
        self.assertEqual(job["error"]["details"]["failure_origin"], "infrastructure")
        self.assertTrue(gateway._infrastructure_failure(job))
        self.assertFalse(gateway._cacheable_gateway_outcome(job))

    def test_error_contract_and_projection(self):
        for reason in LOCAL_INFRASTRUCTURE_REASONS:
            with self.subTest(reason=reason):
                error = local._error(reason, "test failure", "req-test", worker=1)
                self.assertEqual(error["trace_id"], "req-test")
                self.assertEqual(error["details"]["worker"], 1)
                job = {"status": "failed", "result": None, "error": error}
                self.assert_infrastructure(job, reason)
                response = gateway._agent_gateway_failure(job, None, generalized=False)
                self.assertEqual(response["error"]["code"], "gateway_infrastructure")
                self.assertFalse(response["repairable"])

    def test_command_and_request_errors_are_not_retried_as_infrastructure(self):
        for reason in ("command_failed", "profiler_failed", "output_too_large", "validation_error"):
            with self.subTest(reason=reason):
                error = local._error(reason, "test failure")
                job = {"status": "failed", "error": error}
                self.assertNotEqual(error["error_class"], "infra")
                self.assertNotIn("failure_origin", error["details"])
                self.assertFalse(gateway._infrastructure_failure(job))

    def test_restart_marks_running_jobs_infra_and_preserves_queued_jobs(self):
        running = self.claim()
        queued, _ = self.store.create("dev", {}, "queued-trace")
        self.store.close()
        self.store = local.JobStore(self.root / "jobs.db")
        self.assert_infrastructure(self.store.get(running), "scheduler_restarted")
        self.assertEqual(self.store.get(queued["job_id"])["status"], "queued")

    def test_shutdown_records_infra_before_terminated_child_can_publish_failure(self):
        job_id = self.claim()
        child = Mock()
        thread = Mock()
        thread.is_alive.side_effect = [True, False]
        self.scheduler._threads = [thread]
        self.scheduler._processes[job_id] = child

        def terminated(_process):
            self.assert_infrastructure(self.store.get(job_id), "scheduler_stopped")
            # Model the worker waking up after SIGTERM and processing its exit code.
            self.scheduler._complete_job(
                job_id, "dev", {}, self.root,
                {"exit_code": -15, "stdout": "", "stderr": ""},
            )

        with patch.object(self.scheduler, "_terminate", side_effect=terminated) as terminate:
            self.scheduler.stop()
        terminate.assert_called_once_with(child)
        self.assert_infrastructure(self.store.get(job_id), "scheduler_stopped")

    def test_whole_job_timeout_and_setup_exception_are_infrastructure(self):
        job_id = self.claim()
        with patch.object(
            self.scheduler, "_prepare_job",
            return_value=([sys.executable, "-c", "import time; time.sleep(30)"], 0.05),
        ):
            self.scheduler._execute(job_id, "dev", {})
        self.assert_infrastructure(self.store.get(job_id), "command_timeout")
        job_id = self.claim()
        with patch.object(self.scheduler, "_prepare_job", side_effect=FileNotFoundError("tool missing")):
            self.scheduler._execute(job_id, "dev", {})
        self.assert_infrastructure(self.store.get(job_id), "execution_error")

    def test_real_dev_nonzero_exit_remains_a_command_failure(self):
        job_id = self.claim()
        with patch.object(
            self.scheduler, "_prepare_job",
            return_value=([sys.executable, "-c", "raise SystemExit(2)"], 5),
        ):
            self.scheduler._execute(job_id, "dev", {})
        job = self.store.get(job_id)
        self.assertEqual(job["error"]["reason"], "command_failed")
        self.assertEqual(job["result"]["exit_code"], 2)
        self.assertFalse(gateway._infrastructure_failure(job))
        self.assertTrue(gateway._cacheable_gateway_outcome(job))

    def test_structured_negative_verdicts_remain_results(self):
        for kind, filename, result in (
            ("eval", "eval_output/eval_result.json", {"all_pass": False, "reason": "candidate_timeout"}),
            ("compile", "diagnostic-result.json", {"passed": False, "failure_stage": "compile"}),
            ("disassemble", "diagnostic-result.json", {"passed": False, "failure_stage": "launch"}),
        ):
            with self.subTest(kind=kind):
                job_id = self.claim(kind)
                workdir = self.root / kind
                path = workdir / filename
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(result))
                self.scheduler._complete_job(
                    job_id, kind, {}, workdir, {"exit_code": 1, "stdout": "", "stderr": ""},
                )
                job = self.store.get(job_id)
                self.assertEqual(job["status"], "succeeded")
                self.assertEqual(job["result"], result)
                self.assertIsNone(job["error"])
                self.assertTrue(gateway._cacheable_gateway_outcome(job))

    def test_missing_and_unreadable_driver_results_are_infrastructure(self):
        for kind, filename, missing, invalid in (
            ("eval", "eval_output/eval_result.json", "evaluator_failed", "invalid_eval_result"),
            ("compile", "diagnostic-result.json", "diagnostic_failed", "invalid_diagnostic_result"),
        ):
            for has_invalid_file in (False, True):
                with self.subTest(kind=kind, invalid=has_invalid_file):
                    job_id = self.claim(kind)
                    workdir = self.root / f"{kind}-{has_invalid_file}"
                    workdir.mkdir()
                    if has_invalid_file:
                        path = workdir / filename
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text("{truncated")
                    self.scheduler._complete_job(
                        job_id, kind, {}, workdir, {"exit_code": 1, "stdout": "", "stderr": ""},
                    )
                    self.assert_infrastructure(
                        self.store.get(job_id), invalid if has_invalid_file else missing,
                    )

    def test_standalone_scheduler_entrypoint_can_import_contract(self):
        result = subprocess.run(
            [sys.executable, str(Path(local.__file__).resolve()), "--help"],
            cwd=self.root, capture_output=True, text=True, timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
