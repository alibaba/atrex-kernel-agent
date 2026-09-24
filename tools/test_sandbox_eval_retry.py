"""Offline regressions: eval failures must not change the evaluator route."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools import sandbox


def process(code: int = 2, *, stderr: str = "", job: dict | None = None):
    return subprocess.CompletedProcess(
        ["agate", "eval"], code, json.dumps(job) if job else "", stderr
    )


class EvalRetryTests(unittest.TestCase):
    def test_source_rejection_retries_eval_and_retains_original_error(self):
        action = mock.Mock(side_effect=[
            process(stderr="source validation failed: blocked import: ctypes"),
            process(0, job={"job_id": "ev_ok", "status": "succeeded", "result": {}}),
        ])
        output = io.StringIO()
        with mock.patch.object(sandbox, "EVAL_RETRY_DELAYS", (5, 15)), \
             mock.patch.object(sandbox.time, "sleep") as sleep, \
             contextlib.redirect_stderr(output):
            result = sandbox._run_eval_with_retry(action, generalized=False)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(action.call_count, 2)
        sleep.assert_called_once_with(5)
        self.assertIn("blocked import: ctypes", output.getvalue())
        self.assertIn("retrying the same eval", output.getvalue())

    def test_persistent_rejection_is_bounded_and_returns_error(self):
        action = mock.Mock(return_value=process(stderr="source validation failed"))
        with mock.patch.object(sandbox, "EVAL_RETRY_DELAYS", (0, 0, 0)), \
             mock.patch.object(sandbox.time, "sleep"), \
             contextlib.redirect_stderr(io.StringIO()):
            result = sandbox._run_eval_with_retry(action, generalized=False)
        self.assertEqual(action.call_count, 4)
        self.assertEqual(result.returncode, 2)
        self.assertIn(sandbox.EVAL_RETRIES_EXHAUSTED, result.stderr)
        self.assertIn("source validation failed", result.stderr)

    def test_completed_kernel_failures_and_code_errors_are_not_retried(self):
        for job in (
            {"job_id": "ev_test", "status": "failed", "result": {"all_pass": False}},
            {"job_id": "ev_test", "status": "failed", "error": {
                "error_class": "code", "reason": "code_execution_failed",
                "message": "source validation failed in candidate output",
            }},
            {"job_id": "ev_test", "status": "failed", "error": {
                "error_class": "infra", "reason": "native_crash",
            }},
        ):
            with self.subTest(job=job):
                action = mock.Mock(return_value=process(1, job=job))
                sandbox._run_eval_with_retry(action, generalized=False)
                action.assert_called_once_with()

    def test_infrastructure_job_retries_without_replacing_request(self):
        action = mock.Mock(side_effect=[
            process(job={"job_id": "ev_fail", "status": "failed", "error": {
                "error_class": "infra", "reason": "submit_failed",
            }}),
            process(0, job={"job_id": "ev_ok", "status": "succeeded", "result": {}}),
        ])
        with mock.patch.object(sandbox, "EVAL_RETRY_DELAYS", (0,)), \
             mock.patch.object(sandbox.time, "sleep"), \
             contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(sandbox._run_eval_with_retry(action, generalized=False).returncode, 0)
        self.assertEqual(action.call_count, 2)

    def test_http_source_rejection_retries_without_dev(self):
        action = mock.Mock(side_effect=[
            sandbox.GatewayHTTPError(400, "source validation failed"),
            process(0),
        ])
        with mock.patch.object(sandbox, "EVAL_RETRY_DELAYS", (0,)), \
             mock.patch.object(sandbox.time, "sleep"), \
             contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(sandbox._run_eval_with_retry(action, generalized=False).returncode, 0)
        self.assertEqual(action.call_count, 2)

    def test_http_retries_exhausted_keep_error_and_marker(self):
        action = mock.Mock(side_effect=sandbox.GatewayHTTPError(503, "unavailable"))
        output = io.StringIO()
        with mock.patch.object(sandbox, "EVAL_RETRY_DELAYS", (0,)), \
             mock.patch.object(sandbox.time, "sleep"), \
             contextlib.redirect_stderr(output), \
             self.assertRaises(sandbox.GatewayHTTPError):
            sandbox._run_eval_with_retry(action, generalized=False)
        self.assertEqual(action.call_count, 2)
        self.assertIn(sandbox.EVAL_RETRIES_EXHAUSTED, output.getvalue())

    def test_generalized_retry_logs_do_not_expose_hidden_cases(self):
        action = mock.Mock(side_effect=[
            process(stderr="source validation failed: hidden-shape-secret"),
            process(0),
        ])
        output = io.StringIO()
        with mock.patch.object(sandbox, "EVAL_RETRY_DELAYS", (0,)), \
             mock.patch.object(sandbox.time, "sleep"), \
             contextlib.redirect_stderr(output):
            sandbox._run_eval_with_retry(action, generalized=True)
        self.assertNotIn("hidden-shape-secret", output.getvalue())
        self.assertIn("details withheld", output.getvalue())

    def test_typed_command_uses_canonical_eval_entry(self):
        args = argparse.Namespace(url=None, gateway_profile=None, hardware="test", timeout=60, env=[])
        request = {"options": {"num_correctness_cases": 1, "bench_iters": 2},
                   "reference": {"operator": "test"}}
        command = sandbox._typed_agate_command(
            "agate", args, Path("/workspace"), "run", request, 0,
            reference_dir=Path("/reference"),
        )
        self.assertEqual(command[:4], ["agate", "eval", "--backend", "atrex"])

    def test_failed_typed_eval_returns_error_not_dev_fallback_sentinel(self):
        request = {"reference": {"shapes": {"0": {}}}}
        args = argparse.Namespace(
            url=None, gateway_profile=None, hardware="test", timeout=60, env=[],
            profiler=None, profile_level="sol", profile_counter=[], kernel_regex=None,
            top_kernels=None, dry_run=False,
        )
        for generalized in (False, True):
            output = io.StringIO()
            with self.subTest(generalized=generalized), tempfile.TemporaryDirectory() as tmp, \
                 mock.patch.object(sandbox, "_is_generalized_workspace", return_value=generalized), \
                 mock.patch.object(sandbox, "_typed_request", return_value=request), \
                 mock.patch.object(sandbox, "_find_agate", return_value="agate"), \
                 mock.patch.object(sandbox, "_shape_batch_reference"), \
                 mock.patch.object(sandbox, "_typed_agate_command", return_value=["agate", "eval"]), \
                 mock.patch.object(sandbox, "EVAL_RETRY_DELAYS", (0,)), \
                 mock.patch.object(sandbox.time, "sleep"), \
                 mock.patch.object(sandbox, "_run_agate_with_cancel_retry", return_value=process(
                     stderr="source validation failed: hidden-shape-secret")) as runner, \
                 contextlib.redirect_stderr(output):
                result = sandbox._run_typed_gateway(args, Path(tmp), [], "run", [], 0)
            self.assertEqual(result, 2)
            self.assertEqual(runner.call_count, 2)
            self.assertIn(sandbox.EVAL_RETRIES_EXHAUSTED, output.getvalue())
            if generalized:
                self.assertNotIn("hidden-shape-secret", output.getvalue())


if __name__ == "__main__":
    unittest.main()
