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
    def test_source_rejection_is_terminal(self):
        action = mock.Mock(return_value=process(
            stderr="source validation failed: blocked import: ctypes"
        ))
        with mock.patch.object(sandbox.time, "sleep") as sleep:
            result = sandbox._run_eval_with_retry(action, generalized=False)
        self.assertEqual(result.returncode, 2)
        action.assert_called_once_with()
        sleep.assert_not_called()

    def test_structured_source_rejection_is_terminal(self):
        action = mock.Mock(return_value=process(1, job={
            "job_id": "ev_reject", "status": "failed", "error": {
                "error_class": "input", "reason": "source_validation_failed",
            },
        }))
        with mock.patch.object(sandbox.time, "sleep") as sleep:
            result = sandbox._run_eval_with_retry(action, generalized=False)
        self.assertEqual(result.returncode, 1)
        action.assert_called_once_with()
        sleep.assert_not_called()

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

    def test_cancellation_without_outcome_uses_only_outer_retry(self):
        action = mock.Mock(side_effect=[
            process(1, job={"job_id": "ev_cancelled", "status": "cancelled"}),
            process(0, job={"job_id": "ev_ok", "status": "succeeded", "result": {}}),
        ])
        with mock.patch.object(sandbox, "EVAL_RETRY_DELAYS", (0,)), \
             mock.patch.object(sandbox.time, "sleep"), \
             contextlib.redirect_stderr(io.StringIO()):
            result = sandbox._run_eval_with_retry(action, generalized=False)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(action.call_count, 2)

    def test_direct_eval_can_disable_nested_cancel_resubmission(self):
        gateway = mock.Mock(side_effect=[
            {"job_id": "ev_cancelled"},
            {"job_id": "ev_cancelled", "status": "cancelled"},
        ])
        with mock.patch.object(sandbox, "_gateway_json", gateway):
            result = sandbox._run_direct_job(
                url="https://gateway.invalid",
                kind="eval",
                payload={},
                timeout=60,
                queue_wait_grace=0,
                retry_cancelled_without_outcome=False,
            )
        self.assertEqual(result.returncode, 1)
        self.assertEqual(gateway.call_count, 2)

    def test_http_source_rejection_is_terminal(self):
        action = mock.Mock(side_effect=sandbox.GatewayHTTPError(
            400, "source validation failed"
        ))
        with mock.patch.object(sandbox.time, "sleep") as sleep, \
             self.assertRaises(sandbox.GatewayHTTPError):
            sandbox._run_eval_with_retry(action, generalized=False)
        action.assert_called_once_with()
        sleep.assert_not_called()

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
            process(stderr="connection reset by peer: hidden-shape-secret"),
            process(0),
        ])
        output = io.StringIO()
        with mock.patch.object(sandbox, "EVAL_RETRY_DELAYS", (0,)), \
             mock.patch.object(sandbox.time, "sleep"), \
             contextlib.redirect_stderr(output):
            sandbox._run_eval_with_retry(action, generalized=True)
        self.assertNotIn("hidden-shape-secret", output.getvalue())
        self.assertIn("details withheld", output.getvalue())

    def test_default_budget_allows_one_outer_retry(self):
        self.assertEqual(sandbox.EVAL_RETRY_DELAYS, (5,))

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
                 mock.patch.object(sandbox, "_run_agate_once", return_value=process(
                     stderr="connection reset by peer: hidden-shape-secret")) as runner, \
                 contextlib.redirect_stderr(output):
                result = sandbox._run_typed_gateway(args, Path(tmp), [], "run", [], 0)
            self.assertEqual(result, 2)
            self.assertEqual(runner.call_count, 2)
            self.assertIn(sandbox.EVAL_RETRIES_EXHAUSTED, output.getvalue())
            if generalized:
                self.assertNotIn("hidden-shape-secret", output.getvalue())


if __name__ == "__main__":
    unittest.main()
