"""Cancelled jobs remain auditable, but cannot become permanent measurement facts."""

from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from supervisor import gateway
from supervisor import test_gateway_deduplication as fixtures
from supervisor.errors import RuntimeStateError

REAL_RETRY = gateway._run_agate_with_cancel_retry


def cancelled(error=None, *, status="cancelled", result=None):
    return {"job_id": "cancelled-job", "status": status, "result": result, "error": error}


def process(job):
    return subprocess.CompletedProcess([], 1, json.dumps(job), "")


class CancelledOutcomeTests(unittest.TestCase):
    setUp = fixtures.GatewayDeduplicationTest.setUp
    run_task = fixtures.GatewayDeduplicationTest.run_task
    dev = fixtures.GatewayDeduplicationTest.dev
    assert_duplicate = fixtures.GatewayDeduplicationTest.assert_duplicate
    typed_response = staticmethod(fixtures.GatewayDeduplicationTest.typed_response)
    dev_response = staticmethod(fixtures.GatewayDeduplicationTest.dev_response)

    def assert_no_measurement(self, output):
        self.assertNotIn(gateway.SUPERVISOR_MEASUREMENT_PREFIX, output)
        public = json.loads(output)
        self.assertEqual(public["error"]["code"], "gateway_infrastructure")
        self.assertEqual(public["error"]["reason"], "cancelled_without_outcome")
        self.assertFalse(public["repairable"])
        self.assertNotIn("all_pass", public)
        self.assertNotIn("exit_code", public)

    def test_empty_cancellations_are_infrastructure_not_candidate_verdicts(self):
        for status in ("cancelled", "canceled"):
            for error in (None, {}):
                for result in (None, {}):
                    job = cancelled(error, status=status, result=result)
                    with self.subTest(job=job):
                        self.assertTrue(gateway._infrastructure_failure(job))
                        self.assertEqual(
                            gateway._infrastructure_reason(job), "cancelled_without_outcome"
                        )
                        self.assertFalse(gateway._cacheable_gateway_outcome(job))

    def test_typed_cancellations_are_recorded_but_do_not_poison_dedup(self):
        for kind in ("run", "profile", "check", "disassemble"):
            for error in (None, {}):
                evidence = self.root / f"{kind}-{error is None}"
                with (
                    self.subTest(kind=kind, error=error),
                    patch.dict(
                        os.environ,
                        {
                            gateway.SUPERVISOR_EVIDENCE_ROOT_ENV: str(evidence),
                            gateway.REUSE_GATEWAY_RESULTS_ENV: "1",
                        },
                    ),
                ):
                    self.typed.side_effect = lambda *a, **kw: [process(cancelled(error))]
                    code, output = self.run_task("--kind", kind)
                    self.assertNotEqual(code, 0)
                    self.assert_no_measurement(output)
                    self.assertEqual(list((evidence / "gateway-tasks").glob("*.json")), [])
                    record = gateway._visible_gateway_records(self.workspace)[-1]
                    self.assertEqual(record["execution_status"], "cancelled")
                    raw = json.loads((record["record_dir"] / "raw-result.json").read_text())
                    self.assertEqual(raw, cancelled(error))
                    visible = gateway._gateway_record_public_result(record["record_id"], record)
                    self.assertEqual(visible["result"]["error"]["code"], "gateway_infrastructure")
                    self.assertNotIn("all_pass", visible["result"])
                    self.typed.side_effect = self.typed_response
                    self.assertEqual(self.run_task("--kind", kind)[0], 0)
                    self.assertEqual(len(list((evidence / "gateway-tasks").glob("*.json"))), 1)

    def test_inline_dev_and_profile_cancellations_do_not_poison_dedup(self):
        for profile in (False, True):
            for error in (None, {}):
                evidence = self.root / f"inline-{profile}-{error is None}"
                call = (
                    (lambda: self.run_task("--kind", "profile", "--include-raw-profile"))
                    if profile
                    else self.dev
                )
                with (
                    self.subTest(profile=profile, error=error),
                    patch.dict(
                        os.environ,
                        {
                            gateway.SUPERVISOR_EVIDENCE_ROOT_ENV: str(evidence),
                            gateway.REUSE_GATEWAY_RESULTS_ENV: "1",
                        },
                    ),
                ):
                    self.dev_run.return_value = process(cancelled(error))
                    code, output = call()
                    self.assertNotEqual(code, 0)
                    self.assert_no_measurement(output)
                    self.assertEqual(list((evidence / "gateway-tasks").glob("*.json")), [])
                    self.dev_run.return_value = self.dev_response()
                    self.assertEqual(call()[0], 0)
                    self.assertEqual(len(list((evidence / "gateway-tasks").glob("*.json"))), 1)

    def test_direct_http_cancellation_is_not_cached(self):
        with (
            patch.object(gateway, "_find_agate", return_value=None),
            patch.object(
                gateway,
                "_run_direct_gateway",
                return_value=process(cancelled()),
            ) as submit,
        ):
            code, output = self.dev()
            self.assertNotEqual(code, 0)
            self.assert_no_measurement(output)
            self.assertEqual(list((self.evidence / "gateway-tasks").glob("*.json")), [])
            submit.return_value = self.dev_response()
            self.assertEqual(self.dev()[0], 0)
            self.assertEqual(submit.call_count, 2)

    def test_partial_payload_does_not_turn_cancellation_into_success(self):
        partial = {"exit_code": 0, "stdout": "partial output", "all_pass": True}
        self.dev_run.return_value = process(cancelled(result=partial))
        with patch.dict(os.environ, {gateway.REUSE_GATEWAY_RESULTS_ENV: "1"}):
            code, output = self.dev()
        self.assertNotEqual(code, 0)
        self.assertNotIn(gateway.SUPERVISOR_MEASUREMENT_PREFIX, output)
        self.assertFalse(json.loads(output)["ok"])
        self.assertEqual(list((self.evidence / "gateway-tasks").glob("*.json")), [])

    def test_retry_exhaustion_does_not_prevent_a_later_fresh_submission(self):
        self.dev_run.side_effect = REAL_RETRY
        with (
            patch.object(gateway, "_run_agate_once", return_value=process(cancelled({}))) as submit,
            patch.object(
                gateway,
                "DEFAULT_INFRASTRUCTURE_RETRIES",
                1,
            ),
            patch.object(gateway.time, "sleep"),
        ):
            code, output = self.dev()
            self.assertNotEqual(code, 0)
            self.assert_no_measurement(output)
            self.assertEqual(submit.call_count, 2)
            self.assertEqual(list((self.evidence / "gateway-tasks").glob("*.json")), [])
            submit.return_value = self.dev_response()
            self.assertEqual(self.dev()[0], 0)
            self.assertEqual(submit.call_count, 3)

    def poison(self, root, *, hidden_status=False):
        source = (self.workspace / "kernel.py").read_bytes()
        digest = gateway._gateway_task_digest(
            "run", hashlib.sha256(source).hexdigest(), {"seed": "test"}
        )
        with patch.dict(os.environ, {gateway.SUPERVISOR_EVIDENCE_ROOT_ENV: str(root)}):
            record = gateway._record_episode_evaluation(
                self.workspace,
                {"all_pass": True, "latency_us_by_shape": {"0": 1.0}}
                if hidden_status
                else {"status": "cancelled", "error": None},
                gateway_kind="run",
                private_result=cancelled(),
                gateway_task_digest=digest,
            )
            if not hidden_status:
                # Older records had only the status inside the normalized result.
                path = root / "gateway-records" / record["record_id"] / "result.json"
                value = json.loads(path.read_text())
                value.pop("execution_status", None)
                path.write_text(json.dumps(value))
            # Reproduce the completed marker produced by the old implementation.
            marker = gateway._gateway_task_root(self.workspace) / f"{digest}.json"
            marker.write_text(
                json.dumps({"status": "completed", "gateway_record_id": record["record_id"]})
            )
        return digest, record, marker

    def test_task_record_cannot_cache_a_cancelled_job_with_an_apparent_success_payload(self):
        source = (self.workspace / "kernel.py").read_bytes()
        digest = gateway._gateway_task_digest("run", hashlib.sha256(source).hexdigest(), {})
        with gateway._gateway_task(self.workspace, digest, source) as task:
            record = task.record(
                {"all_pass": True, "latency_us_by_shape": {"0": 1.0}},
                gateway_kind="run",
                private_result=cancelled(),
            )
        self.assertEqual(list((self.evidence / "gateway-tasks").glob("*.json")), [])
        with (
            patch.dict(os.environ, {gateway.REUSE_GATEWAY_RESULTS_ENV: "1"}),
            redirect_stdout(io.StringIO()) as output,
        ):
            gateway._emit_supervisor_measurement(self.workspace, record["record_id"], reused=False)
            self.assertEqual(output.getvalue(), "")

    def test_old_current_and_historical_poisoned_markers_are_bypassed(self):
        for historical in (False, True):
            for hidden_status in (False, True):
                case = self.root / f"poison-{historical}-{hidden_status}"
                current = case / "current"
                history = case / "episodes"
                root = history / "e0001" / "supervisor_runtime" if historical else current
                digest, record, marker = self.poison(root, hidden_status=hidden_status)
                original = (
                    root / "gateway-records" / record["record_id"] / "result.json"
                ).read_bytes()
                with (
                    self.subTest(historical=historical, hidden_status=hidden_status),
                    patch.dict(
                        os.environ,
                        {
                            gateway.SUPERVISOR_EVIDENCE_ROOT_ENV: str(current),
                            gateway.SUPERVISOR_HISTORY_ROOT_ENV: str(history),
                        },
                    ),
                ):
                    with gateway._gateway_task(
                        self.workspace, digest, b"class Model: pass\n"
                    ) as task:
                        self.assertIsNotNone(task.owner)
                        self.assertIsNone(task.previous_record_id)
                        task.record(
                            {"all_pass": True, "latency_us_by_shape": {"0": 5.0}},
                            gateway_kind="run",
                        )
                    self.assertEqual(
                        gateway._reserve_gateway_task(self.workspace, digest),
                        (None, task.record_id),
                    )
                self.assertEqual(
                    (root / "gateway-records" / record["record_id"] / "result.json").read_bytes(),
                    original,
                )
                if historical:
                    self.assertEqual(
                        json.loads(marker.read_text())["gateway_record_id"], record["record_id"]
                    )

    def test_direct_completion_reuse_and_receipt_paths_guard_old_poisoned_records(self):
        digest, record, marker = self.poison(self.evidence)
        with (
            patch.dict(os.environ, {gateway.REUSE_GATEWAY_RESULTS_ENV: "1"}),
            redirect_stdout(io.StringIO()) as output,
        ):
            with self.assertRaises(RuntimeStateError):
                gateway._reusable_gateway_record(self.workspace, digest, record["record_id"])
            gateway._emit_supervisor_measurement(self.workspace, record["record_id"], reused=True)
            self.assertEqual(output.getvalue(), "")
            owner, _ = gateway._reserve_gateway_task(self.workspace, digest)
            gateway._complete_gateway_task(self.workspace, digest, owner, record["record_id"])
            self.assertFalse(marker.exists())

    def test_real_candidate_failure_remains_deduplicated_and_reusable(self):
        self.typed.side_effect = lambda *a, **kw: [
            process(
                {
                    "job_id": "candidate-failed",
                    "status": "failed",
                    "error": {"error_class": "candidate", "reason": "compile_failed"},
                }
            )
        ]
        self.assertNotEqual(self.run_task("--kind", "check")[0], 0)
        record = gateway._visible_gateway_records(self.workspace)[-1]
        self.assert_duplicate(lambda: self.run_task("--kind", "check"), record["record_id"])
        with patch.dict(os.environ, {gateway.REUSE_GATEWAY_RESULTS_ENV: "1"}):
            code, output = self.run_task("--kind", "check")
            self.assertNotEqual(code, 0)
            self.assertIn(gateway.SUPERVISOR_MEASUREMENT_PREFIX, output)
        self.assertEqual(self.typed.call_count, 1)
