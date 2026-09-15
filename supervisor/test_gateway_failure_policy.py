"""Private unknown diagnostics and bounded retries for ambiguous whole-job deadlines."""

from __future__ import annotations

import io
import json
import os
import subprocess
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from supervisor import gateway
from supervisor import test_gateway_deduplication as fixtures
from tools.local_gateway import _error

REAL_RETRY = gateway._run_agate_with_cancel_retry


def failed(reason="command_timeout", error_class="unknown"):
    return {
        "job_id": "failed-job", "status": "failed", "result": None,
        "error": {"error_class": error_class, "reason": reason, "message": "PRIVATE_SHAPE=12345"},
    }


class GatewayFailurePolicyTest(unittest.TestCase):
    setUp = fixtures.GatewayDeduplicationTest.setUp
    run_task = fixtures.GatewayDeduplicationTest.run_task
    typed_response = staticmethod(fixtures.GatewayDeduplicationTest.typed_response)
    dev_response = staticmethod(fixtures.GatewayDeduplicationTest.dev_response)

    def test_unknown_messages_and_reasons_stay_private_even_with_infra_metadata(self):
        for error_class in (None, "", "unknown"):
            for status in ("failed", "cancelled", "canceled"):
                for origin in (None, "infrastructure"):
                    with self.subTest(error_class=error_class, status=status, origin=origin):
                        job = failed("PRIVATE_REASON", error_class)
                        job["status"] = status
                        job["error"]["details"] = {"failure_origin": origin, "private": "PRIVATE_DETAILS"}
                        for generalized in (False, True):
                            result = gateway._agent_gateway_failure(job, None, generalized=generalized)
                            self.assertEqual("PRIVATE_" in json.dumps(result), not generalized)
                            self.assertEqual(result["error"]["reason"], "unknown" if generalized else "PRIVATE_REASON")
                            self.assertFalse(result["repairable"])

    def test_immediate_and_historical_unknown_diagnostics_are_masked(self):
        for kind in ("run", "profile", "dev", "check", "disassemble", "same_allocation_abba"):
            with self.subTest(kind=kind):
                job = failed("PRIVATE_REASON")
                record = gateway._record_episode_evaluation(
                    self.workspace, job, gateway_kind=kind, private_result=job,
                    kernel_bytes=(self.workspace / "kernel.py").read_bytes(),
                )
                # Record projection must use the same error policy, including Dev's
                # secondary projection; an ABBA record also has its two Kernel roles.
                value = {
                    "gateway_kind": kind, "result": job, "kernel_id": record["kernel_id"],
                    "kernel_subject_ids": {"incumbent": record["kernel_id"], "candidate": record["kernel_id"]},
                }
                immediate = gateway._agent_gateway_failure(job, record, generalized=True)
                history = gateway._gateway_record_public_result(record["record_id"], value, generalized=True)
                self.assertEqual(history["result"]["error"], immediate["error"])
                self.assertNotIn("PRIVATE_", json.dumps(history))
                private = self.evidence / "gateway-records" / record["record_id"] / "raw-result.json"
                self.assertEqual(json.loads(private.read_text()), job)

    def test_generalized_evaluate_and_record_read_use_safe_unknown_errors(self):
        job = failed("PRIVATE_REASON")
        self.typed.side_effect = lambda *a, **kw: [subprocess.CompletedProcess([], 1, json.dumps(job), "")]
        private = self.root / "private-reference"
        private.mkdir()
        for name in ("reference.py", "input.py", "shapes.json"):
            (private / name).write_bytes((self.workspace / name).read_bytes())
        with (
            patch.object(gateway, "_is_generalized_workspace", return_value=True),
            patch.dict(os.environ, {gateway.PRIVATE_REFERENCE_ENV: str(private)}),
        ):
            code, output = self.run_task("--kind", "run", "--mode", "full")
            self.assertNotEqual(code, 0)
            immediate = json.loads(output)
            code, output = self.run_task("--kind", "record-read", "--record-id", immediate["gateway_record_id"])
            self.assertEqual(code, 0)
            historical = json.loads(output.removeprefix(gateway.RECORD_RESULT_PREFIX))
        self.assertNotIn("PRIVATE_", json.dumps(immediate))
        self.assertNotIn("PRIVATE_", json.dumps(historical))

    def test_current_and_legacy_timeouts_are_not_candidate_verdicts(self):
        for error_class in (None, "unknown", "local_gateway", "infra"):
            with self.subTest(error_class=error_class):
                job = failed(error_class=error_class)
                self.assertTrue(gateway._command_timeout(job))
                self.assertFalse(gateway._infrastructure_failure(job))
                self.assertFalse(gateway._cacheable_gateway_outcome(job))
                public = gateway._agent_gateway_failure(job, None, generalized=True)
                self.assertEqual(public["error"]["code"], "gateway_command_timeout")
                self.assertEqual(public["error"]["error_class"], "unknown")
                self.assertFalse(public["repairable"])
                self.assertNotIn("PRIVATE_", json.dumps(public))
        candidate = failed(error_class="candidate")
        self.assertFalse(gateway._command_timeout(candidate))
        self.assertTrue(gateway._cacheable_gateway_outcome(candidate))
        self.assertEqual(_error("command_timeout", "deadline")["error_class"], "unknown")

    def run_transport(self, transport, outcomes, *, wait_budget=120, expire_on_sleep=False):
        jobs = iter(outcomes)
        submissions = []
        clock = [0.0]

        def submit_cli(**kwargs):
            job = {**next(jobs), "job_id": f"job-{len(submissions)}"}
            submissions.append(job["job_id"])
            return subprocess.CompletedProcess([], 0 if job["status"] == "succeeded" else 1, json.dumps(job), "")

        def request(url, method, path, payload, timeout):
            if method == "POST" and not path.endswith("/cancel"):
                job_id = f"job-{len(submissions)}"
                submissions.append(job_id)
                return {"job_id": job_id}
            if method == "GET":
                return {**next(jobs), "job_id": submissions[-1]}
            return {}

        def sleep(delay):
            clock[0] = float(wait_budget) if expire_on_sleep else clock[0] + delay

        with (
            patch.object(gateway, "_run_agate_once", side_effect=submit_cli),
            patch.object(gateway, "_gateway_json", side_effect=request),
            patch.object(gateway.time, "monotonic", side_effect=lambda: clock[0]),
            patch.object(gateway.time, "sleep", side_effect=sleep),
        ):
            if transport == "cli":
                result = REAL_RETRY(
                    agate=["agate", "dev"], executable="agate", url="", gateway_profile=None,
                    command_timeout=30, wait_budget=wait_budget,
                )
            else:
                try:
                    result = gateway._run_direct_job(
                        url="https://gateway.invalid", kind="dev", payload={},
                        timeout=wait_budget, queue_wait_grace=0,
                    )
                except TimeoutError:
                    if not expire_on_sleep:
                        raise
                    result = None
        return result, submissions

    def test_timeout_retry_cap_applies_to_both_transports_and_legacy_errors(self):
        for transport in ("cli", "http"):
            for error_class in ("unknown", "local_gateway", "infra"):
                with self.subTest(transport=transport, error_class=error_class):
                    result, jobs = self.run_transport(transport, [failed(error_class=error_class)] * 8)
                    self.assertEqual(jobs, ["job-0", "job-1"])
                    self.assertEqual(json.loads(result.stdout)["error"]["reason"], "command_timeout")

    def test_timeout_retry_count_survives_intervening_infra_and_preserves_success(self):
        infra = failed("scheduler_stopped", "infra")
        success = {"status": "succeeded", "result": {"passed": True}}
        for transport in ("cli", "http"):
            with self.subTest(transport=transport):
                result, jobs = self.run_transport(transport, [failed(), infra, failed(), success])
                self.assertEqual(len(jobs), 3)
                self.assertEqual(json.loads(result.stdout)["status"], "failed")
                result, jobs = self.run_transport(transport, [failed(), success])
                self.assertEqual(len(jobs), 2)
                self.assertEqual(json.loads(result.stdout)["status"], "succeeded")
                result, jobs = self.run_transport(transport, [infra] * 8, wait_budget=600)
                self.assertEqual(len(jobs), gateway.DEFAULT_INFRASTRUCTURE_RETRIES + 1)

    def test_deadline_can_stop_timeout_retry_early(self):
        for transport in ("cli", "http"):
            with self.subTest(transport=transport):
                _, jobs = self.run_transport(transport, [failed()] * 3, expire_on_sleep=True)
                self.assertEqual(jobs, ["job-0"])

    def test_timeout_has_no_completed_dedup_or_supervisor_receipt(self):
        self.typed.side_effect = lambda *a, **kw: [subprocess.CompletedProcess([], 1, json.dumps(failed()), "")]
        with patch.dict(os.environ, {gateway.REUSE_GATEWAY_RESULTS_ENV: "1"}):
            code, output = self.run_task("--kind", "check")
            self.assertNotEqual(code, 0)
            self.assertNotIn(gateway.SUPERVISOR_MEASUREMENT_PREFIX, output)
            self.assertEqual(list((self.evidence / "gateway-tasks").glob("*.json")), [])
            record_id = json.loads(output)["gateway_record_id"]
            with redirect_stdout(io.StringIO()) as captured:
                gateway._emit_supervisor_measurement(self.workspace, record_id, reused=True)
            self.assertEqual(captured.getvalue(), "")

    def test_evaluate_stops_remaining_measurement_rounds_after_timeout(self):
        self.typed.side_effect = lambda *a, **kw: [subprocess.CompletedProcess([], 1, json.dumps(failed()), "")]
        code, output = self.run_task("--kind", "run", "--mode", "full")
        self.assertNotEqual(code, 0)
        self.assertEqual(json.loads(output)["error"]["code"], "gateway_command_timeout")
        self.assertEqual(self.typed.call_count, 1)

    def test_generalized_fallback_error_does_not_echo_unknown_diagnostics(self):
        job = failed("PRIVATE_REASON")
        job["error"]["message"] += " source validation failed"
        self.typed.side_effect = lambda *a, **kw: [subprocess.CompletedProcess([], 1, json.dumps(job), "")]
        with (
            patch.object(gateway, "_is_generalized_workspace", return_value=True),
            patch.dict(os.environ, {gateway.PRIVATE_REFERENCE_ENV: str(self.workspace)}),
        ):
            with self.assertRaises(SystemExit) as failure:
                self.run_task("--kind", "run", "--mode", "full")
        self.assertNotIn("PRIVATE_", str(failure.exception.code))


if __name__ == "__main__":
    unittest.main()
