"""Local infra responses use remote retry/cache policy, including old saved errors."""

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
from supervisor.gateway_errors import LOCAL_INFRASTRUCTURE_REASONS
from tools.local_gateway import _error

REAL_RETRY = gateway._run_agate_with_cancel_retry


def failure(reason, *, legacy=False):
    error = (
        {"error_class": "local_gateway", "reason": reason, "message": "old failure", "details": {}}
        if legacy else _error(reason, "local failure")
    )
    return {"job_id": "failed-local-job", "status": "failed", "result": None, "error": error}


def process(job):
    return subprocess.CompletedProcess([], 1, json.dumps(job), "")


class LocalGatewayOutcomeTests(unittest.TestCase):
    setUp = fixtures.GatewayDeduplicationTest.setUp
    run_task = fixtures.GatewayDeduplicationTest.run_task
    dev = fixtures.GatewayDeduplicationTest.dev
    assert_duplicate = fixtures.GatewayDeduplicationTest.assert_duplicate
    typed_response = staticmethod(fixtures.GatewayDeduplicationTest.typed_response)
    dev_response = staticmethod(fixtures.GatewayDeduplicationTest.dev_response)

    def test_old_infrastructure_errors_are_recognized_without_mutating_evidence(self):
        for reason in LOCAL_INFRASTRUCTURE_REASONS:
            job = failure(reason, legacy=True)
            original = json.dumps(job)
            with self.subTest(reason=reason):
                self.assertTrue(gateway._infrastructure_failure(job))
                self.assertFalse(gateway._cacheable_gateway_outcome(job))
                public = gateway._agent_gateway_failure(job, None, generalized=False)
                self.assertEqual(public["error"]["code"], "gateway_infrastructure")
                self.assertFalse(public["repairable"])
                self.assertEqual(json.dumps(job), original)
        # A coincidentally identical reason in a Candidate error is not local infra.
        job["error"] = {"error_class": "candidate", "reason": "command_timeout"}
        self.assertFalse(gateway._infrastructure_failure(job))

    def test_typed_operations_release_failed_reservations_and_allow_resubmission(self):
        for kind in ("run", "profile", "check", "disassemble"):
            for legacy in (False, True):
                evidence = self.root / f"{kind}-{legacy}"
                job = failure("scheduler_stopped", legacy=legacy)
                with (
                    self.subTest(kind=kind, legacy=legacy),
                    patch.dict(os.environ, {
                        gateway.SUPERVISOR_EVIDENCE_ROOT_ENV: str(evidence),
                        gateway.REUSE_GATEWAY_RESULTS_ENV: "1",
                    }),
                ):
                    self.typed.side_effect = lambda *a, **kw: [process(job)]
                    code, output = self.run_task("--kind", kind)
                    self.assertNotEqual(code, 0)
                    self.assertNotIn(gateway.SUPERVISOR_MEASUREMENT_PREFIX, output)
                    public = json.loads(output)
                    self.assertEqual(public["error"]["code"], "gateway_infrastructure")
                    self.assertFalse(public["repairable"])
                    self.assertEqual(list((evidence / "gateway-tasks").glob("*.json")), [])
                    record = gateway._visible_gateway_records(self.workspace)[-1]
                    raw = json.loads((record["record_dir"] / "raw-result.json").read_text())
                    self.assertEqual(raw, job)
                    self.typed.side_effect = self.typed_response
                    self.assertEqual(self.run_task("--kind", kind)[0], 0)

    def test_dev_retries_fresh_jobs_and_exhaustion_does_not_poison_cache(self):
        self.dev_run.side_effect = REAL_RETRY
        for legacy in (False, True):
            for reason in ("command_timeout", "scheduler_stopped", "scheduler_restarted"):
                evidence = self.root / f"retry-{legacy}-{reason}"
                with (
                    self.subTest(legacy=legacy, reason=reason),
                    patch.dict(os.environ, {gateway.SUPERVISOR_EVIDENCE_ROOT_ENV: str(evidence)}),
                    patch.object(gateway, "_run_agate_once", return_value=process(failure(reason, legacy=legacy))) as submit,
                    patch.object(gateway, "DEFAULT_INFRASTRUCTURE_RETRIES", 1),
                    patch.object(gateway.time, "sleep"),
                ):
                    code, output = self.dev()
                    self.assertNotEqual(code, 0)
                    self.assertEqual(submit.call_count, 2)
                    self.assertFalse(json.loads(output)["repairable"])
                    self.assertEqual(list((evidence / "gateway-tasks").glob("*.json")), [])
                    submit.return_value = self.dev_response()
                    self.assertEqual(self.dev()[0], 0)
                    self.assertEqual(submit.call_count, 3)

    def test_old_current_and_historical_cache_entries_are_not_reused(self):
        source = (self.workspace / "kernel.py").read_bytes()
        for historical in (False, True):
            for reason in LOCAL_INFRASTRUCTURE_REASONS:
                case = self.root / f"poison-{historical}-{reason}"
                current, history = case / "current", case / "episodes"
                evidence = history / "e0001" / "supervisor_runtime" if historical else current
                digest = gateway._gateway_task_digest("run", hashlib.sha256(source).hexdigest(), {})
                job = failure(reason, legacy=True)
                with patch.dict(os.environ, {gateway.SUPERVISOR_EVIDENCE_ROOT_ENV: str(evidence)}):
                    record = gateway._record_episode_evaluation(
                        self.workspace, job, gateway_kind="run", private_result=job,
                        gateway_task_digest=digest,
                    )
                    marker = gateway._gateway_task_root(self.workspace) / f"{digest}.json"
                    marker.write_text(json.dumps({
                        "status": "completed", "gateway_record_id": record["record_id"],
                    }))
                path = evidence / "gateway-records" / record["record_id"] / "result.json"
                original = path.read_bytes()
                with (
                    self.subTest(historical=historical, reason=reason),
                    patch.dict(os.environ, {
                        gateway.SUPERVISOR_EVIDENCE_ROOT_ENV: str(current),
                        gateway.SUPERVISOR_HISTORY_ROOT_ENV: str(history),
                        gateway.REUSE_GATEWAY_RESULTS_ENV: "1",
                    }),
                    redirect_stdout(io.StringIO()) as output,
                ):
                    with self.assertRaises(RuntimeStateError):
                        gateway._reusable_gateway_record(self.workspace, digest, record["record_id"])
                    gateway._emit_supervisor_measurement(self.workspace, record["record_id"], reused=True)
                    self.assertEqual(output.getvalue(), "")
                    loaded = gateway._load_gateway_record(self.workspace, record["record_id"])
                    public = gateway._gateway_record_public_result(record["record_id"], loaded)
                    self.assertEqual(public["result"]["error"]["code"], "gateway_infrastructure")
                    with gateway._gateway_task(self.workspace, digest, source) as task:
                        self.assertIsNotNone(task.owner)
                        self.assertIsNone(task.previous_record_id)
                        task.record({"all_pass": True, "latency_us_by_shape": {"0": 5}}, gateway_kind="run")
                    self.assertEqual(gateway._reserve_gateway_task(self.workspace, digest), (None, task.record_id))
                self.assertEqual(path.read_bytes(), original)

    def test_real_local_command_failure_is_still_deduplicated(self):
        self.dev_run.return_value = process(failure("command_failed"))
        self.assertNotEqual(self.dev()[0], 0)
        record = gateway._visible_gateway_records(self.workspace)[-1]
        self.assert_duplicate(self.dev, record["record_id"])
        self.assertEqual(self.dev_run.call_count, 1)


if __name__ == "__main__":
    unittest.main()
