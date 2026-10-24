"""Regression tests for Agent-visible Agate job-deadline failures."""

import json
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aka.light.evaluation.native_abba import _job_deadline_process  # noqa: E402
from aka.light.evaluation.projection import gateway_error_from_process  # noqa: E402


class JobDeadlineProjectionTests(unittest.TestCase):
    def test_nested_abba_deadline_preserves_original_agate_error(self):
        error = {
            "reason": "timeout",
            "message": "job deadline exceeded",
            "trace_id": "req-deadline",
            "error_class": "timeout",
        }
        job = {
            "job_id": "ev-deadline",
            "status": "succeeded",
            "error": None,
            "result": {
                "abba": {
                    "sdk_results": [{
                        "abba": {
                            "runs": [{"result": {"error": error}}],
                        },
                    }],
                },
            },
        }
        process = subprocess.CompletedProcess([], 0, json.dumps(job), "polling")

        failed = _job_deadline_process(job, process)
        self.assertIsNotNone(failed)
        projected = gateway_error_from_process(failed, "same_allocation_abba")

        self.assertEqual(projected["status"], "failed")
        self.assertEqual(projected["job_id"], "ev-deadline")
        self.assertEqual(projected["error"], error)

    def test_unrelated_incomplete_evidence_keeps_generic_recovery_path(self):
        job = {
            "job_id": "ev-incomplete",
            "status": "succeeded",
            "result": {"abba": {"sdk_results": []}},
        }
        process = subprocess.CompletedProcess([], 0, json.dumps(job), "")

        self.assertIsNone(_job_deadline_process(job, process))

    def test_plain_string_deadline_is_also_preserved(self):
        job = {
            "job_id": "ev-string-deadline",
            "status": "succeeded",
            "result": {"abba": {"sdk_results": [{"error": "job deadline exceeded"}]}},
        }
        process = subprocess.CompletedProcess([], 0, json.dumps(job), "")

        failed = _job_deadline_process(job, process)
        projected = gateway_error_from_process(failed, "same_allocation_abba")

        self.assertEqual(projected["error"], {
            "reason": "timeout",
            "message": "job deadline exceeded",
            "error_class": "timeout",
        })


if __name__ == "__main__":
    unittest.main()
