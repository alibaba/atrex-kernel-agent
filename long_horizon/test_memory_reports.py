"""Canonical reports contain results, not Supervisor Git bookkeeping."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from orchestrator.constants import REPO_ROOT
from orchestrator.session_io import _record_local_test_result
from tools.memory_manager import SCHEMA_TEMPLATE

from .campaign import LongHorizonCampaign
from .models import VerificationResult, VerificationRun


class MemoryReportTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.workspace = Path(temporary.name)
        self.runner = LongHorizonCampaign(SimpleNamespace(
            workspace=self.workspace, private_reference_dir=None,
            platform="B200", arch="sm_100",
        ))
        self.experiment = {
            "experiment_id": "experiment_" + "1" * 32,
            "name": "layout change", "analysis": "lower latency",
            "gateway_record_ids": ["gateway-123-0123456789ab"],
        }
        self.journal = {
            "runtime_managed": True,
            "experiments": [self.experiment],
            "candidate_commit": "a" * 40,
            "base_commit": "b" * 40,
            "outcome": {"summary": "layout change", "next_directions": ["prefetch"]},
        }
        self.result = {
            "all_pass": True, "latency_us_geomean": 8.0,
            "latency_us_arith_mean": 8.0, "latency_us_by_shape": {"0": 8.0},
        }

    def assert_no_commit_fields(self, record: object) -> None:
        if isinstance(record, dict):
            for key, value in record.items():
                self.assertNotIn(key, {"git_commit_hash", "candidate_commit", "base_commit"})
                self.assert_no_commit_fields(value)
        elif isinstance(record, list):
            for value in record:
                self.assert_no_commit_fields(value)

    def test_promoted_report_retains_facts_and_journal_without_commit_fields(self) -> None:
        verification = VerificationResult(
            gate="PASS", candidate_latency_us=8.0, incumbent_latency_us=10.0,
            improvement_pct=20.0,
            runs=[VerificationRun("candidate", 1, 0, self.result)],
        )
        report = self.runner._memory_record(
            version=2, journal=self.journal, verification=verification,
            episode_workspace=self.workspace,
        )
        self.assert_no_commit_fields(report)
        self.assertEqual(report["performance"]["latency_us_by_shape"], {"0": 8.0})
        self.assertEqual(report["quality_gate"]["result"], "PASS")
        self.assertEqual(report["experience"]["experiments"], [
            {**self.experiment, "index": 1},
        ])
        self.assertEqual(self.journal["candidate_commit"], "a" * 40)

    def test_nonpromoted_and_recovery_reports_omit_commits(self) -> None:
        for status in ("candidate_ready", "pivot", "blocked", "interrupted"):
            with self.subTest(status=status):
                report = self.runner._outcome_memory_record(
                    version=2, status=status, violation="not accepted", journal=self.journal,
                )
                self.assert_no_commit_fields(report)
                self.assertEqual(report["long_horizon"]["status"], status)
                self.assertEqual(report["quality_gate"]["result"], "FAIL")
                self.assertEqual(report["experience"]["recorded_experiment_count"], 1)

    def test_baseline_measurement_rewrite_drops_obsolete_commit(self) -> None:
        memory = self.workspace / "memory/v0.json"
        memory.parent.mkdir()
        memory.write_text(json.dumps({"git_commit_hash": "a" * 40}))
        output = _record_local_test_result(self.workspace, "v0", self.result)
        report = json.loads(output.read_text())
        self.assert_no_commit_fields(report)
        self.assertEqual(report["correctness"]["status"], "PASS")
        self.assertEqual(report["performance"]["latency_us_by_shape"], {"0": 8.0})

    def test_report_templates_have_no_commit_fields(self) -> None:
        self.assert_no_commit_fields(SCHEMA_TEMPLATE)
        self.assert_no_commit_fields(json.loads(
            (REPO_ROOT / "reference/v_iteration.schema.json").read_text()
        ))


if __name__ == "__main__":
    unittest.main()
