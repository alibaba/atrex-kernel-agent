from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from long_horizon import journal


QUERY_ID = "wiki-query-0123456789abcdef0123456789abcdef"
WIKI_ID = "gpu_wiki::nvidia.blackwell.example.strategy"


class WikiAttributionJournalTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / "journal.json"
        journal.initialize(
            self.path,
            episode=1,
            base_commit="base",
            branch="episode-1",
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def append(self, **overrides: object) -> dict[str, object]:
        experiment: dict[str, object] = {
            "name": "trial",
            "wiki_usage_status": "declared",
            "wiki_query_ids": [QUERY_ID],
            "wiki_usage": [{
                "query_id": QUERY_ID,
                "wiki_id": WIKI_ID,
                "disposition": "applied",
                "use": "selected the launch shape",
                "evidence": "candidate passed",
            }],
            "evaluation": {
                "correctness": "pass",
                "performance": "improved",
                "latency_us": 12.5,
                "kernel_hash": "abc123",
            },
        }
        experiment.update(overrides)
        value = journal.append_experiment(self.path, experiment)
        return value["experiments"][-1]

    def test_declared_usage_preserves_emitted_ids_and_outcome(self) -> None:
        entry = self.append()
        self.assertEqual(entry["wiki_query_ids"], [QUERY_ID])
        self.assertEqual(entry["wiki_usage"][0]["query_id"], QUERY_ID)
        self.assertEqual(entry["wiki_usage"][0]["wiki_id"], WIKI_ID)
        self.assertEqual(entry["evaluation"]["latency_us"], 12.5)
        self.assertNotIn("wiki_usage_errors", entry)
        self.assertNotIn("evaluation_errors", entry)

    def test_malformed_telemetry_is_dropped_without_blocking_append(self) -> None:
        entry = self.append(
            wiki_query_ids=["invented-query"],
            wiki_usage=[{
                "query_id": "invented-query",
                "wiki_id": "invented-record",
                "disposition": "applied",
            }],
            evaluation={
                "correctness": "pass",
                "performance": "improved",
                "latency_us": float("nan"),
                "kernel_hash": "abc123",
            },
        )
        self.assertEqual(entry["wiki_usage"], [])
        self.assertIsNone(entry["evaluation"])
        self.assertTrue(any("emitted Wiki query_id" in error
                            for error in entry["wiki_usage_errors"]))
        self.assertIn(
            "evaluation.latency_us must be a non-negative number or null",
            entry["evaluation_errors"],
        )
        json.dumps(journal.load(self.path), allow_nan=False)

    def test_explicit_no_use_and_not_queried_flows_are_preserved(self) -> None:
        no_use = self.append(
            wiki_usage_status="no_material_use",
            wiki_query_ids=[QUERY_ID],
            wiki_usage=[],
        )
        self.assertEqual(no_use["wiki_usage_status"], "no_material_use")
        self.assertNotIn("wiki_usage_errors", no_use)

        not_queried = self.append(
            wiki_usage_status="not_queried",
            wiki_query_ids=None,
            wiki_usage=[],
        )
        self.assertEqual(not_queried["wiki_usage_status"], "not_queried")
        self.assertNotIn("wiki_usage_errors", not_queried)

    def test_nonfinite_latency_is_rejected_but_large_integer_is_valid(self) -> None:
        for value in (float("inf"), float("-inf")):
            with self.subTest(value=value):
                normalized, errors = journal.normalize_experiment_evaluation({
                    "correctness": "pass",
                    "performance": "improved",
                    "latency_us": value,
                    "kernel_hash": None,
                })
                self.assertIsNone(normalized)
                self.assertTrue(errors)

        normalized, errors = journal.normalize_experiment_evaluation({
            "correctness": "pass",
            "performance": "improved",
            "latency_us": 10**1000,
            "kernel_hash": None,
        })
        self.assertEqual(errors, [])
        self.assertEqual(normalized["latency_us"], 10**1000)


if __name__ == "__main__":
    unittest.main()
