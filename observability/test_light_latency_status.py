from __future__ import annotations

import json
import contextlib
import io
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

from observability.light_latency_status import (
    _accepted_evidence,
    _search_accepted_candidates,
    accepted_best,
    best_passing_in_store,
    eligible_full_eval,
    finite_latency,
    parse_args,
)
from aka.light.experiment.store import source_identity


def evaluation_record(*, mode="full", all_pass=True, shape_id=False):
    command = ["python3", "test_kernel.py"]
    if shape_id:
        command.extend(["--shape-id", "shape-1"])
    result = {
        "all_pass": all_pass,
        "mode": mode,
        "latency_us_by_shape": {"shape-1": 10.0, "shape-2": 20.0},
        "latency_us_geomean": 14.1421356237,
    }
    return {
        "operation": "evaluate",
        "cacheable": True,
        "request": {
            "operation": "evaluate",
            "options": {
                "command": command,
                "evaluation_mode": mode,
                "evaluation_policy": {"mode": mode},
            },
        },
        "response": {
            "exit_code": 0,
            "stdout": "[test_kernel] RESULT_JSON=" + json.dumps(result) + "\n",
            "stderr": "",
        },
        "kernels": {"candidate": {
            "kernel_id": "kernel-" + "a" * 32,
            "kernel_artifact_digest": "sha256:" + "b" * 64,
        }},
    }


class FakeStore:
    def __init__(self, root: Path, records: dict[str, dict]):
        self.root = root
        self.records = records
        for record_id in records:
            path = root / "records" / record_id / "record.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()

    def read(self, record_id: str) -> dict:
        return self.records[record_id]


class LightLatencyStatusTest(unittest.TestCase):
    def test_finite_latency_rejects_invalid_values(self):
        self.assertEqual(finite_latency(12), 12.0)
        for value in (None, True, 0, -1, float("inf"), float("nan")):
            self.assertIsNone(finite_latency(value))

    def test_search_candidates_include_seed_and_terminal_retained_only(self):
        seed = {"latency_us_geomean": 10.0, "commit": "seed"}
        retained = {"latency_us_geomean": 9.0, "commit": "accepted"}
        ignored = {"latency_us_geomean": 8.0, "commit": "still-running"}
        state = {
            "seed": seed,
            "active": {
                "jobs": [
                    {"status": "completed", "result": {"retained_kernel": retained}},
                    {"status": "running", "result": {"retained_kernel": ignored}},
                    {"status": "completed", "result": {"retained_kernel": None}},
                ]
            },
        }
        self.assertEqual(list(_search_accepted_candidates(state)), [seed, retained])

    def test_accepted_best_includes_completed_active_job(self):
        with TemporaryDirectory(dir=Path.cwd()) as temporary:
            workspace = Path(temporary) / "campaign"
            root = Path(temporary) / "runtime"
            workspace.mkdir()
            search = root / "search"
            search.mkdir(parents=True)
            (search / "state.json").write_text(json.dumps({
                "seed": {"latency_us_geomean": 10.0, "commit": "seed"},
                "active": {"jobs": [{
                    "status": "completed",
                    "result": {"retained_kernel": {
                        "latency_us_geomean": 8.0,
                        "commit": "accepted",
                        "trajectory": 2,
                        "episode": 3,
                    }},
                }]},
            }))
            result = accepted_best(workspace, root)
        self.assertEqual(result["latency_us_geomean"], 8.0)
        self.assertEqual(result["git_commit"], "accepted")

    def test_accepted_evidence_verifies_record_against_git_source(self):
        record_id = "gateway-" + "1" * 32
        with TemporaryDirectory(dir=Path.cwd()) as temporary:
            workspace = Path(temporary) / "trajectory"
            attempt = workspace / ".atrex_long_horizon/episodes/e0003/attempt.json"
            attempt.parent.mkdir(parents=True)
            attempt.write_text(json.dumps({
                "episode": 3,
                "version": 2,
                "accepted": True,
                "verification": {"gate": "PASS", "gateway_record_id": record_id},
            }))
            runtime = Path(temporary) / "runtime"
            measurements = runtime / "measurements"
            measurements.mkdir(parents=True)
            (measurements / "identity.key").write_bytes(b"x" * 32)
            record = {
                "operation": "same_allocation_abba",
                "kernels": {"candidate": source_identity(b"candidate")},
            }
            store = mock.Mock()
            store.read.return_value = record
            with mock.patch("observability.light_latency_status.supervisor_campaign_root", return_value=runtime), \
                    mock.patch("observability.light_latency_status.MeasurementStore", return_value=store), \
                    mock.patch("observability.light_latency_status.from_git", return_value=b"candidate"):
                value = _accepted_evidence(workspace, {
                    "episode": 3, "version": 2, "git_commit": "a" * 40,
                })
        self.assertEqual(value["gateway_record_id"], record_id)
        self.assertEqual(value["verification_evidence"], "verified")
        self.assertEqual(value["verification_operation"], "same_allocation_abba")

    def test_standard_full_eval_is_eligible(self):
        result = eligible_full_eval(evaluation_record())
        self.assertIsNotNone(result)
        self.assertTrue(result["all_pass"])

    def test_failed_correctness_and_subset_are_ineligible(self):
        self.assertIsNone(eligible_full_eval(evaluation_record(all_pass=False)))
        self.assertIsNone(eligible_full_eval(evaluation_record(shape_id=True)))
        self.assertIsNone(eligible_full_eval(evaluation_record(mode="correctness_only")))

    def test_non_evaluate_operation_is_ineligible(self):
        record = evaluation_record()
        record["operation"] = "profile"
        self.assertIsNone(eligible_full_eval(record))

    def test_store_scan_returns_only_eligible_records(self):
        passing_id = "gateway-" + "1" * 32
        rejected_id = "gateway-" + "2" * 32
        with TemporaryDirectory(dir=Path.cwd()) as temporary:
            store = FakeStore(Path(temporary), {
                passing_id: evaluation_record(),
                rejected_id: evaluation_record(all_pass=False),
            })
            rows = best_passing_in_store(store, Path("/campaign"))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["gateway_record_id"], passing_id)
        self.assertEqual(rows[0]["kernel_id"], "kernel-" + "a" * 32)
        self.assertEqual(rows[0]["kernel_artifact_digest"], "sha256:" + "b" * 64)
        self.assertAlmostEqual(rows[0]["latency_us_geomean"], 14.1421356237)

    def test_watch_must_be_positive(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args(["/tmp/campaign", "--watch", "0"])
        options = parse_args(["/tmp/campaign", "--watch", "2.5"])
        json_options = parse_args(["/tmp/campaign", "--json"])
        self.assertTrue(json_options.json)
        self.assertEqual(options.watch, 2.5)


if __name__ == "__main__":
    unittest.main()
