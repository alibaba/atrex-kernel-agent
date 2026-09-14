"""ABBA resume tests use fake remote jobs, never a model or GPU allocation."""

from __future__ import annotations

import io
import json
import math
import os
import re
import subprocess
import unittest
from collections import Counter
from contextlib import redirect_stderr
from pathlib import Path
from threading import Lock
from unittest.mock import patch

from supervisor import gateway, test_measurement_reuse as reuse_fixture
from supervisor.abba_checkpoints import AbbaBatchStore, validate_batch


class AbbaCheckpointTest(unittest.TestCase):
    git = reuse_fixture.MeasurementReuseTest.git
    measure = reuse_fixture.MeasurementReuseTest.measure
    verify = reuse_fixture.MeasurementReuseTest.verify

    def bridge(self, *args: object, **kwargs: object) -> subprocess.CompletedProcess:
        try:
            return reuse_fixture.MeasurementReuseTest.bridge(self, *args, **kwargs)
        except SystemExit as exc:
            # Production observes a child exit, not an in-process Python exception.
            return subprocess.CompletedProcess([], 1, "", str(exc.code))

    def setUp(self) -> None:
        reuse_fixture.MeasurementReuseTest.setUp(self)
        self.lock = Lock()
        self.physical_calls: Counter[tuple[int, str]] = Counter()
        self.fail_once: tuple[int, str] | None = None
        self.bad_once: tuple[int, str] | None = None
        self.incorrect: tuple[int, str] | None = None
        self.interrupt_once: tuple[int, str] | None = None
        self.args.requirement = ["custom-kernel==1"]
        self.args.deps_mode = "no_deps"
        (self.workspace / "shapes.json").write_text(json.dumps({"0": {}, "1": {}}))
        (self.workspace / "metadata.json").write_text(
            json.dumps(
                {
                    "shapes": {
                        "0": {"production_performance": {"performance_us": 40}},
                        "1": {"production_performance": {"performance_us": 400}},
                    }
                }
            )
        )

    def remote(self, command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        if len(command) < 2 or not str(command[1]).endswith("supervisor/gateway.py"):
            return self.real_run(command, **kwargs)
        self.assertEqual(command[command.index("--requirement") + 1], "custom-kernel==1")
        self.assertEqual(command[command.index("--deps-mode") + 1], "no_deps")
        request_path = Path(kwargs["cwd"]) / command[-2]
        request = json.loads(request_path.read_text())
        repetition = int(re.search(r"measurement-(\d+)", request_path.name)[1])
        invocation = request["command"]
        shape = invocation[invocation.index("--shape-id") + 1]
        key = repetition, shape
        with self.lock:
            self.calls += 1
            self.physical_calls[key] += 1
            call = self.physical_calls[key]
        if self.fail_once == key and call == 1:
            return subprocess.CompletedProcess(
                command, 1, "", "infra: Ray runtime environment failed"
            )
        if self.interrupt_once == key and call == 1:
            raise KeyboardInterrupt()
        runs = []
        for step in request["schedule"]:
            baseline = step["revision"] == "incumbent"
            values = (
                ((10, 12, 11) if shape == "0" else (100, 110, 1000))
                if baseline
                else ((8, 100, 7) if shape == "0" else (80, 700, 70))
            )
            latency = values[repetition - 1]
            passed = baseline or self.incorrect != key
            runs.append(
                {
                    **step,
                    "exit_code": 0 if passed else 1,
                    "result": {
                        "all_pass": passed,
                        "latency_us_geomean": latency if passed else None,
                        "latency_us_by_shape": {shape: latency} if passed else {},
                        "performance_score": (40 if shape == "0" else 400) / latency
                        if passed
                        else None,
                    },
                }
            )
        payload = {"schema_version": 1, "runs": runs, "error": None}
        if self.bad_once == key and call == 1:
            runs[0]["result"]["latency_us_by_shape"] = {"unknown": 1.0}
        return subprocess.CompletedProcess(
            command,
            0,
            gateway.ABBA_RESULT_PREFIX + json.dumps(payload),
            f"submitted job_id=dv_rep{repetition}_shape{shape}_try{call}",
        )

    def test_partial_failure_resumes_only_missing_batches_and_uses_shape_medians(self) -> None:
        self.fail_once = (2, "0")
        with self.assertRaisesRegex(SystemExit, "abba_comparison_unavailable"):
            self.measure()
        completed = list((self.root / "private/abba-batches").glob("*.json"))
        self.assertEqual(len(completed), 3)
        frozen = {path: path.read_bytes() for path in completed}
        self.assertEqual(self.calls, 4)
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            result = self.verify()
        self.assertTrue(result.passed, result.error)
        self.assertIn("reusing completed measurement", stderr.getvalue())
        self.assertEqual(self.calls, 7)
        self.assertEqual(self.physical_calls[(2, "0")], 2)
        self.assertTrue(
            all(count == 1 for key, count in self.physical_calls.items() if key != (2, "0"))
        )
        self.assertAlmostEqual(result.candidate_latency_us, math.sqrt(8 * 80))
        self.assertAlmostEqual(result.incumbent_latency_us, math.sqrt(11 * 110))
        for path, content in frozen.items():
            self.assertEqual(path.read_bytes(), content)
        raw = json.loads(Path(result.artifact).with_name("raw-result.json").read_text())
        self.assertEqual(len(raw["physical_batches"]), 6)
        self.assertEqual(
            {item["identity"]["measurement_repetition"] for item in raw["physical_batches"]},
            {"measurement-1", "measurement-2", "measurement-3"},
        )
        self.assertTrue(all("job_id=dv_" in item["stderr"] for item in raw["physical_batches"]))
        self.assertTrue(self.verify().reused)
        self.assertEqual(self.calls, 7)
        with self.assertRaisesRegex(SystemExit, "duplicate_gateway_task"):
            self.measure()

    def test_recovery_reuses_batches_from_authorized_history_after_workspace_change(self) -> None:
        self.fail_once = (2, "0")
        history = self.root / "episodes"
        old = history / "e0001/supervisor_runtime"
        with patch.dict(os.environ, {gateway.SUPERVISOR_EVIDENCE_ROOT_ENV: str(old)}):
            with self.assertRaises(SystemExit):
                self.measure()
        frozen = {path: path.read_bytes() for path in (old / "abba-batches").glob("*.json")}
        with patch.dict(os.environ, {gateway.SUPERVISOR_HISTORY_ROOT_ENV: str(history)}):
            self.assertEqual(self.measure()[0], 0)
        self.assertEqual(self.calls, 7)
        self.assertEqual(len(list((self.root / "private/abba-batches").glob("*.json"))), 3)
        for path, content in frozen.items():
            self.assertEqual(path.read_bytes(), content)

    def test_unrelated_history_is_not_reused(self) -> None:
        old = self.root / "other-campaign/e0001/supervisor_runtime"
        with patch.dict(os.environ, {gateway.SUPERVISOR_EVIDENCE_ROOT_ENV: str(old)}):
            self.measure()
        self.measure()
        self.assertEqual(self.calls, 12)

    def test_final_record_failure_does_not_discard_six_completed_batches(self) -> None:
        with patch.object(
            gateway, "_record_episode_evaluation", side_effect=OSError("disk failure")
        ):
            with self.assertRaisesRegex(SystemExit, "gateway_record_unavailable"):
                self.measure()
        self.assertEqual(self.calls, 6)
        self.assertEqual(self.measure()[0], 0)
        self.assertEqual(self.calls, 6)

    def test_correctness_failure_in_one_round_is_preserved_and_never_retried(self) -> None:
        self.incorrect = (2, "0")
        result = self.verify()
        self.assertFalse(result.passed)
        self.assertEqual(self.calls, 6)
        self.assertTrue(self.verify().reused)
        self.assertEqual(self.calls, 6)
        raw = json.loads(Path(result.artifact).with_name("raw-result.json").read_text())
        negative = [
            item
            for item in raw["physical_batches"]
            if any(row["result"]["all_pass"] is False for row in item["payload"]["runs"])
        ]
        self.assertEqual(len(negative), 1)

    def test_malformed_shape_coverage_is_not_persisted_or_scored_as_rejection(self) -> None:
        self.bad_once = (2, "0")
        result = self.verify()
        self.assertEqual(result.gate, "ERROR")
        self.assertEqual(len(list((self.root / "private/abba-batches").glob("*.json"))), 3)
        self.assertTrue(self.verify().passed)
        self.assertEqual(self.calls, 7)

    def test_interruption_releases_outer_reservation_and_preserves_completed_batches(self) -> None:
        self.interrupt_once = (2, "0")
        with self.assertRaises(KeyboardInterrupt):
            self.measure()
        self.assertEqual(self.measure()[0], 0)
        self.assertEqual(self.calls, 7)

    def test_changed_pair_contract_and_schedule_start_new_measurements(self) -> None:
        self.measure()
        (self.workspace / "input.py").write_text("def _make_inputs(): return {'x': 1}\n")
        self.measure(reuse=True)
        (self.workspace / "kernel.py").write_text("class Model: value = 2\n")
        self.measure(reuse=True)
        self.args.comparison_repeats = 3
        self.args.timeout = 1200
        self.measure(reuse=True)
        self.assertEqual(self.calls, 24)

    def test_corrupt_checkpoint_fails_closed_without_overwrite(self) -> None:
        self.fail_once = (2, "0")
        with self.assertRaises(SystemExit):
            self.measure()
        # Corrupt the already completed first round, so no new job can run before rejection.
        paths = list((self.root / "private/abba-batches").glob("*.json"))
        path = next(
            path
            for path in paths
            if json.loads(path.read_text())["record"]["identity"]["measurement_repetition"]
            == "measurement-1"
        )
        value = json.loads(path.read_text())
        value["record"]["payload"]["runs"][0]["result"]["latency_us_geomean"] = 0.01
        path.write_text(json.dumps(value))
        frozen = path.read_bytes()
        result = self.verify()
        self.assertEqual(result.gate, "ERROR")
        self.assertIn("checkpoint is invalid", result.error)
        self.assertEqual(path.read_bytes(), frozen)
        self.assertEqual(self.calls, 4)

    def test_checkpoint_public_result_does_not_expose_batch_internals(self) -> None:
        _, stdout = self.measure()
        public = json.loads(
            next(
                line.removeprefix(gateway.ABBA_RESULT_PUBLIC_PREFIX)
                for line in stdout.splitlines()
                if line.startswith(gateway.ABBA_RESULT_PUBLIC_PREFIX)
            )
        )
        for field in (
            "physical_batches",
            "measurements",
            "measurement_aggregation",
            "checkpoint_contract",
            "completed_at",
        ):
            self.assertNotIn(field, public)

    def test_store_is_write_once_and_validation_rejects_incomplete_rows(self) -> None:
        self.measure()
        path = next((self.root / "private/abba-batches").glob("*.json"))
        value = json.loads(path.read_text())["record"]
        identity = value["identity"]
        frozen = path.read_bytes()
        store = AbbaBatchStore(self.root / "private", [])
        self.assertEqual(
            store.save(identity, value["payload"], stdout="different", stderr="different"), value
        )
        self.assertEqual(path.read_bytes(), frozen)
        for modification in ("missing", "schedule", "timeout", "extra_shape", "nan", "zero"):
            payload = json.loads(json.dumps(value["payload"]))
            first = payload["runs"][0]
            if modification == "missing":
                first["result"] = None
            elif modification == "schedule":
                first["repeat"] = 999
            elif modification == "timeout":
                first["exit_code"] = -1
            elif modification == "extra_shape":
                first["result"]["latency_us_by_shape"]["unknown"] = 10
            else:
                first["result"]["latency_us_geomean"] = float("nan") if modification == "nan" else 0
            with self.subTest(modification=modification), self.assertRaises(ValueError):
                validate_batch(payload, identity["schedule"], identity["shape_ids"])


if __name__ == "__main__":
    unittest.main()
