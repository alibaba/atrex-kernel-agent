from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from long_horizon.verifier import ABBA_RESULT_PREFIX, verification_schedule
from teacher_distill.abba import (
    TeacherABBAValidator,
    score_teacher_abba_payload,
)
from teacher_distill.benchmark import MaterializedTeacherWorkspace
from teacher_distill.models import AbbaStatus


def _result(geomean: float, by_shape: dict[str, float], *, all_pass: bool = True) -> dict:
    return {
        "all_pass": all_pass,
        "latency_us_geomean": geomean,
        "latency_us_by_shape": by_shape,
        "failures": [] if all_pass else ["correctness"],
    }


def _payload(
    teacher: tuple[dict, dict],
    candidate: tuple[dict, dict],
) -> dict:
    schedule = verification_schedule(2)
    values = {
        ("incumbent", 0): teacher[0],
        ("incumbent", 1): teacher[1],
        ("candidate", 0): candidate[0],
        ("candidate", 1): candidate[1],
    }
    return {
        "schema_version": 1,
        "error": None,
        "runs": [
            {
                "revision": row["revision"],
                "repeat": row["repeat"],
                "exit_code": 0,
                "result": values[(row["revision"], row["repeat"])],
                "stdout_tail": "",
                "stderr_tail": "",
            }
            for row in schedule
        ],
    }


class TeacherABBAScoringTest(unittest.TestCase):
    def test_passes_within_geomean_and_per_shape_ratios(self) -> None:
        payload = _payload(
            teacher=(
                _result(100.0, {"a": 80.0, "b": 125.0}),
                _result(102.0, {"a": 82.0, "b": 126.0}),
            ),
            candidate=(
                _result(103.0, {"a": 85.0, "b": 127.0}),
                _result(104.0, {"a": 86.0, "b": 128.0}),
            ),
        )

        result = score_teacher_abba_payload(
            payload,
            schedule=verification_schedule(2),
            repeats=2,
            expected_shape_keys=("a", "b"),
            geomean_ratio=1.05,
            shape_ratio=1.10,
        )

        self.assertEqual(result.status, AbbaStatus.PASS)
        self.assertLessEqual(result.candidate_to_teacher_ratio, 1.05)
        self.assertLessEqual(result.worst_shape_ratio, 1.10)
        self.assertEqual(result.worst_shape_key, "a")

    def test_fails_when_one_shape_exceeds_the_guardrail(self) -> None:
        payload = _payload(
            teacher=(
                _result(100.0, {"a": 80.0, "b": 125.0}),
                _result(100.0, {"a": 80.0, "b": 125.0}),
            ),
            candidate=(
                _result(103.0, {"a": 90.0, "b": 124.0}),
                _result(103.0, {"a": 90.0, "b": 124.0}),
            ),
        )

        result = score_teacher_abba_payload(
            payload,
            schedule=verification_schedule(2),
            repeats=2,
            expected_shape_keys=("a", "b"),
            geomean_ratio=1.05,
            shape_ratio=1.10,
        )

        self.assertEqual(result.status, AbbaStatus.FAIL)
        self.assertEqual(result.worst_shape_key, "a")
        self.assertGreater(result.worst_shape_ratio, 1.10)
        self.assertIn("shape ratio", result.error)

    def test_timeout_or_missing_result_is_infrastructure_error(self) -> None:
        payload = _payload(
            teacher=(
                _result(100.0, {"a": 100.0}),
                _result(100.0, {"a": 100.0}),
            ),
            candidate=(
                _result(100.0, {"a": 100.0}),
                _result(100.0, {"a": 100.0}),
            ),
        )
        payload["runs"][1]["exit_code"] = -1
        payload["runs"][1]["result"] = None
        result = score_teacher_abba_payload(
            payload,
            schedule=verification_schedule(2),
            repeats=2,
            expected_shape_keys=("a",),
            geomean_ratio=1.05,
            shape_ratio=1.10,
        )

        self.assertEqual(result.status, AbbaStatus.INFRA_ERROR)
        self.assertIn("execute", result.error)

    def test_correctness_failure_is_fail_but_malformed_schedule_is_infra_error(self) -> None:
        correctness = _payload(
            teacher=(
                _result(100.0, {"a": 100.0}),
                _result(100.0, {"a": 100.0}),
            ),
            candidate=(
                _result(100.0, {"a": 100.0}, all_pass=False),
                _result(100.0, {"a": 100.0}),
            ),
        )
        failed = score_teacher_abba_payload(
            correctness,
            schedule=verification_schedule(2),
            repeats=2,
            expected_shape_keys=("a",),
            geomean_ratio=1.05,
            shape_ratio=1.10,
        )
        self.assertEqual(failed.status, AbbaStatus.FAIL)

        malformed = dict(correctness)
        malformed["runs"] = list(reversed(correctness["runs"]))
        error = score_teacher_abba_payload(
            malformed,
            schedule=verification_schedule(2),
            repeats=2,
            expected_shape_keys=("a",),
            geomean_ratio=1.05,
            shape_ratio=1.10,
        )
        self.assertEqual(error.status, AbbaStatus.INFRA_ERROR)
        self.assertIn("schedule", error.error)


class TeacherABBAValidatorTest(unittest.TestCase):
    def _git(self, workspace: Path, *args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=workspace, check=True, capture_output=True, text=True
        ).stdout.strip()

    def test_validator_keeps_teacher_snapshots_private_and_uses_exact_abba(self) -> None:
        with tempfile.TemporaryDirectory(prefix="teacher-abba-") as temp_dir:
            root = Path(temp_dir)
            private = root / "private-teacher"
            private.mkdir()
            (private / "kernel.py").write_text("# teacher kernel\n", encoding="utf-8")
            (private / "solution.json").write_text(
                json.dumps({"sources": [{"path": "kernel.py"}]}), encoding="utf-8"
            )
            (private / "test_kernel.py").write_text("# harness\n", encoding="utf-8")
            materialized = MaterializedTeacherWorkspace(
                workspace=private,
                kind="sol",
                expected_shape_keys=("a", "b"),
                workload_hash="1" * 64,
                evaluator_hash="2" * 64,
                measurement_config_hash="3" * 64,
            )

            candidate = root / "candidate"
            candidate.mkdir()
            (candidate / "kernel.py").write_text("# candidate kernel\n", encoding="utf-8")
            (candidate / "solution.json").write_text(
                json.dumps({"sources": [{"path": "kernel.py"}]}), encoding="utf-8"
            )
            self._git(candidate, "init", "-q")
            self._git(candidate, "config", "user.email", "test@local")
            self._git(candidate, "config", "user.name", "test")
            self._git(candidate, "add", "-A")
            self._git(candidate, "commit", "-q", "-m", "candidate")
            commit = self._git(candidate, "rev-parse", "HEAD")
            calls: list[tuple[Path, list[str]]] = []

            payload = _payload(
                teacher=(
                    _result(100.0, {"a": 80.0, "b": 125.0}),
                    _result(100.0, {"a": 80.0, "b": 125.0}),
                ),
                candidate=(
                    _result(103.0, {"a": 84.0, "b": 128.0}),
                    _result(103.0, {"a": 84.0, "b": 128.0}),
                ),
            )

            def sandbox(workspace: Path, command: list[str]):
                calls.append((workspace, command))
                request_path = workspace / command[-2]
                request = json.loads(request_path.read_text(encoding="utf-8"))
                self.assertEqual(request["schedule"], verification_schedule(2))
                self.assertEqual(request["command"][-2:], ["--multi-seed", "5"])
                self.assertEqual(set(request["manifests"]), {"incumbent", "candidate"})
                for manifest in request["manifests"].values():
                    for source in manifest.values():
                        if source is not None:
                            self.assertTrue((request_path.parent / source).is_file())
                return subprocess.CompletedProcess(
                    args=["sandbox"],
                    returncode=0,
                    stdout=ABBA_RESULT_PREFIX + json.dumps(payload) + "\n",
                    stderr="",
                )

            validator = TeacherABBAValidator(
                sandbox_runner=sandbox,
                repeats=2,
                geomean_ratio=1.05,
                shape_ratio=1.10,
            )
            result = validator.verify(
                candidate_workspace=candidate,
                candidate_commit=commit,
                teacher=materialized,
            )

            self.assertEqual(result.status, AbbaStatus.PASS)
            self.assertEqual(calls[0][0], private)
            self.assertFalse(any("teacher kernel" in path.read_text(errors="ignore") for path in candidate.rglob("*" ) if path.is_file() and ".git" not in path.parts))
            self.assertTrue(Path(result.artifact).is_file())
            self.assertTrue(str(result.artifact).startswith(str(private)))


if __name__ == "__main__":
    unittest.main()
