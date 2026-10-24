"""Regression tests for the bounded native-ABBA run-timeout recovery policy."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aka.light.evaluation.native_abba import (  # noqa: E402
    ABBA_WALL_CLOCK_TIMEOUT_REASON,
    compare,
    is_terminal_abba_timeout,
)
from aka.light.experiment.records import _completed_abba_timeout  # noqa: E402
from aka.light.acceptance.recorded_verifier import RecordedABBAValidator  # noqa: E402


TIMEOUT = "candidate exceeded 1200.0s wall-clock timeout"


def _run(index: int, revision: str, label: str, repeat: int, latency_ms: float, *, timeout=False):
    performance = {
        "7": {
            "error": TIMEOUT if timeout else None,
            "samples": [] if timeout else [{"end_to_end_time_ms": latency_ms}],
            "kernel_events": [],
        }
    }
    return {
        "index": index,
        "revision": revision,
        "label": label,
        "repeat": repeat,
        "result": {
            "error": None,
            "passed": {
                "compile": {"7": {"status": "passed", "reason": None}},
                "correctness": {"7": {"status": "passed", "reason": None}},
                "performance": {
                    "7": {
                        "status": "failed" if timeout else "passed",
                        "reason": TIMEOUT if timeout else None,
                    }
                },
            },
            "correctness": {"shapes": {"7": {"cases": []}}},
            "performance": {"shapes": performance},
        },
    }


def _job(job_id: str, *, timed_out_runs=()):
    rows = [
        _run(0, "baseline", "A", 0, 5_600.0, timeout=0 in timed_out_runs),
        _run(1, "candidate", "B", 0, 0.83, timeout=1 in timed_out_runs),
        _run(2, "candidate", "B", 1, 0.84, timeout=2 in timed_out_runs),
        _run(3, "baseline", "A", 1, 5_700.0, timeout=3 in timed_out_runs),
    ]
    return {
        "job_id": job_id,
        "status": "succeeded",
        "error": None,
        "result": {"abba": {"sdk_results": [{"abba": {"runs": rows}}]}},
    }


class _Gateway:
    def __init__(self, jobs):
        self.jobs = list(jobs)
        self.identity_extras = []

    def read_workspace_override(self, workspace, path, **_kwargs):
        return (workspace / path).read_text()

    def _typed_workspace_limitation(self, *_args):
        return None

    def build_typed_request(self, *_args, **_kwargs):
        return {
            "candidate": "def kernel():\n    return None\n",
            "spec": {"target_hardware": ["NVIDIA L20D"]},
            "reference": {"shapes": {"7": {}}},
            "runner_overrides": {},
        }

    @staticmethod
    def shape_id_sort_key(shape_id):
        return (0, int(shape_id))

    @staticmethod
    def find_agate():
        return None

    @staticmethod
    def parse_job_response(stdout):
        return json.loads(stdout)

    def run_direct_job(self, **kwargs):
        self.identity_extras.append(kwargs.get("identity_extra"))
        job = self.jobs.pop(0)
        return subprocess.CompletedProcess([], 0, json.dumps(job), "")


def _args():
    return SimpleNamespace(
        kind="run",
        evaluation_mode="full",
        command=[],
        evaluation_input_path=None,
        evaluation_shapes_path=None,
        shape_id=None,
        ssh="",
        baseline_path="baseline.py",
        multi_seed=0,
        timed_runs=None,
        hardware="NVIDIA L20D",
        timeout=6000,
        env=[],
        requirement=[],
        deps_mode="runtime",
        comparison_repeats=2,
        comparison_run_timeout=1200,
        shape_batch_size=1,
        dry_run=False,
        url="https://gateway.invalid",
        gateway_profile="",
    )


def _result(stdout: str):
    prefix = "[sandbox] ABBA_JSON="
    line = next(line for line in stdout.splitlines() if line.startswith(prefix))
    return json.loads(line[len(prefix) :])


class AbbaWallClockRetryTests(unittest.TestCase):
    def _compare(self, jobs):
        gateway = _Gateway(jobs)
        stdout, stderr = io.StringIO(), io.StringIO()
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            workspace = Path(directory)
            (workspace / "kernel.py").write_text("def kernel():\n    return None\n")
            (workspace / "baseline.py").write_text("def kernel():\n    return None\n")
            with redirect_stdout(stdout), redirect_stderr(stderr):
                code = compare(gateway, _args(), workspace, 60)
        return code, _result(stdout.getvalue()), stderr.getvalue(), gateway

    def test_timeout_is_retried_once_and_success_is_used(self):
        code, result, stderr, gateway = self._compare(
            [_job("ev-first", timed_out_runs=(3,)), _job("ev-retry")]
        )

        self.assertEqual(code, 0)
        self.assertTrue(result["correct"])
        self.assertIsNone(result["error"])
        self.assertEqual(gateway.identity_extras, [None, {"abba_wall_clock_retry": 1}])
        self.assertIn("retrying the complete ABBA batch once", stderr)

    def test_second_timeout_returns_exact_reason_as_completed_failure(self):
        code, result, _stderr, gateway = self._compare(
            [
                _job("ev-first", timed_out_runs=(3,)),
                _job("ev-retry", timed_out_runs=(0, 3)),
            ]
        )

        self.assertEqual(code, 1)
        self.assertFalse(result["correct"])
        self.assertTrue(is_terminal_abba_timeout(result))
        self.assertEqual(result["error"]["reason"], ABBA_WALL_CLOCK_TIMEOUT_REASON)
        self.assertEqual(result["error"]["message"], TIMEOUT)
        self.assertEqual(result["error"]["details"]["attempts"], 2)
        self.assertEqual(
            {(row["attempt"], row["side"], row["repeat"], row["shape_id"])
             for row in result["error"]["details"]["failures"]},
            {(1, "A", 1, "7"), (2, "A", 0, "7"), (2, "A", 1, "7")},
        )
        self.assertEqual(gateway.identity_extras, [None, {"abba_wall_clock_retry": 1}])

    def test_twice_timed_out_result_is_cacheable_only_with_terminal_jobs(self):
        error = {
            "error_class": "timeout",
            "reason": ABBA_WALL_CLOCK_TIMEOUT_REASON,
            "message": TIMEOUT,
            "details": {"attempts": 2, "failures": []},
        }
        sample = {"correct": False, "error": error}

        self.assertTrue(
            _completed_abba_timeout(
                sample,
                [{"phase": "terminal"}, {"phase": "terminal"}],
                truncated=False,
            )
        )
        self.assertFalse(
            _completed_abba_timeout(
                sample,
                [{"phase": "terminal"}, {"phase": "accepted"}],
                truncated=False,
            )
        )
        self.assertFalse(
            _completed_abba_timeout(
                sample,
                [{"phase": "terminal"}, {"phase": "terminal"}],
                truncated=True,
            )
        )

    def test_acceptance_converts_exhausted_timeout_to_failed_promotion(self):
        result = {
            "gateway_record_id": "gateway-timeout",
            "comparison": {"method": "abba", "repeats": 2},
            "correct": False,
            "error": {
                "error_class": "timeout",
                "reason": ABBA_WALL_CLOCK_TIMEOUT_REASON,
                "message": TIMEOUT,
                "details": {"attempts": 2, "failures": []},
            },
        }
        response = {
            "exit_code": 1,
            "stdout": "[sandbox] ABBA_JSON=" + json.dumps(result) + "\n",
            "stderr": "",
        }
        verifier = RecordedABBAValidator(
            execute=lambda _workspace, _argv: response,
            hardware="NVIDIA L20D",
            timeout=6000,
            repeats=2,
            per_run_timeout=1200,
            shape_batch_size=1,
        )

        def install(path, source):
            path.mkdir(parents=True, exist_ok=True)
            (path / "kernel.py").write_bytes(source)

        def publish(path, name, source):
            target = path / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source)

        with tempfile.TemporaryDirectory(dir=ROOT) as directory, patch(
            "aka.light.workspace.sources.from_git",
            side_effect=[b"candidate", b"baseline"],
        ), patch(
            "aka.light.workspace.sources.unpack", return_value=(None, {})
        ), patch(
            "aka.light.workspace.sources.install", side_effect=install
        ), patch(
            "aka.light.acceptance.recorded_verifier.warn_kernel_mismatch"
        ), patch(
            "aka.light.acceptance.recorded_verifier.read_input",
            side_effect=FileNotFoundError,
        ), patch(
            "aka.light.acceptance.recorded_verifier.publish", side_effect=publish
        ):
            verdict = verifier.verify(
                Path(directory),
                base_commit="base",
                candidate_commit="candidate",
                changed_paths=["kernel.py"],
            )

        self.assertEqual(verdict.gate, "FAIL")
        self.assertFalse(verdict.passed)
        self.assertEqual(verdict.gateway_record_id, "gateway-timeout")
        self.assertIn(TIMEOUT, verdict.error)


if __name__ == "__main__":
    unittest.main()
