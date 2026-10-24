"""Regression tests for Light's Agate bench-version routing."""

from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aka.light.evaluation.gateway import (  # noqa: E402
    _runtime_pins,
    build_typed_agate_command,
)


def _args():
    return SimpleNamespace(
        url="",
        gateway_profile="prod",
        hardware="NVIDIA L20D",
        timeout=600,
        profile_level="sol",
        profiler=None,
        profile_counter=[],
        kernel_regex=None,
        kernel_name=None,
        profile_source=False,
        launch_skip=None,
        launch_count=None,
        top_kernels=None,
    )


def _request():
    return {
        "candidate": "def kernel():\n    pass\n",
        "spec": {"target_hardware": ["NVIDIA L20D"], "num_gpus": 1},
        "reference": {"operator": "example"},
        "options": {
            "num_correctness_cases": 1,
            "bench_iters": 10,
            "atol": 0.01,
            "rtol": 0.05,
        },
        "env_vars": {},
        "atrex_bench_version": "0.1.1.dev20261020020502",
    }


class ProfileBenchVersionTests(unittest.TestCase):
    def test_profile_omits_eval_only_bench_version(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "kernel.py").write_text(_request()["candidate"])
            command = build_typed_agate_command(
                "agate",
                _args(),
                workspace,
                "profile",
                _request(),
                60,
                reference_dir=workspace,
                request_sidecar_dir=workspace / "sidecar",
            )

        self.assertEqual(command[:2], ["agate", "profile"])
        self.assertNotIn("--atrex-bench-version", command)

    def test_eval_keeps_bench_version(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "kernel.py").write_text(_request()["candidate"])
            command = build_typed_agate_command(
                "agate",
                _args(),
                workspace,
                "run",
                _request(),
                60,
                reference_dir=workspace,
                request_sidecar_dir=workspace / "sidecar",
            )

        index = command.index("--atrex-bench-version")
        self.assertEqual(command[index + 1], "0.1.1.dev20261020020502")

    def test_non_eval_request_does_not_receive_bench_version(self):
        from unittest.mock import patch

        with patch.dict(
            "os.environ",
            {"ATREX_SANDBOX_ATREX_BENCH_VERSION": "0.1.1.dev20261020020502"},
            clear=False,
        ):
            self.assertNotIn(
                "atrex_bench_version",
                _runtime_pins({}, include_bench_version=False),
            )
            self.assertEqual(
                _runtime_pins({})["atrex_bench_version"],
                "0.1.1.dev20261020020502",
            )


if __name__ == "__main__":
    unittest.main()
