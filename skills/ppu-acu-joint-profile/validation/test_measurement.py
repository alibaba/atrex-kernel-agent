from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from merge import _read_measurement
from timeline import SAMPLE_PREFIX, measure


class MeasurementTest(unittest.TestCase):
    def sample_command(self, latency_ms: float) -> list[str]:
        sample = {
            "latency_ms": latency_ms,
            "correctness": "passed",
            "synchronized": True,
            "workload_identity": "rows=16384,dtype=bf16",
            "warmup": 1,
            "iterations": 2,
            "device_identity": {"physical_device": 0, "serial": "ppu-0"},
            "allocation_identity": "pod-1/job-1",
            "kernel_sha256": "a" * 64,
            "runtime_identity": {"compiler": "hggc", "runtime": "sdk"},
            "cache_policy": "warm",
            "clock_configuration": "locked",
        }
        return [
            sys.executable,
            "-c",
            f"print({(SAMPLE_PREFIX + json.dumps(sample))!r})",
        ]

    def test_measurement_binds_identity_and_commands(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "measurement.json"
            measurement = measure(
                self.sample_command(1.0),
                self.sample_command(1.1),
                "rows=16384,dtype=bf16",
                1,
                2,
                ["AB", "BA"],
                10.0,
                output,
            )
            self.assertEqual(measurement["schema"], "ppu-timeline-measurement/v2")
            self.assertEqual(_read_measurement(output, "measurement"), measurement)

            measurement["runs"][0]["command_sha256"] = "b" * 64
            output.write_text(json.dumps(measurement), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "command hash drifted"):
                _read_measurement(output, "measurement")


if __name__ == "__main__":
    unittest.main()
