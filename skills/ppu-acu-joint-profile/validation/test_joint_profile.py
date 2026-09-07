from __future__ import annotations

import unittest
from pathlib import Path

import sys

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from merge import _metric_summary


class JointProfileTest(unittest.TestCase):
    def test_metric_summary_is_interval_weighted(self) -> None:
        summary = _metric_summary(
            [
                {
                    "packet_index": 0,
                    "metric_name": "compute",
                    "metric_value": 0.0,
                    "interval_ns": 1.0,
                },
                {
                    "packet_index": 0,
                    "metric_name": "compute",
                    "metric_value": 100.0,
                    "interval_ns": 3.0,
                },
            ]
        )
        self.assertEqual(summary["packet_0:compute"]["time_weighted_mean"], 75.0)


if __name__ == "__main__":
    unittest.main()
