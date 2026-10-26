"""Original local evaluator result parsing and memory projection."""

from __future__ import annotations
import json
from pathlib import Path

TEST_RESULT_PREFIX = "[test_kernel] RESULT_JSON="


def _test_result_from_stdout(stdout: str) -> dict:
    """Read the structured result emitted by the active sandbox harness."""
    for line in reversed(stdout.splitlines()):
        if line.startswith(TEST_RESULT_PREFIX):
            result = json.loads(line[len(TEST_RESULT_PREFIX) :])
            if isinstance(result, dict):
                return result
    raise RuntimeError("sandbox test output has no structured RESULT_JSON line")
