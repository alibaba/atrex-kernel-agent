"""Direct JSON transport for one standard Evaluate result.

New Light execution emits the object itself.  The historical text prefix is
accepted only while reading old Records and older evaluator harnesses.
"""

from __future__ import annotations

import json

LEGACY_TEST_RESULT_PREFIX = "[test_kernel] RESULT_JSON="
_RESULT_KEYS = {
    "all_pass",
    "correct",
    "correctness",
    "failures",
    "latency_us_by_shape",
    "latency_us_geomean",
    "performance_score",
}


def parse_evaluation_result_line(line: str) -> dict | None:
    """Decode a direct result object or one historical prefixed result line."""
    value = line.strip()
    if value.startswith(LEGACY_TEST_RESULT_PREFIX):
        value = value[len(LEGACY_TEST_RESULT_PREFIX) :]
    elif not value.startswith("{"):
        return None
    try:
        result = json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(result, dict) or not (_RESULT_KEYS & result.keys()):
        return None
    return result


def evaluation_result_json(result: dict) -> str:
    """Encode one result without a log-label wrapper."""
    return json.dumps(result, ensure_ascii=False, allow_nan=False)
