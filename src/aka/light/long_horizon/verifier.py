from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from aka.legacy.evaluation.abba import (
    verification_schedule, geomean as _geomean, merge_batch_results as _merge_batch_results,
    result_latency as _result_latency, result_performance_score as _result_performance_score,
)
from aka.light.evaluation.policy import SHAPE_BATCH_SIZE, BATCH_WORKERS, validate_abba_repeats

from .models import VerificationResult, VerificationRun

ABBA_RESULT_PREFIX = "__ATREX_LONG_HORIZON_ABBA_RESULT__="
DEFAULT_SHAPE_BATCH_SIZE = SHAPE_BATCH_SIZE
DEFAULT_SHAPE_BATCH_WORKERS = BATCH_WORKERS


def score_verification_payload(
    payload: object, *, schedule: list[dict[str, int | str]], repeats: int,
    min_improvement_pct: float, artifact: str = "",
) -> VerificationResult:
    """Light retention policy: strict GeoMean latency reduction, as in Runtime.

    Main's score-based policy remains unchanged in aka.legacy.evaluation.abba.
    Scores are optional diagnostic values and cannot enable or veto promotion.
    """
    if not math.isfinite(min_improvement_pct) or min_improvement_pct < 0:
        raise ValueError("Minimum latency improvement must be finite and non-negative")
    rows = payload.get("runs") if isinstance(payload, dict) else None
    if (not isinstance(rows, list) or payload.get("error")
            or any(not isinstance(row, dict) for row in rows)
            or [{"revision": row.get("revision"), "repeat": row.get("repeat")} for row in rows] != schedule):
        return VerificationResult("ERROR", None, None, None, artifact=artifact, error="Incomplete ABBA schedule")
    sides = {"candidate": [], "incumbent": []}
    scores = {"candidate": [], "incumbent": []}
    runs = []
    for row in rows:
        result = row.get("result")
        runs.append(VerificationRun(row["revision"], row["repeat"], row.get("exit_code", -1), result))
        latency = _result_latency(result)
        if isinstance(result, dict) and isinstance(result.get("latency_us_geomean"), bool):
            latency = None
        if row.get("exit_code") == 0 and isinstance(result, dict) and result.get("all_pass") is True and latency:
            sides[row["revision"]].append(latency)
            if score := _result_performance_score(result):
                scores[row["revision"]].append(score)
    candidate, incumbent = (_geomean(sides[side]) for side in ("candidate", "incumbent"))
    verdict = VerificationResult("FAIL", candidate, incumbent, None, runs=runs, artifact=artifact,
        candidate_performance_score=sum(scores["candidate"]) / len(scores["candidate"]) if scores["candidate"] else None,
        incumbent_performance_score=sum(scores["incumbent"]) / len(scores["incumbent"]) if scores["incumbent"] else None)
    if any(len(values) != repeats for values in sides.values()) or not candidate or not incumbent:
        verdict.error = "Not every ABBA run passed with a valid GeoMean latency"
        return verdict
    verdict.improvement_pct = (incumbent - candidate) / incumbent * 100
    if (verdict.improvement_pct > min_improvement_pct
            and not math.isclose(verdict.improvement_pct, min_improvement_pct, rel_tol=1e-12, abs_tol=1e-12)):
        verdict.gate = "PASS"
    else:
        verdict.error = f"GeoMean latency reduction {verdict.improvement_pct:.3f}% did not exceed {min_improvement_pct:.3f}%"
    return verdict


def parse_abba_payload(stdout: str) -> dict[str, Any]:
    """Extract the long-horizon ABBA payload from ordinary sandbox stdout."""
    for line in reversed(stdout.splitlines()):
        if not line.startswith(ABBA_RESULT_PREFIX):
            continue
        try:
            payload = json.loads(line[len(ABBA_RESULT_PREFIX) :])
        except json.JSONDecodeError as exc:
            raise ValueError("malformed ABBA result sentinel") from exc
        if not isinstance(payload, dict):
            raise ValueError("ABBA result sentinel must contain a JSON object")
        return payload
    raise ValueError("missing ABBA result sentinel")


def _verification_shape_batches(
    workspace: Path, private_reference_dir: Path | None, batch_size: int
) -> tuple[list[list[str] | None], list[str]]:
    harness = workspace / "test_kernel.py"
    if not harness.is_file() or "--shape-id" not in harness.read_text(encoding="utf-8"):
        return [None], []
    shapes_path = (private_reference_dir or workspace) / "shapes.json"
    if not shapes_path.is_file():
        return [None], []
    shapes = json.loads(shapes_path.read_text(encoding="utf-8"))
    shape_ids = sorted(
        (str(shape_id) for shape_id in shapes),
        key=lambda shape_id: (
            0,
            int(shape_id),
        )
        if shape_id.isdigit()
        else (1, shape_id),
    )
    return (
        [
            shape_ids[offset : offset + batch_size]
            for offset in range(0, len(shape_ids), batch_size)
        ],
        shape_ids,
    )


def merge_abba_batch_payloads(
    payloads: list[dict[str, Any]],
    schedule: list[dict[str, int | str]],
    shape_ids: list[str],
) -> dict[str, Any]:
    """Combine batch results for a shared ABBA schedule and complete Shape set."""
    if len(payloads) == 1:
        return payloads[0]
    for payload in payloads:
        rows = payload.get("runs")
        if (
            payload.get("schema_version") != 1
            or payload.get("error")
            or not isinstance(rows, list)
            or len(rows) != len(schedule)
        ):
            raise ValueError(str(payload.get("error") or "invalid shape batch result"))
        if any(
            not isinstance(row, dict) or not isinstance(row.get("result"), dict)
            for row in rows
        ):
            raise ValueError("shape batch has no valid run result")
        actual = [
            {"revision": row.get("revision"), "repeat": row.get("repeat")}
            for row in rows
        ]
        if actual != schedule:
            raise ValueError("shape batch returned an invalid ABBA schedule")
    runs = []
    for index, step in enumerate(schedule):
        rows = [payload["runs"][index] for payload in payloads]
        result = _merge_batch_results([row["result"] for row in rows], shape_ids)
        # Main's shared merger requires reference scores; Light's latency gate
        # requires complete measured Shapes, not optional reference metadata.
        result["all_pass"] = (set(result["latency_us_by_shape"]) == set(shape_ids)
                              and all(row["result"].get("all_pass") is True for row in rows))
        from aka.light.evaluation.evaluation_feedback import merge_feedback
        result.update(merge_feedback(row["result"] for row in rows))
        runs.append(
            {
                **step,
                "exit_code": 0
                if result["all_pass"] and all(row["exit_code"] == 0 for row in rows)
                else 1,
                "result": result,
                "stdout_tail": "\n".join(
                    str(row.get("stdout_tail", "")) for row in rows
                )[-3000:],
                "stderr_tail": "\n".join(
                    str(row.get("stderr_tail", "")) for row in rows
                )[-3000:],
            }
        )
    return {
        "schema_version": 1,
        "runs": runs,
        "error": "; ".join(
            str(payload["error"]) for payload in payloads if payload.get("error")
        )
        or None,
        "shape_batch_count": len(payloads),
    }


class GatewayABBAValidator:
    def __init__(
        self,
        *,
        hardware: str,
        profile: str = "",
        url: str = "",
        ssh: str = "",
        ssh_init: str = "",
        health_command: str = "",
        timeout: int = 600,
        repeats: int = 2,
        per_run_timeout: int = 120,
        min_improvement_pct: float = 0.0,
        queue_wait_grace: int = 14_400,
        private_reference_dir: Path | None = None,
        shape_batch_size: int = DEFAULT_SHAPE_BATCH_SIZE,
    ):
        self.hardware = hardware
        self.profile = profile
        self.url = url
        self.ssh = ssh
        self.ssh_init = ssh_init
        self.health_command = health_command
        self.timeout = timeout
        validate_abba_repeats(repeats)
        self.repeats = repeats
        self.per_run_timeout = per_run_timeout
        self.min_improvement_pct = min_improvement_pct
        self.queue_wait_grace = queue_wait_grace
        self.private_reference_dir = private_reference_dir
        self.shape_batch_size = shape_batch_size
