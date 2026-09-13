from __future__ import annotations

import hashlib
import json
import math
import subprocess
import uuid
from pathlib import Path
from typing import Any

from supervisor.runner_assets import evaluator_path

from . import main_adapter
from .git_episode import _git
from .models import VerificationResult, VerificationRun
from .store import VERIFY_DIR

ABBA_RESULT_PREFIX = "__ATREX_LONG_HORIZON_ABBA_RESULT__="


def _payload_from_stdout(stdout: str) -> dict[str, Any]:
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


def verification_schedule(repeats: int) -> list[dict[str, int | str]]:
    schedule: list[dict[str, int | str]] = []
    for repeat in range(max(1, repeats)):
        revisions = ("incumbent", "candidate") if repeat % 2 == 0 else ("candidate", "incumbent")
        schedule.extend({"revision": revision, "repeat": repeat} for revision in revisions)
    return schedule


def _geomean(values: list[float]) -> float | None:
    if not values or any(value <= 0.0 or not math.isfinite(value) for value in values):
        return None
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _verification_shape_batches(
    workspace: Path, private_reference_dir: Path | None, batch_size: int
) -> tuple[list[list[str] | None], list[str]]:
    harness = evaluator_path(workspace)
    if not harness.is_file() or "--shape-id" not in harness.read_text(encoding="utf-8"):
        return [None], []
    shapes_path = (private_reference_dir or workspace) / "shapes.json"
    if not shapes_path.is_file():
        return [None], []
    shapes = json.loads(shapes_path.read_text(encoding="utf-8"))
    shape_ids = sorted(
        (str(shape_id) for shape_id in shapes),
        key=lambda shape_id: (
            (
                0,
                int(shape_id),
            )
            if shape_id.isdigit()
            else (1, shape_id)
        ),
    )
    return (
        [
            shape_ids[offset : offset + batch_size]
            for offset in range(0, len(shape_ids), batch_size)
        ],
        shape_ids,
    )


def _merge_batch_results(results: list[dict[str, Any]], shape_ids: list[str]) -> dict[str, Any]:
    latency_by_shape = {
        str(shape_id): float(latency)
        for result in results
        for shape_id, latency in (result.get("latency_us_by_shape") or {}).items()
    }
    missing_shape_ids = [shape_id for shape_id in shape_ids if shape_id not in latency_by_shape]
    latencies = [
        latency_by_shape[shape_id] for shape_id in shape_ids if shape_id in latency_by_shape
    ]
    weighted_scores: list[tuple[float, int]] = []
    for result in results:
        raw_score = result.get("performance_score", result.get("speedup_vs_ref_geomean"))
        by_shape = result.get("latency_us_by_shape")
        if (
            isinstance(raw_score, (int, float))
            and not isinstance(raw_score, bool)
            and raw_score > 0.0
            and math.isfinite(float(raw_score))
            and isinstance(by_shape, dict)
        ):
            weighted_scores.append((float(raw_score), len(by_shape)))
    measured_score_shapes = sum(count for _score, count in weighted_scores)
    performance_score = (
        sum(score * count for score, count in weighted_scores) / measured_score_shapes
        if measured_score_shapes == len(shape_ids) and measured_score_shapes > 0
        else None
    )
    merged = dict(results[-1])
    failures = [str(failure) for result in results for failure in (result.get("failures") or [])]
    if missing_shape_ids:
        failures.append("shape batches did not return latency for: " + ", ".join(missing_shape_ids))
    merged.update(
        {
            "all_pass": not missing_shape_ids
            and performance_score is not None
            and all(result.get("all_pass") for result in results),
            "failures": failures,
            "latency_us_geomean": _geomean(latencies),
            "latency_us_arith_mean": (sum(latencies) / len(latencies) if latencies else 0.0),
            "latency_us_by_shape": latency_by_shape,
            "speedup_vs_ref_mean": performance_score,
            "speedup_vs_ref_geomean": None,
            "performance_score": performance_score,
            "max_abs_err": max(float(result.get("max_abs_err") or 0.0) for result in results),
            "max_rel_err": max(float(result.get("max_rel_err") or 0.0) for result in results),
        }
    )
    return merged


def _merge_batch_payloads(
    payloads: list[dict[str, Any]],
    schedule: list[dict[str, int | str]],
    shape_ids: list[str],
) -> dict[str, Any]:
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
            not isinstance(row, dict) or not isinstance(row.get("result"), dict) for row in rows
        ):
            raise ValueError("shape batch has no valid run result")
        actual = [{"revision": row.get("revision"), "repeat": row.get("repeat")} for row in rows]
        if actual != schedule:
            raise ValueError("shape batch returned an invalid ABBA schedule")
    runs = []
    for index, step in enumerate(schedule):
        rows = [payload["runs"][index] for payload in payloads]
        result = _merge_batch_results([row["result"] for row in rows], shape_ids)
        runs.append(
            {
                **step,
                "exit_code": 0
                if result["all_pass"] and all(row["exit_code"] == 0 for row in rows)
                else 1,
                "result": result,
                "stdout_tail": "\n".join(str(row.get("stdout_tail", "")) for row in rows)[-3000:],
                "stderr_tail": "\n".join(str(row.get("stderr_tail", "")) for row in rows)[-3000:],
            }
        )
    return {
        "schema_version": 1,
        "runs": runs,
        "error": "; ".join(str(payload["error"]) for payload in payloads if payload.get("error"))
        or None,
        "shape_batch_count": len(payloads),
    }


def _result_latency(result: object) -> float | None:
    if not isinstance(result, dict) or not result.get("all_pass"):
        return None
    value = result.get("latency_us_geomean")
    if not isinstance(value, (int, float)):
        performance = result.get("performance")
        value = performance.get("latency_us") if isinstance(performance, dict) else None
    if not isinstance(value, (int, float)) or value <= 0 or not math.isfinite(float(value)):
        return None
    return float(value)


def _result_performance_score(result: object) -> float | None:
    if not isinstance(result, dict) or not result.get("all_pass"):
        return None
    value = result.get("performance_score", result.get("speedup_vs_ref_geomean"))
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or value <= 0
        or not math.isfinite(float(value))
    ):
        return None
    return float(value)


def score_verification_payload(
    payload: object,
    *,
    schedule: list[dict[str, int | str]],
    repeats: int,
    min_improvement_pct: float,
    artifact: str = "",
) -> VerificationResult:
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        return VerificationResult("ERROR", None, None, None, error="unsupported result schema")
    rows = payload.get("runs")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        return VerificationResult("ERROR", None, None, None, error="runs must be a list of objects")
    runs = [
        VerificationRun(
            revision=str(row.get("revision", "")),
            repeat=int(row.get("repeat", -1)),
            exit_code=int(row.get("exit_code", -1)),
            result=row.get("result") if isinstance(row.get("result"), dict) else None,
            stdout_tail=str(row.get("stdout_tail", "")),
            stderr_tail=str(row.get("stderr_tail", "")),
        )
        for row in rows
    ]
    if payload.get("error"):
        return VerificationResult(
            "ERROR",
            None,
            None,
            None,
            runs=runs,
            error=str(payload["error"]),
            artifact=artifact,
        )
    actual = [{"revision": run.revision, "repeat": run.repeat} for run in runs]
    if actual != schedule:
        return VerificationResult(
            "ERROR",
            None,
            None,
            None,
            runs=runs,
            error="remote verifier did not execute the exact ABBA schedule",
            artifact=artifact,
        )
    candidate_latency_values = [
        value
        for run in runs
        if run.revision == "candidate" and run.exit_code == 0
        if (value := _result_latency(run.result)) is not None
    ]
    incumbent_latency_values = [
        value
        for run in runs
        if run.revision == "incumbent" and run.exit_code == 0
        if (value := _result_latency(run.result)) is not None
    ]
    candidate_score_values = [
        value
        for run in runs
        if run.revision == "candidate" and run.exit_code == 0
        if (value := _result_performance_score(run.result)) is not None
    ]
    incumbent_score_values = [
        value
        for run in runs
        if run.revision == "incumbent" and run.exit_code == 0
        if (value := _result_performance_score(run.result)) is not None
    ]
    candidate_latency = _geomean(candidate_latency_values)
    incumbent_latency = _geomean(incumbent_latency_values)
    candidate_score = (
        sum(candidate_score_values) / len(candidate_score_values)
        if candidate_score_values
        else None
    )
    incumbent_score = (
        sum(incumbent_score_values) / len(incumbent_score_values)
        if incumbent_score_values
        else None
    )
    if (
        len(candidate_latency_values) != repeats
        or len(incumbent_latency_values) != repeats
        or len(candidate_score_values) != repeats
        or len(incumbent_score_values) != repeats
    ):
        return VerificationResult(
            "FAIL",
            candidate_latency,
            incumbent_latency,
            None,
            runs=runs,
            error=("not every authoritative ABBA run passed with a valid performance score"),
            artifact=artifact,
            candidate_performance_score=candidate_score,
            incumbent_performance_score=incumbent_score,
        )
    improvement = (
        ((candidate_score / incumbent_score) - 1.0) * 100.0
        if candidate_score and incumbent_score
        else None
    )
    if improvement is None or improvement <= min_improvement_pct:
        return VerificationResult(
            "FAIL",
            candidate_latency,
            incumbent_latency,
            improvement,
            runs=runs,
            error=(
                f"candidate improvement {improvement if improvement is not None else 'unknown'} "
                f"in performance score did not exceed {min_improvement_pct:.3f}%"
            ),
            artifact=artifact,
            candidate_performance_score=candidate_score,
            incumbent_performance_score=incumbent_score,
        )
    return VerificationResult(
        "PASS",
        candidate_latency,
        incumbent_latency,
        improvement,
        runs=runs,
        artifact=artifact,
        candidate_performance_score=candidate_score,
        incumbent_performance_score=incumbent_score,
    )


def _git_blob(workspace: Path, revision: str, relative: str) -> bytes | None:
    result = _git(workspace, "show", f"{revision}:{relative}", check=False, binary=True)
    return result.stdout if result.returncode == 0 else None


def _score_gateway_measurement(
    stdout: str,
    *,
    incumbent: bytes,
    candidate: bytes,
    repeats: int,
    min_improvement_pct: float,
) -> VerificationResult:
    """Judge Runtime facts; never infer authority from Agent summaries or Dev stdout."""
    from supervisor.gateway import SUPERVISOR_MEASUREMENT_PREFIX

    receipt = next(
        (
            json.loads(line.removeprefix(SUPERVISOR_MEASUREMENT_PREFIX))
            for line in reversed(stdout.splitlines())
            if line.startswith(SUPERVISOR_MEASUREMENT_PREFIX)
        ),
        None,
    )
    if not isinstance(receipt, dict):
        raise ValueError("Runtime omitted its authoritative measurement receipt")
    subjects = {
        "incumbent": hashlib.sha256(incumbent).hexdigest(),
        "candidate": hashlib.sha256(candidate).hexdigest(),
    }
    if (
        receipt.get("gateway_kind") != "same_allocation_abba"
        or receipt.get("kernel_subjects") != subjects
        or receipt.get("kernel_sha256") != subjects["candidate"]
    ):
        raise ValueError("Runtime measurement does not match the committed Kernel pair")
    result = receipt.get("result")
    if not isinstance(result, dict) or result.get("comparison") != {
        "method": "abba",
        "repeats": repeats,
    }:
        raise ValueError("Runtime measurement does not match the ABBA policy")
    # The Runtime already aggregates the three identical calls per Shape. Score these
    # authoritative sides once; do not average raw repetitions again in Supervisor.
    rows = []
    for revision, side in (("incumbent", "baseline"), ("candidate", "candidate")):
        metrics = result.get(side)
        if not isinstance(metrics, dict):
            raise ValueError(f"Runtime measurement omitted {side} metrics")
        passed = result.get("correct") is True and metrics.get("correct") is True
        rows.append(
            {
                "revision": revision,
                "repeat": 0,
                "exit_code": 0 if passed else 1,
                "result": {**metrics, "all_pass": passed},
            }
        )
    verification = score_verification_payload(
        {"schema_version": 1, "runs": rows, "error": result.get("error")},
        schedule=verification_schedule(1),
        repeats=1,
        min_improvement_pct=min_improvement_pct,
        artifact=str(receipt["artifact"]),
    )
    verification.gateway_record_id = str(receipt["gateway_record_id"])
    verification.reused = receipt.get("reused") is True
    return verification


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
    ):
        self.hardware = hardware
        self.profile = profile
        self.url = url
        self.ssh = ssh
        self.ssh_init = ssh_init
        self.health_command = health_command
        self.timeout = timeout
        self.repeats = max(1, repeats)
        self.per_run_timeout = per_run_timeout
        self.min_improvement_pct = min_improvement_pct
        self.queue_wait_grace = queue_wait_grace
        self.private_reference_dir = private_reference_dir

    def verify(
        self,
        workspace: Path,
        *,
        base_commit: str,
        candidate_commit: str,
        changed_paths: list[str],
    ) -> VerificationResult:
        try:
            if any(path != "kernel.py" for path in changed_paths):
                raise ValueError("ABBA verification supports committed kernel.py changes only")
            incumbent = _git_blob(workspace, base_commit, "kernel.py")
            candidate = _git_blob(workspace, candidate_commit, "kernel.py")
            if incumbent is None or candidate is None:
                raise ValueError("committed Kernel source is missing")
            if (workspace / "kernel.py").read_bytes() != candidate:
                raise ValueError("workspace Kernel differs from the committed candidate")
            relative_dir = f"{VERIFY_DIR}/{uuid.uuid4().hex}"
            directory = workspace / relative_dir
            directory.mkdir(parents=True, exist_ok=False)
            (directory / "incumbent.py").write_bytes(incumbent)
            shape_batches, _ = _verification_shape_batches(workspace, self.private_reference_dir, 1)
            from supervisor.gateway import DEFAULT_ABBA_BATCH_WORKERS, MEASUREMENT_REPETITIONS

            workers = 1 if self.ssh else DEFAULT_ABBA_BATCH_WORKERS
            waves = math.ceil(len(shape_batches) / workers)

            process = main_adapter.run_sandbox(
                workspace,
                self.hardware,
                self.profile,
                self.url,
                self.timeout,
                [],
                ssh=self.ssh,
                ssh_init=self.ssh_init,
                health_command=self.health_command,
                sync=(),
                gateway_kind="run",
                private_reference_dir=self.private_reference_dir,
                gateway_options=(
                    "--mode",
                    "full",
                    "--baseline-path",
                    f"{relative_dir}/incumbent.py",
                    "--comparison-repeats",
                    str(self.repeats),
                ),
                reuse_completed=True,
                comparison_run_timeout=self.per_run_timeout,
                wall_timeout=MEASUREMENT_REPETITIONS
                * waves
                * (self.timeout + self.queue_wait_grace + 120),
            )
            try:
                return _score_gateway_measurement(
                    process.stdout,
                    incumbent=incumbent,
                    candidate=candidate,
                    repeats=self.repeats,
                    min_improvement_pct=self.min_improvement_pct,
                )
            except ValueError as exc:
                raise ValueError(
                    f"{exc}; gateway exit={process.returncode}: {process.stderr[-2000:]}"
                ) from exc
        except (
            OSError,
            subprocess.SubprocessError,
            RuntimeError,
            ValueError,
            KeyError,
            TypeError,
        ) as exc:
            return VerificationResult(
                "ERROR",
                None,
                None,
                None,
                error=f"gateway ABBA verification failed: {exc}",
            )
