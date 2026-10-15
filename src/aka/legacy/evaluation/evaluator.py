"""Prepare original evaluator commands and interpret their captured results.

The injected host owns execution, retry, cancellation and dependency lifetime.
Preparation and interpretation never submit commands or run a model.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import shutil
from typing import Sequence
import uuid

from aka.contracts.evaluation import (
    EvaluationFailure,
    EvaluationConfiguration,
    EvaluationRequest,
    PreparedEvaluation,
    VerificationResult,
    VerificationRun,
)
from aka.contracts.execution import CommandSpec, JobResult
from aka.contracts.workspace import CandidateWorkspace
from aka.legacy.task.candidate_workspace.excludes import VERIFY_DIR
from .abba import (
    _merge_batch_payloads,
    _payload_from_stdout,
    _verification_shape_batches,
    interpret_verification_payload,
    verification_schedule,
)
from .drivers import remote_abba
from .results import (
    EPISODE_EVALUATIONS_PATH,
    _latest_complete_canonical_performance,
    _latest_complete_episode_performance,
)

RESULT_PREFIX = "[test_kernel] RESULT_JSON="


def atomic_write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def result_from_stdout(stdout: str) -> dict:
    for line in reversed(stdout.splitlines()):
        if line.startswith(RESULT_PREFIX):
            result = json.loads(line[len(RESULT_PREFIX) :])
            if isinstance(result, dict):
                return result
    raise RuntimeError("sandbox test output has no structured RESULT_JSON line")


@dataclass(frozen=True)
class _Plan:
    request: EvaluationRequest
    schedule: tuple[dict, ...] = ()
    shape_ids: tuple[str, ...] = ()
    result_paths: tuple[Path, ...] = ()
    directory: Path | None = None
    error: str = ""


class LegacyEvaluator:
    def __init__(
        self,
        configuration: EvaluationConfiguration = EvaluationConfiguration(),
        *,
        candidate_workspace: CandidateWorkspace | None = None,
        abba_driver: Path | None = None,
    ):
        self.configuration = configuration
        self.candidate_workspace = candidate_workspace
        self.abba_driver = abba_driver or Path(remote_abba.__file__)

    def prepare(self, request: EvaluationRequest) -> PreparedEvaluation:
        config = self.configuration
        if request.mode == "fast":
            return PreparedEvaluation((), 1, _Plan(request))
        if request.mode in {"v0", "framework"}:
            (request.workspace / ".atrex_long_horizon/ci_gateway_failure.json").unlink(missing_ok=True)
            command = ["python", "test_kernel.py", "--version", f"v{request.version}"]
            if request.mode == "framework":
                command += ["--multi-seed", "5"]
                if config.native:
                    command += ["--timed-runs", "5"]
            command.append("--no-memory")
            return PreparedEvaluation(
                (
                    CommandSpec(
                        request.workspace, tuple(command), timeout=config.timeout
                    ),
                ),
                1,
                _Plan(request),
            )
        if request.mode != "abba":
            raise ValueError(f"unsupported evaluation mode: {request.mode}")
        schedule = verification_schedule(config.repeats)
        if config.per_run_timeout * len(schedule) + 30 > config.timeout:
            return PreparedEvaluation(
                (),
                config.max_parallelism,
                _Plan(
                    request,
                    error="ABBA schedule cannot fit in one gateway allocation timeout",
                ),
            )
        if self.candidate_workspace is None:
            raise ValueError("ABBA preparation requires an injected CandidateWorkspace")
        batches, shape_ids = _verification_shape_batches(
            request.workspace, config.private_reference_dir, config.shape_batch_size
        )
        relative_dir = f"{VERIFY_DIR}/{uuid.uuid4().hex}"
        directory = request.workspace / relative_dir
        directory.mkdir(parents=True, exist_ok=False)
        shutil.copy2(self.abba_driver, directory / "test_kernel.py")
        manifests = {
            "incumbent": self.candidate_workspace.write_revision(
                request.workspace,
                request.base_commit,
                list(request.changed_paths),
                directory,
                "incumbent",
            ),
            "candidate": self.candidate_workspace.write_revision(
                request.workspace,
                request.candidate_commit,
                list(request.changed_paths),
                directory,
                "candidate",
            ),
        }
        steps, result_paths = [], []
        for index, ids in enumerate(batches):
            request_relative = f"{relative_dir}/request-{index:04d}.json"
            result_relative = f"{relative_dir}/result-{index:04d}.json"
            command = ["python3", "test_kernel.py", "--version", "vlong", "--no-memory"]
            for shape_id in ids or []:
                command += ["--shape-id", shape_id]
            atomic_write_json(
                request.workspace / request_relative,
                {
                    "schema_version": 1,
                    "schedule": schedule,
                    "manifests": manifests,
                    "command": command,
                    "run_timeout_seconds": config.per_run_timeout,
                },
            )
            steps.append(
                CommandSpec(
                    request.workspace,
                    (
                        "python3",
                        f"{relative_dir}/test_kernel.py",
                        request_relative,
                        result_relative,
                    ),
                    timeout=config.timeout,
                    operation="dev",
                )
            )
            result_paths.append(request.workspace / result_relative)
        return PreparedEvaluation(
            tuple(steps),
            max(1, min(config.max_parallelism, len(steps))),
            _Plan(
                request,
                tuple(schedule),
                tuple(shape_ids),
                tuple(result_paths),
                directory,
                error=(
                    "gateway ABBA verification failed: max_workers must be greater than 0"
                    if not steps and config.max_parallelism != 1
                    else ""
                ),
            ),
        )

    def record_step(
        self, prepared: PreparedEvaluation, index: int, result: JobResult
    ) -> None:
        """Persist a completed ABBA batch before other workers finish or fail."""
        plan = prepared.context
        if plan.request.mode == "abba" and result.transport_returncode == 0:
            atomic_write_json(
                plan.result_paths[index], _payload_from_stdout(result.stdout)
            )

    def interpret(
        self, prepared: PreparedEvaluation, results: Sequence[JobResult]
    ) -> VerificationResult:
        plan = prepared.context
        request = plan.request
        if plan.error:
            return VerificationResult("ERROR", None, None, None, error=plan.error)
        if len(results) != len(prepared.steps):
            raise ValueError("evaluation requires one result per planned step")
        if request.mode == "fast":
            return self._fast_result(request)
        if request.mode in {"v0", "framework"}:
            return self._baseline_result(request, results[0])
        payloads = []
        for index, result in enumerate(results):
            if result.transport_returncode != 0:
                raise RuntimeError(
                    f"gateway ABBA command exited {result.transport_returncode}: "
                    + (result.stdout + "\n" + result.stderr)[-3000:]
                )
            payloads.append(_payload_from_stdout(result.stdout))
        payload = _merge_batch_payloads(
            payloads, list(plan.schedule), list(plan.shape_ids)
        )
        result_path = plan.directory / "result.json"
        try:
            atomic_write_json(result_path, payload)
        except (OSError, TypeError, ValueError) as exc:
            return VerificationResult(
                "ERROR",
                None,
                None,
                None,
                error=f"cannot persist gateway ABBA result: {type(exc).__name__}: {exc}",
            )
        return interpret_verification_payload(
            payload,
            schedule=list(plan.schedule),
            repeats=max(1, self.configuration.repeats),
            artifact=str(result_path),
        )

    def _fast_result(self, request):
        artifact = str(request.workspace / EPISODE_EVALUATIONS_PATH)
        expected = (
            set(request.expected_shape_ids)
            if request.expected_shape_ids is not None
            else None
        )
        candidate = _latest_complete_episode_performance(
            request.workspace,
            expected_shape_ids=expected,
            required_performance_objective="shape_speedup_arithmetic_mean",
        )
        if candidate is None:
            return VerificationResult(
                "FAIL",
                None,
                None,
                None,
                error="fast mode requires one complete passing evaluator result for the final kernel.py and solution.json pair",
                artifact=artifact,
            )
        latency, score = float(candidate["latency_us_geomean"]), float(
            candidate["performance_score"]
        )
        run = VerificationRun("candidate", 0, 0, candidate)
        incumbent = _latest_complete_canonical_performance(
            request.canonical_workspace,
            before_version=request.memory_version,
            expected_shape_ids=expected,
            required_performance_objective="shape_speedup_arithmetic_mean",
        )
        if incumbent is None:
            return VerificationResult(
                "FAIL",
                latency,
                None,
                None,
                runs=[run],
                error="fast mode could not find complete canonical incumbent performance",
                artifact=artifact,
            )
        performance, _ = incumbent
        incumbent_latency = float(performance["latency_us_geomean"])
        incumbent_score = float(performance["performance_score"])
        improvement = (score / incumbent_score - 1.0) * 100.0
        return VerificationResult(
            "PASS",
            latency,
            incumbent_latency,
            improvement,
            runs=[run],
            artifact=artifact,
            candidate_performance_score=score,
            incumbent_performance_score=incumbent_score,
        )

    def _baseline_result(self, request, observation):
        failure_path = request.workspace / ".atrex_long_horizon/ci_gateway_failure.json"
        if observation.transport_returncode != 0 and failure_path.is_file():
            details = json.loads(failure_path.read_text())
            category = details.get("failure_class", "protocol")
            repairable = category == "request" and not details.get("reference_rejected") and bool(
                details.get("candidate_rejected") or details.get("io_contract_rejected"))
            message = ("candidate request rejected; repair source-policy violations and make "
                       "Model.forward accept exactly the trusted input keys"
                       if repairable else f"gateway validation pending: failure_class={category}; "
                       "no correctness result was produced")
            failure = EvaluationFailure(category, f"gateway-http-{details.get('http_status', 'unknown')}",
                                        message, candidate_repairable=repairable)
            return VerificationResult("FAIL" if repairable else "ERROR", None, None, None,
                                      error=message, failure=failure)
        if request.mode == "framework" and observation.transport_returncode != 0:
            return VerificationResult(
                "FAIL",
                None,
                None,
                None,
                error=f"combined validation command failed (exit={observation.transport_returncode})",
            )
        result = result_from_stdout(observation.stdout)
        run = VerificationRun("candidate", 0, observation.transport_returncode, result)
        if request.mode == "v0":
            error = (
                ""
                if observation.transport_returncode == 0 and result.get("all_pass")
                else "sandbox V0 baseline failed correctness/performance validation"
            )
        else:
            error = framework_result_problem(result, request.workspace)
        return VerificationResult(
            "FAIL" if error else "PASS",
            result.get("latency_us_geomean"),
            None,
            None,
            runs=[run],
            error=error,
        )


def framework_result_problem(result, workspace):
    from .memory import read_memory

    if not result.get("all_pass"):
        return "combined multi-seed correctness validation failed"
    latency = result.get("latency_us_geomean")
    if not isinstance(latency, (int, float)) or latency <= 0:
        return "validation reported no usable latency_us_geomean"
    score = result.get("performance_score")
    if (
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or score <= 0
        or not math.isfinite(float(score))
    ):
        return "validation reported no usable performance_score"
    baseline = set(
        ((read_memory(workspace, 0) or {}).get("performance") or {}).get(
            "latency_us_by_shape", {}
        )
    )
    measured = set(result.get("latency_us_by_shape") or {})
    if baseline and measured != baseline:
        return (
            "latency_us_by_shape does not cover the same workloads as v0 "
            + f"(missing {sorted(baseline - measured)}, unexpected {sorted(measured - baseline)})"
        )
    return ""
