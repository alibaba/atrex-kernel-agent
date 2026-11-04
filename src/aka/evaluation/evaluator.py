"""Prepare original evaluator commands and interpret their captured results.

The injected host owns execution, retry, cancellation and dependency lifetime.
Preparation and interpretation never submit commands or run a model.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
import stat
from typing import Sequence
import uuid

from aka.contracts.evaluation import (
    EvaluationFailure,
    EvaluationConfiguration,
    EvaluationRequest,
    PreparedEvaluation,
    PreparationFile,
    VerificationResult,
    VerificationRun,
    MeasurementArtifact,
    StepEvaluation,
    EpisodeEvaluationContent,
)
from aka.contracts.execution import CommandSpec, JobResult
from aka.contracts.workspace import CandidateWorkspace
from aka.contracts.problem import EvaluationInput
from .runtime_paths import VERIFY_DIR
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
    error: str = ""
    gateway: object = None
    native: bool = False
    baseline: object = None
    candidate_source: MeasurementArtifact | None = None


class LegacyEvaluator:
    def __init__(
        self,
        configuration: EvaluationConfiguration = EvaluationConfiguration(),
        *,
        candidate_workspace: CandidateWorkspace | None = None,
        abba_driver: Path | None = None,
        evaluation_input: EvaluationInput | None = None,
    ):
        self.configuration = configuration
        self.evaluation_input = evaluation_input
        self.candidate_workspace = candidate_workspace
        self.abba_driver = abba_driver or Path(remote_abba.__file__)

    def _command_evaluator(self):
        from .gateway_command import GatewayCommandEvaluator
        return GatewayCommandEvaluator(measurement_reader=self.configuration.measurement_reader,
            execution_capabilities=self.configuration.execution_capabilities,
            evaluation_input=self.evaluation_input, shape_batch_size=self.configuration.shape_batch_size,
            measurement_repetitions=self.configuration.measurement_repetitions)

    def prepare(self, request: EvaluationRequest) -> PreparedEvaluation:
        prepared = self._prepare(request)
        if getattr(prepared.context, "native", False):
            return prepared
        if request.mode in {"v0", "framework"} and self.configuration.hardware:
            command = replace(prepared.steps[0], hardware=self.configuration.hardware,
                              environment=self.configuration.environment)
            gateway = self._command_evaluator().prepare(command)
            prepared = replace(prepared, steps=gateway.steps, max_parallelism=gateway.max_parallelism,
                               context=replace(prepared.context, gateway=gateway), fallback_steps=gateway.fallback_steps)
        if self.evaluation_input is not None and (request.mode in {"abba", "numerical"} or
                request.mode in {"v0", "framework"} and not self.configuration.hardware):
            from .private_inputs import selected_content, portable_inputs
            # Numerical preparation already freezes its trusted inputs exactly
            # once. Only the declared runtime is added to its execution view.
            content = {} if request.mode == "numerical" else selected_content(self.evaluation_input)
            from .gateway_request import _candidate_languages, _json_object
            requirements = ()
            item = self.evaluation_input.worker_requirements
            if item is not None:
                text = item.source.read_text(encoding="utf-8") if item.source is not None else item.text
                kernel = request.workspace / "kernel.py"
                source = kernel.read_text(encoding="utf-8") if kernel.is_file() else ""
                from aka.execution.environment_query import select_requirements
                requirements = tuple(select_requirements(text, _candidate_languages(
                    _json_object(request.workspace / "solution.json") or {}, candidate_sources=(source,))))
            prepared = replace(prepared, steps=tuple(replace(step, requirements=requirements) for step in prepared.steps))
            prepared = replace(prepared, steps=tuple(portable_inputs(step, self.evaluation_input, content,
                evaluator=True, kind="run", capabilities=self.configuration.execution_capabilities,
                inject_private=request.mode != "numerical", prepared_files=prepared.inputs) for step in prepared.steps))
        if request.mode == "numerical":
            return prepared
        if request.mode in {"v0", "framework"} and getattr(prepared.context, "gateway", None) is None:
            prepared = replace(prepared, steps=prepared.steps * self.configuration.measurement_repetitions)
        if request.mode == "abba":
            config = self.configuration
            prepared = replace(prepared, retry_keys=tuple(
                f"abba:{request.base_commit}:{request.candidate_commit}:"
                f"{max(1, config.repeats)}:{config.shape_batch_size}:{Path(command.command[-2]).name}"
                for command in prepared.steps))
        from .measurement import legacy_identities
        return replace(prepared, measurement_identities=legacy_identities(
            prepared, self.configuration, self.evaluation_input))

    def _prepare(self, request: EvaluationRequest) -> PreparedEvaluation:
        config = self.configuration
        if self.evaluation_input is not None:
            from aka.task.problem.resources import validate_resources
            validate_resources(self.evaluation_input)
            from .resources import validate_harness
            validate_harness(request.workspace, self.evaluation_input)
        if request.mode == "numerical":
            from .numerical_preparation import prepare_numerical
            return prepare_numerical(request, config, self.evaluation_input, output_root=VERIFY_DIR)
        if request.mode == "fast":
            return PreparedEvaluation((), 1, _Plan(request))
        if request.mode in {"v0", "framework"}:
            command = ["python", "test_kernel.py", "--version", f"v{request.version}"]
            if request.mode == "framework":
                command += ["--multi-seed", "5"]
            command.append("--no-memory")
            return PreparedEvaluation(
                (
                    CommandSpec(
                        request.workspace, tuple(command), timeout=config.timeout, environment=config.environment
                    ),
                ),
                1,
                _Plan(request, baseline=(config.measurement_reader.baseline(request.workspace)
                    if not config.hardware and config.measurement_repetitions > 1 and config.measurement_reader is not None else None)),
            )
        if request.mode != "abba":
            raise ValueError(f"unsupported evaluation mode: {request.mode}")
        if config.comparison_method == "native-abba":
            from .native_abba import prepare_comparison
            return prepare_comparison(request, config, self.candidate_workspace, self.evaluation_input, _Plan(request))
        from .metadata import distributed_evaluation_world_size
        metadata_path = (config.private_reference_dir or request.workspace) / "metadata.json"
        metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else None
        num_gpus = distributed_evaluation_world_size(metadata)
        schedule = verification_schedule(config.repeats)
        if self.candidate_workspace is None:
            raise ValueError("ABBA preparation requires an injected CandidateWorkspace")
        from .resources import evaluator_harness
        batches, shape_ids = _verification_shape_batches(
            request.workspace, config.private_reference_dir, config.shape_batch_size,
            harness=evaluator_harness(self.evaluation_input)
        )
        relative_dir = f"{VERIFY_DIR}/{uuid.uuid4().hex}"
        inputs = [PreparationFile(
            f"{relative_dir}/test_kernel.py", self.abba_driver.read_bytes(),
            stat.S_IMODE(self.abba_driver.stat().st_mode) & 0o777, "text/x-python",
        )]
        manifests = {}
        for label, revision in (("incumbent", request.base_commit), ("candidate", request.candidate_commit)):
            content = self.candidate_workspace.read_revision(
                request.workspace, revision, list(request.changed_paths))
            manifest = {}
            for index, relative in enumerate(request.changed_paths):
                data = content.files[relative]
                snapshot = f"snapshots/{label}/{index:04d}.bin" if data is not None else None
                manifest[relative] = snapshot
                if data is not None:
                    inputs.append(PreparationFile(f"{relative_dir}/{snapshot}", data))
            manifests[label] = manifest
        steps = []
        for index, ids in enumerate(batches):
            request_relative = f"{relative_dir}/request-{index:04d}.json"
            result_relative = f"{relative_dir}/result-{index:04d}.json"
            command = ["python3", "test_kernel.py", "--version", "vlong", "--no-memory"]
            for shape_id in ids or []:
                command += ["--shape-id", shape_id]
            if self.evaluation_input is not None and self.evaluation_input.adapter == "legacy-native-v1" and self.evaluation_input.harness is None:
                from .measurement_policy import native_command
                command = list(native_command(command, config.environment))
            value = {
                "schema_version": 1,
                "schedule": schedule,
                "manifests": manifests,
                "command": command,
                "run_timeout_seconds": config.per_run_timeout,
            }
            inputs.append(PreparationFile(request_relative,
                (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8"),
                content_type="application/json"))
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
                    environment=config.environment,
                    num_gpus=num_gpus,
                )
            )
        return PreparedEvaluation(
            tuple(steps),
            max(1, min(config.max_parallelism, len(steps))),
            _Plan(
                request,
                tuple(schedule),
                tuple(shape_ids),
                error=(
                    "gateway ABBA verification failed: max_workers must be greater than 0"
                    if not steps and config.max_parallelism != 1
                    else ""
                ),
            ),
            output_directory=Path(relative_dir),
            inputs=tuple(inputs),
        )

    def fallback(self, prepared, results):
        gateway = getattr(prepared.context, "gateway", None)
        if gateway is None:
            return ()
        return self._command_evaluator().fallback(gateway, results)

    def observe_step(
        self, prepared: PreparedEvaluation, index: int, result: JobResult
    ) -> StepEvaluation:
        """Return completed ABBA content for immediate Experiment persistence."""
        plan = prepared.context
        if getattr(plan, "gateway", None) is not None:
            return self._command_evaluator().observe_step(plan.gateway, index, result)
        assessment = self.diagnose_step(prepared, index, result)
        if plan.native:
            source = (plan.candidate_source,) if index == 0 and plan.candidate_source is not None else ()
            return StepEvaluation((MeasurementArtifact(f"receipt-{index:04d}.json", result.stdout.encode(), "measurement_batch"), *source),
                assessment, workspace_files=result.files)
        if plan.request.mode in {"v0", "framework"}:
            from .baseline import BaselinePreparation
            BaselinePreparation.print_v0_evaluator_output(result)
        if plan.request.mode == "abba" and result.returncode != 0:
            failure = RuntimeError(f"gateway ABBA command exited {result.returncode}: "
                                   + (result.stdout + "\n" + result.stderr)[-3000:])
            failure.measurement_assessment = assessment
            raise failure
        if plan.request.mode == "abba" and result.returncode == 0:
            return StepEvaluation((self._artifact(f"result-{index:04d}.json",
                _payload_from_stdout(self._dev_stdout(plan.request, result.stdout)), "measurement_batch"),), assessment, workspace_files=result.files)
        return StepEvaluation(assessment=assessment, workspace_files=result.files)

    def diagnose_step(self, prepared, index, result):
        """Explain unavailable or partial facts before a host raises on transport."""
        if not prepared.measurement_identities or index >= len(prepared.measurement_identities):
            return None
        if prepared.context.request.mode == "numerical":
            from .numerical_preparation import assess_numerical
            return assess_numerical(prepared.context, prepared.context.step_cases[index], result)
        gateway = getattr(prepared.context, "gateway", None)
        if gateway is not None:
            return self._command_evaluator().diagnose_step(gateway, index, result)
        from .measurement import assess_payload
        identity = prepared.measurement_identities[index]
        payload = None
        try:
            if prepared.context.native:
                from .native_abba import parse_native_abba
                shapes = list(dict.fromkeys(point.shape_id for point in identity.points))
                payload = parse_native_abba(json.loads(result.stdout), self.configuration.repeats, shapes)
                offset = min(point.repeat for point in identity.points)
                payload["runs"] = [{**row, "repeat": row["repeat"] + offset} for row in payload["runs"]]
            elif prepared.context.request.mode == "abba":
                payload = _payload_from_stdout(self._dev_stdout(prepared.context.request, result.stdout))
            else:
                measured = result_from_stdout(result.stdout)
                payload = {"runs": [{"revision": "candidate", "repeat": identity.points[0].repeat,
                    "exit_code": result.returncode, "result": measured}]}
        except (ValueError, RuntimeError, TypeError, KeyError, AttributeError):
            pass  # The absence of evidence remains explicit in the assessment.
        return assess_payload(identity, result, payload)

    def interpret(
        self, prepared: PreparedEvaluation, results: Sequence[JobResult]
    ) -> VerificationResult:
        plan = prepared.context
        request = plan.request
        if request.mode == "numerical":
            from .numerical_preparation import interpret_numerical
            return interpret_numerical(prepared, results)
        if plan.error:
            return VerificationResult("ERROR", None, None, None, error=plan.error)
        if len(results) != len(prepared.steps) and not (prepared.fallback_steps and len(results) == 1):
            raise ValueError("evaluation requires one result per planned step")
        if request.mode == "fast":
            return self._fast_result(request)
        if plan.native:
            from .native_abba import parse_native_abba, interpret_payloads
            payloads = []
            assessments = tuple(self.diagnose_step(prepared, index, result) for index, result in enumerate(results))
            try:
                for index, result in enumerate(results):
                    if result.returncode:
                        raise ValueError("native ABBA execution did not produce a complete receipt")
                    shapes = list(dict.fromkeys(point.shape_id for point in prepared.measurement_identities[index].points))
                    payloads.append(parse_native_abba(json.loads(result.stdout), self.configuration.repeats, shapes))
                verification = interpret_payloads(payloads, list(plan.shape_ids), self.configuration.repeats,
                    self.configuration.measurement_repetitions)
            except (ValueError, TypeError, KeyError, AttributeError) as error:
                verification = VerificationResult("ERROR", None, None, None, error=str(error),
                    failure=EvaluationFailure("incomplete", "native-abba-incomplete", "Native ABBA evidence is incomplete; candidate invalidity is not established"))
            return replace(verification, measurements=assessments,
                artifacts=tuple(MeasurementArtifact(f"receipt-{index:04d}.json", result.stdout.encode(), "measurement_batch")
                    for index, result in enumerate(results)))
        if request.mode in {"v0", "framework"}:
            if getattr(plan, "gateway", None) is not None:
                result = self._command_evaluator().interpret(plan.gateway, results).result
            else:
                from .gateway_measurement import prepare_dev_output
                result = replace(results[0], stdout=prepare_dev_output(request.workspace, results[0].stdout,
                    evaluator=True, private=False))
                if self.configuration.measurement_repetitions > 1:
                    from .aggregation import aggregate
                    from .transport_results import _with_workspace_performance_score
                    samples = [result_from_stdout(prepare_dev_output(request.workspace, item.stdout,
                        evaluator=True, private=False)) for item in results]
                    measured = aggregate(samples, "evaluate")
                    measured.pop("performance_objective", None)
                    measured = _with_workspace_performance_score(request.workspace, measured, baseline=plan.baseline)
                    result = replace(results[-1], transport_returncode=0 if all(item.returncode == 0 for item in results) else 1,
                        exit_code=None, stdout=RESULT_PREFIX + json.dumps(measured) + "\n")
            from .baseline import BaselinePreparation
            if getattr(plan, "gateway", None) is not None:
                BaselinePreparation.print_v0_evaluator_output(result)
            verification = self._baseline_result(request, result)
            return replace(verification, episode_evaluations=tuple(
                EpisodeEvaluationContent("run", json.dumps(run.result).encode())
                for run in verification.runs if run.result is not None),
                measurements=tuple(assessment for index, raw in enumerate(results)
                if (assessment := self.diagnose_step(prepared, index, raw)) is not None))
        payloads = []
        for index, result in enumerate(results):
            if result.returncode != 0:
                raise RuntimeError(
                    f"gateway ABBA command exited {result.returncode}: "
                    + (result.stdout + "\n" + result.stderr)[-3000:]
                )
            payloads.append(_payload_from_stdout(self._dev_stdout(request, result.stdout)))
        assessments = tuple(assessment for index, raw in enumerate(results)
            if (assessment := self.diagnose_step(prepared, index, raw)) is not None)
        try:
            payload = _merge_batch_payloads(
                payloads, list(plan.schedule), list(plan.shape_ids)
            )
        except (ValueError, KeyError, TypeError) as error:
            error.measurement_assessments = assessments
            raise
        result = interpret_verification_payload(
            payload,
            schedule=list(plan.schedule),
            repeats=max(1, self.configuration.repeats),
            artifact="result.json",
        )
        failure = result.failure
        if failure is None and any(item.validity == "unknown" for item in assessments):
            timeout = any(diagnostic.category in {"timeout", "deadline"}
                for item in assessments for diagnostic in item.diagnostics)
            failure = EvaluationFailure("timeout" if timeout else "incomplete",
                "abba-timeout" if timeout else "missing-measurement",
                "ABBA measurement remains incomplete; this does not establish candidate invalidity")
        return replace(result, artifacts=(self._artifact("result.json", payload, "evaluation"),),
            measurements=assessments, failure=failure)

    @staticmethod
    def _dev_stdout(request, stdout):
        from .gateway_measurement import prepare_dev_output
        return prepare_dev_output(request.workspace, stdout, evaluator=True, private=False)

    @staticmethod
    def _artifact(name, payload, role):
        content = (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
        return MeasurementArtifact(name, content, role)

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
            measurement_reader=self.configuration.measurement_reader,
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
        from .measurement import shape_latency_facts
        if observation.returncode != 0 and observation.primary_diagnostic is not None:
            primary, *secondary = observation.diagnostics
            details = {**primary.details, "failure_class": primary.category,
                       "secondary_diagnostics": [{"failure_class": item.category} for item in secondary]}
            category = details.get("failure_class", "protocol")
            explicit_repairable = details.get("candidate_repairable")
            repairable = (
                explicit_repairable
                if isinstance(explicit_repairable, bool)
                else category == "request"
                and not details.get("reference_rejected")
                and bool(
                    details.get("candidate_rejected")
                    or details.get("io_contract_rejected")
                )
            )
            secondary_classes = [
                str(item.get("failure_class") or "protocol")
                for item in details.get("secondary_diagnostics", [])
                if isinstance(item, dict)
            ]
            secondary_context = (
                "; subsequent diagnostics=" + ",".join(secondary_classes)
                if secondary_classes
                else ""
            )
            if details.get("metadata_invalid"):
                message = (
                    "invalid performance metadata before GPU submission: "
                    f"hardware={details.get('hardware', 'unknown')}; "
                    f"sid={details.get('shape_id', 'unknown')}; "
                    f"reason={details.get('metadata_reason', 'invalid')}"
                    + secondary_context
                )
                failure_code = "invalid-performance-baseline"
            else:
                message = (
                    "candidate request rejected; repair source-policy violations "
                    "and make Model.forward accept exactly the trusted input keys"
                    + secondary_context
                    if repairable
                    else f"gateway validation pending: failure_class={category}; "
                    "no correctness result was produced"
                    + secondary_context
                )
                failure_code = (
                    f"gateway-http-{details.get('http_status', 'unknown')}"
                )
            failure = EvaluationFailure(
                category,
                failure_code,
                message,
                candidate_repairable=repairable,
            )
            return VerificationResult("FAIL" if repairable else "ERROR", None, None, None,
                                      error=message, failure=failure)
        framework_transport_failed = (
            request.mode == "framework" and observation.returncode != 0
        )
        try:
            result = result_from_stdout(observation.stdout)
        except (RuntimeError, json.JSONDecodeError):
            if not framework_transport_failed:
                raise
            return VerificationResult(
                "FAIL",
                None,
                None,
                None,
                error=f"combined validation command failed (exit={observation.returncode})",
            )
        run = VerificationRun("candidate", 0, observation.returncode, result)
        evaluation_failure = result.get("evaluation_failure")
        if isinstance(evaluation_failure, dict):
            category = str(evaluation_failure.get("category") or "infrastructure")
            code = str(evaluation_failure.get("code") or "evaluation-unavailable")
            message = str(
                evaluation_failure.get("message") or "evaluation unavailable"
            )
            failure = EvaluationFailure(category, code, message)
            return VerificationResult(
                "ERROR",
                None,
                None,
                None,
                runs=[run],
                error=message,
                failure=failure,
            )
        if framework_transport_failed:
            return VerificationResult(
                "FAIL",
                None,
                None,
                None,
                runs=[run],
                error=f"combined validation command failed (exit={observation.returncode})",
            )
        if request.mode == "v0":
            error = (
                ""
                if observation.returncode == 0 and result.get("all_pass")
                else "sandbox V0 baseline failed correctness/performance validation"
            )
        else:
            error = framework_result_problem(result, request.workspace, measurement_reader=self.configuration.measurement_reader)
        return VerificationResult(
            "FAIL" if error else "PASS",
            result.get("latency_us_geomean"),
            None,
            None,
            runs=[run],
            error=error,
            candidate_latency_us_by_shape=shape_latency_facts(result.get("latency_us_by_shape")),
        )


def _framework_actionable_repair_hint(result: dict) -> str:
    """Return a bounded deterministic hint for a known candidate-runtime defect."""
    diagnostics = result.get("actionable_diagnostics")
    if not isinstance(diagnostics, list):
        return ""
    for row in diagnostics[:8]:
        if not isinstance(row, dict):
            continue
        message = row.get("message")
        if (
            row.get("stage") == "candidate_runtime"
            and isinstance(message, str)
            and "Argument 'kernelParams' is not a valid type" in message
        ):
            return (
                "CUDA launch ABI repair: replace version-sensitive tuple packing with "
                "typed ctypes argument storage plus a native ctypes void** kernelParams "
                "array containing each argument address; keep both alive through "
                "cuLaunchKernel and pass extra=0"
            )
    return ""


def framework_result_problem(result, workspace, *, measurement_reader=None):

    if not result.get("all_pass"):
        problem = "combined multi-seed correctness validation failed"
        repair_hint = _framework_actionable_repair_hint(result)
        return f"{problem}; {repair_hint}" if repair_hint else problem
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
    fact = measurement_reader.baseline(workspace) if measurement_reader is not None else None
    baseline = set(fact.result.get("latency_us_by_shape") or {}) if fact is not None else set()
    measured = set(result.get("latency_us_by_shape") or {})
    if baseline and measured != baseline:
        return (
            "latency_us_by_shape does not cover the same workloads as v0 "
            + f"(missing {sorted(baseline - measured)}, unexpected {sorted(measured - baseline)})"
        )
    return ""
