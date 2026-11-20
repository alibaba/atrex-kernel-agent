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
    MeasurementArtifact,
    StepEvaluation,
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


class Evaluator:
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
            measurement_repetitions=self.configuration.measurement_repetitions,
            measurement_defaults=self.configuration.measurement_defaults)

    def prepare(self, request: EvaluationRequest) -> PreparedEvaluation:
        prepared = self._prepare(request)
        if request.cancel_event is not None:
            prepared = replace(prepared, steps=tuple(replace(step, cancel_event=request.cancel_event)
                                                     for step in prepared.steps))
        if getattr(prepared.context, "typed_numerical", False):
            return prepared
        if getattr(prepared.context, "native", False):
            return prepared
        if self.evaluation_input is not None and request.mode in {"abba", "numerical"}:
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
            return prepare_numerical(request, replace(config, timeout=request.numerical_probe.timeout),
                                     self.evaluation_input, output_root=VERIFY_DIR)
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
        if getattr(plan, "typed_numerical", False):
            artifacts = (plan.source_artifact,) if index == 0 else ()
            return StepEvaluation(artifacts, assessment, workspace_files=result.files)
        if getattr(plan, "native", False):
            source = (plan.candidate_source,) if index == 0 and plan.candidate_source is not None else ()
            return StepEvaluation((MeasurementArtifact(f"receipt-{index:04d}.json", result.stdout.encode(), "measurement_batch"), *source),
                assessment, workspace_files=result.files)
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
        if getattr(prepared.context, "typed_numerical", False):
            from .typed_numerical import assess
            return assess(prepared.context, index, result)
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
        if getattr(prepared.context, "typed_numerical", False):
            from .typed_numerical import interpret
            return interpret(prepared, results)
        if request.mode == "numerical":
            from .numerical_preparation import interpret_numerical
            return interpret_numerical(prepared, results)
        if plan.error:
            return VerificationResult("ERROR", None, None, None, error=plan.error)
        if len(results) != len(prepared.steps) and not (prepared.fallback_steps and len(results) == 1):
            raise ValueError("evaluation requires one result per planned step")
        if getattr(plan, "native", False):
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
