"""Prepare Legacy measurement commands and interpret raw execution receipts."""
from dataclasses import replace
import json
from pathlib import Path
from aka.contracts.execution import CommandSpec, ExecutionFile, UnsupportedExecution
from aka.contracts.evaluation import PreparedEvaluation, StepEvaluation, MeasurementArtifact, CommandEvaluation, VerificationResult, EpisodeEvaluationContent
from aka.execution.command_syntax import requested_operation, is_test_kernel_command
from .transport_results import TEST_RESULT_PREFIX
PROFILE_RESULT_PREFIX = "[sandbox] PROFILE_JSON="
from .gateway_request import prepare_request, _candidate_languages, _json_object
from .gateway_policy import _typed_workspace_limitation
from .gateway_measurement import GatewayMeasurement, prepare_dev_output, with_measurement_conditions
from .metadata import distributed_evaluation_world_size
from .command_inputs import (validate_command, workload_environment,
    nondefault_profile, validate_prepared_profile, PROFILE_ENVIRONMENT_KEYS)


class GatewayCommandEvaluator:
    def __init__(self, *, measurement_reader=None, execution_capabilities=None,
                 evaluation_input=None, shape_batch_size=4):
        self.measurement_reader = measurement_reader
        self.execution_capabilities = execution_capabilities
        self.evaluation_input = evaluation_input
        if type(shape_batch_size) is not int or shape_batch_size < 1:
            raise ValueError("shape batch size must be positive")
        self.shape_batch_size = shape_batch_size


    def prepare(self, command):
        validate_command(command)
        if command.operation == "preflight" or command.submission is not None:
            identities = ()
            measurement = None
            kind = command.operation
            private = self.evaluation_input is not None and self.evaluation_input.private_reference_dir is not None
            if command.submission is not None:
                prepared = json.loads(command.submission)
                if not isinstance(prepared, dict) or prepared.get("schema_version") != 1:
                    raise ValueError("unsupported prepared gateway request")
                kind = prepared.get("kind")
                if kind not in {"run", "profile", "check", "disassemble"} or command.operation not in {"auto", kind}:
                    raise ValueError("prepared request differs from the command operation")
                request = prepared.get("request")
                if not isinstance(request, dict):
                    raise ValueError("prepared request must be an object")
                references = prepared.get("reference_files", {})
                if not isinstance(references, dict) or any(not isinstance(name, str)
                        or name in {"", ".", ".."} or Path(name).name != name or not isinstance(content, str)
                        for name, content in references.items()):
                    raise ValueError("prepared reference files require safe filenames and text")
                captured_environment = request.get("env_vars", {})
                if (not isinstance(captured_environment, dict) or any(not isinstance(key, str)
                        or not isinstance(value, str) for key, value in captured_environment.items())):
                    raise ValueError("prepared workload environment must be an object")
                from .gateway_request import _parse_env_items
                _parse_env_items(f"{key}={value}" for key, value in captured_environment.items())
                if any(captured_environment.get(key) != value for key, value in workload_environment(command).items()):
                    raise ValueError("workload environment differs from the frozen prepared request")
                if kind in {"run", "profile"}:
                    if (not isinstance(request.get("candidate"), str) or not isinstance(request.get("reference"), dict)
                            or not isinstance(request["reference"].get("shapes"), dict) or not request["reference"]["shapes"]):
                        raise ValueError("prepared measurement requires candidate source and shapes")
                    from .measurement import gateway_identity
                    identities = (gateway_identity(request, references, kind),)
                if kind == "profile":
                    if private:
                        raise UnsupportedExecution("generalized profiles require the portable profile runner")
                    validate_prepared_profile(request, command.profile_options)
                    if PROFILE_ENVIRONMENT_KEYS.intersection(workload_environment(command)):
                        raise UnsupportedExecution("PROFILE_* controls require the portable profile runner")
                    measurement = GatewayMeasurement(request, workspace=command.workspace,
                        hardware=command.hardware, private=private, batch_size=1, profiling=True)
                    if not measurement.shape_ids:
                        raise ValueError("profile measurement requires at least one shape")
            step = replace(command, profile_options=None) if command.profile_options is not None else command
            return PreparedEvaluation((step,), 1, {"command": command, "measurement": measurement,
                "kind": kind, "private": private, "evaluator": False},
                measurement_identities=identities)
        baseline = (self.measurement_reader.baseline(command.workspace)
                    if self.measurement_reader is not None else None)
        environment = command.environment
        batch_size = int(environment.get("ATREX_EVAL_SHAPE_BATCH_SIZE", self.shape_batch_size))
        if batch_size < 1:
            raise ValueError("shape batch size must be positive")
        explicit_environment = workload_environment(command)
        kind = requested_operation(command.operation, list(command.command))
        evaluator = is_test_kernel_command(list(command.command))
        private = self.evaluation_input is not None and self.evaluation_input.private_reference_dir is not None
        options = command.profile_options
        profile_level = (options.level or "sol") if options is not None else "sol"
        context = {"command": command, "kind": kind, "private": private,
                   "evaluator": evaluator, "measurement": None, "baseline": baseline}
        if command.operation == "preflight" or not (evaluator or kind == "profile"):
            context["kind"] = "dev"
            command = replace(command, operation="run") if command.operation == "auto" else command
            return PreparedEvaluation((command,), 1, context)
        from .private_inputs import selected_content, portable_inputs
        content = selected_content(self.evaluation_input)
        reference_content = {name: item.content for name, item in content.items()}
        reference = self.evaluation_input.directory if self.evaluation_input is not None else command.workspace
        if private:
            if "shapes.json" not in content or "metadata.json" not in content:
                raise ValueError("private Task evaluation requires frozen shapes.json and metadata.json")
            if "agent_problem.json" in content and content["agent_problem.json"].content != (command.workspace / "agent_problem.json").read_bytes():
                raise ValueError("workspace public contract differs from the selected Task evaluation input")
        metadata = json.loads(content["metadata.json"].content) if "metadata.json" in content else _json_object(reference / "metadata.json")
        metadata_gpus = distributed_evaluation_world_size(metadata)
        if command.num_gpus != 1 and metadata_gpus != 1 and command.num_gpus != metadata_gpus:
            raise ValueError("command GPU count differs from the Task metadata")
        num_gpus = max(command.num_gpus, metadata_gpus)
        context["num_gpus"] = num_gpus
        command = replace(command, num_gpus=num_gpus)
        context["command"] = command
        solution = _json_object(command.workspace / "solution.json") or {}
        candidate_path = command.workspace / "kernel.py"
        candidate = candidate_path.read_text(encoding="utf-8") if candidate_path.is_file() else ""
        requirements = list(command.requirements)
        if self.evaluation_input is not None and self.evaluation_input.worker_requirements is not None:
            item = self.evaluation_input.worker_requirements
            text = item.source.read_text(encoding="utf-8") if item.source is not None else item.text
            from aka.execution.environment_query import select_requirements
            frozen = tuple(select_requirements(text, _candidate_languages(solution, candidate_sources=(candidate,))))
            if requirements and tuple(requirements) != frozen:
                raise ValueError("command requirements differ from the frozen Task requirements")
            requirements = list(frozen)
        elif not requirements:
            requirements_path = environment.get("ATREX_SANDBOX_REQUIREMENTS_FILE", "").strip()
            if requirements_path:
                from aka.execution.environment_query import select_requirements
                text = Path(requirements_path).read_text(encoding="utf-8")
                if len(text.encode("utf-8")) > 128 * 1024:
                    raise ValueError("runtime requirements file exceeds 128 KiB")
                requirements = list(select_requirements(text, _candidate_languages(solution, candidate_sources=(candidate,))))
        command = replace(command, requirements=tuple(requirements))
        context["command"] = command
        capabilities = self.execution_capabilities
        portable_only = capabilities is not None and not capabilities.prepared_requests
        profile_controls = kind == "profile" and PROFILE_ENVIRONMENT_KEYS.intersection(explicit_environment)
        limitation = ("selected executor uses the portable runner" if portable_only else
            "PROFILE_* controls require the portable profile runner" if profile_controls else
            "custom dev options" if command.input_files else
            _typed_workspace_limitation(command.workspace, list(command.command), kind, reference_dir=reference, reference_content=reference_content, private=private))
        if kind == "dev" or limitation:
            if kind == "profile" and nondefault_profile(command.profile_options):
                raise UnsupportedExecution("requested typed Profile options cannot use the portable route: " + str(limitation))
            if kind == "profile" and num_gpus > 1:
                raise ValueError("distributed profile requires the typed route")
            from .measurement import portable_profile_identity
            identities = (portable_profile_identity(command, reference),) if kind == "profile" else ()
            return PreparedEvaluation((portable_inputs(replace(command, operation="dev", profile_options=None), self.evaluation_input, content, evaluator=evaluator, kind=kind, capabilities=capabilities),), 1, context,
                                      measurement_identities=identities)
        if kind == "profile" and profile_level == "deep" and not (options.kernel_regex if options is not None else None):
            raise ValueError("--profile-level deep requires --kernel-regex")
        try:
            request = prepare_request(command.workspace, command.hardware, command.timeout, [f"{key}={value}" for key, value in explicit_environment.items()],
                list(command.command), kind, reference_dir=reference, reference_content=reference_content, requirements=requirements,
                runtime_version=environment.get("ATREX_SANDBOX_ATREX_BENCH_VERSION", "").strip(),
                profiler=options.profiler if options is not None else None, profile_level=profile_level,
                counters=options.counters if options is not None else (),
                kernel_regex=options.kernel_regex if options is not None else None, top_kernels=options.top_kernels if options is not None else None)
            if num_gpus > 1:
                request["spec"]["num_gpus"] = num_gpus
        except (OSError, UnicodeDecodeError, ValueError) as error:
            # A malformed performance baseline deliberately propagates; it must not submit GPU work.
            from aka.task.problem.inputs import PerformanceBaselineError
            if isinstance(error, PerformanceBaselineError):
                raise
            if nondefault_profile(command.profile_options):
                raise
            return PreparedEvaluation((portable_inputs(replace(command, operation="dev", profile_options=None), self.evaluation_input, content, evaluator=evaluator, kind=kind, capabilities=capabilities),), 1, context)
        measurement = GatewayMeasurement(request, workspace=command.workspace, hardware=command.hardware,
            private=private, batch_size=batch_size, profiling=kind == "profile", baseline=baseline)
        if not measurement.batches:
            raise ValueError("measurement requires at least one shape")
        context["measurement"] = measurement
        steps = []
        identities = []
        for index in range(len(measurement.batches)):
            reference_files = {name: value.decode("utf-8") for name, value in measurement.reference_content(index).items()}
            from .measurement import gateway_identity
            identities.append(gateway_identity(measurement.request_for(index), reference_files, kind))
            submission = json.dumps({"schema_version": 1, "kind": kind,
                "request": measurement.request_for(index), "reference_files": reference_files},
                ensure_ascii=False, allow_nan=False).encode("utf-8")
            steps.append(replace(command, operation=kind, submission=submission, artifacts=(), profile_options=None))
        return PreparedEvaluation(tuple(steps), min(4, len(steps)), context,
            fallback_steps=() if nondefault_profile(command.profile_options) or (self.evaluation_input is not None and (capabilities is None or not capabilities.prepared_inputs)) else
                (portable_inputs(replace(command, operation="dev", submission=None, profile_options=None), self.evaluation_input, content, evaluator=evaluator, kind=kind, capabilities=capabilities),),
            measurement_identities=tuple(identities))

    def fallback(self, prepared, results):
        """Only a settled unsupported route can authorize the prepared dev alternative."""
        if not prepared.fallback_steps or not results:
            return ()
        if any(result.facts.disposition not in {"terminal", "not_submitted"} or result.facts.cancellation_requested or result.transport_status in {"cancelled", "timed_out", "interrupted"}
               or any(item.category in {"cancelled", "authentication"} for item in result.diagnostics)
               for result in results):
            return ()
        if any(result.returncode and result.facts.disposition in {"terminal", "not_submitted"}
               and any(item.category == "capability" and item.code == "typed-source-unsupported" for item in result.diagnostics)
               for result in results):
            return prepared.fallback_steps
        return ()

    def observe_step(self, prepared, index, result):
        assessment = self.diagnose_step(prepared, index, result)
        if prepared.context["measurement"] is None:
            return StepEvaluation(assessment=assessment, workspace_files=result.files)
        return StepEvaluation((MeasurementArtifact(f"receipt-{index:04d}.json", result.stdout.encode(), "measurement_batch"),), assessment,
            workspace_files=() if prepared.context["kind"] == "profile" and index < len(prepared.steps) else result.files)

    def diagnose_step(self, prepared, index, result):
        """Pure per-batch explanation; collecting an original Job is the host's action."""
        if not prepared.measurement_identities or index >= len(prepared.measurement_identities):
            return None
        from .measurement import assess_payload
        from .transport_results import _optimizer_result_from_eval
        identity = prepared.measurement_identities[index]
        payload = None
        if identity.method != "atrex-bench/profile":
            try:
                context = prepared.context
                measurement = context["measurement"]
                request = measurement.request_for(index) if measurement is not None else json.loads(context["command"].submission)["request"]
                job = _job_response(result.stdout)
                if job and isinstance(job.get("result"), dict):
                    measured = _optimizer_result_from_eval(job["result"],
                        [point.shape_id for point in identity.points if point.shape_id is not None],
                        request["reference"].get("metadata"),
                        require_performance=request.get("mode") != "correctness_only", hardware=context["command"].hardware)
                    payload = {"runs": [{"revision": "candidate", "repeat": 0,
                        "exit_code": result.returncode, "result": measured}]}
            except (TypeError, ValueError, KeyError, json.JSONDecodeError):
                pass
        return assess_payload(identity, result, payload)

    def interpret(self, prepared, results):
        context = prepared.context
        command = context["command"]
        measurement = context["measurement"]
        if measurement is None or (len(results) == 1 and any(line.startswith(TEST_RESULT_PREFIX) for line in results[0].stdout.splitlines())):
            result = results[0]
            stdout = prepare_dev_output(command.workspace, result.stdout,
                evaluator=context["evaluator"], private=context["private"], baseline=context.get("baseline"))
            return self._command_evaluation(replace(result, stdout=stdout), context, results)
        jobs = []
        for result in results:
            job = _job_response(result.stdout)
            if result.returncode or job is None or job.get("status") != "succeeded" or not isinstance(job.get("result"), dict):
                result = replace(result, transport_returncode=result.returncode or 2)
                if context["private"]:
                    return self._command_evaluation(replace(result, stdout="", stderr="[sandbox] generalized measurement unavailable; details withheld\n"), context, results)
                return self._command_evaluation(result, context, results)
            jobs.append(job)
        files = []
        if context["kind"] == "run":
            payload = measurement.interpret(jobs)
            stdout = TEST_RESULT_PREFIX + json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n"
            stdout = with_measurement_conditions(stdout, gateway_warmup_iters=command.environment.get("AKA_GATEWAY_WARMUP_ITERS", "5"))
            code = 0 if payload["all_pass"] else 1
        else:
            stdout = PROFILE_RESULT_PREFIX + json.dumps(jobs[0]["result"], ensure_ascii=False) + "\n"
            code = 0
            for relative in command.artifacts:
                path = Path(relative)
                if path.is_absolute() or ".." in path.parts:
                    raise ValueError("profile destination must be workspace-relative")
                if path.parts and path.parts[0] == "profiles":
                    files.append(ExecutionFile(str(path / "gateway_profile.json"),
                        (json.dumps(jobs[0], indent=2, ensure_ascii=False) + "\n").encode()))
                    for output in results[0].files:
                        files.append(replace(output, path=str(path / output.path)))
            limit = command.output_options.max_file_bytes if command.output_options is not None else None
            if limit is not None:
                skipped = [item.path for item in files if len(item.content) > limit]
                files = [item for item in files if len(item.content) <= limit]
            else:
                skipped = []
        stderr = [item.stderr.rstrip() for item in results if item.stderr]
        if context["kind"] == "profile" and skipped:
            stderr.append("[sandbox] outputs skipped: " + ", ".join(skipped))
        result = replace(results[-1], transport_returncode=code, exit_code=None, stdout=stdout,
            stderr="\n".join(stderr),
            files=tuple(files), file_receipts=(),
            attempts=tuple(attempt for item in results for attempt in item.attempts))

        return self._command_evaluation(result, context, results)

    @staticmethod
    def _command_evaluation(result, context, results):
        from .evaluator import result_from_stdout
        try:
            payload = result_from_stdout(result.stdout)
        except (RuntimeError, json.JSONDecodeError):
            return CommandEvaluation(result)
        verification = VerificationResult("PASS" if payload.get("all_pass") else "FAIL",
            payload.get("latency_us_geomean"), None, None)
        content = json.dumps(payload).encode()
        retained = ()
        if context["evaluator"]:
            # Physical Job identity is execution-owned; no stdout Job discovery.
            job_id = ",".join(reference.identifier for item in results for reference in item.job_references) or None
            retained = (EpisodeEvaluationContent(context["kind"], content, job_id),)
        return CommandEvaluation(result, verification,
            (MeasurementArtifact("evaluation.json", content, "evaluation"),), retained)


def _job_response(stdout):
    """Interpret the method's complete typed JSON receipt without transport policy."""
    try:
        result = json.loads(stdout)
    except (TypeError, json.JSONDecodeError):
        return None
    return result if isinstance(result, dict) and isinstance(result.get("job_id"), str) else None
