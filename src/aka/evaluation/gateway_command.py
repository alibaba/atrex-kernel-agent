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
                 evaluation_input=None, shape_batch_size=4, measurement_repetitions=1):
        self.measurement_reader = measurement_reader
        self.execution_capabilities = execution_capabilities
        self.evaluation_input = evaluation_input
        if type(shape_batch_size) is not int or shape_batch_size < 1:
            raise ValueError("shape batch size must be positive")
        self.shape_batch_size = shape_batch_size
        if type(measurement_repetitions) is not int or measurement_repetitions not in {1, 3}:
            raise ValueError("measurement repetitions must be 1 or 3")
        self.measurement_repetitions = measurement_repetitions


    def prepare(self, command):
        from .diagnostic_preparation import matches, prepare
        if matches(command):
            return prepare(command)
        prepared = self._prepare(command)
        context = prepared.context
        kind = context["kind"]
        request = (json.loads(command.submission)["request"] if command.submission is not None else
                   context["measurement"].request if context["measurement"] is not None else {})
        count = self.measurement_repetitions if kind in {"run", "same_allocation_abba"} and request.get("mode") != "correctness_only" else 1
        context["repetitions"] = count
        context["batch_count"] = len(prepared.steps)
        from aka.contracts.measurement import MeasurementPoint
        method_inputs = {}
        expected_shapes = ()
        if self.evaluation_input is not None:
            expected_shapes = self.evaluation_input.case_ids
            import hashlib
            for name, item in context.get("task_content", {}).items():
                method_inputs[name] = hashlib.sha256(item.content).hexdigest()
                if name == "shapes.json" and not expected_shapes:
                    expected_shapes = tuple(json.loads(item.content))
            from .resources import evaluator_harness
            harness = evaluator_harness(self.evaluation_input)
            if harness is not None and harness.path not in method_inputs:
                content = harness.source.read_bytes() if harness.source is not None else harness.text.encode()
                method_inputs[harness.path] = hashlib.sha256(content).hexdigest()
        explicit_subset = any(token == "--shape-id" or token.startswith("--shape-id=") for token in command.command)
        full_scope = (kind == "run" and bool(expected_shapes) and context.get("task_matches", False)
                      and not explicit_subset and not command.input_files
                      and request.get("mode") in {"full", "correctness_only"}
                      and set(request.get("reference", {}).get("shapes", {})) == set(expected_shapes))
        if request.get("candidate") is not None:
            from .source_archives import request_sources
            context["source_parameters"], context["candidate_artifact"] = request_sources(request, command.source_archives)
        from .measurement import gateway_identity
        frozen_identities = []
        for index, identity in enumerate(prepared.measurement_identities):
            if identity.method.startswith("atrex-bench/"):
                step_request = json.loads(prepared.steps[index].submission)
                identity = gateway_identity(step_request["request"], step_request.get("reference_files", {}),
                    kind if kind in {"profile", "check", "disassemble"} else "run", full_request=request, method_inputs=method_inputs,
                    workload_scope="full-task" if full_scope else "subset-or-custom" if expected_shapes else "unknown",
                    source_parameters=context.get("source_parameters"))
            if identity.method == "legacy-eval/command" and self.evaluation_input is not None and self.evaluation_input.adapter == "legacy-native-v1":
                from .measurement import _digest
                step = prepared.steps[index]
                closure = step.prepared_inputs
                frozen_inputs = {item.path: hashlib.sha256(item.content).hexdigest()
                    for item in (*closure.files, *closure.runtime_files)} if closure is not None else {}
                parameters = {**identity.parameters, "command": list(step.command),
                    "evaluation_method_inputs": method_inputs}
                identity = replace(identity, parameters=parameters,
                    input_digest=_digest({"inputs": frozen_inputs, "parameters": parameters}))
            frozen_identities.append(identity)
        identities = tuple(replace(identity,
            points=tuple(MeasurementPoint(point.side, point.revision, point.repeat + outer * (
                int(identity.parameters.get("abba_sdk_blocks", 0)) * 2 or 1), point.shape_id) for point in identity.points),
            parameters={**identity.parameters, "measurement_repetitions": count, "measurement_repetition": outer})
            for outer in range(count) for identity in frozen_identities)
        return replace(prepared, steps=prepared.steps * count, measurement_identities=identities,
                       fallback_steps=() if count > 1 else prepared.fallback_steps)

    def _prepare(self, command):
        validate_command(command)
        if command.operation == "preflight" or command.submission is not None:
            identities = ()
            measurement = None
            kind = command.operation
            private = self.evaluation_input is not None and self.evaluation_input.private_reference_dir is not None
            from .private_inputs import selected_content
            content = {}
            task_matches = True
            if command.submission is not None:
                prepared = json.loads(command.submission)
                if not isinstance(prepared, dict) or prepared.get("schema_version") != 1:
                    raise ValueError("unsupported prepared gateway request")
                kind = prepared.get("kind")
                native = command.operation == "same_allocation_abba"
                if kind not in {"run", "profile", "check", "disassemble"} or (command.operation not in {"auto", kind} and not (native and kind == "run")):
                    raise ValueError("prepared request differs from the command operation")
                request = prepared.get("request")
                if not isinstance(request, dict):
                    raise ValueError("prepared request must be an object")
                if kind in {"run", "profile"}:
                    content = selected_content(self.evaluation_input)
                # An explicit wire request is a full Task measurement only when
                # its trusted content equals the selected frozen input bundle.
                reference = request.get("reference", {})
                for name, field in (("input.py", "input_py"), ("reference.py", "reference_py"),
                                    ("shapes.json", "shapes"), ("metadata.json", "metadata"), ("roofline.json", "roofline")):
                    item = content.get(name)
                    if item is not None:
                        expected = json.loads(item.content) if name.endswith(".json") else item.content.decode()
                        task_matches = task_matches and isinstance(reference, dict) and reference.get(field) == expected
                if request.get("abba") and not native:
                    raise ValueError("native ABBA requires the same_allocation_abba operation")
                if native:
                    abba = request.get("abba")
                    if (not isinstance(abba, dict) or not isinstance(abba.get("baseline"), (str, dict)) or not abba["baseline"]
                            or type(abba.get("repeats")) is not int or abba["repeats"] < 1):
                        raise ValueError("native ABBA requires baseline source and SDK block count")
                    if self.execution_capabilities is None:
                        raise UnsupportedExecution("native ABBA requires bound execution capabilities")
                    self.execution_capabilities.require(command)
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
                    if (not isinstance(request.get("candidate"), (str, dict)) or not isinstance(request.get("reference"), dict)
                            or not isinstance(request["reference"].get("shapes"), dict) or not request["reference"]["shapes"]):
                        raise ValueError("prepared measurement requires candidate source and shapes")
                    from .measurement import gateway_identity
                    if native and isinstance(request["candidate"], str) and not request["candidate"].strip():
                        raise ValueError("native ABBA requires nonempty candidate source")
                if kind in {"run", "profile", "check", "disassemble"} and request.get("candidate") is not None:
                    from .source_archives import request_sources
                    source_facts, _ = request_sources(request, command.source_archives)
                    from .measurement import gateway_identity
                    identities = (gateway_identity(request, references, kind, source_parameters=source_facts),)
                if kind == "profile":
                    native_manifest = (isinstance(request.get("candidate"), dict)
                                       and request["candidate"].get("entry_point", {}).get("type") == "cuda")
                    if private and not native_manifest:
                        raise UnsupportedExecution("generalized profiles require the portable profile runner")
                    validate_prepared_profile(request, command.profile_options)
                    if PROFILE_ENVIRONMENT_KEYS.intersection(workload_environment(command)):
                        raise UnsupportedExecution("PROFILE_* controls require the portable profile runner")
                    measurement = GatewayMeasurement(request, workspace=command.workspace,
                        hardware=command.hardware, private=private, batch_size=1, profiling=True)
                    if not measurement.shape_ids:
                        raise ValueError("profile measurement requires at least one shape")
                elif kind == "run":
                    measurement = GatewayMeasurement(request, workspace=command.workspace,
                        hardware=command.hardware, private=private,
                        batch_size=self.shape_batch_size)
            step = replace(command, profile_options=None) if command.profile_options is not None else command
            steps = (step,)
            if measurement is not None and kind == "run":
                from .measurement import gateway_identity
                steps = tuple(replace(step, submission=json.dumps({**prepared, "request": measurement.request_for(index),
                    "reference_files": {name: item.decode() for name, item in measurement.reference_content(index).items()}}, allow_nan=False).encode())
                    for index in range(len(measurement.batches)))
                identities = tuple(gateway_identity(measurement.request_for(index),
                    {name: item.decode() for name, item in measurement.reference_content(index).items()}, kind,
                    source_parameters=source_facts) for index in range(len(measurement.batches)))
            return PreparedEvaluation(steps, min(4, len(steps)), {"command": command, "measurement": measurement,
                "kind": "same_allocation_abba" if command.operation == "same_allocation_abba" else kind, "private": private, "evaluator": False,
                "task_content": content, "task_matches": task_matches},
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
        if evaluator and self.evaluation_input is not None and self.evaluation_input.adapter == "legacy-native-v1":
            from .resources import validate_harness
            validate_harness(command.workspace, self.evaluation_input)
        private = self.evaluation_input is not None and self.evaluation_input.private_reference_dir is not None
        options = command.profile_options
        profile_level = (options.level or "sol") if options is not None else "sol"
        context = {"command": command, "kind": kind, "private": private,
                   "evaluator": evaluator, "measurement": None, "baseline": baseline}
        if command.operation == "preflight" or not (evaluator or kind == "profile"):
            context["kind"] = "dev"
            command = replace(command, operation="run") if command.operation == "auto" else command
            return PreparedEvaluation((command,), 1, context)
        if self.execution_capabilities is None:
            raise UnsupportedExecution("measurement preparation requires bound execution capabilities")
        from .private_inputs import selected_content, portable_inputs
        content = selected_content(self.evaluation_input)
        context["task_content"] = content
        context["task_matches"] = True
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
            "closed request inputs require the portable runner" if command.prepared_inputs is not None else
            "PROFILE_* controls require the portable profile runner" if profile_controls else
            "custom dev options" if command.input_files else
            _typed_workspace_limitation(command.workspace, list(command.command), kind, reference_dir=reference, reference_content=reference_content, private=private))
        if kind == "dev" or limitation:
            if kind == "profile" and nondefault_profile(command.profile_options):
                raise UnsupportedExecution("requested typed Profile options cannot use the portable route: " + str(limitation))
            if kind == "profile" and num_gpus > 1:
                raise ValueError("distributed profile requires the typed route")
            from .measurement import portable_profile_identity
            frozen_files = ()
            identities = ()
            if kind in {"run", "profile"} and candidate:
                from aka.contracts.workspace import RevisionContent
                from .source_archives import source_artifact, source_parameters
                source = RevisionContent({"kernel.py": candidate.encode()})
                source_facts = source_parameters(source)
                context["candidate_artifact"], _ = source_artifact(source)
                context["source_parameters"] = source_facts
                if kind != "profile":
                    from aka.contracts.measurement import MeasurementIdentity, MeasurementPoint
                    from .measurement import _digest
                    sol = self.evaluation_input is not None and self.evaluation_input.adapter == "legacy-sol-v1"
                    shapes = self.evaluation_input.case_ids if self.evaluation_input is not None else ()
                    sol_facts = {}
                    if sol:
                        from .sol_evidence import trusted_inputs, input_identity, workload_ids
                        content = trusted_inputs(self.evaluation_input)
                        shapes = workload_ids(content)
                        context["sol_workload_ids"] = shapes
                        from aka.contracts.execution import ExecutionFile
                        from .portable_selection import public_inputs
                        frozen_files = public_inputs(command, evaluator=True,
                            prepared_files=(ExecutionFile("kernel.py", candidate.encode()),))
                        frozen = {item.path: item for item in frozen_files}
                        if command.input_files:
                            raise ValueError("SOL evaluation does not accept explicit workspace overrides")
                        from .sol_evidence import single_source_solution
                        single_source = single_source_solution(frozen)
                        sol_facts = {"sol_input_binding": "closed-task/1", "sol_input_digest": input_identity(content),
                            "sol_candidate_inputs_digest": input_identity({name: item for name, item in frozen.items()
                                if name not in content})}
                    parameters = {"mode": "full", "workload_scope": "full-task" if sol and tuple(command.command) == ("python3", "test_kernel.py") else "unknown",
                        "hardware": command.hardware, "command": list(command.command),
                        "environment_digest": _digest(dict(command.environment)), **source_facts, **sol_facts}
                    if sol and not single_source:
                        parameters["workload_scope"] = "subset-or-custom"
                    identity = MeasurementIdentity("legacy-sol/eval" if sol else "legacy-eval/command",
                        tuple(MeasurementPoint("candidate", "sha256:" + source_facts["candidate_source_sha256"], 0, shape) for shape in shapes or (None,)),
                        _digest({"task": {name: item.content.hex() for name, item in content.items()}, "parameters": parameters}),
                        parameters, tuple(command.requirements), "explicit-requirements" if command.requirements else "unknown")
                    identities = (identity,)
            step = portable_inputs(replace(command, operation="dev", profile_options=None),
                self.evaluation_input, content, evaluator=evaluator, kind=kind,
                capabilities=capabilities, prepared_files=frozen_files)
            if kind == "profile":
                identity = portable_profile_identity(step, reference)
                identities = (replace(identity, parameters={**identity.parameters,
                    **context.get("source_parameters", {})}),)
            return PreparedEvaluation((step,), 1, context,
                                      measurement_identities=identities)
        if kind == "profile" and profile_level == "deep" and not (options.kernel_regex if options is not None else None):
            raise ValueError("--profile-level deep requires --kernel-regex")
        try:
            request = prepare_request(command.workspace, command.hardware, command.timeout, [f"{key}={value}" for key, value in explicit_environment.items()],
                list(command.command), kind, reference_dir=reference, reference_content=reference_content, requirements=requirements,
                runtime_version=environment.get("ATREX_SANDBOX_ATREX_BENCH_VERSION", "").strip(), measurement_environment=environment,
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
        if "diagnostic" in prepared.context:
            from .diagnostic_preparation import observe
            return observe(prepared, index, result)
        assessment = self.diagnose_step(prepared, index, result)
        source = ((prepared.context["candidate_artifact"],)
                  if index == 0 and "candidate_artifact" in prepared.context else ())
        if prepared.context["measurement"] is None and prepared.context["kind"] != "same_allocation_abba":
            return StepEvaluation(source, assessment=assessment, workspace_files=result.files)
        return StepEvaluation((MeasurementArtifact(f"receipt-{index:04d}.json", result.stdout.encode(), "measurement_batch"), *source), assessment,
            workspace_files=() if prepared.context["kind"] == "profile" and index < len(prepared.steps) else result.files)

    def diagnose_step(self, prepared, index, result):
        """Pure per-batch explanation; collecting an original Job is the host's action."""
        if "diagnostic" in prepared.context:
            from .diagnostic_preparation import diagnose
            return diagnose(prepared, index, result)
        if not prepared.measurement_identities or index >= len(prepared.measurement_identities):
            return None
        from .measurement import assess_payload
        from .transport_results import _optimizer_result_from_eval
        identity = prepared.measurement_identities[index]
        if identity.method == "legacy-profile/1":
            assessment = assess_payload(identity, result)
            if result.returncode:
                from aka.contracts.measurement import MeasurementDiagnostic
                assessment = replace(assessment, diagnostics=(*assessment.diagnostics,
                    MeasurementDiagnostic("execution", "profile-command-failed",
                        basis=("public-command-exit",))))
            return assessment
        payload = None
        if identity.method != "atrex-bench/profile":
            try:
                context = prepared.context
                measurement = context["measurement"]
                if measurement is None and context["command"].submission is None:
                    from .evaluator import result_from_stdout
                    measured = result_from_stdout(result.stdout)
                    if identity.method == "legacy-sol/eval":
                        from .sol_evidence import validate_result
                        validate_result(measured, context["sol_workload_ids"])
                    payload = {"runs": [{"revision": "candidate", "repeat": identity.points[0].repeat,
                        "exit_code": result.returncode, "result": measured}]}
                    return assess_payload(identity, result, payload)
                request = measurement.request_for(index % context["batch_count"]) if measurement is not None else json.loads(context["command"].submission)["request"]
                job = _job_response(result.stdout)
                if job is not None:
                    from .device_failures import embedded_device_failure
                    if embedded_device_failure(job):
                        from aka.contracts.measurement import MeasurementDiagnostic
                        assessment = assess_payload(identity, result)
                        return replace(assessment, diagnostics=(*assessment.diagnostics,
                            MeasurementDiagnostic("infrastructure", "device-unavailable",
                                basis=("explicit-device-diagnostic",))))
                if job and isinstance(job.get("result"), dict):
                    shapes = list(dict.fromkeys(point.shape_id for point in identity.points if point.shape_id is not None))
                    if context["kind"] == "same_allocation_abba":
                        from .native_abba import parse_native_abba
                        payload = parse_native_abba(job, request["abba"]["repeats"] * 2, shapes)
                        offset = min(point.repeat for point in identity.points)
                        payload["runs"] = [{**row, "repeat": row["repeat"] + offset} for row in payload["runs"]]
                    else:
                        measured = _optimizer_result_from_eval(job["result"], shapes,
                            request["reference"].get("metadata"),
                            require_performance=request.get("mode") != "correctness_only", hardware=context["command"].hardware)
                        payload = {"runs": [{"revision": "candidate", "repeat": identity.points[0].repeat,
                            "exit_code": result.returncode, "result": measured}]}
            except (TypeError, ValueError, KeyError, json.JSONDecodeError):
                pass
        return assess_payload(identity, result, payload)

    def interpret(self, prepared, results):
        if "diagnostic" in prepared.context:
            from .diagnostic_preparation import interpret
            return interpret(prepared, results)
        context = prepared.context
        command = context["command"]
        measurement = context["measurement"]
        if context["kind"] == "profile" and measurement is None:
            from .gateway_measurement import prepare_profile_output
            result = results[0]
            return CommandEvaluation(replace(result, stdout=prepare_profile_output(
                result.stdout, private=context["private"], failed=result.returncode != 0)))
        if context["kind"] in {"check", "disassemble"}:
            result = results[0]
            job = _job_response(result.stdout)
            payload = job.get("result") if job and job.get("status") in {"succeeded", "failed"} else None
            if not isinstance(payload, dict):
                payload = {"status": "unknown", "passed": None}
            prefix = "[sandbox] CHECK_JSON=" if context["kind"] == "check" else "[sandbox] DISASSEMBLE_JSON="
            return CommandEvaluation(replace(result, stdout=prefix + json.dumps(payload, allow_nan=False) + "\n"))
        if context["kind"] == "same_allocation_abba":
            from .native_abba import parse_native_abba, interpret_payloads
            request = json.loads(command.submission)["request"]
            shapes = list(request["reference"]["shapes"])
            payloads = [parse_native_abba(_job_response(result.stdout) or {}, request["abba"]["repeats"] * 2,
                list(measurement.batches[index % context["batch_count"]]) if measurement is not None else shapes)
                for index, result in enumerate(results)]
            verification = interpret_payloads(payloads, shapes, request["abba"]["repeats"] * 2, context["repetitions"])
            verification.measurements = tuple(self.diagnose_step(prepared, index, result) for index, result in enumerate(results))
            content = json.dumps(verification.as_dict(), ensure_ascii=False, allow_nan=False).encode()
            return CommandEvaluation(results[-1], verification,
                (MeasurementArtifact("evaluation.json", content, "evaluation"),))
        if measurement is None or (len(results) == 1 and any(line.startswith(TEST_RESULT_PREFIX) for line in results[0].stdout.splitlines())):
            result = results[0]
            if context.get("sol_workload_ids"):
                checked = self._command_evaluation(result, context, results)
                if checked.verification is None or checked.verification.gate == "ERROR":
                    return checked
            stdout = prepare_dev_output(command.workspace, result.stdout,
                evaluator=context["evaluator"], private=context["private"], baseline=context.get("baseline"))
            if context["repetitions"] > 1:
                from .evaluator import result_from_stdout
                from .aggregation import aggregate
                from .transport_results import _with_workspace_performance_score
                payload = aggregate([result_from_stdout(prepare_dev_output(command.workspace, item.stdout,
                    evaluator=context["evaluator"], private=context["private"], baseline=context.get("baseline"))) for item in results], "evaluate")
                payload.pop("performance_objective", None)
                payload = _with_workspace_performance_score(command.workspace, payload, baseline=context.get("baseline"))
                stdout = TEST_RESULT_PREFIX + json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n"
                result = replace(results[-1], transport_returncode=0 if all(item.returncode == 0 for item in results) else 1,
                    exit_code=None)
            return self._command_evaluation(replace(result, stdout=stdout), context, results)
        jobs = []
        for result in results:
            job = _job_response(result.stdout)
            if job is not None and context["kind"] == "run":
                from .device_failures import embedded_device_failure
                if embedded_device_failure(job):
                    payload = {"all_pass": None, "evaluation_failure": {
                        "category": "infrastructure", "code": "device-unavailable",
                        "message": "GPU evaluation unavailable: worker reports an unavailable device; backend recovery is required"}}
                    return self._command_evaluation(replace(result,
                        transport_returncode=2, exit_code=None,
                        stdout=TEST_RESULT_PREFIX + json.dumps(payload) + "\n"), context, results)
            settled = (job is not None and job.get("status") in {"succeeded", "failed"}
                and isinstance(job.get("result"), dict)
                and result.facts.disposition == "terminal" and result.transport_status in {"succeeded", "failed"})
            if settled and context["kind"] == "run":
                from .native_abba import complete_evaluation
                from .device_failures import embedded_device_failure
                request = measurement.request_for(len(jobs) % context["batch_count"])
                settled = (not embedded_device_failure(job) and complete_evaluation(job["result"], list(request["reference"]["shapes"]),
                    correctness_only=request.get("mode") == "correctness_only"))
            if not settled:
                result = replace(result, transport_returncode=result.returncode or 2)
                if context["private"]:
                    return self._command_evaluation(replace(result, stdout="", stderr="[sandbox] generalized measurement unavailable; details withheld\n"), context, results)
                return self._command_evaluation(result, context, results)
            jobs.append(job)
        files = []
        if context["kind"] == "run":
            batch_count = context["batch_count"]
            samples = [measurement.interpret(jobs[offset:offset + batch_count]) for offset in range(0, len(jobs), batch_count)]
            from .aggregation import aggregate
            failed = next((sample for sample in samples if sample.get("all_pass") is not True), None)
            payload = failed if failed is not None else aggregate(samples, "evaluate")
            if len(samples) > 1 and failed is None:
                from .transport_results import _metadata_speedup_mean
                score, failures = _metadata_speedup_mean(measurement.request["reference"].get("metadata"),
                    measurement.shape_ids, payload["latency_us_by_shape"], command.hardware)
                payload.update(performance_score=score, speedup_vs_ref_mean=score)
                if failures:
                    payload.update(all_pass=False, failures=failures)
            stdout = TEST_RESULT_PREFIX + json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n"
            stdout = with_measurement_conditions(stdout, gateway_warmup_iters=measurement.request.get("runner_overrides", {}).get("warmup_iters", "default"))
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
        from .measurement import shape_latency_facts
        try:
            payload = result_from_stdout(result.stdout)
        except (RuntimeError, json.JSONDecodeError):
            return CommandEvaluation(result)
        error = None
        if context.get("sol_workload_ids"):
            from .sol_evidence import validate_result
            try:
                for item in results:
                    measured = result_from_stdout(item.stdout)
                    validate_result(measured, context["sol_workload_ids"])
                    if item.transport_status not in {"succeeded", "failed"} or item.facts.disposition != "terminal":
                        raise ValueError("SOL execution has not reached a confirmed terminal state")
                    if measured["all_pass"] and item.returncode != 0:
                        raise ValueError("SOL passing receipt has a failed command exit")
            except (ValueError, RuntimeError) as failure:
                error = str(failure)
                payload = {"all_pass": None, "error": error}
                result = replace(result, transport_returncode=2, exit_code=None,
                    stdout=TEST_RESULT_PREFIX + json.dumps(payload, allow_nan=False) + "\n")
        failure = None
        unavailable = payload.get("evaluation_failure")
        if isinstance(unavailable, dict):
            from aka.contracts.evaluation import EvaluationFailure
            failure = EvaluationFailure(str(unavailable["category"]), str(unavailable["code"]),
                str(unavailable["message"]))
            error = failure.message
        verification = VerificationResult("ERROR" if error else "PASS" if payload.get("all_pass") else "FAIL",
            payload.get("latency_us_geomean"), None, None,
            error=error,
            failure=failure,
            candidate_performance_score=payload.get("performance_score"),
            candidate_latency_us_by_shape=shape_latency_facts(payload.get("latency_us_by_shape")))
        content = json.dumps(payload).encode()
        retained = ()
        if context["evaluator"]:
            # Physical Job identity is execution-owned; no stdout Job discovery.
            job_id = (",".join(reference.identifier for item in results
                if item.stdout.lstrip().startswith("{") for reference in item.job_references) or None
                if context["measurement"] is not None else None)
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
