"""Prepare Legacy measurement commands and interpret raw execution receipts."""
from dataclasses import replace
import json
from pathlib import Path
from aka.contracts.execution import CommandSpec, ExecutionFile
from aka.contracts.evaluation import PreparedEvaluation, StepEvaluation, MeasurementArtifact
from aka.legacy.execution.gateway import sandbox as transport
from aka.legacy.execution.legacy.runtime import current_transport_arguments
from .gateway_request import prepare_request, _candidate_languages
from .gateway_policy import _typed_workspace_limitation
from .gateway_measurement import GatewayMeasurement, prepare_dev_output, with_measurement_conditions
from .metadata import distributed_evaluation_world_size


class GatewayCommandEvaluator:
    def prepare(self, command):
        if command.operation == "preflight" or command.submission is not None:
            identities = ()
            if command.submission is not None:
                prepared = json.loads(command.submission)
                if prepared.get("kind") in {"run", "profile"}:
                    from .measurement import gateway_identity
                    identities = (gateway_identity(prepared["request"], prepared.get("reference_files", {}), prepared["kind"]),)
            return PreparedEvaluation((command,), 1, {"command": command, "measurement": None,
                "kind": command.operation, "private": False, "evaluator": False},
                measurement_identities=identities)
        environment = command.environment
        argv = ["--hardware", command.hardware, "--workspace", str(command.workspace),
                "--timeout", str(command.timeout), "--kind", command.operation,
                *current_transport_arguments(), "--", *command.command]
        parser = transport.build_parser()
        parser.set_defaults(hardware=command.hardware, timeout=command.timeout,
            ssh=environment.get("ATREX_SANDBOX_SSH", ""), url=environment.get("ATREX_SANDBOX_URL", ""),
            gateway_profile=environment.get("ATREX_SANDBOX_PROFILE") or None,
            shape_batch_size=int(environment.get("ATREX_EVAL_SHAPE_BATCH_SIZE", transport.DEFAULT_EVAL_SHAPE_BATCH_SIZE)))
        args = parser.parse_args(argv)
        if args.shape_batch_size < 1:
            raise ValueError("shape batch size must be positive")
        kind = transport._requested_gateway_kind(args.kind, list(command.command))
        evaluator = transport._is_test_kernel_command(command.command)
        private = transport._is_generalized_workspace(command.workspace)
        for relative in command.artifacts:
            transport._safe_relative(relative)
        context = {"command": command, "args": args, "kind": kind, "private": private,
                   "evaluator": evaluator, "measurement": None}
        if command.operation == "preflight" or not (evaluator or kind == "profile"):
            context["kind"] = "dev"
            command = replace(command, operation="run") if command.operation == "auto" else command
            return PreparedEvaluation((command,), 1, context)
        reference = Path(environment.get("ATREX_PRIVATE_REFERENCE_DIR") or command.workspace)
        # Validate public/private association before any submission.
        if private:
            from .private_inputs import private_reference_dir
            reference = private_reference_dir(command.workspace, environment) or reference
        metadata = transport._json_object(reference / "metadata.json")
        num_gpus = distributed_evaluation_world_size(metadata)
        context["num_gpus"] = num_gpus
        command = replace(command, num_gpus=num_gpus)
        context["command"] = command
        ssh = args.ssh
        limitation = ("SSH uses the portable runner" if ssh else
            "custom dev options" if args.input or args.keep_pod or args.include_raw_profile else
            _typed_workspace_limitation(command.workspace, list(command.command), kind, reference_dir=reference))
        if kind == "dev" or limitation:
            if kind == "profile" and num_gpus > 1:
                raise ValueError("distributed profile requires the typed route")
            if ssh and evaluator and num_gpus > 1:
                raise ValueError("distributed evaluator is unsupported by SSH")
            return PreparedEvaluation((replace(command, operation="dev"),), 1, context)
        if kind == "profile" and args.profile_level == "deep" and not args.kernel_regex:
            raise ValueError("--profile-level deep requires --kernel-regex")
        requirements_path = environment.get("ATREX_SANDBOX_REQUIREMENTS_FILE", "").strip()
        solution = transport._json_object(command.workspace / "solution.json") or {}
        candidate = (command.workspace / "kernel.py").read_text(encoding="utf-8")
        requirements = transport._runtime_requirements(requirements_path,
            candidate_languages=_candidate_languages(solution, candidate_sources=(candidate,))) if requirements_path else []
        try:
            request = prepare_request(command.workspace, command.hardware, command.timeout, args.env,
                list(command.command), kind, reference_dir=reference, requirements=requirements,
                profiler=args.profiler, profile_level=args.profile_level, counters=args.profile_counter,
                kernel_regex=args.kernel_regex, top_kernels=args.top_kernels)
        except (OSError, UnicodeDecodeError, ValueError) as error:
            # A malformed performance baseline deliberately propagates; it must not submit GPU work.
            from aka.legacy.task.problem.inputs import PerformanceBaselineError
            if isinstance(error, PerformanceBaselineError):
                raise
            return PreparedEvaluation((replace(command, operation="dev"),), 1, context)
        measurement = GatewayMeasurement(request, workspace=command.workspace, hardware=command.hardware,
            private=private, batch_size=args.shape_batch_size, profiling=kind == "profile")
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
            steps.append(replace(command, operation=kind, submission=submission, artifacts=()))
        return PreparedEvaluation(tuple(steps), min(4, len(steps)), context,
            fallback_steps=(replace(command, operation="dev", submission=None),),
            measurement_identities=tuple(identities))

    def fallback(self, prepared, results):
        """Only a settled unsupported route can authorize the prepared dev alternative."""
        if not prepared.fallback_steps or not results:
            return ()
        if any(result.facts.disposition == "unknown" or result.transport_status in {"cancelled", "timed_out", "interrupted"}
               or any(item.category in {"cancelled", "authentication"} for item in result.diagnostics)
               for result in results):
            return ()
        if any(result.returncode and transport._typed_fallback_allowed(result.stdout + "\n" + result.stderr)
               for result in results):
            return prepared.fallback_steps
        return ()

    def observe_step(self, prepared, index, result):
        assessment = self.diagnose_step(prepared, index, result)
        if prepared.context["measurement"] is None:
            return StepEvaluation(assessment=assessment)
        return StepEvaluation((MeasurementArtifact(f"receipt-{index:04d}.json", result.stdout.encode(), "measurement_batch"),), assessment)

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
                job = transport._job_response(result.stdout)
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
        if measurement is None or (len(results) == 1 and any(line.startswith(transport.TEST_RESULT_PREFIX) for line in results[0].stdout.splitlines())):
            result = results[0]
            stdout = prepare_dev_output(command.workspace, result.stdout,
                evaluator=context["evaluator"], private=context["private"])
            return replace(result, stdout=stdout)
        jobs = []
        for result in results:
            job = transport._job_response(result.stdout)
            if result.returncode or job is None or job.get("status") != "succeeded" or not isinstance(job.get("result"), dict):
                result = replace(result, transport_returncode=result.returncode or 2)
                if context["private"]:
                    return replace(result, stdout="", stderr="[sandbox] generalized measurement unavailable; details withheld\n")
                return result
            jobs.append(job)
        files = []
        if context["kind"] == "run":
            payload = measurement.interpret(jobs)
            stdout = transport.TEST_RESULT_PREFIX + json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n"
            stdout = with_measurement_conditions(stdout, gateway_warmup_iters=command.environment.get("AKA_GATEWAY_WARMUP_ITERS", "5"))
            code = 0 if payload["all_pass"] else 1
        else:
            stdout = transport.PROFILE_RESULT_PREFIX + json.dumps(jobs[0]["result"], ensure_ascii=False) + "\n"
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
        return replace(results[-1], transport_returncode=code, exit_code=None, stdout=stdout,
            stderr="\n".join(item.stderr.rstrip() for item in results if item.stderr), files=tuple(files),
            attempts=tuple(attempt for item in results for attempt in item.attempts))
