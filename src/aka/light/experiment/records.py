"""Compose private records with the existing Gateway CLI/result contract."""
from __future__ import annotations
from aka.light.evaluation.policy import CORRECTNESS_CASES

import json
import logging
import subprocess
from pathlib import Path

from aka.light.execution.gateway_jobs import cacheable_job, job_from_process
from aka.light.experiment.store import (
    ABBA_BATCH_ROOT_ENV,
    BATCH_ROOT_ENV,
    JOB_ROOT_ENV,
    DuplicateTask,
    RecordNotFound,
    private_write,
    private_write_bytes,
    read_json,
)
from aka.light.workspace.files import MAX_FILE_BYTES, publish, relative_path
from aka.light.agent.session_tail import read_regular_bytes

from aka.light.evaluation.provider import PREFIXES, result_from_response
from aka.light.evaluation.public_result import seal_response
from aka.light.runtime.request_errors import AgentRequestError, RuntimeStateError, error_response, field_error
from aka.light.experiment.identifiers import GATEWAY_RECORD_ID_RE, KERNEL_RECORD_ID_RE
QUERY_KINDS = {"record-read", "kernel-read", "kernel-records", "kernel-pareto-frontier"}


def _completed_abba_timeout(sample: object, states: list[dict], *, truncated: bool) -> bool:
    """A twice-timed-out native ABBA is final only after both Jobs are terminal."""
    from aka.light.evaluation.native_abba import is_terminal_abba_timeout
    return bool(
        isinstance(sample, dict)
        and sample.get("correct") is False
        and is_terminal_abba_timeout(sample)
        and states
        and all(state.get("phase") == "terminal" for state in states)
        and not truncated
    )


def _read_record(reader, identifier, *, field, pattern):
    if not isinstance(identifier, str) or pattern.fullmatch(identifier) is None:
        raise field_error(field, f"{field} must be a complete ID returned by this Campaign")
    try:
        return reader(identifier)
    except (FileNotFoundError, RecordNotFound) as error:
        raise field_error(field, "Record is not available in this Campaign; check the original tool result for its exact ID",
                          code="record_not_found") from error
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise RuntimeStateError("Saved evidence is unreadable or inconsistent; ask the operator to restore it",
                                code="record_unavailable") from error


def query(store, args, workspace: Path) -> dict:
    if args.command or args.input or args.baseline_path:
        raise ValueError("Record queries take IDs, not commands or GPU inputs")
    if args.kind == "record-read":
        record = _read_record(store.read, args.record_id, field="record_id", pattern=GATEWAY_RECORD_ID_RE)
        return seal_response(record["response"], record["request"],
                             record_id=record["gateway_record_id"], kernels=record["kernels"])
    if args.kind == "kernel-pareto-frontier":
        if args.kernel_id or args.record_id or args.output_path:
            raise ValueError("kernel-pareto-frontier takes no IDs or output path; it queries this Campaign")
        from .frontier import kernel_pareto_frontier
        result = kernel_pareto_frontier(store)
    elif args.kind == "kernel-records":
        result = {"kernel_id": args.kernel_id, "gateway_records": _read_record(
            store.kernel_records, args.kernel_id, field="kernel_id", pattern=KERNEL_RECORD_ID_RE)}
    else:
        path = relative_path(args.output_path or "")
        if len(path.parts) < 2 or path.parts[0] != "scratch":
            raise ValueError("--output-path must name a file (or tree directory) inside scratch/")
        source = _read_record(store.read_kernel, args.kernel_id, field="kernel_id", pattern=KERNEL_RECORD_ID_RE)
        from aka.light.workspace.candidate import Artifact
        exported = Artifact(source).export(workspace / path)
        result = {"ok": True, "kernel_id": args.kernel_id, "file": str(path), "bytes": len(source)}
        if exported:
            result.update(exported, directory=str(path))
    return {"exit_code": 0, "stdout": json.dumps(result) + "\n", "stderr": ""}


def correctness_identity(request: dict) -> dict | None:
    """A narrow equivalence: full/untimed six-case evaluation of the same inputs.

    Only mode, performance repetitions and timing iterations may differ. All
    evaluator bytes, input/source hashes, tolerances, dependencies, target and
    environment remain bound. Old records lack the resolved seed policy and
    cannot prove this contract. This is not an Agent duplicate/reuse policy.
    """
    if request.get("operation") != "evaluate":
        return None
    options = dict(request.get("options", {}))
    policy = options.get("evaluation_policy")
    if (not isinstance(policy, dict) or policy.get("schema_version") != 1
            or policy.get("num_correctness_cases") != CORRECTNESS_CASES
            or policy.get("mode") not in {"full", "correctness_only"}
            or options.get("evaluation_input_path") or options.get("evaluation_shapes_path")):
        return None
    command = options.get("command", [])
    # Canonical entry point only; arbitrary launcher semantics are not evidence
    # for the Supervisor's standard evaluation. Reject all explicit subsets.
    if command[:2] != ["python3", "test_kernel.py"]:
        return None
    normalized = command[:2]
    index = 2
    while index < len(command):
        flag = command[index].split("=", 1)[0]
        if flag == "--shape-id":
            return None
        if flag in {"--multi-seed", "--timed-runs"}:
            index += 1 if "=" in command[index] else 2
            continue
        normalized.append(command[index])
        index += 1
    options["command"] = normalized
    options.pop("evaluation_mode", None)
    options["evaluation_policy"] = {key: value for key, value in policy.items() if key != "mode"}
    supervisor_policy = dict(request.get("supervisor_policy", {}))
    supervisor_policy.pop("repetitions", None)
    return dict(request, options=options, supervisor_policy=supervisor_policy)


def reuse_correctness_result(store, request: dict, kernel_id: str) -> dict | None:
    expected = correctness_identity(request)
    if expected is None:
        return None
    rows = sorted(store.kernel_records(kernel_id), key=lambda row: row["created_at"], reverse=True)
    for row in rows:
        if row["operation"] != "evaluate":
            continue
        try:
            record = store.read(row["gateway_record_id"])
            if record.get("cacheable") is not True or correctness_identity(record["request"]) != expected:
                continue
            response = record["response"]
            result = result_from_response(response, "evaluate")
            if (response.get("exit_code") == 0 and not response.get("truncated")
                    and result and result.get("all_pass") is True and not result.get("error")
                    and _typed_correctness_evidence(store, row["gateway_record_id"])):
                return dict(response, reused=True)
        except (ValueError, KeyError, TypeError, FileNotFoundError):
            # Corrupt/old evidence is not a correctness pass. Other IO failures
            # propagate: a storage outage must not silently cause new GPU work.
            logging.getLogger(__name__).warning("Skipping invalid correctness evidence %s", row["gateway_record_id"])
    return None


def _typed_correctness_evidence(store, record_id: str) -> bool:
    """Do not elevate a Dev script's result marker into a typed correctness gate."""
    found = False
    for path in (store.root / "records" / record_id).glob("repetition-*/jobs/*/state.json"):
        state = read_json(path)
        identity = state.get("identity", {})
        if not isinstance(identity, dict):
            return False
        payload = identity.get("payload") if identity.get("kind") == "eval" else (
            identity.get("request") if identity.get("kind") == "run" else None
        )
        if (state.get("phase") != "terminal" or not isinstance(payload, dict)
                or not isinstance(payload.get("options"), dict)
                or payload.get("options", {}).get("num_correctness_cases") != CORRECTNESS_CASES
                or payload.get("mode") not in {"full", "correctness_only"}):
            return False
        found = True
    return found


def execute(store, evaluation, effects, staged, args, argv, environment, command, *,
            reuse_completed=False, reuse_correctness=False) -> dict:
    from aka.light.runtime.errors import RequestDispatchTimeout
    from aka.light.evaluation.projection import (
        GATEWAY_ERROR_PREFIX,
        SOURCE_ERROR_PREFIX,
        candidate_source_rejection,
        source_error_from_stdout,
    )
    try:
        plan = evaluation.prepare(args, staged, environment, identity_key=store.identity_key,
                                  mode=effects.mode, repetitions=effects.repetitions)
        request, inputs = plan.request, plan.inputs
    except ValueError as error:
        # Preparation has not dispatched an executor or reserved a GPU task.
        # Let the HTTP boundary attach the effective CLI contract and redaction.
        raise AgentRequestError(str(error)) from error
    operation = request["operation"]
    from aka.light.workspace.candidate import measurement_artifacts
    kernels = {side: store.kernel(artifact.snapshot)
               for side, artifact in measurement_artifacts(request, inputs).items()}
    if reuse_correctness and reuse_completed and "candidate" in kernels:
        response = reuse_correctness_result(store, request, kernels["candidate"]["kernel_id"])
        if response is not None:
            return response
    try:
        with store.reserve(request, kernels, deadline=effects.deadline, request_id=effects.request_id) as task:
            for name, source in inputs.items():
                private_write_bytes(task.directory / "inputs" / name, source)
            repetitions = plan.repetitions
            samples, responses, cacheable, pending = [], [], True, False
            # The identity carries only a keyed fingerprint of launcher values.
            # Retain evidence, but never cache a result measured with custom
            # environment or SSH initialization as an interchangeable task.
            custom_launcher = bool(args.env or args.ssh_init)
            for repetition in range(repetitions):
                directory = task.directory / f"repetition-{repetition + 1}"
                env = environment | {
                    JOB_ROOT_ENV: str(directory),
                    "ATREX_AKA_IDENTITY_KEY": store.identity_key.hex(),
                }
                if operation in {"evaluate", "same_allocation_abba"}:
                    # A task may finish only a subset of Shape batches before
                    # an infrastructure outage. Keep completed batches at a
                    # Campaign location so a fresh Gateway Record dispatches
                    # only missing Shapes. Evaluate is task/repetition-scoped;
                    # ABBA keeps its compatible repetition scope. Every
                    # checkpoint verifies its exact batch request, while the
                    # Eval task digest also binds policy and runtime pins.
                    recovery_root = (
                        store.root
                        / "eval-recovery"
                        / task.key
                        / f"repetition-{repetition + 1}"
                        if operation == "evaluate"
                        else store.root
                        / "abba-recovery"
                        / f"repetition-{repetition + 1}"
                    )
                    env[BATCH_ROOT_ENV] = str(recovery_root)
                    if operation == "same_allocation_abba":
                        # Compatibility for older standalone ABBA entrypoints.
                        env[ABBA_BATCH_ROOT_ENV] = str(recovery_root)
                if "version" in plan.runtime_pins:
                    env["ATREX_SANDBOX_ATREX_BENCH_VERSION"] = plan.runtime_pins["version"]
                if "requirements" in plan.runtime_pins:
                    pinned = directory / "runtime-requirements.txt"
                    private_write_bytes(pinned, ("\n".join(plan.runtime_pins["requirements"]) + "\n").encode())
                    env["ATREX_SANDBOX_REQUIREMENTS_FILE"] = str(pinned)
                if effects.private_reference_dir is not None:
                    env["ATREX_PRIVATE_REFERENCE_DIR"] = str(task.directory / "inputs")
                try:
                    process = effects.run(command, staged, env)
                except RequestDispatchTimeout as error:
                    if repetition or any(task.directory.glob("repetition-*/jobs/*/state.json")):
                        from aka.light.execution.retry import InfrastructureUnavailable
                        raise InfrastructureUnavailable(
                            "Measurement partially dispatched; durable job recovery required"
                        ) from error
                    raise
                private_write(directory / "executor.json", {"returncode": process.returncode,
                              "stdout": process.stdout, "stderr": process.stderr})
                effects.audit(argv, process, task.record_id)
                response = evaluation.project(process, plan,
                    generalized=effects.private_reference_dir is not None, environment=env,
                    private_paths=(str(staged), str(store.root), *effects.private_paths))
                responses.append(response)
                sample = evaluation.result(response, operation)
                states = [read_json(path) for path in directory.glob("**/jobs/*/state.json")]
                pending = pending or any(state.get("phase") != "terminal" for state in states)
                cacheable = cacheable and bool(states) and all(
                    state.get("phase") == "terminal" and cacheable_job(job_from_process(subprocess.CompletedProcess(
                        [], state["process"]["returncode"], state["process"]["stdout"], state["process"]["stderr"])),
                        state.get("identity"))
                    for state in states)
                cacheable = cacheable and not response.get("truncated")
                if operation == "same_allocation_abba":
                    terminal_abba_timeout = _completed_abba_timeout(
                        sample,
                        states,
                        truncated=response.get("truncated") is True,
                    )
                    if terminal_abba_timeout:
                        # The native ABBA executor has already submitted one
                        # fresh retry.  Two terminal wall-clock failures are a
                        # complete negative promotion result, not uncertain
                        # infrastructure that the Supervisor should retry forever.
                        cacheable = True
                    else:
                        cacheable = (
                            cacheable
                            and sample is not None
                            and isinstance(sample.get("correct"), bool)
                        )
                if sample is not None:
                    samples.append(sample)
                if process.returncode != 0 or sample is None or not cacheable:
                    break
            response = responses[-1]
            aggregated_result = None
            if repetitions > 1 and len(samples) == repetitions and all(row.get("exit_code") == 0 for row in responses):
                aggregated_result = evaluation.aggregate(samples, plan)
            # IDs are attached to the same public payload stored and re-read.
            prefix = PREFIXES.get(operation)
            identity = {"gateway_record_id": task.record_id}
            if "candidate" in kernels:
                identity["kernel_id"] = kernels["candidate"]["kernel_id"]
            if "baseline" in kernels:
                identity["baseline_kernel_id"] = kernels["baseline"]["kernel_id"]
            # Keep projected warnings/progress/diagnostics in place. Only the
            # last result marker represents the sample selected for aggregation.
            lines = response["stdout"].splitlines()
            markers = [index for index, line in enumerate(lines)
                       if (prefix and line.startswith(prefix)) or line.startswith((SOURCE_ERROR_PREFIX, GATEWAY_ERROR_PREFIX))]
            for index in markers:
                marker_prefix = next((item for item in (SOURCE_ERROR_PREFIX, GATEWAY_ERROR_PREFIX)
                                      if lines[index].startswith(item)), prefix)
                value = (aggregated_result if aggregated_result is not None and index == markers[-1]
                         else json.loads(lines[index][len(marker_prefix):]))
                lines[index] = marker_prefix + json.dumps(value | identity)
            if not markers:
                lines.append("[sandbox] RECORD_JSON=" + json.dumps(identity | {"operation": operation,
                             "status": "succeeded" if response["exit_code"] == 0 else "failed"}))
            response = dict(response, stdout="\n".join(lines) + "\n")
            response = seal_response(response, request, record_id=task.record_id, kernels=kernels)
            task.finish(response, cacheable=cacheable and not custom_launcher, pending=pending)
            # A source rejection is definitive and repairable, not missing GPU
            # evidence. Require private checkpoints too; a Dev-printed marker
            # alone must not authorize an acceptance classification.
            source_rejected = source_error_from_stdout(response["stdout"]) is not None and bool(states) and all(
                state.get("phase") == "rejected" and candidate_source_rejection(subprocess.CompletedProcess(
                    [], state["process"]["returncode"], state["process"]["stdout"], state["process"]["stderr"])) is not None
                for state in states
            )
            if reuse_completed and not cacheable and not source_rejected:
                from aka.light.execution.gateway_jobs import retry_kind
                from aka.light.execution.retry import InfrastructureUnavailable

                if states and all(state.get("phase") == "terminal" for state in states) and any(
                    retry_kind(job_from_process(subprocess.CompletedProcess(
                        [], state["process"]["returncode"], state["process"]["stdout"], state["process"]["stderr"]))) == "infra"
                    for state in states
                ):
                    raise InfrastructureUnavailable("Recorded GPU validation infrastructure failure")
                raise InfrastructureUnavailable(
                    "Acceptance measurement is incomplete or uncertain; inspect its private Gateway Record"
                )
            if operation == "evaluate" and repetitions > 1 and samples:
                from aka.light.evaluation.gateway import EPISODE_EVALUATIONS_PATH
                # The old report compiler must see the same aggregate as the
                # Agent, not whichever repetition happened to finish last.
                log = read_regular_bytes(staged / EPISODE_EVALUATIONS_PATH, limit=MAX_FILE_BYTES + 1)
                if len(log) > MAX_FILE_BYTES:
                    raise ValueError("Evaluation log exceeds the size limit")
                if log.strip():
                    row = json.loads(log.splitlines()[-1])
                    row["result"] = result_from_response(response, operation)
                    row["gateway_record_id"] = task.record_id
                    publish(staged, str(EPISODE_EVALUATIONS_PATH), (json.dumps(row) + "\n").encode())
            # Record publication precedes legacy-output publication: a later
            # filesystem error cannot cause a completed GPU task to run again.
            effects.publish_log(staged)
            if response["exit_code"] == 0:
                effects.publish_outputs(staged, args)
            return response
    except DuplicateTask as error:
        if reuse_completed and error.record_id:
            # reserve() has already validated the exact task identity and its
            # cacheability. This switch is never accepted from HTTP/Agent argv.
            return dict(store.read(error.record_id)["response"], reused=True)
        return task_conflict_response(error)


def task_conflict_response(error: DuplicateTask) -> dict:
    info = error.response()
    recovery = ([{"tool": "record-read", "request": {"record_id": error.record_id},
                  "command": "python3 tools/sandbox.py --kind record-read --record-id " + error.record_id,
                  "instruction": "Reuse the completed result; do not resubmit this exact Kernel task."}]
                if error.record_id else [{"instruction": info["next_action"]}])
    response = error_response(info.pop("message"), repairable=False, recovery=recovery, **info)
    response["previous_gateway_record_id"] = error.record_id
    return {"exit_code": 2, "stdout": "", "stderr": json.dumps(response) + "\n"}
