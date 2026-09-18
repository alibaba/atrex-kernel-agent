"""Compose private records with the existing Gateway CLI/result contract."""
from __future__ import annotations

import json
import math
import statistics
import subprocess
from pathlib import Path

from supervisor.gateway_jobs import cacheable_job, job_from_process
from supervisor.measurement_records import (
    JOB_ROOT_ENV, DuplicateTask, private_write, private_write_bytes, read_json,
)
from supervisor.workspace import MAX_FILE_BYTES, publish, relative_path
from orchestrator.session_tail import read_regular_bytes

PREFIXES = {"evaluate": "[test_kernel] RESULT_JSON=", "profile": "[sandbox] PROFILE_JSON=",
            "check": "[sandbox] CHECK_JSON=", "disassemble": "[sandbox] DISASSEMBLE_JSON=",
            "same_allocation_abba": "[sandbox] ABBA_JSON="}
QUERY_KINDS = {"record-read", "kernel-read", "kernel-records"}


def query(store, args, workspace: Path) -> dict:
    if args.command or args.input or args.baseline_path:
        raise ValueError("Record queries take IDs, not commands or GPU inputs")
    if args.kind == "record-read":
        try:
            return store.read(args.record_id or "")["response"]
        except FileNotFoundError as error:
            raise ValueError("Gateway record is not available in this Campaign; use an ID returned here") from error
    if args.kind == "kernel-records":
        result = {"kernel_id": args.kernel_id, "gateway_records": store.kernel_records(args.kernel_id or "")}
    else:
        path = relative_path(args.output_path or "")
        if len(path.parts) < 2 or path.parts[0] != "scratch":
            raise ValueError("--output-path must name a file inside scratch/")
        source = store.read_kernel(args.kernel_id or "")
        publish(workspace, str(path), source)
        result = {"ok": True, "kernel_id": args.kernel_id, "file": str(path), "bytes": len(source)}
    return {"exit_code": 0, "stdout": json.dumps(result) + "\n", "stderr": ""}


def result_from_response(response: dict, operation: str) -> dict | None:
    prefix = PREFIXES.get(operation)
    for line in reversed(response["stdout"].splitlines()):
        if prefix and line.startswith(prefix):
            value = json.loads(line[len(prefix):])
            return value if isinstance(value, dict) else None
    return None


def _median_side(sides: list[dict]) -> dict:
    maps = [side.get("latency_us_by_shape", {}) for side in sides]
    if not maps[0] or any(set(values) != set(maps[0]) for values in maps):
        raise ValueError("Repeated measurement Shape coverage differs; no aggregate is available")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0
           for values in maps for value in values.values()):
        raise ValueError("Repeated measurement contains an invalid latency")
    result = dict(sides[-1])
    values = {key: statistics.median(row[key] for row in maps) for key in maps[0]}
    result["latency_us_by_shape"] = values
    result["latency_us_geomean"] = math.exp(statistics.mean(math.log(value) for value in values.values()))
    result["latency_us_arith_mean"] = statistics.mean(values.values())
    for key in ("max_abs_err", "max_rel_err"):
        errors = [side[key] for side in sides if isinstance(side.get(key), (int, float))]
        if errors:
            result[key] = max(errors)
    # Scores computed on an individual run must not be mislabeled as median facts.
    for key in ("performance_score", "speedup_vs_ref_geomean", "speedup_vs_ref_mean", "utilization_pct", "sol_utilization_pct"):
        result.pop(key, None)
    return result


def aggregate(results: list[dict], operation: str) -> dict:
    if len(results) == 1:
        return results[0]
    if operation == "evaluate":
        if any(row.get("all_pass") is not True for row in results):
            raise ValueError("Cannot aggregate incomplete or rejected evaluations")
        return _median_side(results)
    if any(row.get("correct") is not True for row in results):
        raise ValueError("Cannot aggregate incomplete or rejected comparisons")
    value = dict(results[-1])
    value["baseline"] = _median_side([row["baseline"] for row in results])
    value["candidate"] = _median_side([row["candidate"] for row in results])
    value["speedup"] = value["baseline"]["latency_us_geomean"] / value["candidate"]["latency_us_geomean"]
    value["improvement_pct"] = (value["speedup"] - 1) * 100
    value.pop("measurements", None)
    return value


def execute(runtime, capability, staged, args, argv, environment, command) -> dict:
    from supervisor.gateway import measurement_inputs, metadata_speedup_mean
    from orchestrator.supervisor_runtime import ROOT, RequestDispatchTimeout
    from supervisor.projection import project_response
    store = runtime.measurements
    try:
        request, inputs = measurement_inputs(args, staged, environment)
    except ValueError as error:
        return {"exit_code": 2, "stdout": "", "stderr": f"sandbox: {error}\n"}
    request["supervisor_policy"] = {"mode": runtime.config.optimization_mode,
                                    "repetitions": runtime.measurement_repetitions}
    operation = request["operation"]
    kernels = {"candidate": store.kernel(inputs["kernel.py"])} if "kernel.py" in inputs else {}
    if args.baseline_path:
        kernels["baseline"] = store.kernel(inputs[args.baseline_path])
    try:
        with store.reserve(request, kernels) as task:
            for name, source in inputs.items():
                private_write_bytes(task.directory / "inputs" / name, source)
            repetitions = runtime.measurement_repetitions if operation in {"evaluate", "same_allocation_abba"} else 1
            if args.evaluation_mode == "correctness_only":
                repetitions = 1
            samples, responses, cacheable, pending = [], [], True, False
            for repetition in range(repetitions):
                directory = task.directory / f"repetition-{repetition + 1}"
                env = environment | {JOB_ROOT_ENV: str(directory)}
                if runtime.config.private_reference_dir is not None:
                    env["ATREX_PRIVATE_REFERENCE_DIR"] = str(task.directory / "inputs")
                try:
                    process = runtime.run_executor(command, staged, env, capability)
                except RequestDispatchTimeout as error:
                    if repetition or any(task.directory.glob("repetition-*/jobs/*/state.json")):
                        raise RuntimeError("Measurement partially dispatched; durable job recovery required") from error
                    raise
                private_write(directory / "executor.json", {"returncode": process.returncode,
                              "stdout": process.stdout, "stderr": process.stderr})
                runtime.audit_process(capability, "gateway", argv, process, task.record_id)
                response = project_response(process, generalized=runtime.config.private_reference_dir is not None,
                    private_paths=(str(staged), str(runtime.root), str(store.root), str(ROOT), str(runtime.config.private_reference_dir or ""),
                                   runtime.config.url, str(runtime.config.atrex_bench_root or "")))
                responses.append(response)
                sample = result_from_response(response, operation)
                states = [read_json(path) for path in directory.glob("**/jobs/*/state.json")]
                if not states:
                    states = [read_json(path) for path in (directory / "jobs").glob("*/state.json")]
                pending = pending or any(state.get("phase") != "terminal" for state in states)
                cacheable = cacheable and bool(states) and all(
                    state.get("phase") == "terminal" and cacheable_job(job_from_process(subprocess.CompletedProcess(
                        [], state["process"]["returncode"], state["process"]["stdout"], state["process"]["stderr"])),
                        state.get("identity"))
                    for state in states)
                cacheable = cacheable and not response.get("truncated")
                if operation == "same_allocation_abba":
                    cacheable = cacheable and sample is not None and isinstance(sample.get("correct"), bool)
                if sample is not None:
                    samples.append(sample)
                if process.returncode != 0 or sample is None or not cacheable:
                    break
            response = responses[-1]
            if len(samples) == repetitions and all(row.get("exit_code") == 0 for row in responses):
                result = aggregate(samples, operation) if operation in {"evaluate", "same_allocation_abba"} else samples[-1]
                if operation == "evaluate" and repetitions > 1 and "metadata.json" in inputs:
                    latencies = result.get("latency_us_by_shape", {})
                    score, failures = metadata_speedup_mean(json.loads(inputs["metadata.json"]), list(latencies), latencies)
                    if not failures and score is not None:
                        result.update(performance_score=score, speedup_vs_ref_mean=score)
                response = dict(response, stdout=PREFIXES[operation] + json.dumps(result) + "\n")
            # IDs are attached to the same public payload stored and re-read.
            prefix = PREFIXES.get(operation)
            identity = {"gateway_record_id": task.record_id}
            if "candidate" in kernels:
                identity["kernel_id"] = kernels["candidate"]["kernel_id"]
            if "baseline" in kernels:
                identity["baseline_kernel_id"] = kernels["baseline"]["kernel_id"]
            lines = []
            for line in response["stdout"].splitlines():
                if prefix and line.startswith(prefix):
                    line = prefix + json.dumps(json.loads(line[len(prefix):]) | identity)
                lines.append(line)
            if not any(prefix and line.startswith(prefix) for line in lines):
                lines.append("[sandbox] RECORD_JSON=" + json.dumps(identity | {"operation": operation,
                             "status": "succeeded" if response["exit_code"] == 0 else "failed"}))
            response = dict(response, stdout="\n".join(lines) + "\n")
            task.finish(response, cacheable=cacheable, pending=pending)
            if operation == "evaluate" and repetitions > 1 and samples:
                from supervisor.gateway import EPISODE_EVALUATIONS_PATH
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
            runtime.publish_evaluation_log(capability.workspace, staged)
            if response["exit_code"] == 0:
                runtime.publish_legacy_outputs(capability.workspace, staged, args)
            return response
    except DuplicateTask as error:
        return {"exit_code": 2, "stdout": "", "stderr": json.dumps({"error": error.response()}) + "\n"}
