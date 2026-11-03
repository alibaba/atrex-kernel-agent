"""Interpret complete native SDK ABBA receipts without owning jobs or records."""
from __future__ import annotations

import math
import statistics
import json
import ast
from dataclasses import replace
from pathlib import Path

from aka.contracts.evaluation import VerificationResult, VerificationRun
from .abba import verification_schedule
from .aggregation import aggregate
from .correctness_summary import correctness_summary
from .transport_results import _compile_failures


def prepare_comparison(request, configuration, workspace, evaluation_input, context):
    """Freeze public Workspace revisions and Task inputs in typed Job requests."""
    from aka.contracts.evaluation import PreparedEvaluation, MeasurementArtifact
    from aka.contracts.execution import CommandSpec, UnsupportedExecution
    from aka.contracts.measurement import MeasurementPoint
    from .gateway_request import prepare_request, _candidate_languages
    from .command_inputs import workload_environment
    from .private_inputs import selected_content
    from .shapes import batch_shape_ids, shape_batch_request
    from .transport_results import _shape_batch_reference
    from .measurement import gateway_identity
    capabilities = configuration.execution_capabilities
    if workspace is None or capabilities is None:
        raise UnsupportedExecution("native ABBA requires selected Workspace and Execution capabilities")
    if type(configuration.repeats) is not int or configuration.repeats < 2 or configuration.repeats % 2:
        raise ValueError("native ABBA requires an even per-side repeat count")
    if (request.workspace / "workload.jsonl").is_file():
        raise UnsupportedExecution("native ABBA does not replace SOL's full workload evaluator")
    from .source_archives import prepare_source, source_parameters
    paths = ["kernel.py"]
    has_solution = "solution.json" in request.changed_paths or (request.workspace / "solution.json").is_file()
    if has_solution:
        paths.append("solution.json")
    revisions = []
    for revision in (request.base_commit, request.candidate_commit):
        reader = getattr(workspace, "read_source", None)
        if callable(reader):
            frozen = reader(request.workspace, revision=revision)
            solution_content = workspace.read_revision(request.workspace, revision, ("solution.json",)) if has_solution else None
            solution_bytes = solution_content.files.get("solution.json") if solution_content is not None else None
        else:
            # Older injected test/embedding providers can still supply the
            # existing read_revision capability for a single-file source.
            frozen = workspace.read_revision(request.workspace, revision, paths)
            solution_bytes = frozen.files.get("solution.json")
            from aka.contracts.workspace import RevisionContent
            frozen = RevisionContent({"kernel.py": frozen.files.get("kernel.py")})
        source = frozen.files.get("kernel.py")
        if not isinstance(source, bytes) or not source.strip():
            raise ValueError("native ABBA requires two nonempty frozen kernel.py revisions")
        solution = json.loads(solution_bytes) if solution_bytes else {}
        if not isinstance(solution, dict):
            raise ValueError("native ABBA source metadata must be an object")
        descriptor, archive, artifact, format_name = prepare_source(frozen,
            role="baseline" if revision == request.base_commit and not revisions else "candidate")
        text = source.decode("utf-8")
        if archive is None:
            if any(isinstance(item, dict) and item.get("path") not in {None, "kernel.py"} for item in solution.get("sources", [])):
                raise UnsupportedExecution("auxiliary sources require a complete declared source snapshot")
            try:
                tree = ast.parse(text)
            except SyntaxError:
                tree = None
            imports = set()
            for node in ast.walk(tree) if tree is not None else ():
                if isinstance(node, ast.Import):
                    imports.update(alias.name.split(".", 1)[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imports.add(node.module.split(".", 1)[0])
            if any(root not in {"input", "reference", "kernel"} and (
                    (request.workspace / (root + ".py")).is_file() or (request.workspace / root / "__init__.py").is_file()) for root in imports):
                raise UnsupportedExecution("native ABBA cannot omit imported local candidate sources")
        identity_reader = getattr(workspace, "source_identity", None)
        facts = source_parameters(frozen, identity=identity_reader(frozen) if callable(identity_reader) else None)
        revisions.append((text, solution, frozen, descriptor, archive, artifact, facts))
    baseline, candidate = revisions
    declared = set(baseline[2].files) | set(candidate[2].files) | {"solution.json"}
    if set(request.changed_paths) - declared:
        raise UnsupportedExecution("native ABBA cannot omit changed undeclared sources")
    archives = tuple(value[4] for value in (baseline, candidate) if value[4] is not None)
    source_facts = {**candidate[6], "baseline_source_sha256": baseline[6]["candidate_source_sha256"]}
    if candidate[4] is not None:
        import hashlib
        source_facts["candidate_source_archive_sha256"] = hashlib.sha256(candidate[4].content).hexdigest()
    if baseline[4] is not None:
        import hashlib
        source_facts["baseline_source_archive_sha256"] = hashlib.sha256(baseline[4].content).hexdigest()
    content = selected_content(evaluation_input)
    references = {name: item.content for name, item in content.items()}
    reference = evaluation_input.directory if evaluation_input is not None else configuration.private_reference_dir or request.workspace
    command = CommandSpec(request.workspace, ("python3", "test_kernel.py", "--version", "vlong", "--no-memory"),
        timeout=configuration.timeout, operation="same_allocation_abba", hardware=configuration.hardware,
        environment=configuration.environment)
    environment = workload_environment(command)
    requirements = ()
    if evaluation_input is not None and evaluation_input.worker_requirements is not None:
        item = evaluation_input.worker_requirements
        text = item.source.read_text(encoding="utf-8") if item.source is not None else item.text
        from aka.execution.environment_query import select_requirements
        requirements = tuple(select_requirements(text, _candidate_languages(candidate[1],
            candidate_sources=(baseline[0], candidate[0]))))
    elif configuration.environment.get("ATREX_SANDBOX_REQUIREMENTS_FILE"):
        from aka.execution.environment_query import select_requirements
        text = Path(configuration.environment["ATREX_SANDBOX_REQUIREMENTS_FILE"]).read_text()
        requirements = tuple(select_requirements(text, _candidate_languages(candidate[1],
            candidate_sources=(baseline[0], candidate[0]))))
    payload = prepare_request(request.workspace, configuration.hardware, configuration.timeout,
        [f"{key}={value}" for key, value in environment.items()], list(command.command), "run",
        reference_dir=reference, reference_content=references, requirements=requirements,
        runtime_version=configuration.environment.get("ATREX_SANDBOX_ATREX_BENCH_VERSION", ""),
        candidate_source=candidate[0], candidate_solution=candidate[1], measurement_environment=configuration.environment)
    payload["candidate"] = candidate[3]
    payload["abba"] = {"baseline": baseline[3], "repeats": configuration.repeats // 2}
    payload["spec"]["languages"] = _candidate_languages(candidate[1], candidate_sources=(baseline[0], candidate[0]))
    payload["runner_overrides"]["perf_timeout_s"] = configuration.per_run_timeout
    shapes = list(payload["reference"]["shapes"])
    if request.expected_shape_ids is not None and set(shapes) != set(request.expected_shape_ids):
        raise ValueError("native ABBA Task Shape coverage differs from the requested workload")
    batches = batch_shape_ids(shapes, configuration.shape_batch_size)
    if not batches:
        raise ValueError("native ABBA requires at least one Task Shape")
    steps, identities = [], []
    for outer in range(configuration.measurement_repetitions):
        for ids in batches:
            batch = shape_batch_request(payload, ids)
            files = {name: value.decode("utf-8") for name, value in _shape_batch_reference(batch["reference"]).items()}
            step = replace(command, requirements=requirements, source_archives=archives, num_gpus=payload["spec"].get("num_gpus", 1),
                submission=json.dumps({"schema_version": 1, "kind": "run", "request": batch,
                    "reference_files": files}, ensure_ascii=False, allow_nan=False).encode())
            capabilities.require(step)
            identity = gateway_identity(batch, files, source_parameters=source_facts)
            identity = replace(identity, points=tuple(MeasurementPoint(point.side,
                request.base_commit if point.side == "incumbent" else request.candidate_commit,
                point.repeat + outer * configuration.repeats, point.shape_id) for point in identity.points),
                parameters={**identity.parameters, "measurement_repetitions": configuration.measurement_repetitions,
                    "measurement_repetition": outer})
            steps.append(step)
            identities.append(identity)
    return PreparedEvaluation(tuple(steps), max(1, min(configuration.max_parallelism, len(steps))),
        replace(context, schedule=tuple(verification_schedule(configuration.repeats)), shape_ids=tuple(shapes), native=True,
            candidate_source=candidate[5]),
        measurement_identities=tuple(identities))


def complete_evaluation(result: dict, shapes: list[str], *, correctness_only=False) -> bool:
    """Absent stage evidence cannot establish a completed candidate rejection."""
    passed = result.get("passed")
    if not shapes or not isinstance(passed, dict) or result.get("error"):
        return False
    compilation = passed.get("compile")
    if not isinstance(compilation, dict):
        return False
    correctness = passed.get("correctness") or {}
    performance = (result.get("performance") or {}).get("shapes") or {}
    if not isinstance(correctness, dict) or not isinstance(performance, dict):
        return False
    for sid in shapes:
        compiled = compilation if "status" in compilation else compilation.get(sid, {})
        if not isinstance(compiled, dict):
            return False
        if compiled.get("status") == "failed":
            continue
        if compiled.get("status") != "passed":
            return False
        checked = correctness.get(sid, {})
        if not isinstance(checked, dict):
            return False
        if checked.get("status") == "failed":
            continue
        if checked.get("status") != "passed":
            return False
        if correctness_only:
            continue
        timed = performance.get(sid, {})
        if not isinstance(timed, dict) or not isinstance(timed.get("samples"), list):
            return False
        values = [row.get("end_to_end_time_ms") for row in timed["samples"] if isinstance(row, dict)]
        if not values or any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0 for value in values):
            return False
    return True


def summarize(raw: dict, shapes: list[str]) -> dict:
    if not complete_evaluation(raw, shapes):
        raise ValueError("Native ABBA run has incomplete evaluator evidence")
    passed = raw["passed"]
    statuses = passed.get("correctness") or {}
    details = (raw.get("correctness") or {}).get("shapes") or {}
    correct = (not _compile_failures(passed.get("compile"), shapes)
               and all(statuses.get(sid, {}).get("status") == "passed" and sid in details for sid in shapes))
    latencies = {}
    performance = (raw.get("performance") or {}).get("shapes") or {}
    for sid in shapes:
        measured = performance.get(sid) or {}
        values = [sample["end_to_end_time_ms"] for sample in measured.get("samples", [])
                  if isinstance(sample, dict) and type(sample.get("end_to_end_time_ms")) in (int, float)
                  and math.isfinite(sample["end_to_end_time_ms"]) and sample["end_to_end_time_ms"] > 0]
        if values and not measured.get("error"):
            latencies[sid] = statistics.median(values) * 1000
    correct = bool(correct and len(latencies) == len(shapes) and shapes)
    result = {"all_pass": correct, "latency_us_by_shape": latencies,
              "latency_us_geomean": math.exp(statistics.mean(math.log(value) for value in latencies.values())) if correct else None,
              "error": None if correct else "compile, correctness, or performance did not pass"}
    result["correctness"] = correctness_summary(raw, passed=correct)
    from .feedback import evaluation_feedback
    result.update(evaluation_feedback(raw, shapes))
    return result


def parse_native_abba(job: dict, repeats: int, shapes: list[str]) -> dict:
    """repeats counts measurements per side; each SDK block contains two."""
    if type(repeats) is not int or repeats < 2 or repeats % 2:
        raise ValueError("Native ABBA requires a positive even per-side repeat count")
    if not shapes or len(shapes) != len(set(shapes)) or job.get("status") != "succeeded" or job.get("error"):
        raise ValueError("Native ABBA job did not complete")
    from .device_failures import embedded_device_failure
    if embedded_device_failure(job):
        raise ValueError("Native ABBA device unavailability is not a candidate verdict")
    result = job.get("result") or {}
    blocks = (result.get("abba") or {}).get("sdk_results")
    if not isinstance(blocks, list) or len(blocks) != repeats // 2:
        raise ValueError("Native ABBA result lacks complete SDK blocks")
    expected = [("baseline", "A", 0), ("candidate", "B", 0),
                ("candidate", "B", 1), ("baseline", "A", 1)]
    runs = []
    for block_index, block in enumerate(blocks):
        rows = (block.get("abba") or {}).get("runs") if isinstance(block, dict) else None
        if not isinstance(rows, list) or len(rows) != 4:
            raise ValueError("Native ABBA block lacks four runs")
        for index, (row, (revision, label, repeat)) in enumerate(zip(rows, expected)):
            if (not isinstance(row, dict) or any(type(row.get(key)) is not type(value) or row.get(key) != value
                    for key, value in {"index": index, "revision": revision, "label": label, "repeat": repeat}.items())
                    or not isinstance(row.get("result"), dict)):
                raise ValueError("Native ABBA run order or evaluator evidence differs from the request")
            runs.append({"revision": "incumbent" if revision == "baseline" else "candidate",
                         "repeat": block_index * 2 + repeat, "exit_code": 0,
                         "result": summarize(row["result"], shapes)})
    return {"schema_version": 1, "transport": "agate_native_eval_abba", "runs": runs, "error": None}


def interpret_payloads(payloads: list[dict], shapes: list[str], repeats: int, repetitions: int = 1) -> VerificationResult:
    """Merge batches in each repetition, then median each Shape before GeoMean."""
    if not payloads or len(payloads) % repetitions:
        raise ValueError("Native ABBA lacks a complete measurement repetition")
    schedule = verification_schedule(repeats)
    batch_count = len(payloads) // repetitions
    samples, runs = [], []
    for outer in range(repetitions):
        batches = payloads[outer * batch_count:(outer + 1) * batch_count]
        sides = {"incumbent": [], "candidate": []}
        for index, step in enumerate(schedule):
            by_shape, measured = {}, []
            for batch in batches:
                rows = batch.get("runs", [])
                if len(rows) != len(schedule) or any(
                        {"revision": row.get("revision"), "repeat": row.get("repeat")} != expected
                        for row, expected in zip(rows, schedule)):
                    raise ValueError("Native ABBA batch schedule differs")
                result = rows[index]["result"]
                values = result.get("latency_us_by_shape", {})
                if set(values).intersection(by_shape):
                    raise ValueError("Native ABBA batches contain duplicate Shapes")
                by_shape.update(values)
                measured.append(result)
            passed = set(by_shape) == set(shapes) and all(value.get("all_pass") is True for value in measured)
            merged = {"all_pass": passed, "latency_us_by_shape": by_shape,
                      "latency_us_geomean": math.exp(statistics.mean(math.log(value) for value in by_shape.values())) if passed else None,
                      "correctness": correctness_summary(measured, passed=passed)}
            runs.append(VerificationRun(str(step["revision"]), outer * repeats + int(step["repeat"]), 0, merged))
            sides[str(step["revision"])].append(merged)
        if any(row.get("all_pass") is not True for values in sides.values() for row in values):
            return VerificationResult("FAIL", None, None, None, runs=runs,
                error="Native ABBA recorded a completed compile, correctness, or performance failure")
        def side_metrics(values):
            by_shape = {sid: math.exp(statistics.mean(math.log(row["latency_us_by_shape"][sid]) for row in values)) for sid in shapes}
            return {"all_pass": True, "latency_us_by_shape": by_shape,
                    "latency_us_geomean": math.exp(statistics.mean(math.log(value) for value in by_shape.values())),
                    "correctness": correctness_summary(values, passed=True)}
        baseline = side_metrics(sides["incumbent"])
        candidate = side_metrics(sides["candidate"])
        samples.append({"correct": True, "baseline": baseline, "candidate": candidate})
    combined = aggregate(samples, "same_allocation_abba")
    baseline, candidate = combined["baseline"]["latency_us_geomean"], combined["candidate"]["latency_us_geomean"]
    return VerificationResult("PASS", candidate, baseline, (baseline / candidate - 1) * 100, runs=runs,
        candidate_latency_us_by_shape=combined["candidate"]["latency_us_by_shape"],
        incumbent_latency_us_by_shape=combined["baseline"]["latency_us_by_shape"])
