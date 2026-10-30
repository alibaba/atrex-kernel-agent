"""Freeze method inputs and interpret their evidence without scheduling a job."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from aka.contracts.measurement import (
    MeasurementIdentity, MeasurementPoint, MeasurementAssessment, MeasurementDiagnostic,
)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _bytes(value):
    return hashlib.sha256(value).hexdigest()


def _content_inputs(workspace, reference, excluded=(), evaluation_input=None):
    """Content names are logical roles; local checkout paths never enter identity."""
    content = {}
    skipped = {".git", ".atrex_verify", ".atrex_long_horizon", ".atrex_plugins",
               ".claude", ".agents", ".qoder", "__pycache__", "memory", "profiles",
               "plans", "gpu-wiki", "reference-projects", "skills", "verification_artifacts",
               "eval_output", "profile_output", ".mypy_cache", ".pytest_cache", ".ruff_cache",
               ".atrex_environment"}
    excluded = set(excluded)
    for path in sorted(workspace.rglob("*")):
        relative = path.relative_to(workspace)
        if path.is_file() and not skipped.intersection(relative.parts) and (
                path.suffix in {".py", ".json", ".txt", ".sh", ".so"}) and relative.as_posix() not in excluded:
            content["workspace/" + relative.as_posix()] = _bytes(path.read_bytes())
    for name in ("reference.py", "input.py", "shapes.json", "metadata.json", "roofline.json"):
        path = reference / name
        if path.is_file():
            content["reference/" + name] = _bytes(path.read_bytes())
    if evaluation_input is not None:
        for item in (*evaluation_input.files, evaluation_input.harness, evaluation_input.worker_requirements):
            if item is not None:
                data = item.source.read_bytes() if item.source is not None else item.text.encode()
                content["task/" + item.path] = _bytes(data)
        runtime = evaluation_input.runtime_root
        if runtime is not None:
            for directory in (runtime / "scripts", runtime / "src"):
                if directory.is_dir():
                    for path in sorted(directory.rglob("*")):
                        if path.is_file() and "__pycache__" not in path.parts:
                            content["runtime/" + path.relative_to(runtime).as_posix()] = _bytes(path.read_bytes())
    return content


def _requirements(configuration, evaluation_input=None):
    item = evaluation_input.worker_requirements if evaluation_input is not None else None
    if item is not None:
        data = item.source.read_bytes() if item.source is not None else item.text.encode()
        text, policy = data.decode(), "frozen-task"
    else:
        path = configuration.environment.get("ATREX_SANDBOX_REQUIREMENTS_FILE", "")
        text = Path(path).read_text() if path else ""
        policy = "explicit-requirements" if path else "unknown"
    return tuple(line.strip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")), policy


def _environment_parameters(environment):
    # Values identify potentially material runtime configuration without exposing
    # credentials or machine-local paths through an evidence document.
    keys = ("ATREX_SANDBOX_ATREX_BENCH_VERSION", "AKA_GATEWAY_WARMUP_ITERS",
            "ATREX_SANDBOX_URL", "ATREX_SANDBOX_PROFILE", "ATREX_SANDBOX_SSH",
            "ATREX_SANDBOX_SSH_INIT", "ATREX_SANDBOX_SSH_GPU",
            "ATREX_CORRECTNESS_MAX_REL_L2", "AKA_GATEWAY_DEV_DEPENDENCIES")
    return {key: _bytes(str(environment[key]).encode()) for key in keys if key in environment}


def legacy_identities(prepared, configuration, evaluation_input=None):
    plan = prepared.context
    request = getattr(plan, "request", None)
    if request is None:
        return ()  # An injected opaque plan must provide its own measurement identity.
    if not prepared.steps:
        return ()
    if getattr(plan, "gateway", None) is not None:
        result = []
        for identity in plan.gateway.measurement_identities:
            points = tuple(MeasurementPoint(point.side,
                request.candidate_commit or point.revision, point.repeat, point.shape_id)
                for point in identity.points)
            result.append(MeasurementIdentity(identity.method, points, identity.input_digest,
                identity.parameters, identity.requirements, identity.dependency_policy))
        return tuple(result)
    reference = configuration.private_reference_dir or request.workspace
    content = _content_inputs(request.workspace, reference, request.changed_paths, evaluation_input)
    requirements, policy = _requirements(configuration, evaluation_input)
    parameters = {"mode": request.mode, "version": request.version, "native": configuration.native,
                  "hardware": configuration.hardware, "repeats": configuration.repeats,
                  "per_run_timeout": configuration.per_run_timeout,
                  "command_timeout": configuration.timeout,
                  "runtime_configuration": _environment_parameters(configuration.environment)}
    parameters["measurement_repetitions"] = configuration.measurement_repetitions if request.mode in {"v0", "framework"} else 1
    result = []
    shape_path = reference / "shapes.json"
    full_shapes = tuple(str(value) for value in json.loads(shape_path.read_text())) if shape_path.is_file() else ()
    for index, command in enumerate(prepared.steps):
        frozen = {}
        for item in prepared.inputs:
            relative = Path(item.path)
            if prepared.output_directory is not None:
                relative = relative.relative_to(prepared.output_directory)
            if relative.name.startswith("request-") and relative.name != f"request-{index:04d}.json":
                continue
            frozen[relative.as_posix()] = _bytes(item.content)
        input_digest = _digest({"content": content, "prepared": frozen})
        if request.mode == "abba":
            batch = next(item for item in prepared.inputs if Path(item.path).name == f"request-{index:04d}.json")
            batch_request = json.loads(batch.content)
            argv = batch_request["command"]
            shapes = tuple(argv[position + 1] for position, token in enumerate(argv[:-1]) if token == "--shape-id") or full_shapes or (None,)
            points = tuple(MeasurementPoint(str(step["revision"]),
                (request.base_commit if step["revision"] == "incumbent" else request.candidate_commit) or "sha256:" + input_digest,
                int(step["repeat"]), shape)
                for step in plan.schedule for shape in shapes)
            method = "legacy-abba/1"
        else:
            shapes = tuple(sorted(request.expected_shape_ids)) if request.expected_shape_ids is not None else full_shapes or (None,)
            points = tuple(MeasurementPoint("candidate", request.candidate_commit or "sha256:" + input_digest, index, shape) for shape in shapes)
            method = "legacy-eval/1"
        result.append(MeasurementIdentity(method, points, input_digest,
            {**parameters, "num_gpus": command.num_gpus}, requirements, policy))
    return tuple(result)


def gateway_identity(request, reference_files, kind="run"):
    # The human job name and local sidecar paths are not measurement inputs.
    frozen = {key: value for key, value in request.items() if key not in {"name", "idempotency_key"}}
    input_digest = _digest({"request": frozen, "reference_files": reference_files})
    shapes = tuple(str(value) for value in request["reference"]["shapes"])
    candidate = "sha256:" + _bytes(request["candidate"].encode())
    abba = request.get("abba")
    if abba:
        from .abba import verification_schedule
        baseline = "sha256:" + _bytes(abba["baseline"].encode())
        points = tuple(MeasurementPoint(str(step["revision"]), baseline if step["revision"] == "incumbent" else candidate,
            int(step["repeat"]), shape) for step in verification_schedule(int(abba["repeats"]) * 2) for shape in shapes)
    else:
        points = tuple(MeasurementPoint("candidate", candidate, 0, shape) for shape in shapes)
    parameters = {key: value for key, value in frozen.items() if key not in {"candidate", "reference", "requirements", "env_vars", "abba"}}
    if abba:
        parameters["abba_sdk_blocks"] = abba["repeats"]
        parameters["comparison_method"] = "native-abba"
    if request.get("env_vars"):
        parameters["worker_environment_digest"] = _digest(request["env_vars"])
    requirements = tuple(request.get("requirements", ()))
    policy = str(request.get("deps_mode") or ("explicit-requirements" if requirements else "unknown"))
    method = "atrex-bench/profile" if kind == "profile" else "atrex-bench/abba" if abba else "atrex-bench/eval"
    return MeasurementIdentity(method, points,
        input_digest, parameters, requirements, policy)


def portable_profile_identity(command, reference):
    """Identify the unchanged wrapper method without inventing typed controls."""
    from .command_inputs import PROFILE_ENVIRONMENT_KEYS
    content = _content_inputs(command.workspace, reference)
    explicit = [{"path": item.path, "mode": item.mode, "sha256": _bytes(item.content)}
                for item in command.input_files]
    input_digest = _digest({"content": content, "explicit_inputs": explicit})
    candidate_path = command.workspace / "kernel.py"
    revision = "sha256:" + (_bytes(candidate_path.read_bytes()) if candidate_path.is_file() else input_digest)
    shape = command.environment.get("PROFILE_SHAPE_ID")
    points = (MeasurementPoint("candidate", revision, 0, shape),)
    controls = {key: _bytes(value.encode()) for key, value in command.environment.items()
                if key in PROFILE_ENVIRONMENT_KEYS}
    parameters = {"hardware": command.hardware, "num_gpus": command.num_gpus,
                  "command": list(command.command), "profile_environment": controls}
    return MeasurementIdentity("legacy-profile/1", points, input_digest, parameters,
        tuple(command.requirements), "explicit-requirements" if command.requirements else "unknown")


def assess_payload(identity, result, payload=None):
    """Interpret one physical receipt without altering the incumbent/candidate score."""
    from .abba import _result_latency, _result_performance_score
    observed = []
    diagnostics = []
    rows = payload.get("runs", ()) if isinstance(payload, dict) else ()
    if not isinstance(rows, list):
        rows = ()
    valid_rows, invalid, unresolved = [], False, bool(isinstance(payload, dict) and payload.get("error"))
    matched = set()
    expected_schedule = list(dict.fromkeys((point.side, point.repeat) for point in identity.points))
    actual_schedule = [(row.get("revision"), row.get("repeat")) for row in rows if isinstance(row, dict)]
    if actual_schedule != expected_schedule[:len(actual_schedule)]:
        diagnostics.append(MeasurementDiagnostic("protocol", "unexpected-measurement-order"))
        unresolved = True
    for row in rows:
        if not isinstance(row, dict):
            unresolved = True
            continue
        side, repeat = row.get("revision"), row.get("repeat")
        points = tuple(point for point in identity.points if point.side == side and point.repeat == repeat)
        if not isinstance(side, str) or type(repeat) is not int or not points or (side, repeat) in matched:
            diagnostics.append(MeasurementDiagnostic("protocol", "unexpected-or-duplicate-measurement-row"))
            unresolved = True
            continue
        matched.add((side, repeat))
        measured = row.get("result")
        timed_out = row.get("termination") == "timed_out" or (
            row.get("exit_code") == -1 and row.get("stderr_tail") == "evaluation timed out")
        if timed_out:
            unresolved = True
            diagnostics.extend(MeasurementDiagnostic("timeout", "abba-run-deadline", point,
                ("owned-measurement-timeout",)) for point in points)
            continue
        if not isinstance(measured, dict):
            unresolved = True
            continue
        by_shape = measured.get("latency_us_by_shape")
        for point in points:
            if point.shape_id is None or isinstance(by_shape, dict) and point.shape_id in by_shape:
                observed.append(point)
        correctness_only = identity.parameters.get("mode") == "correctness_only"
        passed = measured.get("all_pass") is True and row.get("exit_code", 0) == 0
        raw_latency = measured.get("latency_us_geomean")
        if raw_latency is None and isinstance(measured.get("performance"), dict):
            raw_latency = measured["performance"].get("latency_us")
        numeric = type(raw_latency) in (int, float) and math.isfinite(raw_latency) and raw_latency > 0
        shape_numeric = all(point.shape_id is None or isinstance(by_shape, dict)
            and type(by_shape.get(point.shape_id)) in (int, float)
            and math.isfinite(by_shape[point.shape_id]) and by_shape[point.shape_id] > 0 for point in points)
        if correctness_only and passed:
            observed.extend(point for point in points if point not in observed)
            valid_rows.append(True)
        elif passed and numeric and shape_numeric and _result_latency(measured) is not None and (
                identity.parameters.get("comparison_method") == "native-abba" or _result_performance_score(measured) is not None):
            valid_rows.append(True)
        elif measured.get("all_pass") is False:
            invalid = True
            diagnostics.extend(MeasurementDiagnostic("measurement", "measured-check-failure", point,
                ("structured-measurement-result",)) for point in points)
        else:
            unresolved = True
    missing = tuple(point for point in identity.points if point not in observed)
    diagnostics.extend(MeasurementDiagnostic("incomplete", "missing-measurement", point) for point in missing)
    def causes(items):
        for item in items:
            yield item
            yield from causes(item.causes)
    for cause in causes(getattr(result, "causes", ())):
        diagnostics.append(MeasurementDiagnostic(cause.category, cause.code, basis=("public-execution-cause",)))
        if cause.category in {"timeout", "deadline"}:
            unresolved = True
    for diagnostic in result.diagnostics:
        diagnostics.append(MeasurementDiagnostic(diagnostic.category, diagnostic.code,
            basis=("public-execution-diagnostic",)))
        if diagnostic.category in {"timeout", "deadline"}:
            unresolved = True
    status = result.transport_status
    if status in {"timed_out", "cancelled", "interrupted", "infrastructure", "protocol", "rejected"}:
        unresolved = True
        category = "timeout" if status == "timed_out" else status
        diagnostics.extend(MeasurementDiagnostic(category, "execution-" + status, point,
            ("public-execution-result",)) for point in missing)
    shape_known = bool(identity.points) and all(point.shape_id is not None for point in identity.points)
    completeness = "complete" if not missing and shape_known else "partial" if shape_known else "unknown"
    validity = "invalid" if invalid else "valid" if valid_rows and not unresolved and not missing else "unknown"
    environment = result.environment_identity
    comparability = "comparable" if environment and identity.dependency_policy != "unknown" else "unknown"
    return MeasurementAssessment(identity, completeness, validity, tuple(observed), missing,
        tuple(diagnostics), comparability, environment)
