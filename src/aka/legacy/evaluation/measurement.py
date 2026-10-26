"""Freeze method inputs and interpret their evidence without scheduling a job."""
from __future__ import annotations

import hashlib
import json
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
               "eval_output", "profile_output"}
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
            points = tuple(MeasurementPoint("candidate", request.candidate_commit or "sha256:" + input_digest, 0, shape) for shape in shapes)
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
            int(step["repeat"]), shape) for step in verification_schedule(int(abba["repeats"])) for shape in shapes)
    else:
        points = tuple(MeasurementPoint("candidate", candidate, 0, shape) for shape in shapes)
    parameters = {key: value for key, value in frozen.items() if key not in {"candidate", "reference", "requirements", "env_vars", "abba"}}
    if request.get("env_vars"):
        parameters["worker_environment_digest"] = _digest(request["env_vars"])
    requirements = tuple(request.get("requirements", ()))
    policy = str(request.get("deps_mode") or ("explicit-requirements" if requirements else "unknown"))
    method = "atrex-bench/profile" if kind == "profile" else "atrex-bench/abba" if abba else "atrex-bench/eval"
    return MeasurementIdentity(method, points,
        input_digest, parameters, requirements, policy)
