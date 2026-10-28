"""Freeze a supplemental suite and interpret its original per-case receipts."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import uuid

from aka.contracts.evaluation import (
    EvaluationFailure, MeasurementArtifact, NumericalProbeResult,
    PreparationFile, PreparedEvaluation, VerificationResult,
)
from aka.contracts.execution import CommandSpec
from aka.contracts.measurement import MeasurementAssessment, MeasurementDiagnostic, MeasurementIdentity, MeasurementPoint
from .drivers import remote_numerical
from .measurement import _environment_parameters, _requirements
from .numerical import interpret_numerical_output

SCHEMA = "aka.evaluation.numerical-probe.v1"


@dataclass(frozen=True)
class NumericalPlan:
    request: object
    suite: dict
    shapes: dict
    schedule: tuple[dict, ...]
    step_cases: tuple[int, ...]
    identities: tuple[MeasurementIdentity, ...]


def _sha(content):
    return hashlib.sha256(content).hexdigest()


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _fp4(value):
    if isinstance(value, str):
        value = value.casefold().replace("-", "").replace("_", "")
        return "fp4" in value or "float4" in value
    return False


def _metadata_fp4(value):
    if isinstance(value, dict):
        return any(_fp4(value.get(key)) for key in ("dtype", "dtype_compute")) or any(_metadata_fp4(item) for item in value.values())
    return isinstance(value, list) and any(_metadata_fp4(item) for item in value)


def _destination(workspace, relative):
    root = workspace.resolve()
    target = root / relative
    if target.is_symlink() or target.exists() and not target.is_file():
        raise ValueError("numerical snapshot destination is not an ordinary file")
    for parent in target.parents:
        if parent == root:
            break
        if parent.is_symlink() or parent.exists() and not parent.is_dir():
            raise ValueError("numerical snapshot destination has an unsafe parent")


def prepare_numerical(request, configuration, evaluation_input, *, output_root):
    if evaluation_input is None or evaluation_input.harness is None:
        raise ValueError("numerical preparation requires the frozen Task harness")
    probe = request.numerical_probe
    for identity in (probe.rotation, probe.candidate_identity):
        if len(identity) != 64 or any(char not in "0123456789abcdef" for char in identity):
            raise ValueError("numerical input requires lowercase SHA256 identities")
    suite = remote_numerical.validate_suite(json.loads(probe.suite_json))
    reference = configuration.private_reference_dir or request.workspace
    cache = {}
    def read(path):
        key = path.resolve()
        if key not in cache:
            cache[key] = path.read_bytes()
        return cache[key]
    metadata_path = reference / "metadata.json"
    metadata_content = read(metadata_path) if metadata_path.is_file() else None
    metadata = json.loads(metadata_content.decode("utf-8")) if metadata_content is not None else None
    basename_fp4 = _fp4(reference.name)
    limit = 0.2 if _metadata_fp4(metadata) or basename_fp4 else remote_numerical.MAX_REL_L2
    shapes_content = read(reference / "shapes.json")
    shapes = json.loads(shapes_content.decode("utf-8"))
    if not isinstance(shapes, dict) or any(not isinstance(key, str) or not key or not isinstance(value, dict) for key, value in shapes.items()):
        raise ValueError("numerical validation requires a Shape object")
    schedule = tuple(remote_numerical.validation_schedule(suite, shapes, probe.rotation, "thorough"))
    protected = {"input.py", "reference.py", "shapes.json", "metadata.json", "roofline.json", "agent_problem.json", "test_kernel.py", "README.md", "CLAUDE.md", "profile_driver.py", "definition.json", "workload.jsonl"}
    if evaluation_input is not None:
        protected.update(item.path for item in (*evaluation_input.files, evaluation_input.harness, evaluation_input.worker_requirements) if item is not None)
    protected_roots = {".git", Path(output_root).parts[0]}
    if evaluation_input is not None and evaluation_input.runtime_root is not None:
        protected_roots.add("atrex-bench")
    rows = []
    for item in probe.candidate_files:
        path = PurePosixPath(item.path)
        if path.as_posix() != item.path or "\\" in item.path or path.parts[0] in protected_roots or item.path in protected:
            raise ValueError("numerical candidate path must be canonical and cannot replace trusted inputs")
        _destination(request.workspace, item.path)
        rows.append({"path": item.path, "mode": item.mode, "sha256": _sha(item.content)})
    rows.sort(key=lambda row: row["path"])
    paths = {row["path"] for row in rows}
    if any(parent.as_posix() in paths | protected for path in paths for parent in PurePosixPath(path).parents):
        raise ValueError("numerical candidate files have conflicting parent paths")
    if len({row["path"] for row in rows}) != len(rows) or _sha(_json(rows)) != probe.candidate_identity:
        raise ValueError("numerical candidate content differs from its declared identity")
    directory = f"{output_root}/{uuid.uuid4().hex}"
    inputs = []
    candidate_manifest, trusted_manifest = [], []
    def snapshot(item, role, manifest):
        relative = f"snapshots/{role}/{len(manifest):04d}.bin"
        inputs.append(PreparationFile(f"{directory}/{relative}", item.content, item.mode, item.content_type))
        manifest.append({"path": item.path, "snapshot": relative, "mode": item.mode, "sha256": _sha(item.content)})
    for item in sorted(probe.candidate_files, key=lambda item: item.path):
        snapshot(item, "candidate", candidate_manifest)
    trusted = {"shapes.json": PreparationFile("shapes.json", shapes_content, content_type="application/json")}
    for name in ("input.py", "reference.py"):
        path = reference / name if (reference / name).is_file() else request.workspace / name
        trusted[name] = PreparationFile(name, read(path), stat.S_IMODE(path.stat().st_mode) & 0o777, "text/x-python")
    if metadata_content is not None:
        trusted["metadata.json"] = PreparationFile("metadata.json", metadata_content, content_type="application/json")
    if evaluation_input is not None:
        for item in (*evaluation_input.files, evaluation_input.harness):
            if item is not None and item.path not in trusted:
                content = read(item.source) if item.source is not None else item.text.encode("utf-8")
                trusted[item.path] = PreparationFile(item.path, content)
    if "test_kernel.py" not in trusted:
        trusted["test_kernel.py"] = PreparationFile("test_kernel.py", read(request.workspace / "test_kernel.py"))
    for item in sorted(trusted.values(), key=lambda item: item.path):
        _destination(request.workspace, item.path)
        snapshot(item, "trusted", trusted_manifest)
    driver_path = Path(remote_numerical.__file__)
    driver = PreparationFile(f"{directory}/test_kernel.py", read(driver_path), content_type="text/x-python")
    inputs.append(driver)
    requirements, dependency_policy = _requirements(configuration, evaluation_input)
    parameters = {
        "comparison": {"metric": "relative_l2", "max_rel_l2": limit},
        "rotation": probe.rotation, "candidate_identity": probe.candidate_identity,
        "suite_sha256": _sha(probe.suite_json.encode("utf-8")), "per_case_timeout": 540,
        "metadata_present": metadata_content is not None,
        "metadata_sha256": _sha(metadata_content) if metadata_content is not None else None,
        "basename_fp4": basename_fp4, "world_size": suite["world_size"],
        "hardware": configuration.hardware, "num_gpus": 1,
        "runtime_configuration": _environment_parameters(configuration.environment),
    }
    frozen = {item.path.removeprefix(directory + "/"): _sha(item.content) for item in inputs}
    task_runtime, task_content = {}, {}
    if evaluation_input is not None:
        for item in (*evaluation_input.files, evaluation_input.harness, evaluation_input.worker_requirements):
            if item is not None:
                task_content[item.path] = _sha(read(item.source) if item.source is not None else item.text.encode("utf-8"))
    if evaluation_input is not None and evaluation_input.runtime_root is not None:
        runtime = evaluation_input.runtime_root
        for subdirectory in ("scripts", "src"):
            for path in sorted((runtime / subdirectory).rglob("*")):
                if path.is_file() and "__pycache__" not in path.parts:
                    task_runtime[path.relative_to(runtime).as_posix()] = _sha(read(path))
    identities, steps, step_cases, retry_keys = [], [], [], []
    for index, plan in enumerate(schedule):
        value = {"suite": suite, "case_ids": [plan["case_id"]], "rotation": probe.rotation,
            "mode": "thorough", "per_case_timeout": 540, "max_rel_l2": limit,
            "candidate_identity": probe.candidate_identity, "candidate_manifest": candidate_manifest,
            "trusted_manifest": trusted_manifest}
        input_digest = _sha(_json({"inputs": frozen, "request": value, "runtime": task_runtime, "task": task_content}))
        points = tuple(MeasurementPoint("candidate", "sha256:" + probe.candidate_identity, repeat, shape)
            for repeat, _ in enumerate(plan["seeds"]) for shape in plan["shape_ids"])
        identity = MeasurementIdentity("supplemental-numerical/1", points, input_digest,
            {**parameters, "case_id": plan["case_id"]}, requirements, dependency_policy)
        identities.append(identity)
        if plan.get("status") == "unsupported":
            continue
        request_path = f"{directory}/request-{index:04d}.json"
        inputs.append(PreparationFile(request_path, _json(value), content_type="application/json"))
        steps.append(CommandSpec(request.workspace, ("python3", driver.path, request_path), timeout=600, operation="dev"))
        step_cases.append(index)
        retry_keys.append(f"numerical-probe:{probe.rotation}:{plan['case_id']}")
    context = NumericalPlan(request, suite, shapes, schedule, tuple(step_cases), tuple(identities))
    return PreparedEvaluation(tuple(steps), 1, context, tuple(retry_keys),
        output_directory=Path(directory), inputs=tuple(inputs),
        measurement_identities=tuple(identities[index] for index in step_cases))


def probe_result(plan, case_index, result=None):
    case = plan.schedule[case_index]
    if case.get("status") == "unsupported":
        return NumericalProbeResult("unsupported", "unsupported", "missing", diagnosis=case["diagnosis"])
    if result is None:
        return interpret_numerical_output("", case, plan.suite, plan.shapes)
    unavailable = result.transport_status in {"timed_out", "cancelled", "interrupted", "infrastructure", "protocol", "rejected"}
    return interpret_numerical_output(result.stdout, case, plan.suite, plan.shapes,
        transport_returncode=result.returncode or int(unavailable))


def assess_numerical(plan, case_index, result=None):
    interpreted = probe_result(plan, case_index, result)
    identity = plan.identities[case_index]
    complete = (interpreted.status in {"passed", "needs_repair"} and bool(identity.points)
        and interpreted.expected_probes is not None and interpreted.observed_probes == interpreted.expected_probes)
    observed = identity.points if complete else ()
    missing = () if complete else identity.points
    validity = "valid" if interpreted.status in {"passed", "needs_repair"} else "unknown"
    diagnostics = [MeasurementDiagnostic("numerical", interpreted.reason, basis=("interpreted-numerical-receipt",))]
    if not complete and interpreted.observed_probes:
        diagnostics.append(MeasurementDiagnostic("incomplete", "numerical-observed-points-unlocated", basis=("aggregate-receipt-counts",)))
    diagnostics += [MeasurementDiagnostic("incomplete", "missing-numerical-point", point) for point in missing]
    environment = result.environment_identity if result is not None else None
    if result is not None:
        def causes(items):
            for item in items:
                yield item
                yield from causes(item.causes)
        for cause in causes(result.causes):
            diagnostics.append(MeasurementDiagnostic(cause.category, cause.code, basis=("public-execution-cause",)))
        for diagnostic in result.diagnostics:
            diagnostics.append(MeasurementDiagnostic(diagnostic.category, diagnostic.code, basis=("public-execution-diagnostic",)))
        if result.transport_status in {"timed_out", "cancelled", "interrupted", "infrastructure", "protocol", "rejected"}:
            diagnostics.append(MeasurementDiagnostic(result.transport_status, "numerical-execution-unavailable"))
            validity = "unknown"
    return MeasurementAssessment(identity, "complete" if complete else "unknown",
        validity, observed, missing, tuple(diagnostics),
        "comparable" if environment and identity.dependency_policy != "unknown" else "unknown", environment)


def interpret_numerical(prepared, results):
    plan = prepared.context
    if len(results) > len(prepared.steps):
        raise ValueError("numerical evaluation received unexpected step results")
    observed = dict(zip(plan.step_cases, results))
    probes = [{"case_id": case["case_id"], **(fact := probe_result(plan, index, observed.get(index))).as_dict(),
               "receipt_status": fact.receipt_status} for index, case in enumerate(plan.schedule)]
    assessments = tuple(assess_numerical(plan, index, observed.get(index)) for index in range(len(plan.schedule)))
    reference_probes = [probe for probe in probes if {"reference", "unknown"}.intersection(probe["nonfinite_outputs"])]
    reference_failure = bool(reference_probes)
    status = ("needs_validation" if reference_failure else "needs_repair" if any(probe["status"] == "needs_repair" for probe in probes)
        else "needs_validation" if any(probe["status"] == "needs_validation" for probe in probes)
        else "advisory" if any(probe["status"] == "unsupported" for probe in probes) else "passed")
    reason = ("nonfinite_role_unknown" if "unknown" in reference_probes[0]["nonfinite_outputs"] else "reference_nonfinite") if reference_failure else next(
        (probe["reason"] for probe in probes if probe["status"] == status), "unsupported" if status == "advisory" else status)
    counted = [probe for probe in probes if probe["status"] != "unsupported"]
    def count(name):
        values = [probe[name] for probe in counted]
        return sum(values) if values and all(type(value) is int and value >= 0 for value in values) else None
    summary = {"status": status, "reason": reason, "expected_probes": count("expected_probes"),
        "observed_probes": count("observed_probes"), "failed_probes": count("failed_probes")}
    payload = {"schema": SCHEMA, "summary": summary, "probes": probes,
        "measurements": [item.as_dict() for item in assessments]}
    failure = None if status == "passed" else EvaluationFailure("numerical" if status == "needs_repair" else "incomplete",
        reason, "supplemental numerical candidate failure" if status == "needs_repair" else "supplemental numerical evidence unavailable",
        candidate_repairable=status == "needs_repair")
    return VerificationResult("PASS" if status == "passed" else "FAIL" if status == "needs_repair" else "ERROR",
        None, None, None, error=failure.message if failure else "", failure=failure,
        artifact="numerical_result.json", artifacts=(MeasurementArtifact("numerical_result.json", _json(payload), "numerical_probe"),),
        measurements=assessments)
