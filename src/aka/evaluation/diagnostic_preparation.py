"""Prepare and explain a whole-tree diagnostic through the shared Dev contract."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

from aka.contracts.evaluation import PreparedEvaluation, StepEvaluation, CommandEvaluation, MeasurementArtifact
from aka.contracts.execution import CommandSpec, ExecutionFile, PreparedExecutionInputs
from aka.contracts.measurement import MeasurementIdentity, MeasurementPoint, MeasurementAssessment
from aka.contracts.workspace import RevisionContent
from .measurement import _digest
from .source_archives import source_artifact, source_parameters

COMMAND = ("python3", "__atrex_diagnostic.py", "request.json")
METHOD = "aka-source-diagnostic/1"
_PREFIXES = {"profile": "[sandbox] PROFILE_JSON=", "check": "[sandbox] CHECK_JSON=", "disassemble": "[sandbox] DISASSEMBLE_JSON="}


def command(provisional, request, source, kind, args, requirements):
    from . import source_diagnostics, diagnostic_clocks
    from .resources import ncu_profile_script
    from aka.execution.resources import profiling_resource
    if args.profiler not in {None, "ncu"} or args.disassembly_format == "isa":
        raise ValueError("whole-tree diagnostics currently require NVIDIA NCU and auto/sass/ptx")
    if kind == "profile" and args.profile_level == "deep" and not args.kernel_regex:
        raise ValueError("deep Profile requires an explicit kernel regex")
    shapes = request["reference"]["shapes"]
    shape = args.profile_shape_id or next(iter(shapes))
    if shape not in shapes:
        raise ValueError("diagnostic Shape is outside the selected Task input")
    parameters = {key: value for key, value in (("profiler", args.profiler), ("level", args.profile_level),
        ("counters", args.profile_counter), ("kernel_regex", args.kernel_regex), ("kernel_name", args.kernel_name),
        ("source", args.profile_source), ("launch_skip", args.launch_skip), ("launch_count", args.launch_count),
        ("top_kernels", args.top_kernels), ("arch", args.arch), ("sanitize", args.sanitize), ("fmt", args.disassembly_format))
        if value is not None}
    diagnostic = {"aka_evaluation_method": METHOD, "operation": kind, "entrypoint": "kernel.py", "package_root": ".",
        "runtime_requirements": [], "requirements": list(requirements), "shapes": {shape: shapes[shape]},
        "parameters": parameters, "lock_clocks": True, "timeout_s": provisional.timeout}
    files = tuple(ExecutionFile("candidate/" + name, data) for name, data in source.files.items() if data is not None)
    files += (ExecutionFile("input.py", request["reference"]["input_py"].encode()),
        ExecutionFile("request.json", json.dumps(diagnostic, allow_nan=False).encode()),
        ExecutionFile("__atrex_diagnostic.py", Path(source_diagnostics.__file__).read_bytes()),
        ExecutionFile("__atrex_abba.py", Path(diagnostic_clocks.__file__).read_bytes()),
        ExecutionFile("ncu_profile.py", ncu_profile_script().read_bytes()),
        ExecutionFile("classify_ncu.py", profiling_resource("nvidia/classify_ncu.py").read_bytes()),
        ExecutionFile("ncu_classification.py", profiling_resource("nvidia/ncu_classification.py").read_bytes()))
    return replace(provisional, command=COMMAND, operation="dev", timeout=provisional.timeout + 30, requirements=tuple(requirements),
        prepared_inputs=PreparedExecutionInputs(files, private_paths=tuple(item.path for item in files)),
        input_files=(), source_archives=(), submission=None, profile_options=None)


def matches(command):
    return command.operation == "dev" and tuple(command.command) == COMMAND and command.prepared_inputs is not None


def prepare(command):
    files = {item.path: item for item in command.prepared_inputs.files}
    request = json.loads(files["request.json"].content)
    if request.get("aka_evaluation_method") != METHOD or request.get("operation") not in _PREFIXES:
        raise ValueError("prepared diagnostic method differs from its request")
    source = RevisionContent({name[len("candidate/"):]: item.content for name, item in files.items() if name.startswith("candidate/")})
    artifact, _ = source_artifact(source)
    parameters = {"operation": request["operation"], "workload_scope": "diagnostic", "parameters": request["parameters"],
        "timeout_s": command.timeout, "hardware": command.hardware, **source_parameters(source)}
    identity = MeasurementIdentity("source-tree/" + request["operation"],
        tuple(MeasurementPoint("candidate", "sha256:" + parameters["candidate_source_sha256"], 0, sid) for sid in request["shapes"]),
        _digest({name: item.content.hex() for name, item in sorted(files.items())}), parameters,
        tuple(command.requirements), "explicit-requirements" if command.requirements else "unknown")
    return PreparedEvaluation((command,), 1, {"diagnostic": request, "artifact": artifact}, measurement_identities=(identity,))


def payload(result):
    from .source_diagnostics import DIAGNOSTIC_PREFIX
    text = result.stdout
    try:
        job = json.loads(text)
        if isinstance(job, dict) and isinstance(job.get("result"), dict):
            if isinstance(job["result"].get("passed"), bool) and job["result"].get("operation") in _PREFIXES:
                return job["result"]
            text = job["result"].get("stdout", "")
    except ValueError:
        pass
    for line in reversed(text.splitlines()):
        if line.startswith(DIAGNOSTIC_PREFIX):
            value = json.loads(line[len(DIAGNOSTIC_PREFIX):])
            if isinstance(value, dict) and isinstance(value.get("passed"), bool):
                return value
    return None


def diagnose(prepared, index, result):
    identity = prepared.measurement_identities[index]
    value = payload(result)
    complete = (result.facts.disposition == "terminal" and result.transport_status in {"succeeded", "failed"}
        and value is not None and value.get("operation") == prepared.context["diagnostic"]["operation"])
    return MeasurementAssessment(identity, "complete" if complete else "partial", "valid" if complete and value["passed"] else "unknown",
        identity.points if complete else (), () if complete else identity.points,
        comparability="comparable" if result.environment_identity and identity.dependency_policy != "unknown" else "unknown",
        environment_identity=result.environment_identity)


def observe(prepared, index, result):
    return StepEvaluation((MeasurementArtifact(f"receipt-{index:04d}.json", result.stdout.encode(), "diagnostic-receipt"),
        prepared.context["artifact"]), diagnose(prepared, index, result), workspace_files=result.files)


def interpret(prepared, results):
    result = results[0]
    value = payload(result)
    operation = prepared.context["diagnostic"]["operation"]
    if value is None or value.get("operation") != operation:
        return CommandEvaluation(result)
    stdout = _PREFIXES[operation] + json.dumps(value, allow_nan=False) + "\n"
    return CommandEvaluation(replace(result, stdout=stdout), artifacts=(MeasurementArtifact("diagnostic.json", json.dumps(value).encode(), "diagnostic"),))
