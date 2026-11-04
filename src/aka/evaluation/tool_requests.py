"""Convert explicit tool choices to shared command requests without execution."""
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
from pathlib import Path

from aka.contracts.execution import CommandSpec, OutputOptions, ProfileOptions, UnsupportedExecution


@dataclass(frozen=True)
class ToolRequest:
    kind: str
    request: CommandSpec | None
    args: object


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError(message)


def parse_tool_arguments(argv):
    parser = _Parser(add_help=False, allow_abbrev=False)
    parser.add_argument("--kind", choices=("auto", "run", "profile", "dev", "check", "disassemble", "env",
        "record-read", "kernel-read", "kernel-records", "kernel-pareto-frontier"), default="auto")
    for flag in ("--record-id", "--kernel-id", "--output-path", "--version", "--baseline-path", "--input-path", "--shapes-path",
                 "--kernel-name", "--kernel-regex", "--profile-shape-id", "--arch", "--sanitize", "--deps-mode", "--env-gpu"):
        parser.add_argument(flag)
    parser.add_argument("--mode", dest="evaluation_mode", choices=("full", "correctness_only"))
    parser.add_argument("--correctness-max-rel-l2", type=float)
    for flag in ("--multi-seed", "--timed-runs", "--warmup", "--shape-batch-size", "--comparison-repeats", "--comparison-run-timeout",
                 "--launch-skip", "--launch-count", "--top-kernels"):
        parser.add_argument(flag, type=int)
    for flag in ("--candidate-timeout-s", "--perf-timeout-s"):
        parser.add_argument(flag, type=float)
    for flag in ("--shape-id", "--profile-counter", "--requirement", "--input", "--sync"):
        parser.add_argument(flag, action="append", default=[])
    parser.add_argument("--env", "-e", action="append", default=[])
    parser.add_argument("--profile-level", choices=("survey", "sol", "deep"), default="sol")
    parser.add_argument("--profiler", choices=("ncu", "nsys", "rocprofv3", "acu"))
    parser.add_argument("--format", dest="disassembly_format", choices=("auto", "sass", "ptx", "isa"), default="auto")
    for flag in ("--no-sync", "--include-raw-profile", "--inline-output", "--profile-source", "--dry-run", "--env-capabilities", "--env-force"):
        parser.add_argument(flag, action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(list(argv))
    if args.command[:1] == ["--"]:
        args.command = args.command[1:]
    return args


def _relative(name):
    path = Path(name)
    if not name or path.is_absolute() or ".." in path.parts or path == Path("."):
        raise ValueError("tool input paths must be workspace-relative")
    return path.as_posix()


def request_from_argv(argv, *, workspace, evaluation_input, configuration, workspace_provider,
                      defaults=None, filesystem_access=None):
    """Use only host-selected configuration and the supplied composition policy.

    Evaluation supplies shared timing defaults; explicit controls take precedence.
    Workflows may select correctness cases through defaults.
    Endpoint, credentials, retries and execution availability stay with Execution.
    """
    from aka.execution.command_syntax import requested_operation
    from .command_inputs import workload_environment
    from .gateway_request import prepare_request, _candidate_languages
    from .private_inputs import selected_content
    args = parse_tool_arguments(argv)
    if args.kind in {"record-read", "kernel-read", "kernel-records", "kernel-pareto-frontier", "env"}:
        return ToolRequest(args.kind, None, args)
    if args.no_sync and args.sync:
        raise ValueError("--no-sync and --sync are mutually exclusive")
    if args.correctness_max_rel_l2 is not None:
        import math
        if not math.isfinite(args.correctness_max_rel_l2) or args.correctness_max_rel_l2 <= 0:
            raise ValueError("correctness threshold must be finite and positive")
        if args.kind not in {"auto", "run"} or args.command:
            raise ValueError("correctness threshold requires an implicit Evaluate request")
    workspace = Path(workspace)
    defaults = dict(defaults or {})
    kind = "run" if args.kind == "auto" and not args.command else requested_operation(args.kind, args.command)
    if args.baseline_path and (kind != "run" or args.evaluation_mode == "correctness_only"):
        raise ValueError("baseline comparison requires full run")
    if args.command and (any(value is not None for value in (args.version, args.multi_seed, args.timed_runs, args.warmup, args.candidate_timeout_s, args.perf_timeout_s)) or args.shape_id):
        raise ValueError("top-level evaluator controls require an implicit canonical command")
    if kind != "run" and any(value is not None for value in (args.warmup, args.candidate_timeout_s, args.perf_timeout_s)):
        raise ValueError("timing controls require a Run request")
    command = tuple(args.command)
    sol = evaluation_input is not None and evaluation_input.adapter == "legacy-sol-v1"
    if not command and kind == "run" and sol:
        if args.evaluation_mode == "correctness_only" or args.multi_seed is not None or any(value is not None for value in (args.timed_runs, args.warmup, args.candidate_timeout_s, args.perf_timeout_s)) or args.shape_id:
            raise UnsupportedExecution("SOL's selected evaluator owns its complete workload correctness policy")
        command = ("python3", "test_kernel.py")
    elif not command and kind == "run":
        cases = args.multi_seed + 1 if args.multi_seed is not None else defaults.get("correctness_cases", 1)
        if args.shape_id and args.multi_seed is None:
            cases = 1
        from .measurement_policy import DEFAULT_BENCH_ITERS
        timed = args.timed_runs if args.timed_runs is not None else defaults.get("bench_iters", DEFAULT_BENCH_ITERS)
        if type(cases) is not int or cases < 1 or type(timed) is not int or timed < 1:
            raise ValueError("correctness cases and timing iterations must be positive")
        command = ("python3", "test_kernel.py", "--version", args.version or "v0", "--multi-seed", str(cases - 1), "--timed-runs", str(timed))
        command += tuple(value for shape in args.shape_id for value in ("--shape-id", shape))
        for flag, value in (("--warmup", args.warmup), ("--candidate-timeout-s", args.candidate_timeout_s),
                            ("--perf-timeout-s", args.perf_timeout_s)):
            if value is not None:
                command += (flag, str(value))
    elif not command and kind == "profile":
        command = ("python3", "profile_driver.py")
    elif not command and kind == "dev":
        raise ValueError("Dev requires an explicit command")
    artifacts = tuple([] if args.no_sync else args.sync or ["profiles"])
    output = OutputOptions(True if args.inline_output else None, True if args.include_raw_profile else None)
    host_environment = dict(configuration.environment)
    explicit = dict(host_environment)
    for item in args.env:
        if "=" not in item:
            raise ValueError("--env requires KEY=VALUE")
        key, value = item.split("=", 1)
        explicit[key] = value
    provisional = CommandSpec(workspace, command, timeout=configuration.timeout, environment=explicit,
        hardware=configuration.hardware, artifacts=artifacts, operation=kind, filesystem_access=filesystem_access, output_options=output)
    environment = workload_environment(provisional)
    if sol and kind == "run" and not args.baseline_path:
        return ToolRequest(kind, provisional, args)
    inputs = ()
    if args.input:
        from aka.execution.input_snapshot import snapshot_command_inputs
        inputs, skipped = snapshot_command_inputs(workspace, tuple(_relative(value) for value in args.input),
            filesystem_access=filesystem_access, max_file_bytes=None)
        if skipped:
            raise ValueError("required tool inputs are unavailable")
    if kind == "dev" or args.command and kind not in {"check", "disassemble"}:
        options = ProfileOptions(args.profile_level, args.profiler, tuple(args.profile_counter), args.kernel_regex, args.top_kernels) if kind == "profile" else None
        return ToolRequest(kind, replace(provisional, input_files=inputs, profile_options=options), args)
    if inputs:
        raise ValueError("typed measurement inputs use --input-path/--shapes-path; --input is for Dev commands")
    source = workspace_provider.read_source(workspace)
    from .source_archives import prepare_source
    descriptor, candidate_archive, _, _ = prepare_source(source)
    archives = (candidate_archive,) if candidate_archive is not None else ()
    candidate = source.files.get('kernel.py') or source.files.get('solution.json')
    if not isinstance(candidate, bytes) or not candidate.strip():
        raise ValueError("typed operation requires nonempty candidate source")
    content = selected_content(evaluation_input)
    references = {name: item.content for name, item in content.items()}
    for name, field in ((args.input_path, "input.py"), (args.shapes_path, "shapes.json")):
        if name is not None:
            references[field] = workspace_provider.read_working(workspace, (_relative(name),)).files[_relative(name)]
    solution_bytes = workspace_provider.read_working(workspace, ("solution.json",)).files.get("solution.json")
    solution = json.loads(solution_bytes) if solution_bytes else {}
    requirements = []
    if evaluation_input is not None and evaluation_input.worker_requirements is not None:
        item = evaluation_input.worker_requirements
        text = item.source.read_text() if item.source is not None else item.text
        from aka.execution.environment_query import select_requirements
        requirements.extend(select_requirements(text, _candidate_languages(solution, candidate_sources=(candidate.decode(),))))
    requirements.extend(value for value in args.requirement if value not in requirements)
    request = prepare_request(workspace, configuration.hardware, configuration.timeout, [f"{key}={value}" for key, value in environment.items()],
        list(command), "run" if kind == "run" and args.evaluation_mode != "correctness_only" else "profile", reference_dir=evaluation_input.directory if evaluation_input is not None else workspace,
        reference_content=references, requirements=requirements, candidate_source=candidate.decode(), candidate_solution=solution,
        runtime_version=host_environment.get("ATREX_SANDBOX_ATREX_BENCH_VERSION", ""), profiler=args.profiler,
        profile_level=args.profile_level, counters=args.profile_counter, kernel_regex=args.kernel_regex, top_kernels=args.top_kernels)
    if args.deps_mode is not None:
        request["deps_mode"] = args.deps_mode
    provisional = replace(provisional, num_gpus=request["spec"].get("num_gpus", 1),
        languages=tuple(request["spec"].get("languages", ())))
    native_manifest = source.files.get('kernel.py') is None
    if native_manifest and kind in {'check', 'disassemble'}:
        raise ValueError('Compiled manifests support Evaluate and Profile; use Evaluate correctness_only for a build/correctness check')
    if candidate_archive is not None and not native_manifest and kind in {"profile", "check", "disassemble"}:
        from .diagnostic_preparation import command as diagnostic_command
        return ToolRequest(kind, diagnostic_command(provisional, request, source, kind, args, requirements), args)
    request["candidate"] = descriptor
    operation = kind
    if kind == "run":
        request["mode"] = args.evaluation_mode or "full"
        if args.evaluation_mode == "correctness_only":
            from .gateway_request import _fp4_correctness_max_rel_l2
            threshold = _fp4_correctness_max_rel_l2(request["reference"].get("metadata"), request["reference"].get("operator"))
            if threshold is not None:
                request["options"]["correctness_max_rel_l2"] = threshold
            request["runner_overrides"] = {"skip_kernel_attribution": True}
            for field in ("profiler", "level", "counters", "kernel_regex", "top_kernels"):
                request.pop(field, None)
        if args.correctness_max_rel_l2 is not None:
            request["options"]["correctness_max_rel_l2"] = args.correctness_max_rel_l2
        from .measurement_policy import runner_overrides
        request["runner_overrides"].update(runner_overrides(command, explicit, defaults.get("runner_overrides")))
        if "lock_clocks" in defaults:
            request["options"]["lock_clocks"] = defaults["lock_clocks"]
        if args.baseline_path:
            baseline_path = _relative(args.baseline_path)
            baseline = (workspace_provider.read_source(workspace / baseline_path) if (workspace / baseline_path).is_dir()
                        else workspace_provider.read_working(workspace, (baseline_path,)))
            if (workspace / baseline_path).is_dir():
                baseline_bytes = baseline.files.get('kernel.py') or baseline.files.get('solution.json')
            else:
                baseline_bytes = baseline.files[baseline_path]
            if not isinstance(baseline_bytes, bytes) or not baseline_bytes.strip():
                raise ValueError("baseline comparison requires nonempty frozen source")
            repeats = args.comparison_repeats if args.comparison_repeats is not None else configuration.repeats
            if type(repeats) is not int or not 2 <= repeats <= 16 or repeats % 2:
                raise ValueError("comparison repetitions must be even, in 2..16")
            from aka.contracts.workspace import RevisionContent
            baseline_source = baseline if (workspace / baseline_path).is_dir() else RevisionContent({"kernel.py": baseline_bytes})
            baseline_descriptor, baseline_archive, _, _ = prepare_source(baseline_source, role="baseline")
            if baseline_archive is not None:
                archives = (*archives, baseline_archive)
            request["abba"] = {"baseline": baseline_descriptor, "repeats": repeats // 2}
            available = (configuration.timeout - 30) // (2 * repeats)
            seconds = (min(configuration.per_run_timeout, available)
                       if args.comparison_run_timeout is None else args.comparison_run_timeout)
            if seconds <= 0 or seconds > available:
                raise ValueError("comparison run timeout must be positive and fit the full ABBA allocation schedule")
            request["runner_overrides"]["perf_timeout_s"] = seconds
            operation = "same_allocation_abba"
    else:
        request["options"] = {key: value for key, value in request["options"].items() if key in {"bench_iters", "timeout_s"}}
        for field, value in (("kernel_name", args.kernel_name), ("source", args.profile_source or None),
                             ("launch_skip", args.launch_skip), ("launch_count", args.launch_count)):
            if value is not None:
                request[field] = value
        if args.profile_shape_id:
            request["reference"]["shapes"] = {args.profile_shape_id: request["reference"]["shapes"][args.profile_shape_id]}
        if kind in {"check", "disassemble"}:
            first = next(iter(request["reference"]["shapes"].values()))
            request = {key: request[key] for key in ("spec", "candidate", "env_vars")}
            request.update(init_kwargs=first.get("init_kwargs", {}), timeout_s=configuration.timeout)
            if requirements:
                request["requirements"] = requirements
            if args.deps_mode is not None:
                request["deps_mode"] = args.deps_mode
            if kind == "check":
                request.update({key: value for key, value in (("arch", args.arch), ("sanitize", args.sanitize)) if value is not None})
            else:
                request["fmt"] = args.disassembly_format
    submission = json.dumps({"schema_version": 1, "kind": kind, "request": request,
        "reference_files": {name: item.decode() for name, item in references.items() if Path(name).name == name}}, allow_nan=False).encode()
    return ToolRequest(operation, replace(provisional, operation=operation, submission=submission,
        requirements=tuple(requirements), source_archives=archives,
        num_gpus=request["spec"].get("num_gpus", 1), languages=tuple(request["spec"].get("languages", ()))), args)
