"""Experiment entry for Legacy Agent measurement commands and old CLI aliases."""
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
from aka.contracts.execution import CommandSpec, ProfileOptions, OutputOptions, UnsupportedExecution
from aka.evaluation.gateway_command import GatewayCommandEvaluator
from aka.execution.gateway import sandbox as transport
from aka.execution.legacy.runtime import transport_arguments, ExecutionHooks, SandboxRequest
from aka.execution.processes import spawn_owned_session
from aka.execution.configuration import queue_wait_grace
from aka.execution.recovery import raise_if_environment_blocked
from aka.legacy.workflow.records import WorkflowMeasurementReader
from .runner import LegacyExperiment
from .file_records import FileExperimentRecords


def run_sandbox(executor, request, *, hooks):
    from aka.execution.selection import _completed_process
    binder = getattr(executor, "bind_sandbox", None)
    if binder is not None:
        executor = binder(replace(request, launcher=(sys.executable, "-m", "aka.execution.gateway.sandbox")), hooks)
    elif any((request.profile, request.url, request.ssh, request.ssh_init, request.health_command, request.wall_timeout is not None)):
        from aka.contracts.execution import UnsupportedExecution
        raise UnsupportedExecution("selected executor owns endpoint configuration")
    environment = dict(request.environment)
    for key, value in (("ATREX_SANDBOX_SSH", request.ssh), ("ATREX_SANDBOX_URL", request.url),
                       ("ATREX_SANDBOX_PROFILE", request.profile)):
        if any((request.ssh, request.url, request.profile)):
            environment[key] = value
    from aka.evaluation.command_inputs import legacy_command_environment
    from aka.execution.legacy.runtime import current_transport_arguments
    # The bound request owns the original host launch environment. Only explicit
    # workload variables and adapter configuration belong to a public command.
    arguments = transport.build_parser().parse_args(list(current_transport_arguments()))
    environment = legacy_command_environment(environment, arguments.env)
    kind = "preflight" if request.preflight else transport._requested_gateway_kind(request.gateway_kind, list(request.command))
    profile = None
    if (arguments.profile_level != "sol" or arguments.profiler is not None or arguments.profile_counter
            or arguments.kernel_regex is not None or arguments.top_kernels is not None):
        if kind != "profile":
            raise UnsupportedExecution("profile options require a profile command")
        profile = ProfileOptions(arguments.profile_level, arguments.profiler, tuple(arguments.profile_counter),
                                 arguments.kernel_regex, arguments.top_kernels)
    output = None
    byte_limit = getattr(arguments, "max_output_file_bytes", None)
    if (arguments.inline_output is not None or arguments.include_raw_profile is not None
            or byte_limit is not None or arguments.max_output_file_mb != 512):
        output = OutputOptions(arguments.inline_output, arguments.include_raw_profile,
            byte_limit if byte_limit is not None else
                arguments.max_output_file_mb * 1024 * 1024 if arguments.max_output_file_mb != 512 else None)
    inputs = ()
    if arguments.input:
        from aka.execution.command_transport import freeze_cli_inputs
        inputs, skipped = freeze_cli_inputs(request.workspace, arguments.input,
                                           arguments.max_input_file_mb * 1024 * 1024)
        if skipped:
            print("[sandbox] inputs skipped: " + ", ".join(skipped), file=sys.stderr)
    command = CommandSpec(request.workspace, request.command, request.timeout, environment, request.hardware,
        request.sync, kind if profile is not None else "preflight" if request.preflight else request.gateway_kind,
        request.cancel_event, submission=request.submission, num_gpus=request.num_gpus,
        on_observation=request.on_observation, filesystem_access=getattr(request, "filesystem_access", None),
        profile_options=profile, input_files=inputs, output_options=output)
    records = FileExperimentRecords(request.workspace.resolve().parent / ".aka-experiments", str(request.workspace.resolve()))
    result = LegacyExperiment(executor, records=records, command_evaluator=GatewayCommandEvaluator(
        measurement_reader=WorkflowMeasurementReader())).run_command(command)
    result.require_transport()
    return _completed_process(command, result)


def main(argv=None):
    from aka.execution.selection import execution_scope, execution_environment
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = transport.build_parser()
    args = parser.parse_args(arguments)
    if not args.hardware:
        parser.error("--hardware or ATREX_SANDBOX_GPU is required")
    if args.no_sync and args.sync:
        parser.error("--no-sync and --sync are mutually exclusive")
    if sum(value is not None for value in (args.ssh, args.url, args.gateway_profile)) > 1:
        parser.error("sandbox endpoints are mutually exclusive")
    if args.dry_run:
        command = CommandSpec(Path(args.workspace).resolve(), tuple(transport._command_parts(args.command)),
            args.timeout, dict(os.environ), args.hardware, tuple(args.sync), args.kind)
        before = arguments[:arguments.index("--")] if "--" in arguments else arguments
        with transport_arguments(tuple(before)):
            plan = GatewayCommandEvaluator(measurement_reader=WorkflowMeasurementReader()).prepare(command)
        measurement = plan.context["measurement"]
        if measurement is not None:
            print(json.dumps({"hardware": args.hardware, "workspace": str(command.workspace),
                "kind": plan.context["kind"], "shape_count": "private" if plan.context["private"] else len(measurement.shape_ids),
                "shape_batch_count": len(plan.steps), "num_gpus": plan.context["num_gpus"],
                "mode": measurement.request.get("mode"), "options": measurement.request["options"]}, indent=2))
            return 0
        return transport.main(arguments)
    options = arguments[:arguments.index("--")] if "--" in arguments else arguments
    # Remove fields represented by SandboxRequest, leaving the full validated transport surface.
    fields = {"--hardware", "--workspace", "--kind", "--timeout", "--url", "--gateway-profile", "--ssh", "--ssh-init", "--health-command", "--sync"}
    extra = []
    index = 0
    while index < len(options):
        name = options[index].split("=", 1)[0]
        if name in fields:
            index += 1 if "=" in options[index] else 2
        elif name in {"--no-sync", "--preflight"}:
            index += 1
        else:
            extra.append(options[index]); index += 1
    environment = execution_environment(os.environ)
    request = SandboxRequest(Path(args.workspace).resolve(), (sys.executable, "-m", "aka.execution.gateway.sandbox"),
        args.hardware, args.gateway_profile or "", args.url or "", args.timeout,
        tuple(transport._command_parts(args.command)), environment, args.ssh or "", args.ssh_init,
        args.health_command if args.ssh else "", tuple([] if args.no_sync else args.sync or transport.DEFAULT_SYNC_PATHS),
        gateway_kind=args.kind, preflight=args.preflight or args.check_health)
    hooks = ExecutionHooks(spawn_owned_session, raise_if_environment_blocked, queue_wait_grace)
    from aka.task.problem.inputs import PerformanceBaselineError
    try:
        with execution_scope() as executor, transport_arguments(tuple(extra)):
            result = run_sandbox(executor, request, hooks=hooks)
    except PerformanceBaselineError as error:
        print(f"[sandbox] invalid performance baseline: {error.reason}; GPU submission skipped", file=sys.stderr)
        return 2
    sys.stdout.write(result.stdout or "")
    sys.stderr.write(result.stderr or "")
    return result.returncode
