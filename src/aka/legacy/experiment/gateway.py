"""Experiment entry for Legacy Agent measurement commands and old CLI aliases."""
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
from aka.contracts.execution import CommandSpec
from aka.legacy.evaluation.gateway_command import GatewayCommandEvaluator
from aka.legacy.execution.gateway import sandbox as transport
from aka.legacy.execution.legacy.runtime import transport_arguments, ExecutionHooks, SandboxRequest
from aka.legacy.execution.processes import spawn_owned_session
from aka.legacy.execution.configuration import queue_wait_grace
from aka.legacy.execution.recovery import raise_if_environment_blocked
from aka.legacy.workflow.records import WorkflowMeasurementReader
from .runner import LegacyExperiment
from .file_records import FileExperimentRecords


def run_sandbox(executor, request, *, hooks):
    from aka.legacy.execution.selection import _completed_process
    binder = getattr(executor, "bind_sandbox", None)
    if binder is not None:
        executor = binder(replace(request, launcher=(sys.executable, "-m", "aka.legacy.execution.gateway.sandbox")), hooks)
    elif any((request.profile, request.url, request.ssh, request.ssh_init, request.health_command, request.wall_timeout is not None)):
        from aka.contracts.execution import UnsupportedExecution
        raise UnsupportedExecution("selected executor owns endpoint configuration")
    environment = dict(request.environment)
    for key, value in (("ATREX_SANDBOX_SSH", request.ssh), ("ATREX_SANDBOX_URL", request.url),
                       ("ATREX_SANDBOX_PROFILE", request.profile)):
        if any((request.ssh, request.url, request.profile)):
            environment[key] = value
    command = CommandSpec(request.workspace, request.command, request.timeout, environment, request.hardware,
        request.sync, "preflight" if request.preflight else request.gateway_kind, request.cancel_event)
    records = FileExperimentRecords(request.workspace.resolve().parent / ".aka-experiments", str(request.workspace.resolve()))
    result = LegacyExperiment(executor, records=records, command_evaluator=GatewayCommandEvaluator(
        measurement_reader=WorkflowMeasurementReader())).run_command(command)
    result.require_transport()
    return _completed_process(command, result)


def main(argv=None):
    from aka.legacy.execution.selection import execution_scope, execution_environment
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
    request = SandboxRequest(Path(args.workspace).resolve(), (sys.executable, "-m", "aka.legacy.execution.gateway.sandbox"),
        args.hardware, args.gateway_profile or "", args.url or "", args.timeout,
        tuple(transport._command_parts(args.command)), environment, args.ssh or "", args.ssh_init,
        args.health_command if args.ssh else "", tuple([] if args.no_sync else args.sync or transport.DEFAULT_SYNC_PATHS),
        gateway_kind=args.kind, preflight=args.preflight or args.check_health)
    hooks = ExecutionHooks(spawn_owned_session, raise_if_environment_blocked, queue_wait_grace)
    from aka.legacy.task.problem.inputs import PerformanceBaselineError
    try:
        with execution_scope() as executor, transport_arguments(tuple(extra)):
            result = run_sandbox(executor, request, hooks=hooks)
    except PerformanceBaselineError as error:
        print(f"[sandbox] invalid performance baseline: {error.reason}; GPU submission skipped", file=sys.stderr)
        return 2
    sys.stdout.write(result.stdout or "")
    sys.stderr.write(result.stderr or "")
    return result.returncode
