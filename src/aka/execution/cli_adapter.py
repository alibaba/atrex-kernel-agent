"""Bind the retained command CLI without owning measurement or experiments."""
import os
from pathlib import Path
import sys

from aka.contracts.execution import CommandSpec, ExecutionContext, OutputOptions, ProfileOptions, UnsupportedExecution
from .gateway import sandbox as transport
from .legacy.runtime import ExecutionHooks, SandboxRequest, transport_arguments
from .selection import execution_environment, execution_scope, read_execution_patches, selected_execution


def _parser():
    parser = transport.build_parser()
    parser.add_argument("--profile", default=None)
    parser.add_argument("--profiles-dir", type=Path)
    parser.add_argument("--patch", type=Path, action="append", default=[])
    parser.add_argument("--wall-timeout", type=int)
    return parser


def _transport_argv(arguments):
    """Remove host selection/supervision flags from the raw transport preview."""
    result, index = [], 0
    while index < len(arguments):
        value = arguments[index]
        if value == "--":
            return [*result, *arguments[index:]]
        if value.split("=", 1)[0] in {"--profile", "--profiles-dir", "--patch", "--wall-timeout"}:
            index += 1 if "=" in value else 2
        else:
            result.append(value)
            index += 1
    return result


def _transport_options(args):
    # Public files/Profile/output fields are encoded by the selected backend.
    # Only physical CLI controls stay in this Execution-local scope.
    result = []
    for item in args.env:
        result.extend(("--env", item))
    if args.max_input_file_mb != 16:
        result.extend(("--max-input-file-mb", str(args.max_input_file_mb)))
    for item in args.ssh_runtime_bind or ():
        result.extend(("--ssh-runtime-bind", item))
    for flag, value in (("--ssh-gpu", args.ssh_gpu),
                        ("--runtime-health-command", args.runtime_health_command or None)):
        if value is not None:
            result.extend((flag, str(value)))
    for flag, enabled in (("--check-health", args.check_health), ("--keep-pod", args.keep_pod)):
        if enabled:
            result.append(flag)
    return tuple(result)


def _host_environment(args):
    environment = execution_environment(os.environ)
    explicit = {"ATREX_SANDBOX_SSH": args.ssh, "ATREX_SANDBOX_URL": args.url,
                "ATREX_SANDBOX_PROFILE": args.gateway_profile}
    if any(value is not None for value in explicit.values()):
        # Explicit endpoint selection replaces inherited endpoint choices.
        environment.update({key: value or "" for key, value in explicit.items()})
    environment["ATREX_SANDBOX_SSH_INIT"] = args.ssh_init
    if environment.get("ATREX_SANDBOX_SSH"):
        environment["ATREX_SANDBOX_HEALTH_COMMAND"] = args.health_command
    return environment


def _command(args, environment):
    command = tuple(transport._command_parts(args.command))
    profile = None
    if (args.profile_level != "sol" or args.profiler is not None or args.profile_counter
            or args.kernel_regex is not None or args.top_kernels is not None):
        profile = ProfileOptions(args.profile_level, args.profiler, tuple(args.profile_counter),
                                 args.kernel_regex, args.top_kernels)
    limit = args.max_output_file_bytes
    if limit is None and args.max_output_file_mb != 512:
        limit = args.max_output_file_mb * 1024 * 1024
    output = (OutputOptions(args.inline_output, args.include_raw_profile, limit)
              if args.inline_output is not None or args.include_raw_profile is not None or limit is not None else None)
    inputs = ()
    if args.input:
        from .command_transport import freeze_cli_inputs
        inputs, skipped = freeze_cli_inputs(Path(args.workspace).resolve(), args.input,
                                            args.max_input_file_mb * 1024 * 1024)
        if skipped and not args.dry_run:
            print("[sandbox] inputs skipped: " + ", ".join(skipped), file=sys.stderr)
    return CommandSpec(Path(args.workspace).resolve(), command, args.timeout, environment, args.hardware,
        tuple([] if args.no_sync else args.sync or transport.DEFAULT_SYNC_PATHS),
        "preflight" if args.preflight or args.check_health else args.kind,
        num_gpus=args.num_gpus, profile_options=profile, input_files=inputs, output_options=output)


def _bind(executor, command, args, environment):
    if not getattr(executor, "sandbox_cli_options", False):
        if args.keep_pod or args.ssh_runtime_bind or args.ssh_gpu is not None or args.runtime_health_command or args.check_health:
            raise UnsupportedExecution("selected executor does not support sandbox transport options")
    if getattr(executor, "sandbox_cli_endpoint_environment", False):
        if (environment.get("ATREX_SANDBOX_SSH") or args.ssh_runtime_bind or args.ssh_gpu is not None
                or args.runtime_health_command or args.check_health or args.preflight):
            raise UnsupportedExecution("gateway profile does not support SSH or health configuration")
        if args.wall_timeout is not None:
            raise UnsupportedExecution("selected executor owns its local wait budget")
        return executor.bind(ExecutionContext(environment, command.hardware, command.timeout))
    binder = getattr(executor, "bind_sandbox", None)
    if binder is None:
        if any((args.gateway_profile, args.url, args.ssh, args.ssh_init,
                environment.get("ATREX_SANDBOX_HEALTH_COMMAND"), args.wall_timeout is not None)):
            raise UnsupportedExecution("selected executor owns endpoint configuration")
        return executor.bind(ExecutionContext(environment, command.hardware, command.timeout)) if callable(getattr(executor, "bind", None)) else executor
    from .configuration import queue_wait_grace
    from .processes import spawn_owned_session
    from .recovery import raise_if_environment_blocked
    request = SandboxRequest(command.workspace, (sys.executable, "-m", "aka.execution.gateway.sandbox"),
        command.hardware, environment.get("ATREX_SANDBOX_PROFILE", ""), environment.get("ATREX_SANDBOX_URL", ""),
        command.timeout, command.command, environment, environment.get("ATREX_SANDBOX_SSH", ""),
        args.ssh_init, args.health_command if environment.get("ATREX_SANDBOX_SSH") else "", command.artifacts,
        wall_timeout=args.wall_timeout, gateway_kind=args.kind,
        preflight=command.operation == "preflight", num_gpus=command.num_gpus)
    return binder(request, ExecutionHooks(spawn_owned_session, raise_if_environment_blocked, queue_wait_grace))


def run_cli(argv=None, *, prepare_command, run_command, preview_command, health_only=False):
    """Keep host binding private; callbacks consume public commands and CLI choices."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = _parser()
    args = parser.parse_args(arguments)
    if not args.hardware:
        parser.error("--hardware or ATREX_SANDBOX_GPU is required")
    if args.no_sync and args.sync:
        parser.error("--no-sync and --sync are mutually exclusive")
    if sum(value is not None for value in (args.ssh, args.url, args.gateway_profile)) > 1:
        parser.error("sandbox endpoints are mutually exclusive")
    if args.timeout < 1 or args.shape_batch_size < 1 or (args.wall_timeout is not None and args.wall_timeout < 1):
        parser.error("command, shape batch and wall budgets must be positive")
    if args.max_input_file_mb < 1 or args.max_output_file_mb < 1 or (args.max_output_file_bytes is not None and args.max_output_file_bytes < 1):
        parser.error("file byte limits must be positive")
    health = args.preflight or args.check_health
    if health_only and not health:
        parser.error("this Execution entry requires --preflight or --check-health")
    if not transport._command_parts(args.command) and not health:
        parser.error("provide a command after -- or use --preflight")
    selection = dict(selected_execution())
    if args.profile is not None:
        selection["profile"] = args.profile
    if args.patch:
        selection["patches"] = [*selection.get("patches", ()), *read_execution_patches(args.patch)]
    if args.profiles_dir is not None:
        selection["profiles_dir"] = str(args.profiles_dir.expanduser().resolve())
    with execution_scope(selection=selection) as executor:
        environment = _host_environment(args)
        if health and not environment.get("ATREX_SANDBOX_SSH"):
            parser.error("--check-health/--preflight requires --ssh")
        command = _command(args, environment)
        bound = _bind(executor, command, args, environment)
        if not health:
            command = prepare_command(command, environment_items=tuple(args.env), shape_batch_size=args.shape_batch_size)
        with transport_arguments(_transport_options(args)):
            if args.dry_run and not health:
                if preview_command(command, keep_pod=args.keep_pod, input_requested=bool(args.input)):
                    return 0
                return transport.main(_transport_argv(arguments))
            result = bound.run(command) if health else run_command(bound, command,
                keep_pod=args.keep_pod, input_requested=bool(args.input))
        result.require_transport()
        sys.stdout.write(result.stdout or "")
        sys.stderr.write(result.stderr or "")
        return result.returncode
