"""Call the selected Agent tool; execution and records remain Experiment-owned."""
from dataclasses import asdict, replace
import hashlib
import os
from pathlib import Path
from threading import Event, Thread
import sys
import uuid

from aka.agent.tools.client import call_tool, ToolSessionError
from aka.agent.tools.protocol import ENDPOINT_ENV, TOKEN_ENV
from aka.contracts.execution import CommandSpec, OutputOptions, ProfileOptions
from aka.contracts.filesystem_access import require_path_access
from aka.execution.configuration import validate_workload_environment
from .command_access import decode_access, read_input, snapshot_inputs
from .command_host import TOOL_NAME
from .command_protocol import encode_request, decode_response
from .file_records import _job
from .scoped_publication import publish, validate_target


class CommandEndpointError(RuntimeError):
    def __init__(self, category, code, outcome=None, request_id=None):
        super().__init__("Experiment command " + category + "/" + code + "; do not resubmit")
        self.category, self.code, self.outcome, self.request_id = category, code, outcome, request_id


class _Cancellation:
    def __init__(self, inherited):
        self.inherited, self.requested = inherited, Event()

    def is_set(self):
        return self.requested.is_set() or self.inherited is not None and self.inherited.is_set()

    def set(self):
        self.requested.set()


def execute(command, *, endpoint, token, input_paths=(), max_input_file_bytes=None,
            stdout_sink=None, stderr_sink=None, on_skipped=None):
    """Submit once; observe only the same request and retain confirmed receipts on loss."""
    if not endpoint or not token:
        raise CommandEndpointError("request", "missing_session")
    if command.prepared_inputs is not None or command.submission is not None:
        raise ValueError("host-only prepared inputs cannot enter Agent command transport")
    validate_workload_environment(command.environment)
    environment = {ENDPOINT_ENV: endpoint, TOKEN_ENV: token}
    workspace = command.workspace.absolute()
    def invoke(message, inputs=(), cancel_event=None):
        return call_tool(TOOL_NAME, encode_request(message, inputs), workspace=workspace,
                         environment=environment, cancel_event=cancel_event)
    try:
        access_message, _, _, _, _ = decode_response(invoke({"action": "access"}))
    except ToolSessionError:
        raise CommandEndpointError("request", "session_unavailable") from None
    if access_message.get("error"):
        raise CommandEndpointError(**access_message["error"])
    access = decode_access(access_message["access"])
    files, skipped = snapshot_inputs(workspace, access, input_paths, max_file_bytes=max_input_file_bytes)
    if skipped and on_skipped is not None:
        on_skipped(skipped)
    files = (*command.input_files, *files)
    if len({item.path for item in files}) != len(files):
        raise ValueError("execution input paths must be unique")
    for item in files:
        if read_input(access, workspace / item.path) != (item.content, item.mode):
            raise PermissionError("input differs from its authorized snapshot")
    value = {"command": list(command.command), "timeout": command.timeout,
        "hardware": command.hardware, "operation": command.operation, "artifacts": list(command.artifacts),
        "num_gpus": command.num_gpus, "languages": list(command.languages), "requirements": list(command.requirements),
        "environment": dict(command.environment), "profile_options": asdict(command.profile_options) if command.profile_options else None,
        "output_options": asdict(command.output_options) if command.output_options else None,
        "input_files": [{"path": item.path, "mode": item.mode} for item in files]}
    request_id = uuid.uuid4().hex
    cancellation = _Cancellation(command.cancel_event)
    done, returned = Event(), {}
    confirmed, cursor = None, 0
    def run():
        try:
            returned["payload"] = invoke({"action": "run", "request_id": request_id, "command": value}, files, cancellation)
        except BaseException as error:
            returned["error"] = error
        finally:
            done.set()
    worker = Thread(target=run, daemon=True)
    worker.start()
    try:
        while True:
            try:
                message, observed, _, _, _ = decode_response(invoke({"action": "find", "request_id": request_id, "after_observation": cursor, "wait": True}))
                cursor = message.get("after_observation", cursor)
                for data in message.get("observations", ()):
                    confirmed = _job(data)
                    if command.on_observation is not None:
                        command.on_observation(confirmed)
                if observed is not None:
                    confirmed = observed
                if message.get("error"):
                    raise CommandEndpointError(**message["error"], outcome=confirmed, request_id=request_id)
                if done.is_set():
                    break
                if not message.get("observations"):
                    done.wait(0.05)
            except KeyboardInterrupt:
                cancellation.set()  # Explicit intent only; keep waiting for the original receipt.
        worker.join()
        if "error" in returned:
            error = returned["error"]
            opaque = getattr(error, "payload", None)
            if opaque is not None:
                _, received, captured_stdout, captured_stderr, _ = decode_response(opaque)
                if received is not None:
                    confirmed = replace(received, stdout=captured_stdout, stderr=captured_stderr)
            # Read-only reconciliation; never invoke run/resume after a lost reply.
            try:
                recovered, received, captured_stdout, captured_stderr, _ = decode_response(invoke({"action": "find", "request_id": request_id, "after_observation": cursor, "wait": False}))
                if recovered.get("observations"):
                    confirmed = _job(recovered["observations"][-1])
                if received is not None:
                    confirmed = replace(received, stdout=captured_stdout, stderr=captured_stderr)
            except (OSError, ValueError, ToolSessionError):
                pass
            raise CommandEndpointError("transport", "reply_unavailable", confirmed, request_id) from None
        message, result, stdout, stderr, outputs = decode_response(returned["payload"])
        if result is not None:
            confirmed = result
        if message.get("error"):
            raise CommandEndpointError(**message["error"], outcome=confirmed, request_id=request_id)
        if result is None:
            raise ValueError("Experiment command lacks a result")
        targets = []
        for output in outputs:
            if not any(Path(output.path) == Path(name) or Path(output.path).is_relative_to(Path(name)) for name in command.artifacts):
                raise PermissionError("result file is outside requested output paths")
            target = require_path_access(access, workspace / output.path, write=True)
            validate_target(target)
            receipts = [receipt for receipt in result.file_receipts if receipt.path == output.path]
            if len(receipts) != 1 or (receipts[0].mode, receipts[0].size_bytes, receipts[0].sha256) != (output.mode, len(output.content), hashlib.sha256(output.content).hexdigest()):
                raise ValueError("Experiment output integrity mismatch")
            targets.append(target)
        for output, target in zip(outputs, targets):
            publish(target, output.content, mode=output.mode)
        if stdout_sink is not None: stdout_sink(stdout)
        if stderr_sink is not None: stderr_sink(stderr)
        return replace(result, stdout=stdout if stdout_sink is None else "", stderr=stderr if stderr_sink is None else "", files=outputs)
    except CommandEndpointError:
        raise
    except (OSError, ValueError, TypeError, KeyError, ToolSessionError) as error:
        if confirmed is None and not isinstance(error, (ConnectionError, ToolSessionError)):
            raise
        category = "request" if isinstance(error, PermissionError) else "transport"
        code = "unauthorized" if isinstance(error, PermissionError) else "reply_unavailable"
        raise CommandEndpointError(category, code, confirmed, request_id) from None


def main(argv=None):
    from aka.evaluation.command_inputs import PROFILE_ENVIRONMENT_KEYS, legacy_command_environment
    from aka.execution.cli_adapter import build_parser
    from aka.execution.command_syntax import command_parts, requested_operation
    from aka.evaluation.cli_adapter import register_measurement_options
    parser = build_parser(register_measurement_options=register_measurement_options)
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(arguments)
    options = arguments[:arguments.index("--")] if "--" in arguments else arguments
    forbidden = {"--gateway-profile", "--url", "--ssh", "--ssh-init", "--health-command", "--ssh-runtime-bind",
                 "--ssh-gpu", "--runtime-health-command", "--check-health", "--dry-run",
                 "--profile", "--profiles-dir", "--patch", "--wall-timeout"}
    if any(value.split("=", 1)[0] in forbidden for value in options):
        parser.error("the host owns Execution selection and lifecycle configuration")
    if args.no_sync and args.sync:
        parser.error("--no-sync and --sync are mutually exclusive")
    if args.max_input_file_mb <= 0 or args.max_output_file_mb <= 0:
        parser.error("file byte limits must be positive")
    command = tuple(command_parts(args.command))
    operation = "preflight" if args.preflight else requested_operation(args.kind, list(command))
    profile = None
    if (args.profile_level != "sol" or args.profiler is not None or args.profile_counter
            or args.kernel_regex is not None or args.top_kernels is not None):
        if operation != "profile":
            parser.error("Profile options require a profile command")
        profile = ProfileOptions(args.profile_level, args.profiler, tuple(args.profile_counter), args.kernel_regex, args.top_kernels)
    byte_limit = args.max_output_file_bytes
    if byte_limit is None and args.max_output_file_mb != 512:
        byte_limit = args.max_output_file_mb * 1024 * 1024
    output = (OutputOptions(args.inline_output, args.include_raw_profile, byte_limit)
              if args.inline_output is not None or args.include_raw_profile is not None or byte_limit is not None else None)
    environment = {key: os.environ[key] for key in PROFILE_ENVIRONMENT_KEYS if key in os.environ}
    environment = legacy_command_environment(environment, args.env)
    request = CommandSpec(Path(args.workspace).absolute(), command, args.timeout, environment, args.hardware,
        tuple([] if args.no_sync else args.sync or ("profiles",)), operation, num_gpus=args.num_gpus,
        profile_options=profile, output_options=output,
        on_observation=lambda result: print("[experiment] observation=" + json.dumps(result.as_dict()), file=sys.stderr, flush=True))
    try:
        result = execute(request, endpoint=os.environ.get(ENDPOINT_ENV), token=os.environ.get(TOKEN_ENV),
            input_paths=args.input, max_input_file_bytes=args.max_input_file_mb * 1024 * 1024,
            stdout_sink=sys.stdout.write, stderr_sink=sys.stderr.write,
            on_skipped=lambda names: print("[sandbox] inputs skipped: " + ", ".join(names), file=sys.stderr))
        return result.returncode
    except CommandEndpointError as error:
        print("[experiment] " + error.category + "/" + error.code + "; do not resubmit", file=sys.stderr)
        return error.outcome.returncode or 1 if error.outcome else 1
    except (OSError, ValueError, TypeError, KeyError):
        print("[experiment] command_endpoint_unavailable; do not resubmit", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
