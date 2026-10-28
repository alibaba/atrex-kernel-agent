"""Call an explicit Experiment session endpoint without selecting dependencies."""
from dataclasses import asdict, replace
from contextlib import contextmanager
import codecs
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
from threading import Event, Lock, Thread

from aka.contracts.execution import CommandSpec, ExecutionFile, OutputOptions, ProfileOptions
from aka.contracts.filesystem_access import require_path_access
from aka.evaluation.command_inputs import PROFILE_ENVIRONMENT_KEYS, workload_environment, transport_environment_key, legacy_command_environment
from .command_access import decode_access, read_input, snapshot_inputs
from .command_protocol import ENDPOINT_ENV, TOKEN_ENV, receive, receive_json, send, send_json
from .file_records import _job
from .scoped_publication import publish, validate_target


class CommandEndpointError(RuntimeError):
    def __init__(self, category, code, outcome=None):
        super().__init__("Experiment command " + category + "/" + code + "; do not resubmit")
        self.category, self.code, self.outcome = category, code, outcome


def execute(command, *, endpoint, token, input_paths=(), max_input_file_bytes=None,
            stdout_sink=None, stderr_sink=None, on_skipped=None):
    """Make one submission; a lost reply is surfaced without retry or fallback."""
    confirmed = None
    def receipt(result):
        nonlocal confirmed
        confirmed = result
    def observe(result):
        receipt(result)
        if command.on_observation is not None:
            command.on_observation(result)
    try:
        return _execute(replace(command, on_observation=observe), endpoint=endpoint, token=token,
            input_paths=input_paths, max_input_file_bytes=max_input_file_bytes,
            stdout_sink=stdout_sink, stderr_sink=stderr_sink, on_skipped=on_skipped, on_receipt=receipt)
    except CommandEndpointError:
        raise
    except (OSError, ValueError, TypeError, KeyError) as error:
        if confirmed is None and not isinstance(error, ConnectionError):
            raise
        category = "request" if isinstance(error, PermissionError) else "transport"
        code = "unauthorized" if isinstance(error, PermissionError) else "reply_unavailable"
        raise CommandEndpointError(category, code, confirmed) from None


@contextmanager
def _controls(writer, cancellation):
    stopped, lock = Event(), Lock()
    def cancel():
        with lock:
            send_json(writer, "cancel", {})
    def monitor():
        while not stopped.wait(0.05):
            if cancellation is not None and cancellation.is_set():
                try:
                    cancel()
                except OSError:
                    pass
                return
    worker = Thread(target=monitor, daemon=True) if cancellation is not None else None
    if worker is not None:
        worker.start()
    try:
        yield cancel
    finally:
        stopped.set()
        if worker is not None:
            worker.join()


def _execute(command, *, endpoint, token, input_paths, max_input_file_bytes,
             stdout_sink, stderr_sink, on_skipped, on_receipt):
    if not endpoint or not token:
        raise CommandEndpointError("request", "missing_session")
    workspace = command.workspace.absolute()
    if any(transport_environment_key(key) for key in command.environment):
        raise ValueError("transport configuration cannot enter workload environment")
    workload_environment(command)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.connect(endpoint)
        with connection.makefile("rb") as reader, connection.makefile("wb", buffering=0) as writer:
            send_json(writer, "session", {"version": 1, "token": token, "workspace": str(workspace)})
            channel, data = receive(reader)
            if channel == "result":
                reply = json.loads(data)
                raise CommandEndpointError(**reply["error"], outcome=_job(reply["result"]))
            if channel != "access":
                raise ValueError("invalid Experiment session reply")
            access = decode_access(json.loads(data))
            files, skipped = snapshot_inputs(workspace, access, input_paths,
                                             max_file_bytes=max_input_file_bytes)
            if skipped and on_skipped is not None:
                on_skipped(skipped)
            files = (*command.input_files, *files)
            paths = [f.path for f in files]
            if len(paths) != len(set(paths)):
                raise ValueError("execution input paths must be unique")
            for item in files:
                if read_input(access, workspace / item.path) != (item.content, item.mode):
                    raise PermissionError("input differs from its authorized snapshot")
            value = {"command": list(command.command), "timeout": command.timeout,
                "hardware": command.hardware, "operation": command.operation, "artifacts": list(command.artifacts),
                "num_gpus": command.num_gpus, "languages": list(command.languages), "requirements": list(command.requirements),
                "environment": dict(command.environment), "profile_options": asdict(command.profile_options) if command.profile_options else None,
                "output_options": asdict(command.output_options) if command.output_options else None,
                "input_files": [{"path": f.path, "mode": f.mode} for f in files]}
            send_json(writer, "command", value)
            for item in files:
                send(writer, "input", item.content)
            with _controls(writer, command.cancel_event) as cancel:
                try:
                    while True:
                        channel, data = receive(reader)
                        if channel == "observation":
                            if command.on_observation is not None:
                                command.on_observation(_job(json.loads(data)))
                            continue
                        if channel == "cancel_refused":
                            continue
                        if channel != "result":
                            raise ValueError("invalid Experiment command reply")
                        reply = json.loads(data)
                        break
                except KeyboardInterrupt:
                    cancel()
                    # Cancellation requested does not imply remote termination.
                    while True:
                        channel, data = receive(reader)
                        if channel == "observation" and command.on_observation is not None:
                            command.on_observation(_job(json.loads(data)))
                        if channel == "result":
                            reply = json.loads(data)
                            break
            result = _job(reply["result"])
            if result is not None:
                on_receipt(result)

            def text_output(expected, sink):
                collected = bytearray()
                decoder = codecs.getincrementaldecoder("utf-8")()
                def consume(channel, data):
                    if channel != expected:
                        raise ValueError("invalid Experiment output stream")
                    if sink is None:
                        collected.extend(data)
                    else:
                        sink(decoder.decode(data))
                channel, _ = receive(reader, consume=consume)
                if channel != expected:
                    raise ValueError("invalid Experiment output stream")
                if sink is not None:
                    sink(decoder.decode(b"", final=True))
                return collected.decode()

            stdout = text_output("stdout", stdout_sink)
            stderr = text_output("stderr", stderr_sink)
            outputs, targets = [], []
            for item in reply["files"]:
                channel, content = receive(reader)
                if channel != "output":
                    raise ValueError("invalid Experiment file stream")
                output = ExecutionFile(item["path"], content, item["mode"])
                if not any(Path(output.path) == Path(name) or Path(output.path).is_relative_to(Path(name))
                           for name in command.artifacts):
                    raise PermissionError("result file is outside requested output paths")
                target = require_path_access(access, workspace / output.path, write=True)
                validate_target(target)
                receipts = [r for r in result.file_receipts if r.path == output.path] if result else []
                if len(receipts) != 1 or (receipts[0].mode, receipts[0].size_bytes, receipts[0].sha256) != (
                        output.mode, len(content), hashlib.sha256(content).hexdigest()):
                    raise ValueError("Experiment output integrity mismatch")
                outputs.append(output)
                targets.append(target)
            if receive_json(reader, "complete") != {}:
                raise ValueError("invalid Experiment completion receipt")
            for output, target in zip(outputs, targets):
                publish(target, output.content, mode=output.mode)
            if result is not None:
                result = replace(result, stdout=stdout, stderr=stderr, files=tuple(outputs))
            if reply["error"] is not None:
                raise CommandEndpointError(**reply["error"], outcome=result)
            if result is None:
                raise ValueError("Experiment command lacks a result")
            return result


def main(argv=None):
    from aka.execution.gateway.sandbox import build_parser, _command_parts, _requested_gateway_kind
    parser = build_parser()
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(arguments)
    options = arguments[:arguments.index("--")] if "--" in arguments else arguments
    forbidden = {"--gateway-profile", "--url", "--ssh", "--ssh-init", "--health-command", "--ssh-runtime-bind",
                 "--ssh-gpu", "--runtime-health-command", "--check-health", "--keep-pod", "--dry-run"}
    if any(value.split("=", 1)[0] in forbidden for value in options):
        parser.error("the host owns Execution selection and lifecycle configuration")
    if args.no_sync and args.sync:
        parser.error("--no-sync and --sync are mutually exclusive")
    if args.max_input_file_mb <= 0 or args.max_output_file_mb <= 0:
        parser.error("file byte limits must be positive")
    command = tuple(_command_parts(args.command))
    operation = "preflight" if args.preflight else _requested_gateway_kind(args.kind, list(command))
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
