"""Scoped Experiment operations borrowed by the common Agent tool transport."""
from dataclasses import dataclass, field, replace
import json
from pathlib import Path
from threading import Condition, Event, Lock

from aka.contracts.execution import CommandSpec, ExecutionFile, JobResult, OutputOptions, ProfileOptions
from aka.contracts.filesystem import FilesystemAccess
from aka.contracts.filesystem_access import require_path_access
from aka.execution.configuration import transport_environment_key, validate_workload_environment
from .command_access import access_data, client_access, read_input
from .command_protocol import decode_request, encode_response
from .file_records import _job
from .scoped_publication import validate_target

TOOL_NAME = "experiment.command"


@dataclass
class _Operation:
    workspace: Path
    cancel: object
    condition: object = field(default_factory=Condition)
    observations: list = field(default_factory=list)
    done: bool = False


class ExperimentCommandHost:
    """Operation handler only; Agent owns endpoint, tokens, admission and revocation."""
    def __init__(self, open_experiment, capabilities, *, hardware=""):
        if not callable(open_experiment):
            raise TypeError("command handler requires an Experiment context opener")
        self.open_experiment, self.capabilities, self.hardware = open_experiment, capabilities, hardware
        self._lock, self._operations, self._closed = Lock(), {}, False

    def handle(self, payload, context):
        try:
            message, inputs = decode_request(payload)
            if not isinstance(context.filesystem_access, FilesystemAccess):
                raise PermissionError("command tool requires the original filesystem scope")
            require_path_access(context.filesystem_access, context.workspace)
            action = message.get("action")
            with self._lock:
                if self._closed:
                    raise PermissionError("command handler is closed")
            if action == "access" and set(message) == {"action"}:
                return encode_response({"access": access_data(client_access(context.filesystem_access))})
            if action not in {"run", "find"} or not isinstance(message.get("request_id"), str) or not message["request_id"]:
                raise ValueError("invalid command operation identity")
            request_id = message["request_id"]
            if action == "find":
                if (set(message) != {"action", "request_id", "after_observation", "wait"} or type(message["after_observation"]) is not int
                        or message["after_observation"] < 0 or type(message["wait"]) is not bool):
                    raise ValueError("invalid observation request")
                return self._find(context, request_id, message["after_observation"], message["wait"])
            if set(message) != {"action", "request_id", "command"}:
                raise ValueError("invalid command request")
            command = self._command(context, message["command"], inputs)
            operation = _Operation(context.workspace, context.cancel_event)
            with self._lock:
                if self._closed:
                    raise PermissionError("command handler is closed")
                if request_id in self._operations:
                    raise ValueError("command request was already admitted; query its original record")
                self._operations[request_id] = operation
            def observe(result):
                data = result.as_dict()
                data["stdout"] = data["stderr"] = ""
                with operation.condition:
                    operation.observations.append(data)
                    operation.condition.notify_all()
            if self.capabilities.cancellation:
                command = replace(command, cancel_event=context.cancel_event)
            command = replace(command, on_observation=observe)
            try:
                with self.open_experiment(context.workspace, context.filesystem_access) as experiment:
                    if not callable(getattr(experiment, "run_recorded", None)) or not callable(getattr(experiment, "find", None)):
                        raise TypeError("command tools require the existing RecordedExperiment capability")
                    record = experiment.run_recorded(command, request_id=request_id)
                    result = self._completed_result(experiment, record)
                self._result_files(context, command, result)
                return encode_response({"error": None}, result)
            except BaseException as error:
                outcome = getattr(error, "outcome", None) or getattr(error, "execution_outcome", None)
                if not isinstance(outcome, JobResult):
                    with operation.condition:
                        outcome = _job(operation.observations[-1]) if operation.observations else None
                if outcome is not None:
                    outcome = replace(outcome, stdout="", stderr="", files=())
                return encode_response({"error": self._error(error)}, outcome)
            finally:
                with operation.condition:
                    operation.done = True
                    operation.condition.notify_all()
        except Exception as error:
            return encode_response({"error": self._error(error)})

    @staticmethod
    def _record_scope(record, context):
        if record.kind != "command" or Path(record.request["workspace"]) != context.workspace.resolve():
            raise PermissionError("command record belongs to a different workspace")

    def _find(self, context, request_id, after, wait):
        with self._lock:
            operation = self._operations.get(request_id)
        if operation is not None:
            if operation.workspace != context.workspace:
                raise PermissionError("command request belongs to a different workspace")
            with operation.condition:
                while wait and not operation.done and len(operation.observations) <= after and not context.cancel_event.is_set():
                    operation.condition.wait(0.05)
                observations = operation.observations[after:]
                cursor, done = len(operation.observations), operation.done
            if done:
                with self._lock:
                    if self._operations.get(request_id) is operation:
                        self._operations.pop(request_id, None)
            if observations or not done:
                return encode_response({"observations": observations, "after_observation": cursor, "pending": True, "error": None})
        with self.open_experiment(context.workspace, context.filesystem_access) as experiment:
            try:
                record = experiment.find(request_id)
            except LookupError:
                return encode_response({"observations": [], "after_observation": after, "pending": True, "error": None})
            self._record_scope(record, context)
            try:
                result = self._completed_result(experiment, record)
            except LookupError:
                latest = next((attempt.result for attempt in reversed(record.attempts) if attempt.result is not None), None)
                if record.failure is not None:
                    names = {"ExecutionPending": ("execution", "pending"), "ExecutionUnavailable": ("execution", "unavailable"),
                        "UnsupportedExecution": ("execution", "unsupported"), "EvaluationUnavailable": ("evaluation", "unavailable"),
                        "ExperimentRecordError": ("record", "unavailable"), "PermissionError": ("request", "unauthorized"),
                        "ValueError": ("request", "invalid_request"), "TypeError": ("request", "invalid_request")}
                    category, code = names.get(record.failure.exception_type, ("experiment", "operation_failed"))
                    latest = replace(latest, stdout="", stderr="", files=()) if latest is not None else None
                    return encode_response({"observations": [], "after_observation": after, "pending": True,
                        "error": {"category": category, "code": code}}, latest)
                if operation is None and latest is not None:
                    data = latest.as_dict(); data["stdout"] = data["stderr"] = ""
                    return encode_response({"observations": [data], "after_observation": after + 1, "pending": True, "error": None})
                return encode_response({"observations": [], "after_observation": after, "pending": True, "error": None})
            return encode_response({"observations": [], "after_observation": after, "pending": record.phase != "complete", "error": None}, result)

    @staticmethod
    def _completed_result(experiment, record):
        artifact = next((artifact for artifact in reversed(record.artifacts) if artifact.role == "command_result"), None)
        if artifact is None:
            raise LookupError("command final receipt is not recorded")
        result = _job(json.loads(experiment.read_artifact(artifact)))
        if result is None:
            raise ValueError("command receipt is invalid")
        files = []
        for receipt in result.file_receipts:
            content = next((artifact for artifact in record.artifacts if artifact.sha256 == receipt.sha256 and artifact.size_bytes == receipt.size_bytes), None)
            if content is None:
                raise ValueError("command output artifact is unavailable")
            files.append(ExecutionFile(receipt.path, experiment.read_artifact(content), receipt.mode))
        return replace(result, files=tuple(files))

    def _command(self, session, value, inputs):
        allowed = {"command", "timeout", "hardware", "operation", "artifacts", "num_gpus", "languages",
                   "requirements", "environment", "profile_options", "output_options", "input_files"}
        if not isinstance(value, dict) or set(value) - allowed:
            raise ValueError("command transport accepts only public inputs")
        if value.get("hardware") not in (None, "", self.hardware):
            raise ValueError("Agent cannot change selected hardware")
        if value.get("requirements"):
            raise ValueError("Agent cannot change Task runtime requirements")
        for name in ("command", "artifacts", "languages", "requirements"):
            if not isinstance(value.get(name, []), list) or any(not isinstance(v, str) for v in value.get(name, [])):
                raise ValueError("command inputs must contain strings")
        environment = value.get("environment", {})
        if (not isinstance(environment, dict) or any(not isinstance(k, str) or not isinstance(v, str)
                or transport_environment_key(k) for k, v in environment.items())):
            raise ValueError("transport configuration cannot enter workload environment")
        command = CommandSpec(session.workspace, tuple(value.get("command", ())),
            timeout=value.get("timeout", 600), hardware=self.hardware, operation=value.get("operation", "run"),
            artifacts=tuple(value.get("artifacts", ())), environment=environment,
            num_gpus=value.get("num_gpus", 1), languages=tuple(value.get("languages", ())),
            filesystem_access=session.filesystem_access, input_files=inputs,
            profile_options=ProfileOptions(**value["profile_options"]) if value.get("profile_options") is not None else None,
            output_options=OutputOptions(**value["output_options"]) if value.get("output_options") is not None else None)
        validate_workload_environment(command.environment)
        self.capabilities.require(command)
        for item in inputs:
            frozen = read_input(session.filesystem_access, session.workspace / item.path)
            if frozen != (item.content, item.mode):
                raise PermissionError("input differs from its authorized snapshot")
        for name in command.artifacts:
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts or relative == Path("."):
                raise ValueError("command output must be workspace-relative")
            target = require_path_access(session.filesystem_access, session.workspace / relative, write=True)
            validate_target(target / ".command-output" if target.is_dir() else target)
        return command

    @staticmethod
    def _result_files(session, command, result):
        for item in result.files:
            relative = Path(item.path)
            if not any(relative == Path(name) or relative.is_relative_to(Path(name)) for name in command.artifacts):
                raise PermissionError("result file is outside requested output paths")
            require_path_access(session.filesystem_access, session.workspace / relative)

    @staticmethod
    def _error(error):
        from aka.contracts.execution import ExecutionPending, ExecutionUnavailable, UnsupportedExecution
        from aka.contracts.evaluation import EvaluationUnavailable
        from aka.contracts.experiment import ExperimentRecordError
        if isinstance(error, UnsupportedExecution):
            return {"category": "execution", "code": "unsupported"}
        if isinstance(error, ExecutionPending):
            return {"category": "execution", "code": "pending"}
        if isinstance(error, ExecutionUnavailable):
            return {"category": "execution", "code": "unavailable"}
        if isinstance(error, EvaluationUnavailable):
            return {"category": "evaluation", "code": "unavailable"}
        if isinstance(error, ExperimentRecordError):
            return {"category": "record", "code": "unavailable"}
        if isinstance(error, PermissionError):
            return {"category": "request", "code": "unauthorized"}
        if isinstance(error, (ValueError, TypeError, KeyError)):
            return {"category": "request", "code": "invalid_request"}
        return {"category": "experiment", "code": "operation_failed"}

    def close(self):
        with self._lock:
            self._closed = True
            operations = tuple(self._operations.values())
        for operation in operations:
            if self.capabilities.cancellation:
                operation.cancel.set()
            with operation.condition:
                operation.condition.notify_all()
