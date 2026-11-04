"""Capture launcher facts over an owned pipe, independently of user output/files."""
from __future__ import annotations

from dataclasses import replace
from contextlib import contextmanager
from contextvars import ContextVar
import uuid
import json
import base64
import os
import select
import signal
import subprocess
import threading
import time

from aka.contracts.execution import (
    ExecutionAttempt, ExecutionBudget, ExecutionDiagnostic, ExecutionFacts, JobResult, ResourceIdentity,
)
from aka.contracts.execution_recovery import ExecutionCause, JobReference
from .gateway.protocol import classify_gateway_outcome, protocol_failure
from .transport import INFRASTRUCTURE_EXIT_CODE, INFRASTRUCTURE_MARKER

FACTS_FD = "AKA_EXECUTION_FACTS_FD"
_attempt = ContextVar("execution_backend_attempt", default=None)
_SAFE_DETAILS = frozenset((
    "http_status", "job_id", "candidate_rejected", "reference_rejected",
    "io_contract_rejected", "source_validation_failed", "metadata_invalid",
    "metadata_reason", "shape_id", "hardware", "response_type", "agate_return_code",
))


def diagnostic(record):
    return ExecutionDiagnostic(
        str(record.get("failure_class", "protocol")), str(record.get("reason", "unknown")),
        {key: value for key, value in record.items()
         if key in _SAFE_DETAILS and type(value) in (str, int, bool)},
    )


def emit_fact(record):
    """Only the owned launcher inherits this descriptor; worker output cannot write it."""
    value = os.environ.get(FACTS_FD)
    if value is None:
        return
    if _attempt.get() is not None:
        record = {"attempt_id": _attempt.get(), **record}
    data = (json.dumps(record, separators=(",", ":")) + "\n").encode()
    # Writes within PIPE_BUF stay atomic across concurrent shape workers.
    if len(data) > 4096:
        identifier = uuid.uuid4().hex
        pieces = [data[index:index + 2400] for index in range(0, len(data), 2400)]
        for index, piece in enumerate(pieces):
            emit_fact({"event": "fact_chunk", "identifier": identifier, "index": index,
                       "count": len(pieces), "content": base64.b64encode(piece).decode()})
        return
    try:
        os.write(int(value), data)
    except BrokenPipeError:
        pass  # The owning host has stopped; no new submission is initiated here.


@contextmanager
def backend_attempt(operation, backend="gateway"):
    token = _attempt.set(uuid.uuid4().hex)
    try:
        emit_fact({"event": "attempt_started", "operation": operation, "backend": backend})
        yield
    finally:
        _attempt.reset(token)


def emit_diagnostic(record):
    item = diagnostic(record)
    emit_fact({"event": "diagnostic", "record": {
        "failure_class": item.category, "reason": item.code, **item.details}})


def observe_attempt(process, *, operation, backend="gateway", attempt_id=None):
    """Capture a raw backend receipt before measurement redaction or log merging."""
    try:
        job = json.loads(process.stdout or "")
    except (TypeError, json.JSONDecodeError):
        job = None
    job = job if isinstance(job, dict) else {}
    record = protocol_failure(process, lambda _: None)
    status = job.get("status")
    terminal = status in {"succeeded", "failed", "cancelled"}
    rejected = bool(record and record.get("http_status") in {400, 401, 403, 404, 405, 409, 413, 422, 429})
    result = job.get("result")
    code = result.get("exit_code") if isinstance(result, dict) else None
    identifier = job.get("job_id")
    from .gateway.diagnostics import causes_from_job, remote_termination
    causes = causes_from_job(job)
    native_diagnostics = []
    if causes:
        category = {"infra": "infrastructure", "validation": "request", "code": "measurement"}.get(causes[0].category, causes[0].category)
        native_diagnostics.append({"failure_class": category, "reason": causes[0].code})
    def encode(cause):
        return {"category": cause.category, "code": cause.code, "message": cause.message,
                "details": _json(cause.details), "causes": [encode(item) for item in cause.causes]}
    emit_fact({
        **({"attempt_id": attempt_id} if attempt_id is not None else {}),
        "event": "attempt", "operation": operation, "backend": backend,
        "identifier": identifier if isinstance(identifier, str) else None,
        "disposition": "terminal" if terminal else "not_submitted" if rejected else "unknown",
        "termination": "cancelled" if status == "cancelled" else "exited" if terminal else None,
        "exit_code": code if type(code) is int else None,
        "diagnostics": native_diagnostics + ([record] if record else []),
        "causes": [encode(item) for item in causes],
        "remote_terminated": remote_termination(job),
    })
    return process


def _json(value):
    from collections.abc import Mapping
    if isinstance(value, Mapping):
        return {key: _json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json(item) for item in value]
    return value


def _cause(value):
    data = dict(value)
    data["causes"] = tuple(_cause(item) for item in data.get("causes", ()))
    return ExecutionCause(**data)


class FactChannel:
    def __init__(self, forward=None):
        self.forward = forward
        self.reader, self.writer = os.pipe()
        self.records = []
        self.chunks = {}
        self.forward_error = None
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        pending = b""
        while True:
            ready, _, _ = select.select([self.reader], [], [], .1)
            if not ready:
                if self.stopped.is_set():
                    if pending or self.chunks:
                        self.records.append({"event": "invalid"})
                    return
                continue
            data = os.read(self.reader, 65536)
            if not data:
                if pending or self.chunks:
                    self.records.append({"event": "invalid"})
                return
            pending += data
            while b"\n" in pending:
                line, pending = pending.split(b"\n", 1)
                try:
                    record = json.loads(line)
                except (ValueError, UnicodeDecodeError):
                    self.records.append({"event": "invalid"})
                else:
                    record = record if isinstance(record, dict) else {"event": "invalid"}
                    self._accept(record)

    def _accept(self, record):
        if record.get("event") == "fact_chunk":
            try:
                key, index, count = record["identifier"], record["index"], record["count"]
                if type(count) is not int or type(index) is not int or not 0 <= index < count <= 32768:
                    raise ValueError("invalid fact chunk")
                pieces = self.chunks.setdefault(key, [None] * count)
                if len(pieces) != count or pieces[index] is not None:
                    raise ValueError("conflicting fact chunk")
                pieces[index] = base64.b64decode(record["content"], validate=True)
                if any(piece is None for piece in pieces):
                    return
                record = json.loads(b"".join(pieces))
                del self.chunks[key]
                if not isinstance(record, dict):
                    raise ValueError("invalid reassembled fact")
            except (KeyError, TypeError, ValueError):
                record = {"event": "invalid"}
        self.records.append(record)
        if self.forward is not None:
            try:
                self.forward(record)
            except BaseException as error:
                self.forward_error = self.forward_error or error

    def close(self):
        os.close(self.writer)
        self.stopped.set()
        self.thread.join()
        os.close(self.reader)


def captured_result(process, *, resource, hardware="", records=(), status=None,
                    cancelled=False, cleanup="not_required", cleanup_errors=(), local_stopped=True, decode_output=True):
    outcome = classify_gateway_outcome(process) if decode_output else None
    records_diagnostics = []
    attempts = []
    submitted = {}
    job_references = []
    budgets = []
    for record in records:
        if record.get("event") == "submitted":
            submitted[record["identifier"]] = record
            if record.get("backend", "gateway") == "gateway":
                reference = JobReference("gateway", record["identifier"], record.get("operation", "run"),
                    record.get("endpoint", ""), record.get("profile", ""), record.get("request_identity", ""),
                    record.get("idempotency_key"))
                if reference not in job_references:
                    job_references.append(reference)
        elif record.get("event") == "budget":
            budgets.append(ExecutionBudget(**{key: record[key] for key in
                ("scope", "requested", "effective", "hard_limit", "reason") if key in record}))
    # Only the latest observation per Job belongs in each public snapshot.
    waiting = {record["identifier"]: record for record in records if record.get("event") == "job_wait"}
    for record in waiting.values():
        successful = next((item for item in reversed(records) if item.get("event") == "job_wait"
            and item.get("identifier") == record["identifier"] and item.get("query_ok")), None)
        if successful is not None:
            record = {**record, "last_successful_query_at_ms": successful["observed_at_ms"]}
        records_diagnostics.append(ExecutionDiagnostic("observation", "gateway-job-wait", {
            key: value for key, value in record.items() if key not in {"event", "attempt_id"}
            and type(value) in (str, int, bool)
        }))
    for record in records:
        if record.get("event") != "attempt":
            continue
        submission = next((item for item in submitted.values() if
            (record.get("attempt_id") and item.get("attempt_id") == record["attempt_id"]) or
            (record.get("identifier") and item.get("identifier") == record["identifier"])), {})
        intent = next((item for item in records if item.get("event") == "submission_intent" and
            record.get("attempt_id") and item.get("attempt_id") == record["attempt_id"]), {})
        details = tuple(diagnostic(item) for item in record.get("diagnostics", ()))
        attempts.append(ExecutionAttempt(
            record["operation"], record["disposition"],
            ResourceIdentity(record.get("backend", resource.backend), record.get("identifier")),
            record.get("termination"), record.get("exit_code"), details, record.get("attempt_id"),
            tuple(_cause(item) for item in record.get("causes", ())),
            record.get("idempotency_key") or submission.get("idempotency_key") or intent.get("idempotency_key"),
            record.get("request_identity") or submission.get("request_identity") or intent.get("request_identity"),
        ))
        records_diagnostics.extend(details)
    starts = {record["attempt_id"]: record for record in records if record.get("event") == "attempt_started"}
    completed = {record.get("attempt_id") for record in records if record.get("event") == "attempt"}
    for attempt_id, record in starts.items():
        if attempt_id in completed:
            continue
        submission = next((item for item in submitted.values() if item.get("attempt_id") == attempt_id), {})
        intent = next((item for item in records if item.get("event") == "submission_intent" and item.get("attempt_id") == attempt_id), {})
        attempts.append(ExecutionAttempt(record["operation"], resource=ResourceIdentity(
            record.get("backend", "gateway"), submission.get("identifier")), attempt_id=attempt_id,
            idempotency_key=intent.get("idempotency_key"), request_identity=intent.get("request_identity")))
    observed = {item.resource.identifier for item in attempts if item.resource}
    for identifier, record in submitted.items():
        if identifier not in observed:
            attempts.append(ExecutionAttempt(record.get("operation", "unknown"),
                resource=ResourceIdentity(record.get("backend", "gateway"), identifier)))
    # Receipt-channel facts are authoritative for real launchers. Decoding text is
    # reserved for explicitly captured historical transport fixtures/adapters.
    diagnostics = tuple(diagnostic(record["record"]) for record in records if record.get("event") == "diagnostic")
    if outcome is not None:
        diagnostics += tuple(diagnostic(item) for item in (
            ([outcome.primary_diagnostic] if outcome.primary_diagnostic else [])
            + list(outcome.secondary_diagnostics)))
    diagnostics += tuple(item for item in records_diagnostics if item not in diagnostics)
    rank = {"cancelled": 0, "authentication": 1, "request": 2, "timeout": 3, "infrastructure": 4, "observation": 99}
    diagnostics = tuple(sorted(diagnostics, key=lambda item: rank.get(item.category, 4)))
    has_result = bool(outcome and outcome.has_result) or any(record.get("event") == "result_available" for record in records)
    if attempts and attempts[-1].disposition == "terminal" and attempts[-1].exit_code is not None:
        has_result = True
    disposition = "unknown"
    termination = None
    if attempts and all(item.disposition != "unknown" for item in attempts):
        disposition = "terminal" if any(item.disposition == "terminal" for item in attempts) else "not_submitted"
        termination = attempts[-1].termination if disposition == "terminal" else None
    if not attempts and diagnostics and all(item.category in {"request", "authentication"} for item in diagnostics):
        disposition = "not_submitted"
    if any(record.get("event") == "invalid" for record in records):
        disposition, termination = "unknown", None
        diagnostics += (ExecutionDiagnostic("protocol", "invalid-fact-channel"),)
    if status is None:
        status = "succeeded" if process.returncode == 0 else "failed"
        if (process.returncode == INFRASTRUCTURE_EXIT_CODE and (
                any(item.category == "infrastructure" for item in diagnostics)
                or decode_output and INFRASTRUCTURE_MARKER in (process.stderr or "").splitlines())):
            status = "infrastructure"
        elif process.returncode and not has_result and diagnostics and diagnostics[0].category in {"request", "authentication"}:
            status = "rejected"
        elif process.returncode and not has_result and diagnostics and diagnostics[0].category == "cancelled":
            status = "cancelled"
        elif process.returncode and diagnostics and diagnostics[0].category == "timeout":
            status = "timed_out"
        elif process.returncode and not has_result and diagnostics:
            status = "infrastructure" if diagnostics[0].category == "infrastructure" else "protocol"
        elif process.returncode and attempts and attempts[-1].termination == "cancelled":
            status = "cancelled"
    remote_cleanup = "unknown"
    confirmations = {}
    for record in records:
        if record.get("event") != "attempt":
            continue
        if record.get("disposition") == "not_submitted":
            continue
        confirmation = record.get("remote_terminated")
        if "remote_terminated" not in record and type(record.get("exit_code")) is int:
            confirmation = True  # An owned backend process-exit receipt.
        confirmations[record.get("identifier", record.get("attempt_id"))] = confirmation
    values = tuple(confirmations.values())
    remote_terminated = False if False in values else True if values and all(value is True for value in values) and disposition == "terminal" else None
    for record in records:
        if record.get("event") == "cleanup":
            if remote_cleanup != "failed":
                remote_cleanup = record["status"]
            if record["status"] == "failed":
                cleanup_errors = (*cleanup_errors, ExecutionDiagnostic("cleanup", record.get("code", "remote-cleanup-failed")))
        if record.get("event") == "cancel_requested":
            cancelled = True
    return JobResult(
        process.returncode, process.stdout or "", process.stderr or "",
        exit_code=attempts[-1].exit_code if attempts else None,
        resource=resource, transport_status=status,
        facts=ExecutionFacts(disposition, termination, cancelled, local_stopped,
            remote_terminated, cleanup, tuple(cleanup_errors),
            ("launcher-reaped",) if local_stopped else (), remote_cleanup),
        attempts=tuple(attempts), diagnostics=diagnostics, hardware=hardware,
        causes=tuple(cause for attempt in attempts for cause in attempt.causes),
        job_references=tuple(job_references), budgets=tuple(budgets),
    )


def run_owned(argv, *, environment, workspace, timeout, resource, hardware="",
              cancel_event=None, spawn=None, check_environment=lambda: None, input_payload=None,
              on_observation=None, prepared_input_payload=None):
    """Return operational failures with cleanup evidence; preserve host interrupts."""
    channel = FactChannel()
    if on_observation is not None:
        channel.forward = lambda _record: on_observation(captured_result(
            subprocess.CompletedProcess(argv, 0, "", ""), resource=resource, hardware=hardware,
            records=channel.records, status="unknown", local_stopped=False, decode_output=False))
    import tempfile
    from .gateway.returned_files import FILES_FD, read as read_files
    returned = tempfile.TemporaryFile()
    payload = tempfile.TemporaryFile() if input_payload is not None else None
    prepared_inputs = tempfile.TemporaryFile() if prepared_input_payload is not None else None
    if payload is not None:
        payload.write(input_payload)
        payload.seek(0)
        argv = [*argv]
        position = argv.index("--") if "--" in argv else len(argv)
        argv[position:position] = ["--prepared-request-fd", str(payload.fileno())]
    if prepared_inputs is not None:
        prepared_inputs.write(prepared_input_payload)
        prepared_inputs.seek(0)
        argv = [*argv]
        position = argv.index("--") if "--" in argv else len(argv)
        argv[position:position] = ["--prepared-inputs-fd", str(prepared_inputs.fileno())]
    descriptors = ((channel.writer, returned.fileno())
                   + ((payload.fileno(),) if payload is not None else ())
                   + ((prepared_inputs.fileno(),) if prepared_inputs is not None else ()))
    process = None
    stdout = stderr = ""
    status = None
    cleanup = "not_required"
    cleanup_errors = []
    interruption = None
    cancelled = False
    try:
        check_environment()
        if cancel_event is not None and cancel_event.is_set():
            returned.close()
            return JobResult(1, resource=resource, hardware=hardware, transport_status="cancelled",
                facts=ExecutionFacts("not_submitted", cancellation_requested=True,
                    local_stopped=True, cleanup="not_required", basis=("cancelled-before-spawn",)))
        environment = {**environment, FACTS_FD: str(channel.writer), FILES_FD: str(returned.fileno())}
        options = dict(cwd=str(workspace), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       text=True)
        if spawn is None:
            process = subprocess.Popen(argv, env=environment, start_new_session=True, pass_fds=descriptors, **options)
        else:
            process = spawn(argv, role="sandbox", environment=environment, inherited_descriptors=descriptors, **options)
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            if channel.forward_error is not None:
                raise channel.forward_error
            if cancel_event is not None and cancel_event.is_set():
                cancelled = True
                status = "cancelled"
                break
            remaining = .25 if deadline is None else deadline - time.monotonic()
            if remaining <= 0:
                status = "timed_out"
                break
            try:
                stdout, stderr = process.communicate(timeout=min(.25, remaining))
                check_environment()
                break
            except subprocess.TimeoutExpired:
                check_environment()
    except BaseException as error:
        interruption = error
        status = "interrupted"
    finally:
        if process is not None and (status is not None or process.poll() is None):
            cleanup = "completed"
            try:
                if process.poll() is None:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                stdout, stderr = process.communicate(timeout=20)
            except BaseException as error:
                cleanup_errors.append(ExecutionDiagnostic("cleanup", type(error).__name__))
                try:
                    if process.poll() is None:
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                    stdout, stderr = process.communicate()
                except BaseException as fallback:
                    cleanup = "failed"
                    cleanup_errors.append(ExecutionDiagnostic("cleanup", type(fallback).__name__))
        channel.close()
        if payload is not None:
            payload.close()
        if prepared_inputs is not None:
            prepared_inputs.close()
    code = process.returncode if process is not None and process.returncode is not None else 1
    result = captured_result(subprocess.CompletedProcess(argv, code, stdout, stderr),
        resource=resource, hardware=hardware, records=channel.records, status=status,
        cancelled=cancelled, cleanup=cleanup, cleanup_errors=cleanup_errors,
        local_stopped=process is None or process.poll() is not None, decode_output=False)
    try:
        try:
            result = replace(result, files=read_files(returned))
        except (ValueError, KeyError, TypeError):
            result = replace(result, diagnostics=(*result.diagnostics,
                ExecutionDiagnostic("protocol", "invalid-returned-files")), transport_status="protocol",
                transport_returncode=result.returncode or 2)
    finally:
        returned.close()
    if process is None:
        result = replace(result, facts=replace(result.facts, disposition="not_submitted"))
    if interruption is None and channel.forward_error is not None:
        interruption = channel.forward_error
    if interruption is not None:
        # EnvironmentUnavailable/KeyboardInterrupt keep the established host boundary,
        # with the complete attempt and cleanup facts available to recovery owners.
        interruption.execution_outcome = result
        raise interruption
    return result
