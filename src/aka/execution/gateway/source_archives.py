"""Opaque source upload transport; preparation/interpretation remain with owners."""
from __future__ import annotations

import base64
from contextlib import contextmanager
import fcntl
import hashlib
import http.client
import json
import os
from pathlib import Path
import stat
import subprocess
import time
from urllib.parse import urlsplit
import uuid

from aka.contracts.execution import PreparedSourceArchive, UnsupportedExecution
from aka.core.filesystem import open_private_directory

ARCHIVES = "__aka_source_archives"


class SourceArchiveError(RuntimeError):
    """Safe pre-admission transport error, with retained reservation facts."""
    def __init__(self, code, *, status=None):
        self.code, self.status = code, status if type(status) is int else None
        super().__init__("Source archive transport failed: " + code)


def source_descriptor(value):
    if isinstance(value, dict) and value.get('schema_version') == 1 and isinstance(value.get('entry_point'), dict):
        source = value.get('source')
        if isinstance(source, dict) and source.get('type') == 'oss_archive':
            value = {'archive': source.get('archive'), 'entry_point': 'solution.json'}
    if not isinstance(value, dict) or set(value) != {"archive", "entry_point"}:
        return False
    try:
        PreparedSourceArchive("candidate", value["archive"], value["entry_point"], b"descriptor")
    except (TypeError, ValueError):
        return False
    return True


def archive_matches(value, archive):
    if value == {'archive': archive.path, 'entry_point': archive.entry_point}:
        return True
    return (source_descriptor(value) and archive.entry_point == 'solution.json'
            and value.get('source', {}).get('archive') == archive.path
            and value['source'].get('sha256') == hashlib.sha256(archive.content).hexdigest())


def validate_archives(prepared, archives, operation):
    if not archives:
        for value in (prepared.get("request", {}).get("candidate"),
                      prepared.get("request", {}).get("abba", {}).get("baseline")):
            if isinstance(value, dict):
                raise UnsupportedExecution("source archive descriptors require retained archive bytes")
        return
    if operation not in {"run", "same_allocation_abba", "profile"} or prepared.get("kind") not in {"run", "profile"}:
        raise UnsupportedExecution("source archives require a typed Eval or Profile operation")
    request = prepared.get("request", {})
    roles = {item.role for item in archives}
    for item in archives:
        descriptor = request.get("candidate") if item.role == "candidate" else request.get("abba", {}).get("baseline")
        if not archive_matches(descriptor, item):
            raise ValueError("source archive bytes differ from their selected descriptor")
    for role, value in (("candidate", request.get("candidate")),
                        ("baseline", request.get("abba", {}).get("baseline"))):
        if isinstance(value, dict) and (role not in roles or not source_descriptor(value)):
            raise ValueError("source descriptor has no exact retained archive")
    if request.get("oss_files"):
        raise UnsupportedExecution("source archives cannot mix caller-owned upload references")


def prepared_payload(command):
    if command.submission is None:
        if command.source_archives:
            raise UnsupportedExecution("source archives require an immutable typed submission")
        return None
    prepared = json.loads(command.submission)
    validate_archives(prepared, command.source_archives, command.operation)
    key = command.idempotency_key
    old_key = prepared.get("request", {}).get("idempotency_key")
    if key is not None:
        if old_key not in (None, key):
            raise ValueError("prepared request differs from the physical idempotency key")
        prepared["request"]["idempotency_key"] = key
    if command.source_archives and key is None:
        raise UnsupportedExecution("source archives require an explicit physical idempotency key")
    if not command.source_archives:
        return command.submission if key is None else json.dumps(prepared, separators=(",", ":"), allow_nan=False).encode("utf-8")
    prepared[ARCHIVES] = [
        {"role": item.role, "path": item.path, "entry_point": item.entry_point,
         "content": base64.b64encode(item.content).decode("ascii")}
        for item in command.source_archives
    ]
    return json.dumps(prepared, separators=(",", ":"), allow_nan=False).encode("utf-8")


def read_archives(prepared):
    values = prepared.get(ARCHIVES, [])
    if not isinstance(values, list):
        raise ValueError("invalid source archive transport envelope")
    archives = []
    for value in values:
        if not isinstance(value, dict) or set(value) != {"role", "path", "entry_point", "content"}:
            raise ValueError("invalid source archive transport member")
        archives.append(PreparedSourceArchive(value["role"], value["path"], value["entry_point"],
            base64.b64decode(value["content"], validate=True)))
    if len({item.role for item in archives}) != len(archives) or len({item.path for item in archives}) != len(archives):
        raise ValueError("duplicate source archive transport identities")
    return tuple(archives)


def upload_bytes(url, data, *, timeout):
    """PUT exact bytes with storage credentials only; redirects are rejected."""
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError:
        raise SourceArchiveError("upload-invalid-url") from None
    if (parts.scheme not in {"http", "https"} or not parts.hostname
            or parts.username or parts.password or parts.fragment):
        raise ValueError("invalid source archive upload URL")
    connection = (http.client.HTTPSConnection if parts.scheme == "https" else http.client.HTTPConnection)(
        parts.hostname, port, timeout=timeout)
    try:
        target = parts.path or "/"
        if parts.query:
            target += "?" + parts.query
        connection.request("PUT", target, body=data, headers={"Content-Length": str(len(data))})
        response = connection.getresponse()
        response.read(4096)
        if not 200 <= response.status < 300:
            raise SourceArchiveError("upload-rejected", status=response.status)
    except (OSError, http.client.HTTPException):
        raise SourceArchiveError("upload-unavailable") from None
    finally:
        connection.close()


class SourceArchiveState:
    """One private reservation/wire journal under an exclusive physical-request lock."""
    def __init__(self, directory, name, *, request, archives, route, key, clock=time.monotonic, kind='eval'):
        if not isinstance(key, str) or not key:
            raise UnsupportedExecution("source archive transport requires its original physical request key")
        self.directory, self.name, self.clock = directory, name, clock
        self.request, self.archives, self.route, self.key = request, archives, route, key
        self.kind = kind
        descriptors = [{"role": item.role, "path": item.path, "entry_point": item.entry_point,
                        "bytes": len(item.content), "sha256": hashlib.sha256(item.content).hexdigest()}
                       for item in archives]
        identity_parts = [request, descriptors] if kind == "eval" else [kind, request, descriptors]
        self.logical_identity = hashlib.sha256(json.dumps(identity_parts,
            sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        self.state = self._read()
        if self.state is not None and (self.state.get("logical_identity") != self.logical_identity
                                      or self.state.get("route") != list(route)
                                      or self.state.get("key") != key):
            raise ValueError("retained source upload belongs to another request or route")
        self.descriptors = descriptors

    def _read(self):
        try:
            descriptor = os.open(self.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=self.directory)
        except FileNotFoundError:
            return None
        with os.fdopen(descriptor, "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise ValueError("retained upload state must be a regular file")
            value = json.loads(stream.read())
        if not isinstance(value, dict) or value.get("schema") != 1:
            raise ValueError("invalid retained source upload state")
        return value

    def save(self):
        temporary = self.name + ".tmp-" + uuid.uuid4().hex
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                             0o600, dir_fd=self.directory)
        try:
            with os.fdopen(descriptor, "w") as stream:
                json.dump(self.state, stream, allow_nan=False, separators=(",", ":"))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.name, src_dir_fd=self.directory, dst_dir_fd=self.directory)
            os.fsync(self.directory)
        finally:
            try:
                os.unlink(temporary, dir_fd=self.directory)
            except FileNotFoundError:
                pass

    def prepare(self, *, json_call, upload=upload_bytes, deadline, cancelled=lambda: False):
        if self.state is None:
            if cancelled():
                raise SourceArchiveError("upload-cancelled")
            remaining = deadline - self.clock()
            if remaining <= 0:
                raise SourceArchiveError("upload-budget-exhausted")
            self.state = {"schema": 1, "logical_identity": self.logical_identity,
                          "route": list(self.route), "key": self.key, "phase": "reserving"}
            self.save()
            try:
                reservation = json_call("POST", "/v1/uploads", {
                    "kind": self.kind, "spec": self.request["spec"],
                    "files": [{key: item[key] for key in ("path", "bytes", "sha256")} for item in self.descriptors],
                    **({"placement": self.request["placement"]} if "placement" in self.request else {}),
                }, remaining)
            except (OSError, RuntimeError, subprocess.SubprocessError) as error:
                raise SourceArchiveError("reservation-unavailable", status=getattr(error, "status", None)) from None
            self.state["reservation"] = reservation
            self.state["phase"] = "reserved"
            self.state["uploaded"] = []
            self.save()
        elif self.state.get("phase") == "reserving":
            raise SourceArchiveError("reservation-unconfirmed")
        if self.state.get("phase") not in {"reserved", "ready", "submission_intent", "submitted"}:
            raise ValueError("invalid retained source upload phase")
        reservation = self.state.get("reservation")
        uploads = reservation.get("uploads") if isinstance(reservation, dict) else None
        if (not isinstance(reservation, dict) or not isinstance(reservation.get("job_id"), str)
                or not reservation["job_id"] or not isinstance(uploads, list)
                or len(uploads) != len(self.archives)):
            raise ValueError("source upload reservation is incomplete")
        packed = {item.path: item.content for item in self.archives}
        if {item.get("path") for item in uploads if isinstance(item, dict)} != set(packed):
            raise ValueError("source upload reservation paths differ")
        for item in uploads:
            if (not isinstance(item.get("put_url"), str) or not item["put_url"]
                    or not isinstance(item.get("upload_ref"), str) or not item["upload_ref"]):
                raise ValueError("source upload reservation references are incomplete")
        uploaded = self.state.get("uploaded")
        if (not isinstance(uploaded, list) or any(not isinstance(path, str) for path in uploaded)
                or len(set(uploaded)) != len(uploaded) or not set(uploaded) <= set(packed)
                or self.state["phase"] != "reserved" and set(uploaded) != set(packed)):
            raise ValueError("retained source upload progress is invalid")
        if self.state["phase"] == "reserved":
            for item in uploads:
                if item["path"] in self.state["uploaded"]:
                    continue
                if cancelled():
                    raise SourceArchiveError("upload-cancelled")
                remaining = deadline - self.clock()
                if remaining <= 0:
                    raise SourceArchiveError("upload-budget-exhausted")
                upload(item["put_url"], packed[item["path"]], timeout=remaining)
                self.state["uploaded"].append(item["path"])
                self.save()
            wire = dict(self.request, oss_files=[{"path": item["path"], "upload_ref": item["upload_ref"]}
                                                for item in uploads], idempotency_key=self.key)
            self.state["wire"] = wire
            self.state["phase"] = "ready"
            self.save()
        expected_wire = dict(self.request, oss_files=[{"path": item["path"], "upload_ref": item["upload_ref"]}
                                                     for item in uploads], idempotency_key=self.key)
        if self.state.get("wire") != expected_wire:
            raise ValueError("retained source wire identity is invalid")
        return self.state["wire"]

    def begin_submission(self):
        self.state["phase"] = "submission_intent"
        self.save()

    def submitted(self, job):
        if not isinstance(job, dict) or not isinstance(job.get("job_id"), str) or not job["job_id"]:
            raise ValueError("submitted Job identity is unavailable")
        self.state["job_id"] = job["job_id"]
        self.state["phase"] = "submitted"
        self.save()
        if job["job_id"] != self.state["reservation"]["job_id"]:
            raise ValueError("submitted Job differs from the retained upload reservation")


@contextmanager
def retained_upload(workspace, request, archives, *, route, key, clock=time.monotonic, kind='eval'):
    workspace = Path(workspace).absolute()
    identity_parts = [str(workspace), list(route), key]
    if kind != "eval":
        identity_parts.insert(0, kind)
    name = hashlib.sha256(json.dumps(identity_parts, separators=(",", ":")).encode()).hexdigest()
    with open_private_directory(workspace.parent / ".aka-execution-uploads") as directory:
        descriptor = os.open(name + ".lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600, dir_fd=directory)
        try:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise SourceArchiveError("upload-already-active") from None
            yield SourceArchiveState(directory, name + ".json", request=request, archives=archives,
                                     route=route, key=key, clock=clock, kind=kind)
        finally:
            os.close(descriptor)


def run_archived(args, workspace, grace, request, archives, *, transport):
    """Prepare once, then submit or observe the original allocation identity."""
    import subprocess
    from .jobs import GatewayJobs
    from aka.contracts.execution_recovery import JobReference
    from aka.execution.outcomes import emit_fact, observe_attempt

    endpoint, profile = transport._submission_route(args.url)
    if not endpoint:
        raise UnsupportedExecution("source archives require a confirmed HTTP gateway route")
    key = request.get("idempotency_key")
    kind = 'profile' if args.kind == 'profile' else 'eval'
    from aka.execution.configuration import client_wait_timeout
    started = time.monotonic()
    wait_timeout = client_wait_timeout(args.timeout, grace)
    wait_deadline = None if wait_timeout is None else started + wait_timeout
    # Upload has its own finite operational budget; it must not bound GPU waiting.
    deadline = started + (args.timeout if wait_timeout is None else wait_timeout)
    json_call = lambda method, path, payload, timeout: transport._gateway_json(
        endpoint, method, path, payload, timeout)
    with retained_upload(workspace, request, archives, route=(endpoint, profile), key=key, kind=kind) as state:
        try:
            wire = state.prepare(json_call=json_call, deadline=deadline)
        except SourceArchiveError as error:
            emit_fact({"event": "attempt", "operation": "source-upload", "backend": "gateway",
                "disposition": "not_submitted", "idempotency_key": key,
                "request_identity": state.logical_identity,
                "diagnostics": [{"failure_class": "infrastructure", "reason": error.code}],
                "causes": [{"category": "infrastructure", "code": error.code, "message": str(error)}]})
            raise
        identity = hashlib.sha256(json.dumps(wire, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        reserved = state.state["reservation"]["job_id"]
        if state.state["phase"] in {"submission_intent", "submitted"}:
            identifier = state.state.get("job_id", reserved)
            reference = JobReference("gateway", identifier, kind, endpoint, profile, identity, key)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise SourceArchiveError("upload-budget-exhausted")
            jobs = GatewayJobs(endpoint=endpoint, profile=profile, http=lambda url, method, path, payload, timeout:
                               transport._gateway_json(url, method, path, payload, timeout))
            if state.state["phase"] == "submission_intent":
                # Admission may have succeeded before its receipt was retained.
                # A failed lookup is never permission to create another Job.
                try:
                    observed = jobs._read(reference, remaining)
                except (OSError, ValueError, RuntimeError, subprocess.SubprocessError):
                    raise SourceArchiveError("admission-unconfirmed") from None
                try:
                    state.submitted(observed)
                except ValueError:
                    if state.state.get("phase") == "submitted":
                        emit_fact({"event": "submitted", "identifier": state.state["job_id"], "operation": kind,
                            "backend": "gateway", "endpoint": endpoint, "profile": profile,
                            "request_identity": identity, "idempotency_key": key})
                    raise
            emit_fact({"event": "submitted", "identifier": identifier, "operation": kind,
                       "backend": "gateway", "endpoint": endpoint, "profile": profile,
                       "request_identity": identity, "idempotency_key": key})
            return transport._wait_direct_job(url=endpoint, kind=kind, job_id=identifier,
                deadline=wait_deadline, started=started, profile=profile, identity=identity, key=key)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise SourceArchiveError("upload-budget-exhausted")
        state.begin_submission()
        return transport._run_direct_job(url=endpoint, kind=kind, payload=wire,
            timeout=remaining, queue_wait_grace=None if grace is None else 0, on_submitted=state.submitted)


def replay_archived(jobs, command, *, idempotency_key, retry):
    """Reuse retained wire identity; confirmed old Jobs are always queried."""
    from dataclasses import replace
    from .requests import operation_request, effective_budget, hard_limit
    from .jobs import unknown_result, client_wait_budget
    from aka.contracts.execution import ExecutionAttempt, ExecutionDiagnostic, ExecutionFacts, JobResult, ResourceIdentity
    from aka.contracts.execution_recovery import ExecutionCause
    from aka.contracts.execution_recovery import JobReference
    if command.idempotency_key != idempotency_key:
        raise ValueError("archive replay requires the original physical idempotency key")
    prepared = json.loads(prepared_payload(command))
    archives = read_archives(prepared)
    request = operation_request(prepared["request"], prepared["kind"])
    budget = effective_budget(command.timeout, limit=hard_limit(command.environment, "run"), emit=False)
    request.setdefault("options", {})["timeout_s"] = budget.effective
    elapsed_budget = min(command.timeout, retry.max_elapsed_s) if retry.delays else command.timeout
    deadline = jobs.clock() + elapsed_budget
    wait_budgets, wait_diagnostics = client_wait_budget("replay-wait", command.timeout, elapsed_budget)
    budgets = (*wait_budgets, budget)
    def publish(result):
        result = replace(result, budgets=(*budgets, *result.budgets),
                         diagnostics=(*wait_diagnostics, *result.diagnostics))
        if command.on_observation is not None:
            command.on_observation(result)
        return result
    json_call = lambda method, path, payload, timeout: jobs.http(jobs.endpoint, method, path, payload, timeout)
    with retained_upload(command.workspace, request, archives, route=(jobs.endpoint, jobs.profile),
                         key=idempotency_key, clock=jobs.clock, kind='profile' if prepared['kind'] == 'profile' else 'eval') as state:
        def preparation_failure(error):
            cancelled = error.code == "upload-cancelled"
            category = "cancelled" if cancelled else "request" if error.status in {400, 401, 403, 404, 405, 409, 413, 422} else "infrastructure"
            diagnostic = ExecutionDiagnostic(category, error.code,
                {"http_status": error.status} if type(error.status) is int else {})
            causes = (ExecutionCause(category, error.code, str(error)),)
            attempt = ExecutionAttempt("source-upload", "not_submitted", ResourceIdentity("gateway"),
                diagnostics=(diagnostic,), causes=causes, idempotency_key=idempotency_key,
                request_identity=state.logical_identity)
            return publish(JobResult(1 if cancelled else 2 if category == "request" else 75,
                resource=attempt.resource, transport_status="cancelled" if cancelled else "rejected" if category == "request" else "infrastructure",
                facts=ExecutionFacts("not_submitted", cancellation_requested=cancelled, local_stopped=True, cleanup="not_required"),
                attempts=(attempt,), diagnostics=(diagnostic,), causes=causes))
        try:
            wire = state.prepare(json_call=json_call, deadline=deadline,
                cancelled=lambda: command.cancel_event is not None and command.cancel_event.is_set())
        except SourceArchiveError as error:
            return preparation_failure(error)
        identity = hashlib.sha256(json.dumps(wire, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        def unconfirmed_admission(code):
            diagnostic = ExecutionDiagnostic("infrastructure", code)
            cause = ExecutionCause("infrastructure", diagnostic.code)
            attempt = ExecutionAttempt("replay", "unknown", ResourceIdentity("gateway"),
                diagnostics=(diagnostic,), causes=(cause,), idempotency_key=idempotency_key,
                request_identity=identity)
            return publish(JobResult(75, resource=attempt.resource, transport_status="infrastructure",
                facts=ExecutionFacts("unknown", local_stopped=True, cleanup="not_required"),
                attempts=(attempt,), diagnostics=(diagnostic,), causes=(cause,)))
        if state.state["phase"] in {"submission_intent", "submitted"}:
            identifier = state.state.get("job_id", state.state["reservation"]["job_id"])
            reference = JobReference("gateway", identifier, state.kind, jobs.endpoint, jobs.profile, identity, idempotency_key)
            remaining = deadline - jobs.clock()
            if remaining <= 0:
                if state.state["phase"] == "submission_intent":
                    return unconfirmed_admission("query-budget-exhausted")
                return publish(unknown_result(reference, "query-budget-exhausted"))
            if state.state["phase"] == "submission_intent":
                try:
                    state.submitted(jobs._read(reference, remaining))
                except (OSError, ValueError, RuntimeError, subprocess.SubprocessError):
                    if state.state.get("phase") == "submitted":
                        confirmed = replace(reference, identifier=state.state["job_id"])
                        return publish(unknown_result(confirmed, "source-upload-reservation-identity-mismatch"))
                    return unconfirmed_admission("source-upload-admission-unconfirmed")
            remaining = deadline - jobs.clock()
            if remaining <= 0:
                return publish(unknown_result(reference, "query-budget-exhausted"))
            if command.cancel_event is not None and command.cancel_event.is_set():
                return publish(jobs.cancel(reference, timeout=remaining))
            result = jobs.query(reference, timeout=remaining, retry=retry)
            return publish(result)
        def observed(result):
            try:
                if result.job_references:
                    state.submitted({"job_id": result.job_references[0].identifier})
            finally:
                if command.on_observation is not None:
                    command.on_observation(result)
        envelope = {key: value for key, value in prepared.items() if key != ARCHIVES}
        envelope["request"] = wire
        remaining = deadline - jobs.clock()
        if remaining <= 0:
            return preparation_failure(SourceArchiveError("upload-budget-exhausted"))
        if command.cancel_event is not None and command.cancel_event.is_set():
            return preparation_failure(SourceArchiveError("upload-cancelled"))
        state.begin_submission()
        return jobs._replay_prepared(replace(command, source_archives=(), submission=json.dumps(envelope).encode(),
                                   on_observation=observed), idempotency_key=idempotency_key, retry=retry,
                                   deadline=deadline, request_identity=identity)
