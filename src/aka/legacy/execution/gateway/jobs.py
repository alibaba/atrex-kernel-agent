"""Query confirmed gateway jobs without creating replacements or changing routes."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import subprocess
import time
from urllib.parse import quote

from aka.contracts.execution import (
    ExecutionAttempt, ExecutionDiagnostic, ExecutionFacts, JobResult,
    ResourceIdentity, UnsupportedExecution,
)
from aka.contracts.execution_recovery import (
    ExecutionCause, InfrastructureRetryPolicy, JobReference,
)


from .diagnostics import causes_from_job, remote_termination


def result_from_job(job, reference, *, operation="query", diagnostics=(), cancellation_requested=False):
    if not isinstance(job, dict) or job.get("job_id") != reference.identifier:
        return unknown_result(reference, "job-identity-mismatch", operation=operation,
                              cancellation_requested=cancellation_requested)
    terminal = job.get("status") in {"succeeded", "failed", "cancelled"}
    raw = job.get("result")
    exit_code = raw.get("exit_code") if isinstance(raw, dict) else None
    exit_code = exit_code if type(exit_code) is int else None
    causes = causes_from_job(job)
    root = causes[0] if causes else None
    category = root.category if root else ""
    status = "unknown"
    if terminal:
        status = "succeeded" if job["status"] == "succeeded" else "failed"
        if job["status"] == "cancelled":
            status = "cancelled"
        elif category in {"infra", "infrastructure"}:
            status = "infrastructure"
        elif category == "timeout":
            status = "timed_out"
    termination = "cancelled" if job.get("status") == "cancelled" else "exited" if terminal else None
    resource = ResourceIdentity(reference.backend, reference.identifier)
    observed = ExecutionAttempt(operation, "terminal" if terminal else "unknown", resource,
                                termination, exit_code, tuple(diagnostics))
    facts = ExecutionFacts(
        "terminal" if terminal else "unknown", termination,
        cancellation_requested=cancellation_requested, local_stopped=True,
        remote_terminated=remote_termination(job), cleanup="not_required",
        basis=("confirmed-job-query",) + (("backend-job-terminal",) if terminal else ()),
        remote_cleanup="unknown",
    )
    return JobResult(0 if status == "succeeded" else 75 if status == "infrastructure" else 1,
        stdout=json.dumps(job), exit_code=exit_code, resource=resource,
        transport_status=status, facts=facts, attempts=(observed,), diagnostics=tuple(diagnostics),
        causes=causes, job_references=(reference,))


def unknown_result(reference, code, *, operation="query", cancellation_requested=False):
    diagnostic = ExecutionDiagnostic("infrastructure", code, {"job_id": reference.identifier})
    return JobResult(75, resource=ResourceIdentity(reference.backend, reference.identifier),
        transport_status="infrastructure", facts=ExecutionFacts("unknown",
        cancellation_requested=cancellation_requested, local_stopped=True,
        cleanup="not_required", basis=("confirmed-job-identity",)),
        attempts=(ExecutionAttempt(operation, resource=ResourceIdentity(reference.backend, reference.identifier),
                                   diagnostics=(diagnostic,)),), diagnostics=(diagnostic,),
        job_references=(reference,))


class GatewayJobs:
    """Backend-specific observation. Retry schedules apply only to original-job reads."""
    def __init__(self, *, endpoint="", profile="", executable=None,
                 http=None, runner=None, clock=time.monotonic, sleep=time.sleep,
                 idempotent_replay=False):
        self.endpoint, self.profile, self.executable = endpoint, profile, executable
        if http is None:
            from .sandbox import _gateway_json
            http = _gateway_json
        self.http, self.runner = http, runner or subprocess.run
        self.clock, self.sleep = clock, sleep
        self.idempotent_replay = idempotent_replay

    def _require(self, reference):
        if reference.backend != "gateway":
            raise UnsupportedExecution("this executor cannot query the selected job backend")
        if reference.endpoint != self.endpoint or reference.profile != self.profile:
            raise UnsupportedExecution("job recovery must retain the original selected route")
        if not self.endpoint and not self.executable:
            raise UnsupportedExecution("job recovery needs the original gateway route")

    def _read(self, reference, timeout):
        if self.endpoint and not self.executable:
            return self.http(self.endpoint, "GET", "/v1/jobs/" + quote(reference.identifier, safe=""), None, timeout)
        argv = [self.executable, "get"]
        if self.endpoint:
            argv += ["--url", self.endpoint]
        elif self.profile:
            argv += ["--profile", self.profile]
        argv += ["--http-timeout", str(timeout), reference.identifier]
        result = self.runner(argv, capture_output=True, text=True, timeout=timeout)
        if result.returncode:
            raise RuntimeError("original-job-query-failed")
        return json.loads(result.stdout)

    def query(self, reference, *, timeout, retry=InfrastructureRetryPolicy()):
        self._require(reference)
        if type(timeout) not in (int, float) or timeout <= 0:
            raise ValueError("query timeout must be positive")
        elapsed_budget = min(timeout, retry.max_elapsed_s) if retry.delays else timeout
        deadline = self.clock() + elapsed_budget
        attempts = []
        diagnostics = []
        result = None
        for index in range(len(retry.delays) + 1):
            remaining = deadline - self.clock()
            if remaining <= 0:
                break
            try:
                job = self._read(reference, remaining)
                result = result_from_job(job, reference)
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
                result = unknown_result(reference, type(error).__name__)
            attempts.extend(result.attempts)
            diagnostics.extend(result.diagnostics)
            if result.transport_status != "infrastructure" or result.facts.disposition == "terminal":
                break
            if index < len(retry.delays):
                delay = retry.delays[index]
                if delay >= deadline - self.clock():
                    break
                self.sleep(delay)
        if result is None:
            result = unknown_result(reference, "query-budget-exhausted")
        return replace(result, attempts=tuple(attempts) or result.attempts,
                       diagnostics=tuple(diagnostics) or result.diagnostics)

    def cancel(self, reference, *, timeout):
        self._require(reference)
        if type(timeout) not in (int, float) or timeout <= 0:
            raise ValueError("cancel timeout must be positive")
        deadline = self.clock() + timeout
        request_diagnostics = ()
        try:
            if self.endpoint and not self.executable:
                self.http(self.endpoint, "POST", "/v1/jobs/" + quote(reference.identifier, safe="") + "/cancel", {}, timeout)
            else:
                argv = [self.executable, "cancel"]
                if self.endpoint:
                    argv += ["--url", self.endpoint]
                elif self.profile:
                    argv += ["--profile", self.profile]
                argv += ["--http-timeout", str(timeout), reference.identifier]
                result = self.runner(argv, capture_output=True, text=True, timeout=timeout)
                if result.returncode:
                    raise RuntimeError("cancel-request-failed")
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
            request_diagnostics = (ExecutionDiagnostic("cleanup", type(error).__name__),)
        remaining = deadline - self.clock()
        result = self.query(reference, timeout=remaining) if remaining > 0 else unknown_result(reference, "cancel-confirmation-budget-exhausted")
        return replace(result, facts=replace(result.facts, cancellation_requested=True),
                       diagnostics=(*request_diagnostics, *result.diagnostics))

    def replay(self, command, *, idempotency_key, retry=InfrastructureRetryPolicy()):
        """Resend exact prepared bytes/key only to an explicitly supported unified API."""
        if not self.idempotent_replay or not self.endpoint or self.executable:
            raise UnsupportedExecution("this selected transport does not guarantee stable-key replay")
        if not isinstance(idempotency_key, str) or not idempotency_key:
            raise ValueError("replay requires the original idempotency key")
        if command.submission is None:
            raise UnsupportedExecution("replay requires an immutable prepared request")
        prepared = json.loads(command.submission)
        kind = prepared.get("kind")
        if prepared.get("schema_version") != 1 or kind not in {"run", "profile", "check", "disassemble"}:
            raise UnsupportedExecution("unsupported prepared replay operation")
        payload = dict(prepared["request"])
        if payload.get("idempotency_key") not in (None, idempotency_key):
            raise ValueError("replay cannot change the prepared idempotency key")
        payload["idempotency_key"] = idempotency_key
        identity = hashlib.sha256(command.submission).hexdigest()
        timeout = command.timeout
        deadline = self.clock() + (min(timeout, retry.max_elapsed_s) if retry.delays else timeout)
        attempts = []
        diagnostics = []
        last = None
        for index in range(len(retry.delays) + 1):
            remaining = deadline - self.clock()
            if remaining <= 0:
                break
            try:
                accepted = self.http(self.endpoint, "POST", "/v1/jobs/" + {"run": "eval", "check": "compile"}.get(kind, kind), payload, remaining)
                identifier = accepted.get("job_id")
                if not isinstance(identifier, str):
                    raise RuntimeError("submission-identity-unavailable")
                reference = JobReference("gateway", identifier, kind, self.endpoint, self.profile,
                                         identity, idempotency_key)
                last = result_from_job(accepted, reference, operation="replay")
                attempts.extend(last.attempts)
                # Acceptance, queued/running and all confirmed IDs proceed only to GET.
                remaining = deadline - self.clock()
                if last.facts.disposition != "terminal" and remaining > 0:
                    last = self.query(reference, timeout=remaining, retry=retry)
                    attempts.extend(last.attempts)
                break
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
                status = getattr(error, "status", None)
                rejected = status in {400, 401, 403, 405, 413, 422, 429}
                diag = ExecutionDiagnostic("infrastructure" if status in {None, 408, 429} or status >= 500 else "request",
                                           type(error).__name__, {"http_status": status} if type(status) is int else {})
                attempt = ExecutionAttempt("replay", "not_submitted" if rejected else "unknown",
                                           ResourceIdentity("gateway"), diagnostics=(diag,),
                                           idempotency_key=idempotency_key, request_identity=identity)
                attempts.append(attempt)
                diagnostics.append(diag)
                last = JobResult(75 if diag.category == "infrastructure" else 2,
                    resource=ResourceIdentity("gateway"), transport_status="infrastructure" if diag.category == "infrastructure" else "rejected",
                    facts=ExecutionFacts(attempt.disposition, local_stopped=True, cleanup="not_required"),
                    attempts=(attempt,), diagnostics=(diag,))
                if diag.category != "infrastructure" or index == len(retry.delays):
                    break
                delay = retry.delays[index]
                if delay >= deadline - self.clock():
                    break
                self.sleep(delay)
        if last is None:
            raise ValueError("replay budget exhausted before submission")
        return replace(last, attempts=tuple(attempts), diagnostics=tuple(diagnostics) + last.diagnostics)
