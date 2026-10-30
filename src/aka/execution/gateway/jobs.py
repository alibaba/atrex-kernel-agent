"""Query confirmed gateway jobs without creating replacements or changing routes."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import subprocess
import time
import os
import io
import base64
import tarfile
import tempfile
import math
from pathlib import Path
from urllib.parse import quote

from aka.contracts.execution import (
    ExecutionAttempt, ExecutionBudget, ExecutionDiagnostic, ExecutionFacts, ExecutionFile, JobResult,
    ResourceIdentity, UnsupportedExecution,
)
from aka.contracts.execution_recovery import (
    ExecutionCause, InfrastructureRetryPolicy, JobReference,
)


from .diagnostics import causes_from_job, remote_termination


def causes_from_error(error):
    try:
        value = json.loads(getattr(error, "detail", ""))
    except (TypeError, ValueError):
        return (ExecutionCause("infrastructure", type(error).__name__, str(error)),)
    return causes_from_job({"error": value.get("error", value) if isinstance(value, dict) else value})


def client_wait_budget(scope, requested, effective):
    """Record allocated client waiting without claiming a backend runtime limit."""
    reason = "infrastructure-retry-budget" if effective < requested else "requested"
    if int(requested) == requested and int(effective) == effective:
        return (ExecutionBudget(scope, int(requested), int(effective), reason=reason),), ()
    # The inherited ExecutionBudget encodes whole seconds. Preserve fractional
    # API waits exactly instead of rounding them into an untrue effective budget.
    return (), (ExecutionDiagnostic("budget", "fractional-client-wait-budget",
        {"scope": scope, "requested_s": str(requested), "effective_s": str(effective), "reason": reason}),)


def gateway_environment(environment):
    """Freeze CLI config authentication once; credentials never become job facts."""
    value = dict(environment)
    try:
        config = json.loads((Path(value.get("HOME", str(Path.home()))) / ".atrex/config.json").read_text())
    except (OSError, ValueError):
        config = {}
    if not isinstance(config, dict):
        config = {}
    for key, name in (("token", "AGATE_TOKEN"), ("ak", "AGATE_AK"), ("sk", "AGATE_SK")):
        if not value.get(name) and isinstance(config.get(key), str):
            value[name] = config[key]
    return value, config


def selected_route(endpoint="", profile="", *, environment=None, executable=None):
    """Use the installed CLI's route contract, including its actual default profile."""
    environment, config = gateway_environment(os.environ if environment is None else environment)
    if endpoint:
        profile = ""
    elif not profile:
        endpoint = environment.get("AGATE_URL") or config.get("url", "")
    if not endpoint and executable:
        argv = [executable, "config"] + (["--profile", profile] if profile else [])
        try:
            resolved = subprocess.run(argv, env=environment, capture_output=True, text=True, timeout=10)
            value = json.loads(resolved.stdout)
            if not resolved.returncode and isinstance(value, dict):
                endpoint = value.get("url", "")
        except (OSError, ValueError, subprocess.SubprocessError):
            pass  # Unknown route remains unknown; never guess a production endpoint.
    if endpoint:
        JobReference("gateway", "route-validation", endpoint=endpoint)
    return endpoint, profile or "", environment


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
                                termination, exit_code, tuple(diagnostics), causes=causes,
                                idempotency_key=reference.idempotency_key,
                                request_identity=reference.request_identity)
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
                                   diagnostics=(diagnostic,), idempotency_key=reference.idempotency_key,
                                   request_identity=reference.request_identity),), diagnostics=(diagnostic,),
        job_references=(reference,))


def collect_job_files(job, reference, *, timeout):
    """Collect only the original terminal job's returned files; publish nowhere."""
    from . import sandbox
    raw = job.get("result") or {}
    if not isinstance(raw, dict):
        return (), None
    files = []
    stdout = raw.get("stdout", "")
    deadline = time.monotonic() + timeout
    with tempfile.TemporaryDirectory(prefix="aka-job-collection-") as temporary:
        root = Path(temporary)
        for artifact in raw.get("artifacts") or ():
            if not isinstance(artifact, dict):
                raise ValueError("original job artifact must be an object")
            name = sandbox._safe_relative(str(artifact.get("name") or ""))
            destination = root / name
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("original-job-artifact-budget-exhausted")
            sandbox._download_oss_artifact(artifact, destination, timeout=remaining)
            if name == sandbox.OSS_OUTPUT_ARCHIVE:
                with tarfile.open(destination, mode="r:gz") as archive:
                    files.extend(ExecutionFile(*item) for item in sandbox._extract_output_tar(archive, root))
            else:
                files.append(ExecutionFile(name, destination.read_bytes()))
        if isinstance(stdout, str) and sandbox.OUTPUT_BEGIN in stdout and sandbox.OUTPUT_END in stdout:
            command_stdout, framed = stdout.rsplit(sandbox.OUTPUT_BEGIN, 1)
            encoded, trailing = framed.split(sandbox.OUTPUT_END, 1)
            payload = base64.b64decode("".join(encoded.split()), validate=True)
            with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
                files.extend(ExecutionFile(*item) for item in sandbox._extract_output_tar(archive, root))
            stdout = (command_stdout + trailing).rstrip("\n")
    return tuple(files), stdout if reference.operation == "dev" else None


class GatewayJobs:
    """Backend-specific observation. Retry schedules apply only to original-job reads."""
    def __init__(self, *, endpoint="", profile="", executable=None,
                 http=None, runner=None, clock=time.monotonic, sleep=time.sleep,
                 idempotent_replay=False, environment=None):
        self.endpoint, self.profile, self.executable = endpoint, profile, executable
        if http is None:
            from .sandbox import _gateway_json
            frozen = dict(os.environ if environment is None else environment)
            http = lambda endpoint, method, path, payload, timeout: _gateway_json(
                endpoint, method, path, payload, timeout, environment=frozen)
        self.http, self.runner = http, runner or subprocess.run
        self.clock, self.sleep = clock, sleep
        self.idempotent_replay = idempotent_replay
        self.environment = dict(os.environ if environment is None else environment)

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
        result = self.runner(argv, env=self.environment, capture_output=True, text=True, timeout=timeout)
        if result.returncode:
            raise RuntimeError("original-job-query-failed")
        return json.loads(result.stdout)

    def query(self, reference, *, timeout, retry=InfrastructureRetryPolicy()):
        self._require(reference)
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("query timeout must be positive")
        elapsed_budget = min(timeout, retry.max_elapsed_s) if retry.delays else timeout
        wait_budgets, wait_diagnostics = client_wait_budget("query-wait", timeout, elapsed_budget)
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
                if result.facts.disposition == "terminal":
                    try:
                        files, stdout = collect_job_files(job, reference, timeout=deadline - self.clock())
                        result = replace(result, files=files)
                        if stdout is not None and result.exit_code is not None:
                            result = replace(result, stdout=stdout, stderr=str((job.get("result") or {}).get("stderr") or ""),
                                             output_scope="command")
                    except (OSError, ValueError, RuntimeError, tarfile.TarError) as error:
                        diag = ExecutionDiagnostic("infrastructure", "original-job-artifact-unavailable")
                        result = replace(result, transport_returncode=75, transport_status="infrastructure",
                            diagnostics=(*result.diagnostics, diag),
                            causes=(*result.causes, ExecutionCause("infrastructure", type(error).__name__, str(error))))
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
                result = unknown_result(reference, type(error).__name__)
                result = replace(result, causes=causes_from_error(error), attempts=tuple(
                    replace(item, causes=causes_from_error(error)) for item in result.attempts))
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
                       diagnostics=(tuple(diagnostics) or result.diagnostics) + wait_diagnostics,
                       budgets=(*result.budgets, *wait_budgets))

    def cancel(self, reference, *, timeout):
        self._require(reference)
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("cancel timeout must be positive")
        deadline = self.clock() + timeout
        wait_budgets, wait_diagnostics = client_wait_budget("cancel-wait", timeout, timeout)
        request_diagnostics = ()
        request_causes = ()
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
                result = self.runner(argv, env=self.environment, capture_output=True, text=True, timeout=timeout)
                if result.returncode:
                    raise RuntimeError("cancel-request-failed")
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
            request_diagnostics = (ExecutionDiagnostic("cleanup", type(error).__name__),)
            request_causes = causes_from_error(error)
        remaining = deadline - self.clock()
        result = self.query(reference, timeout=remaining) if remaining > 0 else unknown_result(reference, "cancel-confirmation-budget-exhausted")
        return replace(result, diagnostics=(*request_diagnostics, *result.diagnostics, *wait_diagnostics),
                       budgets=(*result.budgets, *wait_budgets),
                       causes=(*request_causes, *result.causes),
                       facts=replace(result.facts, cancellation_requested=True,
                           cleanup_diagnostics=request_diagnostics,
                           remote_cleanup="failed" if request_diagnostics else "completed"))

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
        from .requests import operation_request, prepared_operation, effective_budget, hard_limit
        if command.operation == "same_allocation_abba" or "abba" in prepared.get("request", {}):
            prepared_operation(prepared, command.operation)
        payload = operation_request(prepared["request"], kind)
        if prepared["request"].get("idempotency_key") not in (None, idempotency_key):
            raise ValueError("replay cannot change the prepared idempotency key")
        payload["idempotency_key"] = idempotency_key
        elapsed_budget = min(command.timeout, retry.max_elapsed_s) if retry.delays else command.timeout
        budgets, budget_diagnostics = client_wait_budget("replay-wait", command.timeout, elapsed_budget)
        if kind in {"run", "profile"}:
            runtime = effective_budget(command.timeout, limit=hard_limit(command.environment, kind), emit=False)
            payload.setdefault("options", {})["timeout_s"] = runtime.effective
            budgets = (*budgets, runtime)
        else:
            budget_diagnostics = (*budget_diagnostics, ExecutionDiagnostic("budget", "job-runtime-budget-unconfirmed",
                                    {"operation": kind}))
        identity = hashlib.sha256(command.submission).hexdigest()
        deadline = self.clock() + elapsed_budget
        attempts, diagnostics = [], []
        last = None
        for index in range(len(retry.delays) + 1):
            remaining = deadline - self.clock()
            if remaining <= 0:
                break
            if command.cancel_event is not None and command.cancel_event.is_set():
                last = JobResult(1, resource=ResourceIdentity("gateway"), transport_status="cancelled",
                    facts=ExecutionFacts("not_submitted", cancellation_requested=True,
                        local_stopped=True, cleanup="not_required"), budgets=budgets, diagnostics=budget_diagnostics)
                break
            try:
                accepted = self.http(self.endpoint, "POST", "/v1/jobs/" + {"run": "eval", "check": "compile"}.get(kind, kind), payload, remaining)
                identifier = accepted.get("job_id")
                if not isinstance(identifier, str):
                    raise RuntimeError("submission-identity-unavailable")
                reference = JobReference("gateway", identifier, kind, self.endpoint, self.profile,
                                         identity, idempotency_key)
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
                status = getattr(error, "status", None)
                rejected = status in {400, 401, 403, 404, 405, 409, 413, 422, 429}
                category = "infrastructure" if status is None or status in {408, 429} or status >= 500 else "request"
                diag = ExecutionDiagnostic(category, type(error).__name__, {"http_status": status} if type(status) is int else {})
                native_causes = causes_from_error(error)
                attempt = ExecutionAttempt("replay", "not_submitted" if rejected else "unknown",
                    ResourceIdentity("gateway"), diagnostics=(diag,), causes=native_causes,
                    idempotency_key=idempotency_key, request_identity=identity)
                attempts.append(attempt); diagnostics.append(diag)
                last = JobResult(75 if category == "infrastructure" else 2,
                    resource=ResourceIdentity("gateway"), transport_status="infrastructure" if category == "infrastructure" else "rejected",
                    facts=ExecutionFacts(attempt.disposition, local_stopped=True, cleanup="not_required"),
                    attempts=(attempt,), diagnostics=(diag, *budget_diagnostics), causes=native_causes, budgets=budgets)
                if command.on_observation is not None:
                    command.on_observation(last)
                if category != "infrastructure" or index == len(retry.delays):
                    break
                delay = retry.delays[index]
                if delay >= deadline - self.clock():
                    break
                self.sleep(delay)
                continue
            # Callback failure after confirmed admission must never enter POST retry.
            last = replace(result_from_job(accepted, reference, operation="replay"), budgets=budgets,
                           diagnostics=budget_diagnostics)
            attempts.extend(last.attempts)
            if command.on_observation is not None:
                command.on_observation(last)
            remaining = deadline - self.clock()
            if command.cancel_event is not None and command.cancel_event.is_set() and remaining > 0:
                queried = self.cancel(reference, timeout=remaining)
                last = replace(queried, budgets=(*budgets, *queried.budgets),
                               diagnostics=(*budget_diagnostics, *queried.diagnostics))
                attempts.extend(last.attempts)
            elif last.facts.disposition != "terminal" and remaining > 0:
                queried = self.query(reference, timeout=remaining, retry=retry)
                last = replace(queried, budgets=(*budgets, *queried.budgets),
                               diagnostics=(*budget_diagnostics, *queried.diagnostics))
                attempts.extend(last.attempts)
            break  # Confirmed IDs proceed exclusively to original-job GET.
        if last is None:
            raise ValueError("replay budget exhausted before submission")
        last = replace(last, attempts=tuple(attempts), diagnostics=tuple(diagnostics) + last.diagnostics,
                       causes=tuple(cause for attempt in attempts for cause in attempt.causes))
        if command.on_observation is not None:
            command.on_observation(last)
        return last
