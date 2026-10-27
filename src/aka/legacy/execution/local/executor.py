"""Own scoped local command execution without a trusted-launcher fact channel."""
from contextlib import nullcontext
import os
import signal
import subprocess
import time

from aka.contracts.execution import (BackendCapabilities, ExecutionAttempt, ExecutionBudget,
    ExecutionDiagnostic, ExecutionFacts, JobResult, ResourceIdentity, UnsupportedExecution)
from aka.legacy.execution.filesystem_scope import bound_command, require_scope_backend
from . import filesystem


class LocalCommandExecutor:
    resource_identity = ResourceIdentity("local-command")
    sandbox_cli_options = False
    sandbox_cli_endpoint_environment = False

    def __init__(self, *, bwrap="bwrap", context=None):
        self.bwrap = bwrap
        self.context = context
        self._availability = None

    @property
    def capabilities(self):
        if self._availability is None:
            self._availability = filesystem.availability(self.bwrap)
        return BackendCapabilities(operations=frozenset(("run", "dev")), cancellation=True,
            filesystem_isolation=self._availability[0] is not None)

    def bind(self, context):
        require_scope_backend(context, self.capabilities)
        executor = type(self)(bwrap=self.bwrap, context=context)
        executor._availability = self._availability
        return executor

    def run(self, command):
        if self.context is not None:
            command = bound_command(self.context, command)
        self.capabilities.require(command)
        if command.filesystem_access is None:
            prepared = nullcontext((command.command, dict(command.environment), command.workspace, ()))
        else:
            binary, reason = self._availability
            if binary is None:
                raise UnsupportedExecution(reason)
            prepared = filesystem.launch(binary, command)
        cancelled = command.cancel_event is not None and command.cancel_event.is_set()
        budgets = (ExecutionBudget("local-run", command.timeout, command.timeout),)
        if cancelled:
            result = JobResult(1, resource=self.resource_identity, transport_status="cancelled",
                facts=ExecutionFacts("not_submitted", cancellation_requested=True, local_stopped=True,
                    cleanup="not_required", basis=("cancelled-before-spawn",), remote_cleanup="not_required"),
                hardware=command.hardware, budgets=budgets)
            if command.on_observation is not None:
                command.on_observation(result)
            return result
        # No facts, returned-files, input or caller descriptors cross into argv.
        try:
            with prepared as (argv, environment, working, descriptors):
                process = subprocess.Popen(argv, cwd=working, env=environment, stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True,
                    close_fds=True, pass_fds=descriptors)
        except OSError as error:
            diagnostic = ExecutionDiagnostic("spawn", type(error).__name__, {"errno": error.errno or 0})
            result = JobResult(1, resource=self.resource_identity, transport_status="failed",
                facts=ExecutionFacts("not_submitted", local_stopped=True, cleanup="not_required",
                    basis=("local-spawn-rejected",), remote_cleanup="not_required"),
                diagnostics=(diagnostic,), hardware=command.hardware, budgets=budgets)
            if command.on_observation is not None:
                command.on_observation(result)
            return result
        deadline = time.monotonic() + command.timeout
        status = None
        interruption = None
        cleanup_errors = ()
        cleanup = "completed"
        try:
            while True:
                if command.cancel_event is not None and command.cancel_event.is_set():
                    cancelled, status = True, "cancelled"
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    status = "timed_out"
                    break
                try:
                    stdout, stderr = process.communicate(timeout=min(.25, remaining))
                    break
                except subprocess.TimeoutExpired:
                    pass
        except BaseException as error:
            interruption, status = error, "interrupted"
        if status is not None or process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                stdout, stderr = process.communicate(timeout=20)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                cleanup_errors = (ExecutionDiagnostic("cleanup", "termination-grace-exceeded"),)
                try:
                    stdout, stderr = process.communicate(timeout=20)
                except subprocess.TimeoutExpired as error:
                    # A trusted native process may have detached while retaining
                    # output pipes. Do not turn an owned timeout into an endless
                    # collection wait or claim that all descendants were cleaned.
                    stdout, stderr = error.output or b"", error.stderr or b""
                    process.stdout.close()
                    process.stderr.close()
                    process.wait(timeout=20)
                    cleanup = "unknown"
                    cleanup_errors += (ExecutionDiagnostic("cleanup", "output-pipes-not-closed"),)
        code = process.returncode
        status = status or ("succeeded" if code == 0 else "failed")
        termination = "cancelled" if cancelled else "timed_out" if status == "timed_out" else "interrupted" if status == "interrupted" else "exited"
        facts = ExecutionFacts("terminal", termination, cancelled,
            True, None, cleanup, cleanup_errors, ("owned-local-process-reaped",), "not_required")
        result = JobResult(0, stdout.decode("utf-8", errors="replace"), stderr.decode("utf-8", errors="replace"),
            code, self.resource_identity, "command", status, facts,
            (ExecutionAttempt(command.operation, "terminal", self.resource_identity, facts.termination, code),),
            hardware=command.hardware, budgets=budgets)
        if command.on_observation is not None:
            command.on_observation(result)
        if interruption is not None:
            interruption.execution_outcome = result
            raise interruption
        return result
