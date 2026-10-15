"""Workflow's internal process entry; Bootstrap owns its assembly lifetime."""
from pathlib import Path
import os
import signal
import sys

from aka.legacy.application.execution import application_execution
from aka.legacy.application.processes import invocation_environment, use_launch_environment
from orchestrator.environment_recovery import (
    ENVIRONMENT_TEMPFAIL, EnvironmentUnavailable, current_recovery_context,
    environment_is_blocked, launch_recovery_monitor, signal_restart_ready,
)
from orchestrator.trace_retention import write_trace_retention_manifest
from .invocation import decode_launch
from .preparation import prepare_request, FrameworkDispatchComplete


class WorkflowStartup:
    def __init__(self, workflow):
        self._workflow = workflow

    def run(self, invocation):
        launch = decode_launch(invocation.argv)
        script = invocation.bindings.get("optimizer_script")
        if not script:
            raise ValueError("Workflow launch requires an optimizer_script process binding")
        with invocation_environment(invocation.environment), use_launch_environment(invocation.environment), application_execution():
            try:
                result = self._run(launch, Path(script))
            except EnvironmentUnavailable:
                result = ENVIRONMENT_TEMPFAIL
            if result == ENVIRONMENT_TEMPFAIL or environment_is_blocked():
                context = current_recovery_context()
                supervised = os.environ.get("ATREX_ENVIRONMENT_RESTART_SUPERVISED") == "1"
                pid = launch_recovery_monitor(context) if context is not None and not supervised else None
                if pid is not None:
                    print("[orchestrator] remote GPU environment unavailable; optimization stopped; "
                          f"recovery monitor pid={pid} state={context.directory}", file=sys.stderr, flush=True)
                return ENVIRONMENT_TEMPFAIL
            return result

    def _run(self, launch, script):
        # These are private calls within the Workflow implementation, not a
        # second module Interface. Problem is loaded once, before preparation.
        problem = self._workflow._load_problem(launch.request)
        try:
            request = prepare_request(launch, problem, script)
        except FrameworkDispatchComplete as completed:
            return completed.exit_code
        campaign = self._workflow._campaign(request, problem)
        trace_status = "failed"
        previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGHUP)}

        def interrupt(signum, _frame):
            nonlocal trace_status
            trace_status = "interrupted"
            raise SystemExit(128 + signum)

        try:
            for sig in previous:
                signal.signal(sig, interrupt)
            campaign.prepare(recovery_ready=signal_restart_ready if request.runtime.ssh else None)
            campaign.run()
            trace_status = "completed"
            return 0
        except KeyboardInterrupt:
            trace_status = "interrupted"
            raise
        finally:
            try:
                write_trace_retention_manifest(campaign.workspace, trace_status, hardware={
                    "platform": request.settings.platform, "arch": request.settings.arch or "",
                    "sandbox_hardware": request.runtime.hardware or request.settings.platform})
            finally:
                for sig, handler in previous.items():
                    signal.signal(sig, handler)
