"""Workflow's internal process entry; Bootstrap owns its assembly lifetime."""
from pathlib import Path
import os
import signal
import sys

from aka.legacy.execution.selection import execution_scope
from aka.legacy.workflow.process_environment import invocation_environment, use_launch_environment
from aka.legacy.execution.recovery import ENVIRONMENT_TEMPFAIL, EnvironmentUnavailable, current_recovery_context, environment_is_blocked, launch_recovery_monitor, signal_restart_ready
from aka.legacy.workflow.trace_retention import write_trace_retention_manifest
from .invocation import decode_launch
from .preparation import prepare_request, FrameworkDispatchComplete


class WorkflowStartup:
    def __init__(self, workflow, *, configuration=None, observer=None, evaluator_factory=None):
        self._workflow = workflow
        self.configuration = configuration or {}
        self.observer = observer
        self.evaluator_factory = evaluator_factory

    def _notify(self, name, *args):
        callback = getattr(self.observer, name, None)
        if callback:
            try:
                callback(*args)
            except Exception as error:
                print(f"[workflow] auxiliary observer failed: {type(error).__name__}", file=sys.stderr)

    def _effective_launch(self, launch):
        from dataclasses import replace
        settings, runtime = launch.request.settings, launch.request.runtime
        if launch.provided_settings is not None and "conversion_budget_action" not in launch.provided_settings:
            settings = replace(settings, conversion_budget_action=self.configuration.get("conversion_budget_action", "fail"))
        if launch.provided_runtime is not None and "shape_batch_size" not in launch.provided_runtime:
            runtime = replace(runtime, shape_batch_size=getattr(self.evaluator_factory, "shape_batch_size", runtime.shape_batch_size))
        return replace(launch, request=replace(launch.request,settings=settings,runtime=runtime),
                       provided_settings=None, provided_runtime=None)

    def run(self, invocation):
        launch = decode_launch(invocation.argv)
        script = invocation.bindings.get("optimizer_script")
        if not script:
            script = str(Path(__file__).with_name("entry.py"))
        from dataclasses import replace
        from .invocation import encode_launch
        launch = self._effective_launch(launch)
        values = dict(invocation.environment)
        selected = self.configuration.get("execution_selection")
        if selected:
            import json
            inherited = values.get("AKA_EXECUTION_SELECTION") or os.environ.get("AKA_EXECUTION_SELECTION")
            if inherited and json.loads(inherited) != selected:
                raise ValueError("execution selection conflicts with selected Workflow profile")
            values["AKA_EXECUTION_SELECTION"] = json.dumps(selected, sort_keys=True)
        # Freeze the effective request for children; child dispatch uses canonical arguments.
        launch = replace(launch, child_arguments=(), child_bindings=())
        if Path(script).resolve() == Path(__file__).with_name("entry.py").resolve():
            launch = replace(launch, restart_arguments=encode_launch(replace(launch, restart_arguments=())))
        result, error = 1, ""
        with invocation_environment(values), use_launch_environment(values):
            self._notify("run_started", launch)
            try:
                with execution_scope():
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
                        result = ENVIRONMENT_TEMPFAIL
                    return result
            except BaseException as caught:
                result = 130 if isinstance(caught,KeyboardInterrupt) else (caught.code if isinstance(caught,SystemExit) and type(caught.code) is int else 1)
                error = type(caught).__name__
                raise
            finally:
                self._notify("run_finished", result, error)

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
            campaign.prepare(recovery_ready=signal_restart_ready if campaign.execution.uses_ssh else None)
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
