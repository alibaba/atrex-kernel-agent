"""Run the selected Workflow; all child campaigns stay inside Workflow."""
import os
import signal
import sys

from aka.legacy.workflow.process_environment import invocation_environment, use_launch_environment
from aka.execution.recovery import ENVIRONMENT_TEMPFAIL, EnvironmentUnavailable, current_recovery_context, environment_is_blocked
from .recovery_runtime import launch_recovery_monitor, signal_restart_ready
from aka.legacy.workflow.trace_retention import write_trace_retention_manifest
from .invocation import decode_launch
from .preparation import prepare_request


class WorkflowStartup:
    def __init__(self, workflow, *, configuration=None, observer=None, evaluation_defaults=None,
                 execution_environment=None, evaluator_factory=None):
        self._workflow = workflow
        self.configuration = configuration or {}
        from .rule_sets import validate_config
        validate_config(self.configuration)
        self.observer = observer
        from aka.contracts.evaluation import EvaluationConfiguration
        self.evaluation_defaults = evaluation_defaults if evaluation_defaults is not None else EvaluationConfiguration()
        if not isinstance(self.evaluation_defaults, EvaluationConfiguration):
            raise TypeError("evaluation defaults must use EvaluationConfiguration")
        self.execution_environment = dict(execution_environment or {})

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
        profile_settings = {"conversion_budget_action": self.configuration.get("conversion_budget_action", "fail"),
                            **self.configuration.get("settings_defaults", {})}
        profile_runtime = {"shape_batch_size": self.evaluation_defaults.shape_batch_size,
                           **self.configuration.get("runtime_defaults", {})}
        if launch.provided_settings is not None:
            updates = {key: tuple(value) if isinstance(value, list) else value
                       for key, value in profile_settings.items() if key not in launch.provided_settings}
            settings = replace(settings, **updates)
        if launch.provided_runtime is not None:
            runtime = replace(runtime, **{key: value for key, value in profile_runtime.items()
                                         if key not in launch.provided_runtime})
        problem = launch.request.problem
        original = launch.request.settings
        if launch.provided_settings is not None and any(
            getattr(original, name) != getattr(settings, name)
            for name in ("optimization_mode", "framework", "platform")
        ):
            from .cli_problem import problem_request
            def generated(setting):
                return problem_request(problem.directory, setting.optimization_mode,
                    workspace=launch.workspace_base,
                    suffix=launch.workspace_suffix or f"{setting.framework}_{setting.platform}")
            if problem.snapshot_directory == generated(original).snapshot_directory:
                problem = generated(settings)
        return replace(launch, request=replace(launch.request,problem=problem,settings=settings,runtime=runtime),
                       provided_settings=None, provided_runtime=None)

    def run(self, invocation):
        launch = decode_launch(invocation.argv)
        from dataclasses import replace
        from .invocation import encode_launch
        launch = self._effective_launch(launch)
        values = dict(invocation.environment)
        # External recovery replays the effective request, never profile defaults.
        launch = replace(launch, restart_arguments=encode_launch(replace(launch, restart_arguments=())))
        result, error = 1, ""
        with invocation_environment(values), use_launch_environment(values):
            self._notify("run_started", launch)
            try:
                try:
                    result = self._run(launch)
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

    def _run(self, launch):
        import shutil
        runtime = launch.request.settings.agent_cli
        if shutil.which(runtime) is None:
            raise ValueError(f'Agent CLI is unavailable: {runtime}')
        if self.execution_environment.get("ATREX_SANDBOX_SSH") and any(shutil.which(command) is None for command in ('ssh', 'scp')):
            raise ValueError('selected SSH execution requires ssh and scp')
        # These are private calls within the Workflow implementation, not a
        # second module Interface. Problem is loaded once, before preparation.
        problem = self._workflow._load_problem(launch.request)
        from .checkpoint import CONTEXT
        from .export import document
        context = document(launch.request.workspace / CONTEXT)
        if context is not None:
            frozen = decode_launch(tuple(context['argv'])).request
            if launch.request != frozen:
                raise ValueError('existing campaign requires its frozen request for resume')
        request, frameworks = prepare_request(launch, problem, frozen=context is not None)
        if not launch.request.settings.framework:
            from .dispatch import dispatch_framework_campaigns
            from dataclasses import replace
            from aka.task.hardware import framework_workspace_suffix
            def run_framework(framework):
                # Each selected DSL keeps its own trusted preflight, without
                # reloading Task or rebuilding the selected modules.
                if self.execution_environment.get("ATREX_SANDBOX_SSH"):
                    from aka.execution.health import runtime_health_command
                    from aka.execution.sandbox import check_ssh_environment
                    os.environ["ATREX_SANDBOX_RUNTIME_HEALTH_COMMAND"] = runtime_health_command(
                        sol=problem.initial_source.kind == "sol", framework=framework)
                    check_ssh_environment(launch.workspace_base,
                        request.runtime.hardware or request.settings.platform,
                        self.execution_environment["ATREX_SANDBOX_SSH"],
                        self.execution_environment.get("ATREX_SANDBOX_SSH_INIT", ""),
                        self.execution_environment.get("ATREX_SANDBOX_HEALTH_COMMAND", ""))
                suffix = framework_workspace_suffix(framework, request.settings.platform, request.settings.optimization_mode)
                child_request = replace(request, settings=replace(request.settings, framework=framework))
                child_request = replace(child_request,
                    workspace=launch.workspace_base / f"kernel_opt_{problem.name}_{suffix}",
                    name=problem.name + "_" + suffix)
                return self._run_campaign(child_request, problem)

            return dispatch_framework_campaigns(frameworks, run_framework)
        return self._run_campaign(request, problem)

    def _run_campaign(self, request, problem):
        from .public_state import campaign_lease
        with campaign_lease(request.workspace, writer=True):
            return self._run_campaign_with_lease(request, problem)

    def _run_campaign_with_lease(self, request, problem):
        campaign = self._workflow._campaign(request, problem)
        from .checkpoint import record_context
        record_context(campaign)
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
                if sys.exc_info()[0] is None:
                    record_context(campaign)
                else:
                    try:
                        record_context(campaign)
                    except Exception as error:
                        print(f'[workflow] checkpoint context unavailable: {type(error).__name__}', file=sys.stderr)
                write_trace_retention_manifest(campaign.workspace, trace_status, hardware={
                    "platform": request.settings.platform, "arch": request.settings.arch or "",
                    "sandbox_hardware": request.runtime.hardware or request.settings.platform})
            finally:
                for sig, handler in previous.items():
                    signal.signal(sig, handler)
