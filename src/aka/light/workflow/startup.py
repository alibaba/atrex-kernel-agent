"""Launch AKA Light without Legacy Campaign state or checkpoint files."""

import os
import signal
import sys

from aka.legacy.execution.recovery import (
    ENVIRONMENT_TEMPFAIL,
    EnvironmentUnavailable,
    current_recovery_context,
    environment_is_blocked,
    launch_recovery_monitor,
    signal_restart_ready,
)
from aka.legacy.execution.selection import execution_scope
from aka.legacy.workflow.invocation import decode_launch, encode_launch
from aka.legacy.workflow.preparation import prepare_request
from aka.legacy.workflow.process_environment import (
    invocation_environment,
    use_launch_environment,
)


class LightWorkflowStartup:
    """Run Light while keeping all durable state in the Light Campaign layout."""

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
            except Exception as error:  # noqa: BLE001 - observers are fail-open
                print(
                    f"[workflow] auxiliary observer failed: {type(error).__name__}",
                    file=sys.stderr,
                )

    def _effective_launch(self, launch):
        from dataclasses import replace

        settings, runtime = launch.request.settings, launch.request.runtime
        if (
            launch.provided_settings is not None
            and "conversion_budget_action" not in launch.provided_settings
        ):
            settings = replace(
                settings,
                conversion_budget_action=self.configuration.get(
                    "conversion_budget_action", "fail"
                ),
            )
        return replace(
            launch,
            request=replace(launch.request, settings=settings, runtime=runtime),
            provided_settings=None,
            provided_runtime=None,
        )

    def run(self, invocation):
        from dataclasses import replace

        launch = self._effective_launch(decode_launch(invocation.argv))
        values = dict(invocation.environment)
        launch = replace(
            launch,
            restart_arguments=encode_launch(
                replace(launch, restart_arguments=())
            ),
        )
        result, error = 1, ""
        with invocation_environment(values), use_launch_environment(values):
            self._notify("run_started", launch)
            try:
                with execution_scope():
                    try:
                        result = self._run(launch)
                    except EnvironmentUnavailable:
                        result = ENVIRONMENT_TEMPFAIL
                    if result == ENVIRONMENT_TEMPFAIL or environment_is_blocked():
                        context = current_recovery_context()
                        supervised = (
                            os.environ.get("ATREX_ENVIRONMENT_RESTART_SUPERVISED") == "1"
                        )
                        pid = (
                            launch_recovery_monitor(context)
                            if context is not None and not supervised
                            else None
                        )
                        if pid is not None:
                            print(
                                "[orchestrator] remote GPU environment unavailable; "
                                "optimization stopped; "
                                f"recovery monitor pid={pid} state={context.directory}",
                                file=sys.stderr,
                                flush=True,
                            )
                        result = ENVIRONMENT_TEMPFAIL
                    return result
            except BaseException as caught:
                result = (
                    130
                    if isinstance(caught, KeyboardInterrupt)
                    else caught.code
                    if isinstance(caught, SystemExit) and type(caught.code) is int
                    else 1
                )
                error = type(caught).__name__
                raise
            finally:
                self._notify("run_finished", result, error)

    def _run(self, launch):
        import shutil

        runtime = launch.request.settings.agent_cli
        if shutil.which(runtime) is None:
            raise ValueError(f"Agent CLI is unavailable: {runtime}")
        from aka.legacy.execution.configuration import selected_ssh

        if selected_ssh() and any(
            shutil.which(command) is None for command in ("ssh", "scp")
        ):
            raise ValueError("selected SSH execution requires ssh and scp")
        problem = self._workflow._load_problem(launch.request)
        # Light resume identity is validated by campaign.json.  It must not
        # read or create Legacy .atrex_long_horizon checkpoint context.
        request, frameworks = prepare_request(launch, problem, frozen=False)
        if not launch.request.settings.framework and launch.request.settings.specify_dsl:
            return self._run_framework_campaigns(
                launch, request, problem, frameworks
            )
        return self._run_campaign(request, problem)

    def _run_framework_campaigns(self, launch, request, problem, frameworks):
        from dataclasses import replace

        from aka.legacy.task.hardware import framework_workspace_suffix
        from aka.legacy.workflow.dispatch import dispatch_framework_campaigns

        def run_framework(framework):
            from aka.legacy.execution.configuration import legacy_settings, selected_ssh

            if selected_ssh():
                from aka.legacy.execution.health import runtime_health_command
                from aka.legacy.execution.sandbox import check_ssh_environment

                os.environ["ATREX_SANDBOX_RUNTIME_HEALTH_COMMAND"] = (
                    runtime_health_command(
                        sol=problem.initial_source.kind == "sol",
                        framework=framework,
                    )
                )
                connection = legacy_settings()
                check_ssh_environment(
                    launch.workspace_base,
                    request.runtime.hardware or request.settings.platform,
                    connection.ssh,
                    connection.ssh_init,
                    connection.health_command,
                )
            suffix = framework_workspace_suffix(
                framework,
                request.settings.platform,
                request.settings.optimization_mode,
            )
            child_request = replace(
                request,
                settings=replace(request.settings, framework=framework),
            )
            child_request = replace(
                child_request,
                workspace=launch.workspace_base / f"kernel_opt_{problem.name}_{suffix}",
                name=problem.name + "_" + suffix,
            )
            return self._run_campaign(child_request, problem)

        return dispatch_framework_campaigns(frameworks, run_framework)

    def _run_campaign(self, request, problem):
        from .lease import campaign_writer_lease

        with campaign_writer_lease(request.workspace):
            return self._run_campaign_with_lease(request, problem)

    def _run_campaign_with_lease(self, request, problem):
        campaign = self._workflow._campaign(request, problem)
        previous = {
            selected: signal.getsignal(selected)
            for selected in (signal.SIGTERM, signal.SIGHUP)
        }

        def interrupt(signum, _frame):
            raise SystemExit(128 + signum)

        try:
            for selected in previous:
                signal.signal(selected, interrupt)
            campaign.prepare(
                recovery_ready=(
                    signal_restart_ready if campaign.execution.uses_ssh else None
                )
            )
            campaign.run()
            return 0
        finally:
            for selected, handler in previous.items():
                signal.signal(selected, handler)
