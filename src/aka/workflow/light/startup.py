"""Launch only the persistent-session Light Workflow."""
import os
import signal
import sys
from aka.execution.recovery import ENVIRONMENT_TEMPFAIL, EnvironmentUnavailable, current_recovery_context, environment_is_blocked
from aka.execution.selection import execution_scope
from aka.contracts.workflow_launch import decode_launch, encode_launch
from .process_environment import invocation_environment, use_launch_environment
from .recovery_runtime import launch_recovery_monitor, signal_restart_ready

class WorkflowStartup:
    """Run Light while keeping all durable state in the Light Campaign layout."""

    def __init__(self, workflow, *, configuration=None, observer=None, evaluation_defaults=None, execution_environment=None):
        self._workflow = workflow
        self.configuration = configuration or {}
        self.observer = observer
        self.evaluation_defaults = evaluation_defaults
        self.execution_environment = execution_environment or {}

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

        return replace(launch, provided_settings=None, provided_runtime=None)

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
        from dataclasses import replace
        from aka.execution.configuration import selected_ssh
        from .runtime_preparation import prepare_runtime
        if selected_ssh() and any(shutil.which(command) is None for command in ("ssh", "scp")):
            raise ValueError("selected SSH execution requires ssh and scp")
        problem = self._workflow._load_problem(launch.request)
        arch, _endpoint = prepare_runtime(launch, problem)
        request = replace(launch.request, settings=replace(launch.request.settings, arch=arch))
        return self._run_campaign(request, problem)

    def _run_campaign(self, request, problem):
        from .long_horizon.lease import campaign_writer_lease

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
                    signal_restart_ready if self.execution_environment.get("ATREX_SANDBOX_SSH") else None
                )
            )
            campaign.run()
            return 0
        finally:
            for selected, handler in previous.items():
                signal.signal(selected, handler)
