"""Default observer for optional run fact consumers."""
from aka.core.keys import ServiceKey

WORKFLOW_OBSERVER = ServiceKey("workflow_observer", "WorkflowObserver", module="aka.contracts.workflow")
name = "observer"
provide = ("workflow_observer",)
identity_files = ("observer.py",)
Config = {"type": "object", "properties": {}, "additionalProperties": False}
Defaults = {}


class DefaultWorkflowObserver:
    def run_started(self, launch):
        pass

    def run_finished(self, exit_code, error=""):
        pass

    def campaign_started(self, request):
        pass


def apply(ctx, config):
    contribute_phase_instructions(ctx)
    ctx.provide(WORKFLOW_OBSERVER, DefaultWorkflowObserver())


def contribute_phase_instructions(ctx):
    """Observation discipline remains available when every Wiki is disabled."""
    from aka.contracts.agent_instructions import InstructionSpec
    value = InstructionSpec(
        name="workflow.phase-evidence",
        phases=("setup", "episode", "fast_episode", "framework_baseline"),
        template=("### Phase evidence discipline\n\n"
                  "Run phase-start and phase-end while the work happens, as standalone tool calls. "
                  "End the active phase before starting another. Start correctness before its full "
                  "command and end it after the result; use the same order for benchmark. "
                  "Each repeated evidence cycle owns its interval. Emit markers during that interval."),
    )
    ctx.contribute("instructions", value.name, value)
