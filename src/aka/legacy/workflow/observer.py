"""Default observer for optional run fact consumers."""
from aka.core.keys import ServiceKey

WORKFLOW_OBSERVER = ServiceKey("workflow_observer", "WorkflowObserver", module="aka.contracts.workflow")
name = "observer"
provide = ("workflow_observer",)
identity_files = ("observer.py",)
Config = {"type": "object", "properties": {}, "additionalProperties": False}
Defaults = {}


class DefaultWorkflowObserver:
    def campaign_started(self, request):
        pass


def apply(ctx, config):
    ctx.provide(WORKFLOW_OBSERVER, DefaultWorkflowObserver())
