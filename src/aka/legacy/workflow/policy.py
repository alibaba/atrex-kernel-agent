"""Default mandatory conversion policy with a publicly replaceable decision."""
from aka.core.keys import ServiceKey

WORKFLOW_POLICY = ServiceKey("workflow_policy", "WorkflowPolicy", module="aka.contracts.workflow")
name = "policy"
provide = ("workflow_policy",)
identity_files = ("policy.py",)
Config = {"type": "object", "properties": {}, "additionalProperties": False}
Defaults = {}


class DefaultWorkflowPolicy:
    def conversion_budget(self, facts):
        return None


def apply(ctx, config):
    ctx.provide(WORKFLOW_POLICY, DefaultWorkflowPolicy())
