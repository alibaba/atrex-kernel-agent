"""Provide Workflow with the selected Agent module, leaving Core outside its business objects."""

from aka.core.keys import ServiceKey

WORKFLOW = ServiceKey("workflow", "Workflow", module="aka.contracts.workflow")
name = "workflow"
provide = ("workflow",)
inject = ("agent",)
identity_packages = ("aka.legacy.workflow", "aka.contracts.workflow")
Config = {"type": "object", "properties": {}, "additionalProperties": False}


def apply(ctx, config):
    from .assembly import LegacyWorkflow

    workflow = LegacyWorkflow(ctx.agent)
    ctx.effect(workflow.close, label="close Workflow selection")
    ctx.provide(WORKFLOW, workflow)
