"""Provide Workflow with the selected Agent module, leaving Core outside its business objects."""

from aka.core.keys import ServiceKey

WORKFLOW = ServiceKey("workflow", "Workflow", module="aka.contracts.workflow")
name = "workflow"
provide = ("workflow",)
inject = ("agent", "problem", "source", "candidate")
identity_packages = (
    "aka.legacy.workflow",
    "aka.contracts.workflow",
    "aka.legacy.compat",
)
Config = {"type": "object", "properties": {}, "additionalProperties": False}


def apply(ctx, config):
    from aka.legacy.compat.assembly import default_workflow

    workflow = default_workflow(
        agent=ctx.agent,
        task=dict(problem=ctx.problem, source=ctx.source, workspace=ctx.candidate),
    )
    ctx.effect(workflow.close, label="close Workflow selection")
    ctx.provide(WORKFLOW, workflow)
