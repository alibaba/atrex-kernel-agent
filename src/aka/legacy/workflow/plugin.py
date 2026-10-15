"""Provide Workflow with the selected Agent module, leaving Core outside its business objects."""

from aka.core.keys import ServiceKey

WORKFLOW = ServiceKey("workflow", "Workflow", module="aka.contracts.workflow")
name = "workflow"
provide = ("workflow",)
inject = ("agent", "problem", "source", "candidate", "evaluator_factory")
identity_packages = (
    "aka.legacy.workflow",
    "aka.contracts.workflow",
    "aka.contracts.experiment",
    "aka.contracts.evaluation",
    "aka.contracts.acceptance",
    "aka.legacy.evaluation",
    "aka.legacy.acceptance",
    "aka.legacy.experiment",
    "aka.legacy.application",
    "aka.legacy.execution",
    "aka.contracts.startup",
)
Config = {"type": "object", "properties": {"startup": {"type": "boolean"}}, "additionalProperties": False}

Defaults = {"startup": False}


def apply(ctx, config):
    from .composition import compose_workflow

    workflow = compose_workflow(
        agent=ctx.agent,
        task=dict(problem=ctx.problem, source=ctx.source, workspace=ctx.candidate),
        evaluator_factory=ctx.evaluator_factory,
    )
    ctx.effect(workflow.close, label="close Workflow selection")
    ctx.provide(WORKFLOW, workflow)

    if config.get("startup", False):
        from .startup import WorkflowStartup
        ctx.provide("startup", WorkflowStartup(workflow))
