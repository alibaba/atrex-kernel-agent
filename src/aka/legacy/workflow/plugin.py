"""Provide Workflow with the selected Agent module, leaving Core outside its business objects."""

from aka.core.keys import ServiceKey

WORKFLOW = ServiceKey("workflow", "Workflow", module="aka.contracts.workflow")
name = "workflow"
provide = ("workflow",)
inject = ("agent", "problem", "source", "candidate", "evaluator_factory", "workflow_observer")
identity_packages = (
    "aka.legacy.workflow",
    "aka.legacy.task",
    "aka.legacy.resources",
    "aka.contracts.workflow",
    "aka.contracts.experiment",
    "aka.contracts.evaluation",
    "aka.contracts.acceptance",
    "aka.legacy.evaluation",
    "aka.legacy.acceptance",
    "aka.legacy.experiment",
    "aka.legacy.execution",
    "aka.contracts.startup",
)
Config = {"type": "object", "properties": {
    "startup": {"type": "boolean"},
    "reference_projects": {"type": "array", "items": {"type": "string"}},
    "conversion_budget_action": {"type": "string", "enum": ["fail", "retain_verified"]},
}, "additionalProperties": False}
Defaults = {"startup": False, "conversion_budget_action": "fail", "reference_projects": []}


def apply(ctx, config):
    from .resources import contribute
    contribute(ctx, config)
    from .composition import compose_workflow

    workflow = compose_workflow(
        agent=ctx.agent,
        task=dict(problem=ctx.problem, source=ctx.source, workspace=ctx.candidate),
        evaluator_factory=ctx.evaluator_factory,
        workflow_observer=ctx.workflow_observer,
    )
    ctx.effect(workflow.close, label="close Workflow selection")
    ctx.provide(WORKFLOW, workflow)

    if config.get("startup", False):
        from .startup import WorkflowStartup
        ctx.provide("startup", WorkflowStartup(workflow, configuration=config, observer=ctx.workflow_observer,
                                              evaluator_factory=ctx.evaluator_factory))
