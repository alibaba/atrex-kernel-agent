"""Select the managed light Workflow without changing legacy behavior."""
from aka.legacy.workflow.plugin import WORKFLOW

name = "workflow"
provide = ("workflow",)
inject = ("agent", "problem", "source", "candidate", "tool_runtime",
          "evaluator_factory", "workflow_observer")
identity_packages = ("aka.light", "aka.contracts", "aka.legacy.workflow",
                     "aka.legacy.execution")
Config = {"type": "object", "properties": {
    "startup": {"type": "boolean"},
    "numerical_validation": {"type": "boolean"},
    "goal_episodes": {"type": "boolean"},
    "generate_memory": {"type": "boolean"},
    "output_limit_recovery_retries": {"type": "integer", "minimum": 0, "maximum": 10},
    "episode_timeout_seconds": {"type": "number", "minimum": 0},
    "source_tree": {"type": "object", "properties": {
        "manifest": {"type": "string", "minLength": 1},
        "repository": {"type": "string", "minLength": 1},
        "directory": {"type": "string", "minLength": 1},
    }, "required": ["manifest"], "additionalProperties": False},
    "search": {"type": "object", "properties": {
        "epochs": {"type": "integer", "minimum": 1, "maximum": 10000},
        "trajectories": {"type": "integer", "minimum": 1, "maximum": 64},
        "episodes_per_trajectory": {"type": "integer", "minimum": 1, "maximum": 100},
        "max_parallel_episodes": {"type": "integer", "minimum": 1, "maximum": 64},
    }, "additionalProperties": False},
}, "additionalProperties": False}
Defaults = {"startup": False, "numerical_validation": True, "goal_episodes": False, "generate_memory": False,
            "output_limit_recovery_retries": 2, "episode_timeout_seconds": 28800, "search": {}}


def validate_config(config):
    from .provider import WorkflowOptions
    WorkflowOptions(**{key: value for key, value in config.items() if key != "startup"})


def apply(ctx, config):
    from .provider import LightWorkflow, WorkflowOptions
    workflow = LightWorkflow(ctx.agent, ctx.problem, ctx.source, ctx.candidate,
                             ctx.tool_runtime, workflow_observer=ctx.workflow_observer,
                             agent_resources=ctx.get("agent_resources"),
                             options=WorkflowOptions(**{
                                 key: value for key, value in config.items() if key != "startup"
                             }))
    ctx.effect(workflow.close, label="close light Workflow and private runtimes")
    ctx.provide(WORKFLOW, workflow)
    if config.get("startup", False):
        from aka.legacy.workflow.startup import WorkflowStartup
        ctx.provide("startup", WorkflowStartup(
            workflow, configuration=config, observer=ctx.workflow_observer,
            evaluator_factory=ctx.evaluator_factory,
        ))
