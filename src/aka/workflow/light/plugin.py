"""Select the persistent-session Light Workflow."""
from aka.core.keys import ServiceKey

WORKFLOW = ServiceKey("workflow", "Workflow", module="aka.contracts.workflow")

name = "light"
provide = ("workflow",)
inject = ("agent", "problem", "source", "candidate", "executor", "execution_context",
          "evaluator_factory", "evaluation_defaults", "experiment_factory", "measurement_parallelism", "workflow_observer")
identity_packages = ("aka.workflow.light", "aka.contracts",
                     "aka.execution")
Config = {"type": "object", "properties": {
    "file_system": {"type": "string", "enum": ["single_file", "source_tree", "cuda_manifest"]},
    "startup": {"type": "boolean"},
    "source_tree": {"type": "object"},
    "dependency_policy_recovery_retries": {"type": "integer", "minimum": 0, "maximum": 10},
    "goal_episodes": {"type": "boolean"},
    "output_limit_recovery_retries": {"type": "integer", "minimum": 0, "maximum": 10},
    "wiki": {"type": "boolean"},
    "wiki_options": {"type": "object"},
    "tool_modules": {"type": "array", "items": {"type": "string", "enum": ["directions", "experiments"]}},
    "inherit_state": {"type": "boolean"},
    "request_timeout": {"type": "number", "minimum": 0.001},
    "queue_timeout": {"type": "number", "minimum": 0.001},
    "search": {"type": "object", "properties": {
        "synchronization": {"type": "string", "enum": ["synchronous", "asynchronous"]},
        "role_specialization": {"type": "boolean"},
        "epochs": {"type": "integer", "minimum": 1, "maximum": 10000},
        "trajectories": {"type": "integer", "minimum": 1, "maximum": 64},
        "episodes_per_trajectory": {"type": "integer", "minimum": 1, "maximum": 100},
        "max_parallel_episodes": {"type": "integer", "minimum": 1, "maximum": 64},
    }, "additionalProperties": False},
}, "additionalProperties": False}
Defaults = {"startup": True,
            "wiki": True, "tool_modules": [], "inherit_state": True, "queue_timeout": 60,
            "output_limit_recovery_retries": 2, "search": {}}


def validate_config(config):
    from .provider import WorkflowOptions
    WorkflowOptions(**{key: value for key, value in config.items() if key != "startup"})


def apply(ctx, config):
    from aka.contracts.agent import AgentProcessPolicy
    from aka.execution.recovery import environment_is_blocked
    def unavailable(environment):
        if environment_is_blocked():
            return 75, "Remote execution environment unavailable; stopped for durable recovery"
        return None
    ctx.contribute("agent_process_policies", "workflow", AgentProcessPolicy(
        unavailable=unavailable))
    from .provider import LightWorkflow, WorkflowOptions
    workflow = LightWorkflow(ctx.agent, ctx.problem, ctx.source, ctx.candidate,
                             executor=ctx.executor, execution_context=ctx.execution_context,
                             experiment_factory=ctx.experiment_factory, evaluator_factory=ctx.evaluator_factory,
                             evaluation_defaults=ctx.evaluation_defaults, measurement_parallelism=ctx.measurement_parallelism,
                             workflow_observer=ctx.workflow_observer,
                             options=WorkflowOptions(**{
                                 key: value for key, value in config.items() if key != "startup"
                             }))
    ctx.effect(workflow.close, label="close light Workflow and private runtimes")
    ctx.provide(WORKFLOW, workflow)
    if config.get("startup", Defaults["startup"]):
        from .startup import WorkflowStartup
        ctx.provide("startup", WorkflowStartup(
            workflow, configuration=config, observer=ctx.workflow_observer,
            evaluation_defaults=ctx.evaluation_defaults, execution_environment=ctx.execution_context.environment,
        ))
