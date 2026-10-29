"""Provide Workflow with the selected Agent module, leaving Core outside its business objects."""

from dataclasses import asdict
from aka.core.keys import ServiceKey
from .rule_sets import DEFAULT_RULE_SET, LegacyRuleSet, defaults_schema, select_rule_set, validate_config
from aka.contracts.workflow import WorkflowSettings, CampaignRuntimeSettings

WORKFLOW = ServiceKey("workflow", "Workflow", module="aka.contracts.workflow")
name = "workflow"
provide = ("workflow",)
inject = ("agent", "problem", "source", "candidate", "executor", "execution_context", "measurement_parallelism",
          "evaluator_factory", "evaluation_defaults", "experiment_factory", "workflow_observer", "canonical_measurement_reader",
          "experiment_episode_evidence", "evaluation_episode_interpreter")
identity_packages = (
    "aka.legacy.workflow",
    "aka.task",
    "aka.resources",
    "aka.contracts.workflow",
    "aka.contracts.experiment",
    "aka.contracts.evaluation",
    "aka.contracts.acceptance",
    "aka.evaluation",
    "aka.legacy.acceptance",
    "aka.experiment",
    "aka.execution",
    "aka.contracts.startup",
)
Config = {"type": "object", "properties": {
    "startup": {"type": "boolean"},
    "reference_projects": {"type": "array", "items": {"type": "string"}},
    "conversion_budget_action": {"type": "string", "enum": ["fail", "retain_verified"]},
    "rule_set": {"type": "string", "enum": ["legacy-v1", DEFAULT_RULE_SET]},
    "settings_defaults": defaults_schema(WorkflowSettings),
    "runtime_defaults": defaults_schema(CampaignRuntimeSettings),
    "conversion_tolerance": {"type": "number", "minimum": 0},
    "goal_after_episodes": {"type": "integer", "minimum": 0},
    "goal_stall_threshold": {"type": "integer", "minimum": 0},
    "goal_handoff_resumes": {"type": "integer", "minimum": 0},
}, "additionalProperties": False}
Defaults = {"startup": False, "conversion_budget_action": "fail", "reference_projects": []}
Defaults.update(rule_set=DEFAULT_RULE_SET, settings_defaults={}, runtime_defaults={}, **asdict(LegacyRuleSet()))


def apply(ctx, config):
    from .resources import contribute
    contribute(ctx, config)
    workflow = select_rule_set(config).assemble(
        agent=ctx.agent,
        task=dict(problem=ctx.problem, source=ctx.source, workspace=ctx.candidate),
        evaluator_factory=ctx.evaluator_factory,
        executor=ctx.executor,
        execution_environment=ctx.execution_context.environment,
        measurement_parallelism=ctx.measurement_parallelism,
        experiment_factory=ctx.experiment_factory,
        canonical_measurement_reader=ctx.canonical_measurement_reader,
        experiment_episode_evidence=ctx.experiment_episode_evidence,
        evaluation_episode_interpreter=ctx.evaluation_episode_interpreter,
        workflow_observer=ctx.workflow_observer,
    )
    ctx.effect(workflow.close, label="close Workflow selection")
    ctx.provide(WORKFLOW, workflow)

    if config.get("startup", False):
        from .startup import WorkflowStartup
        ctx.provide("startup", WorkflowStartup(workflow, configuration=config, observer=ctx.workflow_observer,
            evaluation_defaults=ctx.evaluation_defaults, execution_environment=ctx.execution_context.environment))
