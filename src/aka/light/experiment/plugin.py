from aka.core.keys import ServiceKey
EXPERIMENTS = ServiceKey("light_experiments", "ExperimentFactory", module="aka.light.contracts")
name = "experiment"
provide = ("light_experiments",)
inject = ("light_evaluation",)
identity_packages = ("aka.light.experiment", "aka.light.experiment.store", "aka.light.execution.gateway_jobs")
Config = {"type": "object", "properties": {}, "additionalProperties": False}

def apply(ctx, config):
    from .provider import RecordedExperimentFactory
    ctx.provide(EXPERIMENTS, RecordedExperimentFactory(ctx.light_evaluation))
