from aka.core.keys import ServiceKey
EXPERIMENTS = ServiceKey("light_experiments", "ExperimentFactory", module="aka.light.contracts")
name = "experiment"
provide = ("light_experiments",)
inject = ("light_evaluation",)
identity_packages = ("aka.light.experiment", "aka.light.supervisor.measurement_records", "aka.light.supervisor.gateway_jobs")
Config = {"type": "object", "properties": {}, "additionalProperties": False}

def apply(ctx, config):
    from .provider import RecordedExperimentFactory
    ctx.provide(EXPERIMENTS, RecordedExperimentFactory(ctx.light_evaluation))
