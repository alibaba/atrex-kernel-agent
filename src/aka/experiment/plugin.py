"""Select Experiment storage and command evaluation for the Workflow host."""
from aka.core.keys import ServiceKey
from .factory import LegacyExperimentFactory
from aka.contracts.execution_recovery import InfrastructureRetryPolicy

EXPERIMENT_FACTORY = ServiceKey("experiment_factory", "ExperimentFactory", module="aka.contracts.experiment")
name = "experiment"
provide = ("experiment_factory",)
inject = ("canonical_measurement_reader",)
identity_packages = ("aka.experiment", "aka.contracts.experiment")
Config = {"type": "object", "properties": {"records_root": {"type": "string"},
    "retry_delays": {"type": "array", "items": {"type": "number", "minimum": 0}},
    "retry_budget_s": {"type": "number", "minimum": 0}}, "additionalProperties": False}
Defaults = {"retry_delays": list(InfrastructureRetryPolicy().delays),
            "retry_budget_s": InfrastructureRetryPolicy().max_elapsed_s}

def validate_config(config):
    InfrastructureRetryPolicy(tuple(config["retry_delays"]), config["retry_budget_s"])
interpolate = ("records_root",)


def apply(ctx, config):
    factory = LegacyExperimentFactory(measurement_reader=ctx.canonical_measurement_reader,
        records_root=config.get("records_root"),
        retry_policy=InfrastructureRetryPolicy(tuple(config["retry_delays"]), config["retry_budget_s"]))
    ctx.effect(factory.close, label="close Experiment selection")
    ctx.provide(EXPERIMENT_FACTORY, factory)
