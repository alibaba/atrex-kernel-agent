"""Select Experiment storage and command evaluation for the Workflow host."""
from aka.core.keys import ServiceKey
from .factory import LegacyExperimentFactory

EXPERIMENT_FACTORY = ServiceKey("experiment_factory", "ExperimentFactory", module="aka.contracts.experiment")
name = "experiment"
provide = ("experiment_factory",)
inject = ("canonical_measurement_reader",)
identity_packages = ("aka.legacy.experiment", "aka.contracts.experiment")
Config = {"type": "object", "properties": {"records_root": {"type": "string"}}, "additionalProperties": False}
Defaults = {}
interpolate = ("records_root",)


def apply(ctx, config):
    factory = LegacyExperimentFactory(measurement_reader=ctx.canonical_measurement_reader,
        records_root=config.get("records_root"))
    ctx.effect(factory.close, label="close Experiment selection")
    ctx.provide(EXPERIMENT_FACTORY, factory)
