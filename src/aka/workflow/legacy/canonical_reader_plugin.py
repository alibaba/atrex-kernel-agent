"""Publish Workflow-owned canonical facts independently of its consumers."""
from aka.core.keys import ServiceKey
from .records import WorkflowMeasurementReader

CANONICAL_MEASUREMENT_READER = ServiceKey("canonical_measurement_reader", "CanonicalMeasurementReader",
    module="aka.contracts.workflow_records")
name = "canonical-reader-plugin"
provide = ("canonical_measurement_reader",)
identity_packages = ("aka.workflow.legacy", "aka.contracts.workflow_records")
Config = {"type": "object", "properties": {}, "additionalProperties": False}
Defaults = {}


def apply(ctx, config):
    ctx.provide(CANONICAL_MEASUREMENT_READER, WorkflowMeasurementReader())
