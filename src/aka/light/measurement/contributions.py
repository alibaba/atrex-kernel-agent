"""Measurement tools and usage instructions; no Harness or feature imports."""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

from aka.contracts.contributions import InstructionContribution, ModuleContributions, ToolContribution


class MeasurementExecution(Protocol):
    def __call__(self, context: object, argv: list[str], *, prepare_only: bool = False) -> dict | None: ...


def object_schema(properties, required=()):
    return {"type": "object", "properties": properties, "required": list(required), "additionalProperties": False}


def declarations() -> ModuleContributions:
    text = {"type": "string", "minLength": 1}
    return ModuleContributions("measurement", "1", tools=(
        ToolContribution("measurement.evaluate", "Evaluate the current candidate using the selected evaluator; returns its retained measurement result.",
            object_schema({"mode": {"type": "string", "enum": ["full", "correctness_only"]},
                           "shape_ids": {"type": "array", "items": text},
                           "multi_seed": {"type": "integer", "minimum": 0},
                           "timed_runs": {"type": "integer", "minimum": 1},
                           "baseline_path": text}), {"type": "object"}),
        ToolContribution("measurement.record_read", "Read one retained measurement by its record ID; performs no GPU submission.",
                         object_schema({"record_id": text}, ("record_id",)), {"type": "object"}),
        ToolContribution("measurement.kernel_records", "List the measurements for a retained Kernel ID; performs no GPU submission.",
                         object_schema({"kernel_id": text}, ("kernel_id",)), {"type": "object"}),
    ), instructions=(InstructionContribution("measurement.usage", Path(__file__).with_name("usage.md").read_text()),))


def arguments(tool_id: str, value: dict) -> list[str]:
    if tool_id == "measurement.evaluate":
        result = ["--kind", "run", "--mode", value.get("mode", "full")]
        for name in ("multi_seed", "timed_runs", "baseline_path"):
            if name in value:
                result += ["--" + name.replace("_", "-"), str(value[name])]
        for shape in value.get("shape_ids", ()):
            result += ["--shape-id", shape]
        return result
    if tool_id == "measurement.record_read":
        return ["--kind", "record-read", "--record-id", value["record_id"]]
    if tool_id == "measurement.kernel_records":
        return ["--kind", "kernel-records", "--kernel-id", value["kernel_id"]]
    raise ValueError("unknown Measurement tool")


def handlers(execute: MeasurementExecution):
    """Return independent bindings; the composition root owns their registration."""
    result = {}
    for tool in declarations().tools:
        def invoke(context, value, *, tool_id=tool.id):
            return execute(context, arguments(tool_id, value))

        def prepare(context, value, *, tool_id=tool.id):
            return execute(context, arguments(tool_id, value), prepare_only=True)

        result[tool.id] = (invoke, prepare)
    return result
