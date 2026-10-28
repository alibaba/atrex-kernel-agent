"""Own retained measurement CLI conversion and its preparation preview."""
from dataclasses import replace
import json

from aka.contracts.execution import UnsupportedExecution
from aka.execution.gateway import sandbox as transport
from .command_inputs import legacy_command_environment, validate_command, workload_environment


def prepare_command(command, *, environment_items=(), shape_batch_size):
    """Convert explicit CLI options into the existing public command contract."""
    kind = transport._requested_gateway_kind(command.operation, list(command.command))
    if command.profile_options is not None and kind != "profile":
        raise UnsupportedExecution("profile options require a profile command")
    environment = legacy_command_environment(command.environment, environment_items)
    environment["ATREX_EVAL_SHAPE_BATCH_SIZE"] = str(shape_batch_size)
    command = replace(command, environment=environment,
        operation=kind if command.profile_options is not None else command.operation)
    workload_environment(command, environment_items)
    validate_command(command)
    return command


def preview_command(command, *, evaluator):
    """Print the measurement preview without exposing opaque context to callers."""
    plan = evaluator.prepare(command)
    context = plan.context
    measurement = context["measurement"]
    if measurement is None:
        return False
    print(json.dumps({"hardware": command.hardware, "workspace": str(command.workspace),
        "kind": context["kind"], "shape_count": "private" if context["private"] else len(measurement.shape_ids),
        "shape_batch_count": len(plan.steps), "num_gpus": context["num_gpus"],
        "mode": measurement.request.get("mode"), "options": measurement.request["options"]}, indent=2))
    return True
