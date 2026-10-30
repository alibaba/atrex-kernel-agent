"""Own retained measurement CLI conversion and its preparation preview."""
from dataclasses import replace
import json

from aka.contracts.execution import UnsupportedExecution
from aka.execution.command_syntax import requested_operation
from .command_inputs import legacy_command_environment, validate_command, workload_environment


def register_measurement_options(parser):
    """Register measurement semantics; Execution retains its physical CLI options."""
    parser.add_argument("--shape-batch-size", type=int, default=None)
    parser.add_argument("--profile-level", choices=("survey", "sol", "deep"), default="sol")
    parser.add_argument("--profiler", choices=("ncu", "nsys", "rocprofv3", "acu"), default=None)
    parser.add_argument("--profile-counter", action="append", default=[], metavar="METRIC")
    parser.add_argument("--kernel-regex", default=None)
    parser.add_argument("--top-kernels", type=int, default=None)


def prepare_command(command, *, environment_items=(), shape_batch_size=None,
                    profile_level="sol", profiler=None, profile_counter=(), kernel_regex=None, top_kernels=None):
    """Convert explicit CLI options into the existing public command contract."""
    kind = requested_operation(command.operation, list(command.command))
    if command.profile_options is not None and kind != "profile":
        raise UnsupportedExecution("profile options require a profile command")
    from aka.contracts.execution import ProfileOptions
    options = command.profile_options
    selected = ProfileOptions(level=profile_level, profiler=profiler, counters=tuple(profile_counter),
                              kernel_regex=kernel_regex, top_kernels=top_kernels)
    if kind == "profile" and options is None:
        options = selected
    elif kind != "profile" and (profile_level != "sol" or profiler is not None or profile_counter or kernel_regex is not None or top_kernels is not None):
        raise UnsupportedExecution("profile options require a profile command")
    environment = legacy_command_environment(command.environment, environment_items)
    if shape_batch_size is not None:
        if type(shape_batch_size) is not int or shape_batch_size < 1:
            raise ValueError("shape batch size must be positive")
        environment["ATREX_EVAL_SHAPE_BATCH_SIZE"] = str(shape_batch_size)
    command = replace(command, environment=environment, profile_options=options,
        operation=kind if options is not None else command.operation)
    workload_environment(command, environment_items)
    validate_command(command)
    return command


def preview_command(command, *, evaluator):
    """Print the measurement preview without exposing opaque context to callers."""
    plan = evaluator.prepare(command)
    context = plan.context
    if not isinstance(context, dict) or "measurement" not in context:
        print(json.dumps({"hardware": command.hardware, "workspace": str(command.workspace),
            "operation": command.operation, "step_count": len(plan.steps),
            "measurement_identities": [identity.digest for identity in plan.measurement_identities]}, indent=2))
        return True
    measurement = context["measurement"]
    if measurement is None:
        return False
    print(json.dumps({"hardware": command.hardware, "workspace": str(command.workspace),
        "kind": context["kind"], "shape_count": "private" if context["private"] else len(measurement.shape_ids),
        "shape_batch_count": len(plan.steps), "num_gpus": context["num_gpus"],
        "mode": measurement.request.get("mode"), "options": measurement.request["options"]}, indent=2))
    return True
