"""Run measurement commands through composition-selected public services."""
from dataclasses import replace
import argparse
import os
from pathlib import Path
import sys

from aka.core.keys import ServiceKey

PROFILES_DIR = Path(__file__).with_name("profiles")
CANONICAL_READER = ServiceKey("canonical_measurement_reader", "CanonicalMeasurementReader",
    module="aka.contracts.workflow_records")


def _composition_options(parser):
    parser.add_argument("--composition-profile", type=Path, default=PROFILES_DIR / "measurement.json",
                        help="Core profile selecting Execution, Evaluation and Experiment services")
    parser.add_argument("--composition-patch", type=Path, action="append", default=[],
                        help="patch the selected command composition")


def main(argv=None):
    if os.environ.get("AKA_EXPERIMENT_COMMAND_ENDPOINT"):
        from aka.experiment.command_client import main as command_main
        return command_main(argv)
    from aka.execution.cli_adapter import build_parser, run_cli
    from aka.evaluation.cli_adapter import prepare_command, preview_command, register_measurement_options
    from aka.evaluation.plugin import EVALUATOR_FACTORY, EVALUATION_DEFAULTS
    from aka.experiment.plugin import EXPERIMENT_FACTORY
    from aka.execution.assembly import EXECUTOR
    from aka.execution.plugin import EXECUTION_CONTEXT, MEASUREMENT_PARALLELISM
    from aka.core.boot import boot, compose
    from aka.task.problem.inputs import PerformanceBaselineError

    arguments = list(sys.argv[1:] if argv is None else argv)
    boundary = arguments.index("--") if "--" in arguments else len(arguments)
    host_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    _composition_options(host_parser)
    options, remaining = host_parser.parse_known_args(arguments[:boundary])
    remaining.extend(arguments[boundary:])

    def register_options(parser):
        register_measurement_options(parser)
        _composition_options(parser)

    if any(item in {"--help", "-h"} for item in arguments[:boundary]):
        build_parser(register_measurement_options=register_options).parse_args(remaining)
        return 0

    profile = options.composition_profile.expanduser().resolve()
    selected = compose(profile.stem, profiles_dir=profile.parent,
                       patch_files=options.composition_patch)
    tokens = (EXECUTOR, EVALUATOR_FACTORY, EVALUATION_DEFAULTS, CANONICAL_READER, EXPERIMENT_FACTORY)
    report = boot(selected, tokens=(*tokens, EXECUTION_CONTEXT, MEASUREMENT_PARALLELISM),
                  required_services=tuple(token.name for token in tokens))
    try:
        executor = report.service(EXECUTOR.name)
        factory = report.service(EXPERIMENT_FACTORY.name)
        evaluator_factory = report.service(EVALUATOR_FACTORY.name)
        defaults = report.service(EVALUATION_DEFAULTS.name)
        reader = report.service(CANONICAL_READER.name)

        def command_evaluator(command, capabilities):
            configuration = replace(defaults, timeout=command.timeout, hardware=command.hardware,
                environment=command.environment, execution_capabilities=capabilities,
                measurement_reader=reader)
            return evaluator_factory.create_command(configuration, evaluation_input=None)

        def run(bound, command):
            method = command_evaluator(command, bound.capabilities)
            with factory.open(workspace=command.workspace, purpose="development",
                              executor=bound, command_evaluator=method) as experiment:
                return experiment.run_command(command)

        def preview(command, *, execution_capabilities):
            return preview_command(command, evaluator=command_evaluator(command, execution_capabilities))

        return run_cli(remaining, prepare_command=prepare_command, run_command=run,
            preview_command=preview, register_measurement_options=register_options, executor=executor)
    except PerformanceBaselineError as error:
        print(f"[sandbox] invalid performance baseline: {error.reason}; GPU submission skipped", file=sys.stderr)
        return 2
    finally:
        report.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
