"""Compose retained measurement commands from their selected module owners."""
import os
import sys


def main(argv=None):
    if os.environ.get("AKA_EXPERIMENT_COMMAND_ENDPOINT"):
        from aka.experiment.command_client import main as command_main
        return command_main(argv)
    from aka.execution.cli_adapter import run_cli
    from aka.evaluation.cli_adapter import prepare_command, preview_command, register_measurement_options
    from aka.evaluation.evaluator import LegacyEvaluator
    from aka.evaluation.gateway_command import GatewayCommandEvaluator
    from aka.experiment.factory import LegacyExperimentFactory
    from aka.experiment.cli import run_command
    from aka.task.problem.inputs import PerformanceBaselineError
    from .records import WorkflowMeasurementReader

    reader = WorkflowMeasurementReader()
    factory = LegacyExperimentFactory(measurement_reader=reader)
    evaluator = LegacyEvaluator()

    def run(executor, command):
        return run_command(command, factory=factory, executor=executor, evaluator=evaluator)

    def preview(command, *, execution_capabilities):
        return preview_command(command, evaluator=GatewayCommandEvaluator(
            measurement_reader=reader, execution_capabilities=execution_capabilities))

    try:
        return run_cli(argv, prepare_command=prepare_command, run_command=run, preview_command=preview,
                       register_measurement_options=register_measurement_options)
    except PerformanceBaselineError as error:
        print(f"[sandbox] invalid performance baseline: {error.reason}; GPU submission skipped", file=sys.stderr)
        return 2
    finally:
        factory.close()


if __name__ == "__main__":
    raise SystemExit(main())
