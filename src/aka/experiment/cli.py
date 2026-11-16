"""Run a CLI command in its injected Experiment Factory's namespace."""


def run_command(command, *, factory, executor, evaluator):
    with factory.open(workspace=command.workspace, purpose="development",
                      executor=executor, evaluator=evaluator) as experiment:
        return experiment.run_command(command)
