"""Supported Workflow launch and continuation entry points."""

def run_workflow(*args, **kwargs):
    from .launch import run_workflow as launch
    return launch(*args, **kwargs)

def adapt_argv(argv):
    """Convert the retained optimization CLI into canonical WorkflowLaunch arguments."""
    from .cli_conversion import adapt_argv as adapt
    return adapt(argv)

def use_optimizer_entrypoint(*args, **kwargs):
    from .launcher import use_optimizer_entrypoint as scope
    return scope(*args, **kwargs)

def invocation_environment(values):
    from .process_environment import invocation_environment as scope
    return scope(values)
