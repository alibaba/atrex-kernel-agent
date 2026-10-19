"""Workflow implementation and retained CLI argument conversion."""

def adapt_argv(argv):
    """Convert the retained optimization CLI into canonical WorkflowLaunch arguments."""
    from .cli_conversion import adapt_argv as adapt
    return adapt(argv)
