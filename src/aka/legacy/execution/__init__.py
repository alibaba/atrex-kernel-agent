"""Execution mechanisms and providers, independent of application policy."""


def compose_execution(**options):
    """Own an Executor and its resources selected by the Execution profile."""
    from .assembly import compose_execution as compose
    return compose(**options)
