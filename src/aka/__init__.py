"""Submit and reconnect to contract-bound optimization tasks."""
__version__ = "0.2.1"


def submit(**request):
    """Submit an optimization request using the installed task implementation."""
    from aka.light.tasks.api import submit as implementation
    return implementation(**request)


def job(task_id, *, root=None):
    """Reconnect to an existing optimization job."""
    from aka.light.tasks.api import Job
    return Job(task_id, root)
