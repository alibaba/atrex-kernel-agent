"""Persistent task-contract entry point for parent agents and custom evaluators."""

def service(root):
    from .service import TaskService
    return TaskService(root)
