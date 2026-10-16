"""Scope explicitly declared launch environment and resource import paths."""
from contextlib import contextmanager
import os
from pathlib import Path
import sys


@contextmanager
def selection_runtime(environment, python_paths=()):
    previous = {name: os.environ.get(name) for name in environment}
    added = []
    try:
        for name, value in environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        for path in reversed(python_paths):
            value = str(Path(path).resolve())
            if value not in sys.path:
                sys.path.insert(0, value)
                added.append(value)
        yield
    finally:
        for value in added:
            if value in sys.path:
                sys.path.remove(value)
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
