"""Invocation environment retained for a persistent Light campaign and SSH recovery."""
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
import os
from aka.bootstrap.continuation import SELECTION_ENV, DIGEST_ENV, digest, read

_launch_environment = ContextVar("light_launch_environment", default=None)

@contextmanager
def use_launch_environment(environment):
    token = _launch_environment.set(dict(environment))
    try:
        yield
    finally:
        _launch_environment.reset(token)


def child_environment():
    return dict(_launch_environment.get() or {})


def recovery_environment(directory: Path, *, persist=True):
    environment = child_environment()
    payload = read(environment)
    if payload is None:
        return {}
    path = directory / "launch-selection.json"
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise RuntimeError("refusing to replace recovery launch selection with a different composition")
    elif persist:
        from aka.core.filesystem import durable_write_text
        durable_write_text(path, payload)
    return {**environment, SELECTION_ENV: str(path), DIGEST_ENV: digest(payload)}


@contextmanager
def invocation_environment(values):
    previous = {name: os.environ.get(name) for name in values}
    def apply(environment):
        for name, value in environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    apply(values)
    try:
        yield
    finally:
        apply(previous)
