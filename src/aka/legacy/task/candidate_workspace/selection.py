"""Bind the selected workspace provider to original application callers."""
from contextlib import contextmanager
from contextvars import ContextVar

from aka.legacy.task.candidate_workspace.composition import compose_candidate

_candidate = ContextVar("aka_candidate_workspace", default=None)


@contextmanager
def candidate_workspace(provider=None, **selection):
    if provider is not None and selection:
        raise ValueError("an injected workspace cannot also select a plugin")
    current = _candidate.get()
    if provider is None and current is not None and not selection:
        yield current
        return
    owner = None
    if provider is None:
        if selection:
            owner = compose_candidate(**selection)
            provider = owner.candidate
        else:
            from aka.legacy.task.candidate_workspace.provider import GitCandidateWorkspace
            provider = GitCandidateWorkspace()
    token = _candidate.set(provider)
    try:
        yield provider
    finally:
        _candidate.reset(token)
        if owner is not None:
            owner.dispose()

