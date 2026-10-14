"""Select or borrow initial sources; workspace operations remain independent."""
from contextlib import contextmanager
from aka.legacy.task.source.composition import compose_source


@contextmanager
def source_provider(kind="kernel", *, provider=None, selection=None, **config):
    if provider is not None:
        if config or selection:
            raise ValueError("an injected source cannot also select or configure a plugin")
        yield provider
    else:
        with compose_source(kind, config, **(selection or {})) as owner:
            yield owner.source

