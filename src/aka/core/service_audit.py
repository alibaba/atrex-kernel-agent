"""Request-local service evidence, with storage owned by the calling workflow."""
from contextlib import contextmanager
from contextvars import ContextVar
import logging

_OBSERVER = ContextVar("aka_service_audit_observer", default=None)


@contextmanager
def capture_service_exchanges(observer):
    token = _OBSERVER.set(observer)
    try:
        yield
    finally:
        _OBSERVER.reset(token)


def audit_service_exchange(**exchange):
    observer = _OBSERVER.get()
    if observer is not None:
        try:
            observer(**exchange)
        except Exception:
            # Audit failure must not cause replay of an accepted service call.
            logging.getLogger(__name__).exception("Failed to persist service exchange audit")
