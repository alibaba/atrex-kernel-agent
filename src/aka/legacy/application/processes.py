"""Translate neutral continuation facts for preserved optimizer process launches."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
import json
import os

from aka.bootstrap.continuation import SELECTION_ENV, DIGEST_ENV, digest, read
from .recovery import MIGRATION_FILE, has_selection

_launch_environment = ContextVar("aka_legacy_launch_environment", default=None)
_schema3_default = ContextVar("aka_legacy_schema3_default", default=False)


@contextmanager
def default_recovery_profile():
    """Permit the legacy owner to record the built-in default after validation."""
    token = _schema3_default.set(True)
    try:
        yield
    finally:
        _schema3_default.reset(token)


def schema3_recovery_allowed():
    return _schema3_default.get()


def pending_recovery_directory():
    value = os.environ.get("ATREX_ENVIRONMENT_STATE_FILE", "").strip()
    if not value:
        return None
    directory = Path(value).expanduser().resolve().parent
    return directory if has_selection(directory / MIGRATION_FILE) else None


def requires_default_recovery():
    """Recognize selection-free main state before resolving or invoking plugins."""
    state_value = os.environ.get("ATREX_ENVIRONMENT_STATE_FILE", "").strip()
    if not state_value:
        return False
    state = Path(state_value).expanduser().resolve()
    selection = state.with_name("launch-selection.json")
    if not state.with_name("restart.json").is_file():
        if has_selection(selection):
            raise RuntimeError("recovery launch selection exists without restart metadata")
        return False
    try:
        existing = json.loads(state.with_name("restart.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"cannot validate active recovery metadata: {exc}") from exc
    if not isinstance(existing, dict):
        raise RuntimeError("cannot validate active recovery metadata: expected an object")
    if existing.get("schema_version") == 3 and "launch_environment" not in existing:
        if has_selection(selection):
            raise RuntimeError("schema 3 recovery has an unreferenced launch selection")
        return True
    # New records must be reconstructed, never resolved as a fresh default.
    recorded = existing.get("launch_environment")
    if not isinstance(recorded, dict):
        raise RuntimeError("active recovery metadata requires its recorded launch selection")
    recorded_payload, supplied_payload = read(recorded), read(os.environ)
    if recorded_payload is None or supplied_payload is None:
        raise RuntimeError("active recovery metadata requires its recorded launch selection")
    if recorded_payload != supplied_payload:
        raise RuntimeError("active recovery metadata has a different recorded launch selection")
    return False


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
        from orchestrator.durable_state import durable_write_text
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

