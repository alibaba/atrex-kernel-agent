"""Select the optimizer's Python launch script independently of module identity."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Iterator, Sequence

_entrypoint: ContextVar[Path | None] = ContextVar("optimizer_entrypoint", default=None)
_resources: ContextVar[tuple[Path, ...]] = ContextVar("optimizer_resources", default=())
_DEFAULT_ENTRYPOINT = Path(__file__).resolve().with_name("optimize.py")


def optimizer_entrypoint() -> Path:
    """Return the script shared by framework children and environment recovery."""
    selected = _entrypoint.get()
    return _DEFAULT_ENTRYPOINT if selected is None else selected


def optimizer_resources() -> tuple[Path, ...]:
    """Return explicit runtime bindings which the launcher reconstructs."""
    return _resources.get()


@contextmanager
def use_optimizer_entrypoint(script: Path, *, resources: Sequence[Path] = ()) -> Iterator[Path]:
    """Use a readable Python script for this invocation; restore it on every exit.

    The caller invokes optimize.main inside this scope. New processes execute
    the selected script with the existing interpreter and argv; that script is
    responsible for rebuilding its adapters. Bootstrap records the effective
    composition, and resources declare launcher-selected implementation inputs.

    Selection is local to the current context; worker threads must explicitly
    propagate that context or select their entrypoint again.
    """
    selected = Path(script).expanduser().resolve()
    if not selected.is_file():
        raise ValueError(f"optimizer entrypoint is not a file: {selected}")
    with selected.open("rb") as stream:
        stream.read(1)
    selected_resources = tuple(Path(path).expanduser().resolve() for path in resources)
    for path in selected_resources:
        if not path.exists():
            raise ValueError(f"optimizer resource does not exist: {path}")
    token = _entrypoint.set(selected)
    resource_token = _resources.set(selected_resources)
    try:
        yield selected
    finally:
        _resources.reset(resource_token)
        _entrypoint.reset(token)
