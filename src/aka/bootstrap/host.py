"""Own assembly, profile-selected invocation and cleanup for one process."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

from aka.contracts.startup import Invocation
from aka.core.boot import boot
from aka.core.effects import call_sync, require_sync
from .continuation import SELECTION_ENV, DIGEST_ENV, capture, digest, read, restore
from .profile import resolve_profile


def run_profile(profile: Path, invocation: Invocation, **options) -> int:
    """Start from a profile; recorded selections require explicit resume()."""
    if SELECTION_ENV in os.environ or DIGEST_ENV in os.environ:
        raise ValueError("launch selection already supplied; use resume() or clear AKA_LAUNCH_SELECTION and AKA_LAUNCH_DIGEST")
    return run_selection(resolve_profile(profile, **options), invocation)


def resume(invocation: Invocation, *, environment=None, tokens=()) -> int:
    """Reconstruct a recorded selection; its target owns business-state recovery."""
    payload = read(os.environ if environment is None else environment)
    if payload is None:
        raise ValueError("no continuation selection was supplied")
    return run_selection(restore(payload, tokens=tokens), invocation)


def run_selection(selection, invocation: Invocation) -> int:
    selection, payload = capture(selection)
    with tempfile.TemporaryDirectory(prefix="aka-launch-") as tmp:
        path = Path(tmp) / "selection.json"
        path.write_text(payload)
        path.chmod(0o600)
        continuation = {**selection.environment, SELECTION_ENV: str(path), DIGEST_ENV: digest(payload)}
        report = boot(selection.composition, tokens=selection.tokens,
                      required_services=selection.required_services)
        try:
            report.verify_composition()
            target = report.service(selection.target)
            entry = getattr(target, "run", None)
            if not callable(entry):
                raise TypeError("selected startup target must implement run(invocation)")
            require_sync(entry, "startup.run")
            result = call_sync(entry, Invocation(invocation.argv, continuation, selection.bindings))
            if type(result) is not int:
                raise TypeError("startup.run must return an integer exit code")
            return result
        finally:
            report.dispose()
