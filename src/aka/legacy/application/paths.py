"""Resolve a required matching checkout without depending on package depth."""

import os
import sys
import importlib.util
from pathlib import Path


def checkout_root():
    configured = os.environ.get("AKA_LEGACY_REPO_ROOT")
    if configured:
        root = Path(configured).expanduser().resolve()
        if not (root / "orchestrator/optimize.py").is_file():
            raise RuntimeError("AKA_LEGACY_REPO_ROOT does not identify an AKA checkout")
        return root
    for parent in Path(__file__).resolve().parents:
        if (parent / "orchestrator/optimize.py").is_file():
            return parent
    # Installed adapters use the checkout already selected by the application.
    loaded = sys.modules.get("orchestrator")
    origin = getattr(loaded, "__file__", None)
    locations = getattr(loaded, "__path__", ())
    if origin is None:
        spec = importlib.util.find_spec("orchestrator")
        origin = spec.origin if spec is not None else None
        locations = spec.submodule_search_locations if spec is not None else ()
    if origin:
        root = Path(origin).resolve().parent.parent
        if (root / "orchestrator/optimize.py").is_file():
            return root
    # orchestrator is a namespace package in matching source checkouts.
    for location in locations or ():
        root = Path(location).resolve().parent
        if (root / "orchestrator/optimize.py").is_file():
            return root
    raise RuntimeError("a matching AKA checkout is required; set AKA_LEGACY_REPO_ROOT")
