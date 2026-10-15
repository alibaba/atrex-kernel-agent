"""Locate the explicitly supported checkout resources without importing old code."""
import os
import sys
from pathlib import Path


def checkout_root():
    configured = os.environ.get("AKA_LEGACY_REPO_ROOT")
    def is_checkout(root):
        return (root / "tools/sandbox.py").is_file() and (root / "reference").is_dir()
    if configured:
        root = Path(configured).expanduser().resolve()
        if not is_checkout(root):
            raise RuntimeError("AKA_LEGACY_REPO_ROOT does not identify AKA checkout resources")
        return root
    for root in (*Path(__file__).resolve().parents, Path.cwd(), *(Path(p) for p in sys.path if p)):
        if is_checkout(root):
            return root.resolve()
    raise RuntimeError("matching AKA checkout resources are required; set AKA_LEGACY_REPO_ROOT")
