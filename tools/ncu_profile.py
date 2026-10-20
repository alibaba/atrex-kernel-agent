#!/usr/bin/env python3
"""Checkout launcher; Gateway bundles the self-contained implementation."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aka.legacy.evaluation.ncu_profile import main

if __name__ == "__main__":
    raise SystemExit(main())
