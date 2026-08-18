#!/usr/bin/env python3
"""wiki-gate: paths and tunables.

Everything location-dependent lives here so gate.py and match.py stay portable.
"""
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parent.parent
GPU_WIKI = SKILL_DIR.parent.parent                       # gpu-wiki/
KERNEL_WIKI = GPU_WIKI / "kernel_wiki"
RECORDS_ROOT = KERNEL_WIKI / "records"
INDEX_PATH = RECORDS_ROOT / "index.json"
CONFLICTS_DIR = KERNEL_WIKI / "conflicts"
SCHEMA_PATH = GPU_WIKI / "schema" / "kernel" / "schema.json"

MATCH_CONFIDENCE_THRESHOLD = 0.8
SCHEMA_VERSION = "clean-1.3"
