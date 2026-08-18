#!/usr/bin/env python3
# Copyright 2026 Alibaba Group.
# Licensed under the Apache License, Version 2.0.

"""Canonical product identities shared by every hardware-wiki entry point.

Keys are the exact product addresses accepted internally by ``hardware_wiki``.
``recorded`` means an exact spec-sheet currently exists; the other identities
are still recognized so lookup can return a safe ``not-recorded`` procedure.
Architecture identifiers such as gfx942 are deliberately absent: one
architecture can back several products and must not silently select one SKU.
"""
from __future__ import annotations

import re


HARDWARE_IDENTITIES = {
    # NVIDIA Ampere
    "a100": {"vendor": "nvidia", "arch": "ampere", "recorded": False},
    "a800": {"vendor": "nvidia", "arch": "ampere", "recorded": False},
    "a30": {"vendor": "nvidia", "arch": "ampere", "recorded": False},
    "a10": {"vendor": "nvidia", "arch": "ampere", "recorded": False},
    # NVIDIA Ada
    "l20": {"vendor": "nvidia", "arch": "ada", "recorded": False},
    "l40s": {"vendor": "nvidia", "arch": "ada", "recorded": False},
    "l4": {"vendor": "nvidia", "arch": "ada", "recorded": False},
    # NVIDIA Hopper
    "h100": {"vendor": "nvidia", "arch": "hopper", "recorded": False},
    "h200": {"vendor": "nvidia", "arch": "hopper", "recorded": False},
    "h800": {"vendor": "nvidia", "arch": "hopper", "recorded": False},
    "h20": {"vendor": "nvidia", "arch": "hopper", "recorded": False},
    "gh200": {"vendor": "nvidia", "arch": "hopper", "recorded": False},
    # NVIDIA Blackwell
    "b100": {"vendor": "nvidia", "arch": "blackwell", "recorded": False},
    "b200": {"vendor": "nvidia", "arch": "blackwell", "recorded": True},
    "gb200": {"vendor": "nvidia", "arch": "blackwell", "recorded": False},
    # NVIDIA Blackwell Ultra
    "b300": {"vendor": "nvidia", "arch": "blackwell-ultra", "recorded": True},
    "gb300": {"vendor": "nvidia", "arch": "blackwell-ultra", "recorded": False},
    # NVIDIA Blackwell GeForce/workstation. The store's architecture-level
    # sheet uses ``sm120`` as its exact product address.
    "sm120": {"vendor": "nvidia", "arch": "blackwell-geforce", "recorded": True},
    "rtxpro5000": {"vendor": "nvidia", "arch": "blackwell-geforce", "recorded": False},
    "rtx5090": {"vendor": "nvidia", "arch": "blackwell-geforce", "recorded": False},
    "rtx5080": {"vendor": "nvidia", "arch": "blackwell-geforce", "recorded": False},
    # AMD CDNA
    "mi300x": {"vendor": "amd", "arch": "cdna3", "recorded": True},
    "mi300a": {"vendor": "amd", "arch": "cdna3", "recorded": False},
    "mi308x": {"vendor": "amd", "arch": "cdna3", "recorded": True},
    "mi350x": {"vendor": "amd", "arch": "cdna4", "recorded": False},
    "mi355x": {"vendor": "amd", "arch": "cdna4", "recorded": True},
}

PRODUCT_ARCH = {name: row["arch"] for name, row in HARDWARE_IDENTITIES.items()}
RECORDED_PRODUCTS = frozenset(
    name for name, row in HARDWARE_IDENTITIES.items() if row["recorded"]
)


def _compact(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


_CANONICAL = {_compact(name): name for name in HARDWARE_IDENTITIES}

_PREFIXES = ("advancedmicrodevices", "nvidia", "amd", "geforce", "instinct")
_SUFFIXES = ("accelerator", "gpu")


def normalize_product_name(value: str) -> str:
    """Normalize formatting without translating one hardware identity to another."""
    token = _compact(value)
    changed = True
    while token and changed:
        changed = False
        for prefix in _PREFIXES:
            if token.startswith(prefix) and len(token) > len(prefix):
                token = token[len(prefix):]
                changed = True
                break
        for suffix in _SUFFIXES:
            if token.endswith(suffix) and len(token) > len(suffix):
                token = token[:-len(suffix)]
                changed = True
                break
    return _CANONICAL.get(token, token)
