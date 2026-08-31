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
    # PPU. The part is CUDA-source-compatible and its runtime reports sm_89, but it is
    # not NVIDIA silicon: giving it a distinct vendor/arch stops an Ada spec sheet from
    # ever being served as this product's numbers. The line ships one public part, so
    # the product name doubles as the architecture address.
    "zwm890p": {"vendor": "ppu", "arch": "zwm890p", "recorded": False},
}

PRODUCT_ARCH = {name: row["arch"] for name, row in HARDWARE_IDENTITIES.items()}
RECORDED_PRODUCTS = frozenset(
    name for name, row in HARDWARE_IDENTITIES.items() if row["recorded"]
)

# Superseded spellings of a part that is already in the table above. These are
# not identities of their own: they fold onto a canonical name so an older
# spelling still resolves. ``zw890`` addressed this chip before its
# architecture token and product name were unified, and deployed campaign
# configs still carry it.
LEGACY_SPELLINGS = {"zw890": "zwm890p"}


def _compact(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


_CANONICAL = {_compact(name): name for name in HARDWARE_IDENTITIES}

_PREFIXES = ("advancedmicrodevices", "nvidia", "amd", "geforce", "instinct", "ppu", "thead")
_SUFFIXES = ("accelerator", "gpu")


def extract_product_names(text: str) -> list[str]:
    """Return explicit public GPU product identities in textual order.

    Product spellings may vary in case and may insert spaces, underscores, or
    hyphens at letter/number boundaries (for example ``B-200`` or
    ``MI_300_X``). This performs identity normalization only; it never maps an
    internal resource alias to a public product name.
    """
    matches = []
    for product in HARDWARE_IDENTITIES:
        chunks = re.findall(r"[a-z]+|[0-9]+", product.casefold())
        pattern = r"(?<![a-z0-9])" + r"[\s_-]*".join(
            re.escape(chunk) for chunk in chunks
        ) + r"(?![a-z0-9])"
        found = re.search(pattern, text, flags=re.IGNORECASE)
        if found:
            matches.append((found.start(), product))
    matches.sort()
    return [product for _, product in matches]


def normalize_product_name(value: str) -> str:
    """Normalize formatting, folding a superseded spelling onto its canonical name.

    This never translates one part into a different part; it only reconciles
    spellings that address the same silicon.
    """
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
    token = LEGACY_SPELLINGS.get(token, token)
    return _CANONICAL.get(token, token)


def target_table_errors(target_table: dict) -> list[str]:
    """Validate a trace-target map against canonical public identities.

    The mining skill may intentionally map a product alias such as ``rtx5090``
    to the architecture-level recorded address ``sm120``. Both the input
    identity and output address must nevertheless agree on vendor and
    architecture, preventing the two code-owned tables from drifting silently.
    """
    errors = []
    for token, triple in sorted(target_table.items()):
        if not isinstance(triple, (tuple, list)) or len(triple) != 3:
            errors.append("%s: expected (vendor, arch, product)" % token)
            continue
        vendor, arch, product = map(str, triple)
        source = normalize_product_name(str(token))
        target = normalize_product_name(product)
        source_row = HARDWARE_IDENTITIES.get(source)
        target_row = HARDWARE_IDENTITIES.get(target)
        if source_row is None:
            errors.append("%s: unknown public input identity" % token)
            continue
        if target_row is None:
            errors.append("%s: unknown output product %s" % (token, product))
            continue
        expected = (vendor, arch)
        if (source_row["vendor"], source_row["arch"]) != expected:
            errors.append(
                "%s: input identity is %s/%s, table says %s/%s"
                % (token, source_row["vendor"], source_row["arch"], vendor, arch)
            )
        if (target_row["vendor"], target_row["arch"]) != expected:
            errors.append(
                "%s: output identity %s is %s/%s, table says %s/%s"
                % (token, target, target_row["vendor"], target_row["arch"],
                   vendor, arch)
            )
    return errors
