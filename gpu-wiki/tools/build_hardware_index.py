#!/usr/bin/env python3
"""Build records/index.json for the hardware store.

The index is the lookup table: it carries only what is needed to ADDRESS a
record (identity plus availability), never the facts themselves. A lookup
resolves an address here, then reads the one record file it needs -- so adding
records does not make a lookup slower.

    python3 tools/build_hardware_index.py
    python3 tools/build_hardware_index.py --check    # exit 1 if the committed index is stale
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GPU_WIKI = os.path.dirname(HERE)
STORE = os.path.join(GPU_WIKI, "hardware_wiki")
RECORDS = os.path.join(STORE, "records")
INDEX = os.path.join(RECORDS, "index.json")
SCHEMA_VERSION = "hw-1.0"
BUILDER_VERSION = "1"


def record_paths() -> list[str]:
    out = []
    for dirpath, _dirs, files in os.walk(RECORDS):
        for name in sorted(files):
            if name.endswith(".json") and name != "index.json":
                out.append(os.path.join(dirpath, name))
    return sorted(out)


def entry_for(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        record = json.load(handle)
    identity = record["identity"]
    availability = identity.get("availability") or {}
    return {
        "id": record["id"],
        "type": record["type"],
        "status": record["status"],
        "path": os.path.relpath(path, STORE),
        "vendor": identity["vendor"],
        "arch": identity["arch"],
        "product": identity.get("product"),
        "compute_capability": identity.get("compute_capability"),
        "sm_arch": identity.get("sm_arch"),
        "mnemonic": identity.get("mnemonic"),
        "feature": identity.get("feature"),
        "available_sm_arch": availability.get("sm_arch") or [],
        "available_products": availability.get("products") or [],
        "evidence_class": record["provenance"]["evidence_class"],
    }


def build() -> dict:
    entries = [entry_for(p) for p in record_paths()]
    entries.sort(key=lambda e: e["id"])
    return {"schema": SCHEMA_VERSION, "generated_by": "build_index.py",
            "builder_version": BUILDER_VERSION, "count": len(entries),
            "records": entries}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    index = build()
    rendered = json.dumps(index, ensure_ascii=False, indent=1) + "\n"

    if args.check:
        current = ""
        if os.path.exists(INDEX):
            with open(INDEX, encoding="utf-8") as handle:
                current = handle.read()
        if current != rendered:
            print("STALE %s -- rerun: python3 tools/build_hardware_index.py" % INDEX,
                  file=sys.stderr)
            return 1
        print("OK index.json is current (%d records)" % index["count"])
        return 0

    with open(INDEX, "w", encoding="utf-8") as handle:
        handle.write(rendered)
    print("wrote records/index.json (%d records)" % index["count"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
