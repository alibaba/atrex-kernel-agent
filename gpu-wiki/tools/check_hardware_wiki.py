#!/usr/bin/env python3
"""Validation gates for the hardware store.

Each gate catches a failure that is otherwise SILENT -- nothing crashes, the
lookup just starts returning something subtly wrong:

  schema        a record drifts from hw-1.0 and a consumer reads a missing key
  ids           an id stops matching its path, so a citation resolves nowhere
  index         index.json disagrees with records/, so a record is unreachable
  provenance    a number carries no evidence class, and nobody can tell whether
                it was published, divided out of a system total, or guessed
  no-advice     a recommendation leaks in ("prefer X", "usually faster"), which
                is measured experience and belongs in the experience wiki
  fabrication   a spec-sheet claims a number the source does not publish instead
                of leaving it null and saying how to obtain it

    python3 tools/check_hardware_wiki.py
"""
from __future__ import annotations

import json
import os
import re
import sys

import jsonschema

HERE = os.path.dirname(os.path.abspath(__file__))
GPU_WIKI = os.path.dirname(HERE)
STORE = os.path.join(GPU_WIKI, "hardware_wiki")
RECORDS = os.path.join(STORE, "records")
SCHEMA_PATH = os.path.join(GPU_WIKI, "schema", "hardware", "schema.json")
SCHEMA_VERSION = "hw-1.0"

# Phrases that turn a fact into a recommendation. A hardware record states what
# the silicon is; which legal choice is faster is measured, ranked knowledge.
ADVICE = re.compile(
    r"\b(?:usually faster|is faster than|we recommend|you should use|"
    r"best choice|outperforms|prefer(?:red)? (?:to use|using)|"
    r"success rate|retained in \d+%)\b", re.I)


def load_records() -> list[tuple[str, dict]]:
    out = []
    for dirpath, _dirs, files in os.walk(RECORDS):
        for name in sorted(files):
            if not name.endswith(".json") or name == "index.json":
                continue
            path = os.path.join(dirpath, name)
            with open(path, encoding="utf-8") as handle:
                out.append((path, json.load(handle)))
    return sorted(out, key=lambda pair: pair[1]["id"])


def gate_schema(records) -> list[str]:
    with open(SCHEMA_PATH, encoding="utf-8") as handle:
        schema = json.load(handle)
    validator = jsonschema.Draft202012Validator(schema)
    errors = []
    for path, record in records:
        for err in validator.iter_errors(record):
            errors.append("%s: %s at %s" % (
                os.path.relpath(path, STORE), err.message,
                "/".join(str(p) for p in err.absolute_path) or "<root>"))
    return errors


def gate_ids(records) -> list[str]:
    errors = []
    seen = {}
    for path, record in records:
        rid, rtype = record["id"], record["type"]
        identity = record["identity"]
        expected = "%s.%s.%s." % (identity["vendor"], identity["arch"], rtype)
        if not rid.startswith(expected):
            errors.append("%s: id %r does not start with %r" % (rid, rid, expected))
        slug = rid[len(expected):]
        want = os.path.join(RECORDS, rtype, identity["vendor"], identity["arch"],
                            slug + ".json")
        if os.path.abspath(path) != os.path.abspath(want):
            errors.append("%s: path %s does not match id (want %s)" % (
                rid, os.path.relpath(path, STORE), os.path.relpath(want, STORE)))
        if rid in seen:
            errors.append("%s: duplicate id" % rid)
        seen[rid] = path
    return errors


def gate_index(records) -> list[str]:
    path = os.path.join(RECORDS, "index.json")
    if not os.path.isfile(path):
        return ["index.json is missing; run tools/build_hardware_index.py"]
    with open(path, encoding="utf-8") as handle:
        index = json.load(handle)
    if index.get("schema") != SCHEMA_VERSION:
        return ["index.json schema is %r, want %s" % (index.get("schema"),
                                                      SCHEMA_VERSION)]
    indexed = {e["id"] for e in index["records"]}
    actual = {record["id"] for _p, record in records}
    errors = []
    for missing in sorted(actual - indexed):
        errors.append("%s: on disk but not in index.json" % missing)
    for extra in sorted(indexed - actual):
        errors.append("%s: in index.json but not on disk" % extra)
    for entry in index["records"]:
        full = os.path.join(STORE, entry["path"])
        if not os.path.isfile(full):
            errors.append("%s: index path does not exist (%s)" % (entry["id"],
                                                                  entry["path"]))
    return errors


def gate_provenance(records) -> list[str]:
    errors = []
    for _path, record in records:
        rid = record["id"]
        prov = record["provenance"]
        if prov["evidence_class"] == "derived-from-system-total" and not prov.get("derivation"):
            errors.append("%s: derived-from-system-total without a derivation" % rid)
        if record["type"] != "spec-sheet":
            continue
        facts = record["facts"]
        for entry in facts["peak_compute"]:
            if entry.get("provenance") == "derived-from-system-total" and not entry.get("derivation"):
                errors.append("%s: peak_compute[%s] is derived but states no divisor"
                              % (rid, entry["dtype"]))
        for group in ("memory", "compute_units"):
            block = facts.get(group) or {}
            for field in (block.get("provenance_overrides") or {}):
                if field not in block:
                    errors.append("%s: %s.provenance_overrides names unknown field %r"
                                  % (rid, group, field))
    return errors


def gate_no_advice(records) -> list[str]:
    errors = []
    for _path, record in records:
        blob = json.dumps(record["facts"], ensure_ascii=False)
        for hit in set(ADVICE.findall(blob)):
            errors.append("%s: facts read as a recommendation (%r) -- measured "
                          "guidance belongs in the experience wiki" % (record["id"], hit))
    return errors


def gate_fabrication(records) -> list[str]:
    """A null field must be explained, and an explained field must not also
    carry a value: otherwise a reader cannot tell which one to trust."""
    errors = []
    for _path, record in records:
        if record["type"] != "spec-sheet":
            continue
        rid = record["id"]
        facts = record["facts"]
        unavailable = facts.get("unavailable") or {}
        for group in ("memory", "compute_units"):
            block = facts.get(group) or {}
            for field, value in block.items():
                if field in ("provenance", "provenance_overrides", "note"):
                    continue
                if value is None and field not in unavailable:
                    errors.append("%s: %s.%s is null but 'unavailable' does not say "
                                  "how to obtain it" % (rid, group, field))
                if value is not None and field in unavailable:
                    errors.append("%s: %s.%s has both a value and an 'unavailable' "
                                  "note" % (rid, group, field))
    return errors


GATES = (("schema", gate_schema), ("ids", gate_ids), ("index", gate_index),
         ("provenance", gate_provenance), ("no-advice", gate_no_advice),
         ("fabrication", gate_fabrication))

BLURB = {"schema": "records match hw-1.0",
         "ids": "ids unique, deterministic, matching paths",
         "index": "index.json agrees with records/",
         "provenance": "every number states where it came from",
         "no-advice": "facts stay facts, no ranked recommendations",
         "fabrication": "missing numbers are declared, never invented"}


def main() -> int:
    records = load_records()
    print("records: %d" % len(records))
    counts: dict = {}
    for _p, record in records:
        counts[record["type"]] = counts.get(record["type"], 0) + 1
    print("  by type: %s\n" % ", ".join("%s=%d" % kv for kv in sorted(counts.items())))

    failed = 0
    for name, gate in GATES:
        errors = gate(records)
        if errors:
            failed += 1
            print("FAIL %-12s %s" % (name, BLURB[name]))
            for err in errors[:12]:
                print("       - %s" % err)
            if len(errors) > 12:
                print("       ... and %d more" % (len(errors) - 12))
        else:
            print("PASS %-12s %s" % (name, BLURB[name]))
    print("\n%d of %d gates failed" % (failed, len(GATES)))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
