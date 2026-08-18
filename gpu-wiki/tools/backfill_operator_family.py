#!/usr/bin/env python3
"""Backfill retrieval.scope.operator_family on staged trace/session records.

Why this exists: trace mining leaves scope.operator_family as null on most
records even though the operator is recorded elsewhere in the same record. That
null breaks two things downstream:

  * the admission gate routes the record to records/<type>/<vendor>/<arch>/<dsl>/misc/,
    losing the operator grouping the whole directory layout is built on;
  * the admission gate's scope filter then finds no same-scope candidates, so a record
    that should have been compared against the existing records for that
    operator is inserted unexamined.

The operator is recovered from three fallbacks, in order of directness:
  1. payload.problem.operator_id  -- already a slug, use verbatim
  2. retrieval.scope.operators[0] -- an identifier, convert to slug
  3. payload.problem.operator     -- a human-readable name, slugify

The convention being matched is the main store's: operator_family is the slug of
the specific operator (e.g. hyena-fft-size-padding-rfft), NOT the coarse
workload family (ssm-linear-attention), which lives in generality.workload_family.

    python3 backfill_operator_family.py --dry-run   # report, change nothing
    python3 backfill_operator_family.py             # write the fix back
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
KERNEL_WIKI = HERE.parent / "kernel_wiki"
STAGING_DIRS = ("trace_wiki", "session_wiki")


def slugify(text: str) -> str:
    """Same shape the main store uses: lowercase, non-alnum runs become dashes."""
    out = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return out


def recover_family(record: dict) -> tuple[str | None, str]:
    """Return (family, which_source) or (None, reason)."""
    scope = (record.get("retrieval") or {}).get("scope") or {}
    problem = (record.get("payload") or {}).get("problem") or {}

    oid = problem.get("operator_id")
    if oid:
        return oid, "problem.operator_id"

    operators = scope.get("operators") or []
    if operators:
        return slugify(operators[0]), "scope.operators[0]"

    name = problem.get("operator")
    if name:
        return slugify(name), "problem.operator"

    return None, "no operator information anywhere in the record"


def staged_files() -> list[str]:
    out = []
    for sub in STAGING_DIRS:
        root = KERNEL_WIKI / sub
        if not root.is_dir():
            continue
        for path in glob.glob(str(root / "**" / "*.json"), recursive=True):
            if os.path.basename(path) == "index.json":
                continue
            out.append(path)
    return sorted(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would change without writing.")
    args = parser.parse_args()

    files = staged_files()
    if not files:
        print("no staged records found under %s" % ", ".join(STAGING_DIRS))
        return 0

    already = 0
    fixed: list[tuple[str, str, str]] = []
    failed: list[tuple[str, str]] = []
    by_source: collections.Counter = collections.Counter()
    by_family: collections.Counter = collections.Counter()

    for path in files:
        with open(path, encoding="utf-8") as handle:
            record = json.load(handle)
        scope = (record.get("retrieval") or {}).get("scope") or {}

        if scope.get("operator_family"):
            already += 1
            continue

        family, source = recover_family(record)
        if not family:
            failed.append((record.get("id", os.path.basename(path)), source))
            continue

        by_source[source] += 1
        by_family[family] += 1
        fixed.append((record.get("id", ""), family, source))

        if not args.dry_run:
            record["retrieval"]["scope"]["operator_family"] = family
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, indent=1) + "\n")

    verb = "would set" if args.dry_run else "set"
    print("staged records scanned : %d" % len(files))
    print("already had a family   : %d" % already)
    print("%-22s : %d" % (verb, len(fixed)))
    if failed:
        print("could not recover      : %d" % len(failed))

    if by_source:
        print("\nrecovered from:")
        for source, count in by_source.most_common():
            print("  %-24s %d" % (source, count))

    if by_family:
        print("\nresulting operator_family:")
        for family, count in by_family.most_common():
            print("  %-40s %d" % (family, count))

    if failed:
        print("\nunrecoverable records:")
        for rid, reason in failed:
            print("  %s -- %s" % (rid[:70], reason))

    if args.dry_run:
        print("\ndry run: nothing written. Re-run without --dry-run to apply.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
