#!/usr/bin/env python3
"""Rebuild kernel_wiki/records/index.json from the records already on disk.

build_kernel_wiki.py regenerates the whole store from the markdown corpus, which
is the wrong tool once records have been edited or moved in place (audit
demotions, wiki-gate inserts, field backfills). This reads the record files and
re-derives exactly the fields the index carries, reusing the builder's own
derivations so the two can never drift.

    python3 tools/reindex_kernel_wiki.py --check
    python3 tools/reindex_kernel_wiki.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import build_kernel_records as builder          # noqa: E402
import wiki_score                              # noqa: E402

GPU_WIKI = os.path.dirname(HERE)
RECORDS = os.path.join(GPU_WIKI, "kernel_wiki", "records")
INDEX = os.path.join(RECORDS, "index.json")


def record_files() -> list[str]:
    out = []
    for root, _dirs, names in os.walk(RECORDS):
        for name in sorted(names):
            if name.endswith(".json") and name != "index.json":
                out.append(os.path.join(root, name))
    return sorted(out)


def entry_for(path: str, record: dict) -> dict:
    # Index paths are relative to kernel_wiki/, i.e. they start with "records/".
    rel_path = os.path.relpath(path, os.path.dirname(RECORDS)).replace(os.sep, "/")
    measured_on = (record["evidence"]["summary"].get("measured_on") or {})
    return {
        "id": record["id"],
        "type": record["type"],
        "level": record["level"],
        "status": record["status"],
        "episode_key": record["episode_key"],
        "path": rel_path,
        "title": builder.index_title(record),
        "retrieval": record["retrieval"],
        "worth_score": record["worth"]["rank"]["score"],
        "tier": record["worth"]["rank"]["tier"],
        "gain_pct": record["worth"]["gain"].get("pct"),
        "page_path": record["evidence"]["raw"].get("page_path"),
        "search_text": builder.search_text(record),
        # Omitted when empty, matching the seeder: two writers of one index must
        # agree on the shape of "nothing", or every record compares unequal.
        "tracks": sorted(set(filter(None,
                                    measured_on.get("track", "").split("/")))) or None,
        "speedup_x": (record["retrieval"]["signals"]["metrics"]
                      .get("page_best_speedup_x")),
        "refk_status": (record["evidence"]["raw"].get("evidence_extra") or {}
                        ).get("manifest_status"),
    }


def build() -> dict:
    entries = [entry_for(p, json.load(open(p, encoding="utf-8")))
               for p in record_files()]
    entries.sort(key=lambda e: e["id"])
    return {"schema": builder.SCHEMA,
            "generated_by": builder.__name__ + ".py",
            "builder_version": wiki_score.BUILDER_VERSION,
            "count": len(entries), "records": entries}


def describe_drift(current: dict | None, fresh: dict) -> list[str]:
    """What actually differs, not just how many records there are.

    Reporting counts while comparing content produced messages like "has 847
    entries, records/ derives 847", which reads as a contradiction and leaves the
    maintainer no way to act.
    """
    if current is None:
        return ["index.json is missing (%d records derived)" % fresh["count"]]
    notes = []
    for key in ("schema", "generated_by", "builder_version", "count"):
        if current.get(key) != fresh.get(key):
            notes.append("%s: index=%r derived=%r"
                         % (key, current.get(key), fresh.get(key)))
    ci = {e["id"]: e for e in current.get("records", [])}
    fi = {e["id"]: e for e in fresh["records"]}
    for label, ids in (("only in index", set(ci) - set(fi)),
                       ("only in records/", set(fi) - set(ci))):
        if ids:
            notes.append("%d %s (e.g. %s)" % (len(ids), label, sorted(ids)[0]))
    fields: dict[str, int] = {}
    for rid in set(ci) & set(fi):
        for key in set(ci[rid]) | set(fi[rid]):
            if ci[rid].get(key) != fi[rid].get(key):
                fields[key] = fields.get(key, 0) + 1
    for key, n in sorted(fields.items()):
        notes.append("field %r differs in %d record(s)" % (key, n))
    return notes or ["difference outside the compared fields"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="Report whether index.json is current; write nothing.")
    args = ap.parse_args()

    fresh = build()
    current = json.load(open(INDEX, encoding="utf-8")) if os.path.isfile(INDEX) else None

    if args.check:
        if current == fresh:
            print("OK index.json is current (%d records)" % fresh["count"])
            return 0
        print("STALE " + "; ".join(describe_drift(current, fresh)))
        return 1

    if current == fresh:
        print("index.json already current (%d records)" % fresh["count"])
        return 0

    with open(INDEX, "w", encoding="utf-8") as fh:
        json.dump(fresh, fh, ensure_ascii=False, indent=1)
        fh.write("\n")
    print("wrote records/index.json (%d records, was %d)"
          % (fresh["count"], current["count"] if current else 0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
