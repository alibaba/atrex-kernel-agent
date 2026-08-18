#!/usr/bin/env python3
"""Feed usage and outcome evidence into the JSON store's worth.track layer.

Everything lands in feedback/events.jsonl, an append-only log. Records are never
edited here; rebuild_importance.py folds the log into worth.track.counters. That
split exists so counters stay recomputable: the log is the truth, the counters
in the records are a cache.

Two event shapes, both idempotent:

  snapshot   replaces the counters this source contributes for one record.
             Re-running the same source overwrites instead of accumulating.
  increment  adds once, deduplicated on `key`. Re-ingesting the same retrieval
             log line is a no-op.

Three sources:

  --corpus-bootstrap   Offline cold start. For each (operator, technique) the
                       corpus already shows how often that lever was kept and
                       how often it was reverted ON THAT OPERATOR. That is a
                       genuine effective/ineffective count and is not the same
                       thing as the global per-technique success rate, which
                       lives in worth.track.prior.
  --retrieval-logs     Replays wiki_history/retrieved.jsonl written by the
                       running agents. These record which markdown page was
                       returned, so attribution is page-level: every record
                       distilled from a returned page is credited. Per-record
                       resolution only starts once the json backend serves ids.
  --report             The online path, for the optimization loop to call when a
                       version that cited a record is kept or reverted.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
GPU_WIKI = os.path.dirname(HERE)
# worth.track lives on kernel-experience records, so the feedback log and
# the records it folds into both belong to that store.
OUT_ROOT = os.path.join(GPU_WIKI, "kernel_wiki")

sys.path.insert(0, HERE)


EVENTS = os.path.join(OUT_ROOT, "feedback", "events.jsonl")

COUNTER_NAMES = ("query_count", "served_count", "applied_count",
                 "verified_effective", "verified_ineffective", "fallback_served")


def load_index() -> dict:
    path = os.path.join(OUT_ROOT, "records", "index.json")
    if not os.path.exists(path):
        raise SystemExit("index.json missing; run build_kernel_records.py first")
    return json.load(open(path))


def append(events: list[dict]) -> None:
    os.makedirs(os.path.dirname(EVENTS), exist_ok=True)
    with open(EVENTS, "a") as fh:
        for event in events:
            fh.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
    print("appended %d event(s) to %s" % (len(events), EVENTS))


# ------------------------------------------------------------ corpus bootstrap

def report(record_id: str, outcome: str, note: str, now: float) -> list[dict]:
    index = load_index()
    if record_id not in {entry["id"] for entry in index["records"]}:
        raise SystemExit("unknown record id: %s" % record_id)
    counts = {"applied": {"applied_count": 1},
              "effective": {"applied_count": 1, "verified_effective": 1},
              "ineffective": {"applied_count": 1, "verified_ineffective": 1},
              "served": {"served_count": 1}}[outcome]
    return [{
        "ts": now, "record_id": record_id, "kind": "increment",
        "source": "agent-report", "key": "report:%s:%s:%f" % (record_id, outcome, now),
        "counts": counts, "note": note,
    }]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", metavar="RECORD_ID",
                        help="Record id the outcome is about.")
    parser.add_argument("--outcome",
                        choices=("served", "applied", "effective", "ineffective"),
                        help="What happened when the record was used.")
    parser.add_argument("--note", default="",
                        help="Free-text context kept with the event.")
    args = parser.parse_args()

    now = time.time()
    events: list[dict] = []
    if args.report:
        if not args.outcome:
            parser.error("--report needs --outcome")
        events += report(args.report, args.outcome, args.note, now)
    if not events:
        parser.error("nothing to do; pass --report <id> --outcome <outcome>")
    append(events)
    print("now run: python3 rebuild_importance.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
