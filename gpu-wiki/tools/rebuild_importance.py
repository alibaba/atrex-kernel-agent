#!/usr/bin/env python3
"""Fold feedback/events.jsonl into worth.track.counters and rescore every record.

The event log is the truth; the counters stored in each record are a cache, so
this is safe to re-run at any time and always converges to the same state for a
given log. It rewrites worth.rank (score and tier) and nothing else.
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

import wiki_score                                                # noqa: E402

EVENTS = os.path.join(OUT_ROOT, "feedback", "events.jsonl")
COUNTER_NAMES = ("query_count", "served_count", "applied_count",
                 "verified_effective", "verified_ineffective", "fallback_served")


def fold_events() -> dict[str, dict]:
    """record_id -> counters, from the append-only log.

    snapshot  last event per (record, source) wins, so a re-run of a source
              replaces its contribution instead of doubling it.
    increment adds once per `key`, so replaying a log is a no-op.
    """
    if not os.path.exists(EVENTS):
        return {}
    snapshots: dict[tuple[str, str], dict] = {}
    increments: dict[str, dict] = {}
    timestamps: dict[str, list[float]] = defaultdict(list)

    with open(EVENTS) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except ValueError:
                continue
            rid = event.get("record_id")
            if not rid:
                continue
            ts = event.get("ts")
            if ts:
                timestamps[rid].append(float(ts))
            if event.get("kind") == "snapshot":
                snapshots[(rid, event.get("source", ""))] = event
            else:
                key = event.get("key") or json.dumps(event, sort_keys=True)
                increments[key] = event

    folded: dict[str, dict] = defaultdict(lambda: {name: 0 for name in COUNTER_NAMES})
    for event in list(snapshots.values()) + list(increments.values()):
        target = folded[event["record_id"]]
        for name, value in (event.get("counts") or {}).items():
            if name in COUNTER_NAMES:
                target[name] += int(value)
    for rid, stamps in timestamps.items():
        if rid in folded:
            folded[rid]["first_seen_ts"] = min(stamps)
            folded[rid]["last_used_ts"] = max(stamps)
    return dict(folded)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=OUT_ROOT)
    parser.add_argument("--now", type=float, default=None,
                        help="Override the clock, for reproducible rescoring.")
    args = parser.parse_args()

    now = args.now if args.now is not None else time.time()
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(now))
    folded = fold_events()

    index_path = os.path.join(args.out, "records", "index.json")
    index = json.load(open(index_path))
    touched = changed = 0

    for entry in index["records"]:
        path = os.path.join(args.out, entry["path"])
        record = json.load(open(path))
        worth = record["worth"]
        # A seeded record has never been served, so it carries no engine ledger
        # yet. Bootstrap an empty one rather than refusing to rank it.
        track = worth.setdefault("track", {"counters": wiki_score.empty_counters(),
                                           "prior": {}})
        track.setdefault("prior", {})
        counters = wiki_score.empty_counters()
        counters.update(folded.get(record["id"], {}))
        rank = wiki_score.rank(track["prior"], counters, record["type"],
                               worth.get("gain") or {}, now_ts=now)
        rank["computed_at"] = stamp
        before = worth["rank"]["score"]
        worth["rank"] = rank
        worth["track"]["counters"] = counters
        with open(path, "w") as fh:
            json.dump(record, fh, ensure_ascii=False, indent=1)
            fh.write("\n")
        entry["worth_score"] = rank["score"]
        entry["tier"] = rank["tier"]
        touched += 1
        if abs(before - rank["score"]) > 1e-9:
            changed += 1

    with open(index_path, "w") as fh:
        json.dump(index, fh, ensure_ascii=False, indent=1)
        fh.write("\n")

    with_feedback = len(folded)
    print("records rescored: %d (%d changed)" % (touched, changed))
    print("records with feedback: %d (%.1f%%)"
          % (with_feedback, 100.0 * with_feedback / max(1, touched)))
    scores = sorted(entry["worth_score"] for entry in index["records"])
    if scores:
        def pick(fraction):
            return scores[min(len(scores) - 1, int(len(scores) * fraction))]
        print("score distribution: min=%.3f p25=%.3f median=%.3f p75=%.3f max=%.3f"
              % (scores[0], pick(0.25), pick(0.5), pick(0.75), scores[-1]))
    tiers = Counter(entry["tier"] for entry in index["records"])
    print("tiers: %s" % ", ".join("%s=%d" % (name, tiers[name])
                                  for name in wiki_score.TIERS if tiers[name]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
