#!/usr/bin/env python3
"""Fill worth.rank.score, and write records/index.json.

Both are placeholders the distillation cannot produce: a score has to be
comparable across records, which means one model over the whole set, and an index
is a projection of the records rather than a thing an agent writes.

The scorer is `score.py` in this skill, which reproduces the shared ranking
model's published curve so a score here stays comparable with the committed store's
without depending on it. Two degenerate inputs are filtered first, because leaving
them in was measured once: a `step_gain_pct` of null reads as a measured zero and a
lone `n_independent_runs: 1` reads as a weighted zero, and between them they
floored most of a store at one identical score, which destroys the ranking the
score exists to provide.

Usage: STM_SET=<name> python3 score_records.py [--dry-run]
"""
import argparse
import json
import sys
from collections import Counter

import config as c
import score as scoring


def known(d):
    """Drop the values that make the shared scorer misread absence as zero."""
    out = {k: v for k, v in (d or {}).items() if v is not None}
    if out.get("n_independent_runs", 0) <= 1:
        out.pop("n_independent_runs", None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    set_name, _cfg, _root = c.require_set()
    records_dir = c.records(set_name)
    paths = sorted(p for p in records_dir.rglob("*.json") if p.name != "index.json")
    if not paths:
        raise SystemExit("no records under %s" % records_dir)

    changed, hist, index = 0, Counter(), []
    for path in paths:
        rec = json.loads(path.read_text())
        worth = rec.setdefault("worth", {})
        rank = worth.setdefault("rank", {})
        track = worth.get("track") or {}
        gain = worth.get("gain") or {}
        # compute() returns a decomposition, not a number. `components` stays out
        # of the record on purpose: the main store treats the breakdown as
        # engine-side only, and raw-isolation strips it from the projection, so
        # persisting it would only put ranking internals near the agent.
        detail = scoring.compute(known(track.get("prior")),
                            known(track.get("counters")),
                            rec.get("type", "strategy"))
        score = round(float(detail["value"] if isinstance(detail, dict)
                            else detail), 4)
        formula = detail.get("formula") if isinstance(detail, dict) else None
        tier = scoring.tier_for(score, rec.get("type"), gain.get("basis"))
        if rank.get("score") != score or rank.get("tier") != tier:
            rank["score"] = score
            rank["tier"] = tier
            if formula:
                rank["formula"] = formula
            rank["builder_version"] = "session-trace-0.1"
            changed += 1
            if not args.dry_run:
                path.write_text(json.dumps(rec, ensure_ascii=False, indent=1)
                                + "\n")
        hist[round(score, 1)] += 1
        index.append({"id": rec.get("id"), "retrieval": rec.get("retrieval")})

    if not args.dry_run:
        (records_dir / "index.json").write_text(json.dumps(
            {"schema": c.DERIVED_NAME, "count": len(index),
             "records": sorted(index, key=lambda e: e["id"] or "")},
            ensure_ascii=False, indent=1) + "\n")

    print("records    %d  (%d rescored)" % (len(paths), changed))
    print("scores     %s" % dict(sorted(hist.items())))
    if not args.dry_run:
        print("-> %s" % (records_dir / "index.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
