#!/usr/bin/env python3
"""Fill worth.rank, and write the staging records/index.json.

Both are placeholders the distillation cannot produce. A score has to be
comparable across records, which means one model over the whole store rather than
a judgement per record; an index is a projection of the records rather than
something an agent writes.

The scorer is this repository's own `tools/wiki_score.py`, imported rather than
reimplemented so a score here means the same thing as a score in the live store.
Two degenerate inputs are filtered first, because they read as measured zeros
rather than as absence: a null `step_gain_pct` scores as "measured 0% gain", and a
lone `n_independent_runs: 1` adds a weighted zero.

`worth.rank` is written here and nowhere else, so re-running after every
distillation batch is cheap and is the only way the index stays in step.

Usage: RTM_TRACE=<trace dir> python3 score_records.py [--dry-run]
"""
import argparse
import json
import re
import sys
from collections import Counter

import config as c


def known(d):
    """Drop the values that make the shared scorer misread absence as zero."""
    out = {k: v for k, v in (d or {}).items() if v is not None}
    if out.get("n_independent_runs", 0) <= 1:
        out.pop("n_independent_runs", None)
    return out


def title_of(record):
    """One line for the index, taken from what the record already says.

    `payload.goal` is the record's own first line, so the index cannot drift from
    it -- and there is no separate title field to keep in sync.
    """
    goal = ((record.get("payload") or {}).get("goal") or "").strip()
    goal = re.split(r"(?<=[.!?])\s", goal)[0] if goal else ""
    return goal[:160] or (record.get("id") or "")


def search_text(record):
    """Flat lowercase text for substring search, the same idea the store uses."""
    payload = record.get("payload") or {}
    retrieval = record.get("retrieval") or {}
    parts = [record.get("id"), title_of(record), payload.get("goal"),
             payload.get("change"), payload.get("mechanism"),
             payload.get("attempted"), payload.get("lesson"),
             (payload.get("problem") or {}).get("statement"),
             (payload.get("problem") or {}).get("observed_symptom")]
    parts += retrieval.get("technique_tags") or []
    parts += retrieval.get("triggers") or []
    text = " ".join(str(p) for p in parts if p)
    return re.sub(r"\s+", " ", text).strip().lower()[:2000]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    scoring = c.load_wiki_score()
    paths = sorted(p for p in c.RECORDS.rglob("*.json") if p.name != "index.json")
    if not paths:
        raise SystemExit(
            "no records under %s; distil something first (see "
            "references/distill-brief.md)" % c.RECORDS)

    changed, hist, index = 0, Counter(), []
    for path in paths:
        record = json.loads(path.read_text())
        worth = record.setdefault("worth", {})
        rank = worth.setdefault("rank", {})
        track = worth.get("track") or {}
        gain = worth.get("gain") or {}
        prior = known(track.get("prior"))
        counters = known(track.get("counters"))
        rtype = record.get("type", "strategy")

        # compute() returns a decomposition, not a number. `components` stays out
        # of the record on purpose: the store treats the breakdown as engine-only
        # and strips it from the projection, so persisting it would only park
        # ranking internals next to the agent.
        detail = scoring.compute(prior, counters, rtype)
        score = round(float(detail["value"]), 4)
        tier = scoring.tier(rtype, gain, counters,
                            scoring.reference_rate(prior, rtype))
        if rank.get("score") != score or rank.get("tier") != tier:
            rank["score"] = score
            rank["tier"] = tier
            rank["formula"] = detail["formula"]
            rank["builder_version"] = c.BUILDER_VERSION
            changed += 1
            if not args.dry_run:
                path.write_text(json.dumps(record, ensure_ascii=False, indent=1)
                                + "\n")
        hist[round(score, 1)] += 1
        index.append({
            "id": record.get("id"),
            "type": rtype,
            "level": record.get("level"),
            "status": record.get("status"),
            "episode_key": record.get("episode_key"),
            "path": str(path.relative_to(c.STORE)),
            "title": title_of(record),
            "retrieval": record.get("retrieval"),
            "worth_score": score,
            "tier": tier,
            "gain_pct": gain.get("pct"),
            "search_text": search_text(record),
        })

    if not args.dry_run:
        # The store's `index` gate compares this file against records/ and
        # requires the schema name of the records themselves, not the profile's.
        (c.RECORDS / "index.json").write_text(json.dumps(
            {"schema": c.SCHEMA_NAME, "generated_by": "score_records.py",
             "builder_version": c.BUILDER_VERSION, "count": len(index),
             "records": sorted(index, key=lambda e: e["id"] or "")},
            ensure_ascii=False, indent=1) + "\n")

    print("records    %d  (%d rescored)" % (len(paths), changed))
    print("scores     %s" % dict(sorted(hist.items())))
    tiers = Counter(e["tier"] for e in index)
    print("tiers      %s" % dict(sorted(tiers.items())))
    if not args.dry_run:
        print("-> %s" % (c.RECORDS / "index.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
