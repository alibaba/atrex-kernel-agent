#!/usr/bin/env python3
# Copyright 2026 Alibaba Group.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Score retrieval translation against real requests captured from real runs.

Ten optimizing agents were asked to record, at every moment they wanted knowledge,
both the prose they would say to a colleague and the flag query they would actually
have typed -- the latter written down BEFORE they saw whether it worked. That pairing
is what makes this measurable rather than a matter of taste:

  * the flag queries are the BASELINE. Running them against the store tells us how
    the interface performs when the caller has to translate for itself, and it is
    free and deterministic to compute;
  * the natural-language requests are the INPUT to the bridge. Running the bridge on
    them and scoring the same way says whether the bridge is actually better, on the
    same requests, against the same store.

The properties scored are the ones that decide whether a caller was helped or misled,
not whether the flags look similar to a human's:

  * did the query run at all, or die on a vocabulary token;
  * did it return records, or zero;
  * was the answer real matches, or a labelled random sample;
  * was the architecture actually pinned (an unscoped query that "succeeds" is worse
    than a failure, because another chip's measurements are not evidence here).

Usage:
    python3 tools/test_nl_translation.py --baseline            # free, deterministic
    python3 tools/test_nl_translation.py --baseline --verbose
    python3 tools/test_nl_translation.py --bridge 5 --seed 1   # costs agent calls
    python3 -m unittest test_nl_translation                    # static assertions only
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import query_nl                                                    # noqa: E402

SAMPLE_ROOT = Path("/root/wiki_query_traces/nl_samples")
DEFAULT_STORE = query_nl.SIBLING_INTERNAL if (
    query_nl.SIBLING_INTERNAL / "tools" / "query_wiki.py").is_file() \
    else query_nl.OWN_STORE_ROOT

# Words a request uses when the bottleneck is still a guess. A symptom axis spent on
# a guess is the expensive mistake, so scoring has to be able to tell the two apart.
HEDGES = ("i suspect", "i think", "probably", "must be", "i haven't profiled",
          "not profiled", "still a guess", "hypothesis", "i assume", "my guess")


def load_samples(root: Path = SAMPLE_ROOT) -> list[dict]:
    out = []
    for path in sorted(root.glob("task*/samples.json")):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(doc, dict):
            doc = doc.get("samples") or doc.get("moments") or []
        for entry in doc:
            if not isinstance(entry, dict) or not entry.get("nl_request"):
                continue
            entry["_task"] = path.parent.name
            out.append(entry)
    return out


# Flags that address the FACTS store. A request for a part's peak is not a search,
# and scoring it against the experience tool would report a tool mismatch as if the
# caller had used a bad token.
HARDWARE_FLAGS = ("--product", "--instruction", "--feature", "--capability",
                  "--vs", "--field")


def run_query(store_root: Path, flags: list[str]) -> dict:
    """Run one flag query and reduce it to the few things that decide its worth."""
    hardware = any(f in HARDWARE_FLAGS for f in flags)
    tool = "query_hardware.py" if hardware else "query_wiki.py"
    argv = [sys.executable, str(store_root / "tools" / tool)] + \
        [f for f in flags if f not in query_nl.SURVEY_ONLY_FLAGS]
    if not hardware:
        argv += ["--emit-json", "--limit", "8"]
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=180)

    if hardware:
        # exit 4 is the store declining to guess and handing back a procedure. That
        # is a correct answer to "what is this part's peak", not a failed query.
        if proc.returncode == 4:
            return {"ran": True, "hardware": True, "kind": "not-recorded",
                    "served": 0, "arch": None}
        if proc.returncode != 0:
            reason = (proc.stderr or "").strip().splitlines()
            return {"ran": False, "error": reason[0][:120] if reason else "unknown"}
        return {"ran": True, "hardware": True, "kind": "fact", "served": 1,
                "arch": None}

    if proc.returncode != 0:
        reason = (proc.stderr or "").strip().splitlines()
        return {"ran": False, "error": reason[0][:120] if reason else "unknown"}
    try:
        doc = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {"ran": False, "error": "unparseable output"}
    res, scope = doc.get("result", {}), doc.get("query", {})
    return {"ran": True, "kind": res.get("kind"), "served": res.get("served", 0),
            "pool": res.get("scoped_pool"), "arch": scope.get("arch"),
            "over_constrained": bool(res.get("empty_because"))}


def score(entry: dict, outcome: dict) -> dict:
    """Was the caller helped, misled, or stonewalled?"""
    hedged = any(h in (entry.get("nl_request") or "").lower() for h in HEDGES)
    if not outcome.get("ran"):
        verdict = "died-on-vocabulary"
    elif outcome.get("hardware"):
        # Both are correct outcomes for a facts lookup: the number, or the
        # documented way to obtain it. Neither is a retrieval failure.
        verdict = "fact" if outcome["kind"] == "fact" else "fact-not-recorded"
    elif outcome["kind"] == "fallback":
        verdict = "misleading-fallback"
    elif not outcome.get("served"):
        verdict = "empty"
    elif not outcome.get("arch"):
        verdict = "unscoped-hit"          # returned something, but not about this chip
    else:
        verdict = "scoped-hit"
    return {"task": entry["_task"], "phase": entry.get("phase", "?"),
            "verdict": verdict, "served": outcome.get("served", 0),
            "hedged_request": hedged, "detail": outcome}


def summarize(rows: list[dict], label: str) -> dict:
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    total = len(rows) or 1
    good = (counts.get("scoped-hit", 0) + counts.get("fact", 0)
            + counts.get("fact-not-recorded", 0))
    print("\n=== %s: %d moments ===" % (label, len(rows)))
    for verdict in ("scoped-hit", "fact", "fact-not-recorded", "unscoped-hit",
                    "empty", "misleading-fallback", "died-on-vocabulary",
                    "bridge-failed"):
        n = counts.get(verdict, 0)
        if n:
            print("  %-20s %3d  (%4.1f%%)" % (verdict, n, 100.0 * n / total))
    print("  %-20s %3d  (%4.1f%%)" % ("USABLE", good, 100.0 * good / total))
    return counts


def baseline(store_root: Path, verbose: bool) -> list[dict]:
    """How the flag interface performs when the caller translates for itself."""
    rows = []
    for entry in load_samples():
        flags = entry.get("structured_query_i_would_have_written") or []
        if isinstance(flags, str):
            flags = flags.split()
        flags = [str(f) for f in flags if str(f).strip()]
        if not flags:
            continue
        row = score(entry, run_query(store_root, flags))
        row["flags"] = flags
        rows.append(row)
        if verbose:
            print("  [%-16s %-26s] %-20s %s" % (
                row["task"][:16], row["phase"][:26], row["verdict"],
                " ".join(flags)[:80]))
    return rows


def bridge(store_root: Path, n: int, seed: int, agent_cli: str,
           verbose: bool, skip: int = 0) -> list[dict]:
    """How the bridge performs on the SAME requests. Costs one agent call each.

    ``skip`` exists so a failure found in a long run can be re-examined without
    paying for the moments that already passed.
    """
    samples = load_samples()
    random.Random(seed).shuffle(samples)
    rows = []
    for entry in samples[skip:skip + n]:
        argv = [sys.executable, str(HERE / "query_nl.py"), entry["nl_request"],
                "--store-root", str(store_root), "--agent-cli", agent_cli]
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=1800)
        if proc.returncode != 0:
            error = (proc.stderr or proc.stdout or "").strip()[-400:]
            rows.append({"task": entry["_task"], "phase": entry.get("phase", "?"),
                         "verdict": "bridge-failed", "served": 0,
                         "hedged_request": False, "detail": {"error": error}})
            if verbose:
                print("  [%-16s %-26s] %-20s exit=%d %s" % (
                    entry["_task"][:16], entry.get("phase", "?")[:26],
                    "bridge-failed", proc.returncode, error.replace("\n", " ")[:160]))
            continue
        answer = json.loads(proc.stdout)
        records = answer.get("records") or {}
        served = len(records)
        kernel = [r for r in records.values() if r.get("source") == "kernel_wiki"]
        arches = {r.get("applies_to", {}).get("arch") for r in kernel}
        outcome = {"ran": True, "served": served,
                   "kind": "matches" if served else "empty",
                   "arch": next((a for a in arches if a), None),
                   "queries": None}
        row = score(entry, outcome)
        row["notes"] = len(answer.get("notes") or [])
        rows.append(row)
        if verbose:
            print("  [%-16s %-26s] %-20s served=%d" % (
                row["task"][:16], row["phase"][:26], row["verdict"],
                served))
    return rows


# ------------------------------------------------------------------ static tests

class SampleCorpusTests(unittest.TestCase):
    """Guard the corpus itself, so a broken sample file is not read as a result."""

    @classmethod
    def setUpClass(cls):
        cls.samples = load_samples()

    def test_the_corpus_exists(self):
        if not SAMPLE_ROOT.is_dir():
            self.skipTest("no sample corpus on this machine")
        self.assertGreaterEqual(len(self.samples), 20)

    def test_every_moment_pairs_prose_with_a_flag_attempt(self):
        if not self.samples:
            self.skipTest("no sample corpus on this machine")
        paired = [s for s in self.samples
                  if s.get("structured_query_i_would_have_written")]
        self.assertGreater(len(paired) / len(self.samples), 0.8)

    def test_prose_was_not_pre_translated_into_flag_speak(self):
        """A request that is already flags is not a natural-language sample."""
        if not self.samples:
            self.skipTest("no sample corpus on this machine")
        for s in self.samples:
            text = s["nl_request"]
            self.assertGreater(len(text), 200, s["_task"])
            self.assertLess(text.count("--") * 40, len(text), s["_task"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--bridge", type=int, default=0, metavar="N")
    ap.add_argument("--store-root", default=str(DEFAULT_STORE))
    ap.add_argument("--agent-cli", default="qodercli")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip", type=int, default=0,
                    help="Skip this many shuffled moments before --bridge N.")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out", default=None,
                    help="Persist the scored rows as JSON. Do use it: the first run "
                         "of this harness lost the error text of two failures because "
                         "nothing was written down, and they never recurred.")
    args = ap.parse_args()
    if not args.baseline and not args.bridge:
        ap.error("give --baseline and/or --bridge N")
    root = Path(args.store_root).resolve()
    print("store: %s" % root)
    collected: dict[str, list] = {}
    if args.baseline:
        collected["baseline"] = baseline(root, args.verbose)
        summarize(collected["baseline"], "flag baseline (caller translates)")
    if args.bridge:
        collected["bridge"] = bridge(root, args.bridge, args.seed, args.agent_cli,
                                     args.verbose, args.skip)
        summarize(collected["bridge"], "natural-language bridge")
    if args.out:
        Path(args.out).write_text(json.dumps(collected, ensure_ascii=False, indent=1),
                                  encoding="utf-8")
        print("rows written to %s" % args.out)
