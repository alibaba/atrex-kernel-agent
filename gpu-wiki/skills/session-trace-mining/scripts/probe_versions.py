#!/usr/bin/env python3
"""Phase-0 probe: is a version mechanically recoverable, and is it pairable?

Throwaway by design. It answers one question before any schema exists: can we
cut one record per version, or does a corpus need a different unit? The bars are
written down here so the answer is a pass/fail and not an impression.

Usage: python3 probe_versions.py <transcript> [<transcript> ...]
"""
import json
import re
import sys
from collections import OrderedDict

import transcripts as T

# `v233: committed (+0.200%) ...` in a git log echo, or a `vN.json` filename, or
# the `### [v1/S1]` heading of a hand-kept iteration log.
VER_TOKEN = re.compile(r"\bv(\d+)\b")
GITLOG_RE = re.compile(
    r"^\s*([0-9a-f]{7,40})\s+v(\d+)\s*:\s*(.+)$", re.M)
VN_JSON_RE = re.compile(r"memory/v(\d+)\.json")
NOTES_HEAD_RE = re.compile(r"^###\s*\[v(\d+)(?:/[^\]]*)?\]\s*(.*)$", re.M)

# A metric is any of the shapes the corpus actually prints. Kept deliberately
# loose here -- the probe measures availability, not the final parser.
METRIC_RES = (
    re.compile(r"(\d+(?:\.\d+)?)\s*(us|µs|μs|ms|ns)\b", re.I),
    re.compile(r"speedup\s*=?\s*(\d+(?:\.\d+)?)\s*x", re.I),
    re.compile(r"([-+]\d+(?:\.\d+)?)\s*%"),
    re.compile(r"(\d+(?:\.\d+)?)\s*(TFLOP/?s|TFLOPS)", re.I),
    re.compile(r'"latency_us(?:_geomean)?"\s*:\s*(\d+(?:\.\d+)?)'),
)


def has_metric(text):
    return any(r.search(text) for r in METRIC_RES)


def regions(events):
    """Split into monotone regions: line order is history only inside one.

    A compaction or a rollback means an earlier line can describe a later state,
    so a pair that straddles one is not evidence of anything.
    """
    out, cur = [], []
    for e in events:
        if e.kind == "discontinuity":
            if cur:
                out.append(cur)
            cur = []
            continue
        cur.append(e)
    if cur:
        out.append(cur)
    return out


def probe(path):
    fmt, events = T.parse(path)
    if not fmt:
        print("  UNKNOWN FORMAT")
        return {}

    kinds = {}
    for e in events:
        kinds[e.kind] = kinds.get(e.kind, 0) + 1
    tiers = {}
    for e in events:
        if e.tier:
            tiers[e.tier] = tiers.get(e.tier, 0) + 1

    # ---- version sightings, by channel
    gitlog = OrderedDict()      # vN -> (sha, subject, line_no)
    vn_json = OrderedDict()     # vN -> line_no of a full document
    notes = OrderedDict()       # vN -> heading text
    for e in events:
        if e.kind != "tool-output":
            continue
        for sha, n, subj in GITLOG_RE.findall(e.text):
            gitlog.setdefault("v" + n, (sha, subj.strip()[:90], e.line_no))
        for n, head in NOTES_HEAD_RE.findall(e.text):
            notes.setdefault("v" + n, head.strip()[:90])
    for e in events:
        if e.kind not in ("tool-output", "edit", "tool-call"):
            continue
        blob = e.text
        if e.kind == "edit":
            blob = " ".join(e.meta.get("files") or {})
        for n in VN_JSON_RE.findall(blob):
            # A full document, not just a mention: it must contain the
            # performance block.
            if '"performance"' in e.text or e.kind != "tool-output":
                vn_json.setdefault("v" + n, e.line_no)

    versions = sorted(set(gitlog) | set(vn_json) | set(notes),
                      key=lambda v: int(v[1:]))

    # ---- pairability: does each version have an edit and a metric in one region
    edits = [e for e in events if e.kind == "edit" and e.meta.get("code_files")]
    metric_events = [e for e in events
                     if e.kind == "tool-output" and e.tier in (T.TIER_T1, T.TIER_T2)
                     and has_metric(e.text)]
    regs = regions(events)
    reg_of = {}
    for ri, reg in enumerate(regs):
        for e in reg:
            reg_of[e.line_no] = ri

    paired = []
    for v in versions:
        anchor = (gitlog.get(v, (None, None, None))[2] or vn_json.get(v)
                  or None)
        if anchor is None:
            continue
        ri = reg_of.get(anchor)
        near_edit = [e for e in edits
                     if reg_of.get(e.line_no) == ri and e.line_no <= anchor]
        near_metric = [e for e in metric_events
                       if reg_of.get(e.line_no) == ri]
        if near_edit and near_metric:
            paired.append(v)

    verbatim_edits = sum(
        1 for e in edits
        for f in (e.meta.get("files") or {}).values() if f.get("verbatim"))

    print("  format          %s" % fmt)
    print("  events          %s" % dict(sorted(kinds.items())))
    print("  tiers           %s" % dict(sorted(tiers.items())))
    print("  code edits      %d  (verbatim-diff channel: %d)"
          % (len(edits), verbatim_edits))
    print("  metric outputs  %d T1/T2" % len(metric_events))
    print("  regions         %d (discontinuities: %d)"
          % (len(regs), kinds.get("discontinuity", 0)))
    print("  versions seen   %d   gitlog=%d vN.json=%d notes=%d"
          % (len(versions), len(gitlog), len(vn_json), len(notes)))
    print("  PAIRED          %d  %s"
          % (len(paired), ", ".join(paired[:18])))
    for v in list(gitlog)[:4]:
        sha, subj, ln = gitlog[v]
        print("    L%-6d %s %s: %s" % (ln, sha[:9], v, subj[:70]))
    for v in list(notes)[:4]:
        print("    notes  %s: %s" % (v, notes[v]))
    return {"format": fmt, "versions": len(versions), "paired": len(paired),
            "edits": len(edits), "verbatim": verbatim_edits,
            "metrics": len(metric_events)}


def main():
    out = {}
    for p in sys.argv[1:]:
        print("\n=== %s" % p)
        out[p] = probe(p)
    print("\n--- summary ---")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
