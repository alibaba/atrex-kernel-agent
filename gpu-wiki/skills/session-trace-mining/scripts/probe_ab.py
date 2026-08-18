#!/usr/bin/env python3
"""Phase-0 probe, third question: what can the codex sets yield instead?

Some sets have no version ladder at all. One measured example: 3 `[vN]` headings
across six transcripts (all in the one file that keeps a notes log) and 0 `git log`
echoes with version subjects -- but 70 code edits with verbatim diffs and 72 metric
outputs in a single file. So the unit there is not a version but a *measured A/B*:
one code change with a number before it and a number after it, inside one monotone
region.

This counts those, and separately counts the ready-made variant comparisons the
harness prints on one line (`static= 546.67 us  clc= 485.81 us speedup=1.1253x`),
which are A/B pairs that need no pairing at all.

Usage: python3 probe_ab.py <set-root>
"""
import re
import sys
from collections import Counter

import transcripts as T
from probe_versions import regions

# A single line that already contains both sides of a comparison. The harness
# prints these deliberately, so they are the cheapest honest candidate.
INLINE_AB = re.compile(
    r"(\w+)\s*=\s*(\d+(?:\.\d+)?)\s*(us|µs|ms|ns)\s+"
    r"(\w+)\s*=\s*(\d+(?:\.\d+)?)\s*(us|µs|ms|ns)", re.I)
SPEEDUP = re.compile(r"speedup\s*=?\s*(\d+(?:\.\d+)?)\s*x", re.I)
# `A -> B us` / `A→B us`, the arrow form used in notes and commit subjects.
ARROW = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:->|→|=>)\s*(\d+(?:\.\d+)?)\s*(us|µs|ms|ns)", re.I)
# A bare timing line: `430.72 us 0 2.796875` or `[bench ...] 437.9 us  2511 TFLOP/s`
BARE_TIME = re.compile(r"(\d+(?:\.\d+)?)\s*(us|µs|μs|ms|ns)\b", re.I)
# ncu csv gives the unit in its own column, so it needs its own detector.
NCU_CSV = re.compile(r'"gpu__time_duration\.sum","(ns|us|ms)","(\d+(?:\.\d+)?)"')


def timings(text):
    """Every timing in one output, normalized to us. Unit is never guessed."""
    out = []
    for unit, val in NCU_CSV.findall(text):
        out.append(_us(float(val), unit))
    for val, unit in BARE_TIME.findall(text):
        out.append(_us(float(val), unit))
    return out


def _us(v, unit):
    unit = unit.lower().replace("µ", "u").replace("μ", "u")
    return v * {"ns": 1e-3, "us": 1.0, "ms": 1e3}[unit]


def main():
    root = sys.argv[1]
    files = [p for p in T.iter_transcripts(root) if "subagents" not in p.parts]
    totals = Counter()
    for p in files:
        fmt, events = T.parse(p)
        if not fmt:
            continue
        inline, arrows, paired = 0, 0, 0
        for e in events:
            if e.kind != "tool-output" or e.tier not in (T.TIER_T1, T.TIER_T2):
                continue
            for m in INLINE_AB.finditer(e.text):
                if m.group(1).lower() != m.group(4).lower():
                    inline += 1
            arrows += len(ARROW.findall(e.text))
        # edit -> metric pairing inside one monotone region
        for reg in regions(events):
            edits = [e for e in reg if e.kind == "edit"
                     and any(f.get("verbatim")
                             for f in (e.meta.get("files") or {}).values())]
            metric = [e for e in reg if e.kind == "tool-output"
                      and e.tier in (T.TIER_T1, T.TIER_T2) and timings(e.text)]
            for ed in edits:
                before = [m for m in metric if m.line_no < ed.line_no]
                after = [m for m in metric if m.line_no > ed.line_no]
                if before and after:
                    paired += 1
        print("  %-64s inline_ab=%-4d arrow=%-4d edit_paired=%-4d"
              % (p.name[:64], inline, arrows, paired))
        totals["inline_ab"] += inline
        totals["arrow"] += arrows
        totals["edit_paired"] += paired
        totals["files"] += 1
    print("\nset %s" % root)
    print("totals %s" % dict(totals))


if __name__ == "__main__":
    main()
