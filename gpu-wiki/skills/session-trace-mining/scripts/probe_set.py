#!/usr/bin/env python3
"""Phase-0 probe, second question: is the ladder recoverable at SET level?

The first probe assumed one session holds many versions. It does not: where the
orchestrator constrains a session to a single optimisation cycle, one session file
== one version's work, and the `git log --oneline` echoes inside it are the
*history* of every earlier version, not this session's content. Pairing per file
therefore had to fail.

So the real question is whether, across a whole set, we can attribute each
version to the session that produced it, and find that session's edits and
numbers. That is what this measures.

Usage: python3 probe_set.py <set-root> [--limit N]
"""
import json
import re
import sys
from collections import OrderedDict, Counter

import transcripts as T
from probe_versions import GITLOG_RE, has_metric, regions

# The version this session is working on, as the session itself names it. Three
# channels, in falling order of directness.
OWN_VN_JSON = re.compile(r"memory/v(\d+)\.json")
OWN_COMMIT = re.compile(r"git\s+commit[^\n]*?\bv(\d+)\s*:", re.I)
OWN_MM = re.compile(r"memory_manager\.py\s+(?:create|update)\s+v?(\d+)", re.I)


def own_version(events):
    """Which version this session produced, and how we know.

    Write/Edit of `memory/vN.json` is the strongest signal: the run writes that
    file exactly once per cycle, at the end, for its own version.
    """
    votes = Counter()
    how = {}
    for e in events:
        if e.kind == "edit":
            for path in (e.meta.get("files") or {}):
                m = OWN_VN_JSON.search(path)
                if m:
                    votes["v" + m.group(1)] += 5
                    how.setdefault("v" + m.group(1), "wrote memory/vN.json")
        elif e.kind == "tool-call":
            for m in OWN_COMMIT.finditer(e.text):
                votes["v" + m.group(1)] += 3
                how.setdefault("v" + m.group(1), "git commit vN:")
            for m in OWN_MM.finditer(e.text):
                votes["v" + m.group(1)] += 2
                how.setdefault("v" + m.group(1), "memory_manager create/update")
    if not votes:
        return None, None
    top = max(votes, key=lambda v: (votes[v], int(v[1:])))
    return top, how.get(top)


def main():
    root = sys.argv[1]
    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])

    files = list(T.iter_transcripts(root))
    # Subagent transcripts are a different unit (a delegated probe, not a
    # version), so they are not candidates for owning a version.
    main_files = [p for p in files if "subagents" not in p.parts]
    if limit:
        main_files = main_files[:limit]

    ladder = OrderedDict()      # vN -> (sha, subject) from any git log echo
    owned = OrderedDict()       # vN -> dict(file, how, edits, metrics)
    no_owner, fmt_count = [], Counter()

    for p in main_files:
        fmt, events = T.parse(p)
        fmt_count[fmt] += 1
        if not fmt:
            continue
        for e in events:
            if e.kind != "tool-output":
                continue
            for sha, n, subj in GITLOG_RE.findall(e.text):
                ladder.setdefault("v" + n, (sha, subj.strip()))
        ver, how = own_version(events)
        if not ver:
            no_owner.append(p.name)
            continue
        regs = regions(events)
        reg_of = {}
        for ri, reg in enumerate(regs):
            for e in reg:
                reg_of[e.line_no] = ri
        code_edits = [e for e in events
                      if e.kind == "edit" and e.meta.get("code_files")]
        verbatim = sum(1 for e in code_edits
                       for f in e.meta["files"].values() if f.get("verbatim"))
        metrics = [e for e in events if e.kind == "tool-output"
                   and e.tier in (T.TIER_T1, T.TIER_T2) and has_metric(e.text)]
        same_region = any(
            reg_of.get(a.line_no) == reg_of.get(b.line_no)
            for a in code_edits for b in metrics if a.line_no < b.line_no)
        prev = owned.get(ver)
        row = {"file": p.name, "how": how, "n_edits": len(code_edits),
               "n_verbatim": verbatim, "n_metrics": len(metrics),
               "paired": bool(same_region and code_edits and metrics),
               "regions": len(regs)}
        # More than one session can touch a version (a retry); keep the one that
        # actually has code and numbers.
        if prev is None or (row["paired"] and not prev["paired"]):
            owned[ver] = row

    full = [v for v, r in owned.items() if r["paired"] and r["n_verbatim"]]
    print("set              %s" % root)
    print("files            %d main (%s)" % (len(main_files), dict(fmt_count)))
    print("ladder versions  %d from git-log echoes  (max %s)"
          % (len(ladder),
             max(ladder, key=lambda v: int(v[1:])) if ladder else "-"))
    print("owned versions   %d attributed to a session" % len(owned))
    print("  with edits+metrics in one region : %d"
          % sum(1 for r in owned.values() if r["paired"]))
    print("  and a verbatim diff channel      : %d" % len(full))
    print("sessions with no own version       : %d" % len(no_owner))
    print("how attributed   %s"
          % dict(Counter(r["how"] for r in owned.values())))
    print("\nsample owned:")
    for v in sorted(owned, key=lambda x: int(x[1:]))[:12]:
        r = owned[v]
        sha, subj = ladder.get(v, ("-", "(not in any git log echo)"))
        print("  %-5s %s edits=%-3d verb=%-3d metr=%-3d paired=%-5s | %s"
              % (v, sha[:9], r["n_edits"], r["n_verbatim"], r["n_metrics"],
                 r["paired"], subj[:62]))
    print("\nladder-only (no session owns them, history from echoes): %d"
          % len(set(ladder) - set(owned)))


if __name__ == "__main__":
    main()
