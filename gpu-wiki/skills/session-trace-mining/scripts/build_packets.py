#!/usr/bin/env python3
"""Self-contained distillation packets: everything an agent needs, nothing else.

One packet per segment. The agent that writes a record sees only its packet, so a
field it cannot fill from the packet is a field it must leave empty — that is the
whole point, and it is why the evidence text is assembled here under explicit
trust tiers rather than handed over as a transcript.

Two files per segment:

  packets/<seg>.json   metadata, the tiered evidence text, the target path
  packets/<seg>.diff   the verbatim code change, if the transcript has one

The diff is a separate file read by both the agent and the `verbatim` gate. That
is not tidiness: it is the hardest-won lesson of building these pipelines. When
the gate and the distiller read different renderings of the same change, the gate
passes snippets that do not exist.

Usage: STM_SET=<name> python3 build_packets.py [--limit N] [--only SEG_ID]
"""
import argparse
import hashlib
import json
import re
import sys
from collections import Counter

import config as c
import metrics as M
import transcripts as T

# How much of one transcript span to carry. A packet is read by a model, so the
# budget is real; a span longer than this is truncated at a line boundary and
# marked, never silently cut mid-number.
SPAN_CHARS = 6000
EVIDENCE_CHARS = 60000
DIFF_CHARS = 60000

# Only these tiers may enter the evidence text. T4 (agent prose) and T5 (the
# orchestrator prompt) are excluded: admitting the agent's own prose makes the
# no-fabrication gate vacuous, and the prompt states target percentages, which
# would license any number near the threshold.
CITABLE = (T.TIER_T1, T.TIER_T2, T.TIER_T3)


def truncate(text, limit):
    if len(text) <= limit:
        return text, False
    cut = text[:limit]
    nl = cut.rfind("\n")
    return (cut[:nl] if nl > limit // 2 else cut), True


def metric_bearing(text):
    """Does this span carry a number a record could cite?"""
    return bool(M.timings(text) or M.markdown_times(text)
                or M.SPEEDUP_RE.search(text) or M.PCT_RE.search(text)
                or M.PASS_FRACTION_RE.search(text) or M.TFLOPS_RE.search(text))


def collect_spans(events, want_lines, radius=2):
    """Citable spans: the cited lines, plus nearby measurement output.

    `radius` pulls in the outputs immediately around a citation because a
    benchmark's header (the shape list, the `unit:` line, the correctness verdict)
    is usually one event away from the number itself, and a number without its
    header is exactly what leads to a mis-united record.
    """
    idx = {e.line_no: i for i, e in enumerate(events)}
    keep = set()
    for ln in want_lines:
        i = idx.get(ln)
        if i is None:
            continue
        for j in range(max(0, i - radius), min(len(events), i + radius + 1)):
            keep.add(j)
    out = []
    for j in sorted(keep):
        e = events[j]
        if e.kind == "tool-output" and e.tier in CITABLE:
            out.append(e)
        elif e.kind == "edit":
            out.append(e)
    return out


def render_evidence(spans, cmd_of):
    """The haystack, with a machine-readable tier header on every span.

    The header is what lets a reviewer (and the evidence-tier gate) see whether a
    number came from a benchmark or from the run reading back its own notes.
    """
    parts, sources = [], Counter()
    for e in spans:
        if e.kind != "tool-output":
            continue
        body, cut = truncate(e.text.strip(), SPAN_CHARS)
        if not body:
            continue
        cmd = (e.meta.get("cmd") or cmd_of.get(e.meta.get("call_id"))
               or cmd_of.get(e.meta.get("tool_use_id")) or "")
        cmd = re.sub(r"\s+", " ", cmd)[:220]
        sources[e.tier] += 1
        parts.append("### [%s tool-output line %d]%s\ncmd: %s\n\n%s%s"
                     % (e.tier, e.line_no,
                        " (truncated)" if cut else "", cmd or "(not recorded)",
                        body, "\n...[truncated]" if cut else ""))
    text, _cut = truncate("\n\n".join(parts), EVIDENCE_CHARS)
    return text, dict(sources)


def render_diff(spans, cwd_hint):
    """The verbatim change, restricted to code files under the run's own tree.

    A session edits its notes, its plan and its memory records as well as the
    kernel; a diff that mixed those in would invite a snippet that is documentation
    rather than code.
    """
    parts, channels = [], Counter()
    for e in spans:
        if e.kind != "edit":
            continue
        for path, f in (e.meta.get("files") or {}).items():
            if not T.is_code_path(path):
                continue
            diff = f.get("diff") or ""
            if not diff.strip():
                continue
            channels[f.get("channel")] += 1
            parts.append("# --- line %d  channel=%s verbatim=%s\n%s"
                         % (e.line_no, f.get("channel"), f.get("verbatim"), diff))
    text, cut = truncate("\n".join(parts), DIFF_CHARS)
    return text, dict(channels), cut


def scope_line(seg):
    return "%s / %s / %s / %s" % (seg.get("arch"), seg.get("product"),
                                  seg["dsl"], seg.get("workload_family"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--only", action="append")
    ap.add_argument("--type", choices=("strategy", "anti-strategy",
                                       "reference-kernel"))
    args = ap.parse_args()

    set_name, cfg, root = c.require_set()
    c.ensure_dirs(set_name)
    work, pack = c.work(set_name), c.packets(set_name)
    segments = [json.loads(l) for l in (work / "segments.jsonl").open()]
    meta = json.loads((work / "meta.json").read_text())

    if args.only:
        segments = [s for s in segments if s["seg_id"] in set(args.only)
                    or s["id"] in set(args.only)]
    if args.type:
        segments = [s for s in segments if s["record_type"] == args.type]
    if args.limit:
        segments = segments[:args.limit]

    # Parse each transcript once, however many segments cite it.
    by_path = {}
    for seg in segments:
        by_path.setdefault(seg.get("rel_path"), []).append(seg)

    written, skipped = 0, []
    for rel_path, segs in by_path.items():
        if not rel_path:
            skipped.extend((s["seg_id"], "no owning transcript") for s in segs)
            continue
        path = root / rel_path if (root / rel_path).is_file() else root
        if not path.is_file():
            skipped.extend((s["seg_id"], "transcript not found") for s in segs)
            continue
        fmt, events = T.parse(path)
        cmd_of = {}
        for e in events:
            if e.kind == "tool-call":
                key = e.meta.get("call_id") or e.meta.get("tool_use_id")
                if key:
                    cmd_of[key] = e.text
        for seg in segs:
            spans = collect_spans(events, seg.get("cite_lines") or [])
            evidence, tier_counts = render_evidence(spans, cmd_of)
            diff, channels, diff_cut = render_diff(spans, meta.get("set_root"))
            if not evidence.strip():
                skipped.append((seg["seg_id"], "no citable evidence span"))
                continue

            bundle = (evidence + "\0" + diff).encode("utf-8", "replace")
            pkt = {
                "seg_id": seg["seg_id"],
                "record_id": seg["id"],
                "record_type": seg["record_type"],
                "seg_kind": seg.get("seg_kind"),
                "set": set_name,
                "unit": seg["unit"],
                "target": {
                    "output_dir": "records/%s/nvidia/%s/%s/%s"
                                  % (seg["record_type"], seg.get("arch"),
                                     seg["dsl"],
                                     seg.get("workload_family") or "any"),
                    "id": seg["id"],
                },
                "scope": {
                    "vendor": "nvidia",
                    "arch": seg.get("arch"),
                    "product": seg.get("product"),
                    "dsl": seg["dsl"],
                    "operator_slug": seg["operator_slug"],
                    "workload_family": seg.get("workload_family"),
                    "arch_basis": seg.get("arch_basis"),
                    "dsl_basis": seg.get("dsl_basis"),
                    "summary": scope_line(seg),
                },
                "session": {
                    "set": set_name,
                    "format": fmt,
                    "unit": seg["unit"],
                    "rel_path": rel_path,
                    "sibling_paths": seg.get("sibling_paths") or [],
                    "session_id": seg.get("session_id"),
                    "line_nos": seg.get("cite_lines") or [],
                    "line_digests": seg.get("cite_digests") or {},
                    "timestamp": seg.get("date"),
                    "candidate_id": "%s/%s" % (set_name, seg["seg_id"]),
                    "dedup_key": seg.get("dedup_key"),
                    "diff_coverage": seg.get("diff_coverage"),
                    "number_tiers": seg.get("number_tiers") or {},
                    "verdict": _verdict(seg),
                    "evidence_sha256": hashlib.sha256(bundle).hexdigest(),
                },
                "measurement": {
                    "version": seg["version"],
                    "improve_pct": seg.get("improve_pct"),
                    "geomean_us": seg.get("geomean_us"),
                    "before_us": seg.get("before_us"),
                    "after_us": seg.get("after_us"),
                    "n_shapes": seg.get("n_shapes"),
                    "delta_basis": seg.get("delta_basis"),
                    "side_basis": seg.get("side_basis"),
                    "baseline_label": seg.get("baseline_label"),
                    "candidate_label": seg.get("candidate_label"),
                    "correctness_status": seg.get("correctness_status"),
                    "gate_result": seg.get("gate_result"),
                    "quote": seg.get("quote"),
                },
                "narrative": {
                    "subject": seg.get("subject"),
                    "action_description": seg.get("action_description"),
                    "expected_impact": seg.get("expected_impact"),
                    "gate_failure": seg.get("gate_failure"),
                    "deadend_text": seg.get("deadend_text"),
                    "pitfall_fix": seg.get("pitfall_fix"),
                    "profile_evidence": seg.get("profile_evidence") or {},
                    "open_directions": seg.get("open_directions") or [],
                    "builds_on": seg.get("builds_on") or {},
                },
                # Set when the two sides of the comparison were chosen by
                # print order, i.e. nothing establishes they measure
                # alternatives of the same work. The evidence-tier gate
                # then forbids the record from publishing a gain.
                "claims_no_gain": bool(seg.get("claims_no_gain")),
                "evidence_tiers": tier_counts,
                "evidence_text": evidence,
                "diff_file": "%s.diff" % seg["seg_id"] if diff.strip() else None,
                "diff_channels": channels,
                "diff_truncated": diff_cut,
                "n_spans": len(spans),
            }
            (pack / ("%s.json" % seg["seg_id"])).write_text(
                json.dumps(pkt, ensure_ascii=False, indent=1) + "\n")
            if diff.strip():
                (pack / ("%s.diff" % seg["seg_id"])).write_text(diff)
            written += 1

    print("packets    %d -> %s" % (written, pack))
    if skipped:
        print("skipped    %d" % len(skipped))
        for seg_id, why in skipped[:8]:
            print("  %-28s %s" % (seg_id, why))
    return 0


def _verdict(seg):
    gate = str(seg.get("gate_result") or "").upper()
    if gate in ("REVERT", "REVERTED"):
        return "reverted"
    if gate == "FAIL":
        return "failed"
    if gate in ("PASS", "COMMITTED"):
        return "committed"
    if seg["record_type"] == "strategy":
        return "committed"
    return "unknown"


if __name__ == "__main__":
    sys.exit(main())
