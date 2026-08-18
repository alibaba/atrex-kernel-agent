#!/usr/bin/env python3
"""Build one self-contained packet per segment.

Each packet has two layers and they are not interchangeable:

  agent_facing  everything the agent may read and quote. Every string in here has
                been through the scrubber, so absolute paths, addresses, private
                denylist terms, version ids and workload hashes are already gone.
                The agent therefore cannot write a dangling reference even if it
                copies its input verbatim -- which is the point. Constraining
                this by prompt instead fails on the first agent that quotes a
                commit subject.
  provenance    the raw version, commit and trace label, for `evidence.raw` only.
                That layer is exempt from the anonymisation gate, and the store's
                raw-isolation gate proves it never reaches the served projection.

The diff goes to a sibling `<seg_id>.diff` rather than inline: a large diff
inlined in JSON has exhausted an agent's context before it produced anything.

Usage: RTM_TRACE=<trace dir> python3 build_packets.py [--limit N] [--type T]
"""
import argparse
import json
import re

import anonymize
import check_anonymized
import config as c

# Where a REPORT.md states its conclusion, in order of preference.
PROFILE_ANCHORS = (
    re.compile(r"\*\*Root cause\*\*:\s*(.+?)(?:\n\n|\Z)", re.S),
    re.compile(r"##\s*Key Finding[:\s]*(.+?)(?:\n##|\Z)", re.S),
    re.compile(r"##\s*(?:Diagnosis|Bottleneck Analysis)\s*(.+?)(?:\n##|\Z)", re.S),
    re.compile(r"\*\*Evidence\*\*:\s*(.+?)(?:\n\n|\Z)", re.S),
)

MAX_DIFF_BYTES = 60_000
MAX_REPORT_CHARS = 1200
MAX_KERNEL_BYTES = 40_000


def scrubber():
    """One scrubber per packet.

    Returned as a callable that keeps the instance's state, because the
    `shape-N` labels it allocates are only stable within one instance -- and a
    packet is the unit a reader sees.
    """
    sc = anonymize.Scrubber()

    def clean(text):
        if not text:
            return None
        return sc(str(text))

    clean.shape_label = sc.shape_label
    clean.raw = sc
    return clean


def report_conclusion(seg):
    """The run's own account of the bottleneck, if a REPORT.md states one."""
    for d in seg.get("report_dirs") or []:
        path = c.TRACE / "profiles" / d / "REPORT.md"
        if not path.is_file():
            continue
        text = path.read_text(errors="replace")
        for rx in PROFILE_ANCHORS:
            m = rx.search(text)
            if m and len(m.group(1).strip()) >= 30:
                return d, m.group(1).strip()[:MAX_REPORT_CHARS]
        # No anchor: the head of the report is still the run's framing.
        head = "\n".join(text.splitlines()[:20]).strip()
        if len(head) >= 30:
            return d, head[:MAX_REPORT_CHARS]
    return None, None


def diff_text(seg):
    """The change this version made to the kernel, as a unified diff."""
    sha, parent = seg.get("sha"), seg.get("parent")
    if not sha:
        return ""
    if parent:
        out = c.git("diff", "%s..%s" % (parent, sha), "--", "kernel.py")
    else:
        out = c.git("show", "--format=", sha, "--", "kernel.py")
    return out or ""


def kernel_text(seg):
    """A whole kernel file, for reference-kernel records."""
    rel = seg.get("kernel_file")
    if not rel:
        return ""
    if rel == "kernel.py" and seg.get("sha"):
        return c.git("show", "%s:kernel.py" % seg["sha"])
    path = c.TRACE / rel
    return path.read_text(errors="replace") if path.is_file() else ""


HASH8_RE = re.compile(r"^([0-9a-f]{8})\b")


def shape_table(clean, by_shape):
    """Per-workload latency with the workload hashes relabelled.

    The hash is meaningless to a reader and is a dangling reference, but the
    *shape* it identifies carries the whole point of a per-workload table, so the
    scrubber's stable packet-local labels are used instead of dropping the rows.

    The key is truncated to its 8-hex prefix first: a table is often keyed by a
    full uuid while the prose cites only the prefix, and the scrubber keys labels
    on the exact string it is handed -- so passing the full uuid gives one
    workload two different labels and makes the record contradict itself.
    """
    rows = []
    for key, us in sorted((by_shape or {}).items(), key=lambda kv: kv[1]):
        m = HASH8_RE.match(str(key))
        rows.append({"shape": clean.shape_label(m.group(1) if m else str(key)),
                     "latency_us": us})
    return rows


def build(seg, versions):
    v = versions.get(seg["version"], {})
    clean = scrubber()
    report_dir, conclusion = report_conclusion(seg)
    target = "%s (%s %s)" % ((seg.get("product") or "any").upper(),
                             seg["vendor"], seg["arch"])

    agent = {
        "operator": {
            "what_it_computes": clean(seg.get("operator_description")),
            "fixed_axes": seg.get("constants") or {},
            "variable_axes": seg.get("var_axes") or [],
            "n_benchmarked_shapes": seg.get("n_shapes"),
            "dsl": seg["dsl"],
            "gpu": target,
            "harness": seg.get("harness"),
        },
        "what_the_run_said": {
            "change": clean(v.get("action_description")),
            "commit_subject": clean(v.get("subject")),
            "commit_body": clean(v.get("body")),
            "category_hint": v.get("action_category"),
            "gate_verdict": clean(v.get("gate_failure")),
        },
        "measurements": {
            "geomean_us": v.get("geomean_us"),
            "improve_pct_vs_best_so_far": seg.get("improve_pct"),
            "arith_mean_us": v.get("arith_mean_us"),
            "per_shape_latency": shape_table(clean, v.get("by_shape")),
            "correctness": {
                "status": v.get("correctness_status"),
                "max_abs_err": v.get("max_abs_err"),
                "max_rel_err": v.get("max_rel_err"),
            },
            # Percentages in this run are relative to its own baseline, so the
            # record has to say whether they may be compared with the store's.
            "comparable_with_the_store": seg.get("comparable"),
        },
        "profiler": {
            "conclusion": clean(conclusion),
            # The one thing that decides whether a profiler-backed bottleneck may
            # be claimed at all.
            "ncu_measured_the_kernel_under_test": bool(seg.get("ncu_dirs")),
        },
        "open_directions": [
            {"direction": clean(d.get("direction")),
             "rationale": clean(d.get("rationale"))}
            for d in (v.get("open_directions") or []) if isinstance(d, dict)
        ][:5],
        "pitfalls_recorded_here": [
            {"pitfall": clean(p.get("pitfall")),
             "explanation": clean(p.get("explanation")),
             "fix": clean(p.get("fix"))}
            for p in (v.get("pitfalls") or []) if isinstance(p, dict)
        ][:5],
    }

    if seg["record_type"] == "anti-strategy":
        agent["the_failed_lever"] = clean(seg.get("deadend_text"))
        agent["how_it_was_fixed"] = clean(seg.get("pitfall_fix"))
        agent["split_instruction"] = {
            "mechanical": "This packet already isolates ONE failed lever. Write "
                          "exactly one record for it.",
            "curated": "This packet already isolates ONE recorded pitfall. Write "
                       "exactly one record for it.",
            "agent": "The prose below describes SEVERAL unrelated failed levers. "
                     "Write one record for THIS packet's most substantial lever "
                     "only, and name the others in payload.lesson or "
                     "payload.would_retry_if so they are not lost. An "
                     "anti-strategy payload has no next_steps key.",
        }.get(seg.get("split_by"))
        precheck = seg.get("fact_precheck")
        agent["established_fact_required"] = {
            "rule": "An anti-strategy is admissible only as a FACT: under a "
                    "checkable condition C, doing X necessarily yields the bad "
                    "result. Fill payload.established_fact with (a) condition -- "
                    "at least one of sm_arch / shape_regime / dtype / toolchain, "
                    "since the operator alone is not a condition -- and (b) "
                    "mechanism (>=40 chars), why it NECESSARILY fails. A "
                    "measurement is not a mechanism. Take both only from this "
                    "packet's own evidence.",
            "verdict_must_conclude": "verdict is one of accuracy-gate/ceiling, "
                                     "api-limitation, not-worth-it-here, "
                                     "performance-ceiling. 'unknown' and "
                                     "'unstable' were removed from the enum: a "
                                     "run that ended without a result is not "
                                     "negative knowledge.",
            "precheck": precheck,
            "precheck_note": (None if precheck in (None, "ok") else
                              "The prose reads as %s. Find the condition and the "
                              "cause in the evidence below; if neither is there, "
                              "emit no record for this lever." % precheck),
        }

    if seg["record_type"] == "strategy":
        b = seg.get("builds_on") or {}
        agent["builds_on"] = {
            "was_the_untouched_baseline": bool(b.get("is_baseline")),
            "previous_approach": clean(b.get("description")),
        }

    files = {}
    if seg["record_type"] == "reference-kernel":
        body = kernel_text(seg)
        if body:
            files["kernel"] = anonymize.scrub_code(
                body, clean.raw)[:MAX_KERNEL_BYTES]
    else:
        d = diff_text(seg)
        if d.strip():
            files["diff"] = anonymize.scrub_code(d, clean.raw)[:MAX_DIFF_BYTES]

    provenance = {
        # HUMAN ONLY. Copied into evidence.raw verbatim and never anywhere else:
        # the trace repository and the commit are what a reviewer re-resolves the
        # record against, and there is no page to cite instead.
        "source_repo": seg["source_repo"],
        "version": seg["version"],
        "git_commit": seg.get("sha"),
        "ncu_dirs": seg.get("ncu_dirs") or [],
        "report_dir": report_dir,
        "shape_label_note": ("%d workload hashes were relabelled shape-N"
                             % getattr(clean.raw, "n_shapes", 0)),
    }

    return {
        "seg_id": seg["seg_id"],
        "record_type": seg["record_type"],
        "seg_kind": seg["seg_kind"],
        "target": {
            "id": seg["id"],
            "episode_key": seg["episode_key"],
            # The store's layout, which wiki-gate derives from scope on insert:
            # records/<type>/<vendor>/<arch>/<dsl>/<operator_family>/
            "output_dir": "records/%s/%s/%s/%s/%s" % (
                seg["record_type"], seg["vendor"], seg["arch"], seg["dsl"],
                seg["operator_slug"]),
            "scope": {
                "vendor": seg["vendor"], "arch": seg["arch"],
                "product": seg["product"], "dsl": seg["dsl"],
                # Always the operator slug, including for an anti-strategy whose
                # id keys on the family. A null here silently breaks two
                # consumers: wiki-gate files the record under .../misc/ and then
                # finds no same-scope candidates to compare it against, and
                # recon.py skips it when grouping by operator.
                "operator_family": seg["operator_slug"],
            },
            "workload_family": seg["workload_family"],
            "level": "operator",
            "schema": c.SCHEMA_NAME,
        },
        "agent_facing": agent,
        "provenance": provenance,
        "files": files,
    }


def evidence_text(packet):
    """Everything the no-fabrication gate lets a number be quoted from."""
    parts = []

    def walk(node):
        if isinstance(node, dict):
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
        elif node is not None:
            parts.append(str(node))

    walk(packet["agent_facing"])
    walk(packet["files"])
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--type", dest="rtype")
    ap.add_argument("--seg", action="append",
                    help="only these seg_ids (repeatable)")
    args = ap.parse_args()

    c.require_trace()
    c.ensure_dirs()

    segments = [json.loads(l) for l in (c.WORK / "segments.jsonl").open()]
    versions = {json.loads(l)["version"]: json.loads(l)
                for l in (c.WORK / "versions.jsonl").open()}
    meta = json.loads((c.WORK / "meta.json").read_text())["meta"]

    for seg in segments:
        seg["operator_description"] = meta.get("description")
        seg["constants"] = meta.get("constants")
        seg["var_axes"] = meta.get("var_axes")
        seg["n_shapes"] = meta.get("n_shapes")
        seg["harness"] = meta.get("harness")

    if args.rtype:
        segments = [s for s in segments if s["record_type"] == args.rtype]
    if args.seg:
        wanted = set(args.seg)
        segments = [s for s in segments if s["seg_id"] in wanted]
    if args.limit:
        segments = segments[:args.limit]

    n_diff = n_kernel = 0
    leaks = []
    for seg in segments:
        packet = build(seg, versions)
        # Pre-flight the layer the agent reads against the store's own gate
        # patterns. A leak here becomes a rejected record later, and it is far
        # cheaper to see it now than after a batch has been distilled.
        hits = check_anonymized.scan_text(
            evidence_text({"agent_facing": packet["agent_facing"],
                           "files": {}}))
        if hits:
            leaks.append((seg["seg_id"], hits[:4]))
        packet["evidence_chars"] = len(evidence_text(packet))
        files = packet.pop("files")
        if "diff" in files:
            (c.PACKETS / ("%s.diff" % seg["seg_id"])).write_text(files["diff"])
            packet["agent_facing"]["diff_file"] = "%s.diff" % seg["seg_id"]
            packet["code_bytes"] = len(files["diff"])
            n_diff += 1
        if "kernel" in files:
            (c.PACKETS / ("%s.py" % seg["seg_id"])).write_text(files["kernel"])
            packet["agent_facing"]["kernel_file"] = "%s.py" % seg["seg_id"]
            packet["code_bytes"] = len(files["kernel"])
            n_kernel += 1
        packet.setdefault("code_bytes", 0)
        (c.PACKETS / ("%s.json" % seg["seg_id"])).write_text(
            json.dumps(packet, ensure_ascii=False, indent=1) + "\n")

    print("packets %d -> %s" % (len(segments), c.PACKETS))
    print("  with diff: %d   with kernel: %d" % (n_diff, n_kernel))
    if leaks:
        print("  ANONYMISATION LEAKS in %d packets (the store's gate would "
              "reject a record quoting these):" % len(leaks))
        for seg_id, hits in leaks[:8]:
            print("    %s: %s" % (seg_id, hits))
        return 1
    print("  anonymisation pre-flight: clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
