#!/usr/bin/env python3
"""Audit anti-strategy records against the established-fact rule.

A negative record earns its place in the main store only if it states a FACT:

    under checkable condition C, doing X necessarily yields bad result Y

Three criteria, all mandatory:

  1. checkable condition -- an arch, a shape regime, a dtype or a toolchain
     version. "on this operator" is not a condition: a lever that failed on one
     operator is an observation, not a law.
  2. causal mechanism -- why it necessarily fails. "measured, no gain" is a
     measurement, not a mechanism.
  3. a verdict that concludes -- 'unknown' and 'unstable' mean the run ended
     without a conclusion, so there is nothing to record.

This tool only TRIAGES. Regex cannot judge whether prose explains a mechanism,
so anything not decidable by rule is routed to model review, the same division
of labour wiki-gate uses: the tool narrows, the model decides.

    python3 audit_anti_strategy.py --report
    python3 audit_anti_strategy.py --emit-review /tmp/review.json
    python3 audit_anti_strategy.py --apply /tmp/decisions.json
    python3 audit_anti_strategy.py --apply /tmp/decisions.json --dry-run
"""
from __future__ import annotations

import argparse
import collections
import datetime
import glob
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
GPU_WIKI = HERE.parent
KERNEL_WIKI = GPU_WIKI / "kernel_wiki"
MAIN_ANTI = KERNEL_WIKI / "records" / "anti-strategy"
STAGING_ANTI = KERNEL_WIKI / "trace_wiki" / "records" / "anti-strategy"

NO_CONCLUSION_VERDICTS = {"unknown", "unstable"}

# A condition must pin something the reader can check. Operator alone does not.
CONDITION_RE = re.compile(
    r"\b(sm_?\d{2,3}|b200|b300|h100|a100|blackwell|hopper|ampere"
    r"|[MNK]\s*[=<>≤≥]\s*\d+|small-[mnk]\b|large-[mnk]\b|tiny-[mnk]\b"
    r"|bf16|fp16|fp8|fp6|fp4|fp32|fp64|tf32|int8|e4m3|e5m2"
    r"|triton \d|cuda \d{2}|cutlass \d|ptx \d|driver \d"
    r"|when [a-z]|only (?:if|when|on)|for (?:all )?shapes? (?:with|where|below|above))\b",
    re.I)

# Phrases that describe a measurement instead of a cause.
NON_MECHANISM_RE = re.compile(
    r"(no improvement (?:found|over)|flat[- ]within[- ]noise|within noise"
    r"|all (?:\w+\s+){0,3}(?:approaches|trials|variants|attempts) (?:tested|were|failed|flat)"
    r"|tested \d+ approaches|reverted to an earlier|hw floor (?:confirmed|reconfirmed)"
    r"|no kernel change|sanity[_ ]bench|stall[_ ]count)", re.I)

# Words that signal a causal explanation rather than a result readout.
MECHANISM_RE = re.compile(
    r"\b(because|since|due to|caused by|the reason|as a result of"
    r"|bound by|limited by|gated by|serializ|contention|saturat"
    r"|spill|occupancy|bank conflict|scoreboard|barrier|mbarrier"
    r"|bandwidth[- ]bound|latency[- ]bound|compute[- ]bound"
    r"|wave quantization|tail effect|partial wave|launch overhead"
    r"|does not fit|exceeds|cannot|unsupported|not supported|requires)\b",
    re.I)

MIN_MECHANISM_CHARS = 40


def prose_of(record: dict) -> str:
    """Every field where a mechanism could legitimately be written."""
    payload = record.get("payload") or {}
    keys = ("lesson", "attempted", "observed", "root_cause",
            "would_retry_if", "hypothesis", "verdict")
    parts = [str(payload.get(k) or "") for k in keys]
    fact = payload.get("established_fact") or {}
    if isinstance(fact, dict):
        parts.append(str(fact.get("mechanism") or ""))
    return " ".join(p for p in parts if p)


def structured_condition(record: dict) -> str | None:
    """A condition already pinned in structured fields, if any."""
    retrieval = record.get("retrieval") or {}
    signals = retrieval.get("signals") or {}
    regime = (signals.get("shape_regime") or {}).get("predicate")
    if regime and regime != "*":
        return "shape_regime=%s" % regime
    payload = record.get("payload") or {}
    fact = payload.get("established_fact") or {}
    if isinstance(fact, dict):
        cond = fact.get("condition") or {}
        pinned = [f"{k}={v}" for k, v in cond.items() if v]
        if pinned:
            return ", ".join(pinned)
    return None


def rediscovered(record: dict) -> int:
    payload = record.get("payload") or {}
    trace = payload.get("trace") or {}
    if trace.get("rediscovered"):
        return int(trace["rediscovered"])
    metrics = ((record.get("retrieval") or {}).get("signals") or {}).get("metrics") or {}
    return int(metrics.get("rediscovered") or 0)


def triage(record: dict) -> tuple[str, list[str]]:
    """Return (verdict, reasons). verdict in keep / demote / needs-review."""
    payload = record.get("payload") or {}
    prose = prose_of(record)
    reasons: list[str] = []

    # criterion 3 -- decidable by rule alone
    if payload.get("verdict") in NO_CONCLUSION_VERDICTS:
        return "demote", ["verdict=%s means the run reached no conclusion"
                          % payload.get("verdict")]

    # criterion 1
    struct_cond = structured_condition(record)
    prose_cond = CONDITION_RE.search(prose)
    has_condition = bool(struct_cond or prose_cond)
    if struct_cond:
        reasons.append("condition (structured): %s" % struct_cond)
    elif prose_cond:
        reasons.append("condition (prose): %r" % prose_cond.group(0))
    else:
        reasons.append("no checkable condition")

    # criterion 2
    hollow = NON_MECHANISM_RE.search(prose)
    mech = MECHANISM_RE.search(prose)
    root_cause = (payload.get("root_cause") or "").strip()
    fact = payload.get("established_fact") or {}
    stated = (fact.get("mechanism") or "").strip() if isinstance(fact, dict) else ""

    if stated and len(stated) >= MIN_MECHANISM_CHARS and not NON_MECHANISM_RE.search(stated):
        has_mechanism = True
        reasons.append("mechanism: established_fact.mechanism is filled in")
    elif mech and len(root_cause) >= MIN_MECHANISM_CHARS:
        has_mechanism = True
        reasons.append("mechanism: root_cause states a cause (%r)" % mech.group(0))
    elif mech:
        has_mechanism = None            # plausible, but needs a human/model read
        reasons.append("mechanism: the prose has a causal word (%r) but it never "
                       "reached root_cause" % mech.group(0))
    else:
        has_mechanism = False
        reasons.append("no causal mechanism"
                       + (" (the wording is a measurement result: %r)"
                          % hollow.group(0) if hollow else ""))

    n = rediscovered(record)
    if n > 1:
        reasons.append("independently reproduced %d times" % n)

    if has_condition and has_mechanism is True:
        return "keep", reasons
    if not has_condition and has_mechanism is False:
        return "demote", reasons
    return "needs-review", reasons


CONDITION_KEYS = ("sm_arch", "shape_regime", "dtype", "toolchain")


def validate_fact(fact: dict) -> str | None:
    """Check a model-supplied established_fact the way schema and gate will.

    Returns None when acceptable, else a short reason. Applied before writing so
    a sloppy backfill is demoted rather than silently weakening the store.
    """
    if not isinstance(fact, dict):
        return "established_fact is not an object"

    condition = fact.get("condition")
    if not isinstance(condition, dict):
        return "condition missing"
    unknown = [k for k in condition if k not in CONDITION_KEYS]
    if unknown:
        return "condition has unknown key(s): %s" % ", ".join(sorted(unknown))
    if not any((condition.get(k) or "").strip() for k in CONDITION_KEYS
               if isinstance(condition.get(k), str)):
        return "condition has no non-empty field"

    mechanism = fact.get("mechanism")
    if not isinstance(mechanism, str) or len(mechanism.strip()) < MIN_MECHANISM_CHARS:
        return "mechanism shorter than %d chars" % MIN_MECHANISM_CHARS
    if NON_MECHANISM_RE.search(mechanism):
        return "mechanism reads as a measurement, not a cause"
    return None


def main_records() -> list[str]:
    return sorted(p for p in glob.glob(str(MAIN_ANTI / "**" / "*.json"), recursive=True)
                  if os.path.basename(p) != "index.json")


def demote(path: str, reason: str, dry_run: bool) -> str:
    """Move a record from the main store back to staging. Never deletes."""
    with open(path, encoding="utf-8") as handle:
        record = json.load(handle)

    rel = os.path.relpath(path, MAIN_ANTI)
    dest = STAGING_ANTI / rel

    if dry_run:
        return str(dest.relative_to(KERNEL_WIKI))

    record["status"] = "active"
    record["demoted_from_main"] = True
    record["demoted_reason"] = reason
    record["demoted_at"] = datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(record, ensure_ascii=False, indent=1) + "\n",
                    encoding="utf-8")
    os.remove(path)
    return str(dest.relative_to(KERNEL_WIKI))


def cmd_report() -> int:
    files = main_records()
    buckets: collections.Counter = collections.Counter()
    samples: dict = collections.defaultdict(list)

    for path in files:
        with open(path, encoding="utf-8") as handle:
            record = json.load(handle)
        verdict, reasons = triage(record)
        buckets[verdict] += 1
        if len(samples[verdict]) < 4:
            samples[verdict].append((record["id"], reasons))

    print("main-store anti-strategy records: %d\n" % len(files))
    for name, blurb in (
            ("keep", "all three criteria hold, stays in the main store"),
            ("needs-review", "the rules cannot decide, goes to the model"),
            ("demote", "the rules alone find it unfit, returns to staging")):
        print("%-14s %4d   %s" % (name, buckets[name], blurb))
    print()
    for name in ("demote", "needs-review", "keep"):
        if not samples[name]:
            continue
        print("--- %s samples ---" % name)
        for rid, reasons in samples[name]:
            print("  %s" % rid[:72])
            for r in reasons:
                print("      %s" % r)
        print()
    print("next: --emit-review exports the needs-review records for the model")
    return 0


def cmd_emit_review(dest: str) -> int:
    files = main_records()
    out = []
    for path in files:
        with open(path, encoding="utf-8") as handle:
            record = json.load(handle)
        verdict, reasons = triage(record)
        if verdict != "needs-review":
            continue
        payload = record.get("payload") or {}
        out.append({
            "id": record["id"],
            "path": os.path.relpath(path, KERNEL_WIKI),
            "triage_reasons": reasons,
            "verdict_field": payload.get("verdict"),
            "structured_condition": structured_condition(record),
            "rediscovered": rediscovered(record),
            "lesson": payload.get("lesson"),
            "attempted": payload.get("attempted"),
            "observed": payload.get("observed"),
            "root_cause": payload.get("root_cause"),
            "would_retry_if": payload.get("would_retry_if"),
        })
    Path(dest).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n",
                          encoding="utf-8")
    print("wrote %s (%d records for model review)" % (dest, len(out)))
    print()
    print("Return one of these three for each record:")
    print('  {"id": ..., "decision": "keep",   "reason": "..."}')
    print('  {"id": ..., "decision": "demote", "reason": "..."}')
    print('  {"id": ..., "decision": "backfill",')
    print('   "established_fact": {"condition": {"sm_arch": null, "shape_regime": "large-N",')
    print('                                      "dtype": null, "toolchain": null},')
    print('                        "mechanism": "at least 40 chars, why it '
          'necessarily fails under that condition"}}')
    print()
    print("Use backfill only when the mechanism really is written in this "
          "record's own prose -- never invent one.")
    print("then: --apply <decisions.json> [--dry-run]")
    return 0


def cmd_apply(decisions_path: str, dry_run: bool) -> int:
    decisions = json.loads(Path(decisions_path).read_text(encoding="utf-8"))
    by_id = {d["id"]: d for d in decisions} if isinstance(decisions, list) else decisions

    files = main_records()
    moved: list[tuple[str, str, str]] = []
    backfilled: list[str] = []
    kept = 0
    unresolved: list[str] = []
    rejected_backfill: list[tuple[str, str]] = []

    for path in files:
        with open(path, encoding="utf-8") as handle:
            record = json.load(handle)
        rid = record["id"]
        verdict, reasons = triage(record)

        if verdict == "keep":
            kept += 1
            continue

        if verdict == "demote":
            reason = "; ".join(reasons)
        else:                                        # needs-review
            decision = by_id.get(rid)
            if not decision:
                unresolved.append(rid)
                continue

            call = decision.get("decision")

            if call == "keep":
                kept += 1
                continue

            if call == "backfill":
                fact = decision.get("established_fact") or {}
                problem = validate_fact(fact)
                if problem:
                    # A malformed backfill must not sneak a non-fact into the
                    # store: fall through to demotion with the reason recorded.
                    rejected_backfill.append((rid, problem))
                    reason = "backfill rejected (%s); %s" % (problem, "; ".join(reasons))
                else:
                    if not dry_run:
                        record["payload"]["established_fact"] = fact
                        with open(path, "w", encoding="utf-8") as handle:
                            handle.write(json.dumps(record, ensure_ascii=False,
                                                    indent=1) + "\n")
                    backfilled.append(rid)
                    kept += 1
                    continue
            else:
                reason = decision.get("reason") or "; ".join(reasons)

        dest = demote(path, reason, dry_run)
        moved.append((rid, dest, reason))

    verb = "would move" if dry_run else "moved"
    filled = "would backfill" if dry_run else "backfilled"
    print("kept in main store : %d" % kept)
    print("%-19s: %d" % (filled, len(backfilled)))
    print("%-19s: %d" % (verb, len(moved)))
    if rejected_backfill:
        print("backfill rejected  : %d (demoted instead)" % len(rejected_backfill))
        for rid, why in rejected_backfill[:6]:
            print("    %s -- %s" % (rid[:60], why))
    if unresolved:
        print("unresolved (no model decision, left untouched): %d" % len(unresolved))
        for rid in unresolved[:8]:
            print("    %s" % rid[:72])

    if moved:
        print("\n%s to staging:" % verb)
        for rid, dest, reason in moved[:12]:
            print("  %s" % rid[:70])
            print("      -> %s" % dest)
            print("      reason: %s" % reason[:110])
        if len(moved) > 12:
            print("  ... and %d more" % (len(moved) - 12))

    if not dry_run and moved:
        print("\nnow rebuild the index: python3 tools/build_kernel_wiki.py --all")
    if dry_run:
        print("\ndry run: nothing was written")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--report", action="store_true",
                       help="Triage every record and print the three buckets.")
    group.add_argument("--emit-review", metavar="PATH",
                       help="Export the needs-review records for model adjudication.")
    group.add_argument("--apply", metavar="DECISIONS",
                       help="Demote failures, using the model's decisions for needs-review.")
    parser.add_argument("--dry-run", action="store_true",
                        help="With --apply: report without moving anything.")
    args = parser.parse_args()

    if not MAIN_ANTI.is_dir():
        print("ERROR no anti-strategy records at %s" % MAIN_ANTI, file=sys.stderr)
        return 2

    if args.report:
        return cmd_report()
    if args.emit_review:
        return cmd_emit_review(args.emit_review)
    return cmd_apply(args.apply, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
