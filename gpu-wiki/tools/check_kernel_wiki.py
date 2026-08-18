#!/usr/bin/env python3
"""Validate the JSON knowledge store against schema clean-1.3 and the wiki.

Six independent gates. Each one exists because a specific class of silent
corruption is possible in a generated store:

  1. schema        a payload field drifts and nothing notices until an agent
                   reads a record that is missing what it needs
  2. id            ids must be unique and deterministic, or feedback counters
                   attach to the wrong record after a rebuild
  4. anonymization the agent-facing layers must not leak a contributor, repo,
                   version id or corpus path -- the same rule the markdown
                   already passes
  5. raw isolation evidence.raw is human-only. This asserts that the serving
                   projection drops it entirely, so it cannot leak by omission
                   of a filter somewhere downstream
  6. relations     every internal id referenced by a record must exist
  7. established    an anti-strategy must state a fact -- a checkable condition
     fact           plus the cause -- or a later agent reads "measured, no gain"
                    as "this lever is dead" and abandons a live direction

Exit code is non-zero if any gate fails.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
GPU_WIKI = os.path.dirname(HERE)                     # gpu-wiki/
OUT_ROOT = os.path.join(GPU_WIKI, "kernel_wiki")     # where records/ lives

# This store has no markdown tree: records are the only source of truth, seeded
# once from the retired wiki and extended thereafter by the mining skills. The
# dangling-reference check is therefore expressed as generic leak SHAPES rather
# than by importing a corpus-specific scrubber. A committed denylist would
# publish the very names it guards, so private terms come from a file named by
# ATREX_WIKI_DENYLIST (one substring per line).
LEAK_PATTERNS = {
    "absolute path": re.compile(r"/(?:root|home|Users)/[\w.-]+"),
    "email": re.compile(r"(?<![\w.@])[\w.+-]{2,}@[\w-]+\.[a-z]{2,}\b"),
    "markdown page": re.compile(r"\b[\w][\w/-]*\.md\b"),
}


def private_denylist() -> list[str]:
    path = os.environ.get("ATREX_WIKI_DENYLIST")
    if not path or not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as handle:
        return [ln.strip() for ln in handle
                if ln.strip() and not ln.startswith("#")]


def comments_only(code: str) -> str:
    """Code is checked the way a source file is: comment text only."""
    out = []
    for line in code.splitlines():
        s = line.strip()
        if s.startswith(("#", "//", "*", "/*")):
            out.append(s)
        elif "#" in s:
            out.append(s.split("#", 1)[1])
    return "\n".join(out)

# The established-fact criteria live in the audit tool. Importing them rather
# than restating them means the gate and the audit can never disagree about what
# counts as a fact.
sys.path.insert(0, HERE)
from audit_anti_strategy import (CONDITION_KEYS,                # noqa: E402
                                 MIN_MECHANISM_CHARS,
                                 NON_MECHANISM_RE,
                                 NO_CONCLUSION_VERDICTS)

SCHEMA_PATH = os.path.join(GPU_WIKI, "schema", "kernel", "schema.json")

# Pages that are navigation, not knowledge. They legitimately produce no record.
NAV_BASENAMES = {"README.md"}
NAV_PAGES: set[str] = set()

AGENT_LAYERS = ("payload", "retrieval")

# These payload fields hold source, so version-like identifiers in them are code
# (a local named `v01`), not dangling prose references. Only their comments get
# scanned, mirroring how check_anonymized treats a .py file. Must stay in sync
# with build_kernel_records.CODE_KEYS.
CODEISH_KEYS = {"snippet", "dispatch_snippet", "attempted_code"}


def load_records(out_root: str) -> list[tuple[str, dict]]:
    out = []
    root = os.path.join(out_root, "records")
    for dirpath, _dirs, files in os.walk(root):
        for name in sorted(files):
            if name == "index.json":
                continue    # the generated index lives here too, not a record
            if name.endswith(".json"):
                path = os.path.join(dirpath, name)
                with open(path) as fh:
                    out.append((os.path.relpath(path, out_root), json.load(fh)))
    out.sort()
    return out


# ------------------------------------------------------------------- gate 1

def gate_schema(records: list[tuple[str, dict]]) -> list[str]:
    try:
        import jsonschema
    except ImportError:
        return ["SKIP: jsonschema not installed (pip install jsonschema)"]
    schema = json.load(open(SCHEMA_PATH))
    validator = jsonschema.Draft202012Validator(schema)
    errors = []
    for rel_path, record in records:
        for err in sorted(validator.iter_errors(record), key=lambda e: e.path):
            location = "/".join(str(p) for p in err.path) or "<root>"
            errors.append("%s: %s: %s" % (rel_path, location, err.message[:200]))
            if len(errors) > 40:
                errors.append("... truncated")
                return errors
    return errors


# ------------------------------------------------------------------- gate 2

def gate_ids(records: list[tuple[str, dict]]) -> list[str]:
    errors = []
    seen = {}
    for rel_path, record in records:
        rid = record["id"]
        if rid in seen:
            errors.append("duplicate id %s in %s and %s" % (rid, seen[rid], rel_path))
        seen[rid] = rel_path
        if os.path.basename(rel_path) != rid + ".json":
            errors.append("%s: filename does not match id %s" % (rel_path, rid))
        segments = rid.split(".")
        if len(segments) < 5:
            errors.append("%s: id needs at least 5 dotted segments "
                          "(vendor.product.dsl.operator_family.technique-disc)" % rid)
    return errors


# ------------------------------------------------------------------- gate 4

def agent_text(record: dict) -> tuple[str, str]:
    """Return (prose, code) for the agent-facing layers only."""
    prose_parts, code_parts = [], []

    def walk(node, in_code: bool) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value, in_code or key in CODEISH_KEYS)
        elif isinstance(node, list):
            for item in node:
                walk(item, in_code)
        elif isinstance(node, str):
            (code_parts if in_code else prose_parts).append(node)

    for layer in AGENT_LAYERS:
        walk(record.get(layer), False)
    walk(record["evidence"]["summary"], False)
    # worth.gain is served too; it left the payload in clean-1.3 but the same
    # no-dangling-reference rule follows it. worth.track is never served.
    walk(record["worth"].get("gain"), False)
    return "\n".join(prose_parts), "\n".join(code_parts)


def gate_anonymized(records: list[tuple[str, dict]]) -> list[str]:
    errors = []
    counts = Counter()
    for rel_path, record in records:
        prose, code = agent_text(record)
        # Source fragments are checked the way a .py file is: comments only.
        haystack = prose + "\n" + comments_only(code)
        for label, rx in LEAK_PATTERNS.items():
            for match in rx.finditer(haystack):
                token = match.group(0)
                if label == "version id" and re.match(
                        r"^[vV](?:2|4|8|16|32|64|100)$", token):
                    context = haystack[max(0, match.start() - 12):match.end() + 12]
                    if re.search(r"vec|\.f16|\.bf16|x2|float", context, re.I):
                        continue
                counts[label] += 1
                if len(errors) < 25:
                    errors.append("%s: dangling %s in an agent-facing layer: %s"
                                  % (rel_path, label, token))
    if counts:
        errors.append("dangling reference totals: %s" % dict(counts))
    return errors


# ------------------------------------------------------------------- gate 5

def serve(record: dict) -> dict:
    """The agent-facing projection. Must be the only thing ever served."""
    rank = record["worth"]["rank"]
    return {
        "id": record["id"],
        "type": record["type"],
        "level": record["level"],
        # locator and links are engine-only: a path or an id the agent cannot
        # dereference in this same query is noise at best.
        "retrieval": {key: value for key, value in record["retrieval"].items()
                      if key not in ("locator", "links")},
        "payload": record["payload"],
        "evidence": {"summary": record["evidence"]["summary"]},
        # track and the score decomposition are engine-only: raw counters invite
        # the agent to re-derive a ranking that was already computed, and tier
        # says the same thing in one word.
        "worth": {"rank": {"score": rank["score"], "tier": rank["tier"]},
                  **({"gain": record["worth"]["gain"]}
                     if "gain" in record["worth"] else {})},
    }


def gate_raw_isolation(records: list[tuple[str, dict]]) -> list[str]:
    errors = []
    for rel_path, record in records:
        raw = record["evidence"]["raw"]
        served = json.dumps(serve(record), ensure_ascii=False)
        for engine_only in ('"locator"', '"links"'):
            if engine_only in served:
                errors.append("%s: retrieval.%s reached the served projection"
                              % (rel_path, engine_only.strip('"')))
        # Structural, not substring: evidence.summary.measured_on has a "track"
        # key of its own, so name matching would false-positive here.
        worth = serve(record)["worth"]
        if sorted(worth) != ["gain", "rank"] or sorted(worth["rank"]) != ["score", "tier"]:
            errors.append("%s: served worth is %s / rank %s; only rank.score, "
                          "rank.tier and gain may be served"
                          % (rel_path, sorted(worth), sorted(worth["rank"])))
        if '"raw"' in served:
            errors.append("%s: served projection still contains a raw key" % rel_path)
        for key in ("source_repo", "detail_file", "git_commit"):
            value = raw.get(key)
            if value and str(value) in served:
                errors.append("%s: raw.%s leaked into the served projection: %s"
                              % (rel_path, key, value))
        for path in raw.get("file_paths") or []:
            if path in served:
                errors.append("%s: raw file path leaked into the served projection"
                              % rel_path)
    return errors


# ------------------------------------------------------------------- gate 6

def gate_relations(records: list[tuple[str, dict]], full_build: bool) -> list[str]:
    if not full_build:
        return ["SKIP: relation targets are only complete in a full build"]
    known = {record["id"] for _p, record in records}
    errors = []
    for rel_path, record in records:
        links = record["retrieval"].get("links") or {}
        for kind, targets in links.items():
            for target in ([targets] if isinstance(targets, str) else (targets or [])):
                if target and target not in known:
                    errors.append("%s: links.%s points at unknown id %s"
                                  % (rel_path, kind, target))
                    if len(errors) > 25:
                        return errors
    return errors


# ------------------------------------------------------------------- gate 7

ABSOLUTE_LATENCY = re.compile(r"[\d.]+\s*(?:us|µs|\u03bcs|ms)\b", re.I)
# A benefit metric may never be expressed in a time unit; that is what the
# sign-normalized delta_pct is for.
TIME_UNIT = re.compile(r"^\s*(?:us|µs|\u03bcs|ms|s|sec|ns|cycles?)\s*$", re.I)

# Types that describe a concrete code change, so they must ship code and lineage.
CHANGE_TYPES = {"strategy", "reference-kernel"}


def gate_self_contained(records: list[tuple[str, dict]]) -> list[str]:
    """An agent that retrieves ONE record must be able to act on it alone.

    Each check below corresponds to a question the agent cannot otherwise answer
    without a second query: which operator is this about, what was it built on,
    what is the code, and how much does it buy. The last one moved to worth.gain
    in clean-1.3, which is served with the payload, so the contract is unchanged:
    one query is enough.
    """
    errors = []
    for rel_path, record in records:
        payload = record["payload"]
        rtype = record["type"]

        def fail(message: str) -> None:
            if len(errors) < 30:
                errors.append("%s: %s" % (rel_path, message))

        if not (payload.get("goal") or "").strip():
            fail("payload.goal is empty; nothing orients the agent")

        problem = payload.get("problem") or {}
        if not (problem.get("statement") or "").strip():
            fail("payload.problem.statement is empty; the record does not say what "
                 "problem it solves")
        if not (problem.get("target") or "").strip():
            fail("payload.problem.target is empty; the agent cannot tell which GPU "
                 "the numbers came from")
        if record["level"] == "operator" and rtype in CHANGE_TYPES \
                and not problem.get("operator"):
            fail("operator-level record without payload.problem.operator")

        gain = record["worth"].get("gain") or {}
        if not gain.get("basis"):
            fail("worth.gain.basis missing")

        entries = list(gain.get("metrics") or []) + list(gain.get("regressions") or [])
        # Percentages only. An absolute latency here invites cross-operator
        # comparison, which the shape sets do not support.
        for key, value in gain.items():
            if isinstance(value, str) and ABSOLUTE_LATENCY.search(value):
                fail("worth.gain.%s states an absolute latency (%r); "
                     "gains must be percentages" % (key, value[:60]))
        for entry in entries:
            metric = entry.get("metric")
            if TIME_UNIT.match(str(entry.get("unit") or "")):
                fail("worth.gain metric %r carries a time unit (%r); a latency "
                     "entry may only carry delta_pct" % (metric, entry.get("unit")))
            if metric == "latency" and (entry.get("before") is not None
                                        or entry.get("after") is not None):
                fail("worth.gain latency entry carries an absolute before/after; "
                     "those belong in evidence.raw.effect")
            for field in ("note", "measured_over"):
                text = entry.get(field)
                if isinstance(text, str) and ABSOLUTE_LATENCY.search(text):
                    fail("worth.gain %s.%s states an absolute latency (%r)"
                         % (metric, field, text[:60]))
            # No-fabrication: a number with no stated origin cannot be audited.
            if not entry.get("source"):
                fail("worth.gain metric %r has no source" % metric)
            if entry.get("delta_pct") is None and entry.get("before") is None \
                    and entry.get("after") is None and not entry.get("note"):
                fail("worth.gain metric %r carries no number and no note" % metric)
        if gain.get("basis") == "measured" and rtype == "strategy" \
                and not any(e.get("delta_pct") is not None
                            for e in (gain.get("metrics") or [])):
            fail("measured strategy with no percentage gain in metrics[]")
        # pct is a copy lifted out for ranking, so it must never disagree with
        # the entry it was copied from. A null pct is not a disagreement when
        # there is nothing to copy: `metrics: []` with `pct: null` is exactly how
        # the distillation brief tells an anti-strategy to say "no delta here",
        # and rejecting it would force records to express that by omitting the key
        # instead of stating it.
        headline = [e.get("delta_pct") for e in (gain.get("metrics") or [])
                    if e.get("metric") == gain.get("primary")
                    and e.get("delta_pct") is not None]
        declared_no_delta = gain.get("pct") is None and not headline
        if "pct" in gain and not declared_no_delta and gain["pct"] not in headline:
            fail("worth.gain.pct=%r is not the primary metric's delta_pct %r"
                 % (gain["pct"], headline))
        if "pct" not in gain and headline:
            fail("worth.gain has a primary metric percentage but no pct; ranking "
                 "would have to walk metrics[]")

        if rtype in CHANGE_TYPES:
            builds_on = (payload.get("trace") or {}).get("builds_on") or {}
            if not (builds_on.get("approach") or "").strip():
                fail("payload.trace.builds_on.approach is empty; the agent cannot "
                     "tell what this was built on top of")
            implementation = payload.get("implementation") or {}
            fmt = implementation.get("format")
            if fmt != "none" and not (implementation.get("snippet") or "").strip():
                fail("implementation.format=%r but the snippet is empty" % fmt)
            # The reading convention has to travel with the code, and it now lives
            # in the format value rather than a separate constant field.
            if (implementation.get("snippet") or "").lstrip().startswith(("-", "+")) \
                    and not str(fmt).startswith("unified-diff"):
                fail("snippet looks like a diff but format=%r does not say which "
                     "side is before" % fmt)
    return errors


# ------------------------------------------------------------------- gate 9

# Paths into THIS repository. The agent has no access to it, so a path in the
# payload is something it cannot open: at best noise, at worst an invitation to
# hallucinate a file read. retrieval.locator stays engine-side; the code the agent
# actually gets is implementation.snippet, inlined verbatim in the payload.
WIKI_PATH = re.compile(r"\b[\w][\w/-]*\.md\b|(?:docs|records)/[\w./-]+")
# A record id is as undereferenceable to the agent as a path: it can only be
# resolved by querying again, which payload is required not to need.
RECORD_ID = re.compile(r"\bnvidia\.(?:b200|b300|any)\.[a-z0-9-]+\.[a-z0-9-]+\.")


def _values_for_key(node, wanted):
    if isinstance(node, dict):
        out = []
        for key, value in node.items():
            if key == wanted and isinstance(value, str):
                out.append(value)
            else:
                out += _values_for_key(value, wanted)
        return out
    if isinstance(node, list):
        return [v for item in node for v in _values_for_key(item, wanted)]
    return []


def gate_no_cross_reference(records):
    errors = []
    for rel_path, record in records:
        payload = record["payload"]
        # worth.gain left the payload in clean-1.3 but is still served, so the
        # same rule holds for its notes.
        served_knowledge = {"payload": payload,
                            "gain": record["worth"].get("gain") or {}}
        blob = json.dumps(served_knowledge, ensure_ascii=False)
        # Code bodies are verbatim corpus source, so a path inside a comment there
        # belongs to the original author and is not a reference we minted.
        for key in CODEISH_KEYS:
            for value in _values_for_key(payload, key):
                blob = blob.replace(json.dumps(value, ensure_ascii=False), '""')
        for match in WIKI_PATH.finditer(blob):
            if len(errors) < 25:
                errors.append("%s: served knowledge points into this repository (%s); "
                              "the agent cannot open it, use retrieval.locator instead"
                              % (rel_path, match.group(0)[:70]))
        for match in RECORD_ID.finditer(blob):
            if len(errors) < 25:
                errors.append("%s: served knowledge contains a record id (%s); ids "
                              "belong in retrieval.links, it must read without them"
                              % (rel_path, match.group(0)[:70]))
        implementation = payload.get("implementation") or {}
        for key in ("source_text", "source_truncated"):
            if key in implementation:
                errors.append("%s: implementation.%s is serve-time only and must not "
                              "be stored" % (rel_path, key))
    return errors


def gate_established_fact(records: list[tuple[str, dict]]) -> list[str]:
    """A negative record must state a fact, not report a measurement.

    Without this gate a page that says "tested 3 variants, all flat" enters the
    store as a hard conclusion. That is the most expensive kind of silent
    corruption here: a later agent reads it as "this lever is dead" and abandons
    a direction that was merely unmeasured, and nothing in the store records that
    the original run simply ended without a result.
    """
    errors: list[str] = []
    for rel_path, record in records:
        if record["type"] != "anti-strategy":
            continue
        payload = record["payload"]
        verdict = payload.get("verdict")
        if verdict in NO_CONCLUSION_VERDICTS:
            errors.append("%s: verdict %r means the run ended without a "
                          "conclusion" % (rel_path, verdict))
        fact = payload.get("established_fact")
        if not isinstance(fact, dict):
            errors.append("%s: no established_fact; a negative record must name "
                          "the condition and the cause" % rel_path)
            continue
        condition = fact.get("condition") or {}
        if not any((condition.get(key) or "").strip() for key in CONDITION_KEYS):
            errors.append("%s: established_fact.condition is empty; the operator "
                          "alone is not a checkable condition" % rel_path)
        mechanism = (fact.get("mechanism") or "").strip()
        if len(mechanism) < MIN_MECHANISM_CHARS:
            errors.append("%s: mechanism is %d chars, want >=%d"
                          % (rel_path, len(mechanism), MIN_MECHANISM_CHARS))
        elif NON_MECHANISM_RE.search(mechanism):
            errors.append("%s: mechanism reads as a measurement (%r), not a cause"
                          % (rel_path, NON_MECHANISM_RE.search(mechanism).group(0)))
    return errors[:25]


# ---------------------------------------------------------------------- driver

def gate_index(out_root: str, records: list[tuple[str, dict]]) -> list[str]:
    path = os.path.join(out_root, "records", "index.json")
    if not os.path.exists(path):
        return ["index.json is missing; run build_kernel_records.py"]
    index = json.load(open(path))
    errors = []
    if index.get("schema") != "clean-1.3":
        errors.append("index.json declares schema %r" % index.get("schema"))
    indexed = {entry["id"]: entry for entry in index["records"]}
    if len(indexed) != len(records):
        errors.append("index has %d entries, records/ has %d files"
                      % (len(indexed), len(records)))
    for rel_path, record in records:
        entry = indexed.get(record["id"])
        if entry is None:
            errors.append("%s: not present in index.json" % rel_path)
            continue
        if entry["retrieval"] != record["retrieval"]:
            errors.append("%s: index retrieval layer is stale" % rel_path)
        if entry["worth_score"] != record["worth"]["rank"]["score"]:
            errors.append("%s: index worth_score is stale" % rel_path)
        if entry["tier"] != record["worth"]["rank"]["tier"]:
            errors.append("%s: index tier is stale" % rel_path)
        if entry["gain_pct"] != (record["worth"].get("gain") or {}).get("pct"):
            errors.append("%s: index gain_pct is stale" % rel_path)
    return errors[:25]


GATES = (
    ("schema", "record shape matches clean-1.3"),
    ("ids", "ids unique, deterministic, matching filenames"),
    ("anonymization", "no dangling reference in an agent-facing layer"),
    ("raw-isolation", "evidence.raw and worth.track never reach the projection"),
    ("relations", "internal id references resolve"),
    ("index", "index.json agrees with records/"),
    ("self-contained", "a single served record is usable on its own"),
    ("no-cross-reference", "served knowledge carries no repository path, no id"),
    ("established-fact", "every anti-strategy names a condition and a cause"),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=OUT_ROOT)
    parser.add_argument("--full", action="store_true",
                        help="Also require resolvable "
                             "relations (only true after --all).")
    args = parser.parse_args()

    records = load_records(args.out)
    if not records:
        print("no records found under %s/records" % args.out)
        return 1

    results = {
        "schema": gate_schema(records),
        "ids": gate_ids(records),
        "anonymization": gate_anonymized(records),
        "raw-isolation": gate_raw_isolation(records),
        "relations": gate_relations(records, args.full),
        "index": gate_index(args.out, records),
        "self-contained": gate_self_contained(records),
        "no-cross-reference": gate_no_cross_reference(records),
        "established-fact": gate_established_fact(records),
    }

    by_type = Counter(record["type"] for _p, record in records)
    print("records: %d  (%s)" % (len(records), ", ".join(
        "%s=%d" % kv for kv in sorted(by_type.items()))))
    print()

    failed = 0
    for name, description in GATES:
        problems = results[name]
        skipped = [p for p in problems if p.startswith("SKIP:")]
        real = [p for p in problems if not p.startswith("SKIP:")]
        if skipped and not real:
            print("SKIP %-14s %s -- %s" % (name, description, skipped[0][6:]))
            continue
        if not real:
            print("PASS %-14s %s" % (name, description))
            continue
        failed += 1
        print("FAIL %-14s %s" % (name, description))
        for problem in real[:12]:
            print("       %s" % problem)
        if len(real) > 12:
            print("       ... %d more" % (len(real) - 12))

    print()
    print("%d of %d gates failed" % (failed, len(GATES)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
