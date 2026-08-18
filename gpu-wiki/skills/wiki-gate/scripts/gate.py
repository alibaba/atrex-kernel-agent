#!/usr/bin/env python3
"""wiki-gate: two-step tool for the kernel agent.

Step 1 — match:
    python3 scripts/gate.py --match --input /path/to/incoming.json
    Returns a JSON on stdout with schema validation result, the incoming record's
    key fields, and a list of scoped candidates (each with id, change, mechanism,
    gain direction, shape_contract). The AGENT reads this and decides.

Step 2 — commit:
    python3 scripts/gate.py --commit insert   --input /path/to/incoming.json
    python3 scripts/gate.py --commit confirm  --input /path/to/incoming.json --target <record_id>
    python3 scripts/gate.py --commit conflict --input /path/to/incoming.json --target <record_id>

    Executes the agent's decision: writes the record, increments a counter, or
    flags a conflict. stdout prints the action taken.

Exit codes:
    0  success
    1  schema validation failed
    2  argument error
"""
from __future__ import annotations

import argparse
import datetime
import json
import re
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import config as C


# ---------------------------------------------------------------- schema

# A negative record earns a place in the main store only as a FACT: under a
# checkable condition, this necessarily fails. Checked here rather than left to
# jsonschema because the ImportError branch below only compares a version string,
# and "measured, no gain" entering as a hard conclusion is exactly the corruption
# this gate exists to stop -- a later agent reads it and abandons a live lever.
# Normative definition: references/established-fact-criteria.md
NO_CONCLUSION_VERDICTS = {"unknown", "unstable"}
CONDITION_KEYS = ("sm_arch", "shape_regime", "dtype", "toolchain")
MIN_MECHANISM_CHARS = 40
NON_MECHANISM_RE = re.compile(
    r"(no improvement (?:found|over)|flat[- ]within[- ]noise|within noise"
    r"|all (?:\w+\s+){0,3}(?:approaches|trials|variants|attempts) "
    r"(?:tested|were|failed|flat)|tested \d+ approaches"
    r"|hw floor (?:confirmed|reconfirmed)|consecutive (?:dead-end|stall|revert))",
    re.I)


def validate_established_fact(record: dict) -> list[str]:
    """Errors that make an anti-strategy inadmissible. Empty list means OK."""
    if record.get("type") != "anti-strategy":
        return []
    payload = record.get("payload") or {}
    errors = []
    verdict = payload.get("verdict")
    if verdict in NO_CONCLUSION_VERDICTS:
        errors.append("verdict %r means the run ended without a conclusion, so "
                      "there is no negative knowledge to store" % verdict)
    fact = payload.get("established_fact")
    if not isinstance(fact, dict):
        errors.append("anti-strategy without payload.established_fact: name the "
                      "condition it fails under and why it necessarily fails")
        return errors
    condition = fact.get("condition") or {}
    if not any((condition.get(k) or "").strip() for k in CONDITION_KEYS):
        errors.append("established_fact.condition is empty: the operator alone "
                      "is not a checkable condition")
    mechanism = (fact.get("mechanism") or "").strip()
    if len(mechanism) < MIN_MECHANISM_CHARS:
        errors.append("established_fact.mechanism is %d chars, want >=%d"
                      % (len(mechanism), MIN_MECHANISM_CHARS))
    else:
        hit = NON_MECHANISM_RE.search(mechanism)
        if hit:
            errors.append("established_fact.mechanism reads as a measurement "
                          "(%r), not a cause" % hit.group(0))
    return errors


def validate_store_gates(record: dict) -> list[str]:
    """Run the store's own record-level gates against one incoming record.

    The gate calls itself the only entrance to records/, but it used to validate
    less than tools/check_kernel_wiki.py does, so a record could be admitted here
    and only fail afterwards -- with the bad record already in the main store.
    Measured on the 71 staged records: anonymization would have let 2 dangling
    references through and self-contained 5 records whose worth.gain.pct disagrees
    with their own primary metric.

    Only the four gates that judge a record on its own are applicable. ids,
    coverage, relations and index are store-wide and are checked by insert_record
    (id collision) or by the full run after the batch.
    """
    sys.path.insert(0, str(C.GPU_WIKI / "tools"))
    try:
        import check_kernel_wiki as ck
    except Exception as exc:                      # noqa: BLE001 - report, not skip
        return ["cannot load the store's gates from tools/check_kernel_wiki.py "
                "(%s); refusing rather than admitting a record that only two of "
                "the ten gates have seen" % exc]

    one = [("incoming/" + record.get("id", "?") + ".json", record)]
    errors = []
    for gate_fn in (ck.gate_anonymized, ck.gate_raw_isolation,
                    ck.gate_self_contained, ck.gate_no_cross_reference):
        errors += [e for e in gate_fn(one) if not e.startswith("SKIP:")]
    return errors


def validate_schema(record: dict) -> list[str]:
    errors = validate_established_fact(record) + validate_store_gates(record)
    try:
        import jsonschema
        schema = json.loads(C.SCHEMA_PATH.read_text())
        validator = jsonschema.Draft202012Validator(schema)
        return errors + [e.message for e in validator.iter_errors(record)]
    except ImportError:
        if record.get("schema") != C.SCHEMA_VERSION:
            errors.append("schema version mismatch: got %r, want %s"
                          % (record.get("schema"), C.SCHEMA_VERSION))
        return errors


# ---------------------------------------------------------------- index & scope

def load_index() -> list[dict]:
    if not C.INDEX_PATH.is_file():
        return []
    index = json.loads(C.INDEX_PATH.read_text())
    return index.get("records", [])


def load_scoped_candidates(incoming: dict, entries: list[dict]) -> list[dict]:
    scope = (incoming.get("retrieval") or {}).get("scope") or {}
    vendor = scope.get("vendor")
    arch = scope.get("arch")
    dsl = scope.get("dsl")
    family = scope.get("operator_family")

    filtered = []
    for e in entries:
        e_scope = (e.get("retrieval") or {}).get("scope") or {}
        if vendor and e_scope.get("vendor") != vendor:
            continue
        if arch and e_scope.get("arch") != arch:
            continue
        # "any" means "not tied to a DSL", so it has to match in both directions.
        # Comparing only against ("any", dsl) made the filter one-way: an incoming
        # whose own dsl is "any" saw nothing but other "any" records, so
        # DSL-agnostic knowledge was judged against an almost empty pool and a
        # rediscovery would be inserted as new. Measured on a real record: an
        # "any" incoming got 0 candidates while its near-duplicate at dsl=triton
        # (+15.5% against the incoming's +15.78%) was already in the store.
        if dsl and dsl != "any" and e_scope.get("dsl") not in (dsl, "any"):
            continue
        if family and e_scope.get("operator_family") != family:
            continue
        filtered.append(e)

    full = []
    for e in filtered:
        path = C.KERNEL_WIKI / e["path"]
        if path.is_file():
            full.append(json.loads(path.read_text()))
    return full


# ---------------------------------------------------------------- candidate summary

def _gain_direction(record: dict) -> str:
    rtype = record.get("type")
    if rtype == "anti-strategy":
        return "negative"
    worth = record.get("worth") or {}
    gain = worth.get("gain") or {}
    pct = gain.get("pct")
    if pct is not None:
        return "positive" if pct > 0 else "negative"
    if rtype == "strategy":
        return "positive"
    return "unknown"


def summarize_candidate(record: dict) -> dict:
    """Extract the fields an agent needs to judge semantic similarity."""
    payload = record.get("payload") or {}
    problem = payload.get("problem") or {}
    worth = record.get("worth") or {}
    gain = worth.get("gain") or {}
    return {
        "id": record["id"],
        "type": record.get("type"),
        # The store's declared semantic identity. Surfaced because an exact match
        # answers deterministically what the agent would otherwise re-derive from
        # prose: measured on one batch, 11 incoming records held only 5 distinct
        # episode_keys, so 6 of them were rediscoveries of each other.
        "episode_key": record.get("episode_key"),
        "change": payload.get("change", ""),
        "mechanism": payload.get("mechanism", ""),
        "goal": payload.get("goal", ""),
        "observed_symptom": problem.get("observed_symptom", ""),
        "operator": problem.get("operator", ""),
        "bottleneck": problem.get("bottleneck"),
        "shape_contract": problem.get("shape_contract"),
        "gain_direction": _gain_direction(record),
        "gain_pct": gain.get("pct"),
        "gain_basis": gain.get("basis"),
    }


# ---------------------------------------------------------------- step 1: match

def do_match(record: dict) -> dict:
    """Validate, scope-filter, and return candidates for the agent to judge."""
    errors = validate_schema(record)
    if errors:
        return {"status": "schema_error", "errors": errors[:10],
                "incoming": None, "candidates": []}

    entries = load_index()
    candidates = load_scoped_candidates(record, entries)
    summaries = [summarize_candidate(c) for c in candidates]

    key = record.get("episode_key")
    exact = [c["id"] for c in summaries if key and c["episode_key"] == key]

    return {
        "status": "ok",
        "incoming": summarize_candidate(record),
        "candidates_count": len(candidates),
        # Same episode_key means the store already holds this semantic identity,
        # so confirm is the mechanical action and no prose comparison is needed.
        "episode_key_matches": exact,
        "candidates": summaries,
    }


# ---------------------------------------------------------------- step 2: commit

def append_to_index(record: dict, rel_path: str) -> None:
    """Add one entry to records/index.json.

    Without this the record is on disk but invisible: the next --match cannot see
    it, so a second incoming that duplicates it would be inserted again, and the
    store's own "index" gate fails because the file count no longer matches. The
    entry is derived by tools/reindex_kernel_wiki.py rather than rebuilt here, so
    the gate and the builder cannot drift apart.
    """
    tools = C.GPU_WIKI / "tools"
    sys.path.insert(0, str(tools))
    import reindex_kernel_wiki as reindex

    index = json.loads(C.INDEX_PATH.read_text(encoding="utf-8"))
    abs_path = str(C.KERNEL_WIKI / rel_path)
    index["records"].append(reindex.entry_for(abs_path, record))
    index["records"].sort(key=lambda e: e["id"])
    index["count"] = len(index["records"])
    C.INDEX_PATH.write_text(
        json.dumps(index, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def insert_record(record: dict) -> str:
    scope = record["retrieval"]["scope"]
    rtype = record["type"]
    vendor = scope.get("vendor") or "nvidia"
    arch = scope.get("arch") or "blackwell"
    dsl = scope.get("dsl") or "any"
    family = scope.get("operator_family") or "misc"
    rid = record["id"]

    rel_dir = os.path.join("records", rtype, vendor, arch, dsl, family)
    abs_dir = C.KERNEL_WIKI / rel_dir
    abs_dir.mkdir(parents=True, exist_ok=True)
    dest = abs_dir / (rid + ".json")
    # An id collision means the store already holds knowledge under this identity.
    # Overwriting it would delete a record to make room for one the caller
    # believed was new, so refuse: either the incoming is a rediscovery (confirm)
    # or its id was minted wrong upstream. The index is consulted too, because the
    # same id under a different scope lands in a different directory and would
    # otherwise pass here and only be caught later by the store-wide ids gate.
    existing = next((e["path"] for e in load_index() if e.get("id") == rid), None)
    if dest.exists() or existing:
        where = existing or str(dest.relative_to(C.KERNEL_WIKI))
        raise ValueError(
            "id %s already exists at %s; insert would overwrite or duplicate it. "
            "Use --commit confirm --target %s if this is a rediscovery, otherwise "
            "fix the incoming id." % (rid, where, rid))
    dest.write_text(json.dumps(record, ensure_ascii=False, indent=1) + "\n",
                    encoding="utf-8")
    rel_path = str(dest.relative_to(C.KERNEL_WIKI))
    append_to_index(record, rel_path)
    return rel_path


def confirm_record(record_id: str, incoming_id: str) -> str:
    """Directly increment verified_effective on the target record.

    Idempotent: the (record_id, incoming_id) pair is stored in
    worth.track.confirm_keys so the same incoming cannot bump the counter twice.
    """
    entries = load_index()
    target_path_rel = next((e["path"] for e in entries if e.get("id") == record_id), None)
    if not target_path_rel:
        return f"confirm:WARN record {record_id} not found in index"

    full_path = C.KERNEL_WIKI / target_path_rel
    if not full_path.is_file():
        return f"confirm:WARN file not found: {target_path_rel}"

    record = json.loads(full_path.read_text(encoding="utf-8"))
    track = record.setdefault("worth", {}).setdefault("track", {})
    counters = track.setdefault("counters", {})
    confirm_keys = track.setdefault("confirm_keys", [])

    dedup_key = f"{record_id}:{incoming_id}"
    if dedup_key in confirm_keys:
        return f"confirmed:{record_id} (already counted, skipped)"

    counters["verified_effective"] = counters.get("verified_effective", 0) + 1
    confirm_keys.append(dedup_key)

    full_path.write_text(json.dumps(record, ensure_ascii=False, indent=1) + "\n",
                         encoding="utf-8")
    return f"confirmed:{record_id} (verified_effective={counters['verified_effective']})"


def flag_conflict(incoming: dict, target_id: str) -> str:
    C.CONFLICTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    incoming_id = incoming.get("id", "unknown")
    filename = f"{ts}_{incoming_id}.json"
    dest = C.CONFLICTS_DIR / filename

    # Load the existing record for context
    entries = load_index()
    existing_path = next((e["path"] for e in entries if e.get("id") == target_id), None)
    existing = {}
    if existing_path:
        full_path = C.KERNEL_WIKI / existing_path
        if full_path.is_file():
            existing = json.loads(full_path.read_text())

    existing_payload = existing.get("payload") or {}
    incoming_payload = incoming.get("payload") or {}
    existing_worth = existing.get("worth") or {}
    incoming_worth = incoming.get("worth") or {}

    conflict = {
        "created": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "existing_record_id": target_id,
        "incoming_record": incoming,
        "conflict_reason": (
            f"方向矛盾:原记录 type={existing.get('type')} "
            f"gain={existing_worth.get('gain', {}).get('pct', '?')}%, "
            f"新记录 type={incoming.get('type')} "
            f"gain={incoming_worth.get('gain', {}).get('pct', '?')}%"
        ),
        "context": {
            "existing_shape_contract": existing_payload.get("problem", {}).get("shape_contract"),
            "incoming_shape_contract": incoming_payload.get("problem", {}).get("shape_contract"),
            "existing_bottleneck": existing_payload.get("problem", {}).get("bottleneck"),
            "incoming_bottleneck": incoming_payload.get("problem", {}).get("bottleneck"),
            "existing_observed_symptom": existing_payload.get("problem", {}).get("observed_symptom"),
            "incoming_observed_symptom": incoming_payload.get("problem", {}).get("observed_symptom"),
            "possible_explanations": [
                "shape 集合不同",
                "前置条件 / builds_on 不同",
                "硬件批次或驱动版本差异",
                "其中一方的实测有误",
            ],
        },
        "resolution": None,
    }
    dest.write_text(json.dumps(conflict, ensure_ascii=False, indent=1) + "\n",
                    encoding="utf-8")
    return f"conflict_flagged:{dest.name}"


def mark_source_merged(source_path: Path) -> None:
    """Mark the source record (in the mining skill's staging area) as merged.

    This makes the mining output a staging area: once a record has
    been processed by the gate (inserted, confirmed, or conflict-flagged), it
    is marked so it won't be submitted again.
    """
    if not source_path.is_file():
        return
    record = json.loads(source_path.read_text(encoding="utf-8"))
    record["status"] = "merged"
    record["merged_at"] = datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"
    source_path.write_text(json.dumps(record, ensure_ascii=False, indent=1) + "\n",
                           encoding="utf-8")


def do_commit(record: dict, action: str, target: str | None,
              source_path: Path | None = None) -> tuple[int, str]:
    """Execute the agent's decision. Returns (exit_code, message)."""
    if action == "insert":
        try:
            path = insert_record(record)
        except ValueError as exc:
            return 1, "REFUSED: %s" % exc
        if source_path:
            mark_source_merged(source_path)
        return 0, f"inserted → {path}"

    if not target:
        return 2, "ERROR: --target is required for confirm and conflict actions"

    if action == "confirm":
        msg = confirm_record(target, record.get("id", "unknown"))
        if source_path:
            mark_source_merged(source_path)
        return 0, msg

    if action == "conflict":
        msg = flag_conflict(record, target)
        if source_path:
            mark_source_merged(source_path)
        return 0, msg

    return 2, f"ERROR: unknown action {action!r}"


# ---------------------------------------------------------------- CLI

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True,
                        help="Path to the incoming record JSON.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--match", action="store_true",
                       help="Step 1: validate and return scoped candidates.")
    group.add_argument("--commit", choices=["insert", "confirm", "conflict"],
                       help="Step 2: execute the decision.")
    parser.add_argument("--target", default=None,
                        help="For confirm/conflict: the existing record id.")
    args = parser.parse_args()

    path = Path(args.input)
    if not path.is_file():
        print(f"ERROR: file not found: {path}", file=sys.stderr)
        return 2

    record = json.loads(path.read_text(encoding="utf-8"))

    if args.match:
        result = do_match(record)
        print(json.dumps(result, ensure_ascii=False))
        return 1 if result["status"] == "schema_error" else 0

    # --commit
    errors = validate_schema(record)
    if errors:
        print("SCHEMA ERRORS:", file=sys.stderr)
        for e in errors[:10]:
            print(f"  - {e}", file=sys.stderr)
        return 1

    code, msg = do_commit(record, args.commit, args.target, source_path=path)
    print(msg)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
