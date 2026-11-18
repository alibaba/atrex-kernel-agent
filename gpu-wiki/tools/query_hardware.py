#!/usr/bin/env python3
"""Exact lookup over the hardware-facts store.

This is deliberately NOT the experience wiki's retrieval tool. Hardware facts
are vendor-defined, so a query has a right answer or none at all:

  * no ranking -- there is no "more important" peak-FLOPS number;
  * no random fallback -- answering "what is this part's BF16 peak" with an
    unrelated sample would be worse than answering nothing, because the number
    is used as a roofline denominator;
  * fail-loud on every unknown address -- an unknown product, mnemonic, feature
    or field name is an error listing the known values, never a near match;
  * a declared-missing field returns its "how to obtain it instead" note, so a
    caller learns to query the device rather than invent a constant;
  * a recognized part with no spec sheet here is NOT the same failure as a typo.
    The caller has been told to source every roofline number from this store and
    forbidden to fabricate, so a bare error leaves it choosing between inventing
    a number and borrowing another part's -- both of which silently rescale every
    utilization figure computed afterwards. Such a part gets a disposition
    instead: what is missing, what not to do, and how to obtain each class of
    number legitimately.

Self-contained: standard library only, reads this store's records/index.json.

Usage:
    python3 tools/query_hardware.py --product b300
    python3 tools/query_hardware.py --product b300 --field peak_compute.bf16.dense
    python3 tools/query_hardware.py --product b300 --vs b200
    python3 tools/query_hardware.py --product h20      # not recorded: returns a procedure
    python3 tools/query_hardware.py --instruction tcgen05.ld.red
    python3 tools/query_hardware.py --feature fp4-k96-2cta
    python3 tools/query_hardware.py --capability sm_103 --list instructions
    python3 tools/query_hardware.py --list products
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from hardware_identity import (HARDWARE_IDENTITIES, PRODUCT_ARCH,
                               normalize_product_name)

STORE = Path(__file__).resolve().parent.parent / "hardware_wiki"
SCHEMA_VERSION = "hw-1.0"
NO_STORE_EXIT = 3
LISTABLE = ("products", "instructions", "features", "arches", "capabilities")


def die(msg: str, code: int = 2) -> "NoReturn":  # noqa: F821
    print("ERROR %s" % msg, file=sys.stderr)
    raise SystemExit(code)


def load_index(store: Path) -> dict:
    path = store / "records" / "index.json"
    if not path.is_file():
        die("missing-store no index at %s" % path, NO_STORE_EXIT)
    try:
        index = json.loads(path.read_text())
    except Exception as exc:                                   # noqa: BLE001
        die("unreadable-store %s" % exc, NO_STORE_EXIT)
    if index.get("schema") != SCHEMA_VERSION:
        die("unsupported-schema %r (want %s)" % (index.get("schema"), SCHEMA_VERSION),
            NO_STORE_EXIT)
    return index


def known(entries: list[dict], key: str) -> list:
    """Sorted set of an addressable dimension, for fail-loud error messages."""
    if key == "capabilities":
        out = set()
        for e in entries:
            out.update(e.get("available_sm_arch") or [])
            if e.get("sm_arch"):
                out.add(e["sm_arch"])
        return sorted(out)
    field = {"products": "product", "instructions": "mnemonic",
             "features": "feature", "arches": "arch"}[key]
    return sorted({e[field] for e in entries if e.get(field)})


def read_record(store: Path, entry: dict) -> dict:
    return json.loads((store / entry["path"]).read_text())


# ------------------------------------------------- parts with no spec sheet here

# How to obtain a number this store does not hold, per class of number. Handing
# the caller a procedure is the only answer that neither fabricates nor stalls.
OBTAIN = {
    "memory_and_counts": "query the device at runtime "
                         "(torch.cuda.get_device_properties / "
                         "cudaDeviceGetAttribute / hipDeviceGetAttribute); the "
                         "driver reports these, so no datasheet is needed",
    "peak_compute": "read the vendor datasheet for THIS exact part, or measure a "
                    "dense back-to-back MMA microbenchmark and label the result "
                    "as measured rather than published",
    "clocks": "read the achieved clock at runtime under load; a datasheet boost "
              "clock is not what a sustained kernel sees",
}

NOT_RECORDED_EXIT = 4


def not_recorded(product: str, entries: list[dict]) -> int:
    """Answer a real-but-unrecorded part with a procedure, not a dead end."""
    arch = PRODUCT_ARCH[product]
    has_other = any(e.get("arch") == arch and e["type"] != "spec-sheet"
                    for e in entries)
    print(json.dumps({
        "product": product,
        "status": "not-recorded",
        "architecture": arch,
        "recorded_products": known([e for e in entries
                                    if e["type"] == "spec-sheet"], "products"),
        "do_not": "do not substitute another part's numbers and do not invent "
                  "one. A wrong peak silently rescales every utilization figure "
                  "computed afterwards, and that error is invisible in the "
                  "result.",
        "obtain_instead": OBTAIN,
        "still_available_for_this_architecture": (
            "instruction and arch-feature records exist for %s — query them with "
            "--instruction / --feature / --list features --arch %s" % (arch, arch)
        ) if has_other else None,
    }, ensure_ascii=False))
    print("not-recorded %s (%s): no spec sheet here; follow the returned "
          "obtain_instead procedure" % (product, arch), file=sys.stderr)
    return NOT_RECORDED_EXIT


def pick(entries: list[dict], field: str, value: str, label: str) -> dict:
    hits = [e for e in entries if e.get(field) == value]
    if not hits:
        die("unknown-%s %r (known: %s)" % (label, value,
            ", ".join(known(entries, label + "s")) or "none"))
    if len(hits) > 1:
        die("ambiguous-%s %r matches %d records: %s" % (
            label, value, len(hits), ", ".join(h["id"] for h in hits)))
    return hits[0]


# ------------------------------------------------------------------ field paths

def resolve_field(facts: dict, dotted: str) -> tuple:
    """Resolve a dotted path, treating peak_compute as a dict keyed by dtype.

    Returns (value, provenance) so a number never travels without its evidence
    class -- a published peak and a third-party guess must not look alike.
    """
    parts = dotted.split(".")
    if parts[0] == "peak_compute":
        if len(parts) < 2:
            return facts["peak_compute"], None
        dtype = parts[1]
        rows = {row["dtype"]: row for row in facts["peak_compute"]}
        if dtype not in rows:
            die("unknown-dtype %r (known: %s)" % (dtype, ", ".join(sorted(rows))))
        row = rows[dtype]
        if len(parts) == 2:
            return row, row.get("provenance")
        leaf = parts[2]
        if leaf not in row:
            die("unknown-field %r on peak_compute.%s (known: %s)"
                % (leaf, dtype, ", ".join(sorted(row))))
        return row[leaf], row.get("provenance")

    node = facts
    prov = None
    for i, part in enumerate(parts):
        if not isinstance(node, dict) or part not in node:
            where = ".".join(parts[:i]) or "facts"
            options = ", ".join(sorted(node)) if isinstance(node, dict) else "<not a group>"
            die("unknown-field %r under %s (known: %s)" % (part, where, options))
        if isinstance(node, dict) and "provenance" in node:
            prov = node["provenance"]
        node = node[part]
    if isinstance(node, dict) and "provenance" in node:
        prov = node["provenance"]
    # a per-field override beats the group default
    if len(parts) >= 2:
        parent = facts
        for part in parts[:-1]:
            parent = parent.get(part, {}) if isinstance(parent, dict) else {}
        if isinstance(parent, dict):
            prov = (parent.get("provenance_overrides") or {}).get(parts[-1], prov)
    return node, prov


def field_answer(record: dict, dotted: str) -> dict:
    facts = record["facts"]
    value, prov = resolve_field(facts, dotted)
    answer = {"id": record["id"], "field": dotted, "value": value,
              "provenance": prov or record["provenance"]["evidence_class"]}
    if value is None:
        note = (facts.get("unavailable") or {}).get(dotted.split(".")[-1])
        answer["unavailable"] = note or ("Not recorded. Do not substitute a value "
                                         "from another part; read it from the device.")
    return answer


# ------------------------------------------------------------------------- main

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", default=None, help="Store root (default: this repo).")
    ap.add_argument("--product", default=None, help="Look up one part's spec sheet.")
    ap.add_argument("--field", default=None,
                    help="Dotted path into the spec sheet, e.g. peak_compute.bf16.dense.")
    ap.add_argument("--vs", default=None, help="With --product: the recorded deltas "
                                              "against this other part.")
    ap.add_argument("--instruction", default=None, help="Look up one ISA instruction.")
    ap.add_argument("--feature", default=None, help="Look up one architectural feature.")
    ap.add_argument("--capability", default=None,
                    help="Restrict --list to what exists on this sm_arch.")
    ap.add_argument("--arch", default=None, help="Restrict --list to this architecture.")
    ap.add_argument("--list", dest="list_what", choices=LISTABLE, default=None,
                    help="Enumerate an addressable dimension.")
    ap.add_argument("--human", action="store_true",
                    help="Readable text instead of json.")
    return ap


def scope_entries(entries: list[dict], args) -> list[dict]:
    out = entries
    if args.arch:
        if args.arch not in known(entries, "arches"):
            die("unknown-arch %r (known: %s)" % (args.arch,
                ", ".join(known(entries, "arches"))))
        out = [e for e in out if e["arch"] == args.arch]
    if args.capability:
        caps = known(entries, "capabilities")
        if args.capability not in caps:
            die("unknown-capability %r (known: %s)" % (args.capability, ", ".join(caps)))
        out = [e for e in out
               if args.capability in (e.get("available_sm_arch") or [])
               or e.get("sm_arch") == args.capability]
    return out


def emit(payload, human: bool) -> int:
    if human:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(payload, ensure_ascii=False))
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    store = Path(args.store).resolve() if args.store else STORE
    entries = load_index(store)["records"]

    if args.field and not args.product:
        die("--field needs --product: a field path is only meaningful on a spec sheet")
    if args.vs and not args.product:
        die("--vs needs --product")

    asked = [bool(args.product), bool(args.instruction), bool(args.feature),
             bool(args.list_what)]
    if sum(asked) == 0:
        die("nothing asked: give --product / --instruction / --feature / --list")
    if sum(asked) > 1:
        die("ask one thing at a time: --product, --instruction, --feature and "
            "--list are mutually exclusive")

    if args.list_what:
        pool = scope_entries(entries, args)
        print("scope: %d/%d records" % (len(pool), len(entries)), file=sys.stderr)
        answer = {"list": args.list_what, "values": known(pool, args.list_what)}
        if args.list_what == "products":
            visible_arches = ({args.arch} if args.arch else
                              {row["arch"] for row in HARDWARE_IDENTITIES.values()})
            answer["normalization"] = {
                "case_sensitive": False,
                "ignored_separators": ["space", "-", "_"],
                "ignored_wrappers": ["NVIDIA", "AMD", "GeForce", "Instinct",
                                     "GPU", "accelerator"],
                "identity_translation": False,
            }
            answer["recognized_without_spec_sheet"] = sorted(
                product for product, identity in HARDWARE_IDENTITIES.items()
                if not identity["recorded"] and identity["arch"] in visible_arches
            )
            # A spec sheet with no product name is unreachable by --product, so an
            # empty list next to a non-zero scope count reads as "nothing here"
            # when the truth is "something here that cannot be addressed".
            orphans = [e["id"] for e in pool
                       if e["type"] == "spec-sheet" and not e.get("product")]
            if orphans:
                answer["unaddressable_spec_sheets"] = {
                    "count": len(orphans), "ids": orphans,
                    "why": "these records carry no product name, so --product "
                           "cannot reach them and their numbers must not be "
                           "assumed to describe any particular part",
                }
        return emit(answer, args.human)

    if args.product:
        product = normalize_product_name(args.product)
        sheets = [e for e in entries if e["type"] == "spec-sheet"]
        if product not in {e.get("product") for e in sheets} and product in PRODUCT_ARCH:
            return not_recorded(product, entries)
        entry = pick(sheets, "product", product, "product")
        record = read_record(store, entry)
        print("hit: %s" % record["id"], file=sys.stderr)
        if args.field:
            return emit(field_answer(record, args.field), args.human)
        if args.vs:
            comparison = normalize_product_name(args.vs)
            deltas = [d for d in (record["facts"].get("deltas_vs") or [])
                      if d["product"] == comparison]
            if not deltas:
                die("no-recorded-deltas %s has no recorded comparison against %r "
                    "(recorded: %s)" % (record["id"], args.vs,
                    ", ".join(d["product"] for d in
                              (record["facts"].get("deltas_vs") or [])) or "none"))
            return emit({"id": record["id"], "vs": comparison,
                         "changes": deltas[0]["changes"]}, args.human)
        return emit(record, args.human)

    if args.instruction:
        entry = pick([e for e in entries if e["type"] == "instruction"],
                     "mnemonic", args.instruction, "instruction")
        print("hit: %s" % entry["id"], file=sys.stderr)
        return emit(read_record(store, entry), args.human)

    entry = pick([e for e in entries if e["type"] == "arch-feature"],
                 "feature", args.feature, "feature")
    print("hit: %s" % entry["id"], file=sys.stderr)
    return emit(read_record(store, entry), args.human)


if __name__ == "__main__":
    raise SystemExit(main())
