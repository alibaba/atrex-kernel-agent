#!/usr/bin/env python3
# Copyright 2026 Alibaba Group.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Standalone retrieval over the kernel-experience record store.

Self-contained: reads this store's own ``records/index.json`` and record files,
with no dependency on the markdown wiki or any host tool. It applies the design's
retrieval contract:

  * scope is a HARD identity -- vendor / arch / dsl / operator_family / type /
    tier are filtered before any text ranking, and an unknown filter value fails
    CLOSED (a clear error, never a silent empty result), so a caller can never
    mistake "wrong scope" for "no knowledge here". An unknown value is answered
    with the near misses, because a caller guessing ``moe`` for
    ``moe-expert-compute`` should be corrected, not stonewalled;
  * the caller does not know this store's architecture names. It knows what the
    runtime told it: a compute capability (``sm_90``), a product (``H20``), a
    gfx id (``gfx942``). ``--arch`` accepts all of those and resolves them to the
    architecture family, so a scoped query cannot fail on a name the caller had
    no way to learn. A name that resolves to a real architecture this store has
    no records for says exactly that, which is different from a typo;
  * an arch-scoped query also returns the SAME VENDOR's architecture-neutral
    records, because knowledge written for "any NVIDIA GPU" applies to a Hopper
    part; each hit is labelled ``match.arch`` = ``exact`` or
    ``architecture-neutral`` so the caller can weigh a neutral claim differently,
    and the other vendor's neutral records are still excluded;
  * ranking blends the text match with ``worth.rank.score`` (how well tried and
    how believable the record is), capped so a popular record can never bury a
    much stronger text match;
  * a zero-match query returns a labelled random sample of the scoped pool, so a
    caller always has something and never mistakes the fallback for a hit. The
    label is part of the ANSWER (``result.kind``), not just a stderr aside: a
    caller that reads stdout alone must still be unable to mistake a sample for
    knowledge. Before falling back it also reports whether this vendor holds
    matching records under a SIBLING architecture, because "nothing for your
    chip" and "nothing at all" demand different next moves. ``--cross-arch``
    then serves them labelled ``other-architecture`` -- a deliberate, visible
    widening, which is what the caller needs instead of dropping ``--arch`` and
    silently crossing the vendor line as well;
  * a scope count cannot tell a caller that its own cell is empty: ``--dsl`` also
    admits records scoped to ``any``, so a healthy-looking pool can contain
    nothing written for this architecture and language pair. ``--coverage``
    decomposes the pool by how each record was reached, so an empty cell is
    visible before any ranking happens;
  * the ``worth`` layer is ENGINE-SIDE ONLY. It drives filtering and ordering
    here but is never served: handing our own verdict ("proven", score 1.0) to a
    consuming model would anchor its judgement instead of letting the evidence
    speak;
  * ``--emit-json`` writes ONE self-describing object to stdout: ``query`` (the
    scope as resolved, so the caller can see what was actually asked),
    ``result`` (kind / counts / budget), and ``records`` keyed by rank position.
    Records are keyed by their stable ids. Each carries only ``source``, ``type``,
    ``applies_to``, ``match`` and the isolated payload. Evidence and retrieval
    metadata are deliberately not served: they remain available to validation and
    ranking code but must not bias the consuming agent.
  * retrieval happens inside a finite context, so volume is part of correctness:
    ``--brief`` drops the verbatim code and ``--max-bytes`` enforces a budget by
    dropping the weakest hits and saying how many it dropped.

The payload already carries the change's code as ``implementation.snippet``, so
there is no source-resolution mode: this tool never reads the markdown wiki.

Usage:
    python3 tools/query_wiki.py --arch sm_90 --dsl triton --symptom launch-overhead
    python3 tools/query_wiki.py --type technique-card --explain
    python3 tools/query_wiki.py --operator moe-expert-compute --type anti-strategy
    python3 tools/query_wiki.py --arch sm_90 --emit-json --brief --limit 5
    python3 tools/query_wiki.py --list-symptoms
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

import hardware_identity

STORE = Path(__file__).resolve().parent.parent / "kernel_wiki"
SCHEMA_VERSION = "clean-1.3"
NO_STORE_EXIT = 3
RANK_MODES = ("blend", "text", "importance")
TIERS = ("proven", "promising", "provisional", "cautionary")
DEFAULT_FALLBACK_RATIO = 0.10
DEFAULT_WEIGHT = 1.0
DEFAULT_LIMIT = 8

# The query vocabulary must describe every DSL the orchestrator can dispatch,
# even when a particular store currently has only ``dsl:any`` records for one
# of them.  Otherwise a supported campaign is rejected before those portable
# records can be considered.
SUPPORTED_DSLS = {
    "cuda", "cutedsl", "flydsl", "gluon", "triton", "tilelang", "aiter", "any",
}

FALLBACK_NOTE = ("NOT query matches: nothing in scope matched, so this is a "
                 "random sample of the scoped pool, offered only to show what "
                 "kind of knowledge exists here. Do not treat it as advice "
                 "about the queried problem.")

# retrieval halves that are engine-side only and never served to the agent
ENGINE_RETRIEVAL_KEYS = ("locator", "links")

# The caller never sees this store's architecture names -- it sees what the
# runtime reported. Every token below is a publicly documented name for an
# architecture family, so a query can be addressed in the caller's own
# vocabulary instead of failing on a name it had no way to learn.
ARCH_ALIASES = {
    "sm_80": "ampere", "sm80": "ampere", "sm_86": "ampere",
    "8.0": "ampere", "8.6": "ampere", "a100": "ampere",
    # Ada is its own family: sm_89 parts (L20, L40S, L4) are not Ampere.
    "sm_89": "ada", "sm89": "ada", "8.9": "ada",
    "l20": "ada", "l40s": "ada", "l4": "ada",
    "sm_90": "hopper", "sm90": "hopper", "sm_90a": "hopper", "9.0": "hopper",
    "h100": "hopper", "h200": "hopper", "h800": "hopper", "h20": "hopper",
    "sm_100": "blackwell", "sm100": "blackwell", "sm_100a": "blackwell",
    "10.0": "blackwell", "b200": "blackwell", "gb200": "blackwell",
    "sm_103": "blackwell-ultra", "sm103": "blackwell-ultra",
    "sm_103a": "blackwell-ultra", "10.3": "blackwell-ultra",
    "b300": "blackwell-ultra", "gb300": "blackwell-ultra",
    "sm_120": "blackwell-geforce", "sm120": "blackwell-geforce",
    "sm_120a": "blackwell-geforce", "12.0": "blackwell-geforce",
    "gfx942": "cdna3", "mi300x": "cdna3", "mi300a": "cdna3", "mi308x": "cdna3",
    "gfx950": "cdna4", "mi350x": "cdna4", "mi355x": "cdna4",
    "gfx1200": "rdna4", "gfx1201": "rdna4",
    "any": "generic", "neutral": "generic", "architecture-neutral": "generic",
}
# Product spellings are owned by one shared table. Keeping this update here
# makes kernel scoping and exact hardware lookup agree on every known SKU.
ARCH_ALIASES.update(hardware_identity.PRODUCT_ARCH)
# A superseded spelling resolves to whatever its canonical name addresses, so an
# older campaign config keeps scoping correctly after a rename.
ARCH_ALIASES.update({
    old: hardware_identity.PRODUCT_ARCH[new]
    for old, new in hardware_identity.LEGACY_SPELLINGS.items()
})

# Which vendor an architecture belongs to. Needed because "architecture-neutral"
# is neutral WITHIN a vendor: a record written for any CDNA part is not advice
# about a Hopper part, so neutral recall must never cross this line.
ARCH_VENDOR = {
    "ampere": "nvidia", "ada": "nvidia", "hopper": "nvidia", "blackwell": "nvidia",
    "blackwell-ultra": "nvidia", "blackwell-geforce": "nvidia",
    "cdna3": "amd", "cdna4": "amd", "rdna4": "amd",
    "zwm890p": "ppu",
}
NEUTRAL_ARCH = "generic"

# An exact-architecture record outranks a neutral one on an otherwise equal
# match: both may be true, but only one was written about this chip.
NEUTRAL_RANK_DISCOUNT = 0.85

# ``dsl:any`` keeps portable knowledge reachable from a language-scoped query,
# but it is weaker evidence than a record written for the requested DSL. Without
# this discount a slightly higher seed importance can fill the whole top-k with
# language-neutral records, even when exact-DSL matches have the same text hit.
NEUTRAL_DSL_RANK_DISCOUNT = 0.85

# A sibling architecture's record, discounted harder still. Scheduling-level
# knowledge often ports across a vendor's architectures, so refusing to surface
# it leaves the caller choosing between missing it and dropping --arch, which
# hides the vendor boundary too. Surfacing it LABELLED is the honest option.
CROSS_ARCH_RANK_DISCOUNT = 0.6


def suggest(value: str, allowed: set) -> str:
    """Near misses for a rejected scope value, as a correction hint."""
    import difflib
    near = [a for a in sorted(allowed) if value in a or a in value]
    for a in difflib.get_close_matches(value, sorted(allowed), n=5, cutoff=0.6):
        if a not in near:
            near.append(a)
    return ", ".join(near[:8])


def resolve_arch(value: str, allowed: set) -> str:
    """Map a runtime architecture token onto this store's architecture name.

    Fails closed, but distinguishes the two failures that matter: a token that
    means nothing here, versus a real architecture this store simply has no
    records for. The caller must not read the second as a bad query.
    """
    token = value.strip().lower()
    if token in allowed:
        return token
    product = hardware_identity.normalize_product_name(value)
    family = (ARCH_ALIASES.get(token)
              or ARCH_ALIASES.get(token.replace("sm_", "sm"))
              or hardware_identity.PRODUCT_ARCH.get(product))
    if family and family in allowed:
        return family
    if family:
        die("no-records-for-arch %r resolves to %r, which this store has no "
            "records for (has: %s). This is a gap in the store, not a bad query: "
            "do not retry with a different architecture, and do not treat another "
            "architecture's records as applying here."
            % (value, family, ", ".join(sorted(allowed))))
    die("unknown-arch %r (store: %s; also accepts a compute capability such as "
        "sm_90, a product such as h20/b300/mi300x, or a gfx id such as gfx942)"
        % (value, ", ".join(sorted(allowed))))


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


# ------------------------------------------------------------------ vocabulary

def vocab(entries: list[dict]) -> dict[str, set]:
    v: dict[str, set] = {k: set() for k in
                         ("vendor", "arch", "dsl", "family", "operator", "type",
                          "symptom", "tag")}
    for e in entries:
        scope = e["retrieval"]["scope"]
        v["vendor"].add(scope.get("vendor"))
        v["arch"].add(scope.get("arch"))
        v["dsl"].add(scope.get("dsl"))
        if scope.get("operator_family"):
            v["family"].add(scope["operator_family"])
        for op in scope.get("operators") or []:
            v["operator"].add(op)
        v["type"].add(e["type"])
        for s in e["retrieval"].get("signals", {}).get("symptoms") or []:
            v["symptom"].add(s)
        bn = e["retrieval"].get("signals", {}).get("bottleneck")
        if bn:
            v["symptom"].add(bn)
        for t in e["retrieval"].get("technique_tags") or []:
            v["tag"].add(t)
    v["dsl"].update(SUPPORTED_DSLS)
    # symptoms and tags stay free-form for MATCHING -- a profiler phrase should
    # not have to be spelled the store's way -- but they must be enumerable, or
    # a caller cannot discover the tokens that retrieve well.
    return {k: {x for x in s if x} for k, s in v.items()}


def fold(value: str) -> str:
    """Collapse a token to its letters and digits, for spelling-insensitive match."""
    return re.sub(r"[^a-z0-9]", "", value.strip().lower())


def resolve_vocab(name: str, value: str | None, allowed: set) -> str | None:
    """Accept the caller's spelling of a store token, or fail closed with hints.

    The caller writes the name it knows. A store may spell the same language
    ``cute-dsl`` while every other tool in the system spells it ``cutedsl``, and
    refusing that is a spelling argument, not a scope error. Punctuation and case
    are folded away; anything beyond that still fails closed, because silently
    accepting a near miss would substitute a scope the caller did not ask for.
    """
    if value is None or value in allowed:
        return value
    folded = {fold(a): a for a in allowed}
    hit = folded.get(fold(value))
    if hit is not None:
        return hit
    die(vocab_error(name, value, allowed))


# Beyond this many values, dumping the vocabulary into an error is itself a context
# flood: one mistyped operator family printed 157 tokens at a caller that only needed
# the near misses and a way to search.
VOCAB_DUMP_LIMIT = 25


def vocab_error(name: str, value: str, allowed: set) -> str:
    hint = suggest(value, allowed)
    said = " — did you mean: %s?" % hint if hint else ""
    if len(allowed) > VOCAB_DUMP_LIMIT:
        flag = "--list-%s" % ("operators" if name == "operator" else name)
        return ("unknown-%s %r%s (%d known values; search them with %s --like <substring>)"
                % (name, value, said, len(allowed), flag))
    return "unknown-%s %r%s (known: %s)" % (
        name, value, said, ", ".join(sorted(allowed)) or "none")


def check_vocab(name: str, value: str | None, allowed: set) -> None:
    resolve_vocab(name, value, allowed)


# ---------------------------------------------------------------- text scoring

def normalize_terms(query: list[str]) -> list[str]:
    return [t.lower() for t in re.split(r"\s+", " ".join(query)) if t]


def text_score(entry: dict, terms: list[str], match_any: bool) -> int:
    if not terms:
        return 1
    hay = entry.get("search_text", "")
    hits = sum(1 for t in terms if t in hay)
    if hits == 0 or (not match_any and hits != len(terms)):
        return 0
    return hits


def symptom_match(entry: dict, symptom: str) -> str | None:
    """How a record matches a symptom: as an indexed signal, or only as text.

    The distinction is load-bearing. A record whose recorded bottleneck IS this
    symptom is about the caller's problem; a record that merely mentions the words
    somewhere may be about a different one. Matching them together let an indexed
    symptom return hundreds of confident-looking hits -- one caller asked about
    launch overhead at ONE launch and got advice about fusing many launches.
    """
    sig = entry["retrieval"].get("signals", {})
    if sig.get("bottleneck") == symptom or symptom in (sig.get("symptoms") or []):
        return "signal"
    if symptom.replace("-", " ") in entry.get("search_text", ""):
        return "text"
    return None


# ------------------------------------------------------------- serve projection

CODE_KEYS = ("snippet", "dispatch_snippet")

# What a caller needs in order to decide whether to read a record in full. Brief
# mode serves these and announces the rest: enough to choose, not enough to
# apply. Anything outside this set is omitted BY NAME, never dropped in silence.
BRIEF_KEYS = ("goal", "problem", "what", "when", "verdict", "lesson",
              "attempted", "change", "mechanism", "cost", "established_fact")


def brief_payload(payload: dict) -> dict:
    """The payload trimmed to what supports a read-or-skip decision.

    Trimming only the code was not enough: on a real store the bulk of a record
    is prose, so the flag returned a byte-for-byte identical answer and silently
    did nothing. Every omission is named with its size, so a trimmed record can
    never be mistaken for a thin one.
    """
    out: dict = {}
    omitted: dict[str, int] = {}
    for key, value in payload.items():
        if key in BRIEF_KEYS:
            out[key] = value
        elif key == "implementation" and isinstance(value, dict):
            trimmed = {k: v for k, v in value.items() if k not in CODE_KEYS}
            for k in CODE_KEYS:
                if value.get(k):
                    omitted["implementation." + k] = len(value[k])
            if trimmed:
                out["implementation"] = trimmed
        else:
            omitted[key] = len(json.dumps(value, ensure_ascii=False))
    if omitted:
        out["omitted_by_brief"] = {
            "fields": omitted,
            "how_to_get_them": "re-query this id without --brief",
        }
    return out


def serve_record(entry: dict, store: Path, arch_match: str = "exact",
                 brief: bool = False) -> dict:
    """One lightweight hit with an isolated, unmodified payload.

    Retrieval, worth and evidence remain engine-side.  In particular, evidence is
    intentionally absent because confidence labels can anchor the consuming agent.
    The record id is the key in the enclosing mapping, so it is not duplicated here.
    """
    record = json.loads((store / entry["path"]).read_text())
    retrieval = record["retrieval"]
    payload = record["payload"]
    return {
        "source": "kernel_wiki",
        "type": record["type"],
        "applies_to": retrieval.get("scope", {}),
        "match": {"arch": arch_match},
        "payload": brief_payload(payload) if brief else payload,
    }


# -------------------------------------------------------------------- filtering

def arch_reach(scope: dict, arch: str, strict: bool,
               cross: bool = False) -> str | None:
    """How a record is reached by an arch-scoped query, or None if it is not."""
    rec_arch = scope.get("arch")
    listed = scope.get("architectures") or []
    if rec_arch == arch or arch in listed:
        return "exact"
    if strict:
        return None
    want = ARCH_VENDOR.get(arch)
    vendor = scope.get("vendor")
    if rec_arch == NEUTRAL_ARCH:
        # A neutral record claims to hold for a whole vendor. That is only usable
        # here if it is this vendor's (or nobody's) neutral knowledge.
        if arch == NEUTRAL_ARCH or vendor in (want, "generic", None):
            return "architecture-neutral"
        return None
    # A sibling architecture, only when explicitly asked for, and never across
    # the vendor line: a CDNA record is not advice about a Hopper part.
    if cross and vendor in (want, "generic", None):
        return "other-architecture"
    return None


def scoped(entries, args) -> list[tuple[dict, str]]:
    out = []
    for e in entries:
        scope = e["retrieval"]["scope"]
        reach = "exact"
        if args.arch:
            reach = arch_reach(scope, args.arch, args.strict_arch, args.cross_arch)
            if reach is None:
                continue
        if args.vendor and scope.get("vendor") not in (args.vendor, "generic"):
            continue
        if args.dsl and scope.get("dsl") not in (args.dsl, "any"):
            continue
        if args.family and scope.get("operator_family") != args.family:
            continue
        if args.operator:
            operators = scope.get("operators") or []
            operator_agnostic = not operators and not scope.get("operator_family")
            if args.operator not in operators and not operator_agnostic:
                continue
        if args.type and e["type"] != args.type:
            continue
        if args.level and e["level"] != args.level:
            continue
        if args.status and e.get("status") != args.status:
            continue
        if args.tier and e.get("tier") != args.tier:
            continue
        if args.min_importance is not None and (e.get("worth_score") or 0) < args.min_importance:
            continue
        if args.min_gain is not None and not (
                e.get("gain_pct") is not None and e["gain_pct"] >= args.min_gain):
            continue
        if args.min_speedup is not None and not (
                e.get("speedup_x") is not None and e["speedup_x"] >= args.min_speedup):
            continue
        symptom_strength = None
        if args.symptom:
            symptom_strength = symptom_match(e, args.symptom)
            if symptom_strength is None:
                continue
        if args.exclude and e["id"] in args.exclude:
            continue
        out.append((e, reach, symptom_strength))
    # An indexed signal beats a passing mention. Serving both together buries the
    # records that are actually about this bottleneck, so if any record carries the
    # symptom as a recorded signal, the merely-textual ones are not in scope.
    if args.symptom and any(st == "signal" for _e, _r, st in out):
        out = [x for x in out if x[2] == "signal"]
    return [(e, r) for e, r, _st in out]


def ranked(pool, terms, args) -> list[tuple[float, int, float, dict, str]]:
    scored = []
    for e, reach in pool:
        ts = text_score(e, terms, args.any)
        if ts == 0:
            continue
        imp = e.get("worth_score") or 0.0
        if args.rank == "text":
            key = float(ts)
        elif args.rank == "importance":
            key = imp
        else:
            key = ts * (1.0 + args.weight_importance * imp)
        if reach == "architecture-neutral":
            key *= NEUTRAL_RANK_DISCOUNT
        elif reach == "other-architecture":
            key *= CROSS_ARCH_RANK_DISCOUNT
        scope = e["retrieval"]["scope"]
        if getattr(args, "dsl", None) and scope.get("dsl") == "any":
            key *= NEUTRAL_DSL_RANK_DISCOUNT
        scored.append((key, ts, imp, e, reach))
    scored.sort(key=lambda x: (-x[0], -x[2], x[3]["id"]))
    return scored


SCOPE_FILTERS = ("arch", "dsl", "family", "operator", "type", "level",
                 "status", "tier", "symptom")


def empty_because(entries, args, terms) -> list[dict]:
    """Which filter emptied the result, and what dropping it would return.

    A zero result has two very different meanings -- "this store has no such
    knowledge" and "you combined filters that cannot co-occur" -- and they look
    identical. Callers have mistaken the second for the first and concluded the
    store was empty on a subject it covers well. So when nothing matches, say
    which single filter is responsible.
    """
    out = []
    for name in SCOPE_FILTERS:
        if getattr(args, name, None) is None:
            continue
        probe = argparse.Namespace(**vars(args))
        setattr(probe, name, None)
        n = len(ranked(scoped(entries, probe), terms, probe))
        if n:
            out.append({"drop": "--" + name.replace("_", "-"), "would_match": n})
    if terms:
        n = len(ranked(scoped(entries, args), [], args))
        if n:
            out.append({"drop": "the free-text terms", "would_match": n})
    return sorted(out, key=lambda x: -x["would_match"])


def fallback(pool, ratio, limit, seed) -> list[tuple[dict, str]]:
    if not pool:
        return []
    rng = random.Random(seed)
    n = max(1, min(len(pool), int(len(pool) * ratio) or 1, limit))
    return rng.sample(pool, n)


# ------------------------------------------------------------------------- main

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("query", nargs="*", help="Free-text terms (AND unless --any).")
    ap.add_argument("--json-store", default=None, help="Store root (default: this repo).")
    for name in ("vendor", "arch", "dsl", "family", "operator", "type",
                 "level", "status"):
        ap.add_argument("--" + name, default=None)
    ap.add_argument("--strict-arch", action="store_true",
                    help="Exact architecture only; drop architecture-neutral records.")
    ap.add_argument("--cross-arch", action="store_true",
                    help="Also return this vendor's OTHER architectures, labelled. "
                         "Use when an arch-scoped query finds nothing, instead of "
                         "dropping --arch (which would also cross the vendor line).")
    ap.add_argument("--coverage", action="store_true",
                    help="Report how the scoped pool decomposes, then exit. A pool "
                         "size alone cannot tell you whether your own cell is empty.")
    ap.add_argument("--exclude", default=None,
                    help="Comma-separated record ids to suppress (what you already read).")
    ap.add_argument("--tier", choices=TIERS, default=None)
    ap.add_argument("--symptom", default=None)
    ap.add_argument("--min-importance", type=float, default=None)
    ap.add_argument("--min-gain", type=float, default=None)
    ap.add_argument("--min-speedup", type=float, default=None)
    ap.add_argument("--rank", choices=RANK_MODES, default="blend")
    ap.add_argument("--weight-importance", type=float, default=DEFAULT_WEIGHT)
    ap.add_argument("--any", action="store_true", help="OR the text terms.")
    ap.add_argument("--explain", action="store_true")
    ap.add_argument("--emit-json", action="store_true")
    ap.add_argument("--brief", action="store_true",
                    help="Omit verbatim code from payloads (announced, not silent).")
    ap.add_argument("--max-bytes", type=int, default=None,
                    help="Context budget for --emit-json; weakest hits are dropped.")
    ap.add_argument("--fallback-ratio", type=float, default=DEFAULT_FALLBACK_RATIO)
    ap.add_argument("--no-fallback", dest="fallback", action="store_false", default=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    for name in ("arch", "dsl", "family", "operators", "type", "symptoms", "tags"):
        ap.add_argument("--list-" + name, action="store_true")
    ap.add_argument("--like", default=None,
                    help="Substring filter for any --list-*; a vocabulary of "
                         "hundreds of tokens is not readable as a dump.")
    return ap


def envelope(hits, store, args, pool_size: int, total: int, kind: str) -> dict:
    """The served answer, self-describing so stdout alone cannot mislead.

    ``result.kind`` is the load-bearing field: a caller that pipes stdout and
    ignores stderr must still see that a fallback sample is not an answer.
    """
    records = {e["id"]: serve_record(e, store, reach, args.brief)
               for _k, _ts, _imp, e, reach in hits}
    result = {
        "kind": kind,
        "served": len(records),
        "scoped_pool": pool_size,
        "store_total": total,
        "note": FALLBACK_NOTE if kind == "fallback" else None,
    }
    out = {"query": args.resolved_scope, "result": result, "records": records}
    if args.max_bytes:
        dropped = 0
        while len(json.dumps(out, ensure_ascii=False)) > args.max_bytes and len(records) > 1:
            records.popitem()
            dropped += 1
        if dropped:
            result["dropped_for_budget"] = dropped
            result["served"] = len(records)
        if len(json.dumps(out, ensure_ascii=False)) > args.max_bytes:
            # One record can be bigger than the whole budget. Say so rather than
            # pretend the budget held.
            result["over_budget"] = True
    return out


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.weight_importance < 0:
        die("--weight-importance must not be negative")
    store = Path(args.json_store).resolve() if args.json_store else STORE
    index = load_index(store)
    entries = index["records"]
    vv = vocab(entries)

    for name, key in (("arch", "arch"), ("dsl", "dsl"), ("family", "family"),
                      ("operators", "operator"), ("type", "type"),
                      ("symptoms", "symptom"), ("tags", "tag")):
        if getattr(args, "list_" + name):
            values = sorted(vv[key])
            if args.like:
                needle = args.like.strip().lower()
                values = [v for v in values if needle in v.lower()]
            # Some stores record a signal as a whole sentence rather than a token.
            # Both are matchable, but only one is selectable, so say which is which
            # instead of presenting prose as a pick-list.
            tokens = [v for v in values if " " not in v and len(v) <= 60]
            prose = [v for v in values if v not in tokens]
            try:
                for x in tokens + prose:
                    print(x)
                if prose:
                    print("note: %d of %d entries are free-form text, not "
                          "selectable tokens — match those with free-text terms"
                          % (len(prose), len(values)), file=sys.stderr)
            except BrokenPipeError:      # piped into head; not an error
                pass
            return 0

    # fail-closed scope validation; --arch first, since it is the one filter the
    # caller states in the runtime's vocabulary rather than the store's.
    if args.arch:
        args.arch = resolve_arch(args.arch, vv["arch"])
    args.vendor = resolve_vocab("vendor", args.vendor, vv["vendor"])
    args.dsl = resolve_vocab("dsl", args.dsl, vv["dsl"])
    args.family = resolve_vocab("family", args.family, vv["family"])
    args.operator = resolve_vocab("operator", args.operator, vv["operator"])
    args.type = resolve_vocab("type", args.type, vv["type"])
    args.exclude = {x.strip() for x in args.exclude.split(",") if x.strip()} \
        if args.exclude else set()

    # Symptoms stay free-form on purpose: a profiler phrase should not have to be
    # spelled this store's way. But failing open silently let an out-of-vocabulary
    # token look like a filter that had been applied, so say so instead.
    symptom_note = None
    symptom_alternatives: list[str] = []
    if args.symptom and args.symptom not in vv["symptom"]:
        # Structured, not just prose: a caller that guessed the wrong signal name
        # wants "did you mean one of these" as data it can act on. Free-text
        # degradation is the dangerous case here -- it returns hundreds of
        # confident-looking hits about a different problem.
        hint = suggest(args.symptom, vv["symptom"])
        symptom_alternatives = [a for a in hint.split(", ") if a][:8]
        symptom_note = ("%r is not an indexed symptom; it matched as free text "
                        "only, so hits may be about a different problem%s"
                        % (args.symptom,
                           ". Closest indexed signals: %s" % hint if hint else
                           ". Run --list-symptoms for the indexed set."))
        print("warning: %s" % symptom_note, file=sys.stderr)

    if args.coverage:
        by_reach: dict[str, int] = {}
        by_type: dict[str, int] = {}
        cov = scoped(entries, args)
        for e, reach in cov:
            by_reach[reach] = by_reach.get(reach, 0) + 1
            by_type[e["type"]] = by_type.get(e["type"], 0) + 1
        print(json.dumps({
            "scoped_pool": len(cov), "store_total": len(entries),
            "by_reach": by_reach, "by_type": by_type,
            "reading": "by_reach.exact is the only count written about this exact "
                       "architecture. A pool that is entirely architecture-neutral "
                       "means your own cell is empty even though the pool is not.",
        }, ensure_ascii=False))
        return 0

    pool = scoped(entries, args)
    symptom_mode = None
    if args.symptom:
        symptom_mode = "signal" if any(
            symptom_match(e, args.symptom) == "signal" for e, _r in pool) else "text"
        if symptom_mode == "text":
            print("note: no record carries %r as a recorded signal; matched on text "
                  "only, so these may be about a different problem"
                  % args.symptom, file=sys.stderr)
    terms = normalize_terms(args.query)
    args.resolved_scope = {
        "terms": terms,
        "arch": args.arch, "vendor": args.vendor, "dsl": args.dsl,
        "family": args.family, "operator": args.operator, "type": args.type,
        "tier": args.tier, "symptom": args.symptom,
        "architecture_neutral_included": bool(args.arch) and not args.strict_arch,
        "other_architectures_included": bool(args.arch) and args.cross_arch,
        "vendor_gate": ARCH_VENDOR.get(args.arch) if args.arch and not args.vendor else None,
        "symptom_note": symptom_note,
        "symptom_alternatives": symptom_alternatives,
        "symptom_match_mode": symptom_mode,
    }

    filters = [f"{k}={v}" for k, v in (
        ("vendor", args.vendor), ("arch", args.arch), ("dsl", args.dsl),
        ("family", args.family), ("operator", args.operator), ("type", args.type),
        ("tier", args.tier), ("symptom", args.symptom)) if v]
    for k, v in (("min-importance", args.min_importance), ("min-gain", args.min_gain),
                 ("min-speedup", args.min_speedup)):
        if v is not None:
            filters.append("%s=%g" % (k, v))
    # Commentary on stderr: stdout is the machine-readable channel.
    print("scope: %s — %d/%d records" % (
        "; ".join(filters) if filters else "no filter", len(pool), len(entries)),
        file=sys.stderr)

    hits = ranked(pool, terms, args)[:args.limit]
    kind = "matches"
    elsewhere = None
    over_constrained = None
    if not hits:
        kind = "empty"
        over_constrained = empty_because(entries, args, terms)
        if over_constrained:
            best = over_constrained[0]
            print("empty because the scope is over-constrained: dropping %s would "
                  "match %d record(s). This store is NOT empty on this subject."
                  % (best["drop"], best["would_match"]), file=sys.stderr)
        if args.arch and not args.cross_arch and not args.strict_arch:
            # "Nothing for your architecture" is not "nothing at all". Saying which
            # lets the caller widen deliberately instead of dropping the filter and
            # silently crossing the vendor line as well.
            widened = argparse.Namespace(**vars(args))
            widened.cross_arch = True
            n = len(ranked(scoped(entries, widened), terms, widened))
            if n:
                elsewhere = {
                    "matches_on_other_architectures_of_this_vendor": n,
                    "how_to_reach_them": "re-run with --cross-arch; those hits are "
                                         "labelled match.arch=other-architecture",
                }
                print("no match on %s, but %d match under this vendor's other "
                      "architectures — re-run with --cross-arch (do NOT drop "
                      "--arch)" % (args.arch, n), file=sys.stderr)
        if args.fallback:
            sample = fallback(pool, args.fallback_ratio, args.limit, args.seed)
            if sample:
                kind = "fallback"
                print("no match — random %d%% fallback of the scoped pool (%d "
                      "record(s)); these are NOT query matches"
                      % (int(args.fallback_ratio * 100), len(sample)), file=sys.stderr)
                hits = [(0.0, 0, e.get("worth_score") or 0.0, e, reach)
                        for e, reach in sample]

    if args.emit_json:
        doc = envelope(hits, store, args, len(pool), len(entries), kind)
        if elsewhere:
            doc["result"]["available_elsewhere"] = elsewhere
        if over_constrained:
            doc["result"]["empty_because"] = {
                "diagnosis": "over-constrained scope, not an empty store",
                "single_filter_removals_that_would_match": over_constrained,
            }
        print(json.dumps(doc, ensure_ascii=False))
        return 0

    # The label belongs on the channel the caller reads. In --emit-json that is
    # result.kind; here it is this banner, or a zero-match query would print
    # nothing at all and an empty answer would look like a clean one.
    if kind == "fallback":
        print("  !! NOT MATCHES — random sample of the scoped pool; nothing "
              "matched your query. Do not act on these.")
    elif kind == "empty":
        print("  (no records in scope)")
        for item in (over_constrained or [])[:3]:
            print("  -> dropping %s would match %d record(s)"
                  % (item["drop"], item["would_match"]))
        if elsewhere:
            print("  -> %d match under this vendor's other architectures; "
                  "re-run with --cross-arch"
                  % elsewhere["matches_on_other_architectures_of_this_vendor"])
    for key, ts, imp, e, reach in hits:
        extra = "; text=%d; blend=%.2f" % (ts, key) if args.explain else ""
        if args.explain:
            extra += "; type=%s; id=%s" % (e["type"], e["id"])
        neutral = " [architecture-neutral]" if reach == "architecture-neutral" else ""
        print("  [imp=%.3f %s%s] %s%s — %s" % (
            imp, e.get("tier"), extra, e["path"], neutral, e.get("title", "")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
