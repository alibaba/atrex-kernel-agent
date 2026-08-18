#!/usr/bin/env python3
"""Gate the staging store.

Two layers of checking, and neither substitutes for the other.

This repository's own `tools/check_kernel_wiki.py` is invoked as-is against the
staging root, so all nine of its gates apply -- schema, ids, anonymization,
raw-isolation, relations, index, self-contained, no-cross-reference,
established-fact. Copying those checks here would let the copy drift out of step
with the store they protect, and the drift would only surface as records that
pass locally and are refused by `wiki-gate`.

On top of that, seven gates only this pipeline can run, because only it has the
trace and the packets:

  profile          the record satisfies opt-trace-1.0, this corpus's narrowing of
                   clean-1.3 (see make_schema.py PATCHES)
  layout           the directory equals the record's own scope, the same way
                   wiki-gate derives it on insert. The store's ids gate checks
                   the filename but not the path
  verbatim         a snippet is a substring of the code it claims to quote
  no-fabrication   every number the record states appears in its packet
  provenance       evidence.raw names the trace and a commit that resolves in it
  ncu-attribution  a profiler-backed bottleneck cites a capture that actually
                   measured the kernel under test
  store-overlap    the id is free in the live store, so wiki-gate can insert it

Never weaken a gate to make records pass. When a gate looks wrong, prove it fires:
`--injection-tests` mutates a copy of a record and asserts that the named gate
complains. A gate without an injection test is how a store ends up falsely green.

Usage:
  RTM_TRACE=<trace> python3 validate_store.py [--verbose]
  RTM_TRACE=<trace> python3 validate_store.py --injection-tests
"""
import argparse
import copy
import json
import re
import subprocess
import sys

import config as c
import recon

# A number is auditable only when it is not glued to the right of a letter or a
# dot: that skips the digits inside identifiers and versions (`float32`, `f16`,
# `B200`, `sm_90`) while still catching real values such as 59.14 or 2.42.
NUM_RE = re.compile(r"(?<![A-Za-z0-9.])-?\d+(?:\.\d+)?")

# Where a number in a record may not be invented. payload prose is included: a
# mechanism sentence that states "halves write bandwidth (4B vs 8B)" is making a
# factual claim about the code.
AUDITED_PATHS = (
    ("worth", "gain"),
    ("evidence", "summary"),
    ("payload",),
    ("retrieval", "signals", "metrics"),
)

CODEISH_KEYS = {"snippet", "dispatch_snippet", "attempted_code"}


def norm_ws(s):
    return re.sub(r"\s+", " ", s or "").strip()


def mags(text):
    """Magnitude strings in `text`, sign dropped.

    This gate asks whether a magnitude was invented; the sign is a direction and
    a different question. Dropping it removes a class of false positives where a
    record writes -0.7% while the source states 0.7 without the signed token.
    Both forms of an integer are recorded so 5 and 5.0 match either way.
    """
    out = set()
    for tok in NUM_RE.findall(text or ""):
        try:
            v = abs(float(tok))
        except ValueError:
            continue
        out.add("%g" % v)
        if v == int(v):
            out.add(str(int(v)))
    return out


def load_records(store):
    out = []
    for f in sorted((store / "records").rglob("*.json")):
        if f.name == "index.json":
            continue
        try:
            out.append((f, json.loads(f.read_text())))
        except json.JSONDecodeError as e:
            out.append((f, {"__parse_error__": str(e)}))
    return out


def load_packets():
    """seg_id -> (packet, its code text)."""
    out = {}
    for f in sorted(c.PACKETS.glob("*.json")):
        try:
            p = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        code = ""
        for ext in (".diff", ".py"):
            sib = f.with_suffix(ext)
            if sib.is_file():
                code += sib.read_text(errors="replace")
        out[p["seg_id"]] = (p, code)
    return out


def index_packets(packets):
    """Index packets by the id they targeted AND by (trace, version).

    An agent sometimes rewrites the id the packet specified -- a different
    technique slug or suffix number -- but `evidence.raw.version` always matches
    `provenance.version`, because both are copied from the same source. The
    fallback keeps every record auditable even when the id was renamed.
    """
    by_id = {p["target"]["id"]: (p, code) for p, code in packets.values()}
    by_ver = {}
    for p, code in packets.values():
        prov = p.get("provenance") or {}
        key = (prov.get("source_repo"), prov.get("version"))
        if key[1]:
            by_ver[key] = (p, code)
    return by_id, by_ver


def walk_strings(node, in_code=False):
    """Yield (text, in_code) for every string under `node`."""
    if isinstance(node, dict):
        for k, v in node.items():
            yield from walk_strings(v, in_code or k in CODEISH_KEYS)
    elif isinstance(node, list):
        for v in node:
            yield from walk_strings(v, in_code)
    elif isinstance(node, str):
        yield node, in_code
    elif isinstance(node, (int, float)) and not isinstance(node, bool):
        yield str(node), in_code


def dig(record, path):
    node = record
    for key in path:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node


def packet_of(record, ctx):
    entry = ctx["by_id"].get(record.get("id"))
    if entry:
        return entry
    raw = dig(record, ("evidence", "raw")) or {}
    return ctx["by_ver"].get((raw.get("source_repo"), raw.get("version")))


# ------------------------------------------------------------------- gates

def gate_profile(ctx, fails):
    """Every record satisfies opt-trace-1.0."""
    try:
        import jsonschema
    except ImportError:
        return "SKIP: jsonschema not installed"
    if not c.PROFILE_SCHEMA.is_file():
        fails.append("profile: %s is missing; run make_schema.py"
                     % c.PROFILE_SCHEMA.name)
        return "no profile"
    validator = jsonschema.Draft202012Validator(
        json.loads(c.PROFILE_SCHEMA.read_text()))
    ok = 0
    for f, r in ctx["records"]:
        errors = sorted(validator.iter_errors(r), key=lambda e: list(e.path))
        if not errors:
            ok += 1
        for err in errors[:4]:
            where = "/".join(str(p) for p in err.path) or "<root>"
            fails.append("%s: profile: %s: %s"
                         % (f.name, where, err.message[:140]))
    return "%d records satisfy %s" % (ok, c.PROFILE_NAME)


def gate_layout(ctx, fails):
    """The directory must equal the record's own scope.

    Layout is records/<type>/<vendor>/<arch>/<dsl>/<operator_family>/, derived
    from the fields the engine filters on, which makes a misfiled record a
    mechanical error rather than something a reviewer has to notice. The
    expression is copied from `wiki-gate`'s insert path on purpose: staging has
    to agree with where the record will finally land.
    """
    root = ctx["store"] / "records"
    ok = 0
    for f, r in ctx["records"]:
        scope = (r.get("retrieval") or {}).get("scope") or {}
        want = (root / str(r.get("type") or "?") / str(scope.get("vendor") or "?")
                / str(scope.get("arch") or "?") / str(scope.get("dsl") or "any")
                / str(scope.get("operator_family") or "misc"))
        if f.parent != want:
            fails.append("%s: layout: filed under %s but scope says %s"
                         % (f.name, f.parent.relative_to(root),
                            want.relative_to(root)))
        else:
            ok += 1
    return "%d records in the right cell" % ok


def gate_verbatim(ctx, fails):
    """A snippet must be a substring of the code it claims to quote."""
    checked = 0
    for f, r in ctx["records"]:
        entry = packet_of(r, ctx)
        if not entry:
            continue
        _packet, code = entry
        haystack = norm_ws(code)
        for key in ("implementation", "attempted_code"):
            impl = dig(r, ("payload", key)) or {}
            if isinstance(impl, str):
                impl = {"snippet": impl}
            snippet = (impl or {}).get("snippet")
            if not snippet:
                continue
            checked += 1
            if not haystack:
                fails.append("%s: verbatim: quotes code but the packet carries "
                             "none" % f.name)
                continue
            # Compared line by line: a snippet assembled from two hunks is still
            # verbatim if every line of it is.
            missing = [ln for ln in snippet.splitlines()
                       if len(norm_ws(ln)) > 12 and norm_ws(ln) not in haystack]
            if missing:
                fails.append("%s: verbatim: %d line(s) not in the source, first: "
                             "%r" % (f.name, len(missing),
                                     norm_ws(missing[0])[:90]))
    return "%d snippets verified" % checked


def gate_no_fabrication(ctx, fails):
    """Every number a record states must appear in its packet."""
    audited = 0
    for f, r in ctx["records"]:
        entry = packet_of(r, ctx)
        if not entry:
            fails.append("%s: no-fabrication: no packet for this id or for its "
                         "(trace, version); nothing can audit its numbers"
                         % f.name)
            continue
        packet, code = entry
        pool = mags(json.dumps(packet["agent_facing"], ensure_ascii=False))
        pool |= mags(code)
        # The target block's digits are structural (ids, counts), not claims.
        pool |= mags(json.dumps(packet["target"], ensure_ascii=False))
        audited += 1
        for path in AUDITED_PATHS:
            node = dig(r, path)
            if node is None:
                continue
            for text, in_code in walk_strings(node):
                if in_code:
                    continue
                for m in sorted(mags(text) - pool):
                    fails.append("%s: no-fabrication: %s states %s, absent from "
                                 "the packet" % (f.name, "/".join(path), m))
    return "%d records audited against packets" % audited


def gate_provenance(ctx, fails):
    """evidence.raw must name this trace and a commit that resolves in it.

    There is no page to cite instead, so this is the whole of a record's
    auditability: the trace label says which repository, and the commit says
    which state of it. A record that names neither cannot be re-checked once the
    packets are deleted.
    """
    resolved = 0
    cache = {}
    for f, r in ctx["records"]:
        raw = dig(r, ("evidence", "raw")) or {}
        repo, sha = raw.get("source_repo"), raw.get("git_commit")
        if not repo:
            fails.append("%s: provenance: no evidence.raw.source_repo" % f.name)
        elif ctx["trace_label"] and repo != ctx["trace_label"]:
            fails.append("%s: provenance: source_repo is %r but this trace is %r"
                         % (f.name, repo, ctx["trace_label"]))
        if not sha:
            fails.append("%s: provenance: no evidence.raw.git_commit" % f.name)
            continue
        if sha not in cache:
            cache[sha] = c.git("cat-file", "-t", sha).strip() == "commit"
        if cache[sha]:
            resolved += 1
        else:
            fails.append("%s: provenance: %s is not a commit in the trace"
                         % (f.name, str(sha)[:12]))
    return "%d commit shas resolved in the trace" % resolved


def gate_ncu_attribution(ctx, fails):
    """A profiler-backed claim must cite a capture of the right kernel.

    A capture taken without a `--kernel-name` filter measures whatever ran first,
    typically a harness kernel. Those files are schema-valid and describe the
    wrong kernel, so they are actively misleading rather than merely empty --
    without this gate a record could cite one and look well-evidenced.
    """
    usable = ctx["usable_profiles"]
    checked = 0
    for f, r in ctx["records"]:
        ev = dig(r, ("evidence", "summary", "bottleneck_evidence")) or {}
        if ev.get("basis") != "profiler":
            continue
        checked += 1
        entry = packet_of(r, ctx)
        cited = (entry[0]["provenance"].get("ncu_dirs") if entry else []) or []
        if not any(d in usable for d in cited):
            fails.append(
                "%s: ncu-attribution: claims basis=profiler but no capture for "
                "this version measured the kernel under test (cited: %s)"
                % (f.name, ", ".join(cited) or "none"))
    return "%d profiler claims checked (%d usable captures exist)" % (
        checked, len(usable))


def gate_store_overlap(ctx, fails):
    """The id must still be free in the live store.

    `wiki-gate --commit insert` refuses an id that already exists, because
    overwriting would delete a record to make room for one the caller believed was
    new. Catching it here means renumbering before a batch is distilled rather
    than after. An `episode_key` that already exists is NOT a failure: it means
    the store holds this semantic identity already, and whether that is a
    rediscovery to confirm or a genuinely different record is a judgement the
    agent makes at the gate.
    """
    taken = ctx["store_ids"]
    dupes = 0
    for f, r in ctx["records"]:
        rid = r.get("id")
        if rid in taken:
            dupes += 1
            fails.append("%s: store-overlap: id %s already exists in the live "
                         "store; wiki-gate would refuse the insert" % (f.name, rid))
    same_episode = sorted({r.get("episode_key") for _f, r in ctx["records"]}
                          & ctx["store_episodes"])
    note = "%d ids free of %d" % (len(ctx["records"]) - dupes, len(ctx["records"]))
    if same_episode:
        note += "; %d episode_key(s) already in the store (confirm, not insert)" \
            % len(same_episode)
    return note


GATES = (
    ("profile", gate_profile),
    ("layout", gate_layout),
    ("verbatim", gate_verbatim),
    ("no-fabrication", gate_no_fabrication),
    ("provenance", gate_provenance),
    ("ncu-attribution", gate_ncu_attribution),
    ("store-overlap", gate_store_overlap),
)


def build_context(records, packets):
    by_id, by_ver = index_packets(packets)
    profiles_file = c.WORK / "profiles.jsonl"
    usable = set()
    if profiles_file.is_file():
        for line in profiles_file.open():
            row = json.loads(line)
            if row.get("ncu_usable"):
                usable.add(row["dir"])
    entries = recon.store_entries()
    return {
        "store": c.STORE,
        "records": records,
        "packets": packets,
        "by_id": by_id,
        "by_ver": by_ver,
        "usable_profiles": usable,
        "trace_label": c.trace_label(),
        "store_ids": {e.get("id") for e in entries},
        "store_episodes": {e.get("episode_key") for e in entries
                           if e.get("episode_key")},
    }


def run_store_gates():
    """This repository's own checker, against the staging root."""
    script = c.TOOLS / "check_kernel_wiki.py"
    if not script.is_file():
        return 1, ("%s not found. Refusing rather than skipping: silently "
                   "running four fewer gates is exactly what this layer exists "
                   "to prevent." % script)
    r = subprocess.run([sys.executable, str(script), "--out", str(c.STORE)],
                       capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


# ------------------------------------------------------------ injection tests

def _first(records, pred):
    for f, r in records:
        if pred(r):
            return f, r
    return None, None


def injection_tests(records, packets):
    """Prove each local gate fires on the error it exists to catch.

    Each case mutates a deep copy of one record (and, where the error lives
    there, of its packet) and asserts the named gate complains. Nothing is
    written to disk: no gate mutates state, so an in-memory context is both
    sufficient and instant.
    """
    def wrong_cell(rec, pkt):
        scope = rec["retrieval"]["scope"]
        scope["dsl"] = "cuda" if scope.get("dsl") != "cuda" else "triton"
        return True

    def paraphrase(rec, pkt):
        impl = (rec.get("payload") or {}).get("implementation") or {}
        if not (impl.get("snippet") or "").strip():
            return False
        impl["snippet"] = "+    totally_invented_symbol = compute_nothing()"
        return True

    def fabricate(rec, pkt):
        gain = (rec.get("worth") or {}).get("gain") or {}
        for e in gain.get("metrics") or []:
            e["delta_pct"] = 87.6543
            gain["pct"] = 87.6543
            return True
        # No metrics to corrupt: put the invented number in the mechanism prose,
        # which the gate audits for exactly this reason.
        payload = rec.get("payload") or {}
        for key in ("mechanism", "lesson"):
            if payload.get(key):
                payload[key] = payload[key] + " (87.6543% of the traffic)"
                return True
        return False

    def wrong_commit(rec, pkt):
        rec["evidence"]["raw"]["git_commit"] = "0" * 40
        return True

    def unbacked_profiler(rec, pkt):
        summary = rec["evidence"].setdefault("summary", {})
        summary.setdefault("bottleneck_evidence", {})["basis"] = "profiler"
        if pkt is not None:
            # The claim is only a lie when no capture backs it, so the honest
            # injection has to empty the packet's citation too.
            pkt["provenance"]["ncu_dirs"] = []
        return True

    def collide(rec, pkt, taken=None):
        if not taken:
            return False
        rec["id"] = sorted(taken)[0]
        return True

    def break_profile(rec, pkt):
        # `level: generic` is legal clean-1.3 and forbidden by the profile: one
        # kernel's measurement may not be promoted to the generic tier.
        rec["level"] = "generic"
        return True

    ctx0 = build_context(records, packets)
    cases = [
        ("profile/generic-level", "profile", break_profile, None),
        ("layout/scope-mismatch", "layout", wrong_cell, None),
        ("verbatim/paraphrased-snippet", "verbatim", paraphrase,
         lambda r: ((r.get("payload") or {}).get("implementation")
                    or {}).get("snippet")),
        ("no-fabrication/invented-number", "no-fabrication", fabricate, None),
        ("provenance/unknown-commit", "provenance", wrong_commit, None),
        ("ncu-attribution/unbacked-claim", "ncu-attribution",
         unbacked_profiler, None),
        ("store-overlap/known-id", "store-overlap",
         lambda rec, pkt: collide(rec, pkt, ctx0["store_ids"]), None),
    ]

    rc = 0
    for name, gate, mutate, need in cases:
        f, rec = _first(records, need or (lambda r: True))
        if rec is None:
            print("  %-34s SKIP (no suitable record)" % name)
            continue
        mrec = copy.deepcopy(rec)
        mpackets = copy.deepcopy(packets)
        ctx = build_context([(f, mrec)], mpackets)
        mpkt_entry = packet_of(mrec, ctx)
        mpkt = mpkt_entry[0] if mpkt_entry else None
        if not mutate(mrec, mpkt):
            print("  %-34s SKIP (record lacks the field)" % name)
            continue
        # Rebuild after the mutation so an id change is reflected in the indexes.
        ctx = build_context([(f, mrec)], mpackets)
        fails = []
        try:
            dict(GATES)[gate](ctx, fails)
        except Exception as exc:                                # noqa: BLE001
            fails.append("gate crashed: %s: %s" % (type(exc).__name__, exc))
        if fails:
            print("  %-34s FIRED  %s" % (name, fails[0][:88]))
        else:
            print("  %-34s *** DID NOT FIRE -- gate %s is asleep ***"
                  % (name, gate))
            rc = 1
    return rc


# ---------------------------------------------------------------------- driver

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--skip-store-gates", action="store_true",
                    help="only run the trace-specific gates")
    ap.add_argument("--injection-tests", action="store_true",
                    help="prove each trace-specific gate fires")
    args = ap.parse_args()

    c.require_trace()
    records = load_records(c.STORE)
    packets = load_packets()
    if not records:
        print("no records under %s/records" % c.STORE)
        return 1

    if args.injection_tests:
        print("injection tests against %s" % c.STORE)
        return injection_tests(records, packets)

    print("store=%s  records=%d  packets=%d"
          % (c.STORE, len(records), len(packets)))
    print()

    failed = 0
    if not args.skip_store_gates:
        code, out = run_store_gates()
        print("--- tools/check_kernel_wiki.py --out <staging> ---")
        print(out.rstrip())
        print()
        if code:
            failed += 1

    ctx = build_context(records, packets)
    print("--- trace-specific gates ---")
    for name, fn in GATES:
        fails = []
        try:
            note = fn(ctx, fails)
        except Exception as exc:                                # noqa: BLE001
            fails.append("%s: gate crashed: %r" % (name, exc))
            note = "ERROR"
        if isinstance(note, str) and note.startswith("SKIP:"):
            print("  %-16s %-12s %s" % (name, "SKIP", note[6:]))
            continue
        status = "PASS" if not fails else "FAIL (%d)" % len(fails)
        print("  %-16s %-12s %s" % (name, status, note))
        if fails:
            failed += 1
            shown = fails if args.verbose else fails[:6]
            for msg in shown:
                print("      - %s" % msg)
            if len(fails) > len(shown):
                print("      ... %d more (use --verbose)" % (len(fails) - len(shown)))

    print()
    if failed:
        print("GATES FAILED (%d groups)" % failed)
        return 1
    print("ALL GATES PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
