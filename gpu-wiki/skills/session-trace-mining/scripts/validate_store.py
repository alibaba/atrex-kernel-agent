#!/usr/bin/env python3
"""Sixteen gates for a session-derived store. Mechanical checks only.

The distillation step is a model filling fields, so the trustworthiness of the
whole store rests on checks that need no judgement. There is no repository to
resolve a citation against here: the transcript is the corpus, so provenance is
pinned by line digest and every number must be traceable to a span the packet
carried.

  schema                  jsonschema against session-trace-1.0
  ids                     unique, filename == id, >=5 dotted segments
  layout                  records/<type>/<vendor>/<arch>/<dsl>/<family>/ == scope
  provenance              set/rel_path/line digests/session id/packet hash resolve
  verbatim                implementation.snippet is literally in the packet diff
  no-fabrication          every number in worth.gain appears in evidence_text
  direction               delta_pct sign agrees with the metric's direction
  raw-isolation           the serve projection leaks nothing from raw/locator
  relations               every referenced record id exists
  index                   index.json agrees with records/
  evidence-tier           basis=measured requires a T1 span; source_kind fixed
  diff-coverage           a strategy record must carry a recoverable change
  unit-normalization       delta_pct is reproducible; no absolute time in gain
  wiki-overlap            nothing the committed store already covers (id,
                          episode_key or operator version), and the scan of that
                          store must actually have found records
  pairing-integrity       cited lines lie in one monotone region
  anonymization           the agent-visible layers name no person, host or path

Usage:
  STM_SET=<name> python3 validate_store.py [--verbose]
  python3 validate_store.py --all-sets
  python3 validate_store.py --injection-tests
"""
import argparse
import copy
import json
import re
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

import config as c
import metrics as M
import transcripts as T

# Fixed by the metric name, never stored on the record: storing it would let the
# distiller contradict itself.
DIRECTIONS = {
    "latency": "lower-better",
    "compile_time": "lower-better",
    "memory_footprint": "lower-better",
    "sol": "higher-better",
    "mfu": "higher-better",
    "dram_throughput": "higher-better",
    "compute_throughput": "higher-better",
    "arithmetic_intensity": "higher-better",
}

# Anything that must never reach the agent-visible projection.
RAW_ONLY_KEYS = {"locator", "links", "raw", "source_repo", "detail_file",
                 "git_commit", "file_paths", "session", "line_digests",
                 "rel_path", "set", "sibling_paths", "evidence_sha256",
                 "candidate_id", "dedup_key"}

# A number is auditable only when it does not continue an alphanumeric run: the
# lookbehind skips the digits inside SM100, FA3, fp8, e4m3, v2.8.3 (names, not
# measurements) while real values after whitespace or punctuation still match.
NUM_RE = re.compile(r"(?<![A-Za-z0-9.])-?\d+(?:\.\d+)?")

# A time unit anywhere inside worth.gain means an absolute latency leaked into the
# layer the agent sorts on. Microseconds are meaningless across shape sets.
TIME_IN_GAIN_RE = re.compile(r"\b\d+(?:\.\d+)?\s*(?:ns|us|µs|μs|ms|s)\b", re.I)

# People, hosts and corpus paths that must not reach the agent.
IDENTITY_RE = re.compile(
    r"/home/[a-z0-9_.-]+|/root/[a-z0-9_.-]+|/Users/[a-z0-9_.-]+"
    r"|[A-Za-z]:\\Users\\[A-Za-z0-9_.-]+"
    r"|rollout-\d{4}-\d{2}-\d{2}T[\d-]+|\.claude/projects"
    r"|/tmp/session-trace-mining|sessions\.tar\.gz", re.I)

# The same leak after a path has been flattened into an identifier slug, where no
# slash survives for the pattern above to match. This is how an id of the shape
# `nvidia.b200.gluon.root-<username>-<project>.…` reached the store once: the
# operator name was derived from a claude project directory, which is a home path.
SLUGGED_IDENTITY_RE = re.compile(
    r"(?<![a-z0-9])(?:root|home|users?)-[a-z0-9]+[a-z0-9-]*"
    r"|-root-[a-z0-9-]+", re.I)


def norm_ws(s):
    return re.sub(r"\s+", " ", s or "").strip()


def mags(text):
    """Magnitude strings in `text`, sign dropped.

    no-fabrication audits whether a magnitude was invented; the sign is a
    direction and the direction gate checks it. Dropping it here removes a class
    of false positives where the record wrote -5% but the source states 5 without
    that signed token. Both integer and decimal forms are recorded so 5 and 5.0
    match either way.
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


def load_packets(set_name):
    out = {}
    pack = c.packets(set_name)
    if not pack.is_dir():
        return out
    for f in sorted(pack.glob("*.json")):
        try:
            pkt = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        cid = (pkt.get("session") or {}).get("candidate_id")
        if cid:
            out[cid] = pkt
    return out


def session_of(rec):
    return ((rec.get("evidence") or {}).get("raw") or {}).get("session") or {}


def packet_of(rec, packets):
    return packets.get(session_of(rec).get("candidate_id"))


def serve(rec):
    """What the agent actually receives."""
    out = copy.deepcopy(rec)
    ev = out.get("evidence") or {}
    out["evidence"] = {"summary": ev.get("summary")} if ev.get("summary") else {}
    retr = out.get("retrieval") or {}
    retr.pop("locator", None)
    retr.pop("links", None)
    worth = out.get("worth") or {}
    worth.pop("track", None)
    (worth.get("rank") or {}).pop("components", None)
    return out


def walk_keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            for x in walk_keys(v):
                yield x
    elif isinstance(obj, list):
        for v in obj:
            for x in walk_keys(v):
                yield x


def walk_strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            for x in walk_strings(v):
                yield x
    elif isinstance(obj, list):
        for v in obj:
            for x in walk_strings(v):
                yield x


def gain_of(rec):
    return (rec.get("worth") or {}).get("gain") or {}


# ------------------------------------------------------------------ the gates

def gate_schema(ctx, fails):
    try:
        import jsonschema
    except ImportError:
        return "SKIP (jsonschema not installed)"
    schema = json.loads(c.DERIVED_SCHEMA.read_text())
    validator = jsonschema.Draft202012Validator(schema)
    n = 0
    for f, r in ctx["records"]:
        for err in validator.iter_errors(r):
            fails.append("%s: schema: %s: %s"
                         % (f.name, "/".join(str(p) for p in err.path),
                            err.message[:160]))
            n += 1
    return "%d records, %d violations" % (len(ctx["records"]), n)


def gate_ids(ctx, fails):
    seen = Counter()
    for f, r in ctx["records"]:
        rid = r.get("id", "")
        seen[rid] += 1
        if f.stem != rid:
            fails.append("%s: ids: filename != id (%r)" % (f.name, rid))
        if len(rid.split(".")) < 5:
            fails.append("%s: ids: id needs >=5 dotted segments" % f.name)
    for rid, n in seen.items():
        if n > 1:
            fails.append("ids: duplicate id %s x%d" % (rid, n))
    return "%d unique ids" % len(seen)


def gate_layout(ctx, fails):
    """Directory must equal the record's own type and scope.

    Deriving the path from the same fields the engine filters on makes a misfiled
    record a mechanical error rather than something a human has to notice.
    """
    root = ctx["store"] / "records"
    ok = 0
    for f, r in ctx["records"]:
        scope = (r.get("retrieval") or {}).get("scope") or {}
        gen = (r.get("retrieval") or {}).get("generality") or {}
        want = (r.get("type"), scope.get("vendor"), scope.get("arch"),
                scope.get("dsl"), gen.get("workload_family") or "any")
        try:
            got = tuple(f.relative_to(root).parts[:-1])
        except ValueError:
            fails.append("%s: layout: not under %s" % (f.name, root))
            continue
        if got != want:
            fails.append("%s: layout: stored under %s but scope says %s"
                         % (f.name, "/".join(got),
                            "/".join(str(x) for x in want)))
        else:
            ok += 1
    return "%d records in the right cell" % ok


def gate_provenance(ctx, fails):
    """Re-resolve every citation against the archive.

    No clone to `git cat-file` here, so the transcript line is the provenance:
    the set is looked up by name, the file by a set-relative path, and each cited
    line by a digest of its raw bytes. Retargeting a citation to a different line
    fails immediately, and moving the whole archive to another absolute path still
    passes because only `sets.json` knows where it is.
    """
    checked = 0
    for f, r in ctx["records"]:
        s = session_of(r)
        if not s:
            fails.append("%s: provenance: evidence.raw.session missing" % f.name)
            continue
        root = ctx["sets"].get(s.get("set"))
        if not root:
            fails.append("%s: provenance: unknown set %r" % (f.name, s.get("set")))
            continue
        path = Path(root) / s.get("rel_path", "")
        if not path.is_file():
            fails.append("%s: provenance: %s not found under set %s"
                         % (f.name, s.get("rel_path"), s.get("set")))
            continue
        lines = path.read_bytes().splitlines()
        digests = s.get("line_digests") or {}
        for ln in s.get("line_nos") or []:
            if ln < 1 or ln > len(lines):
                fails.append("%s: provenance: line %d out of range (%d lines)"
                             % (f.name, ln, len(lines)))
                continue
            want = digests.get(str(ln))
            got = T.digest_of(lines[ln - 1])
            if not want:
                # Without this, retargeting a citation to a neighbouring line
                # passes: the shifted number has no digest entry, so there is
                # nothing to compare and the check quietly does nothing.
                fails.append("%s: provenance: line %d is cited with no digest"
                             % (f.name, ln))
            elif want != got:
                fails.append("%s: provenance: line %d digest %s != %s "
                             "(citation retargeted or file changed)"
                             % (f.name, ln, want, got))
        anchor = s.get("line_nos") or []
        if anchor:
            try:
                obj = json.loads(lines[anchor[0] - 1].decode("utf-8", "replace"))
            except (json.JSONDecodeError, IndexError):
                obj = {}
            got_sid = (obj.get("sessionId") or obj.get("session_id")
                       or ((obj.get("payload") or {}).get("session_id")))
            if got_sid and s.get("session_id") and got_sid != s["session_id"]:
                # Only a mismatch on a line that states an id is a failure; most
                # lines do not carry one.
                fails.append("%s: provenance: session_id %r but line %d says %r"
                             % (f.name, s["session_id"], anchor[0], got_sid))
        pkt = packet_of(r, ctx["packets"])
        if pkt is None:
            fails.append("%s: provenance: no packet for candidate_id=%r"
                         % (f.name, s.get("candidate_id")))
            continue
        want_hash = s.get("evidence_sha256")
        got_hash = (pkt.get("session") or {}).get("evidence_sha256")
        if want_hash and got_hash and want_hash != got_hash:
            fails.append("%s: provenance: evidence_sha256 %s != packet %s"
                         % (f.name, want_hash[:12], got_hash[:12]))
        checked += 1
    return "%d citations re-resolved" % checked


def gate_verbatim(ctx, fails):
    """A snippet the distiller paraphrased is worse than none: it looks
    authoritative and will not compile."""
    checked = 0
    for f, r in ctx["records"]:
        impl = (r.get("payload") or {}).get("implementation") or {}
        snippet = impl.get("snippet")
        if not snippet:
            continue
        pkt = packet_of(r, ctx["packets"])
        if pkt is None:
            fails.append("%s: verbatim: no packet to check the snippet against"
                         % f.name)
            continue
        diff_name = pkt.get("diff_file")
        src = ""
        if diff_name:
            p = c.packets(ctx["set"]) / diff_name
            if p.is_file():
                src = norm_ws(p.read_text())
        if not src:
            fails.append("%s: verbatim: packet carries no diff, so a snippet "
                         "cannot be verified" % f.name)
            continue
        missing = [ln for ln in snippet.splitlines()
                   if len(norm_ws(ln)) > 12 and norm_ws(ln) not in src]
        if missing:
            fails.append("%s: verbatim: %d snippet line(s) not in the packet "
                         "diff, first: %r" % (f.name, len(missing),
                                              missing[0][:80]))
        else:
            checked += 1
        fmt = impl.get("format", "")
        if snippet.lstrip().startswith(("+", "-")) and \
                not fmt.startswith("unified-diff"):
            fails.append("%s: verbatim: snippet looks like a diff but format=%r"
                         % (f.name, fmt))
    return "%d snippets verified" % checked


def gate_no_fabrication(ctx, fails):
    """Every number the agent will act on must be traceable to the packet.

    The haystack is the packet's `evidence_text`, which by construction holds only
    T1/T2/T3 spans: the agent's own prose never enters it, so a number it invented
    has nowhere to hide.
    """
    checked = 0
    for f, r in ctx["records"]:
        gain = gain_of(r)
        pkt = packet_of(r, ctx["packets"])
        if pkt is None:
            if gain.get("basis") in ("measured", "reported"):
                fails.append("%s: no-fabrication: no packet, cannot audit numbers"
                             % f.name)
            continue
        pool = mags(pkt.get("evidence_text", ""))
        # Derived deltas are legitimate, so the measured levels the packet
        # extracted are admissible anchors too.
        meas = pkt.get("measurement") or {}
        for key in ("improve_pct", "geomean_us", "before_us", "after_us",
                    "n_shapes"):
            if meas.get(key) is not None:
                pool |= mags("%g" % float(meas[key]))
        derivable = _pct_deriver(pkt, pool)
        for entry in (gain.get("metrics") or []) + (gain.get("regressions") or []):
            for key in ("before", "after", "value"):
                v = entry.get(key)
                if v is None:
                    continue
                if "%g" % abs(float(v)) not in pool:
                    fails.append("%s: no-fabrication: %s=%s not in the packet "
                                 "evidence" % (f.name, key, v))
            dp = entry.get("delta_pct")
            if dp is not None and not derivable(dp):
                fails.append("%s: no-fabrication: delta_pct=%s is neither stated "
                             "nor derivable from the packet evidence"
                             % (f.name, dp))
            for txt in (entry.get("note"), entry.get("measured_over")):
                for m in mags(txt):
                    if m in pool or derivable(m):
                        continue
                    fails.append("%s: no-fabrication: %s in %r not in the "
                                 "packet evidence" % (f.name, m, (txt or "")[:60]))
        for txt in [gain.get("note")] + [
                str(v) for v in (gain.get("correctness") or {}).values()
                if isinstance(v, (int, float, str))]:
            for m in mags(txt):
                if m in pool or derivable(m):
                    continue
                fails.append("%s: no-fabrication: %s in %r not in the packet "
                             "evidence" % (f.name, m, (txt or "")[:60]))
        checked += 1
    return "%d gain blocks audited" % checked


def _pct_deriver(pkt, pool):
    """Is this percentage stated, or produced from an anchor by a fixed formula?

    Two derivations are admitted, both closed-form and both from an anchor that is
    demonstrably present:

      the packet's own measured levels   (before-after)/before
      a speedup the harness printed      1 - 1/x, for each `Nx` in the evidence

    Derivation from arbitrary *pairs* of pool numbers is deliberately not allowed.
    Allowing arbitrary pairs was measured once and never again: a 200-number
    packet offers 40k pairs, so every value becomes derivable and the gate is
    asleep while looking green.
    """
    text = pkt.get("evidence_text", "")
    meas = pkt.get("measurement") or {}
    anchors = []
    before, after = meas.get("before_us"), meas.get("after_us")
    if before and after:
        d = M.delta_from_pair(before, after)
        if d is not None:
            anchors.append(abs(d))
            anchors.append(abs((1.0 - after / before) * 100.0))
    for sp in M.SPEEDUP_RE.findall(text) + M.BARE_SPEEDUP_RE.findall(text):
        try:
            d = M.delta_from_speedup(float(sp))
        except (TypeError, ValueError):
            continue
        if d is not None:
            anchors.append(abs(d))

    def ok(value):
        try:
            v = abs(float(value))
        except (TypeError, ValueError):
            return False
        if "%g" % v in pool:
            return True
        return any(abs(v - a) <= 0.6 for a in anchors)

    return ok


def gate_direction(ctx, fails):
    ok = 0
    for f, r in ctx["records"]:
        gain = gain_of(r)
        for entry in gain.get("metrics") or []:
            name = entry.get("metric")
            want = DIRECTIONS.get(name)
            if want and entry.get("direction") != want:
                fails.append("%s: direction: %s is %s, record says %r"
                             % (f.name, name, want, entry.get("direction")))
            before, after = entry.get("before"), entry.get("after")
            dp = entry.get("delta_pct")
            if None not in (before, after) and dp is not None:
                improved = (after < before) if want == "lower-better" \
                    else (after > before)
                if improved != (dp > 0):
                    fails.append("%s: direction: %s %s->%s disagrees with "
                                 "delta_pct=%s" % (f.name, name, before, after, dp))
            ok += 1
        for entry in gain.get("regressions") or []:
            if (entry.get("delta_pct") or 0) > 0:
                fails.append("%s: direction: a regression may not have a positive "
                             "delta_pct (%s)" % (f.name, entry.get("delta_pct")))
        pct, primary = gain.get("pct"), gain.get("primary")
        if pct is not None and primary:
            match = [e for e in gain.get("metrics") or []
                     if e.get("metric") == primary]
            if match and match[0].get("delta_pct") is not None \
                    and abs(match[0]["delta_pct"] - pct) > 0.01:
                fails.append("%s: direction: gain.pct=%s != primary %s delta_pct=%s"
                             % (f.name, pct, primary, match[0]["delta_pct"]))
    return "%d metric entries checked" % ok


def gate_raw_isolation(ctx, fails):
    leaked = 0
    for f, r in ctx["records"]:
        keys = set(walk_keys(serve(r)))
        bad = keys & RAW_ONLY_KEYS
        if bad:
            fails.append("%s: raw-isolation: served projection contains %s"
                         % (f.name, sorted(bad)))
            leaked += 1
    return "%d records, %d leaking" % (len(ctx["records"]), leaked)


def gate_relations(ctx, fails):
    ids = {r.get("id") for _f, r in ctx["records"]}
    n = 0
    for f, r in ctx["records"]:
        links = (r.get("retrieval") or {}).get("links") or {}
        refs = []
        if links.get("parent"):
            refs.append(links["parent"])
        for key in ("depends_on", "conflicts_with", "see_also", "supersedes",
                    "cited_records"):
            refs.extend(links.get(key) or [])
        for ref in refs:
            n += 1
            # A reference into the committed stores is legitimate; only a dangling
            # reference into this store is an error.
            if ref in ids or ctx["wiki_ids"] and ref in ctx["wiki_ids"]:
                continue
            fails.append("%s: relations: %s does not resolve" % (f.name, ref))
    return "%d references checked" % n


def gate_index(ctx, fails):
    path = ctx["store"] / "records" / "index.json"
    if not path.is_file():
        return "SKIP (no index.json)"
    try:
        idx = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        fails.append("index: unparseable (%s)" % e)
        return "unparseable"
    if idx.get("schema") != c.DERIVED_NAME:
        fails.append("index: schema is %r, expected %r"
                     % (idx.get("schema"), c.DERIVED_NAME))
    on_disk = {r.get("id") for _f, r in ctx["records"]}
    in_index = {e.get("id") for e in idx.get("records") or []}
    for rid in sorted(on_disk - in_index):
        fails.append("index: %s on disk but not in index.json" % rid)
    for rid in sorted(in_index - on_disk):
        fails.append("index: %s in index.json but not on disk" % rid)
    by_id = {r.get("id"): r for _f, r in ctx["records"]}
    for e in idx.get("records") or []:
        rec = by_id.get(e.get("id"))
        if rec and e.get("retrieval") != rec.get("retrieval"):
            fails.append("index: %s retrieval block is stale" % e.get("id"))
    return "%d entries" % len(in_index)


def gate_evidence_tier(ctx, fails):
    """`measured` requires a benchmark span, not the run's own notes.

    Without this, the tier work in the packet is advisory and a record can promote
    a number the agent read back from its own memory file to a measurement.

    This gate also enforces the one claim no mechanical check can validate: when a
    comparison's two sides were chosen by print order, nothing establishes that
    they measure alternatives of the same work. A distilling agent found a
    three-phase CUDA-event breakdown of one call read as phase-vs-phase (78.9%)
    and an approximate path against a vendor library computing a different result
    (-99.5%), and both passed every other gate. Such a record may describe the
    finding in prose but may not publish a gain.
    """
    ok = 0
    for f, r in ctx["records"]:
        gain = gain_of(r)
        if gain.get("source_kind") != "agent-session":
            fails.append("%s: evidence-tier: source_kind must be 'agent-session', "
                         "got %r" % (f.name, gain.get("source_kind")))
        pkt = packet_of(r, ctx["packets"])
        if pkt is None:
            continue
        tiers = pkt.get("evidence_tiers") or {}
        if gain.get("basis") == "measured" and not tiers.get("T1"):
            fails.append("%s: evidence-tier: basis=measured but the packet has no "
                         "T1 (tool-output) span; tiers=%s" % (f.name, tiers))
        conf = ((r.get("evidence") or {}).get("summary") or {}).get("confidence")
        if conf == "measured" and not tiers.get("T1"):
            fails.append("%s: evidence-tier: confidence=measured without a T1 span"
                         % f.name)
        if pkt.get("claims_no_gain"):
            if gain.get("kind") != "none" or gain.get("pct") is not None \
                    or gain.get("metrics"):
                fails.append("%s: evidence-tier: the comparison's sides were "
                             "chosen by print order, so this record may not "
                             "publish a gain (needs kind=none, pct=null, no "
                             "metrics)" % f.name)
        ok += 1
    return "%d records tiered" % ok


def gate_diff_coverage(ctx, fails):
    """A strategy record must carry a change someone can apply."""
    ok = 0
    for f, r in ctx["records"]:
        if r.get("type") != "strategy":
            continue
        s = session_of(r)
        impl = (r.get("payload") or {}).get("implementation") or {}
        if s.get("diff_coverage") == "blind":
            fails.append("%s: diff-coverage: diff_coverage=blind may not be a "
                         "strategy record" % f.name)
        if not (impl.get("snippet") or "").strip():
            fails.append("%s: diff-coverage: strategy record has no snippet"
                         % f.name)
        pkt = packet_of(r, ctx["packets"])
        if pkt is not None and not pkt.get("diff_file"):
            fails.append("%s: diff-coverage: packet has no diff file" % f.name)
        ok += 1
    return "%d strategy records checked" % ok


def gate_unit_normalization(ctx, fails):
    """delta_pct must be reproducible, and absolute time must stay out of gain.

    A record whose percentage cannot be recomputed from the levels the extractor
    measured is either citing a different measurement or has a unit error; both
    are indistinguishable from fabrication to a reader.
    """
    ok = 0
    for f, r in ctx["records"]:
        gain = gain_of(r)
        blob = json.dumps(gain, ensure_ascii=False)
        m = TIME_IN_GAIN_RE.search(blob)
        if m:
            fails.append("%s: unit-normalization: absolute time %r inside "
                         "worth.gain" % (f.name, m.group(0)))
        pkt = packet_of(r, ctx["packets"])
        if pkt is None:
            continue
        meas = pkt.get("measurement") or {}
        before, after = meas.get("before_us"), meas.get("after_us")
        # Reproducibility applies to `metrics` only. A regression entry
        # describes a *different subset* of shapes than the headline geomean --
        # three records in one set report a -4.4% worst-shape regression next to a
        # +5% geomean -- so its levels are not the headline levels, and demanding
        # that it reproduce from them rejects correct records. A regression's sign is
        # checked by `direction` and its magnitude by `no-fabrication`, and the
        # absolute-time ban above already covers the whole gain block.
        for entry in gain.get("metrics") or []:
            if entry.get("metric") != "latency":
                continue
            dp = entry.get("delta_pct")
            if dp is None or None in (before, after) or not before:
                continue
            recomputed = M.delta_from_pair(before, after)
            ratio = (1.0 - after / before) * 100.0 if before else None
            if recomputed is None:
                continue
            if abs(recomputed - dp) > 0.6 and abs((ratio or 0) - dp) > 0.6:
                fails.append("%s: unit-normalization: delta_pct=%s is not "
                             "reproducible from %.4f->%.4f us (would be %.2f%%)"
                             % (f.name, dp, before, after, recomputed))
            ok += 1
    return "%d latency deltas reproduced" % ok


def gate_wiki_overlap(ctx, fails):
    """Nothing the committed store already covers.

    Three collisions, in falling order of how often they happen:

      dedup_key     `<operator>|<version>`. The decision on this store is to skip
                    a colliding version during ingest, so one reaching the
                    records means ingest was bypassed or a record was hand-written.
      id            a session record must not occupy an id the committed store
                    already uses; merging the two stores later would overwrite one.
      episode_key   the same episode written twice under different ids is a
                    duplicate the retrieval engine cannot rank.

    The scan itself is checked. A committed store that resolves to nothing at all
    turns this gate into a no-op that still prints OK, which is the exact failure
    mode the injection tests exist to prevent -- so an empty scan is a failure,
    not a pass.
    """
    if not ctx["wiki_scanned"]:
        fails.append("wiki-overlap: the committed store scan found no records "
                     "under %s, so overlap cannot be detected; fix the path "
                     "before trusting this gate"
                     % ", ".join(str(p) for p in c.COMMITTED_STORES))
        return "committed store empty or missing"
    n = 0
    for f, r in ctx["records"]:
        key = session_of(r).get("dedup_key")
        if key:
            n += 1
            if key in ctx["wiki_versions"]:
                fails.append("%s: wiki-overlap: %s is already covered by %s"
                             % (f.name, key,
                                ", ".join(ctx["wiki_versions"][key][:2])))
        if r.get("id") in ctx["wiki_ids"]:
            fails.append("%s: wiki-overlap: id %s already exists in the "
                         "committed store" % (f.name, r.get("id")))
        ep = r.get("episode_key")
        if ep and ep in ctx["wiki_episodes"]:
            fails.append("%s: wiki-overlap: episode_key %s already exists in "
                         "the committed store" % (f.name, ep))
    return "%d dedup keys, %d committed records" % (n, ctx["wiki_scanned"])


def gate_pairing_integrity(ctx, fails):
    """Cited lines must lie in one monotone region of one transcript.

    Compaction and rollback mean line order stops being history, so a citation
    that straddles one is not evidence of a before/after relationship even though
    the line numbers look ordered.
    """
    ok = 0
    cache = {}
    for f, r in ctx["records"]:
        s = session_of(r)
        root = ctx["sets"].get(s.get("set"))
        if not root or not s.get("line_nos"):
            continue
        key = (root, s.get("rel_path"))
        if key not in cache:
            path = Path(root) / s["rel_path"]
            if not path.is_file():
                continue
            _fmt, events = T.parse(path)
            cuts = [e.line_no for e in events if e.kind == "discontinuity"]
            cache[key] = cuts
        cuts = cache[key]
        lines = sorted(s["line_nos"])
        crossed = [x for x in cuts if lines[0] < x < lines[-1]]
        if crossed:
            fails.append("%s: pairing-integrity: cited lines %s-%s straddle a "
                         "discontinuity at %s" % (f.name, lines[0], lines[-1],
                                                  crossed[:3]))
        else:
            ok += 1
    return "%d citations in one region" % ok


def gate_anonymization(ctx, fails):
    """The agent-visible layers must name no person, host or corpus path.

    Session transcripts are full of `/home/<user>/<project>` working directories,
    and `payload` is what gets served, so this is not a theoretical risk.
    `evidence.raw` is exempt by design -- it is the human-only layer and
    raw-isolation keeps it out of the projection.

    The **id is served knowledge too**, and it needs its own check: a home
    directory flattened into a slug has no slash left for the path pattern to
    match, which is how an id of the shape
    `nvidia.b200.gluon.root-<username>-<project>.…` reached the store once.
    """
    n = 0
    for f, r in ctx["records"]:
        ident = "%s %s" % (r.get("id") or "", r.get("episode_key") or "")
        m = SLUGGED_IDENTITY_RE.search(ident)
        if m:
            fails.append("%s: anonymization: identifier contains %r, which looks "
                         "like a flattened home-directory path"
                         % (f.name, m.group(0)))
            n += 1
            continue
        for text in walk_strings(serve(r)):
            m = IDENTITY_RE.search(text)
            if m:
                fails.append("%s: anonymization: served text contains %r"
                             % (f.name, m.group(0)))
                n += 1
                break
    return "%d records with a leak" % n


GATES = [
    ("schema", gate_schema),
    ("ids", gate_ids),
    ("layout", gate_layout),
    ("provenance", gate_provenance),
    ("verbatim", gate_verbatim),
    ("no-fabrication", gate_no_fabrication),
    ("direction", gate_direction),
    ("raw-isolation", gate_raw_isolation),
    ("relations", gate_relations),
    ("index", gate_index),
    ("evidence-tier", gate_evidence_tier),
    ("diff-coverage", gate_diff_coverage),
    ("unit-normalization", gate_unit_normalization),
    ("wiki-overlap", gate_wiki_overlap),
    ("pairing-integrity", gate_pairing_integrity),
    ("anonymization", gate_anonymization),
]


# ---------------------------------------------------------------- wiki lookups

def wiki_state():
    """What the committed store already holds.

    Returns (record ids, episode keys, (operator, version) -> ids, records
    scanned). `scanned` is returned because a scan that silently finds nothing is
    the failure mode of the overlap gate, and the gate reports it rather than
    passing.

    A version identity is read from whichever shape a record carries it in --
    stores seeded from documentation carry none, stores mined from an
    optimisation ladder carry `evidence.raw.source_repo` plus
    `evidence.raw.version`, and session-derived records carry
    `evidence.raw.session.dedup_key` already assembled.
    """
    ids, episodes, versions, scanned = set(), set(), {}, 0
    for root in c.COMMITTED_STORES:
        if not root.is_dir():
            continue
        for path in root.rglob("*.json"):
            if path.name == "index.json":
                continue
            try:
                rec = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            scanned += 1
            ids.add(rec.get("id"))
            if rec.get("episode_key"):
                episodes.add(rec["episode_key"])
            raw = (rec.get("evidence") or {}).get("raw") or {}
            key = (raw.get("session") or {}).get("dedup_key")
            if not key:
                repo, ver = raw.get("source_repo"), raw.get("version")
                if repo and ver:
                    key = "%s|%s" % (str(repo).rstrip("/").split("/")[-1],
                                     str(ver).lower())
            if key:
                versions.setdefault(key, []).append(rec.get("id"))
    return ids, episodes, versions, scanned


def run(set_name, store=None, verbose=False, quiet=False):
    store = store or c.store(set_name)
    sets_file = c.work(set_name) / "sets.json"
    sets = json.loads(sets_file.read_text()) if sets_file.is_file() else \
        {name: str(c.set_root(name)) for name in c.SETS}
    wiki_ids, wiki_episodes, wiki_versions, wiki_scanned = wiki_state()
    ctx = {"set": set_name, "store": store, "records": load_records(store),
           "packets": load_packets(set_name), "sets": sets,
           "wiki_ids": wiki_ids, "wiki_episodes": wiki_episodes,
           "wiki_versions": wiki_versions, "wiki_scanned": wiki_scanned}
    if not ctx["records"]:
        if not quiet:
            print("%-14s no records under %s" % (set_name, store / "records"))
        return 0, ctx
    fails = []
    for name, fn in GATES:
        before = len(fails)
        try:
            note = fn(ctx, fails)
        except Exception as e:                       # a crashing gate is a failure
            fails.append("%s: gate crashed: %s: %s" % (name, type(e).__name__, e))
            note = "CRASH"
        n = len(fails) - before
        if not quiet:
            print("  %-20s %-44s %s" % (name, note,
                                        "OK" if n == 0 else "%d FAIL" % n))
    if fails and not quiet:
        show = fails if verbose else fails[:5]
        print("")
        for msg in show:
            print("  ! %s" % msg)
        if len(fails) > len(show):
            print("  ... %d more (use --verbose)" % (len(fails) - len(show)))
    return len(fails), ctx


# ------------------------------------------------------------ injection tests

def _first(records, pred):
    for f, r in records:
        if pred(r):
            return f, r
    return None, None


def injection_tests(set_name):
    """Prove each gate fires on the error it exists to catch.

    A gate without an injection test is how a store ends up falsely green while
    looking audited: two of the cases below were asleep on their first run. Each
    case mutates a deep copy of one record (and, where the error lives there, of
    its packet) and asserts that the named gate complains. Nothing is written to
    disk: no gate mutates state, so an in-memory context is both sufficient and
    instant.
    """
    store = c.store(set_name)
    records = load_records(store)
    packets = load_packets(set_name)
    if not records:
        print("no records in %s; distil something first" % store)
        return 1
    sets_file = c.work(set_name) / "sets.json"
    sets = json.loads(sets_file.read_text()) if sets_file.is_file() else {}
    wiki_ids, wiki_episodes, wiki_versions, wiki_scanned = wiki_state()
    if not wiki_versions:
        # The committed store carries no version-bearing record (a store seeded
        # from documentation does not). The gate still has to be exercised, so a
        # known key is seeded here and the record is made to collide with it --
        # a skipped case proves nothing.
        wiki_versions = dict(wiki_versions)
        wiki_versions["injected-operator|v1"] = ["synthetic-collision-probe"]
    collision_key = sorted(wiki_versions)[0]
    committed_ids = sorted(i for i in wiki_ids if i)

    def shift_line(rec, pkt):
        s = session_of(rec)
        if not s.get("line_nos"):
            return False
        s["line_nos"] = [s["line_nos"][0] + 1] + s["line_nos"][1:]
        return True

    def wrong_set(rec, pkt):
        session_of(rec)["set"] = "not-a-set"
        return True

    def promote_basis(rec, pkt):
        # The honest form of this injection has to touch the packet too: claiming
        # `measured` is only a lie when no benchmark span backs it.
        if pkt is None:
            return False
        gain_of(rec)["basis"] = "measured"
        (rec.setdefault("evidence", {}).setdefault("summary", {}))[
            "confidence"] = "measured"
        pkt["evidence_tiers"] = {"T2": 1}
        return True

    def blind_strategy(rec, pkt):
        session_of(rec)["diff_coverage"] = "blind"
        return True

    def break_unit(rec, pkt):
        for e in gain_of(rec).get("metrics") or []:
            if e.get("metric") == "latency" and e.get("delta_pct") is not None:
                e["delta_pct"] *= 1000.0
                gain_of(rec)["pct"] = e["delta_pct"]
                return True
        return False

    def collide(rec, pkt):
        session_of(rec)["dedup_key"] = collision_key
        return True

    def take_committed_id(rec, pkt):
        # The id branch of wiki-overlap needs its own case: a version collision
        # and an id collision are different mistakes and only one of them was
        # checked before.
        if not committed_ids:
            return False
        rec["id"] = committed_ids[0]
        return True

    def leak_path(rec, pkt):
        rec["payload"]["change"] = (rec["payload"].get("change") or "") + \
            " see /home/example-user/kernels/attention/interface.py"
        return True

    def fabricate(rec, pkt):
        for e in gain_of(rec).get("metrics") or []:
            e["delta_pct"] = 87.6543
            gain_of(rec)["pct"] = 87.6543
            return True
        return False

    def paraphrase(rec, pkt):
        impl = rec["payload"].get("implementation") or {}
        if not (impl.get("snippet") or "").strip():
            return False
        impl["snippet"] = "+        totally_invented_symbol = compute_nothing()"
        return True

    def straddle(rec, pkt):
        # Cite a line far from the rest: on a transcript with any compaction or
        # rollback this must cross one.
        s = session_of(rec)
        if not s.get("line_nos"):
            return False
        far = max(s["line_nos"]) + 100000
        s["line_nos"] = s["line_nos"] + [far]
        s["line_digests"][str(far)] = "0" * 12
        return True

    def has_latency_metric(r):
        return any(e.get("metric") == "latency"
                   and e.get("delta_pct") is not None
                   for e in ((r.get("worth") or {}).get("gain") or {}
                             ).get("metrics") or [])

    cases = [
        ("provenance/line-shift", "provenance", shift_line, None),
        ("provenance/wrong-set", "provenance", wrong_set, None),
        ("provenance/straddle-far-line", "provenance", straddle, None),
        ("evidence-tier/promoted-basis", "evidence-tier", promote_basis, None),
        ("diff-coverage/blind-strategy", "diff-coverage", blind_strategy,
         lambda r: r.get("type") == "strategy"),
        # These two need a record that actually states a latency delta: handed the
        # first record in the store they found nothing to corrupt and skipped,
        # and a skipped case proves nothing.
        ("unit-normalization/1000x", "unit-normalization", break_unit,
         has_latency_metric),
        ("wiki-overlap/known-collision", "wiki-overlap", collide, None),
        ("wiki-overlap/id-already-committed", "wiki-overlap", take_committed_id,
         None),
        ("anonymization/home-path", "anonymization", leak_path, None),
        ("no-fabrication/invented-pct", "no-fabrication", fabricate,
         has_latency_metric),
        ("verbatim/paraphrased-snippet", "verbatim", paraphrase, None),
    ]

    rc = 0
    for name, gate, mutate, need in cases:
        f, rec = _first(records, need or (lambda r: True))
        if rec is None:
            print("  %-34s SKIP (no suitable record)" % name)
            continue
        mrec = copy.deepcopy(rec)
        mpackets = copy.deepcopy(packets)
        mpkt = packet_of(mrec, mpackets)
        if not mutate(mrec, mpkt):
            print("  %-34s SKIP (record lacks the field)" % name)
            continue
        fails = []
        # Only the overlap gate is allowed to see the committed store here: with
        # `wiki_ids` populated, a relations injection could resolve its dangling
        # reference through that store and look green.
        visible = gate == "wiki-overlap"
        ctx = {"set": set_name, "store": store, "records": [(f, mrec)],
               "packets": mpackets, "sets": sets,
               "wiki_ids": wiki_ids if visible else set(),
               "wiki_episodes": wiki_episodes if visible else set(),
               "wiki_versions": wiki_versions,
               "wiki_scanned": wiki_scanned or 1}
        try:
            dict(GATES)[gate](ctx, fails)
        except Exception as e:
            fails.append("gate crashed: %s: %s" % (type(e).__name__, e))
        if fails:
            print("  %-34s FIRED  %s" % (name, fails[0][:86]))
        else:
            print("  %-34s *** DID NOT FIRE -- gate %s is asleep ***"
                  % (name, gate))
            rc = 1
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--all-sets", action="store_true")
    ap.add_argument("--injection-tests", action="store_true")
    args = ap.parse_args()

    if args.injection_tests:
        set_name = c.SET or (sorted(c.SETS)[0] if c.SETS else "")
        if not set_name:
            raise SystemExit("no sets registered in config.SETS")
        print("injection tests against %s" % c.store(set_name))
        return injection_tests(set_name)

    names = sorted(c.SETS) if args.all_sets else [c.require_set()[0]]
    total = 0
    for name in names:
        print("== %s" % name)
        n, _ctx = run(name, verbose=args.verbose)
        total += n
    print("\n%s" % ("ALL GATES PASS" if total == 0 else "GATES FAILED (%d)" % total))
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
