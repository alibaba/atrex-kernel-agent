#!/usr/bin/env python3
"""Turn versions into segments: one segment becomes one record.

Three shapes come out of here.

  strategy         one per ratchet milestone. Carries code, so it needs a commit.
  anti-strategy    one per *dead-end*, not one per reverted version. A single
                   reverted commit routinely lists several unrelated failed
                   levers ("composite transform 2x slower, direct path 2x
                   slower, block tuning no effect"); keeping them together
                   produces a record that matches three queries and answers
                   none of them.
  reference-kernel the final kernel, plus a snapshot for each mega milestone.

Ids are allocated monotonically per marker letter starting above whatever the
live store already uses for this operator, because `wiki-gate --commit insert`
refuses an id that already exists and renumbering afterwards is the expensive
part.

The marker letter is a namespace, not a claim: `s` strategy, `t` anti-strategy,
`r` reference-kernel. Whether this run's percentages may be compared with another
run's is expressed by `worth.gain.comparable`, which is a property of the
measurement rather than of the id.

Usage: RTM_TRACE=<trace dir> python3 partition.py
"""
import json
import re

import config as c
import ladder
import recon

MARKER_STRATEGY = "s"
MARKER_ANTI = "t"
MARKER_REFK = "r"

# A dead-end item shorter than this says nothing transferable ("no effect").
MIN_DEADEND_CHARS = 12
# Prose-form reverted subjects get one segment each and the agent splits them,
# but only if there is enough prose to split.
MIN_PROSE_CHARS = 80

# The established-fact criteria are imported, never restated: the same regexes
# decide here whether a dead-end is worth a packet and decide in
# `tools/check_kernel_wiki.py` whether the resulting record may enter the store.
# A local copy would drift, and the drift would show up as records that pass the
# triage and are then rejected by the gate after an agent has written them.
# Normative text: skills/wiki-gate/references/established-fact-criteria.md
_facts = c.load_fact_criteria()
CONDITION_RE = _facts.CONDITION_RE
MECHANISM_RE = _facts.MECHANISM_RE
NON_MECHANISM_RE = _facts.NON_MECHANISM_RE

# Dead-ends whose prose does not read like a fact. They are still emitted, with
# the reason attached, because a regex must narrow and never judge: dropping on
# this check discards genuine facts whose wording is unusual. The agent resolves
# the flag; the schema and the established-fact gate enforce the outcome.
FLAGGED_DEADENDS = []


def fact_candidate(text):
    """Can this dead-end become an established fact? -> (bool, reason)."""
    if not text or len(text) < MIN_DEADEND_CHARS:
        return False, "too short to carry a claim"
    if NON_MECHANISM_RE.search(text) and not MECHANISM_RE.search(text):
        return False, "reports a measurement, names no cause"
    if not CONDITION_RE.search(text) and not MECHANISM_RE.search(text):
        return False, "no checkable condition and no cause"
    return True, ""


def slugify(text, limit=48):
    """Lowercase hyphenated slug, danglers removed.

    Version ids and hashes are stripped rather than substituted: in an id
    "an earlier step" would be noise, so the token simply goes away.
    """
    text = re.sub(r"(?<![\w.])[vV]\d+(?![\w.])", " ", text or "")
    text = re.sub(r"\b[0-9a-f]{7,40}\b", " ", text)
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    if len(slug) > limit:
        slug = slug[:limit].rstrip("-")
    return slug or "unnamed"


# Boilerplate that carries no technique: the verdict and its bookkeeping.
BOILERPLATE_RE = re.compile(
    r"reverted|dead[- ]end recorded|no improvement over[^,;]*"
    r"|flat[- ]within[- ]noise|\d+(?:st|nd|rd|th)\s+consecutive[^,;]*"
    r"|committed|geomean improvement|baseline", re.I)


def technique_slug(v, fallback):
    """Name the lever, as specifically as the trace allows.

    `optimization.action_category` looks like the obvious key but is a coarse,
    uncontrolled vocabulary in practice: one category covers several unrelated
    milestones, and its synonyms are one concept. The description names the
    actual change, so it wins; the category is only the fallback.
    """
    desc = strip_boilerplate(fallback)
    if len(desc) >= 12:
        return slugify(desc)
    cat = (v.get("action_category") or "").strip()
    if cat and cat.lower() not in ("baseline", "exhaustive_search"):
        return slugify(cat)
    return slugify(fallback, 48)


def strip_boilerplate(text):
    text = BOILERPLATE_RE.sub(" ", text or "")
    text = re.sub(r"^\W*", "", text)
    text = re.sub(r"\(\s*\)|\[\s*\]", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" -\u2014:;,.")
    return drop_leading_numerals(text)


def drop_leading_numerals(text):
    """Remove leading tokens with no letters in them.

    Stripping the verdict words out of `reverted (flat-within-noise, +0.04%)`
    leaves a bare percentage, which slugifies to `0-04` and names nothing.
    """
    tokens = text.split()
    while tokens and not re.search(r"[A-Za-z]", tokens[0]):
        tokens.pop(0)
    return " ".join(tokens)


# In a reverted subject the verdict comes first and the content follows an
# em-dash or the `dead-end recorded:` marker.
CONTENT_SPLIT_RE = re.compile(r"dead[- ]end recorded\s*:\s*|\s+[\u2014\u2013-]{1,2}\s+")


def anti_technique(v, item):
    """Name a failed lever from whichever field actually describes it.

    Some reverted subjects say nothing but the verdict
    (`reverted (flat-within-noise, +1.4%) - dead-end recorded`), so the subject
    is only the first candidate. The failure reason and the curated pitfall are
    where those runs put the content.
    """
    if item:
        return slugify(strip_boilerplate(item) or item)
    for text in _name_candidates(v):
        named = strip_boilerplate(text)
        if len(named) >= 12:
            return slugify(named)
    cat = (v.get("action_category") or "").strip()
    return slugify(cat) if cat else "unnamed-dead-end"


def _name_candidates(v):
    subject = v.get("subject") or ""
    parts = CONTENT_SPLIT_RE.split(subject, maxsplit=1)
    if len(parts) > 1:
        yield parts[-1]
    yield v.get("action_description") or ""
    # A gate's failure reason states the verdict first and what was attempted
    # after it.
    gate = v.get("gate_failure") or ""
    m = re.search(r"Attempted\s+(.+)", gate, re.S)
    yield m.group(1) if m else gate
    for p in first_pitfall(v):
        yield p


def pitfall_fields(p):
    """Normalise one pitfalls_and_fixes entry into (what failed, how it was fixed).

    The entry is usually a dict but a trace also contains bare strings and nulls.
    """
    if isinstance(p, dict):
        text = (p.get("pitfall") or "").strip()
        detail = (p.get("explanation") or "").strip()
        if detail and detail not in text:
            text = ("%s. %s" % (text.rstrip("."), detail)).strip(". ")
        return text, (p.get("fix") or "").strip() or None
    return str(p or "").strip(), None


def first_pitfall(v):
    """The curated pitfall text, if the run recorded one."""
    for p in (v.get("pitfalls") or []):
        if isinstance(p, dict):
            text = p.get("pitfall") or p.get("explanation") or ""
        else:
            text = str(p or "")
        if text.strip():
            yield text
            return


class Discs:
    """Monotonic id suffix allocator, seeded above the live store's maxima."""

    def __init__(self, start):
        self._next = dict(start)

    def take(self, marker):
        n = self._next.get(marker, 0) + 1
        self._next[marker] = n
        return n


def deadend_segments(v, discs, base):
    """One segment per failed lever, flagged when it does not read like a fact."""
    out = []
    items = v.get("deadend_items") or []
    if items:
        for item in items:
            if len(item) < MIN_DEADEND_CHARS:
                continue
            ok, reason = fact_candidate(item)
            if not ok:
                FLAGGED_DEADENDS.append((reason, item[:110]))
            out.append(dict(
                base,
                seg_kind="deadend-item",
                # The claim is already isolated, so the agent is told to write it
                # up rather than to find it.
                deadend_text=item,
                split_by="mechanical",
                technique=anti_technique(v, item),
                fact_precheck="ok" if ok else reason,
                disc=discs.take(MARKER_ANTI),
            ))
        return out

    # Prose form: the subject describes several failures in running text, often
    # numbered `(1)...(4)`. Splitting that reliably needs reading comprehension,
    # so the segment carries the whole subject and the agent is asked to write up
    # one lever -- which still has to state a fact.
    text = ladder.text_of(v)
    if len(text) < MIN_PROSE_CHARS:
        return out
    ok, reason = fact_candidate(text)
    if not ok:
        FLAGGED_DEADENDS.append((reason, text[:110]))
    out.append(dict(
        base,
        seg_kind="deadend-prose",
        deadend_text=None,
        split_by="agent",
        technique=anti_technique(v, None),
        fact_precheck="ok" if ok else reason,
        disc=discs.take(MARKER_ANTI),
    ))
    return out


def main():
    c.require_trace()
    c.ensure_dirs()

    versions = [json.loads(l) for l in (c.WORK / "versions.jsonl").open()]
    profiles = [json.loads(l) for l in (c.WORK / "profiles.jsonl").open()]
    meta = json.loads((c.WORK / "meta.json").read_text())["meta"]

    milestones, deadends, rejected = ladder.select(versions)
    slug = meta["operator_slug"]
    family = meta["workload_family"]

    entries = recon.store_entries()
    taken = recon.used_discs(entries, slug, family)
    discs = Discs(taken)
    # An independent run of an operator the store already covers measures against
    # its own baseline, so its percentages are not comparable with what is there.
    other_traces = [r for r in recon.existing_coverage(entries, slug, family)
                    if r != meta["source_repo"]]
    comparable = not other_traces

    usable_prof = {p["dir"] for p in profiles if p["ncu_usable"]}
    prof_by_ver = {}
    for p in profiles:
        prof_by_ver.setdefault(p["version"], []).append(p)

    def common(v):
        """The fields every segment needs, whatever its shape."""
        dirs = prof_by_ver.get(v["version"], [])
        return {
            "version": v["version"],
            "n": v["n"],
            "sha": v.get("sha"),
            "parent": v.get("parent"),
            "date": v.get("date"),
            "dsl": v["dsl"],
            "operator_slug": slug,
            "workload_family": family,
            "vendor": meta["vendor"],
            "arch": meta["arch"],
            "product": meta["product"],
            "source_repo": meta["source_repo"],
            "comparable": comparable,
            # Only whitelisted directories may back a profiler claim; the rest
            # measured something other than the kernel under test.
            "ncu_dirs": [d["dir"] for d in dirs if d["dir"] in usable_prof],
            "report_dirs": [d["dir"] for d in dirs if d["has_report"]],
        }

    segments = []

    # A baseline is where the run started, not a lever it pulled, so it earns no
    # strategy record. It becomes the `builds_on` of the first real milestone.
    baseline = next((m for m in milestones if m["kind"] == "baseline"), None)
    steps = [m for m in milestones if m["kind"] != "baseline"]

    prev = baseline
    for m in steps:
        base = common(m)
        segments.append(dict(
            base,
            seg_id="strategy-%s" % m["version"],
            record_type="strategy",
            seg_kind="milestone-%s" % m["kind"],
            marker=MARKER_STRATEGY,
            technique=technique_slug(m, m.get("action_description")
                                     or m.get("subject") or ""),
            disc=discs.take(MARKER_STRATEGY),
            improve_pct=m.get("improve_pct"),
            geomean_us=m.get("geomean_us"),
            kind=m["kind"],
            builds_on={
                "version": prev["version"] if prev else None,
                "is_baseline": bool(prev and prev["kind"] == "baseline"),
                "geomean_us": prev.get("geomean_us") if prev else None,
                "description": (prev.get("action_description")
                                if prev else None),
            },
        ))
        prev = m

    for v in deadends:
        base = dict(common(v), record_type="anti-strategy", marker=MARKER_ANTI)
        for i, seg in enumerate(deadend_segments(v, discs, base)):
            seg["seg_id"] = "anti-%s-%02d" % (v["version"], i + 1)
            segments.append(seg)

    covered = {s["version"] for s in segments
               if s["record_type"] == "anti-strategy"}

    # A curated pitfall is negative knowledge the reverted path cannot see: most
    # hang off *kept* versions, where the run shipped the change and separately
    # wrote down what had not worked on the way. Versions that already produced a
    # dead-end segment keep their pitfalls as packet evidence instead, so nothing
    # is counted twice.
    by_version = {v["version"]: v for v in versions}
    for ver in sorted(set(by_version) - covered,
                      key=lambda x: by_version[x]["n"]):
        v = by_version[ver]
        for i, p in enumerate(v.get("pitfalls") or []):
            text, fix = pitfall_fields(p)
            if len(text) < MIN_DEADEND_CHARS:
                continue
            # Same rule as a reverted dead-end: this becomes an anti-strategy, so
            # it has to state a fact. The curated fix counts as evidence, since a
            # pitfall that names how it was worked around usually names the cause.
            ok, reason = fact_candidate(" ".join(x for x in (text, fix) if x))
            if not ok:
                FLAGGED_DEADENDS.append((reason, text[:110]))
            segments.append(dict(
                common(v),
                seg_id="pitfall-%s-%02d" % (ver, i + 1),
                record_type="anti-strategy",
                seg_kind="curated-pitfall",
                marker=MARKER_ANTI,
                split_by="curated",
                deadend_text=text,
                pitfall_fix=fix,
                fact_precheck="ok" if ok else reason,
                technique=slugify(strip_boilerplate(text) or text),
                disc=discs.take(MARKER_ANTI),
            ))

    # Reference kernels: the shipped kernel, plus a snapshot for each mega step
    # that has one. `versions/` normally holds snapshots for kept versions only.
    snap_dir = c.TRACE / "versions"
    megas = [m for m in milestones if m["kind"] == "mega"]
    final = max(milestones, key=lambda m: m["n"], default=None)
    if final:
        segments.append(dict(
            common(final),
            seg_id="refk-final",
            record_type="reference-kernel",
            seg_kind="final-kernel",
            marker=MARKER_REFK,
            technique="reference-impl",
            disc=discs.take(MARKER_REFK),
            kernel_file="kernel.py",
        ))
    for i, m in enumerate(megas, 1):
        snap = snap_dir / ("kernel_%s.py" % m["version"])
        if not snap.is_file():
            continue
        segments.append(dict(
            common(m),
            seg_id="refk-%s" % m["version"],
            record_type="reference-kernel",
            seg_kind="mega-snapshot",
            marker=MARKER_REFK,
            technique="reference-impl-milestone%d" % i,
            disc=discs.take(MARKER_REFK),
            kernel_file="versions/%s" % snap.name,
        ))

    for seg in segments:
        seg["id"] = build_id(seg)
        seg["episode_key"] = build_episode_key(seg)

    dupes = [i for i, n in count(s["id"] for s in segments).items() if n > 1]
    if dupes:
        raise SystemExit("allocator produced duplicate ids: %s" % dupes[:5])
    known = {e.get("id") for e in entries}
    clash = sorted({s["id"] for s in segments} & known)
    if clash:
        raise SystemExit(
            "these ids already exist in the live store, so wiki-gate would "
            "refuse them: %s\nRe-run recon.py: the allocator seeds from the "
            "store's maxima, so this means the store changed underneath."
            % clash[:5])

    with open(c.WORK / "segments.jsonl", "w") as fh:
        for seg in segments:
            fh.write(json.dumps(seg, ensure_ascii=False) + "\n")

    write_report(segments, milestones, deadends, rejected, meta, taken,
                 usable_prof, comparable, other_traces)

    kinds = count(s["record_type"] for s in segments)
    print("segments %d -> %s" % (len(segments), c.WORK / "segments.jsonl"))
    for k in ("strategy", "anti-strategy", "reference-kernel"):
        if kinds.get(k):
            print("  %-16s %d" % (k, kinds[k]))
    split = count(s.get("split_by") for s in segments if s.get("split_by"))
    if split:
        print("  dead-end split: %s" % dict(split))
    if FLAGGED_DEADENDS:
        print("  dead-ends needing an established fact from the trace: %d"
              % len(FLAGGED_DEADENDS))
        for reason, n in sorted(count(r for r, _t in FLAGGED_DEADENDS).items()):
            print("      %-38s %d" % (reason, n))
        for _reason, text in FLAGGED_DEADENDS[:3]:
            print("      e.g. %s" % text)
    print("  gain.comparable=%s%s"
          % (comparable, "" if comparable
             else " (the store already holds this operator from %s)"
                  % ", ".join("`%s`" % r for r in other_traces[:3])))
    print("  id suffixes start above: %s" % (dict(sorted(taken.items())) or "{}"))


def build_id(seg):
    """`<vendor>.<product>.<dsl>.<slug-or-family.anti>.<technique>-<marker><NN>`.

    Anti-strategies key on the family and carry an `anti` segment, matching the
    shape the store already uses, so the two namespaces stay comparable. Five
    dotted segments minimum: the store's `ids` gate requires it.
    """
    if seg["record_type"] == "anti-strategy":
        middle = "%s.anti" % seg["workload_family"]
    else:
        middle = seg["operator_slug"]
    return "%s.%s.%s.%s.%s-%s%02d" % (
        seg["vendor"], seg["product"] or "any", seg["dsl"], middle,
        seg["technique"], seg["marker"], seg["disc"])


def build_episode_key(seg):
    """`<arch>|<dsl>|<operator_family>|<technique>|<level>`.

    The store's merge handle: two versions of the same technique on the same
    operator share it, and `wiki-gate --match` reports an exact key match as a
    rediscovery instead of making an agent compare prose.
    """
    technique = re.sub(r"-+", "_", seg["technique"])
    return "%s|%s|%s|%s|operator" % (seg["arch"], seg["dsl"],
                                     seg["operator_slug"], technique)


def count(iterable):
    out = {}
    for x in iterable:
        out[x] = out.get(x, 0) + 1
    return out


def write_report(segments, milestones, deadends, rejected, meta, taken,
                 usable_prof, comparable, other_traces):
    out = []
    w = out.append
    w("# Partition: %s" % meta["operator_name"])
    w("")
    w("Generated by `partition.py`. One segment becomes one record.")
    w("")
    kinds = count(s["record_type"] for s in segments)
    w("| record type | count | marker |")
    w("|---|---:|---|")
    w("| strategy | %d | `%s` |" % (kinds.get("strategy", 0), MARKER_STRATEGY))
    w("| anti-strategy | %d | `%s` |" % (kinds.get("anti-strategy", 0),
                                         MARKER_ANTI))
    w("| reference-kernel | %d | `%s` |" % (kinds.get("reference-kernel", 0),
                                            MARKER_REFK))
    w("| **total** | **%d** | |" % len(segments))
    w("")
    w("Ids look like "
      "`%s.%s.<dsl>.<slug|family.anti>.<technique>-<marker><NN>` and are "
      "numbered above the store's existing maxima (%s), so `wiki-gate --commit "
      "insert` cannot collide."
      % (meta["vendor"], meta["product"],
         ", ".join("`%s`>%02d" % kv for kv in sorted(taken.items()))
         or "namespace empty"))
    w("")
    w("`worth.gain.comparable` = **%s**%s."
      % (comparable,
         "" if comparable else
         " -- the store already holds this operator from %s, and this run's "
         "percentages are relative to its own baseline"
         % ", ".join("`%s`" % r for r in other_traces)))
    w("")

    w("## strategy: the ratchet milestones")
    w("")
    w("| version | kind | geomean us | improvement | dsl | technique | id tail |")
    w("|---|---|---:|---:|---|---|---|")
    for s in [x for x in segments if x["record_type"] == "strategy"]:
        w("| `%s` | %s | %s | %s | %s | `%s` | `-%s%02d` |"
          % (s["version"], s["kind"],
             ("%.2f" % s["geomean_us"]) if s.get("geomean_us") else "-",
             ("%.2f%%" % s["improve_pct"]) if s.get("improve_pct") else "-",
             s["dsl"], s["technique"], s["marker"], s["disc"]))
    w("")

    w("## anti-strategy: one record per failed lever")
    w("")
    mech = [s for s in segments if s.get("split_by") == "mechanical"]
    agent = [s for s in segments if s.get("split_by") == "agent"]
    curated = [s for s in segments if s.get("split_by") == "curated"]
    w("| source | segments | what it is |")
    w("|---|---:|---|")
    w("| mechanical | %d | the commit subject used the list form "
      "`dead-end recorded: A, B, C`, split on bracket depth and sentence "
      "boundary -- one segment = one failed lever |" % len(mech))
    w("| agent | %d | prose form; a regex cannot split it, so the whole subject "
      "goes to the agent, which writes up the most substantial lever and names "
      "the rest in `lesson` / `would_retry_if` |" % len(agent))
    w("| curated | %d | `pitfalls_and_fixes` entries on versions that produced "
      "no dead-end segment -- mostly **kept** versions, where the run shipped "
      "the change and separately wrote down what had not worked. The reverted "
      "path cannot see this knowledge |" % len(curated))
    w("")
    flagged = [s for s in segments if s.get("fact_precheck") not in (None, "ok")]
    w("%d of %d anti-strategy segments carry a `fact_precheck` other than `ok`. "
      "That is a flag, not a verdict: the agent must find the condition and the "
      "mechanism in the packet's own evidence, and emit **no record** when "
      "neither is there."
      % (len(flagged), kinds.get("anti-strategy", 0)))
    w("")
    if mech:
        w("Mechanically split, examples:")
        w("")
        for s in mech[:5]:
            w("- `%s` -> %s" % (s["version"], s["deadend_text"]))
        w("")
    if curated:
        w("Curated pitfalls, examples:")
        w("")
        for s in curated[:5]:
            w("- `%s` -> %s" % (s["version"], (s["deadend_text"] or "")[:150]))
        w("")

    w("## reference-kernel")
    w("")
    for s in [x for x in segments if x["record_type"] == "reference-kernel"]:
        w("- `%s` (%s) <- `%s`" % (s["id"].rsplit(".", 1)[-1], s["seg_kind"],
                                   s["kernel_file"]))
    w("")

    w("## Evidence availability")
    w("")
    with_ncu = [s for s in segments if s["ncu_dirs"]]
    with_report = [s for s in segments if s["report_dirs"]]
    w("| | segments |")
    w("|---|---:|")
    w("| has a whitelisted profiler capture (`basis=profiler` allowed) | %d |"
      % len(with_ncu))
    w("| has a `REPORT.md` (`basis=bench-only`, root cause quotable) | %d |"
      % len(with_report))
    w("| neither (`bottleneck` must stay null) | %d |"
      % len([s for s in segments
             if not s["ncu_dirs"] and not s["report_dirs"]]))
    w("")
    w("%d whitelisted profile directories exist in total; the ncu-attribution "
      "gate accepts only those." % len(usable_prof))
    w("")

    w("## Rejected versions")
    w("")
    rc = count(re.sub(r"\(.*\)", "", r).strip() for _v, r in rejected)
    w("| reason | count |")
    w("|---|---:|")
    for reason, n in sorted(rc.items(), key=lambda kv: -kv[1]):
        w("| %s | %d |" % (reason, n))
    w("")

    path = c.REPORTS / "partition.md"
    path.write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
