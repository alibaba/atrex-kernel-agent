#!/usr/bin/env python3
"""Turn candidates into segments: one segment becomes one record.

Three shapes come out of here, matching the main store's vocabulary:

  strategy         a change that made something measurably faster and whose code
                   is recoverable from the transcript.
  anti-strategy    a change that did not work, or a pitfall the run wrote down.
                   Deliberately mined: in these corpora a reverted version's
                   root-cause analysis is the densest mechanism text there is,
                   and it exists nowhere else.
  reference-kernel the end state, when the transcript contains a whole file.

For version-ladder sets the milestone selection is `ladder.py` in this skill,
which carries the thresholds together with the measured reasoning behind each. For
A/B sets there is no ladder to ratchet, so the admission rule is stated here
instead.

Ids start above whatever the committed store already uses for this operator, so a
later merge needs no renumbering.

Usage: STM_SET=<name> python3 partition.py
"""
import json
import re
import sys
from collections import Counter

import config as c
import families
import ladder

# Markers keep this store's ids out of the main chain's namespace. `s` is
# reserved for the representative run of an operator, which a session archive is
# not: these are sibling runs, so `gain.comparable` is false.
MARKER_STRATEGY = "g"
MARKER_ANTI = "u"
MARKER_REFK = "k"

# A dead-end shorter than this says nothing transferable ("no effect").
MIN_DEADEND_CHARS = 12
# Prose-form reverted subjects get one segment each and the agent splits them,
# but only if there is enough prose to split.
MIN_PROSE_CHARS = 80
# Below this, an A/B is measurement noise in this corpus: the runs themselves
# call anything under about half a percent "flat within noise".
MIN_AB_PCT = 0.5

# An anti-strategy is admissible only as a FACT: under a checkable condition it
# necessarily fails. These regexes narrow, they never judge -- a segment whose
# prose does not read like a fact is still emitted, carrying the reason, because
# dropping on a regex would discard genuine facts along with the ledgers. Hard
# enforcement is the schema's required established_fact plus the
# "established-fact" gate in tools/check_kernel_wiki.py.
# Normative definition: skills/wiki-gate/references/established-fact-criteria.md
FACT_CONDITION_RE = re.compile(
    r"\b(sm_?\d{2,3}|b200|b300|h100|a100|blackwell|hopper|ampere"
    r"|[MNK]\s*[=<>]\s*\d+|small-[mnk]\b|large-[mnk]\b|tiny-[mnk]\b"
    r"|bf16|fp16|fp8|fp4|fp32|tf32|int8|e4m3|e5m2"
    r"|triton \d|cuda \d{2}|cutlass \d|cudnn|cublas|cufft|gluon"
    r"|when [a-z]|only (?:if|when|on))\b", re.I)

FACT_MECHANISM_RE = re.compile(
    r"\b(because|since|due to|caused by|bound by|limited by|serializ|contention"
    r"|saturat|spill|occupancy|bank conflict|scoreboard|barrier|wave quantization"
    r"|tail effect|launch overhead|exceeds|cannot|unsupported|not supported"
    r"|requires|ignored|no-op|already)\b", re.I)

FACT_NON_MECHANISM_RE = re.compile(
    r"(no improvement (?:found|over)|flat[- ]within[- ]noise|within noise"
    r"|all (?:\w+\s+){0,3}(?:approaches|trials|variants|attempts) "
    r"(?:tested|were|failed|flat)|tested \d+ approaches"
    r"|hw floor (?:confirmed|reconfirmed)|consecutive (?:dead-end|stall|revert))",
    re.I)


def fact_precheck(text):
    """"ok", or why this prose does not yet read like an established fact."""
    if not text:
        return "nothing said: no condition and no cause"
    if FACT_NON_MECHANISM_RE.search(text) and not FACT_MECHANISM_RE.search(text):
        return "reports a measurement, names no cause"
    if not FACT_CONDITION_RE.search(text) and not FACT_MECHANISM_RE.search(text):
        return "no checkable condition and no cause"
    return "ok"

# A label naming a whole implementation or a toolchain release rather than a knob
# of the kernel under study. `Atrex ms` vs `TRTLLM-gen (ms)` is a competitive
# benchmark; `static` vs `clc` is a lever. The first is a fact about two products
# and carries no change an agent could apply, so it only earns a record when a
# verbatim diff links it to the change that produced the win.
IMPL_LABEL_RE = re.compile(
    r"\b(atrex|atre[xX]|trtllm[\w-]*|trt|fa3|fa4|flashinfer|cublas\w*|cutlass"
    r"|official|vllm|torch|dsl\s*[\d.]+|\d+\.\d+\.\d+)\b", re.I)


def is_impl_comparison(row):
    """Both sides name products or releases, not settings of one kernel."""
    base = row.get("baseline_label") or ""
    cand = row.get("candidate_label") or ""
    return bool(IMPL_LABEL_RE.search(base) and IMPL_LABEL_RE.search(cand))

BOILERPLATE_RE = re.compile(
    r"reverted|dead[- ]end recorded|no improvement over[^,;]*"
    r"|flat[- ]within[- ]noise|\d+(?:st|nd|rd|th)\s+consecutive[^,;]*"
    r"|committed|record \+ carry-forward profile|record \+ plan"
    r"|geomean improvement|baseline", re.I)


def slugify(text, limit=48):
    """Lowercase hyphenated slug, version ids and hashes removed.

    They go away rather than being substituted: in an id, "an earlier version"
    would be noise.
    """
    text = re.sub(r"(?<![\w.])[vV]\d+(?![\w.])", " ", text or "")
    text = re.sub(r"\b[0-9a-f]{7,40}\b", " ", text)
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    if len(slug) > limit:
        slug = slug[:limit].rstrip("-")
    return slug or "unnamed"


def strip_boilerplate(text):
    text = BOILERPLATE_RE.sub(" ", text or "")
    text = re.sub(r"^\W*", "", text)
    text = re.sub(r"\(\s*\)|\[\s*\]", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" -—:;,.")
    tokens = text.split()
    # Stripping the verdict out of `reverted (flat-within-noise, +0.04%)` leaves a
    # bare percentage, which slugifies to `0-04` and names nothing.
    while tokens and not re.search(r"[A-Za-z]", tokens[0]):
        tokens.pop(0)
    return " ".join(tokens)


def technique_slug(row, fallback=""):
    """Name the lever as specifically as the corpus allows.

    `action_category` looks like the obvious key but is an uncontrolled
    vocabulary here (one value covers four unrelated milestones, and three
    spellings of "memory access" are one concept), so the description wins and the
    category is only the fallback.
    """
    for text in (row.get("action_description"), row.get("subject"), fallback):
        named = strip_boilerplate(text)
        if len(named) >= 12:
            return slugify(named)
    cat = (row.get("action_category") or "").strip()
    if cat and cat.lower() not in ("baseline", "exhaustive_search"):
        return slugify(cat)
    return slugify(fallback or "unnamed")


def ab_technique(row):
    """Name an A/B by what it compared.

    The candidate label is the change and the baseline is what it replaced, so
    `official static ms` vs `atreX CLC ms` becomes `atrex-clc-vs-official-static`.
    That reads as a technique and stays unique per comparison.
    """
    cand = strip_boilerplate(_clean_label(row.get("candidate_label") or ""))
    base = strip_boilerplate(_clean_label(row.get("baseline_label") or ""))
    if cand and base:
        return slugify("%s vs %s" % (cand, base))
    if row.get("edit_files"):
        stem = row["edit_files"][0].split("/")[-1]
        return slugify("change in %s" % stem)
    return slugify(row.get("bench_identity") or "ab")


def _clean_label(label):
    """Drop the unit parenthetical from a column header."""
    return re.sub(r"\(?\b(?:ns|us|µs|μs|ms)\b\)?", " ", label or "",
                  flags=re.I).strip()


def operator_slug(rows):
    """The operator name the wiki uses, from the run's own directory name.

    Voted across the rows rather than read off the first one: a version with no
    owning session falls back to the project *directory* name, and on one set that
    put a flattened home path -- containing a username -- into every record id.

    Two preferences before majority, each from a name that majority got wrong:

      a `kernel_opt_*` directory is the campaign's own operator directory;
      a name `families.family_of` can classify is far more likely to be an
      operator than a repo or product name -- without it a set whose sessions run
      inside a packaging repo files under that repo's name instead of the kernel
      it was optimising, and its family degrades to `misc`.

    Naming lives in `families.py` inside this skill, so a record's directory never
    depends on a module in another tree.
    """
    names = Counter(r.get("operator_id") for r in rows if r.get("operator_id"))
    kernel_like = [n for n in names if re.search(r"kernel_opt|kernel-opt", n)]
    classified = [n for n in names
                  if families.family_of(families.normalise(n)) != "misc"]
    if kernel_like:
        name = max(kernel_like, key=lambda n: names[n])
    elif classified:
        name = max(classified, key=lambda n: names[n])
    elif names:
        name = names.most_common(1)[0][0]
    else:
        name = "unknown"
    # A project-directory name arrives as `-root-user-proj-kernel-opt-<n>-<op>`.
    name = families.normalise(name)
    return families.slugify(name), families.family_of(name)


class Discs:
    """Monotonic id suffix allocator, per marker."""

    def __init__(self, start=None):
        self._next = dict(start or {})

    def take(self, marker):
        n = self._next.get(marker, 0) + 1
        self._next[marker] = n
        return n


def used_discs(op_slug):
    """Highest suffix already used for this operator in the committed store.

    Starting above it means a later merge of this store into the main one does
    not have to renumber anything.
    """
    taken = {}
    for root in c.COMMITTED_STORES:
        if not root.is_dir():
            continue
        for path in root.rglob("*%s*.json" % op_slug):
            m = re.search(r"-([a-z])(\d+)\.json$", path.name)
            if not m:
                continue
            marker, n = m.group(1), int(m.group(2))
            taken[marker] = max(taken.get(marker, 0), n)
    return taken


def pitfall_fields(p):
    """Normalise one pitfalls_and_fixes entry into (what failed, how it was fixed).

    Usually a dict, but the corpus also contains bare strings and nulls.
    """
    if isinstance(p, dict):
        text = (p.get("pitfall") or "").strip()
        detail = (p.get("why") or p.get("explanation") or "").strip()
        if detail and detail not in text:
            text = ("%s. %s" % (text.rstrip("."), detail)).strip(". ")
        return text, (p.get("fix") or "").strip() or None
    return str(p or "").strip(), None


def common(row, op_slug, family, set_name, meta):
    """The fields every segment needs, whatever its shape."""
    scope = row.get("scope") or {}
    return {
        "set": set_name,
        "unit": row["unit"],
        "version": row["version"],
        "n": row["n"],
        "sha": row.get("sha"),
        "date": row.get("date"),
        "dsl": scope.get("dsl") or row.get("dsl") or "any",
        "product": scope.get("product") or "any",
        "arch": scope.get("arch") or "blackwell",
        "arch_basis": scope.get("arch_basis"),
        "dsl_basis": scope.get("dsl_basis"),
        "operator_id": row.get("operator_id"),
        "operator_slug": op_slug,
        "workload_family": family,
        "rel_path": row.get("owner_rel_path"),
        "sibling_paths": row.get("sibling_paths") or [],
        "session_id": row.get("owner_session"),
        "format": meta["formats"] and list(meta["formats"])[0],
        "cite_lines": row.get("cite_lines") or [],
        "cite_digests": row.get("cite_digests") or {},
        "number_tiers": row.get("number_tiers") or {},
        "dedup_key": row.get("dedup_key"),
        "diff_coverage": row.get("diff_coverage"),
        "geomean_us": row.get("geomean_us"),
        "correctness_status": row.get("correctness_status"),
        "gate_result": row.get("gate_result"),
    }


def build_id(seg):
    """`nvidia.<product>.<dsl>.<slug|family.anti>.<technique>-<marker><NN>`.

    Anti-strategies key on the family rather than the operator, matching the shape
    the main store already uses, so the two namespaces stay comparable.
    """
    if seg["record_type"] == "anti-strategy":
        middle = "%s.anti" % (seg.get("workload_family") or "misc")
    else:
        middle = seg["operator_slug"]
    return "%s.%s.%s.%s.%s-%s%02d" % (
        "nvidia", seg.get("product") or "any", seg["dsl"], middle,
        seg["technique"], seg["marker"], seg["disc"])


def partition_ladder(rows, meta, set_name):
    milestones, deadends, rejected = ladder.select(rows)
    by_ver = {r["version"]: r for r in rows}
    segments = []
    op_slug, family = operator_slug(rows)
    discs = Discs(used_discs(op_slug))

    # A baseline is where the run started, not a lever it pulled, so it earns no
    # strategy record; it becomes the `builds_on` of the first real milestone.
    baseline = next((m for m in milestones if m["kind"] == "baseline"), None)
    prev = baseline
    for m in [x for x in milestones if x["kind"] != "baseline"]:
        seg = dict(common(m, op_slug, family, set_name, meta),
                   seg_id="strategy-%s" % m["version"],
                   record_type="strategy",
                   seg_kind="milestone-%s" % m["kind"],
                   marker=MARKER_STRATEGY,
                   technique=technique_slug(m),
                   disc=discs.take(MARKER_STRATEGY),
                   improve_pct=m.get("improve_pct"),
                   before_us=(prev or {}).get("geomean_us"),
                   after_us=m.get("geomean_us"),
                   kind=m["kind"],
                   subject=m.get("subject"),
                   action_description=m.get("action_description"),
                   expected_impact=m.get("expected_impact"),
                   profile_evidence=m.get("profile_evidence") or {},
                   open_directions=m.get("open_directions") or [],
                   n_shapes=len(m.get("by_shape") or {}) or None,
                   builds_on={
                       "version": prev["version"] if prev else None,
                       "is_baseline": bool(prev and prev["kind"] == "baseline"),
                       "geomean_us": prev.get("geomean_us") if prev else None,
                       "description": (prev.get("action_description")
                                       or prev.get("subject")) if prev else None,
                   })
        segments.append(seg)
        prev = m

    covered = set()
    for v in deadends:
        covered.add(v["version"])
        text = " ".join(str(x) for x in (v.get("subject") or "",
                                        v.get("gate_failure") or "",
                                        v.get("action_description") or "") if x)
        if len(text) < MIN_PROSE_CHARS:
            rejected.append((v["version"], "reverted, nothing said"))
            continue
        segments.append(dict(
            common(v, op_slug, family, set_name, meta),
            seg_id="anti-%s" % v["version"],
            record_type="anti-strategy",
            seg_kind="deadend-prose",
            fact_precheck=fact_precheck(text),
            marker=MARKER_ANTI,
            split_by="agent",
            deadend_text=None,
            technique=technique_slug(v),
            disc=discs.take(MARKER_ANTI),
            subject=v.get("subject"),
            action_description=v.get("action_description"),
            gate_failure=v.get("gate_failure"),
            profile_evidence=v.get("profile_evidence") or {},
            open_directions=v.get("open_directions") or []))

    # A curated pitfall is negative knowledge the reverted path cannot see: most
    # hang off *kept* versions, where the run shipped a change and separately
    # wrote down what had not worked on the way there.
    for ver in sorted(set(by_ver) - covered, key=lambda x: by_ver[x]["n"]):
        row = by_ver[ver]
        for i, p in enumerate(row.get("pitfalls") or []):
            text, fix = pitfall_fields(p)
            if len(text) < MIN_DEADEND_CHARS:
                continue
            segments.append(dict(
                common(row, op_slug, family, set_name, meta),
                seg_id="pitfall-%s-%02d" % (ver, i + 1),
                record_type="anti-strategy",
                seg_kind="curated-pitfall",
                marker=MARKER_ANTI,
                split_by="curated",
                deadend_text=text,
                pitfall_fix=fix,
                technique=slugify(strip_boilerplate(text) or text),
                disc=discs.take(MARKER_ANTI),
                subject=row.get("subject")))
    return segments, milestones, deadends, rejected


def partition_ab(rows, meta, set_name):
    segments, rejected = [], []
    op_slug, family = operator_slug(rows)
    discs = Discs(used_discs(op_slug))
    for row in rows:
        pct = row.get("improve_pct_raw") or 0.0
        if row.get("implausible"):
            rejected.append((row["version"], "implausible delta %.0f%%" % pct))
            continue
        if abs(pct) < MIN_AB_PCT:
            rejected.append((row["version"], "within noise (%.2f%%)" % pct))
            continue
        if is_impl_comparison(row) and row.get("diff_coverage") != "full":
            # A competitive benchmark with no change attached: true, but there is
            # nothing here for an agent to apply.
            rejected.append((row["version"],
                             "implementation comparison without a linked change"))
            continue
        base = common(row, op_slug, family, set_name, meta)
        base.update({"improve_pct": round(pct, 3),
                     "before_us": row.get("before_us"),
                     "after_us": row.get("after_us"),
                     "n_shapes": row.get("n_shapes"),
                     "delta_basis": row.get("delta_basis"),
                     "side_basis": row.get("side_basis"),
                     "quote": row.get("quote"),
                     "baseline_label": row.get("baseline_label"),
                     "candidate_label": row.get("candidate_label"),
                     "edit_files": row.get("edit_files") or [],
                     "linked_by": row.get("linked_by") or [],
                     "bench_identity": row.get("bench_identity")})
        # A pairing chosen by print order is not known to compare alternatives
        # of the same work, and no gate can establish that it does. A distilling
        # agent found two: a three-phase CUDA-event breakdown of one call read as
        # phase-vs-phase (78.9%), and an approximate path against a vendor library
        # computing a different result (-99.5%). Both passed every gate. So such a
        # candidate may not be a strategy -- it goes to the weaker type carrying
        # `claims_no_gain`, which the brief turns into `worth.gain.kind = "none"`.
        if row.get("side_basis") == "printed-order":
            segments.append(dict(
                base, seg_id="unverified-%s" % row["version"],
                record_type="anti-strategy", seg_kind="unverified-pairing",
                marker=MARKER_ANTI, split_by="curated", deadend_text=None,
                claims_no_gain=True,
                # A printed-order pairing is an unverified claim, not a failure:
                # there is no bad result and no cause, so it cannot satisfy the
                # established-fact rule from this evidence alone.
                fact_precheck="unverified pairing: no failure and no mechanism",
                technique=ab_technique(row), disc=discs.take(MARKER_ANTI)))
            continue
        if pct > 0:
            if row.get("diff_coverage") == "blind":
                # The numbers are real but the code is not recoverable, so this
                # cannot carry an implementation. It is still worth a record about
                # the configuration choice, filed as the weaker type.
                segments.append(dict(
                    base, seg_id="config-%s" % row["version"],
                    record_type="anti-strategy", seg_kind="config-comparison",
                    marker=MARKER_ANTI, split_by="curated",
                    deadend_text=None,
                    # The measured delta is positive; only the code is missing.
                    # Nothing here says a lever necessarily fails, so it cannot
                    # be filed as negative knowledge.
                    fact_precheck="config comparison with a positive delta: "
                                  "no failure to explain",
                    technique=ab_technique(row), disc=discs.take(MARKER_ANTI)))
                continue
            segments.append(dict(
                base, seg_id="strategy-%s" % row["version"],
                record_type="strategy", seg_kind="ab-improvement",
                marker=MARKER_STRATEGY, kind="milestone",
                technique=ab_technique(row),
                disc=discs.take(MARKER_STRATEGY),
                builds_on={"version": None, "is_baseline": True,
                           "geomean_us": row.get("before_us"),
                           "description": row.get("baseline_label")}))
        else:
            segments.append(dict(
                base, seg_id="anti-%s" % row["version"],
                record_type="anti-strategy", seg_kind="ab-regression",
                marker=MARKER_ANTI, split_by="agent", deadend_text=None,
                fact_precheck=fact_precheck(row.get("subject") or ""),
                technique=ab_technique(row), disc=discs.take(MARKER_ANTI)))
    return segments, [], [], rejected


def main():
    set_name, cfg, _root = c.require_set()
    c.ensure_dirs(set_name)
    work = c.work(set_name)
    rows = [json.loads(l) for l in (work / "versions.jsonl").open()]
    meta = json.loads((work / "meta.json").read_text())
    if not rows:
        raise SystemExit("no candidates in %s; run ingest.py first"
                         % (work / "versions.jsonl"))

    if meta["unit"] == "version-ladder":
        segments, milestones, deadends, rejected = partition_ladder(
            rows, meta, set_name)
    else:
        segments, milestones, deadends, rejected = partition_ab(
            rows, meta, set_name)

    for seg in segments:
        seg["id"] = build_id(seg)
    dupes = [i for i, n in Counter(s["id"] for s in segments).items() if n > 1]
    if dupes:
        raise SystemExit("allocator produced duplicate ids: %s" % dupes[:5])

    with open(work / "segments.jsonl", "w") as fh:
        for seg in segments:
            fh.write(json.dumps(seg, ensure_ascii=False) + "\n")

    write_report(set_name, segments, milestones, deadends, rejected, meta, rows)

    kinds = Counter(s["record_type"] for s in segments)
    print("segments   %d -> %s" % (len(segments), work / "segments.jsonl"))
    for k in ("strategy", "anti-strategy", "reference-kernel"):
        if kinds.get(k):
            print("  %-16s %d" % (k, kinds[k]))
    print("rejected   %d" % len(rejected))
    for why, n in Counter(re.sub(r"[\d.]+", "N", r) for _v, r in rejected
                          ).most_common(6):
        print("  %-40s %d" % (why[:40], n))
    return 0


def write_report(set_name, segments, milestones, deadends, rejected, meta, rows):
    w = []
    a = w.append
    kinds = Counter(s["record_type"] for s in segments)
    a("# Partition report: %s" % set_name)
    a("")
    a("Generated by `partition.py`. Each segment corresponds to one record.")
    a("")
    a("| Record type | Count | Marker |")
    a("|---|---:|---|")
    a("| strategy | %d | `%s` |" % (kinds.get("strategy", 0), MARKER_STRATEGY))
    a("| anti-strategy | %d | `%s` |" % (kinds.get("anti-strategy", 0),
                                         MARKER_ANTI))
    a("| reference-kernel | %d | `%s` |" % (kinds.get("reference-kernel", 0),
                                            MARKER_REFK))
    a("| **Total** | **%d** | |" % len(segments))
    a("")
    a("IDs take the form `nvidia.<product>.<dsl>.<slug|family.anti>.<technique>-<marker><NN>`, "
      "numbered starting above the highest value already in the committed record "
      "store so that merging stores later needs no renumbering. The markers are "
      "`%s`/`%s`/`%s` rather than the main chain's `s`: this set is a sibling run, "
      "and `gain.comparable` is false." % (MARKER_STRATEGY, MARKER_ANTI, MARKER_REFK))
    a("")

    if meta["unit"] == "version-ladder":
        a("## strategy: ratchet milestones")
        a("")
        a("For the selection rule see this skill's `scripts/ladder.py`: a version "
          "counts as a milestone only when it beats the best score so far and the "
          "improvement clears the threshold. Step-by-step deltas cannot serve as a "
          "progress curve -- in a corpus like this, improvements and regressions "
          "come in almost equal numbers of steps.")
        a("")
        a("| Version | kind | geomean us | Improvement | dsl | id tail |")
        a("|---|---|---:|---:|---|---|")
        for s in [x for x in segments if x["record_type"] == "strategy"]:
            a("| `%s` | %s | %s | %s | %s | `-%s%02d` |"
              % (s["version"], s.get("kind"),
                 ("%.2f" % s["geomean_us"]) if s.get("geomean_us") else "—",
                 ("%.2f%%" % s["improve_pct"]) if s.get("improve_pct") else "—",
                 s["dsl"], s["marker"], s["disc"]))
        a("")
    else:
        a("## strategy: A/B improvements")
        a("")
        a("| Version | Baseline → candidate | Improvement | Shapes | Source | diff |")
        a("|---|---|---:|---:|---|---|")
        for s in [x for x in segments if x["record_type"] == "strategy"]:
            a("| `%s` | %s → %s | %.2f%% | %s | `%s` | %s |"
              % (s["version"], (s.get("baseline_label") or "?")[:28],
                 (s.get("candidate_label") or "?")[:28], s["improve_pct"],
                 s.get("n_shapes") or "—", s.get("delta_basis"),
                 s.get("diff_coverage")))
        a("")

    a("## anti-strategy: failures and criteria")
    a("")
    split = Counter(s.get("split_by") for s in segments
                    if s["record_type"] == "anti-strategy")
    a("| Split by | Count | Note |")
    a("|---|---:|---|")
    a("| curated | %d | A pitfall or config comparison the run wrote down "
      "itself, already a standalone entry |" % split.get("curated", 0))
    a("| agent | %d | A prose failure account that regex cannot split, handed "
      "to the agent whole to split by meaning |" % split.get("agent", 0))
    a("")
    a("Regression versions are kept deliberately: in this corpus the root-cause "
      "analysis of a `REVERTED` version (`profile_evidence.evidence_chain` + "
      "`pitfalls_and_fixes`) is the densest mechanism knowledge there is, and it "
      "exists nowhere else.")
    a("")

    a("## Candidates rejected")
    a("")
    rc = Counter(re.sub(r"[-\d.]+", "N", r) for _v, r in rejected)
    a("| Reason | Count |")
    a("|---|---:|")
    for why, n in rc.most_common():
        a("| %s | %d |" % (why, n))
    a("")
    a("%d candidates entered in total, %d were rejected, and %d became records."
      % (len(rows), len(rejected), len(segments)))
    a("")

    path = c.reports(set_name) / "partition.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(w) + "\n")


if __name__ == "__main__":
    sys.exit(main())
