#!/usr/bin/env python3
"""Evidence-density report: what this trace can honestly yield.

Runs before any model is involved, because the shape of the corpus decides the
shape of the product. Three numbers have veto power:

  usable profiler %   how much of the profiler output measured the kernel under
                      test rather than a harness kernel. When it is zero, no
                      record may claim a profiler-backed bottleneck at all.
  citable numbers     how many selected versions carry a geomean. A trace with
                      almost none has value as mechanism and anti-patterns; do
                      not force a gain claim onto every record.
  already covered     what the live store already holds for this operator, and
                      from which trace. If this run is already represented, the
                      increment may be one `confirm` rather than a batch of new
                      records.

Read `reports/<slug>/recon.md` before distilling anything.

Usage: RTM_TRACE=<trace dir> python3 recon.py
"""
import json
import re
from collections import Counter

import config as c
import ladder


def load():
    versions = [json.loads(l) for l in (c.WORK / "versions.jsonl").open()]
    profiles = [json.loads(l) for l in (c.WORK / "profiles.jsonl").open()]
    meta = json.loads((c.WORK / "meta.json").read_text())
    return versions, profiles, meta


# --------------------------------------------------------------- the live store

def store_entries():
    """Index entries of the live store, or [] when there is no store yet.

    The index is read in preference to walking the tree because it carries the
    retrieval layer of all of them in one file; the individual records are opened
    only for the handful that match this operator.
    """
    if c.STORE_INDEX.is_file():
        try:
            return json.loads(c.STORE_INDEX.read_text()).get("records") or []
        except (json.JSONDecodeError, OSError):
            pass
    entries = []
    if not c.STORE_RECORDS.is_dir():
        return entries
    for path in sorted(c.STORE_RECORDS.rglob("*.json")):
        if path.name == "index.json":
            continue
        try:
            rec = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        entries.append({"id": rec.get("id"), "type": rec.get("type"),
                        "episode_key": rec.get("episode_key"),
                        "path": str(path.relative_to(c.KERNEL_WIKI)),
                        "retrieval": rec.get("retrieval") or {}})
    return entries


def matches_operator(entry, slug, family):
    scope = (entry.get("retrieval") or {}).get("scope") or {}
    gen = (entry.get("retrieval") or {}).get("generality") or {}
    return (scope.get("operator_family") == slug
            or gen.get("workload_family") == family
            or ("%s." % slug) in (entry.get("id") or ""))


def existing_coverage(entries, slug, family):
    """What the live store already holds for this operator, by source trace."""
    by_repo = {}
    for entry in entries:
        scope = (entry.get("retrieval") or {}).get("scope") or {}
        if scope.get("operator_family") != slug:
            continue
        path = c.KERNEL_WIKI / (entry.get("path") or "")
        raw = {}
        if path.is_file():
            try:
                raw = (json.loads(path.read_text()).get("evidence")
                       or {}).get("raw") or {}
            except (json.JSONDecodeError, OSError):
                raw = {}
        repo = raw.get("source_repo") or "(unrecorded)"
        by_repo.setdefault(repo, []).append(
            (entry.get("type"), raw.get("version"),
             (entry.get("id") or "").rsplit(".", 1)[-1]))
    return by_repo


def family_coverage(entries, family):
    """Type histogram of everything the store holds for the whole family."""
    hist = Counter()
    for entry in entries:
        gen = (entry.get("retrieval") or {}).get("generality") or {}
        if gen.get("workload_family") == family:
            hist[entry.get("type")] += 1
    return hist


def used_discs(entries, slug, family):
    """Highest id suffix already taken per marker letter, in the live store.

    Two id shapes have to be matched. A strategy id carries the operator slug
    (`nvidia.b200.triton.<slug>.<technique>-s07`), an anti-strategy id carries
    the *family* plus an `anti` segment -- matching only on the slug would miss
    every existing t-number.

    Starting above these keeps the ids merge-ready: `wiki-gate --commit insert`
    refuses an id that already exists, and renumbering after the fact is the
    expensive part.
    """
    maxima = {}
    rx = re.compile(r"-([a-z])(\d+)$")
    for entry in entries:
        rid = entry.get("id") or ""
        if slug not in rid and (".%s.anti." % family) not in rid:
            continue
        m = rx.search(rid)
        if m:
            letter, num = m.group(1), int(m.group(2))
            maxima[letter] = max(maxima.get(letter, 0), num)
    return maxima


def episode_keys(entries, slug):
    keys = Counter()
    for entry in entries:
        scope = (entry.get("retrieval") or {}).get("scope") or {}
        if scope.get("operator_family") == slug and entry.get("episode_key"):
            keys[entry["episode_key"]] += 1
    return keys


# ----------------------------------------------------------------------- report

def main():
    c.require_trace()
    versions, profiles, meta_all = load()
    meta = meta_all["meta"]
    variants = meta_all["memory_variants"]

    milestones, deadends, rejected = ladder.select(versions)
    rmin = ladder.running_min(versions)
    slug = meta["operator_slug"]
    family = meta["workload_family"]

    entries = store_entries()
    covered = existing_coverage(entries, slug, family)
    fam_hist = family_coverage(entries, family)
    discs = used_discs(entries, slug, family)
    keys = episode_keys(entries, slug)

    usable_dirs = [p["dir"] for p in profiles if p["ncu_usable"]]
    usable_kernels = {p["profiled_kernel"] for p in profiles if p["ncu_usable"]}
    kernels = Counter(p["profiled_kernel"] or "(no summary)" for p in profiles)
    dsl_seq = []
    for v in versions:
        if not dsl_seq or dsl_seq[-1][1] != v["dsl"]:
            dsl_seq.append((v["version"], v["dsl"]))

    reverted = [v for v in versions if v.get("reverted")]
    list_form = [v for v in reverted if v.get("deadend_items")]
    prose_form = [v for v in reverted if not v.get("deadend_items")]
    n_items = sum(len(v["deadend_items"]) for v in list_form)
    with_geo = [m for m in milestones if m.get("geomean_us")]

    out = []
    w = out.append
    w("# Recon: %s" % meta["operator_name"])
    w("")
    w("Generated by `skills/opt-trace-mining/scripts/recon.py`. Purely "
      "mechanical: no model was involved, and every number below is a count over "
      "the trace or over the live store.")
    w("")
    w("## What this trace is")
    w("")
    w("| | |")
    w("|---|---|")
    w("| operator | `%s` |" % meta["operator_name"])
    w("| slug / family | `%s` / `%s` |" % (slug, family))
    w("| trace label | `%s` |" % meta["source_repo"])
    w("| HEAD | `%s` |" % meta["head_sha"][:12])
    w("| target | %s / %s / %s |" % (meta["vendor"], meta["arch"],
                                     meta["product"]))
    w("| target basis | %s |" % meta["arch_basis"])
    w("| benchmarked shapes | %d |" % meta["n_shapes"])
    w("| fixed axes | %s |" % (", ".join("%s=%s" % kv for kv in
                                         meta["constants"].items()) or "none"))
    w("| variable axes | %s |" % (", ".join(meta["var_axes"]) or "none"))
    w("")
    w("**DSL per version** (classified from `kernel.py` at each commit): "
      + " -> ".join("%s:`%s`" % (v, d) for v, d in dsl_seq))
    w("")

    w("## Version coverage")
    w("")
    w("| source | count |")
    w("|---|---:|")
    w("| versions (commit union step records) | %d |" % len(versions))
    w("| with a commit | %d |" % sum(1 for v in versions if v["has_commit"]))
    w("| with a step record | %d |" % sum(1 for v in versions if v["has_memory"]))
    w("| with both | %d |" % sum(1 for v in versions
                                 if v["has_commit"] and v["has_memory"]))
    w("| kept | %d |" % sum(1 for v in versions
                            if v["has_commit"] and not v.get("reverted")))
    w("| reverted | %d |" % len(reverted))
    w("| profile directories | %d |" % len(profiles))
    w("| filtered re-measurement / variant files | %d |" % len(variants))
    w("")
    if variants:
        sfx = Counter(v.get("suffix") or v.get("why") or "?" for v in variants)
        w("Filtered suffixes: "
          + ", ".join("`%s` x%d" % kv for kv in sfx.most_common()))
        w("")
        w("These are baseline re-measurements and candidate variants, not "
          "versions. Admitting them would put one version on the ladder several "
          "times with different numbers.")
        w("")

    w("## The performance ladder")
    w("")
    ups = downs = 0
    prev = None
    for v in versions:
        g = v.get("geomean_us")
        if not g:
            continue
        if prev is not None:
            if g < prev:
                ups += 1
            elif g > prev:
                downs += 1
        prev = g
    w("The raw series is **not monotonic** and must not be read as a progress "
      "curve: %d steps improve, %d regress. A large regression is normally a "
      "harness or library warm-up artifact rather than a code regression, which "
      "is why only a best-so-far ratchet is used." % (ups, downs))
    w("")
    if rmin:
        w("- **running minimum**: %d points, %.2fus -> %.2fus, **%.2fx**"
          % (len(rmin), rmin[0][1], rmin[-1][1], rmin[0][1] / rmin[-1][1]))
        w("- " + " -> ".join("`%s` %.1f" % (v, g) for v, g in rmin))
        w("")
    w("After the ratchet (`MILESTONE_PCT=%.1f` / `MEGA_PCT=%.0f` / "
      "`REBASELINE_FACTOR=%.0f`, all defined in `ladder.py`):"
      % (ladder.MILESTONE_PCT, ladder.MEGA_PCT, ladder.REBASELINE_FACTOR))
    w("")
    kinds = Counter(m["kind"] for m in milestones)
    w("| outcome | count |")
    w("|---|---:|")
    for k in ("baseline", "mega", "milestone", "final"):
        if kinds.get(k):
            w("| %s | %d |" % (k, kinds[k]))
    w("| **distillable milestones** | **%d** |" % len(milestones))
    w("| distillable dead-end versions | %d |" % len(deadends))
    w("| rejected | %d |" % len(rejected))
    w("")
    if milestones:
        w("| version | kind | geomean us | vs best-so-far | dsl |")
        w("|---|---|---:|---:|---|")
        for m in milestones:
            w("| `%s` | %s | %s | %s | %s |"
              % (m["version"], m["kind"],
                 ("%.2f" % m["geomean_us"]) if m.get("geomean_us") else "-",
                 ("%.2f%%" % m["improve_pct"]) if m.get("improve_pct") else "-",
                 m["dsl"]))
        w("")
    w("%d of %d milestones carry a geomean, so **%d records may claim "
      "`worth.gain.basis=measured`** and the rest must not."
      % (len(with_geo), len(milestones), len(with_geo)))
    w("")
    if rejected:
        rc = Counter(re.sub(r"\(.*\)", "", r).strip() for _v, r in rejected)
        w("Rejection reasons: "
          + ", ".join("%s x%d" % kv for kv in rc.most_common()))
        w("")

    w("## Failed experiments")
    w("")
    w("| | count |")
    w("|---|---:|")
    w("| reverted versions | %d |" % len(reverted))
    w("| of those, list form `dead-end recorded: A, B, C` | %d |" % len(list_form))
    w("| items split out of the list form | %d |" % n_items)
    w("| of those, prose form (an agent has to split it) | %d |" % len(prose_form))
    w("")
    if list_form:
        w("Mechanically split, %.1f dead-ends per subject. Example (`%s`):"
          % (n_items / len(list_form), list_form[-1]["version"]))
        w("")
        for item in list_form[-1]["deadend_items"]:
            w("- %s" % item)
        w("")
    if reverted:
        w("Prose form is %.0f%% of the reverted versions. A regex cannot split "
          "it, so each such version becomes one segment and the agent writes up "
          "its most substantial lever only."
          % (100.0 * len(prose_form) / max(1, len(reverted))))
        w("")
    w("**Every one of these still has to clear the established-fact bar** "
      "(`skills/wiki-gate/references/established-fact-criteria.md`): a checkable "
      "condition plus a mechanism. A version that merely measured no gain is not "
      "negative knowledge and must not become a record.")
    w("")

    w("## Profiler availability (decides whether `bottleneck` may be filled)")
    w("")
    if profiles:
        w("| profiled kernel | directories |")
        w("|---|---:|")
        for name, n in kernels.most_common():
            flag = " (usable)" if name in usable_kernels else ""
            w("| `%s`%s | %d |" % (name, flag, n))
        w("")
        w("**Usable: %d of %d directories (%.0f%%).** A capture taken without a "
          "`--kernel-name` filter measures whatever ran first -- typically the "
          "harness's input-generation kernel. Those metric files are complete and "
          "describe the wrong kernel, so they are actively misleading rather than "
          "merely empty. Non-target kernels for this run: %s."
          % (len(usable_dirs), len(profiles),
             100.0 * len(usable_dirs) / max(1, len(profiles)),
             ", ".join("`%s`" % k for k in c.NON_TARGET_KERNELS) or "none"))
        w("")
        if usable_dirs:
            w("Whitelist (the ncu-attribution gate accepts only these):")
            w("")
            for d in usable_dirs:
                k = next(p["profiled_kernel"] for p in profiles if p["dir"] == d)
                w("- `%s` -> `%s`" % (d, k))
            w("")
        w("%d directories carry a `REPORT.md`; that is where a run writes its "
          "own root-cause analysis, and it is quotable as `bench-only` evidence "
          "even where the capture is not."
          % sum(1 for p in profiles if p["has_report"]))
    else:
        w("No `profiles/` directory. **No record may claim "
          "`bottleneck_evidence.basis=profiler`**; `payload.problem.bottleneck` "
          "and `retrieval.signals.bottleneck` stay null.")
    w("")

    w("## Overlap with the live store")
    w("")
    w("Scanned `%s` (%d records)."
      % (c.STORE_RECORDS.relative_to(c.GPU_WIKI), len(entries)))
    w("")
    if covered:
        w("The store already holds %d record(s) for `%s`, from %d trace(s):"
          % (sum(len(v) for v in covered.values()), slug, len(covered)))
        w("")
        w("| source trace | records | versions |")
        w("|---|---:|---|")
        for repo, rows in sorted(covered.items()):
            vs = sorted({r[1] for r in rows if r[1]},
                        key=lambda x: int(re.sub(r"\D", "", x) or 0))
            w("| `%s` | %d | %s |" % (repo, len(rows), ", ".join(vs) or "-"))
        w("")
        mine = meta["source_repo"]
        if mine in covered:
            w("**This trace is already represented** (`%s`). Expect little "
              "increment: check version by version, and prefer "
              "`wiki-gate --commit confirm` over inserting a near-duplicate."
              % mine)
        else:
            w("**This trace (`%s`) is not among them**, so it is an independent "
              "run of the same operator. Its percentages are relative to its own "
              "baseline, so set `worth.gain.comparable=false` and say so in "
              "`worth.gain.note`." % mine)
    else:
        w("The store holds nothing for `%s`; the namespace is empty." % slug)
    w("")
    if keys:
        w("Episode keys already present for this operator (an incoming record "
          "with the same key is a rediscovery, and `wiki-gate --match` will say "
          "so):")
        w("")
        for key, n in keys.most_common(12):
            w("- `%s`%s" % (key, "" if n == 1 else " x%d" % n))
        w("")
    if fam_hist:
        w("Family `%s` overall: %s." % (family, ", ".join(
            "%s=%d" % kv for kv in sorted(fam_hist.items()))))
        w("")
    w("Highest id suffix already used, per marker letter: %s. `partition.py` "
      "allocates above these."
      % (", ".join("`%s`->%02d" % kv for kv in sorted(discs.items())) or "none"))
    w("")

    w("## Conclusion")
    w("")
    est_anti = n_items + len(prose_form)
    w("- Expected yield: **%d strategy** (milestones minus the baseline) + "
      "**up to %d anti-strategy** (%d list items + %d prose versions, before the "
      "established-fact bar) + **up to %d reference-kernel**."
      % (max(0, len([m for m in milestones if m["kind"] != "baseline"])),
         est_anti, n_items, len(prose_form),
         1 + len([m for m in milestones if m["kind"] == "mega"])))
    w("- `payload.problem.bottleneck` is supportable for %d profile "
      "director%s; everywhere else it stays null."
      % (len(usable_dirs), "y" if len(usable_dirs) == 1 else "ies"))
    w("- `worth.gain.basis` is `measured` for the %d milestone(s) with a geomean "
      "(this trace's own harness ran the numbers), `qualitative` for every "
      "anti-strategy." % len(with_geo))
    w("")

    c.ensure_dirs()
    path = c.REPORTS / "recon.md"
    path.write_text("\n".join(out) + "\n")
    print("wrote %s (%d lines)" % (path, len(out)))
    print("  milestones=%d deadend-versions=%d rejected=%d"
          % (len(milestones), len(deadends), len(rejected)))
    print("  usable profiles %d/%d" % (len(usable_dirs), len(profiles)))
    print("  store: %d records scanned, %d already cover this operator"
          % (len(entries), sum(len(v) for v in covered.values())))
    print("  est records: %d strategy + <=%d anti"
          % (len([m for m in milestones if m["kind"] != "baseline"]), est_anti))


if __name__ == "__main__":
    main()
