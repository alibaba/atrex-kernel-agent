#!/usr/bin/env python3
"""Parse one optimisation trace into work/versions.jsonl + work/profiles.jsonl.

A trace records the same run three times over and the three disagree, so each
source is read for exactly the thing it is authoritative about:

  git commit subjects   the kept/reverted verdict. Nothing else can supply it:
                        the latency series is non-monotonic for reasons that have
                        nothing to do with whether a change was adopted.
  memory/<ver>.json     the measurements -- geomean, per-shape latency, accuracy
                        error, and the run's own account of what it changed.
  profiles/<ver>*/      profiler output, and mostly a trap: a capture taken
                        without a --kernel-name filter measures whatever ran
                        first, so every directory is tagged with the kernel it
                        actually measured and the usable ones are whitelisted
                        rather than assumed.

Nothing here is optional-by-guess. Hardware is taken from the trace when the
trace states it, otherwise from RTM_ARCH / RTM_PRODUCT or the registry entry, and
which of the two happened is recorded in `arch_basis`. A trace that states no
hardware and has no configured target fails loudly rather than being filed under
a default.

Usage: RTM_TRACE=<trace dir> python3 ingest.py
"""
import json
import re

import config as c
import families

# ---------------------------------------------------------------- git subjects

# Versions are the unit of work and every commit that belongs to one announces it
# in the subject. A commit without this prefix is bookkeeping (`chore(007):`).
VER_PREFIX = re.compile(r"^(v\d+)\s*:", re.I)

# The verdict. `reverted` is the standard word; `dead-end` appears in subjects
# that predate it.
REVERT_RE = re.compile(r"\breverted\b|\bdead-end recorded\b", re.I)

# Three subject grammars coexist in a long run. Each yields a different subset of
# (before, after, pct), so all three are tried and whatever is found is kept -- a
# missing field stays missing rather than being guessed.
#   late  : `... -> 59.6->59.1us (-0.7%)`
#   mid   : `... committed (+3.2%)`
#   early : `... (+3.2% geomean improvement)`
ARROW_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:->|\u2192|=>)\s*(\d+(?:\.\d+)?)\s*(?:us|\u00b5s|\u03bcs)",
    re.I)
PCT_RE = re.compile(r"\(([-+]?\d+(?:\.\d+)?)\s*%", re.I)
BASELINE_RE = re.compile(
    r"no improvement over\s+(\d+(?:\.\d+)?)\s*(?:us|\u00b5s|\u03bcs)", re.I)

# The compound dead-end list. A minority of reverted subjects use it; the rest
# encode the same thing as prose, which is why splitting is left to an agent and
# this only marks which grammar a subject uses.
DEADEND_LIST_RE = re.compile(r"dead-end recorded:\s*(.+?)(?:\s*Next:|$)",
                             re.I | re.S)
STREAK_RE = re.compile(r"(\d+)(?:st|nd|rd|th)\s+consecutive dead-end", re.I)
NEXT_RE = re.compile(r"\bNext:\s*(.+)$", re.I | re.S)


def parse_subject(subject):
    out = {"reverted": bool(REVERT_RE.search(subject))}
    m = ARROW_RE.search(subject)
    if m:
        out["subject_before_us"] = float(m.group(1))
        out["subject_after_us"] = float(m.group(2))
    m = PCT_RE.search(subject)
    if m:
        out["subject_pct"] = float(m.group(1))
    m = BASELINE_RE.search(subject)
    if m:
        out["subject_baseline_us"] = float(m.group(1))
    m = DEADEND_LIST_RE.search(subject)
    if m:
        out["deadend_blob"] = m.group(1).strip()
        out["deadend_items"] = split_top_level(m.group(1))
    m = STREAK_RE.search(subject)
    if m:
        out["deadend_streak"] = int(m.group(1))
    m = NEXT_RE.search(subject)
    if m:
        out["next_direction"] = m.group(1).strip()
    return out


def split_top_level(blob):
    """Split a dead-end list on commas that are not inside brackets.

    Two boundaries matter. A comma inside parentheses belongs to the item
    (`Bluestein 3x slower (M>=2N-1)` is one dead-end), and a sentence boundary
    ends the list -- what follows is commentary on the whole batch. The sentence
    cut only needs a letter after `. `, not an uppercase one: a decimal never has
    a space after its point, so `54.5 us` is safe, while `block tuning no effect.
    the vendor library dominates ...` must split.
    """
    blob = _first_sentence(blob)
    items, depth, buf = [], 0, []
    for ch in blob:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            items.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    if buf:
        items.append("".join(buf).strip())
    # The boundary can also fall inside an item, so trim each one too.
    items = [_first_sentence(i) for i in items]
    return [i for i in items if len(i) > 3]


def _first_sentence(text):
    return re.split(r"\.\s+(?=[A-Za-z])", text, maxsplit=1)[0].rstrip(". ")


def read_commits():
    """One commit per version, newest kept when a version was committed twice.

    A re-commit of the same version is an amend in spirit: the later subject is
    the one whose verdict stuck.
    """
    sep = "\x1e"
    # The separator leads each record rather than trailing it, so splitting can
    # never strand a field of the final record.
    raw = c.git("log", "--reverse", "--format=%s%%H|%%P|%%aI|%%s%%n%%b" % sep)
    commits, skipped = {}, []
    for blob in raw.split(sep):
        blob = blob.strip()
        if not blob:
            continue
        head, _, body = blob.partition("\n")
        try:
            sha, parents, date, subject = head.split("|", 3)
        except ValueError:
            continue
        m = VER_PREFIX.match(subject)
        if not m:
            skipped.append(subject[:90])
            continue
        ver = m.group(1).lower()
        commits[ver] = {
            "version": ver,
            "sha": sha,
            "parent": parents.split()[0] if parents.strip() else None,
            "date": date,
            "subject": subject,
            "body": body.strip(),
            **parse_subject(subject + "\n" + body),
        }
    return commits, skipped


# ------------------------------------------------------------------- memory/

# A large share of step-record files in a long run are re-measurements or
# candidate variants rather than versions. Admitting them would put the same
# version on the ladder several times with different numbers.
VARIANT_RE = re.compile(r"^(v\d+)([_-].+)$", re.I)


def read_memory():
    versions, variants = {}, []
    mem_dir = c.TRACE / "memory"
    if not mem_dir.is_dir():
        return versions, variants
    for path in sorted(mem_dir.glob("*.json")):
        stem = path.stem
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            variants.append({"file": path.name, "why": "unparseable"})
            continue
        m = VARIANT_RE.match(stem)
        if m:
            variants.append({"file": path.name, "base": m.group(1).lower(),
                             "suffix": m.group(2).lstrip("_-")})
            continue
        if not re.fullmatch(r"v\d+", stem, re.I):
            variants.append({"file": path.name, "why": "not a version name"})
            continue
        versions[stem.lower()] = summarise_memory(data, path.name)
    return versions, variants


def summarise_memory(data, filename):
    perf = data.get("performance") or {}
    corr = data.get("correctness") or {}
    gate = data.get("quality_gate") or {}
    opt = data.get("optimization") or {}
    if not isinstance(opt, dict):
        opt = {"action_description": str(opt)}
    geo = perf.get("latency_us_geomean")
    # A zero geomean is a corrupt measurement, not a perfect one.
    if not geo:
        geo = None
    return {
        "memory_file": filename,
        "geomean_us": geo,
        "by_shape": perf.get("latency_us_by_shape") or {},
        "arith_mean_us": perf.get("latency_us_arith_mean"),
        "speedup_vs_ref": perf.get("speedup_vs_ref_geomean"),
        "correctness_status": corr.get("status"),
        "max_abs_err": corr.get("max_abs_err"),
        "max_rel_err": corr.get("max_rel_err"),
        "gate_result": gate.get("result"),
        "gate_failure": gate.get("failure_reason"),
        "action_category": opt.get("action_category"),
        "action_description": opt.get("action_description"),
        "expected_impact": opt.get("expected_impact"),
        "git_commit_hash": data.get("git_commit_hash"),
        "open_directions": data.get("open_directions") or [],
        "pitfalls": data.get("pitfalls_and_fixes") or [],
        "search_log": data.get("search_log") or [],
        "profile_evidence": data.get("profile_evidence") or {},
    }


# ------------------------------------------------------------------ profiles/

KERNEL_RE = re.compile(r"^Kernel:\s*(.+)$", re.M)
PROFILE_DIR_RE = re.compile(r"^(v\d+)(?:[_-](.+))?$", re.I)


def read_profiles():
    rows = []
    prof_root = c.TRACE / "profiles"
    if not prof_root.is_dir():
        return rows
    for d in sorted(p for p in prof_root.iterdir() if p.is_dir()):
        m = PROFILE_DIR_RE.match(d.name)
        if not m:
            continue
        summary = d / "summary.txt"
        text = summary.read_text(errors="replace") if summary.is_file() else ""
        km = KERNEL_RE.search(text)
        kernel = km.group(1).strip() if km else None
        report = d / "REPORT.md"
        rows.append({
            "dir": d.name,
            "version": m.group(1).lower(),
            "variant": m.group(2),
            "profiled_kernel": kernel,
            # The whitelist condition for the ncu-attribution gate: a capture is
            # usable only when it names a kernel and that kernel is not one of
            # the harness kernels that run around the operator under test.
            "ncu_usable": bool(kernel) and kernel not in c.NON_TARGET_KERNELS,
            "has_summary": summary.is_file(),
            "has_report": report.is_file(),
            "has_metrics": (d / "analysis" / "metrics_key_run.json").is_file(),
            "report_lines": (len(report.read_text(errors="replace").splitlines())
                             if report.is_file() else 0),
        })
    return rows


# ---------------------------------------------------------------------- meta

def read_meta():
    def load(name):
        p = c.TRACE / name
        if not p.is_file():
            return {}
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            return {}

    definition = load("definition.json")
    solution = load("solution.json")
    axes = definition.get("axes") or {}
    constants = {k: v.get("value") for k, v in axes.items()
                 if v.get("type") == "const" and v.get("value") is not None}
    var_axes = [k for k, v in axes.items() if v.get("type") == "var"]

    n_shapes = 0
    wl = c.TRACE / "workload.jsonl"
    if wl.is_file():
        n_shapes = sum(1 for line in wl.read_text().splitlines() if line.strip())

    spec = solution.get("spec") or {}
    op_name = definition.get("name") or c.TRACE.name
    name = families.normalise(op_name)
    slug = families.slugify(name)
    family = families.family_of(name)

    final_src = c.TRACE / "kernel.py"
    final_dsl = detect_dsl(final_src.read_text(errors="replace")
                           if final_src.is_file() else "")
    target = resolve_target(spec, definition, final_dsl)
    return {
        "operator_name": op_name,
        "operator_slug": slug,
        "workload_family": family,
        "description": definition.get("description"),
        "constants": constants,
        "var_axes": var_axes,
        "n_shapes": n_shapes,
        "target_hardware": spec.get("target_hardware") or [],
        "languages": spec.get("languages") or [],
        "final_dsl": final_dsl,
        **target,
        "harness": c.target_from_config()[4],
        "source_repo": c.trace_label(),
        "head_sha": c.head_sha(),
    }


def resolve_target(spec, definition, final_dsl):
    """(vendor, arch, product) plus the basis on which each was decided.

    Detection first, configuration second, and nothing third: a record filed
    under a guessed architecture is worse than no record, because the store's
    hard scope filter will serve it to an agent working on different hardware.
    """
    tokens = list(spec.get("target_hardware") or [])
    for key in ("target_hardware", "gpu", "hardware"):
        value = definition.get(key)
        if isinstance(value, str):
            tokens.append(value)
        elif isinstance(value, list):
            tokens += [t for t in value if isinstance(t, str)]

    detected = None
    for token in tokens:
        detected = c.target_from_token(token)
        if detected:
            break

    cfg_vendor, cfg_arch, cfg_product, cfg_dsl, _harness = c.target_from_config()
    if detected:
        vendor, arch, product = detected
        basis = "detected from the trace (%s)" % ", ".join(tokens[:3])
    elif cfg_arch:
        vendor, arch, product = cfg_vendor or "nvidia", cfg_arch, cfg_product
        basis = "configured (RTM_ARCH / registry entry)"
    else:
        raise SystemExit(
            "this trace states no target hardware that config.TARGET_TABLE "
            "knows (looked at solution.json spec.target_hardware and "
            "definition.json), and no target is configured.\n"
            "Set RTM_ARCH (one of the schema's arch enum) and RTM_PRODUCT, or "
            "register the trace in config.TRACES. Guessing here would file "
            "records under hardware nobody measured.")
    return {
        "vendor": vendor, "arch": arch, "product": product or "any",
        "arch_basis": basis,
        "dsl_default": cfg_dsl or final_dsl,
    }


def detect_dsl(text):
    """Classify one kernel source into a value of the schema's `dsl` enum.

    Gluon is checked before Triton because it is imported *from* Triton
    (`from triton.experimental import gluon`), so a Gluon kernel always also
    mentions Triton and the cheaper test would swallow it. CuTe DSL is checked
    before CUDA for the same reason: it is Python that emits device code.
    """
    if not text:
        return "any"
    if "gluon" in text:
        return "gluon"
    if re.search(r"cutlass\.cute|import cutlass|cute\.jit|cutedsl", text, re.I):
        return "cutedsl"
    if re.search(r"load_inline|__global__|cpp_extension|hipcc", text):
        return "cuda"
    if "triton" in text:
        return "triton"
    return "any"


def dsl_at(sha):
    """The DSL of the kernel as it stood at one commit.

    Per version rather than per trace: a long run can migrate from one DSL to
    another partway through, and a single verdict would mislabel every record on
    one side of the migration.
    """
    if not sha:
        return None
    return detect_dsl(c.git("show", "%s:kernel.py" % sha))


# ---------------------------------------------------------------------- main

def main():
    c.require_trace()
    c.ensure_dirs()

    commits, skipped = read_commits()
    memory, variants = read_memory()
    profiles = read_profiles()
    meta = read_meta()

    prof_by_ver = {}
    for row in profiles:
        prof_by_ver.setdefault(row["version"], []).append(row)

    versions = []
    for ver in sorted(set(commits) | set(memory),
                      key=lambda v: int(re.sub(r"\D", "", v) or 0)):
        commit = commits.get(ver, {})
        mem = memory.get(ver, {})
        prof = prof_by_ver.get(ver, [])
        versions.append({
            "version": ver,
            "n": int(re.sub(r"\D", "", ver) or 0),
            "has_commit": bool(commit),
            "has_memory": bool(mem),
            "dsl": dsl_at(commit.get("sha")) or meta["dsl_default"] or "any",
            "profile_dirs": [p["dir"] for p in prof],
            "ncu_usable_dirs": [p["dir"] for p in prof if p["ncu_usable"]],
            "report_dirs": [p["dir"] for p in prof if p["has_report"]],
            **{k: v for k, v in commit.items() if k != "version"},
            **{k: v for k, v in mem.items() if k != "version"},
        })

    write(c.WORK / "versions.jsonl", versions)
    write(c.WORK / "profiles.jsonl", profiles)
    (c.WORK / "meta.json").write_text(
        json.dumps({"meta": meta, "memory_variants": variants,
                    "non_version_commits": skipped},
                   ensure_ascii=False, indent=1) + "\n")

    usable = [p for p in profiles if p["ncu_usable"]]
    dsl_mix = {}
    for v in versions:
        dsl_mix[v["dsl"]] = dsl_mix.get(v["dsl"], 0) + 1
    print("trace    %s" % c.TRACE)
    print("slug     %s" % c.SLUG)
    print("operator %s -> %s / %s  shapes=%d"
          % (meta["operator_name"], meta["operator_slug"],
             meta["workload_family"], meta["n_shapes"]))
    print("target   %s / %s / %s  (%s)"
          % (meta["vendor"], meta["arch"], meta["product"], meta["arch_basis"]))
    print("dsl      final=%s  per-version=%s" % (meta["final_dsl"], dsl_mix))
    print("versions %d  (commit=%d memory=%d both=%d)"
          % (len(versions), len(commits), len(memory),
             len(set(commits) & set(memory))))
    print("  reverted %d / kept %d"
          % (sum(1 for v in versions if v.get("reverted")),
             sum(1 for v in versions if v.get("has_commit")
                 and not v.get("reverted"))))
    print("  dead-end lists parsed: %d subjects, %d items"
          % (sum(1 for v in versions if v.get("deadend_items")),
             sum(len(v.get("deadend_items") or []) for v in versions)))
    print("step-record variants filtered out: %d" % len(variants))
    print("profiles %d dirs, usable %d (%.0f%%)"
          % (len(profiles), len(usable),
             100.0 * len(usable) / max(1, len(profiles))))
    print("-> %s" % (c.WORK / "versions.jsonl"))


def write(path, rows):
    with open(path, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
