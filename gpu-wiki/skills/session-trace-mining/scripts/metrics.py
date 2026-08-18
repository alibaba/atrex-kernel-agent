#!/usr/bin/env python3
"""Reading numbers out of transcript text without inventing any.

Two rules decide every design choice here.

  A unit is never guessed. ncu CSV reports ns, the decode tables report ms, the
  prefill log lines report us, and all three appear in one session. A number
  whose block states no unit is dropped. Guessing wrong publishes a 1000x error
  as a fact.

  A magnitude is never converted into a claim. This module extracts what the
  text says, tagged with where it said it; deciding what it means about a version
  is `ingest.py`'s job, and normalising the sign is `delta_from` alone.
"""
import json
import re

# ------------------------------------------------------------- unit handling

_UNIT_US = {"ns": 1e-3, "us": 1.0, "ms": 1e3, "s": 1e6}


def to_us(value, unit):
    u = (unit or "").lower().replace("µ", "u").replace("μ", "u")
    if u not in _UNIT_US:
        return None
    return float(value) * _UNIT_US[u]


# Seconds are deliberately absent. No kernel metric in these corpora is reported
# in seconds; a bare `35.5s` is always a wall-clock duration -- a pytest suite
# time ("2 passed in 35.5s") or the harness preamble -- and admitting it produced
# a 35-second "latency" A/B against a 25-second one on the first run of a set.
TIME_UNIT = r"(?:ns|us|µs|μs|ms)"

# ------------------------------------------------------------- metric shapes

# `gpu__time_duration.sum` in an ncu --csv dump: the unit is in its own column,
# and it is ns, which is the single most dangerous unit in this corpus.
NCU_CSV_RE = re.compile(
    r'"(gpu__time_duration\.sum|[a-z0-9_]+__[a-z0-9_.]+)"\s*,\s*"(ns|us|ms|%)"'
    r'\s*,\s*"([\d.]+)"')

# `S= 8192 static=    546.67 us clc=    485.81 us speedup=1.1253x`
LABELLED_TIME_RE = re.compile(
    r"(?P<label>[A-Za-z][\w+.\-]*)\s*=\s*(?P<val>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>" + TIME_UNIT + r")\b")
# A label that states how the benchmark was configured, never what it measured.
# `triton.testing.do_bench(warmup=50ms, rep=300ms)` was being read as a two-sided
# A/B between a 50ms "variant" and a 300ms one.
CONFIG_LABEL_RE = re.compile(
    r"^(warmup|warm_up|rep|reps|repeat|timeout|yield_time_ms|max_output_tokens"
    r"|sleep|interval|duration|budget|deadline|window|every)$", re.I)
SPEEDUP_RE = re.compile(
    r"(?:speedup|speed-up)\s*[=:]?\s*(\d+(?:\.\d+)?)\s*x", re.I)
BARE_SPEEDUP_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*x\b")

# `39.51->38.63us`, `5.4213→5.4104us`, `15.638us -> 21.214us`
ARROW_RE = re.compile(
    r"(?P<a>\d+(?:\.\d+)?)\s*(?P<ua>" + TIME_UNIT + r")?\s*(?:->|→|=>)\s*"
    r"(?P<b>\d+(?:\.\d+)?)\s*(?P<ub>" + TIME_UNIT + r")")

# A signed percentage, with the words that fix its direction when present.
PCT_RE = re.compile(r"(?P<sign>[-+])?(?P<val>\d+(?:\.\d+)?)\s*%")
REGRESSION_WORD = re.compile(r"\bREGRESS\w*\b", re.I)
IMPROVE_WORD = re.compile(r"\b(?:improv\w*|faster|win|speedup|drops?)\b", re.I)

TFLOPS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*TFLOP/?s", re.I)
BANDWIDTH_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:GB/s|GBps|gbps)", re.I)
# A timing with its unit attached but no label: `430.72 us`, `437.9 us`. The most
# common shape in the codex sets, and safe because the unit is explicit.
BARE_TIME_RE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*(" + TIME_UNIT + r")\b")
# A block that declares its unit once and then prints bare columns:
# `NCU metric: gpu__time_duration.sum; unit: us; median of 3`
BLOCK_UNIT_RE = re.compile(r"\bunit\s*[:=]\s*(" + TIME_UNIT + r")\b", re.I)
# `2048 1cta_clc_lpt 49.280 [49.280, 49.920, 49.184] (320,1,1)` -- the columnar
# NCU digest: a shape column, a mode label, then the median. The label must be
# allowed to start with a digit (`1cta_clc_lpt`, `2cta_clc`) but must contain a
# letter, otherwise a third numeric column would be read as the mode.
COLUMNAR_RE = re.compile(
    r"^\s*(\d+)\s+(\w*[A-Za-z][\w+.\-]*)\s+(\d+(?:\.\d+)?)\s", re.M)
# `48/48 PASS`, `16/16 PASS`, `312 passed`
PASS_FRACTION_RE = re.compile(r"\b(\d+)\s*/\s*(\d+)\s+(PASS|passed)\b", re.I)
SOL_RE = re.compile(r"\b(?:SOL|sol)\s*[:=]?\s*(\d+(?:\.\d+)?)\s*%")
OCCUPANCY_RE = re.compile(
    r"(?:achieved[_ ]occupancy|occupancy)\D{0,12}(\d+(?:\.\d+)?)\s*%", re.I)
REGISTERS_RE = re.compile(
    r"(?:registers?(?:[_ ]per[_ ]thread)?|regs?)\D{0,8}(\d+)\b", re.I)


def block_times(text):
    """Timings from a block that states its unit once, in a header line.

    Some NCU digests are written this way: `unit: us` on one line, then rows
    of `<shape> <mode> <median> [samples]`. Without the header these numbers
    carry no unit and would have to be dropped, so the header is the only thing
    that makes the densest evidence in that set usable at all.
    """
    m = BLOCK_UNIT_RE.search(text)
    if not m:
        return []
    unit = m.group(1)
    out = []
    for shape, label, val in COLUMNAR_RE.findall(text):
        us = to_us(val, unit)
        if us is None:
            continue
        out.append({"value_us": us, "raw_value": float(val), "raw_unit": unit,
                    "label": "%s@%s" % (label, shape), "kind": "block-columnar"})
    return out


def timings(text, limit=400):
    """Every explicitly-united timing in `text`, normalized to us.

    Returns [{value_us, raw_value, raw_unit, label, kind}]. `label` is the token
    the harness printed next to the number, which is what distinguishes the two
    sides of a one-line A/B (`static=` vs `clc=`).

    The four shapes are layered most-specific first and character spans already
    consumed are not re-read, so one number never enters the result twice under
    two different labels.
    """
    out, taken = [], []

    def free(a, b):
        return not any(a < tb and ta < b for ta, tb in taken)

    for m in NCU_CSV_RE.finditer(text):
        name, unit, val = m.group(1), m.group(2), m.group(3)
        if unit == "%":
            continue
        us = to_us(val, unit)
        if us is None:
            continue
        taken.append((m.start(), m.end()))
        out.append({"value_us": us, "raw_value": float(val), "raw_unit": unit,
                    "label": name, "kind": "ncu-csv"})
        if len(out) >= limit:
            return out
    for m in LABELLED_TIME_RE.finditer(text):
        us = to_us(m.group("val"), m.group("unit"))
        if us is None or not free(m.start(), m.end()):
            continue
        if CONFIG_LABEL_RE.match(m.group("label")):
            # A benchmark knob, not a result. Consume the span so the bare
            # extractor below does not pick the same number up unlabelled.
            taken.append((m.start(), m.end()))
            continue
        taken.append((m.start(), m.end()))
        out.append({"value_us": us, "raw_value": float(m.group("val")),
                    "raw_unit": m.group("unit"), "label": m.group("label"),
                    "kind": "labelled"})
        if len(out) >= limit:
            return out
    for m in BARE_TIME_RE.finditer(text):
        us = to_us(m.group(1), m.group(2))
        if us is None or not free(m.start(), m.end()):
            continue
        taken.append((m.start(), m.end()))
        out.append({"value_us": us, "raw_value": float(m.group(1)),
                    "raw_unit": m.group(2), "label": None, "kind": "bare"})
        if len(out) >= limit:
            return out
    for row in block_times(text):
        out.append(row)
        if len(out) >= limit:
            return out
    return out


def markdown_table(text):
    """Markdown tables as (headers, units, rows), each cell carrying its own unit.

    The unit is read per column from that column's own header, with **no
    table-wide fallback**. The fallback was tried and is exactly how a dimension
    column becomes a fake measurement: `| Total KV | 2CTA (ms) | TRTLLM-gen (ms) |`
    has a unit somewhere in the header row, so a sequence length of 65536
    inherited `ms` and was compared against a real latency, reporting a
    -925389% regression. A column whose header names no unit is dropped.
    """
    tables, headers, units, rows = [], None, None, None
    for line in text.splitlines() + [""]:
        if not line.lstrip().startswith("|"):
            if headers and rows:
                tables.append((headers, units, rows))
            headers, units, rows = None, None, None
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if headers is None:
            if not any(re.search(r"[A-Za-z]", c) for c in cells):
                continue
            headers = cells
            units = []
            for h in cells:
                mu = re.search(r"\(?\b(" + TIME_UNIT + r")\b\)?", h, re.I)
                units.append(mu.group(1) if mu else None)
            rows = []
            continue
        if set("".join(cells)) <= set("-: "):
            continue
        row = {}
        for i, cell in enumerate(cells):
            if i >= len(headers):
                break
            m = re.fullmatch(r"(\d+(?:\.\d+)?)", cell)
            if not m or not units[i]:
                continue
            us = to_us(m.group(1), units[i])
            if us is None:
                continue
            row[headers[i]] = {"raw_value": float(m.group(1)),
                               "raw_unit": units[i], "value_us": us}
        if row:
            rows.append(row)
    if headers and rows:
        tables.append((headers, units, rows))
    return tables


def markdown_times(text):
    """Flat list of every united table cell, for density counting."""
    out = []
    for headers, _units, rows in markdown_table(text):
        for row in rows:
            for label, cell in row.items():
                out.append(dict(cell, label=label, kind="md-table"))
    return out


# A column or label that names the thing being improved upon. Ordered by how
# unambiguous it is; `best` last because "Original best" is a baseline but
# "best" alone sometimes labels the winner.
BASELINE_LABEL_RE = re.compile(
    r"\b(static|baseline|original|before|head|reference|official|ref"
    r"|trtllm[\w-]*|trt|fa3|main|prev\w*|old|best)\b", re.I)


def _geomean(values):
    if not values:
        return None
    acc = 1.0
    for v in values:
        if v <= 0:
            return None
        acc *= v
    return acc ** (1.0 / len(values))


def _pick_sides(labels):
    """(baseline, candidate) from a set of column labels.

    A named baseline wins over position, because the corpus prints the columns in
    both orders (`static | clc` but also `Atrex | TRTLLM-gen`). With no named
    baseline, printed order is the convention and is recorded as such so a reader
    can see the comparison rested on it.
    """
    named = [l for l in labels if BASELINE_LABEL_RE.search(l or "")]
    others = [l for l in labels if l not in named]
    if named and others:
        return named[0], others[0], "named-baseline"
    if len(labels) >= 2:
        return labels[0], labels[1], "printed-order"
    return None, None, None


# A label naming an aggregate, or one property of a single measurement, rather
# than a competing configuration. Pairing two of these produced two false
# comparisons in one A/B corpus that a distilling agent had to withdraw:
# `alone` against `sum` (where the sum *is* the two alone timings added) and `gap`
# against `dur` (two properties of one launch). Publishing either is worse than
# dropping it -- it hands a retrieval agent a lever that does not exist.
AGGREGATE_LABEL_RE = re.compile(
    r"^(sum|total|all|overall|avg|average|mean|geomean|median|min|max|gap|dur"
    r"|duration|elapsed|start|end|begin|finish|overhead|wall|cumulative|count"
    r"|n|iters?|iterations?)$", re.I)


def variant_ab(text):
    """Complete comparisons that live inside a single output.

    This is the candidate unit for the codex sets. A benchmark there is never
    re-run with a byte-identical command -- each run is a freshly written inline
    script -- so pairing a "before" run against an "after" run is structurally
    impossible (measured: 61 distinct benchmark identities in one transcript, 0
    repeated). What the harness does print is both sides at once, either on one
    line or as one column per kernel variant. Those are better evidence anyway:
    same script, same GPU, same run.

    Returns [{baseline, candidate, delta_pct, n_rows, side_basis, kind, quote,
    baseline_us, candidate_us}] where delta_pct is positive when the candidate is
    faster.
    """
    out = []

    # ---- one line carrying both sides, e.g. `static= 546.67 us clc= 485.81 us`
    for line in text.splitlines():
        ts = [t for t in timings(line) if t["kind"] == "labelled"]
        by_label = {}
        for t in ts:
            by_label.setdefault(t["label"], t)
        if len(by_label) < 2:
            continue
        base, cand, basis = _pick_sides(list(by_label))
        if not base or not cand:
            continue
        if AGGREGATE_LABEL_RE.match(base) or AGGREGATE_LABEL_RE.match(cand):
            continue
        b_us = by_label[base]["value_us"]
        c_us = by_label[cand]["value_us"]
        delta = delta_from_pair(b_us, c_us)
        if delta is None:
            continue
        sp = SPEEDUP_RE.search(line)
        if sp:
            # The harness stated the ratio; if our reading of which column is the
            # baseline disagrees with it, the comparison is not usable.
            stated = delta_from_speedup(float(sp.group(1)))
            if stated is not None and abs(stated - delta) > 1.0:
                continue
        out.append({"baseline": base, "candidate": cand, "delta_pct": delta,
                    "n_rows": 1, "side_basis": basis, "kind": "inline-variant",
                    "quote": line.strip()[:300],
                    "baseline_us": b_us, "candidate_us": c_us})

    # ---- a table with one column per variant
    for headers, _units, rows in markdown_table(text):
        labels = [h for h in headers if any(h in r for r in rows)]
        if len(labels) < 2:
            continue
        base, cand, basis = _pick_sides(labels)
        if not base or not cand:
            continue
        ratios, pairs = [], []
        for r in rows:
            if base not in r or cand not in r:
                continue
            b_us, c_us = r[base]["value_us"], r[cand]["value_us"]
            if not b_us or not c_us:
                continue
            ratios.append(b_us / c_us)
            pairs.append((b_us, c_us))
        gm = _geomean(ratios)
        if gm is None or len(pairs) < 1:
            continue
        out.append({"baseline": base, "candidate": cand,
                    "delta_pct": (1.0 - 1.0 / gm) * 100.0,
                    "n_rows": len(pairs), "side_basis": basis,
                    "kind": "table-variant",
                    "quote": " | ".join(headers)[:300],
                    "baseline_us": _geomean([p[0] for p in pairs]),
                    "candidate_us": _geomean([p[1] for p in pairs])})

    # ---- a columnar digest that declares its unit once
    bt = block_times(text)
    if bt:
        by_mode = {}
        for row in bt:
            mode = row["label"].split("@")[0]
            by_mode.setdefault(mode, {})[row["label"].split("@")[-1]] = row
        modes = list(by_mode)
        base, cand, basis = _pick_sides(modes)
        if base and cand:
            shared = set(by_mode[base]) & set(by_mode[cand])
            ratios = [by_mode[base][s]["value_us"] / by_mode[cand][s]["value_us"]
                      for s in sorted(shared)
                      if by_mode[cand][s]["value_us"]]
            gm = _geomean(ratios)
            if gm is not None:
                out.append({
                    "baseline": base, "candidate": cand,
                    "delta_pct": (1.0 - 1.0 / gm) * 100.0,
                    "n_rows": len(ratios), "side_basis": basis,
                    "kind": "columnar-variant",
                    "quote": "modes: %s" % ", ".join(sorted(modes))[:280],
                    "baseline_us": _geomean(
                        [by_mode[base][s]["value_us"] for s in sorted(shared)]),
                    "candidate_us": _geomean(
                        [by_mode[cand][s]["value_us"] for s in sorted(shared)])})
    return out


def inline_ab(text):
    """One-line comparisons the harness already paired for us.

    `static= 546.67 us clc= 485.81 us speedup=1.1253x` is a complete A/B with no
    pairing work and no ambiguity about which run is which, so it is the cheapest
    honest candidate in the codex sets.
    """
    out = []
    for line in text.splitlines():
        ts = [t for t in timings(line) if t["kind"] == "labelled"]
        # Two differently-labelled timings on one line, in the same unit family.
        labels = {t["label"].lower() for t in ts}
        if len(ts) < 2 or len(labels) < 2:
            continue
        sp = SPEEDUP_RE.search(line)
        out.append({"line": line.strip()[:300], "sides": ts,
                    "speedup": float(sp.group(1)) if sp else None})
    return out


def pass_fraction(text):
    """(passed, total) from `48/48 PASS`. A partial pass is not a pass.

    Returned as parsed integers rather than the string, so the correctness gate
    compares numbers instead of trusting the word next to them.
    """
    m = PASS_FRACTION_RE.search(text)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def mechanism_metrics(text):
    """occupancy / registers / SOL: why a change worked, not how much it won.

    These belong in `evidence.summary.mechanism_metrics`, never in `worth.gain`
    -- the base schema is explicit that a mechanism number is not a benefit.
    """
    out = {}
    m = OCCUPANCY_RE.search(text)
    if m:
        out["occupancy_pct"] = float(m.group(1))
    m = REGISTERS_RE.search(text)
    if m:
        out["registers_per_thread"] = int(m.group(1))
    m = SOL_RE.search(text)
    if m:
        out["sol_pct"] = float(m.group(1))
    return out


# --------------------------------------------------------------- delta rules

def delta_from_pair(before_us, after_us):
    """Improvement percent for a lower-better metric. Positive == better."""
    if not before_us or before_us <= 0 or after_us is None:
        return None
    return (before_us - after_us) / before_us * 100.0


def delta_from_speedup(x):
    """`speedup=1.1253x` -> 11.13% latency reduction.

    Not `(x-1)*100`: a 2x speedup halves the time, which is a 50% reduction, not
    100%. Getting this backwards would double every large win.
    """
    if not x or x <= 0:
        return None
    return (1.0 - 1.0 / float(x)) * 100.0


def delta_from_text(text):
    """Pull a stated improvement percent out of prose, with its sign fixed.

    Three conventions coexist in this corpus and the sign cannot be taken from
    the character in front of the number:

      `-0.666% all-geo`            negative == latency down == improvement
      `+35.7% REGRESSION vs HEAD`  positive, and the word says it is worse
      `+3.45%` next to `0.027928 -> 0.026997 ms`  a speedup *ratio*, improvement

    So the magnitude comes from the digits and the direction comes from the words
    and from any before/after pair in the same string; a contradiction between
    them is returned as None rather than resolved by preference.
    """
    m = PCT_RE.search(text)
    if not m:
        return None, None
    mag = float(m.group("val"))
    sign_char = m.group("sign")
    regressed = bool(REGRESSION_WORD.search(text))
    improved = bool(IMPROVE_WORD.search(text)) and not regressed

    pair = ARROW_RE.search(text)
    pair_dir = None
    if pair:
        a, b = float(pair.group("a")), float(pair.group("b"))
        ua = pair.group("ua") or pair.group("ub")
        if ua and pair.group("ub"):
            a_us, b_us = to_us(a, ua), to_us(b, pair.group("ub"))
            if a_us and b_us:
                pair_dir = "improve" if b_us < a_us else "regress"

    if regressed:
        direction, basis = "regress", "stated-pct-regression"
    elif pair_dir:
        direction, basis = pair_dir, "arrow-pair"
    elif sign_char == "-":
        direction, basis = "improve", "stated-pct-negative"
    elif improved:
        direction, basis = "improve", "stated-pct-word"
    elif sign_char == "+":
        # `+3.2%` with no word and no pair is ambiguous in this corpus: it means
        # improvement in the commit-subject grammar and regression in the
        # comparison_with_previous grammar. Refuse rather than pick.
        return None, "ambiguous-plus"
    else:
        direction, basis = "improve", "unsigned-pct"

    if pair_dir and regressed and pair_dir != "regress":
        return None, "sign-conflict"
    return (mag if direction == "improve" else -mag), basis


def arrow_pair(text):
    """(before_us, after_us) from `A -> B us`, or (None, None).

    The unit may appear only after the second number (`39.51->38.63us`), in which
    case it governs both -- they are two readings of the same measurement.
    """
    m = ARROW_RE.search(text)
    if not m:
        return None, None
    ub = m.group("ub")
    ua = m.group("ua") or ub
    return to_us(m.group("a"), ua), to_us(m.group("b"), ub)


# ------------------------------------------------- version documents (vN.json)

def extract_version_docs(text):
    """Every `memory/vN.json`-shaped document embedded in `text`.

    Found by anchoring on the `"performance"` key and decoding from the nearest
    enclosing brace, rather than by trying every `{`: these transcripts are
    megabytes wide and a brute-force scan over one is minutes of work for the
    same answer.
    """
    out = []
    for m in re.finditer(r'"performance"\s*:', text):
        start = text.rfind("{", 0, m.start())
        while start != -1:
            try:
                doc, _end = json.JSONDecoder().raw_decode(text[start:])
            except ValueError:
                start = text.rfind("{", 0, start)
                continue
            if isinstance(doc, dict) and "performance" in doc:
                out.append(doc)
            break
    # Same document is often read back several times in one session.
    seen, uniq = set(), []
    for d in out:
        key = json.dumps(d, sort_keys=True)[:2000]
        if key in seen:
            continue
        seen.add(key)
        uniq.append(d)
    return uniq


def summarise_version_doc(doc):
    """The measurement fields of one version document.

    Deliberately the same key names a miner of a checked-out optimisation run
    would produce, because `ladder.select()` reads them and a rename here would
    silently change which versions become milestones.
    """
    perf = doc.get("performance") or {}
    perf = perf if isinstance(perf, dict) else {}
    corr = doc.get("correctness") or {}
    corr = corr if isinstance(corr, dict) else {}
    gate = doc.get("quality_gate") or {}
    gate = gate if isinstance(gate, dict) else {}
    opt = doc.get("optimization") or {}
    if not isinstance(opt, dict):
        opt = {"action_description": str(opt)}
    cmp_prev = perf.get("comparison_with_previous") or {}
    cmp_prev = cmp_prev if isinstance(cmp_prev, dict) else {}

    # A zero geomean is a corrupt measurement, not a perfect one.
    geo = perf.get("latency_us_geomean") or perf.get("latency_us") or None
    if isinstance(geo, str):
        try:
            geo = float(geo)
        except ValueError:
            geo = None
    return {
        "geomean_us": geo or None,
        "by_shape": perf.get("latency_us_by_shape")
                    or perf.get("latency_us_by_bucket") or {},
        "arith_mean_us": perf.get("latency_us_arith_mean"),
        "speedup_vs_ref": perf.get("speedup_vs_ref_geomean"),
        "tflops": perf.get("tflops"),
        "bandwidth_gbps": perf.get("bandwidth_gbps"),
        "latency_delta_text": cmp_prev.get("latency_delta"),
        "correctness_status": corr.get("status"),
        "max_abs_err": corr.get("max_abs_err"),
        "max_rel_err": corr.get("max_rel_err"),
        "correctness_note": corr.get("note"),
        "gate_result": gate.get("result"),
        "gate_failure": gate.get("failure_reason"),
        "action_category": opt.get("action_category"),
        "action_description": opt.get("action_description"),
        "expected_impact": opt.get("expected_impact"),
        "git_commit_hash": doc.get("git_commit_hash"),
        "open_directions": doc.get("open_directions") or [],
        "pitfalls": doc.get("pitfalls_and_fixes") or [],
        "search_log": doc.get("search_log") or [],
        "profile_evidence": doc.get("profile_evidence") or {},
    }
