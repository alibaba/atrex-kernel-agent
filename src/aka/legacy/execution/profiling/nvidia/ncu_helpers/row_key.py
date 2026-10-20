# Copyright 2026 Alibaba Group.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Stable per-row keys + cross-capture diff.

VeloQ tags every row with a `key` so two captures can be joined for diffing.
Its leaf templates are content-stable (`|line:<file>:<line>`, `|sass:0x<hex>`,
...), but its prefix `launch:<idx>` is *positional* — it breaks if kernel
launches are reordered or added between captures.

We keep VeloQ's leaf templates verbatim (so the contract matches) but build the
prefix from launch *content* via ncu_utils.launch_identity(), so the same
kernel launch maps to the same key across iterations v0/v1/... — exactly what
atrex's per-iteration diff needs.

Leaf templates (from VeloQ source_metrics/mod.rs, warp_stalls.rs, disasm.rs):
    summary  totals               -> "totals"
    line     <prefix>|line:<file>:<line>
    sass     <prefix>|sass:0x<hex>           (cubin-relative address)
    file     <prefix>|file:<file>
    counter  <prefix>|counter:<name>
    reason   <prefix>|reason:<reason>
    kernel   "kernel|<function_name>"        (disasm; content-based already)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def line_key(prefix, file, line):
    return f"{prefix}|line:{file}:{line}"


def sass_key(prefix, rel_address):
    # A None rel_address (PC that couldn't be made cubin-relative) gets a
    # distinct token rather than collapsing onto 0x0 and colliding in diffs.
    if rel_address is None:
        return f"{prefix}|sass:none"
    return f"{prefix}|sass:0x{int(rel_address):x}"


def file_key(prefix, file):
    return f"{prefix}|file:{file}"


def counter_key(prefix, name):
    return f"{prefix}|counter:{name}"


def reason_key(prefix, reason):
    return f"{prefix}|reason:{reason}"


def kernel_key(function_name):
    return f"kernel|{function_name}"


TOTALS_KEY = "totals"


# --- diff --------------------------------------------------------------------

def _rows_by_key(env):
    """Index an envelope's data.rows[] by their `key`."""
    rows = env.get("data", {}).get("rows", [])
    return {r["key"]: r for r in rows if "key" in r}


def _numeric_fields(row):
    """Flatten a row's numeric leaves into {field_path: value} for delta-ing.

    Handles the two row shapes we diff:
      - {"counters": {name: value|null, ...}}  (source-metrics)
      - {"stalls": {reason: count, ...}, "total_samples": n}  (warp-stalls)
      - any top-level numeric field (total_samples, sass_count, ...)
    """
    out = {}
    for k, v in row.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out[k] = float(v)
        elif isinstance(v, dict):
            for ck, cv in v.items():
                if isinstance(cv, (int, float)) and not isinstance(cv, bool):
                    out[f"{k}.{ck}"] = float(cv)
    return out


def diff_captures(env_a, env_b):
    """Join two envelopes by row key and return a list of per-key deltas.

    Each entry: {key, status: added|removed|changed|same, deltas: {field: {a,b,delta}}}.
    Rows are matched purely on the (content-derived) key, so launch reordering
    between captures does not misalign them.
    """
    a = _rows_by_key(env_a)
    b = _rows_by_key(env_b)
    out = []
    for key in sorted(set(a) | set(b)):
        ra, rb = a.get(key), b.get(key)
        if ra is None:
            out.append({"key": key, "status": "added",
                        "deltas": {f: {"a": None, "b": v, "delta": v}
                                   for f, v in _numeric_fields(rb).items()}})
            continue
        if rb is None:
            out.append({"key": key, "status": "removed",
                        "deltas": {f: {"a": v, "b": None, "delta": -v}
                                   for f, v in _numeric_fields(ra).items()}})
            continue
        fa, fb = _numeric_fields(ra), _numeric_fields(rb)
        deltas = {}
        for f in sorted(set(fa) | set(fb)):
            va, vb = fa.get(f), fb.get(f)
            d = (vb or 0.0) - (va or 0.0)
            if va != vb:
                deltas[f] = {"a": va, "b": vb, "delta": d}
        out.append({"key": key,
                    "status": "changed" if deltas else "same",
                    "deltas": deltas})
    return out


def format_diff(diff, *, top=40, sort_field=None):
    """Render a diff (from diff_captures) as a readable text table.

    Ranks changed/added/removed rows by the magnitude of `sort_field` (or the
    largest absolute delta on the row when sort_field is None)."""
    interesting = [d for d in diff if d["status"] != "same"]

    def _rank(d):
        if sort_field and sort_field in d["deltas"]:
            return abs(d["deltas"][sort_field]["delta"])
        if not d["deltas"]:
            return 0.0
        return max(abs(x["delta"]) for x in d["deltas"].values())

    interesting.sort(key=_rank, reverse=True)
    lines = [f"# diff: {len(interesting)} changed rows "
             f"(of {len(diff)} joined), top {min(top, len(interesting))}"]
    for d in interesting[:top]:
        lines.append(f"\n[{d['status']}] {d['key']}")
        for f, x in sorted(d["deltas"].items(),
                           key=lambda kv: abs(kv[1]["delta"]), reverse=True):
            lines.append(f"    {f:<48} {x['a']} -> {x['b']}  (Δ {x['delta']:+g})")
    return "\n".join(lines) + "\n"


def _load(path):
    return json.loads(Path(path).read_text())


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Diff two atrex ncu evidence envelopes by stable row key.")
    ap.add_argument("--a", required=True, help="baseline envelope JSON (e.g. v0)")
    ap.add_argument("--b", required=True, help="new envelope JSON (e.g. v1)")
    ap.add_argument("--output", help="write text diff here (default: stdout)")
    ap.add_argument("--json", action="store_true", help="emit raw diff JSON")
    ap.add_argument("--sort-field", default=None,
                    help="rank rows by |delta| of this field (e.g. total_samples)")
    ap.add_argument("--top", type=int, default=40)
    args = ap.parse_args(argv)

    diff = diff_captures(_load(args.a), _load(args.b))
    if args.json:
        text = json.dumps(diff, indent=2, default=str)
    else:
        text = format_diff(diff, top=args.top, sort_field=args.sort_field)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text)
        print(f"diff -> {args.output}")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
