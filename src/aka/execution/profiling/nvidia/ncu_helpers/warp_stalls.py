#!/usr/bin/env python3
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

"""Warp-stall attribution from periodic warp-state samples.

A faithful Python port of VeloQ's `ncu warp-stalls` verb
(crates/ncu/veloq-ncu/src/warp_stalls.rs + scripts/ncu_export.py), on the same
`ncu_report` API atrex already uses — no veloq binary, no new deps.

This is distinct from the existing extract_stall_hotspots.py, which aggregates
the `smsp__pcsamp_*` *metric* family. This tool reads the raw periodic stream
`action.timed_warp_samples()` directly (every sampled warp state, ~1e5 of them),
which is the source VeloQ uses and gives a `not_issued` signal plus a clean
out-of-cubin / unattributed reconciliation.

Requires the report be captured with warp-state sampling (`--set full` or
`--set source` with the warp sampling section). Reasons are taken from the live
`StallReason_*` enum so names stay correct across ncu versions.

Usage:
    python3 warp_stalls.py --run-dir profile/run \\
        --report profile/run/ncu.ncu-rep --tag run --by reason

Output in <run-dir>/analysis/:
    warp_stalls_<by>_<tag>.json   v1 envelope
    warp_stalls_<by>_<tag>.txt    readable digest
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import ncu_utils as U  # noqa: E402
import envelope as E  # noqa: E402
import row_key as K  # noqa: E402


def aggregate(action):
    """Return per-PC stall histograms + totals. Mirrors ncu_export._warp_stalls."""
    samples, reason_names = U.timed_warp_samples(action)
    base = U.cubin_load_base(action)

    per_pc = {}          # pc -> {reason_name: count}
    per_reason = {}      # reason_name -> count
    not_issued = 0
    for s in samples:
        rname = reason_names.get(s["stall_reason"], "unknown")
        per_pc.setdefault(s["pc"], {})
        per_pc[s["pc"]][rname] = per_pc[s["pc"]].get(rname, 0) + 1
        per_reason[rname] = per_reason.get(rname, 0) + 1
        if s["not_issued"]:
            not_issued += 1

    out_of_cubin = 0
    pc_rows = []         # {rel_address|None, source, reasons:{}, total}
    for pc, reasons in per_pc.items():
        if U.sass_by_pc(action, pc) == U.SASS_OUT_OF_CUBIN:
            out_of_cubin += sum(reasons.values())
            continue
        rel = pc - base if (base is not None and pc >= base) else None
        pc_rows.append({
            "rel_address": rel,
            "source": U.source_ref(action, pc),
            "reasons": reasons,
            "total": sum(reasons.values()),
        })

    return {
        "samples": len(samples),
        "not_issued": not_issued,
        "out_of_cubin": out_of_cubin,
        "per_reason": per_reason,
        "pc_rows": pc_rows,
        "base": base,
    }


def build_reason_rows(agg, prefix):
    rows = [{"key": K.reason_key(prefix, r), "reason": r, "total_samples": c}
            for r, c in agg["per_reason"].items()]
    rows.sort(key=lambda x: x["reason"])
    rows.sort(key=lambda x: x["total_samples"], reverse=True)
    return rows


def build_line_rows(agg, prefix):
    lines = {}  # (file,line) -> {reasons, total}
    unattributed = 0
    for p in agg["pc_rows"]:
        src = p["source"]
        if src is None:
            unattributed += p["total"]
            continue
        fk = (src["file"], src["line"])
        acc = lines.setdefault(fk, {"reasons": {}, "total": 0})
        for r, c in p["reasons"].items():
            acc["reasons"][r] = acc["reasons"].get(r, 0) + c
        acc["total"] += p["total"]
    rows = []
    for (file, line), acc in lines.items():
        rows.append({
            "key": K.line_key(prefix, file, line),
            "file": file, "line": line,
            "total_samples": acc["total"],
            "stalls": E.sorted_counter_map(acc["reasons"]),
        })
    rows.sort(key=lambda x: x["key"])
    rows.sort(key=lambda x: x["total_samples"], reverse=True)
    return rows, unattributed


def build_sass_rows(agg, prefix, file_filter):
    rows = []
    unattributed = 0
    for p in agg["pc_rows"]:
        src = p["source"]
        if src is None:
            unattributed += p["total"]
            if file_filter:
                continue  # null-source rows excluded when filtering by file
        elif file_filter and not _fnmatch(src["file"], file_filter):
            continue
        row = {
            "key": K.sass_key(prefix, p["rel_address"]),
            "rel_address": p["rel_address"],
            "total_samples": p["total"],
            "stalls": E.sorted_counter_map(p["reasons"]),
        }
        if src is not None:
            row["file"] = src["file"]
            row["line"] = src["line"]
        rows.append(row)
    rows.sort(key=lambda x: (x["rel_address"] is None, x["rel_address"] or 0))
    rows.sort(key=lambda x: x["total_samples"], reverse=True)
    return rows, unattributed


def _fnmatch(name, pat):
    import fnmatch
    return fnmatch.fnmatchcase(name, pat)


def run(report, by, limit, file_filter, self_check, ordinal):
    action = U.load_action(report)
    prefix = U.launch_identity(action, ordinal)
    kernel = U._kernel_demangled(action)
    agg = aggregate(action)
    warnings = []
    if agg["samples"] == 0:
        warnings.append("no warp-state samples — recapture with `--set full` "
                        "or a set that includes warp sampling")

    unattributed = 0
    if by == "reason":
        rows = build_reason_rows(agg, prefix)
        # unattributed: PCs with no source (computed for aux regardless)
        unattributed = sum(p["total"] for p in agg["pc_rows"] if p["source"] is None)
    elif by == "line":
        rows, unattributed = build_line_rows(agg, prefix)
    elif by == "sass":
        rows, unattributed = build_sass_rows(agg, prefix, file_filter)
    else:
        raise SystemExit(f"unknown --by: {by}")

    total_matched = len(rows)
    if limit and limit > 0:
        rows = rows[:limit]

    if self_check:
        _assert_reconciles(agg, unattributed)

    aux = {
        "row_id": prefix,
        "kernel_demangled": kernel,
        "total_samples": agg["samples"],
        "not_issued_samples": agg["not_issued"],
        "unattributed_samples": unattributed,
        "out_of_cubin_samples": agg["out_of_cubin"],
        "per_reason_totals": E.sorted_counter_map(agg["per_reason"]),
        "warnings": warnings,
    }
    return E.envelope("ncu.warp-stalls", report, count=len(rows),
                      total_matched=total_matched, rows=rows, auxiliary=aux,
                      axis=by)


def _assert_reconciles(agg, unattributed):
    """Σ(per-PC in-cubin totals) + out_of_cubin == total samples; and the
    in-cubin attributed+unattributed split must equal the in-cubin total."""
    in_cubin = sum(p["total"] for p in agg["pc_rows"])
    if in_cubin + agg["out_of_cubin"] != agg["samples"]:
        raise AssertionError(
            f"sample reconcile failed: in_cubin {in_cubin} + out_of_cubin "
            f"{agg['out_of_cubin']} != total {agg['samples']}")
    attributed = sum(p["total"] for p in agg["pc_rows"] if p["source"] is not None)
    if attributed + unattributed != in_cubin:
        raise AssertionError(
            f"attribution reconcile failed: attributed {attributed} + "
            f"unattributed {unattributed} != in_cubin {in_cubin}")


def _digest(env, max_rows=30):
    d = env["data"]
    by = d.get("axis")
    a = d["auxiliary"]
    lines = [f"# warp-stalls --by {by}  ({d['count']}/{d['total_matched']} rows)",
             f"# kernel: {a.get('kernel_demangled')}",
             f"# total={a['total_samples']} not_issued={a['not_issued_samples']} "
             f"unattributed={a['unattributed_samples']} "
             f"out_of_cubin={a['out_of_cubin_samples']}"]
    for w in a.get("warnings", []):
        lines.append(f"# WARNING: {w}")
    for r in d["rows"][:max_rows]:
        if by == "reason":
            lines.append(f"{r['total_samples']:>10}  {r['reason']}")
        elif by == "line":
            top = sorted(r["stalls"].items(), key=lambda kv: -kv[1])[:3]
            br = ", ".join(f"{k}:{v}" for k, v in top)
            lines.append(f"{r['total_samples']:>10}  {r['file']}:{r['line']}  [{br}]")
        else:
            src = (f"  [{r['file']}:{r['line']}]"
                   if "file" in r else "")
            ra = f"0x{r['rel_address']:x}" if r["rel_address"] is not None else "?"
            lines.append(f"{r['total_samples']:>10}  {ra}{src}")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--by", choices=["reason", "line", "sass"], default="reason")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--file", default=None, help="filter sass rows by source file glob")
    ap.add_argument("--ordinal", type=int, default=0)
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args(argv)

    if not args.report.exists():
        print(f"[skip] {args.report} does not exist", file=sys.stderr)
        return 1

    env = run(args.report, args.by, args.limit, args.file, args.self_check,
              args.ordinal)
    analysis = args.run_dir / "analysis"
    stem = f"warp_stalls_{args.by}_{args.tag}"
    E.write_json(analysis / f"{stem}.json", env)
    (analysis / f"{stem}.txt").write_text(_digest(env))
    print(f"warp-stalls ({args.by}) -> {analysis / (stem + '.json')} "
          f"({env['data']['count']} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
