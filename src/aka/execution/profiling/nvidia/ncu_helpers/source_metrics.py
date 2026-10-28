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

"""Per-source-line / per-SASS metric attribution for an .ncu-rep kernel.

A faithful Python port of VeloQ's `ncu source-metrics` verb
(crates/ncu/veloq-ncu/src/source_metrics/ + scripts/ncu_export.py), built on the
same `ncu_report` Python API atrex already uses — no veloq binary, no new deps.

Where atrex's extract_stall_hotspots.py only attributes *stall samples* to
lines, this attributes *any* per-PC counter (global loads/stores, sectors,
atomics, ...) to lines and SASS addresses, with VeloQ's additivity rule so only
summable counts are rolled up per line.

Requires the report be captured with `ncu --set source --section SourceCounters`
(per-PC instances) and the kernel built with `-lineinfo` (source attribution).

Usage:
    python3 source_metrics.py --run-dir profile/run \\
        --report profile/run/ncu_source.ncu-rep --tag run \\
        --counter 'l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,...' \\
        --by line --limit 50

Output in <run-dir>/analysis/:
    source_metrics_<by>_<tag>.json   v1 envelope
    source_metrics_<by>_<tag>.txt    readable digest
"""
from __future__ import annotations

import argparse
import fnmatch
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import ncu_utils as U  # noqa: E402
import envelope as E  # noqa: E402
import row_key as K  # noqa: E402

# Cubin-walk geometry lives in ncu_utils (single source of truth); aliased here
# for the aux `instruction_stride` field.
INSTRUCTION_STRIDE = U.INSTRUCTION_STRIDE

# Default counter glob: the global/shared/atomic count metrics most useful for
# coalescing / contention attribution. Overridable with --counter.
DEFAULT_COUNTERS = ",".join([
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",
    "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_requests_pipe_lsu_mem_global_op_st.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_atom.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_red.sum",
    "smsp__sass_inst_executed_op_global_ld.sum",
    "smsp__sass_inst_executed_op_global_st.sum",
    "smsp__sass_inst_executed_op_shared_ld.sum",
    "smsp__sass_inst_executed_op_shared_st.sum",
])


def _match_counters(action, globs):
    """Resolve comma-separated globs against the report's metric names (OR)."""
    try:
        names = list(action.metric_names())
    except Exception:
        names = []
    pats = [g.strip() for g in globs.split(",") if g.strip()]
    matched = []
    for n in names:
        if any(fnmatch.fnmatchcase(n, p) or n == p for p in pats):
            matched.append(n)
    return sorted(set(matched))


def _disasm_maps(action, base):
    """Walk the cubin from `base`, returning:
        addr_to_source: {rel_addr: {"file","line"} | None}
        addr_to_instr:  {rel_addr: (opcode, operands)}
    Built on the shared U.walk_cubin generator (single source of truth).
    """
    addr_to_source = {}
    addr_to_instr = {}
    for rel, opcode, operands, source in U.walk_cubin(action, base):
        addr_to_instr[rel] = (opcode, operands)
        addr_to_source[rel] = source
    return addr_to_source, addr_to_instr


def _placement(action, base, pc):
    """Classify a sampled PC. Returns (placement, rel_address|None).
    placement in {"attributed","in_cubin_no_source","out_of_cubin"}.
    """
    if U.source_ref(action, pc) is not None:
        plc = "attributed"
    elif U.sass_by_pc(action, pc) == U.SASS_OUT_OF_CUBIN:
        return "out_of_cubin", None
    else:
        plc = "in_cubin_no_source"
    rel = pc - base if (base is not None and pc >= base) else None
    return plc, rel


def collect(action, matched, base):
    """Run the per-instance rollup. Returns (line_acc, sass_acc, file aggregation
    inputs, aux totals). Mirrors VeloQ rollup.rs."""
    addr_to_source, addr_to_instr = _disasm_maps(action, base)

    # line_acc[(file,line)] = {"counters": {name: sum}, "coverage": {name: set(addr)},
    #                          "addrs": set(addr)}
    line_acc = {}
    # sass_acc[rel] = {"counters": {name: value}, "coverage": {name: 0/1},
    #                  "source": ref|None, "opcode","operands"}
    sass_acc = {}

    unattributed = {}        # name -> sum  (in-cubin, DWARF hole, additive)
    out_of_cubin = {}        # name -> sum
    unattributed_count = 0
    out_of_cubin_count = 0
    skipped = []             # {name, reason}

    additive = {n: U.is_additive(action, n) for n in matched}

    for name in matched:
        try:
            m = action[name]
        except Exception:
            continue
        pcs = U.correlation_pcs(m)
        if not pcs:
            # no per-PC instances at all -> not a source counter
            skipped.append({"name": name, "reason": "not-a-source-counter"})
            continue
        saw_in_cubin = False
        for i, pc in enumerate(pcs):
            if pc is None:
                continue
            val = U.instance_value_f64(m, i)
            if val is None:
                continue
            plc, rel = _placement(action, base, pc)
            if plc == "out_of_cubin":
                out_of_cubin[name] = out_of_cubin.get(name, 0.0) + val
                out_of_cubin_count += 1
                continue
            saw_in_cubin = True
            if rel is None:
                continue
            # sass axis: identity passthrough (additivity does NOT apply)
            s = sass_acc.setdefault(rel, {
                "counters": {}, "coverage": {},
                "source": addr_to_source.get(rel),
                "opcode": addr_to_instr.get(rel, ("", ""))[0],
                "operands": addr_to_instr.get(rel, ("", ""))[1],
            })
            if name not in s["counters"]:
                s["counters"][name] = val
                s["coverage"][name] = 1
            # line axis: only additive counters with a source line
            if additive[name]:
                src = addr_to_source.get(rel)
                if src is not None:
                    fk = (src["file"], src["line"])
                    la = line_acc.setdefault(fk, {
                        "counters": {}, "coverage": {}, "addrs": set()})
                    la["counters"][name] = la["counters"].get(name, 0.0) + val
                    la["coverage"].setdefault(name, set()).add(rel)
                    la["addrs"].add(rel)
                else:
                    unattributed[name] = unattributed.get(name, 0.0) + val
                    unattributed_count += 1
        if not saw_in_cubin:
            skipped.append({"name": name, "reason": "not-a-source-counter"})
        elif not additive[name]:
            # present on sass axis only; excluded from line/file
            skipped.append({"name": name, "reason": "non-additive-rollup"})

    return {
        "line_acc": line_acc, "sass_acc": sass_acc,
        "unattributed": unattributed, "out_of_cubin": out_of_cubin,
        "unattributed_count": unattributed_count,
        "out_of_cubin_count": out_of_cubin_count,
        "skipped": skipped, "additive": additive,
    }


def _full_counter_map(matched, present):
    """Every matched counter as a key; absent -> None (VeloQ Option<f64>)."""
    return {n: (present.get(n) if n in present else None) for n in matched}


def build_line_rows(acc, matched, prefix):
    rows = []
    line_counters_for_file = {}  # file -> list of per-line counter dicts
    for (file, line), la in acc["line_acc"].items():
        present = la["counters"]
        addrs = sorted(la["addrs"])
        cov = {n: len(la["coverage"].get(n, ())) for n in matched if n in present}
        # line axis only carries additive counters
        line_matched = [n for n in matched if acc["additive"].get(n)]
        counters = _full_counter_map(line_matched, present)
        coverage = {n: cov.get(n, 0) for n in line_matched}
        rows.append({
            "key": K.line_key(prefix, file, line),
            "launch_row_id": prefix,
            "file": file,
            "line": line,
            "sass_addresses": addrs,
            "sass_address_range": {"start": addrs[0] if addrs else 0,
                                   "end": addrs[-1] if addrs else 0},
            "sass_count": len(addrs),
            "counters": E.sorted_counter_map(counters),
            "counter_coverage": E.sorted_counter_map(coverage),
        })
        line_counters_for_file.setdefault(file, []).append((present, addrs))
    return rows, line_counters_for_file


def build_sass_rows(acc, matched, prefix):
    rows = []
    for rel, s in acc["sass_acc"].items():
        present = s["counters"]
        counters = _full_counter_map(matched, present)
        coverage = {n: (s["coverage"].get(n, 0) if n in present else 0)
                    for n in matched}
        row = {
            "key": K.sass_key(prefix, rel),
            "launch_row_id": prefix,
            "address": rel,
            "opcode": s["opcode"],
            "operands": s["operands"],
            "counters": E.sorted_counter_map(counters),
            "counter_coverage": E.sorted_counter_map(coverage),
        }
        if s["source"] is not None:
            row["source"] = {"file": s["source"]["file"], "line": s["source"]["line"]}
        rows.append(row)
    return rows


def build_file_rows(line_counters_for_file, matched, acc, prefix):
    """Derive file-axis rows from line rows (additivity already enforced)."""
    line_matched = [n for n in matched if acc["additive"].get(n)]
    rows = []
    for file, entries in line_counters_for_file.items():
        sums = {}
        coverage = {}
        all_addrs = set()
        for present, addrs in entries:
            for n, v in present.items():
                sums[n] = sums.get(n, 0.0) + v
                coverage[n] = coverage.get(n, 0) + len(addrs)
            all_addrs.update(addrs)
        counters = _full_counter_map(line_matched, sums)
        cov = {n: coverage.get(n, 0) for n in line_matched}
        rows.append({
            "key": K.file_key(prefix, file),
            "launch_row_id": prefix,
            "file": file,
            "line_count": len(entries),
            "sass_count": len(all_addrs),
            "counters": E.sorted_counter_map(counters),
            "counter_coverage": E.sorted_counter_map(cov),
        })
    return rows


def _sort_rows(rows, by, sort_spec, matched):
    """Sort by '<counter>[:asc|desc]'; default = first counter desc.
    Tiebreaks: line->(file,line), sass->address, file->file."""
    if sort_spec:
        field, _, direction = sort_spec.partition(":")
        desc = direction.lower() != "asc"
    else:
        field = sorted(matched)[0] if matched else None
        desc = True

    def _val(r):
        v = r.get("counters", {}).get(field) if field else None
        return v if isinstance(v, (int, float)) else -math.inf

    def _tie(r):
        if by == "line":
            return (r.get("file", ""), r.get("line", 0))
        if by == "sass":
            return r.get("address", 0)
        return r.get("file", "")

    rows.sort(key=_tie)
    rows.sort(key=_val, reverse=desc)
    return rows


def run(report, by, counter_glob, limit, sort_spec, file_filter, line_filter,
        self_check, ordinal):
    action = U.load_action(report)
    matched = _match_counters(action, counter_glob)
    warnings = []
    base = U.cubin_load_base(action)
    prefix = U.launch_identity(action, ordinal)
    kernel = U._kernel_demangled(action)

    if base is None:
        warnings.append("launch has no cubin_load_base — recapture with "
                        "`--set source` or `--set full`")
    if not matched:
        warnings.append(f"no metric names matched glob: {counter_glob}")

    acc = (collect(action, matched, base)
           if (base is not None and matched)
           else {"line_acc": {}, "sass_acc": {}, "unattributed": {},
                 "out_of_cubin": {}, "unattributed_count": 0,
                 "out_of_cubin_count": 0, "skipped": [],
                 "additive": {n: U.is_additive(action, n) for n in matched}})

    line_rows, line_for_file = build_line_rows(acc, matched, prefix)
    if by == "line":
        rows = line_rows
    elif by == "sass":
        rows = build_sass_rows(acc, matched, prefix)
    elif by == "file":
        rows = build_file_rows(line_for_file, matched, acc, prefix)
    else:
        raise SystemExit(f"unknown --by: {by}")

    # filters (before sort+limit)
    if file_filter:
        if by == "sass":
            rows = [r for r in rows
                    if r.get("source", {}).get("file") and
                    fnmatch.fnmatchcase(r["source"]["file"], file_filter)]
        else:
            rows = [r for r in rows
                    if fnmatch.fnmatchcase(r.get("file", ""), file_filter)]
    if line_filter is not None:
        if not file_filter:
            raise SystemExit("--line requires --file")
        rows = [r for r in rows if r.get("line") == line_filter or
                r.get("source", {}).get("line") == line_filter]

    rows = _sort_rows(rows, by, sort_spec, matched)
    total_matched = len(rows)
    if limit and limit > 0:
        rows = rows[:limit]

    if self_check and by == "file":
        # file sums must equal sum of their line rows
        _assert_file_reconciles(line_rows, rows, matched, acc)

    aux = {
        "row_id": prefix,
        "kernel_demangled": kernel,
        "counter_glob": counter_glob,
        "matched_counters": matched,
        "skipped_counters": acc["skipped"],
        "unattributed_sass_counter_totals": E.sorted_counter_map(acc["unattributed"]),
        "out_of_cubin_counter_totals": E.sorted_counter_map(acc["out_of_cubin"]),
        "unattributed_sass_count": acc["unattributed_count"],
        "out_of_cubin_instance_count": acc["out_of_cubin_count"],
        "instruction_stride": INSTRUCTION_STRIDE,
        "warnings": warnings,
    }
    return E.envelope("ncu.source-metrics", report, count=len(rows),
                      total_matched=total_matched, rows=rows, auxiliary=aux,
                      axis=by)


def _assert_file_reconciles(line_rows, file_rows, matched, acc):
    line_matched = [n for n in matched if acc["additive"].get(n)]
    per_file = {}
    for r in line_rows:
        f = r["file"]
        d = per_file.setdefault(f, {})
        for n in line_matched:
            v = r["counters"].get(n)
            if isinstance(v, (int, float)):
                d[n] = d.get(n, 0.0) + v
    for fr in file_rows:
        f = fr["file"]
        for n in line_matched:
            want = per_file.get(f, {}).get(n, 0.0)
            got = fr["counters"].get(n) or 0.0
            if abs(want - got) > 1e-6:
                raise AssertionError(
                    f"file reconcile failed for {f}/{n}: line-sum {want} != file {got}")


def _digest(env, max_rows=25):
    d = env["data"]
    by = d.get("axis")
    lines = [f"# source-metrics --by {by}  ({d['count']}/{d['total_matched']} rows)",
             f"# kernel: {d['auxiliary'].get('kernel_demangled')}"]
    for w in d["auxiliary"].get("warnings", []):
        lines.append(f"# WARNING: {w}")
    for r in d["rows"][:max_rows]:
        if by == "line":
            head = f"{r['file']}:{r['line']}  (sass x{r['sass_count']})"
        elif by == "sass":
            src = r.get("source")
            loc = f"  [{src['file']}:{src['line']}]" if src else ""
            head = f"0x{r['address']:x}  {r['opcode']} {r['operands']}{loc}"
        else:
            head = f"{r['file']}  (lines {r['line_count']}, sass {r['sass_count']})"
        lines.append(f"\n{head}")
        for n, v in r["counters"].items():
            if v is not None:
                lines.append(f"    {n:<60} {v:g}")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--counter", default=DEFAULT_COUNTERS,
                    help="comma-separated metric globs (OR-matched)")
    ap.add_argument("--by", choices=["line", "sass", "file"], default="line")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--sort", default=None, help="<counter>[:asc|desc]")
    ap.add_argument("--file", default=None, help="filter rows by source file glob")
    ap.add_argument("--line", type=int, default=None, help="filter by line (needs --file)")
    ap.add_argument("--ordinal", type=int, default=0,
                    help="launch occurrence ordinal for the stable key prefix")
    ap.add_argument("--self-check", action="store_true",
                    help="assert reconciliation identities")
    args = ap.parse_args(argv)

    if not args.report.exists():
        print(f"[skip] {args.report} does not exist", file=sys.stderr)
        return 1

    env = run(args.report, args.by, args.counter, args.limit, args.sort,
              args.file, args.line, args.self_check, args.ordinal)

    analysis = args.run_dir / "analysis"
    stem = f"source_metrics_{args.by}_{args.tag}"
    E.write_json(analysis / f"{stem}.json", env)
    (analysis / f"{stem}.txt").write_text(_digest(env))
    print(f"source-metrics ({args.by}) -> {analysis / (stem + '.json')} "
          f"({env['data']['count']} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
