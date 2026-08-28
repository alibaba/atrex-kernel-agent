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

"""Aggregate per-PC stall samples into per-source-line hotspots.

Requires:
    - .ncu-rep collected with `ncu --set source --section SourceCounters`
    - kernel compiled with `-lineinfo`

Produces in `<run-dir>/analysis/`:
    stall_hotspots_<tag>.txt — top lines ranked by total stall samples,
                               with per-stall-type breakdown, plus per-stall
                               top lines.

Usage:
    python3 extract_stall_hotspots.py --run-dir profile/myrun \\
            --report profile/myrun/reports/source_<tag>.ncu-rep --tag <tag>
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ncu_utils import load_action, metric_value_at  # noqa: E402


STALL_METRICS = [
    "smsp__pcsamp_warps_issue_stalled_long_scoreboard",
    "smsp__pcsamp_warps_issue_stalled_short_scoreboard",
    "smsp__pcsamp_warps_issue_stalled_wait",
    "smsp__pcsamp_warps_issue_stalled_math_pipe_throttle",
    "smsp__pcsamp_warps_issue_stalled_mio_throttle",
    "smsp__pcsamp_warps_issue_stalled_lg_throttle",
    "smsp__pcsamp_warps_issue_stalled_not_selected",
    "smsp__pcsamp_warps_issue_stalled_dispatch_stall",
    "smsp__pcsamp_warps_issue_stalled_drain",
    "smsp__pcsamp_warps_issue_stalled_no_instructions",
    "smsp__pcsamp_warps_issue_stalled_selected",
    "smsp__pcsamp_warps_issue_stalled_branch_resolving",
    "smsp__pcsamp_warps_issue_stalled_barrier",
    "smsp__pcsamp_warps_issue_stalled_tex_throttle",
    "smsp__pcsamp_warps_issue_stalled_sleeping",
    "smsp__pcsamp_warps_issue_stalled_misc",
    "smsp__pcsamp_warps_issue_stalled_membar",
]


def collect_per_pc(action):
    """Return dict[pc] -> dict[stall_name -> count]."""
    per_pc = defaultdict(lambda: defaultdict(int))
    for sn in STALL_METRICS:
        try:
            m = action[sn]
        except Exception:
            continue
        try:
            n = m.num_instances()
        except Exception:
            continue
        if n == 0 or not m.has_correlation_ids():
            continue
        cor = m.correlation_ids()
        for i in range(n):
            try:
                pc = cor.as_uint64(i)
            except Exception:
                try:
                    pc = int(cor.as_double(i))
                except Exception:
                    continue
            v = metric_value_at(m, i)
            if v:
                per_pc[pc][sn] += int(v)
    return per_pc


def aggregate_by_source_line(action, per_pc):
    per_line = defaultdict(lambda: defaultdict(int))
    for pc, stalls in per_pc.items():
        try:
            si = action.source_info(pc)
            if si is None:
                file_, line = "?", 0
            else:
                file_, line = si.file_name(), si.line()
        except Exception:
            file_, line = "?", 0
        for sn, v in stalls.items():
            per_line[(file_, line)][sn] += v
    return per_line


def short_stall_name(full):
    return full.replace("smsp__pcsamp_warps_issue_stalled_", "")


def write_report(per_line, out_path, tag, top_n=30):
    totals = []
    for (f_, ln), stalls in per_line.items():
        total = sum(stalls.values())
        totals.append((total, f_, ln, dict(stalls)))
    totals.sort(key=lambda x: -x[0])

    with open(out_path, "w") as f:
        f.write(f"===== Stall hotspots for {tag} =====\n")
        f.write(f"Total distinct (file, line) entries: {len(totals)}\n\n")
        f.write(f"{'Rank':>4} {'Total':>10} {'File':<30} {'Line':>5}  Breakdown (stall: count)\n")
        f.write("-" * 150 + "\n")
        for i, (total, fn, ln, stalls) in enumerate(totals[:top_n]):
            short = Path(str(fn)).name if fn and fn != "?" else "?"
            breakdown = ", ".join(
                f"{short_stall_name(k)}: {v}"
                for k, v in sorted(stalls.items(), key=lambda x: -x[1]) if v
            )
            f.write(f"{i:>4} {total:>10} {short:<30} {ln:>5}  {breakdown}\n")

        f.write("\n\n===== Per-stall-type top lines =====\n")
        for sn in STALL_METRICS:
            items = [((fn, ln), s.get(sn, 0)) for (fn, ln), s in per_line.items()]
            items = [it for it in items if it[1] > 0]
            items.sort(key=lambda x: -x[1])
            if not items:
                continue
            f.write(f"\n--- {short_stall_name(sn)} ---\n")
            for (fn, ln), v in items[:10]:
                short = Path(str(fn)).name if fn and fn != "?" else "?"
                f.write(f"  {v:>8}  {short}:{ln}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--report", type=Path, action="append", required=True,
                    help="Path(s) to source-level .ncu-rep file(s). Pass multiple with repeated flag.")
    ap.add_argument("--tag", type=str, action="append", required=True)
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args()

    if len(args.report) != len(args.tag):
        ap.error("--report and --tag counts must match")

    analysis_dir = args.run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    for rep, tag in zip(args.report, args.tag):
        if not rep.exists():
            print(f"[skip] {rep} not found", file=sys.stderr)
            continue
        action = load_action(rep)
        per_pc = collect_per_pc(action)
        per_line = aggregate_by_source_line(action, per_pc)
        out = analysis_dir / f"stall_hotspots_{tag}.txt"
        write_report(per_line, out, tag, top_n=args.top)
        print(f"[{tag}] -> {out} ({len(per_line)} distinct lines)")


if __name__ == "__main__":
    main()
