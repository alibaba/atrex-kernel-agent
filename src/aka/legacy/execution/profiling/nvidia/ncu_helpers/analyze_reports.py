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

"""Extract and compare key metrics from .ncu-rep files.

Produces in `<run_dir>/analysis/`:
    metrics_all_<tag>.json    — every metric, archival
    metrics_key_<tag>.txt/json — curated B200-compatible key metrics
    compare_<tag1>_vs_<tag2>.txt (when >= 2 reports given) — side-by-side

Usage examples:
    # Single report
    python3 analyze_reports.py --run-dir profile/myrun \\
            --report profile/myrun/reports/full_<tag>.ncu-rep --tag <tag>

    # Multiple reports → side-by-side compare
    python3 analyze_reports.py --run-dir profile/myrun \\
            --report profile/myrun/reports/full_<tag1>.ncu-rep --tag <tag1> \\
            --report profile/myrun/reports/full_<tag2>.ncu-rep --tag <tag2>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make ncu_utils importable whether we're invoked from the skill dir or a run dir
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ncu_utils import (  # noqa: E402
    B200_KEY_METRICS, dump_all_metrics, load_action, safe,
)


def collect(report_path: Path, tag: str, analysis_dir: Path) -> dict:
    action = load_action(report_path)
    print(f"[{tag}] {report_path.name}: {len(action.metric_names())} metrics, kernel {action.name()}")

    # Full archive
    n = dump_all_metrics(action, analysis_dir / f"metrics_all_{tag}.json")
    print(f"  -> metrics_all_{tag}.json ({n} metrics)")

    # Curated key metrics
    key = {m: safe(action, m) for m in B200_KEY_METRICS}
    key["__kernel_name__"] = action.name()

    (analysis_dir / f"metrics_key_{tag}.json").write_text(
        json.dumps(key, indent=2, default=str)
    )
    with open(analysis_dir / f"metrics_key_{tag}.txt", "w") as f:
        f.write(f"===== {tag} =====\nKernel: {action.name()}\n\n")
        for m, v in key.items():
            if m.startswith("__"):
                continue
            f.write(f"{m:95s} = {v}\n")
    print(f"  -> metrics_key_{tag}.{{json,txt}}")
    return key


def compare(collected: dict, analysis_dir: Path):
    tags = list(collected.keys())
    if len(tags) < 2:
        return
    out_path = analysis_dir / f"compare_{'_vs_'.join(tags)}.txt"
    with open(out_path, "w") as f:
        col_w = max(20, max(len(t) for t in tags) + 2)
        f.write(f"{'Metric':<95}")
        for t in tags:
            f.write(f"{t:>{col_w}}")
        f.write("\n" + "-" * (95 + col_w * len(tags)) + "\n")
        for m in B200_KEY_METRICS:
            f.write(f"{m:<95}")
            for t in tags:
                v = collected[t].get(m, "N/A")
                if isinstance(v, (int, float)):
                    v = f"{v:.4g}"
                f.write(f"{str(v):>{col_w}}")
            f.write("\n")
    print(f"compare -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Extract key NCU metrics and compare.")
    ap.add_argument("--run-dir", type=Path, required=True,
                    help="The profile run directory — outputs go to <run-dir>/analysis/")
    ap.add_argument("--report", type=Path, action="append", required=True,
                    help="Path to a .ncu-rep file. Can be passed multiple times.")
    ap.add_argument("--tag", type=str, action="append", required=True,
                    help="Short tag for each report. Must be passed once per --report.")
    args = ap.parse_args()

    if len(args.report) != len(args.tag):
        ap.error("--report and --tag counts must match")

    analysis_dir = args.run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    collected = {}
    for rep, tag in zip(args.report, args.tag):
        if not rep.exists():
            print(f"[skip] {rep} does not exist", file=sys.stderr)
            continue
        collected[tag] = collect(rep, tag, analysis_dir)

    compare(collected, analysis_dir)


if __name__ == "__main__":
    main()
