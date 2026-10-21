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

"""
NCU Metrics Symptom Classification Tool

Reads key metrics from metrics_key_*.json output by the bundled
profiling/nvidia/ncu_helpers/analyze_reports.py, infers symptoms based on the 14 NCU
diagnosis Patterns (see profiling/profile_guide.md), and generates summary.txt.

Usage:
    python3 profiling/nvidia/classify_ncu.py \
        --metrics <output-dir>/analysis/metrics_key_run.json \
        --output <output-dir>/summary.txt

    python3 profiling/nvidia/classify_ncu.py \
        --metrics <output-dir>/analysis/metrics_key_run.json \
        --json  # Output in JSON format
"""

import argparse
import json
import math
import sys
from pathlib import Path

if not __package__ and Path(__file__).with_name("ncu_classification.py").is_file():
    from ncu_classification import PATTERNS, _get, classify
else:
    from aka.legacy.execution.profiling.nvidia.ncu_classification import PATTERNS, _get, classify


# Where to look to localise a symptom to a source line / SASS address. Points at
# the independent source-evidence artifacts (only present on --source runs); these
# never change the diagnosis. Symptoms without a line-level signal (e.g. occupancy)
# are intentionally absent — for those the scalar metrics_key are the right read.
LOCALIZE_BY_SYMPTOM = {
    "memory-bound": ["source_metrics_line_run.txt", "warp_stalls_line_run.txt"],
    "pipeline-stalls": ["warp_stalls_line_run.txt"],
    "compute-bound": ["warp_stalls_line_run.txt", "disasm_run.txt"],
    "register-pressure": ["disasm_run.txt"],
}


def format_summary(metrics, findings):
    """Generate summary.txt text content."""
    lines = []

    # Part 1: Key metrics
    kernel_name = metrics.get("__kernel_name__", "?")
    duration = _get(metrics, "gpu__time_duration.sum")

    lines.append("===== NCU Profile Summary =====")
    lines.append(f"Kernel: {kernel_name}")
    if duration is not None:
        # gpu__time_duration.sum is reported by ncu in nanoseconds; /1000 -> us.
        lines.append(f"Duration: {duration / 1000:.1f} us")

    sm_tp = _get(metrics, "sm__throughput.avg.pct_of_peak_sustained_elapsed")
    mem_tp = _get(metrics, "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed")
    l2_hit = _get(metrics, "lts__t_sector_hit_rate.pct")
    occupancy = _get(metrics, "sm__warps_active.avg.pct_of_peak_sustained_active")
    tc_pct = _get(metrics, "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed")
    regs = _get(metrics, "launch__registers_per_thread")
    grid = _get(metrics, "launch__grid_size")
    block = _get(metrics, "launch__block_size")
    local_ld = _get(metrics, "smsp__sass_inst_executed_op_local_ld.sum", 0)
    local_st = _get(metrics, "smsp__sass_inst_executed_op_local_st.sum", 0)

    if grid is not None and block is not None:
        lines.append(f"Grid: {int(grid)} blocks x {int(block)} threads")
    if sm_tp is not None:
        lines.append(f"SM Throughput: {sm_tp:.1f}%")
    if mem_tp is not None:
        lines.append(f"Memory Throughput: {mem_tp:.1f}%")
    if l2_hit is not None:
        lines.append(f"L2 Hit Rate: {l2_hit:.1f}%")
    if occupancy is not None:
        lines.append(f"Achieved Occupancy: {occupancy:.1f}%")
    if tc_pct is not None:
        lines.append(f"Tensor Core Utilization: {tc_pct:.1f}%")
    if regs is not None:
        lines.append(f"Registers/Thread: {int(regs)}")
    lines.append(f"Local Ld/St: {int(local_ld)} / {int(local_st)}")

    # Top stall reasons
    stall_metrics = [
        ("long_scoreboard", "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio"),
        ("short_scoreboard", "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio"),
        ("wait", "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio"),
        ("barrier", "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio"),
        ("math_throttle", "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio"),
        ("mio_throttle", "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio"),
        ("lg_throttle", "smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio"),
        ("not_selected", "smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio"),
    ]
    stalls = []
    for name, key in stall_metrics:
        v = _get(metrics, key)
        if v is not None and v > 0.1:
            stalls.append((name, v))
    stalls.sort(key=lambda x: -x[1])

    if stalls:
        lines.append("")
        lines.append("Top Stall Reasons:")
        for name, v in stalls[:5]:
            lines.append(f"  {name}: {v:.2f}")

    # Part 2: Diagnosis
    lines.append("")
    lines.append("===== Diagnosis =====")
    if not findings:
        lines.append("No significant patterns detected.")
    else:
        symptoms = set()
        lines.append(f"Detected {len(findings)} pattern(s):")
        lines.append("")
        for pid, label, confidence, evidence in findings:
            lines.append(f"  Pattern {pid} [{confidence}]: {label}")
            lines.append(f"    Evidence: {evidence}")
            if pid in PATTERNS:
                symptoms.add(PATTERNS[pid][2])

        lines.append("")
        symptom_list = sorted(symptoms)
        lines.append(f"SYMPTOMS: {', '.join(symptom_list)}")

        # Point at the source-level evidence that localises these symptoms.
        # Deduped, in the order symptoms appear; only symptoms with a line-level
        # signal contribute. Files are produced by `profile_nvidia.sh --source`.
        seen = set()
        localize = []
        for s in symptom_list:
            for f in LOCALIZE_BY_SYMPTOM.get(s, ()):
                if f not in seen:
                    seen.add(f)
                    localize.append(f)
        if localize:
            lines.append(f"LOCALIZE: {', '.join('analysis/' + f for f in localize)} "
                         f"(profile with --source to generate; see "
                         f"analysis/source_evidence_manifest.json)")

    # Part 3: Query suggestions
    lines.append("")
    lines.append("===== Suggested Queries =====")

    # Playbook pointer only lists the lettered diagnostic Patterns (A-N) that
    # profile_guide.md documents; supplementary ids like `_mem` are not in it.
    pattern_ids = [f[0] for f in findings if f[0] in PATTERNS and f[0].isalpha()]
    if pattern_ids:
        lines.append(f"Playbook: See profiling/profile_guide.md "
                     f"Pattern {', '.join(pattern_ids)}")

    symptoms_for_wiki = set()
    for f in findings:
        if f[0] in PATTERNS:
            symptoms_for_wiki.add(PATTERNS[f[0]][2])
    for s in sorted(symptoms_for_wiki):
        lines.append(
            f"knowledge: symptom '{s}'; consult enabled tools via .atrex_plugins/instructions.md"
        )

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(
        description="NCU Metrics Symptom Classification Tool")
    parser.add_argument("--metrics", required=True,
                        help="Path to metrics_key_*.json file")
    parser.add_argument("--output", "-o", default=None,
                        help="Output summary.txt path (default: stdout)")
    parser.add_argument("--json", action="store_true",
                        help="Output in JSON format (findings list)")
    args = parser.parse_args()

    metrics_path = Path(args.metrics)
    if not metrics_path.exists():
        print(f"Error: file not found: {metrics_path}", file=sys.stderr)
        sys.exit(1)

    try:
        with open(metrics_path) as f:
            metrics = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: {metrics_path} is not valid JSON: {e}", file=sys.stderr)
        sys.exit(1)
    if not isinstance(metrics, dict):
        print(f"Error: {metrics_path} must contain a JSON object of metric "
              f"name -> value (got {type(metrics).__name__})", file=sys.stderr)
        sys.exit(1)

    findings = classify(metrics)

    if args.json:
        symptoms = sorted(set(
            PATTERNS[f[0]][2] for f in findings if f[0] in PATTERNS
        ))
        localize = sorted({f for s in symptoms
                           for f in LOCALIZE_BY_SYMPTOM.get(s, ())})
        result = {
            "findings": [
                {"pattern": pid, "label": label,
                 "confidence": conf, "evidence": ev}
                for pid, label, conf, ev in findings
            ],
            "symptoms": symptoms,
            "localize": [f"analysis/{f}" for f in localize],
        }
        output = json.dumps(result, indent=2, ensure_ascii=False)
    else:
        output = format_summary(metrics, findings)

    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"Summary saved to: {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
