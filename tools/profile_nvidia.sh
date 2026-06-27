#!/bin/bash
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

#
# NVIDIA kernel profile tool (ncu wrapper)
#
# Usage:
#     bash tools/profile_nvidia.sh <kernel.py> [options]
#
# Examples:
#     bash tools/profile_nvidia.sh my_kernel.py --output-dir ./profile
#     bash tools/profile_nvidia.sh my_kernel.py --kernel-name "mla_decode" --launch-skip 1
#     bash tools/profile_nvidia.sh my_kernel.py --source --output-dir ./profile
#
# Dependencies:
#     ncu (NVIDIA Nsight Compute CLI) in PATH
#     ncu report helpers (for parsing .ncu-rep): bundled in tools/ncu_helpers/
#       (analyze_reports.py, ncu_utils.py); override with NCU_HELPERS / --ncu-helpers
#     classify_ncu.py (symptom classification)
#
# Output:
#     <output-dir>/ncu.ncu-rep                         binary report
#     <output-dir>/ncu_source.ncu-rep                  optional: source-level report
#     <output-dir>/analysis/metrics_key_run.json       key metrics JSON
#     <output-dir>/analysis/metrics_key_run.txt        key metrics text
#     <output-dir>/analysis/metrics_all_run.json       full metrics archive
#     <output-dir>/analysis/stall_hotspots_run.txt     optional (--source): per-line stall hotspots
#     <output-dir>/summary.txt                         final summary (metrics+symptoms+query suggestions)
#
#     With --source, source-level evidence is generated in one step and indexed by:
#       <output-dir>/analysis/source_evidence_manifest.json   what evidence exists + purpose
#       <output-dir>/analysis/disasm_run.{json,txt}           source-correlated SASS (+PTX)
#       <output-dir>/analysis/warp_stalls_{reason,line}_run.{json,txt}  warp-stall attribution
#       <output-dir>/analysis/source_metrics_{line,sass}_run.{json,txt} per-line/SASS metric attribution
#     With --diff PREV_DIR:
#       <output-dir>/analysis/diff_*.txt                 per-row delta vs a previous run
#
#     The source-evidence artifacts are independent evidence (a Python port of
#     VeloQ's ncu verbs onto the same ncu_report API); they do NOT feed
#     classify_ncu.py and never change summary.txt. summary.txt's LOCALIZE
#     section points at them when a localisable symptom fires.
#
# Note:
#     ncu is invoked with '--kill yes': the target process is terminated once the
#     requested launches (--launch-skip/--launch-count) have been profiled. This
#     prevents the script from hanging on long-running apps, but any post-kernel
#     cleanup / teardown in your kernel.py after the profiled launches will NOT
#     run. Raise --launch-count or remove --kill from the script if you need the
#     app to run to completion.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

KERNEL_FILE=""
OUTPUT_DIR="./profiles/v0"
OUTPUT_DIR_EXPLICIT=false
KERNEL_NAME=""
LAUNCH_SKIP=0
LAUNCH_COUNT=1
COLLECT_SOURCE=false
NO_CLASSIFY=false
DIFF_DIR=""

# ncu-report-skill helpers path (can be overridden via environment variable)
NCU_HELPERS="${NCU_HELPERS:-}"

usage() {
    cat <<EOF
Usage: $0 <kernel.py> [options]

Options:
    --output-dir DIR        output directory (default: ./profile_output)
    --kernel-name NAME      ncu kernel name filter
    --launch-skip N         skip warmup dispatches (default: 0)
    --launch-count N        number of dispatches to capture (default: 1)
    --source                additionally collect source-level stall data (--set source)
    --no-classify           collect only, skip classification
    --diff PREV_DIR         after analysis, diff this run's envelopes vs PREV_DIR/analysis
    --ncu-helpers DIR       ncu-report-skill helpers directory path
    -h, --help              show help

Note: ncu runs with '--kill yes' — the target is stopped after the profiled
launches, so post-kernel cleanup in kernel.py will not run.

Output:
    <output-dir>/ncu.ncu-rep                    ncu binary report
    <output-dir>/analysis/metrics_key_run.json  key metrics JSON
    <output-dir>/summary.txt                    final summary
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-dir)
            OUTPUT_DIR="$2"
            OUTPUT_DIR_EXPLICIT=true
            shift 2
            ;;
        --kernel-name)
            KERNEL_NAME="$2"
            shift 2
            ;;
        --launch-skip)
            LAUNCH_SKIP="$2"
            shift 2
            ;;
        --launch-count)
            LAUNCH_COUNT="$2"
            shift 2
            ;;
        --source)
            COLLECT_SOURCE=true
            shift
            ;;
        --no-classify)
            NO_CLASSIFY=true
            shift
            ;;
        --diff)
            DIFF_DIR="$2"
            shift 2
            ;;
        --ncu-helpers)
            NCU_HELPERS="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            if [[ -z "$KERNEL_FILE" ]]; then
                KERNEL_FILE="$1"
            else
                echo "Error: unknown argument $1"
                exit 1
            fi
            shift
            ;;
    esac
done

if [[ -z "$KERNEL_FILE" ]]; then
    echo "Error: kernel file must be specified"
    usage
fi

# ============================================================
# Workspace-aware output directory
# ============================================================
# When --output-dir is not explicitly specified, detect if CWD is a workspace
# (contains memory/ and profiles/ directories) and default to profiles/latest/.
if [[ "$OUTPUT_DIR_EXPLICIT" == false ]]; then
    if [[ -d "memory" && -d "profiles" ]]; then
        OUTPUT_DIR="profiles/latest"
        echo "[INFO] Workspace detected (memory/ + profiles/ present in CWD)."
        echo "[INFO] Output directory set to: $OUTPUT_DIR"
        echo "[INFO] Override with --output-dir if needed."
        echo ""
    fi
fi

# ============================================================
# Environment check
# ============================================================
if ! command -v ncu &>/dev/null; then
    echo "Error: ncu not found. Please ensure NVIDIA Nsight Compute is installed and ncu is in PATH"
    echo "  Common paths: /usr/local/cuda/bin/ncu or /usr/local/cuda-*/nsight-compute-*/ncu"
    exit 1
fi

if ! command -v nvidia-smi &>/dev/null; then
    echo "Error: nvidia-smi not found. Please ensure you are running in an NVIDIA GPU environment"
    exit 1
fi

# Detect ncu report helpers path (bundled copy ships in tools/ncu_helpers/)
if [[ -z "$NCU_HELPERS" ]]; then
    SEARCH_PATHS=(
        "$SCRIPT_DIR/ncu_helpers"
        "$HOME/.claude/skills/ncu-report-skill/helpers"
        "$HOME/.config/opencode/skills/ncu-report-skill/helpers"
        "$HOME/.codex/skills/ncu-report-skill/helpers"
    )
    for p in "${SEARCH_PATHS[@]}"; do
        if [[ -f "$p/analyze_reports.py" && -f "$p/ncu_utils.py" ]]; then
            NCU_HELPERS="$(cd "$p" && pwd)"
            break
        fi
    done
fi

if [[ -z "$NCU_HELPERS" || ! -f "$NCU_HELPERS/analyze_reports.py" ]]; then
    echo "Warning: ncu-report-skill helpers not found, skipping automatic metrics parsing"
    echo "  Set the NCU_HELPERS environment variable or use --ncu-helpers to specify the path"
    echo "  Required: analyze_reports.py + ncu_utils.py"
    NCU_HELPERS=""
fi

mkdir -p "$OUTPUT_DIR"

KERNEL_FILTER=()
if [[ -n "$KERNEL_NAME" ]]; then
    KERNEL_FILTER=(--kernel-name "$KERNEL_NAME")
fi

# ============================================================
# Step 1: Collect ncu full report
# ============================================================
echo "=========================================="
echo "  Step 1: Collect ncu full report"
echo "=========================================="

ncu --set full \
    --launch-skip "$LAUNCH_SKIP" \
    --launch-count "$LAUNCH_COUNT" \
    --kill yes \
    ${KERNEL_FILTER[@]+"${KERNEL_FILTER[@]}"} \
    -o "$OUTPUT_DIR/ncu" \
    python "$KERNEL_FILE"

echo ""
echo "ncu full report: $OUTPUT_DIR/ncu.ncu-rep"

# ============================================================
# Step 2: Optional source-level stall collection
# ============================================================
if [[ "$COLLECT_SOURCE" == true ]]; then
    echo ""
    echo "=========================================="
    echo "  Step 2: Collect source-level stall data"
    echo "=========================================="

    ncu --set source \
        --section SourceCounters \
        --launch-skip "$LAUNCH_SKIP" \
        --launch-count "$LAUNCH_COUNT" \
        --kill yes \
        ${KERNEL_FILTER[@]+"${KERNEL_FILTER[@]}"} \
        -o "$OUTPUT_DIR/ncu_source" \
        python "$KERNEL_FILE"

    echo ""
    echo "ncu source report: $OUTPUT_DIR/ncu_source.ncu-rep"
fi

# ============================================================
# Step 3: Parse metrics (ncu-report-skill)
# ============================================================
if [[ -n "$NCU_HELPERS" ]]; then
    echo ""
    echo "=========================================="
    echo "  Step 3: Parse key metrics"
    echo "=========================================="

    python3 "$NCU_HELPERS/analyze_reports.py" \
        --run-dir "$OUTPUT_DIR" \
        --report "$OUTPUT_DIR/ncu.ncu-rep" \
        --tag run

    # Optional: source-level stall hotspots
    if [[ "$COLLECT_SOURCE" == true && -f "$OUTPUT_DIR/ncu_source.ncu-rep" ]]; then
        if [[ -f "$NCU_HELPERS/extract_stall_hotspots.py" ]]; then
            python3 "$NCU_HELPERS/extract_stall_hotspots.py" \
                --run-dir "$OUTPUT_DIR" \
                --report "$OUTPUT_DIR/ncu_source.ncu-rep" \
                --tag run
        else
            echo "Warning: extract_stall_hotspots.py not found in $NCU_HELPERS, skipping stall hotspots"
        fi
    fi
fi

# ============================================================
# Step 3b: source-level evidence (independent of classify)
# ============================================================
# Only on --source runs: that flag means "I want to localise a symptom to a
# source line / SASS address", so the whole evidence bundle is generated here
# and indexed in source_evidence_manifest.json. Best-effort (|| true): a missing
# helper or unsupported report degrades to today's behaviour. These artifacts do
# NOT feed classify_ncu.py and never change summary.txt.
if [[ "$COLLECT_SOURCE" == true && -n "$NCU_HELPERS" \
      && -f "$NCU_HELPERS/source_evidence.py" && -f "$OUTPUT_DIR/ncu.ncu-rep" ]]; then
    echo ""
    echo "=========================================="
    echo "  Step 3b: source-level evidence (disasm / warp-stalls / source-metrics)"
    echo "=========================================="

    python3 "$NCU_HELPERS/source_evidence.py" \
        --run-dir "$OUTPUT_DIR" \
        --report "$OUTPUT_DIR/ncu.ncu-rep" \
        --source-report "$OUTPUT_DIR/ncu_source.ncu-rep" \
        --tag run || true
fi

# ============================================================
# Step 3c: Optional cross-run diff (--diff PREV_DIR)
# ============================================================
if [[ -n "$DIFF_DIR" && -n "$NCU_HELPERS" && -f "$NCU_HELPERS/row_key.py" ]]; then
    echo ""
    echo "=========================================="
    echo "  Step 3c: diff vs $DIFF_DIR"
    echo "=========================================="
    PREV_AN="$DIFF_DIR/analysis"
    CUR_AN="$OUTPUT_DIR/analysis"
    # (envelope basename, ranking field)
    for spec in "source_metrics_line_run:" "warp_stalls_line_run:total_samples" "warp_stalls_reason_run:total_samples"; do
        name="${spec%%:*}"
        field="${spec##*:}"
        if [[ -f "$PREV_AN/$name.json" && -f "$CUR_AN/$name.json" ]]; then
            sort_arg=()
            [[ -n "$field" ]] && sort_arg=(--sort-field "$field")
            python3 "$NCU_HELPERS/row_key.py" \
                --a "$PREV_AN/$name.json" --b "$CUR_AN/$name.json" \
                "${sort_arg[@]}" --output "$CUR_AN/diff_$name.txt" || true
        fi
    done
fi

# ============================================================
# Step 4: Symptom classification
# ============================================================
if [[ "$NO_CLASSIFY" != true && -f "$OUTPUT_DIR/analysis/metrics_key_run.json" ]]; then
    echo ""
    echo "=========================================="
    echo "  Step 4: Symptom classification"
    echo "=========================================="

    python3 "$SCRIPT_DIR/classify_ncu.py" \
        --metrics "$OUTPUT_DIR/analysis/metrics_key_run.json" \
        --output "$OUTPUT_DIR/summary.txt"

    echo ""
    cat "$OUTPUT_DIR/summary.txt"
elif [[ "$NO_CLASSIFY" != true && -z "$NCU_HELPERS" ]]; then
    echo ""
    echo "Skipping symptom classification (ncu-report-skill helpers not available)"
fi

echo ""
echo "=========================================="
echo "  Profile complete"
echo "=========================================="
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "Next steps:"
if [[ -f "$OUTPUT_DIR/summary.txt" ]]; then
    echo "  1. See $OUTPUT_DIR/summary.txt for bottleneck diagnosis"
    echo "  2. See $OUTPUT_DIR/analysis/metrics_key_run.txt for detailed metrics"
    if [[ "$COLLECT_SOURCE" == true ]]; then
        echo "  3. See $OUTPUT_DIR/analysis/stall_hotspots_run.txt for hotspot instructions"
    fi
    echo "  4. Query gpu-wiki based on the diagnosis for optimization suggestions"
elif [[ -f "$OUTPUT_DIR/analysis/metrics_key_run.txt" ]]; then
    # Metrics were parsed but classification was skipped (e.g. --no-classify).
    echo "  1. See $OUTPUT_DIR/analysis/metrics_key_run.txt for detailed metrics"
    echo "  2. Symptom classification was skipped; re-run without --no-classify for a summary.txt diagnosis"
else
    # No helpers available: only the raw report was produced.
    echo "  1. See $OUTPUT_DIR/ncu.ncu-rep (open with: ncu --import $OUTPUT_DIR/ncu.ncu-rep --page raw)"
    echo "  2. Metrics parsing was skipped: provide ncu helpers via --ncu-helpers or NCU_HELPERS, then re-run"
fi
