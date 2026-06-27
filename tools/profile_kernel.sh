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
# Instruction-level profiling tool for Gluon kernels
#
# Usage:
#     bash tools/profile_kernel.sh <kernel.py> [Options]
#
# Examples:
#     bash tools/profile_kernel.sh my_kernel.py --output-dir ./profile
#     bash tools/profile_kernel.sh my_kernel.py --kernel-regex "matmul_kernel" --pmc-only
#
# [TODO] Ensure rocprofv3 is in PATH and run this on an AMD GPU environment

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Auto-detect rocprof-trace-decoder library location.
# Priority: 1) ROCPROF_TRACE_DECODER_DIR env var  2) local clone under tools/  3) /opt/rocm/lib
# Install from: https://github.com/ROCm/rocprof-trace-decoder.git
if [[ -n "${ROCPROF_TRACE_DECODER_DIR:-}" ]]; then
    DECODER_DIR="$ROCPROF_TRACE_DECODER_DIR"
elif [[ -d "$SCRIPT_DIR/rocprof-trace-decoder" ]]; then
    # Prefer the repo cloned by install.sh into tools/rocprof-trace-decoder
    if [[ -f "$SCRIPT_DIR/rocprof-trace-decoder/releases/linux_glibc_2_28_x86_64/librocprof-trace-decoder.so" ]]; then
        DECODER_DIR="$SCRIPT_DIR/rocprof-trace-decoder/releases/linux_glibc_2_28_x86_64"
    else
        DECODER_DIR="$SCRIPT_DIR/rocprof-trace-decoder"
    fi
elif [[ -f /opt/rocm/lib/librocprof-trace-decoder.so ]]; then
    DECODER_DIR="/opt/rocm/lib"
else
    DECODER_DIR="/opt/rocm/lib"
fi

KERNEL_FILE=""
OUTPUT_DIR="./profiles/v0"
KERNEL_REGEX=""
RUN_ATT=true
RUN_PMC=true
RUN_ASM=true
# Iteration filter for ATT. Use the list/range syntax accepted by
# rocprofv3 (e.g. "[1]", "[0,2]", "1-5"). Default skips dispatch 0 which is
# typically the warmup/registration call that has no ATT-decodable payload.
ITERATION_RANGE="[1]"

usage() {
    cat <<EOF
Usage: $0 <kernel.py> [Options]

Options:
    --output-dir DIR        Output directory (default: ./profile_output)
    --kernel-regex REGEX    Filter kernel names
    --iteration-range RANGE dispatch iteration range, list or range syntax
                            (e.g. "[1]", "[0,2]", "1-5"; default: "[1]" to
                            skip the warmup/registration dispatch)
    --pmc-only              Collect hardware counters only
    --att-only              Collect instruction-level trace only
    --asm-only              Extract AMDGCN assembly only
    -h, --help              Show help

Output:
    <output-dir>/att/       instruction-level trace (stats_*.csv)
    <output-dir>/pmc/       hardware counter results (*.csv)
    <output-dir>/asm/       AMDGCN assembly (*.amdgcn)
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --kernel-regex)
            KERNEL_REGEX="$2"
            shift 2
            ;;
        --iteration-range)
            ITERATION_RANGE="$2"
            shift 2
            ;;
        --pmc-only)
            RUN_ATT=false
            RUN_PMC=true
            RUN_ASM=false
            shift
            ;;
        --att-only)
            RUN_ATT=true
            RUN_PMC=false
            RUN_ASM=false
            shift
            ;;
        --asm-only)
            RUN_ATT=false
            RUN_PMC=false
            RUN_ASM=true
            shift
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
    echo "Error: kernel file is required"
    usage
fi

if ! command -v rocprofv3 &>/dev/null; then
    echo "Error: rocprofv3 was not found. Ensure ROCm is installed and rocprofv3 is in PATH"
    exit 1
fi

if [[ ! -f "$DECODER_DIR/librocprof-trace-decoder.so" ]]; then
    echo "Error: ATT decoder library not found at $DECODER_DIR"
    echo "Expected: $DECODER_DIR/librocprof-trace-decoder.so"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# Make the decoder discoverable by both rocprofv3 (--att-library-path) and the
# dlopen() it performs at finalization (LD_LIBRARY_PATH).
export LD_LIBRARY_PATH="$DECODER_DIR:/opt/rocm/lib64:/opt/rocm/lib:${LD_LIBRARY_PATH:-}"

KERNEL_FILTER=()
if [[ -n "$KERNEL_REGEX" ]]; then
    KERNEL_FILTER=(--kernel-include-regex "$KERNEL_REGEX")
fi

# ============================================================
# Step 0: AMDGCN Assembly extraction
# ============================================================
# Force the compiler to dump assembly into <output-dir>/asm/. Two backends
# are supported in this repo:
#   - Triton/Gluon: TRITON_CACHE_DIR drops .amdgcn files into a hash dir
#   - FlyDSL:       FLYDSL_DUMP_IR=1 + FLYDSL_DUMP_DIR drops *.s/*.mlir/*.ll
#                   (FLYDSL_DUMP_IR=1 also bypasses the JIT cache, so the
#                   compilation actually runs even on cache hit.)
# Done before ATT/PMC because it is the cheapest step and useful even if
# rocprofv3 fails later.
if [[ "$RUN_ASM" == true ]]; then
    echo "=========================================="
    echo "  Step 0: Extracting AMDGCN assembly"
    echo "=========================================="

    ASM_DIR="$OUTPUT_DIR/asm"
    TRITON_DIR="$OUTPUT_DIR/.triton_cache"
    rm -rf "$ASM_DIR" "$TRITON_DIR"
    mkdir -p "$ASM_DIR" "$TRITON_DIR"

    if env TRITON_CACHE_DIR="$TRITON_DIR" \
           FLYDSL_DUMP_IR=1 \
           FLYDSL_DUMP_DIR="$ASM_DIR" \
           python "$KERNEL_FILE" >"$ASM_DIR/compile.log" 2>&1; then
        echo "  kernel run OK (log: $ASM_DIR/compile.log)"
    else
        echo "  Warning: kernel run exited non-zero; ASM may still be present (log: $ASM_DIR/compile.log)"
    fi

    # Pull Triton/Gluon .amdgcn files (if any) up next to the FlyDSL dumps.
    TRITON_COUNT=0
    while IFS= read -r -d '' f; do
        hash_dir="$(basename "$(dirname "$f")")"
        cp "$f" "$ASM_DIR/${hash_dir}_$(basename "$f")"
        TRITON_COUNT=$((TRITON_COUNT + 1))
    done < <(find "$TRITON_DIR" -name '*.amdgcn' -print0 2>/dev/null)
    rm -rf "$TRITON_DIR"

    FLYDSL_ASM_COUNT=$(find "$ASM_DIR" -name '*_final_isa.s' 2>/dev/null | wc -l)
    TOTAL=$((TRITON_COUNT + FLYDSL_ASM_COUNT))

    if [[ "$TOTAL" -gt 0 ]]; then
        echo "  Collected $TRITON_COUNT .amdgcn + $FLYDSL_ASM_COUNT FlyDSL ISA file(s) -> $ASM_DIR"
    else
        echo "  Warning: no .amdgcn or *_final_isa.s files were produced. Check $ASM_DIR/compile.log."
        echo "  Common causes: the kernel file does not trigger compilation when run,"
        echo "  or the backend writes ASM to a non-standard location."
    fi
fi

# ============================================================
# Step 1: Instruction-level Trace (ATT)
# ============================================================
if [[ "$RUN_ATT" == true ]]; then
    echo "=========================================="
    echo "  Step 1: Collecting instruction-level trace (ATT)"
    echo "=========================================="

    ATT_DIR="$OUTPUT_DIR/att"
    mkdir -p "$ATT_DIR"

    # Settings explained:
    #   --att-library-path       — points rocprofv3 at the decoder .so so the
    #                              raw .out trace data gets turned into
    #                              stats_*.csv + ui_output/*.json.
    #   --output-format csv json — without this rocprofv3 defaults to rocpd
    #                              (an opaque results.db) and never writes
    #                              the stats CSV the rest of this script
    #                              consumes.
    #   --att-target-cu/-shader-engine-mask/-simd-select/-buffer-size
    #                            — required ATT geometry. With ROCm 7.2 the
    #                              decoder produces 0-byte output if these
    #                              are left at defaults, even on real waves.
    #   We deliberately do NOT pass --att-activity here. It implies a hardware
    #   activity-counter mode that on this rocprofv3/decoder build short-circuits
    #   the trace decode and leaves only raw .out code-object dumps behind.
    rocprofv3 --att \
        --att-library-path "$DECODER_DIR" \
        --att-target-cu 1 \
        --att-shader-engine-mask 0xf \
        --att-simd-select 0xf \
        --att-buffer-size 0x6000000 \
        --output-format csv json \
        "${KERNEL_FILTER[@]}" \
        --kernel-iteration-range "$ITERATION_RANGE" \
        -d "$ATT_DIR" \
        -- python "$KERNEL_FILE"

    echo ""
    echo "ATT output directory: $ATT_DIR"

    # Each matched dispatch produces its own stats CSV. Pick the largest one —
    # warmup/registration dispatches end up as header-only 77-byte files, while
    # the real target kernel produces hundreds of KB.
    STATS_FILE=$(find "$ATT_DIR" -name "stats_*.csv" -printf '%s %p\n' 2>/dev/null \
                 | sort -rn | head -1 | cut -d' ' -f2-)
    if [[ -n "$STATS_FILE" ]]; then
        # The Instruction column contains commas inside quotes, so plain
        # `sort -t,` splits the row mid-field and sorts by garbage. Parse with
        # Python's csv module instead so column indices stay aligned.
        python - "$STATS_FILE" <<'PY'
import csv, sys
path = sys.argv[1]
with open(path, newline='') as f:
    rows = list(csv.reader(f))
header, body = rows[0], rows[1:]
def col(name):
    return header.index(name)
ci, cl, cs = col('Instruction'), col('Latency'), col('Stall')

def top(n, key):
    return sorted(body, key=lambda r: int(r[key]), reverse=True)[:n]

def fmt(row):
    return f"  lat={row[cl]:>8}  stall={row[cs]:>8}  hit={row[col('Hitcount')]:>6}  {row[ci]}"

print('\n--- Top 15 high-latency instructions ---')
for r in top(15, cl):
    print(fmt(r))

print('\n--- Top 10 high-stall instructions ---')
for r in top(10, cs):
    print(fmt(r))

print('\n--- ds_bpermute instructions ---')
hits = [r for r in body if 'ds_bpermute' in r[ci]]
print('(No ds_bpermute instructions)' if not hits else '\n'.join(fmt(r) for r in hits))

print('\n--- scratch operations ---')
hits = [r for r in body if 'scratch' in r[ci].lower()]
print('(no scratch operations)' if not hits else '\n'.join(fmt(r) for r in hits))
PY
    else
        echo "Warning: stats_*.csv file was not found under $ATT_DIR."
        echo "  Verify that the decoder library at $DECODER_DIR loaded successfully."
        echo "  (Look for 'rocprof-trace-decoder' messages in rocprofv3 stderr.)"
    fi
fi

# ============================================================
# Step 2: hardware counters (PMC)
# ============================================================
if [[ "$RUN_PMC" == true ]]; then
    echo ""
    echo "=========================================="
    echo "  Step 2: Collecting hardware counters (PMC)"
    echo "=========================================="

    PMC_DIR="$OUTPUT_DIR/pmc"
    mkdir -p "$PMC_DIR"

    # [TODO] Counters may have mutual-exclusion limits; collect in batches if needed
    rocprofv3 --pmc \
        SQ_LDS_BANK_CONFLICT,SQ_INSTS_VMEM_RD,SQ_INSTS_VMEM_WR,SQ_INSTS_LDS \
        --output-format csv \
        "${KERNEL_FILTER[@]}" \
        -d "${PMC_DIR}/batch1" \
        -- python "$KERNEL_FILE" || echo "Warning: batch 1 counter collection failed"

    rocprofv3 --pmc \
        SPI_RA_VGPR_SGPR_FULL_CSN,SPI_RA_LDS_CU_FULL_CSN,SPI_RA_WAVE_SIMD_FULL_CSN \
        --output-format csv \
        "${KERNEL_FILTER[@]}" \
        -d "${PMC_DIR}/batch2" \
        -- python "$KERNEL_FILE" || echo "Warning: batch 2 counter collection failed"

    rocprofv3 --pmc \
        TCP_TOTAL_READ,TCP_TCC_MISS \
        --output-format csv \
        "${KERNEL_FILTER[@]}" \
        -d "${PMC_DIR}/batch3" \
        -- python "$KERNEL_FILE" || echo "Warning: batch 3 counter collection failed"

    echo ""
    echo "PMC output directory: $PMC_DIR"
fi

echo ""
echo "=========================================="
echo "  Profile complete"
echo "=========================================="
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "  1. Inspect $OUTPUT_DIR/att/**/stats_*.csv to analyze hot instructions"
echo "  2. Inspect $OUTPUT_DIR/pmc/**/*.csv to analyze hardware counters"
echo "  3. Inspect $OUTPUT_DIR/asm/*.amdgcn to analyze generated assembly"
echo "  4. optimize using references/common_optimizations.md and references/patterns/"
