# Profile Tool Usage Guide

> Source: gpu-wiki (`/tmp/gpu-wiki/docs/ref-docs/`)
> - AMD: `amd/common/rocprofv3-profiling-guide.md`, `amd/gluon/gfx942/profiling_guide.md`
> - NVIDIA: `nvidia/common/ncu-profiling-guide.md`, `nvidia/gluon/sm90/profiling_guide.md`

This guide consolidates profile tool usage for both AMD and NVIDIA platforms.
All profiling evidence must come from these official tools 

---

## AMD: rocprofv3 + ATT (CDNA3/CDNA4)

### Quick Start: profile_kernel.sh

The recommended entry point for AMD profiling:

```bash
# Full profile (ATT + PMC + ASM)
bash tools/profile_kernel.sh kernel.py --output-dir profiles/v<N>

# PMC only (hardware counters)
bash tools/profile_kernel.sh kernel.py --output-dir profiles/v<N> --pmc-only

# ATT only (instruction-level trace)
bash tools/profile_kernel.sh kernel.py --output-dir profiles/v<N> --att-only

# ASM only (assembly extraction)
bash tools/profile_kernel.sh kernel.py --output-dir profiles/v<N> --asm-only

# Filter specific kernel
bash tools/profile_kernel.sh kernel.py --output-dir profiles/v<N> \
    --kernel-regex "<kernel_name>" --iteration-range 0-0
```

Default output directory is `./profile_output` when `--output-dir` is not specified.

Output structure:
```
<output-dir>/
├── att/                # Instruction-level trace
│   ├── stats_*.csv    # Per-dispatch latency/stall statistics
│   └── ui_output/     # Decoded trace UI data (JSON)
├── pmc/               # Hardware counter results
│   ├── batch1/        # SQ counters (bank conflict, VMEM, LDS)
│   ├── batch2/        # SPI counters (VGPR/LDS/wave stalls)
│   └── batch3/        # TCP counters (cache read/miss)
└── asm/               # AMDGCN assembly
    ├── *.amdgcn       # Triton/Gluon kernel assembly
    ├── *_final_isa.s  # FlyDSL kernel assembly
    └── compile.log    # Kernel compilation output log
```

### ATT: Instruction-Level Trace

#### Prerequisites

ATT decoding requires the `rocprof-trace-decoder` library ([GitHub](https://github.com/ROCm/rocprof-trace-decoder)). The `profile_kernel.sh` script **auto-detects** the decoder at:
```
1) $ROCPROF_TRACE_DECODER_DIR (if set)
2) /opt/rocm/lib
```
The script automatically sets `--att-library-path` and `LD_LIBRARY_PATH` — no manual configuration required.

#### ATT Command-Line Parameters

`profile_kernel.sh` uses the following hardcoded ATT geometry parameters (required for ROCm 7.2+ decoder to produce valid output):

```bash
rocprofv3 --att \
    --att-library-path "$DECODER_DIR" \
    --att-target-cu 1 \
    --att-shader-engine-mask 0xf \
    --att-simd-select 0xf \
    --att-buffer-size 0x6000000 \
    --output-format csv json \
    --kernel-iteration-range "[1]" \
    -d <output-dir>/att \
    -- python <kernel.py>
```

| Parameter | Value | Description |
|-----------|-------|-------------|
| `--att-target-cu` | `1` | Trace 1 CU per Shader Engine |
| `--att-shader-engine-mask` | `0xf` | Collect from all 4 Shader Engines |
| `--att-simd-select` | `0xf` | Collect all 4 SIMDs on the target CU |
| `--att-buffer-size` | `0x6000000` | 96MB trace buffer |
| `--output-format` | `csv json` | Only CSV + JSON (avoids heavy otf2/pftrace) |
| `--kernel-iteration-range` | `[1]` | Skip dispatch 0 (warmup/registration) |

**Configuration Notes**:

| Issue | Impact | Solution |
|-------|--------|----------|
| Missing `--output-format csv json` | rocprofv3 defaults to opaque `results.db`, no stats CSV | Always specify `csv json` |
| `output_format` includes otf2/pftrace | ~340MB useless files + ~10s finalization | Only use `csv json` |
| ATT geometry left at defaults (ROCm 7.2) | Decoder produces 0-byte output | Use explicit `--att-*` params as above |

#### Manual ATT Command (if not using profile_kernel.sh)

```bash
env LD_LIBRARY_PATH=/opt/rocm/lib64:/opt/rocm/lib:$LD_LIBRARY_PATH \
    rocprofv3 --att \
    --att-library-path <rocprof-trace-decoder-lib-dir> \
    --att-target-cu 1 \
    --att-shader-engine-mask 0xf \
    --att-simd-select 0xf \
    --att-buffer-size 0x6000000 \
    --output-format csv json \
    --kernel-iteration-range "[1]" \
    -d <output-dir>/att \
    -- python <kernel.py>
```

#### Locating the Target Kernel

The Gluon kernel is typically the **last dispatch** (largest dispatch_id) with the **largest stats CSV file** (~115KB vs <35KB for others). The script auto-selects the largest `stats_*.csv` for analysis.

```bash
# Find largest stats file = target Gluon kernel
ls -la <output-dir>/att/stats_*.csv | sort -k5 -nr | head -3
```

#### Analyzing stats_*.csv

Column definitions:

| Column | Meaning | Importance |
|--------|---------|------------|
| Instruction | Assembly instruction | Identify operation type |
| Hitcount | Execution count across all traced waves | Hotness |
| Latency | Total latency cycles = Stall + Issue/Execute | **Core metric** |
| Stall | Pipeline stall cycles | **Bottleneck indicator** |
| Idle | Idle cycles (register/data dependencies) | Dependency chain signal |
| Source | Source code line number | Pinpoint Python source |

Analysis commands:

```bash
STATS_FILE="profiles/v<N>/att/stats_ui_output_agent_<agent>_dispatch_<N>.csv"

# Top 15 high-latency instructions
sort -t',' -k5 -nr "$STATS_FILE" | head -16

# Top 10 high-stall instructions
sort -t',' -k6 -nr "$STATS_FILE" | head -11

# Check specific patterns
grep 'ds_bpermute' "$STATS_FILE"           # Layout conversion
grep 'scratch' "$STATS_FILE"               # Register spill
grep 'v_mfma' "$STATS_FILE"               # Matrix multiply
grep 's_waitcnt' "$STATS_FILE" | sort -t',' -k6 -nr  # Wait instructions
grep 'buffer_load_dword ' "$STATS_FILE"    # Narrow global loads (not dwordx4)
grep 'ds_read_b32' "$STATS_FILE"           # Narrow LDS reads (not b128)
```

### PMC: Hardware Counters

Counters are collected in batches due to mutual-exclusion limits. **If a batch fails, the script outputs a Warning and continues** (does not abort):

```bash
# Batch 1: SQ counters
rocprofv3 --pmc \
    SQ_LDS_BANK_CONFLICT,SQ_INSTS_VMEM_RD,SQ_INSTS_VMEM_WR,SQ_INSTS_LDS \
    --output-format csv \
    -d <output-dir>/pmc/batch1 \
    -- python kernel.py || echo "Warning: batch 1 counter collection failed"

# Batch 2: SPI counters
rocprofv3 --pmc \
    SPI_RA_VGPR_SGPR_FULL_CSN,SPI_RA_LDS_CU_FULL_CSN,SPI_RA_WAVE_SIMD_FULL_CSN \
    --output-format csv \
    -d <output-dir>/pmc/batch2 \
    -- python kernel.py || echo "Warning: batch 2 counter collection failed"

# Batch 3: TCP counters
rocprofv3 --pmc \
    TCP_TOTAL_READ,TCP_TCC_MISS \
    --output-format csv \
    -d <output-dir>/pmc/batch3 \
    -- python kernel.py || echo "Warning: batch 3 counter collection failed"
```

> **Note**: Each batch failure produces a Warning but does not terminate the script. Partial PMC results are still usable.

**Key Counter Categories**:

| Counter | Description | Diagnoses |
|---------|-------------|-----------|
| `SQ_LDS_BANK_CONFLICT` | Bank conflict stall cycles | swizzle/layout issues |
| `SQ_INSTS_VMEM_RD` | Vector memory read instructions | load throughput |
| `SQ_INSTS_VMEM_WR` | Vector memory write instructions | store throughput |
| `SQ_INSTS_LDS` | Total LDS instructions | LDS operation frequency |
| `SPI_RA_VGPR_SGPR_FULL_CSN` | Insufficient VGPR stall | register spilling |
| `SPI_RA_LDS_CU_FULL_CSN` | Insufficient LDS space | LDS over-limit |
| `SPI_RA_WAVE_SIMD_FULL_CSN` | Insufficient wave slots | occupancy issues |
| `TCP_TOTAL_READ` | L1 cache reads | cache utilization |
| `TCP_TCC_MISS` | L2 cache miss | cache efficiency |

### ASM: Assembly Analysis

The script supports two compilation backends for ASM extraction:

| Backend | Mechanism | Output Files |
|---------|-----------|-------------|
| Triton/Gluon | `TRITON_CACHE_DIR` redirect | `*.amdgcn` |
| FlyDSL | `FLYDSL_DUMP_IR=1` + `FLYDSL_DUMP_DIR` | `*_final_isa.s`, `*.mlir`, `*.ll` |

The ASM step runs the kernel with both environment variables set simultaneously. Compilation output is captured in `asm/compile.log` for debugging. If the kernel run exits non-zero, a Warning is printed but any produced ASM files are still collected.

Key patterns to check:

| Pattern | Good | Bad | Issue |
|---------|------|-----|-------|
| Global load width | `buffer_load_dwordx4` | `buffer_load_dword` | Insufficient vectorization |
| LDS read width | `ds_read_b128` | `ds_read_b32` | Narrow LDS access |
| LDS write width | `ds_write_b128` | `ds_write_b32` | Narrow LDS access |
| Register spill | None | `scratch_load/store` | VGPR overflow |
| Layout ops | Minimal | Many `ds_bpermute` | Excessive layout conversion |

### AMD Hot Instruction → Optimization Mapping

| Assembly Pattern | High Stall? | High Idle? | Issue | Action |
|-----------------|:-----------:|:----------:|-------|--------|
| `buffer_load_dword` (not x4) | — | — | Insufficient load width | Increase vectorization |
| `ds_read_b32` (not b128) | — | — | Narrow LDS read | Widen LDS access |
| `ds_read/write` | ✅ | — | Bank conflict | XOR swizzle layout |
| `ds_bpermute_b32` | — | — | Layout conversion overhead | Reduce permutations |
| `scratch_load/store` | ✅ | — | Register spilling | Reduce VGPR pressure |
| `buffer_load_*` | ✅ | — | Memory not overlapped | Software pipelining |
| `buffer_load_*` | — | ✅ | Dependency chain too long | Reorder instructions |
| `v_mfma_*` | — | ✅ | MFMA waiting for data | Prefetch / double buffer |

---

## NVIDIA: Nsight Compute (ncu) (Hopper sm_90)

### Quick Start

#### Step 0: Locate Target Kernel Launch Index (MANDATORY)

> **Critical**: PyTorch setup triggers 10-30 internal kernel launches before the target Gluon kernel.
> Blindly using `--launch-skip 10` will likely profile a wrong kernel.

```bash
ncu --print-summary per-kernel python <kernel.py>
```

Find the target Gluon kernel name and its launch index (e.g., index 23).

**⚠️ Diagnosis**: If the profiled kernel name is not your target (e.g., `distribution_elementwise_grid_stride_kernel`), the `--launch-skip` value is wrong.

#### Step 1: Collect Full Profile

```bash
ncu --set full \
    --launch-skip <N> --launch-count 1 \
    -o profiles/v<N>/ncu \
    python kernel.py
```

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `--set full` | All sections (most comprehensive) | Default |
| `--launch-skip N` | Skip first N launches | **Use index from Step 0** |
| `--launch-count 1` | Profile only 1 launch | 1 |
| `-o <file>` | Output .ncu-rep report | Required |

#### Alternative: Filter by Kernel Name

```bash
ncu --set full \
    --kernel-name "chunk_gated_delta_rule_fwd" \
    --launch-count 1 \
    -o profiles/v<N>/ncu \
    python kernel.py
```

#### Quick Metrics Only (Faster)

```bash
ncu --metrics \
    sm__throughput.avg.pct_of_peak_sustained_elapsed,\
    dram__throughput.avg.pct_of_peak_sustained_elapsed,\
    l1tex__throughput.avg.pct_of_peak_sustained_elapsed,\
    launch__occupancy \
    --launch-skip <N> --launch-count 1 \
    python kernel.py
```

### Viewing Profile Data (CLI)

```bash
# View full report
ncu --import profile.ncu-rep --page raw

# View specific metrics
ncu --import profile.ncu-rep --metrics \
    sm__throughput.avg.pct_of_peak_sustained_elapsed,\
    dram__throughput.avg.pct_of_peak_sustained_elapsed

# View SASS source
ncu --import profile.ncu-rep --page source --print-source sass

# Export as CSV
ncu --import profile.ncu-rep --page raw --csv > metrics.csv
```

### Key Metrics to Extract

#### Compute Throughput

| Metric | Meaning |
|--------|---------|
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | SM throughput (% of peak) |
| `sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed` | Tensor Core activity |
| `smsp__inst_executed.sum` | Total executed instructions |

#### Memory Throughput

| Metric | Meaning |
|--------|---------|
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | DRAM throughput (% of peak) |
| `dram__bytes_read.sum` | Total DRAM read bytes |
| `dram__bytes_write.sum` | Total DRAM write bytes |
| `l1tex__throughput.avg.pct_of_peak_sustained_elapsed` | L1/Tex/Shared memory throughput |

#### Shared Memory Bank Conflicts

| Metric | Meaning |
|--------|---------|
| `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` | Shared memory load bank conflicts |
| `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum` | Shared memory store bank conflicts |

#### Occupancy / Resources

| Metric | Meaning |
|--------|---------|
| `launch__occupancy` | Achieved occupancy (%) |
| `launch__registers_per_thread` | Registers per thread |
| `launch__shared_mem_per_block_dynamic` | Dynamic shared memory (bytes) |

#### Warp Stall Reasons

| Metric | Meaning | Optimization |
|--------|---------|--------------|
| `smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct` | Waiting for global memory | Improve memory pipeline |
| `smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct` | Waiting for shared memory / L1 | Reduce bank conflicts |
| `smsp__warp_issue_stalled_wait_per_warp_active.pct` | Waiting for synchronization | Reduce barriers |
| `smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct` | Math pipeline backpressure | Compute units saturated |
| `smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct` | MIO pipeline backpressure | smem access bottleneck |
| `smsp__warp_issue_stalled_not_selected_per_warp_active.pct` | Warp ready but not scheduled | Insufficient occupancy |

### SASS Instruction → Optimization Mapping

| SASS Pattern | Meaning | Issue | Action |
|--------------|---------|-------|--------|
| `LDG.E.32` | 32-bit global load | Load width insufficient | Vectorize to 128-bit |
| `LDG.E.128` | 128-bit global load | ✅ Optimal | — |
| `STS.32` | 32-bit shared store | Narrow smem access | Widen to 128-bit |
| `STS.128` / `LDS.128` | 128-bit shared access | ✅ Optimal | — |
| `LDSM.16.M88.4` | Load shared matrix | wgmma data loading | — |
| `HGMMA.64x*x*.F32.BF16` | Hopper GMMA | ✅ Tensor Core used | — |
| `STL` | Store to local memory | Register spilling ❌ | Reduce register pressure |
| `LDL` | Load from local memory | Register spilling ❌ | Reduce register pressure |
| `LDGSTS` | CP_ASYNC (global→shared) | ✅ async_copy used | — |
| `BAR.SYNC` | Barrier synchronization | May be bottleneck | Reduce sync points |

### Bottleneck Analysis via SOL%

The **SpeedOfLight** section is the starting point:

- **High Compute SOL%** → compute-bound
- **High Memory SOL%** → memory-bound
- **Both low** → latency or occupancy issues

---

## Complete Profile Workflow

### AMD (CDNA3/CDNA4)

By default, `profile_kernel.sh` executes **all three steps** (ASM → ATT → PMC) in sequence. Use `--asm-only`, `--att-only`, or `--pmc-only` to run individual steps.

```
Step 1: Measure kernel time
  └─ python tools/measure_kernel_time.py kernel.py

Step 2: Compute utilization
  └─ python tools/compute_utilization.py --gpu <gpu> --dtype <dtype> \
       --flops-expr '<expr>' --bytes-expr '<expr>' --time-ms <ms>

Step 3: If utilization < 90%, run full profile (default: ASM + ATT + PMC)
  └─ bash tools/profile_kernel.sh kernel.py --output-dir profiles/v<N>
  └─ Script auto-executes: Step 0 (ASM) → Step 1 (ATT) → Step 2 (PMC)

Step 4: Locate target kernel (auto-done by script, or manually)
  └─ ls -la profiles/v<N>/att/stats_*.csv | sort -k5 -nr | head -3

Step 5: Analyze hot instructions
  └─ sort/grep stats CSV → Identify bottleneck instructions

Step 6: Confirm diagnosis with hardware counters
  └─ Check profiles/v<N>/pmc/batch{1,2,3}/*.csv

Step 7: Inspect assembly for structural issues
  └─ Check profiles/v<N>/asm/*.amdgcn or *_final_isa.s

Step 8: Execute optimization based on diagnosis

Step 9: Clean up trace files (~68MB per iteration)
```

### NVIDIA (Hopper sm_90)

```
Step 0: Locate target kernel launch index ⚠️ (MANDATORY)
  └─ ncu --print-summary per-kernel python kernel.py
  └─ Find target kernel name and index N

Step 1: Measure kernel time
  └─ python tools/measure_kernel_time.py kernel.py

Step 2: Compute utilization
  └─ python tools/compute_utilization.py --gpu <gpu> --dtype <dtype> \
       --flops-expr '<expr>' --bytes-expr '<expr>' --time-ms <ms>

Step 3: If utilization < 90%, run ncu profile
  └─ ncu --set full --launch-skip <N> --launch-count 1 -o profiles/v<N>/ncu python kernel.py
  └─ ⚠️ Confirm kernel name is target, not PyTorch internal kernel

Step 4: View key metrics
  └─ ncu --import profiles/v<N>/ncu.ncu-rep --page raw

Step 5: Analyze warp stall reasons
  └─ smsp__warp_issue_stalled_* → Locate bottleneck type

Step 6: View SASS (optional, deeper)
  └─ ncu --import profiles/v<N>/ncu.ncu-rep --page source --print-source sass

Step 7: Execute optimization based on diagnosis
```

---

## Evidence Format

All profile conclusions must be written as:

```
evidence -> inference -> optimization action
```

Examples:

- `PMC shows SQ_LDS_BANK_CONFLICT = 125000` → `LDS bank conflicts are significant` → `try XOR16 swizzled layout`
- `ASM shows many buffer_load_dword and few dwordx4` → `global memory vectorization insufficient` → `adjust alignment and vector width`
- `ncu shows long_scoreboard dominates warp stalls at 45%` → `latency hiding insufficient` → `try double buffering or software pipelining`
- `ATT stats shows v_mfma high Idle (45%)` → `MFMA waiting for data` → `overlap loads with compute via software pipelining`

---

## Common Issues

### AMD

| Problem | Cause | Solution |
|---------|-------|----------|
| `rocprofv3: command not found` | ROCm not in PATH | `export PATH=/opt/rocm/bin:$PATH` |
| ATT decoder fails | Missing library | Install `rocprof-trace-decoder` from [GitHub](https://github.com/ROCm/rocprof-trace-decoder) or set `ROCPROF_TRACE_DECODER_DIR` |
| Stats CSV empty | Wrong dispatch | Use `--iteration-range "[1]"` to skip warmup |
| PMC batch fails with Warning | Counter mutual exclusion or GPU busy | Retry; partial results from other batches are still usable |
| No `.amdgcn` files produced | Kernel uses JIT cache, no recompilation | Check `asm/compile.log`; clear Triton cache or set `TRITON_CACHE_DIR` manually |
| Large trace files (~500MB) | Missing `--output-format csv json` | Script sets this by default; for manual runs always specify it |

### NVIDIA

| Problem | Cause | Solution |
|---------|-------|----------|
| Permission denied / ERR_NVGPUCTRPERM | Needs root or perf permission | `sudo ncu ...` or `echo 0 > /proc/sys/kernel/perf_event_paranoid` |
| Profiled wrong kernel | `--launch-skip` incorrect | Re-run `ncu --print-summary per-kernel` to find correct index |
| Sections show N/A | Used `--metrics` not `--set full` | Use `--set full` for complete data |
| Very long profile time | Too many launches | Add `--launch-count 1` and `--kill yes` |
