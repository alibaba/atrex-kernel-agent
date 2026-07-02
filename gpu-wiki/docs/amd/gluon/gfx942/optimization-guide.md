# Gluon Kernel Performance Optimization Guide

## Applicability

This guide covers:
- "Optimize the performance of this Gluon kernel"
- "This kernel is too slow, help me analyze it"
- "Help me improve the compute utilization of this GPU kernel"
- "Analyze where the bottleneck is in this kernel"
- "Profile this Gluon operator"

## Prerequisites

This guide assumes the input code is already a **correctly compiling and runnable Gluon kernel**. If conversion from Triton to Gluon is needed, first use the CDNA3 conversion guidance in `../../../../converter/amd/cdna3/conversion-guide.md`.

## Acceptance Checklist

Use the following criteria to decide whether an optimization pass is complete:

- Gluon TFLOPS is at least 1.5x the Triton baseline, or
- Gluon TFLOPS reaches at least 195T, or
- the §1.6 checklist is exhausted and every applicable item is marked done or not applicable with a reason.

Record these artifacts with the result:

- Roofline analysis: bottleneck type, tile AI, CU utilization, and performance upper bound.
- Completed optimization checklist from §1.6.
- Accuracy and performance verification results.
- Baseline versus optimized TFLOPS comparison.

### Anti-pattern: Conversion without Optimization (BLOCKING)

The following situations are **strictly prohibited** from being considered as optimization completion:
- ❌ Directly benchmarking after converting from Triton to Gluon without entering the Step 1-5 optimization loop
- ❌ Stopping with the justification that "the performance gap is reasonable because Triton has automatic optimizations (in_thread_transpose / sched_barrier / num_stages)"
- ❌ Accuracy passes + performance numbers exist = done (**having numbers ≠ optimization complete**)

**Conversion is not optimization.** After conversion, run the Step 1-5 optimization loop until the stopping criteria are met.

This guide may directly reference the following local wiki content to avoid duplication:
- **Gluon API Reference**: `../../../../converter/amd/common/porting_rules.md`
- **Layout Types**: `../../../../converter/amd/cdna3/layouts.md`
- **Memory Access Patterns**: `../../../../converter/amd/cdna3/memory_access.md`
- **Matrix Multiplication Patterns**: `../../../../converter/amd/cdna3/matrix_multiply.md`
- **Pipeline Patterns**: `../../../../converter/amd/cdna3/pipeline.md`
- **Accuracy verification**: run the local accuracy check used by the consuming harness
- **Performance benchmarking**: run the local benchmark used by the consuming harness
- **TTGIR/layout inspection**: inspect generated IR or layout metadata when layout changes are involved

## Core Optimization Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│ Gluon Kernel optimizationworkflow │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌────────────────────────────┐                                 │
│ │ Step 1: bottleneckanalysisutilizationevaluation │◄────────────────────────┐ │
│  │         (Roofline Model)    │                         │      │
│  └──────────┬─────────────────┘                         │      │
│             │                                           │      │
│      ┌──────▼──────┐                                    │      │
│ │ utilization ≥ 90% │──YES──► outputoptimizationsummary, │ │
│ │ ornoneoptimization? │ │ │
│      └──────┬──────┘                                    │      │
│             │NO                                         │      │
│             ▼                                           │      │
│  ┌──────────────────────┐                               │      │
│ │ Step 2: instruction-level Profile │ │ │
│ │ (rocprofv3 analysis) │ │ │
│  └──────────┬───────────┘                               │      │
│             │                                           │      │
│             ▼                                           │      │
│  ┌──────────────────────────────┐                         │      │
│ │ Step 3: iterationoptimization │ │ │
│ │ 3.0 coalesced ⚠️ (step zero) │ │ │
│ │ 3.1 load/store │ │ │
│ │ 3.2 swizzle bankconflict │ │ │
│ │ 3.3 ds_bpermute │ │ │
│ │ 3.4 scratch │ │ │
│ │ 3.5 stall optimization │ │ │
│  │  3.6 warp_pipeline_stage      │                        │      │
│ │ stage ⭐ (GEMMcritical) │ │ │
│ │ 3.7 layout+num_warps │ │ │
│ │ ⭐ (Tilecritical) │ │ │
│ │ 3.8 Attention optimization ⭐ │ │ │
│ │ (full/maskedseparate, │ │ │
│ │ SE-level zigzag) │ │ │
│ │ 3.9 matrix/lowCUutilization ⭐ │ │ │
│ │ (split+BLOCK_K) │ │ │
│ │ (verificationaccuracy+performance) │ │ │
│  └──────────┬───────────────────┘                        │      │
│             │                                           │      │
│             └───────────────────────────────────────────┘      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Step 1: Bottleneck Analysis and Utilization Assessment (Roofline Model)
Before optimization, **you must first determine whether the operator is compute-bound or memory-bound**, as this determines the subsequent optimization direction and utilization assessment metrics.

### 1.1 Build a Roofline Model — Determine the Bottleneck Type

> For the **complete methodology**, see [AMD GPU Roofline Analysis Methodology](../../common/roofline-analysis-methodology.md), including tile-level AI computation, Ridge Point tables, bottleneck determination rules, and tile size selection decision flow.

### 1.2 Measure the Actual Performance of the Current Case

#### Latency Measurement

```bash
# use local benchmarkelapsed time
python <benchmark-command> <kernel.py> <ref.py> \
    --wrapper-name <wrapper> --setup-name <setup>

# or use the single-kernel measurement helper
python tools/measure_kernel_time.py <kernel.py> --wrapper-name <wrapper> --setup-name <setup>
```

#### Calculate Actual Compute Throughput and Actual Bandwidth

```
actual (TFLOPS) = FLOPs / elapsed time (seconds) / 1e12
actualbandwidth (TB/s) = Bytes_transferred / elapsed time (seconds) / 1e12
```

### 1.3 Assess Utilization

> For **utilization formulas and peak parameters**, see [AMD GPU Roofline Analysis Methodology §4](../../common/roofline-analysis-methodology.md#4-utilization-assessment).

**Gluon Tool Invocation**:

```bash
# data: bandwidthupper bound
python tools/measure_bandwidth_ceiling.py --bytes <Bytes> --dtype bf16 --emit-flag

# completeutilizationanalysis
python tools/compute_utilization.py \
    --flops <FLOPs> --bytes <Bytes> --time-ms <time> \
    --gpu mi300x --dtype bf16 \
    --measured-bandwidth-tb-s 3.200
```

### 1.4 CU Utilization Pre-check

> For the **CU utilization quick reference table and decision rules**, see [AMD GPU Roofline Analysis Methodology §3](../../common/roofline-analysis-methodology.md#3-cu-utilization-pre-check). When CU_ratio < 10%, skip ISA-level optimization and proceed directly to §3.9.

### 1.5 Theoretical Peak Performance Assessment

> For **calculation methods and examples**, see [AMD GPU Roofline Analysis Methodology §5](../../common/roofline-analysis-methodology.md#5-theoretical-performance-upper-bound-assessment). When the theoretical peak performance is far below the hardware peak, the problem lies not in ISA-level optimization but at the operator configuration level.

**Gluon Tool Invocation**:

```bash
python tools/compute_utilization.py \
    --gpu mi308x --dtype bf16 \
    --flops-expr "2*BM*BN*K" --bytes-expr "(BM*K + BN*K + BM*BN)*2" \
    --time-ms 0.5 --grid-blocks 64 --num-cus 80
```

### 1.6 Stop Conditions

- **Utilization ≥ 90%** (use compute utilization for compute-bound operators, bandwidth utilization for memory-bound operators) → Stop optimization; output results.
- **Checklist exhausted** → Optimization can be deemed to have "no room for improvement" only when all applicable items in the checklist below have been attempted or have a clearly stated reason for being skipped.
- Otherwise → Proceed to Step 2.
**⚠️ The following do not satisfy the stop condition; stopping is prohibited:**
- "Precision passed" — Precision is a prerequisite, not a stop condition.
- "Benchmark numbers are available" — Having numbers only means it runs, not that optimization is complete.
- "The gap is reasonable because Triton has auto-tuning" — This guide's checklist contains equivalent manual optimization items (§3.6 warp_pipeline_stage, §3.7 num_warps joint tuning, etc.) that must be empirically tested.
- "Conversion is done" — Conversion is separate from optimization and is not a stop condition for this guide.

**⚠️ Mandatory Checklist Audit When Utilization < 90% (Blocking Requirement)**

When utilization < 90%, it is **prohibited** to stop on the grounds of "no room for improvement." Each item in the following checklist must be reviewed, and a status must be filled in for each before stopping:

| # | Optimization Item | Applicable Condition | Status (Required) |
|---|-------------------|---------------------|--------------------|
| 3.0 | Coalesced memory access + order correctness | All kernels | ✅ Done / ⬚ Not done / ➖ N/A (reason) |
| 3.1 | load/store width dwordx4 | All kernels | ✅ / ⬚ / ➖ |
| 3.2 | swizzle bank conflict | Kernels using shared memory | ✅ / ⬚ / ➖ |
| 3.3 | ds_bpermute elimination | Kernels with convert_layout | ✅ / ⬚ / ➖ |
| 3.4 | scratch elimination | All GEMM/Attention | ✅ / ⬚ / ➖ |
| 3.5 | Memory stall / software pipeline | Kernels with num_stages > 1 | ✅ / ⬚ / ➖ |
| 3.6 | warp_pipeline_stage | GEMM kernels | ✅ / ⬚ / ➖ |
| 3.7 | layout + num_warps joint tuning | Large-tile GEMM (BM≥128) | ✅ / ⬚ / ➖ |
| 3.8 | Attention-specific optimization | Attention kernels | ✅ / ⬚ / ➖ |
| 3.9 | Small matrix / CU utilization | grid_blocks < num_CUs | ✅ / ⬚ / ➖ |
| XCD | XCD/PID remapping | MI308X (4 XCD) | ✅ / ⬚ / ➖ |**Rules:**
1. All **⬚ Not Done** items must be attempted before returning to this checklist
2. All **➖ Not Applicable** items must have a reason stated (e.g., "1D reduction has no MFMA, N/A for §3.6")
3. Only when all applicable items are ✅ or ➖ can the "checklist exhausted" determination be made and the process stopped
4. **Strictly prohibited** from skipping any ⬚ Not Done items on the grounds of "estimated minimal benefit" — must verify through actual measurement

### Tool Invocation

```bash
# Tile Roofline analysis(recommended)
# --flops / --flops-expr: single tile FLOPs
# --bytes / --bytes-expr: single tile HBM total bytes
# --time-ms: entire kernel elapsed time(toolautomatic grid blocks obtain per-tile elapsed time)
python tools/compute_utilization.py \
    --gpu mi308x \
    --dtype bf16 \
    --flops-expr "2*BM*BN*K" \
    --bytes-expr "(BM*K + BN*K + BM*BN)*2" \
    --time-ms 0.5 \
    --grid-blocks 64

# data kernel: usebandwidthupper bound
python tools/compute_utilization.py \
    --flops-expr "2*BM*BN*K" \
    --bytes-expr "(BM*K + BN*K + BM*BN)*2" \
    --time-ms 0.05 \
    --grid-blocks 16 \
    --gpu mi308x \
    --dtype bf16 \
    --measured-bandwidth-tb-s 3.2

# ordirect tile
python tools/compute_utilization.py \
    --flops 134217728 \
    --bytes 1212416 \
    --time-ms 0.5 \
    --grid-blocks 64 \
    --gpu mi308x \
    --dtype bf16
```

---

## Step 2: Instruction-Level Profile Analysis

### Using rocprofv3 for Instruction-Level Analysis

```bash
# Instruction-level profile (configure kernel path and wrapper name)
bash tools/profile_kernel.sh <kernel.py> --wrapper-name <wrapper> --output-dir ./profile_output
```

### Analysis Steps

1. **Collect instruction-level trace** (verified command)
   ```bash
   env LD_LIBRARY_PATH=/opt/rocm/lib64:/opt/rocm/lib:$LD_LIBRARY_PATH \
       rocprofv3 --att \
       --att-library-path ./tools/rocprof-trace-decoder-amd-mainline/releases/linux_glibc_2_28_x86_64 \
       -i tools/input_att.yaml \
       -- python <kernel.py>
   ```

2. **Collect hardware counters**
   ```bash
   rocprofv3 --pmc \
       SQ_LDS_BANK_CONFLICT,SQ_INSTS_VMEM_RD,SQ_INSTS_VMEM_WR,\
       SPI_RA_VGPR_SGPR_FULL_CSN,TCP_TCC_MISS \
       -d ./pmc_output -- python <kernel.py>
   ```

3. **Analyze hotspot instructions**
   ```bash
 # by Latency descending ordercolumn
   sort -t',' -k5 -nr ./profile_output/stats_*.csv | head -20

 # by Stall descending ordercolumn
   sort -t',' -k6 -nr ./profile_output/stats_*.csv | head -20
   ```

4. **Clean up trace files** (must be performed after analysis)
   ```bash
 # trace output ~400-500MB, analysiscompletemustcleanup
 rm -rf tt_test # or input_att.yaml configuration output_directory
   ```

For detailed profile interpretation guide, see `../../../../ref-docs/amd/gluon/gfx942/profiling_guide.md` (including complete cleanup strategy).

### Profile Output Format (stats_*.csv)

| Column Name | Meaning |
|------|------|
| Instruction | Assembly instruction |
| Hitcount | Execution count |
| Latency | Total latency cycles = Stall + Issue |
| Stall | Pipeline stall cycles (TCP/LDS backpressure) |
| Idle | Idle cycles (register dependency, icache miss) |
| Source | Corresponding source code line number |

### Instruction Types to Watch

| Instruction Pattern | Potential Issue | Corresponding Optimization Step |
|----------|---------|-------------|
| `buffer_load_dword` (not dwordx4) | Insufficient load width | Step 3.1 |
| `buffer_store_dword` (not dwordx4) | Insufficient store width | Step 3.1 |
| `ds_read_b32` (not b128) | Insufficient LDS read width | Step 3.1 |
| `ds_write_b32` (not b128) | Insufficient LDS write width | Step 3.1 |
| `ds_read/write` high Stall | Bank conflict | Step 3.2 |
| `ds_bpermute_b32` | Layout conversion overhead | Step 3.3 |
| `buffer_load/store` + scratch | Register spilling | Step 3.4 |
| `buffer_load` high Stall + low Idle | Memory access not overlapped | Step 3.5 |## Step 3: Iterative Optimization

### ⚠️ Core Principles

**File Editing Strategy (to save context):**
- **Do not** modify the original file directly → revert → re-verify (this wastes a significant amount of context tokens)
- **Must** create a new file for modifications (e.g., original file `kernel.py` → create `kernel_v2.py`), and iterate on the new file
- Each round of optimization iteration operates on a new file; after verification passes, overwrite the original file with the new file's content
- All intermediate files (`_v2`, `_v3`, etc.) are **deleted at once** after final verification passes

**Verification is required after each optimization step:**
1. **Precision Verification** — Use local accuracy validation
2. **Performance Verification** — Use `tools/measure_kernel_time.py` or local benchmark

**Serial Verification:**

After completing the code modification for each optimization point, run verification and wait for the conclusion before proceeding to the next optimization point.

**Why Serial Waiting**: There are dependencies between optimization points (3.2 is based on the result file of 3.1). If 3.1 verification fails, all subsequent modifications based on it become invalid. You must wait for the verification conclusion to confirm before proceeding.

**Why Summarize Verification Output**: The output of verification scripts (validate.py logs, benchmark data) can be large. Keep a concise pass/fail summary with key performance data in the optimization notes.

```
optimization workflow (verification + iteration):

optimization 3.1 -> create kernel_v2.py
  │
  ├─► run validate.py + benchmark
  │
  ├─ pass -> keep v2, optimization 3.2 -> create kernel_v3.py
  │
  └─ fail -> discard v2, revise the file or abandon that optimization
```

**Verification Record:**
- File under test.
- Reference file.
- Verification command.
- Whether precision passes, including max diff when available.
- Performance data, including latency in microseconds.

If a certain optimization step causes precision regression → continue adjusting on the new file without modifying the original.
If a certain optimization step causes performance regression → discard the new file, record the reason, and continue to the next step.

### 3.1 Ensure Coalesced Memory Access + Maximize buffer_load/buffer_store Instruction Width

> For general principles, diagnostic methods, and repair strategies, see [Coalesced Memory Access and Load/Store Instruction Width Optimization](../../common/coalesced-access-load-store-width.md).

**Gluon-Specific Operations**:
- Confirm that the `order` of `BlockedLayout` is consistent with the tensor's actual storage layout in HBM (the dimension with stride=1 = the innermost dimension of order)
- Increase the `size_per_thread` of the contiguous dimension (for bf16, must be ≥ 8 to achieve dwordx4)
- When the original layout in TTGIR is inherently narrow or non-coalesced, re-select the layout, but must ensure functional equivalence
- Verification: `python tools/extract_asm.py <kernel.py> --check-load-width`

### 3.2 Check for Swizzle Bank Conflicts

**Goal**: Eliminate bank conflicts in shared memory (LDS).

**Diagnosis**:
- Check if the hardware counter `SQ_LDS_BANK_CONFLICT` is elevated
- Check if the stall cycles for `ds_read/ds_write` instructions are high
- Use the `gl.bank_conflicts(layout)` API tracer for pre-compilation checks

**Common Causes and Fixes**:

| Cause | Fix |
|------|---------|
| Improper SwizzledSharedLayout parameters | Adjust `vec`, `perPhase`, `maxPhase` parameters |
| Data layout causes simultaneous access to the same bank by multiple threads | Modify swizzle parameters to eliminate conflicts |

**Bank Conflict Check API Provided by Gluon**:
```python
# check layout whether there is bank conflict
conflicts = gl.bank_conflicts(shared_layout)
# returns 0 meansnoneconflict
```

See `../../../../ref-docs/amd/gluon/gfx942/common_optimizations.md` § 3.2

---

### 3.3 Eliminate ds_bpermute Instructions

**Goal**: Eliminate unnecessary `ds_bpermute_b32` instructions.

**Background**: `ds_bpermute` is usually introduced by `gl.convert_layout()`. When the Gluon compiler needs to convert data between different layouts, it performs cross-lane shuffles via LDS's `ds_bpermute`.

**Diagnosis**:
- Check for `ds_bpermute_b32` instructions in the assembly
- If present, trace back to the corresponding Gluon source code, typically `gl.convert_layout()` calls

**Common Causes and Fixes**:

| Cause | Fix |
|------|---------|
| Explicit `gl.convert_layout()` can be avoided | Reorganize the data flow so that upstream directly outputs the target layout |
| Implicit conversion between different layouts | Unify layouts to reduce layout switching |
| convert_layout immediately after load | Load directly in the target layout (adjust BlockedLayout) |

**Note**: Not all `ds_bpermute` can be eliminated. Some layout conversions are required by the algorithm. Prioritize eliminating `ds_bpermute` inside high-frequency loops.

See `../../../../ref-docs/amd/gluon/gfx942/common_optimizations.md` § 3.3

---

### 3.4 Eliminate Scratch Operations (Register Spilling)

> For general principles, VGPR limits, diagnostic methods, and fix strategies, see [Eliminate Scratch Operations (Register Spilling)](../../common/scratch-elimination-vgpr-spill.md), which includes a VGPR budget reference table and num_warps alternatives.

**Gluon-Specific Operations**:
- Check the VGPR usage count and `scratch_size` in the TTGIR compilation output
- Search for `buffer_load` / `buffer_store` pointing to scratch space in the assembly
- Use `tools/extract_asm.py <kernel.py> --check-scratch` for automatic detection

---

### 3.5 Optimize Memory Access Stalls (Compute-Memory Overlap)

**Goal**: Ensure that memory load latency is effectively covered by compute.

**Diagnosis**:
- High Stall column + low Idle column for `buffer_load` instructions
- Load and compute are not interleaved within the loop

**Common Causes and Fixes**:

| Cause | Fix |
|------|---------|
| Software pipelining not implemented | Implement a three-stage pipeline (prologue + main loop + epilogue), see `../../../../converter/amd/cdna3/pipeline.md` |
| Load and compute are serialized | Reorder code so that load for the next iteration overlaps with compute for the current iteration |
| Insufficient prefetch distance | Increase the prefetch lead distance |

See `../../../../ref-docs/amd/gluon/gfx942/common_optimizations.md` § 3.5

---

### 3.6 Pack All Stages with warp_pipeline_stage (Key GEMM Optimization) ⭐

**Goal**: Pack all stages of the GEMM loop (ds_read, MFMA, ds_write) using `warp_pipeline_stage` and hand them off to the compiler, allowing the WarpPipeliner to automatically perform ping-pong scheduling across iterations. Additionally, place `buffer_load` (global loads) between pipeline stages, giving the LLVM scheduler freedom to arrange global memory accesses.

**This is the most important optimization technique for GEMM kernels**, achieving a cumulative +27% performance improvement over three rounds (161→182→193→204 TFLOPS, surpassing Triton's 200 TFLOPS).

**Key Rules**:
1. **All LDS operations and compute must be packed** — ds_read uses `"prep"`, MFMA uses `"compute"`, ds_write uses `"prep"`; none can be omitted
2. **Do not manually write `gl.barrier()`** — let the compiler's Membar pass insert them automatically (manual `gl.barrier()` only generates `s_barrier` without `s_waitcnt`, and it limits the compiler's pipelining optimization space)
3. **Insert ds_write between two groups of MFMA** — place the ds_write's `"prep"` stage between subslice 2's `"compute"` and subslice 3's `"compute"`, rather than combining subslice 2 and 3's MFMA together. This allows the compiler to emit ds_write during MFMA execution gaps, achieving compute↔store overlap (+5.8%, 193→204 TFLOPS)
4. **Place buffer_load between pipeline stages, not inside any stage** — this was the second-round key optimization (+6.3%):
   - `buffer_load` is a global memory access (400+ cycle latency) with different scheduling characteristics from LDS operations
   - Placing it outside stages lets the LLVM standard scheduler handle global memory access, while the WarpPipeliner only manages LDS↔MFMA ping-pong
   - **Split A/B buffer_loads to different positions**: place A's load at the very beginning of the loop (before subslice 0), and B's load after subslice 1. Increasing the distance between the two loads maximizes latency hiding

See `../../../../ref-docs/amd/gluon/gfx942/common_optimizations.md` § 3.6.

**Diagnosis**: If the assembly shows alternating patterns of `sched_barrier(0)` + `s_barrier`, it indicates that WarpPipeliner has taken effect successfully.

See `../../../../ref-docs/amd/gluon/gfx942/warp_pipeline_stage.md` for details.

---

### 3.7 Joint Layout + num_warps Tuning (Critical for Large Tile GEMM) ⭐

**Objective**: By jointly adjusting three dimensions — `mma_layout.warps_per_cta` (determines num_warps), `b_load_layout.threads_per_warp`, and `b_shared_layout.order` — overcome the VGPR bottleneck and unlock pipeline optimization opportunities.

**When this step is needed**:
- §3.6's `warp_pipeline_stage` causes performance regression rather than improvement
- §3.4 reveals scratch spillage that cannot be resolved by reducing block size (block size is fixed by the upper-level algorithm)
- Accumulator occupies VGPRs ≥ 384 (approaching the 512 limit)

**Core insight**: GEMM performance depends not only on the pipeline strategy but also on the combination of three layout parameters. These parameters have strong interaction effects and cannot be optimized individually — they must be jointly searched.

**Search dimensions**:

| Dimension | Candidate Values | Impact |
|------|--------|------|
| `mma warps_per_cta` | [2,2]4w, [2,4]8w, [4,2]8w, [1,4]4w, [1,8]8w, [8,1]8w | Per-warp accumulator size → VGPR pressure → pipeline feasibility |
| `b_load threads_per_warp` | [2,32], [4,16], [8,8] | Global→register load pattern → buffer_load instruction width/efficiency |
| `b_shared order` | [1,0], [0,1] | LDS store layout → bank conflicts → ds_read/ds_write efficiency |

**VGPR Budget Quick Reference** (256×256 tile, MFMA 32×32×8):

| warps_per_cta | num_warps | per-warp acc VGPRs | pipeline headroom |
|---------------|-----------|-------------------|--------------|
| [2,2] / [1,4] / [4,1] | 4 | 512 | **0 (at the limit! Pipeline will necessarily spill)** |
| [2,4] / [4,2] / [1,8] / [8,1] | 8 | 256 | **256 VGPRs (sufficient for pipeline)** |

**Method**: Use `CONFIG = STRATEGY * 1000 + LAYOUT * 10 + WARP_CFG` to encode all variants into a single kernel, dispatch via `constexpr if`, and run a one-shot full ablation test.

**Key rules**:
1. **num_warps is the top priority** — If VGPRs are already full under 4-warp, all pipeline optimizations are ineffective. Try 8-warp first.
2. **The warps_per_cta of the load layout must match the num_warps of the MMA** — 8-warp MMA requires an 8-warp load layout.
3. **b_shared order significantly impacts pipeline patterns** — order=[0,1] (column-major) is usually more pipeline-friendly because K-dimension continuity reduces bank conflicts.
4. **Strong interaction effects** — The same b_shared order may be optimal under a simple strategy but worst under a pipeline strategy, and vice versa. Joint testing is mandatory.
5. **num_warps=8 may significantly accelerate even the simple strategy** — Simply increasing warp count without pipeline optimization can yield a 2× improvement (doubling occupancy).
6. **Some warp_config settings may cause precision issues** — Precision must be independently validated for each combination.

**Passing num_warps at launch**:
```python
kernel[grid](c, a, b, ..., num_stages=1, num_warps=8)  # 8-warp launch
```

See `../../../../ref-docs/amd/gluon/gfx942/warp_pipeline_stage.md` and `pattern_overview.md` (pitfalls 12-15) for details.

---

### 3.8 Attention-Specific Optimizations ⭐

**Objective**: Specialized optimizations for Flash Attention style kernels (involving online softmax + causal masking + dual MFMA). These optimizations complement the GEMM §3.6/§3.7 optimizations and apply to scenarios with complex control flow between MFMA operations.

**When to use**: When the kernel is an Attention-type operator (QK matmul → softmax → PV matmul), rather than a pure GEMM.

**Core optimization checklist** (ordered by priority):

#### 3.8.1 Separating Full Blocks / Masked Blocks (Highest Priority)

Extract the inner loop into a sub-function, using the `gl.constexpr` parameter `DO_MASK` to control whether masking logic is executed:
- **Full blocks** (DO_MASK=False): Completely skip `gl.where` and causal mask computation, allowing the compiler to generate branch-free code
- **Masked blocks** (DO_MASK=True): Only the trailing `BLOCK_M // BLOCK_N + 1` blocks execute masking**Measured Performance**: In non-causal scenarios, went from trailing Triton by 5% to **leading by 8-13%**; in causal scenarios, improved by +7-14%.

```python
if n_full_blocks > 0:
    acc, d_i, m_i = _attn_inner(..., DO_MASK=False)
if masked_blocks > 0:
    acc, d_i, m_i = _attn_inner(..., DO_MASK=True)
```

#### 3.8.2 V Uses convert_layout Instead of smem

The V matrix (non-transposed) directly `buffer_load → gl.convert_layout(v, dot_op1)` enters MFMA. Only K (transposed) goes through smem. Reduces LDS usage and VGPR pressure.

#### 3.8.3 K Uses Immediate smem (value=)

Within each loop iteration, `allocate_shared_memory(value=k)` is allocated immediately, rather than using persistent depth=1. Compatible with the sub-function pattern, the compiler automatically reuses LDS space.

#### 3.8.4 `tl.assume` Compiler Hints

Add `tl.assume(stride > 0)` to all stride parameters. Zero cost; should be added by default.

#### 3.8.5 `1/d_i` Reciprocal Multiplication

Use `d_recip = 1.0 / d_i; acc = acc * d_recip` instead of `acc / d_i`. Triggers the compiler's Newton-Raphson optimization.

#### 3.8.6 `iglp_opt(2)` Scheduling Hints

`gl.amd.cdna3.iglp_opt(2)` is an attention-specific instruction-group-level parallelism hint that helps the hardware interleave TRANS operations and MFMA. Place it before the loop.

**Note**: `warp_pipeline_stage` (§3.6) does not apply to Attention — within the Attention inner loop, there is complex control flow between MFMAs (online softmax), making it impossible to pack into pure prep/compute stages.

#### 3.8.7 SE-level Zigzag Remap (Causal Attention Load Balancing) ⭐

In causal attention, the workload varies drastically across different M-blocks (block 0 performs 2 iterations, block N-1 performs 2N iterations). The MI308X has 4 XCDs × 4 SEs/XCD = 16 SEs, and the hardware assigns consecutive PIDs to different SEs in a round-robin fashion. Without remapping, some SEs receive only heavy blocks while others receive only light blocks, causing severe load imbalance.

**Implementation Key Points**:
1. **1D grid launch**: `grid = (num_m_blocks * num_bh,)` replaces 2D grid
2. **Logical PIDs arranged as start_m-first**: `logical_pid = start_m * num_bh + bh_idx`, grouping different batch-heads of the same start_m (same workload) together
3. **Zigzag remapping**: Reverse odd-numbered waves, ensuring that blocks received by the same SE have complementary workloads
4. **Non-causal keeps bh-first**: `off_bs_head = flat_pid // num_m_blocks`, preserving L2 cache locality

```python
if IS_CAUSAL:
    NUM_SES: gl.constexpr = 16  # MI308X: 4 XCD × 4 SE
    total_blocks = num_m_blocks * num_bh
    wave = flat_pid // NUM_SES
    pos = flat_pid % NUM_SES
    is_odd = wave % 2
    if is_odd:
        logical_pid = wave * NUM_SES + (NUM_SES - 1 - pos)
    else:
        logical_pid = flat_pid
    logical_pid = tl.minimum(logical_pid, total_blocks - 1)
    start_m = logical_pid // num_bh
    off_bs_head = logical_pid % num_bh
else:
    off_bs_head = flat_pid // num_m_blocks
    start_m = flat_pid % num_m_blocks
```

**Why the batch-head dimension must be included**: When zigzag is performed only on the start_m dimension, if M-blocks ≤ 16 SEs (e.g., S=2048 only has 16 blocks), each SE receives only 1 block and cannot pair. By introducing the batch-head dimension, the total number of blocks = num_m × bs × heads, far exceeding 16 SEs, allowing zigzag to fully pair blocks.

**Measured Performance** (bs=4, h=32, dim=64, MI308X):

| Seq Len | No remap | start_m-only remap | start_m × bh remap | Final Improvement |
|---------|---------|-------------------|-------------------|---------|
| S=4096 (32 M-blocks) | 107 TFLOPS | **145 TFLOPS (+35%)** | 144 TFLOPS | +35% |
| S=2048 (16 M-blocks) | 78 TFLOPS | 78 TFLOPS (+0%) | **129 TFLOPS (+65%)** | +65% |
| S=1024 (8 M-blocks) | 70 TFLOPS | 70 TFLOPS (+0%) | **107 TFLOPS (+52%)** | +52% |
| S=512 (4 M-blocks) | 54 TFLOPS | 54 TFLOPS (+0%) | **72 TFLOPS (+34%)** | +34% |For details, see `se_level_zigzag.md` (SE-level Causal Attention Load Balancing Pitfalls 38-40)

---

### 3.9 Small Matrix / Low CU Utilization Optimization ⭐

> **Complete optimization strategy** is detailed in [Small Matrix / Low CU Utilization Optimization](../../common/small-matrix-cu-utilization.md), including reducing tile size to increase grid parallelism (measured 2.5-2.85× improvement), increasing BLOCK_SIZE_K, ISA micro-optimizations, and inapplicable optimization lists.

**Gluon-specific Notes**:
- After modifying BLOCK_SIZE_M/N, **all** layout parameters will change. Use `extract_ttgir.py` to obtain new layouts
- Systematic search method: Create a temporary Triton script with the target tile's `triton.Config`, extract TTGIR, build Gluon kernel, verify accuracy, and benchmark each one
- When modifying BLOCK_SIZE_K, you must simultaneously modify `blocked_a`/`blocked_b` layout and Shared memory allocation size

For details, see `optimization_strategy.md` and `../../../../ref-docs/amd/gluon/gfx942/key_conclusions.md`

---

## Step 4: Re-evaluate

After optimization is complete, return to Step 1 to rebuild the Roofline Model, and recalculate the corresponding utilization based on the bottleneck type (compute bottleneck → compute utilization, bandwidth bottleneck → bandwidth utilization).

---

## Step 5: Output Optimization Results

### When utilization reaches 90% or Checklist is exhausted

**⚠️ All output items below are mandatory. Missing any item = task incomplete.**

Output includes:

1. **Optimized Gluon Code** (file path)

2. **Roofline Analysis Summary**:
   - Bottleneck Type: Compute Bound / Memory Bound
   - Tile AI vs Ridge Point
   - CU Utilization (grid_blocks / num_CUs)
   - Theoretical Performance Upper Bound in TFLOPS
2. **Optimization Checklist Status Table** (copied from §1.6, fill in each item):
   ```
   | # | Optimization Item | Status | Notes |
   |---|------------------|--------|-------|
   | 3.0 | Merge memory access + order | ✅ | order=[1,0] matches row-major |
   | 3.1 | load/store width | ✅ | Confirmed dwordx4 |
   | 3.2 | swizzle bank conflict | ✅ | bank_conflicts=0 |
   | 3.3 | ds_bpermute elimination | ➖ | No convert_layout |
   | ... | ... | ... | ... |
   ```
   **Rule: No ⬚ undone items allowed. Every item must be ✅ or ➖ + reason.**
4. **Performance Comparison Table** (for each test size):
   ```
   | Size | Triton TFLOPS | Gluon TFLOPS (Before) | Gluon TFLOPS (After) | vs Triton |
   |------|--------------|---------------------|---------------------|-----------|
   | 1024 | 52.6 | 44.5 | 51.2 | 97.3% |
   | ... | ... | ... | ... | ... |
   ```

5. **Optimization Summary**:
   - Key optimization measures and their respective performance improvements
   - Unoptimized items and reasons (corresponding to ➖ items in Checklist)
   - Root cause analysis of remaining performance gap (if any)

---

## Validation And Inspection Guidance

| Tool | Purpose | When to Invoke |
|------|---------|----------------|
| `tools/compute_utilization.py` | Roofline bottleneck analysis + compute/bandwidth utilization calculation | Step 1, Step 4 |
| `tools/measure_bandwidth_ceiling.py` | Measure bandwidth upper bound at specified data volume (Gluon memcpy) | Step 1 (Memory Bound) |
| `tools/profile_kernel.sh` | rocprofv3 instruction-level profiling | Step 2 |
| `tools/extract_asm.py` | Extract and analyze assembly code | Step 3 (each sub-step) |
| `tools/measure_kernel_time.py` | Measure kernel elapsed time | After each optimization step |
| local accuracy validation | Accuracy verification | After each optimization step |
| local benchmark | Compare performance with Triton | Final verification |
| TTGIR/layout inspection | Extract TTGIR to analyze layout | When layout modification is needed |

---

## ⚠️ Editing Strategy

Build on the converter guide's source transformation strategy:
- Write each candidate implementation as a coherent file or coherent function-level change
- **Ignore LSP false positives** (type mismatches involving `constexpr` and `gl.*Layout`)
- Only trust `validate.py` and runtime results, not LSP diagnostics

### New File Iteration Strategy (Mandatory)

**Purpose**: Avoid repeatedly editing and reverting the original file.

**Process**:
1. **Create a new file**: Copy the original file (e.g., `kernel.py`) as `kernel_v2.py`, and make all modifications on the new file
2. **Verify the new file**: Run accuracy verification and performance measurement on `kernel_v2.py`
3. **Iterate**: If further modifications are needed, create `kernel_v3.py`, etc., and rewrite the new candidate coherently rather than making scattered line edits
4. **Confirm pass**: Once verification is passed, **do not overwrite the original file**, retain the final version file (e.g., `kernel_v5.py`)
5. **Record final candidate**: keep the final optimized filename in the optimization notes rather than silently overwriting the original file.

**Prohibited**:
- edit on the original file → verification fails → revert → edit again cycle
- Directly overwriting the original file after optimization is complete (user may need comparison or rollback)

---

## Reference Documents

| Document | Content |
|------|------|
| `docs/hardware-specs/hardware_specs_mi308x.md` | AMD GPU Hardware Compute Specification Sheet |
| `../../../../ref-docs/amd/gluon/gfx942/profiling_guide.md` | rocprofv3 Instruction-Level Profiling Detailed Guide |
| `../../../../ref-docs/amd/gluon/gfx942/common_optimizations.md` | General ISA Optimization Checklist (§3.0-§3.5, applicable to all AMD CDNA3 kernels) |
| `../../../../ref-docs/amd/gluon/gfx942/` | Standard GEMM Optimization Topic (WPS pipeline, final configuration template, pitfalls 7-23, 44-59) |
| `../../../../ref-docs/amd/gluon/gfx942/` | Small Matrix GEMM Optimization Topic (Tile size search, pitfalls 32-37, 41-43) |
| `../../../../ref-docs/amd/gluon/gfx942/` | Flash Attention Optimization Topic (SE-level zigzag, anti-patterns, pitfalls 24-31, 38-40) |
| `../../../../ref-docs/amd/gluon/gfx942/isa_patterns.md` | CDNA3 ISA Instruction Patterns and Optimization Reference |
| `../../../../ref-docs/amd/gluon/gfx942/ck_gemm_optimization_reference.md` | CK Optimization Technology Reference |
| `../../../../ref-docs/amd/gluon/gfx942/gluon-amd-gfx942-optimization.md` | Gluon AMD API Feature and Optimization Technology Reference |

### Cross-Reference (converter guide)

| Document | Referenced Content |
|------|---------|
| `../../../../converter/amd/common/porting_rules.md` | Gluon Complete API Reference |
| `../../../../converter/amd/cdna3/layouts.md` | AMD Layout Types and TTGIR Mapping |
| `../../../../converter/amd/cdna3/pipeline.md` | AMD Pipeline Implementation Patterns |
| `../../../../converter/amd/cdna3/memory_access.md` | AMD Memory Access Patterns |
| `../../../../converter/amd/cdna3/common_pitfalls.md` | AMD Common Pitfalls and Solutions |

---

## Version Information

**Guide Version**: v2.3
**Last updated**: 2026-03-18
**Design Principle**: Closed-loop optimization based on compute utilization + instruction-level profile-driven
**v1.1 Update**: Added Step 3.6 warp_pipeline_stage full-stage packing (key GEMM optimization) + Appendix C practical pitfalls experience (including 6 lessons learned)
**v1.2 Update**: §3.6 added buffer_load placement between pipeline stages + A/B load splitting technique (182→193 TFLOPS, +6.3%) + Appendix C pitfall 7
**v1.3 Update**: §3.6 added ds_write inserted between MFMA instructions to achieve compute↔store overlap technique (193→204 TFLOPS, +5.8%, surpasses Triton) + Appendix C pitfall 8 + final best code template
**v1.4 Update**: ⚠️ buffer_load other=0.0 causes WarpPipeliner degradation (134→204.7, +53%) + sched_group_barrier vs warp_pipeline_stage comparison + epilogue simplification + Appendix C pitfalls 9/10/11
**v1.5 Update**: Added §3.7 layout+num_warps joint tuning (key for large Tile GEMM) — 8-warp unlocks VGPR-limit tile pipeline optimization + layout must be jointly adjusted with strategy + systematic ablation methodology + Appendix C pitfalls 12-15 (including 84 combinations measured data for 256×256×64 fp16 GEMM)
**v1.6 Update**: Added Appendix D medium Tile (128×128×64) optimization pitfalls experience 16-23 — MFMA instr_shape selection, depth=2 LDS overload trap, MI308X XCD remapping (4 XCD), tl.assume compiler hint, scalar vs tensor offset stepping, blocked_b K-contiguous vs N-contiguous layout, SUBK=32 vs SUBK=16 applicable scenarios, reference implementation comparison methodology. Synchronously updated gluon-amd-gfx942-optimization.md §4.4 PID reordering as a complete three-step combination solution
**v1.7 Update**: Added §3.8 Attention Specialized Optimization + Appendix E Flash Attention pitfalls experience 24-31 — full/masked block separation (non-causal flips from negative to positive), V goes through convert_layout not smem, K instant smem vs persistent smem, P→smem actually slower (ds_bpermute vs ds_read_u16 trade-off), shared layout order must match dot_op read pattern, `1/d_i` reciprocal multiplication triggers Newton-Raphson, pointer arithmetic replacing offset accumulation. Includes complete measured data for multiple configurations (Gluon comprehensively surpasses Triton)**v1.8 Update**: Added §3.9 Small Matrix/Low CU Utilization Special Optimization + Appendix F Small Matrix GEMM Pitfall Experience 32-37 — CU utilization is the primary bottleneck for small matrices (priority above all ISA optimizations), BLOCK_SIZE_K doubling is the highest ROI optimization (+17%, surpassing Triton by 10-17%), warp_pipeline_stage/num_warps=8/value= is detrimental in low loop count scenarios (-13~-14%), "always do" optimizations (removing other=0.0 + tl.assume) are effective for all sizes. Includes complete measured data for 8 variants (128×256 tile, fp16, MI308X).
**v1.9 Update**: §3.9 Major Update — Tile reduction elevated to highest priority (⭐⭐), priority elevated from §3.9.2 to §3.9.1. Added Appendix F pitfalls 41-43: tile reduction from 128×256→32×32 measured improvement of **2.5-2.85×** (far exceeding BLOCK_K doubling's 1.17×), Gluon faster than Triton by **2.6-2.85×** on all test cases. Includes complete grid/performance comparison data for 5 tile configurations + TTGIR layout extraction comparison table. Key conclusion: tile reduction ≫ BLOCK_K ≫ ISA optimizations ≫ pipeline/warp tuning.
**v1.9 Update**: Added §3.8.7 SE-level Zigzag Remap (Causal Attention Load Balancing) + Appendix G pitfalls 38-40 — Causal workload balancing across MI308X 16 SEs, start_m × batch-head joint zigzag arrangement (significant causal improvement), must include batch-head dimension (start_m-only remap is ineffective when M-blocks ≤ NUM_SES), non-causal must maintain bh-first arrangement to avoid L2 cache hit rate degradation. Includes complete measured data for multiple seq_len × remap strategies.
**v2.0 Update**: Added Appendix H General GEMM (B=KN Layout) Multi-Strategy Parallel Tuning Pitfalls 44-52 — Tile size is the largest lever (128×128 vs 128×256 difference of 37%, far exceeding pipeline tuning's 3%), BLOCK_K=16 restricts warp_pipeline (requires ≥32 for 4 subslice), TTGIR's shared_b order is unsuitable for scenarios without convert_layout transpose pattern (change order=[1,0] for B=KN, +24%), sched_group_barrier outperforms warp_pipeline_stage in small tile scenarios (+2.5%), 8-warp is universally effective for all large tiles (not just VGPR limits), optimization priority order: Tile>BLOCK_K>num_warps>pipeline>schedule>layout. Includes complete measured comparison data for 7 strategies (fp16 4096³ from Triton 130.98→Gluon 175.90 TFLOPS, +34.3%).
**v2.1 Update**: Added Appendix I Pitfall 53 — blocked_b order=[0,1] with B=KN layout causes 3x performance loss (60.9→181.3 TFLOPS). This is the most insidious performance trap: no errors, no accuracy impact, ASM main loop structure appears correct, but buffer_load degrades to 16-bit element-by-element loading (buffer_load_ushort vs buffer_load_dwordx4). order is "step zero", priority updated to: **order correctness > Tile size > BLOCK_K > num_warps > pipeline > schedule > layout**. Also records removing in-loop masks to reduce v_cndmask interference. Includes complete comparison data for 6 matrix sizes (bf16, MI308X, 1024³~8192³, small sizes surpass Triton by 100.5%, large sizes reach 88-95%).
**v2.2 Update**: Added Appendix J 256×256×64 Large Tile GEMM Optimization Pitfalls 54-59 — The decisive impact of `instr_shape=[16,16,16]` vs `[32,32,8]` on large tiles (+22.6%, 155→190 TFLOPS), `in_thread_transpose` completely infeasible at BLOCK_K≤32 (-68%, ds_bpermute overhead far exceeds benefit), `in_thread_transpose` also inferior to not doing it even on 256×256 tile (+22.6%), WPS subslice count must be minimized (4 sub >> 8 sub), multi-kernel mixed benchmarks are severely distorted (requires isolated process testing), tile selection strategy (≤1024 use 128×128, ≥2048 use 256×256). Includes complete measured data for fp16 GEMM 5 sizes × 2 tiles (MI308X, Gluon 256×256 achieves 240 TFLOPS at 8192³ = 103% peak). Optimization priority updated to: **order > instr_shape > Tile size > BLOCK_K > num_warps > pipeline > in_thread_transpose (usually skip) > schedule > layout**.
**v2.3 Update**: Added acceptance checklist guidance that prevents the "convert only, don't optimize" anti-pattern. Clearly defines completion criteria (≥85% Triton or checklist exhausted), mandatory artifacts (Roofline analysis, completed checklist status table, performance comparison table), acceptance protocol. §1.6 Stop Conditions adds 4 explicit "unsatisfactory stop condition" counterexamples. Step 5 output requirements changed from advisory to mandatory (missing items = incomplete).


## Related

- [Changelog for Preview 0.1.4](CHANGELOG.md)
- [AMD MI308X (gfx942) GEMM Optimization Techniques Reference](ck_gemm_optimization_reference.md)
- [ISA Optimization Detailed Checklist](common_optimizations.md)
- [Stopping Conditions](final_config_template.md)
- [Gluon AMD gfx942 (CDNA3 / MI300) API & Performance Optimization Guide](gluon-amd-gfx942-optimization.md)
- [Gluon Kernel Performance Optimization Guide (AMD CDNA4)](../gfx950/optimization-guide.md)
- [Gluon Kernel Performance Optimization Guide (NVIDIA Hopper)](../../../nvidia/hopper/gluon/optimization-guide.md)
- [Triton Embraces Tile IR: Beyond SIMT](../../../nvidia/common/triton/triton-tile-ir-beyond-simt.md)
- [Composable Kernel (CK) Architecture Overview](../../common/ck-architecture-overview.md)
- [Gluon Tutorial 07: Persistent Kernels and Pipeline Optimization](../../../nvidia/common/gluon/gluon-07-persistent-kernel-pipeline.md)
- [Document Relationship Diagram](../../../RELATIONS.md)
