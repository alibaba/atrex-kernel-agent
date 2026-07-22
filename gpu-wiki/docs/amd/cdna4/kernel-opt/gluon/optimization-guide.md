# Gluon Kernel Performance Optimization Guide (AMD CDNA4)

> **This guide is specifically for the AMD CDNA4 (MI355X, gfx950) target.**
> For CDNA3 (MI300 series), please use the related guide: `docs/amd/cdna3/mi308x/kernel-opt/gluon/optimization-guide.md`
> For NVIDIA GPUs, please use the related guide: `docs/nvidia/hopper/kernel-opt/gluon/optimization-guide.md`

## Applicability

This guide covers:
- "Optimize the performance of this Gluon kernel"
- "This kernel is too slow, help me analyze it"
- "Help me improve the compute utilization of this GPU kernel"
- "Analyze where the bottleneck of this kernel is"
- "Profile this Gluon operator"

## Prerequisites

This guide assumes the input code is already a **correctly compilable and runnable Gluon kernel (CDNA4 target)**. If conversion from Triton to Gluon is needed, first use the CDNA4 conversion guidance in `docs/amd/cdna4/conversion-guide.md`.

This guide may directly reference the following local wiki content to avoid duplication:
- **CDNA4 API Reference**: `docs/amd/cdna4/api_mapping.md`
- **Layout Types**: `docs/amd/cdna4/layouts.md`
- **Memory Access Patterns**: `docs/amd/cdna4/memory_access.md`
- **Matrix Multiplication Patterns**: `docs/amd/cdna4/matrix_multiply.md`
- **Pipeline Patterns**: `docs/amd/cdna4/pipeline.md`
- **Precision verification**: run the local precision check used by the consuming harness
- **Performance benchmarking**: run the local benchmark used by the consuming harness
- **TTGIR/layout inspection**: inspect generated IR or layout metadata when layout changes are involved

---

## Core Optimization Workflow

```
┌──────────────────────────────────────────────────────────────────────┐
│                   Gluon Kernel Optimization Workflow (CDNA4)        │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌────────────────────────────┐                                      │
│  │ Step 0: Compute Pattern Recognition                           │   │
│  │  → Route to topic documents   │                                      │
│  └──────────┬─────────────────┘                                      │
│             │                                                        │
│  ┌──────────▼─────────────────┐                                      │
│  │ Step 1: Bottleneck Analysis and Utilization Evaluation       │   │◄────────────────────────────┐       │
│  │   1.1 Roofline Model       │                            │       │
│  │   1.4 CU Utilization Pre-check  │                             │       │
│  │   1.5 Theoretical Performance Upper Bound           │                             │       │
│  └──────────┬─────────────────┘                            │       │
│             │                                              │       │
│      ┌──────▼──────┐                                       │       │
│      │ Utilization ≥ 90% │──YES──► Output optimization summary, done      │       │
│      │ or no optimization space? │                                       │       │
│      └──────┬──────┘                                       │       │
│             │NO                                            │       │
│             ▼                                              │       │
│  ┌──────────────────────┐                                  │       │
│  │ Step 2: Instruction-Level Profile │                                 │       │
│  │  (rocprofv3 analysis)     │                                  │       │
│  └──────────┬───────────┘                                  │       │
│             │                                              │       │
│             ▼                                              │       │
│  ┌──────────────────────────────┐                          │       │
│  │ Step 3: Iterative Optimization              │                          │       │
│  │  Execute per topic/common checklist         │                           │       │
│  └──────────┬───────────────────┘                          │       │
│             │                                              │       │
│             └──────────────────────────────────────────────┘       │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

## Step 0: Compute Pattern Recognition

Read the kernel code and identify the primary compute pattern. Once matched, go to the corresponding topic document to obtain optimization strategies and common pitfalls for that pattern.

### Recognition Rules (by priority, larger patterns first)

```
Read kernel code
  │
  ├─ Has Q×K^T → softmax → ×V three stages?
  │   └─ YES → Fused Attention (fused_attention.md)
  │
  ├─ Has mfma/mfma_scaled? Main loop iterates along K dimension without cross-iteration dependencies?
  │   └─ YES → Standard GEMM (docs/amd/cdna4/ref-docs/gluon/matmul.md)
  │
  ├─ No mfma? Pure reduction / element-wise?
  │   └─ YES → Softmax/Reduce (docs/amd/cdna4/ref-docs/gluon/softmax_reduce.md)
  │
  └─ None match
      └─ Fall back to common optimization (docs/amd/cdna4/ref-docs/gluon/common_optimizations.md)
```

| Pattern | Recognition Characteristics | Topic Document |
|------|---------|---------|
| Fused Attention | Q×K^T → softmax → ×V; causal mask; online softmax | `fused_attention.md` |
| Standard GEMM | Has mfma/mfma_scaled; K-dimension iteration without cross-iteration dependencies | `docs/amd/cdna4/ref-docs/gluon/matmul.md` |
| Softmax / Reduce / Element-wise | No mfma; pure reduction or element-wise | `docs/amd/cdna4/ref-docs/gluon/softmax_reduce.md` |
| No Match | None of the above match | `common_optimizations.md` |

---

## Step 1: Bottleneck Analysis and Utilization Evaluation (Roofline Model)

Before optimizing, **you must first determine whether the operator is compute-bound or memory-bound**, as this determines the subsequent optimization direction and utilization evaluation metrics.

### 1.1 Building the Roofline Model — Determining the Bottleneck Type

**Key Principle**: Roofline analysis should be performed at the **tile (block)** level, not the global FLOPs/Bytes of the entire kernel.

#### Computing Tile-Level Arithmetic Intensity (AI)

```
Tile AI = Tile_FLOPs / Tile_Bytes (unit: FLOPs/Byte)
```

| Item | Calculation Method | Description |
|------|---------|------|
| **Tile_FLOPs** | Floating-point operations for a single tile | GEMM tile: `2×BM×BN×K`; element-wise tile: `BM×BN` per op |
| **Tile_Bytes** | Total bytes read/written from/to HBM for a single tile | Input tile bytes + Output tile bytes (considering data type width) |

#### Roofline Ridge Point

```
Ridge Point = Peak Compute (FLOPS) / Peak Bandwidth (Bytes/s)   (unit: FLOPs/Byte)
```

| GPU | Precision | Peak Compute (TFLOPS) | Peak Bandwidth (TB/s) | Ridge Point (FLOPs/Byte) |
|-----|------|-------------------|-----------------|--------------------------|
| MI355X | FP16/BF16 dense Matrix | 2,500 | 8.0 | **312.5** |
| MI355X | FP8 dense Matrix | 5,000 | 8.0 | **625** |
| MI355X | FP32 | 157.3 | 8.0 | **19.7** |

> These are the dense peaks and bandwidth published on AMD's official MI355X
> product page. Structured-sparsity peaks are separate ceilings and must not be
> used to classify a dense kernel. See
> `docs/amd/cdna4/hardware-specs/hardware_specs_mi355x.md`.

#### Determining the Bottleneck

```
If Tile AI ≥ Ridge Point:
    → Compute Bound
    → Evaluation metric: Compute utilization
    → Optimization focus: Improve MFMA instruction throughput, eliminate stalls, increase compute-memory overlap

If Tile AI < Ridge Point:
    → Memory Bound
    → Evaluation metric: Bandwidth utilization
    → Optimization focus: Reduce memory access, increase cache hit rate, increase data reuse, optimize load/store width
```

### 1.2 Measuring Actual Performance of the Current Case

```bash
python <benchmark-command> <kernel.py> <ref.py> \
    --wrapper-name <wrapper> --setup-name <setup>
```

```
actual (TFLOPS) = FLOPs / elapsed time (seconds) / 1e12
actualbandwidth (TB/s) = Bytes_transferred / elapsed time (seconds) / 1e12
```

### 1.3 Evaluating Utilization

**Select the corresponding utilization metric based on the bottleneck type determined in 1.1**:

- **Compute Bound** → `Compute Utilization (%) = Actual Compute / Peak Compute × 100%`
- **Memory Bound** → `Bandwidth Utilization (%) = Actual Bandwidth / Bandwidth Upper Bound × 100%`

**Choice of bandwidth upper bound**: Kernels with small data volumes cannot achieve peak theoretical bandwidth. The denominator should use the **measured bandwidth upper bound for the same data volume** (measured with a memcpy kernel).

### 1.4 CU Utilization Pre-check (Must-check for Small Matrices)

```
grid_blocks = cdiv(M, BLOCK_SIZE_M) × cdiv(N, BLOCK_SIZE_N)
CU_ratio = grid_blocks / num_CUs   (MI355X: 256 CUs)
```

| CU_ratio | Judgment | Action |
|----------|------|------|
| < 10% | Tile too large | Reduce tile size or go to §3.6 Small Matrix Special Case |
| 10%-100% | Somewhat low | Check whether tile size can be reduced to improve utilization |
| ≥ 100% | Sufficient | Proceed normally to Step 2 |

### 1.5 Theoretical Performance Upper Bound Evaluation

```
tile_time_min = Tile_FLOPs/peak (Compute) or Tile_Bytes/bandwidthupper bound (Memory)
num_waves = ceil(grid_blocks / num_CUs)
theoretical_time = num_waves × tile_time_min
```

> When the theoretical performance upper bound is far below the hardware peak, the issue lies at the **operator configuration level**, and algorithm-level optimization should be prioritized.

### 1.6 Stop Conditions

- **Utilization ≥ 90%** → Stop optimization, output results
- **Checklist exhausted** → Reference the corresponding checklist based on the hit pattern

| Hit Pattern | Applicable Checklist |
|------------|----------------------|
| GEMM | Full 3.0-3.6 of `common_optimizations.md` |
| Attention | Checklist of `fused_attention.md` |
| Others | 3.0-3.6 of `common_optimizations.md` |

**⚠️ Mandatory Checklist Audit When Utilization Does Not Reach 90%**

When utilization < 90%, it is **forbidden** to stop on the grounds of "no optimization headroom". Each checklist item must be audited item by item, and the status must be filled in before stopping:

| # | Optimization Item | Applicable Condition | Status (Required) |
|---|-------------------|---------------------|-------------------|
| 3.0 | Coalesced memory access + order correctness | All kernels | ✅Done / ⬚Not Done / ➖N/A (reason) |
| 3.1 | load/store width dwordx4 | All kernels | ✅ / ⬚ / ➖ |
| 3.2 | Swizzle bank conflict | Kernels using shared memory | ✅ / ⬚ / ➖ |
| 3.3 | ds_bpermute elimination | Kernels with convert_layout | ✅ / ⬚ / ➖ |
| 3.4 | Scratch elimination | All GEMM/Attention | ✅ / ⬚ / ➖ |
| 3.5 | Memory stall / software pipeline | Kernels with num_stages > 1 | ✅ / ⬚ / ➖ |
| 3.6 | warp_pipeline_stage | GEMM kernels | ✅ / ⬚ / ➖ |
| 3.7 | Layout + num_warps joint tuning | Large tile GEMM (BM≥128) | ✅ / ⬚ / ➖ |
| 3.8 | Attention-specific optimization | Attention kernels | ✅ / ⬚ / ➖ |
| 3.9 | Small matrix / CU utilization | grid_blocks < num_CUs | ✅ / ⬚ / ➖ |
| XCD | XCD/PID remapping | MI355X (8 XCD) | ✅ / ⬚ / ➖ |

**Rules:**
1. All **⬚Not Done** items must be attempted first, then return to this checklist
2. All **➖N/A** items must include a reason
3. Only when all applicable items are ✅ or ➖ can "checklist exhausted" be determined and stopped
4. **Strictly forbidden** to skip items with ⬚Not Done on the grounds of "estimated marginal benefit"—must verify through actual measurement

### Tool Invocation

```bash
python tools/compute_utilization.py \
    --gpu mi355x --dtype bf16 \
    --flops-expr "2*BM*BN*K" --bytes-expr "(BM*K + BN*K + BM*BN)*2" \
    --time-ms 0.5 --grid-blocks 64
```

---

## Step 2: Instruction-Level Profile Analysis

> For detailed profile interpretation guide, see `docs/amd/cdna4/ref-docs/gluon/profiling_guide.md`.

### Key Steps

0. **Locate the launch index of the target kernel (required)**
   ```bash
   rocprofv3 --stats python <kernel.py>
   ```

1. **Collect instruction-level trace**
   ```bash
   env LD_LIBRARY_PATH=/opt/rocm/lib64:/opt/rocm/lib:$LD_LIBRARY_PATH \
       rocprofv3 --att \
       --att-library-path ./tools/rocprof-trace-decoder/releases/linux_glibc_2_28_x86_64 \
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
 sort -t',' -k5 -nr ./profile_output/stats_*.csv | head -20 # by Latency
 sort -t',' -k6 -nr ./profile_output/stats_*.csv | head -20 # by Stall
   ```

4. **Clear trace files** (must be executed after analysis)
   ```bash
   rm -rf tt_test
   ```

### SASS/ISA Quick Reference

| ISA Instruction Pattern | Possible Issue | Corresponding Optimization |
|------------------------|----------------|----------------------------|
| `buffer_load_dword` (non-dwordx4) | Insufficient load width | §3.1 |
| `ds_read_b32` (non-b128) | Insufficient LDS read width | §3.1 |
| `ds_read/write` high Stall | Bank conflict | §3.2 |
| `ds_bpermute_b32` | Layout conversion overhead | §3.3 |
| `buffer_load/store` + scratch | Register spill | §3.4 |
| `buffer_load` high Stall + low Idle | Memory access not overlapped | §3.5 |

## Step 3: Iterative Optimization

Based on the pattern matched in Step 0:

1. **Pattern matched** → Execute optimization strategies from the specialized documentation
   - Encounter general ISA issues (coalesced access, load width, etc.) → Refer to `docs/amd/cdna4/ref-docs/gluon/common_optimizations.md`
   - Encounter pitfalls → Refer to `docs/amd/cdna4/ref-docs/gluon/pitfalls.md` or inline pitfall experience from the specialization

2. **No match** → Execute in order of 3.0 → 3.1 → ... → 3.6 in `docs/amd/cdna4/ref-docs/gluon/common_optimizations.md`

> For detailed optimization checklist, see `docs/amd/cdna4/ref-docs/gluon/common_optimizations.md`.

---

## Step 4: Re-evaluation

After optimization is complete, return to Step 1 to rebuild the Roofline Modelestic and recalculate the corresponding utilization ratios based on the bottleneck type.

---

## Step 5: Output Optimization Results

### When utilization reaches 90% or no further optimization is possible

Output includes:
1. **Optimized Gluon code**
2. **Optimization summary report**:
   - Original utilization vs. optimized utilization
   - Optimization content and effects for each step
   - Latency comparison (before optimization vs. after optimization)
   - Unoptimized items and reasons
3. **Verification results**:
   - Precision verification pass confirmation
   - Performance data

---

## Validation And Inspection Guidance

| Tool | Purpose | When to Use |
|------|------|----------|
| `tools/compute_utilization.py` | Roofline bottleneck analysis + compute/bandwidth utilization calculation | Step 1, Step 4 |
| `tools/measure_bandwidth_ceiling.py` | Measure bandwidth upper bound at specified data size | Step 1 (Memory Bound) |
| `tools/profile_kernel.sh` | rocprofv3 instruction-level profile | Step 2 |
| `tools/extract_asm.py` | Extract and analyze assembly code | Step 3 (each sub-step) |
| `tools/measure_kernel_time.py` | Measure kernel latency | After each optimization step |
| local accuracy validation | Precision verification | After each optimization step |
| local benchmark | Compare performance with Triton | Final verification |
| TTGIR/layout inspection | Extract TTGIR to analyze layout | When layout modification is needed |

---

## ⚠️ Editing Strategy

Build on the docs/ guide's source transformation strategy:
- Write each candidate implementation as a coherent file or coherent function-level change
- **Ignore LSP false positives** (type mismatches involving `constexpr`, `gl.*Layout`)
- Trust only `validate.py` and runtime results, do not trust LSP diagnostics

### New File Iteration Strategy (Mandatory)

**Purpose**: Avoid repeatedly editing and reverting the original file.

**Process**:
1. **Create a new file**: Copy the original file (e.g., `kernel.py`) as `kernel_v2.py`, and make all modifications on the new file
2. **Verify the new file**: Run precision verification and performance measurement on `kernel_v2.py`
3. **Iterate**: If further modifications are needed, create `kernel_v3.py` and similar files, and rewrite the new candidate coherently rather than making scattered line edits
4. **Confirmation passed**: After verification passes, **do not overwrite the original file**, retain the final version file (e.g., `kernel_v5.py`)
5. **Record final candidate**: keep the final optimized filename in the optimization notes rather than silently overwriting the original file.

**Prohibited**:
- The cycle of edit on the original file → verification failure → revert → edit again
- Directly overwriting the original file after optimization is complete (the user may need to compare or roll back)

---

## Reference Documentation

| Document | Content |
|------|------|
| `docs/amd/cdna4/hardware-specs/hardware_specs_mi355x.md` | AMD CDNA4 GPU hardware compute specification table |
| `docs/amd/cdna4/ref-docs/gluon/profiling_guide.md` | rocprofv3 instruction-level profile details |
| `docs/amd/cdna4/ref-docs/gluon/common_optimizations.md` | General ISA optimization checklist (§3.0-3.6) + quick reference by bottleneck/type |
| `docs/amd/cdna4/ref-docs/gluon/pitfalls.md` | Practical pitfall experience index (by pattern tag) |
| `docs/amd/cdna4/ref-docs/gluon/matmul.md` | Standard GEMM optimization special topic |
| `docs/amd/cdna4/ref-docs/gluon/softmax_reduce.md` | Softmax / Reduction / Element-wise optimization special topic |
| `fused_attention.md` | Fused Attention optimization special topic |

### Cross-references (cdna4-docs/ guide)

| Document | Referenced Content |
|------|---------|
| `docs/amd/cdna4/api_mapping.md` | CDNA4 Gluon API reference |
| `docs/amd/cdna4/layouts.md` | CDNA4 Layout types and TTGIR mapping |
| `docs/amd/cdna4/pipeline.md` | async_copy pipeline implementation patterns |
| `docs/amd/cdna4/memory_access.md` | CDNA4 memory access patterns |
| `docs/amd/cdna4/matrix_multiply.md` | MFMA matrix multiplication patterns |
| `docs/amd/cdna4/common_pitfalls.md` | CDNA4 common errors and solutions |

## Version Information

**Guide Version**: v1.0
**Last Updated**: 2026-03-28
**Target Architecture**: AMD CDNA4 (MI355X, gfx950)
**Design Principles**: Pattern Recognition → Topic Routing + Compute Utilization Closed-Loop Optimization + Instruction-Level Profile-Driven
**v1.0**: Refactored to pattern recognition architecture. Added Step 0 compute pattern recognition, split §3.0-3.9 into `common_optimizations.md`, split pitfall experiences into `pitfalls.md`, added 3 pattern-specific topic documents (matmul, softmax_reduce, fused_attention)
