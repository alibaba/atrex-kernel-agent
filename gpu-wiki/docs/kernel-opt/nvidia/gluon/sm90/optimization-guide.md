# Gluon Kernel Performance Optimization Guide (NVIDIA Hopper)

> **This guide is specifically for NVIDIA Hopper (sm_90, H20/H100/H200) targets.**
> For AMD GPUs, please use the related guide: `kernel-opt/amd/gluon/`

## Applicability

This guide covers:
- "Optimize the performance of this Gluon kernel"
- "This kernel is too slow, help me analyze it"
- "Help me improve the compute utilization of this GPU kernel"
- "Analyze where the bottleneck is in this kernel"
- "Profile this Gluon operator"

## Prerequisites

This guide assumes the input code is already a **correctly compilable and runnable Gluon kernel (Hopper target)**. If conversion from Triton to Gluon is needed, first use the Hopper conversion guidance in `../../../../converter/nvidia/hopper/conversion-guide.md`.

This guide can directly reference the following local wiki content to avoid duplication:
- **Hopper API Reference**: `../../../../converter/nvidia/hopper/api_mapping.md`
- **Layout Types**: `../../../../converter/nvidia/hopper/layouts.md`
- **Memory Access Patterns**: `../../../../converter/nvidia/hopper/memory_access.md`
- **Matrix Multiplication Patterns**: `../../../../converter/nvidia/hopper/matrix_multiply.md`
- **Pipeline Patterns**: `../../../../converter/nvidia/hopper/pipeline.md`
- **Precision verification**: run the local precision check used by the consuming harness
- **Performance benchmarking**: run the local benchmark used by the consuming harness
- **TTGIR/layout inspection**: inspect generated IR or layout metadata when layout changes are involved

---

## Core Optimization Workflow

```
┌──────────────────────────────────────────────────────────────────────┐
│                   Gluon Kernel Optimization Workflow (Hopper)        │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌────────────────────────────┐                                      │
│  │ Step 0: Compute Pattern Recognition                           │   │
│  │  → Route to topic documents   │                                      │
│  └──────────┬─────────────────┘                                      │
│             │                                                        │
│  ┌──────────▼─────────────────┐                                      │
│  │ Step 1: Bottleneck Analysis and Utilization Assessment       │   │◄──────────────────────-───┐         │
│  │   1.1 Roofline Model       │                            │         │
│  │   1.4 SM Utilization Pre-check  │                             │         │
│  │   1.5 Theoretical Performance Upper Bound           │                             │         │
│  └──────────┬─────────────────┘                            │         │
│             │                                              │         │
│      ┌──────▼──────┐                                       │         │
│      │ Utilization ≥ 90% │──YES──► Output optimization summary, done      │         │
│      │ or no optimization space? │                                       │         │
│      └──────┬──────┘                                       │         │
│             │NO                                            │         │
│             ▼                                              │         │
│  ┌──────────────────────┐                                  │         │
│  │ Step 2: Instruction-Level Profile │                                 │         │
│  │  (ncu / Nsight analysis)  │                                  │         │
│  └──────────┬───────────┘                                  │         │
│             │                                              │         │
│             ▼                                              │         │
│  ┌──────────────────────────────┐                          │         │
│  │ Step 3: Iterative Optimization              │                          │         │
│  │  Execute per topic/common checklist         │                           │         │
│  └──────────┬───────────────────┘                          │         │
│             │                                              │         │
│             └──────────────────────────────────────────────┘         │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

## Step 0: Compute Pattern Recognition

Read the kernel code and identify the primary compute pattern. Once identified, refer to the corresponding specialized documentation for optimization strategies and common pitfalls.

### Recognition Rules (by priority, larger patterns first)

```
Read kernel code
  │
  ├─ Has Q×K^T → softmax → ×V three stages?
  │   └─ YES → Fused Attention (fused_attention.md)
  │
  ├─ Main loop has h[i+1]=f(h[i]) serial dependency? chunk_size split time dimension?
  │   └─ YES → Recurrent / Linear Attention
  │            (docs/ref-docs/nvidia/gluon/sm90/linear_attention.md)
  │
  ├─ Has wgmma? Main loop iterates along K dimension without cross-iteration dependency?
  │   └─ YES → Standard GEMM (docs/ref-docs/nvidia/gluon/sm90/matmul.md)
  │
  ├─ No wgmma? Pure reduction / element-wise?
  │   └─ YES → Softmax/Reduce (softmax_reduce.md)
  │
  └─ None match
      └─ Fall back to common optimization (docs/ref-docs/nvidia/gluon/sm90/common_optimizations.md)
```

| Pattern | Recognition Features | Specialized Docs |
|------|---------|---------|
| Fused Attention | Q×K^T → softmax → ×V; causal mask; paged KV cache | `fused_attention.md` |
| Chunk Linear Attention / Recurrent | h[i+1]=f(h[i]); chunk_size split; iterative state matrix update | `docs/ref-docs/nvidia/gluon/sm90/linear_attention.md` |
| Standard GEMM | Contains wgmma; K-dimension iteration with no cross-iteration dependency | `docs/ref-docs/nvidia/gluon/sm90/matmul.md` |
| Softmax / Reduce / Element-wise | No wgmma; pure reduction or element-wise | `softmax_reduce.md` |
| No Match | None of the above match | `common_optimizations.md` |

---

## Step 1: Bottleneck Analysis and Utilization Assessment (Roofline Model)

Before optimizing, **you must first determine whether the operator is compute-bound or memory-bound**, as this determines the subsequent optimization direction and utilization evaluation metrics.

### 1.1 Building a Roofline Model — Determine the Bottleneck Type

**Key Principle**: Roofline analysis should be performed at the **tile (block)** level, not based on the global FLOPs/Bytes of the entire kernel.

#### Calculate Tile-Level Arithmetic Intensity (AI)

```
Tile AI = Tile_FLOPs / Tile_Bytes (unit: FLOPs/Byte)
```

| Item | Calculation Method | Description |
|------|---------|------|
| **Tile_FLOPs** | Total floating-point operations per tile | GEMM tile: `2×BM×BN×K`; element-wise tile: `BM×BN` per op |
| **Tile_Bytes** | Total bytes read/written from HBM per tile | Input tile bytes + output tile bytes (considering data type width) |

#### Roofline Ridge Point

```
Ridge Point = Peak Compute (FLOPS) / Peak Bandwidth (Bytes/s)   (unit: FLOPs/Byte)
```

| GPU | Precision | Peak Compute (TFLOPS) | Peak Bandwidth (TB/s) | Ridge Point (FLOPs/Byte) |
|-----|------|-------------------|-----------------|--------------------------|
| H100 SXM | FP16/BF16 | 989.4 | 3.35 | **295** |
| H100 SXM | FP8 | 1,978.9 | 3.35 | **591** |
| H100 SXM | FP32 | 67.0 | 3.35 | **20.0** |
| H20 | FP16/BF16 | 148.0 | 4.0 | **37** |
| H20 | FP8 | 296.0 | 4.0 | **74** |
| H200 | FP16/BF16 | 989.4 | 4.8 | **206** |
| H200 | FP8 | 1,978.9 | 4.8 | **412** |

> For detailed specifications, see `docs/hardware-specs/hardware_specs_hopper.md`.

#### Determine Bottleneck

```
If Tile AI ≥ Ridge Point:
    → Compute Bound
    → Evaluation metric: Compute utilization
    → Optimization focus: Improve wgmma instruction throughput, eliminate stalls, increase compute-memory overlap

If Tile AI < Ridge Point:
    → Memory Bound
    → Evaluation metric: Bandwidth utilization
    → Optimization focus: Reduce memory access, increase cache hit rate, increase data reuse, optimize load/store width
```

### 1.2 Measure Actual Performance of the Current Case

```bash
python <benchmark-command> <kernel.py> <ref.py> \
    --wrapper-name <wrapper> --setup-name <setup>
```

```
actual (TFLOPS) = FLOPs / elapsed time (seconds) / 1e12
actualbandwidth (TB/s) = Bytes_transferred / elapsed time (seconds) / 1e12
```

### 1.3 Evaluate Utilization

**Based on the bottleneck type determined in 1.1, select the corresponding utilization metric**:

- **Compute Bound** → `Compute Utilization (%) = Actual Compute / Peak Compute × 100%`
- **Memory Bound** → `Bandwidth Utilization (%) = Actual Bandwidth / Bandwidth Upper Bound × 100%`**Bandwidth Ceiling Selection**: Small-data kernels cannot reach peak theoretical bandwidth. Use the **measured bandwidth ceiling for the same data size** (measured with a memcpy kernel) as the denominator.

### 1.4 SM Utilization Pre-Check (Mandatory for Small Matrices)

```
grid_blocks = cdiv(M, BLOCK_SIZE_M) × cdiv(N, BLOCK_SIZE_N)
SM_ratio = grid_blocks / num_SMs   (H20: 78, H100: 132, H200: 132)
```

| SM_ratio | Judgment | Action |
|----------|------|------|
| < 10% | Tile is too large | Reduce tile size or proceed to §3.6 Small-Matrix Specialization |
| 10%-100% | On the low side | Check if tile size can be reduced to improve utilization |
| ≥ 100% | Sufficient | Proceed normally to Step 2 |

### 1.5 Theoretical Performance Ceiling Evaluation

```
tile_time_min = Tile_FLOPs/peak (Compute) or Tile_Bytes/bandwidthupper bound (Memory)
num_waves = ceil(grid_blocks / num_SMs)
theoretical_time = num_waves × tile_time_min
```

> When the theoretical performance ceiling is far below the hardware peak, the issue lies at the **operator configuration level**, and algorithm-level optimization should be prioritized.

### 1.8 Stopping Conditions

- **Utilization ≥ 90%** → Stop optimization and output results
- **Latency-Bound and within 1.3× of theoretical lower bound** → Stop ISA-level optimization (see `docs/ref-docs/nvidia/gluon/sm90/linear_attention.md`)
- **Checklist exhausted** → Refer to the corresponding checklist by hit pattern

| Hit Pattern | Applicable Checklist |
|---------|---------------|
| Recurrent | Stopping conditions of `docs/ref-docs/nvidia/gluon/sm90/linear_attention.md` |
| GEMM | Full 3.0-3.6 of `common_optimizations.md` |
| Attention | Checklist of `fused_attention.md` |
| Other | 3.0-3.6 of `common_optimizations.md` |

**⚠️ Mandatory Checklist Audit When Utilization < 90%**

When utilization < 90%, it is **forbidden** to stop optimization claiming "no optimization opportunities exist." Each item must be checked against the checklist and the status filled in before stopping:

| # | Optimization Item | Applicable Condition | Status (Required) |
|---|--------|---------|------------|
| 3.0 | Coalesced memory access + order correctness | All kernels | ✅Done / ⬚Not Done / ➖N/A (Reason) |
| 3.1 | load/store width LDG.128 | All kernels | ✅ / ⬚ / ➖ |
| 3.2 | Shared memory bank conflicts | Kernels using shared memory | ✅ / ⬚ / ➖ |
| 3.3 | Scratch/local memory elimination | All GEMM/Attention | ✅ / ⬚ / ➖ |
| 3.4 | async_copy pipeline optimization | Kernels with num_stages > 1 | ✅ / ⬚ / ➖ |
| 3.5 | wgmma fence/wait correctness | Kernels using wgmma | ✅ / ⬚ / ➖ |
| 3.6 | SM utilization / Tile size tuning | grid_blocks < num_SMs | ✅ / ⬚ / ➖ |

**Rules:**
1. All **⬚Not Done** items must be attempted before returning to this checklist
2. All **➖N/A** items must include a reason
3. Only when all applicable items are either ✅ or ➖ can "checklist exhausted" be declared and optimization stopped
4. **Strictly forbidden** to skip any ⬚ item on the grounds of "expected minimal gains" — actual measurement and verification are mandatory

### Tool Invocation

```bash
python tools/compute_utilization.py \
    --gpu h20 --dtype bf16 \
    --flops-expr "2*BM*BN*K" --bytes-expr "(BM*K + BN*K + BM*BN)*2" \
    --time-ms 0.5 --grid-blocks 64
```

---

## Step 2: Instruction-Level Profile Analysis

> For detailed profile interpretation guide, see `docs/ref-docs/nvidia/gluon/sm90/profiling_guide.md`.

### Key Steps

0. **Locate the launch index of the target kernel (Required)**
   ```bash
   ncu --print-summary per-kernel python <kernel.py>
   ```

1. **Collect profile data**
   ```bash
   ncu --set full --launch-skip <N> --launch-count 1 -o ./profile_output python <kernel.py>
   ```

2. **Inspect key metrics**: `sm__throughput`, `dram__throughput`, `smsp__warp_issue_stalled_*`

3. **SASS instruction analysis**: Check `LDG.E.128` vs `LDG.E.32`, `STL`/`LDL`, `LDGSTS`

### SASS Quick Reference Table

| SASS Instruction Pattern | Potential Issue | Corresponding Optimization |
|-------------|---------|---------|
| `LDG.E.32` (not 128) | Insufficient load width | §3.1 |
| `STL` / `LDL` | Register spilling | §3.3 |
| Missing `LDGSTS` | async_copy not used | §3.4 |
| Too many `BAR.SYNC` | Excessive synchronization | §3.4 |

---

## Step 3: Iterative Optimization

Based on the pattern hit in Step 0:

1. **Pattern matched** → Follow the optimization strategy in the specialized documentation
   - Encountering generic ISA issues (coalesced access, load width, etc.) → Refer to `docs/ref-docs/nvidia/gluon/sm90/common_optimizations.md`
   - Encountering pitfalls → Refer to `docs/ref-docs/nvidia/gluon/sm90/pitfalls.md` or inline pitfall experiences in the specialization documentation

2. **No pattern matched** → Follow the optimization order in `docs/ref-docs/nvidia/gluon/sm90/common_optimizations.md`: 3.0 → 3.1 → ... → 3.6> See `docs/ref-docs/nvidia/gluon/sm90/common_optimizations.md` for the detailed optimization checklist.

---

## Step 4: Re-evaluate

After optimization is complete, return to Step 1 to rebuild the Roofline Model Cond and recalculate the corresponding utilization rate based on the bottleneck type.

---

## Step 5: Output Optimization Results

### When utilization reaches 90% or no further optimization opportunities exist

Output includes:
1. **Optimized Gluon code**
2. **Optimization summary report**:
   - Original utilization vs. optimized utilization
   - Optimization content and effect of each step
   - Latency comparison (before vs. after optimization)
   - Unoptimized items and reasons
3. **Verification results**:
   - Precision verification pass confirmation
   - Performance data

---

## Validation And Inspection Guidance

| Tool | Purpose | When to Invoke |
|------|---------|----------------|
| `tools/compute_utilization.py` | Roofline bottleneck analysis + compute/bandwidth utilization calculation | Step 1, Step 4 |
| `ncu` (Nsight Compute) | GPU kernel profiling | Step 2 |
| local accuracy validation | Precision verification | Verify after each optimization step |
| local benchmark | Compare performance with Triton | Final verification |
| TTGIR/layout inspection | Extract TTGIR to analyze layout | When layout modification is needed |

---

## ⚠️ Editing Strategy

Build on the converter guide's source transformation strategy:
- Write each candidate implementation as a coherent file or coherent function-level change
- **Ignore LSP false positives** (type mismatches involving `constexpr`, `gl.*Layout`)
- Only trust `validate.py` and runtime results; do not trust LSP diagnostics

### New File Iteration Strategy (Mandatory)

**Purpose**: Avoid repeatedly editing and reverting the original file.

**Workflow**:
1. **Create a new file**: Copy the original file (e.g., `kernel.py`) as `kernel_v2.py`, and make all modifications on the new file
2. **Verify the new file**: Run precision verification and performance measurement on `kernel_v2.py`
3. **Iterate**: If further modifications are needed, create `kernel_v3.py`, etc., and rewrite the new candidate coherently rather than making scattered line edits
4. **Confirm pass**: After verification passes, **do not overwrite the original file**; keep the final version file (e.g., `kernel_v5.py`)
5. **Record final candidate**: keep the final optimized filename in the optimization notes rather than silently overwriting the original file.

**Prohibited**:
- The cycle of: edit on original file → verification fails → revert → edit again
- Directly overwriting the original file after optimization is complete (users may need to compare or roll back)

---

## Reference Documents

| Document | Content |
|----------|---------|
| `docs/hardware-specs/hardware_specs_hopper.md` | NVIDIA Hopper GPU hardware compute specification table (H20/H100/H200) |
| `docs/ref-docs/nvidia/gluon/sm90/profiling_guide.md` | Nsight Compute (ncu) profiling details |
| `docs/ref-docs/nvidia/gluon/sm90/isa_patterns.md` | Hopper sm_90 SASS instruction patterns and optimization reference |
| `docs/ref-docs/nvidia/gluon/sm90/common_optimizations.md` | General ISA optimization checklist (§3.0-3.6) + quick reference by bottleneck/type + Hopper vs AMD differences |
| `docs/ref-docs/nvidia/gluon/sm90/pitfalls.md` | Practical pitfall experience index (by pattern tag) |
| `docs/ref-docs/nvidia/gluon/sm90/matmul.md` | Standard GEMM optimization topic |
| `docs/ref-docs/nvidia/gluon/sm90/linear_attention.md` | Chunk Linear Attention / Recurrent optimization topic |
| `softmax_reduce.md` | Softmax / Reduction / Element-wise optimization topic |
| `fused_attention.md` | Fused Attention optimization topic |

### Cross-Reference (hopper-converter guide)

| Document | Referenced Content |
|----------|--------------------|
| `../../../../converter/nvidia/hopper/api_mapping.md` | Hopper Gluon API reference |
| `../../../../converter/nvidia/hopper/layouts.md` | Hopper Layout types and TTGIR mapping |
| `../../../../converter/nvidia/hopper/pipeline.md` | async_copy pipeline implementation patterns |
| `../../../../converter/nvidia/hopper/memory_access.md` | Hopper memory access patterns |
| `../../../../converter/nvidia/hopper/matrix_multiply.md` | wgmma matrix multiplication patterns |
| `../../../../converter/nvidia/hopper/common_pitfalls.md` | Hopper common pitfalls and resolutions |

---

## Version Information

**Guide Version**: v2.0
**Last Updated**: 2026-03-20
**Target Architecture**: NVIDIA Hopper (sm_90, H20/H100/H200)
**Design Principles**: Pattern recognition → topic routing + compute utilization closed-loop optimization + instruction-level profile-driven
**v1.0**: Macro-level optimization (Roofline analysis, coalesced memory access, load width, bank conflicts, register spilling, async_copy pipelining, wgmma correctness, SM utilization)
**v1.1**: Added Latency-Bound kernel identification + tile dimension tuning methodology + practical pitfall experience
**v2.0**: Refactored to pattern recognition architecture. Added Step 0 compute pattern recognition, moved §3.0-3.6 to `common_optimizations.md`, moved pitfall experience to `pitfalls.md`, added 4 pattern-specific topic documents (matmul, linear_attention, softmax_reduce, fused_attention), removed `optimization_checklist.md`
