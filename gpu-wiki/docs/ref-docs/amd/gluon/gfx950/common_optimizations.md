# CDNA4 (gfx950) Generic ISA Optimization Checklist

**Last Updated**: 2026-03-28 (v1.0: Based on submission_gluon_v2-v7 tested optimization experience)

---

## ⚠️ Core Principles

1. **Execute in order** — **3.0** → 3.1 → 3.2 → 3.3 → 3.4 → 3.5 → ..., each step's fix may eliminate issues in subsequent steps
2. **Verify at each step** — Verify accuracy and performance after each optimization step
3. **Accuracy first** — If an optimization step causes accuracy regression, roll back immediately
4. **Record changes** — Record the specific modifications and performance changes for each step

**File editing strategy (save context):**
- **Do not** modify the original file directly → revert → re-verify (wastes excessive context tokens)
- **Must** create a new file for modifications (e.g., original file `kernel.py` → create `kernel_v2.py`), iterate on the new file
- Each round of optimization iteration operates on a new file; keep the final version file after verification passes
- All intermediate files (`_v2`, `_v3`, etc.) are **deleted all at once** after final verification passes

**After each optimization step, must verify:**
1. **Accuracy verification** — run the local accuracy check used by the consuming harness
2. **Performance verification** — run `tools/measure_kernel_time.py` or the local benchmark used by the consuming harness

**Serial verification:**
After completing code modifications for each optimization point, run verification and wait for the result before proceeding to the next optimization point.

**Why wait serially**: There are dependencies between optimization points. If 3.1 verification fails, all subsequent modifications based on it are invalidated. Must wait for verification conclusion before advancing.

If an optimization step causes accuracy regression → continue adjusting on a new file, do not touch the original file.
If an optimization step causes performance regression → abandon the new file, record the reason, continue to the next step.

---

## 3.0 Coalesced Memory Access Pre-check (Step Zero) ⚠️⚠️⚠️

> **This is the prerequisite for all optimizations. Incorrect order does not produce compilation errors or affect accuracy, but can cause multiple times of performance loss. Must be completed before any other optimizations.**

### Objective

Ensure that the `order` of `BlockedLayout` matches the tensor's actual memory layout in HBM, ensuring coalesced access.

### Check Method

For each `BlockedLayout`, verify that the first element of `order` = the dimension with stride=1 of the tensor in HBM:

```python
# 1. tensor stride
# Q[total_q, num_heads, 576]: stride reshape layout
# KV[total_kv, 288]: stride_kc=288, stride_ck=1 -> K (dim 1) contiguous -> order=[1, 0]

# 2. spt contiguousdimension ≥ 8 (bf16/fp16) or ≥ 16 (fp4 packed) dwordx4 load
```

### Quick Diagnosis

If kernel performance is far below expectation (>2x gap) and all optimizations are ineffective:
```bash
# check load
# buffer_load_ubyte / buffer_load_ushort -> coalesced！check order
```

---

## 3.1 Ensure Coalesced Access + Maximize Load/Store Instruction Width

### Objective

On the premise of **ensuring coalesced access**, maximize the instruction width of global memory and shared memory:
- Global load/store: `buffer_load_dwordx4` (128-bit)
- Shared memory: `ds_read_b128` / `ds_write_b128` (128-bit)

**Coalesced access is the prerequisite, instruction width is the means**. If the addresses of 64 threads in a wavefront are not contiguous, even if each thread uses `dwordx4`, the hardware cannot coalesce them into efficient memory transactions.

### CDNA4 Coalesced Access Principle

The GPU's memory controller will **coalesce** the memory access requests of multiple threads within the same wavefront into as few memory transactions as possible.

### Common Causes and Fixes

| Cause | Fix |
|------|---------|
| `order` does not match memory layout → no coalescing | Adjust `order` so that the innermost dimension = the dimension with stride=1 in the tensor |
| `size_per_thread` is too small along the contiguous dimension → narrow load | Increase `size_per_thread` along the contiguous dimension (bf16 requires ≥ 8 to achieve dwordx4) |
| The original layout in TTGIR is inherently narrow or non-coalesced | Need to reselect the layout, but must ensure functional equivalence |

### FP4 Special Handling

FP4 data is packed as uint8 (2 elements per byte). To achieve dwordx4 loading:
- `size_per_thread` × 4 bits ≥ 128 bits → `size_per_thread` ≥ 32
- Or use `[1, 4]` layout with appropriate `threads_per_warp`

---

## 3.2 Check LDS Write Width + Bank Conflicts

### Objective

Eliminate **narrow writes** and bank conflicts in shared memory (LDS).

### Diagnosis

```bash
# critical: check ds_write
grep -o "ds_write_[a-z0-9_]*" $ASM | sort | uniq -c
# if ds_write_b16 / ds_write_b16_d16_hi -> write！
# ds_write_b128 (128-bit vectorwrite)
```

### Common Causes and Fixes

| Cause | Fix |
|------|---------|
| **Narrow tile (BV≤16) via smem.store()** | ⭐ Use `gl.convert_layout(data, dot_op)` instead of the smem path |
| Improper SwizzledSharedLayout parameters | Adjust `vec`, `perPhase`, `maxPhase` parameters |
| Data arrangement causing multiple threads to access the same bank simultaneously | Modify swizzle parameters to eliminate conflicts |

### ⭐ Narrow Tile ds_write_b16 Fix (Measured +6-13%)

When the tile width is ≤ 16 (e.g., a [64,16] bf16 tile with BV=16), ``smem.index(0).store(data)`` cannot form 128-bit vector writesarman, and degrades to element-wise 16-bit writes.

````python
# ❌ tile smem -> 44×ds_write_b16 (performance)
smem_b.index(0).store(b_h.to(gl.bfloat16))
h_dot = smem_b.index(0).load(dot_op1)

# ✅ convert_layout -> write, ds_read_b64_tr_b16
h_dot = gl.convert_layout(b_h.to(gl.bfloat16), dot_op1)
````

**Guideline**: Use `convert_layout` when BV ≤ 16, and `smem` when BV ≥ 32.
**Source**: chunk-GDN fwd_h optimization practice (2026-04-24).

MI355X LDS has 32 banks; the swizzle parameter should ensure that 32 consecutive elements are distributed across different banks.

---

## 3.3 Eliminate ds_bpermute Instructions

### Objective

Eliminate unnecessary ``ds_bpermute_b32`` instructions.

### Background

``ds_bpermute`` is typically introduced by ``gl.convert_layout()``. When the Gluon compiler needs to convert data between different layouts, it performs cross-lane shuffles through LDS ``ds_bpermute``.

### Diagnosis

- Check whether the assembly contains ``ds_bpermute_b32`` instructions
- If present, trace back to the corresponding Gluon source code, usually a ``gl.convert_layout()`` call

### Common Causes and Fixes

| Cause | Fix |
|------|---------|
| Explicit ``gl.convert_layout()`` is avoidable | Reorganize data flow so the upstream directly outputs the target layout |
| Implicit conversions between different layouts | Unify layouts to reduce layout switching |
| `convert_layout` immediately after load | Load directly with the target layout (adjust BlockedLayout) |

**Note**: Not all ``ds_bpermute`` can be eliminated. Some layout conversions are algorithmically necessary. Prioritize eliminating ``ds_bpermute`` inside high-frequency loops.

---

## 3.4 Eliminate Scratch Operations (Register Spills)

### Objective

Eliminate scratch-space `buffer_load`/`buffer_store` instructions (i.e., register spills to device memory).

### Diagnosis

- Check whether the assembly contains ``buffer_load`` / ``buffer_store`` targeting scratch space
- Check whether the hardware counter ``SPI_RA_VGPR_SGPR_FULL_CSN`` is high

### Common Causes and Fixes

| Cause | Fix |
|------|---------|
| Block size too large, insufficient VGPRs | Reduce BLOCK_SIZE_M / BLOCK_SIZE_N / BLOCK_SIZE_K |
| Too many variables simultaneously live inside loops | Rearrange code to narrow variable live ranges |
| Unnecessary intermediate variables | Combine computations to reduce temporary tensors |
| Pipeline depth too deep | Reduce num_stages / pipeline depth |

### CDNA4 VGPR Limits

- Per wave: maximum 512 VGPRs (architectural)
- Actual availability depends on occupancy requirements
- More VGPR use → lower occupancy → fewer concurrent waves → harder to hide latency

---

## 3.5 Optimize Memory Access Stalls (Compute-Memory Overlap)

### Objective

Ensure that memory load latency is effectively hidden by compute.

### Diagnosis

- The Stall column for ``buffer_load`` instructions is high while the Idle column is low
- Load and compute instructions are not interleaved within the loop

### Common Causes and Fixes

| Cause | Fix |
|------|---------|
| Software pipelining not implemented | Implement a three-phase pipeline (prologue + main loop + epilogue) |
| Load and compute arranged serially | Rearrange code so the current iteration's compute overlaps with the next iteration's load |
| Insufficient prefetch distance | Increase prefetch lead time |

### Special Case: MLA Decode

MLA Decode attention's inner loop has complex online softmax control flow, which is not suitable for ``warp_pipeline_stage`` full-stage packing. Use instead:
- Full/Tail block splitting (OPT-3)
- Scalar base address precomputation (OPT-1,2)
- Dynamic V format selection (fp8 for large kv, bf16 for small kv)

---

## 3.6 warp_pipeline_stage Full-Stage Packing (Key GEMM Optimization) ⭐

### Objective

Pack all stages of the GEMM loop (ds_read, MFMA, ds_write) with ``warp_pipeline_stage`` and hand them off to the compiler, allowing the WarpPipeliner to automatically perform ping-pong cross-iteration pipeline scheduling.

**This is the most important optimization technique for GEMM kernels.**

### Key Rules

1. **All LDS operations and compute must be packed** — use ``"prep"`` for ds_read, ``"compute"`` for MFMA, and ``"prep"`` for ds_write; none can be omitted
2. **Do not manually write ``gl.barrier()``** — let the compiler's Membar pass auto-insert them
3. **Place ds_write between the two MFMA groups** — put the ds_write ``"prep"`` stage between subslice 2's ``"compute"`` and subslice 3's ``"compute"``
4. **Place buffer_load between pipeline stages, not inside any stage**

### Diagnosis

If the assembly shows a regular alternation of ``sched_barrier(0)`` + ``s_barrier``, it indicates the WarpPipeliner has successfully taken effect.

### Not Applicable to MLA Decode

Due to the complexity of online softmax control flow in MLA Decode, ``warp_pipeline_stage`` is **not applicable**. Use the software pipelining approach described in §3.5 instead.

## 3.7 Combined Layout + num_warps Tuning (Key for Large Tile GEMM) ⭐

### Objective

Jointly adjust `mma_layout.warps_per_cta` (determines num_warps), `b_load_layout.threads_per_warp`, and `b_shared_layout.order` to break through the VGPR bottleneck and unlock pipeline optimization opportunities.

### When This Step Is Needed

- §3.6's `warp_pipeline_stage` causes performance regression instead of improvement
- §3.4 detects scratch overflow that cannot be resolved by reducing block size
- Accumulator occupies ≥ 384 VGPRs (approaching the 512 limit)

### Search Dimensions

| Dimension | Candidate Values | Impact |
|-----------|------------------|--------|
| `mma warps_per_cta` | [2,2]4w, [2,4]8w, [4,2]8w, [1,4]4w, [1,8]8w, [8,1]8w | Per-warp accumulator size → VGPR pressure → pipeline feasibility |
| `b_load threads_per_warp` | [2,32], [4,16], [8,8] | Global→register load pattern → buffer_load instruction width/efficiency |
| `b_shared order` | [1,0], [0,1] | LDS store layout → bank conflicts → ds_read/ds_write efficiency |

### VGPR Budget Quick Reference (256×256 tile, MFMA 32×32×8)

| warps_per_cta | num_warps | per-warp acc VGPRs | pipeline headroom |
|---------------|-----------|-------------------|-------------------|
| [2,2] / [1,4] / [4,1] | 4 | 512 | **0 (limit! pipeline will definitely spill)** |
| [2,4] / [4,2] / [1,8] / [8,1] | 8 | 256 | **256 VGPRs (sufficient for pipeline)** |

---

## 3.8 Attention-Specific Optimizations ⭐

See `mla_decode.md` and the fused-attention topic for details.

Core optimizations:
- OPT-3: Full/Tail block splitting
- OPT-6: XCD-aware PID remapping
- OPT-7: Split-K refinement
- OPT-8: Scales memory access coalescing
- OPT-9: V FP8 load

---

## 3.9 Small Matrix / Low CU Utilization Optimizations ⭐

### Objective

For small matrix scenarios (where grid size is much smaller than the number of CUs), maximize actual throughput by adjusting tiling strategy and loop parameters.

### Diagnosis — CU Utilization Quick Reference

```
grid_blocks = cdiv(M, BLOCK_SIZE_M) × cdiv(N, BLOCK_SIZE_N)
CU_utilization = grid_blocks / num_CUs   (MI355X: 256 CUs)
```

| grid_blocks / CUs | CU Utilization | Optimization Direction |
|-------------------|----------------|------------------------|
| ≥ 50% | Normal | Apply §3.1–§3.8 ISA-level optimizations |
| 10%–50% | Low | Consider reducing tile size + ISA optimizations |
| < 10% | **Severely insufficient** | **Prioritize adjusting tiling strategy** |

### Optimization Checklist (Sorted by Priority)

#### 3.9.1 Reduce BLOCK_SIZE_M / BLOCK_SIZE_N to Increase Grid Parallelism (Highest Priority ⭐⭐)

**This is the single most impactful optimization for small matrix scenarios**, yielding measured improvements of **2.5–2.85×**.

When grid blocks are extremely sparse (< 10% CU), reducing tile size greatly increases the number of parallel blocks, directly boosting CU utilization.

#### 3.9.2 Increase BLOCK_SIZE_K to Reduce Loop Overhead (⭐)

For small K dimensions (K < 256), **the loop iteration count is low and loop control overhead is proportionally high**. Increasing BLOCK_SIZE_K significantly reduces the number of iterations.

#### 3.9.3 Optimizations Not Applicable to Small Matrices (Avoid Negative Optimization)

The following optimizations were found to be ineffective or harmful on small matrices (under large tile configurations):

| Optimization | Small Matrix Result | Reason |
|--------------|---------------------|--------|
| **warp_pipeline_stage** | ❌ -14% | Too few iterations; pipeline fill/drain overhead outweighs overlap benefit |
| **num_warps increase** | ❌ -14% | Adding warps cannot improve occupancy with very few grid blocks |
| **XCD remapping** | ≈ 0% | Grid too small; no load imbalance across XCDs |

---

## XCD/PID Remapping (MI355X Load Balancing) ⭐

The MI355X has 8 XCDs, each with 32 CUs. Hardware assigns consecutive PIDs to different XCDs in a round-robin fashion. Without remapping, some XCDs receive all heavy blocks while others receive all light blocks, causing severe load imbalance.

```python
NUM_XCDS = 8
pid_raw = gl.program_id(0)
wave = pid_raw // NUM_XCDS
pos_in_wave = pid_raw % NUM_XCDS
is_odd_wave = wave % 2
remapped_pos = tl.where(is_odd_wave, NUM_XCDS - 1 - pos_in_wave, pos_in_wave)
pid = wave * NUM_XCDS + remapped_pos
pid = tl.minimum(pid, total_blocks - 1)
```

**Effect**: Ensures adjacent blocks are distributed across different XCDs, improving HBM bandwidth utilization. Applicable to MLA decode, GEMM, and other large-grid kernels.

---

## Quick Reference by Bottleneck Type

| Bottleneck Type | Priority Optimizations |
|-----------------|----------------------|
| **Memory Bound** | 3.0 → 3.1 → 3.5 → 3.8(OPT-9) |
| **Compute Bound** | 3.3 → 3.4 → 3.6 → 3.7 |
| **Small Matrix (CU < 10%)** | 3.9.1 → 3.9.2 → 3.0 → 3.1 |
| **Attention** | `mla_decode.md` OPT-3 → OPT-9 |

## Quick Reference by Operator Type

| Operator Type | Applicable Optimizations |
|---------|-----------|
| **Standard GEMM** | 3.0 → 3.1 → 3.3 → 3.6 → 3.7 |
| **MLA Decode** | `mla_decode.md` (OPT-3 → OPT-9) |
| **Fused Attention** | Fused-attention topic |
| **Element-wise / Reduce** | 3.0 → 3.1 → 3.9 |

---

## CDNA4 vs Hopper Differences

| Feature | CDNA4 (gfx950) | Hopper (sm_90) |
|------|---------------|---------------|
| **Warp size** | 64 threads | 32 threads |
| **Matrix instruction** | MFMA (v_mfma_*) | WGMMA |
| **Async copy** | async_copy.buffer_load_to_shared | cp.async.bulk |
| **In-thread transpose** | Disabled | Enabled |
| **kpack** | Fixed at 1 | Configurable |
| **LDS capacity** | 160 KB/CU | 228 KB/block max |

These differences mean:
- CDNA4 cannot use in-thread transpose optimization; layout must be explicitly managed
- CDNA4's kpack is fixed at 1 and cannot rely on kpack packing
- CDNA4's ping-pong scheduling is only activated when using async_copy
