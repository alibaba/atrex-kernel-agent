# Hopper (sm_90) General ISA Optimization Checklist

**Last updated**: 2026-03-20 (v2.0: consolidated and deduplicated from `optimization_checklist.md` and `optimization-guide.md` §3.0-3.6)

---

## ⚠️ Core Principles

1. **Execute in order** — **3.0** → 3.1 → 3.2 → 3.3 → 3.4 → 3.5 → 3.6; fixes at each step may resolve issues in subsequent steps
2. **Verify at each step** — After each optimization step, you must verify accuracy and performance
3. **Accuracy first** — If an optimization step causes accuracy regression, roll back immediately
4. **Record changes** — Log specific modifications and performance changes for each step

**File editing strategy (to save context):**
- **Do NOT** modify the original file directly → rollback → re-verify (wastes significant context tokens)
- **Must** create a new file for modifications (e.g., original file `kernel.py` → create `kernel_v2.py`), iterate on the new file
- Each round of optimization iteration operates on a new file; after verification passes, keep the final version file
- All intermediate files (`_v2`, `_v3`, etc.) are **deleted in one batch** after final verification passes

**After each optimization step, you must verify:**
1. **Accuracy verification** — run the local accuracy check used by the consuming harness
2. **Performance verification** — run `tools/measure_kernel_time.py` or the local benchmark used by the consuming harness

**Serial verification:**
After completing code changes for each optimization point, run verification and wait for the conclusion before proceeding to the next optimization point.

**Why wait serially**: Optimization points have dependencies (3.2 is based on 3.1's result file). If 3.1 verification fails, all subsequent modifications based on it are invalid. You must wait for the verification conclusion to confirm before proceeding.

If an optimization step causes accuracy regression → continue adjusting on the new file, do not touch the original file.
If an optimization step causes performance regression → discard that new file, log the reason, and proceed to the next step.

---

## 3.0 Coalesced Access Pre-check (Step Zero) ⚠️⚠️⚠️

> **This is the prerequisite for all optimizations. Incorrect order does not cause compilation errors or affect accuracy, but can result in multiple times performance loss. This must be completed before any other optimization.**

### Goal

Ensure that the `order` of `BlockedLayout` matches the tensor's actual memory layout in HBM, guaranteeing coalesced access.

### Checking Method

For each `BlockedLayout`, verify that the first element of `order` = the dimension of the tensor in HBM where stride=1:

```python
# 1. Confirm tensor stride
# A[M, K]: stride_am=K, stride_ak=1 → K dimension (dim 1) contiguous → order=[1, 0]
# B[K, N]: stride_bk=N, stride_bn=1 → N dimension (dim 1) contiguous → order=[1, 0]
# B[N, K]: stride_bn=K, stride_bk=1 → K dimension (dim 1) contiguous → order=[1, 0]

# 2. Confirm spt in contiguous dimension ≥ 8 (bf16/fp16) or ≥ 4 (fp32) to achieve 128-bit load
```

### B Matrix Layout Quick Reference Table

| B Layout | B shape | stride_bn | Contiguous dimension | blocked_b order | spt high-value position |
|--------|---------|-----------|---------|----------------|-------------|
| **B=KN** | [K, N] | **1** | dim 1 (N) | **[1, 0]** | spt=[*, **8**] |
| **B=NK** | [N, K] | K | dim 1 (K) | **[1, 0]** | spt=[*, **8**] |
| **B=KN.T** | [K, N].T → [N, K] | K | dim 0 (N→K) | **[0, 1]** | spt=[**8**, *] |

> ⚠️ **Absolutely do not copy order from TTGIR or other cases blindly**. You must determine based on the actual value of runtime `tensor.stride()`.

### Quick Diagnosis

If kernel performance is far below expectations (>2x gap) and all optimizations are ineffective:
```bash
# Check load instruction width in SASS
# Large amounts of LDG.E.32 / LDG.E.U16 → Non-coalesced access! Check order immediately
ncu --import profile.ncu-rep --page source --print-source sass | grep -c 'LDG.E.32'
ncu --import profile.ncu-rep --page source --print-source sass | grep -c 'LDG.E.128'
```

---

## 3.1 Ensure Coalesced Access + Maximize Load/Store Instruction Width

### Goal

On the **premise of guaranteeing coalesced access**, maximize the instruction width of global memory and shared memory:
- Global load/store: `LDG.E.128` / `STG.E.128` (128-bit)
- Shared memory: `STS.128` / `LDS.128` (128-bit)
- CP_ASYNC: `LDGSTS.E.128` (128-bit)

**Coalesced access is the prerequisite, and instruction width is the means**. If the addresses of the 32 threads in a warp are not contiguous, even if each thread uses 128-bit loads, the hardware cannot coalesce them into efficient memory transactions.

### Hopper Coalesced Access Principle

The GPU's memory controller will **coalesce** the memory access requests of multiple threads in the same warp into as few memory transactions as possible:

```
coalesced (coalesced):
  thread 0 → addr 0x1000
  thread 1 → addr 0x1010 ← contiguous access
  thread 2 → addr 0x1020
  ...
  → coalesced sectors, high bandwidth utilization

coalesced (strided):
  thread 0 → addr 0x1000
  thread 1 → addr 0x2000 ← strided access
  thread 2 → addr 0x3000
  ...
  → independent sectors, low bandwidth utilization
```

**Hopper Coalescing Rules**:
- Memory access requests from a warp (32 threads) can be coalesced into 1 transaction if the addresses fall within the same **128B sector**
- Ideal case: 32 threads × 16B/thread (LDG.E.128) = 512B, covering 4 sectors, resulting in 4 transactions
- Note: **warp size = 32** (not AMD's 64)

### Relationship Between Coalesced Access and BlockedLayout

**Key Rule**: The `order` of `BlockedLayout` determines which dimension is contiguous in memory. **This dimension must match the actual storage layout of the tensor in HBM**.

```python
# Matrix A: shape [M, K], stride_am=K, stride_ak=1 → K dimension contiguous in memory
# → blocked_a's order should make K dimension (dim 1) contiguous
blocked_a: gl.constexpr = gl.BlockedLayout(
    size_per_thread=[1, 8],    # K dimension spt=8 → dwordx4 (bf16)
    threads_per_warp=[32, 1],  # Hopper: warp size = 32
    warps_per_cta=[4, 1],
    order=[1, 0]               # dim 1 (K) contiguous → match memory layout
)
```

### Relationship Between Instruction Width and Layout

Instruction width is determined by the value of `size_per_thread` in the contiguous dimension of `BlockedLayout`:

```
Bytes per thread = size_per_thread[contiguous_dim] × element_size_bytes
Instruction width = min(bytes per thread, 16)  # Max 128-bit = 16 bytes
```

| Data Type | element_size | size_per_thread required for 128-bit load ≥ |
|-----------|-------------|--------------------------------------|
| fp32 (4B) | 4 | 4 |
| bf16/fp16 (2B) | 2 | 8 |
| fp8/int8 (1B) | 1 | 16 |

### Diagnosis

1. Check SASS for the presence of `LDG.E.32` / `LDG.E.64` (rather than `LDG.E.128`)
2. Verify that `order` of `BlockedLayout` matches the actual storage layout of the tensor in HBM

### Common Causes and Fixes

| Cause | Fix |
|-------|-----|
| `order` does not match memory layout → no coalescing | Adjust `order` so the innermost dimension = the dimension with stride=1 in the tensor |
| `size_per_thread` is too small in the contiguous dimension → narrow load | Increase `size_per_thread` in the contiguous dimension (bf16 requires ≥ 8 elements to reach 128-bit) |
| The original layout in TTGIR is inherently narrow or non-coalescing | Re-select the layout, but must guarantee functional equivalence |

### ⚠️ Notes

- **Coalesced access takes priority over instruction width** — A coalesced 64-bit load is more efficient than a non-coalesced 128-bit load
- **Do NOT** blindly increase size_per_thread — the layout semantics must remain correct, and the order must match the memory layout
- **warp size = 32**: The product of `threads_per_warp` across all `BlockedLayout` must equal 32
- After modifying the load layout, downstream layouts may also need to be adjusted

---

## 3.2 Checking for Shared Memory Bank Conflicts

### Goal

Eliminate shared memory bank conflicts to reduce the latency of `LDS`/`STS`.

### Hopper Shared Memory Bank Configuration

- **32 banks, 4-byte granularity per bank** (same as AMD)
- Bank calculation: `bank = (byte_address / 4) % 32`
- Multiple threads in the same warp accessing different addresses within the same bank → bank conflict → serialization

### Diagnostic Method

```bash
# 1. ncu metrics check bank conflicts
ncu --import profile.ncu-rep --metrics \
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum

# 2. Gluon API check
# Add to kernel code (for debugging):
conflicts = gl.bank_conflicts(shared_layout)
```

### Hopper Specifics: NVMMASharedLayout

wgmma requires operands to be in `NVMMASharedLayout`:

```python
gl.NVMMASharedLayout(
    swizzle_byte_width=128,   # swizzle mode (32/64/128)
    element_bitwidth=16,       # element bit width (bf16=16)
    transposed=False           # whether transposed
)
```

- `NVMMASharedLayout` has built-in swizzle patterns to eliminate bank conflicts
- `swizzle_byte_width` parameter is determined by TTGIR and generally does not need manual adjustment
- **When extracting layouts from TTGIR, the swizzle of NVMMASharedLayout is already optimized**

### SwizzledSharedLayout Parameter Meanings

For non-wgmma paths (e.g., ordinary smem → register load), `SwizzledSharedLayout` is still used:

```python
gl.SwizzledSharedLayout(vec, perPhase, maxPhase, order=[1, 0])
```

| Parameter | Meaning | Impact on Bank Conflicts |
|-----------|---------|--------------------------|
| `vec` | Contiguous vector width (unit: number of elements) | Larger → more elements accessed contiguously per access |
| `perPhase` | Rows per phase | Controls the XOR swizzle period |
| `maxPhase` | Maximum number of phases | Controls the XOR swizzle range |

### Common Causes and Fixes

| Cause | Fix |
|------|---------|
| Improper SwizzledSharedLayout parameters | Adjust `vec`, `perPhase`, `maxPhase` parameters |
| Inappropriate swizzle_byte_width for NVMMASharedLayout | Try different swizzle_byte_width values (32/64/128) |

### ⚠️ Notes

- After modifying swizzle parameters, related downstream layouts may also need adjustment
- Bank conflict = 0 is the ideal target, but a small amount of 2-way conflicts has minimal impact
- Prioritize eliminating bank conflicts on high-frequency execution paths (e.g., inside the main loop)

---

## 3.3 Eliminating Scratch/Local Memory Operations (Register Spills)

### Objective

Eliminate local memory loads/stores (i.e., register spills to DRAM).

### Diagnostic Methods

```bash
# 1. Check STL/LDL instructions in SASS
ncu --import profile.ncu-rep --page source --print-source sass | grep -E 'STL|LDL'

# 2. Check ncu local memory metrics
ncu --import profile.ncu-rep --metrics \
    lts__t_sectors_op_read_lookup_hit.sum,\
    lts__t_sectors_op_write.sum

# 3. Check compiler spill information
# View registers per thread in Launch Statistics of ncu report
```

### Common Causes and Fixes

| Cause | Fix |
|------|---------|
| Block size too large, insufficient registers | Reduce BLOCK_SIZE_M / BLOCK_SIZE_N / BLOCK_SIZE_K |
| Too many simultaneously live variables in the loop | Reorder code to narrow variable live ranges |
| Unnecessary intermediate variables | Combine computations, reduce temporary tensors |
| Pipeline depth too deep (num_stages too large) | Reduce num_stages |

### Hopper Register Limits

| Parameter | Value | Impact |
|------|------|------|
| Total registers per SM | 65,536 | Shared across all active threads |
| Max registers per thread | 255 | Exceeding → spills to local memory |
| Relationship between occupancy and registers | `max_threads_per_SM / (blocks × threads_per_block)` | More registers → lower occupancy |

**Register vs Occupancy Trade-off**:

| Registers per Thread | Max Threads per SM | Occupancy (2048 max) |
|-----------------|------------------|---------------------|
| 32 | 2048 | 100% |
| 64 | 1024 | 50% |
| 128 | 512 | 25% |
| 255 | 256 | 12.5% |

> The performance penalty of register spills (STL/LDL) far outweighs reduced occupancy. Prioritize eliminating spills.

---

## 3.4 async_copy Pipeline Optimization

### Objective

Ensure `async_copy_global_to_shared` (CP_ASYNC DMA) is used for global → shared memory data transfers, and implement software pipelining to overlap data loading with computation.

### Background

Hopper's CP_ASYNC DMA can transfer data directly from global memory to shared memory, **bypassing registers**.

| Transfer Method | SASS Instruction | Performance |
|---------|----------|------|
| ✅ CP_ASYNC DMA | `LDGSTS.E.128` | Optimal (bypasses registers) |
| ❌ Two-step transfer | `LDG.E.128` + `STS.128` | **50%+ slower** (via registers) |

### Diagnostics

```bash
# Check if async_copy (LDGSTS) is used
ncu --import profile_output.ncu-rep --page source --print-source sass | grep -c 'LDGSTS'

# If = 0 and has LDG+STS → async_copy not used → must fix
```

### Correct Implementation

```python
from triton.experimental.gluon.language.nvidia.hopper import async_copy

# ✅ CP_ASYNC DMA (bypass registers, optimal performance)
async_copy.async_copy_global_to_shared(
    smem.index(slot),
    ptr + gl.cast(offsets, gl.int32),
    mask=mask
)
async_copy.commit_group()
# ... can insert computation to overlap ...
async_copy.wait_group(num_outstanding=0)
```

### Software Pipelining Pattern

```python
# Allocate multi-slot smem buffer
a_smem = gl.allocate_shared_memory(dtype, [num_stages, BM, BK], smem_layout)

# Prologue: Preload first num_stages-1 stages
for s in range(num_stages - 1):
    async_copy.async_copy_global_to_shared(a_smem.index(s), ...)
    async_copy.commit_group()

# Main loop
for i in range(num_stages - 1, num_iters):
    # 1. Initiate next stage load
    slot_next = i % num_stages
    async_copy.async_copy_global_to_shared(a_smem.index(slot_next), ...)
    async_copy.commit_group()

    # 2. Wait for current stage ready
    async_copy.wait_group(num_outstanding=num_stages - 1)

    # 3. Compute using current stage data
    slot_cur = (i - num_stages + 1) % num_stages
    fence_async_shared()
    acc = warpgroup_mma(a_smem.index(slot_cur), b_smem.index(slot_cur), acc, is_async=True)
    acc = warpgroup_mma_wait(num_outstanding=0, deps=(acc,))

# Epilogue: Process remaining stages
for s in range(min(num_stages - 1, num_iters)):
    async_copy.wait_group(num_outstanding=num_stages - 2 - s)
    # ... compute ...
```

### ⚠️ Notes

- Larger `num_stages` → more smem usage → potential occupancy decrease
- Hopper smem max is 228 KB, much larger than AMD's 64 KB, supporting more stages
- In `wait_group(num_outstanding=N)`, N indicates a maximum of N groups of asynchronous operations are allowed to be incomplete
- N=0 means wait for all to complete (safest but shallowest pipeline)

See `converter/nvidia/hopper/pipeline.md`

---

## 3.5 wgmma fence/wait Correctness

### Objective

Ensure the three-step pattern for wgmma (warp group matrix multiply-accumulate) is complete and correct.

### Three-Step Pattern (All Three Are Required)

```python
# Step 1: fence — Ensure smem writes visible to wgmma
fence_async_shared()

# Step 2: Async wgmma — Operands in NVMMASharedLayout shared memory
acc = warpgroup_mma(a_smem, b_smem, acc, is_async=True)

# Step 3: wait — Wait for wgmma completion before using results
acc = warpgroup_mma_wait(num_outstanding=0, deps=(acc,))
```

### Common Errors and Consequences

| Error | Consequence | SASS Symptom |
|------|------|----------|
| Missing `fence_async_shared()` | Reading incomplete smem writes → incorrect results | No `FENCE.PROXY.ASYNC.SHARED` |
| Missing `warpgroup_mma_wait()` | Using incomplete computation results → incorrect results | No `WGMMA.WAIT` |
| `num_outstanding` too large | Insufficient waiting → incorrect results | Incorrect parameter for `WGMMA.WAIT` |
| Missing `deps=(acc,)` | Compiler may optimize away the wait | wait is optimized out |
| Operand not in NVMMASharedLayout | Compilation error or incorrect results | No `WGMMA` instruction |

### Interaction with async_copy

In a pipeline, the correct sequence for wgmma and async_copy:

```python
# Correct order:
async_copy.wait_group(...)          # 1. Wait for smem data ready
fence_async_shared()                 # 2. fence: smem writes visible to wgmma
acc = warpgroup_mma(a, b, acc, ...) # 3. Initiate wgmma
acc = warpgroup_mma_wait(...)       # 4. Wait for wgmma completion
```

---

## 3.6 SM Utilization / Tile Size Tuning (Small Matrix Specialization)

### Objective

For scenarios involving small matrices (grid size far less than SM count), maximize actual throughput by adjusting tiling strategies.

### SM Utilization Quick Reference

```
grid_blocks = cdiv(M, BLOCK_SIZE_M) × cdiv(N, BLOCK_SIZE_N)
SM_utilization = grid_blocks / num_SMs

H20: 78 SMs, H100/H200: 132 SMs
```

| grid_blocks / SMs | SM Utilization | Optimization Direction |
|-------------------|----------|---------|
| ≥ 50% | Normal | Apply optimizations from §3.1–§3.5 |
| 10%–50% | Low | Consider reducing tile size + ISA optimization |
| < 10% | **Severely insufficient** | **Must prioritize adjusting tiling strategy** |

### 3.6.1 Reducing BLOCK_SIZE_M / BLOCK_SIZE_N to Increase Grid Parallelism (Highest Priority ⭐⭐)

**Core Principle**: In small matrix scenarios, SM utilization is the primary bottleneck. Reducing tile size can significantly increase the number of parallel blocks.

**⚠️ Must Re-Extract TTGIR**: After modifying BLOCK_SIZE_M/N, **all** layout parameters will change. Use `extract_ttgir.py` to obtain the new layout.

### 3.6.2 Increasing BLOCK_SIZE_K to Reduce Loop Overhead

For small K dimensions (K < 256), with few loop iterations, loop control overhead accounts for a high proportion.

| BLOCK_SIZE_K | K=64 Iterations | K=128 Iterations | Notes |
|-------------|------------|-------------|------|
| 16 | 4 | 8 | Default value, high overhead ratio |
| **32** | **2** | **4** | **Recommended**, loop overhead halved |
| 64 | 1 | 2 | Aggressive, may cause excessive register pressure |

### 3.6.3 Platform-Agnostic Micro-Optimizations

The following optimizations are safe and effective on all GPUs:

1. **Remove unnecessary `other=0.0`**: When the mask guarantees all elements will be loaded, the `other=0.0` parameter generates additional conditional selection instructions. Removing it reduces instruction count.

2. **`tl.assume(stride > 0)`**: Informs the compiler that all stride parameters are positive, helping the compiler optimize address computation. Zero cost, should be added by default.

```python
# Add at beginning of kernel function
tl.assume(stride_am > 0)
tl.assume(stride_ak > 0)
tl.assume(stride_bk > 0)
tl.assume(stride_bn > 0)
```

3. **Avoid unnecessary `gl.cast`**: Ensure offset types are correct to reduce type conversion instructions.

---

## Appendix A: Optimization Decision Quick Reference

### Select Optimization Focus by Bottleneck Type

| Bottleneck Type | Primary Optimization (Highest Return) | Secondary Optimization | Tertiary Optimization (Fine-Tuning) |
|---------|-------------------|---------|----------------|
| **Compute Bound** | 3.3 Eliminate scratch/spill | 3.5 wgmma correctness | 3.2 Bank conflicts |
| **Memory Bound** | 3.0+3.1 Coalesced access + wide load | 3.4 async_copy pipeline | 3.2 Bank conflicts |
| **Latency-Bound** | Tile dimension tuning (see `linear_attention.md`) | 3.0 Coalesced access pre-check | ❌ Most ISA optimizations ineffective |
| **Insufficient SM Utilization** | 3.6 Tile size tuning | 3.6.2 BLOCK_K tuning | 3.6.3 Micro-optimizations |

### Selecting Optimizations by Kernel Type

| Kernel Type | Required | Optional | Not Applicable | Topic Docs |
|------------|------|------|--------|---------|
| **Large Tile GEMM** | 3.0, 3.1, 3.3, 3.4, 3.5 | 3.2 | 3.6 (SM sufficient) | `matmul.md` |
| **Small Matrix GEMM** | 3.0, 3.6 | 3.1, 3.3 | 3.4 (loop too short) | `matmul.md` |
| **Attention** | 3.0, 3.1, 3.3, 3.4, 3.5 | 3.2 | — | Fused attention topic |
| **Recurrent State Update** | Increase chunk_size, 3.0, 3.1 | 3.5 | 3.3, 3.4 (may degrade performance) | `linear_attention.md` |
| **Element-wise** | 3.0, 3.1 | — | 3.3, 3.4, 3.5 (no wgmma) | Softmax/reduce topic |
| **Reduction** | 3.0, 3.1 | 3.2 | 3.4, 3.5 (no wgmma) | Softmax/reduce topic |

---

## Appendix B: Hopper vs AMD Optimization Differences Quick Reference

| Optimization Dimension | AMD (CDNA3) | Hopper (sm_90) |
|---------|-------------|----------------|
| Warp Size | 64 | **32** |
| Matrix Multiply | `mfma` (operands in registers) | `wgmma` (operands in smem) |
| DotOperandLayout | Required | **Not required** |
| Global Load | `buffer_load` | `gl.load` / `LDG` |
| Pipeline Data | Manual `buffer_load` → smem | `async_copy` / `LDGSTS` (CP_ASYNC DMA) |
| Smem Capacity | 64 KB/CU | **228 KB/SM** (configurable) |
| Profiler | `rocprofv3 --att` | `ncu` (Nsight Compute) |
| Registers | 512 VGPR/wave | 255 reg/thread, 65536/SM |
| Scheduling Hints | `warp_pipeline_stage`, `sched_group_barrier` | **No equivalent** |
| Multi-die | XCD remap (MI300X: 8 XCD) | **Not applicable** |
| `ds_bpermute` Elimination | Important optimization | **Not applicable** (Hopper has no such instruction) |
| Load Width Target | `buffer_load_dwordx4` | `LDG.E.128` / `LDGSTS.E.128` |
| Bank Conflict | 32 banks, 4B/bank | **Same**: 32 banks, 4B/bank |

## Related

- **Prerequisites**: [Hopper Hardware Specifications](../../common/hardware-specs/hopper.md)
- **Cross-Architecture Reference**: [CDNA3 ISA Optimization Checklist](../../../amd/gluon/gfx942/common_optimizations.md) | [CDNA4 ISA Optimization Checklist](../../../amd/gluon/gfx950/common_optimizations.md)
- **ISA Reference**: [Hopper SASS Instruction Patterns](isa_patterns.md) — Detailed LDG/STG/WGMMA instruction descriptions
- **Profiling**: [Hopper ncu Profiling Guide](profiling_guide.md)
