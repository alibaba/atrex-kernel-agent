# ISA Optimization Detailed Checklist

**Last updated**: 2026-03-23 (v2.4: restructured to a general optimization checklist; the GEMM-specific §3.6 WPS is covered in `warp_pipeline_stage.md`, and `optimization_checklist.md` has been removed)

> **Note**: This document contains general ISA optimization steps (§3.0-§3.5) applicable to all AMD CDNA3 kernels.
> For GEMM-specific `warp_pipeline_stage` optimizations, see `warp_pipeline_stage.md`.
> See Appendix A of this document for a quick-reference optimization decision table for various kernel types.

---

## ⚠️ Core Principles

1. **Execute in order** — **3.0** → 3.1 → 3.2 → 3.3 → 3.4 → 3.5 → 3.6/3.7 (GEMM) or 3.8 (Attention). Fixing issues at each step may eliminate problems in subsequent steps.
2. **Verify at each step** — Validate both accuracy and performance after each optimization step.
3. **Accuracy first** — If an optimization step causes accuracy regression, roll back immediately.
4. **Document changes** — Record specific modifications and performance changes for each step.

---

## 3.0 Coalesced Memory Access Pre-Check (Step Zero) ⚠️⚠️⚠️

> **This is the prerequisite for all optimizations. An incorrect order does not trigger compilation errors or affect accuracy, but it causes a 3× performance penalty (Pitfall 53). Must be completed before any other optimizations.**

### Inspection Method

For each `BlockedLayout`, verify that the first element of `order` corresponds to the dimension of the tensor with stride=1 in HBM:

```python
# 1. Confirm tensor stride
# A[M, K]: stride_am=K, stride_ak=1 → K dimension (dim 1) is contiguous → order=[1, 0]
# B[K, N]: stride_bk=N, stride_bn=1 → N dimension (dim 1) is contiguous → order=[1, 0]
# B[N, K]: stride_bn=K, stride_bk=1 → K dimension (dim 1) is contiguous → order=[1, 0]

# 2. Confirm spt in contiguous dimension ≥ 8 (bf16/fp16) or ≥ 4 (fp32) to achieve dwordx4
```

### B Matrix Layout Quick Reference

| B Layout | B shape | stride_bn | Contiguous Dim | blocked_b order | High-Value Position in spt |
|----------|---------|-----------|----------------|-----------------|----------------------------|
| **B=KN** | [K, N] | **1** | dim 1 (N) | **[1, 0]** | spt=[*, **8**] |
| **B=NK** | [N, K] | K | dim 1 (K) | **[1, 0]** | spt=[*, **8**] |
| **B=KN.T** | [K, N].T → [N, K] | K | dim 0 (N→K) | **[0, 1]** | spt=[**8**, *] |

> ⚠️ **Never blindly copy the order from TTGIR or other examples**. You must determine it based on the actual runtime value of `tensor.stride()`.

### Quick Diagnosis

If the kernel performs far below expectations (gap > 2×) and all scheduling strategies fail:
```bash
grep -c 'buffer_load_ushort' kernel.amdgcn    # > 0 → Non-coalesced access! Check order immediately
grep -c 'buffer_load_dwordx4' kernel.amdgcn   # = 0 → Same as above
```

See Appendix I Pitfall 53 for details.

---

## 3.1 Ensure Coalesced Access + Maximize buffer_load/buffer_store Instruction Width

### Objective

**On the premise of ensuring coalesced memory access**, maximize the instruction width for global memory and LDS:
- Global load/store: `buffer_load_dwordx4` / `buffer_store_dwordx4` (128-bit)
- LDS read/write: `ds_read_b128` / `ds_write_b128` (128-bit)

**Coalesced access is the prerequisite; instruction width is the means**. If the addresses of the 64 threads in a wavefront are not contiguous, even if each thread uses `dwordx4`, the hardware cannot coalesce them into efficient memory transactions. Instead, it produces a large number of independent requests with extremely low bandwidth utilization.

### Coalesced Access Principles

The GPU's memory controller **coalesces** memory access requests from multiple threads within the same wavefront into as few memory transactions as possible:
Ideal coalesced access:
  thread 0 → addr 0x1000
  thread 1 → addr 0x1010    ← Adjacent threads access contiguous addresses
  thread 2 → addr 0x1020
  thread 3 → addr 0x1030
  → Merged into 1 large transaction, high bandwidth utilization

Non-coalesced access (uncoalesced / strided):
  thread 0 → addr 0x1000
  thread 1 → addr 0x2000    ← Adjacent threads access non-contiguous addresses (large stride)
  thread 2 → addr 0x3000
  thread 3 → addr 0x4000
  → Split into multiple independent transactions, low bandwidth utilization
**CDNA3 Coalescing Rules**:
- Memory access requests from a wavefront (64 threads) can be coalesced into 1 transaction if the addresses fall within the same **128B cache line**.
- Ideal case: 64 threads × 16B/thread (dwordx4) = 1024B, spanning 8 cache lines, resulting in 8 transactions.
- Worst case: addresses of 64 threads are scattered across 64 different cache lines → 64 transactions (8× bandwidth waste).

### Relationship Between Coalesced Access and BlockedLayout

**Key Rule**: The `order` of `BlockedLayout` determines which dimension is contiguous in memory, and **this dimension must match the actual storage layout of the tensor in HBM**.

```python
# Matrix A: shape [M, K], stride_am=K, stride_ak=1 → K dimension is contiguous in memory
# → blocked_a's order should make K dimension (dim 1) contiguous
blocked_a = gl.SwizzledSharedLayout(
    vec=8,                    # K dimension spt=8 → dwordx4 (bf16)
    perPhase=1,
    maxPhase=8,
    threads_per_warp=[16, 4],  # K dimension tpw=4, adjacent threads contiguous in K dimension
    warps_per_cta=[4, 1],
    order=[1, 0]               # dim 1 (K) contiguous → matches memory layout
)

# Matrix B: shape [K, N], stride_bk=N, stride_bn=1 → N dimension is contiguous in memory
# → blocked_b's order should make N dimension (dim 1) contiguous
blocked_b = gl.SwizzledSharedLayout(
    vec=8,                    # N dimension spt=8 → dwordx4 (bf16)
    perPhase=1,
    maxPhase=8,
    threads_per_warp=[16, 4],  # N dimension tpw=4, adjacent threads contiguous in N dimension
    warps_per_cta=[4, 1],
    order=[1, 0]               # dim 1 (N) contiguous → matches memory layout
)
```

**How to Determine Whether Memory Access is Coalesced**:

| Condition | Coalesced? | Description |
|------|---------|------|
| Innermost dimension of `order` = contiguous dimension of tensor in HBM | ✅ Coalesced | Adjacent threads access contiguous addresses |
| Innermost dimension of `order` ≠ contiguous dimension of tensor in HBM | ❌ Not coalesced | Adjacent threads access strided addresses |
| Stride of tensor in contiguous dimension = 1 | ✅ Coalesced | Elements are tightly packed in memory |
| Stride of tensor in contiguous dimension > 1 | ❌ Not coalesced | Gaps exist between elements |

### Diagnostic Methods

```bash
# 1. Extract assembly
# 2. Check load width
grep -c 'buffer_load_dword ' kernel.asm         # dwordx1 (32-bit) → Not ideal
grep -c 'buffer_load_dwordx2' kernel.asm        # dwordx2 (64-bit) → Acceptable
grep -c 'buffer_load_dwordx4' kernel.asm        # dwordx4 (128-bit) → Optimal

# 3. Check LDS width
grep -c 'ds_read_b32' kernel.asm                # 32-bit → Not ideal
grep -c 'ds_read_b64' kernel.asm                # 64-bit → Acceptable
grep -c 'ds_read_b128' kernel.asm               # 128-bit → Optimal

# 4. Store
grep -c 'ds_write_b32' kernel.asm
grep -c 'ds_write_b64' kernel.asm
grep -c 'ds_write_b128' kernel.asm

# 5. Check coalesced access (need to analyze with kernel source code)
# Confirm blocked layout's order matches tensor's memory layout
```

### Relationship Between Instruction Width and Layout

The instruction width is determined by the value of `size_per_thread` of `BlockedLayout` in the contiguous dimension:

```
Bytes per thread = size_per_thread[contiguous_dim] × element_size_bytes
Instruction width = min(Bytes per thread, 16)  # Max dwordx4 = 16 bytes
```

| Data Type | element_size | size_per_thread required to achieve dwordx4 |
|---------|-------------|--------------------------------------|
| fp32 (4B) | 4 | 4 |
| bf16/fp16 (2B) | 2 | 8 |
| fp8/int8 (1B) | 1 | 16 |

### Fix Methods

#### Step 1: Confirm the Tensor's Memory Layout

```python
# Analyze tensor's stride to determine which dimension is contiguous in memory
# Dimension with smallest stride = contiguous dimension in memory
# For example: stride_am=K, stride_ak=1 → K dimension is contiguous (stride=1)
# For example: stride_bk=N, stride_bn=1 → N dimension is contiguous (stride=1)
```

#### Step 2: Set BlockedLayout's Order to Match the Memory Layout

```python
# First element of order = innermost (contiguous) dimension
# If K dimension (dim 1) is contiguous in memory → order=[1, 0]
# If M dimension (dim 0) is contiguous in memory → order=[0, 1]
```

#### Step 3: Maximize size_per_thread in the Contiguous Dimension

```python
# ❌ Non-coalesced + narrow load
# Matrix A: stride_am=K, stride_ak=1 (K dimension contiguous)
# But order=[0, 1] makes M dimension contiguous → adjacent threads expand in M dimension → strided access
blocked_a_wrong = gl.SwizzledSharedLayout(
    vec=2,                    # M dimension spt=2, but M dimension stride=K → non-contiguous
    perPhase=1,
    maxPhase=8,
    threads_per_warp=[16, 4],
    warps_per_cta=[4, 1],
    order=[0, 1]               # ❌ M dimension contiguous, but K dimension is contiguous in memory
)

# ✅ Coalesced + wide load
# order=[1, 0] makes K dimension contiguous → matches memory layout → coalesced access
blocked_a_correct = gl.SwizzledSharedLayout(
    vec=8,                    # K dimension spt=8 → bf16 × 8 = 16B → dwordx4
    threads_per_warp=[16, 4],  # K dimension tpw=4, adjacent threads expand in K dimension
    warps_per_cta=[4, 1],
    order=[1, 0]               # ✅ K dimension contiguous → matches memory layout
)
```

**Constraint**: The product of `size_per_thread × threads_per_warp × warps_per_cta` must equal the size of each dimension of the block. When adjusting size_per_thread, threads_per_warp must be adjusted accordingly.

#### Step 4: Re-extract Layout from TTGIR (Optional)

If manual adjustments cause functional errors, re-extract the original layout using `extract_ttgir.py` to confirm whether the layout in TTGIR is inherently narrow or non-coalesced.

### ⚠️ Notes

- **Coalesced memory access takes precedence over instruction width** — a coalesced `dwordx2` is more efficient than a non-coalesced `dwordx4`
- **Do not** blindly increase size_per_thread — the layout semantics must remain correct, and the order must match the memory layout
- After modifying the load layout, downstream layouts such as SliceLayout, DotOperandLayout, etc. may also need adjustment
- In some scenarios, the tensor's contiguous dimension conflicts with the layout required for computation (e.g., K-contiguous vs N-contiguous for the B matrix), requiring a trade-off between coalesced memory access and LDS bank conflicts (see §3.2 and Appendix D Pitfall 21)
- Verify accuracy before confirming the changes

## 3.2 Check Swizzle for Bank Conflicts

### Objective

Eliminate LDS (Shared Memory) bank conflicts to reduce stall cycles in `ds_read`/`ds_write`.

### CDNA3 LDS Bank Configuration

- 32 banks, 4-byte granularity per bank
- Bank calculation: `bank = (byte_address / 4) % 32`
- Multiple threads in the same wavefront accessing different addresses in the same bank → bank conflict → serialization

### Diagnostic Methods

```bash
# 1. Hardware counters
rocprofv3 --pmc SQ_LDS_BANK_CONFLICT -d ./pmc -- python <kernel.py>

# 2. Check ds_read/ds_write Stall in ATT
grep 'ds_read\|ds_write' stats_*.csv | sort -t',' -k6 -nr

# 3. Gluon API check
# Add in kernel code (for debugging):
conflicts = gl.bank_conflicts(shared_layout)
```

### SwizzledSharedLayout Parameter Reference

```python
gl.SwizzledSharedLayout(vec, perPhase, maxPhase, order=[1, 0])
```

| Parameter | Meaning | Impact on Bank Conflicts |
|------|------|-------------------|
| `vec` | Consecutive vector width (unit: number of elements) | Larger → more elements accessed consecutively at a time |
| `perPhase` | Number of rows per phase | Controls the XOR swizzle period |
| `maxPhase` | Maximum number of phases | Controls the XOR swizzle range |
| `order` | Dimension order | `[1, 0]` = column-major (commonly used for MFMA) |

### XOR Swizzle Principle

Swizzling eliminates bank conflicts by applying an XOR transformation to the column address:
```
swizzled_col = (col / vec) XOR ((row / perPhase) % maxPhase) * vec + (col % vec)
```

### Fixes

| Scenario | Adjustment Direction |
|------|---------|
| MFMA read bank conflicts | Increase `maxPhase` to ensure accesses from different phases are spread across different banks |
| bf16 data bank conflicts | `vec=4` (4 × 2B = 8B), `perPhase=1`, `maxPhase=16` typically conflict-free |
| fp32 data bank conflicts | `vec=2` (2 × 4B = 8B), `perPhase=1`, `maxPhase=16` |

### ⚠️ Notes

- After modifying swizzle parameters, the corresponding `DotOperandLayout.k_width` may also need adjustment
- Zero bank conflicts is the ideal target, but a small number of 2-way conflicts has minimal impact
- Prioritize eliminating bank conflicts on the MFMA read path (high-frequency execution)

---

## 3.3 Eliminate ds_bpermute Instructions

### Objective

Reduce or eliminate `ds_bpermute_b32` instructions, which are high-latency cross-lane data exchange operations.

### Background

`ds_bpermute` uses LDS hardware but does not write to LDS storage. It is used Yaml arbitrary lane-to-lane data reads within a warp. Latency is approximately 50 cycles.

In Gluon/Triton, `ds_bpermute` is typically introduced by the following operations:
- Explicit `gl.convert_layout(tensor, target_layout)` calls
- Implicit layout conversions (automatically inserted by the compiler when an operation requires a different layout)

### Diagnostic Methods

```bash
# 1. Check ds_bpermute count in assembly
grep -c 'ds_bpermute' kernel.asm

# 2. Check ds_bpermute latency contribution in ATT
grep 'ds_bpermute' stats_*.csv | awk -F',' '{sum+=$5} END {print "Total latency:", sum}'

# 3. Search for convert_layout calls in Gluon source code
grep 'convert_layout' kernel.py
```

### Fixes

#### Method A: Eliminate Unnecessary convert_layout

```python
# ❌ convert_layout immediately after load
data = gl.amd.cdna3.buffer_load(ptr=base, offsets=offs, mask=mask, other=0.0)  # blocked layout
data_mma = gl.convert_layout(data, mma)  # Triggers ds_bpermute

# ✅ If data will eventually enter shared memory and then do MFMA, load directly with blocked layout
# Then naturally convert layout through shared memory path
data = gl.amd.cdna3.buffer_load(ptr=base, offsets=offs, mask=mask, other=0.0)
data_smem = gl.allocate_shared_memory(data.dtype, shape, shared_layout, value=data)
data_dot = data_smem.load(layout=dot_op)  # Through smem path, usually won't generate ds_bpermute
```

#### Method B: Unify Upstream and Downstream Layouts

```python
# ❌ Use mma slice layout for load, then store in blocked layout smem
row_idx = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, mma))      # mma slice
data = gl.amd.cdna3.buffer_load(...)                                  # mma layout data
# If blocked layout is needed later, compiler will insert convert_layout

# ✅ Use blocked slice layout for load, consistent with storage layout
row_idx = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, blocked))   # blocked slice
data = gl.amd.cdna3.buffer_load(...)                                  # blocked layout data
data_smem = gl.allocate_shared_memory(..., value=data)                # No conversion needed
```

#### Method C: Accept ds_bpermute (When Unavoidable)

Certain layout conversions are algorithmically necessary (e.g., scattering back to different distributions after reduction). In these cases:
- Ensure that `ds_bpermute` is not inside a high-frequency loop
- If inside a loop, consider whether convert_layout can be hoisted outside the loop

### ⚠️ Notes

- Not all `ds_bpermute` can be eliminated — some are algorithmically necessary
- Prioritize eliminating `ds_bpermute` **inside loop bodies** (high-frequency execution)
- `ds_bpermute` outside loops have minimal performance impact and can be ignored

---

## 3.4 Eliminating Scratch Operations (Register Spilling)

### Objective

Eliminate `buffer_load`/`buffer_store` or `scratch_load`/`scratch_store` targeting scratch space — these indicate VGPR spilling to video memory.

### Background

CDNA3 VGPR hierarchy:
1. **VGPR** (fastest) — up to 512 32-bit registers per wave
2. **AGPR** (fast) — accumulator registers, overflow transfers go here first when VGPRs are exhausted
3. **Scratch** (extremely slow) — scratch space in video memory, equivalent to global memory latency

Spilling path: VGPR insufficient → spill to AGPR (`v_accvgpr_read/write`) → AGPR also insufficient → spill to scratch (`scratch_store/load`)

### Diagnostic Methods

```bash
# 1. Check scratch operations in assembly
grep -c 'scratch_load\|scratch_store' kernel.asm

# 2. Check VGPR usage count and spill count
grep 'vgpr_count\|vgpr_spill_count\|sgpr_spill_count' kernel.asm
# .vgpr_count:       256
# .vgpr_spill_count: 0     ← Must be 0!

# 3. Check AGPR to VGPR transfers (warning sign)
grep -c 'v_accvgpr_read_b32\|v_accvgpr_write_b32' kernel.asm

# 4. Hardware counters
rocprofv3 --pmc SPI_RA_VGPR_SGPR_FULL_CSN -d ./pmc -- python <kernel.py>
```

### Remediation Methods

#### Method A: Reduce Block Size

```python
# ❌ Block size too large, not enough VGPRs
BLOCK_M, BLOCK_N, BLOCK_K = 128, 256, 64  # Accumulator 128×256 = 32768 elements

# ✅ Reduce block size
BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 64  # Accumulator 128×128 = 16384 elements
```

#### Method B: Narrow Variable Live Ranges

```python
# ❌ All variables simultaneously active
a = gl.amd.cdna3.buffer_load(...)  # a becomes active
b = gl.amd.cdna3.buffer_load(...)  # b becomes active
c = gl.amd.cdna3.buffer_load(...)  # c becomes active
# Here a, b, c all active → high VGPR pressure
result_a = process(a)
result_b = process(b)
result_c = process(c)

# ✅ Stagger active ranges
a = gl.amd.cdna3.buffer_load(...)
result_a = process(a)  # a dies after use
b = gl.amd.cdna3.buffer_load(...)
result_b = process(b)  # b dies after use
c = gl.amd.cdna3.buffer_load(...)
result_c = process(c)
```

#### Method C: Reduce Pipeline Depth

```python
# ❌ Deep pipeline = more simultaneously active buffers
smem = gl.allocate_shared_memory(dtype, [3, M, K], layout)  # depth=3, 3 copies of data simultaneously active

# ✅ Shallow pipeline
smem = gl.allocate_shared_memory(dtype, [1, M, K], layout)  # depth=1, only 1 copy
```

#### Method D: Merge Temporary Variables

```python
# ❌ Unnecessary intermediate variables
tmp1 = a * b
tmp2 = tmp1 + c
tmp3 = gl.exp(tmp2)
result = tmp3 * d

# ✅ Chained computation (compiler easier to reuse registers)
result = gl.exp(a * b + c) * d
```

### VGPR Budget Estimation

| Content | VGPR Consumption (Approx.) |
|------|-----------------|
| MFMA Accumulator (M×N fp32) | M × N / 64 VGPRs |
| 2D Tensor (M×N bf16) | M × N / 128 VGPRs |
| 1D Vector (M bf16) | M / 128 VGPRs |
| Scalar/Pointer | 1–2 VGPRs |

Example: 128×128 fp32 accumulator = 128×128/64 = 256 VGPRs → all VGPRs already exhausted!

### ⚠️ Notes

- `vgpr_spill_count = 0` is a hard target
- A small amount of AGPR ↔ VGPR movement is normal (MFMA requires AGPR), but heavy movement indicates proximity to the spilling boundary
- Reducing block size is the most effective approach, but it decreases data reuse → trade-off required

---

## 3.5 Optimizing Memory Access Stalls (Compute-Memory Overlap)

### Objective

Ensure memory load latency is effectively hidden by compute operations, avoiding pipeline stalls.

### Latency Reference

| Memory Level | Approximate Latency (Cycles) |
|---------|---------------|
| LDS | ~20 |
| L1 Cache | ~20 |
| L2 Cache | ~80 |
| MALL (LLC) | ~200 |
| HBM | ~400+ |

### Diagnostic Methods

```bash
# 1. Check buffer_load Stall and Idle
grep 'buffer_load' stats_*.csv | sort -t',' -k6 -nr | head -10

# 2. Check MFMA Idle (waiting for data ready)
grep 'v_mfma' stats_*.csv | sort -t',' -k7 -nr | head -10

# 3. Overall stall distribution
awk -F',' 'NR>1 {total_lat+=$5; total_stall+=$6; total_idle+=$7}
    END {printf "Latency: %d, Stall: %d (%.1f%%), Idle: %d (%.1f%%)\n",
         total_lat, total_stall, total_stall/total_lat*100,
         total_idle, total_idle/total_lat*100}' stats_*.csv
```

### Fix Methods

#### Method A: Implement Software Pipelining

If the kernel has a main loop and has not yet implemented pipelining:

See `converter/amd/cdna3/pipeline.md` for details.

Core pattern:
```python
# Prologue: Prefetch first batch
data_0 = gl.amd.cdna3.buffer_load(...)
smem.index(0).store(data_0)

# Main loop: compute current + prefetch next batch
for i in range(NT - 1):
    next_data = gl.amd.cdna3.buffer_load(...)     # Next batch load (async)
    dot = smem.index(0).load(layout=dot_op)       # Consume current data
    acc = gl.amd.cdna3.mfma(dot, other, acc)      # Compute (covers load latency)
    smem.index(0).store(next_data)                # Store prefetch result

# Epilogue: Process last batch
dot = smem.index(0).load(layout=dot_op)
acc = gl.amd.cdna3.mfma(dot, other, acc)
```

#### Method B: Reorder Code Sequence

```python
# ❌ Load and compute serial
data_a = gl.amd.cdna3.buffer_load(...)  # load A
result_a = compute(data_a)               # compute A (wait for A to arrive)
data_b = gl.amd.cdna3.buffer_load(...)  # load B
result_b = compute(data_b)               # compute B (wait for B to arrive)

# ✅ Load issued early, compute delayed consumption
data_a = gl.amd.cdna3.buffer_load(...)  # load A (issued)
data_b = gl.amd.cdna3.buffer_load(...)  # load B (issued, overlaps with A)
result_a = compute(data_a)               # compute A (A may have arrived)
result_b = compute(data_b)               # compute B (B may have arrived)
```

#### Method C: Increase Occupancy

More concurrent waves → the GPU switches to another wave for execution while one wave is waiting for memory.

- Reduce VGPR usage → allow more waves to run concurrently
- Reduce LDS usage → allow more workgroups to run concurrently

| VGPR/wave | Max waves/SIMD (CDNA3) | Occupancy |
|-----------|----------------------|-----------|
| ≤ 64 | 8 | 100% |
| ≤ 96 | 5 | 62.5% |
| ≤ 128 | 4 | 50% |
| ≤ 256 | 2 | 25% |
| ≤ 512 | 1 | 12.5% |

### ⚠️ Notes

- Pipelining is the most effective means of overlap, but it increases VGPR pressure (holding multiple copies of data simultaneously)
- If Step 3.4 already has register pressure, pipelining may actually worsen the situation → a trade-off between the two must be made
- Occupancy and VGPR usage are in a trade-off relationship

---

## 3.6 warp_pipeline_stage Full-Stage Packing (GEMM-Specific Optimization)

> ⚠️ **This optimization applies only to large-tile GEMM**, not to Flash Attention or small-matrix GEMM.
> Detailed content is covered in `warp_pipeline_stage.md`.

**Applicable Conditions**:
- Large-tile GEMM (grid_blocks ≥ num_CUs, sufficient loop iterations)
- Not applicable to Flash Attention (complex control flow between MFMAs)
- Not applicable to small-matrix GEMM (too few loop iterations, overhead > benefit)

**Key Points** (see `warp_pipeline_stage.md` for details):
1. All LDS operations and computations must be packed — ds_read uses `"prep"`, MFMA uses `"compute"`, ds_write uses `"prep"`
2. buffer_load is placed between pipeline stages and not packed inside any stage
3. buffer_load must not use the `other=0.0` parameter
4. Do not manually write `gl.barrier()`
5. Insert ds_write between MFMAs to achieve compute↔store overlap

**Expected Benefit**: 161 → 204.7 TFLOPS (+27%, surpassing Triton)

---

## Verification Checklist

Must perform after each optimization step:

```bash
# 1. Accuracy verification
python <validation-command> \
    optimized_kernel.py reference_kernel.py --var-name result_gold

# 2. Performance verification
python tools/measure_kernel_time.py optimized_kernel.py \
    --wrapper-name <wrapper> --setup-name <setup>

# 3. Compare before and after optimization
echo "Before optimization: X.XXX ms → After optimization: Y.YYY ms (improved by Z%)"
```

If precision verification fails → **roll back immediately**, do not proceed with subsequent optimizations.
If performance does not improve or regresses → roll back, record the reason, continue to the next step.

---

## Appendix A: Optimization Decision Quick Reference

### By Bottleneck Type

| Bottleneck Type | Tier 1 Optimization | Tier 2 Optimization | Tier 3 Optimization |
|---------|---------|---------|----------|
| Compute Bound | 3.3 Eliminate scratch/spill | 3.5 compute-memory overlap | 3.2 bank conflicts |
| Memory Bound | 3.0+3.1 coalesced access+wide load | 3.5 compute-memory overlap | 3.2 bank conflicts |
| Insufficient CU Utilization | Tile size tuning | BLOCK_K tuning | Micro-optimizations |

### By Kernel Type

| Kernel Type | Required | Optional | Not Applicable | Specific Docs |
|------------|------|------|--------|----------|
| Large-tile GEMM | 3.0, 3.1, 3.3, 3.5, 3.6 | 3.2, 3.4 | — | `warp_pipeline_stage.md` |
| Small-matrix GEMM | 3.0, tile tuning | 3.1, 3.3 | 3.5, 3.6 | Small GEMM topic notes |
| Flash Attention | 3.0, 3.1, 3.3 | 3.2 | 3.6 (N/A) | Fused attention topic |

## Related

- **Prerequisites**: [CDNA3 Hardware Specifications](../../hardware-specs/mi300x.md) — Required for Roofline analysis
- **Cross-Architecture Reference**: [CDNA4 ISA Optimization Checklist](../gfx950/common_optimizations.md) | [Hopper ISA Optimization Checklist](../../../nvidia/hopper/gluon/common_optimizations.md)
- **ISA Reference**: [CDNA3 ISA Instruction Patterns](isa_patterns.md) — Detailed descriptions of instructions referenced in this document
- **🔴 Conflict Note**: This document considers manual ISA optimization to be effective (+1-4%), but [Hopper pitfalls #1](../../../nvidia/hopper/gluon/pitfalls.md) found that manual refactoring on sm_90 is almost always a negative optimization
