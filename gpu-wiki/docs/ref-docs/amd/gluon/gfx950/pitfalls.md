# CDNA4 (gfx950) Practical Pitfalls

**Last Updated**: 2026-03-28
**Source**: `submission_gluon_v2-v7.py` measured optimization experience + mixed-mla report.md

---

## Pitfall Index

| # | Title | Mode Tags | Scope | Severity |
|---|------|---------|---------|--------|
| 1 | Passing E8M0 scale directly causes precision errors | [MLA], [FP4] | mfma_scaled | 🔴 Fatal |
| 2 | mask/other inside loop cause v_cndmask explosion | [MLA], [Attention] | All kernels with mask | 🔴 Fatal |
| 3 | Excessive Split-K partitioning causes stage2 overhead | [MLA], [GEMM] | split-K kernel | 🟠 High |
| 4 | Scales narrow load instruction count explosion | [MLA], [FP4] | mxfp4 QK | 🟡 Medium |
| 5 | V fp8 regresses in small kv scenarios | [MLA] | decode attention | 🟡 Medium |
| 6 | Uneven XCD load leads to low bandwidth utilization | [MLA], [GEMM] | Large grid kernel | 🟡 Medium |
| 7 | warp_pipeline_stage causes negative optimization in attention | [Attention] | online softmax | 🟠 High |
| 8 | async_copy not activating ping-pong scheduling | [GEMM] | pipeline kernel | 🟡 Medium |
| 9 | order does not match memory layout | [General] | All kernels | 🔴 Fatal |
| 10 | gfx950 in-thread transpose disabled causes layout errors | [General] | MFMA | 🔴 Fatal |

---

## Pitfall 1: Passing E8M0 Scale Directly Causes Precision Errors

**Mode**: [MLA], [FP4]
**Severity**: 🔴 Fatal (completely incorrect precision)

### Symptoms

```python
# ❌ Wrong: direct int8 scale
q_pe_bf16 = gl.amd.cdna4.scaled_upcast_fp4(q_pe_fp4_dot, q_pe_scales_raw, gl.bfloat16, 1)
```

Compilation succeeds, but runtime outputs NaN or completely wrong values.

### Cause

CDNA4's `scaled_upcast_fp4` does not support `useShiftedScale=true`'s i8 path. The unbiased exponent in E8M0 format must be manually shifted into the BF16 exponent field.

### Fix

```python
# ✅ Correct: e8m0 i8 -> bf16 shifted scale
q_pe_scales_i16 = gl.cast(q_pe_scales_reshaped, gl.int16)      # zero-extend i8→i16
q_pe_scales_shifted = q_pe_scales_i16 << 7                      # shl 7: place exp in bf16 exp field
q_pe_scales_bf16 = gl.cast(q_pe_scales_shifted, gl.bfloat16, bitcast=True)
q_pe_bf16 = gl.amd.cdna4.scaled_upcast_fp4(q_pe_fp4_dot, q_pe_scale_dot, gl.bfloat16, 1)
```

### Verification Method

Compare against the Triton reference implementation output; max diff < 1e-3 passes.

---

## Pitfall 2: mask/other Inside Loop Causes v_cndmask Explosion

**Mode**: [MLA], [Attention]
**Severity**: 🔴 Fatal (~10% performance loss)

### Symptoms

ASM profile shows ~174 `v_cndmask_b32` instructions, with extremely high stall cycles.

### Cause

In the original code, all `buffer_load` carry `mask` and `other=0`, and masks are added even when full blocks won't go out of bounds. The compiler generates conditional select instructions for each load.

### Fix

Split the loop into full blocks (no mask) + tail block (with mask):

```python
split_len = split_end - split_start
num_full_blocks = split_len // BLOCK_N
full_end = split_start + num_full_blocks * BLOCK_N

# FULL BLOCKS LOOP — no mask, no other!
for start_n in range(split_start, full_end, BLOCK_N):
 k_nope_fp4 = gl.amd.cdna4.buffer_load(ptr=KV_fp4, offsets=k_nope_fp4_offs) # none mask/other
    # ... compute ...

# TAIL BLOCK - requires mask
if split_len % BLOCK_N != 0:
    start_n = full_end
    # ... with mask/other ...
```

### Effect

Eliminated ~174 `v_cndmask` instructions; the hot path is completely branch-free.

---

## Pitfall 3: Excessive Split-K Partitioning Causes Stage2 Overhead

**Mode**: [MLA], [GEMM]
**Severity**: 🟠 High (+10-20% in small kv scenarios)

### Symptoms

When bs=4, kv=1024 MAD splitted into 16 splits, each split has only 1 iteration, and stage2 reduce accounts for > 30% of total time.

### Cause

The Split-K strategy does not consider the minimum iterations per split constraint, and excessive partitioning causes stage2 reduce overhead to dominate.

### Fix

Introduce a `min_iters_per_split=2` constraint:

```python
min_iters_per_split = 2
max_splits_by_iters = max(1, num_kv_blocks // min_iters_per_split)
desired_splits = max(1, target_total_blocks // batch_parallelism)
raw = min(desired_splits, max_splits_by_iters, num_kv_blocks)
```### Effect

2-6% improvement in 6/8 scenarios, avoiding stage2 reduce overhead dominating runtime.

---

## Pitfall 4: Scales narrow load instruction bloat

**Pattern**: [MLA], [FP4]
**Severity**: 🟡 Medium (performance loss ~4%)

### Symptom

K nope scales uses `size_per_thread=[1,1]` (1B/thread), generating `buffer_load_ubyte` instructions.

### Cause

The layout design did not account for memory access coalescing; the contiguous dimension's `size_per_thread` is too small.

### Fix

Changed to `size_per_thread=[1,4]` (4B/thread), generating `buffer_load_dword`:

```python
s1_blocked2_wide: gl.constexpr = gl.BlockedLayout(
    size_per_thread=[1, 4], threads_per_warp=[4, 16], warps_per_cta=[4, 1], order=[1, 0])
```

### Effect

The number of K nope scales load instructions inside the hot loop is reduced by 4x.

---

## Pitfall 5: V fp8 regresses in small kv scenarios

**Pattern**: [MLA]
**Severity**: 🟡 Medium (small kv +4-14%, large kv -3-13%)

### Symptom

After V fp8 load optimization, performance regresses by +4% to +14% in small kv (≤1024) scenarios.

### Cause

In small kv scenarios, the data volume is small and bandwidth is not saturated; the fp8→bf16 cast overhead becomes a burden instead.

### Fix

Dynamically select the V format based on `max_kv_len`:

```python
def custom_kernel(data, config):
    max_kv_len = int((kv_indptr[1:] - kv_indptr[:-1]).max().item())

    if max_kv_len >= 4096:
        return _mla_decode_fp8(data, config)  # v7
    else:
        return _mla_decode_bf16(data, config)  # v6
```

### Effect

Optimal across all scenarios — use bf16 for small kv (avoiding cast overhead) and fp8 for large kv (bandwidth benefit).

---

## Pitfall 6: XCD load imbalance causing low bandwidth utilization

**Pattern**: [MLA], [GEMM]
**Severity**: 🟡 Medium (performance loss ~5%)

### Symptom

The MI355X has 8 XCDs, but HBM bandwidth utilization is only 60-70%.

### Cause

The hardware assigns consecutive PIDs to different XCDs in a round-robin fashion. Without reordering, some XCDs receive all heavy blocks while other XCDs receive all light blocks.

### Fix

Zigzag remap:

```python
NUM_XCDS = 8
pid_raw = gl.program_id(0)
wave = pid_raw // NUM_XCDS
pos_in_wave = pid_raw % NUM_XCDS
is_odd_wave = wave % 2
remapped_pos = tl.where(is_odd_wave, NUM_XCDS - 1 - pos_in_wave, pos_in_wave)
pid = wave * NUM_XCDS + remapped_pos
```

### Effect

Ensures adjacent blocks are distributed across different XCDs, improving HBM bandwidth utilization.

---

## Pitfall 7: `warp_pipeline_stage` negatively impacts attention

**Pattern**: [Attention]
**Severity**: 🟠 High (performance regression ~10%)

### Symptom

After applying `warp_pipeline_stage` full-stage pipelining to MLA decode attention, performance regresses rather than improves.

### Cause

The inner loop of MLA Decode contains complex online softmax control flow (updating e_max, e_sum, rescaling accumulator), which cannot be packed into pure prep/compute stages. Forcing packaging cuts off the compiler's optimization space.

### Fix

Switch to software pipelining mode:
- Full/Tail block splitting (OPT-3)
- Scalar base address precomputation (OPT-1, 2)
- Dynamic V format selection

### Applicable Scenarios

`warp_pipeline_stage` applies only to Standard GEMM (pure matrix multiplication with no cross-iteration dependencies).

---

## Pitfall 8: `async_copy` does not activate ping-pong scheduling

**Pattern**: [GEMM]
**Severity**: 🟡 Medium (performance loss ~10-20%)

### Symptom

When using the conventional `buffer_load + smem.store` two-step transfer, benchmarks report performance below expectations.

### Cause

CDNA4's ping-pong scheduling optimization is only activated by the compiler when using `async_copy.buffer_load_to_shared`. Regular `buffer_load` does not trigger this optimization.

### Fix

```python
from triton.experimental.gluon.language.amd.cdna4 import async_copy

# ✅ Correct: async_copy DMA
async_copy.buffer_load_to_shared(smem.index(slot), ptr, offsets, mask=mask)
async_copy.commit_group()
async_copy.wait_group(num_outstanding=0)
data = async_copy.load_shared_relaxed(smem.index(slot), layout=dot_op)
```

### Note

MLA Decode is not suitable for this optimization due to complex control flow, but Standard GEMM must use it.

---

## Pitfall 9: order does not match memory layout

**Pattern**: [General]
**Severity**: 🔴 Fatal (multiple times performance loss)

### Symptom

Kernel performance is far below expectations (>2x gap) and all optimizations are ineffective.

### Cause

The `order` of `BlockedLayout` does not match the actual memory layout of the tensor in HBM, resulting in uncoalesced memory access.

### Fix

Verify the tensor's stride and make the first element of `order` correspond to the stride=1 dimension:

```python
# B[K, N]: stride_bk=N, stride_bn=1 -> N (dim 1) contiguous -> order=[1, 0]
blocked_b: gl.constexpr = gl.BlockedLayout(
 size_per_thread=[*, 8], # N contiguous
    threads_per_warp=[...],
    warps_per_cta=[...],
 order=[1, 0]) # dim 1
```### Diagnosis

Check whether the assembly has a large number of `buffer_load_ubyte` / `buffer_load_ushort` rather than `dwordx4`.

---

## Pitfall 10: gfx950 in-thread transpose disabled causes layout errors

**Pattern**: [General]
**Severity**: 🔴 Critical (MFMA input error)

### Symptoms

MFMA produces incorrect output or compilation fails.

### Cause

The Triton compiler disables the in-thread transpose optimization for gfx950. CDNA3 can rearrange register data via in-thread transpose, but CDNA4 does not support it.

### Fix

Ensure data already has the correct layout for MFMA consumption **before** being stored to shared memory:

```python
# ✅ Correct: DotOperandLayout
dot_op0 = gl.DotOperandLayout(operand_index=0, parent=mma, k_width=4)
a_dot = a_smem.load(dot_op0) # directCorrect layout load
```

Do not rely on the compiler to automatically transpose register data.

---

## Quick Checklist

Before submitting optimization results, go through the following checklist item by item:

| # | Check Item | Status |
|---|--------|------|
| 1 | E8M0 scale has been converted to bf16 shifted scale | ⬚ |
| 2 | Full/Tail block splitting is complete | ⬚ |
| 3 | Split-K min_iters_per_split ≥ 2 | ⬚ |
| 4 | Scales load width ≥ dword | ⬚ |
| 5 | V format dynamically selected based on kv length | ⬚ |
| 6 | XCD zigzag remap has been implemented | ⬚ |
| 7 | Attention does not use warp_pipeline_stage | ⬚ |
| 8 | GEMM uses async_copy to enable ping-pong | ⬚ |
| 9 | order matches memory layout | ⬚ |
| 10 | gfx950 does not use in-thread transpose | ⬚ |

## Related Documents

- **Cross-architecture Comparison**: [Hopper Pitfalls](../../../nvidia/gluon/sm90/pitfalls.md) — 11 Hopper-specific pitfalls
- **Referenced By**: [matmul](matmul.md) references #1-5 | [fused_attention](../../../../kernel-opt/amd/gluon/gfx950/fused_attention.md) references #2,7 | [mla_decode](mla_decode.md) references #2-6
- **🔴 #7 Conflicts with CDNA3**: [CDNA3 WPS](../gfx942/warp_pipeline_stage.md) considers WPS to yield +27%, while #7 in this document finds it is a negative optimization for attention
- **#10 CDNA4-specific Regression**: in-thread transpose is disabled on gfx950, but works normally on CDNA3 (gfx942)
