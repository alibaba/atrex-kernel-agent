# MLA Decode Attention Optimization Guide

> This document focuses on optimizations specific to the **MLA (Multi-Head Latent Attention) Decode** kernel.

**Last updated**: 2026-06-30

> For the general ISA optimization checklist, see `common_optimizations.md`.

---

## ⚠️ Lossy Optimization Notice

**OPT-9 (V FP8 Load)** is a **lossy optimization** (reduced precision):
- V is reduced from bf16 to fp8, introducing quantization error
- Before implementing this optimization, **you must explicitly ask the user whether they accept precision loss**
- Recommended approach: dynamically select based on `max_kv_len` (use fp8 for large KV, bf16 for small KV), and only apply when the user confirms they need performance improvement

---

## Pattern Characteristics

| Feature | Description |
|------|------|
| **Core Computation** | QK matmul → online softmax → PV matmul |
| **Data Format** | Q/K: mxfp4 (e2m1) + e8m0 scales; V: bf16 or fp8 |
| **Main Loop Structure** | Iterate along KV seq len, load K/V tiles each step, perform MFMA scaled QK + softmax + MFMA PV |
| **Key Bottleneck** | **V bf16 bandwidth accounts for ~77% of total bandwidth** (absolute bottleneck) |
| **ISA Signatures** | `v_mfma_scale_bf16` (QK), `v_mfma_bf16` (PV), `buffer_load_dwordx4`, `v_cndmask_b32` (to be eliminated) |

**Identification Conditions**:
- Contains `mfma_scaled` (mxfp4 QK) + `mfma` (bf16 PV)
- Online softmax: updates `e_max`, `e_sum`, rescales accumulator at each step
- Paged KV cache: indexes variable-length sequences via `kv_indptr`

---

## Bottleneck Analysis

### Roofline Analysis (bs=256, kv=8192, BLOCK_N=64)

Data volume per iteration:

| Data | Format | Size | Proportion |
|------|------|------|------|
| K nope fp4 | mxfp4 | 16 KB | 19% |
| K PE fp4 | mxfp4 | 2 KB | 2% |
| **V bf16** | **bf16** | **64 KB** | **77%** |
| K scales | e8m0 | 1.1 KB | 1% |
| **Total** | | **~83 KB** | **100%** |

**Conclusion**: V bf16 is the absolute bandwidth bottleneck. Any optimization that reduces V bandwidth yields significant benefits.

### Tile AI vs Ridge Point

```
Tile_FLOPs = 2 × BLOCK_H × BLOCK_N × (BLOCK_C + BLOCK_R)
           = 2 × 16 × 64 × (512 + 64) = 1.18M
Tile_Bytes = 83 KB
Tile_AI    = 1.18M / 83KB ≈ 14 FLOPs/Byte
```

MI355X Ridge Point ≈ 245 BF16. **Tile AI << Ridge Point → Memory Bound**.

---

## Optimization Strategy Priority

| Priority | Optimization | Description | Expected Gain |
|--------|--------|------|---------|
| ⭐⭐⭐ | Advance V prefetch (OPT-12) | Overlap V[N+1] load with QK[N] MFMA | -6% ~ -13% |
| ⭐⭐⭐ | V fp8 load (OPT-9) | Reduce 50% of the 77% bandwidth | -3% ~ -13% (large KV) |
| ⭐⭐⭐ | Full/Tail split (OPT-3) | Eliminate hot-path v_cndmask | +5% ~ +10% |
| ⭐⭐ | Refined Split-K (OPT-7) | Avoid excessive splitting, reduce stage2 overhead | +2% ~ +6% |
| ⭐⭐ | Scales access merging (OPT-8) | dword → dwordx4 | +2% ~ +4% |
| ⭐ | XCD remap (OPT-6) | Load balancing | +2% ~ +5% |
| ⭐ | Scalar base address precomputation (OPT-1,2) | Reduce SALU in loop | +1% ~ +3% |
| — | warp_pipeline_stage | Not applicableasca (complex control flow) | N/A |

---

## Detailed Optimization Techniques

### OPT-3: Full/Tail Block Splitting (Eliminate Hot-Path Masking)

**Problem**: In the original code, all `buffer_load` carry `mask` and `other=0`, causing the compiler to generate a large number of `v_cndmask_b32` instructions (~174), which is the #1 bottleneck identified by ASM profiling.

**Solution**: Split the loop into full blocks (no mask) + tail block (with mask).

```python
split_len = split_end - split_start
num_full_blocks = split_len // BLOCK_N
full_end = split_start + num_full_blocks * BLOCK_N
has_tail = split_len % BLOCK_N != 0

# FULL BLOCKS LOOP — no mask, no other, no v_cndmask!
for start_n in range(split_start, full_end, BLOCK_N):
 k_nope_fp4 = gl.amd.cdna4.buffer_load(ptr=KV_fp4, offsets=k_nope_fp4_offs) # none mask/other
    # ... compute ...

# TAIL BLOCK - requires mask
if has_tail:
    start_n = full_end
    # ... with mask/other ...
```

**Effect**: Eliminates ~174 `v_cndmask` instructions, making the hot path completely branch-free.

---

### OPT-6: XCD-aware PID Remap (Load Balancing)

MI355X has 8 XCD × 32 CU/XCD = 256 CUs. Hardware assigns consecutive PIDs to different XCDs in a round-robin fashion. Without remapping, some XCDs may receive entirely heavy blocks while others receive entirely light blocks, resulting in severe load imbalance.```python
NUM_XCDS = 8
pid_raw = gl.program_id(0)
wave = pid_raw // NUM_XCDS
pos_in_wave = pid_raw % NUM_XCDS
is_odd_wave = wave % 2
remapped_pos = tl.where(is_odd_wave, NUM_XCDS - 1 - pos_in_wave, pos_in_wave)
pid = wave * NUM_XCDS + remapped_pos
pid = tl.minimum(pid, total_blocks - 1)
```

**Effect**: Ensures adjacent blocks are distributed across different XCDs, improving HBM bandwidth utilization.

---

### OPT-7: Refined Split-K Strategy

**Problem**: In v3, for small kv scenarios (kv=1024), split=16 means each split has only 1 iteration, and the stage2 reduce overhead becomes disproportionately large.

**Improvement**:
- Introduced the `min_iters_per_split=2` constraint to ensure at least 2 iterations per split
- Reduced target from 3x CU oversubscription to 2x (512 blocks)
- More conservative per-tier caps: `[4, 8, 16, 64]` (vs v3's `[8, 16, 32, 64]`)

**Split-K Configuration Comparison**:

| bs | kv | v3 splits | v3 iters/split | v6 splits | v6 iters/split | Change |
|---|---|---|---|---|---|---|
| 4 | 1024 | 16 | 1 | 8 | 2 | Reduced excessive splitting |
| 4 | 8192 | 64 | 2 | 64 | 2 | Unchanged |
| 32 | 1024 | 16 | 1 | 8 | 2 | Reduced excessive splitting |
| 256 | 8192 | 4 | 32 | 2 | 64 | Reduced splits |

**Effect**: 2%~6% improvement in 6/8 scenarios, avoiding stage2 reduce overhead dominating.

---

### OPT-8: K Nope Scales Memory Access Coalescing

**Problem**: K nope scales used `size_per_thread=[1,1]` (1B/thread), generating `buffer_load_ubyte` instructions.

**Improvement**: Changed to `size_per_thread=[1,4]` (4B/thread), generating `buffer_load_dword`.

```python
# ❌ : size_per_thread=[1,1] -> buffer_load_ubyte (1B/thread)
s1_blocked2: gl.constexpr = gl.BlockedLayout(
    size_per_thread=[1, 1], threads_per_warp=[4, 16], warps_per_cta=[4, 1], order=[1, 0])

# ✅ : size_per_thread=[1,4] -> buffer_load_dword (4B/thread)
s1_blocked2_wide: gl.constexpr = gl.BlockedLayout(
    size_per_thread=[1, 4], threads_per_warp=[4, 16], warps_per_cta=[4, 1], order=[1, 0])
```

**Effect**: Instruction count for K nope scales loads in the hot loop reduced by 4x.

---

### OPT-9: V FP8 Load (2x Bandwidth Reduction) ⭐

**Problem**: V bf16 is the largest bandwidth consumer (~77% of total bandwidth), 2B/elem per token.

**Improvement**: Load pre-quantized fp8 V cache (1B/elem) from `kv_data["fp8"]`, cast to bf16 before MFMA.

```python
# load fp8 (1B/elem) bf16 (2B/elem)
v_fp8 = gl.amd.cdna4.buffer_load(ptr=KV_fp8, offsets=v_offs)
v_bf16 = v_fp8.to(gl.bfloat16)

# PV dot , outputstage per-tensor scale
out_val = acc * kv_fp8_scale / e_sum_broadcast
```

**Dequant**: `v_bf16 = v_fp8.to(bf16) * kv_fp8_scale` (per-tensor scale).

**Effect**:

| bs | kv | v6 (μs) | v7 (μs) | v7 vs v6 |
|---|---|---|---|---|
| 4 | 1024 | 113 | 129 | +14.2% ❌ |
| 32 | 8192 | 181 | 175 | **-3.3%** ✅ |
| 64 | 8192 | 255 | 222 | **-12.9%** ✅ |
| 256 | 8192 | 565 | 520 | **-8.0%** ✅ |

**Key Findings**:
- **Large kv (≥8192)**: Bandwidth is a significant bottleneck; fp8 reduces bandwidth → clear gains (-3% ~ -13%)
- **Small kv (≤1024)**: Small data volume, bandwidth is unsaturated; fp8→bf16 cast overhead becomes a burden (+4% ~ +14%)

**Recommendation**: Dynamically select V bf16 (small kv) or V fp8 (large kv) based on `max_kv_len`.

---

### OPT-12: V Prefetch Moved Earlier + sched_barrier (V Load Overlaps with QK MFMA) ⭐⭐⭐

**Problem**: In the original code, V prefetch occurred after softmax and before PV MFMA. At that point, QK computation had already completed, and V load could not overlap with any computation, resulting in purely serial HBM data waiting.

**Improvement**: Moved the V[N+1] prefetch to the beginning of the loop (immediately after K prefetch), allowing V's HBM load to run in parallel with QK[N]'s MFMA computation.

```python
for start_n in range(split_start, full_end, BLOCK_N):
 # 1. Prefetch next K nope, K pe
    next_k_nope = gl.amd.cdna3.buffer_load(...)
    next_k_pe = gl.amd.cdna3.buffer_load(...)

    # 2. [OPT-12] Prefetch next V EARLY — overlap with QK MFMA
    next_v = gl.amd.cdna3.buffer_load(...)

    # 3. sched_barrier: prevent compiler reordering
    gl.amd.cdna3.sched_barrier(0x0)

    # 4. QK MFMA (V load is in flight, overlapping with this)
    qk = gl.amd.cdna4.mfma(q_nope_dot, k_nope_dot, ...)
    qk = gl.amd.cdna4.mfma(q_pe_dot, k_pe_dot, qk)

    # 5. Softmax
    ...

    # 6. PV MFMA (V data should be ready by now)
    v_dot = v_smem.load(dot_op1_mma1)
    acc = gl.amd.cdna4.mfma(p_dot, v_dot, acc)
```**Key Principles**:
- V[N+1]'s HBM load (~32KB/iter, ~4μs latency) overlaps with QK[N]'s MFMA computation (~2μs)
- `sched_barrier(0x0)` prevents the compiler from reordering the V load after softmax
- Benefits are more significant in large kv scenarios (more iterations, accumulated overlap effect)

**Results**:

| bs | kv | v3_fp8 (μs) | **v4_fp8 (μs)** | delta |
|----|----|-------------|-----------------|-------|
| 4 | 1024 | 18 | **17** | -6% |
| 4 | 8192 | 39 | **35** | -10% |
| 32 | 8192 | 79 | **72** | -9% |
| 64 | 8192 | 122 | **108** | -11% |
| 256 | 1024 | 88 | **82** | -7% |
| 256 | 8192 | 404 | **352** | -13% |

---

### OPT-1,2: Precompute Static Indices and Scalar Base Addresses

**OPT-1**: Hoist the `arange` indices, which are recomputed inside the loop, out of the loop.

**OPT-2**: Hoist scalar multiplications such as `kv_token_base * stride` out of the loop, performing only additive updates inside the loop.

```python
# loopcompute
k_nope_fp4_cur = kv_token_base * stride_kv_fp4_row
step_fp4 = BLOCK_N * stride_kv_fp4_row

# loopaddition
for start_n in range(split_start, full_end, BLOCK_N):
    k_nope_fp4_offs = gl.cast(k_nope_fp4_cur + offs_kn_cols_2d * stride_kv_fp4_row + ..., gl.int32)
    # ... compute ...
 k_nope_fp4_cur = k_nope_fp4_cur + step_fp4 # additionmultiplication
```

**Results**: Reduces SALU pressure inside the loop, +1% ~ +3%.

---

## Typical Optimization Paths

### Path 1: Large kv MLA Decode (kv ≥ 4096)

```
§3.0 order check -> OPT-3 full/tail split -> OPT-9 V fp8 -> OPT-12 V prefetch -> OPT-7 split-K -> OPT-8 scales merge
```

### Path 2: Small kv MLA Decode (kv ≤ 2048)

```
§3.0 order check -> OPT-3 full/tail split -> OPT-12 V prefetch -> OPT-7 split-K -> OPT-6 XCD remap
( OPT-9 V fp8, kv )
```

### Path 3: Dynamic Format Selection (Recommended)

```python
def custom_kernel(data, config):
    max_kv_len = int((kv_indptr[1:] - kv_indptr[:-1]).max().item())

    if max_kv_len >= 4096:
        return _mla_decode_fp8(data, config)  # v7
    else:
        return _mla_decode_bf16(data, config)  # v6
```

---

## CDNA4-Specific Notes

### E8M0 Scale Conversion

E8M0 format scales (int8) cannot be passed directly to `scaled_upcast_fp4`. They must be converted to bf16 shifted scales:

```python
q_pe_scales_i16 = gl.cast(q_pe_scales_reshaped, gl.int16)
q_pe_scales_shifted = q_pe_scales_i16 << 7
q_pe_scales_bf16 = gl.cast(q_pe_scales_shifted, gl.bfloat16, bitcast=True)
```

### Disable In-Thread Transpose on gfx950

Ensure data has the correct layout for MFMA consumption **before** storing it to shared memory.

---

## References

- General Optimization: `common_optimizations.md`
- GEMM Topic: `matmul.md`
- Lessons Learned: `pitfalls.md` (tagged [MLA], [Attention])


## Related

- [chunk-GDN (Gated Delta Net) Optimization Summary](chunk_gdn_lessons.md)
- [CDNA4 (gfx950) Generic ISA Optimization Checklist](common_optimizations.md)
- [Fused Attention Optimization Guide](fused_attention.md)
- [Standard GEMM / Batched GEMM Optimization Guide](matmul.md)
- [Gluon Kernel Performance Optimization Guide (AMD CDNA4)](optimization-guide.md)
- [Composable Kernel (CK) Architecture Overview](../../common/ck-architecture-overview.md)
- [CK Tile Quantized GEMM and MX Format](../../common/ck-quantization-mx.md)
- [Document Relationship Diagram](../../../RELATIONS.md)
