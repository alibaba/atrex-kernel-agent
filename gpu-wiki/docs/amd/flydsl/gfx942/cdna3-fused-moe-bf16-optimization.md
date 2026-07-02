# FlyDSL Fused MoE BF16 Step-by-Step Optimization (MI308X gfx942)

Applicability: backend: flydsl; hardware: amd; topic: reference


**Last updated**: 2026-06-30

This document records the complete process of implementing and progressively optimizing a Fused MoE kernel from scratch on the AMD MI308X (CDNA3, gfx942, 80 CUs) using FlyDSL. The model configuration is Mixtral-7B: hidden=4096, intermediate=14336, experts=8, top_k=2, BF16.

Final performance: **All token counts (1~4096) achieve better E2E performance than aiter (CK ASM)**, reaching 185 TFLOPS (89.9% peak) at tc=4096.

## Target Architecture

| Parameter | Value |
|------|-----|
| GPU | AMD Instinct MI308X |
| Architecture | CDNA3 (gfx942) |
| CU Count | 80 |
| BF16 Peak | 206 TFLOPS |
| HBM Bandwidth | 5.3 TB/s |
| LDS Capacity/CU | 64 KB |
| MFMA Instruction | mfma_f32_16x16x16bf16_1k |

## MoE Computation Logic

```
input: hidden_states [tc, 4096] BF16
      topk_weights  [tc, 2]    F32
      topk_ids      [tc, 2]    I32

Stage 1: sorted_x @ W1^T → gate [tc*2, 14336] + up [tc*2, 14336]
         activated = SiLU(gate) * up

Stage 2: activated @ W2^T → result [tc*2, 4096]
         output[token_id] += result * weight  (atomic f32 accumulation)
```

## Optimization Journey (V1 → V27)

### V1-V2: Baseline Implementation

- V1: Basic MFMA preshuffle GEMM + host-side SiLU + host-side weighted scatter
- V2: Optimized token sorting, removed unnecessary synchronization
- **Baseline Performance**: tc=4096 ~60 TFLOPS

### V4: Preshuffle Weight Layout

**Motivation**: Eliminate address computation overhead within GEMM

Pre-transform W1/W2 weights into a preshuffle layout, allowing `buffer_copy_gmem16_dwordx4` to load continuously in 128-bit chunks, avoiding runtime stride calculations.

```python
w1_ps = preshuffle_weight(w1_flat)  # [E*2*inter, hidden] → preshuffled
w2_ps = preshuffle_weight(w2_flat)  # [E*hidden, inter] → preshuffled
```

**Gain**: ~20% bandwidth utilization improvement

### V5: LDS-based Cooperative Epilogue (Stage 2)

**Motivation**: S2 output needs to go from registers → GMEM, but the MFMA C fragment layout is not suitable for coalesced stores

Use LDS as an intermediate buffer: write the MFMA C fragment to LDS, then read from LDS in a coalesced pattern and write to GMEM:

```
Thread C fragments → LDS (scatter by MFMA layout) → LDS (gather coalesced) → GMEM
```

**Note**: This approach was later replaced by the atomic accumulation approach in V17.

### V6: Expand tile_k to 128

**Motivation**: Reduce the number of K-dimension loop iterations, increasing MFMA instruction density

tile_k from 64 → 128: K loop iterations halved, executing more MFMA instructions per iteration. Requires more LDS spaceBR> to hold A/B tiles.

**Paired with tile_m=64**: Use the same tile configuration across all token counts.

### V7: Grid Dimension Swap (bn, num_m_tiles)

**Motivation**: Improve L2 cache locality for A-tile reuse

```python
# Before: grid = (num_m_tiles, bn, 1) -> different block different A
# After: grid = (bn, num_m_tiles, 1) -> block shared A tile
```

All N-blocks of the same M-tile are scheduled consecutively, reusing A data in L2.

**Gain**: tc=4096 ~8% improvement

### V8: Fine-Grained Instruction Scheduling Hints

**Motivation**: Profiling shows stalls between MFMA and memory instructions

Use FlyDSL scheduling hints to control instruction interleaving:

```python
sched_dsrd # LDS read
sched_mfma # MFMA
sched_vmem # VMEM load
sched_dswr # LDS write
```

Within the K loop, alternate scheduling by `vmem_load → dswr → dsrd → mfma` to ensure sufficient overlap between MFMA and memory operations.

### V10: tile_n 64→128

**Motivation**: Profiling shows insufficient MFMA instruction ratio; too few MFMA operations per A load

Double tile_n: execute 2x the number of MFMA instructions per A tile load (corresponding to 2 sub-tiles in the N direction).

```
Before: A load -> 4 MFMA (tile_n=64)
After: A load -> 8 MFMA (tile_n=128) ← MFMA/load
```

**Gain**: tc=4096 from ~110 → 152 TFLOPS (+38%)

### V13: Adaptive tile_m

**Motivation**: For small token counts, tile_m=64 causes excessive padding waste (e.g., tc=1 → 2 sorted tokens → pad to 64 → 97% wasted)

```python
TILE_M = 32 if token_count <= 128 else 64
```

Use tile_m=32 for small token counts to reduce padding, and tile_m=64 for large token counts to maintain high utilization. Requires compiling two separate kernel groups (cached via `@functools.lru_cache`).**Benefit**: tc=1~128 performance doubled

### V15: Cross-tile A0 LDS Prefetch

**Motivation**: The LDS read in the first iteration of the K loop must wait for the GMEM→LDS copy to complete (cold start)

Before the K loop starts, pre-launch the LDS→register transfer of the first A tile. Use a double-buffering pattern inside the loop:

```
Prefetch A[k=0] LDS→reg
for k in range(K_TILES):
    if k < K_TILES - 1:
        load A[k+1] GMEM→LDS
    MFMA(A[k], B[k])
    if k < K_TILES - 1:
        LDS→reg A[k+1]  (prefetch next)
```

Also place `s_barrier` after the scheduler hint to prevent barrier from interrupting the MFMA pipeline.

### V17: Fused Atomic Accumulation in S2 Epilogue (Key Optimization)

**Motivation**: After S2 there are 3 host-side operations taking ~1500us:
1. `result_sorted.float()` — bf16→f32 cast (~300us)
2. `result_f32 * sorted_weights` — weight multiplication (~400us)
3. `output_accum.index_add_(0, ...)` — scattered accumulation (~800us)

**Approach**: Directly complete weight multiply + atomic f32 accumulation in the S2 kernel epilogue:

```python
# S2 epilogue ( LDS cooperative store)
for ii in range_constexpr(WARP_M_STEPS):
    for kk in range_constexpr(WMMA_C_FRAG):
        token_id = buffer_load(sorted_tids_rsrc, m_global + kk, i32)
        weight = buffer_load(sorted_wts_rsrc, m_global + kk, f32)
    for jj in range_constexpr(WARP_N_STEPS):
        for kk in range_constexpr(WMMA_C_FRAG):
            val_f32 = vector.extract(c_frag, [kk])
            weighted = val_f32 * weight[kk]
            byte_off = (token_id[kk] * hidden_size + n_col) * 4
            rocdl.raw_ptr_buffer_atomic_fadd(weighted, out_rsrc, byte_off, 0, 0)
```

32 scalar f32 atomics per thread. Extremely low contention: when top_k=2, at most 2 sorted rows per token, and they are in different M-tiles.

**Benefit**: tc=4096 improved from ~152 → 181 TFLOPS (eliminating ~1500us host-side overhead)

### V18: In-kernel X Loading via Indirection

**Motivation**: Before Stage 1, `sorted_x = hidden_states[sorted_token_ids]` pre-gather is needed (~200us)

Pass `sorted_token_ids` into the S1 kernel, and load `hidden_states[sorted_token_ids[m]]` on-demand inside the kernel:

```python
def moe_stage1_kernel(arg_out, arg_x, arg_w1_ps, arg_expert_ids, arg_sorted_token_ids):
 # compute M row token index
    token_indices = [buffer_load(tid_rsrc, m_offset + i) for i in range(WARP_M_STEPS * 4)]
 # A load: token_index * hidden_size computeglobal
    for k_step:
        for m_frag:
            byte_off = token_indices[m_frag] * hidden_size * 2 + k_offset * 2
            a_data = buffer_load(x_rsrc, byte_off, ...)
```

**Benefit**: Eliminates ~200us pre-gather; the S1 kernel itself increases by ~50us due to indirect addressing

### V19-V21: Sort Optimization + Pre-allocated Buffers

- V19: Optimize `moe_sort_tokens`, reducing Python overhead
- V20: Pre-allocate `activated_buf` and `output_buf` to avoid allocation on every `fused_moe` call
- V21: Reorder output zeroing to after sort (overlapping with `.item()` sync)

### V27: Zero-Sync Fused Sort + Kernel Early-Exit Guard (Key Optimization)

**Motivation**: `moe_sort_tokens` returns `num_m_tiles` which requires `.item()` to move the GPU scalar to CPU (~30-50us HIP sync). At small tc, sort itself is also slow (multiple kernel launches).

**Migration/comparison note: fused sort**

A historical Triton fused-sort experiment showed that replacing argsort +
scatter + tile with one GPU-side pass removes most of the small-token fixed
overhead. For this FlyDSL target, keep that as migration context only: do not
switch the backend to Triton. Carry the lesson into a FlyDSL-compatible design
where the sort path produces GPU-resident metadata and the FlyDSL S1/S2 kernels
consume it without a CPU synchronization.

**FlyDSL path: GPU-side num_m_tiles + Kernel Early-Exit Guard**

The sort kernel writes `num_m_tiles` to a GPU tensor, and the S1/S2 kernels read it at entry via `buffer_load` and apply a guard:

```python
def moe_stage1_kernel(arg_out, arg_x, arg_w1_ps, arg_expert_ids,
                      arg_sorted_token_ids, arg_num_valid_tiles):
 # Early-exit guard: if block_idx >= num_valid_tiles, directreturns
    bid_m = fx.block_idx.y                    # i32
    nv_rsrc = buffer_ops.create_buffer_resource(arg_num_valid_tiles, max_size=True)
    nv_i32 = buffer_ops.buffer_load(nv_rsrc, fx.Index(0), vec_width=1, dtype=T.i32)
    blk_valid = arith.cmpi(arith.CmpIPredicate.ult, bid_m, nv_i32)
    if_blk = scf.IfOp(blk_valid)
    with _if_then(if_blk):
 # ... complete kernel body ( valid tiles execute GEMM)
```

At launch, `max_tiles` is used as the grid size, and the kernel internally determines whether it is a valid tile.

**Key Implementation Details**:
- `fx.block_idx.y` returns `i32` (not an index type) in FlyDSL, and can be directly compared with `i32` (returned by `buffer_load`) using `arith.cmpi`
- The `_if_then` context manager wraps the then-block of `scf.IfOp` + `scf.YieldOp`
- Invalid tiles execute only 3 instructions (buffer_load + cmp + branch), with negligible overhead

**Benefit**: tc=512 drops from 2663us → 2596us (surpassing aiter's 2624us)

## Final Performance

MI308X, Mixtral-7B BF16, E2E (including sort + GEMM + accumulation):

| tc | FlyDSL(us) | aiter(us) | Speedup | FlyDSL TFLOPS | peak% |
|----|-----------|-----------|--------|--------------|-------|
| 1 | 303.7 | 612.9 | 2.02x | 2.32 | 1.2% |
| 2 | 518.4 | 1148.3 | 2.22x | 2.72 | 1.4% |
| 4 | 780.4 | 1727.2 | 2.21x | 3.61 | 1.8% |
| 8 | 859.0 | 1853.1 | 2.16x | 6.56 | 3.3% |
| 16 | 1026.2 | 2276.9 | 2.22x | 10.99 | 5.5% |
| 32 | 1026.7 | 1253.4 | 1.22x | 21.96 | 11.0% |
| 64 | 1029.3 | 2285.8 | 2.22x | 43.81 | 21.9% |
| 128 | 1219.7 | 2294.2 | 1.88x | 73.94 | 37.0% |
| 256 | 1575.6 | 2319.1 | 1.47x | 114.49 | 57.3% |
| 512 | 2595.7 | 2624.7 | 1.01x | 138.99 | 69.6% |
| 1024 | 4803.9 | 4906.9 | 1.02x | 150.20 | 75.2% |
| 2048 | 8390.2 | 8810.0 | 1.05x | 172.00 | 86.1% |
| 4096 | 15588.4 | 16795.8 | 1.08x | 185.15 | 89.9% |

## Key Lessons Learned

### 1. Optimization Benefit Distribution

| Optimization | Version | Large tc Benefit | Small tc Benefit | Type |
|------|------|-----------|-----------|------|
| Preshuffle layout | V4 | +20% | +20% | Bandwidth |
| tile_n 64→128 | V10 | **+38%** | Moderate | Compute density |
| Fused atomic epilogue | V17 | **+19%** | Large | Eliminate host ops |
| In-kernel X indirection | V18 | +3% | +15% | Eliminate pre-gather |
| GPU-side fused sort migration lesson | V27 | ~0% | **+50%** | Eliminate sort overhead without making Triton the FlyDSL implementation backend |
| Zero-sync early-exit | V27 | ~0% | +5% | Eliminate HIP sync |

### 2. Compute Density Is Critical for Large tc

For compute-bound configurations (tc≥512), **MFMA instruction density** determines performance:
- Number of MFMAs per A load (determined by tile_n)
- Overlap of MFMA and memory operations (scheduling hints)
- K loop prefetch (double buffering)

### 3. Fixed Overhead Is the Bottleneck for Small tc

For memory-bound configurations (tc≤128), **fixed overhead** dominates E2E time:
- Sort (host argsort versus the historical fused-sort comparison: ~200us difference)
- CPU↔GPU sync (.item() calls: ~30-50us)
- Tensor allocation (eliminated by pre-allocation)
- Pre-gather (eliminated by in-kernel indirection)

### 4. FlyDSL-Specific Patterns

- **`scf.IfOp` + `_if_then` guard**: used for kernel early-exit, paired with zero-sync sort
- **`buffer_ops.buffer_load` reads GPU scalar**: avoids `.item()` synchronization
- **`rocdl.raw_ptr_buffer_atomic_fadd`**: S2 epilogue fused accumulation
- **`fx.block_idx.y` returns i32**: no need for `arith.index_cast`, directly participates in `arith.cmpi`
- **Scheduling hints**: `sched_dsrd/mfma/vmem/dswr` controls instruction interleaving order

### 5. Fused Sort Migration Lesson

- Treat the Triton fused-sort result as migration evidence, not as a FlyDSL
  implementation recipe.
- Keep `num_m_tiles` GPU-resident so FlyDSL kernels can read it with
  `buffer_ops.buffer_load` and avoid `.item()` synchronization.
- Preserve the one-pass count + pad + scatter + tile-expert structure when
  adapting the sort path to the consuming FlyDSL harness.

---

## Reference Implementations

- [fused_moe_flydsl.py](../../../../reference-kernels/amd/cdna/flydsl/FlyDSL/fused_moe_mixtral_bf16.py) — Complete implementation code
- [moe_gemm_2stage.py](../../../../reference-kernels/amd/cdna/flydsl/FlyDSL/moe_gemm_2stage.py) — FlyDSL MoE 2-stage reference implementation

## Related

- [Fused MoE Optimization (W4A16)](cdna3-fused-moe-flydsl.md) — Kimi-K2.5 W4A16 mixed-precision MoE optimization
- [FlyDSL Programming Guide](../flydsl-programming-guide.md) — FlyDSL programming guide
- [AMD MFMA Matrix Core Programming Guide](../../common/amd-mfma-matrix-cores.md) — MFMA instruction reference
