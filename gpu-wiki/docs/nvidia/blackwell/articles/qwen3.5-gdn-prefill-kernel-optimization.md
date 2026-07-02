# Qwen3.5 GDN Prefill Kernel Optimization

Practical optimization of Qwen3.5 GDN prefill kernels using Gluon on Blackwell, covering TMA pipelining, async scatter, bitmask causal masking, and zero-copy transpose.


**Last updated**: 2026-06-30

---

## 1. Algorithm Overview

Qwen3.5 GDN prefill uses a chunk-wise algorithm: sequence length T is divided into ⌈T/C⌉ chunks (C=64). Inter-chunk operations serially recur the hidden state, while intra-chunk operations compute attention in parallel. This document focuses on the two core kernels: `chunk_gated_delta_rule_fwd_h` and `chunk_o`.

---

## 2. Gluon vs Triton

| Feature | Triton | Gluon |
| --- | --- | --- |
| Data movement | `tl.load/store` (compiler-scheduled) | TMA async copy (explicit mbarrier) |
| Matrix multiply | `tl.dot` (compiler selects instruction) | `tcgen05_mma` (direct Blackwell TC call) |
| Accumulator | Registers (compiler-allocated) | **TMEM** (explicitly allocated) |
| Layout | Compiler-automatic | BlockedLayout / NVMMASharedLayout |
| Memory sync | Implicit | mbarrier explicit |

---

## 3. Blackwell Feature Overview

- **TMEM**: 128 rows × 512 cols × 32 bit = **256 KB / CTA**, serves directly as tcgen05_mma output accumulator, eliminating the Hopper WGMMA register accumulator bottleneck
- **tcgen05_mma**: Inputs from SMEM/TMEM, output to TMEM, throughput > WGMMA; requires `tcgen05_commit + mbarrier.wait` for explicit synchronization
- **TMA**: Describes multi-dimensional tensors via Tensor Descriptors; supports async_copy_global_to_shared / async_copy_shared_to_global / async_scatter
- **mbarrier**: Phase-flip mechanism: `mbarrier.expect → async operation → wait(phase) → phase ^= 1`

---

## 4. chunk_delta_h: Recurrence + Pipelining

Each chunk update has 4 steps: (1) store h_t → (2) v_new = v - w @ h_t → (3) apply gate (v_new *= exp(g_last-g), h *= exp(g_last)) → (4) h_{t+1} = h_t + k^T @ v_new.

Pipeline design:

1. **w prologue prefetch**: Initiate TMA load of w[0] before the main loop
2. **v + k early prefetch**: Launch at iteration start, overlapping with gate scalar computation / h store / w wait / MMA1
3. **w next-iteration prefetch**: After MMA1 completes and w_smem is free, immediately prefetch w[t+1]
4. **h TMA store overlaps with MMA1**: Using different smem buffers for parallelism

---

## 5. Varlen async_scatter (Key Optimization)

`tma.async_copy_shared_to_global` writes in fixed blocks, which can overwrite the next sequence. The 1D scatter approach:

```python
t_offsets = gl.arange(0, BT)
row_valid = t_offsets < t_limit_right
x_offsets = gl.where(row_valid, bos + i_t*BT + t_offsets, 0x7FFFFFFF)
tma.async_scatter(v_new_desc, x_offsets, i_h*V + i_v*BV, v_new_smem)
```

Out-of-bounds indices are set to `0x7FFFFFFF` (int32 max); the TMA hardware automatically skips them, preserving asynchronous behavior. **Measured +12% improvement.** TMA gather/scatter was originally designed for sparse matrices.

---

## 6. chunk_o: Bitmask Causal Mask

Original approach: BT×BT element-wise comparison + where, producing many setp instructions.

Bitmask approach (16 elements per group):

```python
@gluon.jit
def _mask_scalar(A, col_limit_right, s, i):
    col_lim_right_cur = max(col_limit_right - s, 0)
    mask = -1 << col_lim_right_cur
    return gl.where((mask & (1 << i)) == 0, A, 0.0)

@gluon.jit
def _apply_causal_mask(A, col_limit_right):
    offs_n = gl.arange(0, A.shape[1])[None, :]
    s = offs_n & ~0xF   # group start
    i = offs_n & 0xF    # intra-group offset
    return gl.map_elementwise(_mask_scalar, A, col_limit_right, s, i)
```

`-1 << col_lim` sets all "visible column" positions to 0; the R2P instruction extracts 16 predicates at once; `gl.map_elementwise` interleaves mask/where instructions (mask[0], where[0], mask[1], where[1]...) to facilitate ptxas SASS optimization and reduce register pressure. NCU measurements confirm significantly reduced instruction count.

---

## 7. Transposed State Zero-Copy

GDN hidden state `[N,H,K,V]` requires `[N,H,V,K]` (K-dimension contiguous) for the decode CuTeDSL kernel. Triton modifies stride/order; Gluon uses **smem.permute()** to change only the access view — zero-copy transpose without extra register transpose overhead.

---

## 8. cumsum Vectorization

The original `chunk_local_cumsum_scalar_kernel` processes only 1 head per chunk per block, with uncoalesced memory access. Optimization: merge BH heads into the same block, expanding [BT] → [BT, BH]. The blocked layout derivation produces single-warp `ld/st.global.v4.b32` vectorized memory access (replacing the original 8-warp scalar access).

---

## 9. Notes

- chunk_size=64 aligns well with Hopper MMA performance; Blackwell theoretically allows tuning but this PR maintains consistency
- Only supports sm100


## Related

- [Comprehensive Guide to NVIDIA Blackwell Architecture](blackwell-architecture-comprehensive-guide.md)
- [GPGPU Architecture: Blackwell Instruction Analysis](blackwell-architecture-instruction-analysis.md)
- [Blackwell GPGPU Architecture New Features Overview](blackwell-gpgpu-new-features-overview.md)
- [NVIDIA Blackwell Tensor Core Analysis (Part 2): B300](blackwell-tensor-core-analysis-b300.md)
- [NVIDIA Blackwell Tensor Core Analysis (Part 1)](blackwell-tensor-core-analysis-part1.md)
- [Composable Kernel (CK) Architecture Overview](../../../amd/common/ck-architecture-overview.md)
- [Gluon Tutorial 07: Persistent Kernels and Pipeline Optimization](../../common/gluon/gluon-07-persistent-kernel-pipeline.md)
