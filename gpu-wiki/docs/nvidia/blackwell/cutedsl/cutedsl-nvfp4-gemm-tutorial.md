# CuTeDSL Tutorial: Learning nvfp4_gemm_0.py

A walkthrough of the CuTe DSL NVFP4 GEMM example, covering TMA tensor construction, MMA configuration, TMEM layout, and the producer-consumer mainloop pattern.


**Last updated**: 2026-06-30

---

## 1. Regular vs TMA Tensor Syntax

### 1.1 wgmma_sm90.cu (Without TMA)

```cpp
Tensor mA = make_tensor(make_gmem_ptr(A), select<0,2>(shape_MNK), dA); // (M,K)
Tensor mB = make_tensor(make_gmem_ptr(B), select<1,2>(shape_MNK), dB); // (N,K)
Tensor mC = make_tensor(make_gmem_ptr(C), select<0,1>(shape_MNK), dC); // (M,N)
auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});
Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{});
Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{});
```

`mA` is the full global tensor; `gA` is the tile this CTA operates on — straightforward.

### 1.2 wgmma_tma_sm90.cu (With TMA)

```cpp
auto [M, N, K] = shape_MNK;
Tensor mA = tma_a.get_tma_tensor(make_shape(M,K)); // (M,K) TMA Tensor
Tensor mB = tma_b.get_tma_tensor(make_shape(N,K)); // (N,K) TMA Tensor
Tensor mC = make_tensor(make_gmem_ptr(C), make_shape(M,N), dC); // (M,N)
// gA uses the same syntax as before
```

The `mA` construction changes to `tma_a.get_tma_tensor(...)` — additional boilerplate.

### 1.3 nvfp4_gemm_0.py (Python DSL)

```python
b_tensor = cute.make_tensor(
    b_ptr,
    cute.make_layout(
        (n, cute.assume(k, 32), l),
        stride=(cute.assume(k, 32), 1, cute.assume(n * k, 32)),
    ),
)
tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
    cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
    b_tensor, b_smem_layout, self.mma_tiler, tiled_mma,
    self.cluster_layout_vmnk.shape,
)
```

A regular tensor is transformed via `make_tiled_tma_atom_B` into `tma_tensor_b`.

---

## 2. Key Configuration in nvfp4_gemm_0.py

```python
mma_tiler_mn  = (128, 256)
mma_inst_shape_k = 64
ab_dtype = cutlass.Float4E2M1FN
sf_dtype = cutlass.Float8E4M3FN
c_dtype  = cutlass.Float16
sf_vec_size = 16

self.threads_per_cta   = 128
self.num_tmem_alloc_cols = 512
self.num_acc_stage = 1
self.num_ab_stage  = 4
```

**Constraints:**

- m, n, k must be divisible by tile dimensions (128, 256, 256)
- Scale factor vector size = 16
- A/B: k-dimension contiguous; C: n-dimension contiguous
- A/B = Float4E2M1FN (NVFP4), SFA/SFB = Float8E4M3FN (E4M3 scale factors)

---

## 3. MMA Op and Tiled MMA Construction

```python
mma_inst_tile_k = 4
self.mma_tiler = (mma_tiler_mn[0], mma_tiler_mn[1], mma_inst_shape_k * mma_inst_tile_k)  # (128,256,256)

mma_op = tcgen05.MmaMXF4NVF4Op(
    sf_dtype,
    (*mma_tiler_mn, mma_inst_shape_k),
    tcgen05.CtaGroup.ONE,
    tcgen05.OperandSource.SMEM,
)
tiled_mma = cute.make_tiled_mma(mma_op)
```

---

## 4. SFA/SFB Layout Chain

```python
sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(a_tensor.shape, sf_vec_size)
# ((Atom_M, Rest_M),(Atom_K, Rest_K),RestL)
sfa_tensor = cute.make_tensor(sfa_ptr, sfa_layout)
```

---

## 5. Four TMA Atoms

```python
tma_atom_a, tma_tensor_a   = cute.nvgpu.make_tiled_tma_atom_A(CopyBulkTensorTileG2SOp(CtaGroup.ONE), a_tensor, a_smem_layout, ...)
tma_atom_b, tma_tensor_b   = cute.nvgpu.make_tiled_tma_atom_B(..., b_tensor, b_smem_layout, ...)
tma_atom_sfa, tma_tensor_sfa = cute.nvgpu.make_tiled_tma_atom_A(..., sfa_tensor, sfa_smem_layout, ..., internal_type=cutlass.Int16)
tma_atom_sfb, tma_tensor_sfb = cute.nvgpu.make_tiled_tma_atom_B(..., sfb_tensor, sfb_smem_layout, ..., internal_type=cutlass.Int16)
```

SFA/SFB use `internal_type=Int16` to avoid FP8-to-FP conversion overhead by treating them as raw 16-bit integers during transfer.

---

## 6. TMEM Layout: Accumulator + SFA + SFB Compact Packing

```python
acc_tmem_ptr = tmem.retrieve_ptr(cutlass.Float32)
tCtAcc = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

# SFA immediately after accumulator
sfa_tmem_ptr = cute.recast_ptr(
    acc_tmem_ptr + tcgen05.find_tmem_tensor_col_offset(tCtAcc),
    dtype=sf_dtype,
)
tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)

# SFB immediately after SFA
sfb_tmem_ptr = cute.recast_ptr(
    acc_tmem_ptr + tcgen05.find_tmem_tensor_col_offset(tCtAcc)
                 + tcgen05.find_tmem_tensor_col_offset(tCtSFA),
    dtype=sf_dtype,
)
```

---

## 7. S2T Copy (SMEM to TMEM)

```python
copy_atom_s2t = cute.make_copy_atom(
    tcgen05.Cp4x32x128bOp(tcgen05.CtaGroup.ONE),
    sf_dtype,
)
tiled_copy_s2t_sfa = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSFA_compact)
```

---

## 8. Mainloop (Single-Warp Producer, warp_idx == 0)

```python
if warp_idx == 0:
    acc_empty = acc_producer.acquire_and_advance()
    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

    for k_tile in cutlass.range(k_tile_cnt, prefetch_stages=self.num_ab_stage - 2):
        ab_empty = ab_producer.acquire_and_advance()

        # 4 TMA loads (A / B / SFA / SFB) sharing the same mbarrier ab_empty.barrier
        cute.copy(tma_atom_a,   tAgA[(None, ab_empty.count)],   tAsA[(None, ab_empty.index)],   tma_bar_ptr=ab_empty.barrier)
        cute.copy(tma_atom_b,   tBgB[(None, ab_empty.count)],   tBsB[(None, ab_empty.index)],   tma_bar_ptr=ab_empty.barrier)
        cute.copy(tma_atom_sfa, tAgSFA[(None, ab_empty.count)], tAsSFA[(None, ab_empty.index)], tma_bar_ptr=ab_empty.barrier)
        cute.copy(tma_atom_sfb, tBgSFB[(None, ab_empty.count)], tBsSFB[(None, ab_empty.index)], tma_bar_ptr=ab_empty.barrier)

        ab_full = ab_consumer.wait_and_advance()

        # S2T copy SFA/SFB -> TMEM
        cute.copy(tiled_copy_s2t_sfa, tCsSFA_compact_s2t[(None, None, None, None, ab_full.index)], tCtSFA_compact_s2t)
        cute.copy(tiled_copy_s2t_sfb, tCsSFB_compact_s2t[(None, None, None, None, ab_full.index)], tCtSFB_compact_s2t)

        # Inner k-block loop
        num_kblocks = cute.size(tCrA, mode=[2])
        for kblock_idx in cutlass.range(num_kblocks, unroll_full=True):
            tiled_mma.set(tcgen05.Field.SFA, tCtSFA[(None, None, kblock_idx)].iterator)
            tiled_mma.set(tcgen05.Field.SFB, tCtSFB[(None, None, kblock_idx)].iterator)
            cute.gemm(tiled_mma, tCtAcc, tCrA[(None, None, kblock_idx, ab_full.index)],
                                          tCrB[(None, None, kblock_idx, ab_full.index)], tCtAcc)
            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

        ab_full.release()

    acc_empty.commit()
```

---

## 9. Epilogue

```python
op = tcgen05.Ld32x32bOp(tcgen05.Repetition.x128, tcgen05.Pack.NONE)
copy_atom_t2r = cute.make_copy_atom(op, cutlass.Float32)
tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tCtAcc)
# ... T2R transfer to registers -> cast to fp16 -> SIMT STG writes output C
acc_full.release()
tmem.free(acc_tmem_ptr)
```

---

## 10. Naming Convention Explained

The naming convention `tAsA` is standard across CuTe and CUTLASS. It reads as "partitioning pattern `tA` applied to tensor `sA`". By applying the same partitioning pattern `tA` to both `sA` and `gA`, logical consistency is preserved — enabling `cute::copy` to verify that two tensors use the same partitioning via lexicographic validation.

Therefore `tCtSFB` = partitioning pattern `tC` applied to `tSFB` (a TMEM tensor). This per-CTA view preserves logical consistency for copy operations. The prefix is necessary because it reflects the per-CTA partitioning scope.


## Related

- [SM100 Blackwell CuTeDSL Panorama](blackwell-cutedsl-sm100.md)
- [Blackwell GEMM: Low-Precision Data Types and Block Scaling](blackwell-gemm-low-precision.md)
- [CUTLASS Tutorial: Blackwell GEMM with Tensor Memory](blackwell-gemm-tensor-memory.md)
- [Blackwell GEMM: Thread Block Clusters, TMA Multicast, and Pair-UMMA](blackwell-gemm-thread-block-cluster.md)
- [Building a tcgen05 GEMM from Scratch: Reaching 98% of cuBLAS on Blackwell](blackwell-tcgen05-gemm-from-scratch.md)
- [CUTLASS GEMM Optimization Strategy](../../common/cutedsl/cutlass-gemm-optimization.md)
- [Community GEMM Optimization Practical Summary](../../../generic/gemm-optimization-guide.md)
