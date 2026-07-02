# High-Performance Memory-Bound Kernels on B200: An MXFP8 Quantization Case Study

This document discusses optimization techniques for memory-bound kernels on NVIDIA B200, using an MXFP8 blockscaled quantization kernel as a practical example. Topics include persistent kernel design, 256-bit vectorized access, occupancy tuning, and asynchronous 1D TMA for scale factor storage.


**Last updated**: 2026-06-30

---

## 1. Background

This kernel was submitted to the SGLang community as part of a CUTLASS-based MXFP8 Blockscaled Grouped GEMM implementation. The quantization kernel converts multiple fp16/bf16 input matrices to fp8 (e4m3) and outputs the MXFP8 blockscaled scale factors required by the GEMM.

It is a classic memory-bound kernel. The target architecture B200 provides 7.7 TB/s HBM3e bandwidth (nearly 2x H200), making effective bandwidth utilization the core optimization challenge. The techniques discussed are broadly applicable to other memory-bound kernels and architectures.

## 2. Kernel Design Overview

In the Grouped GEMM scenario, per-group input sizes are variable and unknown at compile time. The standard approach is a **persistent kernel**: launch a fixed number of ThreadBlocks that dynamically process multiple data tiles at runtime via scheduling logic.

The activation matrix is Row-Major (M, K), with one block of 32 elements along K producing one e8m0 (8-bit) scale factor. Tiles must align with block boundaries, so `BLOCK_K` is a multiple of 32 to avoid cross-ThreadBlock data exchange.

### 2.1 Scale Factor Tile Layout

A critical constraint: cuBLAS/CUTLASS blockscaled scale factors are not stored in simple (M, ceil(K/32)) Row-Major format. The (M, ceil(K/32)) SF matrix is partitioned into (128, 4) tiles, where each tile's 512 elements must be stored in a special swizzled layout occupying exactly 512 contiguous bytes.

TensorRT-LLM enforces padding: M aligned to 128, ceil(K/32) aligned to 4:

```cpp
enum class QuantizationSFLayout {
    // Block scale factors stored in swizzled layout for cutlass FP4 kernel.
    // 512-byte blocks in global memory, each block 128x4 FP8 values.
    // SF rows padded to multiple of 128, columns to multiple of 4.
    // For SF row 'i': maps to data block row (i % 4) * 32 + (i / 4)
    SWIZZLED,
    LINEAR
};
```

Each (128, 4) SF tile corresponds to a (128, 128) block of the original input matrix, so each ThreadBlock processing a (128, 128) tile naturally produces one contiguous 512-byte SF output.

## 3. Vectorized and Coalesced Access

Each ThreadBlock loads one (128, 128) fp16/bf16 tile, stores one (128, 128) fp8 tile, plus 512 bytes of scale factors.

Key points:
- Row-contiguous data with consecutive threads processing consecutive elements ensures coalesced access
- **Blackwell extends per-thread load/store width from 128-bit to 256-bit** (16 fp16/bf16 elements), which is critical for saturating HBM bandwidth
- fp8 store uses 128-bit STG (sufficient, not the bottleneck)

CuTe TiledCopy implementation:

```cpp
using ThrLayout = Layout<Shape<_16, _8>, Stride<_8, _1>>;
using ValLayout = Layout<Shape<_1, _16>>;
ThrLayout thr_layout{};
ValLayout val_layout{};

using CopyOpG2R = UniversalCopy<cutlass::AlignedArray<T_IN, size(val_layout)>>;
using CopyAtomG2R = cute::Copy_Atom<CopyOpG2R, T_IN>;
auto tiled_copy_g2r = cute::make_tiled_copy(CopyAtomG2R{}, thr_layout, val_layout);

using CopyOpR2G = UniversalCopy<cutlass::AlignedArray<cutlass::float_e4m3_t, size(val_layout)>>;
using CopyAtomR2G = cute::Copy_Atom<CopyOpR2G, cutlass::float_e4m3_t>;
auto tiled_copy_r2g = cute::make_tiled_copy(CopyAtomR2G{}, thr_layout, val_layout);
```

Store reuses the same thread-to-element mapping as Load, avoiding inter-thread data exchange.

## 4. Occupancy Optimization

For traditional SIMT-based memory-bound kernels, **high occupancy is nearly essential** to saturate bandwidth — sufficient schedulable warps are needed to keep the LSU fully utilized. The persistent design also requires launching enough ThreadBlocks.

Grid size is determined using the occupancy API:

```cpp
int max_active_blocks_per_sm = -1;
AT_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &max_active_blocks_per_sm,
    mxfp8_group_quant<T_IN, decltype(tiled_copy_g2r), decltype(tiled_copy_r2g), decltype(tiled_copy_r2s)>,
    THREAD_BLOCK_SIZE, 0));
dim3 grid(at::cuda::getCurrentDeviceProperties()->multiProcessorCount * max_active_blocks_per_sm, 1, 1);
```

### 4.1 Loop Unrolling vs Register Pressure

Counter-intuitive finding: the inner loop has 8 compile-time-visible iterations. Full unrolling raises registers per thread to 40, dropping occupancy from 100% to 75%. Manual `#pragma unroll 4` restores 100% occupancy with better actual performance:

```cpp
constexpr int tile_loop_count = size<1>(tiled_tensor_s);  // 8
#pragma unroll 4
for (int t = 0; t < tile_loop_count; t++) {
    // ...
}
```

ILP vs register pressure for memory-bound kernels has no theoretical answer — empirical measurement is required.

## 5. TMA for Scale Factor Storage

The 512-byte SF tile has internally shuffled rows (not Row-Major):

```
Row 0  -> Row 0
Row 1  -> Row 4
Row 2  -> Row 8
Row 3  -> Row 12
...
Row 32 -> Row 1
Row 33 -> Row 5
...
```

This is expressible as a CuTe Layout:

```cpp
using ScaleFactorTileLayout = Layout<Shape<Shape<_32, _4>, _4>, Stride<Stride<_16, _4>, _1>>;
```

The shuffle is unfriendly to coalesced access — warps that would naturally write 16 consecutive SFs now scatter across 4 rows. TensorRT-LLM handles this with simple `STG.8` (single-byte stores).

The optimized approach: **assemble the SF tile in shared memory matching the global memory layout, then issue a single shared-to-global copy**. Treating the SF tile as a 1D 512-byte tensor enables **1D async TMA** (no TMA descriptor needed, no dynamic update overhead, simpler programming).

Async TMA enables compute/store overlap: after issuing TMA, immediately proceed to the next (128, 128) tile computation. Synchronize only before rewriting SMEM.

The `cp.async.bulk.wait_group` instruction with `.read` modifier allows early return (data visibility to the issuing thread is not required), preventing pipeline stalls.

The fundamental benefit of async TMA: **issuing memory transactions more densely, saturating L2/HBM bandwidth**.

## 6. Performance Results

NCU profiling shows:
- End-to-end HBM bandwidth utilization: 75%+
- Peak bandwidth utilization (PM sampling): 87%+
- Gap between end-to-end and peak is primarily due to load imbalance

The load imbalance comes from the current per-group scheduling strategy: a single group of (4096, 4096) only fills 1024 ThreadBlocks, far fewer than the grid size (2368, 1, 1). Subsequent groups only use the first 1024 ThreadBlocks, leaving the rest idle. Cross-group scheduling optimization addresses this.

## 7. Summary

Optimization techniques for B200 memory-bound kernels:

1. **Persistent kernel + (128, 128) tile** — aligned with SF (128, 4) swizzled layout
2. **Blackwell 256-bit per-thread load/store** — critical for saturating 7.7 TB/s bandwidth
3. **`cudaOccupancyMaxActiveBlocksPerMultiprocessor`** — determine grid size for 100% theoretical occupancy
4. **Manual `#pragma unroll` control** — prevent compiler from inflating register pressure
5. **Shared memory staging + 1D async TMA** for scale factors — significantly outperforms `STG.8`
6. **`.read` modifier on `cp.async.bulk.wait_group`** — enable early return for better pipeline utilization

## 8. Design Trade-offs

**Why not 2D TMA for loads?** 2D TMA requires SMEM space and typically uses multi-stage buffering that further reduces occupancy. For Grouped GEMM, dynamic TMA descriptor updates add significant overhead. SIMT vector loads are better suited to this scenario.

**Small shapes (e.g., single (4096, 4096) group) not saturating bandwidth?** This is a limitation of the current parallelism strategy. Small data should use (16, 128) tiles per ThreadBlock, increasing grid size by 8x. SF output falls back to STG/SMEM staging since small data is fundamentally latency-bound and TMA's latency becomes unfavorable.

**Persistent vs Warp Specialization?** These are orthogonal — non-persistent kernels can also use warp specialization. However, for this memory-bound scenario, warp specialization is counterproductive: multi-buffer SMEM consumption, consumer warps not issuing memory requests, and reduced overall concurrency conflict with the goal of having all 64 warps saturating bandwidth.


## Related

- [SM100 Blackwell CuTeDSL Panorama](blackwell-cutedsl-sm100.md)
- [Blackwell GEMM: Low-Precision Data Types and Block Scaling](blackwell-gemm-low-precision.md)
- [CUTLASS Tutorial: Blackwell GEMM with Tensor Memory](blackwell-gemm-tensor-memory.md)
- [Blackwell GEMM: Thread Block Clusters, TMA Multicast, and Pair-UMMA](blackwell-gemm-thread-block-cluster.md)
- [Building a tcgen05 GEMM from Scratch: Reaching 98% of cuBLAS on Blackwell](blackwell-tcgen05-gemm-from-scratch.md)
- [Occupancy Optimization](../../../amd/common/occupancy-optimization.md)
- [Occupancy Tuning Differences Across Architectures](../../common/occupancy-tuning-by-arch.md)
- [CUTLASS/CuTe Core Concepts and Layout Algebra](../../common/cutedsl/cutlass-cute-fundamentals.md)
