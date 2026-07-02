# Blackwell GPGPU Architecture New Features Overview

A summary of Blackwell compute GPU (SM100) architecture updates derived from analysis of CUTLASS, PTX documentation, and Hopper feature comparisons, combined with information disclosed at GTC 2025.


**Last updated**: 2026-06-30

---

## 1. Base Specifications

The full Blackwell die contains 160 SMs, but **each die has only 80 SMs** — a significant reduction from Hopper's full 144 SMs. The reasons are clear:
- Each SM gains 256 KiB of TMEM
- Tensor FMA/SM/clock throughput doubles
- Blackwell uses the same TSMC 4nm process as Hopper, with minimal density improvement

Consequently, SM area increases substantially. Since Hopper enables at most 132 SMs, the GB200's single B200 die is estimated to enable approximately **148±4 SMs** with an inferred boost clock around 2.06 GHz. In practice, GB200-B200 ships with **152 SMs at 1965 MHz boost**, while the standalone B200 has 148 SMs at 1965 MHz — heavy thermal throttling under sustained load.

The two dies are connected via a **10 TB/s NV-HBI** coherent interconnect. Official L2 cache size is 126 MB. The dual-die architecture resembles GA100/GH100's dual L2 partitions, except each partition is now one full reticle die. **Estimated cache line width is 12,288 bytes; measured bandwidth can easily reach 30 TB/s** thanks to the L2 Request Coalescer (LRC). For reference, H100's 5120B theoretical bandwidth is ~10 TB/s, yet measured bandwidth reaches 22 TB/s.

---

## 2. 5th Generation Tensor Core

The fifth-generation Tensor Core introduces new FP4/FP6 number formats and achieves **2x FMA/SM/clock throughput**. FP64 support is not doubled — it is halved.

To achieve the throughput doubling, the 5th-gen Tensor Core introduces **CTA Pair**: two SMs with a total of 8 Tensor Cores work cooperatively. The corresponding SASS instruction is **UTCMMA**. UTCMMA enables greater data sharing compared to Hopper's WGMMA, reducing pressure on L2 cache and SMEM bandwidth. The throughput doubling is straightforward: larger BlockTile and MMA tile sizes enable more data reuse.

To relieve register pressure, Blackwell introduces TMEM: **256 KiB/SM enables pingpong buffering for a 128×256 BlockTile**. On Hopper, the maximum BlockTile for pingpong is approximately 128×192, requiring higher cache bandwidth (measured: H100-SXM 128×160 works at low frequency but struggles at high frequency).

Each warp can only access 32 lanes — **1/4 of the TMEM**. TMEM also enables novel convolution optimizations (Thor's AI compute uses the Blackwell architecture).

The Tensor Core is likely still located within each SM's four subcores, but **a dedicated data-sharing and synchronization channel between two SMs** allows a single thread from one CTA to control 8 Tensor Cores across both SMs, with broadcast capability for the B operand.

**B-operand broadcast is critical**:
- Storing half of B in each SM's SMEM reduces external bandwidth demand
- Broadcasting B to 8 Tensor Cores across both SMs during reads reduces SMEM bandwidth requirements

UTCMMA's programming model is simpler and more efficient than Hopper's WGMMA. It resolves Hopper's issue where CUDA Core instruction dispatch from the same warp and inter-subcore warp synchronization slightly impacted GMMA dispatch efficiency. Additionally, **UTCMMA natively supports block scaling** — Hopper required CUDA Cores for this computation, which impacted peak throughput (as demonstrated in DeepGEMM).

---

## 3. Thread Block Clusters

```cpp
auto cluster_layout_vmnk = tiled_divide(make_layout(cluster_shape), make_tile(typename TiledMma::AtomThrID{}));
auto cluster_shape_fallback = cutlass::detail::select_cluster_shape(ClusterShape{}, hw_info.cluster_shape_fallback);
auto cluster_layout_vmnk_fallback = tiled_divide(make_layout(cluster_shape_fallback), make_tile(typename TiledMma::AtomThrID{}));
```

For H100-SXM, cluster size 2 is optimal because any cluster size greater than 2 causes some SMs to remain idle due to yield limitations. Estimate: the SXM's 132 SMs consist of 3 GPCs with full 18 SMs, 3 GPCs with 16, and one GPC with only 8+6. Using cluster size 4 would leave only 120 SMs usable.

Blackwell introduces **Preferred Thread Block Clusters**: a single grid can launch with two different cluster sizes, naturally solving the fragmentation problem.

Note: NVIDIA has not provided examples of DSMEM-optimized GEMM — unclear whether it cannot accelerate or is simply unnecessary (Split-K and Stream-K excluded from consideration).

---

## 4. Cluster Launch Control (CLC)

Cluster Launch Control first appeared with CUDA 12.8 / CUTLASS 3.8. It upgrades Hopper's Static Tile Scheduling to **Dynamic Tile Scheduling**: previously, SM context switches caused severe tail effects under static scheduling. The core PTX instruction is `clusterlaunchcontrol`, with `UGETNEXTWORKID` as the corresponding SASS instruction.

---

## 5. New PTX/SASS Instructions

- **FP32x2**: FFMA2, FMUL2, FADD2
- **FP8x4**: QFMA4, QMUL4, QADD4
- **Mixed Precision FP**: FHFMA, FHADD

---

## 6. Blackwell Ultra (B300)

Blackwell Ultra achieves **50% FP4 dense throughput improvement to 15 PF**. Comparing HGX B300 official data against B200:
- FP64 tensor and CUDA core support is essentially removed — only 5/148 SMs retain it (for compatibility)
- INT8 tensor throughput reduced to 1/32 of B200 (FP16/TF32 can emulate FP64 where needed)
- FP4 with sparsity vs. without sparsity is no longer a 2x ratio — only FP4 without sparsity increased (suggesting the architecture is bandwidth-limited; the initial Blackwell L2 bandwidth budget appears generous)

Another notable addition is **New Attention Instructions**. Based on community analysis, these are likely **MUFU-related instructions** that accelerate SoftMax computation in MHA. After Blackwell doubled tensor throughput relative to Hopper, if `MUFU.EX2` throughput remained unchanged, MMA could no longer hide Softmax computation latency. **The SFU count likely doubled as well** — BF16/FP16 precision and range are still inferior to FP32.


## Related

- [Comprehensive Guide to NVIDIA Blackwell Architecture](blackwell-architecture-comprehensive-guide.md)
- [GPGPU Architecture: Blackwell Instruction Analysis](blackwell-architecture-instruction-analysis.md)
- [NVIDIA Blackwell Tensor Core Analysis (Part 2): B300](blackwell-tensor-core-analysis-b300.md)
- [NVIDIA Blackwell Tensor Core Analysis (Part 1)](blackwell-tensor-core-analysis-part1.md)
- [Blackwell Ultra (B300): NVIDIA AI Chip Evolution and Roadmap](blackwell-ultra-b300-chip-evolution.md)
- [PTX Programming Model and Basics](../../common/ptx/ptx-programming-model.md)
- [PTX Core Instruction Set](../../common/ptx/ptx-instruction-set.md)
- [CUTLASS/CuTe Core Concepts and Layout Algebra](../../common/cutedsl/cutlass-cute-fundamentals.md)
