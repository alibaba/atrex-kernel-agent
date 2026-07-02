# Blackwell GEMM: Thread Block Clusters, TMA Multicast, and Pair-UMMA

How to leverage thread block clusters for Blackwell GEMM, combining TMA multicast for efficient data sharing and 2-SM UMMA (pair-UMMA) for doubled arithmetic intensity. Part 2 of the Colfax Research CUTLASS Blackwell series.


**Last updated**: 2026-06-30

---

## 1. Introduction

In Part 1, we covered Blackwell's Tensor Memory and UMMA instructions (`tcgen05.mma`). This document explains how to exploit thread block clusters and 2-SM UMMA for Blackwell GEMM:

- Using TMA with thread block clusters and multicast to share global memory transfers among participating CTAs
- Using Blackwell 2-SM UMMA with CTA pairs to increase MMA arithmetic intensity
- Combining TMA multicast and 2-SM UMMA in the GEMM main loop with correct synchronization

## 2. Thread Block Clusters

A thread block cluster groups physically proximate SMs (e.g., on the same GPC). CTAs within a cluster are guaranteed to be co-scheduled on the same GPU Processing Cluster. Introduced on Hopper, clusters give developers a new level of hierarchy for cooperation between adjacent thread blocks. Notably, CTAs in a cluster can access each other's shared memory — a capability called **distributed shared memory**.

### 2.1 Using Thread Block Clusters

Cluster shape is a launch-time parameter (like grid or block size), defined as a `dim3` tuple `<cluster.x, cluster.y, cluster.z>`. The maximum portable cluster size is 8, though some GPUs (Hopper H100, Blackwell B200) allow extending to 16 via opt-in. A `<1,1,1>` cluster is called trivial. The cluster shape must evenly divide the grid size.

In CUTLASS, a special launch utility `launch_kernel_on_cluster` handles cluster launches.

## 3. TMA Multicast

TMA multicast loads a tensor tile into the SMEM of multiple CTAs in the same cluster in a single operation. A set of CTAs within the cluster cooperate to simultaneously load a data tile into their respective shared memories, reducing global memory traffic when multiple CTAs need the same data.

For example, with 4 participating CTAs, each loads one quarter of the data, reducing total TMA load volume by 4x.

## 4. CuTe Example: GEMM with TMA Multicast

Multicast naturally fits GEMM tiling: each tile of operands A and B is used to compute multiple output tiles.

Consider a `<2,2,1>` cluster shape where each CTA handles one (bM, bN) output tile, so the cluster processes a 2x2 block of 4 output tiles. Each main-loop iteration requires loading a (bM, bK) tile of A and a (bN, bK) tile of B per CTA.

With naive TMA, each output tile loads 2 tiles, totaling 8 tiles per cluster. TMA multicast reduces this to the minimum: **4 tiles**.

Specifically: each CTA's required A tile is identical to all other CTAs in the same row; each CTA's required B tile is identical to all CTAs in the same column. Each CTA participates in two TMA multicast operations — one for A (shared with same-row CTAs) and one for B (shared with same-column CTAs).

### 4.1 Synchronizing TMA Participants

TMA multicast participation is specified via `ctaMask` — a bitmask where bit i determines whether the CTA with cluster index i participates. Blackwell's maximum cluster size of 16 yields a 16-bit bitmask.

For a 4x4x1 cluster, CTA 0 has `tma_bitmask_a = 0x1111` and `tma_bitmask_b = 0x000f`. Each bit corresponds to a CTA; multi-dimensional cluster shapes map to 1D CTA order via column-major layout.

Same-row CTAs use identical A bitmasks (e.g., `0x1111` for the top row); same-column CTAs use identical B bitmasks (e.g., `0x000f` for the leftmost column). A CTA only waits for the 6 other CTAs participating in its operand multicast loads.

The `mbarrier` object has two internal counters: a pending arrival count and a pending transaction count (tx-count, in bytes). The phase completes when both reach 0. The tx-count is set to the expected TMA load size via `cute::set_barrier_transaction_bytes`.

### 4.2 Post-UMMA Synchronization

UMMA is asynchronous, so explicit completion waiting is required. Additionally, other CTAs must finish consuming operand data in SMEM before multicast overwrites it.

A simple solution is `cute::cluster_sync()`. A better approach: each CTA waits only for the 3 CTAs sharing its A tile and the 3 sharing its B tile. This sub-cluster synchronization uses new Blackwell instructions, specifically `tcgen05.commit` or its CUTLASS wrapper `cutlass::arch::umma_arrive_multicast`.

The MMA bitmask = TMA_A bitmask OR TMA_B bitmask. For CTA 0: `tma_bitmask_a | tma_bitmask_b = 0x111f`. Each CTA's barrier waits for 7 arrivals (all CTAs in the mask, including itself).

## 5. CuTe Example: Pair-UMMA with TMA Multicast

Blackwell adds the ability for two adjacent CTAs in the same cluster to cooperatively execute UMMA. This is called 2-SM UMMA or **pair-UMMA**.

### 5.1 Thread Block Cluster for CTA Pairs

CTA pairs must reside in a single cluster. Within the cluster, **CTAs whose indices differ in bit 0** (e.g., 0 and 1, 2 and 3) form a pair. The even-indexed CTA is the "even CTA" and the odd-indexed is the "odd CTA."

With pair-UMMA, `AtomThrID{}` is 2, and `tiled_divide` tiles along dimension 0 of the cluster shape with size-(2) tiles.

`mma_coord_vmnk` is a composite coordinate: mode 0 is the peer-CTA coordinate within a single MMA, while modes 1-3 are the MMA's global coordinates. **On Blackwell, MMA is pair-local; on Hopper, it was CTA-local.**

### 5.2 Pair-UMMA Details

In pair-UMMA, both CTAs in the pair cooperatively process the same MMA tile. Each CTA loads half of each operand tile and holds half the accumulator in its TMEM. For a 256x256x16 MMA, each CTA loads a 128x16 slice of A and B, and holds a 128x256 accumulator in TMEM.

In terms of arithmetic intensity, this behaves like a 256x256 MMA: compared to two CTAs independently executing 128x256 MMAs, the pair-UMMA performs the same FLOPs but transfers half the operand data.

Pair-UMMA is issued in PTX as `tcgen05.mma` with qualifier `cta_group::2`. **Supported M sizes are 128 and 256**; the accumulator is always split along M between the two CTAs.

The pair-UMMA must be launched from a single thread in the designated leader CTA. In CUTLASS, the **even CTA is always the leader**.

### 5.3 Constructing Bitmasks

In the 2-SM case, each CTA is responsible for a non-overlapping half of the MMA tile. The even CTA does not need data from the odd CTA and vice versa. Therefore, **TMA multicast only needs to multicast to CTAs of the same parity**. The MMA, however, uses the entire tile and needs data from both parities.

For a `<4,4,1>` cluster (producing a 4D cluster shape `<2, 2, 4, 1>`), CTA 0's bitmasks:

- `tma_mcast_mask_a: 0x1111`
- `tma_mcast_mask_b: 0x0005`
- `mma_mcast_mask_c: 0x333f`

CUTLASS utility: `create_tma_multicast_mask<Modes...>(cluster_layout_vmnk, cta_in_cluster_coord_vmnk)` produces a bitmask of all CTAs differing from the specified CTA only in the given modes. `create_tma_multicast_mask<2>` creates the mask for CTAs participating in this A tile's TMA load; `create_tma_multicast_mask<0,2>` creates the mask for CTAs participating in MMA using this A tile.

### 5.4 Synchronizing Pair-UMMA

Since launch comes from the even CTA, UMMA arrival instructions must also come from the even CTA. In `cluster_shape_vmnk`, the M mode has size 2 and N mode has size 4. The participant count (arrival count) is 5, despite involving 10 CTAs.

Pair-UMMA arrival uses the special CUTLASS function `umma_arrive_multicast_2x1SM`, because `tcgen05.commit` with `cta_group::1` and `cta_group::2` are processed in different pipelines.

### 5.5 Synchronizing TMA Multicast for 2-SM

TMA is launched with a bitmask restricting multicast to same-parity CTAs. But `wait_barrier` for TMA is called only from the even CTA, and it must wait for the entire MMA tile.

CUTLASS solution: SM100 introduces a `cta_group` qualifier for TMA copy instructions. Setting it to `cta_group::2` allows TMA copy to arrive at the executing CTA's or its peer CTA's mbarrier. `Sm100MmaPeerBitMask` is `0xFEFFFFFF` — a CTA can find its leader CTA's mbarrier address by clearing bit 24 of its own mbarrier address.

Pair-aware copy atoms are created via `make_tma_atom_[A|B]_sm100()`.

## 6. Conclusion

TMA multicast reduces global memory traffic by sharing loads among CTAs in a cluster. Pair-UMMA doubles arithmetic intensity by having two CTAs cooperatively execute a single larger MMA. For both features, the complex indexing and bitmask logic is largely abstracted by CuTe layouts and CUTLASS utility functions.

Part 3 covers low-precision GEMM and native block-scaling support on Blackwell.


## Related

- [SM100 Blackwell CuTeDSL Panorama](blackwell-cutedsl-sm100.md)
- [Blackwell GEMM: Low-Precision Data Types and Block Scaling](blackwell-gemm-low-precision.md)
- [CUTLASS Tutorial: Blackwell GEMM with Tensor Memory](blackwell-gemm-tensor-memory.md)
- [Building a tcgen05 GEMM from Scratch: Reaching 98% of cuBLAS on Blackwell](blackwell-tcgen05-gemm-from-scratch.md)
- [Blackwell Matrix Multiplication Part 1: Fundamentals](colfax-blackwell-gemm-part1-basics.md)
- [CUTLASS GEMM Optimization Strategy](../../common/cutedsl/cutlass-gemm-optimization.md)
- [Community GEMM Optimization Practical Summary](../../../generic/gemm-optimization-guide.md)
- [CUTLASS/CuTe Core Concepts and Layout Algebra](../../common/cutedsl/cutlass-cute-fundamentals.md)
