# Blackwell Matrix Multiplication Part 4: Beyond SOTA

Persistent kernels with CLC scheduling, TMEM circular buffering, and block swizzle techniques that push Blackwell GEMM to 100.6% of cuBLAS on 4096³ and beyond for production workloads.


**Last updated**: 2026-06-30

---

## 1. Kernel 8: CLC-Based Persistent Kernel

Two remaining overheads:
- Shared memory and barriers must be re-initialized between waves
- Writes to C matrix cannot be overlapped

### 1.1 What is a Persistent Kernel?

A **wave** refers to a batch of thread blocks that can be simultaneously allocated to all SMs. A **persistent kernel** means the kernel author controls tile coordinate scheduling rather than hardware.

- Launch CTAs equal to the number of SMs
- Each CTA performs MMA + store
- After store completes, fetch the next tile coordinate
- Repeat for `total_tiles / num_SMs` iterations

Static persistent kernels have problems: they assume exclusive SM ownership, are unaware of SM busy/idle state, and can cause work starvation.

### 1.2 CLC (Cluster Launch Control) Scheduler

Blackwell introduces an on-chip scheduler:
- Launches the number of thread blocks needed by the problem shape
- A scheduling warp (warp 4 of the first CTA in the cluster) attempts to "dequeue" work blocks for idle SMs
- The scheduling warp writes coordinates to dedicated shared memory (16 bytes) within all CTAs in the cluster
- All CTAs read coordinates from shared memory

Communication between the scheduling warp and load warp uses `empty_mbar` / `full_mbar`, enabling overlap of the next wave's TMA loads with the current wave's C writes.

### 1.3 CLC Fetch Pipelining

2-level pipeline: Each CTA reserves two shared memory locations for CLC responses; the scheduling warp writes the next coordinate one step ahead.

### 1.4 TMEM as Circular Buffer

Problem with a single TMEM accumulator:
- Epilogue warps (TMEM→SMEM→GMEM) force producer/consumer warps to idle (avoiding overwrite of data being read)
- Output and next-wave MMA execute sequentially (data dependency on TMEM between MMA warp and output warp)

**Solution**: When MMA_N == 256, use TMEM as a circular buffer — epilogue warps write columns 0-255 while MMA warp simultaneously accumulates into columns 256-511. New `accum_full` / `accum_empty` barriers coordinate access.

**Performance: 4096³ → 1772.9 TFLOPS = 100.6% of cuBLAS.** However, 8192³ only achieves ~90%.

---

## 2. Kernel 9: Block Swizzle

Improves L2 hit rate.

- 6×5 cluster = 28 cluster-level tiles, 9 SMs per wave
- Default Wave 0: A tile ([0,5], k) loaded 6 times, B tile ([0,1], k) loaded 2 times
- Introduce `block_swizzle_size` (number of steps along N before stepping down in M) — sawtooth pattern enables tile groups to share coordinates
- block_swizzle_size=2: A tile ([0,4], k) 5 loads, B tile ([0,1], k) 2 loads (reduces 1 global load)

Optimal swizzle_size depends on problem shape + number of cluster-level tiles.

---

## 3. Production Shapes

In actual LLM workloads, M = batch × prompt_length ≈ 10²-10³, while N/K depend on the model (Gemma 3 27B has wide range).

Example: Gemma 3 serving with M×N×K = 512×8192×5376
- 256×256 2×SM MMA → 512/128 × 8192/256 = **128 CTAs** (B200 has 148 SMs → wasted)
- Switch to 256×224 → **148 CTAs**, perfectly mapping to 148 SMs

Mojo includes a built-in auto-tuning framework called **kbench**. MAX running Gemma 3 27B on Blackwell **matches or exceeds SOTA by up to +6%**.

---

## 4. Summary

> GPUs will become more powerful and sophisticated... Mojo is positioned to meet this challenge.

The complete optimization journey from 5 TFLOPS (naive 4-line kernel) to 1772.9 TFLOPS (100.6% cuBLAS) demonstrates that with proper understanding of hardware features — TMA, tcgen05 MMA, TMEM, CLC scheduling, and pipeline design — peak Blackwell performance is achievable at the language level without resorting to hand-tuned assembly.


## Related

- [SM100 Blackwell CuTeDSL Panorama](blackwell-cutedsl-sm100.md)
- [Blackwell GEMM: Low-Precision Data Types and Block Scaling](blackwell-gemm-low-precision.md)
- [CUTLASS Tutorial: Blackwell GEMM with Tensor Memory](blackwell-gemm-tensor-memory.md)
- [Blackwell GEMM: Thread Block Clusters, TMA Multicast, and Pair-UMMA](blackwell-gemm-thread-block-cluster.md)
- [Building a tcgen05 GEMM from Scratch: Reaching 98% of cuBLAS on Blackwell](blackwell-tcgen05-gemm-from-scratch.md)
- [CUTLASS GEMM Optimization Strategy](../../common/cutedsl/cutlass-gemm-optimization.md)
- [Community GEMM Optimization Practical Summary](../../../generic/gemm-optimization-guide.md)
- [Composable Kernel (CK) Architecture Overview](../../../amd/common/ck-architecture-overview.md)
