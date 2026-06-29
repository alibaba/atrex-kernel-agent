# Triton to SASS: TMA, Multicast, and Warp Specialization Debugging

Practical notes on debugging Triton-generated SASS code for Blackwell, focusing on TMA (Tensor Memory Accelerator), multicast operations, and warp specialization patterns.

---

## 1. Overview

This document covers debugging techniques when working with Triton kernels that compile down to SASS instructions utilizing Blackwell-specific features: TMA for asynchronous data movement, multicast for efficient data distribution across SMs in a cluster, and warp specialization for overlapping compute with memory operations.

---

## 2. Key Concepts

- **TMA (Tensor Memory Accelerator)**: Hardware unit for asynchronous bulk data transfers between global memory and shared memory, described via tensor descriptors
- **Multicast**: A single TMA load can distribute data to shared memory of multiple CTAs within the same cluster, reducing L2 cache traffic
- **Warp Specialization**: Dedicating different warps to different roles (producer warps for data loading, consumer warps for computation) to maximize pipeline overlap

---

## 3. Debugging Approach

When Triton kernels using these features produce incorrect results or suboptimal performance:

1. Dump PTX/SASS using Triton's compilation options to inspect generated instructions
2. Use Nsight Compute to verify TMA utilization, multicast efficiency, and warp scheduling
3. Check mbarrier synchronization patterns for correctness
4. Verify tensor descriptor setup matches actual memory layout
5. Inspect SMEM bank conflicts in the swizzled layouts
