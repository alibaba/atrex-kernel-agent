# GPU Architecture Resource Model

Use this page before changing tile size, stage count, block size, or occupancy

**Last updated**: 2026-06-30

assumptions. The key is to separate the physical resource that bottlenecks the
kernel from the programming interface used to reach it.

## Architecture Comparison

| Resource | SM90 Hopper | SM100 Blackwell datacenter | SM120 Blackwell desktop/workstation |
|---|---|---|---|
| Tensor-core path | WGMMA | tcgen05 | closer to register-fragment tensor-core paths |
| Accumulator storage | registers | TMEM | registers |
| Shared memory | large, TMA-friendly | larger, TMA/multicast-friendly | smaller than datacenter Blackwell |
| Scheduling tool | persistent kernels, Stream-K-like schedulers | CLC plus persistent kernels | persistent kernels without datacenter cluster/TMEM assumptions |
| Narrow precision | FP8 | FP4/FP6/FP8 block-scaled paths | FP4/FP6 exists but datacenter tcgen05/TMEM assumptions do not all carry |

## Decision Rules

- If the kernel is GEMM-like and targets SM100, first ask whether a tcgen05/TMEM path exists in CUTLASS, CuTe DSL, Triton 3.6+, or a tracked PR.
- If the kernel is memory-bound, architecture differences mostly change the bandwidth/L2/occupancy tradeoff; coalescing and vectorized loads remain first checks.
- If the kernel is tail-limited, SM100 CLC may be the right primitive; on SM90/SM120 use persistent scheduling or tile splitting.
- If porting from Hopper to Blackwell, do not just enlarge tiles. Re-check accumulator location, shared-memory usage, TMA overlap, and resident blocks.

## Profiling Questions

1. Is the limiting resource tensor pipe, DRAM, L2, shared memory, registers, TMEM, or launch/scheduling?
2. Did a larger tile improve reuse while lowering active warps too far?
3. Does the shape fill full waves of SMs, or is last-wave utilization the main loss?
4. Does the architecture actually support the feature being assumed: TMEM, CLC, clusters, or the target tensor-core instruction?


## Related

- [Two-SM Cooperative MMA](2sm-cooperative.md)
- [Cluster Launch Control (CLC)](clc.md)
- [CUDA Memory Hierarchy for Kernel Optimization](cuda-memory-hierarchy.md)
- [mbarrier](mbarrier.md)
- [NVFP4 and Block-Scaled Narrow Precision](nvfp4.md)
- [Occupancy Optimization](../../../amd/common/occupancy-optimization.md)
- [Occupancy Tuning Differences Across Architectures](../../common/occupancy-tuning-by-arch.md)
- [Tensor Core from Volta to Blackwell](../../common/tensor-core-volta-to-blackwell.md)
