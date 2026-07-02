# Register Budgeting


**Last updated**: 2026-06-30

## Overview

SM occupancy is inversely proportional to registers-per-thread. For memory-bound kernels, higher occupancy = more warps to hide memory latency. `-maxrregcount` and `__launch_bounds__` force the compiler to stay within a budget.

## Pattern

```cuda
// Aggressive: 32 registers/thread → ~4 blocks per SM at 256 threads/block
__launch_bounds__(256, 4)
__global__ void gemv_memory_bound(...) {
    // Compiler will spill to local memory if needed
}

// Or via nvcc flag:
// nvcc -maxrregcount=32 -arch=sm_100a ...
```

## Compiler Tradeoffs

Lower register count → compiler may:
- Spill frequently-used values to local memory (bad)
- Recompute values instead of storing them (neutral)
- Use fewer unrolled iterations (bad for compute-bound)

For memory-bound kernels, spills can be hidden by memory latency anyway, so aggressive budgeting often wins.

## GPU Mode NVFP4 GEMV Results

| Rank | Register count | Latency |
|------|---------------|---------|
| 1 | 32 | 18.5μs |
| 3 | 45 | ~20μs |

The measurable difference between 32 and 45 registers shows occupancy dominates for memory-bound NVFP4 GEMV.

## When To Use

- Memory-bound kernels (first priority: occupancy)
- Kernels where register pressure comes from inner loop, not accumulators (TMEM handles accumulators)
- Sub-byte types with heavy decode/scale computation

## When NOT To Use

- Compute-bound GEMM (let compiler use what it needs)
- Kernels where spills to local memory would serialize


## Related

- [Cache Policy Differentiation](cache-policy.md)
- [Chunk-Based Parallelism](chunk-parallelism.md)
- [CUDA GEMM Optimization Ladder](cuda-gemm-optimization-ladder.md)
- [Double/Multi-Buffering Patterns](double-buffering.md)
- [Epilogue Fusion](epilogue-fusion.md)
- [Occupancy Optimization](../../../amd/common/occupancy-optimization.md)
- [Occupancy Tuning Differences Across Architectures](../../common/occupancy-tuning-by-arch.md)
- [Composable Kernel (CK) Architecture Overview](../../../amd/common/ck-architecture-overview.md)
