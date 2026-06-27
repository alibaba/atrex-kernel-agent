# Memory Bandwidth Bound

## Symptom

Nsight Compute shows high DRAM throughput but low tensor core utilization. Arithmetic intensity below the roofline knee point.

## Likely Causes

1. **Low arithmetic intensity**: Operations like GEMV, small batch decode, or reduction kernels
2. **Poor data reuse**: Each data element used only once
3. **Inefficient memory access**: Uncoalesced loads, L1 cache thrashing

## Candidate Techniques

| Technique | Effect |
|---|---|
| [Vectorized loads](../techniques/vectorized-loads.md) | 128/256-bit loads maximize bandwidth utilization |
| [Cache policies](../techniques/vectorized-loads.md) | L1::no_allocate for streaming, L1::evict_last for reuse |
| [Register budgeting](../techniques/vectorized-loads.md) | -maxrregcount increases occupancy |
| [TMA multicast](../hardware/tma.md) | Share loaded data across SMs in cluster |
| [Swizzling](../techniques/swizzling.md) | Eliminate bank conflicts in shared memory |

## Examples

```cuda
// NVFP4 GEMV: memory-bound optimization
// Key insight: profile FIRST to confirm memory-bound behavior
// "The single most important thing could have been running Nsight Compute"
// -- Amandeep (12 Attempts at an FP4 Kernel)

// Optimization priorities for memory-bound kernels:
// 1. Maximize memory bandwidth (wide loads, coalescing)
// 2. Reduce register count (higher occupancy)
// 3. Differentiate cache policies per access pattern
// 4. DON'T optimize compute (it's not the bottleneck)
```

## Caveats
- Always profile before optimizing -- wrong assumption wastes effort
- B200 has 8 TB/s bandwidth; speed-of-light calculation determines achievable performance
- ILP and compute optimizations have diminishing returns for memory-bound kernels
