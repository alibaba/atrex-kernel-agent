# CUDA Memory Hierarchy for Kernel Optimization

For kernel work, treat memory as a set of bottleneck candidates rather than a

**Last updated**: 2026-06-30

generic hierarchy. The likely next edit depends on which level is stressed.

## What To Inspect

| Symptom | Likely memory level | First edits |
|---|---|---|
| High DRAM throughput, low tensor/SM pipe | global memory | coalescing, vectorized loads, fewer bytes, fusion |
| Many global sectors per useful element | global memory | layout fix, contiguous access, packed load/store |
| Shared bank conflicts | shared memory | padding, swizzle, transpose layout, warp partitioning |
| Long scoreboard / async waits | global-to-shared pipeline | stage count, TMA overlap, barrier placement |
| Low occupancy from registers | register file | launch bounds, smaller per-thread tile, move to TMEM path on SM100 |
| Large accumulator tile on SM100 | TMEM | use tcgen05/TMEM path instead of register-resident accumulation |

## Minimal Debug Snippet

```bash
ncu --section MemoryWorkloadAnalysis --section Occupancy --section SpeedOfLight --kernel-name "kernel" ./bench
cuobjdump -res-usage ./bench
```

Use the report to classify the next edit. Do not change tile size, cache policy,
and vector width in the same attempt unless the ledger records the interaction.

## Blog-Code Anchors

- `blog-simon-boehm-sgemm`: progression from uncoalesced global access to vectorized loads, bank-conflict fixes, and double buffering.
- `blog-lei-mao-cuda-gemm`: staged GEMM variants with explicit coalescing and tiling code.
- `blog-veitner-cuda-cute`: transpose/swizzle code paths that isolate shared-memory layout effects.


## Related

- [Two-SM Cooperative MMA](2sm-cooperative.md)
- [Cluster Launch Control (CLC)](clc.md)
- [GPU Architecture Resource Model](gpu-architecture-resource-model.md)
- [mbarrier](mbarrier.md)
- [NVFP4 and Block-Scaled Narrow Precision](nvfp4.md)
- [GPU Memory Hierarchy and Optimization](../../../generic/gpu-memory-hierarchy.md)
- [Composable Kernel (CK) Architecture Overview](../../../amd/common/ck-architecture-overview.md)
