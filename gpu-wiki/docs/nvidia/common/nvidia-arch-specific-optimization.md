# NVIDIA Architecture-Specific Optimization Techniques

This document indexes optimization techniques that require specific compute capability, as well as tuning differences across architectures. Detailed content for each technique has been broken out into separate documents.


**Last updated**: 2026-06-30

| File | Technique | Minimum CC | Description |
|------|------|---------|------|
| [L2 Cache Persistence Control](l2-cache-persistence.md) | L2 Persistence | 8.0+ | Partitions L2 cache into persistent/streaming regions, keeping hot data resident |
| [Async Global-to-Shared Memory Copy](async-global-to-shared-copy.md) | `memcpy_async` | 8.0+ | DMA copy bypassing registers, supporting pipeline overlap |
| [Thread Block Cluster](thread-block-cluster.md) | Cluster + TMA | 9.0+ | Multi-block clusters, distributed shared memory with accelerated tensor data movement |
| [Occupancy Tuning Differences](occupancy-tuning-by-arch.md) | Occupancy | All architectures | Differences in block size, shared memory, and registers across CC 7.5/8.0/9.0/10.x |
| [Profiling Tools](profiling/profiling-tools-by-arch.md) | nsys / ncu | All architectures | Profiling tool selection and key metrics across different architectures |


## Related

- [Async Global-to-Shared Memory Copy (CC 8.0+)](async-global-to-shared-copy.md)
- [FlashAttention 1–4: GPU Generational Evolution](flash-attention-1-to-4-gpu-evolution.md)
- [FlashInfer: Efficient and Customizable Attention Engine for LLM Inference](flashinfer-efficient-attention-engine.md)
- [GPU Architecture Deep Dive](gpu-architecture-deep-dive.md)
- [Memory-Bound Kernel Optimization: Hierarchical Reduction](hierarchical-reduction-memory-bound.md)
