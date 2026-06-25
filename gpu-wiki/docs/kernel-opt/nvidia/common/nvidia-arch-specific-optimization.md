# NVIDIA Architecture-Specific Optimization Techniques

This document indexes optimization techniques that require specific compute capability, as well as tuning differences across architectures. Detailed content for each technique has been broken out into separate documents.

| File | Technique | Minimum CC | Description |
|------|------|---------|------|
| [L2 Cache Persistence Control](l2-cache-persistence.md) | L2 Persistence | 8.0+ | Partitions L2 cache into persistent/streaming regions, keeping hot data resident |
| [Async Global-to-Shared Memory Copy](async-global-to-shared-copy.md) | `memcpy_async` | 8.0+ | DMA copy bypassing registers, supporting pipeline overlap |
| [Thread Block Cluster](thread-block-cluster.md) | Cluster + TMA | 9.0+ | Multi-block clusters, distributed shared memory with accelerated tensor data movement |
| [Occupancy Tuning Differences](occupancy-tuning-by-arch.md) | Occupancy | All architectures | Differences in block size, shared memory, and registers across CC 7.5/8.0/9.0/10.x |
| [Profiling Tools](profiling-tools-by-arch.md) | nsys / ncu | All architectures | Profiling tool selection and key metrics across different architectures |
