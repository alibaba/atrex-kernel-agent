# NVIDIA Common Optimization Knowledge

NVIDIA GPU general optimization knowledge: Compute Capability comparison, architecture-specific optimizations, PTX ISA, NCU Profiling (DSL-agnostic).

> Complete reference articles have been migrated to [ref-docs/nvidia/common/](../../../ref-docs/nvidia/common/)

---

| File | Description |
|------|------|
| [blackwell/](blackwell/) | Blackwell / Hopper kernel optimization knowledge cards (51 cards): hardware mechanisms, typical kernels, language interfaces, migration rules, bottleneck patterns, and optimization techniques |
| [hands-on/](hands-on/) | Blackwell (SM100) Kernel Optimization Hands-on: tcgen05/TMEM, three-role warp specialization, CLC, 2CTA, etc. |
| [sm90/](sm90/) | Hopper (SM90) specific optimizations and hands-on |
| [NVIDIA Compute Capability Comparison Table](nvidia-compute-capabilities.md) | CC 7.5-12.x threads/warps/blocks/registers comparison |
| [NVIDIA Architecture-Specific Optimization Techniques (Index)](nvidia-arch-specific-optimization.md) | Index page for architecture-specific optimization techniques |
| [L2 Cache Persistence Control (CC 8.0+)](l2-cache-persistence.md) | L2 set-aside, access policy window, hitRatio tuning |
| [Asynchronous Global-to-Shared Memory Copy (CC 8.0+)](async-global-to-shared-copy.md) | memcpy_async bypassing registers, pipeline double-buffering mode |
| [Thread Block Cluster (CC 9.0+)](thread-block-cluster.md) | Distributed shared memory, TMA tensor acceleration |
| [Occupancy Tuning Differences Across Architectures](occupancy-tuning-by-arch.md) | CC 7.5-10.x block size, shared memory, register comparison |
| [Profiling Tools Across Architectures](profiling-tools-by-arch.md) | nsys/ncu tool selection and key sections |
