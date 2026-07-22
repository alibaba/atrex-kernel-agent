# GPU Memory Hierarchy and Optimization

The core of GPU performance optimization is understanding the memory hierarchy and properly leveraging the bandwidth and latency characteristics of each storage level.

## Memory Types Overview

| Memory Type | Location | Scope | Lifetime | Latency | Bandwidth | Typical Size |
|-------------|----------|-------|----------|---------|-----------|--------------|
| Registers | On-chip (SM) | Thread-private | Kernel | Lowest (~1 cycle) | Highest | 255 per thread |
| Shared Memory | On-chip (SM) | Shared within block | Kernel | Low (~5 cycles) | High | 64–228 KB/SM |
| L1 Cache | On-chip (SM) | Automatic | Automatic | Low | High | Shared unified cache with shared memory |
| L2 Cache | On-chip (global) | Global | Automatic | Medium | Medium | Several MB |
| Global Memory | Off-chip (HBM/GDDR) | Global | Application lifetime | High (~400–800 cycles) | Depends on VRAM | Tens of GB |
| Local Memory | Off-chip (same as global) | Thread-private | Kernel | High | Same as global | 512 KB/thread |
| Constant Memory | Off-chip + cache | Global, read-only | Application lifetime | Low (cache hit) | High (broadcast) | 64 KB |
| Texture Memory | Off-chip + cache | Global, read-only | Application lifetime | Low (cache hit) | High (2D locality) | — |

## Register Optimization

Registers are the fastest storage, but their quantity is limited (typically 64K 32-bit registers per SM).

### Register Pressure Management

- The more registers each thread uses, the fewer threads can run simultaneously on an SM (reducing occupancy)
- Variables exceeding the limit will **spill** to local memory (off-chip), severely impacting performance
- Register usage can be limited via compiler options: `--maxrregcount=N` or `__launch_bounds__(maxThreadsPerBlock)`
- Moderately reducing register usage → more blocks can be scheduled → higher occupancy → may offset spilling overhead

### Register Dependency Latency

- Instruction-Level Parallelism (ILP): Multiple independent instructions within a single thread can be pipelined, hiding register read/write latency
- Thread-Level Parallelism (TLP): A sufficient number of active warps can also hide latency (warp switching masks wait times)

## Global Memory Optimization

Global memory is the largest but slowest storage. The key to optimization is **memory coalescing**.

### Coalesced Access

The GPU accesses global memory in units of **32-byte transactions**. Memory requests from a warp (32 threads) are combined into the minimum number of transactions.

**Best case**: Adjacent threads access adjacent addresses
- 32 threads each read 4 bytes → 128 bytes → 4 × 32-byte transactions → **100% utilization**

**Worst case**: Adjacent threads access addresses with a stride ≥ 32 bytes
- Each thread triggers an independent transaction → 1024 bytes transferred but only 128 bytes used → **12.5% utilization**

```c
// ✅ Coalesced access: Adjacent threads read adjacent elements
int idx = blockIdx.x * blockDim.x + threadIdx.x;
float val = data[idx];

// ❌ Non-coalesced access: Strided access
int idx = (blockIdx.x * blockDim.x + threadIdx.x) * stride;
float val = data[idx];  // stride > 1 wastes bandwidth
```

### Alignment Requirements

- GPU memory allocation APIs typically guarantee at least 256-byte alignment
- When the starting address is misaligned, the first and last transactions may only utilize partial bytes
- For structs, alignment modifiers are recommended to ensure 8- or 16-byte alignment

### Avoiding Strided Access

With a stride of 2, utilization is only 50%; larger strides are even worse. Solutions:
- Use shared memory as an intermediary to convert irregular access patterns into coalesced accesses
- Reorganize data layout: AoS (Array of Structures) → SoA (Structure of Arrays)

## Shared Memory Optimization

Shared memory is on-chip programmable cache with bandwidth far exceeding that of global memory.

### Typical Use Cases

1. **Caching global data**: When multiple threads within a block reuse the same data, load it into shared memory first
2. **Fixing non-coalesced access**: Use shared memory as a staging buffer to rearrange access patterns
3. **Inter-thread communication**: Threads within a block exchange data through shared memory (requires block-level synchronization barrier)

### Bank Conflict

Shared memory is divided into **32 banks**, with consecutive 32-bit words mapped to consecutive banks.

- **No conflict**: Threads within the same warp access different banks, or all threads access the same address (broadcast)
- **Conflict**: Multiple threads access different addresses in the same bank → serialization

```c
// ❌ 32-way bank conflict: Column access of 32×32 array
__shared__ float s[32][32];
float val = s[threadIdx.x][0];  // All threads access bank 0

// ✅ Eliminate bank conflict: Padding
__shared__ float s[32][33];  // Extra 1 column padding, stagger bank mapping
float val = s[threadIdx.x][0];  // Each thread accesses different bank
```

### Balancing Shared Memory and Occupancy

The total shared memory per SM is limited. The more shared memory a single block consumes, the fewer blocks can reside simultaneously. A balance must be struck between data reuse benefits and the resulting occupancy decrease.

## Local Memory

The name is misleading—local memory actually resides in **off-chip device memory**, with latency and bandwidth identical to global memory.

The compiler places variables in local memory under the following conditions:
- Array indices cannot be determined as constants at compile time
- Large structs or arrays exceed register capacity
- Register spill

The latency of local memory can be partially mitigated through L1/L2 caches.

## Constant Memory

- Total 64 KB, globally read-only
- Ideal for scenarios where all threads read the same value (broadcast within a warp, a single read suffices)
- When threads within a warp read different addresses, accesses are serialized; in such cases, global memory is preferable

## Host-Device Data Transfer Optimization

### Reducing Transfer Volume

PCIe bandwidth (~16 GB/s) is far lower than device memory bandwidth (hundreds of GB/s to several TB/s).
- Keep intermediate results on the GPU to avoid back-and-forth transfers
- Even if certain computations are faster on the CPU, the overhead of frequent transfers may be greater

### Using Pinned Memory

Page-locked (pinned) memory offers higher transfer speeds than regular pageable memory and supports asynchronous transfers. However, it should not be over-allocated — it occupies non-pageable system memory.

### Overlapping Async Transfers with Computation

Use asynchronous memory copies and multiple streams to overlap data transfers with kernel execution:

```
Pseudo-code:
for each stream i:
    async_copy(host_data[i] → device_data[i], stream[i])
    launch_kernel(device_data[i], stream[i])
    async_copy(device_result[i] → host_result[i], stream[i])
```

By splitting data into chunks and assigning them to different streams, you can achieve pipelined overlap between transfers and computation, fully utilizing the DMA engine and compute units.

## Effective Bandwidth as a Performance Metric

Key metric for measuring memory optimization effectiveness:

```
Effective bandwidth (GB/s) = (bytes read + bytes written) / time / 10^9
```

Compare the effective bandwidth against the hardware's theoretical peak bandwidth to gauge how well the kernel's memory usage is optimized. The closer the ratio is to 1, the more thoroughly optimized it is.

## Related Documentation

- **Same-layer complementary**: [GPU Execution Model and Thread Optimization](gpu-execution-model.md) — Together with this document, forms Tier 0 foundational knowledge
- **Content overlap**: [GPU Application-Level Optimization Strategies](gpu-application-optimization.md) — The host-device transfer section is more comprehensive
- **NVIDIA-specific**: [NVIDIA Architecture-Specific Optimization Techniques](../../nvidia/common/kernel-opt/nvidia-arch-specific-optimization.md) — L2 persistence, TMA, and other NVIDIA-specific memory technologies
- **AMD-specific**: [LDS Bank Conflict Optimization](../../amd/common/kernel-opt/lds-bank-conflict-optimization.md) — LDS bank conflicts, XOR swizzle
- **🔴 Conflict Note**: This document claims shared memory has 32 banks (generic), but AMD CDNA4 actually has **64 banks**. See [Hardware Specification Comparison](../../amd/common/hardware-specs/hardware-comparison-cdna3-cdna4.md)
