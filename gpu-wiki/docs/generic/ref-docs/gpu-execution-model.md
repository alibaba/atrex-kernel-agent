# GPU Execution Model and Thread Optimization

## SIMT Execution Model

GPUs use the SIMT (Single Instruction, Multiple Threads) model:

- **Thread**: The smallest execution unit; each thread has its own registers and program counter
- **Warp/Wavefront**: A group of threads (NVIDIA: 32 threads called a warp; AMD: 64 threads called a wavefront) that execute the same instruction in **lockstep**
- **Block**: Multiple warps form a thread block; threads within a block can communicate via shared memory
- **Grid**: Multiple blocks form a grid, corresponding to a single kernel launch

### Thread Index Calculation

```c
// 1D
int tid = blockIdx.x * blockDim.x + threadIdx.x;

// 2D
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;

// Linearized order: x changes fastest (consecutive threadIdx.x corresponds to consecutive threads)
```

Multi-dimensional indexing is only a programming convenience and does not affect performance. The key point is that threads with contiguous indices in the innermost dimension belong to the same warp/wavefront.

## Warp Divergence

When threads within the same warp take different branch paths, the GPU must execute each path serially — all threads execute all paths, but only threads that satisfy the condition commit their results.

### Optimization Principles

```c
// ❌ Thread-level divergence: Threads within same warp take different paths
if (threadIdx.x % 2 == 0) {
    do_something();
} else {
    do_other();
}

// ✅ Warp-level divergence: Entire warp takes same path
if (threadIdx.x / 32 % 2 == 0) {  // Branch by warp
    do_something();
} else {
    do_other();
}
```

### Branch Predication

The compiler uses **predicated execution** instead of branching for short conditional code blocks:
- All threads execute both paths, but only results from threads satisfying the condition are committed
- Avoids the overhead of branch instructions
- Only applicable when the conditional body is very short (a few instructions)

## Occupancy Optimization

Occupancy = Active warps / Maximum warps supported by an SM

### Influencing Factors

| Resource | Impact |
|------|------|
| Registers per thread | More registers → fewer warps can fit |
| Shared memory per block | More shared memory → fewer blocks can fit |
| Block size | A block that is too large or too small may waste SM capacity |

### Guidelines for Choosing Block Size

1. **Block size must be a multiple of the warp/wavefront size** (NVIDIA: 32, AMD: 64)
2. Recommended starting point: **128 or 256 threads per block**
3. Using multiple smaller blocks is better than one large block — it improves SM utilization
4. The grid size should be large enough to cover all SMs

### Using APIs for Automatic Calculation

Each GPU platform provides occupancy calculation APIs that can automatically recommend optimal block sizes based on the kernel's register and shared memory usage. It is recommended to use these APIs during tuning rather than guessing manually.

### Higher Occupancy Is Not Always Better

High occupancy helps hide memory latency, but in some scenarios:
- Moderate occupancy + more registers → better instruction-level parallelism → higher performance
- The key is balancing TLP (Thread-Level Parallelism) and ILP (Instruction-Level Parallelism)

## Latency Hiding

GPUs hide various types of latency through massive parallel threads:

| Latency Type | Typical Cycles | Hiding Method |
|----------|-----------|----------|
| Arithmetic instruction latency | ~10 cycles | ILP (other independent instructions in the same thread) |
| Global memory latency | 400-800 cycles | TLP (switching to other warps) |
| Shared memory latency | ~5 cycles | ILP or TLP |
| Branch latency | ~20 cycles | TLP |

**Active warps needed** ≈ Latency × Throughput. For example, 800 cycles of memory latency with 1 memory instruction per cycle → approximately 800 / 32 ≈ 25 warps needed to fully hide the latency.

## Concurrent Kernel Execution

Multiple kernels can execute concurrently through different streams (command queues), suitable for scenarios where a single kernel cannot fully utilize the GPU.

Key concepts:
- **Stream**: An ordered GPU command queue; operations within the same stream execute sequentially
- **Multi-Stream Concurrency**: Operations across different streams can overlap in execution
- **Default Stream**: Usually implicitly synchronizes with all other streams; care must be taken to avoid accidental serialization

### GPU Context Management

- Avoid creating multiple compute contexts on the same GPU — this leads to time-slice rotation and context switching overhead
- Use a single context + multiple streams Westminster to achieve concurrency

## Related Documents

- **Same-tier Complementary**: [GPU Memory Hierarchy and Optimization](gpu-memory-hierarchy.md) — Together with this document, forms the Tier 0 foundational knowledge
- **Advanced**: [GPU Instruction-Level Optimization](gpu-instruction-optimization.md) — Extension of warp divergence optimization techniques
- **NVIDIA Specific**: [NVIDIA Compute Capability Reference Table](../../nvidia/common/kernel-opt/nvidia-compute-capabilities.md) — Specific warp/block/SM limits for each CC
- **AMD Specific**: [Occupancy Optimization](../../amd/common/kernel-opt/occupancy-optimization.md) — AMD VGPR→occupancy mapping table
- **⚠️ Difference Note**: This document recommends block sizes of 128-256, but Hopper requires multiples of 128 (wgmma), and AMD recommends 1024+ (64-thread wavefront)
