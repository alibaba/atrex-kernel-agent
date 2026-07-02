# GPU Application-Level Optimization Strategies


**Last updated**: 2026-06-30

## Parallelization Assessment

### Amdahl's Law (Strong Scaling)

Maximum speedup S = 1 / ((1 - P) + P/N)

- P = parallelizable fraction
- N = number of processors
- Even as N → ∞, speedup limit = 1/(1-P)
- Insight: **Maximize the parallelizable fraction P first**

### Gustafson's Law (Weak Scaling)

S = N + (1 - P)(1 - N)

- Problem size grows linearly with the number of processors
- The serial portion decreases proportionally as problem size increases
- More suitable for evaluating large-scale parallel scenarios such as GPUs

## Data Transfer Optimization

### Minimizing Host-Device Transfers

Host-device interconnect bandwidth is far lower than device memory bandwidth:

| Channel | Typical Bandwidth |
|----------|----------|
| PCIe Gen3 x16 | ~16 GB/s |
| PCIe Gen4 x16 | ~32 GB/s |
| PCIe Gen5 x16 | ~64 GB/s |
| GPU Interconnect (varies by generation) | 50-900 GB/s |
| HBM (Device Memory) | 2-5 TB/s |

**Core Principle**: Keep data on the GPU as much as possible

- Avoid transferring data back to the CPU between kernels
- Even if a computation step runs faster on the CPU, the transfer overhead may be greater
- Consider moving the entire pipeline to the GPU

### Transfer Optimization Techniques

1. **Batched Transfers**: Merge small transfers into larger ones to reduce launch overhead
2. **Pinned Memory**: Page-locked memory can be 2-3x faster than pageable memory for transfers
3. **Asynchronous Transfers**: Use multiple streams to overlap transfers and computation

## Memory Allocation Strategy

### Avoid Frequent Allocation/Deallocation

GPU memory allocation/deallocation is a heavyweight operation:

- Pre-allocate and reuse
- Use memory pools for sub-allocation
- Consider using stream-ordered memory allocators to reduce synchronization overhead

### NUMA Awareness

On multi-socket systems, the NUMA affinity of CPU threads should match the PCIe topology where the GPU is located; otherwise, Host-Device transfer bandwidth will degrade.

## Leverage Existing Libraries

Before writing custom kernels, prioritize using vendor-optimized libraries. All GPU platforms provide optimized libraries in the following categories:

| Category | Use Case |
|----|------|
| BLAS | Linear algebra (GEMM, etc.) |
| DNN | Deep learning primitives (convolution, normalization, etc.) |
| FFT | Fast Fourier Transform |
| Sparse | Sparse matrix operations |
| Parallel Algorithm Primitives | scan, reduce, sort, etc. |
| Customizable GEMM | Matrix multiplication with custom epilogue support |

## Kernel Design Patterns

### Tiling

Decompose large problems into sub-problems that fit within shared memory:

```
1. Each block is responsible for one tile of the output
2. Load the input tile into shared memory
3. Compute on shared memory
4. Write results back to global memory
```

Benefits:
- Improved data reuse
- Reduced global memory accesses
- Enables coalesced access

### Reduction

Standard pattern for parallel reduction:

```
1. Thread-level: Each thread accumulates multiple elements (reduces thread count)
2. Warp-level: Use warp shuffle instructions (no shared memory needed)
3. Block-level: Cross-warp reduction through shared memory
4. Grid-level: Atomic operations or multi-pass reduction
```

### Fusion (Operator Fusion)

Fuse multiple kernels into one, reducing:
- Kernel launch overhead
- Global memory reads/writes (intermediate results kept in registers/shared memory)
- Memory bandwidth pressure

## Debugging and Validation

### Numerical Accuracy Verification

- GPU floating-point computations may produce slight differences from CPU results due to FMA, instruction reordering, etc.
- It is recommended to write a CPU reference implementation for result comparison and verification
- Pay attention to precision degradation in FP16/BF16 mixed-precision scenarios

### Error Checking

Production code must check return values of every GPU API callega, and check for asynchronous errors after kernel launches. Asynchronous errors are easily overlooked in GPU programming; it is recommended to encapsulate unified error-checking macros.

## Deployment Recommendations

### Forward Compatibility

- Include intermediate representation (IR) at compile time to support future architectures
- Also include native binaries for the target architecture charms to achieve optimal performance
- Use multi-architecture binaries to support multiple GPUs simultaneously

### GPU Compute Mode

Most GPUs support configuring compute modes:
- **Shared Mode**: Multiple processes can share the GPU
- **Exclusive Mode**: A single process monopolizes the GPU, commonly used in cluster environments to avoid resource contention
- Configure via platform-specific management tools

## Related

- **Prerequisites**: [GPU Execution Model](gpu-execution-model.md) + [GPU Memory Hierarchy](gpu-memory-hierarchy.md)
- **Content Overlap**: The host-device transfer section overlaps with `gpu-memory-hierarchy.md`, and this document is more comprehensive
- **Advanced**: [GPU Instruction-Level Optimization](gpu-instruction-optimization.md) — Instruction-level optimization after operator fusion
