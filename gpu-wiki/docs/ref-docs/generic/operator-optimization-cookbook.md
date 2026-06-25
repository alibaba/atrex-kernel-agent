# Community Common Operator Optimization Manual

Comprehensive core knowledge from the Zhihu community on optimizing common GPU operators (softmax, LayerNorm, reduce, etc.). Supplements CUDA-level operator optimization details not covered in [Online Softmax & Flash Attention Hands-on](../../kernel-opt/generic/hands-on/online-softmax-flash-attention.md) and [Fused Kernel Patterns Hands-on](../../kernel-opt/generic/hands-on/fused-kernel-patterns.md).

> **Source Note**: This document synthesizes core knowledge from approximately 19 related articles on the Zhihu community, after deduplication, filtering, and structured organization.

---

## Table of Contents

- [1. Reduce Optimization](#1-reduce-optimization)
- [2. Softmax Optimization](#2-softmax-optimization)
- [3. LayerNorm / RMSNorm Optimization](#3-layernorm-rmsnorm-optimization)
- [4. Transpose / Data Reordering Optimization](#4-transformose-data-rearrangement-optimization)
- [5. Elementwise Optimization](#5-elementwise-optimization)
- [6. Other Operator Optimization](#6-other-operator-optimizations)
- [7. General Optimization Principles Quick Reference](#7-general-optimization-principles-quick-reference)
- [Related Documents](#related-documents)

---

## 1. Reduce Optimization

Reduce is one of the most fundamental and critical operations on GPUs. Nearly all operators that involve summation and maximum value computation depend on it. Reduce is a typical **memory-bound** operation, and the optimization goal is to approach the theoretical limit of memory bandwidth.

### 1.1 Classic Seven-Step Optimization Path

Taking the reduce sum of 32M float data on V100 as an example, the complete path from baseline to extreme optimization:

| Optimization Step | Bandwidth (GB/s) | Time (ms) | Key Improvement |
|---------|------------|----------|---------|
| Baseline | 166.7 | 3.23 | Naive implementation with warp divergence |
| Eliminate warp divergence | 232.7 | 2.32 | Thread index remapping so adjacent threads follow the same branch |
| Eliminate bank conflict | 239.7 | 2.25 | Reverse stride traversal to ensure threads within a warp access different banks |
| Utilize idle threads | 463.9 | 1.16 | Do an extra addition per thread during load, halving the number of blocks |
| Unroll within warp | 802.8 | 0.67 | The last 32 threads do not need `__syncthreads()` |
| Full loop unrolling | 798.2 | 0.672 | Eliminate for-loop overhead (marginal benefit on V100) |
| Shuffle instructions | 863.4 | 0.619 | Use `__shfl_down_sync` instead of shared memory communication |

Ultimately reaching **858 GB/s** on V100, approaching the HBM2 theoretical bandwidth of 897 GB/s (~95% bandwidth utilization).

### 1.2 Key Optimization Techniques in Detail

**Eliminate Warp Divergence**: Map thread indices into a contiguous block of active threads, rather than using modulo to produce alternating active/idle threads.

```cuda
// Bad: tid % (2*s) == 0 causes threads within warp to take different paths
// Good: Use consecutive thread indices
int index = 2 * s * tid;
if (index < blockDim.x) {
    sdata[index] += sdata[index + s];
}
```

**Eliminate Bank Conflict**: Traverse stride in reverse (from blockDim.x/2 down to 1), ensuring threads within the same warp access different banks.

```cuda
// Good: Reverse stride, no bank conflict
for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
        sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
}
```

**Warp-Level Shuffle Reduce**: Use `__shfl_down_sync` to exchange data directly through registers within a warp, avoiding shared memory access.

```cuda
__device__ __forceinline__ float warpReduceSum(float sum) {
    sum += __shfl_down_sync(0xffffffff, sum, 16);
    sum += __shfl_down_sync(0xffffffff, sum, 8);
    sum += __shfl_down_sync(0xffffffff, sum, 4);
    sum += __shfl_down_sync(0xffffffff, sum, 2);
    sum += __shfl_down_sync(0xffffffff, sum, 1);
    return sum;
}
```

**Two-Level Reduce Pattern (Warp + Block)**: First perform shuffle reduce within each warp, then write each warp's result to shared memory, and use the first warp Grac final reduction.

```cuda
// Level 1: Reduce within warp
float sum = warpReduceSum(val);
if (laneId == 0) warpLevelSums[warpId] = sum;
__syncthreads();
// Level 2: Reduce across warps
if (tid < 32) {
    sum = (tid < blockDim.x / 32) ? warpLevelSums[tid] : 0.f;
    sum = warpReduceSum(sum);
}
```

### 1.3 Thread Coarsening

Reducing the number of blocks and having each thread perform partial reduction during the load phase is an important means of improving reduce performance. In LeetGPU's Reduce benchmark, thread coarsening delivered a **3.94x** speedup (0.28ms -> 0.071ms), far surpassing other techniques.

Core idea: During the prefetch phase, have each thread process multiple elements using a grid-stride loop, reducing them in advance to the shared memory size.

```cuda
float sum = 0;
for (int i = tx; i < ITEMS_PER_BLOCK; i += BLOCK_SIZE_N) {
    sum += (b_base + i < N) ? input[b_base + i] : 0.0f;
}
s_input[tx] = sum;  // Partial reduction already done
```### 1.4 ncu Metric Diagnosis

- `Avg. Not Predicated Off Threads Per Warp`: Measures the effective compute ratio of threads, with an ideal value of 32. If it is far below 32, it indicates a large number of idle threads.
- `smsp__average_warp_latency_issue_stalled_dispatch_stall.pct`: Warp dispatch stall ratio; a high value indicates excessive synchronization operations.
- When LSU (Load/Store Unit) utilization is too high, consider reducing the number of shared memory reads/writes and replacing them with register variables.

---

## 2. Softmax Optimization

Softmax is an extremely high-frequency operator in large models. Its core consists of two reduce operations (finding max and sum) and one elementwise operation (normalization), making it a typical memory-bound operation.

### 2.1 From 0.44ms to 0.04ms: 10x Optimization on A100

Taking FP32 softmax with shape=[49152, 128] on A100 as an example:

| Kernel Version | Time (ms) | Bandwidth (GB/s) | Bandwidth Utilization | Cumulative Speedup |
|------------|----------|------------|----------|---------|
| Naive Baseline (single thread per row) | 0.4419 | 113.9 | 7.3% | 1.0x |
| Shared Memory + Tree Reduction | 0.3274 | 153.7 | 9.9% | 1.35x |
| Warp Shuffle (1D block) | 0.0852 | 590.4 | 38.0% | 5.19x |
| Warp Shuffle (2D block) | 0.0585 | 860.8 | 55.4% | 7.55x |
| Multi-Row Per Warp | 0.0447 | 1124.8 | 72.3% | 9.88x |

### 2.2 Key Optimization Techniques

**Warp Shuffle Replacing Shared Memory**: The largest single optimization leap for softmax (3.84x). Warp shuffle transfers data directly between registers with a latency of about 1 cycle, whereas the shared memory approach requires approximately 20 cycles (write + synchronize + read).

**2D Block Improves Occupancy**: With a 1D block of only 32 threads, occupancy is limited to 50% due to the hardware limit of at most 32 blocks per SM. After switching to `dim3(32, 4)`, each block has 128 threads and occupancy improves to 100%. This change requires modifying only 2 lines of code, yet yields a **1.46x** speedup.

```cuda
// 1D: int row_idx = blockIdx.x;
// 2D: Each warp (threadIdx.y) handles different rows
int row_idx = blockIdx.x * blockDim.y + threadIdx.y;
```

**Multi-Row Processing Utilizing Registers**: The A100 allows up to 256 registers per thread, while a simple softmax uses only about 8. Let each warp process 4 rows simultaneously using a 2D register array `float buf[4][4]`, leveraging ILP (Instruction-Level Parallelism) to pipeline computation across different rows.

### 2.3 Segmented Function Strategy

OneFlow's softmax optimization adopts a segmented strategy, selecting different implementations based on `num_cols`:

| num_cols Range | Strategy | Data Cache Location |
|-------------|------|-----------|
| <= 1024 | Warp-level processing, each warp handles 1-2 rows | Registers |
| 1024 ~ Shared Memory upper limit | Block-level processing | Shared Memory |
| Very large | Block-level processing, repeated reads of x | L2 Cache (uncached) |

**Performance Watershed**: When dim < 1024, Warp Shuffle dominates; when dim >= 1024, the Shared Memory approach dominates. In practice, the implementation should be adaptively selected based on dim.

### 2.4 Safe Softmax + Float4 Vectorization

Key points for safe softmax implementation, often tested in interviews: first perform block reduce max, then exp and block reduce sum, and finally normalize. Combining with float4 vectorization can reduce the instruction count by a factor of 4.

```cuda
// float4 vectorized safe softmax core logic
// First compute local max within thread
// Compute exp and sum
// ... Similar processing for y, z, w
// Normalize output
```

---

## 3. LayerNorm / RMSNorm Optimization

LayerNorm and RMSNorm are core normalization operators in large models. Their optimization methods are highly consistent with softmax: both perform reduce operations on each row of data charms, making them memory-bound operators.

### 3.1 Welford Algorithm

There are three methodsega to compute variance:

| Method | Number of Passes | Numerical Stability | Applicability |
|------|--------|----------|---------|
| Two-pass | 2 | Good | Small data sizes |
| Naive (accumulating x^2) | 1 | Poor (potential precision loss) | Not recommended |
| Welford | 1 | Good | **Recommended**, standard approach in modern frameworks |

The core of the Welford algorithm lies in incrementally updating the mean and variance, avoiding precision loss from subtracting large numbers:

```
M_n = M_{n-1} + (x_n - M_{n-1}) / n
S_n = S_{n-1} + (x_n - M_{n-1}) * (x_n - M_n)
variance = S_n / n
```

Welford can be combined with warp shuffle (WelfordWarpAllReduce) to merge statistics from different threads during reduction.

### 3.2 OneFlow's Three-Tier LayerNorm Implementation

OneFlow's LayerNorm comprehensively outperforms NVIDIA Apex and PyTorch on A100. Its core strategy is segmented functions + vectorized memory access:**num_cols <= 1024 (Warp-level)**: Each warp processes 1 row, with data cached in registers. When num_cols is very small, use half-warp (thread_group_width=16/8/4) and have each thread process 2 rows to increase parallelism. The key template parameter `pack_size` controls vectorized read/write granularity.

**num_cols > 1024 (Block + Shared Memory)**: Each block processes 1 row, with data cached in shared memory. Use `cudaOccupancyMaxActiveBlocksPerMultiprocessor` to determine whether launch is possible. The principle for selecting block_size: make it as large as possible without reducing the number of blocks that an SM can schedule.

**Very large num_cols (Block without caching)**: Do not use shared memory; x is read twice (the second pass relies on L2 cache). Use a larger block_size to reduce the number of concurrent blocks per SMega, improving cache hit rate.

### 3.3 RMSNorm Optimization

RMSNorm is simpler than LayerNorm (no needym to compute the mean, only the sum of squares), but the optimization approach is entirely consistent. Key implementations:

1. **Naive**: Shared memory tree reduction
2. **Shared Memory + Cached x**: Avoid repeated global memory reads
3. **Warp Reduce**: Use `__shfl_down_sync` instead of tree reduction
4. **Float4 Vectorization**: Read/write 4 elements at a time

// RMSNorm warp-level + float4 core logic
// ... cross-warp reduction
// Output: y = x * rms_ * weight

---

## 4. Transformose / Data Rearrangement Optimization

Matrix transformose appears simple, but on GPUs it is a classic problem that tests memory access optimization skills: coalesced reads mean scattered writes, and coalesced writes mean scattered reads.

### 4.1 Optimization Path and Performance Data

Using a 8192x2048 float matrix tested on an RTX 5060 (theoretical bandwidth 384 GB/s):

| Version | Bandwidth (GB/s) | Speedup vs PyTorch |
|------|------------|----------------|
| PyTorch torch.transpose | 118.5 | 1.00x |
| Naive coalesced read | 258.6 | 2.18x |
| Shared Memory relay | 279.3 | 2.36x |
| Smem + Padding to elimite bank conflict | 301.1 | 2.54x |
| Smem + Float4 vectorization + Padding | 320.5 | 2.71x |
| Smem + Float4 + XOR Swizzle | 323.6 | **2.73x** |

The final version achieves 84.2% of theoretical bandwidth, with ncu showing compute memory throughput reaching 92%.

### 4.2 Shared Memory Relay (Corner Turn)

Core idea: Use shared memory as a staging area to achieve "coalesced read + coalesced write".

```cuda
// 1. Coalesced read global -> shared (by row)
// 2. Swap x/y coordinates
// 3. Coalesced write shared -> global (by row, but physically columns of original matrix)
```

### 4.3 Bank Conflict Resolution: Padding vs Swizzle

**Padding**: Change `tile[32][32]` to `tile[32][33]`, adding one extra element per row so that column accesses stagger across banks. Simple but wastes space and breaks 128-bit alignment.

**XOR Swizzle**: Use the address transform of `(row, col) -> (row, col ^ row)` to scramble storage locations. Does not waste space, maintains alignment, and is the standard approach used by high-performance libraries like CUTLASS.

```cuda// Swizzle when writing
// Reverse swizzle when reading```

Mathematical proof of XOR swizzle's effectiveness: For a fixed constant C, the mapping f(x) = x XOR C is a bijection (one-to-one mapping), so 32 distinct row numbers map to 32 distinct banks, with no conflicts.

### 4.4 In-thread 4x4 Transformose (No Shared Memory)

Another approach: Each thread reads 4 elements from each of 4 rows using float4, performs a 4x4 small matrix transformose within registers, then writes out using float4. With appropriate block configuration, both reads and writes can be coalesced, achieving bandwidth of 490+ GB/s (RTX 4080).

The key is that the block configuration must ensure memory coalescing during writes. Analysis shows that block_width must be an even value <= 16.

---

## 5. Elementwise Optimization

Elementwise operations (sigmoid, silu, add, etc.) appear to be the simplest, but they too have significant optimization potential.

### 5.1 Float4 Vectorization

The most critical optimization technique. A single 128-bit load/store instruction processes 4 floats, reducing the instruction count by 4x.

```cuda
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])

__global__ void sigmoid_vec4(float* x, float* y, int N) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx < N) {
        float4 reg_x = FLOAT4(x[idx]);
        float4 reg_y;
        reg_y.x = 1.0f / (1.0f + expf(-reg_x.x));
        reg_y.y = 1.0f / (1.0f + expf(-reg_x.y));
        reg_y.z = 1.0f / (1.0f + expf(-reg_x.z));
        reg_y.w = 1.0f / (1.0f + expf(-reg_x.w));
        FLOAT4(y[idx]) = reg_y;
    }
}
```### 5.2 Kernel Fusion (Example: RoPE)

RoPE (Rotary Position Embedding) is a standard operator in large models. A naive PyTorch implementation requires multiple reads and writes to global memory (roughly 11× data volume), whereas a hand-written fused kernel only needs 1 read and 1 write (2× data volume), reducing ineffective memory access by 80%+.

A key technique in fused kernels—"compute instead of read": compute sin/cos on the fly inside the kernel (using `__sincosf` hardware instructions), rather than reading precomputed trigonometric function tables from global memory.

On the RTX 5060, the hand-written RoPE kernel achieves an effective bandwidth of 325 GB/s (physical limit 337 GB/s, utilization **96.4%**), and ncu shows compute memory throughput of 97%. Compared to the version optimized by torch.compile, it still has a several-fold advantage.

### 5.3 Numerical Stability Considerations

Activation functions like SiLU need to watch out for numerical overflow. Recommended sigmoid implementation:

```cuda
__device__ __forceinline__ float sigmoid(float x) {
    if (x >= 0.0f) {
        return 1.0f / (1.0f + __expf(-x));
    } else {
        float exp_x = __expf(x);
        return exp_x / (1.0f + exp_x);
    }
}
```

---

## 6. Other Operator Optimizations

### 6.1 Embedding / EmbeddingBag

PyTorch's `torch.nn.EmbeddingBag` has two subtle performance issues:

1. **CUDA_KERNEL_ASSERT causing register bloat**: Input bounds-checking code increases registers from 48 to 74, reducing occupancy from 62.5% to 37.5%. Solution: extract parameter checking into a separate kernel (takes only about 3 us).
2. **Unreasonable block configuration**: The underlying index_select operator of `torch.nn.Embedding` has its block upper limit set to `sm_count * 8`, resulting in only 48.5% occupancy.

Optimization results (A800, batch=2048, bag=150, dim=128):

| Version | GPU kernel time |
|---------|-----------------|
| PyTorch official | 383.4 us |
| Optimized | 144.2 us |

### 6.2 Sort (Radix Sort)

Optimization of `torch.sort` for sorting 100 million int32 integers (A800): 14.2 ms -> **6.4 ms** (2.2× speedup). Three optimization points:

1. **Eliminate OpaqueType**: PyTorch uses `OpaqueType<8>` as a replacement for int64 to reduce template instantiation, but the nvcc compiler generates inefficient 16-bit load instructions for OpaqueType. After switching to the original type directly, occupancy improved from 19% to 56%, and registers dropped from 101 to 49.
2. **Eliminate redundant copies**: RadixSort is not an in-place sort, so there is no need to pre-copy input to output.
3. **Optimize indices type**: int64 -> int32 (in most cases, N < 2 billion).

### 6.3 GEMV (Matrix-Vector Multiplication)

Optimization points for when GEMM degenerates into GEMV during the LLM decode phase:

- Each warp handles one row of the matrix, with reduction performed within the warp
- When N is large, use float4 vectorized reads; a single warp can consume 128 elements at once
- When N is small (< 32), split multiple rows within one warp (`ROW_PER_WARP`)

### 6.4 Conv2d

The foundation of convolution optimization is clarifying the mapping relationships among 6 layers of loops (output_channel, height, width, input_channel, kernel_h, kernel_w), as well as the correspondence of input and output memory. Further optimization typically involves converting to im2col + GEMM calling cuBLAS, or directly using cuDNN.

### 6.5 Scan (Prefix Sum)

GPU optimization path for prefix sum:
1. Naive: compute directly on global memory
2. Shared memory: reduce global memory access
3. Double buffer: eliminate shared memory read/write contention
4. Warp scan + block scan: use `__shfl_up_sync` to perform prefix sum within a warp, then merge across warps
5. Thread coarsening: each thread processes `batchSize` elements, reducing the number of blocks

### 6.6 Integer-Type Operators

**Low-precision integer optimization**: If the value range allows, using int32 instead of int64 can yield significant speedup. For example, `torch.unique` (based on RadixSort) is about 43% faster under int32 than int64 (84 us -> 48 us), because radix sort complexity is linearly related to the number of bits.

**TensorFlow pitfalls**: Some TF operators (e.g., `tf.greater_equal`, `tf.equal`) do not support int32 GPU kernels and will silently fall back to CPU execution, incurring D2H/H2D copy overhead.

---

## 7. General Optimization Principles Quick Reference

### 7.1 Performance Measurement

Don't just focus on "how many times faster over CPU"; pay attention to:
- **Effective bandwidth utilization** = actual data movement volume / kernel execution time / theoretical peak bandwidth
- **Compute Intensity**: the amount of computation required per byte of data, determining whether an operator is compute-bound or memory-bound

Reference: for memory-bound operators, a bandwidth utilization of 80-95% is considered excellent.

### 7.2 Memory Hierarchy Optimization Priority

1. **Global Memory**: Must ensure coalesced access; one warp accesses a contiguous 128 B
2. **Vectorization**: Use float4/int4 to reduce the number of instructions and improve bandwidth utilization (prerequisite: address alignment)
3. **Shared Memory**: Use as a data staging area (Corner Turn) to solve mismatched read/write directions
   - Watch out for bank conflicts; resolve with padding or swizzling
4. **Registers**: The fastest storage level; cache repeatedly accessed data into registers whenever possible
5. **Warp Shuffle**: Direct register-to-register communication, ~1 cycle latency, replacing shared memory's ~20 cycles

### 7.3 Parallelism Optimization

- **Occupancy**: Improve by adjusting block size, reducing register usage, and controlling shared memory usage
- **2D Block**: A simple change (1D -> 2D) can significantly boost occupancy
- **Thread Coarsening**: Appropriately reduce the number of blocks Arg, have each thread process more data, which can alleviate warp stalls and improve instruction reordering
- **ILP (Instruction-Level Parallelism)**: Let one thread process multiple independent data items (e.g., multiple rows), allowing the compiler to pipeline### 7.4 Reduce Unnecessary Overhead

- Avoid redundant data copies between kernels (e.g., unnecessary copies in `torch.sort`)
- Be mindful of the negative impact of debugging code such as `CUDA_KERNEL_ASSERT` on registers and occupancy
- Use hardware intrinsics like `__expf`, `__sincosf` in place of standard library functions
- Fuse kernels to reduce launch overhead and global memory read/write of intermediate results

---

## Related Documents

### This Knowledge Base

- [Online Softmax and Flash Attention Hands-On](../../kernel-opt/generic/hands-on/online-softmax-flash-attention.md) -- Triton implementation of the online softmax algorithm and Flash Attention
- [Fused Kernel Patterns Hands-On](../../kernel-opt/generic/hands-on/fused-kernel-patterns.md) -- A systematic methodology for kernel fusion
- [GPU General Optimization Theory](README.md) -- Fundamental theoretical framework for GPU optimization

### Original Zhihu Articles

- [A Deep Dive into GPU Optimization Series: Reduce Optimization](https://zhuanlan.zhihu.com/p/426978026) -- Seven-step reduce optimization on V100 achieving 858 GB/s bandwidth
- [CUDA Optimization: LayerNorm Performance Optimization in Practice](https://zhuanlan.zhihu.com/p/443026261) -- OneFlow's three-level LayerNorm segmented optimization strategy
- [[CUDA] Softmax Implementation and Optimization](https://zhuanlan.zhihu.com/p/719205928) -- Softmax implementation from naive to warp reduce
- [From 0.44ms to 0.04ms - 10x Softmax Performance Improvement via CUDA Optimization](https://zhuanlan.zhihu.com/p/1964020134839576011) -- Five-step softmax optimization on A100
- [Matrix Transpose - From Padding to Swizzle](https://zhuanlan.zhihu.com/p/2005711788013028455) -- A complete path for shared memory optimization
- [RoPE - The Role of Hand-Written Operators: Kernel Fusion](https://zhuanlan.zhihu.com/p/2011045579652895890) -- A fusion optimization example using computation instead of memory reads
- [Performance Optimization of Embedding Computation on GPU](https://zhuanlan.zhihu.com/p/17838433578) -- A case study of CUDA_KERNEL_ASSERT causing occupancy degradation
- [Performance Optimization of torch.sort on GPU](https://zhuanlan.zhihu.com/p/21819376841) -- The impact of OpaqueType on PTX code generation
- [Common Interview Hand-Written CUDA Operator Implementation and Optimization](https://zhuanlan.zhihu.com/p/1904550020918801886) -- Code templates for softmax/LayerNorm/RMSNorm/GEMV
