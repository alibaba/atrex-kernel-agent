# Community GEMM Optimization Practical Summary

A comprehensive summary of core CUDA GEMM optimization knowledge from the Zhihu community, covering the complete optimization path from naive implementation to near-cuBLAS performance. Supplements CUDA-level optimization details not covered in [Triton Grouped GEMM / DeepGEMM Hands-on](../../kernel-opt/generic/hands-on/grouped-gemm-deepgemm.md).

> **Source Note**: This document synthesizes core knowledge from approximately 70 related articles in the Zhihu community, after deduplication, filtering, and structured organization. Key reference articles include Simon Boehm's SGEMM Worklog series, MegEngine's Ma Jun's Ultimate Guide to Matrix Multiplication Optimization, Li Shaoxia's GEMM Theoretical Performance Analysis, Zheng Size's Tensor Core GEMM Hands-on, Meng Yuan's From Zero to GEMM Optimization, and Aleksa Gordic's In-depth Analysis of GPU matmul, among others.

---

## 1. GEMM Optimization Panorama: From Naive to Extreme

GEMM (General Matrix Multiplication) is the most classic compute-intensive task in GPU programming. In large model training and inference, core operations such as convolution, fully connected layers, and QKV projections in Attention can all be transformed into GEMM operations, accounting for nearly all FLOPs.

### 1.1 Typical Optimization Path and Performance Ladder

Taking 4092x4092 FP32 matrix multiplication on an A6000 GPU as an example (cuBLAS peak ~23.2 TFLOPS), performance improvements at each optimization stage:

| Optimization Stage | Performance (GFLOPS) | Relative to cuBLAS | Core Optimization Techniques |
|----------|-------------|-------------|-------------|
| Naive | 309 | 1.3% | Each thread computes 1 C element |
| GMEM Coalescing | 1,987 | 8.5% | Coalesced global memory access |
| SMEM Caching | 2,980 | 12.8% | Shared Memory block caching |
| 1D Blocktiling | 8,475 | 36.5% | Each thread computes multiple C elements (1D) |
| 2D Blocktiling | 15,972 | 68.7% | Each thread computes TM x TN C elements |
| Vectorized Access | 18,237 | 78.4% | float4 vectorized read/write |
| Autotuning | 19,721 | 84.8% | Automatic block parameter tuning |
| Warptiling | 21,779 | 93.7% | Warp-level tiling + register reuse |

The Tensor Core version can further achieve 99.9% of cuBLAS performance (A100 measured at 25.08 vs 25.11 TFLOPS in TF32 mode).

### 1.2 Basic Methods for Performance Analysis

Before writing any kernel, the first step is **Roofline Analysis** — understanding the three major bottleneck ceilings of the hardware:

**Theoretical Peak Compute** (taking H100 SXM BF16 as an example):
```
Peak performance = max clock frequency × tensor core count × FLOPs per core per cycle
         = 1830 MHz x 528 x 1024 ≈ 989 TFLOPS
```

**Compute-to-Memory Ratio Analysis for GEMM**:
- Compute: `2 * M * N * K` FLOPs
- Minimum data transfer: `(M*K + K*N + M*N) * sizeof(dtype)` bytes
- For square matrix N=4092, theoretical compute is 137 GFLOPS, minimum memory transfer is 268 MB
- Given GPU compute of 30 TFLOPS and bandwidth of 768 GB/s, compute time is ~4.5ms and memory access time is ~0.34ms
- Compute time is approximately 10x memory access time, indicating that **an optimized GEMM should be compute-bound**

**Key Conclusion**: The core goal of optimization is to increase the compute-to-memory ratio so that compute units remain continuously busy and memory access latency is fully hidden.

---

## 2. Naive Implementation and Global Memory Optimization

### 2.1 Problems with Naive Implementation

```cpp
__global__ void sgemm_naive(int M, int N, int K, float alpha,
    const float *A, const float *B, float beta, float *C) {
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < M && y < N) {
        float tmp = 0.0;
        for (int i = 0; i < K; ++i) {
            tmp += A[x * K + i] * B[i * N + y];
        }
        C[x * N + y] = alpha * tmp + beta * C[x * N + y];
    }
}
```

**Performance Problem Analysis**:

1. **Extremely low compute-to-memory ratio**: The main loop consists of 2 Load instructions + 1 FMA instruction, with compute accounting for only 1/3
2. **Extensive repeated reads from global memory**: Each thread independently reads an entire row of A and an entire column of B, resulting in a total read count of `2*M*N*K`
3. **Uncoalesced memory access**: Threads within the same warp may access discontinuous memory addresses

**Bandwidth Perspective Analysis** (Titan V as example):
- Each warp performs 32 FFMA = 64 OP per iteration, requiring 1 element from A + 32 elements from B = 132 bytes
- Compute-to-memory ratio = 64 OP / 132 B = 0.48
- Even with 100% L2 cache hit rate (1.9 TB/s), the theoretical upper limit is only 0.912 TFLOPS, far below the 14.9 TFLOPS hardware compute capability

### 2.2 Global Memory Coalescing

**Core Principle**: Sequential memory accesses by 32 threads within the same warp can be coalesced by hardware into fewer memory transactions. GPU supports 32B/64B/128B memory transactions. If each thread loads a 4B float, the warp's 32*4B=128B can be coalesced into a single 128B transaction.
**Key Requirements**:
- Memory addresses accessed by threads within a warp should collectively lie within a contiguous region
- Accesses need to be aligned (but accesses by adjacent threads within a warp do not have to be strictly sequential)

**Implementation Method**: Change blockDim to one-dimensional and adjust the mapping of threads to matrix C elements:

```cpp
// Change to 1D block
dim3 blockDim(32 * 32);
// Adjust thread mapping to ensure same warp accesses consecutive columns of B
const int x = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
const int y = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);
```**Effect**: Global memory throughput increased from 15 GB/s to 110 GB/s, and performance improved from 300 GFLOPS to ~2000 GFLOPS.

---

## 3. Shared Memory Tiling

### 3.1 Basic Tiling: Global Memory -> Shared Memory

**Core Idea**: Move tiles of matrices A and B from slow Global Memory to fast Shared Memory (on-chip SRAM), reducing Global Memory accesses through data reuse.

**Tiling Strategy**:
- Matrix C is tiled by `BM x BN`, with each block computing one C sub-block
- Along the K dimension, load A sub-tile `[BM, BK]` and B sub-tile `[BK, BN]` of thickness `BK` into Shared Memory each iteration
- Perform sub-matrix multiplication on Shared Memory and accumulate
- Iterate `K/BK` times to complete the full computation

**Memory Access Analysis** (using BM=BN=128, BK=8 as an example):
- Global Memory reads per block: `K * (BM + BN)` elements
- Total read volume: `M*N*K * (1/BM + 1/BN)`
- Compared to Naive's `2*M*N*K`, memory access is reduced to `1/2 * (1/BM + 1/BN) ≈ 1/128` of the original

**Shared Memory Bandwidth Characteristics**:
- SMEM is composed of SRAM and is private to each SM
- Bandwidth is roughly 16x that of Global Memory (Volta measured: SMEM 12,080 GiB/s vs GMEM 750 GiB/s)
- Maximum 48 KB available per block (configurable L1/SMEM ratio)

### 3.2 Theoretical Analysis of Compute-to-Memory Ratio

For `M_tile x N_tile` thread block tile:
- Compute: `M_tile * N_tile * K * 2` FLOPs
- Memory access: `(M_tile + N_tile) * K * 4` bytes (FP32)
- Compute-to-memory ratio: `M_tile * N_tile / [2 * (M_tile + N_tile)]`

| M_tile x N_tile | Compute-to-Memory Ratio | Sufficient Margin for Peak Throughput |
|----------------|-----------|-------------------|
| 32 x 32 | 8 | Insufficient (~16 needed) |
| 64 x 64 | 16 | Tipping point |
| 128 x 128 | 32 | Sufficient |

**Titan V Example**: With a 64x64 tile, the compute-to-memory ratio is 16, so the required average memory bandwidth > 14.9 TFLOPS / 16 = 931 GB/s. The actual global memory read bandwidth is about 520 GB/s, and with a 30% L2 Cache hit rate (1.9 TB/s), the weighted bandwidth is 934 GB/s, just surpassing the tipping point.
---

## 4. Register Tiling (Register-Level Tiling and Data Reuse)

### 4.1 Why Register Tiling Is Needed

Using only Shared Memory tiling, where each thread computes 1 element of C, the inner loop structure is:
```
ld.shared A_val     // Read A from SMEM
ld.shared B_val     // Read B from SMEM
fma c, A_val, B_val // Compute
```

**Problem**: Before each FMA, there are 2 SMEM loads, making SMEM access the bottleneck. Analysis: each warp iteration performs 32 FFMA operations corresponding to 256B of SMEM reads, while GV100's SMEM egress bandwidth per SM is 128B/cycle, requiring 2 cycles for data transfer. However, 32 FFMA operations only need 0.5 cycles (64 FFMA/SM/cycle), meaning SMEM bandwidth can only utilize 1/4 of the compute power.

### 4.2 Thread Tile: Each Thread Computes TM x TN Results

Have each thread compute `TM x TN` elements of C, loading A and B data into registers for reuse:

```cpp
// Load TM A values and TN B values to registers
    // TM × TN FMA operations, only TM + TN SMEM reads
```

**Compute-to-Memory Ratio Improvement**:
- SMEM read count: `TM + TN` times
- FMA count: `TM * TN` times
- Compute-to-memory ratio: `TM * TN / (TM + TN)`

| TM x TN | Compute-to-Memory Ratio | FMA Count |
|---------|-----------|---------|
| 1 x 1 | 0.5 | 1 |
| 4 x 4 | 2.0 | 16 |
| 8 x 8 | 4.0 | 64 |

### 4.3 Warp Tile Design

Above the Thread Tile level there is also a Warp Tile hierarchy. A typical three-level tiling structure:

```
Thread Block Tile (128 x 128)
  -> Warp Tile (64 x 32)
    -> Thread Tile (8 x 8)
```

- 256 threads = 8 warps, arranged as a 2x4 grid
- Each warp is responsible for a 64x32 sub-block of C
- Each thread is responsible for an 8x8 sub-block of C (64 registers for accumulation results)

### 4.4 Resource Configuration Trade-offs

**Register Budget**: Taking a 128x128 block with 256 threads as an example
- Each thread computes 64 results -> 64 registers for C
- A/B data movement -> ~16 registers
- Other temporary variables -> ~20 registers
- Total ~100 per thread, 256 threads = 25,600 per block
- SM limit is 65,536, can accommodate 2 blocks -> Good Occupancy

**Occupancy Trade-offs**:
- Block size 128, 128 results per thread -> requires 180+ registers/thread -> Active Warps only 8 -> Occupancy 25%
- Block size 256, 64 results per thread -> requires 128 registers/thread -> Active Warps can reach 16 -> Occupancy 50%
- **Higher Occupancy does not necessarily mean higher performance**, but it provides a larger warp switching pool to hide latency

## 5. Vectorized Access and Bank Conflict Elimination

### 5.1 Vectorized Memory Access

Using `float4` (128-bit) vectorized reads and writes can significantly reduce the number of instructions:

```cpp
// Scalar read: 4 LDS.32 instructions
float a0 = smem[idx], a1 = smem[idx+1], a2 = smem[idx+2], a3 = smem[idx+3];

// Vectorized read: 1 LDS.128 instruction
float4 a = *reinterpret_cast<float4*>(&smem[idx]);
```

**Effect**: Read instructions are reduced by a factor of 4, further improving the compute-to-memory-access ratio. To read Shared Memory using LDS.128, matrix A needs to be transposed before writing to SMEM.

### 5.2 Shared Memory Bank Conflict

Shared Memory consists of 32 banks, each 32 bits (4 bytes) wide. **Threads within the same warp cannot simultaneously access different addresses in the same bank**, otherwise accesses are serialized (N-way bank conflict -> N× throughput degradation).

**Typical Bank Conflict in GEMM**:
- For SMEM accesses to matrix B, threads with `threadIdx` differing by 4 access the same 8 banks, resulting in bank conflicts
- Matrix A, after transposition, can often leverage broadcast behavior to avoid conflicts

### 5.3 Solutions: Padding and Swizzle

**Padding Method**: Add padding elements at the end of each SMEM row to offset bank mapping:
```cpp
__shared__ float Bs[BK][BN + PADDING];  // Typically PADDING = 4
```

**Swizzle Method** (more efficient): Remap SMEM addresses using XOR operations so that different threads access different banks:
```cpp
// XOR swizzle: XOR address high bits with low bits
int swizzled_addr = addr ^ ((addr >> shift) & mask);
```

The advantage of swizzle is that it does not waste SMEM space Vintage, and it can be used in conjunction with TMA's hardware swizzle. On the Hopper architecture, TMA supports built-in swizzle modes (128B/64B/32B), which automatically perform address shuffling during data movement.

---

## 6. Double Buffering / Pipelining (Hiding Latency)

### 6.1 Core Idea of Double Buffering

**Problem**: In an implementation without double buffering, two `__syncthreads()` are needed:
1. Synchronization after GMEM → SMEM load completes
2. Synchronization after the current tile computation completes

This causes loading and computation to be completely serialized.

**Double Buffering Solution**: Allocate twice the space for both SMEM and registers, and use them alternately:
- While Buffer 0 is being computed, Buffer 1 asynchronously loads the next tile
- Separate reads and writes eliminate data dependencies, requiring only one `__syncthreads()`

```cpp
__shared__ float As[2][BK][BM];  // Double buffer
__shared__ float Bs[2][BK][BN];
float frag_a[2][TM];             // Register double buffer
float frag_b[2][TN];

// Prologue: Load first tile
load_gmem_to_smem(A, B, As[0], Bs[0]);
__syncthreads();

int write_stage = 1;
for (int k = BK; k < K; k += BK) {
    // Async load next tile to intermediate registers
    load_gmem_to_reg(A, B, ldg_a_reg, ldg_b_reg);

    // Compute current tile (load from SMEM to registers + FMA)
    for (int j = 0; j < BK - 1; ++j) {
        load_smem_to_reg(As[load_stage], j+1, frag_a[(j+1)%2]);
        load_smem_to_reg(Bs[load_stage], j+1, frag_b[(j+1)%2]);
        mma(frag_a[j%2], frag_b[j%2], c);
    }

    // Write intermediate registers to SMEM (intentionally staggered with compute to hide latency via ILP)
    store_reg_to_smem(ldg_a_reg, As[write_stage]);
    store_reg_to_smem(ldg_b_reg, Bs[write_stage]);
    __syncthreads();
    write_stage ^= 1;

    mma(frag_a[1], frag_b[1], c);  // Compute last tile
}
```

### 6.2 The Essence of Instruction-Level Parallelism (ILP)

The performance gains from double buffering do not come from "parallel execution" of code, but from **instruction-level parallelism**:
- On GPUs, memory access and computation correspond to different hardware units that can work in parallel
- Sequential execution of code corresponds to the order in which instructions are issued, and the issue rate is very fast
- Since read and write operations under double buffering use independent data, the GPU can issue more instructions without waiting
- The latency of memory access instructions is filled by compute instructions, achieving overlap between reads and writes

### 6.3 Multi-Stage Pipeline (A100+ cp.async)

A100 introduced the `cp.async` instruction, which supports asynchronous copies from Global Memory to Shared Memory (bypassing registers), enabling deeper pipelines:

```cpp
// 4-stage pipeline
cp.async.cg.shared.global [smem_ptr], [gmem_ptr], 16;  // Async copy 16 bytes
cp.async.commit_group;                                   // Commit a group
cp.async.wait_group 2;                                   // Wait until at most 2 groups are pending
```

- A 3–4 stage pipeline is usually sufficient; further depth increases yield diminishing returns
- SMEM usage = number of stages × single-stage usage, but peak usage can be controlled by sharing SMEM address space among A/B/C

---

## 7. Tensor Core / WMMA / MMA Usage

### 7.1 Tensor Core Fundamentals

Tensor Cores are dedicated matrix multiply units. A single operation can complete a small matrix multiplication (e.g., 16×16×16 FP16 multiply-accumulate), with throughput far exceeding CUDA Cores:

| Architecture | FP32 CUDA Core | TF32 Tensor Core | FP16 Tensor Core |
|------|----------------|-------------------|-------------------|
| A100 | 19.5 TFLOPS | 156 TFLOPS | 312 TFLOPS |
| H100 | 67 TFLOPS | 495 TFLOPS | 989 TFLOPS |### 7.2 Using the WMMA API

WMMA (Warp Matrix Multiply-Accumulate) is the simplest Tensor Core programming interface:

```cpp
// Load fragment
load_matrix_sync(FragA[i], smem_a + offset, leading_dim);
// Execute matrix multiply-accumulate
mma_sync(Accum[idx], FragA[i], FragB[j], Accum[idx]);
```
**Typical configuration** (A100 128×128×32 tiling):
- Thread block: 128 threads = 4 warps (arranged 2×2)
- Each warp processes a 64×64×16 sub-tile
- Each warp requires 4×4=16 wmma invocations

### 7.3 Using PTX MMA Instructions

Using PTX directly provides finer-grained control:

```cpp
// TF32 MMA: m16n8k8 shape
asm volatile(
    "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
    : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
      "r"(b0), "r"(b1),
      "f"(c0), "f"(c1), "f"(c2), "f"(c3)
);
```

In conjunction with the `ldmatrix` instruction for cooperative data loading into fragments:
```cpp
// Warp cooperative load of 4 8x8 matrices
asm volatile(
    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
    : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
    : "r"(smem_addr)
);
```

### 7.4 Real-World Performance Data

Zheng Size implemented FP16-FP32 mixed-precision GEMM (5376×5376×2048) using Tensor Cores on the A100. The optimization process:

| Version | Time (ms) | TFLOPS | Description |
|------|----------|--------|------|
| cuBLAS baseline | 0.755 | 156.7 | FP32 accumulator |
| Naive WMMA | 4.47 | 26.5 | Basic wmma, no pipelining |
| 4-Stage Pipeline | - | ~100+ | cp.async + 4-stage pipeline |
| + SMEM Swizzle | - | ~130+ | Eliminates bank conflicts |
| + ldmatrix substitution | 0.84 | 140.9 | Replaces wmma with ldmatrix+mma |
| Final optimization | - | ~141 | Approximately 90% of cuBLAS |

---

## 8. Stream-K / Split-K Work Distribution Strategies

### 8.1 Tile Quantization Problem

In the traditional Data Parallel approach, each CTA computes one complete C sub-tile. When the number of C sub-tiles is not divisible by the number of SMs, **Tile Quantization** occurs: some SMs become idle during the last wave.

**Example**: 9 tiles, 4 SMs → requires 3 waves → only 1 tile in the last wave → throughput is only 9/12 = 75%.

### 8.2 Split-K

Splits along the K dimension to increase parallelism:
- The K-loop of each tile is split across s CTAs for parallel execution
- After computation, an additional reduction (fixup) operation combines the s partial sums
- Additional IO overhead: reads and writes of size `M * N * (s-1)`

**Advantage**: Keeps BM and BN unchanged, does not affect SMEM data reuse
**Disadvantage**: Extra IO overhead during the fixup stage

### 8.3 Stream-K

Stream-K evenly distributes all MAC operations across SMs by total count, rather than by tile:

```
Total iterations = ceil(M/BM) * ceil(N/BN) * ceil(K/BK)
Iterations per CTA = ceil(total iterations / number of CTAs)
```

- Each CTA may span computations across multiple tiles
- The number of fixups is only `CTA count - 1`, independent of problem size
- Can achieve close to 100% Quantization Efficiency

**Example**: problem size (384, 384, 128), tile (128, 128, 32), 4 SMs → each SM receives exactly 9 MAC iterations → 100% utilization.

---

## 9. GEMV Vectorized Optimization

GEMV (General Matrix-Vector Multiply) is the most dominant operator in LLM inference determinand is a typical memory-bound operation.

### 9.1 Hiding Memory Latency

Cover memory latency through a large number of concurrent warps. Note, however, that ALUs within an SM are time-shared among all warps. The compute delay of a single warp can be stretched by interruptions from other warps, with actual compute latency reaching over 20% of memory latency.

### 9.2 Dual-Pipeline Latency Elimination

For GEMV with a large batch size (using Tensor Cores in practice), dual pipelining can achieve full overlap of load and computation:

```
A[2], B[2];  // double buffer
for k-loop:
    Async Load tile k+1 into A[load_index]
    Async Load tile k+1 into B[load_index]
    Compute tile k with A/B[compute_index]
    swap buffers
```

**Result**: The latency of each loop iteration includes only the load latencyanson, not compute latency.

### 9.3 Practical Suggestions

- GEMV with batch size = 1: The computational workload is too small Wool to effectively overlap; prioritize high concurrency to hide latency.
- Batched GEMV with a large batch size: The computational workload is significant; dual-pipelining is the primary optimization technique.

---

## 10. Performance Comparison and Analysis with cuBLAS

### 10.1 Summary of Measured Data Across Platforms

| Platform | Precision | Handwritten Kernel | cuBLAS | Ratio | Scale |
|------|------|-----------|--------|------|------|
| A6000 | FP32 | 21.8 TFLOPS | 23.2 TFLOPS | 93.7% | 4092^2 |
| RTX 3090 | FP32 | ~33 TFLOPS | ~34 TFLOPS | ~97% | 4096^2 |
| Titan V | FP32 | ~14.2 TFLOPS | ~14.5 TFLOPS | ~98% | Large Matrix |
| A100 | TF32 | 25.08 TFLOPS | 25.11 TFLOPS | 99.9% | 4096^2 |
| A100 | FP16 | ~141 TFLOPS | ~157 TFLOPS | ~90% | 5376^2x2048 |
| RTX 5060M | FP32 | 13.85 ms | 14.33 ms | 103.5% | 4096^2 |
| RTX 5060M | TF32 | - | - | >100% | 4096^2 |

### 10.2 Conditions for Surpassing cuBLAS

Under certain conditions, a handwritten kernel can surpass cuBLAS:
- **Specific matrix sizes**: cuBLAS needs to cover all shapes, while a handwritten kernel can be extremely optimized for a specific shape.
- **Consumer-grade GPUs**: cuBLAS may not be fully optimized for consumer-grade graphics cards.
- **No boundary checks**: The boundary-handling logic required by general-purpose libraries is omitted.
- **Specific precisions**: cuBLAS support for certain precision combinations is relatively recent Tuc, with limited optimization.

---

## 11. Common GEMM Optimization Pitfalls and Lessons

### 11.1 Resource Configuration Pitfalls

- **Register Spilling**: Exceeding the register limit per thread causes spills to local memory (which is actually DRAM), resulting in a cliff-like performance drop.
- **Excessively Low Occupancy**: Double buffering increases register usage, which may cause threads to exceed 128 registers per thread, reducing the number of blocks from 2 to 1 and causing a dramatic performance drop. Solution: streamline variables Nex reuse registers.
- **SMEM Overflow**: Double buffering doubles SMEM usage; confirm that the per-block maximum SMEM limit is not exceeded.

### 11.2 Memory Access Pitfalls

- **The Subtlety of Bank Conflicts**: Bank conflicts only occur during SMEM reads (when threads within a warp concurrently access different addresses of the same bank); conflicts during SMEM writes have a smaller impact.
- **Default Type Promotion**: Inadvertently using the `double` type in computation causes a sharp performance drop (FP64 throughput is much lower than FP32).
- **Unused L2 Cache**: Use Grid Swizzling (arranging blocks in Hilbert curve or Z-order) to improve L2 cache hit rates.

### 11.3 Computation Pitfalls

- **Loop-Carried Dependencies**: In microbenchmarks, writing to the same accumulator serializes SMEM accesses; use multiple independent accumulators to break the dependency chain.
- **Compiler Optimization Interference**: The compiler may merge consecutive additions into multiplications (inflating bandwidth figures); use `volatile` to prevent this.
- **Mismatched Tiling Parameters**: If BK is too small, the main loop iteration count becomes too high, and `__syncthreads()` overhead is large; if BK is too large, SMEM/register limits are exceeded.

### 11.4 Tensor Core Pitfalls

- **Shape Constraints**: Different precisions support different mma shapes (e.g., TF32 only supports m16n8k8), limiting layout flexibility.
- **ldmatrix No Transpose**: TF32 mode does not support the `.trans` modifier; manual transposition or register shuffling is required.
- **Fixed Fragment Layout**: The input/output fragment layout of mma instructions is determined by hardware and cannot be customized.

### 11.5 Power Limit Constraints

Actual performance is affected by power limit constraints—when GPU power consumption approaches TDP, clock speeds are automatically reduced. For example, when the H100 clock drops from 1830 MHz to 1000 MHz, BF16 performance drops from 989 TFLOPS to about 541 TFLOPS.

---

## 12. Guide to Selecting Block Size and Tiling Parameters

Considerations for choosing block size and data tiling size:

### 12.1 Block Size

- Sweet spot: **128 or 256** threads
- At least 4 warps (128 threads) to fully utilize the SM's 4 warp schedulers
- 256 threads = 8 warps, allowing more data to be loaded into SMEM for reuse

### 12.2 Data Tiling Size

**Driven by Compute-to-Memory Ratio**:
```
Compute-to-memory ratio = BM * BN * BK * 2 / [(BM * BK + BK * BN) * sizeof(dtype)]
```
Must exceed the hardware's compute-to-bandwidth ratio (e.g., approximately 25.2 FLOPs/B for RTX 5060M).

**Typical Configurations**:
- SGEMM: BM=BN=128, BK=8~16, 256 threads
- HGEMM/TF32: BM=64~128, BN=128, BK=16~32, 256~512 threads

### 12.3 Choosing BK

- Should not be too small (e.g., 4): The main loop iteration count doubles, and loop control + `__syncthreads()` overhead is large.
- Should not be too large (e.g., 32): SMEM and register usage surges, occupancy drops, and double buffering cannot be supported.
- Sweet spot: **8 or 16**, balancing spatial locality and resource consumption.

---

## Related Documentation

- [GPU Memory Hierarchy and Optimization](gpu-memory-hierarchy.md) — Bandwidth/latency characteristics of registers, SMEM, L1/L2, and HBM
- [GPU Execution Model and Thread Optimization](gpu-execution-model.md) — SIMT, warp, block, occupancy
- [Triton Grouped GEMM / DeepGEMM in Practice](../../kernel-opt/generic/hands-on/grouped-gemm-deepgemm.md) — GEMM optimization at the Triton level
- [Triton Memory Access Optimization Patterns](../../kernel-opt/generic/hands-on/memory-access-optimization.md) — General techniques such as coalesced access and vectorization
- [Persistent Kernel and Tile Scheduling](../../kernel-opt/generic/hands-on/persistent-kernel-tile-scheduling.md) — Scheduling strategies such as Stream-K
- [NVIDIA General Optimization Documentation](../nvidia/common/) — NCU profiling, async copy, TMA, etc.
- [Blackwell GEMM Optimization in Practice](../../kernel-opt/nvidia/common/hands-on) — Tensor Core optimization on SM100 architecture
- [Hopper GEMM Optimization in Practice](../../kernel-opt/nvidia/common/hands-on) — TMA, WGMMA, warp specialization
