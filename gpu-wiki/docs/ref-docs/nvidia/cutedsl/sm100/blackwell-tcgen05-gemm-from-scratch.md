# Building a tcgen05 GEMM from Scratch: Reaching 98% of cuBLAS on Blackwell

A step-by-step implementation of a Blackwell matrix multiplication kernel using raw PTX tcgen05 instructions, progressing from a basic kernel to 98% of cuBLAS performance through swizzling, pipelining, warp specialization, 2-SM MMA, and persistent kernel techniques.

---

## 1. Introduction

`tcgen05` is a set of PTX instructions for programming Tensor Cores on NVIDIA Blackwell GPUs (SM100, not to be confused with consumer Blackwell SM120). This document records the process of learning tcgen05 and reaching 98% of cuBLAS speed on M=N=K=4096.

All B200 work was performed on cloud GPU instances. The complete code is available in the companion repository.

## 2. High-Performance Matrix Multiplication Review

Matrix multiplication of A (shape MxK) and B (shape KxN) produces output C of shape MxN. Mathematically, each element of C is the dot product of a row of A and a column of B.

```python
def matmul(A: Tensor, B: Tensor, C: Tensor, M: int, N: int, K: int):
    for m in range(M):
        for n in range(N):
            acc = 0
            for k in range(K):
                acc += A[m, k] * B[k, n]
            C[m, n] = acc
```

Nearly all matrix multiplication implementations use some form of tiling: selecting blocks of A and B, performing a small matrix multiply, and accumulating results along K.

```python
def tiled_matmul(A: Tensor, B: Tensor, C: Tensor, M: int, N: int, K: int):
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 64
    for m in range(0, M, BLOCK_M):
        for n in range(0, N, BLOCK_N):
            acc = torch.zeros(BLOCK_M, BLOCK_N)
            for k in range(0, K, BLOCK_K):
                A_tile = A[m:m+BLOCK_M, k:k+BLOCK_K]
                B_tile = B[k:k+BLOCK_K, n:n+BLOCK_N]
                acc += mini_matmul(A_tile, B_tile)
            C[m : m + BLOCK_M, n : n + BLOCK_N] = acc
```

Tensor Cores serve as the "mini matmul engine," so we naturally tile our problem according to Tensor Core shapes. The `m` and `n` loops are parallelized across thread blocks (each responsible for BLOCK_M x BLOCK_N of output). Within each thread block, we iterate along K, loading tiles of A and B and executing MMA.

Each NVIDIA GPU generation has its own PTX instructions for loads and compute. Conceptually, a Blackwell matmul kernel is not drastically different from previous generations — we just need to understand the new PTX instructions.

## 3. Basic tcgen05 Kernel

### 3.1 TMA and mbarrier Primer

The Tensor Memory Accelerator (TMA), available since Hopper, issues memory loads with minimal register usage and address computation. Unlike `cp.async` (max 16 bytes per thread), TMA can issue arbitrarily large loads from a single thread. In PTX, TMA corresponds to `cp.async.bulk` (1D) and `cp.async.bulk.tensor` (1D–5D) instructions.

First, create a Tensor Map object on the host to encode how TMA transfers data:

```cpp
#include <cudaTypedefs.h>
#include <cuda_bf16.h>

constexpr int BLOCK_M = 64;
constexpr int BLOCK_N = 64;
constexpr int BLOCK_K = 64;
constexpr int TB_SIZE = 128;

void init_2D_tmap(
    CUtensorMap *tmap, const nv_bfloat16 *ptr,
    uint64_t global_height, uint64_t global_width,
    uint32_t shared_height, uint32_t shared_width
) {
    constexpr uint32_t rank = 2;
    uint64_t globalDim[rank] = {global_width, global_height};
    uint64_t globalStrides[rank-1] = {global_width * sizeof(nv_bfloat16)};
    uint32_t boxDim[rank] = {shared_width, shared_height};
    uint32_t elementStrides[rank] = {1, 1};
    auto err = cuTensorMapEncodeTiled(
        tmap, CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        rank, (void *)ptr, globalDim, globalStrides, boxDim, elementStrides,
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
}
```

Both A and B are assumed K-major: A has shape (M, K) and B has shape (N, K). This matches the common PyTorch `nn.Linear()` pattern where input is (batch_size, in_features) and weight is (out_features, in_features).

TMA operates in an asynchronous agent. The programming model treats TMA as a separate device from the CUDA core perspective. Whenever reading data written by TMA or writing data for TMA to read, we must follow the PTX memory consistency model. NVIDIA provides `mbarrier` as the synchronization mechanism.

The mbarrier tracks two counts: an arrival count (how many threads have "arrived") and a transaction count (how many bytes have been transferred). We set the expected arrival count at mbarrier initialization. For TMA with a single issuing thread, we initialize to 1.

```cpp
// mbarrier initialization
if (tid == 0) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(mbar_addr), "r"(1));
    asm volatile("fence.mbarrier_init.release.cluster;");
}
__syncthreads();

// main loop
int phase = 0;
for (int iter_k = 0; iter_k < K / BLOCK_K; iter_k++) {
    if (tid == 0) {
        const int off_k = iter_k * BLOCK_K;
        tma_2d_gmem2smem(A_smem, &A_tmap, off_k, off_m, mbar_addr);
        tma_2d_gmem2smem(B_smem, &B_tmap, off_k, off_n, mbar_addr);
        constexpr int cp_size = (BLOCK_M + BLOCK_N) * BLOCK_K * sizeof(nv_bfloat16);
        asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
            :: "r"(mbar_addr), "r"(cp_size) : "memory");
    }
    mbarrier_wait(mbar_addr, phase);
    phase ^= 1;
    // issue tcgen05.mma ...
}
```

The `mbarrier_wait()` function spins on the barrier using `mbarrier.try_wait.parity.acquire`:

```cpp
__device__ inline void mbarrier_wait(int mbar_addr, int phase) {
    uint32_t ticks = 0x989680;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1, %2;\n\t"
        "@P1 bra.uni DONE;\n\t"
        "bra.uni LAB_WAIT;\n\t"
        "DONE:\n\t"
        "}" :: "r"(mbar_addr), "r"(phase), "r"(ticks)
    );
}
```

### 3.2 Acquire-Release Semantics

The `.acquire` and `.release` modifiers ensure that anything before the release (on the producer thread) is visible to everything after the acquire (on the consumer thread). In the TMA context: once `mbarrier_wait()` returns, we know the TMA transfer is complete and subsequent shared memory operations see fresh data.

### 3.3 Understanding tcgen05

Key differences across GPU generations for MMA:

| Property | Ampere | Hopper | Blackwell |
|----------|--------|--------|-----------|
| MMA shape | 16x8x16 | 64x256x16 | 128x256x16 |
| Operand A source | Registers | SMEM | SMEM (*or TMEM) |
| Operand B source | Registers | SMEM | SMEM |
| Accumulator | Registers | Registers | Tensor Memory |

MMA_K stays at 32 bytes across generations while MMA_M and MMA_N increase significantly, improving arithmetic intensity.

**Tensor Memory** is a new memory type dedicated to MMA results. Capacity: 128 rows x 512 columns, each element 32-bit (fitting FP32/INT32 accumulators). Usage:

- Allocate with `tcgen05.alloc` (column granularity, always allocates all 128 rows)
- Deallocate with `tcgen05.dealloc` before kernel exit
- Access via `tcgen05.ld` (TMEM→registers), `tcgen05.st` (registers→TMEM), `tcgen05.cp` (SMEM→TMEM)

```cpp
__shared__ int tmem_addr[1];
if (warp_id == 1) {
    const int addr = static_cast<int>(__cvta_generic_to_shared(tmem_addr));
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
        :: "r"(addr), "r"(BLOCK_N));
}
__syncthreads();
const int taddr = tmem_addr[0];
```

### 3.4 Shared Memory Descriptors and Core Matrices

The `tcgen05.mma` instruction requires 64-bit shared memory descriptors for operands A and B. These encode address, layout, and swizzle information.

A critical hidden concept: the **core matrix** — an implicit 8x16B unit (8 rows, 16 bytes per row). The PTX documentation refers to "Leading dimension byte offset (LBO)" and "Stride dimension byte offset (SBO)" relative to these core matrices.

Key insight: each 8x16B core matrix must be contiguous in shared memory. This requires reorganizing the SMEM layout so that [BLOCK_M, 16B] slices are stored contiguously rather than using natural row-major order.

For the initial non-swizzled layout:
```cpp
auto make_desc = [](int addr, int height) -> uint64_t {
    const int LBO = height * 16;
    const int SBO = 8 * 16;
    return desc_encode(addr) | (desc_encode(LBO) << 16ULL)
         | (desc_encode(SBO) << 32ULL) | (1ULL << 46ULL);
};
```

### 3.5 Epilogue: Reading from Tensor Memory

Each warp can only access a portion of Tensor Memory. With MMA_M=128 and `.cta_group::1`, the kernel needs at least 4 warps for the epilogue (Layout D in the PTX documentation).

Using `tcgen05.ld` with `.32x32b.x8`, each thread loads 8 consecutive FP32 accumulator values, converts them to BF16, and stores 16 bytes to global memory:

```cpp
asm volatile("tcgen05.fence::after_thread_sync;");
for (int n = 0; n < BLOCK_N / 8; n++) {
    float tmp[8];
    const int row = warp_id * 32;
    const int col = n * 8;
    const int addr = taddr + (row << 16) + col;
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
        : "=f"(tmp[0]),"=f"(tmp[1]),"=f"(tmp[2]),"=f"(tmp[3]),
          "=f"(tmp[4]),"=f"(tmp[5]),"=f"(tmp[6]),"=f"(tmp[7])
        : "r"(addr));
    asm volatile("tcgen05.wait::ld.sync.aligned;");
    // convert to BF16 and store to global memory
}
```

**v1 result:** Less than 20% of cuBLAS speed (~254 TFLOPS vs ~1507 TFLOPS).

## 4. 128-Byte Global Loads and Swizzling

Swizzling avoids shared memory bank conflicts by distributing data across all 32 banks. TMA supports swizzling at 16-byte granularity. `CU_TENSOR_MAP_SWIZZLE_128B` means within a 128-byte block, eight 16-byte units are rearranged based on their row index via XOR.

For tcgen05.mma with 128B swizzling, the requirement is that each 8x128B tile is a contiguous block in shared memory (compared to 8x16B tiles without swizzling).

The TMA Tensor Map encodes the innermost two dimensions as 8x128B, with 128B swizzle mode. The shared memory descriptor sets bits 61-63 to `2` (128B swizzle) and encodes `SBO = 8 * 128`:

```cpp
auto make_desc = [](int addr) -> uint64_t {
    const int SBO = 8 * 128;
    return desc_encode(addr) | (desc_encode(SBO) << 32ULL)
         | (1ULL << 46ULL) | (2ULL << 61ULL);
};
```

MMA tile selection uses two nested loops: the outer iterates over [BLOCK_M, 128B] TMA tiles within [BLOCK_M, BLOCK_K], and the inner iterates over [BLOCK_M, 32B] MMA tiles within [BLOCK_M, 128B]:

```cpp
for (int k1 = 0; k1 < BLOCK_K / 64; k1++)
    for (int k2 = 0; k2 < 64 / MMA_K; k2++) {
        uint64_t a_desc = make_desc(A_smem + k1 * BLOCK_M * 128 + k2 * 32);
        uint64_t b_desc = make_desc(B_smem + k1 * BLOCK_N * 128 + k2 * 32);
        tcgen05_mma_f16(taddr, a_desc, b_desc, i_desc, enable_acc);
    }
```

**v2 result:** 2.7x speedup over v1, reaching 46% of cuBLAS (~695 TFLOPS).

An important observation: the speedup likely comes from TMA being more efficient with wider innermost dimensions (128B vs 16B), rather than bank conflict elimination per se. The v1 layout (contiguous 8x16B tiles) already spans all 32 banks without conflicts.

## 5. Pipelining

Standard N-stage pipelining: maintain N in-flight global→shared load stages, each with its own shared memory buffer. Use separate mbarriers per stage:

```cpp
auto load = [&](int iter_k) { /* issue TMA */ };
auto compute = [&](int iter_k) { /* issue tcgen05.mma */ };

for (int stage = 0; stage < NUM_STAGES; stage++)
    load(stage);

for (int iter_k = 0; iter_k < K / BLOCK_K; iter_k++) {
    load(iter_k + NUM_STAGES - 1);
    // wait for current load stage
    compute(iter_k);
    // wait for current compute stage
}
```

**v3 result:** 35% speedup, reaching ~940 TFLOPS.

## 6. Warp Specialization

Since only 1 thread issues TMA and 1 thread issues MMA, dedicate separate warps to each task. Each warp runs its own main loop:

```cpp
if (warp_id == 0 && elect_sync()) {
    // TMA warp
    for (int iter_k = 0; iter_k < num_iters; iter_k++) ...
} else if (warp_id == 1 && elect_sync()) {
    // MMA warp
    for (int iter_k = 0; iter_k < num_iters; iter_k++) ...
}
```

On Blackwell (and Hopper), since Tensor Cores operate asynchronously, we can issue multiple `tcgen05.mma` without waiting for completion before moving to the next stage. Two sets of mbarriers coordinate TMA and MMA warps.

A critical detail: after the main loop, `__syncthreads()` alone does not guarantee MMA completion — it only guarantees the last `tcgen05.mma` was *issued*. Use an additional mbarrier with `tcgen05.commit` after the loop to properly synchronize.

**v4 result:** 29% speedup, reaching ~1209 TFLOPS.

## 7. 2-SM MMA

Blackwell allows two CTAs to cooperatively compute a single MMA tile. Each CTA provides half of A and half of B, and holds half of the output accumulator. MMA_M increases to 256 (double the 1-SM case), while each CTA only needs half the B tile.

### 7.1 Cluster Launch

Enable thread block clusters with `__cluster_dims__(2, 1, 1)`:

```cpp
__global__ __cluster_dims__(2, 1, 1) __launch_bounds__(TB_SIZE)
void kernel(...) {
    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));
    ...
}
```

### 7.2 Synchronization Pattern

Only CTA0 issues MMA. Both CTAs' TMA report completion to CTA0's TMA mbarrier. The instruction descriptor encodes `MMA_M = 256` with `.cta_group::2`. For shared memory descriptors, only local addresses are needed — hardware automatically fetches from remote shared memory at the same offset.

Key requirement: shared memory object layout must be identical across all cluster ranks.

For multicast (signaling mbarriers across CTAs), the same offset principle applies — `tcgen05.commit` with `.cta_group::2` multicasts to mbarriers in both CTAs at the same shared memory offset.

**v5 result:** 8% speedup, reaching ~1302 TFLOPS.

## 8. Persistent Kernel with Static Scheduling

In-kernel profiling reveals that epilogue and per-tile setup consume significant time, leaving Tensor Cores idle. The solution: launch exactly N_SM thread blocks (148 on B200), each processing multiple output tiles sequentially.

Benefits:
- One-time setup per SM instead of per tile
- Epilogue overlaps with TMA/MMA for the next tile

### 8.1 Design

Three warp groups with two producer-consumer pairs:

1. **TMA↔MMA pair:** shared memory buffers (same as before)
2. **Main loop↔Epilogue pair:** tensor memory buffers (double-buffered across output tiles)

```cpp
if (warp_id == 0 && elect_sync()) {
    // TMA warp: iterate over output tiles, inner loop over K
} else if (cta_rank == 0 && warp_id == 1 && elect_sync()) {
    // MMA warp: iterate over output tiles, inner loop over K
} else if (warp_id >= 2) {
    // Epilogue warps (4 warps): iterate over output tiles
}
```

Total: 6 warps — 1 TMA, 1 MMA, 4 epilogue (minimum needed to access all tensor memory).

**v6 result:** Reaches 98% of cuBLAS (~1476 TFLOPS).

## 9. Performance Summary

| Kernel | TFLOPS | % of cuBLAS |
|--------|--------|-------------|
| cuBLAS | 1506.74 | 100% |
| v1 (basic tcgen05 + 16B TMA) | 254.62 | 17% |
| v2 (128B swizzled TMA) | 695.43 | 46% |
| v3 (pipelining) | 939.61 | 62% |
| v4 (warp specialization) | 1208.83 | 80% |
| v5 (2-SM MMA) | 1302.29 | 86% |
| v6 (persistent kernel) | 1475.93 | 98% |

## 10. Observations

Tensor Core programming on Blackwell feels easier than previous generations due to dedicated hardware and instructions. Once you understand the requirements for good FLOPS utilization, the design space is relatively small. You can think at the tile level rather than thread level — no complex thread address calculations or manual swizzle computation needed.

However, mixing tcgen05 with CUDA core operations (e.g., attention mechanisms) introduces synchronization challenges between the general agent and async agent. Techniques not yet explored (thread block swizzling for L2 utilization, Cluster Launch Control for dynamic scheduling) could potentially exceed cuBLAS performance.
