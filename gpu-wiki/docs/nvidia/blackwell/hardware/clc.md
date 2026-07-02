# Cluster Launch Control (CLC)


**Last updated**: 2026-06-30

## Overview

Cluster Launch Control (CLC) is a Blackwell hardware mechanism for **dynamic tile scheduling** in persistent kernels. It replaces the static grid scheduling model where the CUDA runtime pre-assigns tile coordinates to CTAs at launch time.

With CLC, persistent CTAs dynamically request work from a hardware queue, enabling:

- **Better load balancing**: No fixed CTA-to-tile mapping; busy SMs consume tiles as they become available.
- **Tail-effect mitigation**: The "tail" of a GEMM (when remaining tiles < number of SMs) is handled efficiently because idle CTAs pick up remaining work.
- **Dynamic cancellation**: Tiles can be cancelled via `try_cancel` when the output is no longer needed (e.g., speculative decoding).

## Static Scheduling vs CLC

### Static Scheduling (Hopper and earlier)

```
Launch grid: 256 CTAs for 256 tiles
CTA 0 -> tile (0,0)     [fixed at launch]
CTA 1 -> tile (0,1)     [fixed at launch]
CTA 2 -> tile (0,2)     [fixed at launch]
...
CTA 255 -> tile (15,15)  [fixed at launch]

Problem: If SM count = 132, first wave = 132 CTAs.
         Second wave = 124 CTAs -> 8 SMs idle = 6% waste.
         For small GEMMs, tail effect dominates.
```

### CLC Dynamic Scheduling (Blackwell)

```
Launch grid: 132 persistent CTAs (= SM count)
CTA 0: request tile -> get (0,0) -> compute -> request tile -> get (2,4) -> ...
CTA 1: request tile -> get (0,1) -> compute -> request tile -> get (2,5) -> ...
...
CTA 131: request tile -> get (0,131) -> compute -> request tile -> ...

All CTAs stay busy until the tile queue is empty.
Tail: last few tiles distributed to first-available CTAs.
```

## How CLC Works

### Hardware Queue

CLC maintains a hardware-managed work queue. Each entry represents a tile coordinate (or cluster coordinate in multi-CTA setups). The queue is populated at kernel launch and drained by CTAs calling `clusterlaunchcontrol.try_acquire`.

### CLC Programming Model

```cuda
__global__ void persistent_gemm_clc(
    const half* A, const half* B, half* C,
    int M, int N, int K,
    int num_tiles_m, int num_tiles_n
) {
    // Allocate persistent resources (TMEM, pipeline state)
    uint32_t tmem_acc = tmem_alloc(256);

    // Shared storage for CLC results (visible to all threads in CTA)
    __shared__ uint2 clc_tile_coord;
    __shared__ int clc_has_tile;

    // CLC tile loop: keep requesting tiles until none remain
    while (true) {
        // Thread 0 acquires the next tile; result goes to shared memory
        if (threadIdx.x == 0) {
            uint2 result;
            int acquired = 0;
            asm volatile(
                "{\n"
                "  .reg .pred p;\n"
                "  clusterlaunchcontrol.try_cancel {%0, %1}, p;\n"
                "  selp.s32 %2, 1, 0, p;\n"
                "}\n"
                : "=r"(result.x), "=r"(result.y), "=r"(acquired)
            );
            clc_tile_coord = result;
            clc_has_tile = acquired;
        }
        __syncthreads();  // All threads see the shared result

        // Exit if no more tiles
        if (!clc_has_tile) break;

        int tile_m = clc_tile_coord.x;
        int tile_n = clc_tile_coord.y;

        // Zero accumulator
        tmem_zero(tmem_acc, 256);

        // Mainloop: iterate over K dimension
        for (int k = 0; k < K / TILE_K; ++k) {
            // TMA load A and B tiles to SMEM
            tma_load_a(smem_a, A, tile_m, k);
            tma_load_b(smem_b, B, k, tile_n);
            wait_barrier();

            // Issue MMA
            if (threadIdx.x == 0) {
                asm volatile(
                    "tcgen05.mma.cta_group::1.kind::f16 "
                    "[%0], %1, %2, %3, 1;"
                    :
                    : "r"(tmem_acc), "l"(desc_a), "l"(desc_b), "r"(0)
                );
            }
        }

        // Epilogue
        asm volatile("tcgen05.mma.fence::before_thread_sync;");
        __syncthreads();
        store_output(tmem_acc, C, tile_m, tile_n);
    }

    // Cleanup
    tmem_dealloc(tmem_acc, 256);
}
```

## try_cancel API

CLC provides a `try_cancel` mechanism to cancel pending tiles. This is useful for speculative execution where some outputs may not be needed.

```cuda
// Cancel a specific tile if it hasn't started execution yet
__device__ bool clc_try_cancel(uint2 tile_coord) {
    bool cancelled = false;
    if (threadIdx.x == 0) {
        asm volatile(
            "clusterlaunchcontrol.try_cancel.async.shared::cta "
            "%0, [%1];"
            : "=r"(cancelled)
            : "r"(&tile_coord)
        );
    }
    return cancelled;
}
```

Use cases for `try_cancel`:
- **Speculative decoding**: Cancel tiles for rejected draft tokens.
- **Early termination**: If an attention mask makes certain output tiles zero, cancel them.
- **Dynamic batching**: Cancel tiles for sequences that have finished.

## CUTLASS Integration

CUTLASS 4.5.0 for SM100 provides CLC support through the `PersistentScheduler` class:

```cuda
// CUTLASS SM100 persistent GEMM with CLC scheduling
using Gemm = cutlass::gemm::device::GemmUniversal<
    cutlass::half_t,              // ElementA
    cutlass::layout::RowMajor,    // LayoutA
    cutlass::half_t,              // ElementB
    cutlass::layout::ColumnMajor, // LayoutB
    cutlass::half_t,              // ElementC
    cutlass::layout::RowMajor,    // LayoutC
    float,                        // ElementAccumulator
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm100,         // Blackwell
    // Tile shape: 128x256x64
    cutlass::gemm::GemmShape<128, 256, 64>,
    // Cluster shape
    cutlass::gemm::GemmShape<2, 1, 1>,
    // Use CLC persistent scheduler
    cutlass::gemm::PersistentScheduler
>;

// Launch: CUTLASS handles CLC internally
Gemm gemm_op;
gemm_op(args, workspace, stream);
```

### CUTLASS CLC Tile Scheduler

```cpp
// Simplified CUTLASS CLC scheduler logic
struct ClcTileScheduler {
    CUTLASS_DEVICE
    WorkTileInfo get_next_work() {
        WorkTileInfo work;
        // Acquire next tile from CLC hardware
        bool valid = clc_try_acquire(&work.tile_coord);
        work.is_valid = valid;

        if (valid) {
            // Convert linear tile index to 2D coordinates
            work.tile_m = work.tile_coord.x;
            work.tile_n = work.tile_coord.y;

            // Apply swizzle for L2 locality
            apply_l2_swizzle(work.tile_m, work.tile_n);
        }
        return work;
    }
};
```

## Performance Impact

CLC delivers significant performance gains, especially for small-to-medium GEMMs where tail effects dominate:

### Tail Effect Mitigation

| GEMM Size | Static Scheduler | CLC Scheduler | Improvement |
|---|---|---|---|
| 2048x2048 (small) | 86% SM utilization | 98% SM utilization | +14% |
| 4096x4096 (medium) | 92% SM utilization | 98% SM utilization | +6.5% |
| 8192x8192 (large) | 97% SM utilization | 99% SM utilization | +2% |

The canonical benchmark from the "tcgen05 for dummies" tutorial shows the jump from 940 TFLOPS (pipelined, static scheduling) to **1476 TFLOPS** (persistent + CLC), approaching 98% of cuBLAS (1507 TFLOPS).

### Why CLC Matters for Inference

Production LLM inference typically hits shapes where tail effects are severe:

```python
# Typical LLM GEMM shapes during decode (batch_size=1-64)
# M is small (batch * seq_len for decode), N and K are large (model dim)
# Example: Llama-70B decode, batch=32
M = 32     # small!
N = 8192   # hidden dim
K = 8192   # hidden dim

# Tile = 128x256 -> tiles_m = 1, tiles_n = 32 -> only 32 tiles total
# On B200 (132 SMs): 100 SMs idle with static scheduling
# CLC: 32 persistent CTAs handle all 32 tiles efficiently
```

## CLC with 2-SM Cooperative Mode

When using 2-SM cooperative MMA (`cta_group::2`), CLC distributes work in **cluster-sized units**:

```cuda
// 2-SM cooperative CLC: each acquisition gets a cluster-sized tile
__device__ void cooperative_clc_loop() {
    while (true) {
        // Acquire tile for the 2-CTA cluster
        ClusterTile tile;
        bool valid = clc_try_acquire_cluster(&tile);
        if (!valid) break;

        // Both CTAs in the cluster share the tile
        // CTA 0 handles rows 0-127, CTA 1 handles rows 128-255
        int my_row_start = (blockIdx.x % 2) * 128;

        // Issue cooperative MMA
        if (threadIdx.x == 0) {
            asm volatile(
                "tcgen05.mma.cta_group::2.kind::f16 "
                "[%0], %1, %2, %3, 1;"
                :
                : "r"(tmem_acc), "l"(desc_a), "l"(desc_b), "r"(0)
            );
        }
        // ...
    }
}
```

## L2 Cache Swizzling with CLC

CLC tile ordering can be customized with swizzle patterns to improve L2 cache hit rates:

```cuda
// Swizzle tile coordinates for better L2 locality
// Tiles are visited in a Z-order (Morton) curve pattern
__device__ void apply_l2_swizzle(int& tile_m, int& tile_n, int swizzle_bits) {
    // Convert linear tile index to swizzled 2D coordinates
    // This groups spatially adjacent tiles together, improving
    // L2 reuse for the B matrix (shared across M tiles)
    int linear = tile_m * num_tiles_n + tile_n;
    int swizzle_mask = (1 << swizzle_bits) - 1;
    int group = linear >> swizzle_bits;
    int within = linear & swizzle_mask;

    tile_m = group / num_tiles_n;
    tile_n = (group % num_tiles_n) ^ (tile_m & swizzle_mask);
}
```

## Comparison: CLC vs Software Persistent Scheduling

| Feature | CLC (Hardware) | Software Atomics |
|---|---|---|
| Scheduling overhead | Near zero (hardware) | atomicAdd contention |
| Tail-effect handling | Optimal | Good with careful design |
| Cancellation | try_cancel API | Complex (flags + barriers) |
| L2 swizzle | Configurable at launch | Manual implementation |
| Portability | SM100+ only | SM70+ |
| CUTLASS support | Built-in | Manual scheduler |


## Related

- [Two-SM Cooperative MMA](2sm-cooperative.md)
- [CUDA Memory Hierarchy for Kernel Optimization](cuda-memory-hierarchy.md)
- [GPU Architecture Resource Model](gpu-architecture-resource-model.md)
- [mbarrier](mbarrier.md)
- [NVFP4 and Block-Scaled Narrow Precision](nvfp4.md)
- [Composable Kernel (CK) Architecture Overview](../../../amd/common/ck-architecture-overview.md)
