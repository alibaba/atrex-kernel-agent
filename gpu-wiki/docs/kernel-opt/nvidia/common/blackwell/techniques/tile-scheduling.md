## Overview

Tile scheduling determines the order in which output tiles of a GEMM (or attention) kernel are assigned to CTAs. The scheduling order affects L2 cache hit rates, tail-effect severity, and overall GPU utilization. On Blackwell, the CLC hardware unit supports dynamic scheduling policies including swizzled raster, while Hopper relies on software-based static stride or swizzled patterns computed at launch time.

## Scheduling Strategies

### Linear Raster (Naive)

Tiles are assigned in row-major order. Simple but poor L2 locality: consecutive tiles share no B-matrix data until the entire M-dimension is traversed.

```cuda
// Linear raster: tile_idx maps directly to (tile_m, tile_n)
__device__ void linear_raster(int tile_idx, int tiles_n,
                               int& tile_m, int& tile_n) {
    tile_m = tile_idx / tiles_n;
    tile_n = tile_idx % tiles_n;
}

// Access pattern for a 4x4 tile grid:
//  0  1  2  3
//  4  5  6  7
//  8  9 10 11
// 12 13 14 15
//
// Problem: tiles 0,1,2,3 all load different B columns.
// By tile 4, B column 0 has been evicted from L2.
```

### Swizzled Raster

Tiles are assigned in a blocked pattern that groups nearby M and N tiles together, maximizing reuse of both A rows and B columns in L2 cache:

```cuda
// Swizzled raster: group tiles into blocks that share A and B data
// swizzle_size controls the block width (typically 4-8)
__device__ void swizzled_raster(int tile_idx, int tiles_m, int tiles_n,
                                 int swizzle_size, int& tile_m, int& tile_n)
{
    // Number of tile columns per swizzle group
    int group_cols = min(swizzle_size, tiles_n);
    int tiles_per_group = tiles_m * group_cols;

    // Which swizzle group
    int group_idx = tile_idx / tiles_per_group;
    int within_group = tile_idx % tiles_per_group;

    // Within the group, iterate in column-major order
    tile_m = within_group / group_cols;
    tile_n = group_idx * group_cols + within_group % group_cols;
}

// Access pattern with swizzle_size=2 on a 4x4 grid:
//  0  1 |  8  9
//  2  3 | 10 11
//  4  5 | 12 13
//  6  7 | 14 15
//
// Tiles 0,1,2,3 share the same B columns (0,1).
// Tiles 0,2,4,6 share the same A rows.
// Much better L2 reuse.
```

### Static Stride (Hopper Persistent)

Each CTA processes tiles at fixed intervals equal to the grid size:

```cuda
// Static stride: CTA i processes tiles i, i+gridDim.x, i+2*gridDim.x, ...
__device__ void static_stride(int cta_id, int total_ctas,
                               int iteration, int tiles_n,
                               int& tile_m, int& tile_n) {
    int tile_idx = cta_id + iteration * total_ctas;
    tile_m = tile_idx / tiles_n;
    tile_n = tile_idx % tiles_n;
}
```

### CLC Dynamic Scheduling (Blackwell)

The CLC hardware scheduler assigns tiles at runtime, combining the benefits of dynamic load balancing with configurable scheduling policies:

```cuda
// CLC-based scheduling on Blackwell
// The scheduling policy is set once during CLC initialization
enum class ClcSchedulePolicy {
    LinearRaster,       // Simple row-major order
    SwizzledRaster,     // Blocked pattern for L2 locality
    ColumnFirst,        // Column-major for specific workloads
    Hilbert             // Space-filling curve (experimental)
};

__device__ void clc_init_scheduler(
    void* clc_buffer,
    int tiles_m, int tiles_n,
    ClcSchedulePolicy policy)
{
    if (threadIdx.x == 0) {
        // Program the CLC with tile grid dimensions and policy
        // CLC handles the swizzle mapping internally
        uint32_t config = encode_clc_config(tiles_m, tiles_n, policy);
        asm volatile(
            "clusterctl.init.shared [%0], %1, %2, %3;"
            : : "r"((uint32_t)clc_buffer),
                "r"(tiles_m), "r"(tiles_n), "r"(config)
        );
    }
    __syncwarp();
}
```

## CUTLASS Tile Schedulers

CUTLASS provides several tile schedulers that abstract these strategies:

```cuda
// CUTLASS tile scheduler selection for SM100
// All persistent schedulers inherit from PersistentTileSchedulerSm100

// 1. Default CLC scheduler with swizzled raster
using Scheduler_Default = cutlass::gemm::PersistentTileSchedulerSm100;

// 2. Stream-K scheduler for better tail handling
//    Splits K-dimension across CTAs for the last wave
using Scheduler_StreamK = cutlass::gemm::StreamKSchedulerSm100;

// 3. Grouped GEMM scheduler for MoE workloads
//    Each group has different M, shared N and K
using Scheduler_Grouped = cutlass::gemm::GroupedTileSchedulerSm100;

// Usage in CUTLASS kernel definition:
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<Int<128>, Int<256>, Int<64>>,   // Tile shape
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementC, LayoutC,
    TiledMma,
    CollectiveMainloop,
    CollectiveEpilogue,
    Scheduler_Default                            // Tile scheduler
>;
```

## L2 Cache Locality Analysis

The choice of scheduling strategy directly impacts L2 cache hit rates. On B200 with 126 MB L2:

```python
# L2 cache reuse analysis for different schedulers
# Problem: M=8192, N=8192, K=4096, BF16
# Tile: 128x256, giving 64x32 = 2048 tiles
# B200: 142 SMs, 126 MB L2

tile_bytes_A = 128 * 4096 * 2  # 1 MB per tile row of A
tile_bytes_B = 4096 * 256 * 2  # 2 MB per tile column of B

# Linear raster: first wave loads 142 tiles across 142/32 = 4.4 column groups
# B data for 5 different column groups = 5 * 2 MB = 10 MB (fits in L2)
# But A data for 142/32 = 4.4 row groups * 4.4 col groups = ~20 distinct A rows
# 20 * 1 MB = 20 MB -> some L2 eviction

# Swizzled raster (swizzle=4): first wave covers 142 tiles in ~36 groups of 4
# Each group uses 1 A row + 4 B columns = 1 + 8 = 9 MB per group
# But groups share A rows: total unique A = ~36 rows * 1 MB = 36 MB
# L2 pressure: 36 MB + 8 MB = 44 MB (fits in B200's 126 MB L2)

# Conclusion: swizzled raster reduces L2 misses by ~2x vs linear for large problems
```

## Tail Effect Mitigation

The "tail effect" occurs when the last wave of tiles does not fully occupy all SMs. Different schedulers handle this differently:

| Scheduler | Tail Handling | SM Utilization (Last Wave) |
|-----------|---------------|---------------------------|
| Linear raster | None | `(total_tiles % num_SMs) / num_SMs` |
| Static stride | None | Same as linear |
| CLC dynamic | Automatic | Fast CTAs steal from slow ones |
| Stream-K | K-splitting | Near 100% (splits partial tiles across SMs) |

For a problem with 150 tiles on 142 SMs:
- Static: last wave has 8 tiles on 8 SMs, 134 SMs idle (5.6% utilization)
- CLC: fast-finishing CTAs from wave 1 absorb the 8 extra tiles
- Stream-K: the 8 remaining tiles are split across all 142 SMs

## When to Use

- **Swizzled raster**: Default choice for large GEMMs. Always better than linear for L2 locality.
- **CLC dynamic**: Recommended on Blackwell for all persistent kernels. Combines dynamic load balancing with swizzled ordering.
- **Stream-K**: Best for small-to-medium problems where the tail effect dominates. Adds complexity for K-dimension synchronization.
- **Grouped scheduler**: Essential for MoE and batched GEMM where problem sizes vary across groups.

## Caveats

- Swizzle size must be tuned per problem shape. Too large a swizzle group exceeds L2 capacity; too small loses the locality benefit.
- CLC scheduling adds a small latency per tile acquisition (~10s of cycles). For extremely small tiles, this overhead is proportionally larger.
- Stream-K requires atomic accumulation where K-splits meet, adding synchronization overhead. Only worthwhile when tail utilization is a proven bottleneck.
