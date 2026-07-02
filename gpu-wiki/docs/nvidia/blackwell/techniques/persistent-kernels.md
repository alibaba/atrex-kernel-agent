# Persistent Kernels with CLC


**Last updated**: 2026-06-30

## Overview

Persistent kernels launch exactly as many CTAs as SMs, and each CTA processes multiple output tiles in a loop rather than exiting after one tile. On Blackwell, the CLC (Cluster Launch Control) hardware unit replaces software-based tile scheduling with a hardware-assisted mechanism. Each CTA queries the CLC for its next tile assignment and can cancel itself when no work remains, using the `try_cancel` pattern.

## CLC Loop Pattern

The core persistent kernel loop on Blackwell uses CLC to dynamically assign tiles:

```cuda
// Persistent kernel with CLC tile scheduling (Blackwell SM100)
__global__ void __launch_bounds__(512)
persistent_gemm_clc(const __grid_constant__ GemmParams params)
{
    // CLC-managed persistent loop: each CTA processes multiple tiles
    while (true) {
        // Query CLC for next tile assignment
        // Returns tile coordinates (tile_m, tile_n) or signals termination
        TileCoord tile;
        bool has_work = clc_try_get_tile(&tile);

        if (!has_work) {
            // No more tiles to process -- CTA exits
            // clc_try_cancel atomically checks if all tiles are done
            if (clc_try_cancel()) {
                return;  // CTA terminates
            }
            continue;  // Race condition: another CTA may have generated work
        }

        // Standard GEMM tile computation
        int tile_m = tile.m;
        int tile_n = tile.n;

        // TMA producer loads A[tile_m, :] and B[:, tile_n] tiles
        // MMA consumer accumulates K-dimension
        // Epilogue writes C[tile_m, tile_n]
        compute_gemm_tile(params, tile_m, tile_n);
    }
}
```

At the PTX level, the CLC interaction uses dedicated instructions:

```ptx
// CLC tile acquisition in PTX
// clusterctl.try_cancel cancels the CTA if no work remains
.reg .pred  %has_work;
.reg .b32   %tile_m, %tile_n;

TILE_LOOP:
    // Attempt to get next tile from CLC hardware scheduler
    clusterctl.query.async.shared  [%smem_tile_desc];
    clusterctl.wait;

    // Check if valid tile was assigned
    ld.shared.b32  %has_work, [%smem_tile_desc + 0];
    @!%has_work bra TRY_CANCEL;

    // Extract tile coordinates
    ld.shared.b32  %tile_m, [%smem_tile_desc + 4];
    ld.shared.b32  %tile_n, [%smem_tile_desc + 8];

    // ... compute tile ...

    bra TILE_LOOP;

TRY_CANCEL:
    // Atomically try to cancel this CTA
    clusterctl.try_cancel  %cancelled;
    @!%cancelled bra TILE_LOOP;   // Another CTA may have pushed work
    ret;
```

## Comparison: CLC vs Static Stride (Hopper)

On Hopper (SM90), persistent kernels use a static stride pattern where each CTA computes tiles at fixed intervals:

```cuda
// Hopper-style static stride persistent kernel
__global__ void hopper_persistent_gemm(GemmParams params)
{
    int cta_id = blockIdx.x;
    int total_ctas = gridDim.x;
    int total_tiles = params.num_tiles_m * params.num_tiles_n;

    // Static stride: CTA i handles tiles i, i+total_ctas, i+2*total_ctas, ...
    for (int tile_idx = cta_id; tile_idx < total_tiles; tile_idx += total_ctas) {
        int tile_m = tile_idx / params.num_tiles_n;
        int tile_n = tile_idx % params.num_tiles_n;
        compute_gemm_tile(params, tile_m, tile_n);
    }
}
```

| Aspect | Hopper Static Stride | Blackwell CLC |
|--------|---------------------|---------------|
| Scheduling | Software loop with fixed stride | Hardware CLC unit assigns tiles |
| Load balancing | Fixed; uneven if tile costs vary | Dynamic; CLC rebalances automatically |
| Tail effect | Last wave may have partial occupancy | CLC minimizes by giving fast CTAs more tiles |
| Launch overhead | Grid launch for each new problem | CLC can chain multiple problems |
| Termination | Implicit when loop ends | Explicit `try_cancel` |
| L2 locality | Depends on stride pattern | CLC can apply swizzled raster |

## CUTLASS PersistentTileSchedulerSm100

CUTLASS 4.5.0 provides `PersistentTileSchedulerSm100` that wraps the CLC hardware:

```cuda
// CUTLASS SM100 persistent tile scheduler (simplified)
template <class TileShape>
struct PersistentTileSchedulerSm100 {

    // Initialize the CLC with the problem geometry
    CUTLASS_DEVICE static void init(
        dim3 problem_tiles,
        void* clc_smem_buffer)
    {
        if (threadIdx.x == 0) {
            // Program CLC with total tile count and scheduling policy
            clc_init(clc_smem_buffer,
                     problem_tiles.x,  // tiles along M
                     problem_tiles.y,  // tiles along N
                     ClcPolicy::SwizzledRaster);
        }
        __syncthreads();
    }

    // Shared storage for CTA-wide CLC result broadcast
    // __shfl_sync is warp-local and cannot reach warps 1-15.
    struct SharedClcState {
        int tile_m, tile_n;
        int valid;       // 1 = got tile, 0 = no more work
        int cancelled;
    };

    // Get next tile assignment from CLC
    CUTLASS_DEVICE static bool get_next_tile(
        void* clc_smem_buffer,
        SharedClcState& shared_clc,
        int& tile_m,
        int& tile_n)
    {
        if (threadIdx.x == 0) {
            int m, n;
            bool v = clc_query_tile(clc_smem_buffer, m, n);
            shared_clc.tile_m = m;
            shared_clc.tile_n = n;
            shared_clc.valid  = v ? 1 : 0;
        }
        __syncthreads();  // All warps see the result
        tile_m = shared_clc.tile_m;
        tile_n = shared_clc.tile_n;
        return shared_clc.valid != 0;
    }

    // Try to cancel the CTA when no more work
    CUTLASS_DEVICE static bool try_cancel(
        void* clc_smem_buffer,
        SharedClcState& shared_clc)
    {
        if (threadIdx.x == 0) {
            shared_clc.cancelled = clc_try_cancel(clc_smem_buffer) ? 1 : 0;
        }
        __syncthreads();
        return shared_clc.cancelled != 0;
    }
};
```

## Performance Impact

The tcgen05-tutorial progression demonstrates the impact of persistent kernels:

```
Without persistence (static grid):  940 TFLOPS  (62% of peak)
With CLC persistent scheduling:    1476 TFLOPS  (98% of cuBLAS)
```

The 57% improvement comes from:
1. **Eliminated tail effect**: CLC dynamically assigns tiles, so fast-completing CTAs absorb extra work rather than sitting idle while the last wave finishes.
2. **Reduced launch overhead**: A single kernel launch covers all tiles; no need to re-launch grids.
3. **Better L2 cache utilization**: CLC can apply a swizzled raster pattern that improves spatial locality across neighboring tiles.

## When to Use

- **Large GEMM problems**: Persistent kernels are most beneficial when the number of output tiles exceeds the SM count by at least 2-3x.
- **Grouped GEMMs / MoE**: CLC can chain multiple problem instances, eliminating inter-kernel launch gaps.
- **Workloads with uneven tile cost**: CLC's dynamic scheduling naturally handles variable-cost tiles (e.g., triangular attention masks).

## Caveats

- CLC is SM100-only; Hopper kernels must use software-based scheduling.
- The `try_cancel` pattern introduces a potential race that must be handled with a retry loop.
- For very small problems (fewer tiles than SMs), CLC overhead may not justify the complexity. A simple single-wave grid launch suffices.


## Related

- [Cache Policy Differentiation](cache-policy.md)
- [Chunk-Based Parallelism](chunk-parallelism.md)
- [CUDA GEMM Optimization Ladder](cuda-gemm-optimization-ladder.md)
- [Double/Multi-Buffering Patterns](double-buffering.md)
- [Epilogue Fusion](epilogue-fusion.md)
- [Composable Kernel (CK) Architecture Overview](../../../amd/common/ck-architecture-overview.md)
