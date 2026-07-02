# CUTLASS Tile Scheduling

CUTLASS 3.x provides multiple tile scheduling strategies that control how persistent kernels assign output tiles to thread blocks. This document covers Persistent Scheduler, Stream-K work stealing, Grouped GEMM scheduling, Dependent Kernel Launch (PDL), tile heuristics, and SM90/SM100/SM120 scheduler differences.


**Last updated**: 2026-06-30

## Tile Scheduler Overview

All CUTLASS 3.x GEMM kernels are persistent kernels -- thread blocks run continuously throughout the kernel's lifetimeuru, fetching the next tile to compute from the tile scheduler. The scheduler is passed to `GemmUniversal` via the template parameter `TileScheduler_`.

### Scheduler Types

```cpp
namespace cutlass::gemm {
 struct PersistentScheduler { }; // default: persistent scheduling
 struct StreamKScheduler { }; // Stream-K
  struct GroupScheduler { };            // Grouped/Ptr-Array GEMM
 struct DynamicPersistentScheduler { }; // persistent(SM100 CLC)
 struct StaticPersistentScheduler { }; // persistent(SM100 fallback)
}
```

`TileSchedulerSelector` routes to the concrete implementation based on the scheduler tag and architecture tag:

| Scheduler Tag | SM90 | SM100/SM103 | SM120 |
|--------------|------|-------------|-------|
| `void` (default) | `PersistentTileSchedulerSm90` | `PersistentTileSchedulerSm100` | `PersistentTileSchedulerSm100` |
| `PersistentScheduler` | `PersistentTileSchedulerSm90` | `PersistentTileSchedulerSm100` | `PersistentTileSchedulerSm100` |
| `StreamKScheduler` | `PersistentTileSchedulerSm90StreamK` | `PersistentTileSchedulerSm100StreamK` | `PersistentTileSchedulerSm100StreamK` |
| `GroupScheduler` | `PersistentTileSchedulerSm90Group` | `PersistentTileSchedulerSm100Group` | `PersistentTileSchedulerSm90Group` |
| `DynamicPersistentScheduler` | -- | `PersistentTileSchedulerSm100` | -- |
| `StaticPersistentScheduler` | `PersistentTileSchedulerSm90` | `StaticPersistentTileScheduler100` | -- |

## Persistent Tile Scheduler

The Persistent scheduler is CUTLASS's default tile scheduling strategy. Thread blocks do not exit after launch; they loop to fetch the next tile until all tiles are completed.

### Core Data Structures

```cpp
struct WorkTileInfo {
 int32_t M_idx = 0; // output tile M
 int32_t N_idx = 0; // output tile N
 int32_t L_idx = 0; // batch dimension
  bool is_valid_tile = false;

  bool is_valid() const { return is_valid_tile; }
 bool is_final_split(uint32_t k_tiles) const { return true; } // split-K true
};
```

### Swizzle-based Rasterization

The Persistent scheduler uses swizzle to improve L2 cache locality. Tile traversal is not simply row-major or column-major; instead, accesses are interleaved within swizzle blocks:

```cpp
// tile_scheduler_params.h swizzle compute
auto log_swizzle_size = get_log_swizzle_size(
    problem_blocks_m, problem_blocks_n, max_swizzle_size);
```

**Raster Order** determines the primary traversal direction:

```cpp
enum class RasterOrder {
 AlongM, // M dimension
 AlongN // N dimension(default)
};

enum class RasterOrderOptions {
 Heuristic, // automatic( M/N )
  AlongM,
  AlongN
};
```

**Swizzle Mechanism**: When mapping a linear tile index to 2D coordinates, tiles are first grouped by swizzle_size, and the M/N directions are interleaved within each group. swizzle_size is a power of 2, with a default upper limit of 1.

```cpp
// SM90 swizzle compute
offset = cluster_id & ((1 << log_swizzle_size) - 1);
extra = cluster_id >> log_swizzle_size;
divmod_cluster_blk_major(cluster_idx_minor_div_swizzle, cluster_idx_major, extra);
cluster_idx_minor = cluster_idx_minor_div_swizzle * (1 << log_swizzle_size) + offset;
```### SM90 Persistent Scheduler

`PersistentTileSchedulerSm90` inherits from `StaticPersistentTileScheduler` (CRTP):

- **Static Scheduling** (`IsDynamicPersistent = false`): tile assignment is determined at kernel launch time
- Thread block computes the initial tile index via `blockIdx` + `gridDim`, stepping by `gridDim` each round
- Grid size is limited by the SM count, ensuring ≥1 block per SM
- No additional workspace required

```cpp
class PersistentTileSchedulerSm90 :
    public StaticPersistentTileScheduler<PersistentTileSchedulerSm90> {
  static constexpr bool IsDynamicPersistent = false;
 // none Pipeline, none CLCResponse -- scheduling
  using Pipeline = PipelineEmpty;
};
```

### SM100 Persistent Scheduler (CLC-based)

`PersistentTileSchedulerSm100` uses Blackwell's CLC (Cluster Launch Control) hardware feature for dynamic scheduling:

```cpp
template<class ClusterShape_, uint32_t Stages_>
class PersistentTileSchedulerSm100 {
  static constexpr bool IsDynamicPersistent = true;
  static constexpr uint32_t Stages = Stages_;

 // CLC response 16 bytes
  struct CLCResponse { uint32_t data[4] = {0}; };

 // CLC asynchronous fetch pipeline
  using Pipeline = PipelineCLCFetchAsync<Stages, ClusterShape>;
 // pipeline
  using ThrottlePipeline = PipelineAsync<Stages>;

  class SharedStorage {
    PipelineStorage pipeline_;
    ThrottlePipelineStorage throttle_pipeline_;
 CLCResponse data_[Stages]; // stage CLC response buffer
  };
};
```

**Key Differences Between SM100 and SM90**:

| Feature | SM90 | SM100/SM120 |
|------|------|-------------|
| Scheduling Method | Static (blockIdx stepping) | Dynamic (CLC hardware dispatch) |
| IsDynamicPersistent | false | true |
| Pipeline | Empty | PipelineCLCFetchAsync |
| Shared Memory | No scheduler overhead | CLCResponse buffer |
| Cluster Support | Software emulation | Hardware CLC |
| Scheduler Pipeline | None | SchedulerPipelineStageCount |
| Accumulator Pipeline | None | AccumulatorPipelineStageCount |

The SM100 scheduler has a dedicated pipeline stage count parameter, allowing tile fetch and MMA computation to overlap, reducing scheduling latency.

## Stream-K Work Stealing

Stream-K is a fine-grained work decomposition strategy that splits GEMM along the K dimension, so that a single "work unit" can span partial K intervals across multiple output tiles.

### Core Concepts

In traditional data-parallel GEMM, each thread block computes a complete output tile (all K tiles). When the number of output tiles is not evenly divisible by the number of SMs, the tail "wave" leads to reduced SM utilization.

Stream-K evenly distributes the total workload by K-tile granularity across thread blocks, where the K-tile range processed by each block may cross tile boundaries.

### DecompositionMode

```cpp
enum class DecompositionMode {
 Heuristic, // automatic data-parallel / split-K / stream-K
 DataParallel, // data-parallel( block completecomplete output tile)
 SplitK, // split-K( splits parameteruse)
 StreamK // stream-K
};
```

### ReductionMode

Stream-K produces partial accumulation results that need reduction to combine:

```cpp
enum class ReductionMode {
 Deterministic, // : by K turnstile (requires)
 Nondeterministic // : atomic
};
```

### WorkTileInfo (Stream-K)

Stream-K's WorkTileInfo is more complex than the basic scheduler, containing K dimension information:

```cpp
struct WorkTileInfo {
  int32_t M_idx = 0;
  int32_t N_idx = 0;
 int32_t K_idx = 0; // current work unit K dimensionbit
  int32_t L_idx = 0;
 uint32_t k_tile_count = 0; // work unit compute K tile
 uint32_t k_tile_remaining = 0; // entire work unit K tile
 bool is_separate_reduction = false; // reduction work unit

  bool is_valid() const {
    return k_tile_count > 0 || is_separate_reduction;
  }
  bool is_final_split(uint32_t k_tiles_per_output_tile) const {
    return (K_idx + k_tile_count) == k_tiles_per_output_tile;
  }
};
```### Stream-K Arguments

```cpp
struct Arguments {
 int splits = 1; // split-K
 int max_swizzle_size = 1; // swizzle upper bound
  RasterOrderOptions raster_order = RasterOrderOptions::Heuristic;
  ReductionMode reduction_mode = ReductionMode::Deterministic;
  DecompositionMode decomposition_mode = DecompositionMode::Heuristic;
};
```

### Workspace

Stream-K requires a device-side workspace to store partial accumulation results:
- The workspace size is calculated by `get_workspace_size()`, depending on splits, number of output tiles, and element size
- Initialized before launch via `initialize_workspace()` (zeroing + barrier initialization)
- The thread block of the final split is responsible for performing the reduction and epilogue

### SM90 vs SM100 Stream-K

| Feature | SM90 | SM100 |
|------|------|-------|
| Implementation class | `PersistentTileSchedulerSm90StreamK` | `PersistentTileSchedulerSm100StreamK` |
| Base scheduler | `PersistentTileSchedulerSm90` | CLC-based |
| Pipeline | Empty | CLC Pipeline |
| Scheduler pipeline stages | None | Configurable |

## Grouped GEMM Scheduler

Grouped kernels execute multiple independent GEMM problems in a single CUDA launch, processing them sequentially through a persistent loop.

### Scheduling Logic

A thread block obtains the next tile to compute via `ProblemVisitor`:

```cpp
ProblemVisitor problem_visitor;

while (problem_visitor.next_tile()) {
 // tile index
 // execute MMA + epilogue
    problem_visitor.advance(gridDim.x);
}
```

**Round-robin allocation**: Tiles are distributed in turn based on thread block ID. For example, with 4 GEMM problems each having 4 tiles and 8 thread blocks, block 0 processes GEMM 0 tile 0, block 1 processes GEMM 0 tile 1, and so on.

### `next_tile()` Workflow

1. Maintains `tile_idx` (initialized to `blockIdx.x`, incremented by `gridDim.x` each round)
2. Starting from the last accessed GEMM, accumulates each GEMM's tile count into `problem_tile_start`
3. Finds the GEMM where `problem_tile_start <= tile_idx < problem_tile_start + tiles_in_problem`
4. Computes the tile coordinates within the GEMM via `tile_idx - problem_tile_start`

### GroupScheduleMode

```cpp
enum class GroupScheduleMode {
 kDeviceOnly, // default: device scheduling
 kHostPrecompute // maincomputescheduling
};
```

**kDeviceOnly**:
- Threads in each warp search in parallel: each thread "owns" a problem, and a warp-wide prefix sum computes the tile starting position
- No host-device communication required
- Suitable for scenarios where problem parameters are generated by a previous kernel

**kHostPrecompute**:
- Precomputes the `(problem_idx, problem_starting_tile)` array on the host and copies it to the device
- Reduces device-side scheduling overhead
- Suitable for small problem groups with low compute intensity, where host work can overlap with other kernels
- Redundant when each block computes multiple tiles of the same problem

### K-Dimension Load Balancing

In grouped GEMM, the K dimensions across problems can vary significantly. Round-robin allocation may cause some blocks to concentrate on large-K problems, leading to load imbalance.

**Solution**: Sort the problem list in descending order by K, allowing round-robin allocation to naturally interleave large and small K tiles.

```cpp
// usesort
grouped_gemm.sort_problems();
```

Sorting can yield approximately 30% speedup in certain scenarios, but is not guaranteed to be effective in all cases.

### SM90 vs SM100 Group Scheduler

| Feature | SM90 | SM100 |
|------|------|-------|
| Implementation class | `PersistentTileSchedulerSm90Group` | `PersistentTileSchedulerSm100Group` |
| Pipeline depth | `SchedulerPipelineStageCount` parameter | `SchedulerPipelineStageCount` parameter |

GroupScheduler on SM120 falls back to using `PersistentTileSchedulerSm90Group`.

## Dependent Kernel Launch (PDL)

PDL allows two adjacent kernels in the same CUDA stream to overlap in execution. Both Hopper and Blackwell architectures support this feature.

### How It Works

1. The preceding kernel (primary) signals when it is about to complete
2. The subsequent kernel (dependent) begins execution upon receiving the signal
3. The dependent kernel programmatically waits for the primary kernel to finish its memory writes
4. The tail of the first kernel and the head of the second kernel can overlap

### How to Enable

At compile time:
```bash
cmake . -DCUTLASS_ENABLE_GDC_FOR_SM90=1   # Hopper
cmake . -DCUTLASS_ENABLE_GDC_FOR_SM100=1  # Blackwell
```Runtime:
```cpp
gemm.run(stream, /*cuda_adapter=*/nullptr, /*launch_with_pdl=*/true);
```

### Weight Prefetch Optimization

In inference scenarios, the weight matrix is not produced by the previous kernel. Using PDL allows you to:
1. Only wait for the primary kernel to flush the activation matrix
2. Prefetch weights into shared memory during the prologue stage
3. Significantly accelerate memory bandwidth-bound problems (small batch, large K)

See CUTLASS example 63 (`hopper_gemm_with_weight_prefetch`) for details.

## Tile Heuristics

CUTLASS provides analytical heuristic tools `nvidia-matmul-heuristics` to assist in selecting kernel configurations.

### Coverage

- **Problem types**: plain dense GEMM (f8, f16, f32)
- **Hardware**: Hopper (SM9x), Blackwell (SM10x)

### Usage

1. Prepare a problem list JSON file (parameters such as m, n, k, dtype, layout)
2. Specify the problem file during CMake build:
```bash
cmake .. -DCUTLASS_NVCC_ARCHS=90a \
    -DCUTLASS_LIBRARY_HEURISTICS_PROBLEMS_FILE=problems.json \
    -DCUTLASS_LIBRARY_HEURISTICS_CONFIGS_PER_PROBLEM=5
```
3. The heuristic automatically ranks candidate kernel configurations for each problem
4. Output a testlist CSV for `cutlass_profiler` autotuning

### CMake Options

| Option | Description |
|------|------|
| `CUTLASS_LIBRARY_HEURISTICS_PROBLEMS_FILE` | Problem list JSON path |
| `CUTLASS_LIBRARY_HEURISTICS_CONFIGS_PER_PROBLEM` | Maximum number of configurations returned per problem |
| `CUTLASS_LIBRARY_HEURISTICS_RESTRICT_KERNELS` | Limit to building only the default kernel set |
| `CUTLASS_LIBRARY_HEURISTICS_TESTLIST_FILE` | Output CSV path |
| `CUTLASS_LIBRARY_HEURISTICS_GPU` | Specify target GPU (e.g., `H100_SXM5`) |

## Quick Reference for Scheduler Configuration

### PersistentScheduler (Default)

```cpp
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<
    cutlass::gemm::kernel::GemmUniversal<
        ProblemShape, CollectiveMainloop, CollectiveEpilogue,
 cutlass::gemm::PersistentScheduler // or void
    >>;

// Arguments
typename Gemm::GemmKernel::TileScheduler::Arguments scheduler_args;
scheduler_args.max_swizzle_size = 1; // swizzle upper bound(2 )
scheduler_args.raster_order = RasterOrderOptions::Heuristic;
```

### StreamKScheduler

```cpp
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<
    cutlass::gemm::kernel::GemmUniversal<
        ProblemShape, CollectiveMainloop, CollectiveEpilogue,
        cutlass::gemm::StreamKScheduler
    >>;

// Arguments
typename Gemm::GemmKernel::TileScheduler::Arguments scheduler_args;
scheduler_args.splits = 1; // >1 split-K
scheduler_args.max_swizzle_size = 1;
scheduler_args.raster_order = RasterOrderOptions::Heuristic;
scheduler_args.reduction_mode = ReductionMode::Deterministic;
scheduler_args.decomposition_mode = DecompositionMode::Heuristic;
```

### GroupScheduler

```cpp
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<
    cutlass::gemm::kernel::GemmUniversal<
        GroupProblemShape, CollectiveMainloop, CollectiveEpilogue,
        cutlass::gemm::GroupScheduler
    >>;
```

## Related

- [CUTLASS 3.x Architecture](cutlass-3x-architecture.md) -- Four-level hierarchy and dispatch policy system
- [CuTe Fundamentals](cutlass-cute-fundamentals.md) -- Layout algebra
- [CUTLASS GEMM Optimization Strategies](cutlass-gemm-optimization.md) -- Tiling and performance tuning
- [CuTeDSL Software Pipeline](cutedsl-pipeline-patterns.md) -- Detailed pipeline patterns
