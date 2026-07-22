# SM100 Blackwell CuTeDSL Panorama

The SM100 (Blackwell) architecture introduces the 5th generation Tensor Core (tcgen05), dedicated accumulator TMEM, 2SM cooperative MMA, and Cluster Launch Control (CLC) dynamic scheduling. This article dissects these new mechanisms from the CUTLASS/CuTeDSL code level, benchmarking against [SM90 CuTeDSL Special Features](../../../hopper/ref-docs/cutedsl/hopper-cutedsl-sm90.md), and compares the differences with SM120 (Blackwell GeForce).

---

## 1. UMMA — 5th Generation Tensor Core

### 1.1 tcgen05.mma Instructions

SM100 replaces Hopper's WGMMA with `tcgen05.mma`, delivering **2x-4x** throughput over Hopper:

| Instruction Kind | Throughput (vs Hopper) | Operand Types |
|-----------|-----------------|-----------|
| `tf32` | 2x | A/B = tf32 |
| `f16` | 2x | A/B = f16/bf16 |
| `i8` | 2x | A/B = i8/u8 |
| `f8f6f4` | 2x | A/B can mix f4/f6/f8 |
| `mxf8f6f4.block_scale` | 2x | block-scaled mx types |
| `mxf4.block_scale` | **4x** | A/B = mxf4 (TN only) |
| `mxf4nvf4.block_scale` | **4x** | A/B = mxf4/nvf4 (TN only) |

### 1.2 UMMA::DescriptorIterator — No SmemCopyAtom

Like SM90 WGMMA, SM100 UMMA reads operands directly from SMEM descriptors, requiring **no explicit smem-to-rmem copy atom**. This is enforced in CUTLASS via static_assert:

```cpp
// sm100_mma_warpspecialized.hpp
static_assert(cute::is_void_v<SmemCopyAtomA>,
    "SM100 UMMA cannot have a non-void copy atom for smem sourced instructions.");
static_assert(cute::is_void_v<SmemCopyAtomB>,
    "SM100 UMMA cannot have a non-void copy atom for smem sourced instructions.");
```

Both operand A and B fragment types derive from `UMMA::DescriptorIterator`:

```cpp
static_assert(cute::is_base_of<cute::UMMA::DescriptorIterator, typename TiledMma::FrgTypeA>::value &&
              cute::is_base_of<cute::UMMA::DescriptorIterator, typename TiledMma::FrgTypeB>::value,
              "MMA atom must source both A and B operand from smem_desc for this mainloop.");
```

### 1.3 1SM vs 2SM MMA Atom

`AtomThrShapeMNK` defines how many SMs cooperate on a single MMA instruction:

```cpp
// TiledMma::ThrLayoutVMNK 0
using AtomThrShapeMNK = Shape<decltype(shape<0>(typename TiledMma::ThrLayoutVMNK{})), _1, _1>;
// 1SM: AtomThrShapeMNK = Shape<_1,_1,_1>  → size = 1
// 2SM: AtomThrShapeMNK = Shape<_2,_1,_1>  → size = 2
```

| Attribute | 1SM | 2SM |
|------|-----|-----|
| Dispatch Policy | `KernelTmaWarpSpecialized1SmSm100` | `KernelTmaWarpSpecialized2SmSm100` |
| MMA Tile M Range | 64-128 | 128-256 |
| TMA copy atom | `SM90_TMA_LOAD` / `SM90_TMA_LOAD_MULTICAST` | `SM100_TMA_2SM_LOAD` / `SM100_TMA_2SM_LOAD_MULTICAST` |
| Cluster M Constraint | No special requirement | Must be a multiple of 2 |
| TMEM Allocator | `Allocator1Sm` | `Allocator2Sm` |

### 1.4 UMMA vs GMMA (SM90) Comparison

| Dimension | SM90 GMMA | SM100 UMMA |
|------|-----------|------------|
| Thread Scale | 1 warpgroup (128 threads) | 1 warp (32 threads) per SM |
| Accumulator Storage | Register file | **TMEM** (dedicated tensor memory) |
| SmemCopyAtom | void (descriptor-based) | void (descriptor-based) |
| Multi-SM Cooperation | None | 2SM MMA (`cta_group::2`) |
| Descriptor Format | GMMA descriptor | UMMA descriptor (compatible with GMMA swizzle modes) |
| SMEM Layout Atoms | `GMMA::Layout_*_Atom` | Reuses `GMMA::Layout_*_Atom` + new `Layout_MN_SW128_32B_Atom` |

### 1.5 Runtime Data Types

SM100 supports selecting FP8/FP6/FP4 types at runtime (rather than being bound at compile time), controlled via the `IsRuntimeDataType` tag:

```cpp
static constexpr bool IsRuntimeDataTypeA = detail::is_sm10x_runtime_f8f6f4<ElementA>();
static constexpr bool IsRuntimeDataTypeB = detail::is_sm10x_runtime_f8f6f4<ElementB>();

// must runtime or static
static_assert((IsRuntimeDataTypeA && IsRuntimeDataTypeB) ||
              (!IsRuntimeDataTypeA && !IsRuntimeDataTypeB),
              "ElementA and ElementB should be both runtime or both static.");
```The runtime type is passed into the kernel via the ``UMMA::MXF8F6F4Format`` enum:

```cpp
using RuntimeDataTypeA = cute::conditional_t<IsRuntimeDataTypeA, cute::UMMA::MXF8F6F4Format, void*>;
using RuntimeDataTypeB = cute::conditional_t<IsRuntimeDataTypeB, cute::UMMA::MXF8F6F4Format, void*>;
```

---

## 2. TMEM — Dedicated Tensor Accumulator Memory

### 2.1 Core Concepts

SM100 introduces **Tensor Memory (TMEM)**, specifically designed bulls for storing MMA accumulator results. This relieves pressure on the register file, allowing the 5 warps to focus on their respective roles without competing for registers.

### 2.2 TmemAllocator

Select different allocators based on 1SM/2SM:

```cpp
// sm100_gemm_tma_warpspecialized.hpp
using TmemAllocator = cute::conditional_t<
    cute::size(cute::shape<0>(typename TiledMma::ThrLayoutVMNK{})) == 1,
    cute::TMEM::Allocator1Sm,
    cute::TMEM::Allocator2Sm>;
```

The MMA warp performs TMEM allocation at kernel entry:

```cpp
// MMA warp entire TMEM
tmem_allocator.allocate(TmemAllocator::Sm100TmemCapacityColumns, &shared_storage.tmem_base_ptr);
__syncwarp();
tmem_allocation_result_barrier.arrive; // notify epilogue warp

uint32_t tmem_base_ptr = shared_storage.tmem_base_ptr;
collective_mainloop.set_tmem_offsets(tmem_storage, tmem_base_ptr);
```

Free before kernel exit:

```cpp
tmem_allocator.release_allocation_lock();
// ... wait epilogue complete ...
tmem_allocator.free(tmem_base_ptr, TmemAllocator::Sm100TmemCapacityColumns);
```

### 2.3 set_tmem_offsets() Configure Base Address

After allocation, the base address must be written into the accumulator tensor's data pointer:

```cpp
template <class TmemStorage>
CUTLASS_DEVICE static void
set_tmem_offsets(TmemStorage& tmem_storage, uint32_t tmem_base_addr) {
    tmem_storage.accumulators.data() = tmem_base_addr;
}
```

### 2.4 Double-Buffered Accumulator (AccumulatorPipelineStageCount)

TMEM accumulators support double buffering — while the MMA writes to one stage, the epilogue can read from another stage:

```cpp
static constexpr uint32_t AccumulatorPipelineStageCount =
    DispatchPolicy::Schedule::AccumulatorPipelineStageCount;

// tensor by stage
// ((MMA_TILE_M,MMA_TILE_N),MMA_M,MMA_N,ACC_PIPE)
Tensor accumulators = cutlass::detail::make_sm100_accumulator<
    AccumulatorPipelineStageCount, IsOverlappingAccum>(
    tiled_mma, acc_shape, EpilogueTile{});
```

After MMA completes a tile, commit:

```cpp
accumulator_pipeline.producer_commit(accumulator_pipe_producer_state);
++accumulator_pipe_producer_state;
```

### 2.5 PipelineUmmaAsync

The TMEM accumulator pipeline uses ``PipelineUmmaAsync<Stages, AtomThrShapeMNK>`` to synchronize the MMA (producer) with the epilogue (consumer):

```cpp
using AccumulatorPipeline = cutlass::PipelineUmmaAsync<
    AccumulatorPipelineStageCount, AtomThrShapeMNK>;
```

In 2SM mode, the producer commit uses multicast arrive to notify the peer SM:

```cpp
// PipelineUmmaAsync::producer_commit
if constexpr (is_2sm_mma) {
    cutlass::arch::umma_arrive_multicast_2x1SM(smem_ptr, tmem_sync_mask_);
} else {
    cutlass::arch::umma_arrive(smem_ptr);
}
```

---

## 3. 2SM MMA Collaboration

### 3.1 Two SMs Collaboratively Execute One MMA

When ``AtomThrShapeMNK = Shape<_2,_1,_1>``, two adjacent CTAs in the cluster share TMEM and collaboratively execute the ``tcgen05.mma.cta_group::2`` instruction. The M-dimension tile is split evenly between the two SMs.

### 3.2 TMA Copy Atom Selection

```cpp
// 1SM usestandard TMA
SM90_TMA_LOAD / SM90_TMA_LOAD_MULTICAST

// 2SM use 2SM TMA
SM100_TMA_2SM_LOAD / SM100_TMA_2SM_LOAD_MULTICAST
```

2SM TMA loads double the data in one shot, because it needs to fill the SMEM of two SMs:

### 3.3 calculate_umma_peer_mask()

Track which CTAs are TMEM peers of the current CTA:

```cpp
template<class ClusterShape, class AtomThrShape_MNK>
CUTLASS_DEVICE
uint16_t calculate_umma_peer_mask(ClusterShape cluster_shape,
                                   AtomThrShape_MNK atom_thr_shape,
                                   dim3 block_id_in_cluster) {
    uint16_t tmem_sync_mask = 0;
    auto cluster_layout = make_layout(cluster_shape);
    int block_id_x = (block_id_in_cluster.x / size<0>(AtomThrShape_MNK{}))
                   * size<0>(AtomThrShape_MNK{});
    int block_id_y = (block_id_in_cluster.y / size<1>(AtomThrShape_MNK{}))
                   * size<1>(AtomThrShape_MNK{});
    for (int x = 0; x < size<0>(AtomThrShape_MNK{}); x++) {
        for (int y = 0; y < size<1>(AtomThrShape_MNK{}); y++) {
            tmem_sync_mask |= (1 << cluster_layout(block_id_x + x, block_id_y + y, Int<0>{}));
        }
    }
    return tmem_sync_mask;
}
```

### 3.4 Cluster Layout and TMA Atom Construction

For 2SM MMA, the M dimension of the cluster is split by `AtomThrID`, affecting the TMA multicast partition:

```cpp
// Cluster layout AtomThrID
auto cluster_layout_vmnk = tiled_divide(
    make_layout(cluster_shape),
    make_tile(typename TiledMma::AtomThrID{}));
auto cta_coord_vmnk = cluster_layout_vmnk.get_flat_coord(block_rank_in_cluster);

// TMA partition use layout
auto [tAgA_mkl, tAsA] = tma_partition(*observed_tma_load_a_,
    get<2>(cta_coord_vmnk), make_layout(size<2>(cta_layout_vmnk)),
    group_modes<0,3>(sA), group_modes<0,3>(tCgA_mkl));
```

---

## 4. CLC (Cluster Launch Control) Pipeline

### 4.1 Dynamic vs. Static Persistent Scheduling

| Scheduling Method | SM90 | SM100 CLC |
|---------|------|-----------|
| Persistence Method | Software static tile scheduler | Hardware CLC dynamic allocation |
| Load Balancing | Unable to sense SM availability | Runtime on-demand tile allocation |
| Core Instruction | None | `clusterlaunchcontrol.try_cancel` |
| Grid Size | SM count order of magnitude | Equal to total output tile count |

### 4.2 PipelineCLCFetchAsync

CLC queries are pipelined, with the scheduler warp asynchronously prefetching the next tile information:

```cpp
template <int Stages_, class ClusterShape = Shape<int,int,_1>>
class PipelineCLCFetchAsync {
public:
    static constexpr uint32_t Stages = Stages_;

    struct Params {
 uint32_t transaction_bytes = 0; // 16 bytes (CLC response)
        ThreadCategory role;
 uint32_t producer_blockid = 0; // 0 CTA producer
 uint32_t producer_arv_count = 1; // 1 elected thread
 uint32_t consumer_arv_count; //
    };
};
```

In the kernel, the CLC pipeline depth is typically **3 stages**:

```cpp
// configuration
static constexpr uint32_t SchedulerPipelineStageCount =
    DispatchPolicy::Schedule::SchedulerPipelineStageCount;  // = 3

using CLCPipeline = cutlass::PipelineCLCFetchAsync<
    SchedulerPipelineStageCount, ClusterShape>;
```### 4.3 Scheduling Loop

The scheduler warp (warp 1) is the producer of the CLC pipeline. Its work loop:

```cpp
// Scheduler warp loop
do {
 // 1. : wait mainloop load
    clc_throttle_pipeline.consumer_wait(clc_pipe_throttle_consumer_state);
    clc_throttle_pipeline.consumer_release(clc_pipe_throttle_consumer_state);

 // 2. CLC , next ClcID
    clc_pipe_producer_state = scheduler.advance_to_next_work(
        clc_pipeline, clc_pipe_producer_state);

 // 3. warp CLC result
    auto [next_work_tile_info, increment_pipe] = scheduler.fetch_next_work(
        work_tile_info, clc_pipeline, clc_pipe_consumer_state);
} while (work_tile_info.is_valid());
```

The first CLC tile comes from `blockIdx` (statically allocated) and does not require a CLC query.

---

## 5. 5-Warp Role Architecture

### 5.1 Role Assignment

The SM100 kernel divides threads within a threadblock into **5 roles**, each role occupying **1 warp (32 threads)** (epilogue can use multiple warps):

```cpp
enum class WarpCategory : int32_t {
 MMA = 0, // 1 warp - execute tcgen05.mma
 Sched = 1, // 1 warp - CLC scheduling( cluster 0 CTA)
 MainloopLoad = 2, // 1 warp - TMA load A/B
 EpilogueLoad = 3, // 1 warp - TMA load C/aux
 Epilogue = 4, // 1-4 warps - epilogue compute + TMA store
};
```

Total thread count and role confirmation:

```cpp
static constexpr uint32_t NumSchedThreads        = 32;  // 1 warp
static constexpr uint32_t NumMMAThreads          = 32;  // 1 warp
static constexpr uint32_t NumMainloopLoadThreads = 32;  // 1 warp
static constexpr uint32_t NumEpilogueLoadThreads = 32;  // 1 warp
static constexpr uint32_t NumEpilogueThreads     = CollectiveEpilogue::ThreadCount;

static constexpr uint32_t MaxThreadsPerBlock =
    NumSchedThreads + NumMainloopLoadThreads + NumMMAThreads +
    NumEpilogueLoadThreads + NumEpilogueThreads;
```

### 5.2 Comparison with SM90's 3 Warp Group

| Dimension | SM90 Hopper | SM100 Blackwell |
|------|-------------|-----------------|
| Basic scheduling unit | Warp Group (128 threads, 4 warps) | Warp (32 threads) |
| MMA thread count | 128 (1 warpgroup) | 32 (1 warp) + TMEM offloaded accumulator |
| Producer threads | 1 warp (32 threads) | 1 warp (32 threads) |
| Scheduler role | No independent role, inlined within MMA warpgroup | Independent Sched warp (warp 1) |
| Epilogue | Consumer warpgroup also handles it | Independent Epilogue warp(s) |
| Ping-Pong | 2 consumer warpgroup alternating | Accumulator double buffering (TMEM) |
| Registers/MMA | 255 regs/thread x 128 threads | 255 regs/thread x 32 threads + TMEM |

### 5.3 Participant Determination

Each warp checks its own role and participation conditions:

```cpp
IsParticipant is_participant = {
    (warp_category == WarpCategory::MMA),                                 // mma
 (warp_category == WarpCategory::Sched) && is_first_cta_in_cluster, // sched: CTA 0
    (warp_category == WarpCategory::MainloopLoad),                        // main_load
 (warp_category == WarpCategory::EpilogueLoad) && is_epi_load_needed, // epi_load: by
    (warp_category == WarpCategory::Epilogue)                             // epilogue
};
```

---

## 6. Dynamic Cluster

### 6.1 IsDynamicCluster

Dynamic cluster is enabled when `ClusterShape` contains non-static dimensions:

```cpp
static constexpr bool IsDynamicCluster = not cute::is_static_v<ClusterShape>;
```

Dynamic cluster allows the optimal cluster shape to be selected at runtime based on problem size and GPU resources.

### 6.2 Fallback TMA Descriptors

Dynamic clusters must prepare a set of TMA descriptors for both the primary cluster shape and the fallback shape:

```cpp
struct Params {
 TMA_A tma_load_a; // main cluster shape TMA
    TMA_B tma_load_b;
 TMA_A tma_load_a_fallback; // fallback cluster shape TMA
    TMA_B tma_load_b_fallback;
    dim3 cluster_shape_fallback;
    // ...
};
```

The device side selects the appropriate TMA based on the actual cluster shape:

```cpp
if constexpr (IsDynamicCluster) {
    const bool is_fallback_cluster =
        (cute::size<0>(cluster_shape_) == params.cluster_shape_fallback.x &&
         cute::size<1>(cluster_shape_) == params.cluster_shape_fallback.y);
    observed_tma_load_a_ = is_fallback_cluster
        ? &params.tma_load_a_fallback : &params.tma_load_a;
    observed_tma_load_b_ = is_fallback_cluster
        ? &params.tma_load_b_fallback : &params.tma_load_b;
}
```

### 6.3 Use Cases

- A fallback is needed when the problem shape is not divisible by the cluster shape
- When multiple kernels coexist, SM resources may be insufficient Brock support large clusters
- Block-scaled GEMM limits each cluster dimension to ≤4 (scale factor multicast limitation)

---

## 7. SM100 Tile Shape and Dispatch Policy Quick Reference

### 7.1 Legacy Types (f16/bf16/tf32/i8)

| SM Mode | Tile Shape M | Tile Shape N | Dispatch Policy |
|---------|-------------|-------------|-----------------|
| 1SM | 64, 128 | 64, 128, 192, 256 | `KernelTmaWarpSpecialized1SmSm100` |
| 2SM | 128, 256 | 64, 128, 192, 256 | `KernelTmaWarpSpecialized2SmSm100` |

The K dimension is always `4 * MMA_K`. All four layouts (TN/TT/NT/NN) are supported.

### 7.2 Narrow Precision (f8/f6/f4)

Non-block-scaled narrow precision only supports tiles with K=128. Layout support depends on the type combination.

### 7.3 Block-Scaled Types

| Block-Scaled Type | Dispatch Policy Suffix | Characteristics |
|---|---|---|
| mxf8/mxf6 mixed | `Mxf8f6f4Sm100` | All four layouts |
| mxf4 x mxf4 (TN only) | `Mxf4Sm100` | 4x throughput, TN only |
| nvf4 x nvf4 (TN only) | `Nvf4Sm100` | 4x throughput, TN only |

---

## 8. SM120 (Blackwell GeForce) Differences

SM120 targets consumer-grade Blackwell GPUs (RTX 50 series), supporting narrow-precision MMA but **without UMMA/TMEM/CLC**.

### 8.1 Architecture Comparison

| Feature | SM100 (B200) | SM120 (RTX 5090) |
|------|-------------|-----------------|
| Tensor Core | tcgen05 (UMMA) | mma.sync (register-resident) |
| Accumulator | TMEM | Register file |
| SmemCopyAtom | void (descriptor direct read) | **Must be explicitly specified** (smem -> rmem) |
| MMA Threads | 1 warp (32) | 8 warps (256) cooperative/pingpong |
| TMA | SM100_TMA_2SM_LOAD or SM90_TMA_LOAD | SM90_TMA_LOAD |
| CLC Scheduling | PipelineCLCFetchAsync | Traditional tile scheduler |
| Cluster | Supports multicast | **Fixed 1x1x1** (no multicast) |
| Layout | TN/TT/NT/NN depending on type | **TN only** |
| 2SM MMA | Supported | Not supported |

### 8.2 SmemCopyAtom is Mandatory

SM120 uses register-resident MMA, so an explicit smem->rmem copy is required:

```cpp
// sm120_mma_tma.hpp
static_assert(not cute::is_void_v<SmemCopyAtomA>,
    "SM120 mainloop must specify a copy atom for A operand smem->rmem reads.");
static_assert(not cute::is_void_v<SmemCopyAtomB>,
    "SM120 mainloop must specify a copy atom for B operand smem->rmem reads.");
```

### 8.3 F8F6F4 unpacksmem Types

SM120 uses `unpacksmem` variants for sub-byte types, with TMA automatically unpacking on load:

```cpp
using TmaInternalElementA = cute::conditional_t<
    cute::is_same_v<ElementA, cutlass::float_e2m1_t>,
    cutlass::detail::float_e2m1_unpacksmem_t,    // 4-bit unpack
    cute::conditional_t<cute::is_same_v<ElementA, cutlass::float_e2m3_t>,
        cutlass::detail::float_e2m3_unpacksmem_t,  // 6-bit unpack
        cute::conditional_t<cute::is_same_v<ElementA, cutlass::float_e3m2_t>,
            cutlass::detail::float_e3m2_unpacksmem_t,
            uint_bit_t<sizeof_bits_v<ElementA>>>>>;
```### 8.4 Kernel Schedule

SM120 uses a warpgroup mode similar to Hopper, with two kernel schedules:

| Schedule | Description |
|----------|------|
| `KernelTmaWarpSpecializedCooperativeSm120` | 8 MMA warps collaborate on the same output tile (default) |
| `KernelTmaWarpSpecializedPingpongSm120` | 2 groups of 4 MMA warps each alternate processing different tiles, overlapping mainloop and epilogue |

SM120 register allocation budget:

```cpp
static constexpr uint32_t LoadRegisterRequirement = 40;
static constexpr uint32_t MmaRegisterRequirement = 232;
```

### 8.5 SM120 Tile Shape Constraints

SM120 tile shape choices are more limited than SM100:

| Type | Available Tile Shapes | Schedule |
|------|----------------|----------|
| f8/f6/f4 (non block-scaled) | 64x64, 64x128, 128x64, 128x128 (K=128) | Cooperative or Pingpong |
| nvf4/mxf4 (block-scaled) | 128x128x128, 128x128x256, 256x128x128 | Cooperative (256x only Cooperative) |
| mxf8f6f4 mixed | 128x128x128 | Cooperative or Pingpong |

---

## 9. Three-Generation Architecture Core Comparison

| Feature | SM90 (Hopper) | SM100 (Blackwell) | SM120 (GeForce) |
|------|--------------|-------------------|-----------------|
| MMA instruction | WGMMA | tcgen05 (UMMA) | mma.sync |
| MMA threads | 128 (warpgroup) | 32 (1 warp) | 256 (8 warps) |
| Accumulator | Register File | TMEM | Register File |
| Smem->Rmem Copy | void | void | Explicit SmemCopyAtom |
| TMA | SM90_TMA_LOAD | SM90/SM100_TMA_2SM | SM90_TMA_LOAD |
| Cluster scheduling | Static persistent | CLC dynamic persistent | Static (1x1x1) |
| Layout support | TN/TT/NT/NN | TN/TT/NT/NN | TN only |
| Block-scaled | Not supported | Supported (mxf4 4x) | Supported (mxf4 4x) |
| Sparse MMA | Not supported | Supported | Supported |
| SMEM capacity | 228 KB | Larger | -- |

---

## 10. Related Documents

- [SM90 CuTeDSL Special Features](../../../hopper/ref-docs/cutedsl/hopper-cutedsl-sm90.md) — Hopper WGMMA/TMA/Pipeline comparison
- [CuTeDSL Pipeline Patterns](../../../common/ref-docs/cutedsl/cutedsl-pipeline-patterns.md) — Producer-Consumer pipeline paradigms
- [NVIDIA CuTeDSL Architecture Primitives](../../../common/ref-docs/cutedsl/nvidia-cutedsl-arch-primitives.md) — TMA, barrier, cluster fundamentals
- [Blackwell SM100 Optimization Practice](../../kernel-opt/hands-on) — tcgen05, TMEM, 2SM, CLC programming practice
