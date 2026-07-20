# CUTLASS 3.x Architecture

CUTLASS 3.x is a major architectural overhaul of CUTLASS 2.x with the core goal of introducing CuTe as a unified Layout algebra system, decoupling the GEMM hierarchy from hardware-coupled designs into algorithm-driven abstraction layers. This document covers the design philosophy, four-level hierarchy, Builder pattern, scheduling strategy system, code organization, and feature matrix of 3.x.

## Changes from 2.x to 3.x

### Problems with 2.x

The hierarchical design of CUTLASS 2.x is tightly coupled to GPU hardware organization (warp-level, threadblock-level, etc.), leading to:

- **Fragile extension**: Hopper's warp-group instructions cannot naturally map to 2.x's warp/thread hierarchy
- **Named type explosion**: a proliferation of homogeneous but differently named types such as `MmaMultistage`, `MmaPlanarComplexMultistage`, `MmaPipelined`
- **Complex iterators**: thread-to-data mapping is implied in the imperative code of 1D iterators, lacking formal algebraic support
- **`default_x_configuration.h` alias hell**: users must mentally substitute type aliases to understand the code

### Design Goals of 3.x

1. **CuTe Layout Algebra**: replace all iterators and layout types with a unified `cute::Layout`
2. **Reduce named types**: replace dozens of named types with `CollectiveMma` + tag-dispatch policy
3. **Correctness first**: statically verify layout consistency at compile time — "if it compiles, it's most likely correct"
4. **Clear performance tuning points**: layout selection is the primary performance knob
5. **Architecture decoupling**: hierarchy is organized around GEMM algorithm structure rather than specific GPU generations

See [CuTe Fundamentals](cutlass-cute-fundamentals.md) for details on Layout algebra.

## Four-Level GEMM Hierarchy

CUTLASS 3.x decomposes GEMM into five conceptual levels (users mainly interact with the first four):

| Level | API Class | Responsibility |
|------|-----------|----------------|
| **Device** | `cutlass::gemm::device::GemmUniversalAdapter` | Host-side handle that manages kernel lifecycle and parameter conversion |
| **Kernel** | `cutlass::gemm::kernel::GemmUniversal` | Grid-level scheduling, tile traversal, warp role assignment |
| **Collective** | `cutlass::gemm::collective::CollectiveMma` + Epilogue | Thread block/cluster-level cooperation, mainloop k-tile iteration |
| **Tiled MMA/Copy** | `cute::TiledMma` / `cute::TiledCopy` | Tiles atoms over threads and data to form composable micro-kernels |
| **Atom** | `cute::Mma_Atom` / `cute::Copy_Atom` | Smallest indivisible hardware instruction unit |

### GEMM Loop Corresponding to Each Level

```cpp
// Kernel (Device launch): cluster tile
for (cluster_m, cluster_n) { // <- Kernel scheduling

 // Collective : k mainloop
  for (k_tile = 0; k_tile < K_tiles; k_tile++) {   // <- CollectiveMma

 // Tiled MMA : unrolled loop
    for (tiled_mma_k, tiled_mma_m, tiled_mma_n) {  // <- cute::gemm()
      mma.call(d, a, b, c);                        // <- Atom
    }
  }
}
```

### Device Level

`GemmUniversalAdapter` is a stateful, reusable host-side handle:

```cpp
template <class GemmKernel_, class Enable = void>
class GemmUniversalAdapter;
```

- **Stateful**: manages the lifecycle of kernel `Params`, converting the user's `Arguments` to the kernel's `Params`
- **Reusable**: the same handle can be invoked multiple times with different matrices
- **Architecture-independent**: automatically distinguishes between 2.x/3.x kernels via the `IsCutlass3GemmKernel` metafunction
- Does not control grid shape; grid shape is determined by the kernel level

### Kernel Level

`GemmUniversal` is a stateless universal kernel entry point:

```cpp
template <
  class ProblemShapeOrThreadblockMma_,  // cute::Shape<int,int,int,int> for 3.x
  class CollectiveMainloop_,
  class CollectiveEpilogue_,
  class TileScheduler_ = void,
  class Enable = void
>
class GemmUniversal;
```

Four major responsibilities:
1. **Tile scheduling**: assigns output tiles to thread blocks via `TileScheduler`
2. **Warp role assignment**: in warp-specialized mode, splits warps into producers and consumers
3. **Grid swizzle**: performs L2-cache-friendly tile traversal order
4. **Tensor slicing**: slices input tensors by cluster tile and passes them to the collective

In 3.x, `ProblemShape` is elevated to a top-level template parameter (enabling fully static optimization with compile-time-known shapes).

### Collective Level

A "Collective" is the largest set of threads that can leverage hardware-accelerated communication and synchronization:

```cpp
template <
  class DispatchPolicy, class TileShape,
  class ElementA, class StrideA,
  class ElementB, class StrideB,
  class TiledMma,
  class GmemTiledCopyA, class SmemLayoutAtomA, class SmemCopyAtomA, class TransformA,
  class GmemTiledCopyB, class SmemLayoutAtomB, class SmemCopyAtomB, class TransformB
>
struct CollectiveMma;
```- Implements the k-tile mainloop (loads tiles from global memory into shared memory, executes MMA)
- Acts as the composition point for mainloop fusion and epilogue fusion
- Selects the concrete implementation via tag dispatch under `DispatchPolicy` (rather than naming a new type)

### Atom and Tiled Layers

- **Atom**: the smallest-granularity wrapper for hardware instructions, describing thread and data dimensions using CuTe Layouts
- **Tiled**: tiles an Atom across more threads and larger data blocks, forming thread-block-level operations
- `cute::gemm()` and `cute::copy()` drive the inner loops at the tiled level

## CollectiveBuilder Pattern

`CollectiveBuilder` is a simplified interface for non-expert users that automatically constructs the optimal `CollectiveMma` using 2.x-style parameters:

```cpp
template <
  class ArchTag,            // arch::Sm90, arch::Sm100, arch::Sm120
  class OpClass,            // arch::OpClassTensorOp
  class ElementA, class GmemLayoutA, int AlignmentA,
  class ElementB, class GmemLayoutB, int AlignmentB,
  class ElementAccumulator,
  class TileShape_MNK,      // cute::Shape<_128, _256, _64>
  class ClusterShape_MNK,   // cute::Shape<_2, _1, _1>
  class StageCountType,     // StageCountAuto or StageCount<N>
  class KernelScheduleType, // KernelScheduleAuto or specific tag
  class Enable = void
>
struct CollectiveBuilder;
```

Internal builder logic:

1. Selects the architecture-specific builder specialization (`sm90_gmma_builder.inl`, `sm100_umma_builder.inl`, etc.) based on `ArchTag`
2. Automatically computes the maximum pipeline depth based on tile size and shared memory capacity during `StageCountAuto`
3. Selects the optimal schedule tag based on data type and architecture during `KernelScheduleAuto`
4. Outputs the type alias `CollectiveOp` for use by the kernel layer

**Builder File Organization** (`include/cutlass/gemm/collective/builders/`):

| File | Architecture | Description |
|------|------|------|
| `sm90_gmma_builder.inl` | SM90 | Hopper GMMA + TMA dense GEMM |
| `sm90_sparse_gmma_builder.inl` | SM90 | Hopper sparse GEMM |
| `sm100_umma_builder.inl` | SM100 | Blackwell UMMA dense GEMM |
| `sm100_blockscaled_umma_builder.inl` | SM100 | Blackwell block-scaled MMA |
| `sm100_sparse_umma_builder.inl` | SM100 | Blackwell sparse GEMM |
| `sm100_9xBF16_umma_builder.inl` | SM100 | Blackwell FastFP32 (9xBF16) |
| `sm100_mixed_input_umma_builder.inl` | SM100 | Blackwell mixed-precision input |
| `sm120_mma_builder.inl` | SM120 | GeForce (Blackwell consumer) |
| `sm120_blockscaled_mma_builder.inl` | SM120 | SM120 block-scaled MMA |
| `sm103_blockscaled_umma_builder.inl` | SM103 | NVF4 Ultra block-scaled |

## Dispatch Policy System

CUTLASS 3.x uses tag-based dispatch in place of named types. Each mainloop dispatch policy contains the `Schedule` type alias, corresponding to a kernel-layer implementation.

### Kernel Schedule Tags

#### SM90 (Hopper) Schedule Tags

| Tag | Description |
|-----|------|
| `KernelMultistage` | Classic multi-stage pipeline (non-warp-specialized) |
| `KernelTma` | TMA + GMMA, static schedule |
| `KernelTmaWarpSpecialized` | TMA + GMMA, warp-specialized dynamic schedule |
| `KernelTmaWarpSpecializedPingpong` | Dual-buffer pingpong, minimizing stalls |
| `KernelTmaWarpSpecializedCooperative` | Multi-warp-group cooperative |
| `KernelCpAsyncWarpSpecialized` | cp.async + GMMA warp-specialized |
| `KernelCpAsyncWarpSpecializedPingpong` | cp.async pingpong variant |
| `KernelCpAsyncWarpSpecializedCooperative` | cp.async cooperative variant |
| `KernelPtrArrayTmaWarpSpecializedCooperative` | Grouped/Ptr-Array GEMM cooperative |
| `KernelPtrArrayTmaWarpSpecializedPingpong` | Grouped/Ptr-Array GEMM pingpong |
| `KernelTmaWarpSpecializedCooperativeFP8Blockwise` | FP8 block-scaled cooperative |
| `KernelTmaWarpSpecializedPingpongFP8Blockwise` | FP8 block-scaled pingpong |
| `KernelTmaWarpSpecializedMixedInput` | Mixed-precision input |

#### SM100 (Blackwell) Schedule Tags

SM100 introduces 1SM/2SM MMA atom differentiation and CLC (Cluster Launch Control) scheduling:

| Tag | Description |
|-----|-------------|
| `KernelTmaWarpSpecialized1SmSm100` | 1SM dense GEMM |
| `KernelTmaWarpSpecialized2SmSm100` | 2SM dense GEMM (2CTA collaboration) |
| `KernelWarpSpecialized1SmSm100` | 1SM without TMA (cp.async) |
| `KernelTmaWarpSpecializedBlockwise1SmSm100` | 1SM blockwise scaling |
| `KernelTmaWarpSpecializedBlockwise2SmSm100` | 2SM blockwise scaling |
| `KernelTmaWarpSpecialized1SmBlockScaledSm100` | 1SM block-scaled MMA |
| `KernelTmaWarpSpecialized2SmBlockScaledSm100` | 2SM block-scaled MMA |
| `KernelSparseTmaWarpSpecialized1SmSm100` | 1SM sparse GEMM |
| `KernelSparseTmaWarpSpecialized2SmSm100` | 2SM sparse GEMM |
| `KernelTmaWarpSpecialized1SmFastFP32Sm100` | 1SM FastFP32 (9xBF16) |
| `KernelTmaWarpSpecialized2SmFastFP32Sm100` | 2SM FastFP32 (9xBF16) |
| `KernelPtrArrayTmaWarpSpecialized1SmSm100` | 1SM Ptr-Array dense |
| `KernelPtrArrayTmaWarpSpecialized2SmSm100` | 2SM Ptr-Array dense |

SM100 Schedule Tags have two template parameters, `SchedulerPipelineStageCount` and `AccumulatorPipelineStageCount`, which control the scheduler pipeline depth and accumulator pipeline depth respectively.

#### SM120 (Blackwell GeForce) Schedule Tags

SM120 inherits SM90's Cooperative/Pingpong modes and adds a scheduler pipeline parameter:

| Tag | Description |
|-----|-------------|
| `KernelTmaWarpSpecializedCooperativeSm120<N>` | Dense cooperative |
| `KernelTmaWarpSpecializedPingpongSm120<N>` | Dense pingpong |
| `KernelTmaWarpSpecializedNvf4Sm120` | NVF4 block-scaled |
| `KernelTmaWarpSpecializedMxf8f6f4Sm120` | MX FP8/FP6/FP4 block-scaled |
| `KernelSparseTmaWarpSpecializedNvf4Sm120` | Sparse NVF4 |
| `KernelTmaWarpSpecializedBlockwiseCooperativeSm120` | Blockwise scaling cooperative |

### Mainloop Dispatch Policies

Mainloop policy packages the schedule tag, pipeline depth, and cluster shape into a single type:

```cpp
// SM90 TMA + GMMA warp-specialized
template<int Stages_, class ClusterShape_, class KernelSchedule>
struct MainloopSm90TmaGmmaWarpSpecialized {
  constexpr static int Stages = Stages_;
  using ClusterShape = ClusterShape_;
  using ArchTag = arch::Sm90;
 using Schedule = KernelSchedule; // kernel
};

// SM100 TMA + UMMA warp-specialized
template<int Stages_, int SchedulerPipelineStageCount_,
         int AccumulatorPipelineStageCount_, class ClusterShape_>
struct MainloopSm100TmaUmmaWarpSpecialized {
  constexpr static int Stages = Stages_;
  using ClusterShape = ClusterShape_;
  using ArchTag = arch::Sm100;
  using Schedule = KernelTmaWarpSpecializedSm100<
      SchedulerPipelineStageCount_, AccumulatorPipelineStageCount_>;
};
```

**Key design**: A single mainloop can compose multiple kernel schedules. For example, `MainloopSm90TmaGmmaWarpSpecialized` can be paired with three schedules: `KernelTmaWarpSpecialized`, `Pingpong`, and `Cooperative`.

## Code Organization

```
include/
  cutlass/
 arch/ # direct(instruction-level GEMM)
    gemm/
 thread/ #
 warp/ # Warp
 collective/ # 3.x Collective
 collective_mma.hpp # CollectiveMma
 collective_builder.hpp # CollectiveBuilder
 builders/ # builder (.inl file)
 threadblock/ # 2.x CTA
 kernel/ # Kernel entry + TileScheduler
        gemm_universal.hpp   #   3.x GemmUniversal
 sm90_gemm_*.hpp # SM90 schedule kernel
 sm100_gemm_*.hpp # SM100 schedule kernel
        sm90_tile_scheduler*.hpp  # SM90 tile scheduler
        sm100_tile_scheduler*.hpp # SM100 tile scheduler
 tile_scheduler.hpp # TileSchedulerSelector
      device/                # Host-side launch
        gemm_universal_adapter.h  # GemmUniversalAdapter
 dispatch_policy.hpp # schedule tag + mainloop policy definition
 layout/ # memorylayoutdefinition
    epilogue/
 collective/ # Collective epilogue

 cute/ # CuTe core(independent CUTLASS)
 algorithm/ # gemm, copy core
 arch/ # PTX
 atom/ # Mma_Atom, Copy_Atom +
      mma_atom.hpp           # cute::Mma_Atom + TiledMma
      copy_atom.hpp          # cute::Copy_Atom + TiledCopy
 *.hpp # Shape, Stride, Layout, Tensor coretype
```### Key Design Patterns

**Tag-based dispatch**: Use empty structs as compile-time tags, selecting implementations through template specialization:

```cpp
// dispatch_policy.hpp definition tag
struct KernelTmaWarpSpecializedCooperative { };

// sm90_gemm_tma_warpspecialized_cooperative.hpp passed enable_if
template <class ProblemShape_, class CollectiveMainloop_, ...>
class GemmUniversal<ProblemShape_, CollectiveMainloop_, ...,
    std::enable_if_t<std::is_base_of_v<KernelTmaWarpSpecializedCooperative,
        typename CollectiveMainloop_::DispatchPolicy::Schedule>>>
```

**CRTP (Curiously Recurring Template Pattern)**: The tile scheduler uses CRTP to implement static polymorphism:

```cpp
template<class Subclass>
class StaticPersistentTileScheduler { ... };

class PersistentTileSchedulerSm90 :
    public StaticPersistentTileScheduler<PersistentTileSchedulerSm90> { ... };
```

**Compile-time configuration**: All tile shapes, cluster shapes, and stage counts are determined at compile time, leveraging `cute::Int<N>` constant types Mug\[support compile-time algebraic operations.\]

## Feature Matrix

### CUTLASS 3.x Device-level GEMM

| Data Type | Compute Capability | CUDA Toolkit | Layout |
|---------|-------------------|--------------|------|
| `f16 * f16 + {f16,f32} => {f16,f32}` | SM90a | 12.0+ | {N,T} x {N,T} |
| `bf16 * bf16 + {f16,f32} => {bf16,f32}` | SM90a | 12.0+ | {N,T} x {N,T} |
| `{f32,tf32} * {f32,tf32} + f32 => f32` | SM90a | 12.0+ | T x N |
| `s8 * s8 + s32 => {s32,s8}` | SM90a | 12.0+ | T x N |
| `f64 * f64 + f64 => f64` | SM90+ | 11.8+ | {N,T} x {N,T} |

### CUTLASS 2.x Device-level GEMM (Still Available)

| Opcode Class | Data Type | Compute Capability |
|-------------|---------|-------------------|
| **Simt** | f32, f64, f16, s8 | SM50-SM80+ |
| **WmmaTensorOp** | f16, s8, s4, b1 | SM70-SM75+ |
| **TensorOp** | f16, bf16, tf32, s8, s4, b1, f64, cf32, cf64 | SM70-SM80+ |
| **SpTensorOp** | f16, bf16, tf32, s8, s4 | SM80+ |

### SM100/SM120 New Capabilities (Supported via Builder)

| Feature | SM100 (Blackwell) | SM120 (GeForce Blackwell) |
|------|-------------------|--------------------------|
| Dense GEMM | 1SM / 2SM | Cooperative / Pingpong |
| Block-Scaled MMA | MXF8/F6/F4, NVF4 | MXF8/F6/F4, NVF4 |
| Sparse GEMM | 1SM / 2SM | F8/F6/F4 |
| FastFP32 (9xBF16) | 1SM / 2SM | -- |
| Planar Complex | 1SM / 2SM | -- |
| Interleaved Complex TF32 | 1SM / 2SM | -- |
| Mixed-Precision Input | 1SM / 2SM | -- |
| Ptr-Array / Grouped | All Supported | All Supported |
| CLC Dynamic Dispatch | Yes | Yes |
| Blockwise Scaling | Yes | Yes |

## Assembling a 3.x GEMM Kernel

Complete four-step assembly process:

```cpp
// Step 1: passed CollectiveBuilder mainloop
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    arch::Sm90, arch::OpClassTensorOp,
    half_t, layout::RowMajor, 8,        // A: half, RowMajor, 128-bit aligned
    half_t, layout::ColumnMajor, 8,     // B: half, ColMajor, 128-bit aligned
    float,                              // accumulator
    Shape<_128, _256, _64>,             // tile shape MNK
    Shape<_2, _1, _1>,                  // cluster shape
 StageCountAuto, // automaticcompute pipeline
 KernelScheduleAuto // automatic schedule
>::CollectiveOp;

// Step 2: epilogue
using CollectiveEpilogue = cutlass::epilogue::collective::DefaultEpilogue<
    half_t, /* stride_C, stride_D, thread_epilogue_op */ ...>;

// Step 3: kernel
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>,  // ProblemShape [M,N,K,L]
    CollectiveMainloop,
    CollectiveEpilogue
>;

// Step 4: host handle
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
```## Related Documents

- [CuTe Fundamentals](cutlass-cute-fundamentals.md) -- Layout Algebra, Tensor Abstraction
- [CUTLASS GEMM Optimization Strategies](cutlass-gemm-optimization.md) -- tiling strategies
- [CuTeDSL Software Pipeline](cutedsl-pipeline-patterns.md) -- producer/consumer state machine
- [CUTLASS Tile Scheduling](cutlass-tile-scheduling.md) -- persistent scheduling, Stream-K, Grouped GEMM
