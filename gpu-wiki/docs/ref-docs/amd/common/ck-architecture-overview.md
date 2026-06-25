# Composable Kernel (CK) Architecture Overview

Composable Kernel (CK) is AMD's official high-performance GPU kernel programming library, implemented in HIP C++. Through two core concepts—the Tile-based programming model and Tensor Coordinate Transformation—it enables portable high-performance kernels across multiple GPU architectures. This document outlines CK's overall architectural layering, the CK-Tile programming model, component classification, and key terminology.

---

## CK Four-Layer Architecture

The overall structure of the CK library is divided into four layers, from bottom to top:

| Layer | Name | Description |
|------|------|------|
| **Layer 1** | Templated Tile Operators | The lowest-level tile operator implementation, including warp-tile and block-tile level operations (e.g., MFMA warp-level matmul, tile load/store). These operators are templated and can produce different execution strategies through different Policy parameter combinations |
| **Layer 2** | Templated Kernel + Invoker | Assembles tile operators into a complete GPU kernel. The kernel template receives components such as Pipeline, Tile Partitioner, and Epilogue. The Invoker is responsible for setting launch parameters and initiating kernel calls |
| **Layer 3** | Instantiated Kernel + Invoker | Instantiates the Layer 2 template with concrete parameters (data type, tile size, layout, etc.) to produce directly compilable kernel instances |
| **Layer 4** | Client API | The top-level user-facing interface, providing a calling style similar to hipBLAS. Users only need to specify problem parameters (shape, dtype), and the instance factory automatically selects the optimal kernel instance |

The core benefit of this layered design is separation of concerns: the bottom layer focuses on efficient execution of a single tile, the middle layer handles kernel assembly and parameterization, and the top layer provides a clean user interface.

## CK-Tile Programming Model

CK-Tile is the next-generation programming model of the CK library, located under the `include/ck_tile/` directory. It is **independent of the legacy CK** (`include/ck/`) and does not require any header files from the legacy CK. CK-Tile reimplements all infrastructure, providing better abstraction and improved composability.

### Core Concepts

CK-Tile is built on two key concepts:

1. **Tensor Coordinate Transformation**: Through transformation primitives such as Merge, Unmerge, Embed, and PassThrough, it describes at compile time how an N-dimensional tensor maps to a one-dimensional memory offset. This is a core concept inherited and improved from the legacy CK. See [TileDistribution and Coordinate Systems](ck-tile-distribution-coordinates.md) for details.

2. **Tile-based Programming Model**: Introduces the Distributed Tensor concept, describing how a group of threads collaboratively processes a tile of data. It includes tile-level APIs (`load_tile`, `store_tile`, `shuffle_tile`, `sweep_tile`, etc.) and compile-time data distribution encoding.

### Relationship with Legacy CK

CK-Tile is the evolution and replacement of legacy CK:

- The legacy CK template system was too complex, resulting in severe instantiation combinatorial explosion
- CK-Tile retains the same mathematical foundation (coordinate transformation) but redesigns the API layer gravity to be more user-friendly
- Currently in a transition period, with legacy CK and CK-Tile coexisting; the ultimate goal is to migrate all operators to CK-Tile

### Kernel Composition in CK-Tile

A typical CK-Tile kernel consists of the following components:

```
Kernel = Pipeline + Tile Partitioner + Epilogue
```

- **Pipeline**: Orchestrates the kernel's execution flow (load -> compute -> store), contains Problem (defines computation content) and Policy (defines data movement strategy)
- **Tile Partitioner**: Defines the mapping from problem dimensions to GPU hierarchy, specifies workgroup-level tile sizes, and calculates grid dimensions
- **Epilogue**: The post-processing stage of the kernel, such as activation functions, bias addition, etc.

---

## Component Classification

CK-Tile's source code is organized into several independent components, each requiring only a single header include to use.

### core (`ck_tile/core`)

All foundational data structures and functions, serving as the basic building blocks for constructing kernels.

```cpp
#include "ck_tile/core.hpp"
```

| Subdirectory | Content |
|--------|------|
| `container/` | `array` (runtime fixed-length array), `tuple` (heterogeneous container), `sequence` (compile-time integer sequence) |
| `numeric/` | GPU data types (`fp16_t`, `bf16_t`, `fp8_t`), type conversion, math functions |
| `algorithm/` | Coordinate transformation system—describes how tensors are constructed via merge/unmerge/embed primitives and how ND coordinates are mapped to 1D memory offsets |
| `tensor/` | Tensor descriptor (TensorDescriptor), distributed tensor (DistributedTensor), tile-level APIs (`load_tile`, `store_tile`, `shuffle_tile`, `slice_tile`, etc.) |
| `arch/` | Device-level foundational constructs, such as MMA instruction wrappers, buffer addressing |
| `utility/` | Host/device common utility functions |

### host (`ck_tile/host`)

Host-side utilities for kernel launching, device memory management, and reference implementations.

```cpp
#include "ck_tile/host.hpp" // orby include singlefile
```

Mainly includes: `kernel_launch.hpp`, `device_memory.hpp`, `stream_config.hpp`, `host_tensor.hpp`, `reference/`, etc.

If you only need to build device libraries with CK-Tile (no host-side executables required), this component can be omitted.

### ops (`ck_tile/ops/`)

Device-side implementations of various operators, organized hierarchically within each operator directory:

| Operator | Header File |
|------|--------|
| GEMM | `ck_tile/ops/gemm.hpp` |
| FMHA (Flash Attention) | `ck_tile/ops/fmha.hpp` |
| Fused MoE | `ck_tile/ops/fused_moe.hpp` |
| LayerNorm / RMSNorm | `ck_tile/ops/layernorm2d.hpp`, `ck_tile/ops/rmsnorm2d.hpp` |
| Softmax | `ck_tile/ops/softmax.hpp` |
| Reduce | `ck_tile/ops/reduce.hpp` |
| Elementwise | `ck_tile/ops/elementwise.hpp` |
| Batched Contraction | `ck_tile/ops/batched_contraction.hpp` |
| Pooling | `ck_tile/ops/pooling.hpp` |
| TopK / TopK+Softmax | `ck_tile/ops/topk.hpp`, `ck_tile/ops/topk_softmax.hpp` |
| FlatMM | `ck_tile/ops/flatmm.hpp` |
| Sparse Attention | `ck_tile/ops/sparse_attn.hpp` |Each operator typically contains:
- **Warp-level**: tile operations within a single wavefront
- **Block-level**: tile operations involving collaboration among multiple warps within a thread block
- **pipeline**: replaceable main loop / epilogue implementation
- **kernel**: template interface for user instantiation

### epilogue (`ck_tile/ops/epilogue`)

The epilogue portion of the kernel, supporting user customization.

### ref (`ck_tile/ref`)

CPU or GPU reference implementation for correctness verification. Include specific header files as needed.

---

## Key Terminology

### Tile-related

| Term | Description |
|------|------|
| **Tile** | A sub-region of a tensor/matrix; the fundamental unit for computation and data movement in CK |
| **Block Tile** | A tile processed by a single work group (thread block) |
| **Wave Tile** | A sub-tile processed by a single wavefront (warp) |
| **TileWindow** | A viewport over a large tensor, defining the current tile's position and boundaries. Created via `make_tile_window()`, supports distributed loading with distribution |
| **TileDistribution** | A compile-time abstraction describing the hierarchical mapping of threads to data. Maps P (thread position) and Y (local data index) to X (global tensor coordinates), then linearized to memory address D |

### Coordinate Spaces

| Space | Meaning |
|------|------|
| **P-space** | Thread Position space (Partition) — identifies a thread's position in the GPU execution hierarchy, such as `[warp_id, lane_id]` |
| **Y-space** | Local data space (Yield) — the data iteration coordinate within each thread |
| **X-space** | Global position space — the actual coordinates in the tensor, such as row/column indices of a matrix |
| **D-space** | Memory address space — the linearized memory offset |
| **R-space** | Replication space — describes the pattern of data shared across multiple threads |

For detailed coordinate system and TileDistribution mechanisms, see [TileDistribution and Coordinate Systems](ck-tile-distribution-coordinates.md).

### Transformation Primitives

| Primitive | Description |
|------|------|
| **MergeTransform** | Merges multiple dimensions into a single linear dimension (e.g., `[4, 5] -> [20]`) |
| **UnmergeTransform** | Splits a single dimension into multiple dimensions (e.g., `[24] -> [3, 4, 2]`) |
| **EmbedTransform** | Maps a linear index to multi-dimensional coordinates using custom strides |
| **PassThroughTransform** | Identity transformation; coordinates remain unchanged |
| **ReplicateTransform** | Broadcast transformation; replicates a scalar into multi-dimensional space |
| **PadTransform** | Adds padding on both ends of a dimension |

### Kernel Structure

| Term | Description |
|------|------|
| **Pipeline** | Orchestrates the kernel's load -> compute -> store flow, containing Problem and Policy |
| **Problem** | Defines the computation — input/output shapes, data types, and mathematical operations |
| **Policy** | Defines data movement strategies — memory access patterns and hardware-specific optimizations |
| **Tile Partitioner** | Maps problem dimensions to GPU hierarchy, specifies tile sizes, and computes grid dimensions |
| **Epilogue** | The kernel's post-processing stage |
| **Descriptor** | Metadata structure that defines tile properties, memory layout, and coordinate transformations |

### Hardware-related

| Term | Description |
|------|------|
| **Compute Unit (CU)** | AMD GPU's parallel processing unit, equivalent to NVIDIA's SM |
| **Wavefront** | AMD's SIMD execution unit (64 threads), equivalent to NVIDIA's Warp (32 threads) |
| **MFMA** | Matrix Fused Multiply-Add, AMD's matrix core instruction |
| **LDS** | Local Data Share, AMD's on-chip shared memory, equivalent to NVIDIA's Shared Memory |
| **VGPR** | Per-thread Vector General Purpose Register |
| **SGPR** | Wave-level Scalar General Purpose Register, shared by all threads |

---

## Typical Usage Pattern

The following is a typical pattern for implementing distributed tile loading using CK-Tile:

```cpp
#include "ck_tile/core.hpp"

// 1. definition(compilation)
using Encoding = ck_tile::tile_distribution_encoding<
 ck_tile::sequence<>, // R: nonedimension
    ck_tile::tuple<
 ck_tile::sequence<4, 2, 8, 4>, // M dimension: [Repeat, WarpPerBlock, ThreadPerWarp, Vector]
 ck_tile::sequence<4, 2, 8, 4>>, // N dimension
    ck_tile::tuple<ck_tile::sequence<1, 2>,
                   ck_tile::sequence<1, 2>>,       // P -> RH major
    ck_tile::tuple<ck_tile::sequence<1, 1>,
                   ck_tile::sequence<2, 2>>,       // P -> RH minor
    ck_tile::sequence<1, 1, 2, 2>,                // Y -> RH major
    ck_tile::sequence<0, 3, 0, 3>                 // Y -> RH minor
>;

// 2. create distribution tensor view
constexpr auto distribution = ck_tile::make_static_tile_distribution(Encoding{});
auto tensor_view = ck_tile::make_naive_tensor_view_packed<
    ck_tile::address_space_enum::global>(ptr, ck_tile::make_tuple(M, N));

// 3. create tile window loaddata
auto window = ck_tile::make_tile_window(
    tensor_view, window_lengths, origin, distribution);
auto tile_data = window.load();

// 4. tile datarowcompute
ck_tile::sweep_tile(tile_data, [&](auto idx) {
    auto val = tile_data(idx);
 // ... compute ...
});
```The encoding `sequence<4, 2, 8, 4>` represents a four-level hierarchical decomposition: 4 repetitions (per thread), 2 warps (per block), 8 threads (per warp participating), 4 elements (vectorized). This hierarchical decomposition maps directly to the GPU hardware organization, ensuring optimal memory access patterns.

---

## Related Documents

- [TileDistribution and Coordinate System](ck-tile-distribution-coordinates.md) -- P/Y/X/D coordinate space, transform composition, encoding internals
- [AMD GPU Kernel Optimization Framework Overview](amd-kernel-optimization-frameworks.md) -- Comparison of CK with other frameworks (FlyDSL, TileLang)
- [MFMA Matrix Core Programming Guide](amd-mfma-matrix-cores.md) -- MFMA instructions used by CK kernels under the hood
- [LDS Bank Conflict Optimization](../../../kernel-opt/amd/common/lds-bank-conflict-optimization.md) -- Techniques such as XOR swizzle in CK ήταν to avoid bank conflicts
