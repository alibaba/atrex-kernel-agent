# CK-Tile TileDistribution and Coordinate System

The core innovation of CK-Tile lies in its compile-time coordinate transformation system, which automatically generates optimal memory access patterns. This article provides a detailed introduction to the five coordinate spaces (P/Y/X/D/R), coordinate transformation primitives, the TileDistribution compile-time abstraction, the thread-to-data mapping mechanism, and the internal implementation of the encoding.


**Last updated**: 2026-06-30

---

## The Five Coordinate Spaces

CK-Tile uses five interrelated coordinate spaces, each representing an abstraction level in the process from thread identification to memory access. The core transformation chain is:

```
Thread ID -> P-space -> (P + Y) -> X-space -> D-space (Memory Address)
```

### Coordinate Space Reference Table

| Space | Full Name | Meaning | Example | Dimension Source |
|------|------|------|------|----------|
| **P** | Partition | The thread's position within the GPU execution hierarchy | `[warp_id=1, lane_id=5]` | Hardware thread ID |
| **Y** | Yield | Local data iteration coordinates within each thread | `[y0=1, y1=0, y2=0, y3=1]` | Hierarchical decomposition defined by encoding |
| **X** | Physical tensor | Actual coordinates in the global tensor | `[row=3, col=7]` | P+Y transformation result |
| **R** | Replication | Pattern of data sharing/replication across multiple threads | `[r0=2, r1=4]` | rs_lengths in encoding |
| **D** | Data/linear | Linearized memory address offset | `offset=31` | X computed via stride |

### P-space: Thread Position Identification

P-space directly reflects the GPU's hardware execution hierarchy. For CK-Tile:

- **1D P-space**: `P = [lane_id]`, using only lane ID
- **2D P-space**: `P = [warp_id, lane_id]`, using warp ID + lane ID

```cpp
// CK-Tile internal implementation
static auto _get_partition_index()
{
    if constexpr(NDimP == 1)
        return array<index_t, 1>{get_lane_id()};
    else if constexpr(NDimP == 2)
        return array<index_t, 2>{get_warp_id(), get_lane_id()};
}
```

P-space coordinates determine:
- Work distribution — which portion of data this thread processes
- Memory coalescing — adjacent threads accessing adjacent memory
- Thread collaboration — usage patterns of shared memory

### Y-space: Local Data Organization

Y-space describes how each thread traverses data elements within its assigned tile. All threads share the same Y-space pattern but correspond to different X-space positions.

Y-space can be hierarchical. A typical 4-level structure is:

```
Y = [Repeat, WarpDim, ThreadDim, VectorDim]
```

Each level corresponds to a different hardware level:
- **Repeat**: Algorithm-level repetition (e.g., multiple attention heads)
- **WarpDim**: Inter-warp collaboration dimension
- **ThreadDim**: Intra-warp thread dimension
- **VectorDim**: SIMD vectorization dimension

The traversal of Y-space can be unrolled at compile time using the `sweep_tile` API:

```cpp
sweep_tile(distributed_tensor, [&](auto y_coord) {
    // y_coord is compile-time multi_index
    // All iterations are fully unrolled at compile time
    auto value = distributed_tensor(y_coord);
    // ... processing ...
});
```

### X-space: Global Tensor Coordinates

X-space represents the tensor coordinates from the user's perspective (e.g., row and column indices), and is the output of the P+Y transformation. Each thread's P coordinate combined with its Y-space traversal coordinates determines the global data position accessed by that thread.

### R-space: Replication Dimension

R-space is used to express patterns where multiple threads share the same data, commonly seen in:
- Multiple warps sharing the same A or B matrix data in GEMM
- Multiple threads collaborating to compute a single result in reduction operations

```cpp
// GEMM: NWarpPerBlock warps share A data, MWarpPerBlock warps share B data
// No replication```

### D-space: Memory Linearization

D-space is the final linear memory address, transformed from X-space through stride computation:

```
D = x0 * stride_0 + x1 * stride_1 + ... + xN * stride_N
```

Multiple layout strategies are supported: Row-major, Column-major, Blocked layouts, etc.

---

## Core Transformation: P + Y -> X

This is the core of TileDistribution, combining the thread identifier (P) with local data coordinates (Y) to determine the global tensor position (X).

Mathematical expression:

```
X = f(P, Y) = BasePosition(P) + LocalOffset(Y)
```

Where:
- `BasePosition(P)` determines the starting position in the tile based on thread position
- `LocalOffset(Y)` determines the offset within the tile based on local iteration coordinates

This transformation is highly configurable through distribution encoding rift. Different distribution strategies can be defined for different algorithms while keeping the mathematical framework unchanged.

---

## Transform Primitives

CK-Tile's transform primitives operate on logical coordinate spaces and **do not involve any data copying**. Each transform defines a bidirectional mapping between the Lower Dimension Space (source) and the Upper Dimension Space (target).

### Transform Type Overview

| Transform | Direction | Description | Application Scenarios |
|------|------|------|----------|
| **MergeTransform** | Multi-D -> 1D | Merges multiple dimensions into a linear index | Thread block mapping, address linearization |
| **UnmergeTransform** | 1D -> Multi-D | Splits a linear index into multi-dimensional coordinates | Hierarchical decomposition, tile substructures |
| **EmbedTransform** | 1D -> Multi-D (strided) | Splitting using custom strides | Non-contiguous layouts, sub-tensor views |
| **PassThroughTransform** | 1D -> 1D | Identity mapping | Dimensions that do not need modification in a transform chain |
| **ReplicateTransform** | 0D -> Multi-D | Broadcast scalar to multi-dimensional space | Broadcast operations |
| **PadTransform** | 1D -> 1D (padded) | Adds padding | Convolution padding, alignment |
| **OffsetTransform** | 1D -> 1D (shifted) | Coordinate translation | Tile window sliding |
| **XorTransform** | 2D -> 2D | XOR coordinate mapping | Avoiding LDS bank conflicts |
| **SliceTransform** | 1D -> 1D (sub) | Extracts a sub-range | Tensor slicing |
| **ModuloTransform** | 1D -> 1D (cyclic) | Modulo cyclic mapping | Circular buffers |### Detailed Explanation of Key Transformations

#### MergeTransform: Multi-Dimensional Merging

Merges multi-dimensional coordinates into a single linear index using row-major order:

```cpp
// [4, 5] -> [20]
auto transform = make_merge_transform(make_tuple(4, 5));

// Forward: (2, 3) -> 2*5 + 3 = 13
// Inverse: 13 -> (13/5, 13%5) = (2, 3)
```

#### UnmergeTransform: Linear Splitting

Splits a linear index into multi-dimensional coordinates. It is the inverse operation of MergeTransform:

```cpp
// [24] -> [3, 4, 2]
auto transform = make_unmerge_transform(make_tuple(3, 4, 2));

// Forward: 14 -> (14/8, (14%8)/2, (14%8)%2) = (1, 3, 0)
// where 8 = 4*2
```

This transform is at the core of the hierarchical decomposition in TileDistribution encoding—splitting a dimension into `[Repeat, WarpPerBlock, ThreadPerWarp, VectorSize]` is an UnmergeTransform.

#### EmbedTransform: Custom Stride Mapping

Similar to UnmergeTransform but uses custom strides, supporting non-contiguous layouts:

```cpp
// Use stride [12, 1] to map linear index to 2D
auto transform = make_embed_transform(make_tuple(2, 3), make_tuple(12, 1));
// 14 -> (14/12, 14%12) = (1, 2)
```

#### Transform Composition

Transforms can be chained together (via tensor adaptors) to build arbitrarily complex coordinate mappings with zero runtime overhead:

```cpp
// Transform [2, 6] tensor to [2, 2, 3] view
auto transformed = transform_tensor_descriptor(
    base_descriptor,
    make_tuple(
        make_pass_through_transform(2),           // Dimension 0 unchanged
        make_unmerge_transform(make_tuple(2, 3))  // Split dimension 1
    ),
    make_tuple(sequence<0>{}, sequence<1>{}),      // Input mapping
    make_tuple(sequence<0>{}, sequence<1, 2>{})    // Output mapping
);
```

---

## TileDistribution Compile-Time Abstraction

TileDistribution is the core API of CK-Tile, assembling the above coordinate space and transformation primitives into a complete compile-time data distribution scheme.

### Template Structure

```cpp
template <typename PsYs2XsAdaptor_,           // P+Y -> X coordinate transform adapter
          typename Ys2DDescriptor_,             // Y -> D linearization descriptor
          typename StaticTileDistributionEncoding_,  // Encoding specification
          typename TileDistributionDetail_>     // Implementation details
struct tile_distribution
{
    // Get current thread's P coordinate
    static auto _get_partition_index();

 // P compute Y-space
    template <typename PartitionIndex>
    static auto calculate_tile_Ys_index(const PartitionIndex& ps_idx);
};
```

Three core components:

| Component | Responsibility |
|------|------|
| `PsYs2XsAdaptor` | Performs P+Y -> X coordinate transform, implementing the mapping from threads to global tensor positions |
| `Ys2DDescriptor` | Handles Y -> D linearization, converting multi-dimensional tile patterns into register allocation schemes |
| `StaticTileDistributionEncoding` | Captures the hierarchical work decomposition scheme—how work is distributed across blocks, warps, and threads |

### Distribution Encoding

Encoding is the core configuration of TileDistribution, fully defining the data distribution scheme through 6 template parameters:

```cpp
template <typename RsLengths_,      // R dimension lengths (replication)
          typename HsLengthss_,     // H dimension hierarchical decomposition
          typename Ps2RHssMajor_,   // P -> RH mapping (major index)
          typename Ps2RHssMinor_,   // P -> RH mapping (minor index)
          typename Ys2RHsMajor_,    // Y -> RH mapping (major index)
          typename Ys2RHsMinor_>    // Y -> RH mapping (minor index)
struct tile_distribution_encoding;
```

#### Parameter Details

**RsLengths -- Replication Dimensions**

Defines the dimensions replicated across multiple processing units:

```cpp
// GEMM: NWarpPerBlock warp shared A data, MWarpPerBlock warp shared B data
using RsLengths = sequence<NWarpPerBlock, MWarpPerBlock>;

// none
using RsLengths = sequence<>;
```

**HsLengthss -- Hierarchical Decomposition**

Performs hierarchical decomposition for each X dimension, which is key to performance optimization:

```cpp
using HsLengthss = tuple<
    sequence<MRepeat, MWarp, MThread, MVec>,  // M dimension: repeat*warp*thread*vector
    sequence<NRepeat, NWarp, NThread, NVec>   // N dimension: same as above
>;

// For example sequence<4, 2, 8, 4>:
// - 4: Repeat 4 times per thread
// - 2: 2 warps participate
// - 8: 8 threads per warp
// - 4: Vectorize 4 elements per operation
// Total: 4 * 2 * 8 * 4 = 256 elements
```**P -> RH Mapping**

Defines how thread IDs map to positions in the hierarchical decomposition:

```cpp
// P0=warp_id maps to component 1 of H dimension group 1
// P1=lane_id maps to component 2 of H dimension group 2
using Ps2RHssMajor = tuple<sequence<1, 2>, sequence<1, 2>>;
using Ps2RHssMinor = tuple<sequence<1, 1>, sequence<2, 2>>;
```

- Major index: which RH dimension group (0=R, 1-N=H)
- Minor index: which component within the group

**Y -> RH Mapping**

Defines how Y-space coordinates map to the hierarchical decomposition:

```cpp
using Ys2RHsMajor = sequence<1, 1, 2, 2>;  // Y0,Y1 map to H1; Y2,Y3 map to H2
using Ys2RHsMinor = sequence<0, 3, 0, 3>;  // Map to components 0,3,0,3 respectively
```

### Encoded Transformation Pipeline

The encoding generates a transformation pipeline at compile time:

```
P + Y coordinates  -->  Replicate  -->  Unmerge  -->  Merge  -->  X coordinates
(thread+local)       (handle replication) (hierarchical) (merge to X dims)  (global position)
```

Corresponding code structure:

```cpp
template <typename Encoding>
auto make_ps_ys_to_xs_adaptor(const Encoding& encoding)
{
    // 1. Create each transform
    constexpr auto replicate_transform = make_replicate_transform(
        encoding.get_rs_lengths());
    constexpr auto unmerge_transform = make_unmerge_transform(
        encoding.get_hs_lengthss());
    constexpr auto merge_transform = make_merge_transform(
        encoding.get_rhs_to_xs_mapping());

    // 2. Chain composition
    constexpr auto chain = chain_transforms(
        replicate_transform, unmerge_transform, merge_transform);

    // 3. Create adapter
    return make_tile_adaptor(chain, encoding.get_lower_dimension_hidden_idss());
}
```

---

## Y -> D Linearization

The Y -> D descriptor is responsible for mapping each thread's multi-dimensional Y-space coordinates to linear offsets in the thread-local registers:

```cpp
template <typename YLengths, typename YStrides>
struct ys_to_d_descriptor
{
    // Calculate register offset from Y coordinates
    template <typename YIndex>
    constexpr index_t calculate_offset(const YIndex& idx_y) const
    {
        index_t offset = 0;
        static_for<0, num_of_dimension, 1>{}([&](auto i) {
            offset += idx_y[i] * YStrides{}[i];
        });
        return offset;
    }

    // Total number of elements per thread
    static constexpr index_t get_element_space_size()
    {
        return reduce_on_sequence(YLengths{}, multiplies{}, number<1>{});
    }
};
```

This descriptor is used in `static_distributed_tensor` to manage the thread-local register buffer:

```cpp
template <typename TileDistribution>
struct static_distributed_tensor
{
    using ys_to_d_descriptor = typename TileDistribution::ys_to_d_descriptor;

    static constexpr index_t thread_buffer_size =
        ys_to_d_descriptor::get_element_space_size();

    DataType thread_buffer_[thread_buffer_size];  // Thread-local storage

    template <typename YIndex>
    DataType& at(const YIndex& idx_y)
    {
        const index_t offset = ys_to_d_descriptor{}.calculate_offset(idx_y);
        return thread_buffer_[offset];
    }
};
```

### GEMM Optimized Layout

For GEMM kernels, the Y -> D layout can be optimized into a vectorization-friendly form:

```cpp
// Layout: [M/VectorSize][N][VectorSize]
// Ensure vectorized load data is contiguous in memory
template <index_t M, index_t N, index_t VectorSize>
using GemmYsToD = tile_descriptor<
    sequence<M/VectorSize, N, VectorSize>,
    sequence<N * VectorSize, VectorSize, 1>>;
```

---

## Tensor Descriptor: Complete Tensor Specification

TensorDescriptor is the complete blueprint for a tensor, encapsulating shape, stride, and the transformation pipeline into a single object.

### Creating Basic Descriptors

```cpp
// Custom stride (with padding)
auto desc = make_naive_tensor_descriptor(
    make_tuple(3, 4),    // shape: [3, 4]
    make_tuple(8, 1)     // strides: [8, 1], 8 elements per row (4 data + 4 padding)
);

// Compact row-major
auto desc_packed = make_naive_tensor_descriptor_packed(make_tuple(3, 4));
// Strides automatically [4, 1]

// Aligned layout
auto desc_aligned = make_naive_tensor_descriptor_aligned(
    make_tuple(4, 5), 8);  // Align each row to 8 element boundary
// Stride becomes [8, 1], total size 4*8=32
```### Transformation Pipeline

Descriptors add transformation layers via `transform_tensor_descriptor`, constructing arbitrarily complex logical views **without moving data**:

```cpp
// Matrix transpose view
auto transposed = transform_tensor_descriptor(
    original,
    make_tuple(make_pass_through_transform(N), make_pass_through_transform(M)),
    make_tuple(sequence<1>{}, sequence<0>{}),  // Swap input dimensions
    make_tuple(sequence<0>{}, sequence<1>{})
);

// 5D -> 3D merge (GPU thread block work mapping)
auto merged = transform_tensor_descriptor(
    base_5d,
    make_tuple(
        make_pass_through_transform(NumIssues),
        make_merge_transform(make_tuple(wavesPerM, wavesPerK)),
        make_merge_transform(make_tuple(WarpSize, KVector))
    ),
    make_tuple(sequence<0>{}, sequence<1, 2>{}, sequence<3, 4>{}),
    make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{})
);
```

---

## Complete Example: GEMM Distribution Encoding

Below is a complete GEMM kernel TileDistribution example, demonstrating how all coordinate spaces work together:

```cpp
// Define GEMM distribution encoding
using GemmEncoding = tile_distribution_encoding<
    sequence<>,                              // R: No replication
    tuple<sequence<4, 2, 8, 4>,             // M dimension: [MRepeat=4, MWarp=2, MThread=8, MVec=4]
          sequence<4, 2, 8, 4>>,            // N dimension: [NRepeat=4, NWarp=2, NThread=8, NVec=4]
    tuple<sequence<1, 2>, sequence<1, 2>>,  // P -> RH major
    tuple<sequence<1, 1>, sequence<2, 2>>,  // P -> RH minor
    sequence<1, 1, 2, 2>,                   // Y -> RH major
    sequence<0, 3, 0, 3>                    // Y -> RH minor
>;

// Create distribution
constexpr auto distribution = make_static_tile_distribution(GemmEncoding{});

// Use in kernel
__global__ void gemm_kernel(...)
{
    // Step 1: Get P coordinates (thread identifier)
    const auto p_coord = distribution.calculate_p_coord();
    // For thread 37: warp_id=1, lane_id=5 -> P=[1, 5]

    // Step 2: Create tile window
    auto c_window = make_tile_window(
        c_view,
        make_tuple(number<256>{}, number<256>{}),  // Tile size
        {blockIdx.x * 256, blockIdx.y * 256},      // origin
        distribution);

    // Step 3: Load data to registers (distributed tensor)
    auto c_tile = c_window.load();

    // Step 4: Traverse Y-space for computation
    sweep_tile(c_tile, [&](auto y_coord) {
        // Compile-time: P+Y -> X transform
        // Runtime: Directly access corresponding memory address
        c_tile(y_coord) = compute_element(...);
    });

    // Step 5: Write back
    c_window.store(c_tile);
}
```

The meaning of the H dimension `sequence<4, 2, 8, 4>`:
- **4 (Repeat)**: Each thread processes 4 tiles, written to the y0/y1 dimensions of Y-space
- **2 (WarpPerBlock)**: 2 warps within the block share the work, indexed by P's warp_id
- **8 (ThreadPerWarp)**: 8 threads within each warp participate, indexed by P's lane_id
- **4 (Vector)**: Process 4 elements per vectorized operation, written to the y2/y3 dimensions of Y-space

In total, each dimension processes 4 * 2 * 8 * 4 = 256 elements, corresponding to a 256x256 output tile.

---

## Performance Impact

The coordinate system framework directly affects the following key performance metrics:

| Optimization Goal | Mechanism |
|----------|------|
| **Memory Coalescing** | The P+Y->X transformation ensures adjacent threads access adjacent memory, achieving maximum memory bandwidth |
| **Cache Efficiency** | Y-space traversal order can be designed for maximum cache reuse (data within a tile stays hot in L1/L2) |
| **Register Optimization** | Y->D linearization minimizes register usage, avoiding register spills |
| **Vectorization** | Coordinate transformations naturally align with vector operations, supporting 4/8/16-element vector loads/stores |
| **LDS Bank Conflict** | Avoid shared memory bank conflicts through transformations such as XorTransform |
| **Zero Runtime Overhead** | All transformations are resolved at compile time, and the generated machine code is comparable to hand-optimized code |

---

## Related

- [CK Architecture Overview](ck-architecture-overview.md) -- CK four-layer architecture, CK-Tile programming model, component classification
- [LDS Bank Conflict Optimization](lds-bank-conflict-optimization.md) -- Application of XorTransform to avoid bank conflicts in CK
- [AMD MFMA Matrix Core Programming Guide](amd-mfma-matrix-cores.md) -- Using MFMA instructions in CK kernels
- GEMM Tuning -- Tile size and distribution strategy tuning for GEMM kernels
