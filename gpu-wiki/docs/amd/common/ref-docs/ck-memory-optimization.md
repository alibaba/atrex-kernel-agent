# Composable Kernel Memory Optimization System

CK (Composable Kernel) memory subsystem design — from the BufferView raw memory abstraction to the TileWindow distributed access window, covering XOR preshuffle for eliminating LDS bank conflicts, Morton ordering spatial locality optimization, Space-Filling Curve cache-friendly traversal, and LDS double buffering / Async DMA prefetch pipelines.

> **Applicable Architectures**: CDNA3 (gfx942 / MI300X), CDNA4 (gfx950 / MI355X)
> **Source Version**: ck_tile framework (composable_kernel/include/ck_tile/)

---

## 1. Memory Hierarchy and Data Flow

CK abstracts the GPU memory hierarchy into a unified address space enumeration, with data flowing through a four-level pipeline:

```
Global Memory (HBM)
  ↓  buffer_load / async DMA
VGPR (Vector Register)
  ↓  ds_write_b128
LDS (Local Data Share, 64KB/CU)
  ↓  ds_read_b128
VGPR → MFMA (Matrix Fused Multiply-Add)
 ↓ result
Global Memory
```

Key hardware parameters (MI300X):

| Parameter | Value |
|------|-----|
| CU count | 304 |
| SIMD units/CU | 4 |
| Threads/wavefront | 64 |
| LDS/CU | 64 KB |
| LDS banks | 32 or 64 |
| Bank width | 4 bytes |
| VGPR/CU | 512 (32-bit each) |

---

## 2. BufferView — Raw Memory Abstraction

`buffer_view` is CK's lowest-level memory access abstraction, distinguished by compile-time template parameters for address space, out-of-bounds handling strategy, and cache coherence:

```cpp
template <address_space_enum AddressSpace,
          typename T,
          typename BufferSizeType,
          bool InvalidElementUseNumericalZeroValue,
          amd_buffer_coherence_enum Coherence>
struct buffer_view;
```

### 2.1 Address Spaces

| AddressSpace | Corresponding Storage | Typical Use |
|-------------|---------|---------|
| `address_space_enum::global` | HBM | Input matrices A/B, output matrix C |
| `address_space_enum::lds` | Local Data Share | Tile data staging |
| `address_space_enum::vgpr` | Vector register file | Intermediate computation results |

### 2.2 Out-of-Bounds Handling

CK provides two OOB (Out-Of-Bounds) modes, both implemented branchlessly:

- **Zero mode** (`InvalidElementUseNumericalZeroValue = true`): Out-of-bounds elements return the numeric value zero, leveraging the hardware characteristics of the `buffer_load` instruction with no additional overhead.
- **Custom Value mode** (`InvalidElementUseNumericalZeroValue = false`): Out-of-bounds elements return a custom value, used for padding scenarios.

### 2.3 Vectorized Access

BufferView supports vectorized reads and writes via `ext_vector_t<T, N>`, accessing N elements at a time:

```cpp
// read 4 float16 (8 bytes)
auto data = buf.get(offset, bool_constant<true>{}); // compilation OOB check

// vectortype
using float16x8 = ext_vector_t<half_t, 8>;  // 128-bit = ds_read_b128
```

### 2.4 Atomic Operations

BufferView supports atomic add, atomic max, and other operations for result reduction in Stream-K GEMM:

```cpp
buf.atomic_add(offset, value);
buf.atomic_max(offset, value);
```

---

## 3. TensorView — Multi-Dimensional Tensor Structure

TensorView overlays `TensorDescriptor` (shape + stride) on top of BufferView, forming a zero-copy view of multi-dimensional tensors:

```cpp
// create packed row-major tensor
auto a_tensor = make_naive_tensor_view_packed(
 a_ptr, // pointer
 make_tuple(M, K), //
 number<1>{}); // stride-1 dimension
```

Key features:

- **Zero-copy transpose**: Only swaps the strides in the TensorDescriptor without moving any data.
- **Row-major / Column-major**: Through stride configuration, `(K, 1)` is row-major, and `(1, M)` is column-major.
- **Custom strides**: Supports arbitrary memory layouts such as strided batch and tiling.

---

## 4. XOR Preshuffle — LDS Bank Conflict Elimination

### 4.1 Problem Background

LDS consists of 32/64 banks, each bank 4 bytes wide. When multiple lanes within the same phase access different addresses in the same bank, a bank conflict occurs, leading to serialized access and performance degradation.

`ds_write_b128` divides the 64-lane wavefront into 8 phases (8 lanes per phase):
- Phase 0: lanes 0-7
- Phase 1: lanes 8-15
- ...
- Phase 7: lanes 56-63

`ds_read_b128` uses a different phase grouping (paired lanes):
- Phase 0: lanes 0:3 + lanes 20:23
- Phase 1: lanes 4:7 + lanes 16:19
- ...

### 4.2 XOR Transformation Principle

XOR preshuffle remaps LDS addresses through **coordinate transformation**, ensuring that lanes within the same phase access different banks. The core idea is to apply an XOR operation to the column index:In CK's official implementation, this is a three-step coordinate transformation process:

1. **XOR Transform**: `K0' = K0 XOR (M % (KPerBlock / KPack * MLdsLayer))`
2. **Unmerge Transform**: Splits the one-dimensional index into multi-dimensional coordinates
3. **Merge Transform**: Merges the transformed coordinates back into a one-dimensional LDS address

### 4.3 CK Implementation

```cpp
template <typename BaseLengths, // LDS
 typename BaseStrides, // LDS
 index_t XORFactor> // XOR
struct LdsIndexSwapping {
    // English comment
    using XorTransform    = XorTransform<BaseLengths[1], XORFactor>;
    using UnmergeTransform = UnmergeTransform<...>;
    using MergeTransform   = MergeTransform<...>;

    __device__ auto operator()(index_t m, index_t k) const {
        auto k_xor = k ^ (m % factor);   // Step 1: XOR
 // Step 2-3: Unmerge + Merge LDS offset
        return compute_offset(m, k_xor);
    }
};
```

### 4.4 MLdsLayer Parameter

When multiple rows share the same set of bank mappings, `MLdsLayer` controls the period of the XOR factor:

```
factor = KPerBlock / (KPack * MLdsLayer)
K0' = K0 XOR (M % factor)
```

The larger `MLdsLayer` is, the longer the XOR period, which can eliminate more inter-row bank conflicts, but it increases register pressure.

### 4.5 Comparison with Traditional Padding

| Method | Extra LDS Overhead | Implementation Complexity | Effectiveness |
|------|-------------|-----------|------|
| Padding (+1 column) | Wastes 1 element per row | Low | Partial elimination |
| XOR Preshuffle | **Zero extra storage** | Medium (coordinate transform) | Complete elimination |

The core advantage of XOR preshuffle is **zero storage overhead** — it does not increase LDS usage and does not affect occupancy.

### 4.6 Usage in GEMM

```cpp
// GEMM kernel LDS
__shared__ float lds_a[MPerBlock][KPerBlock]; // none padding

// write LDS XOR preshuffle
auto lds_offset = lds_index_swapping(m_local, k_local);
lds_a_view.set(lds_offset, value);

// read LDS usesame
auto data = lds_a_view.get(lds_index_swapping(m_local, k_local));
```

---

## 5. Morton Ordering — Z-Order Spatial Locality

### 5.1 Principle

Morton ordering (Z-order curve) maps 2D coordinates to 1D indices through **bit interleaving**, preserving spatial locality:

For coordinate `(y, x)`, the Morton index is `y1 x1 y0 x0` (binary bit interleaving).

Morton indices for a 4x4 Tile:

```
 0  1  4  5
 2  3  6  7
 8  9 12 13
10 11 14 15
```

Compared to row-major ordering (`0,1,2,3,4,...`), Morton ordering ensures that spatially adjacent elements are also adjacent in the 1D index.

### 5.2 CK Implementation

CK implements Morton ordering using two Transform steps:

1. **UnmergeTransform**: Splits the linear tile into 2D sub-blocks
2. **MergeTransform**: Merges coordinate dimensions in a bit-interleaved manner

```cpp
// English comment
morton_index = interleave_bits(row_within_tile, col_within_tile);
```

### 5.3 Combination with XOR Preshuffle

Morton ordering and XOR preshuffle can be used together:

- **Morton ordering**: Optimizes the thread→element mapping, improving cache locality
- **XOR preshuffle**: Optimizes LDS address mapping, eliminating bank conflicts

The two are orthogonal and do not conflict with each other.

---

## 6. Space-Filling Curve — Cache-Friendly Traversal

### 6.1 Template Interface

```cpp
template <index_t NDim,
 typename SFCLengths, // dimension
 typename DimAccessOrder, // dimension
 typename ScalarsPerAccess, // accessscalar
 bool IsSnakeCurved = false> //
struct space_filling_curve;
```

### 6.2 Snake Traversal

When `IsSnakeCurved = true`, even-numbered rows are traversed forward and odd-numbered rows backward, reducing the jump distance between adjacent rows:

```
English description
backward: ←←←←←
English description
backward: ←←←←←
```

### 6.3 Vectorization and Dimension Ordering

- `ScalarsPerAccess`: Controls the vectorization width of the innermost dimension, e.g., `<1,1,8>` means the innermost layer reads 8 scalars at a time
- `DimAccessOrder`: `<0,1,2>` is row-major, `<2,1,0>` is column-major, affecting traversal cache behavior

### 6.4 Integration in LoadStoreTraits

The Space-Filling Curve is automatically selected by `LoadStoreTraits` and embedded into the TileWindow's load/store path; users typically do not need to use it directly.

## 7. TileWindow — Distributed Access Window

### 7.1 Concept

TileWindow is the core data access gateway in CK, binding a rectangular window on a TensorView to thread distribution:

```cpp
template <typename TensorView,
          typename WindowLengths,
          typename TileDistribution>
struct tile_window_with_static_distribution;
```

### 7.2 Load Process

The TileWindow's `load()` operation performs the following steps:

1. **LoadStoreTraits Analysis**: Determines the `vector_dim_y` (stride-1 dimension) and `scalar_per_vector` (vectorization width)
2. **SFC Traversal**: Traverses elements within the window in Space-Filling Curve order
3. **Vectorized Access**: Executes `buffer_load` in units of `scalar_per_vector`
4. **Boundary Handling**: OOB elements are automatically zero-padded (leveraging BufferView hardware features)

### 7.3 Window Movement

```cpp
// O(1) - , copydata
window.set_window_origin(make_tuple(m_offset, k_offset));
```

This makes sliding window iteration over the K dimension highly efficient, modifying only a single coordinate value each time.

### 7.4 Store Operation

Store is symmetric with Load, using the same LoadStoreTraits analysis and SFC traversal path, ensuring that writes also benefit from vectorization and coalesced write optimizations.

---

## 8. LoadStoreTraits — Compile-Time Vectorization Analysis

### 8.1 Function

LoadStoreTraits is a compile-time analysis engine that automatically determines optimal parameters for each TileWindow access:

| Analysis Item | Meaning | Typical Value |
|--------|------|--------|
| `vector_dim_y` | stride-1 dimension (contiguous memory direction) | K dim (row-major A) |
| `scalar_per_vector` | number of scalars per read/write | float32: 4, float16: 8 |
| SFC Type | traversal curve selection | snake / linear |
### 8.2 Vectorization Width Selection

```
scalar_per_vector = min(
 128 / sizeof(T) / 8, // vector (128-bit)
 tile_length_on_vector_dim, // tile contiguous
 alignment_of_base_ptr // pointer
)
```

For `float16`: `128 / 16 / 8 = 8`, i.e., reading 8 FP16 at once = 128 bit = `ds_read_b128`.

### 8.3 Multi-Level Vectorization

When the tile length along the contiguous dimension exceeds the single vector width, LoadStoreTraits automatically generates multiple vectorized accesses, each covering `scalar_per_vector` elements.

---

## 9. LDS Double Buffering

### 9.1 Principle

In the GEMM pipeline, LDS double buffering achieves overlap of computation and data movement by allocating **twice the LDS space**:

```
: t0 t1 t2 t3
Buffer 0: [load tile0] [compute tile0] [load tile2] [compute tile2]
Buffer 1: [load tile1] [compute tile1] [load tile3]
```

### 9.2 CK Implementation

In `gemm_pipeline_ag_bg_cr_comp_async.hpp`:

```cpp
// double buffer
static_assert(DoubleSmemBuffer == true, "pipeline requires double smem buffer");

// LDS size
static constexpr index_t GetSmemSize() {
    return 2 * smem_size;  // Ping-pong buffer
}
```

The key to ping-pong mode is **alternating between two LDS buffers**:

- Even iterations: read from Buffer 0 + write to Buffer 1
- Odd iterations: read from Buffer 1 + write to Buffer 0

### 9.3 Synchronization

- Standard pipeline: uses `block_sync_lds()` to ensure writes are complete before reading
- Async pipeline: uses `block_sync_lds_direct_load()` to wait for async DMA completion

---

## 10. Async DMA Prefetch

### 10.1 Global → LDS Direct Transfer

CK's async pipeline uses `GlobalPrefetchAsync` to achieve direct Global → LDS transfer, bypassing VGPR staging:

```
standardpath: Global -> VGPR -> LDS
Async path: Global -> LDS (, DMA )
```

This frees up VGPRs for computation, reducing register pressure.

### 10.2 Integration with Double Buffering

Async DMA combined with LDS double buffering achieves three-level pipeline overlap:

```
Iteration i:
 - Async DMA: prefetch tile[i+1] -> LDS buffer[(i+1)%2]
 - DS Read: LDS buffer[i%2] read tile[i] -> VGPR
 - MFMA: compute tile[i] matrix
```

### 10.3 Prefetch Depth

The multi-stage prefetch depth is determined by the `MinMemInFlyBytes = 32KB` formula. See [CK GEMM Pipeline Architecture](ck-gemm-pipelines.md#2-multi-stage-prefetch-and-minmeminflybytes) for details.

---

## 11. Practical Recommendations

### Memory Access Optimization Checklist

| Optimization | Expected Benefit | Applicable Scenarios |
|--------|---------|---------|
| XOR Preshuffle | Eliminates LDS bank conflicts, zero additional storage | All GEMM, Attention |
| Vectorization (128-bit) | 4x–8x throughput vs. scalar | Contiguous memory access |
| SFC Snake Traversal | Reduces cache misses | 2D tile traversal |
| LDS Double Buffering | Compute-transfer overlap | Multi-iteration kernels |
| Async DMA | Frees VGPRs, reduces latency | HBM bandwidth-bound kernels |
| Morton Ordering | Improves spatial locality | 2D data access patterns |

## Related Documents
- [CK GEMM Pipeline Architecture](ck-gemm-pipelines.md) — GEMM pipeline variants, scheduling strategies, MFMA instruction selection- [LDS Bank Conflict Optimization](../kernel-opt/lds-bank-conflict-optimization.md) — Bank architecture and conflict checking fundamentals
- [AMD MFMA Matrix Core Programming Guide](amd-mfma-matrix-cores.md) — MFMA instruction naming conventions and register layout
- [aiter Optimization Techniques Summary](aiter-optimization-techniques.md) — CK usage practices in AMD inference operator libraries
