# Shared Memory Swizzling (Blackwell SM100)


**Last updated**: 2026-07-01

## Overview

Shared memory swizzling remaps the linear address layout of a matrix tile in SMEM so that threads accessing consecutive columns (or rows) hit different 32-byte banks rather than the same bank. This eliminates bank conflicts that would otherwise serialize concurrent accesses. On Blackwell (SM100), 128-byte swizzling is mandatory for TMA loads and tcgen05.mma operands. Without it, performance drops to 46% of the achievable throughput for GEMM workloads.

## Why 128-Byte Swizzling is Mandatory on Blackwell

Shared memory has 32 banks, each 4 bytes wide (128 bytes total per bank cycle). When a warp accesses a matrix stored in row-major layout, threads in the same warp reading elements from consecutive rows in the same column hit the same bank, causing a 32-way bank conflict.

The TMA unit on both Hopper and Blackwell encodes the swizzle pattern as part of the tensor descriptor. The tcgen05.mma instruction expects its SMEM operands to already be swizzled in the 128-byte pattern. Using unswizzled data produces incorrect MMA results.

The tcgen05-tutorial benchmark progression shows the impact:

```
Naive (no swizzle):   255 TFLOPS  (17% of cuBLAS)
128B swizzle applied: 695 TFLOPS  (46% of cuBLAS)
                      ---- 2.7x improvement from swizzling alone ----
```

## How 128-Byte Swizzling Works

The swizzle function XORs a portion of the column address with the row address to scatter accesses across banks:

```cuda
// 128-byte swizzle: XOR bits [4:6] of the byte offset with the row index
// This ensures that consecutive rows accessing the same logical column
// map to different physical SMEM banks.
//
// For a tile stored in SMEM with TILE_N columns of 2-byte elements:
//   byte_offset = row * (TILE_N * sizeof(half)) + col * sizeof(half)
//   swizzled_offset = byte_offset ^ ((row & 0x7) << 4)
//
// The mask 0x7 = 3 bits, shift 4 = bits [4:6], giving 8-row periodicity
// across the 128-byte bank group.

__device__ int swizzle_128B(int row, int col, int stride_bytes) {
    int byte_offset = row * stride_bytes + col * sizeof(half);
    // XOR bits [4:6] of byte offset with low 3 bits of row
    int swizzled = byte_offset ^ ((row & 0x7) << 4);
    return swizzled;
}
```

Visually, for an 8-row x 64-column half-precision tile (128 bytes per row):

```
Without swizzle (row-major):
  Row 0: bank 0,1,2,...,31  bank 0,1,2,...,31
  Row 1: bank 0,1,2,...,31  bank 0,1,2,...,31
  Row 2: bank 0,1,2,...,31  bank 0,1,2,...,31
  -> Column access = 8-way bank conflict

With 128B swizzle (XOR pattern):
  Row 0: bank 0,1,2,...,31  bank 0,1,2,...,31
  Row 1: bank 1,2,3,...,0   bank 1,2,3,...,0    (rotated by 1)
  Row 2: bank 2,3,4,...,1   bank 2,3,4,...,1    (rotated by 2)
  -> Column access = conflict-free
```

## TMA Swizzle Encoding

The TMA descriptor encodes the swizzle mode when creating a tensor map. The swizzle mode must match what the consumer (tcgen05.mma or wgmma) expects:

```cuda
// Creating a TMA descriptor with 128-byte swizzle
#include <cuda.h>

CUtensorMap tensor_map;

// Swizzle mode: CU_TENSOR_MAP_SWIZZLE_128B
// This tells TMA to apply the 128-byte XOR swizzle pattern
// when writing data into shared memory
cuTensorMapEncodeTiled(
    &tensor_map,
    CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
    2,                                    // 2D tensor
    global_ptr,                           // global memory base
    global_dims,                          // {N, M} dimensions
    global_strides,                       // {N * sizeof(half), sizeof(half)}
    tile_dims,                            // {TILE_N, TILE_M}
    element_strides,                      // {1, 1}
    CU_TENSOR_MAP_INTERLEAVE_NONE,
    CU_TENSOR_MAP_SWIZZLE_128B,           // 128-byte swizzle
    CU_TENSOR_MAP_L2_PROMOTION_NONE,
    CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
);
```

The available swizzle modes and their use cases:

| Swizzle Mode | Bank Spread | Use Case |
|-------------|-------------|----------|
| `SWIZZLE_NONE` | No remapping | Non-MMA data (flags, scales) |
| `SWIZZLE_32B` | 32-byte groups | Narrow tiles, small data types |
| `SWIZZLE_64B` | 64-byte groups | Medium tiles |
| `SWIZZLE_128B` | 128-byte groups | Standard for BF16/FP16 MMA operands |

## CuTe Swizzle Layout

In CuTe/CUTLASS, swizzle is expressed as a layout composition:

```cuda
// CuTe swizzle layout for 128-byte swizzle pattern
// Swizzle<B, M, S> where:
//   B = number of bits in the base (non-swizzled) portion
//   M = number of bits in the mask
//   S = shift amount
//
// Swizzle<3, 4, 3> encodes the 128B swizzle:
//   3 base bits (8-byte alignment)
//   4 mask bits (16 rows)
//   3 shift bits (8-column groups)

using SmemLayoutAtom = decltype(
    composition(
        Swizzle<3, 4, 3>{},
        Layout<Shape<_8, _64>,
               Stride<_64, _1>>{}
    )
);

// Tile the atom across the full SMEM tile
using SmemLayoutA = decltype(
    tile_to_shape(SmemLayoutAtom{}, Shape<Int<TILE_M>, Int<TILE_K>>{})
);
```

## Verification: Detecting Bank Conflicts

Use `nvprof` or Nsight Compute to verify that swizzling eliminates conflicts:

```python
# Nsight Compute command to check shared memory bank conflicts
# Look for "Shared Memory Bank Conflicts" metric
# ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum \
#     ./my_kernel

# Expected results:
# Without swizzle: bank_conflicts >> 0
# With 128B swizzle: bank_conflicts == 0
```

## When to Use

- **All Blackwell tensor core kernels**: 128-byte swizzling is not optional. Both TMA and tcgen05.mma require it for correct results and peak performance.
- **Hopper wgmma kernels**: Same requirement applies; wgmma expects swizzled SMEM operands.
- **Non-MMA shared memory access**: If multiple warps access the same SMEM tile in a column pattern (e.g., reduction), swizzling prevents serialization.

## Caveats

- The swizzle mode in the TMA descriptor must exactly match the access pattern of the consumer. A mismatch produces silently incorrect results, not a runtime error.
- Swizzled layouts make SMEM address computation non-trivial. Using CuTe's layout algebra avoids manual indexing errors.
- For data types wider than 2 bytes (e.g., FP32 accumulators), the optimal swizzle mode may differ. TMEM accumulators avoid this issue since they use a separate address space.


## Related

- [Cache Policy Differentiation](cache-policy.md)
- [Chunk-Based Parallelism](chunk-parallelism.md)
- [CUDA GEMM Optimization Ladder](cuda-gemm-optimization-ladder.md)
- [Double/Multi-Buffering Patterns](double-buffering.md)
- [Epilogue Fusion](epilogue-fusion.md)
- [Composable Kernel (CK) Architecture Overview](../../../amd/common/ck-architecture-overview.md)
