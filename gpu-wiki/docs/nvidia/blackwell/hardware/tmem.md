# Tensor Memory (TMEM)


**Last updated**: 2026-06-30

## Overview

Tensor Memory (TMEM) is a new addressable memory space introduced in the Blackwell architecture (SM100). Each SM contains **256KB of dedicated TMEM**, used primarily as the accumulator storage for `tcgen05.mma` operations. TMEM eliminates the register pressure that plagued Hopper's wgmma, where accumulators consumed 128+ registers per warpgroup.

## Architecture Layout

TMEM is organized as a 2D matrix:

| Dimension | Size | Description |
|---|---|---|
| Rows | 128 | Mapped to warp lanes (4 warps x 32 lanes) |
| Columns | 512 | 32-bit (4-byte) elements per row |
| Total | 128 x 512 x 4 bytes = 256 KB | Per-SM capacity |

### Row-to-Lane Mapping

TMEM rows are mapped to warp lanes within a CTA:

```
Row 0-31:   Warp 0, lanes 0-31
Row 32-63:  Warp 1, lanes 0-31
Row 64-95:  Warp 2, lanes 0-31
Row 96-127: Warp 3, lanes 0-31
```

Each thread "owns" the TMEM row corresponding to its warp and lane. When reading from TMEM, thread `T` in warp `W` accesses row `W*32 + T%32`.

### Column Addressing

Columns are addressed via a column offset in the TMEM descriptor. A 128x256 MMA accumulator tile occupies:

- 128 rows (all lanes across 4 warps)
- 256 columns of FP32 values = 1024 bytes per row

```
TMEM Layout for 128x256 FP32 accumulator:
  col 0        col 255     col 511
  |------------|-----------|-----------|
  |  Acc Tile  |  (free)   |  (free)   |  row 0   (warp0, lane0)
  |  128x256   |           |           |  row 1   (warp0, lane1)
  |  FP32      |           |           |  ...
  |            |           |           |  row 127  (warp3, lane31)
  |------------|-----------|-----------|
```

## Allocation and Deallocation Lifecycle

TMEM is **explicitly managed** by the programmer. There is no automatic allocation or garbage collection.

### Allocation

```cuda
// Shared storage for CTA-wide broadcast of TMEM address
__shared__ uint32_t s_tmem_addr;

__device__ uint32_t tmem_alloc_cta(uint32_t num_cols) {
    // Only thread 0 allocates; result must reach ALL warps in the CTA.
    // __shfl_sync is warp-local -- it cannot broadcast across warps.
    if (threadIdx.x == 0) {
        uint32_t addr;
        asm volatile(
            "tcgen05.alloc.cta_group::1.sync.aligned.b32 %0, %1;"
            : "=r"(addr)
            : "r"(num_cols)
        );
        s_tmem_addr = addr;
    }
    __syncthreads();  // All warps now see s_tmem_addr
    return s_tmem_addr;
}
```

Key points:
- `num_cols` specifies the number of 32-bit columns to allocate.
- A 128x256 FP32 accumulator needs 256 columns.
- Allocation is **CTA-scoped** -- all threads in the CTA share the same TMEM region.
- Only one thread (typically thread 0) issues the allocation.
- The returned `tmem_addr` is the base column index.

### Deallocation

```cuda
__device__ void tmem_dealloc(uint32_t tmem_addr, uint32_t num_cols) {
    if (threadIdx.x == 0) {
        asm volatile(
            "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
            :
            : "r"(tmem_addr), "r"(num_cols)
        );
    }
    __syncthreads();
}
```

**Warning**: Failure to deallocate TMEM before CTA exit will leak memory and prevent subsequent CTAs from allocating, leading to hangs in persistent kernels.

### Lifecycle in a Persistent GEMM Kernel

```cuda
__global__ void persistent_gemm_kernel(/* ... */) {
    // 1. Allocate TMEM for accumulators
    uint32_t tmem_acc = tmem_alloc(256);  // 256 cols for M=128, N=256

    while (has_more_tiles()) {
        // 2. Zero-initialize TMEM accumulator
        tmem_zero(tmem_acc, 256);

        // 3. Mainloop: accumulate K tiles
        for (int k = 0; k < K_tiles; ++k) {
            issue_tcgen05_mma(tmem_acc, smem_a[k], smem_b[k]);
        }

        // 4. Fence and read results
        asm volatile("tcgen05.mma.fence::before_thread_sync;");
        __syncthreads();

        // 5. Epilogue: read from TMEM, apply bias/activation, write to GMEM
        epilogue_from_tmem(tmem_acc, output);

        // 6. TMEM persists across tiles -- no need to reallocate
    }

    // 7. Deallocate before exit
    tmem_dealloc(tmem_acc, 256);
}
```

## Data Movement Operations

### TMEM Store (Register to TMEM)

```cuda
// Store a register value to TMEM
// Each thread writes to its own TMEM row at the specified column
__device__ void tmem_store_f32(uint32_t tmem_col, float value) {
    asm volatile(
        "tcgen05.st.sync.aligned.32x1b.x1.b32 [%0], {%1};"
        :
        : "r"(tmem_col), "f"(value)
    );
}

// Vectorized store: 4 consecutive FP32 values
__device__ void tmem_store_f32x4(uint32_t tmem_col, float4 values) {
    asm volatile(
        "tcgen05.st.sync.aligned.32x1b.x4.b32 [%0], {%1, %2, %3, %4};"
        :
        : "r"(tmem_col),
          "f"(values.x), "f"(values.y),
          "f"(values.z), "f"(values.w)
    );
}
```

### TMEM Load (TMEM to Register)

```cuda
// Load a single FP32 from TMEM
__device__ float tmem_load_f32(uint32_t tmem_col) {
    float result;
    asm volatile(
        "tcgen05.ld.sync.aligned.32x1b.x1.b32 {%0}, [%1];"
        : "=f"(result)
        : "r"(tmem_col)
    );
    return result;
}

// Vectorized load: 4 consecutive FP32 values
__device__ float4 tmem_load_f32x4(uint32_t tmem_col) {
    float4 result;
    asm volatile(
        "tcgen05.ld.sync.aligned.32x1b.x4.b32 {%0, %1, %2, %3}, [%4];"
        : "=f"(result.x), "=f"(result.y),
          "=f"(result.z), "=f"(result.w)
        : "r"(tmem_col)
    );
    return result;
}
```

### TMEM Zero-Fill

```cuda
// Zero-fill a range of TMEM columns
__device__ void tmem_zero(uint32_t tmem_base_col, uint32_t num_cols) {
    // Each thread zeros its own row
    for (uint32_t c = 0; c < num_cols; c += 4) {
        float4 zero = make_float4(0.f, 0.f, 0.f, 0.f);
        tmem_store_f32x4(tmem_base_col + c, zero);
    }
}
```

### Bulk TMEM Copy via tcgen05.cp

TMEM supports bulk copy operations between TMEM regions:

```ptx
// Copy 256 columns of TMEM from src to dst
tcgen05.cp.cta_group::1.b128 [dst_tmem_col], [src_tmem_col];
```

## Double-Buffering with TMEM

Double-buffering TMEM accumulators enables overlapping the epilogue of the current tile with the MMA accumulation of the next tile:

```cuda
__global__ void double_buffered_gemm(/* ... */) {
    // Allocate two accumulator buffers
    uint32_t tmem_acc[2];
    tmem_acc[0] = tmem_alloc(256);
    tmem_acc[1] = tmem_alloc(256);

    int buf = 0;

    for (int tile = 0; tile < num_tiles; ++tile) {
        // Zero the current buffer
        tmem_zero(tmem_acc[buf], 256);

        // Mainloop: accumulate into current buffer
        for (int k = 0; k < K_tiles; ++k) {
            issue_tcgen05_mma(tmem_acc[buf], smem_a[k], smem_b[k]);
        }
        asm volatile("tcgen05.mma.fence::before_thread_sync;");
        __syncthreads();

        // If not the first tile, the *other* buffer's epilogue
        // was overlapped with this tile's MMA in the pipeline.

        // Start epilogue for current buffer
        // (can overlap with next tile's MMA on the other buffer)
        epilogue_from_tmem(tmem_acc[buf], output_tile[tile]);

        buf ^= 1;  // Swap buffers
    }

    // Cleanup
    tmem_dealloc(tmem_acc[0], 256);
    tmem_dealloc(tmem_acc[1], 256);
}
```

### TMEM Budget Considerations

With 512 total columns per SM:

| Accumulator Size | Columns | Max Buffers | Remaining for Scratch |
|---|---|---|---|
| 128x128 FP32 | 128 cols | 4 | 0 |
| 128x192 FP32 | 192 cols | 2 | 128 |
| 128x256 FP32 | 256 cols | 2 | 0 |
| 128x256 FP32 (2x) | 512 cols | 2 (double-buf) | 0 |

For double-buffered 128x256 tiles, the full 512 columns are consumed. If additional scratch TMEM is needed (e.g., for softmax in attention kernels), reduce the tile size or use a single accumulator buffer.

### When can you pipeline-overlap? (columns / bytes / head-dim threshold)

Pipeline overlap of MMA with the epilogue (or of consecutive MMAs) relies on having a **second accumulator buffer** so one can be filled while the other is drained. That requires both buffers to coexist in the 512-column TMEM:

> **Double-buffer fits ⇔ 2 × (accumulator columns) ≤ 512 ⇔ accumulator ≤ 256 columns.**

Column ↔ byte conversion (per SM): each TMEM column spans 128 lanes × 4 B = **512 B/column**, so 512 columns = 256 KB.

| Per-accumulator footprint | Columns | Bytes | Can double-buffer (overlap)? |
|---|---|---|---|
| ≤ 128 cols | ≤ 128 | ≤ 64 KB | Yes — up to 4 buffers, room for scratch |
| ≤ 256 cols | ≤ 256 | ≤ 128 KB | Yes — exactly 2 buffers, no scratch left |
| > 256 cols | > 256 | > 128 KB | **No** — only one fits; overlap must come from 2-SM (split the accumulator across the pair) or a smaller tile |

For a GEMM tile M×N the accumulator occupies **N columns** (M maps to the 128 lanes), so overlap is feasible when **N ≤ 256**.

For attention, the accumulators are sized by **head-dim and KV-tile**: the O accumulator is **head-dim columns** and the S accumulator is **kv-tile columns**. Practical consequences:

- **head-dim ≤ 128:** O (≤128 cols) leaves ample room to double-buffer O and still hold S + softmax scratch.
- **head-dim = 256:** a single O accumulator already uses 256 cols (= half of TMEM). With an S accumulator alongside it (e.g. 128 cols → S+O = 384 ≤ 512 fits **single-buffered**), there is **no room to also double-buffer O on one SM** — so large head-dim trades away overlap headroom. Options: keep single-stage (`q_stage=1`) on 1 SM, or go 2-SM to split O across the pair's TMEM (then weigh the cluster sync tax — see [2sm-cooperative.md](2sm-cooperative.md)).

Rule of thumb: **overlap is "free" on TMEM up to ~256 accumulator columns (~128 KB); beyond that you must either shrink the tile or use 2-SM.**

## Microbenchmark Data

From published Blackwell microbenchmarks:

| Metric | TMEM | SMEM | Registers |
|---|---|---|---|
| End-to-end latency (cache miss) | ~420 cycles | ~30 cycles | ~4 cycles |
| Bandwidth for large working sets | High (dedicated bus) | Medium | N/A (limited count) |
| Best for | Multi-stage tensor pipelines | Single-shot small matrix | Scalar/vector ALU |

TMEM is **not** a replacement for shared memory. Its strength is in serving as a dedicated accumulator buffer that eliminates register pressure for large MMA tiles. SMEM remains faster for small, frequently accessed data.

## TMEM in the CUTLASS Abstraction

In CUTLASS 4.5.0 for SM100, TMEM is managed through CuTe layouts:

```python
# CuTe-DSL example: TMEM accumulator layout
# 128 rows x 256 columns, FP32
tmem_layout = Layout(
    shape=(128, 256),
    stride=(256, 1),
    memory_space=MemorySpace.TMEM
)

# Allocate TMEM accumulator
acc = tmem_alloc(tmem_layout)

# Issue MMA -- accumulator lives in TMEM
tcgen05_mma(acc, smem_a, smem_b)

# Fence and read
tcgen05_fence()
result = tmem_load(acc)
```

## Common Pitfalls

1. **Forgetting to fence**: Reading TMEM without `tcgen05.mma.fence::before_thread_sync` produces undefined (stale) values.
2. **Forgetting to deallocate**: In persistent kernels, TMEM must be freed before re-acquiring tiles. Otherwise, the next allocation will fail or hang.
3. **Exceeding 512 columns**: Attempting to allocate more than the SM's total column budget silently corrupts data or causes a hang.
4. **Cross-warp reads**: A thread can only directly read/write TMEM rows mapped to its own lane. Accessing another warp's rows requires explicit shuffle or SMEM staging.
5. **Assuming SMEM-like latency**: TMEM has ~420-cycle latency on cache miss vs ~30 cycles for SMEM. Do not use TMEM for low-latency random access patterns.


## Related

- [Two-SM Cooperative MMA](2sm-cooperative.md)
- [Cluster Launch Control (CLC)](clc.md)
- [CUDA Memory Hierarchy for Kernel Optimization](cuda-memory-hierarchy.md)
- [GPU Architecture Resource Model](gpu-architecture-resource-model.md)
- [mbarrier](mbarrier.md)
- [Composable Kernel (CK) Architecture Overview](../../../amd/common/ck-architecture-overview.md)
