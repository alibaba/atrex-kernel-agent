# Tile Rasterization and L2 Cache Locality

How the traversal order (rasterization strategy) of GEMM tiles affects L2 cache hit rate: row-major, column-major, swizzle patterns, persistent kernels, and tile scheduling.

---

## 1. L2 Cache Locality Issues

### 1.1 Data Access Patterns in GEMM

In matrix multiplication C = A * B, each output tile C[i][j] needs to read:
- The i-th row stripe of matrix A (tile_M x K)
- The j-th column stripe of matrix B (K x tile_N)

```
        B (K x N)
        j   j+1  j+2  j+3
       +-+--+-+--+-+--+-+
 | | | | | | | | K row
       | |  | |  | |  | |
       +-+--+-+--+-+--+-+

A (M x K)
     +--------+        C (M x N)
 i   |========| ---->  [C00][C01][C02][C03]
 i+1 |========| ---->  [C10][C11][C12][C13]
 i+2 |========| ---->  [C20][C21][C22][C23]
 i+3 |========| ---->  [C30][C31][C32][C33]
     +--------+

Compute C[0][0] needs: Row 0 of A + Column 0 of B
Compute C[0][1] needs: Row 0 of A + Column 1 of B
                    ↑ Row 0 of A reused!
```

Key observation: Adjacent output tiles share input data. Exploiting this sharing relationship is the core of L2 optimization.

### 1.2 CTA Dispatch Order Determines L2 Reuse

The GPU's hardware scheduler assigns CTAs to SMs in a non-deterministic order. If the execution order of CTAs happens to process tiles sharing data simultaneously (or nearly simultaneously), the shared data remains in the L2 cache, avoiding repeated reads from HBM.

**Worst case**: CTAs are scheduled randomly, each tile reads A and B independently, with zero L2 reuse.

```
Data volume comparison (M=N=K=8192, tile=128):
  No reuse: Each tile reads A_strip + B_strip = 2 * 128 * 8192 * 2B = 4MB
           64x64 = 4096 tiles, total read = 16 TB  ← Disaster
  Perfect reuse: A read once + B read once = 2 * 8192 * 8192 * 2B = 256 MB
```

### 1.3 When L2 Locality Matters

| Condition | Does L2 Locality Matter? |
|-----------|---------------------------|
| Problem is small enough (A+B all fit in L2) | Not important — all accesses hit regardless of traversal |
| K dimension is very small (< 256) | Not very important — each strip is small |
| Both M and N are large (> 4096) | Very important — accounts for 10–30% performance difference |
| Batch GEMM | Moderately important — the batch dimension provides additional reuse |

---

## 2. Rasterization Strategies

### 2.1 Row-Major (Along N / Row-Major)

CTAs traverse the output tile grid by row:

```
English description
  (0,0) → (0,1) → (0,2) → (0,3) → (0,4) → (0,5) →
  (1,0) → (1,1) → (1,2) → (1,3) → (1,4) → (1,5) →
  (2,0) → (2,1) → (2,2) → ...

 N (column) ->
M  +----+----+----+----+----+----+
(  | 0  | 1  | 2  | 3  | 4  | 5  |
row +----+----+----+----+----+----+
 | 6 | 7 | 8 | 9 | 10 | 11 |
English description
)  | 12 | 13 | 14 | 15 | 16 | 17 |
↓  +----+----+----+----+----+----+
```

**L2 behavior**:
- Tiles in the same row share the same row stripe of A → good A reuse
- The column stripes of B switch one by one → when N is large, by the time the N-th column is processed, the B data from column 0 has already been evicted from L2
- The next row needs all columns of B → all of B is reloaded

**Suitable scenario**: M >> N (when there are far more rows than columns, the A stripe is the bottleneck)

### 2.2 Column-Major (Along M / Column-Major)

CTAs traverse by column:

```
 N (column) ->
M  +----+----+----+----+----+----+
(  | 0  | 3  | 6  | 9  | 12 | 15 |
row +----+----+----+----+----+----+
 | 1 | 4 | 7 | 10 | 13 | 16 |
English description
)  | 2  | 5  | 8  | 11 | 14 | 17 |
↓  +----+----+----+----+----+----+
```

**L2 behavior**:
- Tiles in the same column share the same column stripe of B → good B reuse
- The row stripes of A switch one by one → when M is large, A is continuously evicted

**Suitable scenario**: N >> M (when there are far more columns than rows, the B stripe is the bottleneck)

### 2.3 Swizzle (Interleaved Rasterization)

Swizzle is a compromise between row-major and column-major. The core idea: partition the tile grid into small blocks (swizzle blocks), and within each block use a traversal order different from the global order, so that both A and B stripes achieve local reuse.

```
Swizzle Size = 4 example (6x6 tile grid):

Within each 4x4 swizzle block:
  Row 0: 0, 1, 2, 3     (normal order)
  Row 1: 5, 4, 7, 6     (XOR flipped)
  Row 2: 8, 9, 10, 11   (normal order)
  Row 3: 13, 12, 15, 14 (XOR flipped)
```**Swizzle Formula** (from linear index to 2D tile coordinates):

```python
def swizzle_tile_index(linear_idx, num_tiles_major, log_swizzle_size):
    """Map linear tile index to (minor, major) coordinates"""
    swizzle_size = 1 << log_swizzle_size

    # Extract offset within swizzle block
    offset = linear_idx & (swizzle_size - 1)         # Low log_swizzle_size bits
    extra = linear_idx >> log_swizzle_size            # High bits

    # Extract major and minor block indices from high bits
    minor_div_swizzle = extra // num_tiles_major
    major = extra % num_tiles_major

    # Reconstruct minor coordinate: block index * swizzle_size + offset
    minor = minor_div_swizzle * swizzle_size + offset

    return minor, major
```

**L2 Behavior**: Tiles within a swizzle block have locality in both the M and N dimensions. In a 4x4 swizzle block, A's row stripes are used by at most 4 different tiles, and B's column stripes are also used by at most 4 different tiles. Both can survive in L2.

### 2.4 Automatic Rasterization Order Selection (Heuristic)

Production-grade libraries use simple heuristic rules:

```python
def choose_raster_order(tiles_m, tiles_n):
    """Choose rasterization direction"""
        return "AlongM"    # Column-major: N dimension large, keep B columns in L2
        return "AlongN"    # Row-major: M dimension large, keep A rows in L2
```

Intuition: Traverse along the **shorter** dimension as the "major" direction, so that stripes along the **longer** dimension can be reused across more tiles.

---

## 3. Swizzle Size Selection

### 3.1 What is Swizzle Size

`swizzle_size` is the number of tiles in a swizzle block along the minor dimension. It controls the "window" size of local reuse.

```
swizzle_size = 1:  Degrades to pure linear traversal (no swizzle)
swizzle_size = 2:  2 adjacent tiles as a group interleaved
swizzle_size = 4:  4 adjacent tiles as a group
swizzle_size = 8:  8 adjacent tiles as a group (maximum common value)
```

### 3.2 Relationship with L2 Capacity

The upper bound of swizzle size is determined by L2 cache capacity:

```
swizzle_size * tile_K * element_size  <=  L2_effective_capacity

For example:
  tile_K = 128 elements
  element_size = 2 bytes (FP16)
  L2 per SM partition ≈ 256 KB (H100 has 50 MB L2, ~72 partitions)

  swizzle_size <= 256 KB / (128 * 2) = 1024  ← Theoretical upper bound

Actually limited by concurrently active CTA count and other L2 users:
  Actual effective L2 ≈ 64 KB per CTA
  swizzle_size <= 64 KB / (128 * 2) = 256   ← Still large

So the bottleneck is usually not L2 capacity, but tile grid size.
```

### 3.3 Relationship with Tile Grid Size

Swizzle size cannot exceed the number of tiles along the minor dimension, otherwise it is meaningless. The selection logic in production code:

```python
def get_log_swizzle_size(tiles_m, tiles_n, max_swizzle_size):
    """Select swizzle size based on tile grid dimensions"""
    min_dim = min(tiles_m, tiles_n)

    if max_swizzle_size >= 8 and min_dim >= 6:
        return 3    # swizzle_size = 8
    elif max_swizzle_size >= 4 and min_dim >= 3:
        return 2    # swizzle_size = 4
    elif max_swizzle_size >= 2 and min_dim >= 2:
        return 1    # swizzle_size = 2
    else:
        return 0    # swizzle_size = 1 (no swizzle)
```

Note the asymmetry of the thresholds: `min_dim >= 6` is requiredpop to use `swizzle_size=8`, not `>= 8`. This is because the swizzle block size is rounded up to a multiple of swizzle_size, and an overly large swizzle size on a small grid would waste resources (producing out-of-bounds tile indices that need to be skipped).

### 3.4 Impact of Cluster Shape

Hopper introduced Thread Block Clusters, where multiple CTAs form a cluster that shares distributed shared memory. The granularity of swizzle must align with the cluster shape:

```
Actual swizzle granularity = swizzle_size * cluster_shape_minor

For example: cluster_shape = (2, 1), swizzle_size = 4
  → Actual granularity = 4 * 2 = 8 CTAs as a swizzle group
```

The tile grid size is also rounded up to a multiple of `swizzle_size * cluster_shape`, ensuring that each swizzle block is complete.

---

## 4. Persistent Kernel and Tile Scheduling

### 4.1 Why Persistent Kernels Are Needed

In a standard GEMM launch, the kernel launches M_tiles * N_tiles CTAs, and the GPU hardware scheduler decides which CTA runs on which SM. The problem:The hardware scheduler typically assigns CTA IDs in linear order (SM 0 gets CTA 0, 1, ..., SM 1 gets CTA k, k+1, ...), but this does not mean that spatially adjacent tiles are processed simultaneously.

**Persistent Kernel Solution**: Launch exactly N_SM CTAs (or fewer), with each CTA processing multiple tiles. The CTA itself decides which tile to process.

```
Persistent Kernel:
gridDim = (num_SMs, 1, 1)   # e.g., 114 CTAs (H100 has 114 SMs)

  Each CTA:
      tile_idx = atomicAdd(&global_counter, 1)   # Atomic increment to get next tile
      (m, n) = swizzle(tile_idx)                 # Apply swizzle mapping
```

### 4.2 Tile Scheduling Flow

```
                    +-------------------+
                    | Global linear counter     |
                    | atomic_counter = 0 |
                    +--------+----------+
                             |
              +--------------+--------------+
              |              |              |
         CTA 0          CTA 1          CTA 2        ...
              |              |              |
     claim tile 0    claim tile 1    claim tile 2
              |              |              |
      swizzle(0)     swizzle(1)     swizzle(2)
     → (m=0,n=0)    → (m=0,n=1)    → (m=0,n=2)
              |              |              |
     compute(0,0)   compute(0,1)   compute(0,2)
              |              |              |
     claim tile 3    claim tile 4    claim tile 5
      swizzle(3)     swizzle(4)     swizzle(5)
     → (m=0,n=3)    → (m=1,n=0)    → (m=1,n=1)
              |              |              |
            ...            ...            ...
```

### 4.3 Key Implementation Points for advance_to_next_work

```python
"""Get next tile to process"""
    # Atomic increment global counter

    # Check if exceeding total tile count
        return None  # No more work

    # Handle batch dimension

    # Apply swizzle mapping to (m, n)
```

On Hopper and Blackwell, the tile scheduler is also coupled with the pipeline: while fetching the next tile, TMA data prefetching is initiated to hide data loading latency.

### 4.4 Stream-K Partitioning

Stream-K is a more advanced tile scheduling strategy that also partitions the K dimension:

```
Standard partition: Each CTA handles complete K dimension
  CTA 0: C[0,0] = sum_k(A[0,:] * B[:,0])

Stream-K partition: Multiple CTAs share same tile's K dimension
  CTA 0: partial_0 = sum_k0..k1(A[0,:] * B[:,0])
  CTA 1: partial_1 = sum_k1..k2(A[0,:] * B[:,0])
  C[0,0] = partial_0 + partial_1  (requires synchronous reduction)
```

Stream-K improves SM utilization (when the number of tiles is not an integer multiple of the number of SMs), but increases reduction overhead and complexity. It is orthogonal to swizzling — swizzling controls the traversal order of the MxN plane, while Stream-K controls the partitioning of the K dimension.

---

## 5. Performance Impact and Metrics

### 5.1 NCU Profiling Metrics

| Metric | Meaning | Expected Value |
|------|------|-------|
| `lts__t_sectors_srcunit_tex_op_read` | Number of sectors read from L2 | Should decrease after swizzling |
| `lts__t_sector_hit_rate` | L2 hit rate | Should increase after swizzling (> 80%) |
| `dram__bytes_read` | HBM read bytes | Should approach theoretical minimum after swizzling |
| `smsp__inst_executed_pipe_lsu` | Number of load/store instructions | Should not change (swizzling does not change instruction count) |

### 5.2 Theoretical Analysis

For a GEMM with M=N=8192, K=4096, tile=128x128, FP16:

```
Tile grid: 64 x 64 = 4096 tiles

Each tile needs to read:
  A strip: 128 * 4096 * 2 = 1 MB
  B strip: 4096 * 128 * 2 = 1 MB

No reuse (worst case):
  Total HBM read = 4096 * 2 MB = 8 GB

Perfect row-major reuse:
  A: Read each row once, 64 rows * 1 MB = 64 MB
  B: Process each new row, B fully reloaded: 64 * 64 MB = 4 GB
  Total HBM read = 64 MB + 4 GB ≈ 4 GB

Swizzle (size=8) reuse:
  A: Every 8 rows as a group, A reused 8 times within group: 64 MB
  B: Every 8 columns as a group, B reused 8 times within group: 64 MB * (64/8) = 512 MB
  Total HBM read ≈ 64 MB + 512 MB = 576 MB

Theoretical minimum (A+B each read once):
  A: 8192 * 4096 * 2 = 64 MB
  B: 4096 * 8192 * 2 = 64 MB
  Total = 128 MB
```

Swizzle reduces HBM reads from 4 GB to 576 MB, nearly a 4.5x reduction. In memory-bound scenarios, this directly translates into performance improvement.

### 5.3 Measured Empirical Values

| Problem Size | No Swizzle | Swizzle=4 | Swizzle=8 | Improvement |
|--------------|------------|-----------|-----------|-------------|
| M=N=2048, K=2048 | Baseline | +5% | +5% | Small, as the problem nearly fits L2 |
| M=N=4096, K=4096 | Baseline | +15% | +18% | Moderate |
| M=N=8192, K=8192 | Baseline | +22% | +28% | Significant |
| M=16384, N=1024 | Baseline | +8% | +8% | Average (non-square shape, limited swizzle benefit) |

---

## 6. Beyond GEMM

### 6.1 Convolution

Convolution after im2col is equivalent to GEMM Facts apply directly. In implicit convolution (implicit GEMM), the spatial dimension (H*W) corresponds to the M dimensionestic and output channels correspond to the N dimensionestic. Swizzle is equally effective.

### 6.2 Batched Operations

In batched GEMM, the batch dimension provides additional L2 reuse opportunities:

```
Traversal order selection:
  Option A: batch priority → Traverse all tiles within same batch → Good reuse when batch is small
  Option B: tile priority → Same tile across all batches → Good for weight sharing (inference scenario)
```

Production code typically places batch in the outermost position (Option A), because the A/B matrices of different batches are completely different and cannot be reused across batches.

### 6.3 Multi-GPU

In data parallelism, each GPU processes a different batch—this does not affect tile rasterization within a single GPU.

In tensor parallelism, the matrix is partitioned:
```
GPU 0 processes C[:, 0:N/2]    # Only needs B[:, 0:N/2]
GPU 1 processes C[:, N/2:N]    # Only needs B[:, N/2:N]
```
After partitioning, each GPU's subproblem is smaller and may already fit L2, in which case the swizzle benefit is reduced.

---

## 7. Practical Decision Guide

### 7.1 Rules of Thumb

1. **Enable swizzle by default**: Virtually no downside (only adds a few integer arithmetic instructions)
2. **max_swizzle_size = 8 is a safe default**: The library will automatically degrade based on the actual number of tiles
3. **Use a heuristic to select rasterization direction**: Use AlongM when tiles_n > tiles_m, otherwise AlongN
4. **Persistent kernel is a prerequisite for swizzle**: Without a persistent kernel, CTA scheduling is not controllable

### 7.2 When to Apply

- When developing large-scale GEMM kernels (M*N > 4096*4096)
- When profiling reveals L2 hit rate below 60%
- When HBM bandwidth is saturated but compute utilization is not high
- When migrating from a non-persistent kernel to a persistent kernel

### 7.3 Debugging Checklist

```
If L2 hit rate is low:
  1. Check if persistent kernel is used (non-persistent cannot control swizzle)
  2. Check if swizzle size fits current problem scale
  3. Use NCU to compare L2 hit rate for raster_order = AlongM vs AlongN
  4. Check if other kernels run concurrently, competing for L2

If swizzle brings no improvement:
  1. Problem may already fit L2 (check A+B total size vs L2 capacity)
  2. Kernel may be compute-bound rather than memory-bound
  3. Tile size may be too large, each tile's strip already exceeds L2 capacity
```

---

## Further Reading

- [L2 Cache Persistence](../../../kernel-opt/nvidia/common/l2-cache-persistence.md) — L2 set-aside strategy, complementary to swizzle
- [Occupancy Tuning](../../../kernel-opt/nvidia/common/occupancy-tuning-by-arch.md) — CTA count selection for persistent kernels
- [Thread Block Cluster](../../../kernel-opt/nvidia/common/thread-block-cluster.md) — Cluster shape affects swizzle granularity
- NVIDIA CUTLASS `tile_scheduler_params.h` — Production-grade implementation of swizzle and raster order
- NVIDIA CUTLASS `sm90_tile_scheduler.hpp` — Hopper persistent tile scheduler
