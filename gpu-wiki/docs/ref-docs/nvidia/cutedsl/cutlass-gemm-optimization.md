# CUTLASS GEMM Optimization Strategy

## Three-Layer Tiling Structure

CUTLASS maps GEMM to the GPU hardware hierarchy through three-layer tiling:

| Level | Concurrency Unit | Data Storage | Tile Shape |
|------|---------|---------|-----------|
| **Threadblock** | CTA-level parallelism | Global → Shared Memory | `ThreadblockShape::{kM, kN, kK}` |
| **Warp** | Warp-level parallelism | Shared → Register | `WarpShape::{kM, kN, kK}` |
| **Instruction** | Instruction-level parallelism | Register | `InstructionShape::{kM, kN, kK}` |

The inner loop over the K dimension forms the **GEMM mainloop**, where each iteration is a "stage".

---

## Building the GEMM Mainloop (CuTe Style)

### Step 1: Problem Definition and Global Tensor

CuTe convention: A = `(M,K)`, B = `(N,K)`, C = `(M,N)`. K is always in the second mode.

```cpp
auto prob_shape = make_shape(M, N, K);
Tensor mA = make_tensor(make_gmem_ptr(A), select<0,2>(prob_shape), dA);  // (M,K)
Tensor mB = make_tensor(make_gmem_ptr(B), select<1,2>(prob_shape), dB);  // (N,K)
Tensor mC = make_tensor(make_gmem_ptr(C), select<0,1>(prob_shape), dC);  // (M,N)
```

**M-major vs K-major** (replacing BLAS N/T notation):

| BLAS | A Layout | A Stride | B Layout | B Stride |
|------|--------|----------|--------|----------|
| NT | M-major | `(1, ldA)` | N-major | `(1, ldB)` |
| TN | K-major | `(ldA, 1)` | K-major | `(ldB, 1)` |

### Step 2: CTA Tiling

```cpp
auto cta_tiler = make_shape(Int<128>{}, Int<128>{}, Int<8>{});  // (BLK_M, BLK_N, BLK_K)

auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _); // _ keep K dimension
Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{});  // (BLK_M, BLK_K, k)
Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X, _1, _1>{});  // (BLK_N, BLK_K, k)
Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1,  X>{});  // (BLK_M, BLK_N)
```

In the `Step` parameter, `X` marks the skipped mode, and the `_` coordinate preserves the entire K dimension (producing a third mode `k` for the reduction loop).

`local_tile` = `zipped_divide` + coordinate slice.

### Step 3: Shared Memory Setup

```cpp
// Column-major (NT)
auto sA_layout = make_layout(make_shape(bM, bK));           // m-major
auto sB_layout = make_layout(make_shape(bN, bK));           // n-major

// English comment
__shared__ TA smemA[cosize_v<ASmemLayout>];
Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout);  // (BLK_M, BLK_K)
```

`cosize` = layout value domain size = number of elements to allocate. SMEM layout must be static.

---

## Copy Partitioning vs Math Partitioning

### Copy Partitioning (Data Movement)

Use the thread layout `tA` to partition the gmem/smem tensor:

```cpp
// 32x8 = 256 128x8 tile -> 4x1 tensor
auto tA = make_layout(make_shape(Int<32>{}, Int<8>{}));

Tensor tAgA = local_partition(gA, tA, threadIdx.x);  // (THR_M, THR_K, k)
Tensor tAsA = local_partition(sA, tA, threadIdx.x);  // (THR_M, THR_K)

copy(tAgA(_,_,k_tile), tAsA);  // gmem → smem
```

### Math Partitioning (Computation)

Use a **different** thread layout `tC` to partition the smem/register tensor:

```cpp
auto tC = make_layout(make_shape(Int<16>{}, Int<16>{})); // 16x16 = 256

// Step : tC M mode sA, N mode sB
Tensor tCsA = local_partition(sA, tC, threadIdx.x, Step<_1, X>{});  // (THR_M, BLK_K)
Tensor tCsB = local_partition(sB, tC, threadIdx.x, Step< X, _1>{});  // (THR_N, BLK_K)
Tensor tCgC = local_partition(gC, tC, threadIdx.x, Step<_1, _1>{});  // (THR_M, THR_N)
Tensor tCrC = make_tensor_like(tCgC); // register

gemm(tCsA, tCsB, tCrC);
```### Naming Convention

`tAgA` = Partition pattern `tA` applied to tensor `gA`. Using the same partition pattern for the same tensor (gmem/smem) ensures logical correspondence of elements.

---

## Advanced Partitioning: TiledCopy and TiledMMA

### TiledCopy

```cpp
TiledCopy copyA = make_tiled_copy(
 Copy_Atom<UniversalCopy<uint128_t>, TA>{}, // 128-bit
 Layout<Shape<_32, _8>>{}, // layout
 Layout<Shape<_4, _1>>{} //
);

ThrCopy thr_copy = copyA.get_slice(threadIdx.x);
Tensor tAgA = thr_copy.partition_S(gA);   // (CPY, CPY_M, CPY_K, k)
Tensor tAsA = thr_copy.partition_D(sA);   // (CPY, CPY_M, CPY_K)
cute::copy(copyA, tAgA, tAsA);
```

### TiledMMA

```cpp
TiledMMA mmaC = make_tiled_mma(UniversalFMA<TC,TA,TB>{},
                               Layout<Shape<_16,_16,_1>>{});

ThrMMA thr_mma = mmaC.get_slice(threadIdx.x);
Tensor tCsA = thr_mma.partition_A(sA);
Tensor tCsB = thr_mma.partition_B(sB);
Tensor tCrC = thr_mma.make_fragment_C(tCgC);
cute::gemm(mmaC, tCsA, tCsB, tCrC);
```

---

## Mainloop Implementation

```cpp
for (int k_tile = 0; k_tile < K_TILE_MAX; ++k_tile) {
    // 1. Copy: gmem → smem
    copy(tAgA(_,_,k_tile), tAsA);
    copy(tBgB(_,_,k_tile), tBsB);

    cp_async_fence();
    cp_async_wait<0>();
    __syncthreads();

    // 2. Compute: smem → register → accumulate
    gemm(tCsA, tCsB, tCrC);

    __syncthreads();
}
```

---

## SMEM Layout Optimization

### Padding to Eliminate Bank Conflict

```cpp
// m-major: stride = bM
auto sA = make_layout(make_shape(bM, bK));

// Padding: stride = bM + 1, bank access
auto sA = make_layout(make_shape(bM, bK),
                      make_stride(Int<1>{}, bM + Int<1>{}));
```

Simply modify the layout definition; the kernel code remains unchanged.

### Swizzle Modes

Swizzle eliminates bank conflicts through XOR transformations:

```
Swizzle(MBase, BBits, SShift)
result = lowbit XOR highbitbit
```

In CuTeDSL, `SmemLayoutAtomKind` provides predefined swizzle modes:
- `MN_SW32/64/128` — MN-major 32/64/128 byte swizzle
- `K_SW32/64/128` — K-major 32/64/128 byte swizzle

---

## Software Pipelining

### Double-Buffering Strategy

Since the accumulator occupies a large number of registers Site resulting in low occupancy, the GPU cannot hide latency through context switching. CUTLASS employs double-buffering at two levels:

1. **SMEM level**: Allocate two tiles; one serves the current computation while the other buffers the next global memory load
2. **Register level**: Allocate two fragments; one participates in the current MMA while the other pre-fetches data from SMEM for the next iteration

```
Iteration i:     [Load tile i+1 → SMEM_B] + [Compute tile i from SMEM_A]
Iteration i+1:   [Load tile i+2 → SMEM_A] + [Compute tile i+1 from SMEM_B]
```

### Multi-Stage Pipelining

Hopper+ architectures support more stages (3–7), achieving deeper pipeline overlap through mbarrier and TMA.

---

## Parallel Reduction

### Split-K (Across Threadblocks)

Distributes the K dimension across multiple threadblocks, requiring two kernels:

1. **Split GEMM**: Each threadblock computes a partition of K, similar to batched-strided GEMM
2. **Reduction kernel**: Sums the partial results from all partitions

```
m=128, n=128, k=4096, 16 partitions
-> 16 batches, k=256
-> reduction 16 128x128
```

Use case: M and N are small but K is large, unable to fully utilize all SMs.

### Sliced-K (Across Warps)

Within a threadblock, assigns `CtaTileK` to multiple warps, where each warp computes partial sums, followed by a small-scale reduction at the end.

---

## Threadblock Rasterization

Adjusts the mapping order of threadblocks to GEMM coordinates to maximize **L2 cache reuse**.

Maps consecutively launched threadblocks to compact 2D regions, increasing the probability that neighboring threadblocks share global memory data.

---

## Epilogue

Handles the layout conversion from register layout to global memory layout:

1. Threads exchange data through SMEM
2. Cooperatively write back to GMEM using striped access patterns
3. Apply linear scaling, ReLU, or custom elementwise operations

---

## Hopper Warp Specialization### Producer-Consumer Model

Thread blocks are divided into **producer** and **consumer** warp groups:

- **Producer**: Performs global → shared loading via TMA, waiting for the consumer to release empty buffers
- **Consumer**: Waits for filled buffers, initiates TensorCore MMA, and releases buffers upon completion

### Cooperative Design

Persistent threadblocks compute multiple output tiles to amortize launch overhead. Two consumer warp groups split tiles along the M dimension to reduce register pressure.

### Ping-Pong Design

Two consumer warp groups alternate working on **different output tiles**:

```
Consumer A: [Compute tile 0] [Epilogue tile 0] [Compute tile 2] [Epilogue tile 2]
Consumer B:                  [Compute tile 1]  [Epilogue tile 1] [Compute tile 3]
```

One consumer's epilogue overlaps with the other's compute, maximizing tensor core utilization. The Producer uses Ordered Sequence Barriers to alternately fill buffers.

---

## GETT: Tensor Contractions as Multimodal GEMM

CuTe's nested layout supports executing tensor contractions with an **unchanged GEMM kernel**:

```cpp
// standard GEMM: M scalar
auto M = m;
auto bM = Int<128>{};

// GETT: M
auto M = make_shape(m0, m1);
auto bM = Shape<_64, _2>{}; // m0 64, m1 2

// kernel ！
gemm_device<<<dimGrid, dimBlock>>>(prob_shape, cta_tiler, ...);
```

Simply modify the shape to multimodal, the stride to nested, and the tiler to multimodal—the device kernel itself is completely generic.
