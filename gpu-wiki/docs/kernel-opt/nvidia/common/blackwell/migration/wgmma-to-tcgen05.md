# Migrating from wgmma to tcgen05

## Overview

This guide covers the migration from Hopper's `wgmma.mma_async` (SM90) to Blackwell's `tcgen05.mma` (SM100). This is the single most impactful change when porting kernels from H100/H200 to B200. The two instructions differ in:

- **Issuing model**: warpgroup (128 threads) vs single thread
- **Accumulator storage**: registers vs TMEM
- **Operand loading**: ldmatrix + registers vs direct SMEM access
- **Synchronization**: warpgroup barriers vs async fences
- **Available data types**: FP8 (Hopper) vs FP4/FP6/FP8 with block scaling (Blackwell)

## Migration Checklist

```
[ ] Replace wgmma.mma_async with tcgen05.mma
[ ] Move accumulators from registers to TMEM (alloc/dealloc)
[ ] Remove ldmatrix operations (tcgen05 reads directly from SMEM)
[ ] Change SMEM swizzle from 64B to 128B
[ ] Replace warpgroup commit/wait with tcgen05 fences
[ ] Update warp specialization roles (fewer warps needed for MMA)
[ ] Update tile sizes (128xN -> consider 256xN with 2-SM)
[ ] Add TMEM lifecycle management (alloc at start, dealloc at end)
[ ] Update epilogue to read from TMEM instead of registers
```

## Side-by-Side Comparison

### Hopper Kernel Structure (SM90)

```cuda
// SM90 GEMM kernel using wgmma
__global__ void hopper_gemm(
    const half* A, const half* B, half* C,
    int M, int N, int K
) {
    extern __shared__ char smem[];
    half* smem_a = reinterpret_cast<half*>(smem);
    half* smem_b = smem_a + TILE_M * TILE_K;

    // Register accumulators -- HIGH REGISTER PRESSURE
    float acc[MMA_M_FRAGS][MMA_N_FRAGS];  // 128+ registers!
    for (int i = 0; i < MMA_M_FRAGS; ++i)
        for (int j = 0; j < MMA_N_FRAGS; ++j)
            acc[i][j] = 0.0f;

    // Mainloop
    for (int k = 0; k < K / TILE_K; ++k) {
        // Load A to SMEM (TMA or cp.async)
        tma_load(smem_a, A, tile_m, k);
        tma_load(smem_b, B, k, tile_n);

        // Load A fragment from SMEM to registers via ldmatrix
        uint32_t a_frag[4];
        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
            "{%0,%1,%2,%3}, [%4];"
            : "=r"(a_frag[0]), "=r"(a_frag[1]),
              "=r"(a_frag[2]), "=r"(a_frag[3])
            : "r"(smem_a_addr)
        );

        // Issue wgmma -- ALL 128 THREADS IN WARPGROUP participate
        asm volatile(
            "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 "
            "{%0, %1, %2, %3, %4, %5, %6, %7, ...}, "  // acc registers
            "{%N, %N+1, %N+2, %N+3}, "                   // A in registers
            "desc_b, ..."                                  // B descriptor
            : "+f"(acc[0][0]), "+f"(acc[0][1]), ...
            : "r"(a_frag[0]), ...
        );

        // Commit and wait for warpgroup
        asm volatile("wgmma.commit_group.sync.aligned;");
        asm volatile("wgmma.wait_group.sync.aligned 0;");
    }

    // Epilogue: accumulators already in registers
    // Write directly to GMEM (or through SMEM for vectorized stores)
    for (int i = 0; i < MMA_M_FRAGS; ++i)
        for (int j = 0; j < MMA_N_FRAGS; ++j)
            C[row + i][col + j] = (half)acc[i][j];
}
```

### Blackwell Kernel Structure (SM100)

```cuda
// SM100 GEMM kernel using tcgen05
__global__ void blackwell_gemm(
    const half* A, const half* B, half* C,
    int M, int N, int K
) {
    extern __shared__ char smem[];
    half* smem_a = reinterpret_cast<half*>(smem);
    half* smem_b = smem_a + TILE_M * TILE_K;

    // TMEM accumulator -- NO REGISTER PRESSURE
    uint32_t tmem_acc;
    if (threadIdx.x == 0) {
        asm volatile(
            "tcgen05.alloc.cta_group::1.sync.aligned.b32 %0, %1;"
            : "=r"(tmem_acc)
            : "r"(256)  // 256 columns for 128x256 tile
        );
    }
    tmem_acc = __shfl_sync(0xFFFFFFFF, tmem_acc, 0);

    // Zero TMEM accumulator
    tmem_zero(tmem_acc, 256);

    // Mainloop
    for (int k = 0; k < K / TILE_K; ++k) {
        // TMA load with 128B swizzle (mandatory for tcgen05)
        tma_load_128b_swizzle(smem_a, A, tile_m, k);
        tma_load_128b_swizzle(smem_b, B, k, tile_n);
        wait_barrier();

        // NO ldmatrix -- tcgen05 reads directly from SMEM
        // SINGLE THREAD issues MMA (not warpgroup)
        if (threadIdx.x == 0) {
            asm volatile(
                "tcgen05.mma.cta_group::1.kind::f16 "
                "[%0], %1, %2, %3, 1;"
                :
                : "r"(tmem_acc),
                  "l"(make_smem_desc_128b(smem_a)),
                  "l"(make_smem_desc_128b(smem_b)),
                  "r"(0)
            );
        }
        // NO commit/wait -- fully async, fence-based
    }

    // Fence before reading TMEM (replaces wgmma.wait_group)
    asm volatile("tcgen05.mma.fence::before_thread_sync;");
    __syncthreads();

    // Epilogue: read from TMEM, then write to GMEM
    float4 acc_vals = tmem_load_f32x4(tmem_acc + col_offset);
    // Apply bias, activation, etc.
    store_output(acc_vals, C, row, col);

    // Deallocate TMEM (no equivalent needed on Hopper)
    if (threadIdx.x == 0) {
        asm volatile(
            "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
            :
            : "r"(tmem_acc), "r"(256)
        );
    }
}
```

## Step-by-Step Migration

### Step 1: Replace Accumulator Storage

**Before (Hopper):** Accumulators live in registers.

```cuda
// Hopper: 128 FP32 registers for a 64x256 accumulator fragment
float acc[4][32];  // Per-thread fragment of the warpgroup accumulator
```

**After (Blackwell):** Accumulators live in TMEM.

```cuda
// Blackwell: TMEM address replaces register array
uint32_t tmem_acc = tmem_alloc(256);  // 128 rows x 256 cols in TMEM
```

### Step 2: Remove ldmatrix

**Before (Hopper):** Load A operand from SMEM to registers.

```cuda
// Hopper: ldmatrix loads 8x8 matrix fragments into registers
uint32_t a_frag[4];
asm volatile(
    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
    "{%0,%1,%2,%3}, [%4];"
    : "=r"(a_frag[0]), "=r"(a_frag[1]),
      "=r"(a_frag[2]), "=r"(a_frag[3])
    : "r"(smem_addr)
);
```

**After (Blackwell):** No equivalent needed. tcgen05 reads A directly from SMEM via descriptor.

```cuda
// Blackwell: just construct the SMEM descriptor
uint64_t desc_a = make_smem_desc_128b(smem_a_ptr);
// Pass desc_a directly to tcgen05.mma -- no register staging
```

### Step 3: Change SMEM Swizzle from 64B to 128B

**Before (Hopper):** 64-byte or 128-byte swizzle both work for wgmma.

```cuda
// Hopper TMA descriptor: 64B swizzle is common
CUtensorMap desc = create_tma_desc(ptr, M, N, tile_m, tile_n,
                                    CU_TENSOR_MAP_SWIZZLE_64B);
```

**After (Blackwell):** 128-byte swizzle is mandatory.

```cuda
// Blackwell TMA descriptor: MUST use 128B swizzle
CUtensorMap desc = create_tma_desc(ptr, M, N, tile_m, tile_n,
                                    CU_TENSOR_MAP_SWIZZLE_128B);
```

### Step 4: Replace wgmma Issue with tcgen05 Issue

**Before (Hopper):**

```cuda
// Hopper: all 128 threads in warpgroup issue wgmma
asm volatile(
    "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 "
    "{%0,%1,...}, {%N,...}, desc_b, ...;"
    : "+f"(acc[0]), "+f"(acc[1]), ...
    : "r"(a_frag[0]), ...
);
asm volatile("wgmma.commit_group.sync.aligned;");
```

**After (Blackwell):**

```cuda
// Blackwell: single thread issues tcgen05
if (threadIdx.x == 0) {
    asm volatile(
        "tcgen05.mma.cta_group::1.kind::f16 "
        "[%0], %1, %2, %3, 1;"
        :
        : "r"(tmem_acc), "l"(desc_a), "l"(desc_b), "r"(0)
    );
}
// No commit needed -- fully async
```

### Step 5: Replace wgmma.wait_group with tcgen05 Fence

**Before (Hopper):**

```cuda
// Hopper: wait for outstanding wgmma groups
asm volatile("wgmma.wait_group.sync.aligned 0;");
// Accumulators now ready in registers
```

**After (Blackwell):**

```cuda
// Blackwell: fence before reading TMEM
asm volatile("tcgen05.mma.fence::before_thread_sync;");
__syncthreads();
// TMEM accumulators now ready for reading
```

### Step 6: Update Epilogue

**Before (Hopper):**

```cuda
// Hopper: accumulators in registers, directly usable
float result = acc[frag_m][frag_n];
// Apply bias
result += bias[col];
// Apply activation
result = relu(result);
// Store to GMEM
C[row * N + col] = (half)result;
```

**After (Blackwell):**

```cuda
// Blackwell: must read from TMEM first
float result = tmem_load_f32(tmem_acc + col_offset);
// Apply bias
result += bias[col];
// Apply activation
result = relu(result);
// Store to GMEM
C[row * N + col] = (half)result;
```

## Warp Specialization Changes

### Hopper Warp Specialization (Typical)

```
Warp 0-3:  Warpgroup 0 -- MMA producer (all 128 threads issue wgmma)
Warp 4-7:  Warpgroup 1 -- MMA producer (backup/double-buffer)
Warp 8-11: Data movement (TMA loads, SMEM management)
Warp 12:   Tile scheduler

Total: 13+ warps, 416+ threads
```

### Blackwell Warp Specialization (Typical)

```
Warp 0:    MMA producer (single thread issues tcgen05)
Warp 1-2:  TMA data movement (load A, load B, manage barriers)
Warp 3:    Epilogue (read TMEM, apply post-ops, store to GMEM)
Warp 4:    Tile scheduler / CLC management

Total: 5 warps, 160 threads (can be fewer)
```

With tcgen05, far fewer warps are dedicated to MMA because only one thread is needed to issue the instruction. The freed warps can be repurposed for:
- More aggressive data prefetching
- Overlapped epilogue (read TMEM while next tile's MMA is computing)
- Softmax or other reductions (FlashAttention-style)

## Tile Size Migration

| Hopper | Blackwell (1-SM) | Blackwell (2-SM) |
|---|---|---|
| m64 x n256 x k16 | m128 x n256 x k16 | m256 x n256 x k16 |
| m64 x n128 x k16 | m128 x n128 x k16 | m256 x n128 x k16 |

Blackwell's base MMA tile is 2x larger in M (128 vs 64). When migrating:
- CTA tile size of 128x256 maps naturally to a single tcgen05 MMA
- CTA tile size of 64x256 on Hopper should be doubled to 128x256
- For 2-SM mode, consider 256x256 tiles

## Common Migration Pitfalls

1. **Forgetting 128B swizzle**: The most common silent-failure bug. wgmma works with 64B swizzle; tcgen05 does not. Results will be numerically wrong but the kernel won't crash.

2. **Not deallocating TMEM**: On Hopper, register accumulators are freed implicitly when the CTA exits. On Blackwell, TMEM must be explicitly deallocated in persistent kernels.

3. **Synchronization model mismatch**: Replacing `wgmma.wait_group` with `__syncthreads()` alone is insufficient. The `tcgen05.mma.fence::before_thread_sync` must precede the syncthreads.

4. **Over-allocating threads**: On Hopper, you need 128 threads per warpgroup for MMA. Blindly keeping the same thread count on Blackwell wastes resources since only 1 thread issues tcgen05.

5. **Register pressure assumptions**: Code tuned for Hopper's high register pressure (e.g., reduced tile sizes, manual spilling) may be overly conservative on Blackwell. Re-tune tile sizes to take advantage of freed registers.

## CUTLASS Migration

If using CUTLASS, the migration is largely handled by changing the arch tag and kernel schedule:

```cpp
// Hopper CUTLASS GEMM
using GemmHopper = cutlass::gemm::device::GemmUniversal<
    /* ... */
    cutlass::arch::Sm90,
    /* ... */
    cutlass::gemm::collective::KernelScheduleSm90CpAsyncWarpSpecialized
>;

// Blackwell CUTLASS GEMM -- change arch + schedule
using GemmBlackwell = cutlass::gemm::device::GemmUniversal<
    /* ... */
    cutlass::arch::Sm100,   // <-- changed
    /* ... */
    cutlass::gemm::collective::KernelScheduleSm100CpAsyncWarpSpecialized  // <-- changed
>;
```

CUTLASS handles the internal differences (TMEM allocation, descriptor construction, fence insertion, 128B swizzle) automatically through its `MMA_Atom` and `MMA_Traits` abstractions for SM100.
