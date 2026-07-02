# Vectorized Loads and Cache Policy (Blackwell)


**Last updated**: 2026-07-01

## Overview

For memory-bound kernels (low arithmetic intensity), maximizing global memory throughput is critical. Three complementary techniques from the GPU Mode NVFP4 Hackathon achieve this: (1) wide vectorized loads (128-bit and 256-bit) to saturate memory bandwidth per thread, (2) differentiated L1 cache policies to keep reused data hot while bypassing the cache for streaming data, and (3) register budgeting via `-maxrregcount` to increase occupancy. These techniques reduced NVFP4 GEMV latency from 2000us to 22.4us (89x improvement).

## Wide Vectorized Loads

Standard 32-bit loads waste memory bus bandwidth. Wider loads amortize the instruction overhead and saturate the 8 TB/s HBM bandwidth of B200:

```cuda
// Vectorized load widths comparison for FP4 GEMV
// Each thread loads more data per instruction

// 32-bit load: 4 bytes per thread per instruction
float val;
asm volatile("ld.global.b32 %0, [%1];" : "=f"(val) : "l"(ptr));

// 64-bit load: 8 bytes per thread per instruction
uint2 val64;
asm volatile("ld.global.v2.u32 {%0,%1}, [%2];"
    : "=r"(val64.x), "=r"(val64.y) : "l"(ptr));

// 128-bit load: 16 bytes per thread per instruction (preferred)
uint4 val128;
asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];"
    : "=r"(val128.x), "=r"(val128.y), "=r"(val128.z), "=r"(val128.w)
    : "l"(ptr));

// 256-bit load: 32 bytes per thread per instruction
// Requires v4.u64 (4 x 64-bit)
uint64_t v256[4];
asm volatile("ld.global.v4.u64 {%0,%1,%2,%3}, [%4];"
    : "=l"(v256[0]), "=l"(v256[1]), "=l"(v256[2]), "=l"(v256[3])
    : "l"(ptr));
```

For the NVFP4 GEMV, 128-bit and 256-bit loads are essential because FP4 elements are only 0.5 bytes each. A 256-bit load fetches 64 FP4 values in a single instruction:

```cuda
// NVFP4 GEMV: each thread loads 64 FP4 values via 256-bit load
// Then unpacks using PTX byte manipulation instead of bitwise ops
__device__ void load_and_unpack_nvfp4_256bit(
    const uint8_t* fp4_data,  // Packed FP4 data (2 values per byte)
    float* unpacked,          // Output: 64 FP32 values
    int offset)
{
    // 256-bit load: 32 bytes = 64 FP4 values
    uint64_t raw[4];
    const uint64_t* ptr = reinterpret_cast<const uint64_t*>(fp4_data + offset);
    asm volatile(
        "ld.global.v4.u64 {%0,%1,%2,%3}, [%4];"
        : "=l"(raw[0]), "=l"(raw[1]), "=l"(raw[2]), "=l"(raw[3])
        : "l"(ptr)
    );

    // Unpack using PTX mov.b32 byte extraction
    // This avoids the bitwise shift-and-mask overhead
    for (int i = 0; i < 4; i++) {
        uint32_t lo = (uint32_t)(raw[i]);
        uint32_t hi = (uint32_t)(raw[i] >> 32);

        // PTX byte unpack: extract individual bytes from 32-bit word
        uint32_t b0, b1, b2, b3;
        asm volatile("mov.b32 {%0,%1,%2,%3}, %4;"
            : "=r"(b0), "=r"(b1), "=r"(b2), "=r"(b3) : "r"(lo));

        // Each byte contains 2 FP4 values -- decode low word (8 values)
        for (int b = 0; b < 4; b++) {
            uint32_t byte_val = (b == 0) ? b0 : (b == 1) ? b1 : (b == 2) ? b2 : b3;
            unpacked[i * 16 + b * 2]     = decode_fp4(byte_val & 0xF);
            unpacked[i * 16 + b * 2 + 1] = decode_fp4((byte_val >> 4) & 0xF);
        }

        // Unpack high word (next 8 values from same 64-bit element)
        uint32_t hb0, hb1, hb2, hb3;
        asm volatile("mov.b32 {%0,%1,%2,%3}, %4;"
            : "=r"(hb0), "=r"(hb1), "=r"(hb2), "=r"(hb3) : "r"(hi));
        for (int b = 0; b < 4; b++) {
            uint32_t byte_val = (b == 0) ? hb0 : (b == 1) ? hb1 : (b == 2) ? hb2 : hb3;
            unpacked[i * 16 + 8 + b * 2]     = decode_fp4(byte_val & 0xF);
            unpacked[i * 16 + 8 + b * 2 + 1] = decode_fp4((byte_val >> 4) & 0xF);
        }
    }
}
```

## L1 Cache Policy Differentiation

Different data streams have different reuse patterns. Applying the correct cache policy per stream avoids L1 pollution:

```cuda
// Cache policy selection based on data reuse pattern
//
// Matrix A (streamed, each row used once): bypass L1
// Vector B (reused across all rows):      keep in L1

// L1::no_allocate -- data bypasses L1 cache (streaming access)
// Used for matrix A which is read once per GEMV
asm volatile(
    "ld.global.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
    : "=r"(a.x), "=r"(a.y), "=r"(a.z), "=r"(a.w)
    : "l"(matrix_a_ptr)
);

// L1::evict_last -- data stays in L1 as long as possible
// Used for vector B which is reused across all M rows
asm volatile(
    "ld.global.L1::evict_last.v4.u32 {%0,%1,%2,%3}, [%4];"
    : "=r"(b.x), "=r"(b.y), "=r"(b.z), "=r"(b.w)
    : "l"(vector_b_ptr)
);
```

The impact of cache policies from the GPU Mode Hackathon:

```
No cache policy differentiation:  39 us
A: L1::no_allocate, B: L1::evict_last: 27 us  (1.44x faster)
```

The full set of PTX load cache qualifiers:

```ptx
// PTX load qualifiers for cache control

// Default: normal L1 and L2 caching
ld.global.b32          %r, [%addr];

// L1 bypass: skip L1, still cached in L2
ld.global.L1::no_allocate.b32  %r, [%addr];

// L1 keep: prioritize keeping in L1 (evict last)
ld.global.L1::evict_last.b32   %r, [%addr];

// L2 promotion hint (256-byte sector)
// Used by DeepEP for communication overlap
ld.global.nc.L1::no_allocate.L2::256B.b32  %r, [%addr];

// Non-coherent read-only (nc): uses texture path
// Avoids coherence traffic, useful for read-only data
ld.global.nc.b32       %r, [%addr];
```

## Register Budgeting (-maxrregcount)

For memory-bound kernels, occupancy (number of concurrent warps) matters more than per-thread register count. Limiting registers per thread allows more warps to be resident:

```cuda
// Compile-time register budgeting
// Lower register count -> higher occupancy -> better latency hiding

// Problem 1 winner (rank 1): -maxrregcount=32
// This allows 64 warps per SM (100% occupancy on SM100)
// Sufficient for GEMV where each thread does minimal computation

// Problem 1 rank 3: -maxrregcount=45
// Allows 44 warps per SM (~69% occupancy)
// More registers per thread for wider vectorized accumulation
```

The tradeoff in a build system:

```python
# nvcc compilation with register budgeting
# In CMakeLists.txt or build script:

# For memory-bound GEMV kernel:
# nvcc -maxrregcount=32 -arch=sm_100a gemv_kernel.cu -o gemv_kernel

# For compute-bound GEMM kernel:
# nvcc -arch=sm_100a gemm_kernel.cu -o gemm_kernel  (no limit; needs ~128+ regs)

# Per-kernel register limits in the same translation unit:
# Use __launch_bounds__ to control per-kernel

# Memory-bound: maximize occupancy
__global__ void __launch_bounds__(256, 8)  // 256 threads, min 8 blocks/SM
gemv_kernel(/* ... */) { /* ... */ }

# Compute-bound: maximize register availability
__global__ void __launch_bounds__(512, 1)  // 512 threads, min 1 block/SM
gemm_kernel(/* ... */) { /* ... */ }
```

## Complete NVFP4 GEMV Example

Combining all three techniques for the GPU Mode Hackathon Problem 1:

```cuda
// Optimized NVFP4 Batched GEMV
// A: [M, K] NVFP4, B: [1, K] NVFP4, C: [M, 1] FP16
// Memory-bound: maximize bandwidth utilization

// NVFP4 Batched GEMV: each row processed by THREADS_PER_ROW threads
// Memory-bound: maximize bandwidth with wide loads and cache policies

#define BLOCK_M 4       // Rows per thread block
#define THREADS 256
#define THREADS_PER_ROW (THREADS / BLOCK_M)  // 64 threads per row

__global__ void __launch_bounds__(THREADS, 4)
nvfp4_gemv_optimized(
    const uint8_t* __restrict__ A,      // [M, K/2] packed FP4
    const uint8_t* __restrict__ B,      // [1, K/2] packed FP4
    const fp8_e4m3* __restrict__ sfa,   // [M, K/16] block scales
    const fp8_e4m3* __restrict__ sfb,   // [1, K/16] block scales
    half* __restrict__ C,               // [M, 1] output
    float global_scale_a, float global_scale_b,
    int M, int K)
{
    // Map threads to rows: 64 threads per row, 4 rows per block
    int local_row = threadIdx.x / THREADS_PER_ROW;      // 0..3
    int thread_in_row = threadIdx.x % THREADS_PER_ROW;  // 0..63
    int row = blockIdx.x * BLOCK_M + local_row;
    if (row >= M) return;

    float acc = 0.0f;

    // Each of the 64 threads handles a distinct K-chunk (no duplication)
    // 64 elements per load × 64 threads = 4096 K elements per iteration
    for (int k = thread_in_row * 64; k < K; k += THREADS_PER_ROW * 64) {
        // Load B (reused across rows): L1::evict_last, 256-bit
        uint64_t b_raw[4];
        asm volatile(
            "ld.global.L1::evict_last.v4.u64 {%0,%1,%2,%3}, [%4];"
            : "=l"(b_raw[0]), "=l"(b_raw[1]), "=l"(b_raw[2]), "=l"(b_raw[3])
            : "l"((const uint64_t*)(B + k / 2)));

        // Load A (streamed once): L1::no_allocate, 256-bit
        uint64_t a_raw[4];
        asm volatile(
            "ld.global.L1::no_allocate.v4.u64 {%0,%1,%2,%3}, [%4];"
            : "=l"(a_raw[0]), "=l"(a_raw[1]), "=l"(a_raw[2]), "=l"(a_raw[3])
            : "l"((const uint64_t*)(A + row * (K / 2) + k / 2)));

        // Unpack FP4, apply block scale, dot-product
        for (int i = 0; i < 64; i++) {
            float a_val = unpack_fp4(a_raw, i) * get_block_scale(sfa, row, k + i);
            float b_val = unpack_fp4(b_raw, i) * get_block_scale(sfb, 0, k + i);
            acc += a_val * b_val;
        }
    }

    // Two-phase reduction: first within each warp, then across the 2 warps per row
    // Phase 1: warp-level reduction (32 threads → 1 partial sum per warp)
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_xor_sync(0xFFFFFFFF, acc, offset);
    }

    // Phase 2: shared memory reduction across the 2 warps assigned to this row
    __shared__ float smem_reduce[BLOCK_M * 2];  // 2 warp partials per row
    int warp_in_row = thread_in_row / 32;  // 0 or 1
    int lane = thread_in_row % 32;
    if (lane == 0) {
        smem_reduce[local_row * 2 + warp_in_row] = acc;
    }
    __syncthreads();

    // Final sum and store (one thread per row)
    if (thread_in_row == 0) {
        float result = smem_reduce[local_row * 2] + smem_reduce[local_row * 2 + 1];
        C[row] = __float2half(result * global_scale_a * global_scale_b);
    }
}
```

## Optimization Progression (GPU Mode Hackathon Problem 1)

| Step | Technique | Latency | Speedup |
|------|-----------|---------|---------|
| Baseline | Naive C++ | 2000 us | 1.0x |
| Coalesced access | Memory layout fix | 443 us | 4.5x |
| Hardware intrinsics | FP4 decode | 39 us | 51x |
| PTX assembly | Vectorized loads + cache policy | 27 us | 74x |
| ILP + register tuning | Unrolling + maxrregcount | 22.4 us | 89x |
| Speed of light | | ~8.6 us | 233x |

## When to Use

- **GEMV and memory-bound kernels**: Vectorized loads and cache policies are essential. These kernels are entirely limited by memory bandwidth.
- **FP4/FP8 kernels**: Sub-byte data types make wide loads even more impactful since more elements fit in a single wide load.
- **Decode-phase inference**: Single-token GEMV during autoregressive decoding is always memory-bound.

## Caveats

- 256-bit loads require 32-byte aligned addresses. Misaligned access falls back to multiple narrower transactions.
- `L1::no_allocate` is harmful for data that will be reused. Only apply it to truly streaming access patterns.
- `-maxrregcount` that is too low causes register spilling to local memory, which is slower than the occupancy gain. Profile with Nsight Compute to find the optimal point.
- PTX inline assembly bypasses the compiler's register allocator. Excessive inline PTX can interfere with compiler optimizations for surrounding code.


## Related

- [Cache Policy Differentiation](cache-policy.md)
- [Chunk-Based Parallelism](chunk-parallelism.md)
- [CUDA GEMM Optimization Ladder](cuda-gemm-optimization-ladder.md)
- [Double/Multi-Buffering Patterns](double-buffering.md)
- [Epilogue Fusion](epilogue-fusion.md)
- [Composable Kernel (CK) Architecture Overview](../../../amd/common/ck-architecture-overview.md)
