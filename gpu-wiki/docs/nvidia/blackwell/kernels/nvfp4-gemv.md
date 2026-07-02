# NVFP4 Batched GEMV


**Last updated**: 2026-06-30

## Overview

NVFP4 Batched GEMV is a memory-bound kernel computing batched matrix-vector products with NVFP4 (E2M1) block-scaled inputs on B200 GPUs. Unlike compute-bound GEMM, GEMV is dominated by memory bandwidth utilization (each FP4 element is used only once in the dot product), making PTX-level memory access control, cache policy differentiation, and register budgeting the critical optimization levers.

This was Problem 1 of the GPU Mode NVFP4 Hackathon (Nov 2025). The theoretical speed-of-light is ~8.6us for the largest config, limited by B200's 8 TB/s HBM3e bandwidth. Top performers achieved ~22.4us (2.6x off SOL), reflecting the overhead of FP4 decoding and scale factor application.

## Problem Specification

```
Inputs:
  a: (M x K x L) NVFP4 packed matrix
  b: (1 x K x L) NVFP4 packed vector
  sfa: (M x K/16 x L) FP8 E4M3 scale factors for a
  sfb: (1 x K/16 x L) FP8 E4M3 scale factors for b
  sf_a_global, sf_b_global: FP32 per-tensor global scales

Output:
  c: (M x 1 x L) FP16

Computation per element:
  c[m][l] = sum_k( sf_a_global * sfa[m][k/16][l] * deq(a[m][k][l])
                  * sf_b_global * sfb[0][k/16][l] * deq(b[0][k][l]) )

Benchmark configs:
  Config 1: M=7168, K=16384, L=1
  Config 2: M=4096, K=7168,  L=8
  Config 3: M=7168, K=2048,  L=4
```

## Key Optimization: PTX-Level Control

Raw PTX provides critical performance gains over C intrinsics for FP4 decoding and memory access:

```asm
; FP4 to FP16 conversion: hardware instruction
; Converts two packed E2M1 values to a pair of FP16 values
cvt.rn.f16x2.e2m1x2 %result, %fp4_packed;

; Byte unpacking: PTX mov.b32 is faster than manual bitwise ops
; Instead of: val = (packed >> (i*4)) & 0xF  (multiple shifts + masks)
; Use: direct byte decomposition
mov.b32 {tmp0, tmp1, tmp2, tmp3}, %packed_word;

; This eliminates the shift-mask chain entirely
; Key insight: PTX byte unpacking leverages hardware byte decomposition
```

## Cache Policy Differentiation

Different data access patterns require different cache strategies:

```asm
; Matrix A: streamed once per row, never reused across thread blocks
; Bypass L1 to avoid polluting cache with one-shot data
ld.global.L1::no_allocate.v4.u64 {a0,a1,a2,a3}, [addr_a];

; Vector B: reused across all M rows in a thread block
; Keep hot in L1 for fast reuse
ld.global.L1::evict_last.v4.u64 {b0,b1,b2,b3}, [addr_b];

; Rank 1 solution: used different qualifiers depending on K-dimension variant
; K=16384 (large): aggressive L1 bypass for A
; K=2048 (small): different balance since B is smaller relative to cache
```

## Register Budgeting

Lower register counts force higher occupancy, which is critical for memory-bound kernels where latency hiding dominates:

```cpp
// Rank 1: aggressive register limit for maximum occupancy
// nvcc -maxrregcount=32
// Fewer registers -> more warps/SM -> better memory latency hiding

// Rank 3: slightly relaxed for more ILP
// nvcc -maxrregcount=45

// Launch bounds to complement register budgeting
__launch_bounds__(256, 4)  // 256 threads/block, 4+ blocks per SM
__global__ void nvfp4_gemv_kernel(...) {
    // ...
}

// The measurable difference between 32 and 45 registers shows
// that occupancy is the dominant factor for memory-bound kernels
```

## Per-K Specialization

Compile separate kernels per K dimension, each with full loop unrolling and tuned configs:

```cpp
// Each K variant compiled separately with optimal configuration
template <int K_SIZE, int BLOCK_M, int MAX_REG>
__global__ __launch_bounds__(THREADS, MIN_BLOCKS)
void nvfp4_gemv_specialized(
    const uint8_t* __restrict__ a,
    const uint8_t* __restrict__ b,
    const uint8_t* __restrict__ sfa,
    const uint8_t* __restrict__ sfb,
    half* __restrict__ c,
    float sf_a_global, float sf_b_global,
    int M
) {
    float acc = 0.0f;

    // Full unroll: compiler knows K_SIZE at compile time
    #pragma unroll
    for (int k = 0; k < K_SIZE / ELEMENTS_PER_LOAD; k++) {
        // Load FP4 packed data
        uint64_t a_packed = load_fp4(a, row, k);
        uint64_t b_packed = load_fp4(b, 0, k);

        // Dequantize via PTX
        half2 a_vals = cvt_e2m1x2(a_packed);
        half2 b_vals = cvt_e2m1x2(b_packed);

        // Apply block scales
        float sa = sfa[row * (K_SIZE/16) + k/2];
        float sb = sfb[k/2];

        // Accumulate
        acc += (float)a_vals.x * (float)b_vals.x * sa * sb;
        acc += (float)a_vals.y * (float)b_vals.y * sa * sb;
    }

    // Apply global scales and store
    c[row] = __float2half(acc * sf_a_global * sf_b_global);
}

// Dispatch: K=2048 uses BLOCK_M=8, maxreg=32
//           K=7168 uses BLOCK_M=4, maxreg=40
//           K=16384 uses BLOCK_M=2, maxreg=32
```

## Vectorized Loads

128-bit and 256-bit vector loads maximize bandwidth utilization:

```asm
; 128-bit vector load (16 bytes = 32 FP4 elements)
ld.global.v2.u64 {r0, r1}, [addr];

; 256-bit vector load (32 bytes = 64 FP4 elements)
ld.global.v4.u64 {r0, r1, r2, r3}, [addr];

; Only effective when combined with PTX byte unpacking
; to avoid bitwise overhead in the subsequent unpack stage
; Without proper unpacking, wide loads just move the bottleneck
```

## Data Reuse (Rank 2 Approach)

Since B is shape (1 x K x L), all M rows multiply against the same B vector:

```cpp
// Load B vector into shared memory once per thread block
// All BLOCK_M rows reuse the same B data
__shared__ half b_shared[K_TILE];

// Cooperative load: all threads in block load a portion of B
for (int i = threadIdx.x; i < K_TILE; i += blockDim.x) {
    b_shared[i] = dequant_fp4(b[i], sfb[i/16]) * sf_b_global;
}
__syncthreads();

// Each thread computes dot product of its A row with shared B
float acc = 0.0f;
for (int k = 0; k < K_TILE; k++) {
    acc += (float)a_dequant[k] * (float)b_shared[k];
}

// Reduces global memory traffic by BLOCK_M ratio
```

## Performance Progression

Documented progression from Yue's hackathon blog:

| Stage | Technique | Latency |
|-------|-----------|---------|
| CuTe DSL baseline | Basic CuTe partition/copy | ~100us |
| Coalesced access | Fix memory access patterns | 443us -> 39us |
| Hardware intrinsics | cvt.rn.f16x2.e2m1x2 | ~39us |
| PTX assembly | Full PTX with byte unpacking | ~27us |
| ILP optimization | Instruction-level parallelism | ~22.9us |
| Final submission | All combined | 22.392us |

## Key Lessons

1. **Memory-bound kernels need bandwidth-first thinking**: Arithmetic optimizations have minimal impact; focus on memory access patterns, cache policies, and vectorized loads
2. **PTX gives real control on Blackwell**: The gap between C intrinsics and hand-written PTX was substantial (443us to 27us in one journey)
3. **Profile first**: "Run Nsight Compute to confirm memory-bound behavior" (Amandeep's lesson after 12 attempts)
4. **Register budgeting matters**: Lower registers -> higher occupancy -> better memory latency hiding
5. **TMEM is irrelevant**: Memory-bound kernels do not benefit from TMEM (it helps compute-bound only)

## When to Use

- Decode-time MLP with batch size 1 (matrix-vector product)
- Any NVFP4 workload where arithmetic intensity is too low for tensor cores
- Memory-bound operations with FP4 quantized weights

## Caveats

- SM100/SM100a only (native FP4 decode instructions)
- PTX-level optimizations are fragile across CUDA toolkit versions
- Per-K specialization increases binary size (one kernel per K variant)
- Speed-of-light is bounded by B200 memory bandwidth (8 TB/s)

## Sources

- [GPU Mode NVFP4 Hackathon](https://github.com/gpu-mode/reference-kernels)
- [Yue's Hackathon Journey](https://yue-zhang-2025.github.io/2025/12/02/blackwell-nvfp4-kernel-hackathon-journey.html)
- [Twelve Attempts (Amandeep)](https://amandeepsp.github.io/blog/nvfp4-blackwell-gemv/)
- [Simon's NVFP4 GEMV Blog](https://veitner.bearblog.dev/nvfp4-gemv/)


## Related

- [DeepGEMM -- FP8 GEMM with Fine-Grained Scaling](deepgemm.md)
- [FlashAttention-4](flash-attention-4.md)
- [FlashMLA -- Multi-head Latent Attention](flashmla.md)
- [FP8 Block-Scale GEMM](fp8-block-scale-gemm.md)
- [Fused MoE -- FP8 Block-Scale Routing + Dual GEMM](fused-moe.md)
- [CUTLASS GEMM Optimization Strategy](../../common/cutedsl/cutlass-gemm-optimization.md)
- [Community GEMM Optimization Practical Summary](../../../generic/gemm-optimization-guide.md)
- [PTX Programming Model and Basics](../../common/ptx/ptx-programming-model.md)
