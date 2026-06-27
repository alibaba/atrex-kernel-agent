# NVFP4 GEMM -- 4-bit Floating Point Matrix Multiply

## Overview

NVFP4 GEMM is a compute-bound matrix multiplication kernel operating on NVIDIA's native 4-bit floating-point format (E2M1) with block scaling on Blackwell GPUs. Unlike the memory-bound GEMV, GEMM is dominated by tensor core throughput and benefits from Blackwell's native FP4 MMA instructions via tcgen05.mma, TMA bulk loads, TMEM accumulation, and warp specialization.

This kernel was Problem 2 of the GPU Mode NVFP4 Hackathon (Nov-Dec 2025), targeting B200 GPUs. Top entries achieved within 1% of cuBLAS performance using CUTLASS SM100 schedules.

## NVFP4 Data Format

```
NVFP4 (E2M1): 4-bit floating point
  Bit layout: [S][E1][E0][M0]
  Representable values: 0, 0.5, 1, 1.5, 2, 3, 4, 6 (positive and negative)

Block scaling:
  Every 16 FP4 elements share one FP8 (E4M3) scale factor
  Two-level: per-block E4M3 + per-tensor FP32 global scale

  Dequantization: x_hat = s_global * s_block * deq_FP4(q)

Key difference from MXFP4:
  - E4M3 block scale (non-power-of-two) vs UE8M0 (power-of-two only)
  - Block size 16 (tighter) vs 32 (coarser)
```

## CUTLASS SM100 Schedule

The kernel uses the `KernelPtrArrayTmaWarpSpecialized1SmNvf4Sm100` CUTLASS schedule, which combines TMA async loads with warp-specialized MMA execution.

```cpp
// CUTLASS dispatch for NVFP4 GEMM on Blackwell
using Schedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecialized1SmNvf4Sm100;

// Tile configuration
using TileShape = cute::Shape<_128, _256, _128>;  // M, N, K tile
using ClusterShape = cute::Shape<_1, _1, _1>;      // 1-SM mode

// Element types
using ElementA = cutlass::float_e2m1_t;  // NVFP4
using ElementB = cutlass::float_e2m1_t;  // NVFP4
using ElementC = float;                   // FP32 accumulator
using ElementScale = cutlass::float_e4m3_t;  // FP8 E4M3 block scales

// Kernel definition
using Kernel = cutlass::gemm::device::GemmUniversal<
    ElementA, cutlass::layout::RowMajor,
    ElementB, cutlass::layout::ColumnMajor,
    ElementC, cutlass::layout::RowMajor,
    float,  // accumulator type
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm100,
    TileShape, ClusterShape, Schedule
>;
```

## Warp-Specialized Kernel Structure

```cpp
// Warp specialization: TMA producer + MMA consumer + epilogue
// Shared memory holds pipelined A/B tiles + scale factors

constexpr int NUM_STAGES = 4;  // Pipeline depth

// Shared memory layout
struct SharedStorage {
    // Double-buffered across NUM_STAGES
    nvfp4_t A_smem[NUM_STAGES][BLOCK_M * BLOCK_K / 2];  // Packed FP4: 2 per byte
    nvfp4_t B_smem[NUM_STAGES][BLOCK_N * BLOCK_K / 2];
    fp8_t   sfa_smem[NUM_STAGES][BLOCK_M * (BLOCK_K / 16)];  // 1 scale per 16 elements
    fp8_t   sfb_smem[NUM_STAGES][BLOCK_N * (BLOCK_K / 16)];
    uint64_t mbarrier[NUM_STAGES];
};

__global__ void nvfp4_gemm_kernel(
    const nvfp4_t* A, const nvfp4_t* B,
    const fp8_t* sfa, const fp8_t* sfb,
    float sf_a_global, float sf_b_global,
    float* C, int M, int N, int K
) {
    extern __shared__ SharedStorage smem[];
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    if (warp_id == 0 && lane_id == 0) {
        // TMA producer warp: async bulk loads
        for (int k = 0; k < K; k += BLOCK_K) {
            int stage = (k / BLOCK_K) % NUM_STAGES;

            // TMA descriptor-based bulk load (128-byte aligned)
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cluster.global.tile"
                ".mbarrier::complete_tx::bytes"
                " [%0], [%1, {%2, %3}], [%4];"
                :: "r"((uint32_t)&smem->A_smem[stage]),
                   "l"(tma_desc_A),
                   "r"(tile_m), "r"(k),
                   "r"((uint32_t)&smem->mbarrier[stage])
            );
            // Similarly for B, sfa, sfb
        }
    } else if (warp_id == 1 && lane_id == 0) {
        // MMA consumer warp: tcgen05.mma
        uint32_t tmem_addr;
        asm volatile(
            "tcgen05.alloc.cta_group::1.sync.aligned %0, %1;"
            : "=r"(tmem_addr) : "r"(256)  // 256 TMEM columns
        );

        for (int k = 0; k < K; k += BLOCK_K) {
            int stage = (k / BLOCK_K) % NUM_STAGES;
            // Wait for TMA to complete this stage
            mbarrier_wait(&smem->mbarrier[stage]);

            // tcgen05.mma with native block scaling
            // Reads A/B from SMEM, accumulates into TMEM
            asm volatile(
                "tcgen05.mma.cta_group::1.kind::f8f6f4"
                " [%0], %1, %2, %3, %4;"
                :: "r"(tmem_addr),
                   "l"((uint64_t)&smem->A_smem[stage]),
                   "l"((uint64_t)&smem->B_smem[stage]),
                   "r"(packed_scales),
                   "n"(1)  // scale_D enabled
            );
        }

        // Signal epilogue warps
    } else {
        // Epilogue warps: read from TMEM, apply global scales, store to C
        // tmem -> registers -> global
    }
}
```

## 128-Byte TMA Alignment

All TMA operands require 128-byte alignment. For NVFP4 (2 elements per byte), this means K dimensions must be multiples of 256 elements:

```cpp
// Critical: pad tensors to 128-byte boundaries for TMA
// NVFP4: 2 elements per byte, so 256 elements = 128 bytes
static_assert(K % 256 == 0, "K must align to 128 bytes for FP4 TMA");

// For scale factors: FP8 is 1 byte per element
// 128 bytes = 128 scale values
// Since 1 scale per 16 FP4 elements: 128 scales cover 2048 FP4 elements
static_assert((K / 16) % 128 == 0, "Scale array must align to 128 bytes");
```

## Scale Factor Conversion

The tcgen05.mma instruction expects UE8M0 (unsigned power-of-two exponent only) scales, but NVFP4 uses FP8 E4M3 (non-power-of-two). Conversion is needed:

```cpp
// Convert FP8 E4M3 block scales to UE8M0 for tcgen05.mma hardware
// E4M3: 4 exponent bits, 3 mantissa bits (non-power-of-two)
// UE8M0: 8 exponent bits, 0 mantissa bits (power-of-two only)
__device__ uint32_t pack_scales_ue8m0(
    fp8_e4m3_t s0, fp8_e4m3_t s1, fp8_e4m3_t s2, fp8_e4m3_t s3
) {
    uint8_t u0 = fp8_e4m3_to_ue8m0(s0);  // Round to nearest power-of-two
    uint8_t u1 = fp8_e4m3_to_ue8m0(s1);
    uint8_t u2 = fp8_e4m3_to_ue8m0(s2);
    uint8_t u3 = fp8_e4m3_to_ue8m0(s3);
    return (u3 << 24) | (u2 << 16) | (u1 << 8) | u0;
}
```

## Competition Results

Problem 2 top performers (geometric mean across benchmark configs):

| Rank | Participant | Latency (us) |
|------|-------------|--------------|
| 1 | Simon | 10.807 |
| 2 | yue | 10.914 |
| 3 | currybab | 10.931 |

## When to Use

- Inference with 4-bit quantized weights on Blackwell
- MLP layers in LLMs where weight matrices are NVFP4-quantized
- Compute-bound matrix multiplications where tensor core utilization is the bottleneck

## Caveats

- SM100/SM100a only -- no Hopper support for native FP4 tensor core instructions
- Scale factor conversion (E4M3 to UE8M0) adds overhead if not precomputed
- TMA requires 128-byte alignment for all operands
- TMEM size (128x512 per SM) limits maximum output tile to 128 rows x 512 cols (32-bit)

## Sources

- [GPU Mode NVFP4 Hackathon](https://github.com/gpu-mode/reference-kernels)
- [NVIDIA NVFP4 Blog](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)
- [NVFP4 Format Details](https://haroldbenoit.com/notes/ml/engineering/precision/nvfp4-format)
- [CUTLASS SM100 documentation](https://docs.nvidia.com/cutlass/latest/CHANGELOG.html)
