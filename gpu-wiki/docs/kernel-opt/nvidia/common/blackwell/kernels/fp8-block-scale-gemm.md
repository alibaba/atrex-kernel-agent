# FP8 Block-Scale GEMM

## Overview

FP8 GEMM with fine-grained block scales (128x128 weights, 1x128 activations). Preserves more dynamic range than per-tensor FP8 scaling, critical for LLM inference and training where outliers dominate quantization error.

DeepGEMM is the reference implementation; CUTLASS provides SM100 schedules.

## Block Scaling Structure

```
Activations: tile-wise 1x128 scales
  [1x128 values] → 1 scale factor (FP32 or FP8 E4M3)

Weights: block-wise 128x128 scales
  [128x128 values] → 1 scale factor per block

Output accumulator: FP32
  Multiply A × B in FP8, accumulate in FP32, apply scales at MMA boundary
```

## SM90 Path (Hopper, WGMMA)

```cuda
// Hopper accumulator has ~22-bit precision (FP22)
// Every Nc=128 WGMMAs, promote partial sum to FP32 on CUDA cores
// This retains precision without adding MMA overhead

__device__ void sm90_fp8_gemm_with_promotion(...) {
    float acc_fp32 = 0.0f;

    for (int k = 0; k < K; k += 128) {
        float acc_fp22 = 0.0f;
        #pragma unroll 4
        for (int kk = 0; kk < 128; kk += 32) {
            wgmma_fp8_e4m3(acc_fp22, A_frag, B_frag);
        }
        // Promote to FP32 accumulator on CUDA cores
        acc_fp32 += acc_fp22 * scale_a * scale_b;
    }

    // Write acc_fp32 to output
}
```

## SM100 Path (Blackwell, tcgen05.mma)

```cuda
// tcgen05.mma has native UE8M0 block scale support in hardware
// No promotion needed - scales applied inside MMA

__global__ void sm100_fp8_gemm_block_scale(...) {
    uint32_t tmem = tmem_alloc(256);

    for (int k = 0; k < K; k += BLOCK_K) {
        mbarrier_wait(&tma_done);

        // tcgen05.mma.mxf8f6f4.block_scale variant
        // Reads A, B from SMEM; scales from scale SMEM; accumulates in TMEM
        asm volatile(
            "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X "
            "[%0], %1, %2, %3, %4, %5;"
            :: "r"(tmem), "l"(a_desc), "l"(b_desc),
               "r"(sf_a_desc), "r"(sf_b_desc), "n"(1)
        );
    }

    // Read from TMEM, apply global scale, store
    float result = tmem_load(tmem) * global_scale;
    output[row * N + col] = __float2half(result);
}
```

## Memory Layout

```
A (activations) [M, K] packed FP8 E4M3
sf_a            [M, K/128] FP32 or FP8 E4M3 scales  # 1 per 1x128 tile

B (weights)     [N, K] packed FP8 E4M3
sf_b            [N/128, K/128] scales               # 1 per 128x128 block
                # or packed UE8M0 format (Blackwell): 4 scales per int32
```

## Performance

- DeepGEMM on H800: up to 1550 TFLOPS FP8
- CUTLASS SM100 schedules: similar ratio vs peak

## When To Use

- LLM inference with FP8 quantized weights (DeepSeek V3, Qwen2-FP8, etc.)
- Training with FP8 activations (DeepSeek V3 training framework)
- Anywhere per-tensor FP8 accuracy is insufficient due to outliers
