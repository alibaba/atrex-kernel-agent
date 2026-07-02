# Fine-Grained FP8/FP4 Quantization


**Last updated**: 2026-06-30

## Overview

Fine-grained quantization applies per-block (rather than per-tensor) scaling factors to low-precision data, preventing outlier values from destroying the quantization precision of an entire tensor. DeepSeek pioneered the tile-wise 1x128 scaling for activations and block-wise 128x128 scaling for weights in their FP8 training framework. On Blackwell (SM100), native block scaling support in tcgen05.mma enables hardware-accelerated fine-grained quantization using the UE8M0 scale format, while Hopper requires software-managed CUDA core promotion.

## Scaling Granularities

```
Per-tensor scaling (coarsest):
  Entire tensor shares one FP32 scale factor.
  Problem: a single outlier ruins precision for all elements.

  [=== entire MxK matrix === ] -> 1 scale

Per-block 128x128 scaling (weights):
  Each 128x128 block has its own scale factor.
  128x128 = 16,384 elements per scale -> 0.006% overhead.

  +---+---+---+
  |s1 |s2 |s3 |  <- each block has independent scale
  +---+---+---+
  |s4 |s5 |s6 |
  +---+---+---+

Per-tile 1x128 scaling (activations):
  Each row of 128 elements has its own scale factor.
  Captures per-channel activation distributions.

  [s1: ====128 elements====]
  [s2: ====128 elements====]
  [s3: ====128 elements====]
```

## DeepGEMM: Tile-wise and Block-wise Scaling

DeepGEMM implements fine-grained FP8 GEMM with two scaling patterns:

```cuda
// DeepGEMM FP8 GEMM with fine-grained scaling
// A (activations): FP8 E4M3 with tile-wise 1x128 FP32 scales
// B (weights):     FP8 E4M3 with block-wise 128x128 FP32 scales
// C (output):      FP32 accumulator

// Scale tensor shapes:
//   scale_A: [M, K/128] -- one FP32 scale per 128 elements along K for each row
//   scale_B: [K/128, N/128] -- one FP32 scale per 128x128 block

__device__ void deepgemm_fp8_tile(
    const fp8_e4m3* A,    // [TILE_M, TILE_K] in FP8
    const fp8_e4m3* B,    // [TILE_K, TILE_N] in FP8
    const float* scale_A, // [TILE_M, TILE_K/128]
    const float* scale_B, // [TILE_K/128, TILE_N/128]
    float* C,             // [TILE_M, TILE_N] accumulator
    int M, int N, int K)
{
    // For each 128-element K-chunk:
    for (int k_block = 0; k_block < K; k_block += 128) {
        // 1. Load FP8 A tile [TILE_M, 128] and B tile [128, TILE_N]
        // 2. Perform MMA: partial = A_fp8 * B_fp8 (in limited-precision FP32)
        // 3. Apply combined scale: scale_A[m, k_block/128] * scale_B[k_block/128, n_block/128]
        // 4. Accumulate: C[m, n] += partial * combined_scale

        for (int m = 0; m < TILE_M; m++) {
            float sa = scale_A[m * (K / 128) + k_block / 128];
            for (int n_block = 0; n_block < TILE_N; n_block += 128) {
                float sb = scale_B[(k_block / 128) * (N / 128) + n_block / 128];
                float combined_scale = sa * sb;

                // Apply scale to the partial MMA result
                for (int n = n_block; n < n_block + 128; n++) {
                    C[m * TILE_N + n] += partial[m][n] * combined_scale;
                }
            }
        }
    }
}
```

## Hopper: CUDA Core Promotion (Nc=128)

On Hopper (SM90), the wgmma instruction accumulates in Tensor Core registers with limited precision (~FP22, not true FP32). To maintain numerical accuracy, DeepGEMM promotes the partial sum to a separate FP32 accumulator on CUDA Cores every 4 wgmma instructions (Nc=128, since each wgmma processes 32 K-elements):

```cuda
// Hopper FP8 GEMM with CUDA Core promotion (DeepGEMM pattern)
// Every Nc=128 K-elements, promote Tensor Core accumulator to FP32

__device__ void hopper_fp8_gemm_with_promotion(
    const fp8_e4m3* A, const fp8_e4m3* B,
    const float* scale_A, const float* scale_B,
    float* C_accumulator,
    int K)
{
    // Tensor Core limited-precision accumulator (FP22-ish)
    // These are wgmma output registers
    float tc_acc[TILE_M_PER_THREAD][TILE_N_PER_THREAD] = {0};

    // True FP32 accumulator on CUDA Cores
    float fp32_acc[TILE_M_PER_THREAD][TILE_N_PER_THREAD] = {0};

    int promotion_interval = 128;  // Nc = 128 elements = 4 wgmma ops
    int wgmma_count = 0;

    for (int k = 0; k < K; k += 32) {
        // Issue wgmma (accumulates in limited-precision tc_acc)
        wgmma_mma_async(tc_acc, smem_A_ptr, smem_B_ptr);
        wgmma_count++;

        if (wgmma_count == 4) {  // Every 128 K-elements
            // Promote: transfer tc_acc to fp32_acc on CUDA Cores
            wgmma_wait();  // Ensure wgmma is complete

            int k_block = k / promotion_interval;
            for (int m = 0; m < TILE_M_PER_THREAD; m++) {
                float sa = scale_A[/*...*/];
                for (int n = 0; n < TILE_N_PER_THREAD; n++) {
                    float sb = scale_B[/*...*/];
                    // Add scaled partial to true FP32 accumulator
                    fp32_acc[m][n] += tc_acc[m][n] * sa * sb;
                    // Reset TC accumulator for next interval
                    tc_acc[m][n] = 0;
                }
            }
            wgmma_count = 0;
        }
    }
}
```

The Nc=128 interval was chosen because:
- 4 wgmma operations process 128 K-elements (4 x 32)
- At this interval, the accumulated FP22 error is bounded to ~0.1% relative error
- Fewer than 4 ops (Nc=32, Nc=64) adds too much promotion overhead
- More than 4 ops (Nc=256) allows unacceptable precision loss

## Blackwell: Native Block Scaling

On Blackwell, tcgen05.mma supports native block scaling via the UE8M0 (unsigned 8-bit exponent, no mantissa) format. The hardware applies per-block scale factors directly during the MMA operation, eliminating the software promotion step:

```cuda
// Blackwell native block scaling with UE8M0
// Scale format: UE8M0 = pure power-of-two scale (2^exponent)
// Packed: 4 UE8M0 values per 32-bit integer

// DeepGEMM SM100 kernel: scale_A and scale_B are UE8M0 packed
// tcgen05.mma applies scales automatically during accumulation

struct BlockScaleDescriptor {
    // 4 UE8M0 scale values packed into one uint32
    // Each UE8M0 is an 8-bit unsigned exponent: value = 2^(e - 127)
    uint32_t packed_scales;  // Contains 4 block scales

    // Dequantization for block [i]:
    //   scale_i = 2^(((packed >> (i*8)) & 0xFF) - 127)
};

// PTX for tcgen05.mma with block scaling:
// The .scale modifier tells the hardware to apply UE8M0 scales
// from a designated SMEM region alongside the MMA operands
```

```ptx
// tcgen05.mma with native block scaling (Blackwell PTX)
// This instruction applies UE8M0 scales from SMEM during MMA
tcgen05.mma.cta_group::1.kind::f8f6f4
    [%tmem_addr],           // TMEM accumulator destination
    [%desc_a],              // SMEM descriptor for A operand
    [%desc_b],              // SMEM descriptor for B operand
    %scale_d,               // Scale descriptor for D (output)
    %enable_mask,
    [%scale_a_smem],        // UE8M0 scales for A in SMEM
    [%scale_b_smem];        // UE8M0 scales for B in SMEM
```

## UE8M0 vs E4M3 Scale Formats

| Property | UE8M0 (Blackwell native) | E4M3 (NVFP4 hackathon) | FP32 (DeepGEMM Hopper) |
|----------|-------------------------|------------------------|------------------------|
| Bits | 8 | 8 | 32 |
| Representable values | Powers of 2 only | 240 distinct values | Full FP32 range |
| Range | 2^-127 to 2^128 | ~0 to 448 | Full FP32 |
| Block size | 32 (MXFP standard) | 16 (NVFP4) | 128 (DeepGEMM) |
| Hardware support | tcgen05.mma native | Software decode | Software promotion |
| Precision impact | Coarser (power-of-2 only) | Fine (non-power-of-2) | Best (FP32) |

## NVFP4 Two-Level Scaling

The NVFP4 format used in the GPU Mode hackathon has its own scaling scheme:

```cuda
// NVFP4 dequantization with two-level scaling
// Level 1: per-block FP8 E4M3 scale (every 16 FP4 elements)
// Level 2: per-tensor FP32 global scale

__device__ float dequant_nvfp4(
    uint8_t fp4_packed,    // Two FP4 values packed in one byte
    fp8_e4m3 block_scale,  // Per-block scale (every 16 elements)
    float global_scale)    // Per-tensor global scale
{
    // Extract one FP4 value (E2M1 format)
    // Representable: 0, 0.5, 1, 1.5, 2, 3, 4, 6
    float fp4_val = decode_e2m1(fp4_packed & 0x0F);

    // Two-level dequantization
    float result = global_scale * float(block_scale) * fp4_val;
    return result;
}

// Block scale layout for NVFP4:
// For matrix A of shape [M, K]:
//   scale_A shape: [M, K/16] -- one FP8 E4M3 per 16 FP4 elements
//   Finer granularity than MXFP4 (block size 32) or DeepGEMM (block size 128)
```

## When to Use

- **FP8 training**: Use tile-wise 1x128 for activations and block-wise 128x128 for weights (DeepGEMM pattern). This is the validated approach for training 671B+ parameter models.
- **FP4 inference on Blackwell**: Use NVFP4 with E4M3 block scales (block size 16) for highest precision, or MXFP4 with UE8M0 scales (block size 32) for native hardware acceleration.
- **Hopper FP8 inference**: Use CUDA core promotion with Nc=128 interval to maintain precision despite limited TC accumulation.

## Caveats

- UE8M0 scales are power-of-two only. Non-power-of-two distributions (common in activations) lose precision compared to E4M3 or FP32 scales.
- Smaller block sizes (16 for NVFP4 vs 128 for DeepGEMM) provide better precision but higher overhead: more scale values to store, load, and apply.
- The Nc=128 promotion interval on Hopper is a performance-accuracy tradeoff. Reducing Nc improves accuracy but adds more promotion overhead. Increasing Nc risks precision degradation.
- On Blackwell, native block scaling only works with UE8M0. Using E4M3 or FP32 scales still requires software handling.


## Related

- [Cache Policy Differentiation](cache-policy.md)
- [Chunk-Based Parallelism](chunk-parallelism.md)
- [CUDA GEMM Optimization Ladder](cuda-gemm-optimization-ladder.md)
- [Double/Multi-Buffering Patterns](double-buffering.md)
- [Epilogue Fusion](epilogue-fusion.md)
- [Composable Kernel (CK) Architecture Overview](../../../amd/common/ck-architecture-overview.md)
- [CK Tile Quantized GEMM and MX Format](../../../amd/common/ck-quantization-mx.md)
