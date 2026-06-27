# Gated Dual GEMM

## Overview

Gated dual GEMM fuses two matrix multiplications with activation and elementwise operations -- the canonical MLP gate-up pattern used by LLaMA, Qwen, DeepSeek, and most modern LLMs. Fusion eliminates two global memory roundtrips compared to separate gate/up GEMM + SwiGLU.

This was Problem 3 of the GPU Mode NVFP4 Hackathon.

## Fused Operation

```
Given x, W_gate, W_up (weights shared same K dimension)
Standard (unfused):
  gate = x @ W_gate      (GEMM 1: reads x, W_gate, writes gate)
  up   = x @ W_up         (GEMM 2: reads x, W_up, writes up)
  silu = gate * sigmoid(gate)  (elementwise: reads gate, writes silu)
  out  = silu * up        (elementwise: reads silu, up, writes out)

Fused:
  out = SiLU(x @ W_gate) * (x @ W_up)  (single kernel, no intermediate GMEM)
```

## Kernel Structure (Blackwell)

```cuda
template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int NUM_STAGES>
__global__ void gated_dual_gemm_nvfp4(
    const nvfp4_t* __restrict__ X,        // [M, K] input
    const nvfp4_t* __restrict__ W_gate,   // [N, K] gate weights
    const nvfp4_t* __restrict__ W_up,     // [N, K] up weights
    const fp8_t* sf_x, const fp8_t* sf_gate, const fp8_t* sf_up,
    half* __restrict__ output,             // [M, N] output
    int M, int N, int K
) {
    // Two TMEM regions: one accumulator per GEMM
    uint32_t tmem_gate = tmem_alloc(256);
    uint32_t tmem_up   = tmem_alloc(256);

    for (int k = 0; k < K; k += BLOCK_K) {
        int stage = (k / BLOCK_K) % NUM_STAGES;
        mbarrier_wait(&tma_done[stage]);

        // Two MMAs per K-tile: gate and up
        // Both read same X tile, different weight tiles
        tcgen05_mma(x_smem[stage], wg_smem[stage], tmem_gate);
        tcgen05_mma(x_smem[stage], wu_smem[stage], tmem_up);
    }

    // Fused epilogue: SiLU(gate) * up
    float g = tmem_load(tmem_gate);
    float u = tmem_load(tmem_up);
    float s = g / (1.0f + expf(-g));  // SiLU
    output[row * N + col] = __float2half(s * u);

    tmem_dealloc(tmem_gate, 256);
    tmem_dealloc(tmem_up, 256);
}
```

## Key Optimizations

1. **X reuse**: Same X tile feeds both MMAs -- loaded once, used twice
2. **TMEM dual accumulator**: Blackwell's 512-column TMEM fits two 256-col accumulators side-by-side
3. **Fused epilogue**: SiLU and multiply happen after TMEM load, no intermediate SMEM
4. **Shared SFA**: X's block scales apply to both gate and up computations

## When To Use

- MLP layers in modern LLMs (LLaMA, Qwen, DeepSeek, Mistral)
- Any dual-output operation sharing one input
- MoE expert computations (expand to per-expert fused kernels)
