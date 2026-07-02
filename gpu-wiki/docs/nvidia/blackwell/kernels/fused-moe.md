# Fused MoE -- FP8 Block-Scale Routing + Dual GEMM


**Last updated**: 2026-06-30

## Overview

Fused MoE kernels combine the full Mixture-of-Experts forward pass into minimal kernel launches: routing, token dispatch, gate-up dual GEMM, SwiGLU activation, down projection GEMM, and token combine. In unfused implementations this requires 5-7 separate kernel launches; fused variants reduce this to 1-3 launches, eliminating intermediate global memory roundtrips and saving up to 21.9% activation memory traffic.

This is Track A of the FlashInfer MLSys 2026 contest, targeting B200 GPUs with DeepSeek-V3-style MoE parameters.

## MoE Forward Pass Structure

```
Input tokens x [batch, hidden_dim=7168]
    |
    v
[Router] top-k=8 experts from 32, grouped (8 groups, top_group=4)
    |
    v
[Dispatch] Scatter tokens to selected experts
    |
    v
[Gate-Up Dual GEMM]
    gate = x @ W_gate        [batch_expert, hidden -> intermediate]
    up   = x @ W_up          [batch_expert, hidden -> intermediate]
    |
    v
[SwiGLU Activation]
    h = SiLU(gate) * up      Element-wise: SiLU(x) = x * sigmoid(x)
    |
    v
[Down GEMM]
    y = h @ W_down            [batch_expert, intermediate -> hidden]
    |
    v
[Combine] Weighted sum of expert outputs per token (router weights)
    |
    v
Output [batch, hidden_dim=7168]
```

## Kernel Fusion Strategy

```
Unfused (vLLM): 7 kernel launches
  1. Router softmax
  2. Top-k selection
  3. Token dispatch (scatter)
  4. Gate GEMM
  5. Up GEMM
  6. SiLU + multiply
  7. Down GEMM + combine

Partially fused (SGLang): 5 kernel launches
  1. Router + top-k
  2. Dispatch
  3. Gate-Up fused GEMM + SiLU (3 ops -> 1 kernel)
  4. Down GEMM
  5. Combine

Fully fused (ideal): 1-2 launches
  All ops in single kernel, or gate-up-silu + down-combine
```

## Gated Dual GEMM Kernel (Hackathon Problem 3)

The gate-up projection fuses two GEMMs with SiLU activation and element-wise multiply:

```cpp
// Fused gate-up: two GEMMs + SiLU + multiply in one kernel
// Avoids writing intermediate gate and up results to global memory

template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int NUM_STAGES>
__global__ void gated_dual_gemm_fused(
    const fp8_t* __restrict__ X,       // [M, K] input tokens
    const fp8_t* __restrict__ W_gate,  // [N, K] gate weights
    const fp8_t* __restrict__ W_up,    // [N, K] up weights
    const float* __restrict__ sf_x,    // Block scales for X
    const float* __restrict__ sf_gate, // Block scales for W_gate
    const float* __restrict__ sf_up,   // Block scales for W_up
    half* __restrict__ output,         // [M, N] fused output
    int M, int N, int K
) {
    // Two TMEM regions: one for gate accumulator, one for up accumulator
    uint32_t tmem_gate, tmem_up;
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned %0, %1;"
                 : "=r"(tmem_gate) : "r"(256));
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned %0, %1;"
                 : "=r"(tmem_up) : "r"(256));

    // Pipelined main loop
    for (int k = 0; k < K; k += BLOCK_K) {
        int stage = (k / BLOCK_K) % NUM_STAGES;
        barrier_wait(stage);

        // Two MMAs per K-tile: gate and up projections
        // Both read same X tile, different weight tiles
        asm volatile(
            "tcgen05.mma.cta_group::1.kind::f8f6f4"
            " [%0], %1, %2, %3, %4;"
            :: "r"(tmem_gate), "l"(x_smem[stage]),
               "l"(wg_smem[stage]), "r"(scales_gate), "n"(1)
        );
        asm volatile(
            "tcgen05.mma.cta_group::1.kind::f8f6f4"
            " [%0], %1, %2, %3, %4;"
            :: "r"(tmem_up), "l"(x_smem[stage]),
               "l"(wu_smem[stage]), "r"(scales_up), "n"(1)
        );
    }

    // Fused epilogue: SiLU(gate) * up
    // Read both accumulators from TMEM, apply activation, store result
    float gate_val, up_val;
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x1 {%0}, [%1];"
                 : "=r"(gate_val) : "r"(tmem_gate));
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x1 {%0}, [%1];"
                 : "=r"(up_val) : "r"(tmem_up));

    // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
    float silu_gate = gate_val / (1.0f + expf(-gate_val));
    output[row * N + col] = __float2half(silu_gate * up_val);

    // Deallocate TMEM
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned %0, %1;"
                 :: "r"(tmem_gate), "r"(256));
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned %0, %1;"
                 :: "r"(tmem_up), "r"(256));
}
```

## FP8 Block-Scale MoE (FlashInfer API)

```python
# FlashInfer API for fused MoE
# Single function call replaces 5-7 separate kernel launches
import flashinfer

output = flashinfer.fused_moe.trtllm_fp8_block_scale_moe(
    hidden_states=x,            # [batch, 7168] BF16 input
    w_gate_up=w_gate_up,        # [32, 2*2048, 7168] FP8 (fused gate+up weights)
    w_down=w_down,              # [32, 7168, 2048] FP8 (down weights)
    router_weights=router_w,    # [batch, 32] FP32 routing logits
    topk=8,                     # Select top-8 experts per token
    num_groups=8,               # Expert grouping
    topk_group=4,               # Top groups to select from
    block_scale_gate_up=sf_gu,  # Block scales [32, 2*2048, 7168/128] FP8
    block_scale_down=sf_d,      # Block scales [32, 7168, 2048/128] FP8
)
```

## Triton Fused Gate-Up Kernel

```python
import triton
import triton.language as tl

@triton.jit
def fused_moe_gate_up_triton(
    X_ptr, W_gate_ptr, W_up_ptr, Out_ptr,
    expert_ids_ptr, token_counts_ptr,
    sf_x_ptr, sf_gate_ptr, sf_up_ptr,
    N: tl.constexpr, K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Fused gate-up GEMM + SiLU for one expert's tokens."""
    expert_id = tl.program_id(2)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    token_count = tl.load(token_counts_ptr + expert_id)
    m_start = pid_m * BLOCK_M
    if m_start >= token_count:
        return

    n_start = pid_n * BLOCK_N
    offs_m = m_start + tl.arange(0, BLOCK_M)
    offs_n = n_start + tl.arange(0, BLOCK_N)

    gate_acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    up_acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        x_tile = tl.load(X_ptr + offs_m[:, None] * K + offs_k[None, :])
        wg_tile = tl.load(W_gate_ptr + expert_id * N * K
                          + offs_n[:, None] * K + offs_k[None, :])
        wu_tile = tl.load(W_up_ptr + expert_id * N * K
                          + offs_n[:, None] * K + offs_k[None, :])
        gate_acc += tl.dot(x_tile, tl.trans(wg_tile))
        up_acc += tl.dot(x_tile, tl.trans(wu_tile))

    # Fused epilogue: SiLU(gate) * up
    silu_gate = gate_acc * tl.sigmoid(gate_acc)
    result = silu_gate * up_acc
    tl.store(Out_ptr + offs_m[:, None] * N + offs_n[None, :],
             result.to(tl.float16))
```

## Framework Baselines (B200)

| Framework | Batch 4096 TFLOPS | Batch 1 Latency | Kernel Launches |
|-----------|-------------------|-----------------|-----------------|
| SGLang | 1262 | 206.9us | 5 (fused) |
| FlashInfer CuTe DSL | 1225 | 481.9us | 1-2 (fully fused) |
| vLLM | 1117 | 369.5us | 7 (unfused) |

## Challenges

1. **No pre-tuned FP8 MoE config for B200**: Tile sizes and pipeline stages need empirical tuning
2. **FP8 numerical overflow**: Block scaling (block size 128) required for stability
3. **Batch-size sensitivity**: batch=1 is latency-critical (kernel launch overhead dominates); batch=4096 is throughput-critical
4. **Expert load imbalance**: Variable token counts per expert cause tail effects
5. **TMA alignment**: 128-byte alignment required for all TMA descriptors
6. **Dual TMEM allocation**: Gate and up accumulators each need TMEM space, competing for the 256KB budget

## When to Use

- MoE model inference (DeepSeek-V3, Mixtral, etc.)
- Both prefill (high batch, throughput-critical) and decode (low batch, latency-critical)
- When gate-up GEMM fusion provides measurable speedup over separate launches

## Caveats

- Full fusion (routing through combine) is extremely complex to implement correctly
- Expert load imbalance is the primary practical bottleneck
- CUDA graph compatibility requires masked layout (fixed allocation per expert)
- Small expert token counts cause thin-GEMM inefficiency on tensor cores
- FP8 block scaling adds memory overhead for scale factor storage

## Sources

- [FlashInfer MLSys 2026 Contest](https://mlsys26.flashinfer.ai/)
- [GPU Mode NVFP4 Hackathon Problem 3](https://github.com/gpu-mode/reference-kernels)
- [DeepGEMM MoE](https://github.com/deepseek-ai/DeepGEMM)
- [SGLang Fused MoE](https://github.com/sgl-project/sglang)


## Related

- [DeepGEMM -- FP8 GEMM with Fine-Grained Scaling](deepgemm.md)
- [FlashAttention-4](flash-attention-4.md)
- [FlashMLA -- Multi-head Latent Attention](flashmla.md)
- [FP8 Block-Scale GEMM](fp8-block-scale-gemm.md)
- [Gated Delta Net -- Linear Attention](gated-delta-net.md)
- [CUTLASS GEMM Optimization Strategy](../../common/cutedsl/cutlass-gemm-optimization.md)
- [Community GEMM Optimization Practical Summary](../../../generic/gemm-optimization-guide.md)
- [Composable Kernel (CK) Architecture Overview](../../../amd/common/ck-architecture-overview.md)
