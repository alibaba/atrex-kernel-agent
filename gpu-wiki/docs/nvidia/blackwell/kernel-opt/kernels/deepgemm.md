# DeepGEMM -- FP8 GEMM with Fine-Grained Scaling

## Overview

DeepGEMM is DeepSeek's open-source FP8 GEMM library providing high-performance matrix multiplication with fine-grained per-tile/per-block scaling. The core kernel is remarkably compact (~300 lines), yet achieves approximately 90% utilization on H800. It supports both Hopper (SM90 via WGMMA) and Blackwell (SM100 via tcgen05.mma) architectures, and includes specialized MoE grouped GEMM layouts.

The key innovation is the fine-grained quantization scheme: tile-wise 1x128 scaling for activations and block-wise 128x128 scaling for weights, which prevents outlier values from destroying quantization precision.

## Fine-Grained Quantization Scheme

```
Activations (tile-wise 1x128):
  +-----------+-----------+-----------+
  | tile 0    | tile 1    | tile 2    |  <-- each tile: 1 row x 128 cols
  | scale: s0 | scale: s1 | scale: s2 |  <-- one FP32 scale per tile
  +-----------+-----------+-----------+

Weights (block-wise 128x128):
  +---------------+---------------+
  |  block (0,0)  |  block (0,1)  |  <-- each block: 128 rows x 128 cols
  |  scale: s_00  |  scale: s_01  |  <-- one FP32 scale per block
  +---------------+---------------+
  |  block (1,0)  |  block (1,1)  |
  |  scale: s_10  |  scale: s_11  |
  +---------------+---------------+
```

## FP8 Accumulation with Nc=128 CUDA Core Promotion

On Hopper, the Tensor Core accumulator has limited precision (~FP22, not true FP32). DeepGEMM mitigates this by promoting partial sums to a separate FP32 accumulator on CUDA Cores every Nc=128 columns (4 consecutive WGMMA operations).

```cpp
// SM90 path: WGMMA with Nc=128 CUDA Core promotion
// Every 4 WGMMAs, promote accumulated result to FP32 CUDA core accumulator
constexpr int Nc = 128;  // Promotion interval (4 WGMMAs of n=32 each)
constexpr int WGMMA_N = 32;

float cuda_core_acc[TILE_M][TILE_N] = {0};  // FP32 accumulator on CUDA Cores

for (int k = 0; k < K; k += Nc) {
    // Run 4 consecutive WGMMAs with TC-limited precision accumulation
    __half2 tc_acc[TILE_M][WGMMA_N];  // Tensor Core accumulator (~FP22)
    memset(tc_acc, 0, sizeof(tc_acc));

    for (int sub_k = 0; sub_k < Nc; sub_k += WGMMA_K) {
        wgmma_mma_async(tc_acc, A_smem + sub_k, B_smem + sub_k);
    }
    wgmma_wait();

    // Promote: add TC result to CUDA Core FP32 accumulator
    // This prevents precision loss from repeated FP22 accumulation
    for (int m = 0; m < TILE_M; m++)
        for (int n = 0; n < TILE_N; n++)
            cuda_core_acc[m][n] += (float)tc_acc[m][n] * scale_a[m] * scale_b[n];
}
```

On Blackwell (SM100), the tcgen05.mma instruction with TMEM accumulation uses native UE8M0 block scaling, which eliminates the need for explicit CUDA core promotion.

```cpp
// SM100 path: tcgen05.mma with native block scaling
// Scaling factors packed as UE8M0 (4 values per uint32)
// No explicit CUDA core promotion needed -- TMEM accumulates in full precision

// Pack 4 UE8M0 scale factors into a single uint32
uint32_t packed_scales = pack_ue8m0(sf[0], sf[1], sf[2], sf[3]);

// tcgen05.mma reads A/B from SMEM, accumulates into TMEM
// Block scale applied natively during MMA
asm volatile(
    "tcgen05.mma.cta_group::1.kind::f8f6f4"
    " [%0], %1, %2, %3, %4;"
    :
    : "l"(tmem_addr), "l"(a_smem_addr), "l"(b_smem_addr),
      "r"(packed_scales), "n"(SCALE_D_ENABLED)
);
```

## MoE Grouped GEMM Layouts

DeepGEMM provides three grouped GEMM layouts tailored for MoE workloads, where only the M-axis varies (different token counts per expert) while N and K remain fixed:

```
Layout 1: Contiguous (prefill)
  Expert 0: M0 tokens  ──┐
  Expert 1: M1 tokens  ──┤── All packed contiguously in memory
  Expert 2: M2 tokens  ──┘   Index array stores cumulative offsets

Layout 2: Masked (decode with CUDA graphs)
  Fixed-size M_max allocation per expert
  Binary mask indicates valid tokens
  Compatible with CUDA graph capture (no dynamic shapes)

Layout 3: K-grouped (weight gradients)
  Groups along K-axis instead of M-axis
  Used for computing dW in MoE training backward
```

```cpp
// Contiguous grouped GEMM dispatch
// problem_sizes[i] = {M_i, N, K} for expert i
void grouped_gemm_contiguous(
    const fp8_t* A,         // All expert inputs packed
    const fp8_t* B,         // Expert weights [num_experts, N, K]
    float* C,               // Output packed
    const int* offsets,     // Cumulative M offsets per expert
    int num_experts
) {
    // Each thread block picks an expert via tile scheduling
    // M-axis tiles distributed across experts using offset lookup
    int expert_id = binary_search(offsets, num_experts, tile_m_start);
    int local_m = tile_m_start - offsets[expert_id];

    // Standard GEMM tile with expert-specific B matrix
    compute_tile(A + offsets[expert_id] * K,
                 B + expert_id * N * K,
                 C + offsets[expert_id] * N,
                 local_m, N, K);
}
```

## JIT Compilation

DeepGEMM uses JIT compilation via NVRTC to specialize kernels per problem shape at runtime. This avoids the combinatorial explosion of pre-compiled template instantiations while still achieving optimal register allocation and loop unrolling.

```cpp
// Lightweight JIT module: compile per-shape kernel at first call
auto kernel = jit_compile(
    "deepgemm_fp8",
    {{"M", M}, {"N", N}, {"K", K},
     {"BLOCK_M", 128}, {"BLOCK_N", 128}, {"BLOCK_K", 64},
     {"NUM_STAGES", 4}}
);
kernel.launch(A, B, C, scales_a, scales_b, stream);
```

## Memory Layout

SM90 kernels use NT (non-transposed A, transposed B) layout exclusively. SM100 kernels support all four layout combinations (NT, TN, NN, TT), enabled by tcgen05.mma's flexible operand addressing.

## Performance

| GPU | Dtype | Shape | TFLOPS | Utilization |
|-----|-------|-------|--------|-------------|
| H800 | FP8 | M=4096, N=4096, K=4096 | 1550 | ~90% |

## When to Use

- FP8 inference and training where per-tensor quantization loses too much precision
- MoE expert computation with variable token counts per expert
- Situations where fine-grained (1x128 / 128x128) scale granularity is needed

## Caveats

- SM90 path is NT layout only
- JIT compilation adds first-call latency (amortized over repeated calls)
- Fine-grained scaling adds overhead vs. per-tensor scaling -- only beneficial when outlier sensitivity matters

## Sources

- [DeepGEMM GitHub](https://github.com/deepseek-ai/DeepGEMM)
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
