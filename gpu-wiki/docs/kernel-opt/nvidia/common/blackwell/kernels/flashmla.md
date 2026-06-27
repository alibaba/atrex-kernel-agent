# FlashMLA -- Multi-head Latent Attention

## Overview

FlashMLA provides high-performance kernels for DeepSeek's Multi-head Latent Attention (MLA) mechanism, which compresses the KV cache from 327-516 KB/token (standard MHA) down to 70 KB/token through a learned low-rank projection into a latent space. This extreme compression (4.66-7.28x reduction) is critical for serving DeepSeek-V3/V3.2 models at scale.

FlashMLA includes four kernel variants: dense MLA decoding (SM90), sparse MLA decoding (SM90/SM100), dense MLA prefill (SM100), and sparse MLA prefill (SM90/SM100).

## MLA KV Cache Layout

Each token in the MLA KV cache occupies 656 bytes:

```
Token KV Cache Entry (656 bytes total):
+------------------------------------------+
| FP8 compressed KV data    | 512 bytes    |  <-- Latent KV representation
| FP32 scaling factors      |  16 bytes    |  <-- Per-head scales
| BF16 RoPE embeddings      | 128 bytes    |  <-- Position encodings
+------------------------------------------+

Paged KV cache:
  Page size = 64 tokens (dense) or variable (sparse)
  Each page = 64 * 656 = 41,984 bytes
```

## Dense MLA Decoding (SM90)

The decode kernel targets memory-bound inference with paged KV cache (block size 64). It achieves up to 3000 GB/s memory bandwidth and 660 TFLOPS on H800.

```cpp
// Dense MLA decode kernel structure (SM90, BF16)
// Memory-bound: bandwidth utilization is the primary metric

template <int HEAD_DIM, int BLOCK_KV>
__global__ void flashmla_decode_dense(
    const half* __restrict__ Q,       // [batch, num_heads, head_dim]
    const int8_t* __restrict__ KV,    // Paged KV cache (FP8)
    const float* __restrict__ scales, // Per-head scales
    const half* __restrict__ rope,    // RoPE embeddings
    const int* __restrict__ page_table,
    half* __restrict__ O,
    float* __restrict__ L              // Log-sum-exp
) {
    // Each warpgroup handles one query head
    const int head_id = blockIdx.x;
    const int batch_id = blockIdx.y;

    // Load query into registers
    half Q_reg[HEAD_DIM];
    load_query(Q, batch_id, head_id, Q_reg);

    float acc[HEAD_DIM] = {0.0f};
    float lse = -INFINITY;

    // Iterate over KV pages
    for (int page = 0; page < num_pages; page++) {
        int page_idx = page_table[batch_id * max_pages + page];

        // TMA load KV page into shared memory
        __shared__ int8_t KV_smem[BLOCK_KV * 656];
        tma_load_async(KV_smem, KV + page_idx * BLOCK_KV * 656);
        cp_async_wait();

        // Compute attention scores for this page
        for (int t = 0; t < BLOCK_KV; t++) {
            // Dequantize KV: fp8 -> bf16, apply scale
            half K_token[HEAD_DIM], V_token[HEAD_DIM];
            dequant_kv(KV_smem + t * 656, scales, K_token, V_token);

            // Apply RoPE
            apply_rope(K_token, rope + (page * BLOCK_KV + t) * 64);

            // Score and accumulate (online softmax)
            float score = dot_product(Q_reg, K_token, HEAD_DIM);
            float new_lse = logaddexp(lse, score);
            float rescale = exp(lse - new_lse);
            for (int d = 0; d < HEAD_DIM; d++)
                acc[d] = acc[d] * rescale + exp(score - new_lse) * V_token[d];
            lse = new_lse;
        }
    }

    // Write output
    store_output(O, batch_id, head_id, acc);
    L[batch_id * num_heads + head_id] = lse;
}
```

## Sparse MLA (SM90/SM100)

Sparse MLA uses token-level sparsity indices to select only relevant tokens from the KV cache, dramatically reducing memory reads for long sequences.

```cpp
// Sparse MLA: only attend to selected tokens via indices tensor
// Each query has a variable-length list of relevant token indices

template <int HEAD_DIM>
__global__ void flashmla_sparse(
    const half* Q,
    const int8_t* KV_cache,
    const float* scales,
    const int* token_indices,   // Selected token indices per query
    const int* num_selected,    // Number of selected tokens per query
    half* O
) {
    const int query_id = blockIdx.x;
    const int n_tokens = num_selected[query_id];

    // Only load and compute on selected tokens
    for (int i = 0; i < n_tokens; i += BLOCK_SIZE) {
        int tok_idx = token_indices[query_id * MAX_TOKENS + i];

        // Load only the selected token's KV entry (656 bytes)
        load_kv_entry(KV_cache, tok_idx, K_local, V_local);
        dequant_and_accumulate(Q, K_local, V_local, scales, acc, lse);
    }
}
```

## Dense Prefill (SM100)

The SM100 prefill kernel leverages tcgen05.mma and TMEM for the compute-heavy forward and backward passes, achieving 1460 TFLOPS forward and 1000 TFLOPS backward on B200.

```cpp
// SM100 dense prefill: tcgen05.mma with TMEM accumulation
// Uses warp specialization: TMA warps + MMA warps + softmax warps

// Forward pass structure:
// 1. TMA loads Q, K, V tiles into SMEM
// 2. tcgen05.mma computes S = Q @ K^T into TMEM
// 3. Softmax warpgroup applies online softmax on TMEM data
// 4. tcgen05.mma computes O = softmax(S) @ V into TMEM
// 5. TMEM -> SMEM -> Global memory for output

// Key: TMEM holds both S matrix and O accumulator
// No register spill for large tile sizes
```

## Performance

| Variant | GPU | Dtype | TFLOPS | Bandwidth |
|---------|-----|-------|--------|-----------|
| Dense decode | H800 | BF16 | 660 | 3000 GB/s |
| Sparse decode | H800 | FP8 | 410 | -- |
| Sparse decode | B200 | FP8 | 350 | -- |
| Dense prefill fwd | B200 | BF16 | 1460 | -- |
| Dense prefill bwd | B200 | BF16 | 1000 | -- |
| Sparse prefill | H800 | FP8 | 640 | -- |
| Sparse prefill | B200 | FP8 | 1450 | -- |

## Architecture Integration

FlashMLA is deployed in production for DeepSeek-V3 and V3.2 inference:
- SGLang and vLLM provide day-0 support
- CUTLASS SM100 includes MLA attention kernels with fused reduction
- FlashMLA sparse kernels are used by DeepSeek-V3.2-Exp with NSA

## When to Use

- DeepSeek-V3/V3.2 model serving with MLA architecture
- Long-context inference where KV cache size is the bottleneck
- Combined with NSA for sparse attention on long sequences

## Caveats

- MLA-specific: the latent KV cache format (656 bytes/token) is tied to DeepSeek's architecture
- Dense prefill is SM100 only
- Sparse MLA requires a separate indexing pass to select relevant tokens

## Sources

- [FlashMLA GitHub](https://github.com/deepseek-ai/FlashMLA)
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
- [CUTLASS SM100 Attention Changelog](https://docs.nvidia.com/cutlass/latest/CHANGELOG.html)
