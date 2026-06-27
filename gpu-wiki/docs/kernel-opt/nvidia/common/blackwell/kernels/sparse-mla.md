# Sparse MLA (DeepSeek V3.2 Sparse Attention)

## Overview

Sparse Multi-head Latent Attention introduced in DeepSeek V3.2. Two-stage pipeline: (1) Lightning Indexer selects top-K tokens per query via FP8 scorer, (2) MLA runs only over selected tokens. This reduces decode attention compute from O(seqlen) to O(topk=2048) tokens, critical for long-context serving.

## Architecture

```
Query q_t (new token embedding)
   │
   ▼
┌────────────────────────┐
│ Lightning Indexer      │   FP8 scorer, per-query top-K selection
│ - FP8 KV cache          │   h64, d128, topk=2048, page_size=64
│ - Compute q·k_i scores  │
│ - Select top-2048 i     │
└────────────────────────┘
   │ selected_indices [2048]
   ▼
┌────────────────────────┐
│ Sparse MLA              │   Standard MLA but only over selected
│ - Gather selected K,V   │   h16, ckv512, kpe64, topk=2048
│ - Attention compute     │
│ - Output y_t            │
└────────────────────────┘
```

## Token Layout

Each KV cache entry is 656 bytes:
- 512 bytes: FP8 compressed KV data
- 16 bytes: FP32 per-block scale factors
- 128 bytes: BF16 RoPE embeddings (for indexer positional encoding)

Block size fixed at 64 (FlashMLA requirement).

## Kernel Patterns

### Lightning Indexer (FP8 Score Compute)
```cuda
// Score each KV block against query using FP8 block-scale MMA
// Reduce to per-block max, then top-K selection across blocks

__global__ void lightning_indexer_kernel(
    const fp8_t* q_fp8,           // [h, d] query (FP8 quantized)
    const fp8_t* kv_cache_fp8,    // [num_blocks, 64, d] paged
    const fp8_t* kv_scales,
    float* scores_out,            // [num_blocks] per-block max score
    int num_blocks
) {
    // Each threadblock handles one KV block's score
    uint32_t tmem = tmem_alloc(64);
    tcgen05_mma_f8(q_smem, k_block_smem, tmem);
    // Reduce inside block to max score
    // Write per-block score
}

// Separate top-K selection kernel across the num_blocks score array
```

### Sparse Attention Gather
```cuda
// After top-K selection gives indices, gather K,V from paged cache
// Then run standard MLA on the gathered subset

__global__ void sparse_mla_decode_kernel(
    const int* topk_indices,      // [2048] selected block indices
    const fp8_t* kv_cache,        // paged KV (656 bytes/token)
    const half* q,                // [16, 576] MLA query
    half* output
) {
    // Load q into registers/SMEM
    // For each selected block:
    //   gather K,V block (FP8) → dequant to BF16 → MMA
    // Online softmax accumulation
    // Final weighted sum
}
```

## Performance

| Variant | GPU | TFLOPS | Notes |
|---------|-----|--------|-------|
| Dense MLA decode | H800 | 660 (BF16) | 3000 GB/s, compute-bound |
| Sparse MLA decode | H800 | 410 (FP8) | Token-level sparsity |
| Sparse MLA decode | B200 | 350 (FP8) | Lower because bandwidth dominates decode |
| Dense prefill | B200 | 1460 (BF16) | tcgen05 peak |
| Sparse prefill | B200 | 1450 (FP8) | FP8 sparse matches BF16 dense |

## When To Use

- Long-context LLM serving (32K+)
- DeepSeek V3.2 and similar MLA architectures
- Serving workloads where per-token decode latency matters
