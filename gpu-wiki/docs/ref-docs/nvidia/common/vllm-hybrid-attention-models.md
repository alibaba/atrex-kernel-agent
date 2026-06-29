# Hybrid Attention Models as First-Class Citizens in vLLM

This document covers the integration of hybrid attention models (combining standard attention with Mamba/linear attention) into vLLM V1, including the unified allocator design, stride alignment, CUDA Graph optimization, and benchmark results.

---

## 1. The Fundamental Bottlenecks of Attention

KV cache memory consumption grows linearly with sequence length, and prefill latency scales quadratically with prompt length. For contexts exceeding 120K tokens, new architectural paradigms become necessary. **Hybrid attention** architectures (Qwen3-Next, Nemotron Nano 2, MiniMax-Text-01, Granite 4.0) fuse standard attention with Mamba or linear attention, making long-sequence inference both efficient and practical. The vLLM community has elevated hybrid attention models from an experimental "patch" in V0 to a first-class citizen in V1.

Engineering optimizations (PagedAttention, chunked online softmax, Tensor Cores, TMA, quantization) cannot eliminate two fundamental constraints:

1. **KV cache scales linearly with length x batch**: Each generated token appends new key-value pairs; at hundreds of thousands of tokens, the cache consumes all available memory.
2. **Prefill TTFT scales quadratically with prompt length**: Prompts exceeding 120K tokens make inference impractical.

## 2. Why Long Sequences Matter

- **RAG**: User queries concatenated with multiple documents push prompts from thousands to hundreds of thousands of tokens.
- **Agents**: Tool-calling loops append context each round.
- **Reasoning**: Chain-of-thought forces all intermediate tokens to remain in context.

## 3. A Brief History of SSMs

- **S4 (2021)**: A recurrent structure that maps length-T input sequences to outputs through a latent state h. Complexity is linear in T; latent dimension N is fixed regardless of sequence length. Weak at selective copying and in-context reasoning.
- **Mamba-1 (2023)**: Makes matrices A/B/C input-dependent per step, enabling selective attention. However, the parallel implementation underutilizes matrix multiplication and Tensor Cores.
- **Mamba-2 (2024)**: Reveals that SSMs can be expressed as input-to-output matrix transformations. Adds more structure to A for highly efficient implementations. Equivalent to Linear Attention.
- **Linear Attention (2020)**: Approximates softmax as a linear dot product of kernel feature maps. Variants include Lightning Attention (MiniMax-Text-01) and Gated Delta Net (Qwen3-Next).

## 4. V1 Landscape

V1 currently supports hybrid attention models including Nemotron Nano 2, Granite 4.0, Qwen3-Next, MiniMax-Text-01, and Kimi Linear. Hybrid architectures are not a niche experiment but an active design choice across organizations.

## 5. State Management: Merging Two Worlds

- **Attention KV cache**: 16-token blocks, approximately 64 KiB per block.
- **Mamba state**: One large fixed-size state per sequence (approximately 2.57 MiB), updated in-place.

At 128K token sequences, KV cache can be nearly 200x larger than Mamba state — this is precisely the selling point of hybrid approaches.

## 6. V0: The Fragile Patch

Mamba states were allocated separately, with size guessed from `max_num_seqs`: too high causes OOM, too low reduces concurrency. Poor user experience.

## 7. V1: The Unified Allocator

The unified allocator manages both KV cache and Mamba state simultaneously. The original V1 "hybrid allocation" only supported combinations like "full attention + sliding window attention" (Gemma 3, Llama 4, gpt-oss). To support Mamba:

1. **Relaxed same-block-size requirement**: Attention block size is automatically enlarged to align with Mamba pages (e.g., Nemotron-Nano-12B-v2: 672 tokens per attention block). Mamba pages are slightly padded to match.
2. **Block size decoupling**: The block size used for KV cache management is separated from the block size seen by kernels, allowing FlashInfer/TRT-LLM kernels that do not support arbitrary block sizes on Blackwell to still function.

Empirical testing shows irregular block sizes have minimal performance impact (when Mamba/linear layers dominate, attention runtime is a small fraction).

## 8. Stride Adjustment: Perfect Alignment Details

KVCacheGroups share the same KVCacheTensor but with different view layouts:

- **Attention** view (FlashInfer backend): K/V interleaved, stored block by block.
- **Mamba** view: All Conv state blocks followed by all SSM state blocks, not interleaved by default.

Writing a block through the Mamba view would corrupt data in "another block" from the Attention view. The fix adjusts Mamba tensor strides so both views align perfectly. The same applies to FlashAttention: an additional stride adjustment to the Attention view ensures it can also serve as a hybrid attention model backend (FlashAttention is the default vLLM backend).

## 9. Performance Engineering

**Triton launch overhead** (triton#2637): Significant impact on ITL for small batches and models with few active parameters. The Mamba backend introduced CUDA Graph support in stages:

```
eager → piecewise → decode-only full → FULL_AND_PIECEWISE (full graph for decode + piecewise graph for mixed batches)
```

With FULL_AND_PIECEWISE enabled by default, V1 recovers and significantly surpasses V0 performance. Prefix caching for Mamba-2 hybrid models is also supported (experimental stage).

## 10. Benchmarks (vLLM v0.10.2, H100)

Input 32K + output 128, sweeping concurrency:

- **Nemotron-Nano-12B-v2** (dense): FULL_AND_PIECEWISE delivers **+2% to +18%** throughput overall.
- **Granite-4.0-h-tiny** (7B/1B active MoE): At low concurrency, FULL_AND_PIECEWISE achieves up to **+91%** throughput; TTFT and ITL also decrease significantly.

PIECEWISE alone at very low concurrency is actually slower than V0 (CPU launch overhead); FULL_AND_PIECEWISE resolves this.

## 11. Conclusion

Hybrid attention models in V1 benefit from unified memory allocation, CUDA Graph optimization, prefix caching, KV migration, and PD disaggregation. For enterprise-grade long-context workloads, hybrid architectures are now a practical tool.
