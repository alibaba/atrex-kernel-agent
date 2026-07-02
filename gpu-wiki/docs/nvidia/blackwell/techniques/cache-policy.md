# Cache Policy Differentiation


**Last updated**: 2026-06-30

## Overview

PTX cache qualifiers (`L1::no_allocate`, `L1::evict_last`, `L1::evict_first`) let kernels hint to hardware how to handle cache admission for specific loads. Critical for memory-bound kernels where the L1 working set matters more than the compute.

## Pattern

```asm
; Matrix A (streamed once per row, never reused): bypass L1
; Avoids polluting L1 with one-shot data
ld.global.L1::no_allocate.v4.u64 {a0,a1,a2,a3}, [addr_a];

; Vector B (reused across BLOCK_M rows): keep in L1
ld.global.L1::evict_last.v4.u64 {b0,b1,b2,b3}, [addr_b];

; Streaming output: evict immediately after write
st.global.L1::evict_first.v2.u64 [addr_c], {c0, c1};
```

## GPU Mode NVFP4 GEMV Winner Technique

Rank 1 submission used **different qualifiers per K-dimension variant**:
- K=16384 (large): aggressive `L1::no_allocate` on A (huge streaming matrix)
- K=2048 (small): relaxed balance since B is smaller relative to cache

## Measurable Impact

- NVFP4 GEMV: 443μs → 27μs (16x improvement) came partly from cache policy + PTX byte unpacking
- On memory-bound kernels, cache policy can be the dominant lever

## When To Use

- Memory-bound kernels (profile with Nsight Compute first)
- Tensor with clear "streaming" vs "reused" access patterns
- Inputs > L2 cache size (B200: 126MB)
- Separate M and N tile loading patterns in GEMM


## Related

- [Chunk-Based Parallelism](chunk-parallelism.md)
- [CUDA GEMM Optimization Ladder](cuda-gemm-optimization-ladder.md)
- [Double/Multi-Buffering Patterns](double-buffering.md)
- [Epilogue Fusion](epilogue-fusion.md)
- [Fine-Grained FP8/FP4 Quantization](fine-grained-quantization.md)
- [PTX Programming Model and Basics](../../common/ptx/ptx-programming-model.md)
- [PTX Core Instruction Set](../../common/ptx/ptx-instruction-set.md)
