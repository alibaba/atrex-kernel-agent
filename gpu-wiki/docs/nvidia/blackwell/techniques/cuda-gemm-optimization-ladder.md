# CUDA GEMM Optimization Ladder

Use this page as a simple CUDA C++ baseline or diagnostic ladder before

**Last updated**: 2026-06-30

escalating to CUTLASS/CuTe/Triton/TileLang.

## Ladder

| Step | Bottleneck question | Typical edit |
|---|---|---|
| Naive | Is the baseline correct? | one thread computes one C element |
| Coalesced global | Are warp loads contiguous? | map thread axes so adjacent lanes read adjacent addresses |
| Shared tiling | Is global traffic reused? | stage A/B tiles in shared memory |
| Thread/register tiling | Is each load reused enough? | compute multiple C elements per thread |
| Vectorized loads | Is instruction overhead or transaction width limiting? | use `float4`, `half2`, or packed loads when aligned |
| Bank conflict fix | Is shared memory serialized? | pad or swizzle shared tiles |
| Warp tiling | Is work partitioned well across lanes? | assign subtiles per warp |
| Double buffering | Are loads and compute overlapped? | pipeline global-to-shared stages |

## Minimal Coalescing Check

```cuda
// Coalesced: lane i reads element base+i.
int lane = threadIdx.x & 31;
float x = ptr[warp_base + lane];

// Suspicious: lane i reads base+i*stride.
float y = ptr[warp_base + lane * stride];
```

If this check fails, fix layout/thread mapping before increasing tile size.

## When To Stop

Stop hand-tuning scalar CUDA GEMM when the roofline says the kernel needs
tensor-core throughput. At that point, use this ladder to understand the
failure mode, then compare against CUTLASS/CuTe/Triton PR evidence for the
target architecture.


## Related

- [Cache Policy Differentiation](cache-policy.md)
- [Chunk-Based Parallelism](chunk-parallelism.md)
- [Double/Multi-Buffering Patterns](double-buffering.md)
- [Epilogue Fusion](epilogue-fusion.md)
- [Fine-Grained FP8/FP4 Quantization](fine-grained-quantization.md)
- [CUTLASS GEMM Optimization Strategy](../../common/cutedsl/cutlass-gemm-optimization.md)
- [Community GEMM Optimization Practical Summary](../../../generic/gemm-optimization-guide.md)
- [CUTLASS/CuTe Core Concepts and Layout Algebra](../../common/cutedsl/cutlass-cute-fundamentals.md)
