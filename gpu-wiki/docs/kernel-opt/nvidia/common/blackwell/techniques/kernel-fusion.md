# Kernel Fusion

## Overview

Kernel fusion combines multiple operations into a single kernel launch, eliminating intermediate global memory roundtrips. Critical for MoE and attention pipelines where 5-7 sequential launches each incur latency, synchronization, and memory traffic overhead.

## Examples

### Fused Gate-Up Dual GEMM + SwiGLU
```cuda
// Instead of: gate_gemm → up_gemm → silu → multiply (4 kernels)
// Fused: single kernel with two TMEM accumulators
__global__ void fused_gate_up_silu(...) {
    uint32_t tmem_gate = tmem_alloc(256);
    uint32_t tmem_up = tmem_alloc(256);

    for (int k = 0; k < K; k += BLOCK_K) {
        tcgen05_mma(x_smem, w_gate_smem, tmem_gate);
        tcgen05_mma(x_smem, w_up_smem, tmem_up);
    }

    float g = tmem_load(tmem_gate);
    float u = tmem_load(tmem_up);
    output = (g / (1.0f + expf(-g))) * u;  // SwiGLU fused
}
```

### MoE Fusion Progression
- **vLLM (7 kernels)**: softmax → topk → dispatch → gate → up → silu_mul → down+combine
- **SGLang (5 kernels)**: router+topk → dispatch → fused gate-up-silu → down → combine
- **Ideal (1-2 kernels)**: all ops in one launch, saves 21.9% activation memory traffic

## Constraints

- TMEM capacity limits how many accumulators can fuse (256 cols total)
- Register pressure on epilogue if fusing complex activations
- Fusion opportunities depend on dataflow shape (dependency graph must be DAG-compatible with CTA scope)

## Related
- [fused-moe](../kernels/fused-moe.md)
- [epilogue-fusion](epilogue-fusion.md)
