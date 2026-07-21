# Fused MoE Optimization (FlyDSL on MI308X)

Applicability: backend: flydsl; hardware: amd; topic: optimization

## Bottleneck Analysis

In the Kimi-K2.5 model, `fused_moe` accounts for **87.8%** (concurrency=2) to **89.7%** (concurrency=40) of GPU time.

## FlyDSL Programming Model

FlyDSL is based on the MLIR compilation stackNord, using FLIR (Flexible Layout IR) — a layout algebra system inspired by CuTe:

- **Python Native Development**: Written via the `flydsl` package
- **Hierarchical Control**: Block → Warp → Thread decomposition, explicit MFMA instruction mapping
- **Data Movement**: global → LDS → register three-level
- **Compilation Pipeline**: Python → MLIR → Canonicalization/CSE → GPU-to-ROCDL → Binary

Target architectures: `gfx942` (MI308X) and `gfx950` (MI350/MI355X).

## Mixed Precision Strategy

```bash
export FLYDSL_W4A16_HYBRID=w2_bf16
```

- Stage 1 (gate/up projection): W4A16 (4-bit quantized weights, 16-bit activations)
- Stage 2 (down projection): BF16 (full precision)
- Trade-off: Slightly more memory in exchange for better compute throughput and numerical stability

## MoE Key Shapes

| Configuration | tokens | model_dim | inter_dim | Experts | topk |
|------|--------|-----------|-----------|---------|------|
| Large (Dominant) | 16384 | 7168 | 512 | 384 | 8 |
| Medium | 2048 | 4096 | 512 | 64 | 8 |
| Small | 512 | 2048 | 512 | 64 | 4 |

## Kernel-Level Performance

Most critical shape (tokens=16384, E=384, topk=8):

The Triton numbers below are comparison baselines only; this page's target
implementation path is FlyDSL on AMD hardware.

| dtype | Torch | Triton | CK | FlyDSL |
|-------|-------|--------|-----|--------|
| BF16 | 119.82ms | 12.09ms | GPU fault | **8.68ms** |
| W4A16 | 131.33ms | 31.43ms | Not Supported | **9.77ms** |

FlyDSL versus the Triton comparison baseline: BF16 **1.39x**, W4A16
**3.22x** speedup.

## End-to-End Performance

Concurrency=40 (decode-dominant):

| Metric | Baseline | Optimized | Improvement |
|------|---------|--------|------|
| TTFT Mean | 33479ms | 17730ms | -47.0% |
| TPOT Mean | 230.37ms | 70.86ms | **-69.2%** |
| Throughput | 135.39 tok/s | 355.35 tok/s | **+162.4%** |

---

## Related Documents

- [MI308X (CDNA3) Kernel Optimization Practices (Index)](../cdna3-mi308x-kernel-practices.md) -- Index of the case study collection this article belongs to
- [AMD GPU Kernel Optimization Framework Overview](../../../../common/ref-docs/amd-kernel-optimization-frameworks.md) -- FlyDSL's position within the AMD optimization framework
- [AMD MFMA Matrix Core Programming Guide](../../../../common/ref-docs/amd-mfma-matrix-cores.md) -- MFMA instruction reference used by FlyDSL
