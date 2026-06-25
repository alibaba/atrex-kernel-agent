# Grouped GEMM Optimization (MI308X)

## hipBLASLt Grouped GEMM

Pack matrix multiplications of different sizes into a single kernel launch to reduce launch overhead:

```bash
# Phase 1: Enable hipBLASLt
USE_HIPBLASLT_GROUPED_GEMM=1

# Phase 2: Generate tuning parameters
USE_HIPBLASLT_GROUPED_GEMM=2  # → hipblaslt_gemm_tune.txt

# Phase 3: Apply tuning parameters
USE_HIPBLASLT_GROUPED_GEMM=3
```

**Only applicable to CDNA3 architecture.** Advantages: better CU utilization, reduced kernel launch overhead, improved memory scheduling.

## llama.cpp MI308X Performance

| Model | Metric | MI308X Result | vs H100 |
|------|------|-----------|----------|
| DeepSeek V3 671B Q4_K_M | pp4096 | 1,650 t/s | +76% |
| Llama 3.1 70B Q4_K_M (FA) | pp4096 | 4,011 t/s | **+213%** |
| Llama 3.1 8B (grouped GEMM) | pp4096 | — | +29% vs rocBLAS |

**Key Optimizations**:
- `hipMemcpyAsync` calls reduced by ~10x
- Flash Attention (`-fa 1`) critical for large prompts
- Performance improves with increasing prompt length (higher arithmetic intensity better utilizes CDNA3 CUs)

---

## Related Documents

- [MI308X (CDNA3) Kernel Optimization Practices (Index)](cdna3-mi308x-kernel-practices.md) -- Index of case studies to which this document belongs
- AMD GPU Kernel Tuning Guide -- CDNA3 hardware specifications and tuning reference
- AMD GPU Kernel Optimization in Practice -- More AMD kernel optimization case studies
