# SM120 NVFP4 Linear QKVZ ATREX Diagnostics

> **Usability status:** `diagnostic-archive`
>
> This package captures a specific SM120 investigation. It may require `$CUTLASS_DIR` and shape-specific assumptions before it can run.

ATREX CUDA source snapshots for the `linear_qkvz` NVFP4 decode GEMM shape
family on NVIDIA Blackwell-GeForce / SM120.

These files are archived as diagnostic source for the structural DRAM-bandwidth
ceiling lesson. They are not production kernels: the all-shape acceptance gate
was not met, and the best env-gated route remained a partial operator-only
deliverable.

---

| File | Scope |
|---|---|
| [nvfp4gemm_linear_qkvz_atrex_sm120_diagnostic.py](nvfp4gemm_linear_qkvz_atrex_sm120_diagnostic.py) | JIT/build wrapper and environment knob surface for the ATREX Split-K CUDA source. |
| [nvfp4gemm_splitk_linear_qkvz_atrex_sm120_shape_m1_16_n16384_k5120_diagnostic.cu](nvfp4gemm_splitk_linear_qkvz_atrex_sm120_shape_m1_16_n16384_k5120_diagnostic.cu) | CUDA Split-K / CTA-3D / A-staging diagnostic source for `M=1..16,N=16384,K=5120`. |

## Related Docs

- [SM120 NVFP4 Decode GEMM Production Lessons](../../../../../docs/ref-docs/nvidia/cuda/sm120/sm120-nvfp4-decode-gemm-production-lessons.md)
- [SM120 NVFP4 Decode GEMM Production Pitfalls](../../../../../docs/pitfalls/nvidia/cuda/sm120-nvfp4-decode-gemm-production-pitfalls.md)
