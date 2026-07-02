# SM120 NVFP4 Decode GEMV Reference Kernel Notes

This note maps the omoExplore NVFP4 decode GEMM knowledge into the reference
kernel archive. It is not another benchmark report; it tells agents which code
path to study for each reusable idea.

## Archived Code

| File | What to study |
|---|---|
| [gemm_v3_splitk_sm120.cu](gemm_v3_splitk_sm120.cu) | Minimal CUDA C++ / inline-PTX two-stage Split-K GEMV: phase-1 FP32 partials and phase-2 BF16 reduce. |
| [cutlass_splitk_dispatch_sm120.py](cutlass_splitk_dispatch_sm120.py) | vLLM dispatch wrapper pattern: C2-like decode routes to Split-K, C1 and prefill stay on stock CUTLASS. |
| [gemm_v3_splitk_cta3d_tma_sm120_shape_m1_n5120_k17408.cu](gemm_v3_splitk_cta3d_tma_sm120_shape_m1_n5120_k17408.cu) | Scoped CTA-3D TMA follow-up: fused intra-CTA Split-K reduction and B TMA with split encoded as a third tensor-map dimension. |
| [flashinfer_b12x_cta3d_dispatch_sm120.py](flashinfer_b12x_cta3d_dispatch_sm120.py) | vLLM integration pattern that keeps FlashInfer b12x as the default and routes only allowlisted small-M shapes to CTA-3D TMA. |

## Knowledge Boundaries

- The archived CUDA kernel is the clean reference for C2-like small-N / long-K
  Split-K mechanics.
- The dispatch wrapper is the reference for keeping workspace allocation out of
  the hot path and keeping C1 on the original CUTLASS graph boundary.
- CTA-3D TMA is now archived as a scoped shape-specific source snapshot. The
  smaller two-stage CUDA Split-K kernel remains the stable teaching reference.
- b12x/CuTe persistent GEMM structure should be studied under
  [reference-kernels/nvidia/blackwell-geforce/cutedsl/cutlass/](../../cutedsl/cutlass/)
  when comparing against persistent single-wave library routes.
- Diagnostic and experimental sources for adjacent lessons live in
  [nvfp4_linear_qkvz_atrex/](../nvfp4_linear_qkvz_atrex/),
  [nvfp4_prefill_gemm/](../nvfp4_prefill_gemm/), and
  [rmsnorm_mlp_nvfp4_pdl/](../rmsnorm_mlp_nvfp4_pdl/).

## When To Reuse

Use this reference kernel when the target shape has:

- `M=1`;
- small N so the base output grid under-fills SMs;
- long K with `K/S % 128 == 0`;
- stable FP32 workspace lifetime across CUDA Graph capture and replay;
- a scale-factor layout proven to match the current binary.

Do not use it as a generic NVFP4 GEMM, prefill GEMM, or large-N C1 replacement.

## Related Reports

- [SM120 NVFP4 Decode GEMM Production Lessons](../../../../../docs/nvidia/blackwell-geforce/cuda/sm120-nvfp4-decode-gemm-production-lessons.md)
- [SM120 NVFP4 Decode GEMM Production Pitfalls](../../../../../docs/nvidia/blackwell-geforce/cuda/pitfalls/sm120-nvfp4-decode-gemm-production-pitfalls.md)
