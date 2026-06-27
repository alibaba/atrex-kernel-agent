# SM120 Triton — Optimization Report

A complete optimization journey report for Triton kernels on NVIDIA RTX PRO 5000 / 4000 Blackwell-GeForce (`sm_120`).

| File | Kernel | dtype | Gains |
|------|--------|-------|------|
| [sm120-fused-rmsnorm-gated-bf16-optimization.md](sm120-fused-rmsnorm-gated-bf16-optimization.md) | vLLM `_deltanet_post` RMSNormGated + SiLU(z) gating fusion | bf16 in/out, NVFP4 downstream unchanged | 4 version iterations: V1 baseline 3.35× → V2 cache hint 0% → V3 sweep `BLOCK_M=2` **3.87× over V2**. Kernel-only **13.01×** vs eager; **end-to-end deltanet_forward 2.16×** (2402 us → 1112 us); bandwidth 122% memcpy ceiling (R:W=2:1 outperforms ceiling assumption of 1:1). Key techniques: BLOCK_M sweep + `cache_modifier=".cg"` + `multiple_of` / `max_contiguous` vectorization hints. **Strong lesson**: cache hint + LDG.128 hint alone yield zero gains but must be retained; BLOCK_M is the dominant tuning knob for elementwise+reduce kernels like sm_120. |

## Related

- Implementation code: [reference-kernels/nvidia/blackwell-geforce/triton/](../../../../../reference-kernels/nvidia/blackwell-geforce/triton/)
- Quick reference for pitfalls: [docs/pitfalls/nvidia/triton/](../../../../pitfalls/nvidia/triton/)
- CuTeDSL path on same hardware (true fusion / different path): [docs/ref-docs/nvidia/cutedsl/sm120/](../../cutedsl/sm120/)
