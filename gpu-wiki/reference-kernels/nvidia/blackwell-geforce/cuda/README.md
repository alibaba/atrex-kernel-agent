# Blackwell GeForce CUDA Kernels

CUDA C++ reference kernels for Blackwell GeForce / SM120. This collection includes CUDA approaches that do not depend on CuTeDSL, Triton, or Gluon DSL.

---

| Directory | Description |
|-----------|-------------|
| [nvfp4_splitk_gemv/](nvfp4_splitk_gemv/) | NVFP4 decode GEMV Split-K and scoped CTA-3D TMA sources: C2-like small-N / long-K shapes use CUDA Split-K or allowlisted CTA-3D TMA, while C1-like large-N shapes remain on CUTLASS / FlashInfer. |
| [nvfp4_linear_qkvz_atrex/](nvfp4_linear_qkvz_atrex/) | Diagnostic ATREX CUDA source for `linear_qkvz` `M=1..16,N=16384,K=5120`; archived as structural DRAM-bandwidth ceiling evidence, not a production route. |
| [nvfp4_prefill_gemm/](nvfp4_prefill_gemm/) | Experimental task39 SM120 NVFP4 prefill GEMM router and CUDA candidates; source-map evidence for baseline-first and fusion-boundary conclusions. |
| [rmsnorm_mlp_nvfp4_pdl/](rmsnorm_mlp_nvfp4_pdl/) | RMSNorm + MLP input NVFP4 quant, PDL handoff, row-chunk, and row-ready C1 diagnostic sources and probes. |
