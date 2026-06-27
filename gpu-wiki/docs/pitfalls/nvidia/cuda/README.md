# CUDA Pitfalls

Implementation and integration pitfalls encountered with CUDA C++ / inline PTX kernels.

---

| File | Kernel | Hardware | Trap count |
|------|--------|----------|-----------|
| [nvfp4-split-k-gemv-pitfalls.md](nvfp4-split-k-gemv-pitfalls.md) | NVFP4 decode GEMV Split-K with vLLM CUTLASS dispatch | sm_120 (RTX PRO 5000 Blackwell-GeForce) | 7 |
| [sm120-nvfp4-decode-gemm-production-pitfalls.md](sm120-nvfp4-decode-gemm-production-pitfalls.md) | Consolidated NVFP4 decode / prefill GEMM production lessons | sm_120 (RTX PRO 5000 Blackwell-GeForce) | 10 |
| [sm120-rmsnorm-mlp-pdl-pitfalls.md](sm120-rmsnorm-mlp-pdl-pitfalls.md) | RMSNorm + MLP input NVFP4 quant PDL handoff | sm_120 (RTX PRO 5000 Blackwell-GeForce) | 7 |
