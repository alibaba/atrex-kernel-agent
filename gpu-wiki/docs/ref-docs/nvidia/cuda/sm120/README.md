# SM120 CUDA — Optimization Report

Full optimization journey report for CUDA C++ / inline PTX kernels on NVIDIA RTX PRO 5000 / 4000 Blackwell-GeForce (`sm_120`).

---

| File | Kernel | dtype | Gains |
|------|--------|-------|-------|
| [sm120-nvfp4-split-k-gemv-bf16-optimization.md](sm120-nvfp4-split-k-gemv-bf16-optimization.md) | NVFP4 decode GEMV Split-K for C2-like MLP shapes | NVFP4 inputs + ue4m3 SF, FP32 partial, BF16 output | C1 CUTLASS + C2 Split-K path; TP=2 paired E2E reached `16.720385 -> 15.866395 ms/token` (`-5.107%`) under prompt and semantic gates. |
| [sm120-nvfp4-decode-gemm-production-lessons.md](sm120-nvfp4-decode-gemm-production-lessons.md) | Consolidated NVFP4 decode / prefill GEMM knowledge | NVFP4 inputs, BF16 output, CUDA Graph serving | Shape taxonomy, Split-K vs b12x boundary, CTA-3D TMA scoped win, cold-cache residency lever, structural DRAM ceiling, and reference-kernel map. |
| [sm120-rmsnorm-mlp-pdl-fusion-report.md](sm120-rmsnorm-mlp-pdl-fusion-report.md) | RMSNorm + MLP input NVFP4 quant handoff | BF16 hidden state, NVFP4 C1 input | No-PDL route is correctness-safe but served-neutral; whole-A PDL, row-chunk PDL, and C1 wait-cache remain no-go for promotion. |

## Related

- Implementation code: [reference-kernels/nvidia/blackwell-geforce/cuda/nvfp4_splitk_gemv/](../../../../../reference-kernels/nvidia/blackwell-geforce/cuda/nvfp4_splitk_gemv/)
- Pitfalls quick reference: [docs/pitfalls/nvidia/cuda/nvfp4-split-k-gemv-pitfalls.md](../../../../pitfalls/nvidia/cuda/nvfp4-split-k-gemv-pitfalls.md)
- Related scheduling background: [CUTLASS Tile Scheduling](../../cutedsl/cutlass-tile-scheduling.md)
