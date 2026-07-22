# Blackwell GeForce Reference Kernels

A collection of GPU reference kernels for the NVIDIA Blackwell GeForce/workstation SM120 architecture, organized by DSL framework.

---

| Directory | Description |
|------|------|
| [cuda/](cuda/) | CUDA C++ / inline PTX kernels and vLLM integration sources (NVFP4 decode Split-K / CTA-3D TMA, prefill GEMM experiments, RMSNorm-MLP PDL diagnostics) |
| [cutedsl/](cutedsl/) | CuTeDSL framework kernels (CUTLASS, Flash Attention, FlashInfer, task39 diagnostic b12x fork, GDN chunk fwd V113) |
| [triton/](triton/) | Triton framework kernels (vLLM GDN post-processing fused norm+gate; tuned for sm_120) |
