# NVIDIA Pitfalls

Non-obvious traps encountered while implementing or porting GPU kernels on NVIDIA Blackwell hardware.

---

| Directory | Description |
|-----------|-------------|
| [cuda/](cuda/) | CUDA C++ / inline PTX pitfalls (NVFP4 GEMM, RMSNorm-MLP PDL) |
| [cutedsl/](cutedsl/) | CuTeDSL / CUTLASS pitfalls (GDN chunk/decode, NVFP4 GEMM, TMA warp-spec) |
| [gluon/](gluon/) | Gluon pitfalls (Blackwell primitives, tcgen05_mma, layout rules) |
| [triton/](triton/) | Triton pitfalls (fused RMSNorm, sparse decode split-K) |
