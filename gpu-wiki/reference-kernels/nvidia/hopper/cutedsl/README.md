# Hopper CuTeDSL Kernels

CuTeDSL reference kernels on the Hopper (SM90) architecture, organized by source repository.

---

| Directory | Description |
|-----------|-------------|
| [cutlass/](cutlass/) | CUTLASS framework reference kernels (GEMM, FMHA, Norm) |
| [flash-attention/](flash-attention/) | Flash Attention SM90 forward/backward reference implementations |
| [flashinfer/](flashinfer/) | FlashInfer reference kernels (GDN Decode, Norm, SSD) |
| [quack/](quack/) | QuACK high-performance reduction / GEMM kernels |
| [tilelang/](tilelang/) | TileLang CuTeDSL backend: inline PTX utility library (atomic, ldsm, mma, ieee math, quantize, grid sync, etc.) |
