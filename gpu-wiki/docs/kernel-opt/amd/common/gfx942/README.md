# AMD gfx942 (MI308X / CDNA3) General Optimizations

MI308X (gfx942) specific non-DSL optimization documentation.

---

| File | Description |
|------|-------------|
| [MI308X Kernel Optimization Knowledge Index](cdna3-mi308x-kernel-practices.md) | Knowledge entry points, full summary links to ref-docs |
| [Flash Attention (TileLang)](cdna3-flash-attention-tilelang.md) | TileLang declarative programming, 1.53x compared to Triton |
| [Grouped GEMM](cdna3-grouped-gemm.md) | hipBLASLt grouped GEMM, llama.cpp benchmarks |
| [Composable Kernel](cdna3-composable-kernel.md) | CK programming model, TensorDescriptor transformation tree |
