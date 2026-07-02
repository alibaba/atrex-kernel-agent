# MI308X (CDNA3/gfx942) Kernel Optimization Knowledge Index

MI308X (gfx942) kernel optimization knowledge index entry. The full optimization report and experiment summary can be found at `docs/ref-docs/`; this page only retains links to reusable topics.


**Last updated**: 2026-06-30

## Topic Index

| File | Topic | Type |
|------|------|------|
| [cdna3-chunk-gdn-mi308x-wave-specialized-megakernel-optimization.md](../../flydsl/gfx942/cdna3-chunk-gdn-mi308x-wave-specialized-megakernel-optimization.md) | Chunk-GDN megakernel (FlyDSL) | Full report, located at `ref-docs` |
| [cdna3-flash-attention-tilelang.md](cdna3-flash-attention-tilelang.md) | Flash Attention (TileLang) | Optimization knowledge |
| [cdna3-fused-moe-flydsl.md](../../flydsl/gfx942/cdna3-fused-moe-flydsl.md) | Fused MoE (FlyDSL) | Optimization knowledge |
| [cdna3-grouped-gemm.md](cdna3-grouped-gemm.md) | Grouped GEMM | Optimization knowledge |
| [cdna3-composable-kernel.md](cdna3-composable-kernel.md) | Composable Kernel (CK) Programming Model | Optimization knowledge |

## Quick Reference Dimensions

- **Hardware Resources**: LDS capacity, CU count, HBM bandwidth, MFMA instruction shapes.
- **Operator Patterns**: Flash Attention, Grouped GEMM, MoE, Chunk-GDN.
- **Framework Entry Points**: FlyDSL, TileLang, Composable Kernel.


## Related

- [Composable Kernel (CK) Programming Model (MI308X)](cdna3-composable-kernel.md)
- [Flash Attention Optimization (TileLang on MI308X)](cdna3-flash-attention-tilelang.md)
- [Grouped GEMM Optimization (MI308X)](cdna3-grouped-gemm.md)
- [FlyDSL Programming Guide](../../flydsl/flydsl-programming-guide.md)
- [CuTeDSL Gated DeltaNet Chunk Forward (bf16, Precomputed Neumann) on SM120](../../../nvidia/blackwell-geforce/cutedsl/sm120-gdn-chunk-fwd-bf16-neumann-optimization.md)
