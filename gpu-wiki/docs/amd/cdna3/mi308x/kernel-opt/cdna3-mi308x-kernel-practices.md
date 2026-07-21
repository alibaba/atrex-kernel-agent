# MI308X (CDNA3/gfx942) Kernel Optimization Knowledge Index

MI308X (gfx942) kernel optimization knowledge index entry. The full optimization report and experiment summary can be found in the sibling `ref-docs/` directory; this page only retains links to reusable topics.

## Topic Index

| File | Topic | Type |
|------|------|------|
| [cdna3-chunk-gdn-mi308x-wave-specialized-megakernel-optimization.md](../ref-docs/flydsl/cdna3-chunk-gdn-mi308x-wave-specialized-megakernel-optimization.md) | Chunk-GDN megakernel (FlyDSL) | Full report, located at `ref-docs` |
| [cdna3-flash-attention-tilelang.md](cdna3-flash-attention-tilelang.md) | Flash Attention (TileLang) | Optimization knowledge |
| [cdna3-fused-moe-flydsl.md](flydsl/cdna3-fused-moe-flydsl.md) | Fused MoE (FlyDSL) | Optimization knowledge |
| [cdna3-grouped-gemm.md](cdna3-grouped-gemm.md) | Grouped GEMM | Optimization knowledge |
| [cdna3-composable-kernel.md](cdna3-composable-kernel.md) | Composable Kernel (CK) Programming Model | Optimization knowledge |

## Quick Reference Dimensions

- **Hardware Resources**: LDS capacity, CU count, HBM bandwidth, MFMA instruction shapes.
- **Operator Patterns**: Flash Attention, Grouped GEMM, MoE, Chunk-GDN.
- **Framework Entry Points**: FlyDSL, TileLang, Composable Kernel.
