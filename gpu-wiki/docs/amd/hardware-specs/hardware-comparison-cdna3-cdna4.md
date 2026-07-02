# Hardware Specification Comparison: CDNA3 vs CDNA4 vs RDNA4 Architecture Parameters


**Last updated**: 2026-07-01

## Architecture Parameter Comparison Table

| Parameter | CDNA3 MI300X (gfx942) | CDNA3 MI308X (gfx942) | CDNA4 MI355X (gfx950) | RDNA4 (gfx1250) |
|------|----------------------|----------------------|----------------------|-----------------|
| Compute Units | 304 (38/XCD, 8 XCD) | 80 (20/XCD, 4 XCD) | 256 (32/XCD, 8 XCD) | — |
| Wavefront Size | 64 | 64 | 64 | 32 |
| VRAM | 192 GB HBM3 | 128 GB HBM3 | 288 GB HBM3e | — |
| LDS/CU | 64 KB | 64 KB | 160 KB | 128 KB |
| LDS Bank Count | 32 | 32 | 64 | 32 |
| LDS Read Bandwidth | 128 bytes/clock | 128 bytes/clock | 256 bytes/clock | — |
| L1 Vector Cache | 32 KB | 32 KB | 32 KB | — |
| L2 Cache | 32 MB (4/XCD) | 16 MB (4/XCD) | 32 MB (4/XCD) | — |
| L3 Cache | 256 MB | 128 MB | 256 MB | — |
| VGPR File | 512 KB | 512 KB | 512 KB | — |
| SGPR File | 12.5 KB | 12.5 KB | 12.5 KB | — |
| LLVM Target | gfx942 | gfx942 | gfx950 | gfx1250 |
| MFMA Max K | 32 (BF16) | 32 (BF16) | 128 (FP8) | 16 (FP16, WMMA) |
| FP8 Format | FNUZ (non-standard) | FNUZ (non-standard) | OCP (standard) | OCP (standard) |
| Async DMA | No | No | Yes | TDM |
| BF16 Ridge Point | ~247 | ~247 | ~629 | N/A |
| BF16 vs FP16 Performance | BF16 is faster | BF16 is faster | BF16 is faster | Same |

---

## Related

- **Index**: AMD GPU Kernel Tuning Guide — Complete tuning topic index
- **Occupancy**: [Occupancy Optimization](../common/occupancy-optimization.md) — Relationship between VGPR and occupancy
- **LDS**: [LDS Bank Conflict Optimization](../common/lds-bank-conflict-optimization.md) — Impact of different bank counts on optimization strategies
- **CDNA3 Detailed Specs**: [CDNA3 Hardware Compute Specification Table](mi300x.md)
- **CDNA4 Detailed Specs**: [CDNA4 Hardware Compute Specification Table](mi355x.md)
- **Cross-Architecture Compilation**: [Cross-Architecture Conditional Compilation](../common/hands-on/cross-architecture-conditional-compilation.md) — Adapting a single kernel to three architectures
