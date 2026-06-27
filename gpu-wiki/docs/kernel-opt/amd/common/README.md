# AMD General Optimization Knowledge

AMD GPU general optimization framework, MFMA programming, tuning guide, and profiling tools (DSL-independent).

> Complete reference articles have been migrated to [ref-docs/amd/common/](../../../ref-docs/amd/common/)

---

| File | Description |
|------|------|
| [hands-on/](hands-on/) | AMD GPU Kernel optimization hands-on: MFMA selection, LDS swizzle, preshuffle, async DMA, MoE fusion, etc. |
| [gfx942/](gfx942/) | MI308X (CDNA3) specific non-DSL optimizations (CK, TileLang, Grouped GEMM) |
| [gfx950/](gfx950/) | MI355X (CDNA4) specific non-DSL optimizations |
| [Roofline Analysis Methodology](roofline-analysis-methodology.md) | Tile-level Roofline Model, Ridge Point, bottleneck identification, Tile Size selection, CU utilization pre-check, theoretical performance ceiling |
| [Coalesced Access and Instruction Width Optimization](coalesced-access-load-store-width.md) | Coalesced access principles, buffer_load/store dwordx4 targets, layout order matching, size_per_thread requirements |
| [Eliminating Scratch Operations (Register Spilling)](scratch-elimination-vgpr-spill.md) | VGPR spill diagnosis, VGPR limit and budget quick reference, common cause fixes, num_warps alternatives |
| [Small Matrix / Low CU Utilization Optimization](small-matrix-cu-utilization.md) | Small matrix scenario partitioning strategies: reduce tile size to increase grid parallelism, increase BLOCK_K, negative optimization avoidance |
| [Hardware Specification Comparison](../../../hardware-specs/hardware-comparison-cdna3-cdna4.md) | CDNA3 vs CDNA4 architecture parameters (CU, LDS, VGPR, Cache, etc.) |
| [Occupancy Optimization](occupancy-optimization.md) | VGPR and occupancy relationship, tips for reducing register pressure |
| [LDS Bank Conflict Optimization](lds-bank-conflict-optimization.md) | Bank architecture, conflict checking, XOR swizzle to eliminate conflicts |
| [Profiling Tools Overview](profiling-tools-overview.md) | rocprof, PyTorch Profiler, ISA/MLIR debugging, memory debugging |
