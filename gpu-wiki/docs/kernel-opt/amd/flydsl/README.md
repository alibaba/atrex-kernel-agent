# FlyDSL — AMD GPU Kernel DSL

Applicability: backend: flydsl; hardware: amd; topic: optimization

FlyDSL (Flexible Layout DSL) is an MLIR-based AMD GPU kernel programming framework that implements the CuTe Layout algebra.

> The complete reference article has been migrated to [ref-docs/amd/flydsl/](../../../ref-docs/amd/flydsl/)

---

| File | Description |
|------|-------------|
| [gfx942/](gfx942/) | FlyDSL gfx942 (MI308X) specific optimizations: Chunk-GDN warp-specialized megakernel, Fused MoE |
| [gfx950/](gfx950/) | FlyDSL gfx950 (MI355X) specific optimizations: chunk-GDN fwd_h from 3.69x to 0.97x versus the Triton comparison baseline |
