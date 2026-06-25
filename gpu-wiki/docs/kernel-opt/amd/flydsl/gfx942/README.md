# FlyDSL gfx942 (MI308X / CDNA3) Specific Optimizations

FlyDSL-specific optimization cases on MI308X (gfx942).

> Complete reference articles have been migrated to [ref-docs/amd/flydsl/gfx942/](../../../../ref-docs/amd/flydsl/gfx942/)

---

| File | Description |
|------|-------------|
| [Chunk-GDN Wave-Specialized Megakernel Playbook](../../../../ref-docs/amd/flydsl/gfx942/cdna3-chunk-gdn-mi308x-wave-specialized-megakernel-optimization.md) | Skill-facing playbook: FlyDSL megakernel porting FlashQLA warp-specialization concepts on MI308X/CDNA3, covering applicable conditions, delegation contract, LDS 64KB constraint, producer wave, barrier anchor, BDV64/BDV32 shape optimization, and rocprofv3 validation |
| [Fused MoE Optimization (W4A16)](cdna3-fused-moe-flydsl.md) | FlyDSL/MLIR compilation stack MoE, W4A16 mixed precision, end-to-end +162.4% |
