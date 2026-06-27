# AMD gfx942 (MI300X / CDNA3) Gluon Optimization

Kernel optimization knowledge for Gluon DSL on MI300X (gfx942).

> Full reference articles have been migrated to [ref-docs/amd/gluon/gfx942/](../../../../ref-docs/amd/gluon/gfx942/)

---

| File | Description |
|------|------|
| [Optimization Guide](optimization-guide.md) | CDNA3 Gluon kernel optimization guide |
| [Optimization Strategy](optimization_strategy.md) | Small matrix GEMM optimization priority table |
| [Small Matrix GEMM Topic](pattern_overview.md) | Specialized matmul scenarios and pitfalls |
| [SE-level Zigzag](se_level_zigzag.md) | Causal Attention load balancing |
| [Configuration Template](final_config_template.md) | 256x256x64 GEMM configuration and stopping conditions |
| [ROCprof Trace Decoder](rocprof-trace-decoder.md) | rocprofiler-sdk ATT data decoding |

Full summary migrated to ref-docs:

| Reference Article | Description |
|------|------|
| [Key Conclusions](../../../../ref-docs/amd/gluon/gfx942/key_conclusions.md) | CDNA3 Gluon optimization conclusion summary |
| [Flash Attention Results](../../../../ref-docs/amd/gluon/gfx942/optimization_results.md) | Triton vs Gluon-v1 vs Gluon-v2 comparison |
| [Changelog](../../../../ref-docs/amd/gluon/gfx942/CHANGELOG.md) | ROCm 7.0 trace decoder changes |
