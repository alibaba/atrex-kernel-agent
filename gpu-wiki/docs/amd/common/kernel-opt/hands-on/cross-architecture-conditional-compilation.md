# Cross-Architecture Conditional Compilation

Cross-architecture conditional compilation patterns extracted from `reference-kernels/amd/`, supporting the same kernel to adapt to CDNA3, CDNA4, and RDNA4.

---

## Pattern: Runtime Architecture Detection

**Source**: `cdna/flydsl/FlyDSL/` (multiple files supporting both CDNA3+CDNA4)

```python
# FlyDSL supports conditional compilation
@flyc.kernel
def gemm_kernel(a, b, c, ..., TARGET: flyc.constexpr):
    if TARGET == 'gfx942':  # CDNA3
        # 32 banks LDS, 64 KB/CU
        acc = flyc.mfma(a, b, acc, mfma_type='f32_16x16x32_bf16')
    elif TARGET == 'gfx950':  # CDNA4
        # 64 banks LDS, 160 KB/CU
        acc = flyc.mfma(a, b, acc, mfma_type='f32_16x16x128_f8f6f4')

# In Triton/Gluon, branching via constexpr
@triton.jit
def kernel(..., IS_CDNA4: tl.constexpr):
    if IS_CDNA4:
        # CDNA4 path: async copy + mfma_scale
        ...
    else:
        # CDNA3 path: register-based pipeline + standard mfma
        ...
```

**Practical Experience**:
- `constexpr` branches are eliminated at compile timeapr and do not affect runtime performance
- Autotune configurations for cross-architecture kernels also need to be differentiated by architecture
- Main changes in CDNA3 → CDNA4 migration: pipeline approach, LDS swizzle parameters, MFMA instruction selection

---

## Related Documentation

- **Tuning Guide**: AMD GPU Kernel Tuning Guide — CDNA3 vs CDNA4 Hardware Specification Comparison
- **Optimization Frameworks**: [AMD GPU Kernel Optimization Framework Overview](../../ref-docs/amd-kernel-optimization-frameworks.md) — FlyDSL/CK/TileLang Comparison
- **Cross-Architecture Difference Quick Reference**: [Hardware Specification Comparison](../../hardware-specs/hardware-comparison-cdna3-cdna4.md)
