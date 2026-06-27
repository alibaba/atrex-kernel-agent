# RDNA4-Specific Optimizations

RDNA4 (gfx1250)-specific optimization patterns extracted from `reference-kernels/amd/`, including WMMA matrix multiplication and LDS transpose loading.

---

## Pattern: WMMA (Wave32) Matrix Multiplication

**Source**: `rdna4/flydsl/FlyDSL/`, `rdna4/gluon/triton/`

```python
# RDNA4 uses Wave32 (32 threads/wave, CDNA uses 64 threads/wavefront)
# WMMA instruction: v_wmma_f32_16x16x16_f16

# FlyDSL
@flyc.kernel(target='gfx1250')
def rdna4_gemm(a, b, c, ...):
    # 16×16×16 tile per WMMA instruction
    acc = flyc.wmma(a_tile, b_tile, acc)
```

---

## Pattern: ds_load_tr16_b128 Transpose Load

```python
# RDNA4 new LDS transpose load instruction
# One instruction completes 128-bit load + 16-bit element transpose
# Replaces the ds_read + ds_bpermute combination used in CDNA

# Scenario: load the transpose of matrix B (for WMMA's B operand)
b_transposed = ds_load_tr16_b128(lds_b_ptr)
```

**Practical Experience**:
- RDNA4's WMMA throughput is lower than CDNA4's MFMA (consumer-oriented)
- Wave32 means more waves are needed to fill a CU
- `ds_load_tr16_b128` eliminates the overhead of B matrix transposition
- RDNA4's LDS is 128 KB/CU (larger than CDNA3's 64 KB)

---

## Related Documents

- **MFMA Instruction Reference**: [AMD MFMA Matrix Core Programming Guide](../../../../ref-docs/amd/common/amd-mfma-matrix-cores.md) — instruction naming rules, register layout
- **Optimization Frameworks**: [AMD GPU Kernel Optimization Framework Overview](../../../../ref-docs/amd/common/amd-kernel-optimization-frameworks.md) — FlyDSL/CK/TileLang comparison
- **Cross-Architecture Differences Quick Reference**: [Hardware Specification Comparison](../../../../hardware-specs/hardware-comparison-cdna3-cdna4.md)
