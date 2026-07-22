# Preshuffle B Layout

A pre-permuted weight matrix optimization pattern extracted from `reference-kernels/amd/`, avoiding runtime layout conversion overhead.

---

## Pattern: Pre-Permute Weight Matrix to Avoid Runtime Layout Conversion

**Source**: `cdna/flydsl/FlyDSL/preshuffle_gemm.py`

```python
# Problem: MFMA instructions require a specific operand layout
# Standard row-major B matrix requires runtime conversion → many ds_bpermute instructions

# Solution: Pre-shuffle the B matrix on the host side into an MFMA-friendly layout
def preshuffle_b(b_matrix, mfma_shape):
    """Pre-shuffle the B matrix into a layout directly usable by MFMA"""
    # Chunk by MFMA tile size
    # Rearrange elements within each block into a register-direct layout
    # Allows MFMA to read directly from registers without ds_bpermute
    ...
    return b_preshuffled

# In the kernel: directly load the pre-shuffled B, skip layout conversion
b_tile = tl.load(b_preshuffled_ptr + offsets)
c = tl.dot(a_tile, b_tile)  # No ds_bpermute overhead
```

**Practical Experience**:
- Preshuffle is performed on the host side (one-time cost), eliminating the need for `convert_layout` in the kernel
- Eliminating `ds_bpermute_b32` instructions can yield a 10-20% performance improvement
- Suitable for inference scenarios (fixed weights), not suitable for training (weights updated every step)
- Check via ATT trace: if you see a large number of `ds_bpermute_b32`, consider preshuffle

---

## Related Documents

- **MFMA Instruction Reference**: [AMD MFMA Matrix Core Programming Guide](../../ref-docs/amd-mfma-matrix-cores.md) — Instruction naming rules, register layout
- **MFMA Instruction Selection**: [MFMA Instruction Selection and Usage](mfma-instruction-selection.md)
- **CDNA3 ISA**: [CDNA3 ISA Instruction Patterns](../../../cdna3/ref-docs/gluon/isa_patterns.md)
