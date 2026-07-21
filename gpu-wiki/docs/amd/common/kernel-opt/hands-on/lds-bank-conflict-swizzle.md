# LDS Bank Conflict Elimination

LDS bank conflict elimination patterns extracted from `reference-kernels/amd/`, covering CDNA3 (gfx942) and CDNA4 (gfx950).

---

## Pattern: XOR16 Swizzle

**Source**: `cdna/flydsl/FlyDSL/preshuffle_gemm.py`, `cdna4/gluon/triton/`

```python
# CDNA4 LDS has 64 banks (CDNA3 has 32 banks)
# Without swizzle, column-major access will cause 64-way bank conflict

# XOR16 swizzle principle:
# XOR the high bits of the LDS address with the low bits to disperse accesses to the same bank
# addr_swizzled = addr ^ ((addr >> 4) & 0xF) << 4

# In FlyDSL
smem_a = flyc.shared_memory(shape, swizzle='xor16')

# In Gluon
# The compiler usually applies swizzle automatically, but it can be manually specified
a_smem = tl.make_block_ptr(
    base=smem_a,
    shape=(BLOCK_M, BLOCK_K),
    strides=(BLOCK_K, 1),
    offsets=(0, 0),
    block_shape=(BLOCK_M, BLOCK_K),
    order=(1, 0),  # column-major
)
```

**Practical Experience**:
- **Must** swizzle along the K dimension (consecutive threads access consecutive K elements)
- CDNA4's 64 banks means XOR16 is the optimal swizzle (CDNA3 uses XOR8)
- Bank conflicts can be monitored via `rocprofv3 --pmc SQ_LDS_BANK_CONFLICT`
- Eliminating bank conflicts can yield 20-40% LDS throughput improvement

---

## Related Documentation

- **Tuning Guide**: AMD GPU Kernel Tuning Guide — CDNA3 vs CDNA4 Hardware Specification Comparison
- **Profiling**: [AMD rocprofv3 Profiling Guide](../../ref-docs/rocprofv3-profiling-guide.md) — General rocprofv3 usage
- **General Memory Hierarchy**: [GPU Memory Hierarchy and Optimization](../../../../generic/ref-docs/gpu-memory-hierarchy.md)
