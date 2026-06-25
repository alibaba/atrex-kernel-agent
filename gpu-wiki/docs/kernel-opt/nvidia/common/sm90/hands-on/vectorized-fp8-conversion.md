# Vectorized FP8 Conversion via PTX

## Pattern: Efficient type conversion via PTX inline

**Source**: `cutedsl/flashinfer/`, `gluon/triton/`

```python
# Triton's FP8 ↔ FP16/BF16 conversion may generate element-wise instructions
# Manual PTX can use vectorized instructions

# PTX: one instruction converts 4 FP8 → 4 FP16
# cvt.rn.f16x2.e4m3x2 %r0, %r1;  # 2 FP8 → 2 FP16

# In CuTeDSL, cute.cast automatically selects the optimal PTX instruction
result = cute.cast(fp8_data, cute.float16)
```

**Practical experience**:
- FP8 conversion can account for 5-10% of total time in attention kernels
- Vectorized conversion is 4x faster than element-wise
- H100 supports both E4M3 and E5M2 FP8 formats, both with hardware-accelerated conversion instructions

---

## Related Documentation

- **PTX Instruction Set**: [PTX Core Instruction Set](../../../../../ref-docs/nvidia/common/ptx-instruction-set.md) — Floating-point instruction reference
- **ISA Reference**: [Hopper ISA Instruction Patterns](../../../../../ref-docs/nvidia/gluon/sm90/isa_patterns.md) — SASS instruction throughput
- **Instruction-Level Optimization**: [GPU Instruction-Level Optimization](../../../../../ref-docs/generic/gpu-instruction-optimization.md) — Precision vs. throughput trade-offs
- **Reference Kernel**: `reference-kernels/nvidia/hopper/` — 21 Hopper kernel source code
