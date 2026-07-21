# MFMA Instruction Selection and Usage

Extracting MFMA instruction selection optimization patterns from `reference-kernels/amd/`, covering CDNA3 (gfx942) and CDNA4 (gfx950).

---

## Pattern: CDNA3 vs CDNA4 MFMA Instruction Differences

**Source**: `cdna/flydsl/FlyDSL/`, `cdna4/gluon/triton/`

```python
# CDNA3 (gfx942): Standard MFMA
# v_mfma_f32_16x16x32_bf16  — 16×16 output, K=32, BF16 input
# v_mfma_f32_32x32x16_bf16  — 32×32 output, K=16, BF16 input

# CDNA4 (gfx950): New mfma_scale and larger K dimensions
# v_mfma_f32_16x16x128_f8f6f4  — K=128, supports FP8/FP6/FP4 mixed precision
# v_mfma_scale_f32_16x16x128_f8f6f4  — MFMA with block scale
```

**MFMA Calls in FlyDSL**:

```python
@flyc.kernel
def gemm_kernel(a, b, c, ...):
    # FlyDSL automatically selects MFMA instruction
    acc = flyc.mfma(a_tile, b_tile, acc)

    # Specify specific instruction type
    acc = flyc.mfma(a_tile, b_tile, acc,
                     mfma_type='f32_16x16x32_bf16')
```

**MFMA in Gluon**:

```python
# Gluon automatically maps tl.dot to MFMA
# Need to ensure tile size matches MFMA supported shapes
c = tl.dot(a, b)  # Compiler selects optimal MFMA instruction
```

**Practical Experience**:
- CDNA3: `mfma_f32_16x16x32_bf16` is the primary instruction for BF16 GEMM
- CDNA4: `mfma_f32_16x16x128_f8f6f4` expands K dimension from 32 to 128, providing 4x the compute per instruction
- CDNA3 BF16 matrix cores execute **significantly faster** than FP16 (unlike NVIDIA, where both have the same throughput)
- BLOCK_K must be a multiple of MFMA K dimension (CDNA3: 32, CDNA4: 128)

---

## Pattern: mfma_scale for FP8 Quantized GEMM

**Source**: `cdna4/gluon/triton/matmul_gluon_gfx950_*.py`

```python
# CDNA4 mfma_scale: Apply block scale inside MMA instruction
# Avoids separate dequant step

# Gluon usage
from triton.experimental.gluon.language.amd.cdna4 import mfma_scaled

# scale_a, scale_b: One E8M0 scale factor per 32 elements
c = mfma_scaled(a_fp8, b_fp8, c_acc, scale_a, scale_b)
```

**Practical Experience**:
- `mfma_scale` is ~30% faster than "dequant first, then mfma"
- Scale factor format must be E8M0 (8-bit exponent, no mantissa)
- CDNA3 does not support `mfma_scale`; FP8 GEMM requires manual dequant

---

## Related Documents

- **MFMA Instruction Reference**: [AMD MFMA Matrix Core Programming Guide](../../ref-docs/amd-mfma-matrix-cores.md) — instruction naming conventions, register layout
- **Tuning Guide**: AMD GPU Kernel Tuning Guide — CDNA3 vs CDNA4 hardware specification comparison
- **CDNA4 FP8 Hands-on**: [CDNA4 FP8 GEMM Optimization Hands-on](../../../cdna4/ref-docs/cdna4-fp8-gemm-optimization.md)
