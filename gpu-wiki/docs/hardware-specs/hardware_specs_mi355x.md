# AMD GPU Hardware Compute Specification Table (CDNA4)

**Last Updated**: 2026-03-28

---

## MI355X (gfx950 / CDNA4)

| Specification | Value |
|------|-----|
| **Architecture** | CDNA4 |
| **Release** | June 2025 |
| **CU Count** | 256 (8 XCD × 32 CU/XCD) |
| **Stream Processors** | 16,384 (64 per CU) |
| **Matrix Cores** | 1,024 (4 per CU) |
| **Peak Frequency** | 2.4 GHz |
| **Peak Compute (BF16/FP16)** | 5,033.2 TFLOPS (Matrix), 2,516.6 TFLOPS (VFMA) |
| **Peak Compute (FP8/INT8)** | 10,066.4 TFLOPS (Matrix), 5,033.2 TFLOPS (VFMA) |
| **Peak Compute (FP6)** | 20,132.6 TFLOPS (Matrix), 10,066.3 TFLOPS (VFMA) |
| **Peak Compute (FP4)** | 20,132.6 TFLOPS (Matrix), 10,066.3 TFLOPS (VFMA) |
| **Peak Compute (FP32)** | 157.3 TFLOPS |
| **Peak Compute (FP64)** | 78.6 TFLOPS |
| **HBM Capacity** | 288 GB HBM3e |
| **HBM Bandwidth** | 8 TB/s |
| **LDS Capacity/CU** | 160 KB |
| **Warp Size** | 64 threads |
| **Ridge Point (BF16)** | ~629 FLOPs/Byte (Matrix: 5033/8) |
| **Matrix Instruction** | MFMA scaled (v_mfma_scale_bf16, FP4/FP6 support) |
| **Supported Formats** | FP64, FP32, FP16, BF16, FP8 (E4M3/E5M2), FP6, FP4 (E2M1) |
| **Async Copy** | ✅ buffer_load_to_shared (DMA global→shared) |
| **In-thread Transpose** | ❌ Disabled (compiler limitation for gfx950) |
| **kpack** | Fixed at 1 |
| **Structured Sparsity** | 2:4 |
| **Power Consumption** | 1400W max |

**Data Sources**: Glenn Lockwood's Digital Garden, AMD MI355X GPU Brochure, ISSCC 2026

---

## In-thread Transpose Disabled Explanation

**Why is in-thread transpose disabled on gfx950?**

The Triton compiler for CDNA4 (gfx950) disables the in-thread transpose pass. This is a compiler-level limitation, not a hardware limitation.

**Reasons**:
1. **Register layout constraints**: gfx950's MFMA instructions impose stricter requirements on the register layout of input data. In-thread transpose may cause data arrangement to not conform to MFMA input specifications.
2. **kpack=1 constraint**: gfx950 enforces kpack=1, which conflicts with the packing strategy of in-thread transpose.
3. **Compiler simplification**: CDNA4 introduces new scaled MFMA (FP4/FP6), and the compiler chooses to disable certain optimization passes to ensure correctness.

**Impact**:
- Cannot rely on the compiler to automatically transpose register data.
- Must explicitly manage shared memory layout to ensure that data is loaded from smem directly with the correct DotOperandLayout.
- Data must already have the correct layout for MFMA consumption **before** being stored into shared memory.

**Solution**:
```python
# ✅ Correct: Explicitly specify DotOperandLayout to ensure shared memory load produces correct layout directly
dot_op0 = gl.DotOperandLayout(operand_index=0, parent=mma, k_width=4)
a_dot = a_smem.load(dot_op0)  # Load directly with correct layout
acc = gl.amd.cdna4.mfma(a_dot, b_dot, acc)
```

---

## Ridge Points Calculation

Ridge Point = Peak Compute / Peak Bandwidth (unit: FLOPs/Byte)

| GPU | BF16/FP16 (Matrix) | FP8 (Matrix) | FP32 |
|-----|-------------------|-------------|------|
| MI355X | 629 | 1,258 | 19.7 |

**Identifying Bottlenecks**:
- Arithmetic Intensity ≥ Ridge Point → Compute Bound
- Arithmetic Intensity < Ridge Point → Memory Bound

## Related Documents

- **Cross-Architecture Comparison**: [CDNA3 Hardware Specs](hardware_specs_mi300x.md) | [Hopper Hardware Specs](hardware_specs_hopper.md)
- **Detailed Hardware Comparison**: [CDNA4 FP8 GEMM Optimization Practice](../ref-docs/amd/common/gfx950/cdna4-fp8-gemm-optimization.md) — Includes CDNA4 vs CDNA3 hardware difference comparison table
- **Downstream Usage**: [CDNA4 ISA Optimization Checklist](../ref-docs/amd/gluon/gfx950/common_optimizations.md)
- **⚠️ Key Difference**: LDS 160 KB (CDNA3 only 64 KB), 64 LDS banks (CDNA3 only 32), tile configurations cannot be directly reused.
