# Block-Scaled MMA

## Mode: MXF8/MXF4/NVF4 Block-Scaled Matrix Multiplication

**Source**: `cutedsl/cutlass/blockwise_gemm/`, `cutedsl/flashinfer/dense_blockscaled_gemm_sm100.py`

```python
# Blackwell tcgen05 supports block-scaled MMA:
# Matrices are quantized by blocks, each block shares a scale factor

# MXF8: every 32 elements share an E8M0 scale
tiled_mma = cute.make_tiled_mma(
    cute.SM100_MMA_F32MXF8MXF8F32_SS_TN,
    # Inputs are FP8 + E8M0 scale, output is FP32
)

# Scale factors are loaded via separate TMA
cute.copy(tma_scale_a, scale_a_gmem, scale_a_smem)
cute.copy(tma_scale_b, scale_b_gmem, scale_b_smem)

# MMA automatically applies the scale
acc = cute.gemm(tiled_mma, data_smem, scale_smem, acc)
```

## NVF4 (NVIDIA FP4) Mode

```python
# NVF4: 4-bit floating point, every 16 elements share a scale
tiled_mma = cute.make_tiled_mma(
    cute.SM100_MMA_F32NVF4NVF4F32_SS_TN,
)

# NVF4 scale factor is UE8M0 (unsigned E8M0)
# 16:1 quantization ratio, extreme memory compression
```

**Practical experience**:
- The TFLOPS of block-scaled MMA is much higher than dense FP16/BF16 (because the data volume is smaller, it is easier to achieve compute-bound)
- Loading scale factors incurs additional overhead, but the magnitude is much smaller than the data itself
- MXF8 (FP8) is suitable for most LLM inference; NVF4 is suitable for extreme compression scenarios
- The scale block size (32 for MXF8, 16 for NVF4) is fixed in hardware and cannot be adjusted

---

## Related Documents

- **tcgen05 MMA and TMEM**: [tcgen05 MMA and TMEM](tcgen05-mma-tmem.md) — tcgen05 MMA basics
- **MLA Decode**: [MLA Decode](mla-decode.md) — Application of block-scaled FP8 in MLA inference
- **Epilogue Fusion**: [Epilogue Fusion](epilogue-fusion.md) — GEMM + FP4 Quantize fusion
- **PTX MMA Evolution**: [PTX MMA Instruction Evolution](../../../../ref-docs/nvidia/common/nvidia-ptx-mma-instructions.md) — wmma → mma.sync → wgmma → tcgen05
