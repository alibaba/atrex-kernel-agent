# NVFP4 and Block-Scaled Narrow Precision

## Overview

NVFP4 is NVIDIA's 4-bit floating-point format (E2M1) with block scaling, native to Blackwell tensor cores.

## Format Details

```
E2M1: 1 sign bit, 2 exponent bits, 1 mantissa bit
Representable values: 0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6

Block scaling: every 16 FP4 elements share one FP8 E4M3 scale factor
Two-level: per-block E4M3 scale × per-tensor FP32 global scale

Quantization:   q_i = cast_FP4(x_i / (s_global * s_block))
Dequantization: x_hat_i = s_global * s_block * deq_FP4(q_i)
```

## tcgen05 Variants for FP4

| Variant | Description | Throughput vs Hopper |
|---|---|---|
| `tcgen05.mma.mxf4.block_scale` | MX FP4 with block scaling | **4×** |
| `tcgen05.mma.mxf4nvf4.block_scale` | NVFP4 + MX FP4 flexible scaling | **4×** |

## PTX for FP4 Conversion

```ptx
// Convert two FP4 values to two FP16 values
cvt.rn.f16x2.e2m1x2 result, packed_fp4;

// Byte unpacking (faster than bitwise extraction)
mov.b32 {tmp0, tmp1, tmp2, tmp3}, packed_data;
```

## NVFP4 vs MXFP4

| Aspect | NVFP4 | MXFP4 |
|---|---|---|
| Scale format | E4M3 (fractional) | UE8M0 (power-of-2 only) |
| Block size | 16 elements | 32 elements |
| Scale precision | Non-power-of-2 | Power-of-2 only |
| Quantization error | Lower | Higher |

## Related
- [fine-grained-quantization](../techniques/fine-grained-quantization.md) -- Scaling strategies
- [nvfp4-gemm](../kernels/nvfp4-gemm.md) -- NVFP4 GEMM kernel
- [nvfp4-gemv](../kernels/nvfp4-gemv.md) -- NVFP4 GEMV kernel
