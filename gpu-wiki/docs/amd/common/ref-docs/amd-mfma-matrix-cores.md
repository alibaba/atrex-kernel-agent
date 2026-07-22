# AMD MFMA Matrix Core Programming Guide

## MFMA Instruction Overview

MFMA (Matrix Fused Multiply-Add) is the matrix multiply-accumulate instruction for AMD's CDNA architecture, where 64 lanes collaborate to execute a single instruction that completes `D = A * B + C`.

### Instruction Naming Convention

```
__builtin_amdgcn_mfma_ODType_MxNxKInDType(a_reg, b_reg, c_reg, cbsz, abid, blgp)
```

- `ODType`: Output/accumulator type (typically f32)
- `MxNxK`: Matrix dimensions
- `InDType`: Input type
- `cbsz, abid, blgp`: Broadcast flag (typically set to 0)

---

## CDNA3 (gfx942) MFMA Instructions

| Input → Output | Matrix Dimension MxNxK | Cycles |
|-------------|---------------|---------|
| FP64 → FP64 | 16x16x4 | 64 |
| FP32 → FP32 | 32x32x2, 16x16x4 | 64, 32 |
| FP16/BF16 → FP32 | 32x32x8, 16x16x16 | 32, 16 |
| FP8 → FP32 | 16x16x32, 32x32x16 | 16, 32 |

**CDNA3 FP8 variants:** E4M3FNUZ (bias=8) and E5M2FNUZ (bias=16)

### CDNA3 Peak Performance

| Precision | Peak TFLOPS | Relative to FP32 |
|------|-------------|----------|
| FP64 | 163.4 | 1x |
| FP32 | 163.4 | 1x |
| FP16/BF16 | 1307.4 | ~8x |
| FP8 | 2614.9 | ~16x |

---

## CDNA4 (gfx950) New MFMA Instructions

CDNA4 retains all CDNA3 instructions and adds the following:

| Input → Output | Matrix Dimension MxNxK | Cycles |
|-------------|---------------|---------|
| FP16/BF16 → FP32 | 16x16x32, 32x32x16 | 16, 32 |
| FP8/FP6/FP4 → FP32 | 16x16x128, 32x32x64 | 16-64 (depends on type) |
| MXFP8/6/4 → FP32 (block-scaled) | 16x16x128, 32x32x64 | 16-64 |

**CDNA4 FP8 variants:** E4M3FN (OCP, bias=7) and E5M2 (OCP, bias=15)

**Key difference:** FP6/FP4 instructions have fewer cycles than FP8 instructions with the same dimensions.

### CDNA4 Peak Performance

| Precision | Peak TFLOPS | Relative to FP32 |
|------|-------------|----------|
| FP64 | 78.6 | ~0.5x |
| FP32 | 157.3 | 1x |
| FP16/BF16 | 2500 (2.5 PF) | ~16x |
| FP8 | 5000 (5 PF) | ~32x |
| FP6/FP4 | 10000 (10 PF) | ~64x |

### Peak Performance Calculation Formula

```
Peak = 2 * M * N * K * num_matrix_cores * (max_engine_clock / cycle_count) / 10^6
```

---

## Wave-Lane Mapping

Wavefront = 64 lanes, all lanes collaborate to execute a single MFMA.

### Elements Per Thread

| Instruction | A elements/thread | B elements/thread | C/D elements/thread |
|------|-------------|-------------|----------------|
| f32_32x32x2f32 | 1 (scalar) | 1 (scalar) | 16 |
| f32_16x16x16f16 | 4 | 4 | 4 |
| f32_32x32x16_fp8 | 8 | 8 | 16 |
| scale_f32_32x32x64 (FP8) | 32 + 1 scale | 32 + 1 scale | 16 |
| scale_f32_32x32x64 (FP4) | 32 + 1 scale | 32 + 1 scale | 16 |

### FP16 16x16x16 Lane Mapping

```c
// Matrix A (16x16): lane t 4 contiguous FP16
// row = t % 16, column = 4 * (t / 16)
a_reg = *(fp16x4_t*)(A + 4*(t/16) + 16*(t%16));

// Matrix B (16x16): lane t 4
// column = t % 16, row t/16
for (int i = 0; i < 4; i++)
    b_reg[i] = *(B + i*16 + t%16 + (t/16)*64);

// Matrix C/D (16x16): 4 /thread
for (int i = 0; i < 4; i++)
    *(C + i*16 + t%16 + (t/16)*64) = c_reg[i];
```

### FP32 32x32 Output Layout (General)

All 32x32 outputs (FP32/FP8/block-scaled) share the same layout. Each thread holds 16 FP32 values, distributed across 4 groups × 4 rows:

```c
for (int i = 0; i < 4; i++) {
    C[t%32 + (t/32)*4*32 + i*32*8]          = c_reg[i*4];
    C[t%32 + (t/32)*4*32 + 32*1 + i*32*8]   = c_reg[i*4+1];
    C[t%32 + (t/32)*4*32 + 32*2 + i*32*8]   = c_reg[i*4+2];
    C[t%32 + (t/32)*4*32 + 32*3 + i*32*8]   = c_reg[i*4+3];
}
```

---

## HIP Programming Example

### FP16 16x16x16 (CDNA3/CDNA4)

```c
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

using fp16_t = _Float16;
using fp16x4_t = __attribute__((vector_size(4 * sizeof(fp16_t)))) fp16_t;
using fp32x4_t = __attribute__((vector_size(4 * sizeof(float)))) float;

__global__ void mfma_f32_16x16x16_f16(const fp16_t* A, const fp16_t* B, float* C) {
    fp16x4_t a_reg;
    fp16x4_t b_reg;
    fp32x4_t c_reg {};

    a_reg = *(const fp16x4_t*)(A + 4*(threadIdx.x/16) + 16*(threadIdx.x%16));
    for (int i = 0; i < 4; i++)
        b_reg[i] = *(B + i*16 + threadIdx.x%16 + (threadIdx.x/16)*64);

    c_reg = __builtin_amdgcn_mfma_f32_16x16x16f16(a_reg, b_reg, c_reg, 0, 0, 0);

    for (int i = 0; i < 4; i++)
        *(C + i*16 + threadIdx.x%16 + (threadIdx.x/16)*64) = c_reg[i];
}
// Launch: <<<1, 64>>>
```### FP8 32x32x16（CDNA3）

```c
#include <hip/hip_fp8.h>

using fp8_t = __hip_fp8_storage_t;
using fp8x8_t = __attribute__((vector_size(8))) fp8_t;
using fp32x16_t = __attribute__((vector_size(64))) float;

__global__ void mfma_f32_32x32x16_fp8(const fp8_t* A, const fp8_t* B, float* C) {
    fp8x8_t a_reg, b_reg;
    fp32x16_t c_reg {};

    a_reg = *(const fp8x8_t*)(A + (threadIdx.x/32)*8 + (threadIdx.x%32)*16);
    for (int i = 0; i < 8; i++)
        b_reg[i] = *(B + i*32 + threadIdx.x%32 + (threadIdx.x/32)*8*32);

 // note: FP8 intrinsic (long) typeconversion
    c_reg = __builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8(
        (long)a_reg, (long)b_reg, c_reg, 0, 0, 0);
 // ... outputstore ...
}
```

### Block-Scaled FP8 32x32x64（CDNA4 Only）

```c
#include <hip/hip_ext_ocp.h> // CDNA4 use OCP file

using fp8_t = __amd_fp8_storage_t; // CDNA4 usedifferenttype
using fp8x32_t = __attribute__((vector_size(32))) fp8_t;
using fp32x16_t = __attribute__((vector_size(64))) float;

__global__ void mfma_scale_f32_32x32x64_fp8(
    const fp8_t* A, const fp8_t* B, float* C) {
    fp8x32_t a_reg, b_reg;
    fp32x16_t c_reg {};

 // ... load A: 32 FP8/thread, B: 32 FP8/thread ...

    uint8_t scale_a = 127;  // E8M0: 2^(127-127) = 1.0
    uint8_t scale_b = 127;

    c_reg = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
        a_reg, b_reg, c_reg,
        0,        // Atype: 0=E4M3, 1=E5M2, 2=E2M3(FP6), 3=E3M2(BF6), 4=E2M1(FP4)
        0,        // Btype
        0, scale_a,  // OPSEL_A, scale_a
        0, scale_b); // OPSEL_B, scale_b
 // ... outputstore ...
}
```

### FP4 32x32x64（CDNA4 Only）

```c
using fp4x2_t = __amd_fp4x2_storage_t; // uint8_t, 2 FP4 packed
using fp4x64_t = fp4x2_t __attribute__((ext_vector_type(32)));  // 256-bit

// FP4 bytesaccessrequiresfunction
uint8_t val = __amd_extract_fp4(packed_byte, index); // single FP4
fp4x2_t pair = __amd_create_fp4x2(val0, val1); // FP4

// Intrinsic requires 256-bit parameter, 32 FP4 = 128-bit,
c_reg = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
    a_reg, b_reg, c_reg, 4, 4, ...);  // Atype=4(FP4), Btype=4(FP4)
```

---

## Low-Precision Types Quick Reference

| Type | Bits | Exp/Man | Bias | Range | Has Zero | Has NaN |
|------|------|---------|------|------|------|--------|
| FP16 | 16 | 5/10 | 15 | ±65504 | Yes (signed) | Yes |
| BF16 | 16 | 8/7 | 127 | ±3.39e38 | Yes (signed) | Yes |
| E4M3FN (OCP/CDNA4) | 8 | 4/3 | 7 | ±448 | Yes | Yes |
| E4M3FNUZ (CDNA3) | 8 | 4/3 | 8 | ±240 | Unsigned only | Yes |
| E5M2 (OCP/CDNA4) | 8 | 5/2 | 15 | ±57344 | Yes | Yes |
| E5M2FNUZ (CDNA3) | 8 | 5/2 | 16 | ±57344 | Unsigned only | No |
| E8M0 (scale) | 8 | 8/0 | 127 | 2^±127 | No | Yes |
| E2M3 (FP6) | 6 | 2/3 | 1 | ±7.5 | No | No |
| E3M2 (BF6) | 6 | 3/2 | 3 | ±28 | No | No |
| E2M1 (FP4) | 4 | 2/1 | 1 | ±6 | No | No |

---

## CDNA3 vs CDNA4 Programming Differences Summary

| Aspect | CDNA3 (gfx942) | CDNA4 (gfx950) |
|------|----------------|----------------|
| FP8 Header | `hip/hip_fp8.h` | `hip/hip_ext_ocp.h` |
| FP8 Storage Type | `__hip_fp8_storage_t` | `__amd_fp8_storage_t` |
| FP8 Format | FNUZ (bias=8/16) | OCP (bias=7/15) |
| FP6/FP4 | Not supported | Supported (E2M3, E3M2, E2M1) |
| Block-scaled MFMA | Not supported | Supported (E8M0 scale factor) |
| Max FP16 K | 16 | 32 |
| MI16x16 vs MI32x32 | Both available | MI16x16 is more power-efficient |
| LDS Capacity | 64 KB/CU | 160 KB/CU |
| LDS Bank Count | 32 | 64 |
