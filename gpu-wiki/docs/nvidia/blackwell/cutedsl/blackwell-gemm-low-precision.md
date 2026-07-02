# Blackwell GEMM: Low-Precision Data Types and Block Scaling

How Blackwell handles sub-byte (FP4, FP6) data formats for UMMA operations, including TMA unpacking, SMEM layout constraints, and native block-scaling support. Part 3 of the Colfax Research CUTLASS Blackwell series.


**Last updated**: 2026-06-30

---

## 1. Overview

This document covers low-precision computation, focusing on sub-byte (6-bit and 4-bit) formats and their impact on memory layout. Key takeaway: for `f8f6f4`-type mixed-input UMMA (accepting any combination of supported 8-bit, 6-bit, and 4-bit operands), UMMA requires data in a specific unpacked format, and TMA can perform the unpacking during GMEM-to-SMEM loads. However, this imposes additional constraints on tile sizes, leading dimensions, and address alignment in GMEM.

Blackwell also supports **block-scaled** formats: both OCP-standard MX types and NVIDIA's own NVF4 data type.

## 2. Motivation for Low Precision

Hardware and software have co-evolved toward lower precision over the years:

- **Volta (2017):** Tensor Cores support FP16 matrix multiply + FP32 accumulation
- **bfloat16 (2018):** 8 exponent bits (same dynamic range as FP32) + 7 mantissa bits
- **Ampere TF32:** 19-bit format with FP32 range and FP16 precision
- **Hopper FP8:** E4M3 and E5M2 8-bit floating-point formats
- **Blackwell:** Sub-byte precision with 6-bit and 4-bit floating-point types

DeepSeek mitigated FP8 GEMM accuracy loss on Hopper by alternating between Tensor Core accumulation and CUDA core accumulation.

**Block-scaling** divides each group of values by a high-precision scale factor. Grouping options range from per-tensor to per-value, per-row/column, tile-level (128x128), and row-block-level (1x32 or 1x16). Blackwell UMMA natively supports **1x32 or 1x16 block scaling**.

## 3. Data Formats

NVIDIA GPUs support five basic floating-point data types at or below 1 byte:

| Type  | Bits | Exponent | Mantissa | Range/Notes |
|-------|------|----------|----------|-------------|
| E5M2  | 8    | 5        | 2        | max=57344 |
| E4M3  | 8    | 4        | 3        | max=448, higher precision than E5M2 |
| E3M2  | 6    | 3        | 2        | range [-28, 28] |
| E2M3  | 6    | 2        | 3        | range [-7.5, 7.5], higher precision than E3M2 |
| E2M1  | 4    | 2        | 1        | exactly represents {0, 0.5, 1, 1.5, 2, 3, 4, 5, 6} and negatives |

Unlike IEEE formats, the 6-bit and 4-bit types have **no NaN or +/-infinity**.

## 4. Low-Precision UMMA

UMMA data types are determined by the `.kind` qualifier. **`tcgen05.mma` with `.kind::f8f6f4`** supports MMA operations with operands in any of the 5 low-precision types above (with FP32 or FP16 accumulation). A and B need not have the same type, enabling mixed-input UMMA.

### 4.1 Operation Constraints

The f8f6f4 kind imposes several restrictions:

- For dense GEMM, the MMA tile **K extent is always 32**
- Dense GEMM operand tiles must be 32 bytes wide in the K direction
- Operand values for f8f6f4 instructions are padded to 1 byte per value

### 4.2 Dynamic Data Types

In pre-fifth-generation Tensor Core instructions (PTX mma), all data types were encoded in the instruction itself, requiring compile-time knowledge. For `tcgen05.mma` with `.kind::f8f6f4`, data type information is now encoded in the **instruction descriptor** — a runtime parameter constructed on-device. This enables runtime selection of operand types.

## 5. Operand Layout and TMA Loading

### 5.1 SMEM and GMEM Layout

Operand data in SMEM must be stored in a specific **16-byte aligned format**: 16 consecutive 4-bit or 6-bit elements are packed contiguously, then padded to a 16-byte boundary.

SMEM allocation for sub-byte operands uses the same space as byte-sized operands (this is partly what enables dynamic data type passing). **The `.kind::f8f6f4` qualifier does not support fully packed contiguous data in SMEM.**

Ideally, tensors are stored packed in GMEM and expanded to the proper padded format during TMA load. TMA provides exactly this functionality:

- `CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B`: Copies 16 packed 4-bit elements from GMEM to a 16-byte aligned SMEM region, adding 8 bytes of padding
- `CU_TENSOR_MAP_DATA_TYPE_16U6_ALIGN16B`: Copies 16 packed 6-bit elements from GMEM to a 16-byte aligned SMEM region, adding 4 bytes of padding

In PTX, these correspond to `cp.async.bulk.tensor` with data types `.b4x16_p64` or `.b6x16_p32`.

**Additional TMA constraints for these types:**

1. TMA base address must be **32-byte aligned** (not the usual 16-byte)
2. The size along the contiguous dimension (leading dimension) must be a **multiple of 128 elements**
3. Only **128B interleave mode** is supported (or no interleave)

In CUTLASS, `sm1xx_gemm_is_aligned()` checks GMEM alignment and `sm1xx_gemm_check_for_f8f6f4_mix8bit_requirement()` checks tile size requirements. CUTLASS asserts **64-byte alignment** for 4-bit data and **96-byte alignment** for 6-bit data.

A third Tensor Map data type, `CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B` (PTX `.b4x16`), copies packed 4-bit GMEM data to packed, unpadded SMEM format — useful for FP4-only UMMA variants.

### 5.2 TMEM Layout

For TMEM, UMMA expects sub-byte data types padded to **1-byte containers**, including 4-bit data.

The typical process for loading sub-byte data into TMEM for GEMM:

1. Store data packed in global memory
2. Load from GMEM to SMEM using one of the "unpacking" TMA types above
3. Load from SMEM to TMEM using `tcgen05.cp` with optional decompression

## 6. CUTLASS Sub-byte UMMA

CUTLASS defines sub-byte data types in `cutlass/float_subbyte.h`:

- `cutlass::float_e3m2_t`
- `cutlass::float_e2m3_t`
- `cutlass::float_e2m1_t`

All derive from `float_exmy_base`. Different types can be mixed for basic arithmetic, though these operations execute in FP32 without hardware support.

CUTLASS also provides special sub-byte types designed for UMMA and TMA:

- `cutlass::float_e3m2_unpacksmem_t`
- `cutlass::float_e2m3_unpacksmem_t`
- `cutlass::float_e2m1_unpacksmem_t`

These types instruct TMA to use 16-byte padded copies where applicable. For f8f6f4 UMMA kernels, prefer these types. The collective builder converts plain types to these unpacked types via `cutlass::gemm::collective::detail::sm1xx_kernel_input_element_to_mma_input_element`.

For all sub-byte types, the SMEM layout is identical to that of 8-bit data, so `uint8_t` can be used to define SMEM layouts.

### 6.1 Runtime Data Types

To use runtime operand data types, specify:

- `cutlass::type_erased_dynamic_float8_t`
- `cutlass::type_erased_dynamic_float6_t`
- `cutlass::type_erased_dynamic_float4_t`

SMEM layouts require no changes for these types. TMA is format-agnostic (only bit-width matters). For MMA itself, the instruction descriptor must be manually updated. `runtime_data_types` is an integer representation of data types used in the instruction descriptor, specified as members of `cute::UMMA::MXF8F6F4`.

## 7. Conclusion

At the PTX/hardware level, sub-byte UMMA requires **16-byte aligned, padded SMEM formats** and supports **runtime data type selection** via the instruction descriptor. At the CUTLASS level, this translates to creating SMEM layouts with appropriate padding, instructing TMA to format data during transfer, and optionally using runtime data types.

Block-scaling support for MXFP4 and NVF4 data types builds on these foundations, with UMMA natively applying per-block scale factors during the matrix multiply operation.


## Related

- [SM100 Blackwell CuTeDSL Panorama](blackwell-cutedsl-sm100.md)
- [CUTLASS Tutorial: Blackwell GEMM with Tensor Memory](blackwell-gemm-tensor-memory.md)
- [Blackwell GEMM: Thread Block Clusters, TMA Multicast, and Pair-UMMA](blackwell-gemm-thread-block-cluster.md)
- [Building a tcgen05 GEMM from Scratch: Reaching 98% of cuBLAS on Blackwell](blackwell-tcgen05-gemm-from-scratch.md)
- [Blackwell Matrix Multiplication Part 1: Fundamentals](colfax-blackwell-gemm-part1-basics.md)
- [CUTLASS GEMM Optimization Strategy](../../common/cutedsl/cutlass-gemm-optimization.md)
- [Community GEMM Optimization Practical Summary](../../../generic/gemm-optimization-guide.md)
- [CUTLASS/CuTe Core Concepts and Layout Algebra](../../common/cutedsl/cutlass-cute-fundamentals.md)
