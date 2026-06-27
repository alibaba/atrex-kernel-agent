"""Vendored CuTeDSL helpers for NVFP4 quantization on bf16 inputs.

Functions copied verbatim from
  reference-kernels/nvidia/blackwell/cutedsl/flashinfer/
    {fp4_common.py, quantization_cute_dsl_utils.py}
keeping only the bf16 path (no fp16/fp8/MXFP4 variants).

The exact PTX inline-asm contents are critical for bit-exact match against
``vllm._custom_ops.scaled_fp4_quant``.
"""
from __future__ import annotations

from typing import Tuple

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint32, Uint64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op


# ============================================================================
# Constants
# ============================================================================

NVFP4_SF_VEC_SIZE = 16
WARP_SIZE = 32


# ============================================================================
# 128-bit vectorized loads / stores  (fp4_common.py)
# ============================================================================

@dsl_user_op
def ld_global_v4_u32(base_ptr: Int64, *, loc=None, ip=None
                      ) -> Tuple[Uint32, Uint32, Uint32, Uint32]:
    """Load 128 bits (4 x uint32) from global memory."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
        [Int64(base_ptr).ir_value(loc=loc, ip=ip)],
        "ld.global.v4.u32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )
    v0 = llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)
    v1 = llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)
    v2 = llvm.extractvalue(T.i32(), result, [2], loc=loc, ip=ip)
    v3 = llvm.extractvalue(T.i32(), result, [3], loc=loc, ip=ip)
    return Uint32(v0), Uint32(v1), Uint32(v2), Uint32(v3)


@dsl_user_op
def st_global_u64(base_ptr: Int64, value: Uint64, *, loc=None, ip=None):
    """Store 64 bits to global memory."""
    llvm.inline_asm(
        None,
        [Int64(base_ptr).ir_value(loc=loc, ip=ip),
         Uint64(value).ir_value(loc=loc, ip=ip)],
        "st.global.u64 [$0], $1;", "l,l",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def get_ptr_as_int64(tensor: cute.Tensor, offset: Int32,
                     *, loc=None, ip=None) -> Int64:
    """Address of tensor[offset] as int64. Use only with explicit address-space loads."""
    elem_ptr = tensor.iterator + Int32(offset)
    ptr_int = llvm.ptrtoint(T.i64(), elem_ptr.llvm_ptr, loc=loc, ip=ip)
    return Int64(ptr_int)


# ============================================================================
# V1-TMA: shared-memory address + 128-bit shared load (fp4_common.py upstream)
# ============================================================================
# Copied verbatim from
#   reference-kernels/nvidia/blackwell/cutedsl/flashinfer/fp4_common.py
# lines 230-270. Critical: use `elem_ptr.toint()` (preserves SMEM addrspace=3),
# NOT `llvm.ptrtoint(T.i32(), ...)` (would drop addrspace and ld.shared faults).

@dsl_user_op
def get_smem_ptr_as_int32(tensor: cute.Tensor, offset: Int32,
                          *, loc=None, ip=None) -> Int32:
    """Get the shared-memory byte address of tensor[offset] as Int32.

    Uses Pointer.toint() which preserves the SMEM address space (addrspace 3),
    returning a 32-bit SMEM address suitable for ld.shared.* instructions.
    """
    elem_ptr = tensor.iterator + Int32(offset)
    return elem_ptr.toint(loc=loc, ip=ip)


@dsl_user_op
def ld_shared_v4_u32(smem_addr: Int32, *, loc=None, ip=None
                      ) -> Tuple[Uint32, Uint32, Uint32, Uint32]:
    """Load 128 bits (4 x uint32) from shared memory via ld.shared.v4.u32.

    Args:
        smem_addr: 32-bit shared memory address (from get_smem_ptr_as_int32).

    Returns:
        4 Uint32 values (16 bytes total, e.g. 8 packed bf16 elements).
    """
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
        [Int32(smem_addr).ir_value(loc=loc, ip=ip)],
        "ld.shared.v4.u32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )
    v0 = llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)
    v1 = llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)
    v2 = llvm.extractvalue(T.i32(), result, [2], loc=loc, ip=ip)
    v3 = llvm.extractvalue(T.i32(), result, [3], loc=loc, ip=ip)
    return Uint32(v0), Uint32(v1), Uint32(v2), Uint32(v3)


# ============================================================================
# Math intrinsics  (fp4_common.py)
# ============================================================================

@dsl_user_op
def rcp_approx_ftz(a: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(llvm.inline_asm(
        T.f32(), [Float32(a).ir_value(loc=loc, ip=ip)],
        "rcp.approx.ftz.f32 $0, $1;", "=f,f",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    ))


@dsl_user_op
def fmax_f32(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(llvm.inline_asm(
        T.f32(),
        [Float32(a).ir_value(loc=loc, ip=ip),
         Float32(b).ir_value(loc=loc, ip=ip)],
        "max.f32 $0, $1, $2;", "=f,f,f",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    ))


@dsl_user_op
def fabs_f32(a: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(llvm.inline_asm(
        T.f32(), [Float32(a).ir_value(loc=loc, ip=ip)],
        "abs.f32 $0, $1;", "=f,f",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    ))


# ============================================================================
# bfloat2 SIMD intrinsics  (fp4_common.py + quantization_cute_dsl_utils.py)
# ============================================================================

@dsl_user_op
def bfloat2_habs2(x: Uint32, *, loc=None, ip=None) -> Uint32:
    """abs of two bf16 packed in a uint32 — clear sign bits."""
    return Uint32(llvm.inline_asm(
        T.i32(), [Uint32(x).ir_value(loc=loc, ip=ip)],
        "and.b32 $0, $1, 0x7FFF7FFF;", "=r,r",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    ))


@dsl_user_op
def bfloat2_hmax2(a: Uint32, b: Uint32, *, loc=None, ip=None) -> Uint32:
    """element-wise max of 2 bf16 pairs."""
    return Uint32(llvm.inline_asm(
        T.i32(),
        [Uint32(a).ir_value(loc=loc, ip=ip),
         Uint32(b).ir_value(loc=loc, ip=ip)],
        "max.bf16x2 $0, $1, $2;", "=r,r,r",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    ))


@dsl_user_op
def bfloat2_mul(a: Uint32, b: Uint32, *, loc=None, ip=None) -> Uint32:
    """element-wise mul of 2 bf16 pairs."""
    return Uint32(llvm.inline_asm(
        T.i32(),
        [Uint32(a).ir_value(loc=loc, ip=ip),
         Uint32(b).ir_value(loc=loc, ip=ip)],
        "mul.bf16x2 $0, $1, $2;", "=r,r,r",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    ))


@dsl_user_op
def bfloat2_hmax_reduce_to_f32(x: Uint32, *, loc=None, ip=None) -> Float32:
    """Extract max of 2 bf16 values in a bfloat2 as Float32."""
    return Float32(llvm.inline_asm(
        T.f32(), [Uint32(x).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b32 lo, hi;
            .reg .f32 f0, f1;
            and.b32 lo, $1, 0xFFFF;
            shr.b32 hi, $1, 16;
            shl.b32 lo, lo, 16;
            shl.b32 hi, hi, 16;
            mov.b32 f0, lo;
            mov.b32 f1, hi;
            max.f32 $0, f0, f1;
        }
        """,
        "=f,r",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    ))


@dsl_user_op
def bfloat2_to_float2_scaled(bf2: Uint32, scale: Float32,
                              *, loc=None, ip=None) -> Tuple[Float32, Float32]:
    """Convert bfloat16x2 to (Float32, Float32) AND multiply by scale."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [Uint32(bf2).ir_value(loc=loc, ip=ip),
         Float32(scale).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b32 lo, hi;
            .reg .f32 f0, f1;
            and.b32 lo, $2, 0xFFFF;
            shr.b32 hi, $2, 16;
            shl.b32 lo, lo, 16;
            shl.b32 hi, hi, 16;
            mov.b32 f0, lo;
            mov.b32 f1, hi;
            mul.f32 $0, f0, $3;
            mul.f32 $1, f1, $3;
        }
        """,
        "=f,=f,r,f",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )
    f0 = llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip)
    f1 = llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)
    return Float32(f0), Float32(f1)


@dsl_user_op
def bfloat2_to_float2(bf2: Uint32, *, loc=None, ip=None) -> Tuple[Float32, Float32]:
    """Convert bfloat16x2 to (Float32, Float32) without scaling."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [Uint32(bf2).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b32 lo, hi;
            .reg .f32 f0, f1;
            and.b32 lo, $2, 0xFFFF;
            shr.b32 hi, $2, 16;
            shl.b32 lo, lo, 16;
            shl.b32 hi, hi, 16;
            mov.b32 f0, lo;
            mov.b32 f1, hi;
            mov.f32 $0, f0;
            mov.f32 $1, f1;
        }
        """,
        "=f,=f,r",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )
    f0 = llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip)
    f1 = llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)
    return Float32(f0), Float32(f1)


@dsl_user_op
def f32x2_to_bfloat2(a: Float32, b: Float32, *, loc=None, ip=None) -> Uint32:
    """Pack two Float32 to bfloat16x2 in a uint32 (lo=a, hi=b).
    Uses cvt.rn.bf16x2.f32 for round-to-nearest-even."""
    return Uint32(llvm.inline_asm(
        T.i32(),
        [Float32(a).ir_value(loc=loc, ip=ip),
         Float32(b).ir_value(loc=loc, ip=ip)],
        "cvt.rn.bf16x2.f32 $0, $2, $1;",
        "=r,f,f",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    ))


# ============================================================================
# E4M3 / E2M1 conversion  (fp4_common.py)
# ============================================================================

@dsl_user_op
def cvt_f32_to_e4m3(a: Float32, *, loc=None, ip=None) -> Uint32:
    """Float32 -> E4M3 via cvt.rn.satfinite.e4m3x2 (one fp8 in low byte)."""
    return Uint32(llvm.inline_asm(
        T.i32(), [Float32(a).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b16 fp8_pair;
            .reg .f32 zero;
            mov.f32 zero, 0f00000000;
            cvt.rn.satfinite.e4m3x2.f32 fp8_pair, zero, $1;
            cvt.u32.u16 $0, fp8_pair;
        }
        """,
        "=r,f",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    ))


@dsl_user_op
def nvfp4_compute_output_scale(fp8_val: Uint32, global_scale: Float32,
                                *, loc=None, ip=None) -> Float32:
    """output_scale = rcp(float(E4M3(scale)) * rcp(global_scale))
                   = global_scale / float(E4M3(scale)). 0 if scale==0."""
    return Float32(llvm.inline_asm(
        T.f32(),
        [Uint32(fp8_val).ir_value(loc=loc, ip=ip),
         Float32(global_scale).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .pred p_zero;
            .reg .b16 fp8_pair;
            .reg .b32 h2_32;
            .reg .b16 h_lo, h_hi;
            .reg .f32 scale_f32, rcp_gs, product, result;

            cvt.u16.u32 fp8_pair, $1;
            cvt.rn.f16x2.e4m3x2 h2_32, fp8_pair;
            mov.b32 {h_lo, h_hi}, h2_32;
            cvt.f32.f16 scale_f32, h_lo;

            rcp.approx.ftz.f32 rcp_gs, $2;
            mul.f32 product, scale_f32, rcp_gs;
            rcp.approx.ftz.f32 result, product;

            setp.eq.f32 p_zero, scale_f32, 0f00000000;
            selp.f32 $0, 0f00000000, result, p_zero;
        }
        """,
        "=f,r,f",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    ))


@dsl_user_op
def cvt_e2m1x8_f32(v0: Float32, v1: Float32, v2: Float32, v3: Float32,
                   v4: Float32, v5: Float32, v6: Float32, v7: Float32,
                   *, loc=None, ip=None) -> Uint32:
    """8 Float32 -> 8 E2M1 packed as uint32 (4 bytes)."""
    return Uint32(llvm.inline_asm(
        T.i32(),
        [Float32(v0).ir_value(loc=loc, ip=ip),
         Float32(v1).ir_value(loc=loc, ip=ip),
         Float32(v2).ir_value(loc=loc, ip=ip),
         Float32(v3).ir_value(loc=loc, ip=ip),
         Float32(v4).ir_value(loc=loc, ip=ip),
         Float32(v5).ir_value(loc=loc, ip=ip),
         Float32(v6).ir_value(loc=loc, ip=ip),
         Float32(v7).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b8 byte0, byte1, byte2, byte3;
            cvt.rn.satfinite.e2m1x2.f32 byte0, $2, $1;
            cvt.rn.satfinite.e2m1x2.f32 byte1, $4, $3;
            cvt.rn.satfinite.e2m1x2.f32 byte2, $6, $5;
            cvt.rn.satfinite.e2m1x2.f32 byte3, $8, $7;
            mov.b32 $0, {byte0, byte1, byte2, byte3};
        }
        """,
        "=r,f,f,f,f,f,f,f,f",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    ))


# ============================================================================
# Sigmoid+Mul on bfloat16x2 (custom; fused-epilogue helper)
# ============================================================================

@dsl_user_op
def bfloat2_sigmoid_mul(attn_pair: Uint32, gate_pair: Uint32,
                         *, loc=None, ip=None) -> Uint32:
    """For two (attn, gate) bf16 pairs, return bf16x2 of attn * sigmoid(gate).
    sigmoid(g) = 1 / (1 + ex2(-g * log2e))."""
    return Uint32(llvm.inline_asm(
        T.i32(),
        [Uint32(attn_pair).ir_value(loc=loc, ip=ip),
         Uint32(gate_pair).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b32 a_lo32, a_hi32, g_lo32, g_hi32;
            .reg .f32 a_lo, a_hi, g_lo, g_hi;
            .reg .f32 ng_lo, ng_hi, prod_lo, prod_hi;
            .reg .f32 ex_lo, ex_hi, denom_lo, denom_hi;
            .reg .f32 sig_lo, sig_hi, out_lo, out_hi;
            .reg .f32 lge, one;
            mov.f32 lge, 0f3FB8AA3B;          // log2(e) ~ 1.4426950408
            mov.f32 one, 0f3F800000;          // 1.0

            // unpack attn_pair (bf16x2) -> 2 fp32
            and.b32 a_lo32, $1, 0xFFFF;
            shr.b32 a_hi32, $1, 16;
            shl.b32 a_lo32, a_lo32, 16;
            shl.b32 a_hi32, a_hi32, 16;
            mov.b32 a_lo, a_lo32;
            mov.b32 a_hi, a_hi32;

            // unpack gate_pair
            and.b32 g_lo32, $2, 0xFFFF;
            shr.b32 g_hi32, $2, 16;
            shl.b32 g_lo32, g_lo32, 16;
            shl.b32 g_hi32, g_hi32, 16;
            mov.b32 g_lo, g_lo32;
            mov.b32 g_hi, g_hi32;

            // sigmoid(g) = 1 / (1 + ex2(-g * log2e))
            neg.f32 ng_lo, g_lo;
            neg.f32 ng_hi, g_hi;
            mul.f32 prod_lo, ng_lo, lge;
            mul.f32 prod_hi, ng_hi, lge;
            ex2.approx.f32 ex_lo, prod_lo;
            ex2.approx.f32 ex_hi, prod_hi;
            add.f32 denom_lo, ex_lo, one;
            add.f32 denom_hi, ex_hi, one;
            rcp.approx.ftz.f32 sig_lo, denom_lo;
            rcp.approx.ftz.f32 sig_hi, denom_hi;

            // out = attn * sigmoid
            mul.f32 out_lo, a_lo, sig_lo;
            mul.f32 out_hi, a_hi, sig_hi;

            // pack to bf16x2
            cvt.rn.bf16x2.f32 $0, out_hi, out_lo;
        }
        """,
        "=r,r,r",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    ))


# ============================================================================
# High-level helpers (quantization_cute_dsl_utils.py)
# ============================================================================

@cute.jit
def bfloat2_max_abs_8(v0: Uint32, v1: Uint32, v2: Uint32, v3: Uint32,
                       v4: Uint32, v5: Uint32, v6: Uint32, v7: Uint32) -> Uint32:
    """Max abs of 8 bfloat2 (16 bf16) via tree reduction. Returns bfloat2."""
    abs0 = bfloat2_habs2(v0)
    abs1 = bfloat2_habs2(v1)
    abs2 = bfloat2_habs2(v2)
    abs3 = bfloat2_habs2(v3)
    abs4 = bfloat2_habs2(v4)
    abs5 = bfloat2_habs2(v5)
    abs6 = bfloat2_habs2(v6)
    abs7 = bfloat2_habs2(v7)
    max01 = bfloat2_hmax2(abs0, abs1)
    max23 = bfloat2_hmax2(abs2, abs3)
    max45 = bfloat2_hmax2(abs4, abs5)
    max67 = bfloat2_hmax2(abs6, abs7)
    max0123 = bfloat2_hmax2(max01, max23)
    max4567 = bfloat2_hmax2(max45, max67)
    return bfloat2_hmax2(max0123, max4567)


@cute.jit
def bfloat2x8_to_e2m1x16_packed(h0: Uint32, h1: Uint32, h2: Uint32, h3: Uint32,
                                 h4: Uint32, h5: Uint32, h6: Uint32, h7: Uint32,
                                 inv_scale: Float32) -> Uint64:
    """8 bfloat2 (16 bf16) -> 16 e2m1 packed as uint64 (8 bytes)."""
    s0, s1 = bfloat2_to_float2_scaled(h0, inv_scale)
    s2, s3 = bfloat2_to_float2_scaled(h1, inv_scale)
    s4, s5 = bfloat2_to_float2_scaled(h2, inv_scale)
    s6, s7 = bfloat2_to_float2_scaled(h3, inv_scale)
    s8, s9 = bfloat2_to_float2_scaled(h4, inv_scale)
    s10, s11 = bfloat2_to_float2_scaled(h5, inv_scale)
    s12, s13 = bfloat2_to_float2_scaled(h6, inv_scale)
    s14, s15 = bfloat2_to_float2_scaled(h7, inv_scale)

    packed_lo = cvt_e2m1x8_f32(s0, s1, s2, s3, s4, s5, s6, s7)
    packed_hi = cvt_e2m1x8_f32(s8, s9, s10, s11, s12, s13, s14, s15)
    return (Uint64(packed_hi) << Uint64(32)) | Uint64(packed_lo)


# ============================================================================
# Swizzled SF index (quantization_cute_dsl_utils.py)
# ============================================================================

@cute.jit
def compute_sf_index_swizzled_128x4_gpu(
    row_idx: Int32, col_idx: Int32, padded_cols: Int32,
) -> Int32:
    """CUTLASS-style swizzled-128x4 layout offset for one e4m3 scale byte."""
    kColumnGroup0Size = Int32(4)
    kRowGroup0Size = Int32(32)
    kRowGroup1Size = Int32(128)

    columnIdxInGroup0 = col_idx % kColumnGroup0Size
    columnGroupIdx = col_idx // kColumnGroup0Size
    columnGroupStride = Int32(512)

    rowIdxInGroup0 = row_idx % kRowGroup0Size
    rowIdxInGroup1 = (row_idx % kRowGroup1Size) // kRowGroup0Size
    rowGroupIdx = row_idx // kRowGroup1Size

    rowGroup1Stride = Int32(4)
    rowGroup0Stride = Int32(16)
    rowGroupStride = kRowGroup1Size * padded_cols

    offset = (
        columnIdxInGroup0
        + columnGroupIdx * columnGroupStride
        + rowIdxInGroup0 * rowGroup0Stride
        + rowIdxInGroup1 * rowGroup1Stride
        + rowGroupIdx * rowGroupStride
    )
    return offset


__all__ = [
    "NVFP4_SF_VEC_SIZE",
    "WARP_SIZE",
    "ld_global_v4_u32",
    "st_global_u64",
    "get_ptr_as_int64",
    "rcp_approx_ftz",
    "fmax_f32",
    "fabs_f32",
    "bfloat2_habs2",
    "bfloat2_hmax2",
    "bfloat2_mul",
    "bfloat2_hmax_reduce_to_f32",
    "bfloat2_to_float2_scaled",
    "bfloat2_to_float2",
    "f32x2_to_bfloat2",
    "cvt_f32_to_e4m3",
    "nvfp4_compute_output_scale",
    "cvt_e2m1x8_f32",
    "bfloat2_sigmoid_mul",
    "bfloat2_max_abs_8",
    "bfloat2x8_to_e2m1x16_packed",
    "compute_sf_index_swizzled_128x4_gpu",
]
