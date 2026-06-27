"""Minimal SM120 NVFP4 GEMM atom demo using CuTeDSL + inline PTX.

This file intentionally implements a single m16n8k64 NVFP4 GEMM atom.
It is a correctness reference for:

- SM120 warp MMA path
- inline PTX `mma.sync.aligned.kind::mxf4nvf4...`
- register packing for A/B/SFA/SFB operands

The kernel compares against a dequantized NVFP4 reference, not a dense BF16
reference, because the MMA operates on quantized payloads.
"""

import os

os.environ["CUTE_DSL_ARCH"] = "sm_120a"

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
import cutlass.utils as utils
import torch
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.cute.core import _pack_shape
from cutlass.cute.atom import make_atom
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.nvgpu.warp.mma import MmaMXF4NVF4Trait, MmaSM120BlockScaledOp
import cutlass._mlir.dialects.cute_nvgpu as _cute_nvgpu_ir

from flashinfer import SfLayout, e2m1_and_ufp8sf_scale_to_float, nvfp4_quantize


AB_DTYPE = cutlass.Float4E2M1FN
SF_DTYPE = cutlass.Float8E4M3FN
ACC_DTYPE = cutlass.Float32
U8_DTYPE = cutlass.Uint8

TILE_M = 16
TILE_N = 8
TILE_K = 64
THREADS = 32
SF_VEC_SIZE = 16


class MmaMXF4NVF4InlineOp(MmaSM120BlockScaledOp):
    descriptive_name = "sm120 nvfp4 inline-ptx mma op"

    def __init__(self):
        super().__init__(AB_DTYPE, ACC_DTYPE, (TILE_M, TILE_N, TILE_K), SF_DTYPE, SF_VEC_SIZE)

    def _make_trait(self, *, loc=None, ip=None, **kwargs):
        shape_mnk = _pack_shape(self.shape_mnk, loc=loc, ip=ip)
        ty = _cute_nvgpu_ir.MmaAtomSM120BlockScaledType.get(
            shape_mnk.type.attribute,
            SF_VEC_SIZE,
            False,
            self.ab_dtype.mlir_type,
            self.ab_dtype.mlir_type,
            self.acc_dtype.mlir_type,
            self.sf_type.mlir_type,
        )
        return MmaMXF4NVF4Trait(make_atom(ty, loc=loc, ip=ip))

    def _verify_fragment_A(self, input, *, loc=None, ip=None):
        return None

    def _verify_fragment_B(self, input, *, loc=None, ip=None):
        return None


def _ir(val, loc=None, ip=None):
    return val.ir_value(loc=loc, ip=ip) if hasattr(val, "ir_value") else val


def _extract(vec, idx: int, ty, *, loc=None, ip=None):
    del ty
    return llvm.extractelement(_ir(vec, loc, ip), _ir(cutlass.Int32(idx), loc, ip), loc=loc, ip=ip)


@dsl_user_op
def _mma_mxf4nvf4_inline(
    a_regs,
    b_regs,
    c_regs,
    sfa_reg,
    sfb_reg,
    *,
    loc=None,
    ip=None,
):
    a0 = _extract(a_regs, 0, T.i32(), loc=loc, ip=ip)
    a1 = _extract(a_regs, 1, T.i32(), loc=loc, ip=ip)
    a2 = _extract(a_regs, 2, T.i32(), loc=loc, ip=ip)
    a3 = _extract(a_regs, 3, T.i32(), loc=loc, ip=ip)
    b0 = _extract(b_regs, 0, T.i32(), loc=loc, ip=ip)
    b1 = _extract(b_regs, 1, T.i32(), loc=loc, ip=ip)
    c0 = _extract(c_regs, 0, T.f32(), loc=loc, ip=ip)
    c1 = _extract(c_regs, 1, T.f32(), loc=loc, ip=ip)
    c2 = _extract(c_regs, 2, T.f32(), loc=loc, ip=ip)
    c3 = _extract(c_regs, 3, T.f32(), loc=loc, ip=ip)
    rst = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32(), T.f32(), T.f32()]),
        [
            a0,
            a1,
            a2,
            a3,
            b0,
            b1,
            c0,
            c1,
            c2,
            c3,
            _ir(sfa_reg, loc, ip),
            _ir(cutlass.Int16(0), loc, ip),
            _ir(cutlass.Int16(0), loc, ip),
            _ir(sfb_reg, loc, ip),
            _ir(cutlass.Int16(0), loc, ip),
            _ir(cutlass.Int16(0), loc, ip),
        ],
        """mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3
           {$0, $1, $2, $3},
           {$4, $5, $6, $7},
           {$8, $9},
           {$10, $11, $12, $13},
           {$14},
           {$15, $16},
           {$17},
           {$18, $19};""",
        "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f,r,h,h,r,h,h",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    out = cute.make_rmem_tensor((4,), ACC_DTYPE)
    out[0] = cutlass.Float32(llvm.extractvalue(T.f32(), rst, [0]))
    out[1] = cutlass.Float32(llvm.extractvalue(T.f32(), rst, [1]))
    out[2] = cutlass.Float32(llvm.extractvalue(T.f32(), rst, [2]))
    out[3] = cutlass.Float32(llvm.extractvalue(T.f32(), rst, [3]))
    return out.load()


def _load_fp4_nibble(tensor_u8, row, k):
    byte_val = cutlass.Int32(tensor_u8[row, k // cutlass.Int32(2)])
    shift = (k & cutlass.Int32(1)) * cutlass.Int32(4)
    return (byte_val >> shift) & cutlass.Int32(0xF)


@cute.kernel
def _kernel(a_q: cute.Tensor, b_q: cute.Tensor, a_sf: cute.Tensor, b_sf: cute.Tensor, alpha: cute.Tensor, out: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()

    permutation = utils.blackwell_helpers.get_permutation_mnk((TILE_M, TILE_N, TILE_K), SF_VEC_SIZE, False)
    cute.make_tiled_mma(MmaMXF4NVF4InlineOp(), permutation_mnk=permutation)

    gA_u8 = cute.make_tensor(a_q.iterator, cute.make_layout((TILE_M, TILE_K // 2), stride=(TILE_K // 2, 1)))
    gB_u8 = cute.make_tensor(b_q.iterator, cute.make_layout((TILE_N, TILE_K // 2), stride=(TILE_K // 2, 1)))
    gSFA_u8 = cute.make_tensor(a_sf.iterator, cute.make_layout((128, 4), stride=(4, 1)))
    gSFB_u8 = cute.make_tensor(b_sf.iterator, cute.make_layout((128, 4), stride=(4, 1)))

    lane_group = tidx % cutlass.Int32(4)
    lane_row = tidx // cutlass.Int32(4)
    rA = cute.make_rmem_tensor((4,), cutlass.Int32)
    rB = cute.make_rmem_tensor((2,), cutlass.Int32)
    for i in cutlass.range_constexpr(4):
        rA[i] = cutlass.Int32(0)
    for i in cutlass.range_constexpr(2):
        rB[i] = cutlass.Int32(0)

    for linear in cutlass.range_constexpr(32):
        v0 = linear % 8
        v1 = (linear // 8) % 2
        v2 = linear // 16
        logical_row = lane_row + cutlass.Int32(8 * v1)
        logical_k = lane_group * cutlass.Int32(8) + cutlass.Int32(v0) + cutlass.Int32(32 * v2)
        nib = _load_fp4_nibble(gA_u8, logical_row, logical_k)
        reg_idx = cutlass.Int32(linear // 8)
        nib_idx = cutlass.Int32(linear % 8)
        rA[reg_idx] = rA[reg_idx] | (nib << (cutlass.Int32(4) * nib_idx))

    for linear in cutlass.range_constexpr(16):
        v0 = linear % 8
        v1 = linear // 8
        logical_row = lane_row
        logical_k = lane_group * cutlass.Int32(8) + cutlass.Int32(v0) + cutlass.Int32(32 * v1)
        nib = _load_fp4_nibble(gB_u8, logical_row, logical_k)
        reg_idx = cutlass.Int32(linear // 8)
        nib_idx = cutlass.Int32(linear % 8)
        rB[reg_idx] = rB[reg_idx] | (nib << (cutlass.Int32(4) * nib_idx))

    sfa_row = (tidx // cutlass.Int32(4)) + (tidx % cutlass.Int32(2)) * cutlass.Int32(8)
    sfb_row = tidx // cutlass.Int32(4)
    sfa_phys_row = sfa_row * cutlass.Int32(4)
    sfb_phys_row = sfb_row * cutlass.Int32(4)
    sfa_reg = (
        cutlass.Int32(gSFA_u8[sfa_phys_row, 0])
        | (cutlass.Int32(gSFA_u8[sfa_phys_row, 1]) << cutlass.Int32(8))
        | (cutlass.Int32(gSFA_u8[sfa_phys_row, 2]) << cutlass.Int32(16))
        | (cutlass.Int32(gSFA_u8[sfa_phys_row, 3]) << cutlass.Int32(24))
    )
    sfb_reg = (
        cutlass.Int32(gSFB_u8[sfb_phys_row, 0])
        | (cutlass.Int32(gSFB_u8[sfb_phys_row, 1]) << cutlass.Int32(8))
        | (cutlass.Int32(gSFB_u8[sfb_phys_row, 2]) << cutlass.Int32(16))
        | (cutlass.Int32(gSFB_u8[sfb_phys_row, 3]) << cutlass.Int32(24))
    )

    acc = cute.make_rmem_tensor((4,), ACC_DTYPE)
    acc.fill(0.0)
    acc.store(_mma_mxf4nvf4_inline(rA.load(), rB.load(), acc.load(), sfa_reg, sfb_reg))
    alpha_val = alpha[0]
    for i in cutlass.range_constexpr(cute.size(acc)):
        acc[i] = acc[i] * alpha_val

    # Verified output mapping:
    # acc[0] -> (m0, n0)
    # acc[1] -> (m0, n1)
    # acc[2] -> (m8, n0)
    # acc[3] -> (m8, n1)
    gOut = cute.make_tensor(out.iterator, cute.make_layout((TILE_M, TILE_N), stride=(TILE_N, 1)))
    n_base = lane_group * cutlass.Int32(2)
    gOut[lane_row + cutlass.Int32(0), n_base + cutlass.Int32(0)] = acc[0]
    gOut[lane_row + cutlass.Int32(0), n_base + cutlass.Int32(1)] = acc[1]
    gOut[lane_row + cutlass.Int32(8), n_base + cutlass.Int32(0)] = acc[2]
    gOut[lane_row + cutlass.Int32(8), n_base + cutlass.Int32(1)] = acc[3]


@cute.jit
def _launch(a_q, b_q, a_sf, b_sf, alpha, out, stream):
    _kernel(a_q, b_q, a_sf, b_sf, alpha, out).launch(grid=(1, 1, 1), block=(THREADS, 1, 1), stream=stream)


def _global_scale(x: torch.Tensor) -> torch.Tensor:
    return ((448.0 * 6.0) / x.float().abs().nan_to_num().max()).reshape(1).to(torch.float32)


def quantize_problem(a: torch.Tensor, b: torch.Tensor):
    a_g = _global_scale(a)
    b_g = _global_scale(b)
    a_q, a_sf = nvfp4_quantize(a, a_g, sfLayout=SfLayout.layout_128x4, do_shuffle=False, sf_vec_size=SF_VEC_SIZE)
    b_q, b_sf = nvfp4_quantize(b, b_g, sfLayout=SfLayout.layout_128x4, do_shuffle=False, sf_vec_size=SF_VEC_SIZE)
    alpha = (1.0 / (a_g * b_g)).reshape(1).to(torch.float32)
    return a_q.contiguous(), a_sf.contiguous(), b_q.contiguous(), b_sf.contiguous(), a_g, b_g, alpha


def run_demo(device="cuda"):
    torch.manual_seed(0)
    a = torch.randn((TILE_M, TILE_K), device=device, dtype=torch.bfloat16)
    b = torch.randn((TILE_N, TILE_K), device=device, dtype=torch.bfloat16)
    a_q, a_sf, b_q, b_sf, a_g, b_g, alpha = quantize_problem(a, b)

    out = torch.empty((TILE_M, TILE_N), device=device, dtype=torch.float32)
    stream = cutlass_torch.default_stream()
    _launch(
        from_dlpack(a_q, assumed_align=16),
        from_dlpack(b_q, assumed_align=16),
        from_dlpack(a_sf, assumed_align=16),
        from_dlpack(b_sf, assumed_align=16),
        from_dlpack(alpha, assumed_align=4),
        from_dlpack(out, assumed_align=16),
        stream,
    )
    torch.cuda.synchronize()

    a_deq = e2m1_and_ufp8sf_scale_to_float(
        a_q,
        a_sf,
        1.0 / a_g,
        sf_vec_size=SF_VEC_SIZE,
        ufp8_type=1,
        is_sf_swizzled_layout=True,
    ).to(device=device)
    b_deq = e2m1_and_ufp8sf_scale_to_float(
        b_q,
        b_sf,
        1.0 / b_g,
        sf_vec_size=SF_VEC_SIZE,
        ufp8_type=1,
        is_sf_swizzled_layout=True,
    ).to(device=device)
    ref = a_deq.float() @ b_deq.float().T
    dense_ref = a.float() @ b.float().T
    rel = (out - ref).norm() / ref.norm().clamp_min(1e-8)
    max_abs = (out - ref).abs().max()
    dense_rel = (out - dense_ref).norm() / dense_ref.norm().clamp_min(1e-8)
    dense_max_abs = (out - dense_ref).abs().max()
    print("out[0,:8] =", out[0, :8].tolist())
    print("ref[0,:8] =", ref[0, :8].tolist())
    print("dense_ref[0,:8] =", dense_ref[0, :8].tolist())
    print(f"dequant_ref_rel_err={rel.item():.6f}, dequant_ref_max_abs={max_abs.item():.6f}")
    print(f"dense_ref_rel_err={dense_rel.item():.6f}, dense_ref_max_abs={dense_max_abs.item():.6f}")


if __name__ == "__main__":
    run_demo()
