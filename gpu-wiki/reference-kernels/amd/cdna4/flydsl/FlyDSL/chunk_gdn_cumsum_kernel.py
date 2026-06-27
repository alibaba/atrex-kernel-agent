#!/usr/bin/env python3
"""FlyDSL cumsum (inclusive prefix sum) kernel for MI355X.

Input: bf16 [B, T, H], Output: fp32 [B, T, H], optional scale.
Grid: (NT, B*H), Block: 64 (1 wavefront)
Algorithm: Hillis-Steele via ds_bpermute (wave-level, zero barriers).
"""
import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, buffer_ops, gpu, rocdl, vector, range_constexpr
from flydsl.expr.typing import T
from flydsl._mlir import ir
from flydsl._mlir.dialects import scf
from flydsl.runtime.device import get_rocm_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

# O=3 monkey-patch
from flydsl.compiler.backends.rocm import RocmBackend
_orig_pipeline = RocmBackend.pipeline_fragments
def _patched_pipeline(self, *, compile_hints=None, **kw):
    if compile_hints is None: compile_hints = {}
    frags = _orig_pipeline(self, compile_hints=compile_hints, **kw)
    return [f.replace('O=2', 'O=3') if 'rocdl-attach-target' in f else f for f in frags]
RocmBackend.pipeline_fragments = _patched_pipeline

BT = 64
H_SIZE = 32
HEADS_PER_BLOCK = 4  # 4 wavefronts, each handles 1 head
BLOCK_SIZE = 64 * HEADS_PER_BLOCK  # 256 threads

# Hardcoded max NT for grid sizing
MAX_NT = 1024  # supports T up to 65536


def build_cumsum():
    gpu_arch = get_rocm_arch()
    # Still need allocator for kernel launch (even if unused)
    allocator = SmemAllocator(None, arch=gpu_arch, global_sym_name="cumsum_smem")
    # Minimal LDS allocation (SmemAllocator requires >0)
    lds_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_offset + 16

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def cumsum_kernel(
        s_ptr: fx.Tensor,    # input g [B, T, H] bf16
        o_ptr: fx.Tensor,    # output  [B, T, H] fp32
        scale: fx.Float32,   # scale factor (use 1.0 for no scale)
        T_val: fx.Int32,
    ):
        H = H_SIZE
        HPB = HEADS_PER_BLOCK

        i_t = gpu.block_idx.x
        i_bh_base = gpu.block_idx.y  # block handles HPB consecutive heads
        i_b = i_bh_base // (H // HPB)
        i_h_base = (i_bh_base % (H // HPB)) * HPB
        tid = gpu.thread_idx.x

        # Each wavefront handles one head
        wave_id = tid // 64
        lane = tid % 64
        i_h = i_h_base + wave_id

        bos = i_b * T_val
        lane_v = arith.ArithValue(lane)
        zero_f32 = arith.constant(0.0, type=T.f32)
        scale_v = arith.ArithValue(scale)

        rsrc_s = buffer_ops.create_buffer_resource(s_ptr, max_size=True)
        rsrc_o = buffer_ops.create_buffer_resource(o_ptr, max_size=True)

        t_idx = i_t * BT + lane
        s_off = (bos + t_idx) * H + i_h

        in_bounds = arith.cmpi(arith.CmpIPredicate.slt, t_idx, T_val)

        val_bf16 = buffer_ops.buffer_load(rsrc_s, s_off, vec_width=1, dtype=T.bf16)
        val_f32 = arith.extf(T.f32, val_bf16)
        val_f32 = arith.select(in_bounds, val_f32, zero_f32)

        # Hillis-Steele via ds_bpermute (zero barriers, per-wavefront)
        for d_exp in range_constexpr(6):
            d = 1 << d_exp
            d_const = arith.constant(d, type=T.i32)
            has_neighbor = arith.cmpi(arith.CmpIPredicate.sge, lane_v, d_const)
            neighbor_lane = arith.subi(lane_v, d_const)
            safe_lane = arith.select(has_neighbor, neighbor_lane, lane_v)
            four = arith.constant(4, type=T.i32)
            byte_off = arith.muli(safe_lane, four)
            val_i32 = arith.bitcast(T.i32, val_f32)
            neighbor_i32 = rocdl.ds_bpermute(T.i32, byte_off, val_i32)
            neighbor_val = arith.bitcast(T.f32, neighbor_i32)
            new_val = arith.addf(val_f32, neighbor_val)
            val_f32 = arith.select(has_neighbor, new_val, val_f32)

        result_scaled = arith.mulf(val_f32, scale_v)

        o_off = (bos + t_idx) * H + i_h
        store_if = scf.IfOp(in_bounds, results_=[], has_else=True)
        with ir.InsertionPoint(store_if.then_block):
            buffer_ops.buffer_store(result_scaled, rsrc_o, o_off)
            scf.YieldOp([])
        with ir.InsertionPoint(store_if.else_block):
            scf.YieldOp([])

    @flyc.jit
    def launch(
        s_ptr: fx.Tensor, o_ptr: fx.Tensor,
        scale: fx.Float32, T_val: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        cumsum_kernel(s_ptr, o_ptr, scale, T_val).launch(
            grid=(MAX_NT, H_SIZE // HEADS_PER_BLOCK, 1),
            block=(BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    return launch


# ── Python wrapper ──
_cumsum_fn = None

def cumsum_fwd(g, scale_val=1.0):
    """Compute inclusive prefix sum per chunk.
    g: [B, T, H] bf16, returns [B, T, H] fp32.
    """
    global _cumsum_fn
    if _cumsum_fn is None:
        _cumsum_fn = build_cumsum()

    B, TT, H = g.shape
    assert TT <= MAX_NT * BT, f"T={TT} exceeds MAX_NT*BT={MAX_NT*BT}"
    o = torch.empty(B, TT, H, device=g.device, dtype=torch.float32)
    _cumsum_fn(g, o, scale_val, TT)
    return o


if __name__ == "__main__":
    # Quick test
    torch.manual_seed(42)
    B, TT, H = 1, 65536, 32
    g = torch.randn(B, TT, H, device="cuda", dtype=torch.bfloat16) * 0.1

    RCP_LN2 = 1.0 / 0.6931471805599453
    o_fly = cumsum_fwd(g, scale_val=RCP_LN2)

    # Reference: PyTorch cumsum per chunk
    BT = 64
    o_ref = torch.zeros(B, TT, H, device="cuda", dtype=torch.float32)
    for b in range(B):
        for t_start in range(0, TT, BT):
            t_end = min(t_start + BT, TT)
            chunk = g[b, t_start:t_end, :].float()
            o_ref[b, t_start:t_end, :] = chunk.cumsum(dim=0) * RCP_LN2

    diff = (o_fly - o_ref).abs()
    rel = diff / (o_ref.abs() + 1e-8)
    print(f"cumsum: max_abs={diff.max().item():.6e}, max_rel={rel.max().item():.6e}")
    print(f"  → {'PASS' if rel.max().item() < 1e-4 else 'FAIL'}")
