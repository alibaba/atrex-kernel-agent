#!/usr/bin/env python3
"""FlyDSL implementation of chunk_gated_delta_rule_fwd_kernel_h for MI355X.

V17: V11 IfOp style (per-element k-write, per-kt k-read) + grid_y=N*H fix.
O=3 monkey-patch for partial-wait instruction scheduling.

Grid: (V/BV, N*H) = (8, N*32)
Block: 256 threads (4 wavefronts)
"""
import torch
import numpy as np
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, buffer_ops, gpu, rocdl, vector, range_constexpr
from flydsl.expr.typing import T
from flydsl._mlir import ir
from flydsl._mlir.dialects import scf, fly as _fly, llvm as _llvm, math as math_dialect
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

# O=3 monkey-patch for better instruction scheduling
from flydsl.compiler.backends.rocm import RocmBackend
_orig_pipeline = RocmBackend.pipeline_fragments
def _patched_pipeline(self, *, compile_hints=None, **kw):
    if compile_hints is None: compile_hints = {}
    frags = _orig_pipeline(self, compile_hints=compile_hints, **kw)
    return [f.replace('O=2', 'O=3') if 'rocdl-attach-target' in f else f for f in frags]
RocmBackend.pipeline_fragments = _patched_pipeline

# ─── Constants ───
BT = 64
BV = 16
K_SIZE = 128
V_SIZE = 128
H_SIZE = 32
Hg_SIZE = 8
BLOCK_SIZE = 256  # 4 wavefronts
K_LDS_STRIDE = 72  # BT + 8 padding


def build_fwd_h():
    """Build the fwd_h kernel. Returns launch_fn."""
    K = K_SIZE
    V = V_SIZE
    H = H_SIZE
    Hg = Hg_SIZE

    LDS_B_H_ELEMS = 1024
    LDS_B_V_ELEMS = 1024
    LDS_K_ELEMS = 64 * K_LDS_STRIDE  # 4608 per buffer

    gpu_arch = get_hip_arch()
    allocator = SmemAllocator(None, arch=gpu_arch, global_sym_name="fwd_h_smem")

    lds_bh0_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_bh0_offset + LDS_B_H_ELEMS * 2
    lds_bh1_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_bh1_offset + LDS_B_H_ELEMS * 2
    lds_bv_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_bv_offset + LDS_B_V_ELEMS * 2

    # 4 separate k-LDS buffers (2 kc × 2 ping-pong)
    lds_k_kc0_buf0_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_k_kc0_buf0_offset + LDS_K_ELEMS * 2
    lds_k_kc1_buf0_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_k_kc1_buf0_offset + LDS_K_ELEMS * 2
    lds_k_kc0_buf1_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_k_kc0_buf1_offset + LDS_K_ELEMS * 2
    lds_k_kc1_buf1_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_k_kc1_buf1_offset + LDS_K_ELEMS * 2

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def fwd_h_kernel(
        k_ptr: fx.Tensor,
        v_ptr: fx.Tensor,
        w_ptr: fx.Tensor,
        v_new_ptr: fx.Tensor,
        g_ptr: fx.Tensor,
        h_ptr: fx.Tensor,
        ht_ptr: fx.Tensor,
        T_val: fx.Int32,
    ):
        v4f32_type = T.vec(4, T.f32)
        v8bf16_type = T.vec(8, T.bf16)
        v4bf16_type = T.vec(4, T.bf16)
        v1bf16_type = T.vec(1, T.bf16)
        _i32_zero = ir.IntegerAttr.get(ir.IntegerType.get_signless(32), 0)

        def mfma(a, b, c):
            res = rocdl.mfma_f32_16x16x32_bf16(
                v4f32_type, [a, b, c, _i32_zero, _i32_zero, _i32_zero])
            return res.result if hasattr(res, 'result') else res

        def to_idx(v):
            return arith.index_cast(T.index, v)

        def bf16_pack_v4(f32_vals):
            bf16_vals = [arith.truncf(T.bf16, v) for v in f32_vals]
            return vector.from_elements(v4bf16_type, bf16_vals)

        i_v = gpu.block_idx.x
        i_nh = gpu.block_idx.y
        i_n = i_nh // H
        i_h = i_nh % H
        tid = gpu.thread_idx.x
        wave_id = tid // 64
        lane = tid % 64

        bos = i_n * T_val
        NT = (T_val + BT - 1) // BT
        boh = i_n * NT

        stride_v = H * V
        stride_h = H * K * V
        stride_k = Hg * K
        stride_w = H * K

        h_base = (boh * H + i_h) * K * V
        v_base = (bos * H + i_h) * V
        k_base = (bos * Hg + i_h // (H // Hg)) * K
        w_base = (bos * H + i_h) * K
        g_base = bos * H + i_h
        vn_base = v_base

        rsrc_k = buffer_ops.create_buffer_resource(k_ptr, max_size=True)
        rsrc_v = buffer_ops.create_buffer_resource(v_ptr, max_size=True)
        rsrc_w = buffer_ops.create_buffer_resource(w_ptr, max_size=True)
        rsrc_vn = buffer_ops.create_buffer_resource(v_new_ptr, max_size=True)
        rsrc_g = buffer_ops.create_buffer_resource(g_ptr, max_size=True)
        rsrc_h = buffer_ops.create_buffer_resource(h_ptr, max_size=True)
        rsrc_ht = buffer_ops.create_buffer_resource(ht_ptr, max_size=True)

        lds_base_val = allocator.get_base()
        lds_bh0 = SmemPtr(lds_base_val, lds_bh0_offset, T.bf16, shape=(LDS_B_H_ELEMS,)).get()
        lds_bh1 = SmemPtr(lds_base_val, lds_bh1_offset, T.bf16, shape=(LDS_B_H_ELEMS,)).get()
        lds_bv = SmemPtr(lds_base_val, lds_bv_offset, T.bf16, shape=(LDS_B_V_ELEMS,)).get()

        # 4 separate k-LDS memrefs
        lds_k_kc0_buf0 = SmemPtr(lds_base_val, lds_k_kc0_buf0_offset, T.bf16, shape=(LDS_K_ELEMS,)).get()
        lds_k_kc1_buf0 = SmemPtr(lds_base_val, lds_k_kc1_buf0_offset, T.bf16, shape=(LDS_K_ELEMS,)).get()
        lds_k_kc0_buf1 = SmemPtr(lds_base_val, lds_k_kc0_buf1_offset, T.bf16, shape=(LDS_K_ELEMS,)).get()
        lds_k_kc1_buf1 = SmemPtr(lds_base_val, lds_k_kc1_buf1_offset, T.bf16, shape=(LDS_K_ELEMS,)).get()

        lds_bh = [lds_bh0, lds_bh1]
        lds_k_buf0 = [lds_k_kc0_buf0, lds_k_kc1_buf0]
        lds_k_buf1 = [lds_k_kc0_buf1, lds_k_kc1_buf1]

        def v4_extract(v, i):
            return vector.extract(v, static_position=[i], dynamic_position=[])

        def v4_make(vals):
            return vector.from_elements(v4f32_type, vals)

        def v4_scale(v, s):
            elems = [arith.mulf(v4_extract(v, i), s) for i in range_constexpr(4)]
            return v4_make(elems)

        def v4_sub(a, b):
            elems = [arith.subf(v4_extract(a, i), v4_extract(b, i)) for i in range_constexpr(4)]
            return v4_make(elems)

        zero_f32 = arith.constant(0.0, type=T.f32)
        zero_v4 = v4_make([zero_f32] * 4)
        init_h = [zero_v4, zero_v4]

        # ═══════════ Main loop ═══════════
        for i_t, inner_iter_args, loop_results in scf.for_(
            arith.index(0),
            to_idx(NT),
            arith.index(1),
            iter_args=init_h,
        ):
            i_t_i32 = arith.index_cast(T.i32, i_t)
            h_acc = list(inner_iter_args)

            # Compute buffer select: i_t % 2 == 0
            two = arith.constant(2, type=T.i32)
            buf_sel = arith.remui(i_t_i32, two)
            is_buf0 = arith.cmpi(arith.CmpIPredicate.eq, buf_sel, arith.constant(0, type=T.i32))

            # ════════ PHASE 1: h ops + k loads + PRE-LOAD w/v/g ════════

            for kc in range_constexpr(2):
                acc = h_acc[kc]
                f32_vals = [v4_extract(acc, i) for i in range_constexpr(4)]
                bf16_pack = bf16_pack_v4(f32_vals)

                for ii in range_constexpr(4):
                    val_bf16 = vector.extract(bf16_pack, static_position=[ii], dynamic_position=[])
                    row = kc * 64 + wave_id * 16 + (lane // 16) * 4 + ii
                    col = i_v * BV + lane % 16
                    h_off = h_base + i_t_i32 * stride_h + row * V + col
                    buffer_ops.buffer_store(val_bf16, rsrc_h, h_off)

                lds_store_off = (lane % 16) * 64 + wave_id * 16 + (lane // 16) * 4
                vector.store(bf16_pack, lds_bh[kc], [to_idx(lds_store_off)])

            # Pre-load w for both kc blocks (4 loads total)
            w_preloads = []
            for kc in range_constexpr(2):
                for k_half in range_constexpr(2):
                    w_off = w_base + (i_t_i32 * BT + wave_id * 16 + lane % 16) * stride_w + kc * 64 + k_half * 32 + (lane // 16) * 8
                    w_preloads.append(buffer_ops.buffer_load(rsrc_w, w_off, vec_width=8, dtype=T.bf16))

            # Pre-load v (4 scalar loads)
            v_preloads = []
            for ii in range_constexpr(4):
                row = i_t_i32 * BT + wave_id * 16 + (lane // 16) * 4 + ii
                col = i_v * BV + lane % 16
                v_off = v_base + row * stride_v + col
                v_preloads.append(buffer_ops.buffer_load(rsrc_v, v_off, vec_width=1, dtype=T.bf16))

            # Pre-load g (last and per-thread)
            last_idx_candidate = arith.ArithValue(arith.muli((i_t_i32 + 1), arith.constant(BT, type=T.i32)))
            T_val_v = arith.ArithValue(T_val)
            cmp_lt = arith.cmpi(arith.CmpIPredicate.slt, last_idx_candidate, T_val_v)
            last_idx_clamped = arith.select(cmp_lt, last_idx_candidate, T_val_v)
            last_idx = arith.subi(last_idx_clamped, arith.constant(1, type=T.i32))
            g_last_off = g_base + last_idx * H
            g_last = buffer_ops.buffer_load(rsrc_g, g_last_off, vec_width=1, dtype=T.f32)

            t_base = i_t_i32 * BT + wave_id * 16 + (lane // 16) * 4
            g_per_elem = []
            for ii in range_constexpr(4):
                g_elem_off = g_base + (t_base + ii) * H
                g_per_elem.append(buffer_ops.buffer_load(rsrc_g, g_elem_off, vec_width=1, dtype=T.f32))

            # Load k to double-buffered LDS — per-element IfOp for buffer selection
            for kc in range_constexpr(2):
                for load_iter in range_constexpr(2):
                    linear_idx = tid * 16 + load_iter * 8
                    t_local = linear_idx // 64
                    dk_start = linear_idx % 64
                    k_global_off = k_base + (i_t_i32 * BT + t_local) * stride_k + kc * 64 + dk_start

                    k_vec = buffer_ops.buffer_load(rsrc_k, k_global_off, vec_width=8, dtype=T.bf16)

                    for j in range_constexpr(8):
                        k_elem = vector.extract(k_vec, static_position=[j], dynamic_position=[])
                        lds_k_off = (dk_start + j) * K_LDS_STRIDE + t_local
                        k_elem_vec = vector.from_elements(v1bf16_type, [k_elem])
                        write_if = scf.IfOp(is_buf0, results_=[], has_else=True)
                        with ir.InsertionPoint(write_if.then_block):
                            vector.store(k_elem_vec, lds_k_buf0[kc], [to_idx(lds_k_off)])
                            scf.YieldOp([])
                        with ir.InsertionPoint(write_if.else_block):
                            vector.store(k_elem_vec, lds_k_buf1[kc], [to_idx(lds_k_off)])
                            scf.YieldOp([])

            # ════════ BARRIER 1: h and k are in LDS ════════
            gpu.barrier()

            # ════════ PHASE 2: w·h (USE pre-loaded w) ════════
            wh_acc = zero_v4

            for kc in range_constexpr(2):
                b_lds_off_0 = (lane % 16) * 64 + (lane // 16) * 8
                b_pack_0 = vector.load_op(v8bf16_type, lds_bh[kc], [to_idx(b_lds_off_0)])
                b_lds_off_1 = (lane % 16) * 64 + 32 + (lane // 16) * 8
                b_pack_1 = vector.load_op(v8bf16_type, lds_bh[kc], [to_idx(b_lds_off_1)])

                a_pack_0 = w_preloads[kc * 2 + 0]
                a_pack_1 = w_preloads[kc * 2 + 1]

                wh_acc = mfma(a_pack_0, b_pack_0, wh_acc)
                wh_acc = mfma(a_pack_1, b_pack_1, wh_acc)

            # ════════ PHASE 3: v_new (USE pre-loaded v and g) ════════
            v_elems = [arith.extf(T.f32, vp) for vp in v_preloads]
            v_v4 = v4_make(v_elems)
            vn_local = v4_sub(v_v4, wh_acc)

            f32_vals_vn = [v4_extract(vn_local, i) for i in range_constexpr(4)]
            bf16_pack_vn = bf16_pack_v4(f32_vals_vn)
            for ii in range_constexpr(4):
                val_bf16 = vector.extract(bf16_pack_vn, static_position=[ii], dynamic_position=[])
                row = i_t_i32 * BT + wave_id * 16 + (lane // 16) * 4 + ii
                col = i_v * BV + lane % 16
                vn_off = vn_base + row * stride_v + col
                buffer_ops.buffer_store(val_bf16, rsrc_vn, vn_off)

            # Gating (USE pre-loaded g) — per-element gating
            zero_f = arith.constant(0.0, type=T.f32)

            for ii in range_constexpr(4):
                t_abs = t_base + ii
                t_valid_i = arith.cmpi(arith.CmpIPredicate.slt, t_abs, T_val)
                g_diff_i = arith.subf(g_last, g_per_elem[ii])
                g_scale_i = math_dialect.exp2(g_diff_i)
                g_mask_i = arith.select(t_valid_i, g_scale_i, zero_f)
                vn_elem = v4_extract(vn_local, ii)
                vn_elem = arith.mulf(vn_elem, g_mask_i)
                vn_local = vector.insert(vn_elem, vn_local,
                                         static_position=[ii], dynamic_position=[])

            g_last_exp2 = math_dialect.exp2(g_last)
            for kc_idx in range_constexpr(2):
                h_acc[kc_idx] = v4_scale(h_acc[kc_idx], g_last_exp2)

            # ════════ PHASE 4: Stage gated v_new to lds_bv ════════
            f32_vals_vn2 = [v4_extract(vn_local, i) for i in range_constexpr(4)]
            bf16_pack_vn2 = bf16_pack_v4(f32_vals_vn2)
            lds_store_off2 = (lane % 16) * 64 + wave_id * 16 + (lane // 16) * 4
            vector.store(bf16_pack_vn2, lds_bv, [to_idx(lds_store_off2)])

            # ════════ BARRIER 2 ════════
            gpu.barrier()

            # ════════ PHASE 5: k^T · v_new (read k from correct buffer) ════════
            for kc in range_constexpr(2):
                for kt in range_constexpr(2):
                    dk = wave_id * 16 + lane % 16
                    t_start = kt * 32 + (lane // 16) * 8
                    k_lds_off = dk * K_LDS_STRIDE + t_start

                    read_if = scf.IfOp(is_buf0, results_=[v8bf16_type], has_else=True)
                    with ir.InsertionPoint(read_if.then_block):
                        a0 = vector.load_op(v8bf16_type, lds_k_buf0[kc], [to_idx(k_lds_off)])
                        scf.YieldOp([a0])
                    with ir.InsertionPoint(read_if.else_block):
                        a1 = vector.load_op(v8bf16_type, lds_k_buf1[kc], [to_idx(k_lds_off)])
                        scf.YieldOp([a1])
                    a_pack = read_if.results[0]

                    b_lds_off = (lane % 16) * 64 + kt * 32 + (lane // 16) * 8
                    b_pack = vector.load_op(v8bf16_type, lds_bv, [to_idx(b_lds_off)])

                    h_acc[kc] = mfma(a_pack, b_pack, h_acc[kc])

            # NO BARRIER 3 — double-buffered k-LDS makes this safe

            yield h_acc

        # ═══════════ Store final state to ht ═══════════
        ht_base = i_nh * K * V
        final_h = list(loop_results)
        for kc in range_constexpr(2):
            acc = final_h[kc]
            for ii in range_constexpr(4):
                row = kc * 64 + wave_id * 16 + (lane // 16) * 4 + ii
                col = i_v * BV + lane % 16
                ht_off = ht_base + row * V + col
                val_f32 = v4_extract(acc, ii)
                buffer_ops.buffer_store(val_f32, rsrc_ht, ht_off)

    @flyc.jit
    def launch(
        k_ptr: fx.Tensor, v_ptr: fx.Tensor, w_ptr: fx.Tensor,
        v_new_ptr: fx.Tensor, g_ptr: fx.Tensor, h_ptr: fx.Tensor,
        ht_ptr: fx.Tensor,
        T_val: fx.Int32,
        N_val: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        grid_x = V_SIZE // BV
        grid_y = N_val * H_SIZE

        fwd_h_kernel(k_ptr, v_ptr, w_ptr, v_new_ptr, g_ptr, h_ptr, ht_ptr, T_val).launch(
            grid=(grid_x, grid_y, 1),
            block=(BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    return launch


_fwd_h_fn = None

def fwd_h_fwd(k, w, u, g_cumsum):
    """Compute fwd_h: h state recurrence + v_new correction + final state."""
    global _fwd_h_fn
    if _fwd_h_fn is None:
        _fwd_h_fn = build_fwd_h()
    B, TT, Hg, K = k.shape
    H = u.shape[2]
    V = u.shape[3]
    NT = (TT + BT - 1) // BT
    h = torch.empty(B, NT, H, K, V, device=k.device, dtype=k.dtype)
    v_new = torch.empty_like(u)
    ht = torch.empty(B * H, K_SIZE, V_SIZE, device=k.device, dtype=torch.float32)
    _fwd_h_fn(k, u, w, v_new, g_cumsum, h, ht, TT, B)
    return h, v_new
