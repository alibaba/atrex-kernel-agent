#!/usr/bin/env python3
"""FlyDSL fwd_o kernel V6: BV=128, 1 wavefront (64 threads), load-first scheduling.

Computes: o = scale * (exp2(g) * (q @ h) + causal_mask(exp2(g_i - g_j) * (q @ k^T)) @ v_new)

Grid: (NT, B*H, 1), Block: 64 (1 wavefront)
BK=64, BV=128, MFMA 16x16x32 bf16

Matches Triton config: BV=128, 1 wavefront.
Loads issued first, stores batched, to help O=3 schedule overlaps.
LDS: lds_q(9KB, stride=72) + lds_b(17KB, stride=136) = 26KB.
"""
import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, buffer_ops, gpu, rocdl, vector, range_constexpr
from flydsl.expr.typing import T
from flydsl._mlir import ir
from flydsl._mlir.dialects import scf, math as math_dialect, llvm as llvm_d, memref as memref_d
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
BK = 64
BV = 128
K_SIZE = 128
V_SIZE = 128
H_SIZE = 32
Hg_SIZE = 8
BLOCK_SIZE = 64    # 1 wavefront
LDS_STRIDE_Q = 72     # 64 + 8 padding
LDS_STRIDE_B = 136    # 128 + 8 padding

LDS_Q_ELEMS = BT * LDS_STRIDE_Q       # 4608 bf16 = 9216 bytes
LDS_B_ELEMS = BK * LDS_STRIDE_B       # 8704 bf16 = 17408 bytes


def build_fwd_o():
    gpu_arch = get_rocm_arch()
    allocator = SmemAllocator(None, arch=gpu_arch, global_sym_name="fwd_o_smem")

    lds_q_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_q_offset + LDS_Q_ELEMS * 2
    lds_b_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_b_offset + LDS_B_ELEMS * 2

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def fwd_o_kernel(
        q_ptr: fx.Tensor,     # [B, T, Hg, K] bf16
        k_ptr: fx.Tensor,     # [B, T, Hg, K] bf16
        v_ptr: fx.Tensor,     # [B, T, H, V] bf16 (v_new)
        h_ptr: fx.Tensor,     # [B, NT, H, K, V] bf16
        g_ptr: fx.Tensor,     # [B, T, H] fp32 (cumsum gates)
        o_ptr: fx.Tensor,     # [B, T, H, V] bf16 (output)
        scale: fx.Float32,
        T_val: fx.Int32,
    ):
        H = H_SIZE
        Hg = Hg_SIZE
        K = K_SIZE
        V = V_SIZE

        v4f32 = T.vec(4, T.f32)
        v8bf16 = T.vec(8, T.bf16)
        v1bf16 = T.vec(1, T.bf16)
        _z = ir.IntegerAttr.get(ir.IntegerType.get_signless(32), 0)
        v4bf16 = T.vec(4, T.bf16)
        lds_ptr_ty = ir.Type.parse("!llvm.ptr<3>")

        def mfma(a, b, c):
            r = rocdl.mfma_f32_16x16x32_bf16(v4f32, [a, b, c, _z, _z, _z])
            return r.result if hasattr(r, 'result') else r

        def to_idx(v):
            return arith.index_cast(T.index, v)

        def v4e(vec, i):
            return vector.extract(vec, static_position=[i], dynamic_position=[])

        def v4m(vals):
            return vector.from_elements(v4f32, vals)

        def ds_read_tr(lds_elem_off):
            """Read v4bf16 from lds_b using hardware transpose."""
            raw_memref = arith.unwrap(lds_b)
            lds_base = memref_d.extract_aligned_pointer_as_index(raw_memref)
            byte_off = arith.ArithValue(to_idx(lds_elem_off)) * arith.index(2)
            total_byte = arith.ArithValue(lds_base) + byte_off
            addr_i32 = arith.index_cast(T.i32, total_byte)
            ptr = llvm_d.inttoptr(lds_ptr_ty, addr_i32)
            return rocdl.ds_read_tr16_b64(v4bf16, ptr).result

        def ds_read_tr_v8(lds_elem_lo, lds_elem_hi):
            """Read v8bf16 from lds_b using 2x hardware transpose reads."""
            v4_lo = ds_read_tr(lds_elem_lo)
            v4_hi = ds_read_tr(lds_elem_hi)
            elems = []
            for i in range_constexpr(4):
                elems.append(vector.extract(v4_lo, static_position=[i], dynamic_position=[]))
            for i in range_constexpr(4):
                elems.append(vector.extract(v4_hi, static_position=[i], dynamic_position=[]))
            return vector.from_elements(v8bf16, elems)

        # ═══ Thread/Block indices ═══
        # Grid: (B*H, NT, 1) — heads as fastest dim for L2 cache locality
        i_bh = gpu.block_idx.x     # batch*head index
        i_t = gpu.block_idx.y      # chunk index
        i_b = i_bh // H
        i_h = i_bh % H
        tid = gpu.thread_idx.x
        lane = tid  # 1 wavefront: lane == tid

        bos = i_b * T_val
        NT = (T_val + BT - 1) // BT
        i_tg = i_b * NT + i_t
        k_head = i_h // (H // Hg)

        # Buffer resources
        rsrc_q = buffer_ops.create_buffer_resource(q_ptr, max_size=True)
        rsrc_k = buffer_ops.create_buffer_resource(k_ptr, max_size=True)
        rsrc_v = buffer_ops.create_buffer_resource(v_ptr, max_size=True)
        rsrc_h = buffer_ops.create_buffer_resource(h_ptr, max_size=True)
        rsrc_g = buffer_ops.create_buffer_resource(g_ptr, max_size=True)
        rsrc_o = buffer_ops.create_buffer_resource(o_ptr, max_size=True)

        # LDS
        lds_base = allocator.get_base()
        lds_q = SmemPtr(lds_base, lds_q_offset, T.bf16, shape=(LDS_Q_ELEMS,)).get()
        lds_b = SmemPtr(lds_base, lds_b_offset, T.bf16, shape=(LDS_B_ELEMS,)).get()

        zero_f32 = arith.constant(0.0, type=T.f32)
        zero_v4 = v4m([zero_f32] * 4)

        # ═══ Accumulators (4 mt × 8 nt for q@h, 4 mt × 4 na for q@k^T) ═══
        b_o = [[zero_v4 for _ in range(8)] for _ in range(4)]
        b_A = [[zero_v4 for _ in range(4)] for _ in range(4)]

        # ═══════════ K-loop with load-ahead pipelining ═══════════
        # K-iter 0: Load q_0, h_0
        q_off_row_0 = (bos + i_t * BT + lane) * Hg * K + k_head * K + 0 * BK
        q_vecs = [None] * 8
        for j in range_constexpr(8):
            q_vecs[j] = buffer_ops.buffer_load(rsrc_q, q_off_row_0 + j * 8,
                                                vec_width=8, dtype=T.bf16)
        h_off_row_0 = ((i_tg * H + i_h) * K + 0 * BK + lane) * V
        h_vecs = [None] * 16
        for j in range_constexpr(16):
            h_vecs[j] = buffer_ops.buffer_load(rsrc_h, h_off_row_0 + j * 8,
                                                vec_width=8, dtype=T.bf16)

        for i_k in range_constexpr(K_SIZE // BK):

            # ── STORE q, h to LDS ──
            for j in range_constexpr(8):
                vector.store(q_vecs[j], lds_q, [to_idx(lane * LDS_STRIDE_Q + j * 8)])
            for j in range_constexpr(16):
                vector.store(h_vecs[j], lds_b, [to_idx(lane * LDS_STRIDE_B + j * 8)])

            # Prefetch k^T for this iteration (in flight during barrier + Dot 1)
            k_off_row = (bos + i_t * BT + lane) * Hg * K + k_head * K + i_k * BK
            k_vecs = [None] * 8
            for j in range_constexpr(8):
                k_vecs[j] = buffer_ops.buffer_load(rsrc_k, k_off_row + j * 8,
                                                    vec_width=8, dtype=T.bf16)

            gpu.barrier()

            # ── Dot 1: b_o += q @ h  (4 mt × 8 nt × 2 kt = 64 MFMAs) ──
            for kt in range_constexpr(2):
                b_h = [None] * 8
                for nt in range_constexpr(8):
                    gk = kt * 32 + (lane // 16) * 8
                    lds_lo = (gk + (lane % 16) // 4) * LDS_STRIDE_B + nt * 16 + (lane % 4) * 4
                    lds_hi = (gk + 4 + (lane % 16) // 4) * LDS_STRIDE_B + nt * 16 + (lane % 4) * 4
                    b_h[nt] = ds_read_tr_v8(lds_lo, lds_hi)
                for mt in range_constexpr(4):
                    a_r = mt * 16 + lane % 16
                    a_k = kt * 32 + (lane // 16) * 8
                    a = vector.load_op(v8bf16, lds_q,
                                       [to_idx(a_r * LDS_STRIDE_Q + a_k)])
                    for nt in range_constexpr(8):
                        b_o[mt][nt] = mfma(a, b_h[nt], b_o[mt][nt])

            gpu.barrier()

            # ── STORE k to LDS (data arrived during Dot 1) ──
            for j in range_constexpr(8):
                vector.store(k_vecs[j], lds_b, [to_idx(lane * LDS_STRIDE_B + j * 8)])

            # Prefetch q, h for NEXT K-iter (in flight during barrier + Dot 2)
            if i_k < (K_SIZE // BK) - 1:
                next_ik = i_k + 1
                q_off_row_n = (bos + i_t * BT + lane) * Hg * K + k_head * K + next_ik * BK
                for j in range_constexpr(8):
                    q_vecs[j] = buffer_ops.buffer_load(rsrc_q, q_off_row_n + j * 8,
                                                        vec_width=8, dtype=T.bf16)
                h_off_row_n = ((i_tg * H + i_h) * K + next_ik * BK + lane) * V
                for j in range_constexpr(16):
                    h_vecs[j] = buffer_ops.buffer_load(rsrc_h, h_off_row_n + j * 8,
                                                        vec_width=8, dtype=T.bf16)

            gpu.barrier()

            # ── Dot 2: b_A += q @ k^T  (4 mt × 4 na × 2 kt = 32 MFMAs) ──
            for kt in range_constexpr(2):
                b_kt = [None] * 4
                for na in range_constexpr(4):
                    b_c = na * 16 + lane % 16
                    b_k = kt * 32 + (lane // 16) * 8
                    b_kt[na] = vector.load_op(v8bf16, lds_b,
                                               [to_idx(b_c * LDS_STRIDE_B + b_k)])
                for mt in range_constexpr(4):
                    a_r = mt * 16 + lane % 16
                    a_k = kt * 32 + (lane // 16) * 8
                    a = vector.load_op(v8bf16, lds_q,
                                       [to_idx(a_r * LDS_STRIDE_Q + a_k)])
                    for na in range_constexpr(4):
                        b_A[mt][na] = mfma(a, b_kt[na], b_A[mt][na])

            gpu.barrier()

        # ═══════════ Gating ═══════════
        g_rows = [[None]*4 for _ in range(4)]
        for mt in range_constexpr(4):
            for ii in range_constexpr(4):
                row = mt * 16 + (lane // 16) * 4 + ii
                g_rows[mt][ii] = buffer_ops.buffer_load(
                    rsrc_g, (bos + i_t * BT + row) * H + i_h,
                    vec_width=1, dtype=T.f32)

        # Scale b_o by exp2(g_row)
        for mt in range_constexpr(4):
            for ii in range_constexpr(4):
                ge = math_dialect.exp2(g_rows[mt][ii])
                for nt in range_constexpr(8):
                    old = v4e(b_o[mt][nt], ii)
                    b_o[mt][nt] = vector.insert(
                        arith.mulf(old, ge), b_o[mt][nt],
                        static_position=[ii], dynamic_position=[])

        # Load g for columns
        g_cols = [None] * 4
        for na in range_constexpr(4):
            col = na * 16 + lane % 16
            g_cols[na] = buffer_ops.buffer_load(
                rsrc_g, (bos + i_t * BT + col) * H + i_h,
                vec_width=1, dtype=T.f32)

        # Scale b_A by exp2(g_row - g_col) + causal mask
        for mt in range_constexpr(4):
            for na in range_constexpr(4):
                for ii in range_constexpr(4):
                    row = mt * 16 + (lane // 16) * 4 + ii
                    col = na * 16 + lane % 16

                    gd = arith.subf(g_rows[mt][ii], g_cols[na])
                    gs = math_dialect.exp2(gd)
                    sa = arith.mulf(v4e(b_A[mt][na], ii), gs)

                    causal = arith.cmpi(arith.CmpIPredicate.sge, row, col)
                    r_ok = arith.cmpi(arith.CmpIPredicate.slt, i_t * BT + row, T_val)
                    c_ok = arith.cmpi(arith.CmpIPredicate.slt, i_t * BT + col, T_val)
                    mask = arith.andi(arith.andi(causal, r_ok), c_ok)
                    ma = arith.select(mask, sa, zero_f32)
                    b_A[mt][na] = vector.insert(
                        ma, b_A[mt][na],
                        static_position=[ii], dynamic_position=[])

        # ═══════════ Stage b_A → lds_q, v_new → lds_b ═══════════
        for mt in range_constexpr(4):
            for na in range_constexpr(4):
                for ii in range_constexpr(4):
                    row = mt * 16 + (lane // 16) * 4 + ii
                    col = na * 16 + lane % 16
                    val_bf16 = arith.truncf(T.bf16, v4e(b_A[mt][na], ii))
                    vector.store(vector.from_elements(v1bf16, [val_bf16]),
                                 lds_q, [to_idx(row * LDS_STRIDE_Q + col)])

        # v_new LOAD + STORE: 64 threads, each loads 1 row, 128 cols = 16 vec8
        v_off_row = ((bos + i_t * BT + lane) * H + i_h) * V
        v_vecs = [None] * 16
        for j in range_constexpr(16):
            v_vecs[j] = buffer_ops.buffer_load(rsrc_v, v_off_row + j * 8,
                                                vec_width=8, dtype=T.bf16)
        for j in range_constexpr(16):
            vector.store(v_vecs[j], lds_b, [to_idx(lane * LDS_STRIDE_B + j * 8)])

        gpu.barrier()

        # ═══════════ Dot 3: b_o += b_A @ v_new  (4 mt × 8 nt × 2 kt = 64 MFMAs) ═══════════
        for kt in range_constexpr(2):
            b_v = [None] * 8
            for nt in range_constexpr(8):
                gk = kt * 32 + (lane // 16) * 8
                lds_lo = (gk + (lane % 16) // 4) * LDS_STRIDE_B + nt * 16 + (lane % 4) * 4
                lds_hi = (gk + 4 + (lane % 16) // 4) * LDS_STRIDE_B + nt * 16 + (lane % 4) * 4
                b_v[nt] = ds_read_tr_v8(lds_lo, lds_hi)
            for mt in range_constexpr(4):
                a_r = mt * 16 + lane % 16
                a_k = kt * 32 + (lane // 16) * 8
                a = vector.load_op(v8bf16, lds_q,
                                   [to_idx(a_r * LDS_STRIDE_Q + a_k)])
                for nt in range_constexpr(8):
                    b_o[mt][nt] = mfma(a, b_v[nt], b_o[mt][nt])

        # ═══════════ Store o = b_o * scale ═══════════
        scale_v = arith.ArithValue(scale)
        for mt in range_constexpr(4):
            for nt in range_constexpr(8):
                for ii in range_constexpr(4):
                    row = mt * 16 + (lane // 16) * 4 + ii
                    col = nt * 16 + lane % 16
                    val = arith.mulf(v4e(b_o[mt][nt], ii), scale_v)
                    o_off = ((bos + i_t * BT + row) * H + i_h) * V + col
                    buffer_ops.buffer_store(arith.truncf(T.bf16, val), rsrc_o, o_off)

    @flyc.jit
    def launch(
        q_ptr: fx.Tensor, k_ptr: fx.Tensor, v_ptr: fx.Tensor,
        h_ptr: fx.Tensor, g_ptr: fx.Tensor, o_ptr: fx.Tensor,
        scale: fx.Float32, T_val: fx.Int32, B_val: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        NT = T_val // BT
        fwd_o_kernel(q_ptr, k_ptr, v_ptr, h_ptr, g_ptr, o_ptr, scale, T_val).launch(
            grid=(B_val * H_SIZE, NT, 1),
            block=(BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    return launch


_fwd_o_fn = None

def fwd_o_fwd(q, k, v_new, h, g, scale):
    """Compute output o = scale * (exp2(g)*(q@h) + causal(exp2(g_i-g_j)*(q@k^T))@v_new)."""
    global _fwd_o_fn
    if _fwd_o_fn is None:
        _fwd_o_fn = build_fwd_o()
    B, TT, Hg, K = q.shape
    H = v_new.shape[2]
    V = v_new.shape[3]
    o = torch.zeros(B, TT, H, V, device=q.device, dtype=q.dtype)
    _fwd_o_fn(q, k, v_new, h, g, o, scale, TT, B)
    return o
