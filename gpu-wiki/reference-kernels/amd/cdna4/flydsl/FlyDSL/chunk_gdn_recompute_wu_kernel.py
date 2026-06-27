#!/usr/bin/env python3
"""FlyDSL recompute_w_u kernel V2 for MI355X.

V1→V2: Row-major staging + ds_read_tr for B operand (replaces col-major scalar staging).
Grid reordered (B*H, NT, 1) for L2 cache locality.

Computes:
  u[BT, V] = A[BT, BT] @ (v[BT, V] * beta[BT, 1])
  w[BT, K] = A[BT, BT] @ (k[BT, K] * beta[BT, 1] * exp2(g[BT, 1]))

Grid: (B*H, NT, 1), Block: 256 (4 wavefronts)
MFMA: 16x16x32 bf16
LDS: lds_a (A row-major, stride=72, 9KB) + lds_b (vb/kb row-major, stride=136, 17KB) = 26KB
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
K_SIZE = 128
V_SIZE = 128
H_SIZE = 32
Hg_SIZE = 8
BLOCK_SIZE = 256  # 4 wavefronts
LDS_STRIDE_A = 72     # 64 + 8 padding (for A[64, 64])
LDS_STRIDE_B = 136    # 128 + 8 padding (for vb/kb[64, 128])

LDS_A_ELEMS = BT * LDS_STRIDE_A       # 4608 bf16 = 9216 bytes
LDS_B_ELEMS = BT * LDS_STRIDE_B       # 8704 bf16 = 17408 bytes


def build_recompute_wu():
    gpu_arch = get_rocm_arch()
    allocator = SmemAllocator(None, arch=gpu_arch, global_sym_name="recompute_wu_smem")

    lds_a_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_a_offset + LDS_A_ELEMS * 2
    lds_b_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_b_offset + LDS_B_ELEMS * 2

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def recompute_wu_kernel(
        k_ptr: fx.Tensor,    # [B, T, Hg, K] bf16
        v_ptr: fx.Tensor,    # [B, T, H, V] bf16
        beta_ptr: fx.Tensor, # [B, T, H] bf16
        w_ptr: fx.Tensor,    # [B, T, H, K] bf16 (output)
        u_ptr: fx.Tensor,    # [B, T, H, V] bf16 (output)
        A_ptr: fx.Tensor,    # [B, T, H, BT] bf16
        g_ptr: fx.Tensor,    # [B, T, H] fp32
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

        i_bh = gpu.block_idx.x
        i_t = gpu.block_idx.y
        i_b = i_bh // H
        i_h = i_bh % H
        tid = gpu.thread_idx.x
        wave_id = tid // 64
        lane = tid % 64

        bos = i_b * T_val
        k_head = i_h // (H // Hg)

        rsrc_k = buffer_ops.create_buffer_resource(k_ptr, max_size=True)
        rsrc_v = buffer_ops.create_buffer_resource(v_ptr, max_size=True)
        rsrc_beta = buffer_ops.create_buffer_resource(beta_ptr, max_size=True)
        rsrc_w = buffer_ops.create_buffer_resource(w_ptr, max_size=True)
        rsrc_u = buffer_ops.create_buffer_resource(u_ptr, max_size=True)
        rsrc_A = buffer_ops.create_buffer_resource(A_ptr, max_size=True)
        rsrc_g = buffer_ops.create_buffer_resource(g_ptr, max_size=True)

        lds_base = allocator.get_base()
        lds_a = SmemPtr(lds_base, lds_a_offset, T.bf16, shape=(LDS_A_ELEMS,)).get()
        lds_b = SmemPtr(lds_base, lds_b_offset, T.bf16, shape=(LDS_B_ELEMS,)).get()

        zero_f32 = arith.constant(0.0, type=T.f32)
        zero_v4 = v4m([zero_f32] * 4)

        # Thread mapping for staging [64, 128]: 256 threads
        # Each thread handles 2 rows × 16 cols (= 32 elems = 4 vec8)
        stg_row_base = tid // 8   # 0..31
        stg_col_base = (tid % 8) * 16  # 0, 16, 32, ..., 112

        # ═══ Stage A[64, 64] to lds_a (row-major, stride 72) — ONCE ═══
        a_row_base = tid // 8    # 0..31
        a_col_base = (tid % 8) * 8  # 0, 8, 16, ..., 56
        for ei in range_constexpr(2):
            a_row = a_row_base + 32 * ei  # 0..63
            a_off = (bos + i_t * BT + a_row) * H * BT + i_h * BT + a_col_base
            a_vec = buffer_ops.buffer_load(rsrc_A, a_off, vec_width=8, dtype=T.bf16)
            vector.store(a_vec, lds_a, [to_idx(a_row * LDS_STRIDE_A + a_col_base)])

        gpu.barrier()

        # ═══════════ u = A @ (v * beta) ═══════════
        # Stage vb[64, 128] to lds_b ROW-MAJOR (stride 136)
        # 256 threads, each handles 2 rows × 16 cols = 32 elems
        for ei in range_constexpr(2):
            row = stg_row_base + 32 * ei   # 0..63
            v_off = ((bos + i_t * BT + row) * H + i_h) * V + stg_col_base
            beta_off = (bos + i_t * BT + row) * H + i_h
            beta_val = buffer_ops.buffer_load(rsrc_beta, beta_off, vec_width=1, dtype=T.bf16)
            beta_f32 = arith.extf(T.f32, beta_val)

            for j in range_constexpr(2):
                v_vec = buffer_ops.buffer_load(rsrc_v, v_off + j * 8, vec_width=8, dtype=T.bf16)
                # Scale each element by beta and repack
                scaled = []
                for jj in range_constexpr(8):
                    elem = vector.extract(v_vec, static_position=[jj], dynamic_position=[])
                    s = arith.truncf(T.bf16, arith.mulf(arith.extf(T.f32, elem), beta_f32))
                    scaled.append(s)
                scaled_vec = vector.from_elements(v8bf16, scaled)
                vector.store(scaled_vec, lds_b, [to_idx(row * LDS_STRIDE_B + stg_col_base + j * 8)])

        gpu.barrier()

        # MFMA: u = A @ vb  (4 wf × 8 nt × 2 kt = 16 MFMAs per wf)
        u_accs = [zero_v4 for _ in range(8)]
        for kt in range_constexpr(2):
            b_vb = [None] * 8
            for nt in range_constexpr(8):
                gk = kt * 32 + (lane // 16) * 8
                lds_lo = (gk + (lane % 16) // 4) * LDS_STRIDE_B + nt * 16 + (lane % 4) * 4
                lds_hi = (gk + 4 + (lane % 16) // 4) * LDS_STRIDE_B + nt * 16 + (lane % 4) * 4
                b_vb[nt] = ds_read_tr_v8(lds_lo, lds_hi)
            a_r = wave_id * 16 + lane % 16
            a_k = kt * 32 + (lane // 16) * 8
            a_pack = vector.load_op(v8bf16, lds_a, [to_idx(a_r * LDS_STRIDE_A + a_k)])
            for nt in range_constexpr(8):
                u_accs[nt] = mfma(a_pack, b_vb[nt], u_accs[nt])

        # Store u
        for nt in range_constexpr(8):
            for ii in range_constexpr(4):
                row = wave_id * 16 + (lane // 16) * 4 + ii
                col = nt * 16 + lane % 16
                u_off = ((bos + i_t * BT + row) * H + i_h) * V + col
                u_bf16 = arith.truncf(T.bf16, v4e(u_accs[nt], ii))
                buffer_ops.buffer_store(u_bf16, rsrc_u, u_off)

        gpu.barrier()

        # ═══════════ w = A @ (k * beta * exp2(g)) ═══════════
        # Stage kb[64, 128] to lds_b ROW-MAJOR (stride 136)
        for ei in range_constexpr(2):
            row = stg_row_base + 32 * ei
            k_off = ((bos + i_t * BT + row) * Hg + k_head) * K + stg_col_base
            beta_off = (bos + i_t * BT + row) * H + i_h
            beta_val = buffer_ops.buffer_load(rsrc_beta, beta_off, vec_width=1, dtype=T.bf16)
            g_val = buffer_ops.buffer_load(rsrc_g, beta_off, vec_width=1, dtype=T.f32)
            g_exp2 = math_dialect.exp2(g_val)
            scale_val = arith.mulf(arith.extf(T.f32, beta_val), g_exp2)

            for j in range_constexpr(2):
                k_vec = buffer_ops.buffer_load(rsrc_k, k_off + j * 8, vec_width=8, dtype=T.bf16)
                scaled = []
                for jj in range_constexpr(8):
                    elem = vector.extract(k_vec, static_position=[jj], dynamic_position=[])
                    s = arith.truncf(T.bf16, arith.mulf(arith.extf(T.f32, elem), scale_val))
                    scaled.append(s)
                scaled_vec = vector.from_elements(v8bf16, scaled)
                vector.store(scaled_vec, lds_b, [to_idx(row * LDS_STRIDE_B + stg_col_base + j * 8)])

        gpu.barrier()

        # MFMA: w = A @ kb  (16 MFMAs per wf)
        w_accs = [zero_v4 for _ in range(8)]
        for kt in range_constexpr(2):
            b_kb = [None] * 8
            for nt in range_constexpr(8):
                gk = kt * 32 + (lane // 16) * 8
                lds_lo = (gk + (lane % 16) // 4) * LDS_STRIDE_B + nt * 16 + (lane % 4) * 4
                lds_hi = (gk + 4 + (lane % 16) // 4) * LDS_STRIDE_B + nt * 16 + (lane % 4) * 4
                b_kb[nt] = ds_read_tr_v8(lds_lo, lds_hi)
            a_r = wave_id * 16 + lane % 16
            a_k = kt * 32 + (lane // 16) * 8
            a_pack = vector.load_op(v8bf16, lds_a, [to_idx(a_r * LDS_STRIDE_A + a_k)])
            for nt in range_constexpr(8):
                w_accs[nt] = mfma(a_pack, b_kb[nt], w_accs[nt])

        # Store w
        for nt in range_constexpr(8):
            for ii in range_constexpr(4):
                row = wave_id * 16 + (lane // 16) * 4 + ii
                col = nt * 16 + lane % 16
                w_off = ((bos + i_t * BT + row) * H + i_h) * K + col
                w_bf16 = arith.truncf(T.bf16, v4e(w_accs[nt], ii))
                buffer_ops.buffer_store(w_bf16, rsrc_w, w_off)

    @flyc.jit
    def launch(
        k_ptr: fx.Tensor, v_ptr: fx.Tensor, beta_ptr: fx.Tensor,
        w_ptr: fx.Tensor, u_ptr: fx.Tensor,
        A_ptr: fx.Tensor, g_ptr: fx.Tensor,
        T_val: fx.Int32, B_val: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        NT = T_val // BT
        recompute_wu_kernel(k_ptr, v_ptr, beta_ptr, w_ptr, u_ptr, A_ptr, g_ptr, T_val).launch(
            grid=(B_val * H_SIZE, NT, 1),
            block=(BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    return launch


_recompute_wu_fn = None

def recompute_w_u_fwd(k, v, beta, A, g_cumsum):
    """Compute w and u from A, v, k, beta, g_cumsum."""
    global _recompute_wu_fn
    if _recompute_wu_fn is None:
        _recompute_wu_fn = build_recompute_wu()
    B, TT, Hg, K = k.shape
    H = v.shape[2]
    V = v.shape[3]
    w = torch.empty(B, TT, H, K, device=k.device, dtype=k.dtype)
    u = torch.empty(B, TT, H, V, device=v.device, dtype=v.dtype)
    _recompute_wu_fn(k, v, beta, w, u, A, g_cumsum, TT, B)
    return w, u
