#!/usr/bin/env python3
"""FlyDSL fused_kkt_solve kernel V1 for MI355X.

V0→V1: Vec8 k staging, MFMA-based mat_mul_16x16 (replaces scalar accumulation).

Fuses k@k^T + gating + beta + triangular solve + block merge.
Grid: (NT, B*H), Block: 64 (1 wavefront)
BC=16, BK=64, MFMA 16x16x32 bf16
"""
import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, buffer_ops, gpu, rocdl, vector, range_constexpr
from flydsl.expr.typing import T
from flydsl._mlir import ir
from flydsl._mlir.dialects import scf, math as math_dialect
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
BC = 16
BK_TILE = 64
K_SIZE = 128
H_SIZE = 32
Hg_SIZE = 8
BLOCK_SIZE = 64

LDS_K_ROW_STRIDE = BK_TILE + 8  # 72, row-major for k sub-chunks
LDS_MAT_STRIDE = BC + 8         # 24, for 16x16 matrices


def build_kkt_solve():
    gpu_arch = get_rocm_arch()
    allocator = SmemAllocator(None, arch=gpu_arch, global_sym_name="kkt_solve_smem")

    LDS_K_ELEMS = BC * LDS_K_ROW_STRIDE  # 1152
    lds_k0_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_k0_offset + LDS_K_ELEMS * 2
    lds_k1_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_k1_offset + LDS_K_ELEMS * 2
    lds_k2_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_k2_offset + LDS_K_ELEMS * 2
    lds_k3_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_k3_offset + LDS_K_ELEMS * 2

    LDS_MAT_ELEMS = 16 * LDS_MAT_STRIDE  # 384
    lds_mata_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_mata_offset + LDS_MAT_ELEMS * 2
    lds_matb_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_matb_offset + LDS_MAT_ELEMS * 2

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def kkt_solve_kernel(
        k_ptr: fx.Tensor,
        g_ptr: fx.Tensor,
        beta_ptr: fx.Tensor,
        A_ptr: fx.Tensor,
        T_val: fx.Int32,
    ):
        H = H_SIZE
        Hg = Hg_SIZE
        K = K_SIZE

        v4f32_type = T.vec(4, T.f32)
        v8bf16_type = T.vec(8, T.bf16)
        v1bf16_type = T.vec(1, T.bf16)
        v4bf16_type = T.vec(4, T.bf16)
        _i32_zero = ir.IntegerAttr.get(ir.IntegerType.get_signless(32), 0)

        def mfma(a, b, c):
            res = rocdl.mfma_f32_16x16x32_bf16(
                v4f32_type, [a, b, c, _i32_zero, _i32_zero, _i32_zero])
            return res.result if hasattr(res, 'result') else res

        def to_idx(v):
            return arith.index_cast(T.index, v)

        def v4_extract(v, i):
            return vector.extract(v, static_position=[i], dynamic_position=[])

        def v4_make(vals):
            return vector.from_elements(v4f32_type, vals)

        def bf16_pack_v4(f32_vals):
            bf16_vals = [arith.truncf(T.bf16, v) for v in f32_vals]
            return vector.from_elements(v4bf16_type, bf16_vals)

        i_t = gpu.block_idx.x
        i_bh = gpu.block_idx.y
        i_b = i_bh // H
        i_h = i_bh % H
        tid = gpu.thread_idx.x
        lane = tid

        bos = i_b * T_val
        k_head = i_h // (H // Hg)

        rsrc_k = buffer_ops.create_buffer_resource(k_ptr, max_size=True)
        rsrc_g = buffer_ops.create_buffer_resource(g_ptr, max_size=True)
        rsrc_beta = buffer_ops.create_buffer_resource(beta_ptr, max_size=True)
        rsrc_A = buffer_ops.create_buffer_resource(A_ptr, max_size=True)

        lds_base_val = allocator.get_base()
        lds_k = [
            SmemPtr(lds_base_val, lds_k0_offset, T.bf16, shape=(LDS_K_ELEMS,)).get(),
            SmemPtr(lds_base_val, lds_k1_offset, T.bf16, shape=(LDS_K_ELEMS,)).get(),
            SmemPtr(lds_base_val, lds_k2_offset, T.bf16, shape=(LDS_K_ELEMS,)).get(),
            SmemPtr(lds_base_val, lds_k3_offset, T.bf16, shape=(LDS_K_ELEMS,)).get(),
        ]
        lds_mata = SmemPtr(lds_base_val, lds_mata_offset, T.bf16, shape=(LDS_MAT_ELEMS,)).get()
        lds_matb = SmemPtr(lds_base_val, lds_matb_offset, T.bf16, shape=(LDS_MAT_ELEMS,)).get()

        zero_f32 = arith.constant(0.0, type=T.f32)
        zero_v4 = v4_make([zero_f32] * 4)
        zero_bf16 = arith.constant(0.0, type=T.bf16)
        zero_v8bf16 = vector.from_elements(v8bf16_type, [zero_bf16] * 8)

        lane_v = arith.ArithValue(lane)
        c16 = arith.constant(16, type=T.i32)
        c_mat_stride = arith.constant(LDS_MAT_STRIDE, type=T.i32)

        # Helper: check if lane group is valid for MFMA K=16 (only lane//16 < 2)
        lane_div16 = arith.divui(lane_v, c16)
        c2 = arith.constant(2, type=T.i32)
        is_k_valid = arith.cmpi(arith.CmpIPredicate.slt, lane_div16, c2)

        # ════════ Step 1: Load k sub-chunks (vec8) and compute k@k^T ════════
        A00 = zero_v4; A11 = zero_v4; A22 = zero_v4; A33 = zero_v4
        A10 = zero_v4; A20 = zero_v4; A21 = zero_v4
        A30 = zero_v4; A31 = zero_v4; A32 = zero_v4

        # Thread mapping for k staging: 64 threads, 16 rows × 64 cols per sub-chunk
        k_row = tid // 4     # 0..15
        k_col_grp = tid % 4  # 0..3, each handles 16 cols

        for i_k_tile in range_constexpr(K_SIZE // BK_TILE):
            # Vec8 staging: 4 sub-chunks × 2 vec8 loads per thread
            for sc in range_constexpr(4):
                k_row_abs = bos + i_t * BT + sc * BC + k_row
                k_off_base = (k_row_abs * Hg + k_head) * K + i_k_tile * BK_TILE + k_col_grp * 16
                for j in range_constexpr(2):
                    k_off = k_off_base + j * 8
                    k_vec = buffer_ops.buffer_load(rsrc_k, k_off, vec_width=8, dtype=T.bf16)
                    lds_off = k_row * LDS_K_ROW_STRIDE + k_col_grp * 16 + j * 8
                    vector.store(k_vec, lds_k[sc], [to_idx(lds_off)])

            gpu.barrier()

            # k@k^T MFMA (10 blocks)
            for kt in range_constexpr(BK_TILE // 32):
                a_row_m = lane % 16
                a_col_base = kt * 32 + (lane // 16) * 8
                lds_off_k = a_row_m * LDS_K_ROW_STRIDE + a_col_base

                k0 = vector.load_op(v8bf16_type, lds_k[0], [to_idx(lds_off_k)])
                k1 = vector.load_op(v8bf16_type, lds_k[1], [to_idx(lds_off_k)])
                k2 = vector.load_op(v8bf16_type, lds_k[2], [to_idx(lds_off_k)])
                k3 = vector.load_op(v8bf16_type, lds_k[3], [to_idx(lds_off_k)])

                A00 = mfma(k0, k0, A00); A11 = mfma(k1, k1, A11)
                A22 = mfma(k2, k2, A22); A33 = mfma(k3, k3, A33)
                A10 = mfma(k1, k0, A10); A20 = mfma(k2, k0, A20)
                A21 = mfma(k2, k1, A21); A30 = mfma(k3, k0, A30)
                A31 = mfma(k3, k1, A31); A32 = mfma(k3, k2, A32)

            gpu.barrier()

        # ════════ Step 2: Gating + beta ════════
        def apply_gating_and_mask(block_acc, sc_row, sc_col, is_diagonal):
            for ii in range_constexpr(4):
                my_row = (lane // 16) * 4 + ii
                my_col = lane % 16
                abs_row = sc_row * BC + my_row
                abs_col = sc_col * BC + my_col

                g_row_off = (bos + i_t * BT + abs_row) * H + i_h
                g_col_off = (bos + i_t * BT + abs_col) * H + i_h
                g_row_val = buffer_ops.buffer_load(rsrc_g, g_row_off, vec_width=1, dtype=T.f32)
                g_col_val = buffer_ops.buffer_load(rsrc_g, g_col_off, vec_width=1, dtype=T.f32)
                g_diff = arith.subf(g_row_val, g_col_val)
                g_scale = math_dialect.exp2(g_diff)

                beta_off = (bos + i_t * BT + abs_row) * H + i_h
                beta_val = buffer_ops.buffer_load(rsrc_beta, beta_off, vec_width=1, dtype=T.bf16)
                beta_f32 = arith.extf(T.f32, beta_val)

                old_val = v4_extract(block_acc, ii)
                scaled = arith.mulf(arith.mulf(old_val, g_scale), beta_f32)

                if is_diagonal:
                    is_lower = arith.cmpi(arith.CmpIPredicate.sgt,
                                          arith.ArithValue(my_row),
                                          arith.ArithValue(my_col))
                    scaled = arith.select(is_lower, scaled, zero_f32)

                block_acc = vector.insert(scaled, block_acc, static_position=[ii], dynamic_position=[])
            return block_acc

        A00 = apply_gating_and_mask(A00, 0, 0, True)
        A11 = apply_gating_and_mask(A11, 1, 1, True)
        A22 = apply_gating_and_mask(A22, 2, 2, True)
        A33 = apply_gating_and_mask(A33, 3, 3, True)
        A10 = apply_gating_and_mask(A10, 1, 0, False)
        A20 = apply_gating_and_mask(A20, 2, 0, False)
        A21 = apply_gating_and_mask(A21, 2, 1, False)
        A30 = apply_gating_and_mask(A30, 3, 0, False)
        A31 = apply_gating_and_mask(A31, 3, 1, False)
        A32 = apply_gating_and_mask(A32, 3, 2, False)

        # ════════ Step 3: Triangular solve ════════
        col_id_v = arith.remui(lane_v, c16)

        def solve_diagonal(A_block):
            """Invert (I - A_block) where A_block is strictly lower triangular 16x16."""
            f32_vals = [v4_extract(A_block, i) for i in range_constexpr(4)]
            neg_vals = [arith.negf(v) for v in f32_vals]
            bf16_pack = bf16_pack_v4(neg_vals)

            for ii in range_constexpr(4):
                row = (lane // 16) * 4 + ii
                col = lane % 16
                lds_off = col * LDS_MAT_STRIDE + row
                val = vector.extract(bf16_pack, static_position=[ii], dynamic_position=[])
                vector.store(vector.from_elements(v1bf16_type, [val]),
                             lds_mata, [to_idx(lds_off)])

            gpu.barrier()

            for i_idx in scf.for_(arith.index(2), arith.index(16), arith.index(1)):
                i_val = arith.index_cast(T.i32, i_idx)

                a_row_i_off = arith.addi(arith.muli(col_id_v, c_mat_stride), i_val)
                a_row_i_vec = vector.load_op(v1bf16_type, lds_mata, [to_idx(a_row_i_off)])
                a_row_i_f32 = arith.extf(T.f32, vector.extract(a_row_i_vec, static_position=[0], dynamic_position=[]))

                col_lt_i = arith.cmpi(arith.CmpIPredicate.slt, col_id_v, i_val)
                a_row_i_f32 = arith.select(col_lt_i, a_row_i_f32, zero_f32)

                for k in range_constexpr(16):
                    k_const = arith.constant(k, type=T.i32)
                    k_lt_i = arith.cmpi(arith.CmpIPredicate.slt, k_const, i_val)

                    a_ik_off_v = arith.addi(arith.muli(k_const, c_mat_stride), i_val)
                    a_ik_vec = vector.load_op(v1bf16_type, lds_mata, [to_idx(a_ik_off_v)])
                    a_ik = arith.extf(T.f32, vector.extract(a_ik_vec, static_position=[0], dynamic_position=[]))

                    ai_km_off_v = arith.addi(arith.muli(col_id_v, c_mat_stride), k_const)
                    ai_km_vec = vector.load_op(v1bf16_type, lds_mata, [to_idx(ai_km_off_v)])
                    ai_km = arith.extf(T.f32, vector.extract(ai_km_vec, static_position=[0], dynamic_position=[]))

                    product = arith.mulf(a_ik, ai_km)
                    masked_product = arith.select(k_lt_i, product, zero_f32)
                    a_row_i_f32 = arith.addf(a_row_i_f32, masked_product)

                gpu.barrier()

                acc_bf16 = arith.truncf(T.bf16, a_row_i_f32)
                vector.store(vector.from_elements(v1bf16_type, [acc_bf16]),
                             lds_mata, [to_idx(a_row_i_off)])

                gpu.barrier()

            # Add identity
            for ii in range_constexpr(4):
                row = (lane // 16) * 4 + ii
                col = lane % 16
                lds_off = col * LDS_MAT_STRIDE + row
                val_vec = vector.load_op(v1bf16_type, lds_mata, [to_idx(lds_off)])
                val_f32 = arith.extf(T.f32, vector.extract(val_vec, static_position=[0], dynamic_position=[]))
                is_diag = arith.cmpi(arith.CmpIPredicate.eq,
                                     arith.ArithValue(row), arith.ArithValue(col))
                one_f32 = arith.constant(1.0, type=T.f32)
                diag_add = arith.select(is_diag, one_f32, zero_f32)
                val_f32 = arith.addf(val_f32, diag_add)
                vector.store(vector.from_elements(v1bf16_type, [arith.truncf(T.bf16, val_f32)]),
                             lds_mata, [to_idx(lds_off)])

            gpu.barrier()

            result = zero_v4
            for ii in range_constexpr(4):
                row = (lane // 16) * 4 + ii
                col = lane % 16
                lds_off = col * LDS_MAT_STRIDE + row
                val_vec = vector.load_op(v1bf16_type, lds_mata, [to_idx(lds_off)])
                val_f32 = arith.extf(T.f32, vector.extract(val_vec, static_position=[0], dynamic_position=[]))
                result = vector.insert(val_f32, result, static_position=[ii], dynamic_position=[])

            return result

        Ai00 = solve_diagonal(A00)
        Ai11 = solve_diagonal(A11)
        Ai22 = solve_diagonal(A22)
        Ai33 = solve_diagonal(A33)

        # ════════ Step 4: Block merge (MFMA-based mat_mul) ════════
        def mat_mul_16x16(A_acc, B_acc):
            """Compute C = A @ B using MFMA 16x16x32 (K=16, zero-padded to 32)."""
            # Stage A to lds_mata ROW-MAJOR: lds[row * stride + k]
            # Stage B to lds_matb COL-MAJOR: lds[col * stride + k]
            a_bf16 = bf16_pack_v4([v4_extract(A_acc, i) for i in range_constexpr(4)])
            b_bf16 = bf16_pack_v4([v4_extract(B_acc, i) for i in range_constexpr(4)])

            for ii in range_constexpr(4):
                row = (lane // 16) * 4 + ii
                col = lane % 16
                a_val = vector.extract(a_bf16, static_position=[ii], dynamic_position=[])
                b_val = vector.extract(b_bf16, static_position=[ii], dynamic_position=[])
                # A row-major: lds_mata[row * stride + col] where col = K dimension
                vector.store(vector.from_elements(v1bf16_type, [a_val]),
                             lds_mata, [to_idx(row * LDS_MAT_STRIDE + col)])
                # B col-major: lds_matb[col * stride + row] where row = K dimension
                vector.store(vector.from_elements(v1bf16_type, [b_val]),
                             lds_matb, [to_idx(col * LDS_MAT_STRIDE + row)])

            gpu.barrier()

            # MFMA load with conditional zero for K>=16 lanes
            a_row_m = lane % 16
            a_k_base = (lane // 16) * 8
            a_lds_off = a_row_m * LDS_MAT_STRIDE + a_k_base

            a_if = scf.IfOp(is_k_valid, results_=[v8bf16_type], has_else=True)
            with ir.InsertionPoint(a_if.then_block):
                a_real = vector.load_op(v8bf16_type, lds_mata, [to_idx(a_lds_off)])
                scf.YieldOp([a_real])
            with ir.InsertionPoint(a_if.else_block):
                scf.YieldOp([zero_v8bf16])
            a_pack = a_if.results[0]

            b_col_n = lane % 16
            b_k_base = (lane // 16) * 8
            b_lds_off = b_col_n * LDS_MAT_STRIDE + b_k_base

            b_if = scf.IfOp(is_k_valid, results_=[v8bf16_type], has_else=True)
            with ir.InsertionPoint(b_if.then_block):
                b_real = vector.load_op(v8bf16_type, lds_matb, [to_idx(b_lds_off)])
                scf.YieldOp([b_real])
            with ir.InsertionPoint(b_if.else_block):
                scf.YieldOp([zero_v8bf16])
            b_pack = b_if.results[0]

            result = mfma(a_pack, b_pack, zero_v4)

            gpu.barrier()
            return result

        def v4_neg(v):
            vals = [arith.negf(v4_extract(v, i)) for i in range_constexpr(4)]
            return v4_make(vals)

        def v4_add(a, b):
            vals = [arith.addf(v4_extract(a, i), v4_extract(b, i)) for i in range_constexpr(4)]
            return v4_make(vals)

        Ai10 = v4_neg(mat_mul_16x16(mat_mul_16x16(Ai11, A10), Ai00))
        Ai21 = v4_neg(mat_mul_16x16(mat_mul_16x16(Ai22, A21), Ai11))
        Ai32 = v4_neg(mat_mul_16x16(mat_mul_16x16(Ai33, A32), Ai22))
        Ai20 = v4_neg(mat_mul_16x16(Ai22, v4_add(mat_mul_16x16(A20, Ai00), mat_mul_16x16(A21, Ai10))))
        Ai31 = v4_neg(mat_mul_16x16(Ai33, v4_add(mat_mul_16x16(A31, Ai11), mat_mul_16x16(A32, Ai21))))
        Ai30 = v4_neg(mat_mul_16x16(Ai33, v4_add(v4_add(mat_mul_16x16(A30, Ai00), mat_mul_16x16(A31, Ai10)), mat_mul_16x16(A32, Ai20))))

        # ════════ Step 5: Store result ════════
        def store_block(block, sc_row, sc_col):
            bf16_pack = bf16_pack_v4([v4_extract(block, i) for i in range_constexpr(4)])
            for ii in range_constexpr(4):
                row = (lane // 16) * 4 + ii
                col = lane % 16
                t = i_t * BT + sc_row * BC + row
                bt = sc_col * BC + col
                a_off = ((bos + t) * H + i_h) * BT + bt
                val = vector.extract(bf16_pack, static_position=[ii], dynamic_position=[])
                buffer_ops.buffer_store(val, rsrc_A, a_off)

        store_block(Ai00, 0, 0); store_block(Ai11, 1, 1)
        store_block(Ai22, 2, 2); store_block(Ai33, 3, 3)
        store_block(Ai10, 1, 0); store_block(Ai20, 2, 0)
        store_block(Ai21, 2, 1); store_block(Ai30, 3, 0)
        store_block(Ai31, 3, 1); store_block(Ai32, 3, 2)

        # Store zeros in upper triangle
        zero_bf16_store = arith.constant(0.0, type=T.bf16)
        for sc_row in range_constexpr(4):
            for sc_col in range_constexpr(4):
                if sc_col > sc_row:
                    for ii in range_constexpr(4):
                        row = (lane // 16) * 4 + ii
                        col = lane % 16
                        t = i_t * BT + sc_row * BC + row
                        bt = sc_col * BC + col
                        a_off = ((bos + t) * H + i_h) * BT + bt
                        buffer_ops.buffer_store(zero_bf16_store, rsrc_A, a_off)

    @flyc.jit
    def launch(
        k_ptr: fx.Tensor, g_ptr: fx.Tensor, beta_ptr: fx.Tensor,
        A_ptr: fx.Tensor, T_val: fx.Int32, B_val: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        NT = T_val // BT
        kkt_solve_kernel(k_ptr, g_ptr, beta_ptr, A_ptr, T_val).launch(
            grid=(NT, B_val * H_SIZE, 1),
            block=(BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    return launch


_kkt_solve_fn = None

def fused_kkt_solve_fwd(k, g_cumsum, beta):
    """Compute fused kkt + triangular solve."""
    global _kkt_solve_fn
    if _kkt_solve_fn is None:
        _kkt_solve_fn = build_kkt_solve()
    B, TT, Hg, K = k.shape
    H = beta.shape[2]
    A = torch.zeros(B, TT, H, BT, device=k.device, dtype=k.dtype)
    _kkt_solve_fn(k, g_cumsum, beta, A, TT, B)
    return A
