# SPDX-License-Identifier: Apache-2.0
"""FlyDSL dV+dK backward kernel with arbitrary mask support.

Supports arbitrary additive masks via bit-packed u32 bitmask, precomputed
loop bounds, ballot-based tile-skip, and strategic sched_barrier(0) placement.
Tuned for AMD MI308X (CDNA3, gfx942).
"""

import math
import os

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, gpu, range_constexpr, rocdl, vector
from flydsl.expr.typing import T
from atrex.src.flydsl.flash_attn.kernels_common import dtype_to_elem_type
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from flydsl._mlir import ir
from flydsl._mlir.dialects import math as math_dialect
from flydsl._mlir.dialects import scf, fly as _fly, llvm as _llvm

KERNEL_NAME = "attn_bwd_dkdv_kernel"
_LLVM_GEP_DYNAMIC = -2147483648  # LLVM kDynamicIndex sentinel

def _llvm_ptr_ty():
    return ir.Type.parse("!llvm.ptr")

def build_dkdv_module(
    num_heads: int,
    head_dim: int,
    dtype_str: str = "bf16",
    sm_scale: float = 0.125,
    block_m: int = 64,
    block_n: int = 64,
    flat_work_group_size: int = 256,
    num_kv_heads=None,
    is_causal: bool = False,
):
    """Build dV+dK backward kernel launcher."""
    gpu_arch = get_hip_arch()

    BLOCK_M = block_m
    BLOCK_N = block_n
    BLOCK_SIZE = flat_work_group_size
    WARP_SIZE = 64
    NUM_WAVES = BLOCK_SIZE // WARP_SIZE
    ROWS_PER_WAVE = BLOCK_M // NUM_WAVES  # 16 for BLOCK_M=64

    NUM_HEADS = num_heads
    HEAD_DIM = head_dim
    NUM_KV_HEADS = num_heads if num_kv_heads is None else int(num_kv_heads)
    Q_STRIDE_TOKEN = NUM_HEADS * HEAD_DIM
    KV_STRIDE_TOKEN = NUM_KV_HEADS * HEAD_DIM

    K_PAD = 4
    K_STRIDE = HEAD_DIM + K_PAD

    # MFMA constants for gfx942 K=8 path
    K_STEP_QK = 8
    K_STEPS_QK = HEAD_DIM // K_STEP_QK  # 8 for D=64
    MFMA_LANE_K = 4  # K=8 path uses 4 lanes per K subblock
    
    # Bit-packed mask strides are passed as dynamic kernel arguments
    # (mask_stride_b, mask_stride_s) to support arbitrary sequence lengths.

    VEC_WIDTH = 16
    assert HEAD_DIM % VEC_WIDTH == 0
    THREADS_PER_ROW_LOAD = HEAD_DIM // VEC_WIDTH  # 4 for D=64
    assert BLOCK_SIZE % THREADS_PER_ROW_LOAD == 0
    ROWS_PER_BATCH_LOAD = BLOCK_SIZE // THREADS_PER_ROW_LOAD

    if ROWS_PER_BATCH_LOAD >= BLOCK_N:
        NUM_BATCHES_KV = 1
        KV_NEEDS_GUARD = ROWS_PER_BATCH_LOAD > BLOCK_N
    else:
        assert BLOCK_N % ROWS_PER_BATCH_LOAD == 0
        NUM_BATCHES_KV = BLOCK_N // ROWS_PER_BATCH_LOAD
        KV_NEEDS_GUARD = False

    LDS_K_TILE_SIZE = BLOCK_N * K_STRIDE  # bf16 elems

    # dO transpose buffer in LDS: dO^T[d_col, q_row] layout
    # dV MFMA uses 32 q-rows (wave0 single tile), so q_dim = 32
    BLOCK_M_DV = 32  # number of q-rows in one dV MFMA tile
    DOT_PAD = 2  # +2 avoids 4-way LDS bank conflict
    DOT_STRIDE = BLOCK_M_DV + DOT_PAD  # 34
    LDS_DOT_TILE_SIZE = HEAD_DIM * DOT_STRIDE  # 64 * 34 = 2176 bf16 elems

    # P transpose buffer in LDS: P^T[k_row, q_col] layout (bf16)
    PT_PAD = 2
    PT_STRIDE = BLOCK_M_DV + PT_PAD  # 34
    LDS_PT_TILE_SIZE = BLOCK_N * PT_STRIDE  # 32 k_rows * 34 = 1088 bf16 elems

    # dO cooperative load constants (same as K load but for BLOCK_M_DV rows)
    # BLOCK_SIZE threads, each loads HEAD_DIM/VEC_WIDTH=4 cols → ROWS_PER_BATCH rows per batch
    # For BLOCK_SIZE=256: ROWS_PER_BATCH=64 > BLOCK_M_DV=32, 1 batch with guard
    # For BLOCK_SIZE=64:  ROWS_PER_BATCH=16 < BLOCK_M_DV=32, 2 batches no guard
    DOT_ROWS_PER_BATCH = ROWS_PER_BATCH_LOAD
    if DOT_ROWS_PER_BATCH >= BLOCK_M_DV:
        DOT_NUM_BATCHES = 1
        DOT_NEEDS_GUARD = DOT_ROWS_PER_BATCH > BLOCK_M_DV
    else:
        assert BLOCK_M_DV % DOT_ROWS_PER_BATCH == 0
        DOT_NUM_BATCHES = BLOCK_M_DV // DOT_ROWS_PER_BATCH
        DOT_NEEDS_GUARD = False

    # V buffer in LDS: same layout as K (BLOCK_N rows × K_STRIDE cols)
    # Separate from K to avoid reloading K every iteration.
    LDS_V_TILE_SIZE = BLOCK_N * K_STRIDE  # same size as K

    # Q^T reuses lds_dot buffer after dV MFMAs consume dO^T data.
    # Same layout: Q^T[d_col, q_row] with stride DOT_STRIDE (34).
    QT_STRIDE = DOT_STRIDE  # 34 (same padding as DOT)

    allocator = SmemAllocator(
        None,
        arch=gpu_arch,
        global_sym_name="attn_bwd_dv_only_smem",
    )
    lds_k_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_k_offset + LDS_K_TILE_SIZE * 2
    # V buffer: separate from K so K stays resident across iterations
    lds_v_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_v_offset + LDS_V_TILE_SIZE * 2
    # dO^T buffer (also reused for Q^T after dV MFMAs consume dO^T)
    lds_dot_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_dot_offset + LDS_DOT_TILE_SIZE * 2
    # P^T transpose buffer
    lds_pt_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_pt_offset + LDS_PT_TILE_SIZE * 2

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def attn_bwd_dkdv_kernel(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        dV: fx.Tensor,
        dK: fx.Tensor,
        dQ: fx.Tensor,
        Mask: fx.Tensor,
        LSE: fx.Tensor,
        Delta: fx.Tensor,
        dO: fx.Tensor,
        LoopBounds: fx.Tensor,
        mask_stride_b: fx.Int32,
        mask_stride_s: fx.Int32,
        seq_len: fx.Int32,
    ):
        elem_type = dtype_to_elem_type(dtype_str)
        compute_type = T.f32
        
        # Fast math flags for arithmetic operations
        fm_fast = arith.FastMathFlags.fast

        q_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), Q)
        k_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), K)
        v_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), V)
        dv_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), dV)
        dk_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), dK)
        dq_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), dQ)
        mask_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), Mask)
        lse_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), LSE)
        delta_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), Delta)
        do_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), dO)
        bounds_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), LoopBounds)
        mask_stride_b_idx = arith.index_cast(T.index, mask_stride_b)
        mask_stride_s_idx = arith.index_cast(T.index, mask_stride_s)

        v4f16_type = T.vec(4, elem_type)
        vxf16_type = T.vec(VEC_WIDTH, elem_type)
        v16f32_type = T.vec(16, compute_type)
        _v4f32_type = T.vec(4, compute_type)  # v4f32 for vectorized mask load
        mfma_pack_type = T.vec(4, elem_type)  # v4bf16 for MFMA input

        seq_len_v = arith.index_cast(T.index, seq_len)

        base_ptr = allocator.get_base()
        lds_k = SmemPtr(
            base_ptr,
            lds_k_offset,
            elem_type,
            shape=(LDS_K_TILE_SIZE,),
        ).get()
        # V buffer: separate from K to avoid K reload every iteration
        lds_v = SmemPtr(
            base_ptr,
            lds_v_offset,
            elem_type,
            shape=(LDS_V_TILE_SIZE,),
        ).get()
        # dO transpose buffer: dO^T[d_col, q_row] layout in LDS
        # Also reused for Q^T after dV MFMAs (same size and stride)
        lds_dot = SmemPtr(
            base_ptr,
            lds_dot_offset,
            elem_type,
            shape=(LDS_DOT_TILE_SIZE,),
        ).get()
        # P transpose buffer: P^T[k_row, q_col] layout in LDS (bf16)
        lds_pt = SmemPtr(
            base_ptr,
            lds_pt_offset,
            elem_type,
            shape=(LDS_PT_TILE_SIZE,),
        ).get()

        block_id = arith.index_cast(T.index, gpu.block_idx.x)
        tid = arith.index_cast(T.index, gpu.thread_idx.x)

        num_kv_tiles = (seq_len_v + BLOCK_N - 1) // BLOCK_N
        head_idx = block_id % NUM_HEADS
        batch_n_tile = block_id // NUM_HEADS
        n_tile_idx = batch_n_tile % num_kv_tiles
        batch_idx = batch_n_tile // num_kv_tiles
        kv_head_idx = head_idx

        n_start = n_tile_idx * BLOCK_N

        load_row_in_batch = tid // THREADS_PER_ROW_LOAD
        load_lane_in_row = tid % THREADS_PER_ROW_LOAD
        load_col_base = load_lane_in_row * VEC_WIDTH

        # Lane decomposition for MFMA -- mirrors forward kernel_best.py L348-379.
        # All values are kept in 'index' type (NOT i32) so subsequent index
        # arithmetic (q_row = q_start + wave_q_offset + lane_mod_32) is well-typed.
        wave_id = tid // WARP_SIZE
        lane = tid % WARP_SIZE
        lane_mod_32 = lane % 32
        lane_div_32 = lane // 32          # 0/1 for 64-wide wave
        wave_q_offset = wave_id * ROWS_PER_WAVE

        # V1a iter8: switch from token-major (B,S,H,D) to head-major (B,H,S,D)
        # to match host-side .contiguous() layout for dV/dK/dQ outputs.
        def kv_global_idx(token_idx, col):
            head_off = (batch_idx * NUM_KV_HEADS + kv_head_idx) * seq_len_v
            return (head_off + token_idx) * HEAD_DIM + col

        def q_global_idx(token_idx, col):
            head_off = (batch_idx * NUM_HEADS + head_idx) * seq_len_v
            return (head_off + token_idx) * HEAD_DIM + col

        def _gep_load(base_ptr, elem_idx, vec_type):
            idx_i64 = arith.index_cast(T.i64, elem_idx)
            gep = _llvm.GEPOp(_llvm_ptr_ty(), base_ptr, [idx_i64],
                              rawConstantIndices=[_LLVM_GEP_DYNAMIC],
                              elem_type=elem_type,
                              noWrapFlags=0)
            return _llvm.LoadOp(vec_type, gep.result).result

        def _gep_store(val, base_ptr, elem_idx):
            idx_i64 = arith.index_cast(T.i64, elem_idx)
            gep = _llvm.GEPOp(_llvm_ptr_ty(), base_ptr, [idx_i64],
                              rawConstantIndices=[_LLVM_GEP_DYNAMIC],
                              elem_type=elem_type,
                              noWrapFlags=0)
            _llvm.StoreOp(val, gep.result)

        def _gep_load_v4f32(base_ptr, elem_idx):
            """Load 4 consecutive f32 from global memory (buffer_load_dwordx4)."""
            idx_i64 = arith.index_cast(T.i64, elem_idx)
            gep = _llvm.GEPOp(_llvm_ptr_ty(), base_ptr, [idx_i64],
                              rawConstantIndices=[_LLVM_GEP_DYNAMIC],
                              elem_type=T.f32,
                              noWrapFlags=0)
            return _llvm.LoadOp(_v4f32_type, gep.result).result

        def _gep_load_u32(base_ptr, elem_idx):
            """Load a single u32 from global memory (buffer_load_dword)."""
            idx_i64 = arith.index_cast(T.i64, elem_idx)
            gep = _llvm.GEPOp(_llvm_ptr_ty(), base_ptr, [idx_i64],
                              rawConstantIndices=[_LLVM_GEP_DYNAMIC],
                              elem_type=T.i32,
                              noWrapFlags=0)
            return _llvm.LoadOp(T.i32, gep.result).result

        def load_global_f16xN(base_ptr, base_idx):
            return _gep_load(base_ptr, base_idx, vxf16_type)


        def load_global_mfma_pack(base_ptr, base_idx):
            return _gep_load(base_ptr, base_idx, mfma_pack_type)

        # Load per-workgroup loop bounds (precomputed on host from mask sparsity)
        # LoopBounds layout: int32[grid_x * 2], [block_id*2]=start, [block_id*2+1]=end
        _bounds_base = block_id * arith.index(2)
        _loop_start_i32 = _gep_load_u32(bounds_ptr, _bounds_base)
        _loop_end_i32 = _gep_load_u32(bounds_ptr, _bounds_base + arith.index(1))
        loop_start_q = arith.index_cast(T.index, _loop_start_i32)
        loop_end_q = arith.index_cast(T.index, _loop_end_i32)

        # MFMA helper: gfx942 K=8 bf16 path. Mirrors forward kernel_best.py L319-327.
        # Real signature: ods_fn(result_type, a, b, c, cbsz, abid, blgp).result
        # cbsz/abid/blgp are IntegerAttr (NOT arith.constant).
        _mfma_zero_attr = ir.IntegerAttr.get(ir.IntegerType.get_signless(32), 0)

        # bf16 v4 packs must be bitcast to i16x4 before feeding ROCDL mfma intrinsic.
        i16x4_type = T.vec(4, T.i16)

        def mfma_acc(a_pack, b_pack, c_acc):
            a_i16 = vector.bitcast(i16x4_type, a_pack)
            b_i16 = vector.bitcast(i16x4_type, b_pack)
            return rocdl.mfma_f32_32x32x8bf16_1k(
                v16f32_type, [a_i16, b_i16, c_acc,
                _mfma_zero_attr, _mfma_zero_attr, _mfma_zero_attr],
            )

        # bf16 trunc-pack helper: mirrors forward kernel_best.py L484-501.
        # Packs 4 f32 values into v4bf16 via bitwise truncation of the upper 16 bits.
        _v2i32_type = T.vec(2, T.i32)
        _c16_i32 = arith.constant(16, type=T.i32)
        _cmask_i32 = arith.constant(0xFFFF0000, type=T.i32)

        def bf16_trunc_pack_v4(f32_vals):
            a0 = arith.ArithValue(f32_vals[0]).bitcast(T.i32)
            b0 = arith.ArithValue(f32_vals[1]).bitcast(T.i32)
            p0 = arith.OrIOp(arith.AndIOp(b0, _cmask_i32).result,
                             arith.ShRUIOp(a0, _c16_i32).result).result
            a1 = arith.ArithValue(f32_vals[2]).bitcast(T.i32)
            b1 = arith.ArithValue(f32_vals[3]).bitcast(T.i32)
            p1 = arith.OrIOp(arith.AndIOp(b1, _cmask_i32).result,
                             arith.ShRUIOp(a1, _c16_i32).result).result
            return vector.bitcast(mfma_pack_type, vector.from_elements(_v2i32_type, [p0, p1]))

        def k_buf_base():
            return arith.index(0)

        def coop_load_k(tile_start):
            k_base = k_buf_base()
            for batch in range_constexpr(NUM_BATCHES_KV):
                row_offset = batch * ROWS_PER_BATCH_LOAD
                row_idx = tile_start + load_row_in_batch + row_offset
                if KV_NEEDS_GUARD:
                    row_valid = arith.cmpi(
                        arith.CmpIPredicate.ult,
                        load_row_in_batch,
                        arith.index(BLOCK_N),
                    )
                    _if_k = scf.IfOp(row_valid)
                    with ir.InsertionPoint(_if_k.then_block):
                        g_idx = kv_global_idx(row_idx, load_col_base)
                        lds_row = load_row_in_batch + row_offset
                        lds_idx = k_base + lds_row * K_STRIDE + load_col_base
                        vec = load_global_f16xN(k_ptr, g_idx)
                        vector.store(vec, lds_k, [lds_idx])
                        scf.YieldOp([])
                else:
                    g_idx = kv_global_idx(row_idx, load_col_base)
                    lds_row = load_row_in_batch + row_offset
                    lds_idx = k_base + lds_row * K_STRIDE + load_col_base
                    vec = load_global_f16xN(k_ptr, g_idx)
                    vector.store(vec, lds_k, [lds_idx])

        def coop_load_v(tile_start):
            """Load V tile to lds_v (separate buffer from K)."""
            v_base = arith.index(0)
            for batch in range_constexpr(NUM_BATCHES_KV):
                row_offset = batch * ROWS_PER_BATCH_LOAD
                row_idx = tile_start + load_row_in_batch + row_offset
                if KV_NEEDS_GUARD:
                    row_valid = arith.cmpi(
                        arith.CmpIPredicate.ult,
                        load_row_in_batch,
                        arith.index(BLOCK_N),
                    )
                    _if_v = scf.IfOp(row_valid)
                    with ir.InsertionPoint(_if_v.then_block):
                        g_idx = kv_global_idx(row_idx, load_col_base)
                        lds_row = load_row_in_batch + row_offset
                        lds_idx = v_base + lds_row * K_STRIDE + load_col_base
                        vec = load_global_f16xN(v_ptr, g_idx)
                        vector.store(vec, lds_v, [lds_idx])
                        scf.YieldOp([])
                else:
                    g_idx = kv_global_idx(row_idx, load_col_base)
                    lds_row = load_row_in_batch + row_offset
                    lds_idx = v_base + lds_row * K_STRIDE + load_col_base
                    vec = load_global_f16xN(v_ptr, g_idx)
                    vector.store(vec, lds_v, [lds_idx])

        # Load both K and V once before the loop (they stay resident in LDS)
        coop_load_k(n_start)
        coop_load_v(n_start)
        # 1-wave (BLOCK_SIZE=64): no barrier needed — same wavefront LDS writes
        # are visible to subsequent reads without explicit synchronization.

        # (K^T pre-transpose removed in iter2: no longer needed without dQ computation)

        # ---- S = Q @ K^T MFMA computation (V1a iter5) ----
        # Mirrors forward kernel_best.py exactly: A=Q, B=K, output S[q_row, k_row].
        # row = q_row = wave_q_offset + lane_mod_32
        # col_formula = lane_div_32*4 + (r//4)*8 + (r%4)  (this is k_row in S)
        # We will transpose at store time to write dV[batch, head, n=k_row, q=q_row].
        #
        # V1c iter1: Support multiple Q tiles (m_iter loop) for S > BLOCK_M_DV (32).
        # Each iteration processes BLOCK_M_DV=32 rows of Q, accumulating dV and dK.
        
        # Compile-time constants for MFMA tiling (needed both inside loop and for post-loop store)
        D_CHUNKS = HEAD_DIM // 32  # 2 for D=64
        PV_K_STEPS = BLOCK_M_DV // 8  # 4 for BLOCK_M_DV=32

        # Zero vector constant for accumulator initialization
        c_zero_v16f32 = arith.constant_vector(0.0, v16f32_type)

        # Initialize accumulators for dV and dK (4 total: dv0, dv1, dk0, dk1)
        init_args = [c_zero_v16f32, c_zero_v16f32, c_zero_v16f32, c_zero_v16f32]

        # Use precomputed loop bounds from mask sparsity analysis
        loop_start = loop_start_q

        for q_start, inner_iter_args, loop_results in scf.for_(
            loop_start,  # lower bound (from precomputed bounds)
            loop_end_q,  # upper bound (from precomputed bounds)
            arith.index(BLOCK_M_DV),  # step = 32
            iter_args=init_args,
        ):
            # Extract dv_accs and dk_accs from iter_args
            dv_accs = [inner_iter_args[0], inner_iter_args[1]]
            dk_accs = [inner_iter_args[2], inner_iter_args[3]]
            # Save original accumulators for else-branch (before IfOp modifies them)
            _orig_dv0 = dv_accs[0]
            _orig_dv1 = dv_accs[1]
            _orig_dk0 = dk_accs[0]
            _orig_dk1 = dk_accs[1]

            # ---- Tile-skip: ballot-based early exit for zero mask tiles ----
            # Load mask bits (cheap: 1 global load per lane), then wavefront ballot.
            # If ALL lanes have zero mask → this q-tile contributes nothing, skip.
            q_row = q_start + lane_mod_32
            q_in_bounds = arith.cmpi(arith.CmpIPredicate.slt, q_row, seq_len_v)
            q_row_safe = arith.select(q_in_bounds, q_row, arith.index(0))

            _mask_row_base = (batch_idx * mask_stride_b_idx
                              + q_row_safe * mask_stride_s_idx)
            _kv_word_idx = n_start // arith.index(32)
            _mask_bits = _gep_load_u32(mask_ptr, _mask_row_base + _kv_word_idx)

            _c_zero_i32_ballot = arith.constant(0, type=T.i32)
            _mask_nonzero = arith.cmpi(
                arith.CmpIPredicate.ne, _mask_bits, _c_zero_i32_ballot)
            _ballot_result = rocdl.ballot(T.i64, _mask_nonzero)
            _c_zero_i64 = arith.constant(0, type=T.i64)
            _any_active = arith.cmpi(
                arith.CmpIPredicate.ne, _ballot_result, _c_zero_i64)

            _if_tile_active = scf.IfOp(
                _any_active,
                results_=[v16f32_type, v16f32_type, v16f32_type, v16f32_type],
                has_else=True)
            with ir.InsertionPoint(_if_tile_active.then_block):
                c_zero_mfma_pack = arith.constant_vector(0.0, mfma_pack_type)

                q_a_packs = []
                for ks in range_constexpr(K_STEPS_QK):
                    q_col = arith.index(ks * K_STEP_QK) + lane_div_32 * MFMA_LANE_K
                    g_idx = q_global_idx(q_row_safe, q_col)
                    raw = load_global_mfma_pack(q_ptr, g_idx)
                    q_a_packs.append(arith.select(q_in_bounds, raw, c_zero_mfma_pack))

                # K B-operand from LDS (j = lane_mod_32 selects k_row, same address formula).
                s_acc = c_zero_v16f32
                k_base = k_buf_base()

                rocdl.sched_barrier(0)

                # [Pipelined] Issue dO coop load BEFORE GEMM1.
                # VMEM/DS operations overlap with GEMM1's 512-cycle MFMA chain.
                # dO→lds_dot is separate from lds_k, no conflict.
                # dO B-operand for dP GEMM is read from lds_dot (strided scalar reads)
                # instead of separate global loads — saves 8 VMEM per iteration.

                # Cooperative load dO → lds_dot (transposed)
                _v1_type = T.vec(1, elem_type)
                for dot_batch in range_constexpr(DOT_NUM_BATCHES):
                    row_offset = dot_batch * DOT_ROWS_PER_BATCH
                    dot_row_local = load_row_in_batch + row_offset
                    do_q_row = q_start + dot_row_local

                    if DOT_NEEDS_GUARD:
                        do_row_valid = arith.cmpi(arith.CmpIPredicate.ult, load_row_in_batch, arith.index(BLOCK_M_DV))
                        _if_do_load = scf.IfOp(do_row_valid)
                        with ir.InsertionPoint(_if_do_load.then_block):
                            do_g_idx = q_global_idx(do_q_row, load_col_base)
                            do_vec = load_global_f16xN(do_ptr, do_g_idx)
                            for _e in range_constexpr(VEC_WIDTH):
                                elem = vector.extract(do_vec, static_position=[_e], dynamic_position=[])
                                dot_d_col = load_col_base + arith.index(_e)
                                dot_idx = dot_d_col * arith.index(DOT_STRIDE) + dot_row_local
                                v1_elem = vector.from_elements(_v1_type, [elem])
                                vector.store(v1_elem, lds_dot, [dot_idx])
                            scf.YieldOp([])
                    else:
                        do_g_idx = q_global_idx(do_q_row, load_col_base)
                        do_vec = load_global_f16xN(do_ptr, do_g_idx)
                        for _e in range_constexpr(VEC_WIDTH):
                            elem = vector.extract(do_vec, static_position=[_e], dynamic_position=[])
                            dot_d_col = load_col_base + arith.index(_e)
                            dot_idx = dot_d_col * arith.index(DOT_STRIDE) + dot_row_local
                            v1_elem = vector.from_elements(_v1_type, [elem])
                            vector.store(v1_elem, lds_dot, [dot_idx])

                # GEMM1: K @ Q^T (VMEM/DS loads above overlap with these MFMAs)
                for ks in range_constexpr(K_STEPS_QK):
                    col = arith.index(ks * K_STEP_QK) + lane_div_32 * MFMA_LANE_K
                    k_idx = k_base + lane_mod_32 * K_STRIDE + col
                    k_pack = vector.load_op(mfma_pack_type, lds_k, [k_idx])
                    s_acc = mfma_acc(k_pack, q_a_packs[ks], s_acc)

                # ---- V1b iter1: Apply scale and add mask ----
                # s_acc currently holds S = K @ Q^T (no scale). Now:
                #   s_acc = s_acc * scale + mask[q_row, k_row]
                #
                # V1c iter10 CRITICAL FIX: MFMA C output lane_mod_32 = B's lane_mod_32
                # For mfma_acc(K_a, Q_b, s_acc): B=Q, so C lane_mod_32 = Q's lane_mod_32 = q_row_in_tile
                # j_formula corresponds to A(K)'s contribution direction = k_col_in_tile
                # Therefore: s_acc[lane, r] = S[q=lane_mod_32, k=j_formula(lane_div_32, r)]
                #
                # This matches forward kernel_best.py where mask uses:
                #   q_row = q_start + lane_mod_32 (row dimension of mask)
                #   kv_col = lane_div_32*4 + grp*8 + sub (column = j_formula pattern)
                # iter3: fused softmax constants (same optimization as dQ V8b).
                # Original: p = exp2((s * scale + mask - lse) * log2e) = 5 VALU/r
                # Fused:    p = exp2(fma(s, scale_log2e, neg_lse_log2e) + mask_log2e) = 3 VALU/r
                _LOG2E = 1.4426950408889634
                c_scale_log2e = arith.constant(sm_scale * _LOG2E, type=compute_type)
                c_scale = arith.constant(sm_scale, type=compute_type)  # still needed for dS
                c_neg_inf_log2e = arith.constant(-1000000.0 * _LOG2E, type=compute_type)
                c_zero_f32 = arith.constant(0.0, type=compute_type)

                q_row_lse = q_start + lane_mod_32
                lse_idx_base = (batch_idx * arith.index(NUM_HEADS) + head_idx) * seq_len_v + q_row_lse
                lse_i64_base = arith.index_cast(T.i64, lse_idx_base)
                lse_gep_base = _llvm.GEPOp(_llvm_ptr_ty(), lse_ptr, [lse_i64_base],
                                      rawConstantIndices=[_LLVM_GEP_DYNAMIC],
                                      elem_type=T.f32,
                                      noWrapFlags=0)
                lse_raw = _llvm.LoadOp(T.f32, lse_gep_base.result).result
                # Pre-compute neg_lse_log2e = -(lse * log2e)
                c_log2e_pre = arith.constant(_LOG2E, type=compute_type)
                lse_log2e = arith.MulFOp(lse_raw, c_log2e_pre, fastmath=fm_fast).result
                neg_lse_log2e = arith.SubFOp(c_zero_f32, lse_log2e, fastmath=fm_fast).result

                # Bit-packed mask: extract per-position attend bits from pre-loaded u32
                c_zero_i32 = arith.constant(0, type=T.i32)
                c_mask_penalty_log2e = arith.constant(-1.0e6 * _LOG2E, type=compute_type)
                _lane_bit_base = arith.index_cast(T.i32, lane_div_32 * arith.index(4))
                _mask_ps = arith.ShRUIOp(_mask_bits, _lane_bit_base).result
                mask_log2e_vals = [None] * 16
                for grp in range_constexpr(4):
                    for sub in range_constexpr(4):
                        r = grp * 4 + sub
                        _bit_mask = arith.constant(1 << (grp * 8 + sub), type=T.i32)
                        is_attend = arith.cmpi(
                            arith.CmpIPredicate.ne,
                            arith.AndIOp(_mask_ps, _bit_mask).result,
                            c_zero_i32)
                        mask_log2e_vals[r] = arith.select(
                            is_attend, c_zero_f32, c_mask_penalty_log2e)

                # Fused softmax: 3 VALU/r instead of 5
                for r in range_constexpr(16):
                    f32_val = vector.extract(s_acc, static_position=[r], dynamic_position=[])
                    # fma(s, scale_log2e, neg_lse_log2e) = s * scale * log2e - lse * log2e
                    s_fma = math_dialect.fma(f32_val, c_scale_log2e, neg_lse_log2e)
                    # Add mask (0 or -inf*log2e)
                    s_with_mask = arith.AddFOp(s_fma, mask_log2e_vals[r], fastmath=fm_fast).result
                    p_val = rocdl.exp2(T.f32, s_with_mask)
                    s_acc = vector.insert(p_val, s_acc, static_position=[r], dynamic_position=[])

                # ---- V1c iter9: Compute dV = dO^T @ P via DUAL LDS transpose ----
                # s_acc now holds P values from Q@K^T → scale → mask → exp2.
                # Q@K^T MFMA layout: row = lane_mod_32 = q_row, col = lane_div_32*4 + (r//4)*8 + (r%4) = k_col
                # So s_acc[lane, r] = P[q=lane_mod_32, k=col_formula(lane_div_32, r)]
                #
                # dV[k_row, d_col] = sum_q P[q, k_row] * dO[q, d_col] = dO^T[d, q] @ P[q, k]
                # dV MFMA: A=dO^T, B=P → C[i=d_col, j=k_row] += A[i, q] * B[q, j]
                #   A operand: lane_mod_32 = i = d_col → solved by dO LDS transpose
                #   B operand: lane_mod_32 = j = k_row → BUT s_acc has lane_mod_32 = q_row!
                #
                # BUG FIX (iter9): P must also go through LDS transpose.
                # Write P[q=lane_mod_32, k=col_formula] to LDS as P[q_row, k_col],
                # then read back as P^T[k_row, q_col] for B operand (lane_mod_32 = k_row).
                #
                # LDS P^T layout: lds_pt[k_row * PT_STRIDE + q_col], stride=34
                # B operand read: lane_mod_32 selects k_row, pack 4 consecutive q values
                #
                # D=64, split into 2 d_chunks of 32 columns each (D_CHUNKS = HEAD_DIM // 32 = 2)
                # BLOCK_M_DV=32 q-rows in this tile, PV_K_STEPS = 32 // 8 = 4
                # (D_CHUNKS and PV_K_STEPS defined before loop for visibility in post-loop store)

                # ---- Pre-issue Q^T global loads (overlap with P writes + dV MFMAs) ----
                qt_loaded_vecs = []
                for dot_batch in range_constexpr(DOT_NUM_BATCHES):
                    row_offset = dot_batch * DOT_ROWS_PER_BATCH
                    dot_row_local_qt = load_row_in_batch + row_offset
                    q_load_row = q_start + dot_row_local_qt
                    if DOT_NEEDS_GUARD:
                        q_load_valid = arith.cmpi(arith.CmpIPredicate.ult, load_row_in_batch, arith.index(BLOCK_M_DV))
                        q_g_idx = q_global_idx(q_load_row, load_col_base)
                        raw_q = load_global_f16xN(q_ptr, q_g_idx)
                        zero_vec = arith.constant_vector(0.0, vxf16_type)
                        qt_loaded_vecs.append(arith.select(q_load_valid, raw_q, zero_vec))
                    else:
                        q_g_idx = q_global_idx(q_load_row, load_col_base)
                        qt_loaded_vecs.append(load_global_f16xN(q_ptr, q_g_idx))

                # ---- Step 1: Write P to LDS (transpose) ----
                _v1_type_pt = T.vec(1, elem_type)
                q_row_pt = lane_mod_32
                for r in range_constexpr(16):
                    k_col_pt = lane_div_32 * arith.index(4) + arith.index((r // 4) * 8 + r % 4)
                    p_f32 = vector.extract(s_acc, static_position=[r], dynamic_position=[])
                    p_i32 = arith.ArithValue(p_f32).bitcast(T.i32)
                    p_top16 = arith.ShRUIOp(p_i32, arith.constant(16, type=T.i32)).result
                    p_i16 = arith.trunci(T.i16, p_top16)
                    p_bf16 = arith.ArithValue(p_i16).bitcast(elem_type)
                    pt_idx = k_col_pt * arith.index(PT_STRIDE) + q_row_pt
                    v1_p = vector.from_elements(_v1_type_pt, [p_bf16])
                    vector.store(v1_p, lds_pt, [pt_idx])

                def read_P_from_lds(pks):
                    k_row_pt = lane_mod_32
                    q_base_pt = arith.index(pks * 8) + lane_div_32 * arith.index(4)
                    pt_read_idx = k_row_pt * arith.index(PT_STRIDE) + q_base_pt
                    return vector.load_op(mfma_pack_type, lds_pt, [pt_read_idx])

                p_packs = []
                for pks in range_constexpr(PV_K_STEPS):
                    p_packs.append(read_P_from_lds(pks))

                def read_dO_from_lds(pks, d_chunk):
                    d_pos = arith.index(d_chunk * 32) + lane_mod_32
                    q_base = arith.index(pks * 8) + lane_div_32 * arith.index(4)
                    lds_idx = d_pos * arith.index(DOT_STRIDE) + q_base
                    return vector.load_op(mfma_pack_type, lds_dot, [lds_idx])

                # ---- dV = dO^T @ P MFMAs ----
                for pks in range_constexpr(PV_K_STEPS):
                    for d_chunk in range_constexpr(D_CHUNKS):
                        do_a = read_dO_from_lds(pks, d_chunk)
                        dv_accs[d_chunk] = mfma_acc(do_a, p_packs[pks], dv_accs[d_chunk])

                # ==================================================================
                # ---- dK COMPUTATION: dP → Di → dS → dS^T LDS → Q^T LDS → dK MFMA ----
                # ==================================================================

                # ---- Step B: dP GEMM = V @ dO^T ----
                # dO B-operand read from lds_dot (strided scalar reads) instead of
                # global memory — lds_dot still holds dO^T data at this point.
                # V A-operand from lds_v (loaded once before the loop).
                _v1_type_dob = T.vec(1, elem_type)
                def read_dO_B_from_lds(ks):
                    """Read dO^T as B operand from lds_dot using strided scalar reads.
                    B operand: lane_mod_32 = q_col, 4 packed d-elements at stride DOT_STRIDE."""
                    q_col_do = lane_mod_32
                    d_base_do = arith.index(ks * K_STEP_QK) + lane_div_32 * arith.index(MFMA_LANE_K)
                    do_elems = []
                    for _s in range_constexpr(4):
                        d_pos_do = d_base_do + arith.index(_s)
                        lds_idx_do = d_pos_do * arith.index(DOT_STRIDE) + q_col_do
                        v1_do = vector.load_op(_v1_type_dob, lds_dot, [lds_idx_do])
                        do_elems.append(vector.extract(v1_do, static_position=[0], dynamic_position=[]))
                    return vector.from_elements(mfma_pack_type, do_elems)

                dp_acc = c_zero_v16f32
                v_base = arith.index(0)

                for ks in range_constexpr(K_STEPS_QK):
                    col_v = arith.index(ks * K_STEP_QK) + lane_div_32 * MFMA_LANE_K
                    v_idx = v_base + lane_mod_32 * K_STRIDE + col_v
                    v_pack = vector.load_op(mfma_pack_type, lds_v, [v_idx])
                    do_b = read_dO_B_from_lds(ks)
                    dp_acc = mfma_acc(v_pack, do_b, dp_acc)

                # ---- Write pre-loaded Q^T to lds_dot (dO^T fully consumed by dP above) ----
                for dot_batch in range_constexpr(DOT_NUM_BATCHES):
                    row_offset = dot_batch * DOT_ROWS_PER_BATCH
                    dot_row_local_qt = load_row_in_batch + row_offset
                    q_vec = qt_loaded_vecs[dot_batch]
                    for _e_q in range_constexpr(VEC_WIDTH):
                        elem_q = vector.extract(q_vec, static_position=[_e_q], dynamic_position=[])
                        qt_d_col = load_col_base + arith.index(_e_q)
                        qt_idx = qt_d_col * arith.index(QT_STRIDE) + dot_row_local_qt
                        v1_eq = vector.from_elements(_v1_type, [elem_q])
                        vector.store(v1_eq, lds_dot, [qt_idx])

                # dp_acc now holds dP[q=lane_mod_32, k=j_formula] — same layout as s_acc(P)!

                # ---- Step C: Load precomputed Delta[q] = row_sum(dO * O) ----
                # Delta is precomputed externally as Di[q] = sum_d dO[q,d] * O[q,d].
                # This is equivalent to sum_k P[q,k] * dP[q,k] but computed over ALL k,
                # not just the current kv tile's 32 columns. Using precomputed Delta
                # ensures correctness when S > BLOCK_N (multiple kv tiles).
                #
                # Delta layout: [B, H, S] with stride (H*S, S, 1), dtype=f32
                # Index: batch_idx * (NUM_HEADS * seq_len) + head_idx * seq_len + q_row
                # q_row for this lane = q_start + lane_mod_32
                delta_q_row = q_start + lane_mod_32
                delta_idx = (batch_idx * arith.index(NUM_HEADS) + head_idx) * seq_len_v + delta_q_row
                delta_idx_i64 = arith.index_cast(T.i64, delta_idx)
                delta_gep = _llvm.GEPOp(_llvm_ptr_ty(), delta_ptr, [delta_idx_i64],
                                        rawConstantIndices=[_LLVM_GEP_DYNAMIC],
                                        elem_type=T.f32,
                                        noWrapFlags=0)
                _v1f32_type = T.vec(1, T.f32)
                di_vec = _llvm.LoadOp(_v1f32_type, delta_gep.result).result
                di_full = vector.extract(di_vec, static_position=[0], dynamic_position=[])

                # ---- Step D: dS_unscaled = P * (dP - Di) element-wise ----
                # iter3: scale factor deferred to post-loop dK store (same as dQ V9).
                # dK = Q^T @ dS = Q^T @ (scale * P * (dP - Di)) = scale * (Q^T @ (P * (dP - Di)))
                # By computing dS without scale here, we save 16 MulF per iteration.
                # The final dK accumulator is multiplied by scale once in post-loop store.
                ds_acc = c_zero_v16f32
                for r in range_constexpr(16):
                    p_val_ds = vector.extract(s_acc, static_position=[r], dynamic_position=[])
                    dp_val_ds = vector.extract(dp_acc, static_position=[r], dynamic_position=[])
                    dp_minus_di = arith.SubFOp(dp_val_ds, di_full, fastmath=fm_fast).result
                    ds_val = arith.MulFOp(p_val_ds, dp_minus_di, fastmath=fm_fast).result
                    ds_acc = vector.insert(ds_val, ds_acc, static_position=[r], dynamic_position=[])

                # ---- Step E: dS LDS transpose (reuse PT buffer) ----
                q_row_ds = lane_mod_32
                for r in range_constexpr(16):
                    k_col_ds = lane_div_32 * arith.index(4) + arith.index((r // 4) * 8 + r % 4)
                    ds_f32 = vector.extract(ds_acc, static_position=[r], dynamic_position=[])
                    ds_i32 = arith.ArithValue(ds_f32).bitcast(T.i32)
                    ds_top16 = arith.ShRUIOp(ds_i32, arith.constant(16, type=T.i32)).result
                    ds_i16 = arith.trunci(T.i16, ds_top16)
                    ds_bf16 = arith.ArithValue(ds_i16).bitcast(elem_type)
                    ds_pt_idx = k_col_ds * arith.index(PT_STRIDE) + q_row_ds
                    v1_ds = vector.from_elements(_v1_type_pt, [ds_bf16])
                    vector.store(v1_ds, lds_pt, [ds_pt_idx])

                def read_dST_from_lds(pks):
                    k_row_dst = lane_mod_32
                    q_base_dst = arith.index(pks * 8) + lane_div_32 * arith.index(4)
                    dst_read_idx = k_row_dst * arith.index(PT_STRIDE) + q_base_dst
                    return vector.load_op(mfma_pack_type, lds_pt, [dst_read_idx])

                ds_t_packs = []
                for pks in range_constexpr(PV_K_STEPS):
                    ds_t_packs.append(read_dST_from_lds(pks))

                # ---- Step F: Read Q^T from lds_dot ----
                def read_QT_from_lds(pks, d_chunk):
                    d_pos_q = arith.index(d_chunk * 32) + lane_mod_32
                    q_base_q = arith.index(pks * 8) + lane_div_32 * arith.index(4)
                    lds_idx_q = d_pos_q * arith.index(QT_STRIDE) + q_base_q
                    return vector.load_op(mfma_pack_type, lds_dot, [lds_idx_q])

                # ---- Step G: dK MFMA = Q^T @ dS^T ----
                rocdl.sched_barrier(0)
                for pks in range_constexpr(PV_K_STEPS):
                    for d_chunk in range_constexpr(D_CHUNKS):
                        qt_a = read_QT_from_lds(pks, d_chunk)
                        dk_accs[d_chunk] = mfma_acc(qt_a, ds_t_packs[pks], dk_accs[d_chunk])

                # ---- Yield updated accumulators for next iteration ----
                scf.YieldOp([dv_accs[0], dv_accs[1], dk_accs[0], dk_accs[1]])
            with ir.InsertionPoint(_if_tile_active.else_block):
                scf.YieldOp([_orig_dv0, _orig_dv1, _orig_dk0, _orig_dk1])

            # Yield IfOp results as new accumulators
            yield [_if_tile_active.results[0], _if_tile_active.results[1],
                   _if_tile_active.results[2], _if_tile_active.results[3]]

        # ==================================================================
        # ---- POST-LOOP: Store dV and dK from accumulated results ----
        # ==================================================================
        # Extract final accumulators from loop results
        final_dv_accs = [loop_results[0], loop_results[1]]
        final_dk_accs = [loop_results[2], loop_results[3]]

        # ---- Store dV: C layout lane_mod_32=k_row, j_formula=d_col ----
        # iter3: vectorized v4bf16 store (same as dQ V6).
        # MFMA C output d_col formula: lane_div_32*4 + (r//4)*8 + (r%4)
        # For each group of r=[grp*4..grp*4+3], d_col values are consecutive:
        #   lane_div_32*4 + grp*8 + {0,1,2,3}
        # Pack 4 bf16 values into v4bf16 and store once per group.
        _v4_store_type = T.vec(4, elem_type)
        cond_w0_store_dv = arith.cmpi(arith.CmpIPredicate.eq, wave_id, arith.index(0))
        _if_store_dv = scf.IfOp(cond_w0_store_dv)
        with ir.InsertionPoint(_if_store_dv.then_block):
            k_row_dv = n_start + lane_mod_32  # k_row in global coords
            cond_k_dv = arith.cmpi(arith.CmpIPredicate.ult, k_row_dv, seq_len_v)
            _if_k_valid_dv = scf.IfOp(cond_k_dv)
            with ir.InsertionPoint(_if_k_valid_dv.then_block):
                for d_chunk in range_constexpr(D_CHUNKS):
                    for grp in range_constexpr(4):
                        dv_bf16_elems = []
                        for sub in range_constexpr(4):
                            r_idx = grp * 4 + sub
                            dv_f32 = vector.extract(final_dv_accs[d_chunk], static_position=[r_idx], dynamic_position=[])
                            dv_i32 = arith.ArithValue(dv_f32).bitcast(T.i32)
                            dv_top16 = arith.ShRUIOp(dv_i32, arith.constant(16, type=T.i32)).result
                            dv_i16 = arith.trunci(T.i16, dv_top16)
                            dv_bf16_elems.append(arith.ArithValue(dv_i16).bitcast(elem_type))
                        v4_dv = vector.from_elements(_v4_store_type, dv_bf16_elems)
                        d_col_base = arith.index(d_chunk * 32) + lane_div_32 * arith.index(4) + arith.index(grp * 8)
                        g_idx_dv = kv_global_idx(k_row_dv, d_col_base)
                        idx_i64_dv = arith.index_cast(T.i64, g_idx_dv)
                        gep_dv = _llvm.GEPOp(_llvm_ptr_ty(), dv_ptr, [idx_i64_dv],
                                             rawConstantIndices=[_LLVM_GEP_DYNAMIC],
                                             elem_type=elem_type,
                                             noWrapFlags=0)
                        _llvm.StoreOp(v4_dv, gep_dv.result)
                scf.YieldOp([])
            scf.YieldOp([])

        # ---- Store dK: same C layout as dV, vectorized v4bf16 store ----
        # iter3: apply deferred scale factor before bf16 truncation.
        # dK_acc = Q^T @ (P * (dP - Di)), needs * scale for final dK.
        c_scale_dk = arith.constant(sm_scale, type=compute_type)
        cond_w0_store_dk = arith.cmpi(arith.CmpIPredicate.eq, wave_id, arith.index(0))
        _if_store_dk = scf.IfOp(cond_w0_store_dk)
        with ir.InsertionPoint(_if_store_dk.then_block):
            k_row_dk = n_start + lane_mod_32
            cond_k_dk = arith.cmpi(arith.CmpIPredicate.ult, k_row_dk, seq_len_v)
            _if_k_valid_dk = scf.IfOp(cond_k_dk)
            with ir.InsertionPoint(_if_k_valid_dk.then_block):
                for d_chunk in range_constexpr(D_CHUNKS):
                    for grp in range_constexpr(4):
                        dk_bf16_elems = []
                        for sub in range_constexpr(4):
                            r_idx = grp * 4 + sub
                            dk_f32_raw = vector.extract(final_dk_accs[d_chunk], static_position=[r_idx], dynamic_position=[])
                            dk_f32 = arith.MulFOp(dk_f32_raw, c_scale_dk, fastmath=fm_fast).result
                            dk_i32 = arith.ArithValue(dk_f32).bitcast(T.i32)
                            dk_top16 = arith.ShRUIOp(dk_i32, arith.constant(16, type=T.i32)).result
                            dk_i16 = arith.trunci(T.i16, dk_top16)
                            dk_bf16_elems.append(arith.ArithValue(dk_i16).bitcast(elem_type))
                        v4_dk = vector.from_elements(_v4_store_type, dk_bf16_elems)
                        d_col_base = arith.index(d_chunk * 32) + lane_div_32 * arith.index(4) + arith.index(grp * 8)
                        g_idx_dk = kv_global_idx(k_row_dk, d_col_base)
                        idx_i64_dk = arith.index_cast(T.i64, g_idx_dk)
                        gep_dk = _llvm.GEPOp(_llvm_ptr_ty(), dk_ptr, [idx_i64_dk],
                                             rawConstantIndices=[_LLVM_GEP_DYNAMIC],
                                             elem_type=elem_type,
                                             noWrapFlags=0)
                        _llvm.StoreOp(v4_dk, gep_dk.result)
                scf.YieldOp([])
            scf.YieldOp([])

    @flyc.jit
    def launch_attn_bwd_dkdv(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        dV: fx.Tensor,
        dK: fx.Tensor,
        dQ: fx.Tensor,
        Mask: fx.Tensor,
        LSE: fx.Tensor,
        Delta: fx.Tensor,
        dO: fx.Tensor,
        LoopBounds: fx.Tensor,
        mask_stride_b: fx.Int32,
        mask_stride_s: fx.Int32,
        batch_size: fx.Int32,
        seq_len: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        bs_idx = arith.index_cast(T.index, batch_size)
        sl_idx = arith.index_cast(T.index, seq_len)
        num_kv_tiles = (sl_idx + BLOCK_N - 1) // BLOCK_N
        grid_x = bs_idx * num_kv_tiles * NUM_HEADS

        launcher = attn_bwd_dkdv_kernel(Q, K, V, dV, dK, dQ, Mask, LSE, Delta, dO, LoopBounds, mask_stride_b, mask_stride_s, seq_len)

        flat_wg_attr = ir.StringAttr.get(f"{BLOCK_SIZE},{BLOCK_SIZE}")
        for op in ctx.gpu_module_body.operations:
            if getattr(op, "OPERATION_NAME", None) == "gpu.func":
                op.attributes["rocdl.flat_work_group_size"] = flat_wg_attr

        launcher.launch(
            grid=(grid_x, 1, 1),
            block=(BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    return launch_attn_bwd_dkdv
