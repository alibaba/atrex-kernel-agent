# SPDX-License-Identifier: Apache-2.0
"""FlyDSL dQ backward kernel with arbitrary mask support.

Supports arbitrary additive masks via bit-packed u32 bitmask, precomputed
loop bounds, and strategic sched_barrier(0) placement.
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

KERNEL_NAME = "attn_bwd_dq_kernel"
_LLVM_GEP_DYNAMIC = -2147483648  # LLVM kDynamicIndex sentinel

def _llvm_ptr_ty():
    return ir.Type.parse("!llvm.ptr")

def build_dq_module(
    num_heads: int,
    head_dim: int,
    dtype_str: str = "bf16",
    sm_scale: float = 0.125,
    block_m: int = 32,
    block_n: int = 32,
    flat_work_group_size: int = 64,
    num_kv_heads=None,
    is_causal: bool = False,
):
    """Build dQ backward kernel launcher."""
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
    
    # Bit-packed mask stride constants (u32 bitmask shape: B, 1, S_pad, S_pad//32)
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

    # dQ kernel tile size constants
    BLOCK_M_DV = 32  # number of q-rows in one MFMA32x32x8 tile

    # P/dS repack buffer in LDS: dS[q_row, k_col] layout (bf16)
    # s_acc has lane_mod_32=q_row, col=k_col. dV MFMA B operand needs lane_mod_32=k_row.
    # Write P[q, k] to LDS, read back as P^T[k, q] for correct B operand lane mapping.
    # P tile is 32×32 (q_rows × k_cols from wave0), stored with padding.
    PT_PAD = 2  # avoid bank conflict
    PT_STRIDE = BLOCK_M_DV + PT_PAD  # 34 (q dim stride, reading 4 consecutive q values)
    LDS_PT_TILE_SIZE = BLOCK_N * PT_STRIDE  # 32 k_rows * 34 = 1088 bf16 elems
    # Note: only 32 k_rows because dV output is 32-wide tile (one MFMA32x32x8)

    # V buffer in LDS: same layout as K (BLOCK_N rows × K_STRIDE cols)
    # Separate from K to avoid reloading K every iteration.
    LDS_V_TILE_SIZE = BLOCK_N * K_STRIDE  # same size as K

    # K^T reads: derived from lds_k with strided scalar loads (no separate buffer).
    # Eliminates 4352 bytes LDS → occupancy 5→7.

    allocator = SmemAllocator(
        None,
        arch=gpu_arch,
        global_sym_name="attn_bwd_dq_smem",
    )
    lds_k_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_k_offset + LDS_K_TILE_SIZE * 2
    # V buffer: separate from K so K stays resident across iterations
    lds_v_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_v_offset + LDS_V_TILE_SIZE * 2

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def attn_bwd_dq_kernel(
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

        block_id = arith.index_cast(T.index, gpu.block_idx.x)
        tid = arith.index_cast(T.index, gpu.thread_idx.x)

        num_q_tiles = (seq_len_v + BLOCK_M - 1) // BLOCK_M
        head_idx = block_id % NUM_HEADS
        batch_q_tile = block_id // NUM_HEADS
        q_tile_idx = batch_q_tile % num_q_tiles
        batch_idx = batch_q_tile // num_q_tiles
        kv_head_idx = head_idx
        q_start_fixed = q_tile_idx * BLOCK_M  # fixed Q-tile start for this workgroup

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
        _bounds_base = block_id * arith.index(2)
        _loop_start_i32 = _gep_load_u32(bounds_ptr, _bounds_base)
        _loop_end_i32 = _gep_load_u32(bounds_ptr, _bounds_base + arith.index(1))
        loop_start_kv = arith.index_cast(T.index, _loop_start_i32)
        loop_end_kv = arith.index_cast(T.index, _loop_end_i32)

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

        # Initialize accumulators for dQ (2 total: dq0, dq1)
        init_args = [c_zero_v16f32, c_zero_v16f32]

        # ================================================================
        # V7: Hoist loop-invariant Q, dO, LSE, Delta loads BEFORE loop.
        # Q and dO packs are fixed for this Q-tile (q_start_fixed) and
        # do not change across KV-tile iterations. Loading them once
        # before the loop eliminates 16 global loads per iteration.
        # LSE and Delta are also per-q-row scalars, fixed per lane.
        # ================================================================
        q_row = q_start_fixed + lane_mod_32
        q_in_bounds = arith.cmpi(arith.CmpIPredicate.slt, q_row, seq_len_v)
        q_row_safe = arith.select(q_in_bounds, q_row, arith.index(0))
        c_zero_mfma_pack = arith.constant_vector(0.0, mfma_pack_type)

        # Pre-load Q packs (8 v4bf16) — loop invariant
        q_a_packs = []
        for ks in range_constexpr(K_STEPS_QK):
            q_col = arith.index(ks * K_STEP_QK) + lane_div_32 * MFMA_LANE_K
            g_idx = q_global_idx(q_row_safe, q_col)
            raw = load_global_mfma_pack(q_ptr, g_idx)
            q_a_packs.append(arith.select(q_in_bounds, raw, c_zero_mfma_pack))

        # Pre-load dO packs (8 v4bf16) — loop invariant
        do_row_dp = q_start_fixed + lane_mod_32
        do_in_bounds = arith.cmpi(arith.CmpIPredicate.slt, do_row_dp, seq_len_v)
        do_row_safe = arith.select(do_in_bounds, do_row_dp, arith.index(0))

        do_b_packs = []
        for ks in range_constexpr(K_STEPS_QK):
            do_col = arith.index(ks * K_STEP_QK) + lane_div_32 * MFMA_LANE_K
            g_idx_do = q_global_idx(do_row_safe, do_col)
            raw_do = load_global_mfma_pack(do_ptr, g_idx_do)
            do_b_packs.append(arith.select(do_in_bounds, raw_do, c_zero_mfma_pack))

        # V8b: Pre-fused constants for softmax.
        # Original: p = exp2((s * scale + mask - lse) * log2e)
        # Fused:    p = exp2(fma(s, scale_log2e, neg_lse_log2e) + mask_log2e)
        # This saves 2 VALU/r (from 5 to 3) × 16r = 32 VALU per iteration.
        _LOG2E = 1.4426950408889634
        c_scale_log2e = arith.constant(sm_scale * _LOG2E, type=compute_type)
        c_scale = arith.constant(sm_scale, type=compute_type)  # still needed for dS
        c_neg_inf_log2e = arith.constant(-1000000.0 * _LOG2E, type=compute_type)
        c_zero_f32 = arith.constant(0.0, type=compute_type)

        q_row_lse = q_start_fixed + lane_mod_32
        lse_idx_base = (batch_idx * arith.index(NUM_HEADS) + head_idx) * seq_len_v + q_row_lse
        lse_i64_base = arith.index_cast(T.i64, lse_idx_base)
        lse_gep_base = _llvm.GEPOp(_llvm_ptr_ty(), lse_ptr, [lse_i64_base],
                              rawConstantIndices=[_LLVM_GEP_DYNAMIC],
                              elem_type=T.f32,
                              noWrapFlags=0)
        lse_raw = _llvm.LoadOp(T.f32, lse_gep_base.result).result
        # V8b: pre-compute neg_lse_log2e = -(lse * log2e) for fused softmax
        c_log2e_pre = arith.constant(_LOG2E, type=compute_type)
        lse_log2e = arith.MulFOp(lse_raw, c_log2e_pre, fastmath=fm_fast).result
        neg_lse_log2e = arith.SubFOp(c_zero_f32, lse_log2e, fastmath=fm_fast).result

        # Pre-load Delta scalar — loop invariant (per-q-row)
        delta_q_row = q_start_fixed + lane_mod_32
        delta_idx = (batch_idx * arith.index(NUM_HEADS) + head_idx) * seq_len_v + delta_q_row
        delta_idx_i64 = arith.index_cast(T.i64, delta_idx)
        delta_gep = _llvm.GEPOp(_llvm_ptr_ty(), delta_ptr, [delta_idx_i64],
                                rawConstantIndices=[_LLVM_GEP_DYNAMIC],
                                elem_type=T.f32,
                                noWrapFlags=0)
        _v1f32_type = T.vec(1, T.f32)
        di_vec = _llvm.LoadOp(_v1f32_type, delta_gep.result).result
        di_full = vector.extract(di_vec, static_position=[0], dynamic_position=[])

        # Use precomputed loop bounds from mask sparsity analysis
        loop_end = loop_end_kv

        # Pre-compute mask row base (loop-invariant: q_row_safe is fixed per workgroup)
        _mask_row_base = (batch_idx * mask_stride_b_idx
                          + q_row_safe * mask_stride_s_idx)

        # Prologue: load first tile's K, V, K^T to LDS (software pipeline)
        coop_load_k(loop_start_kv)
        coop_load_v(loop_start_kv)

        for n_start, inner_iter_args, loop_results in scf.for_(
            loop_start_kv,
            loop_end,
            arith.index(BLOCK_N),
            iter_args=init_args,
        ):
            # Extract dq_accs from iter_args
            dq_accs = [inner_iter_args[0], inner_iter_args[1]]

            # Pre-load mask bits (overlap with sched_barrier wait for pipelined loads)
            _kv_word_idx = n_start // arith.index(32)
            _mask_bits = _gep_load_u32(mask_ptr, _mask_row_base + _kv_word_idx)

            # K/V data already in LDS from prologue or previous iteration's pipeline.

            # K B-operand from LDS (j = lane_mod_32 selects k_row, same address formula).
            s_acc = c_zero_v16f32
            k_base = k_buf_base()

            rocdl.sched_barrier(0)
            for ks in range_constexpr(K_STEPS_QK):
                col = arith.index(ks * K_STEP_QK) + lane_div_32 * MFMA_LANE_K
                k_idx = k_base + lane_mod_32 * K_STRIDE + col
                k_pack = vector.load_op(mfma_pack_type, lds_k, [k_idx])
                s_acc = mfma_acc(k_pack, q_a_packs[ks], s_acc)

            # ---- Apply scale and add mask ----
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

            # ==================================================================
            # ---- dQ COMPUTATION: dP → dS → K^T LDS → dS repack → dQ MFMA ----
            # ==================================================================

            # ---- Step B: dP GEMM = V @ dO^T ----
            dp_acc = c_zero_v16f32
            v_base = arith.index(0)

            for ks in range_constexpr(K_STEPS_QK):
                col_v = arith.index(ks * K_STEP_QK) + lane_div_32 * MFMA_LANE_K
                v_idx = v_base + lane_mod_32 * K_STRIDE + col_v
                v_pack = vector.load_op(mfma_pack_type, lds_v, [v_idx])
                dp_acc = mfma_acc(v_pack, do_b_packs[ks], dp_acc)

            # ---- Step D: dS_unscaled = P * (dP - Di) element-wise ----
            ds_acc = c_zero_v16f32
            for r in range_constexpr(16):
                p_val_ds = vector.extract(s_acc, static_position=[r], dynamic_position=[])
                dp_val_ds = vector.extract(dp_acc, static_position=[r], dynamic_position=[])
                dp_minus_di = arith.SubFOp(dp_val_ds, di_full, fastmath=fm_fast).result
                ds_val = arith.MulFOp(p_val_ds, dp_minus_di, fastmath=fm_fast).result
                ds_acc = vector.insert(ds_val, ds_acc, static_position=[r], dynamic_position=[])

            # ---- Step E: dS register bypass (V12c) ----
            # Truncate f32→bf16 via top-16-bit extraction (single ShRUI per element)
            _v4_type_ds = T.vec(4, elem_type)
            ds_b_packs = []
            for grp in range_constexpr(PV_K_STEPS):
                ds_bf16_elems = []
                for sub in range_constexpr(4):
                    r_idx = grp * 4 + sub
                    ds_f32 = vector.extract(ds_acc, static_position=[r_idx], dynamic_position=[])
                    ds_i32 = arith.ArithValue(ds_f32).bitcast(T.i32)
                    ds_top16 = arith.ShRUIOp(ds_i32, arith.constant(16, type=T.i32)).result
                    ds_i16 = arith.trunci(T.i16, ds_top16)
                    ds_bf16_elems.append(arith.ArithValue(ds_i16).bitcast(elem_type))
                ds_b_packs.append(vector.from_elements(_v4_type_ds, ds_bf16_elems))

            # ---- Step F2: Pre-read ALL K^T packs from lds_k (strided scalar reads) ----
            # Read K[k_row, d_col] from lds_k and pack as K^T[d_col, k_row] v4bf16.
            # Must complete before coop_load overwrites lds_k.
            _v1_type_kt_read = T.vec(1, elem_type)
            def read_KT_from_lds(pks, d_chunk):
                d_pos_k = arith.index(d_chunk * 32) + lane_mod_32
                k_row_base = arith.index(pks * 8) + lane_div_32 * arith.index(4)
                kt_elems = []
                for _i in range_constexpr(4):
                    k_row = k_row_base + arith.index(_i)
                    k_idx = k_row * arith.index(K_STRIDE) + d_pos_k
                    v1 = vector.load_op(_v1_type_kt_read, lds_k, [k_idx])
                    kt_elems.append(vector.extract(v1, static_position=[0], dynamic_position=[]))
                return vector.from_elements(mfma_pack_type, kt_elems)

            kt_a_packs = []
            for d_chunk in range_constexpr(D_CHUNKS):
                kt_chunk = []
                for pks in range_constexpr(PV_K_STEPS):
                    kt_chunk.append(read_KT_from_lds(pks, d_chunk))
                kt_a_packs.append(kt_chunk)

            # ---- Pipeline: load NEXT iteration's K/V to LDS ----
            # sched_barrier ensures K^T scalar reads from lds_k complete before coop_load overwrites.
            rocdl.sched_barrier(0)
            n_next = n_start + arith.index(BLOCK_N)
            has_next = arith.cmpi(arith.CmpIPredicate.slt, n_next, loop_end)
            _if_has_next = scf.IfOp(has_next)
            with ir.InsertionPoint(_if_has_next.then_block):
                coop_load_k(n_next)
                coop_load_v(n_next)
                scf.YieldOp([])

            # ---- Step G: dQ MFMA = K^T @ dS (overlaps with pipelined coop_load) ----
            for d_chunk in range_constexpr(D_CHUNKS):
                for pks in range_constexpr(PV_K_STEPS):
                    dq_accs[d_chunk] = mfma_acc(kt_a_packs[d_chunk][pks], ds_b_packs[pks], dq_accs[d_chunk])

            # ---- Yield updated dQ accumulators for next iteration ----
            yield [dq_accs[0], dq_accs[1]]

        # ==================================================================
        # ---- POST-LOOP: Store dQ from accumulated results ----
        # ==================================================================
        # Extract final accumulators from loop results
        final_dq_accs = [loop_results[0], loop_results[1]]

        # ---- Store dQ: C layout lane_mod_32=q_row, j_formula=d_col ----
        # dQ MFMA output: lane_mod_32 = q_row (B's), j_formula = d_col (A's)
        # Only wave 0 stores (32 q-rows from wave 0's computation).
        #
        # V6 optimization: vectorized v4bf16 store.
        # MFMA C output d_col formula: lane_div_32*4 + (r//4)*8 + (r%4)
        # For each group of r=[grp*4 .. grp*4+3], d_col values are consecutive:
        #   lane_div_32*4 + grp*8 + {0,1,2,3}
        # So we pack 4 bf16 values into v4bf16 and store once per group.
        # This reduces 32 scalar stores to 8 vectorized v4bf16 stores (4x reduction).
        _v4_store_dq_type = T.vec(4, elem_type)
        cond_w0_store_dq = arith.cmpi(arith.CmpIPredicate.eq, wave_id, arith.index(0))
        _if_store_dq = scf.IfOp(cond_w0_store_dq)
        with ir.InsertionPoint(_if_store_dq.then_block):
            q_row_dq = q_start_fixed + lane_mod_32  # q_row in global coords
            cond_q_dq = arith.cmpi(arith.CmpIPredicate.ult, q_row_dq, seq_len_v)
            _if_q_valid = scf.IfOp(cond_q_dq)
            with ir.InsertionPoint(_if_q_valid.then_block):
                for d_chunk in range_constexpr(D_CHUNKS):
                    for grp in range_constexpr(4):  # 4 groups of 4 consecutive r values
                        # V9: apply deferred scale factor before bf16 truncation.
                        # dQ_acc = K^T @ (P * (dP - Di)), needs * scale for final dQ.
                        dq_bf16_elems = []
                        for sub in range_constexpr(4):
                            r_idx = grp * 4 + sub
                            dq_f32_raw = vector.extract(final_dq_accs[d_chunk], static_position=[r_idx], dynamic_position=[])
                            dq_f32 = arith.MulFOp(dq_f32_raw, c_scale, fastmath=fm_fast).result
                            dq_i32 = arith.ArithValue(dq_f32).bitcast(T.i32)
                            dq_top16 = arith.ShRUIOp(dq_i32, arith.constant(16, type=T.i32)).result
                            dq_i16 = arith.trunci(T.i16, dq_top16)
                            dq_bf16_elems.append(arith.ArithValue(dq_i16).bitcast(elem_type))
                        v4_dq = vector.from_elements(_v4_store_dq_type, dq_bf16_elems)
                        # d_col_base = d_chunk*32 + lane_div_32*4 + grp*8
                        # This is the first of 4 consecutive d_col positions
                        d_col_base = arith.index(d_chunk * 32) + lane_div_32 * arith.index(4) + arith.index(grp * 8)
                        g_idx_dq = q_global_idx(q_row_dq, d_col_base)
                        idx_i64_dq = arith.index_cast(T.i64, g_idx_dq)
                        gep_dq = _llvm.GEPOp(_llvm_ptr_ty(), dq_ptr, [idx_i64_dq],
                                             rawConstantIndices=[_LLVM_GEP_DYNAMIC],
                                             elem_type=elem_type,
                                             noWrapFlags=0)
                        _llvm.StoreOp(v4_dq, gep_dq.result)
                scf.YieldOp([])
            scf.YieldOp([])

    @flyc.jit
    def launch_attn_bwd_dq(
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
        num_q_tiles = (sl_idx + BLOCK_M - 1) // BLOCK_M
        grid_x = bs_idx * num_q_tiles * NUM_HEADS

        launcher = attn_bwd_dq_kernel(Q, K, V, dV, dK, dQ, Mask, LSE, Delta, dO, LoopBounds, mask_stride_b, mask_stride_s, seq_len)

        flat_wg_attr = ir.StringAttr.get(f"{BLOCK_SIZE},{BLOCK_SIZE}")
        for op in ctx.gpu_module_body.operations:
            if getattr(op, "OPERATION_NAME", None) == "gpu.func":
                op.attributes["rocdl.flat_work_group_size"] = flat_wg_attr

        launcher.launch(
            grid=(grid_x, 1, 1),
            block=(BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    return launch_attn_bwd_dq
