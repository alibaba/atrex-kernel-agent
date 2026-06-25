# SPDX-License-Identifier: Apache-2.0
# SageAttention FlyDSL Implementation for AMD CDNA3/4
#
# Algorithm: Quantized Flash Attention (SageAttn)
#   - Q, K: INT8 quantized with per-block descale
#   - V: FP8 (E4M3FN) quantized with per-head descale
#   - GEMM1 (QK): mfma_i32_32x32x16_i8  ->  INT32 -> FP32 -> softmax
#   - GEMM2 (PV): mfma_f32_32x32x16_fp8_fp8  ->  FP32 accumulate
#   - Online softmax (log2-based) with rescaling
#
# Tile: BLOCK_M=128, BLOCK_N=64, 4 waves (256 threads), MFMA32
# GEMM1: K @ Q^T so S/P live in MFMA32 register layout -> fed directly to GEMM2
# K uses LDS with XOR swizzle; V pre-transposed to [B,H,D,S], vectorized LDS stores
#
# Optimizations:
# - "recompute from i32 accumulators" - keep v16i32 accumulators, re-extract
#   during PV phase to reduce register pressure
# - Deferred row-max scaling: find max of unscaled FP32, scale once (-31 MUL)
# - FMA fusion in PV: fma(s_f32, qk_scale, -m_new) replaces mul+sub (-32 VALU)
# - V LDS prefetch in PV: pre-read V for dc+1 while MFMA[dc] executes
# - Scheduling hints: sched_dsrd/sched_mfma for QK MFMA interleaving
#
# Layout: Q/K/V/O are 1D flattened from BSHD (batch, seq_len, num_heads, head_dim)
# Grid: (batch * num_q_tiles * num_heads,)
# Block: (256,)

import math
import os

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, buffer_ops, gpu, range_constexpr, rocdl, vector
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from flydsl._mlir import ir
from flydsl._mlir.dialects import scf, fly as _fly, llvm as _llvm, math as math_dialect

KERNEL_NAME = "sage_attn_kernel"
_LOG2E = math.log2(math.e)  # 1.4426950408889634
_LLVM_GEP_DYNAMIC = -2147483648

# s_waitcnt encoding helpers for gfx942
_VMCNT_LO_MASK = 0xF
_VMCNT_HI_SHIFT = 14
_VMCNT_HI_MASK = 0x3
_LGKMCNT_EXPCNT_BASE = (63 << 8) | (7 << 4)  # lgkmcnt=63 (max), expcnt=7 (max)


def _waitcnt_vm_n(n):
    """Emit s_waitcnt vmcnt(n) only (lgkmcnt=63, expcnt=7)."""
    val = (n & _VMCNT_LO_MASK) | _LGKMCNT_EXPCNT_BASE | (
        ((n >> 4) & _VMCNT_HI_MASK) << _VMCNT_HI_SHIFT
    )
    rocdl.s_waitcnt(val)


def _llvm_ptr_ty():
    return ir.Type.parse("!llvm.ptr")


def _llvm_lds_ptr_ty():
    return ir.Type.parse("!llvm.ptr<3>")


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_sage_attn_module(
    num_heads,
    head_dim,
    sm_scale=None,
    waves_per_eu=3,
    block_m=128,
    block_n=64,
):
    """Build a SageAttention kernel for AMD CDNA3/4.

    Parameters
    ----------
    num_heads : int
        Number of attention heads.
    head_dim : int
        Head dimension (must be divisible by 32, >= 64).
    sm_scale : float or None
        Softmax scale (default: 1/sqrt(head_dim)).
    block_m : int
        Q-tile size along sequence dimension.
    block_n : int
        KV-tile size along sequence dimension.

    Returns
    -------
    callable
        JIT-compiled launcher function.
    """
    gpu_arch = get_rocm_arch()

    # ---- Tile configuration ----
    BLOCK_M = block_m
    BLOCK_N = block_n
    HEAD_DIM = head_dim
    WARP_SIZE = 64
    BLOCK_SIZE = 256          # 4 waves
    NUM_WAVES = BLOCK_SIZE // WARP_SIZE  # 4
    ROWS_PER_WAVE = BLOCK_M // NUM_WAVES  # 32  (matches MFMA-32 tile)
    K_SUB_N = 32              # half of BLOCK_N; each MFMA covers 32 KV positions

    # QK MFMA: INT8 32x32x16
    K_STEP_QK = 16            # K-dimension per INT8 MFMA
    K_STEPS_QK = HEAD_DIM // K_STEP_QK
    MFMA_LANE_K = 8           # i8 values per lane per MFMA (8 bytes -> i64)

    # PV MFMA: FP8 32x32x16
    PV_K_STEP = 16            # K-dimension per FP8 MFMA
    PV_K_STEPS = K_SUB_N // PV_K_STEP  # 2

    D_CHUNK = 32
    D_CHUNKS = HEAD_DIM // D_CHUNK  # 4

    NUM_HEADS = num_heads
    STRIDE_TOKEN = NUM_HEADS * HEAD_DIM

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    assert head_dim % 32 == 0, f"head_dim ({head_dim}) must be divisible by 32"
    assert head_dim >= 64, f"head_dim ({head_dim}) must be >= 64"
    assert BLOCK_M % ROWS_PER_WAVE == 0
    assert BLOCK_N % K_SUB_N == 0

    # ---- LDS layout ----
    # K: INT8, row-major [BLOCK_N, HEAD_DIM] with XOR swizzle at 16-byte granularity
    K_STRIDE = HEAD_DIM  # in i8 elements (= bytes)
    # V: FP8, stored transposed as V^T [HEAD_DIM, BLOCK_N] for direct MFMA B-operand load
    VT_STRIDE = BLOCK_N + 8  # +8 padding for 8-byte alignment (72%8=0), eliminates ds_read_b64 bank conflicts

    # Cooperative vector load config (i8/fp8 = 1 byte each)
    VEC_WIDTH = 16                            # 16 bytes per vector load

    # K load: threads sweep along HEAD_DIM (inner dim of K[BLOCK_N, HEAD_DIM])
    THREADS_PER_ROW = HEAD_DIM // VEC_WIDTH   # 128/16 = 8
    ROWS_PER_BATCH = BLOCK_SIZE // THREADS_PER_ROW  # 256/8 = 32
    if ROWS_PER_BATCH >= BLOCK_N:
        NUM_BATCHES_KV = 1
        KV_NEEDS_GUARD = ROWS_PER_BATCH > BLOCK_N
    else:
        NUM_BATCHES_KV = BLOCK_N // ROWS_PER_BATCH  # 64/32 = 2
        KV_NEEDS_GUARD = False

    # V^T load: threads sweep along BLOCK_N (inner dim of V^T[HEAD_DIM, BLOCK_N])
    # V is pre-transposed in Python to [B, H, D, S], so we load V^T[d, n:n+16]
    # and write directly to LDS V^T with vectorized ds_write_b128.
    VT_THREADS_PER_ROW = BLOCK_N // VEC_WIDTH     # 64/16 = 4
    VT_ROWS_PER_BATCH = BLOCK_SIZE // VT_THREADS_PER_ROW  # 256/4 = 64
    VT_NUM_BATCHES = HEAD_DIM // VT_ROWS_PER_BATCH  # 128/64 = 2

    # LDS sizes (bytes) -- both K and V single-buffered
    NUM_K_BUFS = 1   # K: single buffer (17408B total → 3 WGs/CU → 3 waves/SIMD)
    NUM_V_BUFS = 1   # V: single buffer
    LDS_K_TILE = BLOCK_N * K_STRIDE
    LDS_V_TILE = HEAD_DIM * VT_STRIDE
    LDS_K_TOTAL = NUM_K_BUFS * LDS_K_TILE
    LDS_V_BASE = LDS_K_TOTAL
    LDS_V_TOTAL = NUM_V_BUFS * LDS_V_TILE
    LDS_TOTAL = LDS_K_TOTAL + LDS_V_TOTAL

    allocator = SmemAllocator(None, arch=gpu_arch, global_sym_name="sage_attn_smem")
    lds_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_offset + LDS_TOTAL

    # ---- Resolve MFMA intrinsics ----
    mfma_i32_qk = (
        getattr(rocdl, "mfma_i32_32x32x16_i8", None)
        or getattr(rocdl, "mfma_i32_32x32x16i8", None)
    )
    if mfma_i32_qk is None:
        raise AttributeError(
            "INT8 32x32x16 MFMA not found in rocdl "
            "(expected mfma_i32_32x32x16_i8 or mfma_i32_32x32x16i8)"
        )

    mfma_f32_pv = (
        getattr(rocdl, "mfma_f32_32x32x16_fp8_fp8", None)
        or getattr(rocdl, "mfma_f32_32x32x16fp8fp8", None)
        or getattr(rocdl, "mfma_f32_32x32x16_f8f6f4", None)
    )
    if mfma_f32_pv is None:
        raise AttributeError(
            "FP8 32x32x16 MFMA not found in rocdl "
            "(expected mfma_f32_32x32x16_fp8_fp8)"
        )

    # =====================================================================
    # Kernel
    # =====================================================================

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def sage_attn_kernel(
        Q: fx.Tensor,           # INT8 [B*S*H*D] flattened
        K_tensor: fx.Tensor,    # INT8 [B*S*H*D] flattened
        V: fx.Tensor,           # FP8  [B*S*H*D] flattened
        O: fx.Tensor,           # BF16 [B*S*H*D] flattened
        descale_q: fx.Tensor,   # FP32 [B, H, num_q_tiles]
        descale_k: fx.Tensor,   # FP32 [B, H, num_k_tiles]
        descale_v: fx.Tensor,   # FP32 [B, H]
        seq_len: fx.Int32,
    ):
        # -- Type aliases --
        i8_elem = T.i8
        compute_type = T.f32
        out_elem = T.bf16
        v16i32_type = T.vec(16, T.i32)
        v16f32_type = T.vec(16, compute_type)
        v16i8_type = T.vec(16, i8_elem)
        v1i8_type = T.vec(1, i8_elem)
        v8i8_type = T.vec(8, i8_elem)
        v1i64_type = T.vec(1, T.i64)
        v2i32_type = T.vec(2, T.i32)

        fm_fast = arith.FastMathFlags.fast
        _mfma_zero = ir.IntegerAttr.get(ir.IntegerType.get_signless(32), 0)

        def do_mfma_qk(a_i64, b_i64, c_v16i32):
            """INT8 32x32x16 MFMA: accumulate in INT32."""
            return mfma_i32_qk(
                v16i32_type, a_i64, b_i64, c_v16i32,
                _mfma_zero, _mfma_zero, _mfma_zero,
            ).result

        def do_mfma_pv(a_i64, b_i64, c_v16f32):
            """FP8 32x32x16 MFMA: accumulate in FP32."""
            return mfma_f32_pv(
                v16f32_type, a_i64, b_i64, c_v16f32,
                _mfma_zero, _mfma_zero, _mfma_zero,
            ).result

        # -- Extract raw pointers --
        q_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), Q)
        k_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), K_tensor)
        v_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), V)
        o_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), O)
        dq_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), descale_q)
        dk_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), descale_k)
        dv_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), descale_v)

        seq_len_v = arith.index_cast(T.index, seq_len)

        # -- LDS view (typed as i8, 1 byte per element for INT8/FP8) --
        base_ptr = allocator.get_base()
        lds_mem = SmemPtr(
            base_ptr, lds_offset, i8_elem, shape=(LDS_TOTAL,)
        ).get()

        # -- Thread / block indices --
        block_id = arith.index_cast(T.index, gpu.block_idx.x)
        tid = arith.index_cast(T.index, gpu.thread_idx.x)

        wave_id = tid // WARP_SIZE
        lane = tid % WARP_SIZE
        lane_mod_32 = lane % 32
        lane_div_32 = lane // 32  # 0 or 1: K-subblock selector

        wave_q_offset = wave_id * ROWS_PER_WAVE

        # -- Decompose block_id -> batch, head, q_tile --
        # q_tile-major ordering: consecutive blocks share K/V data for
        # better L2 cache locality (same head's KV stays in cache).
        num_q_tiles = (seq_len_v + BLOCK_M - 1) // BLOCK_M
        q_tile_idx = block_id % num_q_tiles
        batch_head_id = block_id // num_q_tiles
        head_idx = batch_head_id % NUM_HEADS
        batch_idx = batch_head_id // NUM_HEADS
        q_start = q_tile_idx * BLOCK_M

        # -- Cooperative load decomposition (K) --
        load_row_in_batch = tid // THREADS_PER_ROW
        load_lane_in_row = tid % THREADS_PER_ROW
        load_col_base = load_lane_in_row * VEC_WIDTH

        # -- Cooperative load decomposition (V^T) --
        vt_load_row_in_batch = tid // VT_THREADS_PER_ROW  # D-row within batch
        vt_load_col_base = (tid % VT_THREADS_PER_ROW) * VEC_WIDTH  # N-offset

        # -- Helpers: global memory access --
        def global_byte_idx(token_idx, col):
            """Flat byte offset for INT8/FP8 tensors (1 byte per element)."""
            token = batch_idx * seq_len_v + token_idx
            return token * STRIDE_TOKEN + head_idx * HEAD_DIM + col

        def global_vt_byte_idx(d_idx, s_idx):
            """Flat byte offset for V^T[B, H, D, S] layout (1 byte per element)."""
            bh = batch_idx * arith.index(NUM_HEADS) + head_idx
            return (bh * arith.index(HEAD_DIM) + d_idx) * seq_len_v + s_idx

        def global_f32_idx(base_offset):
            """Flat element offset for FP32 scale tensors."""
            return base_offset

        def _gep_load(base_ptr, elem_idx, vec_type, elem_type_ir):
            idx_i64 = arith.index_cast(T.i64, elem_idx)
            gep = _llvm.GEPOp(
                _llvm_ptr_ty(), base_ptr, [idx_i64],
                rawConstantIndices=[_LLVM_GEP_DYNAMIC],
                elem_type=elem_type_ir, noWrapFlags=0,
            )
            return _llvm.LoadOp(vec_type, gep.result).result

        def _gep_store(val, base_ptr, elem_idx, elem_type_ir):
            idx_i64 = arith.index_cast(T.i64, elem_idx)
            gep = _llvm.GEPOp(
                _llvm_ptr_ty(), base_ptr, [idx_i64],
                rawConstantIndices=[_LLVM_GEP_DYNAMIC],
                elem_type=elem_type_ir, noWrapFlags=0,
            )
            _llvm.StoreOp(val, gep.result)

        def load_global_i8xN(ptr, byte_idx):
            """Load VEC_WIDTH consecutive INT8 bytes from global memory."""
            return _gep_load(ptr, byte_idx, v16i8_type, i8_elem)

        def load_global_i64(ptr, byte_idx):
            """Load 8 consecutive bytes as i64 (for MFMA operand)."""
            return _gep_load(ptr, byte_idx, T.i64, i8_elem)

        def load_global_f32(ptr, elem_idx):
            """Load single FP32 scalar."""
            return _gep_load(ptr, elem_idx, compute_type, compute_type)

        def lds_load_i64(byte_idx):
            """Load 8 bytes from LDS as i64 (for MFMA operand packing)."""
            v8 = vector.load_op(v8i8_type, lds_mem, [byte_idx])
            v1 = vector.bitcast(v1i64_type, v8)
            return vector.extract(v1, static_position=[0], dynamic_position=[])

        # -- K XOR swizzle: col ^ ((row & 7) << 4) at 16-byte granularity --
        def _k_swizzle(row_idx, col_idx):
            mask = (row_idx & arith.index(0x7)) << arith.index(4)
            return col_idx ^ mask

        # -- LDS double-buffer base helpers --
        def k_buf_base(buf_id=0):
            # K is single-buffered — always at offset 0 (ignore buf_id)
            return arith.index(0)

        def v_buf_base(buf_id=0):
            # V is single-buffered — always at LDS_V_BASE (ignore buf_id)
            return arith.index(LDS_V_BASE)

        # -- Cooperative K load (INT8, row-major, XOR-swizzled, double-buffered) --
        def coop_load_k(tile_start, buf_id=0):
            kb = k_buf_base(buf_id)
            for batch in range_constexpr(NUM_BATCHES_KV):
                row_offset = batch * ROWS_PER_BATCH
                row_idx = tile_start + load_row_in_batch + row_offset
                if KV_NEEDS_GUARD:
                    row_valid = arith.cmpi(
                        arith.CmpIPredicate.ult,
                        load_row_in_batch,
                        arith.index(BLOCK_N),
                    )
                    _if_k = scf.IfOp(row_valid)
                    with ir.InsertionPoint(_if_k.then_block):
                        g_idx = global_byte_idx(row_idx, load_col_base)
                        lds_row = load_row_in_batch + row_offset
                        swz_col = _k_swizzle(lds_row, load_col_base)
                        lds_idx = kb + lds_row * K_STRIDE + swz_col
                        vec = load_global_i8xN(k_ptr, g_idx)
                        vector.store(vec, lds_mem, [lds_idx])
                        scf.YieldOp([])
                else:
                    g_idx = global_byte_idx(row_idx, load_col_base)
                    lds_row = load_row_in_batch + row_offset
                    swz_col = _k_swizzle(lds_row, load_col_base)
                    lds_idx = kb + lds_row * K_STRIDE + swz_col
                    vec = load_global_i8xN(k_ptr, g_idx)
                    vector.store(vec, lds_mem, [lds_idx])

        # -- Cooperative V^T load (FP8, pre-transposed [B,H,D,S], vectorized) --
        # V is pre-transposed in Python to [B, H, D, S] layout.
        # Each thread loads 16 consecutive N-values for one D-row from global,
        # then writes directly to LDS V^T[D, BLOCK_N] with ds_write_b128.
        # This replaces the old byte-scatter approach (32× ds_write_b8 per iter).
        def coop_load_vt(tile_start, buf_id=0):
            vb = v_buf_base(buf_id)
            for batch in range_constexpr(VT_NUM_BATCHES):
                d_idx = vt_load_row_in_batch + batch * VT_ROWS_PER_BATCH
                s_idx = tile_start + vt_load_col_base
                g_idx = global_vt_byte_idx(d_idx, s_idx)
                vec = load_global_i8xN(v_ptr, g_idx)
                lds_idx = vb + d_idx * VT_STRIDE + vt_load_col_base
                vector.store(vec, lds_mem, [lds_idx])

        # -- Split prefetch: issue global loads (non-blocking) --
        def coop_load_kv_global(tile_start):
            """Issue global loads for K and V^T, return vectors."""
            k_vecs = []
            v_vecs = []
            for batch in range_constexpr(NUM_BATCHES_KV):
                row_offset = batch * ROWS_PER_BATCH
                row_idx = tile_start + load_row_in_batch + row_offset
                g_idx_k = global_byte_idx(row_idx, load_col_base)
                k_vecs.append(load_global_i8xN(k_ptr, g_idx_k))
            for batch in range_constexpr(VT_NUM_BATCHES):
                d_idx = vt_load_row_in_batch + batch * VT_ROWS_PER_BATCH
                s_idx = tile_start + vt_load_col_base
                g_idx_v = global_vt_byte_idx(d_idx, s_idx)
                v_vecs.append(load_global_i8xN(v_ptr, g_idx_v))
            return k_vecs, v_vecs

        def coop_store_kv_lds(k_vecs, v_vecs, buf_id):
            """Store previously-loaded K/V^T data to LDS (vectorized)."""
            kb = k_buf_base(buf_id)
            vb = v_buf_base(buf_id)
            for batch in range_constexpr(NUM_BATCHES_KV):
                lds_row = load_row_in_batch + batch * ROWS_PER_BATCH
                swz_col = _k_swizzle(lds_row, load_col_base)
                lds_idx = kb + lds_row * K_STRIDE + swz_col
                vector.store(k_vecs[batch], lds_mem, [lds_idx])
            for batch in range_constexpr(VT_NUM_BATCHES):
                d_idx = vt_load_row_in_batch + batch * VT_ROWS_PER_BATCH
                lds_idx = vb + d_idx * VT_STRIDE + vt_load_col_base
                vector.store(v_vecs[batch], lds_mem, [lds_idx])

        # ================================================================
        # Q preload: B-operand packs kept in registers for entire kernel
        # ================================================================
        q_row = q_start + wave_q_offset + lane_mod_32
        q_in_bounds = arith.cmpi(arith.CmpIPredicate.slt, q_row, seq_len_v)
        q_row_safe = arith.select(q_in_bounds, q_row, arith.index(0))

        c_zero_i64 = arith.constant(0, type=T.i64)
        q_b_packs = []
        for ks in range_constexpr(K_STEPS_QK):
            q_col = arith.index(ks * K_STEP_QK) + lane_div_32 * MFMA_LANE_K
            g_idx = global_byte_idx(q_row_safe, q_col)
            raw = load_global_i64(q_ptr, g_idx)
            q_b_packs.append(arith.select(q_in_bounds, raw, c_zero_i64))

        # ================================================================
        # Load scales
        # ================================================================
        # Per-block Q scale: descale_q[batch, head, q_tile]
        dq_idx = (batch_idx * arith.index(NUM_HEADS) * num_q_tiles
                  + head_idx * num_q_tiles + q_tile_idx)
        q_scale_raw = load_global_f32(dq_ptr, dq_idx)

        # Pre-multiply Q scale by sm_scale * log2(e) for log2-based softmax
        c_sm_log2e = arith.constant(sm_scale * _LOG2E, type=compute_type)
        q_scale = arith.MulFOp(q_scale_raw, c_sm_log2e, fastmath=fm_fast).result

        # Per-head V scale: descale_v[batch, head]
        dv_idx = batch_idx * arith.index(NUM_HEADS) + head_idx
        v_scale_val = load_global_f32(dv_ptr, dv_idx)

        # ================================================================
        # Constants
        # ================================================================
        c_neg_inf = arith.constant(float("-inf"), type=compute_type)
        c_zero_f = arith.constant(0.0, type=compute_type)
        c_one_f = arith.constant(1.0, type=compute_type)
        c_zero_v16f32 = arith.constant_vector(0.0, v16f32_type)

        # INT32 zero vector for QK accumulator
        zero_i32 = arith.constant(0, type=T.i32)
        c_zero_v16i32 = vector.broadcast(v16i32_type, zero_i32)

        shuf_32_i32 = arith.constant(32, type=T.i32)
        width_i32 = arith.constant(WARP_SIZE, type=T.i32)
        lane_i32 = arith.index_cast(T.i32, lane)

        def reduction_peer(v_f32):
            """Cross-lane reduction via XOR shuffle (lane ^ 32)."""
            return arith.ArithValue(v_f32).shuffle_xor(shuf_32_i32, width_i32)

        # ================================================================
        # KV loop upper bound
        # ================================================================
        kv_upper = seq_len_v

        # ================================================================
        # Main KV loop (online softmax + GEMM1 + GEMM2)
        # Double-buffered: prefetch first tile before loop, then overlap
        # next tile's loads with current tile's compute.
        # ================================================================
        # Prefetch first K/V tile into buffer 0
        coop_load_k(arith.index(0), buf_id=0)
        coop_load_vt(arith.index(0), buf_id=0)
        gpu.barrier()

        # Loop-carried: [m_old, l_old, buf_id, o_acc_chunk_0, ..., o_acc_chunk_{D-1}]
        c_buf_0 = arith.index(0)
        init_args = [c_neg_inf, c_zero_f, c_buf_0]
        for _ in range_constexpr(D_CHUNKS):
            init_args.append(c_zero_v16f32)

        for kv_block_start, inner_iter_args, loop_results in scf.for_(
            arith.index(0),
            kv_upper,
            arith.index(BLOCK_N),
            iter_args=init_args,
        ):
            m_running = inner_iter_args[0]
            l_running = inner_iter_args[1]
            cur_buf = inner_iter_args[2]
            o_accs = [inner_iter_args[3 + i] for i in range_constexpr(D_CHUNKS)]

            # Current buffer bases for K and V reads
            cur_k_base = k_buf_base(cur_buf)
            cur_v_base = v_buf_base(cur_buf)

            # Next buffer id (toggle 0<->1)
            next_buf = arith.index(1) - cur_buf

            # ---- Issue global loads for next tile EARLY (non-blocking) ----
            # VMEM loads will be in-flight during QK + softmax + PV compute.
            # Use safe address (current tile) if no next tile to avoid OOB.
            next_kv_start = kv_block_start + arith.index(BLOCK_N)
            has_next = arith.cmpi(
                arith.CmpIPredicate.ult, next_kv_start, kv_upper)
            safe_pf_start = arith.select(has_next, next_kv_start, kv_block_start)
            pf_k_vecs, pf_v_vecs = coop_load_kv_global(safe_pf_start)

            # ---- Per-block K scale: descale_k[batch, head, k_tile] ----
            k_tile_idx_v = kv_block_start // arith.index(BLOCK_N)
            num_k_tiles = (seq_len_v + arith.index(BLOCK_N) - 1) // arith.index(BLOCK_N)
            dk_idx = (batch_idx * arith.index(NUM_HEADS) * num_k_tiles
                      + head_idx * num_k_tiles + k_tile_idx_v)
            k_scale = load_global_f32(dk_ptr, dk_idx)

            # Pre-compute combined scale: q_scale * k_scale (saves 32 VALU/iter)
            qk_scale = arith.MulFOp(q_scale, k_scale, fastmath=fm_fast).result

            # ============================================================
            # GEMM1: QK = K @ Q^T via INT8 MFMA 32x32x16
            # K is A-operand from LDS; Q^T is B-operand from registers
            # Two 32-row sub-tiles: lo (rows 0-31), hi (rows 32-63)
            # ============================================================
            k_swz_mask = (lane_mod_32 & arith.index(0x7)) << arith.index(4)
            k_hi_offset = arith.index(K_SUB_N * K_STRIDE)

            s_acc_lo = c_zero_v16i32
            s_acc_hi = c_zero_v16i32

            # Software-pipelined QK MFMA: prefetch depth 2 (matches flash_attn)
            _QK_PF_DEPTH = 2
            k_packs_lo = [None] * K_STEPS_QK
            k_packs_hi = [None] * K_STEPS_QK
            for p in range_constexpr(_QK_PF_DEPTH):
                col_p = arith.index(p * K_STEP_QK) + lane_div_32 * MFMA_LANE_K
                swz_p = col_p ^ k_swz_mask
                k_packs_lo[p] = lds_load_i64(cur_k_base + lane_mod_32 * K_STRIDE + swz_p)
                k_packs_hi[p] = lds_load_i64(cur_k_base + k_hi_offset + lane_mod_32 * K_STRIDE + swz_p)

            for ks in range_constexpr(K_STEPS_QK):
                # Scheduling: interleave LDS reads with MFMA
                rocdl.sched_dsrd(2)
                rocdl.sched_mfma(2)
                s_acc_lo = do_mfma_qk(k_packs_lo[ks], q_b_packs[ks], s_acc_lo)
                s_acc_hi = do_mfma_qk(k_packs_hi[ks], q_b_packs[ks], s_acc_hi)
                # Prefetch ks+depth (if within bounds)
                if ks + _QK_PF_DEPTH < K_STEPS_QK:
                    col_nxt = arith.index((ks + _QK_PF_DEPTH) * K_STEP_QK) + lane_div_32 * MFMA_LANE_K
                    swz_nxt = col_nxt ^ k_swz_mask
                    k_packs_lo[ks + _QK_PF_DEPTH] = lds_load_i64(cur_k_base + lane_mod_32 * K_STRIDE + swz_nxt)
                    k_packs_hi[ks + _QK_PF_DEPTH] = lds_load_i64(cur_k_base + k_hi_offset + lane_mod_32 * K_STRIDE + swz_nxt)

            # Barrier: drain all QK MFMAs before softmax
            rocdl.sched_barrier(0)

            # ============================================================
            # Row-max extraction pass: INT32 -> FP32, find max UNSCALED
            # Since qk_scale > 0, max(x * c) == max(x) * c. We defer
            # the qk_scale multiply to after the cross-lane reduction,
            # saving 31 MUL instructions per iteration.
            # ============================================================
            # Combined row-max + value extraction: extract all 32 i32→f32
            # values once, find max, then reuse f32 values in softmax.
            _max_fm = {"fastmath": fm_fast}
            s_raw_lo = []
            s_raw_hi = []
            local_max_raw = c_neg_inf
            for r in range_constexpr(16):
                s_i32_lo = vector.extract(
                    s_acc_lo, static_position=[r], dynamic_position=[])
                s_i32_hi = vector.extract(
                    s_acc_hi, static_position=[r], dynamic_position=[])
                s_f32_lo = arith.SIToFPOp(compute_type, s_i32_lo).result
                s_f32_hi = arith.SIToFPOp(compute_type, s_i32_hi).result
                s_raw_lo.append(s_f32_lo)
                s_raw_hi.append(s_f32_hi)
                local_max_raw = arith.MaxNumFOp(local_max_raw, s_f32_lo, **_max_fm).result
                local_max_raw = arith.MaxNumFOp(local_max_raw, s_f32_hi, **_max_fm).result

            # Cross-lane max (lane ^ 32), then scale once
            peer_max_raw = reduction_peer(local_max_raw)
            row_max_raw = arith.MaxNumFOp(local_max_raw, peer_max_raw, **_max_fm).result
            row_max = arith.MulFOp(row_max_raw, qk_scale, fastmath=fm_fast).result
            m_new = arith.MaxNumFOp(m_running, row_max, **_max_fm).result

            # Correction factor for previous accumulator
            diff_m = arith.SubFOp(m_running, m_new, fastmath=fm_fast).result
            corr = rocdl.exp2(compute_type, diff_m)

            # Rescale only o_accs[0] now; o_accs[1..3] deferred to PV loop
            corr_vec = vector.broadcast(v16f32_type, corr)
            o_accs[0] = arith.MulFOp(
                o_accs[0], corr_vec, fastmath=fm_fast).result

            # ============================================================
            # PV GEMM with on-the-fly recompute from i32 accumulators
            # Uses FMA to fuse mul+sub, and V prefetching to overlap
            # LDS reads with MFMA execution.
            # ============================================================
            def _pack_8_f32_to_i64(vals_8):
                """Pack 8 FP32 values -> 2 i32 (4 fp8 each) -> 1 i64."""
                w0 = rocdl.cvt_pk_fp8_f32(
                    T.i32, vals_8[0], vals_8[1],
                    fx.Int32(0), False)
                w0 = rocdl.cvt_pk_fp8_f32(
                    T.i32, vals_8[2], vals_8[3],
                    w0, True)
                w1 = rocdl.cvt_pk_fp8_f32(
                    T.i32, vals_8[4], vals_8[5],
                    fx.Int32(0), False)
                w1 = rocdl.cvt_pk_fp8_f32(
                    T.i32, vals_8[6], vals_8[7],
                    w1, True)
                v2 = vector.from_elements(v2i32_type, [w0, w1])
                return vector.extract(
                    vector.bitcast(v1i64_type, v2),
                    static_position=[0], dynamic_position=[])

            local_sum = c_zero_f

            # Pre-compute neg_m_new for FMA: fma(s_f32, qk_scale, neg_m_new)
            neg_m_new = arith.SubFOp(c_zero_f, m_new, fastmath=fm_fast).result

            def _v_index_lo(dc_val, pks_val):
                """V LDS index for lo sub-tile (K positions 0-31)."""
                d_pos = arith.index(dc_val * D_CHUNK) + lane_mod_32
                k_b = arith.index(pks_val * PV_K_STEP) + lane_div_32 * 8
                return cur_v_base + d_pos * VT_STRIDE + k_b

            def _v_index_hi(dc_val, pks_val):
                """V LDS index for hi sub-tile (K positions 32-63)."""
                d_pos = arith.index(dc_val * D_CHUNK) + lane_mod_32
                k_b = arith.index(pks_val * PV_K_STEP) + lane_div_32 * 8
                return cur_v_base + d_pos * VT_STRIDE + k_b + arith.index(K_SUB_N)

            # Phase 1: Pre-pack ALL P values using pre-extracted s_raw values.
            # Reuses f32 values from row-max pass (no re-extract/re-convert).
            p_packs = []  # 4 i64 packs: [lo_pks0, lo_pks1, hi_pks0, hi_pks1]
            for sub_tile in range_constexpr(2):
                s_raw = s_raw_lo if sub_tile == 0 else s_raw_hi
                for pks in range_constexpr(PV_K_STEPS):
                    p_group = []
                    p_base = pks * 8
                    for r in range_constexpr(8):
                        s_f32 = s_raw[p_base + r]
                        diff = math_dialect.fma(s_f32, qk_scale, neg_m_new)
                        p_val = rocdl.exp2(compute_type, diff)
                        p_group.append(p_val)
                        local_sum = arith.AddFOp(
                            local_sum, p_val, fastmath=fm_fast).result
                    p_packs.append(_pack_8_f32_to_i64(p_group))

            # Phase 2: PV GEMM with pre-loaded V data (pure MFMA inner loop)
            # Deferred O rescaling: o_accs[1..3] rescaled during first PV
            # sub-tile at pks==0, dc==0,1,2 to overlap VALU with MFMA pipeline.
            for sub_tile in range_constexpr(2):
                v_idx_fn = _v_index_lo if sub_tile == 0 else _v_index_hi
                # Pre-load all V data for this sub-tile (8 i64 values)
                v_preloaded = []
                for pks in range_constexpr(PV_K_STEPS):
                    for dc in range_constexpr(D_CHUNKS):
                        v_preloaded.append(lds_load_i64(v_idx_fn(dc, pks)))
                # Deferred rescale during first sub-tile's first pks
                if sub_tile == 0:
                    for dc_r in range_constexpr(D_CHUNKS - 1):
                        o_accs[dc_r + 1] = arith.MulFOp(
                            o_accs[dc_r + 1], corr_vec,
                            fastmath=fm_fast).result
                # PV MFMA loop with scheduling hints
                for pks in range_constexpr(PV_K_STEPS):
                    rocdl.sched_mfma(4)
                    p_pack = p_packs[sub_tile * PV_K_STEPS + pks]
                    for dc in range_constexpr(D_CHUNKS):
                        o_accs[dc] = do_mfma_pv(
                            v_preloaded[pks * D_CHUNKS + dc],
                            p_pack, o_accs[dc])

            # PV→sum: no sched_barrier. o_accs aren't read until yield;
            # cross-lane sum is independent of o_accs (uses local_sum).

            # Cross-lane sum
            peer_sum = reduction_peer(local_sum)
            tile_sum = arith.AddFOp(
                local_sum, peer_sum, fastmath=fm_fast).result

            # Update running sum: l_new = corr * l_old + tile_sum
            l_corr = arith.MulFOp(corr, l_running, fastmath=fm_fast).result
            l_new = arith.AddFOp(l_corr, tile_sum, fastmath=fm_fast).result

            # ---- Barrier: V is single-buffered, so all waves must finish
            # reading V from LDS (during PV) before any wave overwrites it. ----
            gpu.barrier()

            # ---- Store prefetched K/V to LDS (VMEM loads issued at start) ----
            _pf_if = scf.IfOp(has_next, [], has_else=False)
            with ir.InsertionPoint(_pf_if.then_block):
                _waitcnt_vm_n(0)  # Wait for all VMEM loads to complete
                coop_store_kv_lds(pf_k_vecs, pf_v_vecs, next_buf)
                scf.YieldOp([])

            # ---- Barrier: wait for LDS writes to complete ----
            gpu.barrier()

            # ---- Yield loop-carried values ----
            m_running = m_new
            l_running = l_new
            yield [m_running, l_running, next_buf] + o_accs

        # ================================================================
        # Normalize output and write back
        # ================================================================
        l_final = loop_results[1]
        # loop_results[2] is buf_id (not needed after loop)
        o_finals = [loop_results[3 + dc] for dc in range_constexpr(D_CHUNKS)]

        # O_norm = O_acc / l_final * v_scale
        inv_l = arith.DivFOp(c_one_f, l_final, fastmath=fm_fast).result
        scale_combined = arith.MulFOp(
            inv_l, v_scale_val, fastmath=fm_fast).result

        # Guard: only write if Q row is in bounds
        # Shift-based f32->bf16 truncation: pack pairs of f32 as bf16x2 in i32
        c16 = arith.constant(16, type=T.i32)
        c_ffff0000 = arith.constant(0xFFFF0000, type=T.i32)

        _o_guard = scf.IfOp(q_in_bounds, [], has_else=False)
        with ir.InsertionPoint(_o_guard.then_block):
            for dc in range_constexpr(D_CHUNKS):
                scale_vec = vector.broadcast(v16f32_type, scale_combined)
                o_norm = arith.MulFOp(
                    o_finals[dc], scale_vec, fastmath=fm_fast).result

                # Extract f32 values, convert to bf16 via bit-shift, store
                for r in range_constexpr(0, 16, 2):
                    o_val0 = vector.extract(
                        o_norm, static_position=[r], dynamic_position=[])
                    o_val1 = vector.extract(
                        o_norm, static_position=[r + 1], dynamic_position=[])

                    # Shift-based f32->bf16: pack 2 bf16 into 1 i32
                    bits0 = arith.bitcast(T.i32, o_val0)
                    bits1 = arith.bitcast(T.i32, o_val1)
                    bf16x2_i32 = arith.OrIOp(
                        arith.ShRUIOp(bits0, c16).result,
                        arith.AndIOp(bits1, c_ffff0000).result,
                    ).result

                    # MFMA32 output remap for both elements
                    d_row_rel_0 = lane_div_32 * 4 + (r // 4) * 8 + (r % 4)
                    d_row_rel_1 = lane_div_32 * 4 + ((r + 1) // 4) * 8 + ((r + 1) % 4)
                    d_col_0 = arith.index(dc * D_CHUNK) + d_row_rel_0
                    d_col_1 = arith.index(dc * D_CHUNK) + d_row_rel_1

                    o_token = batch_idx * seq_len_v + q_row
                    o_base = o_token * arith.index(STRIDE_TOKEN) + head_idx * arith.index(HEAD_DIM)

                    # Store i32 (2 packed bf16) using bf16-strided GEP
                    o_global_0 = o_base + d_col_0
                    _gep_store(bf16x2_i32, o_ptr, o_global_0, out_elem)

            scf.YieldOp([])

    # =====================================================================
    # JIT launcher
    # =====================================================================

    @flyc.jit
    def launch_sage_attn(
        Q: fx.Tensor,
        K_tensor: fx.Tensor,
        V: fx.Tensor,
        O: fx.Tensor,
        descale_q: fx.Tensor,
        descale_k: fx.Tensor,
        descale_v: fx.Tensor,
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

        launcher = sage_attn_kernel(
            Q, K_tensor, V, O,
            descale_q, descale_k, descale_v,
            seq_len,
        )

        if waves_per_eu is not None:
            _wpe = int(waves_per_eu)
            if _wpe >= 1:
                for op in ctx.gpu_module_body.operations:
                    if getattr(op, "OPERATION_NAME", None) == "gpu.func":
                        op.attributes["rocdl.waves_per_eu"] = (
                            ir.IntegerAttr.get(T.i32, _wpe)
                        )

        flat_wg_attr = ir.StringAttr.get(f"{BLOCK_SIZE},{BLOCK_SIZE}")
        passthrough_entries = [
            ir.ArrayAttr.get([
                ir.StringAttr.get("denormal-fp-math-f32"),
                ir.StringAttr.get("preserve-sign,preserve-sign"),
            ]),
            ir.ArrayAttr.get([
                ir.StringAttr.get("no-nans-fp-math"),
                ir.StringAttr.get("true"),
            ]),
            ir.ArrayAttr.get([
                ir.StringAttr.get("unsafe-fp-math"),
                ir.StringAttr.get("true"),
            ]),
        ]
        for op in ctx.gpu_module_body.operations:
            if getattr(op, "OPERATION_NAME", None) == "gpu.func":
                op.attributes["rocdl.flat_work_group_size"] = flat_wg_attr
                op.attributes["passthrough"] = ir.ArrayAttr.get(
                    passthrough_entries)

        launcher.launch(
            grid=(grid_x, 1, 1),
            block=(BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    _compile_hints = {
        "fast_fp_math": True,
        "unsafe_fp_math": True,
        "llvm_options": {
            "enable-post-misched": True,
            "lsr-drop-solution": True,
            "amdgpu-early-inline-all": True,
            "misched-postra-direction": 2,
        },
    }

    def _launch(*args, **kwargs):
        with CompilationContext.compile_hints(_compile_hints):
            return launch_sage_attn(*args, **kwargs)

    def _compile(Q, K_tensor, V, O, descale_q, descale_k, descale_v,
                 batch_size, seq_len, stream=None):
        with CompilationContext.compile_hints(_compile_hints):
            return flyc.compile(
                launch_sage_attn, Q, K_tensor, V, O,
                descale_q, descale_k, descale_v,
                batch_size, seq_len, fx.Stream(stream),
            )

    _launch.compile = _compile
    return _launch


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------

_module_cache = {}

def sage_attention(
    Q,           # [B, S, H, D] INT8
    K,           # [B, S, H, D] INT8
    V,           # [B, S, H, D] FP8
    descale_q,   # [B, H, num_q_tiles] FP32
    descale_k,   # [B, H, num_k_tiles] FP32
    descale_v,   # [B, H] FP32
    sm_scale=None,
    **kwargs,
):
    """High-level API: run SageAttention.

    All tensors must be contiguous, flattened BSHD layout.
    Returns O in BF16 with same shape as Q (but BF16).
    """
    import torch

    B, S, H, D = Q.shape
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)
    O = torch.empty(B, S, H, D, dtype=torch.bfloat16, device=Q.device)

    # Flatten to 1D
    Q_flat = Q.contiguous().view(-1)
    K_flat = K.contiguous().view(-1)
    # V must be pre-transposed to [B, H, D, S] layout by the caller.
    # If V is [B, S, H, D], transpose it here (one-time cost).
    if V.shape == (B, S, H, D):
        V = V.permute(0, 2, 3, 1).contiguous()  # [B, H, D, S]
    V_flat = V.contiguous().view(-1)
    O_flat = O.view(-1)

    # Cache built modules to avoid recompilation
    cache_key = (H, D, sm_scale)
    if cache_key not in _module_cache:
        _module_cache[cache_key] = build_sage_attn_module(
            num_heads=H, head_dim=D, sm_scale=sm_scale,
        )
    launcher = _module_cache[cache_key]
    launcher(
        Q_flat, K_flat, V_flat, O_flat,
        descale_q.contiguous(), descale_k.contiguous(), descale_v.contiguous(),
        B, S,
    )
    return O
