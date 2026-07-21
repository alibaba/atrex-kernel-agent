# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""
⚠ TUNED FOR MI308X (gfx942, CDNA3) — stock LLVM (ROCm 6.4+).

Flash Attention forward (causal, prefill) bf16 kernel, optimized for MI308X
with stock LLVM. V11 (v0→v18 optimization journey).

Verified performance (bf16, causal, H=32, D=128, stock LLVM):
  B1S4096:  87.4 TFLOPS (42.4% of 206T peak)
  B1S8192: 102.9 TFLOPS
  B1S16384: 109.7 TFLOPS (53.3% peak — stock LLVM ceiling)
  B2S4096:  93.6 TFLOPS
  B2S8192: 104.4 TFLOPS

vs generic (architecture-agnostic) baseline (`amd/cdna/flydsl/FlyDSL/flash_attn_func.py`):
  - K_PAD=4 bank-conflict-free K LDS (generic uses K_PAD=0)
  - ds_swizzle XOR-N V transpose (generic uses scalar transpose)
  - rocdl.exp2 single-cycle softmax (generic uses arith.exp2)
  - Pre-loaded V + pure MFMA PV loop (generic interleaves V reads)
  - v_perm_b32 bf16 pack (generic uses manual bit ops)
  - Inter-block K prefetch (hidden behind softmax)
  - Split exp2/sum softmax pipeline (breaks serial dependency)
  - LLVM flags: enable-post-misched, amdgpu-early-inline-all, lsr-drop-solution
  - waves_per_eu=3, ds_swizzle has_side_effects=False latency hiding

Optimization report: docs/amd/cdna3/mi308x/ref-docs/flydsl/cdna3-flash-attention-bf16-gqa-optimization.md
Pitfalls:           docs/amd/cdna3/mi308x/pitfalls/flydsl/flash-attn-pitfalls.md

Architecture:
- True MFMA32 remap: `mfma_f32_32x32x16bf16` / `mfma_f32_32x32x16f16` for both GEMM stages.
- Tile shape: BLOCK_M=128 or 256 (auto-selected), BLOCK_N=64.
- BLOCK_M=128: 4 waves (256 threads), BLOCK_M=256: 8 waves (512 threads).
- Per-wave Q rows: 32.
- GEMM1 uses `K @ Q^T` so S/P live in MFMA32 register layout.
- Online softmax over KV dimension is done in registers.
- P is kept in registers and fed directly to GEMM2 (`V^T @ P`) without LDS roundtrip.
- K and V use separate LDS regions with K_PAD=4 padding.
- For H>=32, both M=128 and M=256 variants are built and dispatched at runtime.

Layout: Q/K/V/O are 1D flattened from BSHD (batch, seq_len, num_heads, head_dim).
Grid:   (batch * num_q_tiles * num_heads,) where num_q_tiles = seq_len / BLOCK_M.
Block:  (256,) or (512,) depending on BLOCK_M.

Requires: head_dim % 32 == 0, head_dim >= 64, seq_len % 128 == 0.
"""

import math
import os

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, buffer_ops, gpu, range_constexpr, rocdl, vector
from flydsl.expr.typing import T
from kernels.kernels_common import dtype_to_elem_type
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from flydsl._mlir import ir
from flydsl._mlir.dialects import memref as _memref, scf, fly as _fly, llvm as _llvm, math as math_dialect

# ---- Module-level constants ----

KERNEL_NAME = "flash_attn_func_kernel"

_LOG2E = math.log2(math.e)  # 1.4426950408889634

_LLVM_GEP_DYNAMIC = -2147483648  # LLVM kDynamicIndex sentinel (0x80000000 as signed i32)

def _llvm_ptr_ty():
    return ir.Type.parse("!llvm.ptr")


def _llvm_lds_ptr_ty():
    return ir.Type.parse("!llvm.ptr<3>")

_VMCNT_LO_MASK = 0xF
_LGKMCNT_EXPCNT_BASE = 0x3F70
_VMCNT_HI_SHIFT = 14
_VMCNT_HI_MASK = 0x3


def _waitcnt_vm_n(n):
    """Emit s_waitcnt vmcnt(n) only (lgkmcnt=63, expcnt=7)."""
    val = (n & _VMCNT_LO_MASK) | _LGKMCNT_EXPCNT_BASE | (((n >> 4) & _VMCNT_HI_MASK) << _VMCNT_HI_SHIFT)
    rocdl.s_waitcnt(val)


def build_flash_attn_func_module_primary(
    num_heads,
    head_dim,
    causal=True,
    dtype_str="f16",
    sm_scale=None,
    waves_per_eu=None,
    flat_work_group_size=None,
    block_m=None,
    unsafe_fp_math=True,
    fast_fp_math=True,
    daz=True,
    path_tag="auto",
):
    """Build the flash_attn_func launcher using the post-refactor FlyDSL API."""
    gpu_arch = get_hip_arch()

    BLOCK_N = 64
    K_SUB_N = 32
    WARP_SIZE = 64

    # Auto tile selection: for H>=32, build both M=128 and M=256 variants
    # and dispatch at runtime based on B*S.
    if block_m is None and num_heads >= 32:
        _launcher_m128 = build_flash_attn_func_module_primary(
            num_heads, head_dim, causal, dtype_str, sm_scale, waves_per_eu,
            flat_work_group_size=256, block_m=128,
            unsafe_fp_math=unsafe_fp_math, fast_fp_math=fast_fp_math,
            daz=daz, path_tag=path_tag)
        _launcher_m256 = build_flash_attn_func_module_primary(
            num_heads, head_dim, causal, dtype_str, sm_scale, waves_per_eu,
            flat_work_group_size=512, block_m=256,
            unsafe_fp_math=unsafe_fp_math, fast_fp_math=fast_fp_math,
            daz=daz, path_tag=path_tag)
        _BS_THRESHOLD = 4096 * num_heads

        def _auto_launch(*args, **kwargs):
            B = args[4] if len(args) > 4 else kwargs.get('batch_size', 1)
            S = args[5] if len(args) > 5 else kwargs.get('seq_len', 128)
            bs = (B if isinstance(B, int) else 1) * (S if isinstance(S, int) else 128)
            if bs * num_heads >= _BS_THRESHOLD:
                return _launcher_m256(*args, **kwargs)
            return _launcher_m128(*args, **kwargs)

        return _auto_launch

    if block_m is not None:
        BLOCK_M = block_m
    else:
        BLOCK_M = 128

    if flat_work_group_size is None:
        if BLOCK_M <= 128:
            flat_work_group_size = 256
        else:
            flat_work_group_size = 512
    NUM_WAVES = flat_work_group_size // WARP_SIZE
    BLOCK_SIZE = flat_work_group_size
    ROWS_PER_WAVE = BLOCK_M // NUM_WAVES
    if path_tag.upper() in ("N32", "N128"):
        PATH_TAG = path_tag.upper()
    elif dtype_str in ("f16", "bf16") and causal and head_dim == 128:
        PATH_TAG = "N128"
    else:
        PATH_TAG = "N32"
    BLOCK_N_OUT = 128 if PATH_TAG == "N128" else BLOCK_N
    N_SUBTILES = BLOCK_N_OUT // BLOCK_N
    ENABLE_PREFETCH_3BUF = (
        os.getenv("FLYDSL_FLASH_ATTN_FUNC_ENABLE_PREFETCH3", "0") == "1"
    )
    # buffer_load_dwordx4_lds (16B DMA-to-LDS) requires gfx950+; gfx94x only has dword (4B).
    _has_lds_load_b128 = not gpu_arch.startswith("gfx942")
    ENABLE_DMA = _has_lds_load_b128 and (
        PATH_TAG == "N128" or (
            os.getenv("FLYDSL_FLASH_ATTN_FUNC_ENABLE_DMA", "0") == "1"
        )
    )
    ENABLE_LDS_VEC16 = (
        os.getenv("FLYDSL_FLASH_ATTN_FUNC_ENABLE_LDS_VEC16", "1") == "1"
    )
    REDUCE_MODE = os.getenv("FLYDSL_FLASH_ATTN_FUNC_REDUCE_MODE", "xor").strip().lower()
    if REDUCE_MODE not in ("xor", "ds_bpermute"):
        REDUCE_MODE = "xor"
    # V9.7: K double-buffer for non-DMA path (gfx942). Overlaps K[sub1]
    # global load with sub0's softmax+GEMM2. Requires N_SUBTILES >= 2.
    ENABLE_K_DBUF = (
        not ENABLE_DMA and not ENABLE_PREFETCH_3BUF and N_SUBTILES >= 2 and
        os.getenv("FLYDSL_FA_K_DBUF", "1") == "1"
    )
    NUM_PREFETCH_K = 3 if ENABLE_PREFETCH_3BUF else (2 if (ENABLE_DMA or ENABLE_K_DBUF) else 1)
    NUM_PREFETCH_V = 3 if ENABLE_PREFETCH_3BUF else 1
    CK_LDS_SEQ = (1, 2, 0, 1, 0, 1, 2, 0) if ENABLE_PREFETCH_3BUF else (0,)

    ENABLE_SCHED_HINTS = os.getenv("FLYDSL_FA_SCHED_HINTS", "1") == "1"
    _SCHED_DSRD_QK = int(os.getenv("FLYDSL_FA_SCHED_DSRD_QK", "2"))
    _SCHED_MFMA_QK = int(os.getenv("FLYDSL_FA_SCHED_MFMA_QK", "2"))
    _SCHED_MFMA_PV = int(os.getenv("FLYDSL_FA_SCHED_MFMA_PV", "4"))

    # V9.1: q_tile pair-zigzag remap for causal load balance.
    # Causal attention: q_tile_idx i attends to (i+1) KV blocks → workload
    # grows linearly with q_tile_idx. With head_fast decomposition, early
    # WGs all have q_tile=0 (lightest) and late WGs all have q_tile=N-1
    # (heaviest). Pair-zigzag (i → (i//2) if even else (N-1-i//2))
    # interleaves light/heavy WGs across the time-wise WG dispatch.
    # ONLY enabled for BLOCK_M=256 path — at BLOCK_M=128 the kernel has a
    # pre-existing race condition that zigzag exposes as wild garbage values.
    NUM_SES = 16
    _ZIGZAG_DEFAULT = "zigzag" if (causal and (block_m is None or block_m == 256)) else "linear"
    BLOCK_ORDER = os.getenv(
        "FLYDSL_FA_BLOCK_ORDER", _ZIGZAG_DEFAULT).strip().lower()
    if BLOCK_ORDER not in ("linear", "zigzag"):
        BLOCK_ORDER = "linear"
    USE_ZIGZAG = (BLOCK_ORDER == "zigzag") and causal and (block_m is None or block_m == 256)

    # V9.3: v_pk_mul_f32 (VOP3P) for O rescale — half the VALU count.
    # Env knob: FLYDSL_FA_PK_MUL=1 (default on for gfx942 bf16) to enable.
    USE_PK_MUL = os.getenv("FLYDSL_FA_PK_MUL", "1") == "1"

    ENABLE_PERSISTENT = os.getenv("FLYDSL_FA_PERSISTENT", "0") == "1"
    PERSISTENT_NUM_CUS = int(os.getenv("FLYDSL_FA_NUM_CUS", "80"))
    if ENABLE_PERSISTENT:
        USE_ZIGZAG = False

    ENABLE_K_INTERBLOCK = ENABLE_K_DBUF and os.getenv("FLYDSL_FA_K_INTERBLOCK", "1") == "1"

    # gfx950+ has ds_read_tr16_b64 (HW transpose LDS read); gfx942 needs V^T stored in LDS.
    USE_HW_TR = gpu_arch.startswith("gfx950")

    ENABLE_XSUB_PIPELINE = os.getenv("FLYDSL_FA_XSUB_PIPELINE", "0") == "1"
    ENABLE_PV_PIPELINE = os.getenv("FLYDSL_FA_PV_PIPELINE", "0") == "1"
    _PV_DEPTH = int(os.getenv("FLYDSL_FA_PV_DEPTH", "2"))
    ENABLE_P_PACK_EARLY = os.getenv("FLYDSL_FA_P_PACK_EARLY", "0") == "1"
    ENABLE_SGB_INTERLEAVE = os.getenv("FLYDSL_FA_SGB_INTERLEAVE", "0") == "1"

    # MFMA32 K-dimension: 16 on gfx950+ (CDNA4) for both GEMMs.
    USE_K16 = gpu_arch.startswith("gfx950")
    K_STEP_QK = 16 if USE_K16 else 8
    K_STEPS_QK = head_dim // K_STEP_QK
    D_CHUNK = 32
    D_CHUNKS = head_dim // D_CHUNK
    PV_K_STEP = 16 if USE_K16 else 8
    PV_K_STEPS = K_SUB_N // PV_K_STEP  # 2 steps per sub-tile (K=16) or 4 (K=8)

    assert BLOCK_M % NUM_WAVES == 0
    assert head_dim % 32 == 0, f"head_dim ({head_dim}) must be divisible by 32"
    assert head_dim >= 64, f"head_dim ({head_dim}) must be >= 64"
    assert flat_work_group_size in (128, 256, 512), (
        f"flat_work_group_size must be 128, 256, or 512, got {flat_work_group_size}"
    )
    assert dtype_str in ("f16", "bf16"), "flash_attn_func only supports f16 and bf16"
    assert BLOCK_N % 32 == 0
    assert BLOCK_N_OUT % BLOCK_N == 0

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    NUM_HEADS = num_heads
    HEAD_DIM = head_dim
    CAUSAL = causal
    STRIDE_TOKEN = NUM_HEADS * HEAD_DIM

    ENABLE_K_XOR_SWIZZLE = os.getenv("FLYDSL_FA_K_XOR_SWIZZLE", "0") == "1"
    K_PAD = 0 if ENABLE_K_XOR_SWIZZLE else 4
    K_STRIDE = HEAD_DIM + K_PAD
    if USE_HW_TR:
        V_STRIDE = HEAD_DIM if ENABLE_DMA else HEAD_DIM + 4
    else:
        VT_STRIDE = BLOCK_N + 2
        V_STRIDE = VT_STRIDE

    # Vectorized cooperative load constants.
    VEC_WIDTH = 16 if ENABLE_LDS_VEC16 else 8
    assert HEAD_DIM % VEC_WIDTH == 0
    THREADS_PER_ROW_LOAD = HEAD_DIM // VEC_WIDTH
    assert BLOCK_SIZE % THREADS_PER_ROW_LOAD == 0
    ROWS_PER_BATCH_LOAD = BLOCK_SIZE // THREADS_PER_ROW_LOAD

    if ROWS_PER_BATCH_LOAD >= BLOCK_N:
        NUM_BATCHES_KV = 1
        KV_NEEDS_GUARD = ROWS_PER_BATCH_LOAD > BLOCK_N
    else:
        assert BLOCK_N % ROWS_PER_BATCH_LOAD == 0
        NUM_BATCHES_KV = BLOCK_N // ROWS_PER_BATCH_LOAD
        KV_NEEDS_GUARD = False

    # K/V circular buffers; defaults to 1/1, optional 3/3 with CK-like LDS sequence.
    LDS_K_TILE_SIZE = BLOCK_N * K_STRIDE
    if USE_HW_TR:
        LDS_V_TILE_SIZE = BLOCK_N * V_STRIDE
    else:
        LDS_V_TILE_SIZE = HEAD_DIM * VT_STRIDE
    LDS_K_TOTAL_SIZE = NUM_PREFETCH_K * LDS_K_TILE_SIZE
    LDS_V_BASE = LDS_K_TOTAL_SIZE
    LDS_V_TOTAL_SIZE = NUM_PREFETCH_V * LDS_V_TILE_SIZE
    LDS_KV_TOTAL_SIZE = LDS_K_TOTAL_SIZE + LDS_V_TOTAL_SIZE

    allocator = SmemAllocator(
        None,
        arch=gpu_arch,
        global_sym_name=f"flash_attn_func_smem_{PATH_TAG}",
    )
    lds_kv_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_kv_offset + LDS_KV_TOTAL_SIZE * 2

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def flash_attn_func_kernel(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        O: fx.Tensor,
        seq_len: fx.Int32,
        batch_size: fx.Int32,
    ):
        elem_type = dtype_to_elem_type(dtype_str)
        compute_type = T.f32
        q_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), Q)
        k_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), K)
        v_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), V)
        o_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), O)

        # All FP operations use aggressive fast-math (no NaN/Inf checks, reassociation).
        # The unsafe_fp_math/fast_fp_math builder params control LLVM-level attributes only.
        fm_fast = arith.FastMathFlags.fast
        v4f16_type = T.vec(4, elem_type)
        vxf16_type = T.vec(VEC_WIDTH, elem_type)
        v8f16_type = T.vec(8, elem_type)
        v16f32_type = T.vec(16, compute_type)
        mfma_pack_type = v8f16_type if USE_K16 else v4f16_type
        MFMA_LANE_K = 8 if USE_K16 else 4
        _mfma_zero = ir.IntegerAttr.get(ir.IntegerType.get_signless(32), 0)
        def _mfma(ods_fn, a, b, c):
            return ods_fn(v16f32_type, a, b, c, _mfma_zero, _mfma_zero, _mfma_zero).result
        def mfma_acc(a, b, c):
            if dtype_str == "bf16":
                if USE_K16:
                    return _mfma(rocdl.mfma_f32_32x32x16_bf16, a, b, c)
                a = vector.bitcast(T.i16x4, a)
                b = vector.bitcast(T.i16x4, b)
                return _mfma(rocdl.mfma_f32_32x32x8bf16_1k, a, b, c)
            if USE_K16:
                return _mfma(rocdl.mfma_f32_32x32x16_f16, a, b, c)
            return _mfma(rocdl.mfma_f32_32x32x8f16, a, b, c)

        seq_len_v = arith.index_cast(T.index, seq_len)

        # ---- LDS view ----
        base_ptr = allocator.get_base()
        lds_kv = SmemPtr(
            base_ptr,
            lds_kv_offset,
            elem_type,
            shape=(LDS_KV_TOTAL_SIZE,),
        ).get()

        # ---- Thread / block indices ----
        block_id = arith.index_cast(T.index, gpu.block_idx.x)
        tid = arith.index_cast(T.index, gpu.thread_idx.x)

        # ---- Wave decomposition ----
        wave_id = tid // WARP_SIZE
        lane = tid % WARP_SIZE
        lane_mod_32 = lane % 32
        lane_div_32 = lane // 32  # 0/1

        # ---- ds_read_b64_tr_b16 lane decomposition ----
        # Hardware does 4×4 transpose within blocks of 16 lanes.
        # tr_k_group selects which of 4 K-rows within the block,
        # tr_col_sub selects which 4-column sub-group within 16 columns.
        tr_k_group = (lane % 16) // 4   # 0..3: K-row offset within 4-row group
        tr_col_sub = lane % 4            # 0..3: 4-column sub-group
        tr_col_half = (lane % 32) // 16  # 0 or 1: first/second 16-column half

        # ---- ds_read_b64_tr_b16 helper ----

        def ds_read_tr_v4f16(lds_elem_idx):
            """Read v4f16 from LDS with hardware transpose.

            Within each block of 16 lanes, the hardware performs a 4×4
            transpose across 4 groups of 4 lanes.  After the transpose,
            result[lane, elem_e] = Input[source_lane, lane%4] where
            source_lane = e*4 + (lane%16)//4.  This naturally produces
            the MFMA A-operand layout when per-lane addresses point to
            the correct K-row and D-column sub-group.
            """
            byte_offset = lds_elem_idx * 2 + lds_kv_offset
            byte_i64 = arith.index_cast(T.i64, byte_offset)
            ptr = _llvm.IntToPtrOp(_llvm_lds_ptr_ty(), byte_i64).result
            return rocdl.ds_read_tr16_b64(v4f16_type, ptr).result

        # ---- Wave offsets ----
        wave_q_offset = wave_id * ROWS_PER_WAVE

        # ---- Persistent grid-stride loop (optional) ----
        if ENABLE_PERSISTENT:
            _p_nqt = (seq_len_v + BLOCK_M - 1) // BLOCK_M
            _p_bs = arith.index_cast(T.index, batch_size)
            _p_total = _p_bs * _p_nqt * NUM_HEADS
            _persistent_for = scf.ForOp(
                block_id, _p_total, arith.index(PERSISTENT_NUM_CUS))
            _persistent_body = _persistent_for.regions[0].blocks[0]
            _persistent_ip = ir.InsertionPoint(_persistent_body)
            _persistent_ip.__enter__()
            effective_tile_id = _persistent_body.arguments[0]
        else:
            effective_tile_id = block_id

        # ---- Decompose effective_tile_id ----
        head_idx = effective_tile_id % NUM_HEADS
        batch_q_tile_id = effective_tile_id // NUM_HEADS
        num_q_tiles = (seq_len_v + BLOCK_M - 1) // BLOCK_M
        q_tile_raw = batch_q_tile_id % num_q_tiles
        batch_idx = batch_q_tile_id // num_q_tiles
        if USE_ZIGZAG:
            q_is_odd = q_tile_raw % arith.index(2)
            q_half = q_tile_raw // arith.index(2)
            q_tile_idx = arith.select(
                arith.cmpi(arith.CmpIPredicate.eq, q_is_odd,
                           arith.index(0)),
                q_half,
                num_q_tiles - arith.index(1) - q_half)
        else:
            q_tile_idx = q_tile_raw
        q_start = q_tile_idx * BLOCK_M

        # ---- Cooperative load decomposition ----
        load_row_in_batch = tid // THREADS_PER_ROW_LOAD
        load_lane_in_row = tid % THREADS_PER_ROW_LOAD
        load_col_base = load_lane_in_row * VEC_WIDTH

        # ---- Helper: global flat index ----
        def global_idx(token_idx, col):
            token = batch_idx * seq_len_v + token_idx
            return token * STRIDE_TOKEN + head_idx * HEAD_DIM + col

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

        def load_global_f16x4(base_ptr, base_idx):
            return _gep_load(base_ptr, base_idx, v4f16_type)

        def load_global_mfma_pack(base_ptr, base_idx):
            return _gep_load(base_ptr, base_idx, mfma_pack_type)

        def load_global_f16xN(base_ptr, base_idx):
            return _gep_load(base_ptr, base_idx, vxf16_type)

        def bf16_trunc_pack_v4(f32_vals):
            """Pack 4 f32 values into v4bf16 via bitwise truncation (upper 16 bits)."""
            _v2i32 = T.vec(2, T.i32)
            a0 = arith.ArithValue(f32_vals[0]).bitcast(T.i32)
            b0 = arith.ArithValue(f32_vals[1]).bitcast(T.i32)
            a1 = arith.ArithValue(f32_vals[2]).bitcast(T.i32)
            b1 = arith.ArithValue(f32_vals[3]).bitcast(T.i32)
            if USE_VPERM_PACK:
                p0 = _v_perm_b32(b0, a0)
                p1 = _v_perm_b32(b1, a1)
            else:
                _c16 = arith.constant(16, type=T.i32)
                _cmask = arith.constant(0xFFFF0000, type=T.i32)
                p0 = arith.OrIOp(arith.AndIOp(b0, _cmask).result,
                                 arith.ShRUIOp(a0, _c16).result).result
                p1 = arith.OrIOp(arith.AndIOp(b1, _cmask).result,
                                 arith.ShRUIOp(a1, _c16).result).result
            return vector.bitcast(v4f16_type, vector.from_elements(_v2i32, [p0, p1]))

        def bf16_trunc_pack_v8(f32_vals):
            """Pack 8 f32 values into v8bf16 via bitwise truncation (upper 16 bits)."""
            _v4i32 = T.vec(4, T.i32)
            pairs = []
            for j in range_constexpr(4):
                a = arith.ArithValue(f32_vals[j * 2]).bitcast(T.i32)
                b = arith.ArithValue(f32_vals[j * 2 + 1]).bitcast(T.i32)
                if USE_VPERM_PACK:
                    p = _v_perm_b32(b, a)
                else:
                    _c16 = arith.constant(16, type=T.i32)
                    _cmask = arith.constant(0xFFFF0000, type=T.i32)
                    p = arith.OrIOp(arith.AndIOp(b, _cmask).result,
                                    arith.ShRUIOp(a, _c16).result).result
                pairs.append(p)
            return vector.bitcast(v8f16_type, vector.from_elements(_v4i32, pairs))

        def k_buf_base(buf_id):
            if isinstance(buf_id, int):
                return arith.index(buf_id * LDS_K_TILE_SIZE)
            return buf_id * arith.index(LDS_K_TILE_SIZE)

        def v_buf_base(buf_id):
            return arith.index(LDS_V_BASE + buf_id * LDS_V_TILE_SIZE)

        def _k_swizzle(row_idx, col_idx):
            if ENABLE_K_XOR_SWIZZLE:
                return arith.XOrIOp(col_idx, arith.ShLIOp(
                    arith.AndIOp(row_idx, arith.index(7)).result,
                    arith.index(4)).result).result
            return col_idx

        # ---- Cooperative K load (row-major, XOR-swizzled) ----
        def coop_load_k(tile_start, buf_id=0):
            k_base = k_buf_base(buf_id)
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
                        g_idx = global_idx(row_idx, load_col_base)
                        lds_row = load_row_in_batch + row_offset
                        swz_col = _k_swizzle(lds_row, load_col_base)
                        lds_idx = k_base + lds_row * K_STRIDE + swz_col
                        vec = load_global_f16xN(k_ptr, g_idx)
                        vector.store(vec, lds_kv, [lds_idx])
                        scf.YieldOp([])
                else:
                    g_idx = global_idx(row_idx, load_col_base)
                    lds_row = load_row_in_batch + row_offset
                    swz_col = _k_swizzle(lds_row, load_col_base)
                    lds_idx = k_base + lds_row * K_STRIDE + swz_col
                    vec = load_global_f16xN(k_ptr, g_idx)
                    vector.store(vec, lds_kv, [lds_idx])

        def coop_load_k_global(tile_start):
            """Issue global loads for K, return vectors (non-blocking)."""
            vecs = []
            for batch in range_constexpr(NUM_BATCHES_KV):
                row_offset = batch * ROWS_PER_BATCH_LOAD
                row_idx = tile_start + load_row_in_batch + row_offset
                g_idx = global_idx(row_idx, load_col_base)
                vecs.append(load_global_f16xN(k_ptr, g_idx))
            return vecs

        def coop_store_k_lds(vecs, buf_id=0):
            """Write previously-loaded K vectors to LDS."""
            k_base = k_buf_base(buf_id)
            for batch in range_constexpr(NUM_BATCHES_KV):
                row_offset = batch * ROWS_PER_BATCH_LOAD
                lds_row = load_row_in_batch + row_offset
                swz_col = _k_swizzle(lds_row, load_col_base)
                lds_idx = k_base + lds_row * K_STRIDE + swz_col
                vector.store(vecs[batch], lds_kv, [lds_idx])

        # ---- Cooperative V load ----
        def _v_store_row_major(v_base, lds_row, vec):
            lds_idx = v_base + lds_row * V_STRIDE + load_col_base
            vector.store(vec, lds_kv, [lds_idx])

        _v1_type = T.vec(1, elem_type) if not USE_HW_TR else None
        _v2_type = T.vec(2, elem_type) if not USE_HW_TR else None

        def _ds_swizzle_xor16_inline(src):
            return _llvm.inline_asm(
                ir.IntegerType.get_signless(32),
                [src],
                "s_waitcnt lgkmcnt(0)\n\tds_swizzle_b32 $0, $1 offset:0x401F",
                "=v,v",
                has_side_effects=True,
            )

        def _ds_swizzle_xor8_inline(src):
            return _llvm.inline_asm(
                ir.IntegerType.get_signless(32),
                [src],
                "s_waitcnt lgkmcnt(0)\n\tds_swizzle_b32 $0, $1 offset:0x201F",
                "=v,v",
                has_side_effects=True,
            )

        USE_VPERM_PACK = os.getenv("FLYDSL_FA_VPERM_PACK", "1") == "1"
        _perm_selector = arith.constant(0x07060302, type=T.i32) if USE_VPERM_PACK else None

        def _v_perm_b32(src0, src1):
            """v_perm_b32 dst, src0, src1, sel — byte permute with SGPR selector.
            With sel=0x07060302: dst = {src0[31:16], src1[31:16]}."""
            return _llvm.inline_asm(
                T.i32,
                [src0, src1, _perm_selector],
                "v_perm_b32 $0, $1, $2, $3",
                "=v,v,v,s",
                has_side_effects=False,
            )

        # V9.3: v_pk_mul_f32 for O rescale (CDNA3 VOP3P).
        # Halves the per-rescale VALU instruction count: 16 v_mul_f32 → 8
        # v_pk_mul_f32 per v16f32 accumulator. Operates on 2-DWORD aligned
        # pairs (the MFMA C-operand layout is naturally even-aligned in VGPR
        # banks, so this is safe).
        _v2f32_type = T.vec(2, compute_type)

        def _v_pk_mul_f32_pair_inline(pair_a, pair_b):
            """v_pk_mul_f32 vdst, vsrc0, vsrc1 — 2× FP32 mul packed (CDNA3)."""
            return _llvm.inline_asm(
                _v2f32_type,
                [pair_a, pair_b],
                "v_pk_mul_f32 $0, $1, $2",
                "=v,v,v",
                has_side_effects=False,
            )

        def _pk_mul_f32_v16(acc_v16f32, scale_scalar):
            """Multiply v16f32 by broadcast scalar via 8 v_pk_mul_f32 ops."""
            scale_pair = vector.from_elements(
                _v2f32_type, [scale_scalar, scale_scalar])
            result_elems = []
            for i in range_constexpr(8):
                a = vector.extract(
                    acc_v16f32, static_position=[2 * i], dynamic_position=[])
                b = vector.extract(
                    acc_v16f32, static_position=[2 * i + 1], dynamic_position=[])
                pair_in = vector.from_elements(_v2f32_type, [a, b])
                pair_out = _v_pk_mul_f32_pair_inline(pair_in, scale_pair)
                r0 = vector.extract(
                    pair_out, static_position=[0], dynamic_position=[])
                r1 = vector.extract(
                    pair_out, static_position=[1], dynamic_position=[])
                result_elems.extend([r0, r1])
            return vector.from_elements(v16f32_type, result_elems)

        def _v_store_transposed(v_base, lds_row, vec):
            for _e in range_constexpr(VEC_WIDTH):
                elem = vector.extract(vec, static_position=[_e], dynamic_position=[])
                vt_d = load_col_base + _e
                vt_idx = v_base + vt_d * VT_STRIDE + lds_row
                v1 = vector.from_elements(_v1_type, [elem])
                vector.store(v1, lds_kv, [vt_idx])

        def _v_store_transposed_perm(v_base, lds_row, vec):
            """V transpose via ds_swizzle XOR-N + paired ds_write_b32.
            Halves LDS write count vs scalar transpose (~22-24% kernel speedup).
            """
            num_dwords = VEC_WIDTH // 2
            _vN_i32_type = T.vec(num_dwords, T.i32)
            own_vNi32 = vector.bitcast(_vN_i32_type, vec)

            if THREADS_PER_ROW_LOAD == 16:
                _swz_inline = _ds_swizzle_xor16_inline
            elif THREADS_PER_ROW_LOAD == 8:
                _swz_inline = _ds_swizzle_xor8_inline
            else:
                raise NotImplementedError(
                    f"THREADS_PER_ROW_LOAD={THREADS_PER_ROW_LOAD} not supported")

            peer_dwords = []
            for k in range_constexpr(num_dwords):
                own_dw = vector.extract(
                    own_vNi32, static_position=[k], dynamic_position=[])
                peer_dw = _swz_inline(own_dw)
                peer_dwords.append(peer_dw)

            row_lo_bit = arith.AndIOp(
                arith.index_cast(T.i32, lds_row),
                arith.constant(1, type=T.i32)).result
            is_even = arith.cmpi(
                arith.CmpIPredicate.eq, row_lo_bit,
                arith.constant(0, type=T.i32))

            peer_vNi32 = vector.from_elements(_vN_i32_type, peer_dwords)
            peer_vec = vector.bitcast(vxf16_type, peer_vNi32)

            _if_writer = scf.IfOp(is_even)
            with ir.InsertionPoint(_if_writer.then_block):
                for _e in range_constexpr(VEC_WIDTH):
                    own_elem = vector.extract(
                        vec, static_position=[_e], dynamic_position=[])
                    peer_elem = vector.extract(
                        peer_vec, static_position=[_e], dynamic_position=[])
                    pair = vector.from_elements(
                        _v2_type, [own_elem, peer_elem])
                    vt_d = load_col_base + _e
                    vt_idx = v_base + vt_d * VT_STRIDE + lds_row
                    vector.store(pair, lds_kv, [vt_idx])
                scf.YieldOp([])

        if USE_HW_TR:
            _v_store_to_lds = _v_store_row_major
        else:
            _v_store_to_lds = _v_store_transposed_perm

        def coop_load_v(tile_start, buf_id=0):
            v_base = v_buf_base(buf_id)
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
                        g_idx = global_idx(row_idx, load_col_base)
                        lds_row = load_row_in_batch + row_offset
                        vec = load_global_f16xN(v_ptr, g_idx)
                        _v_store_to_lds(v_base, lds_row, vec)
                        scf.YieldOp([])
                else:
                    g_idx = global_idx(row_idx, load_col_base)
                    lds_row = load_row_in_batch + row_offset
                    vec = load_global_f16xN(v_ptr, g_idx)
                    _v_store_to_lds(v_base, lds_row, vec)

        def coop_load_v_global(tile_start):
            """Issue global loads for V, return vectors (non-blocking)."""
            vecs = []
            for batch in range_constexpr(NUM_BATCHES_KV):
                row_offset = batch * ROWS_PER_BATCH_LOAD
                row_idx = tile_start + load_row_in_batch + row_offset
                g_idx = global_idx(row_idx, load_col_base)
                vecs.append(load_global_f16xN(v_ptr, g_idx))
            return vecs

        def coop_store_v_lds(vecs, buf_id=0):
            """Write previously-loaded V vectors to LDS."""
            v_base = v_buf_base(buf_id)
            for batch in range_constexpr(NUM_BATCHES_KV):
                row_offset = batch * ROWS_PER_BATCH_LOAD
                if KV_NEEDS_GUARD:
                    row_valid = arith.cmpi(
                        arith.CmpIPredicate.ult,
                        load_row_in_batch,
                        arith.index(BLOCK_N),
                    )
                    _if_v = scf.IfOp(row_valid)
                    with ir.InsertionPoint(_if_v.then_block):
                        lds_row = load_row_in_batch + row_offset
                        _v_store_to_lds(v_base, lds_row, vecs[batch])
                        scf.YieldOp([])
                else:
                    lds_row = load_row_in_batch + row_offset
                    _v_store_to_lds(v_base, lds_row, vecs[batch])

        # ---- DMA loading for K (buffer_load_dwordx4 ... lds) ----
        if ENABLE_DMA:
            from flydsl._mlir.dialects import llvm
            k_rsrc = buffer_ops.create_buffer_resource(K, max_size=True)
            _lds_ptr_ty = _llvm_lds_ptr_ty()
            DMA_BYTES = 16  # buffer_load_dwordx4 = 16 bytes per lane
            DMA_BATCH_BYTES = BLOCK_SIZE * DMA_BYTES
            K_TILE_BYTES = BLOCK_N * K_STRIDE * 2
            NUM_DMA_K = K_TILE_BYTES // DMA_BATCH_BYTES
            LANES_PER_K_ROW = HEAD_DIM * 2 // DMA_BYTES
            ROWS_PER_DMA_BATCH = DMA_BATCH_BYTES // (HEAD_DIM * 2)
            lds_kv_base_idx = _memref.extract_aligned_pointer_as_index(lds_kv)
            _dma_size = arith.constant(DMA_BYTES, type=T.i32)
            _dma_soff = arith.constant(0, type=T.i32)
            _dma_off = arith.constant(0, type=T.i32)
            _dma_aux = arith.constant(1, type=T.i32)

            def coop_dma_k(tile_start, buf_id=0):
                """Load K tile via DMA with XOR-swizzled global fetch."""
                if isinstance(buf_id, int):
                    k_lds_byte_base = lds_kv_base_idx + arith.index(buf_id * LDS_K_TILE_SIZE * 2)
                else:
                    k_lds_byte_base = lds_kv_base_idx + buf_id * arith.index(LDS_K_TILE_SIZE * 2)
                for d in range_constexpr(NUM_DMA_K):
                    lds_addr = (k_lds_byte_base
                                + wave_id * arith.index(WARP_SIZE * DMA_BYTES)
                                + arith.index(d * DMA_BATCH_BYTES))
                    lds_i64 = arith.index_cast(T.i64, lds_addr)
                    lds_lane0 = rocdl.readfirstlane(T.i64, lds_i64)
                    lds_ptr = llvm.IntToPtrOp(_lds_ptr_ty, lds_lane0).result

                    row_in_tile = (tid // LANES_PER_K_ROW
                                   + arith.index(d * ROWS_PER_DMA_BATCH))
                    swiz_col_f16 = (tid % LANES_PER_K_ROW) * (DMA_BYTES // 2)
                    xor_mask = (row_in_tile & arith.index(0x7)) << arith.index(4)
                    unsw_col_f16 = swiz_col_f16 ^ xor_mask
                    col_byte = unsw_col_f16 * 2
                    global_row = (batch_idx * seq_len_v + tile_start
                                  + row_in_tile)
                    global_byte = (global_row * arith.index(STRIDE_TOKEN * 2)
                                   + head_idx * arith.index(HEAD_DIM * 2)
                                   + col_byte)
                    voffset = arith.index_cast(T.i32, global_byte)

                    rocdl.raw_ptr_buffer_load_lds(
                        k_rsrc, lds_ptr, _dma_size, voffset,
                        _dma_soff, _dma_off, _dma_aux,
                    )

        # ---- V XOR swizzle: col ^ ((row & 3) << 4) at 16-element granularity ----
        def _v_swizzle(row_idx, col_idx):
            mask = (row_idx & arith.index(0x3)) << arith.index(4)
            return col_idx ^ mask

        # ---- DMA loading for V (buffer_load_dwordx4 ... lds) ----
        if ENABLE_DMA:
            v_rsrc = buffer_ops.create_buffer_resource(V, max_size=True)
            V_TILE_BYTES = BLOCK_N * V_STRIDE * 2
            NUM_DMA_V = V_TILE_BYTES // DMA_BATCH_BYTES
            LANES_PER_V_ROW = HEAD_DIM * 2 // DMA_BYTES
            ROWS_PER_DMA_BATCH_V = DMA_BATCH_BYTES // (HEAD_DIM * 2)

            def coop_dma_v(tile_start, buf_id=0):
                """Load V tile via DMA with XOR-swizzled global fetch."""
                v_lds_byte_base = (lds_kv_base_idx
                                   + arith.index((LDS_V_BASE + buf_id * LDS_V_TILE_SIZE) * 2))
                for d in range_constexpr(NUM_DMA_V):
                    lds_addr = (v_lds_byte_base
                                + wave_id * arith.index(WARP_SIZE * DMA_BYTES)
                                + arith.index(d * DMA_BATCH_BYTES))
                    lds_i64 = arith.index_cast(T.i64, lds_addr)
                    lds_lane0 = rocdl.readfirstlane(T.i64, lds_i64)
                    lds_ptr = llvm.IntToPtrOp(_lds_ptr_ty, lds_lane0).result

                    row_in_tile = (tid // LANES_PER_V_ROW
                                   + arith.index(d * ROWS_PER_DMA_BATCH_V))
                    swiz_col_f16 = (tid % LANES_PER_V_ROW) * (DMA_BYTES // 2)
                    xor_mask = (row_in_tile & arith.index(0x3)) << arith.index(4)
                    unsw_col_f16 = swiz_col_f16 ^ xor_mask
                    col_byte = unsw_col_f16 * 2
                    global_row = (batch_idx * seq_len_v + tile_start
                                  + row_in_tile)
                    global_byte = (global_row * arith.index(STRIDE_TOKEN * 2)
                                   + head_idx * arith.index(HEAD_DIM * 2)
                                   + col_byte)
                    voffset = arith.index_cast(T.i32, global_byte)

                    rocdl.raw_ptr_buffer_load_lds(
                        v_rsrc, lds_ptr, _dma_size, voffset,
                        _dma_soff, _dma_off, _dma_aux,
                    )

        # ---- Preload Q^T B-operand packs once (register-resident) ----
        # B operand uses j = lane_mod_32, k-subblock = lane_div_32*MFMA_LANE_K.
        q_row = q_start + wave_q_offset + lane_mod_32
        q_row_i32 = arith.index_cast(T.i32, q_row)
        q_in_bounds = arith.cmpi(arith.CmpIPredicate.slt, q_row, seq_len_v)
        q_row_safe = arith.select(q_in_bounds, q_row, arith.index(0))
        c_zero_mfma_pack = arith.constant_vector(0.0, mfma_pack_type)
        q_b_packs = []
        for ks in range_constexpr(K_STEPS_QK):
            q_col = arith.index(ks * K_STEP_QK) + lane_div_32 * MFMA_LANE_K
            g_idx = global_idx(q_row_safe, q_col)
            raw = load_global_mfma_pack(q_ptr, g_idx)
            q_b_packs.append(arith.select(q_in_bounds, raw, c_zero_mfma_pack))

        # ---- Constants ----
        c_neg_inf = arith.constant(float("-inf"), type=compute_type)
        c_zero_f = arith.constant(0.0, type=compute_type)
        c_one_f = arith.constant(1.0, type=compute_type)
        c_sm_scale_log2e = arith.constant(sm_scale * _LOG2E, type=compute_type)
        c_zero_v16f32 = arith.constant_vector(0.0, v16f32_type)
        width_i32 = arith.constant(WARP_SIZE, type=T.i32)
        shuf_32_i32 = arith.constant(32, type=T.i32)
        c4_i32 = arith.constant(4, type=T.i32)
        lane_i32 = arith.index_cast(T.i32, lane)
        lane_xor_32_i32 = arith.XOrIOp(lane_i32, shuf_32_i32).result
        lane_xor_32_byte = arith.MulIOp(lane_xor_32_i32, c4_i32).result

        def reduction_peer(v_f32):
            if REDUCE_MODE == "ds_bpermute":
                v_i32 = arith.ArithValue(v_f32).bitcast(T.i32)
                peer_i32 = rocdl.ds_bpermute(T.i32, lane_xor_32_byte, v_i32)
                return arith.ArithValue(peer_i32).bitcast(compute_type)
            return arith.ArithValue(v_f32).shuffle_xor(shuf_32_i32, width_i32)

        # ---- KV loop upper bound ----
        _q_end = q_start + BLOCK_M
        if CAUSAL:
            kv_upper = arith.MinSIOp(_q_end, seq_len_v).result
        else:
            kv_upper = seq_len_v

        # Loop-carried: [m_old, l_old, o_acc_chunks..., (buf_id if DMA dbuf)]
        _use_dma_dbuf = ENABLE_DMA and not ENABLE_PREFETCH_3BUF
        init_args = [c_neg_inf, c_zero_f]
        for _ in range_constexpr(D_CHUNKS):
            init_args.append(c_zero_v16f32)
        if _use_dma_dbuf:
            init_args.append(arith.index(0))
            coop_dma_k(arith.index(0), buf_id=0)
        if ENABLE_K_INTERBLOCK:
            _k_interblock_init = coop_load_k_global(arith.index(0))
            init_args.extend(_k_interblock_init)

        for kv_block_start, inner_iter_args, loop_results in scf.for_(
            arith.index(0),
            kv_upper,
            arith.index(BLOCK_N_OUT),
            iter_args=init_args,
        ):
            m_running = inner_iter_args[0]
            l_running = inner_iter_args[1]
            o_accs = [
                inner_iter_args[2 + i] for i in range_constexpr(D_CHUNKS)
            ]
            _cur_buf_id = inner_iter_args[2 + D_CHUNKS] if _use_dma_dbuf else None
            if ENABLE_K_INTERBLOCK:
                _k_ib_offset = 2 + D_CHUNKS + (1 if _use_dma_dbuf else 0)
                _k_interblock_vecs = [inner_iter_args[_k_ib_offset + i] for i in range(NUM_BATCHES_KV)]
            preload_k_count = (
                NUM_PREFETCH_K if NUM_PREFETCH_K < N_SUBTILES else N_SUBTILES
            )

            if ENABLE_PREFETCH_3BUF:
                for pre_k in range_constexpr(preload_k_count):
                    pre_k_slot = CK_LDS_SEQ[pre_k % len(CK_LDS_SEQ)] % NUM_PREFETCH_K
                    pre_k_start = kv_block_start + pre_k * BLOCK_N
                    if ENABLE_DMA:
                        coop_dma_k(pre_k_start, pre_k_slot)
                    else:
                        coop_load_k(pre_k_start, pre_k_slot)
                if ENABLE_DMA:
                    rocdl.s_waitcnt(0)
                else:
                    rocdl.sched_group_barrier(rocdl.mask_vmem_rd, 1, 0)
                gpu.barrier()

            if ENABLE_XSUB_PIPELINE:
                _XP_VALU_MASK = 0x002
                _xp_kv0 = kv_block_start
                _xp_kv1 = kv_block_start + arith.index(BLOCK_N)

                # ==== Sub0: K setup → GEMM1[0] → alu0[0] → memory ====
                if ENABLE_K_INTERBLOCK:
                    _waitcnt_vm_n(0)
                    coop_store_k_lds(_k_interblock_vecs, 0)
                    gpu.barrier()
                else:
                    coop_load_k(_xp_kv0, 0)
                    gpu.barrier()
                _xp_kb0 = k_buf_base(0)
                _xp_v0 = coop_load_v_global(_xp_kv0)

                _xp_khi = K_SUB_N * K_STRIDE
                def _xp_klo(kb, ks):
                    c = arith.index(ks * K_STEP_QK) + lane_div_32 * MFMA_LANE_K
                    swz = _k_swizzle(lane_mod_32, c)
                    return kb + lane_mod_32 * K_STRIDE + swz
                def _xp_khi_fn(kb, ks):
                    c = arith.index(ks * K_STEP_QK) + lane_div_32 * MFMA_LANE_K
                    row_hi = lane_mod_32 + arith.index(K_SUB_N)
                    swz = _k_swizzle(row_hi, c)
                    return kb + _xp_khi + lane_mod_32 * K_STRIDE + swz

                _XP_PD = 2
                _kl0 = [None] * K_STEPS_QK
                _kh0 = [None] * K_STEPS_QK
                for p in range_constexpr(_XP_PD):
                    _kl0[p] = vector.load_op(mfma_pack_type, lds_kv, [_xp_klo(_xp_kb0, p)])
                    _kh0[p] = vector.load_op(mfma_pack_type, lds_kv, [_xp_khi_fn(_xp_kb0, p)])
                _sl0 = c_zero_v16f32
                _sh0 = c_zero_v16f32
                rocdl.sched_dsrd(_SCHED_DSRD_QK)
                rocdl.sched_mfma(_SCHED_MFMA_QK)
                for ks in range_constexpr(K_STEPS_QK):
                    _sl0 = mfma_acc(_kl0[ks], q_b_packs[ks], _sl0)
                    _sh0 = mfma_acc(_kh0[ks], q_b_packs[ks], _sh0)
                    if ks + _XP_PD < K_STEPS_QK:
                        _kl0[ks + _XP_PD] = vector.load_op(
                            mfma_pack_type, lds_kv, [_xp_klo(_xp_kb0, ks + _XP_PD)])
                        _kh0[ks + _XP_PD] = vector.load_op(
                            mfma_pack_type, lds_kv, [_xp_khi_fn(_xp_kb0, ks + _XP_PD)])
                rocdl.sched_barrier(0)
                _xp_kdbuf = coop_load_k_global(_xp_kv1)

                # s_raw[0] extraction
                _sr_lo0 = [vector.extract(_sl0, static_position=[r], dynamic_position=[]) for r in range(16)]
                _sr_hi0 = [vector.extract(_sh0, static_position=[r], dynamic_position=[]) for r in range(16)]
                if CAUSAL:
                    _xp_kvs0 = arith.index_cast(T.i32, _xp_kv0)
                    _xp_ld32 = arith.index_cast(T.i32, lane_div_32)
                    _xp_qs = arith.index_cast(T.i32, q_start)
                    _xp_mk0 = arith.AddIOp(_xp_kvs0, arith.constant(BLOCK_N - 1, type=T.i32)).result
                    _xp_tm0 = arith.cmpi(arith.CmpIPredicate.ugt, _xp_mk0, _xp_qs)
                    _mif0 = scf.IfOp(_xp_tm0, [T.f32] * 32, has_else=True)
                    with ir.InsertionPoint(_mif0.then_block):
                        _mlo, _mhi = [], []
                        for r in range_constexpr(16):
                            _ro = arith.constant((r % 4) + (r // 4) * 8, type=T.i32)
                            _lo = arith.MulIOp(_xp_ld32, arith.constant(4, type=T.i32)).result
                            _kc = arith.AddIOp(arith.AddIOp(_xp_kvs0, _lo).result, _ro).result
                            _mlo.append(arith.select(
                                arith.cmpi(arith.CmpIPredicate.ugt, _kc, q_row_i32),
                                c_neg_inf, _sr_lo0[r]))
                            _kch = arith.AddIOp(_kc, arith.constant(K_SUB_N, type=T.i32)).result
                            _mhi.append(arith.select(
                                arith.cmpi(arith.CmpIPredicate.ugt, _kch, q_row_i32),
                                c_neg_inf, _sr_hi0[r]))
                        scf.YieldOp(_mlo + _mhi)
                    with ir.InsertionPoint(_mif0.else_block):
                        scf.YieldOp(_sr_lo0 + _sr_hi0)
                    _sr_lo0 = [_mif0.results[i] for i in range(16)]
                    _sr_hi0 = [_mif0.results[16 + i] for i in range(16)]

                # alu0[0]
                _fm = {"fastmath": fm_fast}
                _lm0 = _sr_lo0[0]
                for r in range_constexpr(15):
                    _lm0 = arith.MaxNumFOp(_lm0, _sr_lo0[r + 1], **_fm).result
                for r in range_constexpr(16):
                    _lm0 = arith.MaxNumFOp(_lm0, _sr_hi0[r], **_fm).result
                _pm0 = reduction_peer(_lm0)
                _rm0 = arith.MaxNumFOp(_lm0, _pm0, **_fm).result
                _mn0 = arith.MaxNumFOp(m_running, _rm0, **_fm).result
                _dm0 = arith.SubFOp(m_running, _mn0, fastmath=fm_fast).result
                _ds0 = arith.MulFOp(_dm0, c_sm_scale_log2e, fastmath=fm_fast).result
                _corr0 = rocdl.exp2(T.f32, _ds0)
                _sm0 = arith.MulFOp(c_sm_scale_log2e, _mn0, fastmath=fm_fast).result
                _nsm0 = arith.SubFOp(c_zero_f, _sm0, fastmath=fm_fast).result

                # O_rescale chunk 0 by corr_0
                if USE_PK_MUL:
                    o_accs[0] = _pk_mul_f32_v16(o_accs[0], _corr0)
                else:
                    _cv0 = vector.broadcast(v16f32_type, _corr0)
                    o_accs[0] = arith.MulFOp(o_accs[0], _cv0, fastmath=fm_fast).result

                # Memory: waitcnt → K store slot1 → V store slot0 → barrier
                _waitcnt_vm_n(0)
                coop_store_k_lds(_xp_kdbuf, 1)
                coop_store_v_lds(_xp_v0, 0)
                gpu.barrier()
                m_running = _mn0

                # ==== Phase A: [alu1[0] + P_pack[0]] || [GEMM1[1]] ====
                _xp_kb1 = k_buf_base(1)
                _xp_v1 = coop_load_v_global(_xp_kv1)
                rocdl.sched_barrier(0)
                for _rep in range_constexpr(K_STEPS_QK):
                    rocdl.sched_group_barrier(rocdl.mask_mfma, 2, 0)
                    rocdl.sched_group_barrier(rocdl.mask_dsrd, 2, 0)
                    rocdl.sched_group_barrier(_XP_VALU_MASK, 7, 0)

                # VALU: alu1[0] (exp2 + sum + l_update) + P_pack[0]
                _pv_lo0 = []
                _pv_hi0 = []
                for r in range_constexpr(16):
                    _d = math_dialect.fma(_sr_lo0[r], c_sm_scale_log2e, _nsm0)
                    _pv_lo0.append(rocdl.exp2(T.f32, _d))
                for r in range_constexpr(16):
                    _d = math_dialect.fma(_sr_hi0[r], c_sm_scale_log2e, _nsm0)
                    _pv_hi0.append(rocdl.exp2(T.f32, _d))
                _ls0 = c_zero_f
                for r in range_constexpr(16):
                    _ls0 = arith.AddFOp(_ls0, _pv_lo0[r], fastmath=fm_fast).result
                for r in range_constexpr(16):
                    _ls0 = arith.AddFOp(_ls0, _pv_hi0[r], fastmath=fm_fast).result
                _ps0 = reduction_peer(_ls0)
                _ts0 = arith.AddFOp(_ls0, _ps0, fastmath=fm_fast).result
                _lc0 = arith.MulFOp(_corr0, l_running, fastmath=fm_fast).result
                _ln0 = arith.AddFOp(_lc0, _ts0, fastmath=fm_fast).result
                _pp_lo0 = []
                _pp_hi0 = []
                for pks in range_constexpr(PV_K_STEPS):
                    pb = pks * 4
                    _pp_lo0.append(bf16_trunc_pack_v4(_pv_lo0[pb:pb+4]))
                    _pp_hi0.append(bf16_trunc_pack_v4(_pv_hi0[pb:pb+4]))

                # MFMA: GEMM1[1]
                _kl1 = [None] * K_STEPS_QK
                _kh1 = [None] * K_STEPS_QK
                for p in range_constexpr(_XP_PD):
                    _kl1[p] = vector.load_op(mfma_pack_type, lds_kv, [_xp_klo(_xp_kb1, p)])
                    _kh1[p] = vector.load_op(mfma_pack_type, lds_kv, [_xp_khi_fn(_xp_kb1, p)])
                _sl1 = c_zero_v16f32
                _sh1 = c_zero_v16f32
                for ks in range_constexpr(K_STEPS_QK):
                    _sl1 = mfma_acc(_kl1[ks], q_b_packs[ks], _sl1)
                    _sh1 = mfma_acc(_kh1[ks], q_b_packs[ks], _sh1)
                    if ks + _XP_PD < K_STEPS_QK:
                        _kl1[ks + _XP_PD] = vector.load_op(
                            mfma_pack_type, lds_kv, [_xp_klo(_xp_kb1, ks + _XP_PD)])
                        _kh1[ks + _XP_PD] = vector.load_op(
                            mfma_pack_type, lds_kv, [_xp_khi_fn(_xp_kb1, ks + _XP_PD)])

                if ENABLE_K_INTERBLOCK:
                    rocdl.sched_barrier(0)
                    _xp_nkv = kv_block_start + arith.index(BLOCK_N_OUT)
                    _xp_nkv_s = arith.MinSIOp(
                        _xp_nkv, kv_upper - arith.index(BLOCK_N)).result
                    _xp_nkv_s = arith.MaxSIOp(
                        _xp_nkv_s, arith.index(0)).result
                    _k_interblock_next = coop_load_k_global(_xp_nkv_s)
                l_running = _ln0

                # ==== V[0] pre-read + O_rescale[1-3] + alu0[1] + GEMM2[0] ====
                _xp_vb = v_buf_base(0)
                _xp_steps = [(dc, pks) for dc in range(D_CHUNKS) for pks in range(PV_K_STEPS)]
                _XP_TPV = len(_xp_steps)
                def _xp_rdv(si, vb):
                    dc, pks = _xp_steps[si]
                    dp = arith.index(dc * D_CHUNK) + lane_mod_32
                    kbv = arith.index(pks * PV_K_STEP) + lane_div_32 * 4
                    vli = vb + dp * VT_STRIDE + kbv
                    vhi = vli + arith.index(K_SUB_N)
                    return (vector.load(v4f16_type, lds_kv, [vli]),
                            vector.load(v4f16_type, lds_kv, [vhi]))

                _xp_vl0 = [None] * _XP_TPV
                _xp_vh0 = [None] * _XP_TPV
                for si in range_constexpr(_XP_TPV):
                    _xp_vl0[si], _xp_vh0[si] = _xp_rdv(si, _xp_vb)

                for dc_r in range_constexpr(D_CHUNKS - 1):
                    if USE_PK_MUL:
                        o_accs[dc_r + 1] = _pk_mul_f32_v16(o_accs[dc_r + 1], _corr0)
                    else:
                        _cv0b = vector.broadcast(v16f32_type, _corr0)
                        o_accs[dc_r + 1] = arith.MulFOp(
                            o_accs[dc_r + 1], _cv0b, fastmath=fm_fast).result

                # s_raw[1] + causal mask + alu0[1]
                _sr_lo1 = [vector.extract(_sl1, static_position=[r], dynamic_position=[]) for r in range(16)]
                _sr_hi1 = [vector.extract(_sh1, static_position=[r], dynamic_position=[]) for r in range(16)]
                if CAUSAL:
                    _xp_kvs1 = arith.index_cast(T.i32, _xp_kv1)
                    _xp_mk1 = arith.AddIOp(_xp_kvs1, arith.constant(BLOCK_N - 1, type=T.i32)).result
                    _xp_tm1 = arith.cmpi(arith.CmpIPredicate.ugt, _xp_mk1, _xp_qs)
                    _mif1 = scf.IfOp(_xp_tm1, [T.f32] * 32, has_else=True)
                    with ir.InsertionPoint(_mif1.then_block):
                        _ml1, _mh1 = [], []
                        for r in range_constexpr(16):
                            _ro = arith.constant((r % 4) + (r // 4) * 8, type=T.i32)
                            _lo = arith.MulIOp(_xp_ld32, arith.constant(4, type=T.i32)).result
                            _kc = arith.AddIOp(arith.AddIOp(_xp_kvs1, _lo).result, _ro).result
                            _ml1.append(arith.select(
                                arith.cmpi(arith.CmpIPredicate.ugt, _kc, q_row_i32),
                                c_neg_inf, _sr_lo1[r]))
                            _kch = arith.AddIOp(_kc, arith.constant(K_SUB_N, type=T.i32)).result
                            _mh1.append(arith.select(
                                arith.cmpi(arith.CmpIPredicate.ugt, _kch, q_row_i32),
                                c_neg_inf, _sr_hi1[r]))
                        scf.YieldOp(_ml1 + _mh1)
                    with ir.InsertionPoint(_mif1.else_block):
                        scf.YieldOp(_sr_lo1 + _sr_hi1)
                    _sr_lo1 = [_mif1.results[i] for i in range(16)]
                    _sr_hi1 = [_mif1.results[16 + i] for i in range(16)]

                _lm1 = _sr_lo1[0]
                for r in range_constexpr(15):
                    _lm1 = arith.MaxNumFOp(_lm1, _sr_lo1[r + 1], **_fm).result
                for r in range_constexpr(16):
                    _lm1 = arith.MaxNumFOp(_lm1, _sr_hi1[r], **_fm).result
                _pm1 = reduction_peer(_lm1)
                _rm1 = arith.MaxNumFOp(_lm1, _pm1, **_fm).result
                _mn1 = arith.MaxNumFOp(m_running, _rm1, **_fm).result
                _dm1 = arith.SubFOp(m_running, _mn1, fastmath=fm_fast).result
                _ds1 = arith.MulFOp(_dm1, c_sm_scale_log2e, fastmath=fm_fast).result
                _corr1 = rocdl.exp2(T.f32, _ds1)
                _sm1 = arith.MulFOp(c_sm_scale_log2e, _mn1, fastmath=fm_fast).result
                _nsm1 = arith.SubFOp(c_zero_f, _sm1, fastmath=fm_fast).result

                # GEMM2[0]
                rocdl.sched_mfma(_SCHED_MFMA_PV)
                for si in range_constexpr(_XP_TPV):
                    dc, pks = _xp_steps[si]
                    o_accs[dc] = mfma_acc(_xp_vl0[si], _pp_lo0[pks], o_accs[dc])
                    o_accs[dc] = mfma_acc(_xp_vh0[si], _pp_hi0[pks], o_accs[dc])

                # ==== Between: O_rescale ALL by corr_1 + V[1] store ====
                if USE_PK_MUL:
                    for dc in range_constexpr(D_CHUNKS):
                        o_accs[dc] = _pk_mul_f32_v16(o_accs[dc], _corr1)
                else:
                    _cv1 = vector.broadcast(v16f32_type, _corr1)
                    for dc in range_constexpr(D_CHUNKS):
                        o_accs[dc] = arith.MulFOp(o_accs[dc], _cv1, fastmath=fm_fast).result
                gpu.barrier()
                if ENABLE_K_INTERBLOCK:
                    _waitcnt_vm_n(NUM_BATCHES_KV)
                else:
                    _waitcnt_vm_n(0)
                coop_store_v_lds(_xp_v1, 0)
                gpu.barrier()

                # ==== Epilogue: V[1] pre-read → alu1[1] + P_pack[1] → GEMM2[1] ====
                _xp_vl1 = [None] * _XP_TPV
                _xp_vh1 = [None] * _XP_TPV
                for si in range_constexpr(_XP_TPV):
                    _xp_vl1[si], _xp_vh1[si] = _xp_rdv(si, _xp_vb)

                _pv_lo1 = []
                _pv_hi1 = []
                for r in range_constexpr(16):
                    _d = math_dialect.fma(_sr_lo1[r], c_sm_scale_log2e, _nsm1)
                    _pv_lo1.append(rocdl.exp2(T.f32, _d))
                for r in range_constexpr(16):
                    _d = math_dialect.fma(_sr_hi1[r], c_sm_scale_log2e, _nsm1)
                    _pv_hi1.append(rocdl.exp2(T.f32, _d))
                _ls1 = c_zero_f
                for r in range_constexpr(16):
                    _ls1 = arith.AddFOp(_ls1, _pv_lo1[r], fastmath=fm_fast).result
                for r in range_constexpr(16):
                    _ls1 = arith.AddFOp(_ls1, _pv_hi1[r], fastmath=fm_fast).result
                _ps1 = reduction_peer(_ls1)
                _ts1 = arith.AddFOp(_ls1, _ps1, fastmath=fm_fast).result
                _lc1 = arith.MulFOp(_corr1, l_running, fastmath=fm_fast).result
                _ln1 = arith.AddFOp(_lc1, _ts1, fastmath=fm_fast).result
                _pp_lo1 = []
                _pp_hi1 = []
                for pks in range_constexpr(PV_K_STEPS):
                    pb = pks * 4
                    _pp_lo1.append(bf16_trunc_pack_v4(_pv_lo1[pb:pb+4]))
                    _pp_hi1.append(bf16_trunc_pack_v4(_pv_hi1[pb:pb+4]))

                rocdl.sched_mfma(_SCHED_MFMA_PV)
                for si in range_constexpr(_XP_TPV):
                    dc, pks = _xp_steps[si]
                    o_accs[dc] = mfma_acc(_xp_vl1[si], _pp_lo1[pks], o_accs[dc])
                    o_accs[dc] = mfma_acc(_xp_vh1[si], _pp_hi1[pks], o_accs[dc])
                m_running = _mn1
                l_running = _ln1

            for kv_sub in range_constexpr(0 if ENABLE_XSUB_PIPELINE else N_SUBTILES):
                kv_start = kv_block_start + kv_sub * BLOCK_N

                if ENABLE_PREFETCH_3BUF:
                    k_slot = CK_LDS_SEQ[kv_sub % len(CK_LDS_SEQ)] % NUM_PREFETCH_K
                elif _use_dma_dbuf:
                    if kv_sub % 2 == 0:
                        _k_buf_id = _cur_buf_id
                    else:
                        _k_buf_id = arith.index(1) - _cur_buf_id
                    rocdl.s_waitcnt(0)
                    gpu.barrier()
                    _next_k_buf_id = arith.index(1) - _k_buf_id
                    if kv_sub + 1 < N_SUBTILES:
                        coop_dma_k(
                            kv_block_start + (kv_sub + 1) * BLOCK_N,
                            _next_k_buf_id,
                        )
                    else:
                        _next_kv = kv_block_start + arith.index(BLOCK_N_OUT)
                        _has_next = arith.cmpi(
                            arith.CmpIPredicate.slt, _next_kv, kv_upper)
                        _if_dma = scf.IfOp(_has_next)
                        with ir.InsertionPoint(_if_dma.then_block):
                            coop_dma_k(_next_kv, _next_k_buf_id)
                            scf.YieldOp([])
                    rocdl.sched_barrier(0)
                    k_base = k_buf_base(_k_buf_id)
                elif ENABLE_K_DBUF:
                    k_slot = kv_sub % 2
                    if kv_sub == 0:
                        if ENABLE_K_INTERBLOCK:
                            _waitcnt_vm_n(0)
                            coop_store_k_lds(_k_interblock_vecs, k_slot)
                            gpu.barrier()
                        else:
                            coop_load_k(kv_start, k_slot)
                            gpu.barrier()
                else:
                    k_slot = 0
                    coop_load_k(kv_start, k_slot)
                    gpu.barrier()
                if not _use_dma_dbuf:
                    k_base = k_buf_base(k_slot)

                if not USE_HW_TR or (not ENABLE_DMA and not ENABLE_PREFETCH_3BUF):
                    _v_vecs_prefetch = coop_load_v_global(kv_start)

                # ==== GEMM1: bulk-read all K packs, then pipeline MFMAs ====
                k_hi_offset = K_SUB_N * K_STRIDE

                def _k_idx_lo(ks):
                    col = arith.index(ks * K_STEP_QK) + lane_div_32 * MFMA_LANE_K
                    swz = _k_swizzle(lane_mod_32, col)
                    return k_base + lane_mod_32 * K_STRIDE + swz

                def _k_idx_hi(ks):
                    col = arith.index(ks * K_STEP_QK) + lane_div_32 * MFMA_LANE_K
                    row_hi = lane_mod_32 + arith.index(K_SUB_N)
                    swz = _k_swizzle(row_hi, col)
                    return (k_base + k_hi_offset
                            + lane_mod_32 * K_STRIDE + swz)

                _QK_PREFETCH_DEPTH = 2
                k_packs_lo = [None] * K_STEPS_QK
                k_packs_hi = [None] * K_STEPS_QK
                for p in range_constexpr(_QK_PREFETCH_DEPTH):
                    k_packs_lo[p] = vector.load_op(
                        mfma_pack_type, lds_kv, [_k_idx_lo(p)])
                    k_packs_hi[p] = vector.load_op(
                        mfma_pack_type, lds_kv, [_k_idx_hi(p)])

                if ENABLE_DMA and not ENABLE_PREFETCH_3BUF:
                    coop_dma_v(kv_start, 0)
                    if ENABLE_SCHED_HINTS:
                        rocdl.sched_barrier(0)

                s_acc_lo = c_zero_v16f32
                s_acc_hi = c_zero_v16f32
                if ENABLE_SGB_INTERLEAVE:
                    rocdl.sched_barrier(0)
                elif ENABLE_SCHED_HINTS:
                    rocdl.sched_dsrd(_SCHED_DSRD_QK)
                    rocdl.sched_mfma(_SCHED_MFMA_QK)
                for ks in range_constexpr(K_STEPS_QK):
                    if ENABLE_SGB_INTERLEAVE:
                        rocdl.sched_group_barrier(rocdl.mask_mfma, 2, 0)
                        rocdl.sched_group_barrier(rocdl.mask_dsrd, 2, 0)
                    s_acc_lo = mfma_acc(
                        k_packs_lo[ks], q_b_packs[ks], s_acc_lo)
                    s_acc_hi = mfma_acc(
                        k_packs_hi[ks], q_b_packs[ks], s_acc_hi)
                    if ks + _QK_PREFETCH_DEPTH < K_STEPS_QK:
                        k_packs_lo[ks + _QK_PREFETCH_DEPTH] = vector.load_op(
                            mfma_pack_type, lds_kv,
                            [_k_idx_lo(ks + _QK_PREFETCH_DEPTH)])
                        k_packs_hi[ks + _QK_PREFETCH_DEPTH] = vector.load_op(
                            mfma_pack_type, lds_kv,
                            [_k_idx_hi(ks + _QK_PREFETCH_DEPTH)])

                if ENABLE_K_DBUF and kv_sub + 1 < N_SUBTILES:
                    if ENABLE_SCHED_HINTS:
                        rocdl.sched_barrier(0)
                    _k_dbuf_vecs = coop_load_k_global(
                        kv_block_start + (kv_sub + 1) * BLOCK_N)
                if ENABLE_K_INTERBLOCK and kv_sub == N_SUBTILES - 1:
                    if ENABLE_SCHED_HINTS:
                        rocdl.sched_barrier(0)
                    _next_kv_start = kv_block_start + arith.index(BLOCK_N_OUT)
                    _next_kv_safe = arith.MinSIOp(
                        _next_kv_start, kv_upper - arith.index(BLOCK_N)).result
                    _next_kv_safe = arith.MaxSIOp(
                        _next_kv_safe, arith.index(0)).result
                    _k_interblock_next = coop_load_k_global(_next_kv_safe)

                # ==== Online softmax over 64 KV positions ====
                s_raw_lo = []
                s_raw_hi = []
                for r in range_constexpr(16):
                    s_raw_lo.append(vector.extract(
                        s_acc_lo, static_position=[r], dynamic_position=[]))
                    s_raw_hi.append(vector.extract(
                        s_acc_hi, static_position=[r], dynamic_position=[]))

                if CAUSAL:
                    kv_start_i32 = arith.index_cast(T.i32, kv_start)
                    lane_div_32_i32 = arith.index_cast(T.i32, lane_div_32)
                    q_start_i32 = arith.index_cast(T.i32, q_start)
                    max_kv_col_i32 = arith.AddIOp(
                        kv_start_i32,
                        arith.constant(BLOCK_N - 1, type=T.i32)).result
                    tile_needs_mask = arith.cmpi(
                        arith.CmpIPredicate.ugt, max_kv_col_i32, q_start_i32)
                    _mask_if = scf.IfOp(
                        tile_needs_mask, [T.f32] * 32, has_else=True)
                    with ir.InsertionPoint(_mask_if.then_block):
                        _m_lo = []
                        _m_hi = []
                        for r in range_constexpr(16):
                            # MFMA 32x32 register remap: 16 elements -> (row, col)
                            r_off_i32 = arith.constant(
                                (r % 4) + (r // 4) * 8, type=T.i32)
                            lane_off_i32 = arith.MulIOp(
                                lane_div_32_i32,
                                arith.constant(4, type=T.i32)).result
                            kv_col_lo = arith.AddIOp(
                                arith.AddIOp(
                                    kv_start_i32, lane_off_i32).result,
                                r_off_i32).result
                            is_masked_lo = arith.cmpi(
                                arith.CmpIPredicate.ugt,
                                kv_col_lo, q_row_i32)
                            _m_lo.append(arith.select(
                                is_masked_lo, c_neg_inf, s_raw_lo[r]))
                            kv_col_hi = arith.AddIOp(
                                kv_col_lo,
                                arith.constant(K_SUB_N, type=T.i32)).result
                            is_masked_hi = arith.cmpi(
                                arith.CmpIPredicate.ugt,
                                kv_col_hi, q_row_i32)
                            _m_hi.append(arith.select(
                                is_masked_hi, c_neg_inf, s_raw_hi[r]))
                        scf.YieldOp(_m_lo + _m_hi)
                    with ir.InsertionPoint(_mask_if.else_block):
                        scf.YieldOp(s_raw_lo + s_raw_hi)
                    s_raw_lo = [_mask_if.results[i] for i in range(16)]
                    s_raw_hi = [_mask_if.results[16 + i] for i in range(16)]

                _max_fm = {"fastmath": fm_fast}
                local_max = s_raw_lo[0]
                for r in range_constexpr(15):
                    local_max = arith.MaxNumFOp(local_max, s_raw_lo[r + 1], **_max_fm).result
                for r in range_constexpr(16):
                    local_max = arith.MaxNumFOp(local_max, s_raw_hi[r], **_max_fm).result
                peer_max = reduction_peer(local_max)
                row_max = arith.MaxNumFOp(local_max, peer_max, **_max_fm).result
                m_new_raw = arith.MaxNumFOp(m_running, row_max, **_max_fm).result

                diff_m_raw = arith.SubFOp(m_running, m_new_raw, fastmath=fm_fast).result
                diff_m_scaled = arith.MulFOp(diff_m_raw, c_sm_scale_log2e, fastmath=fm_fast).result
                corr = rocdl.exp2(T.f32, diff_m_scaled)

                scaled_max = arith.MulFOp(c_sm_scale_log2e, m_new_raw, fastmath=fm_fast).result
                neg_scaled_max = arith.SubFOp(c_zero_f, scaled_max, fastmath=fm_fast).result

                p_vals_lo = []
                p_vals_hi = []
                for r in range_constexpr(16):
                    diff_lo = math_dialect.fma(s_raw_lo[r], c_sm_scale_log2e, neg_scaled_max)
                    p_vals_lo.append(rocdl.exp2(T.f32, diff_lo))
                for r in range_constexpr(16):
                    diff_hi = math_dialect.fma(s_raw_hi[r], c_sm_scale_log2e, neg_scaled_max)
                    p_vals_hi.append(rocdl.exp2(T.f32, diff_hi))
                local_sum = c_zero_f
                for r in range_constexpr(16):
                    local_sum = arith.AddFOp(local_sum, p_vals_lo[r], fastmath=fm_fast).result
                for r in range_constexpr(16):
                    local_sum = arith.AddFOp(local_sum, p_vals_hi[r], fastmath=fm_fast).result

                peer_sum = reduction_peer(local_sum)
                tile_sum = arith.AddFOp(local_sum, peer_sum, fastmath=fm_fast).result
                l_corr = arith.MulFOp(corr, l_running, fastmath=fm_fast).result
                l_new = arith.AddFOp(l_corr, tile_sum, fastmath=fm_fast).result

                # ==== Rescale O accumulators ====
                corr_vec = vector.broadcast(v16f32_type, corr)
                if not USE_HW_TR:
                    if USE_PK_MUL:
                        o_accs[0] = _pk_mul_f32_v16(o_accs[0], corr)
                    else:
                        o_accs[0] = arith.MulFOp(o_accs[0], corr_vec, fastmath=fm_fast).result
                else:
                    for dc in range_constexpr(D_CHUNKS):
                        if USE_PK_MUL:
                            o_accs[dc] = _pk_mul_f32_v16(o_accs[dc], corr)
                        else:
                            o_accs[dc] = arith.MulFOp(o_accs[dc], corr_vec, fastmath=fm_fast).result

                if ENABLE_PREFETCH_3BUF and (kv_sub + preload_k_count) < N_SUBTILES:
                    next_k_sub = kv_sub + preload_k_count
                    next_k_start = kv_block_start + next_k_sub * BLOCK_N
                    next_k_slot = (
                        CK_LDS_SEQ[next_k_sub % len(CK_LDS_SEQ)] % NUM_PREFETCH_K
                    )
                    if ENABLE_DMA:
                        coop_dma_k(next_k_start, next_k_slot)
                    else:
                        coop_load_k(next_k_start, next_k_slot)

                def _build_p_packs():
                    nonlocal p_packs_lo, p_packs_hi
                    if dtype_str == "bf16" and not USE_K16:
                        p_packs_lo = []
                        p_packs_hi = []
                        for pks in range_constexpr(PV_K_STEPS):
                            p_base = pks * 4
                            p_packs_lo.append(bf16_trunc_pack_v4(
                                p_vals_lo[p_base:p_base+4]))
                            p_packs_hi.append(bf16_trunc_pack_v4(
                                p_vals_hi[p_base:p_base+4]))
                    elif dtype_str == "bf16" and USE_K16:
                        p_packs_lo = []
                        p_packs_hi = []
                        for pks in range_constexpr(PV_K_STEPS):
                            p_base = pks * 8
                            p_packs_lo.append(bf16_trunc_pack_v8(
                                p_vals_lo[p_base:p_base+8]))
                            p_packs_hi.append(bf16_trunc_pack_v8(
                                p_vals_hi[p_base:p_base+8]))
                    else:
                        p_f16_lo = []
                        p_f16_hi = []
                        for r in range_constexpr(16):
                            p_f16_lo.append(arith.trunc_f(elem_type, p_vals_lo[r]))
                            p_f16_hi.append(arith.trunc_f(elem_type, p_vals_hi[r]))
                        if USE_K16:
                            p_packs_lo = []
                            p_packs_hi = []
                            for pks in range_constexpr(PV_K_STEPS):
                                p_base = pks * 8
                                p_packs_lo.append(vector.from_elements(v8f16_type, [
                                    p_f16_lo[p_base+0], p_f16_lo[p_base+1],
                                    p_f16_lo[p_base+2], p_f16_lo[p_base+3],
                                    p_f16_lo[p_base+4], p_f16_lo[p_base+5],
                                    p_f16_lo[p_base+6], p_f16_lo[p_base+7]]))
                                p_packs_hi.append(vector.from_elements(v8f16_type, [
                                    p_f16_hi[p_base+0], p_f16_hi[p_base+1],
                                    p_f16_hi[p_base+2], p_f16_hi[p_base+3],
                                    p_f16_hi[p_base+4], p_f16_hi[p_base+5],
                                    p_f16_hi[p_base+6], p_f16_hi[p_base+7]]))
                        else:
                            p_packs_lo = []
                            p_packs_hi = []
                            for pks in range_constexpr(PV_K_STEPS):
                                p_base = pks * 4
                                p_packs_lo.append(vector.from_elements(v4f16_type, [
                                    p_f16_lo[p_base], p_f16_lo[p_base+1],
                                    p_f16_lo[p_base+2], p_f16_lo[p_base+3]]))
                                p_packs_hi.append(vector.from_elements(v4f16_type, [
                                    p_f16_hi[p_base], p_f16_hi[p_base+1],
                                    p_f16_hi[p_base+2], p_f16_hi[p_base+3]]))

                p_packs_lo = None
                p_packs_hi = None

                if ENABLE_P_PACK_EARLY:
                    _build_p_packs()

                if ENABLE_PREFETCH_3BUF:
                    v_slot = CK_LDS_SEQ[kv_sub % len(CK_LDS_SEQ)] % NUM_PREFETCH_V
                    v_base = v_buf_base(v_slot)
                    coop_load_v(kv_start, v_slot)
                    if ENABLE_SCHED_HINTS:
                        rocdl.sched_group_barrier(rocdl.mask_dswr, 1, 0)
                    gpu.barrier()
                elif ENABLE_DMA:
                    v_base = v_buf_base(0)
                    rocdl.s_waitcnt(0)
                    gpu.barrier()
                elif ENABLE_K_DBUF:
                    v_slot = 0
                    v_base = v_buf_base(v_slot)
                    if ENABLE_K_INTERBLOCK and kv_sub == N_SUBTILES - 1:
                        _waitcnt_vm_n(NUM_BATCHES_KV)
                    else:
                        _waitcnt_vm_n(0)
                    if kv_sub + 1 < N_SUBTILES:
                        coop_store_k_lds(_k_dbuf_vecs, (kv_sub + 1) % 2)
                    coop_store_v_lds(_v_vecs_prefetch, v_slot)
                    gpu.barrier()
                else:
                    v_slot = 0
                    v_base = v_buf_base(v_slot)
                    _waitcnt_vm_n(0)
                    coop_store_v_lds(_v_vecs_prefetch, v_slot)
                    gpu.barrier()

                if not ENABLE_P_PACK_EARLY:
                    _build_p_packs()

                # Build flat (dc, pks) schedule for interleaved GEMM2.
                _steps = [(dc, pks)
                          for dc in range(D_CHUNKS)
                          for pks in range(PV_K_STEPS)]
                TOTAL_PV = len(_steps)

                def _read_v_pack(step_idx):
                    dc, pks = _steps[step_idx]
                    if USE_HW_TR:
                        d_col = (arith.index(dc * D_CHUNK)
                                 + tr_col_half * 16 + tr_col_sub * 4)
                        k_row = (arith.index(pks * PV_K_STEP)
                                 + lane_div_32 * 4 + tr_k_group)
                        _d_col_eff = _v_swizzle(k_row, d_col) if ENABLE_DMA else d_col
                        lds_lo = v_base + k_row * V_STRIDE + _d_col_eff
                        lds_hi = lds_lo + arith.index(K_SUB_N * V_STRIDE)
                        if USE_K16:
                            vl_a = ds_read_tr_v4f16(lds_lo)
                            vl_b = ds_read_tr_v4f16(
                                lds_lo + arith.index(8 * V_STRIDE))
                            vl = vector.shuffle(
                                vl_a, vl_b, [0, 1, 2, 3, 4, 5, 6, 7])
                            vh_a = ds_read_tr_v4f16(lds_hi)
                            vh_b = ds_read_tr_v4f16(
                                lds_hi + arith.index(8 * V_STRIDE))
                            vh = vector.shuffle(
                                vh_a, vh_b, [0, 1, 2, 3, 4, 5, 6, 7])
                        else:
                            vl = ds_read_tr_v4f16(lds_lo)
                            vh = ds_read_tr_v4f16(lds_hi)
                    else:
                        d_pos = arith.index(dc * D_CHUNK) + lane_mod_32
                        k_base = arith.index(pks * PV_K_STEP) + lane_div_32 * 4
                        v_lo_idx = v_base + d_pos * VT_STRIDE + k_base
                        v_hi_idx = v_lo_idx + arith.index(K_SUB_N)
                        vl = vector.load(v4f16_type, lds_kv, [v_lo_idx])
                        vh = vector.load(v4f16_type, lds_kv, [v_hi_idx])
                    return vl, vh

                if not ENABLE_PV_PIPELINE:
                    v_los = [None] * TOTAL_PV
                    v_his = [None] * TOTAL_PV
                    for si in range_constexpr(TOTAL_PV):
                        v_los[si], v_his[si] = _read_v_pack(si)
                    if not USE_HW_TR:
                        for dc_r in range_constexpr(D_CHUNKS - 1):
                            if USE_PK_MUL:
                                o_accs[dc_r + 1] = _pk_mul_f32_v16(o_accs[dc_r + 1], corr)
                            else:
                                o_accs[dc_r + 1] = arith.MulFOp(
                                    o_accs[dc_r + 1], corr_vec, fastmath=fm_fast,
                                ).result
                    if ENABLE_SCHED_HINTS:
                        rocdl.sched_mfma(_SCHED_MFMA_PV)
                    for si in range_constexpr(TOTAL_PV):
                        dc, pks = _steps[si]
                        o_accs[dc] = mfma_acc(
                            v_los[si], p_packs_lo[pks], o_accs[dc])
                        o_accs[dc] = mfma_acc(
                            v_his[si], p_packs_hi[pks], o_accs[dc])
                else:
                    _pvd = _PV_DEPTH
                    _v_buf_lo = [None] * _pvd
                    _v_buf_hi = [None] * _pvd
                    for p in range_constexpr(_pvd):
                        _v_buf_lo[p], _v_buf_hi[p] = _read_v_pack(p)
                    if ENABLE_SCHED_HINTS:
                        rocdl.sched_mfma(_SCHED_MFMA_PV)
                    for si in range_constexpr(TOTAL_PV):
                        dc, pks = _steps[si]
                        buf = si % _pvd
                        vl = _v_buf_lo[buf]
                        vh = _v_buf_hi[buf]
                        if si + _pvd < TOTAL_PV:
                            _v_buf_lo[buf], _v_buf_hi[buf] = _read_v_pack(si + _pvd)
                        if not USE_HW_TR and pks == 0 and dc > 0:
                            if USE_PK_MUL:
                                o_accs[dc] = _pk_mul_f32_v16(o_accs[dc], corr)
                            else:
                                o_accs[dc] = arith.MulFOp(
                                    o_accs[dc], corr_vec, fastmath=fm_fast,
                                ).result
                        o_accs[dc] = mfma_acc(vl, p_packs_lo[pks], o_accs[dc])
                        o_accs[dc] = mfma_acc(vh, p_packs_hi[pks], o_accs[dc])

                m_running = m_new_raw
                l_running = l_new

            _yield_args = [m_running, l_running] + o_accs
            if _use_dma_dbuf:
                if N_SUBTILES % 2 == 1:
                    _yield_args.append(arith.index(1) - _cur_buf_id)
                else:
                    _yield_args.append(_cur_buf_id)
            if ENABLE_K_INTERBLOCK:
                _yield_args.extend(_k_interblock_next)
            yield _yield_args

        # ---- Normalize and store O (skip OOB rows for partial Q tiles) ----
        l_final = loop_results[1]
        o_finals = [
            loop_results[2 + dc] for dc in range_constexpr(D_CHUNKS)
        ]

        inv_l = arith.DivFOp(
            c_one_f,
            l_final,
            fastmath=fm_fast,
        ).result
        inv_l_vec = vector.broadcast(v16f32_type, inv_l)

        _o_guard = scf.IfOp(q_in_bounds, [], has_else=False)
        with ir.InsertionPoint(_o_guard.then_block):
            for dc in range_constexpr(D_CHUNKS):
                o_norm_vec = arith.MulFOp(
                    o_finals[dc],
                    inv_l_vec,
                    fastmath=fm_fast,
                ).result
                for grp in range_constexpr(4):
                    f32_vals = [
                        vector.extract(
                            o_norm_vec,
                            static_position=[grp * 4 + j],
                            dynamic_position=[],
                        )
                        for j in range_constexpr(4)
                    ]
                    packed = bf16_trunc_pack_v4(f32_vals)
                    d_col_base = (arith.index(dc * D_CHUNK)
                                  + lane_div_32 * 4
                                  + arith.index(grp * 8))
                    o_global = global_idx(q_row, d_col_base)
                    _gep_store(packed, o_ptr, o_global)
            scf.YieldOp([])

        if ENABLE_PERSISTENT:
            scf.YieldOp([])
            _persistent_ip.__exit__(None, None, None)

    @flyc.jit
    def launch_flash_attn_func(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        O: fx.Tensor,
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
        if ENABLE_PERSISTENT:
            grid_x = arith.MinSIOp(grid_x, arith.index(PERSISTENT_NUM_CUS)).result

        launcher = flash_attn_func_kernel(Q, K, V, O, seq_len, batch_size)

        _wpe_val = waves_per_eu if waves_per_eu is not None else int(os.getenv("FLYDSL_FA_WPE", "3"))
        if _wpe_val is not None:
            _wpe = int(_wpe_val)
            if _wpe >= 1:
                for op in ctx.gpu_module_body.operations:
                    if getattr(op, "OPERATION_NAME", None) == "gpu.func":
                        op.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(
                            T.i32,
                            _wpe,
                        )
        if flat_work_group_size is not None:
            _fwgs = int(flat_work_group_size)
            if _fwgs >= 1:
                flat_wg_attr = ir.StringAttr.get(f"{_fwgs},{_fwgs}")
                for op in ctx.gpu_module_body.operations:
                    if getattr(op, "OPERATION_NAME", None) == "gpu.func":
                        op.attributes["rocdl.flat_work_group_size"] = flat_wg_attr

        passthrough_entries = []
        if daz:
            passthrough_entries.append(ir.ArrayAttr.get([
                ir.StringAttr.get("denormal-fp-math-f32"),
                ir.StringAttr.get("preserve-sign,preserve-sign"),
            ]))
            passthrough_entries.append(ir.ArrayAttr.get([
                ir.StringAttr.get("no-nans-fp-math"),
                ir.StringAttr.get("true"),
            ]))
            passthrough_entries.append(ir.ArrayAttr.get([
                ir.StringAttr.get("unsafe-fp-math"),
                ir.StringAttr.get("true"),
            ]))
        for op in ctx.gpu_module_body.operations:
            if getattr(op, "OPERATION_NAME", None) == "gpu.func":
                op.attributes["passthrough"] = ir.ArrayAttr.get(passthrough_entries)

        launcher.launch(
            grid=(grid_x, 1, 1),
            block=(BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    # Best MI355X FMHA numbers so far were measured with ROCm/llvm-project
    # `felix/tune_fmha` at c8cf6da4367c010c7cbbb7789a9c4349e7407619.
    # Other LLVM revisions can compile/run this kernel, but usually leave a
    # few percent of peak throughput on the table.
    _llvm_opts = {
        "enable-post-misched": os.getenv("FLYDSL_FA_POST_MISCHED", "1") == "1",
        "lsr-drop-solution": True,
    }
    if os.getenv("FLYDSL_FA_EARLY_INLINE", "1") == "1":
        _llvm_opts["amdgpu-early-inline-all"] = True
    _fmha_compile_hints = {
        "fast_fp_math": fast_fp_math,
        "unsafe_fp_math": unsafe_fp_math,
        "llvm_options": _llvm_opts,
    }

    def _launch(*args, **kwargs):
        with CompilationContext.compile_hints(_fmha_compile_hints):
            return launch_flash_attn_func(*args, **kwargs)

    def _compile(Q, K, V, O, batch_size, seq_len, stream=None):
        with CompilationContext.compile_hints(_fmha_compile_hints):
            return flyc.compile(
                launch_flash_attn_func, Q, K, V, O, batch_size, seq_len,
                fx.Stream(stream))

    _launch.compile = _compile

    return _launch


build_flash_attn_func_module = build_flash_attn_func_module_primary

