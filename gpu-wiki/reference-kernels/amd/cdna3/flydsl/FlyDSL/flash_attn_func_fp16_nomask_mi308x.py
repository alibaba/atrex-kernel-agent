# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""flash_attn_func FP16 no-mask forward — TUNED FOR AMD MI308X (CDNA3 / gfx942).

⚠ HARDWARE-SPECIFIC FP16 NO-MASK VARIANT. This file is a CDNA3 (gfx942,
   MI308X) tuned FlyDSL FlashAttention forward kernel for the shape family
   `B=1024, H=8, S=316 padded to 320, D=64`, with `dtype_str="f16"` and
   `HAS_MASK=False`. It intentionally coexists with:
   - `reference-kernels/amd/cdna3/flydsl/FlyDSL/flash_attn_func_mi308x.py`
     (MI308X bf16 causal/GQA forward)
   - `reference-kernels/amd/cdna3/flydsl/FlyDSL/flash_attn_func_mask_mi308x.py`
     (MI308X bf16 arbitrary mask / bit-packed mask forward)
   - `reference-kernels/amd/cdna/flydsl/FlyDSL/flash_attn_func.py`
     (generic CDNA baseline)

Documents:
- Optimization journey:
  `docs/amd/cdna3/mi308x/ref-docs/flydsl/cdna3-flash-attention-fp16-nomask-ck-isa-optimization.md`
- Pitfalls:
  `docs/amd/cdna3/mi308x/pitfalls/flydsl/flash-attn-pitfalls.md`

Key differences vs the generic / earlier MI308X variants:
- **No-mask hot path**: `HAS_MASK=False`; all arbitrary-mask and bit-pack logic is
  compiled out. Use `flash_attn_func_mask_mi308x.py` for free-mask workloads.
- **FP16 defaults**: `waves_per_eu=2`, K/V LDS overlay enabled, P-probability
  packing via `v_cvt_pkrtz_f16_f32`, and no-mask softmax `sched_barrier` disabled
  by default.
- **LDS footprint reduction**: K and V share an overlayed LDS slot for FP16 D64,
  reducing final LDS to 8704 B for the target shape.
- **CK-ISA diagnostic knobs kept default-off**: K/V pairpack layouts targeting
  `ds_read_b128`, K dword-to-LDS staging, K chunk32, and K swizzle disable are
  retained only for experimentation. They were not promoted because they either
  regressed performance or produced non-finite output.
- **rocprofv3/ATT-driven comparison**: local CK no-mask reference
  `$COMPOSABLE_KERNEL_ROOT/build/bin/tile_example_fmha_fwd
  -b=1024 -h=8 -v=0 -d=64 -s=316` measured 2.912971 ms / 71.889906 TFLOPS.
  This FlyDSL final measured 3.327254 ms / 62.938751 TFLOPS, i.e. 87.55% of CK
  and 92.16% of the CK95 target. Archive status: optimized but CK95 not met.

Algorithm:
- Tile shape: BLOCK_M=128, BLOCK_N=64, D=64.
- BLOCK_M=128 uses 256 threads (4 waves); per-wave Q rows = 32.
- GEMM1 uses `K @ Q^T`; online softmax stays in registers.
- P is kept in registers and fed directly to GEMM2 (`V^T @ P`) without LDS
  roundtrip.
- K/V are staged through LDS; the accepted FP16 default overlays K and V LDS.

Layout: Q/K/V/O are 1D flattened from BSHD (batch, seq_len, num_heads, head_dim).
Grid:   (batch * num_q_tiles * num_heads,) where num_q_tiles = seq_len / BLOCK_M.
Block:  (256,) for BLOCK_M=128.

Requires: head_dim % 32 == 0, head_dim >= 64, seq_len % BLOCK_N == 0.
"""

import math
import os

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, buffer_ops, gpu, range_constexpr, rocdl, vector
from flydsl.expr.arith import _to_raw as _raw
from flydsl.expr.typing import T
from atrex.src.flydsl.flash_attn.kernels_common import dtype_to_elem_type
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
    num_kv_heads=None,
):
    """Build the flash_attn_func launcher using the post-refactor FlyDSL API."""
    gpu_arch = get_hip_arch()

    BLOCK_N = 64
    K_SUB_N = 32
    WARP_SIZE = 64

    # Auto tile selection: build both M=128 and M=256 variants and dispatch on
    # total Q-row work B*S*H. Threshold 32768 means: H=8 → S≥4096 picks M=256;
    # H=32 → S≥1024; H=64 → S≥512. Empirically B1H64S2k (B*S*H=131k) wants
    # M=256 even though B*S=2048 alone would not have triggered the old gate.
    if block_m is None and num_heads >= 8 and not gpu_arch.startswith("gfx942"):
        _launcher_m128 = build_flash_attn_func_module_primary(
            num_heads, head_dim, causal, dtype_str, sm_scale, waves_per_eu,
            flat_work_group_size=256, block_m=128,
            unsafe_fp_math=unsafe_fp_math, fast_fp_math=fast_fp_math,
            daz=daz, path_tag=path_tag, num_kv_heads=num_kv_heads)
        _launcher_m256 = build_flash_attn_func_module_primary(
            num_heads, head_dim, causal, dtype_str, sm_scale, waves_per_eu,
            flat_work_group_size=512, block_m=256,
            unsafe_fp_math=unsafe_fp_math, fast_fp_math=fast_fp_math,
            daz=daz, path_tag=path_tag, num_kv_heads=num_kv_heads)
        _BSH_THRESHOLD = 32768

        def _auto_launch(*args, **kwargs):
            B = args[7] if len(args) > 7 else kwargs.get('batch_size', 1)
            S = args[8] if len(args) > 8 else kwargs.get('seq_len', 128)
            bs = (B if isinstance(B, int) else 1) * (S if isinstance(S, int) else 128)
            if bs * num_heads >= _BSH_THRESHOLD:
                return _launcher_m256(*args, **kwargs)
            return _launcher_m128(*args, **kwargs)

        return _auto_launch

    if block_m is not None:
        BLOCK_M = block_m
    else:
        BLOCK_M = 128
    if waves_per_eu is None:
        waves_per_eu = 2 if dtype_str == "f16" else 1

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
    _enable_dma_dword = False
    ENABLE_K_DWORD_DMA = (
        gpu_arch.startswith("gfx942")
        and os.getenv("FLYDSL_FLASH_ATTN_FUNC_ENABLE_K_DWORD_DMA", "0") == "1"
        and not ENABLE_PREFETCH_3BUF
    )
    ENABLE_DMA = _has_lds_load_b128 and (
        PATH_TAG == "N128" or (
            os.getenv("FLYDSL_FLASH_ATTN_FUNC_ENABLE_DMA", "0") == "1"
        )
    )
    ENABLE_LDS_VEC16 = (
        os.getenv("FLYDSL_FLASH_ATTN_FUNC_ENABLE_LDS_VEC16", "1") == "1"
    )
    REDUCE_MODE = os.getenv(
        "FLYDSL_FLASH_ATTN_FUNC_REDUCE_MODE", "ds_bpermute_lgkm_sum"
    ).strip().lower()
    if REDUCE_MODE not in (
        "xor",
        "ds_bpermute",
        "ds_bpermute_lgkm",
        "ds_bpermute_lgkm_max",
        "ds_bpermute_lgkm_sum",
    ):
        REDUCE_MODE = "xor"
    ENABLE_NOMASK_SOFTMAX_BARRIER = (
        os.getenv(
            "FLYDSL_FLASH_ATTN_FUNC_NOMASK_SOFTMAX_BARRIER",
            "0" if dtype_str == "f16" else "1",
        ) == "1"
    )
    _nomask_softmax_barrier_mask_env = os.getenv(
        "FLYDSL_FLASH_ATTN_FUNC_NOMASK_SOFTMAX_BARRIER_MASK", "0"
    ).strip().lower()
    if _nomask_softmax_barrier_mask_env.startswith("0x"):
        NOMASK_SOFTMAX_BARRIER_MASK = int(_nomask_softmax_barrier_mask_env, 16)
    else:
        NOMASK_SOFTMAX_BARRIER_MASK = int(_nomask_softmax_barrier_mask_env)
    PV_MFMA_HINT = int(os.getenv("FLYDSL_FLASH_ATTN_FUNC_PV_MFMA_HINT", "0"))
    ENABLE_KV_LDS_OVERLAY = (
        os.getenv(
            "FLYDSL_FLASH_ATTN_FUNC_ENABLE_KV_LDS_OVERLAY",
            "1" if dtype_str == "f16" else "0",
        )
        == "1"
    )
    _qk_prefetch_depth_default = "3" if ENABLE_KV_LDS_OVERLAY else "2"
    QK_PREFETCH_DEPTH = int(os.getenv(
        "FLYDSL_FLASH_ATTN_FUNC_QK_PREFETCH_DEPTH",
        _qk_prefetch_depth_default,
    ))
    if QK_PREFETCH_DEPTH < 1 or QK_PREFETCH_DEPTH > 4:
        QK_PREFETCH_DEPTH = 2
    EARLY_RESCALE_ALL = (
        os.getenv("FLYDSL_FLASH_ATTN_FUNC_EARLY_RESCALE_ALL", "0") == "1"
    )
    FP16_PKRTZ_PACK = (
        os.getenv(
            "FLYDSL_FLASH_ATTN_FUNC_FP16_PKRTZ_PACK",
            "1" if dtype_str == "f16" else "0",
        )
        == "1"
    )
    ENABLE_K_CHUNK32 = (
        gpu_arch.startswith("gfx942")
        and head_dim == 64
        and os.getenv("FLYDSL_FLASH_ATTN_FUNC_ENABLE_K_CHUNK32", "0") == "1"
        and not ENABLE_DMA
        and not ENABLE_PREFETCH_3BUF
        and not ENABLE_K_DWORD_DMA
    )
    ENABLE_K_PAIRPACK_B128 = (
        gpu_arch.startswith("gfx942")
        and dtype_str == "f16"
        and head_dim == 64
        and ENABLE_LDS_VEC16
        and os.getenv("FLYDSL_FLASH_ATTN_FUNC_ENABLE_K_PAIRPACK_B128", "0") == "1"
        and not ENABLE_DMA
        and not ENABLE_PREFETCH_3BUF
        and not ENABLE_K_DWORD_DMA
        and not ENABLE_K_CHUNK32
    )
    ENABLE_V_PAIRPACK_B128 = (
        gpu_arch.startswith("gfx942")
        and dtype_str == "f16"
        and head_dim == 64
        and os.getenv("FLYDSL_FLASH_ATTN_FUNC_ENABLE_V_PAIRPACK_B128", "0") == "1"
        and not gpu_arch.startswith("gfx950")
        and not ENABLE_DMA
        and not ENABLE_PREFETCH_3BUF
    )
    DISABLE_K_SWIZZLE = (
        os.getenv("FLYDSL_FLASH_ATTN_FUNC_DISABLE_K_SWIZZLE", "0") == "1"
    )
    NUM_PREFETCH_K = 3 if ENABLE_PREFETCH_3BUF else (2 if ENABLE_DMA else 1)
    NUM_PREFETCH_V = 3 if ENABLE_PREFETCH_3BUF else 1
    CK_LDS_SEQ = (1, 2, 0, 1, 0, 1, 2, 0) if ENABLE_PREFETCH_3BUF else (0,)
    ENABLE_K_INTERBLOCK = (
        not ENABLE_DMA and not ENABLE_PREFETCH_3BUF
        and not ENABLE_K_DWORD_DMA
        and not ENABLE_K_CHUNK32
        and not ENABLE_K_PAIRPACK_B128
        and not ENABLE_KV_LDS_OVERLAY
        and os.getenv("FLYDSL_FLASH_ATTN_FUNC_K_INTERBLOCK", "1") == "1"
    )

    # gfx950+ has ds_read_tr16_b64 (HW transpose LDS read); gfx942 needs V^T stored in LDS.
    USE_HW_TR = gpu_arch.startswith("gfx950")

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
    HAS_MASK = False  # No-mask variant — all mask logic compiled out
    _BLOCK_ORDER = os.getenv("FLYDSL_FA_BLOCK_ORDER", "head_fast").strip().lower()
    if _BLOCK_ORDER not in ("head_fast", "q_fast"):
        _BLOCK_ORDER = "head_fast"
    # GQA: H_q = num_heads, H_kv = num_kv_heads. Default MHA (H_kv == H_q).
    if num_kv_heads is None:
        NUM_KV_HEADS = num_heads
    else:
        NUM_KV_HEADS = int(num_kv_heads)
    assert NUM_HEADS % NUM_KV_HEADS == 0, (
        f"num_heads ({NUM_HEADS}) must be divisible by num_kv_heads ({NUM_KV_HEADS})"
    )
    GQA_GROUP_SIZE = NUM_HEADS // NUM_KV_HEADS
    Q_STRIDE_TOKEN = NUM_HEADS * HEAD_DIM
    KV_STRIDE_TOKEN = NUM_KV_HEADS * HEAD_DIM
    STRIDE_TOKEN = Q_STRIDE_TOKEN  # backward-compat alias used by Q path

    # K row-major: HEAD_DIM(=128) → 256B/row → 64 dwords. ds_read_b64 across 32
    # lanes hits the same banks (8-way conflict).
    # K_PAD=4 → 264B/row = 66 dwords stride. Stays 16-byte aligned per row
    # (so ds_write_b128 still single-instruction) and reduces conflicts to 2-way.
    # K_PAD=2 would give gcd(65,32)=1 (zero conflict) but breaks 16B alignment,
    # forcing 2× ds_write_b64 — empirically worse.
    _kpad_default = (
        "0" if (ENABLE_K_DWORD_DMA or ENABLE_K_PAIRPACK_B128) else
        ("8" if ENABLE_K_CHUNK32 else ("2" if (not USE_HW_TR) else "0"))
    )
    K_PAD = int(os.getenv("FLYDSL_FLASH_ATTN_FUNC_K_PAD", _kpad_default))
    K_LDS_HEAD_DIM = 32 if ENABLE_K_CHUNK32 else HEAD_DIM
    K_STRIDE = K_LDS_HEAD_DIM + K_PAD
    if USE_HW_TR:
        V_STRIDE = HEAD_DIM if ENABLE_DMA else HEAD_DIM + 4
    else:
        # VT_STRIDE=BLOCK_N+2 → 132B/row → 33-dword stride. gcd(33,32)=1 ⇒
        # bank-conflict-free for ds_read_b64. Tried +8 (matches SageAttention
        # FP8 recipe) but for bf16 it's 144B/row=36 dwords, gcd(36,32)=4 →
        # 4-way conflict that regressed every shape by 5-12%. Keep 66.
        # V11a (2026-05-10) re-tested +8 on bit-packed mask shape — bank
        # conflicts ~5x worse (3.7%→18.4% derived; 70%→421% per-INST), wall
        # clock +0.65%. Confirms +8 is wrong for this stride pattern too.
        VT_STRIDE = BLOCK_N if ENABLE_V_PAIRPACK_B128 else BLOCK_N + 2
        V_STRIDE = VT_STRIDE

    # Vectorized cooperative load constants.
    VEC_WIDTH = 16 if ENABLE_LDS_VEC16 else 8
    assert HEAD_DIM % VEC_WIDTH == 0
    THREADS_PER_ROW_LOAD = HEAD_DIM // VEC_WIDTH
    assert BLOCK_SIZE % THREADS_PER_ROW_LOAD == 0
    ROWS_PER_BATCH_LOAD = BLOCK_SIZE // THREADS_PER_ROW_LOAD
    assert K_LDS_HEAD_DIM % VEC_WIDTH == 0
    K_THREADS_PER_ROW_LOAD = K_LDS_HEAD_DIM // VEC_WIDTH
    assert BLOCK_SIZE % K_THREADS_PER_ROW_LOAD == 0
    K_ROWS_PER_BATCH_LOAD = BLOCK_SIZE // K_THREADS_PER_ROW_LOAD

    if ROWS_PER_BATCH_LOAD >= BLOCK_N:
        NUM_BATCHES_KV = 1
        KV_NEEDS_GUARD = ROWS_PER_BATCH_LOAD > BLOCK_N
    else:
        assert BLOCK_N % ROWS_PER_BATCH_LOAD == 0
        NUM_BATCHES_KV = BLOCK_N // ROWS_PER_BATCH_LOAD
        KV_NEEDS_GUARD = False
    if K_ROWS_PER_BATCH_LOAD >= BLOCK_N:
        NUM_BATCHES_K = 1
        K_NEEDS_GUARD = K_ROWS_PER_BATCH_LOAD > BLOCK_N
    else:
        assert BLOCK_N % K_ROWS_PER_BATCH_LOAD == 0
        NUM_BATCHES_K = BLOCK_N // K_ROWS_PER_BATCH_LOAD
        K_NEEDS_GUARD = False

    # K/V circular buffers; defaults to 1/1, optional 3/3 with CK-like LDS sequence.
    LDS_K_TILE_SIZE = BLOCK_N * K_STRIDE
    if USE_HW_TR:
        LDS_V_TILE_SIZE = BLOCK_N * V_STRIDE
    else:
        LDS_V_TILE_SIZE = HEAD_DIM * VT_STRIDE
    if ENABLE_KV_LDS_OVERLAY:
        LDS_KV_SLOT_SIZE = max(LDS_K_TILE_SIZE, LDS_V_TILE_SIZE)
        LDS_K_TOTAL_SIZE = NUM_PREFETCH_K * LDS_KV_SLOT_SIZE
        LDS_V_BASE = 0
        LDS_V_TOTAL_SIZE = NUM_PREFETCH_V * LDS_KV_SLOT_SIZE
        LDS_KV_TOTAL_SIZE = max(NUM_PREFETCH_K, NUM_PREFETCH_V) * LDS_KV_SLOT_SIZE
    else:
        LDS_KV_SLOT_SIZE = 0
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
        Mask: fx.Tensor,
        mask_stride_b: fx.Int32,
        mask_stride_s: fx.Int32,
        seq_len: fx.Int32,
    ):
        elem_type = dtype_to_elem_type(dtype_str)
        compute_type = T.f32
        q_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), Q)
        k_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), K)
        v_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), V)
        o_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), O)
        mask_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), Mask)
        mask_stride_b_idx = arith.index_cast(T.index, mask_stride_b)
        mask_stride_s_idx = arith.index_cast(T.index, mask_stride_s)

        # All FP operations use aggressive fast-math (no NaN/Inf checks, reassociation).
        # The unsafe_fp_math/fast_fp_math builder params control LLVM-level attributes only.
        fm_fast = arith.FastMathFlags.fast
        v4f16_type = T.vec(4, elem_type)
        vxf16_type = T.vec(VEC_WIDTH, elem_type)
        v8f16_type = T.vec(8, elem_type)
        v16f32_type = T.vec(16, compute_type)
        _v2f32_type = T.vec(2, T.f32)
        mfma_pack_type = v8f16_type if USE_K16 else v4f16_type

        def _v_pk_mul_f32_pair_inline(pair_a, pair_b):
            """v_pk_mul_f32 $0, $1, $2 — multiply two f32 pairs in one cycle."""
            return _llvm.inline_asm(
                _v2f32_type, [pair_a, pair_b],
                "v_pk_mul_f32 $0, $1, $2", "=v,v,v",
                has_side_effects=False)

        def _pk_mul_f32_v16(acc_v16f32, scale_scalar):
            """Multiply 16-element f32 vector by scalar using 8 × v_pk_mul_f32."""
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

        MFMA_LANE_K = 8 if USE_K16 else 4
        _mfma_zero = ir.IntegerAttr.get(ir.IntegerType.get_signless(32), 0)
        # FlyDSL 0.1.6's wrapped rocdl.mfma_* takes (result_type, operands, *).
        # Use the _ods_* variants which keep the per-operand signature
        # (res, a, b, c, cbsz, abid, blgp).
        def _mfma(ods_fn, a, b, c):
            return ods_fn(v16f32_type, a, b, c, _mfma_zero, _mfma_zero, _mfma_zero).result
        def mfma_acc(a, b, c):
            if const_expr(dtype_str == "bf16"):
                if const_expr(USE_K16):
                    return _mfma(rocdl._ods_mfma_f32_32x32x16_bf16, a, b, c)
                a = vector.bitcast(T.i16x4, a)
                b = vector.bitcast(T.i16x4, b)
                return _mfma(rocdl._ods_mfma_f32_32x32x8bf16_1k, a, b, c)
            if const_expr(USE_K16):
                return _mfma(rocdl._ods_mfma_f32_32x32x16_f16, a, b, c)
            return _mfma(rocdl._ods_mfma_f32_32x32x8f16, a, b, c)

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

        # ---- Decompose block_id ----
        # MI308X dispatches blocks SE-ordered (16 SEs × 5 CUs). For causal
        # attention, compute is uniform along H_q but varies along q_tile
        # (later q_tile attends to more KV blocks). Two valid orderings:
        #   FLYDSL_FA_BLOCK_ORDER=head_fast (default): block 0..H-1 = same
        #     q_tile, varying head. Per-wave SE load is uniform; waves get
        #     heavier over time as q_tile increases.
        #   FLYDSL_FA_BLOCK_ORDER=q_fast: block 0..Q-1 = same head, varying
        #     q_tile. Per-wave SE load is mixed (light+heavy concurrently);
        #     each SE accumulates same total work over time.
        num_q_tiles = (seq_len_v + BLOCK_M - 1) // BLOCK_M
        # head_fast block-order only. The original V7 kernel had a runtime
        # `if _BLOCK_ORDER == "q_fast": ... else: ...` selector here, but
        # FlyDSL 0.1.6's ast_rewriter rewrites that `if` into scf_if_dispatch
        # and drops variables first-defined inside both branches, leaving
        # `q_tile_idx` undefined in the post-if scope. We hard-wire the
        # default head_fast layout (matches FLYDSL_FA_BLOCK_ORDER=head_fast,
        # the gpu-wiki tested setting). See memory.md / Stage 1 baseline notes.
        head_idx = block_id % NUM_HEADS
        batch_q_tile_id = block_id // NUM_HEADS
        q_tile_idx = batch_q_tile_id % num_q_tiles
        batch_idx = batch_q_tile_id // num_q_tiles
        q_start = q_tile_idx * BLOCK_M

        # GQA: each group of GQA_GROUP_SIZE Q-heads shares one KV-head.
        # For MHA (GROUP_SIZE=1) this is just head_idx. Wrapped with
        # const_expr so FlyDSL ast_rewriter treats this as a Python-time
        # branch and preserves kv_head_idx in the post-if scope.
        if const_expr(GQA_GROUP_SIZE == 1):
            kv_head_idx = head_idx
        else:
            kv_head_idx = head_idx // arith.index(GQA_GROUP_SIZE)

        # ---- Cooperative load decomposition ----
        load_row_in_batch = tid // THREADS_PER_ROW_LOAD
        load_lane_in_row = tid % THREADS_PER_ROW_LOAD
        load_col_base = load_lane_in_row * VEC_WIDTH
        k_load_row_in_batch = tid // K_THREADS_PER_ROW_LOAD
        k_load_lane_in_row = tid % K_THREADS_PER_ROW_LOAD
        k_load_col_base = k_load_lane_in_row * VEC_WIDTH

        # ---- Helper: global flat indices (Q vs K/V) ----
        def global_idx(token_idx, col):
            """Q global flat offset. BSHD stride = NUM_HEADS * HEAD_DIM per token."""
            token = batch_idx * seq_len_v + token_idx
            return token * Q_STRIDE_TOKEN + head_idx * HEAD_DIM + col

        def kv_global_idx(token_idx, col):
            """K/V global flat offset. BSHD stride = NUM_KV_HEADS * HEAD_DIM per token."""
            token = batch_idx * seq_len_v + token_idx
            return token * KV_STRIDE_TOKEN + kv_head_idx * HEAD_DIM + col

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

        def _gep_load_f32(base_ptr, elem_idx):
            """Load a single f32 from global memory."""
            idx_i64 = arith.index_cast(T.i64, elem_idx)
            gep = _llvm.GEPOp(_llvm_ptr_ty(), base_ptr, [idx_i64],
                              rawConstantIndices=[_LLVM_GEP_DYNAMIC],
                              elem_type=T.f32,
                              noWrapFlags=0)
            return _llvm.LoadOp(T.f32, gep.result).result

        _v4f32_type = T.vec(4, T.f32)

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

        def mask_global_idx(batch, q_pos, kv_pos):
            """Compute flat index into mask tensor (B, 1, S, S) as f32."""
            return batch * mask_stride_b_idx + q_pos * mask_stride_s_idx + kv_pos

        def load_global_f16x4(base_ptr, base_idx):
            return _gep_load(base_ptr, base_idx, v4f16_type)

        def load_global_mfma_pack(base_ptr, base_idx):
            return _gep_load(base_ptr, base_idx, mfma_pack_type)

        def load_global_f16xN(base_ptr, base_idx):
            return _gep_load(base_ptr, base_idx, vxf16_type)

        # f32 → 16-bit pack helpers. For bf16, pack two f32 values into a
        # single dword via VOP3 v_perm_b32 (1 VALU/pair vs 3 VALU for the
        # AND/SHR/OR sequence). bf16 IS the high half of f32, so the perm
        # selector 0x07060302 picks bytes {7,6} from src1 (high_dw) and
        # bytes {3,2} from src0 (low_dw), giving {high_dw[31:16], low_dw[31:16]}
        # = {bf16(b), bf16(a)} packed as v2bf16 with lane[0]=a, lane[1]=b.
        # For fp16 (IEEE binary16) the bit layout differs (5-bit exp vs 8-bit),
        # so we must use arith.trunc_f for proper round-to-nearest conversion.
        _IS_BF16 = (dtype_str == "bf16")
        _perm_sel_07060302 = arith.constant(0x07060302, type=T.i32)
        _v2i32_type = T.vec(2, T.i32)

        def fp16_pkrtz_pack_v4(f32_vals):
            """Pack 4 f32 values to v4f16 with two v_cvt_pkrtz_f16_f32 ops."""
            p0 = _llvm.inline_asm(
                T.i32,
                [f32_vals[0], f32_vals[1]],
                "v_cvt_pkrtz_f16_f32 $0, $1, $2",
                "=v,v,v",
                has_side_effects=False,
            )
            p1 = _llvm.inline_asm(
                T.i32,
                [f32_vals[2], f32_vals[3]],
                "v_cvt_pkrtz_f16_f32 $0, $1, $2",
                "=v,v,v",
                has_side_effects=False,
            )
            return vector.bitcast(v4f16_type, vector.from_elements(_v2i32_type, [p0, p1]))

        def bf16_trunc_pack_v4(f32_vals):
            """Pack 4 f32 values into v4f16/v4bf16. dtype-aware."""
            if const_expr(_IS_BF16):
                # Want lane[0]=bf16(a), lane[1]=bf16(b) packed as one i32.
                # Empirically on this gfx942 toolchain, the operand-order
                # convention of _v_perm_b32_inline produces lane[0] from the
                # SECOND positional arg, not the first; passing (b, a) yields
                # the expected {bf16(b), bf16(a)} → bitcast → {a, b} lane order.
                a0 = arith.ArithValue(f32_vals[0]).bitcast(T.i32)
                b0 = arith.ArithValue(f32_vals[1]).bitcast(T.i32)
                p0 = _v_perm_b32_inline(b0, a0, _perm_sel_07060302)
                a1 = arith.ArithValue(f32_vals[2]).bitcast(T.i32)
                b1 = arith.ArithValue(f32_vals[3]).bitcast(T.i32)
                p1 = _v_perm_b32_inline(b1, a1, _perm_sel_07060302)
                return vector.bitcast(v4f16_type, vector.from_elements(_v2i32_type, [p0, p1]))
            # fp16 path: real round-to-nearest f32→f16 conversion
            elems = [arith.trunc_f(elem_type, v) for v in f32_vals]
            return vector.from_elements(v4f16_type, elems)

        def bf16_trunc_pack_v8(f32_vals):
            """Pack 8 f32 values into v8f16/v8bf16. dtype-aware."""
            if const_expr(_IS_BF16):
                _v4i32 = T.vec(4, T.i32)
                pairs = []
                for j in range_constexpr(4):
                    # Same operand-order swap as bf16_trunc_pack_v4.
                    a = arith.ArithValue(f32_vals[j * 2]).bitcast(T.i32)
                    b = arith.ArithValue(f32_vals[j * 2 + 1]).bitcast(T.i32)
                    p = _v_perm_b32_inline(b, a, _perm_sel_07060302)
                    pairs.append(p)
                return vector.bitcast(v8f16_type, vector.from_elements(_v4i32, pairs))
            # fp16 path
            elems = [arith.trunc_f(elem_type, v) for v in f32_vals]
            return vector.from_elements(v8f16_type, elems)

        def k_buf_base(buf_id):
            if isinstance(buf_id, int):
                if const_expr(ENABLE_KV_LDS_OVERLAY):
                    return arith.index(buf_id * LDS_KV_SLOT_SIZE)
                return arith.index(buf_id * LDS_K_TILE_SIZE)
            if const_expr(ENABLE_KV_LDS_OVERLAY):
                return buf_id * arith.index(LDS_KV_SLOT_SIZE)
            return buf_id * arith.index(LDS_K_TILE_SIZE)

        def v_buf_base(buf_id):
            if const_expr(ENABLE_KV_LDS_OVERLAY):
                if isinstance(buf_id, int):
                    return arith.index(buf_id * LDS_KV_SLOT_SIZE)
                return buf_id * arith.index(LDS_KV_SLOT_SIZE)
            return arith.index(LDS_V_BASE + buf_id * LDS_V_TILE_SIZE)

        # ---- K LDS layout: row-major with +K_PAD padding for bank diversity ----
        # K_PAD>0 supplies bank stride per row, making swizzle unnecessary.
        # K_PAD==0 falls back to the original XOR pattern.
        def _k_swizzle(row_idx, col_idx):
            if const_expr(K_PAD > 0 or DISABLE_K_SWIZZLE):
                return col_idx
            if const_expr(HEAD_DIM == 64):
                mask = (row_idx & arith.index(0x3)) << arith.index(4)
            else:
                mask = (row_idx & arith.index(0x7)) << arith.index(4)
            return col_idx ^ mask

        def _k_pairpack_swizzle(row_idx, col_idx):
            # D=64-safe XOR: row&3 keeps the 16-element bank-rotation inside
            # the row, unlike the old row&7 form which can address columns
            # 64..127 on head_dim=64.
            mask = (row_idx & arith.index(0x3)) << arith.index(4)
            return col_idx ^ mask

        def _permute_k_pairpack_v16(vec):
            # Logical chunks [0:4,4:8,8:12,12:16] become
            # [ks_even lane_half, ks_odd lane_half] for each half-wave. A
            # later v8f16 LDS load can feed two consecutive MFMA K-steps.
            order = (0, 1, 2, 3, 8, 9, 10, 11,
                     4, 5, 6, 7, 12, 13, 14, 15)
            elems = [
                vector.extract(vec, static_position=[order[i]], dynamic_position=[])
                for i in range_constexpr(16)
            ]
            return vector.from_elements(vxf16_type, elems)

        def _split_v8_to_v4_pair(v8):
            lo = vector.from_elements(v4f16_type, [
                vector.extract(v8, static_position=[i], dynamic_position=[])
                for i in range_constexpr(4)
            ])
            hi = vector.from_elements(v4f16_type, [
                vector.extract(v8, static_position=[i + 4], dynamic_position=[])
                for i in range_constexpr(4)
            ])
            return lo, hi

        def _pairpack_half_col(pair_idx):
            return arith.index(pair_idx * 16) + lane_div_32 * arith.index(8)

        def _v_pairpack_n(n_idx):
            n_half = n_idx // arith.index(32)
            n_sub = n_idx % arith.index(32)
            step = n_sub // arith.index(8)
            lane4 = (n_sub % arith.index(8)) // arith.index(4)
            elem = n_sub % arith.index(4)
            return (
                n_half * arith.index(32)
                + (step // arith.index(2)) * arith.index(16)
                + lane4 * arith.index(8)
                + (step % arith.index(2)) * arith.index(4)
                + elem
            )

        # ---- Cooperative K load (row-major, XOR-swizzled) ----
        def _k_global_col(chunk_col_base):
            if isinstance(chunk_col_base, int):
                return arith.index(chunk_col_base) + k_load_col_base
            return chunk_col_base + k_load_col_base

        def coop_load_k(tile_start, buf_id=0, chunk_col_base=0):
            k_base = k_buf_base(buf_id)
            for batch in range_constexpr(NUM_BATCHES_K):
                row_offset = batch * K_ROWS_PER_BATCH_LOAD
                row_idx = tile_start + k_load_row_in_batch + row_offset
                if const_expr(K_NEEDS_GUARD):
                    row_valid = arith.cmpi(
                        arith.CmpIPredicate.ult,
                        k_load_row_in_batch,
                        arith.index(BLOCK_N),
                    )
                    _if_k = scf.IfOp(row_valid)
                    with ir.InsertionPoint(_if_k.then_block):
                        g_idx = kv_global_idx(row_idx, _k_global_col(chunk_col_base))
                        lds_row = k_load_row_in_batch + row_offset
                        swz_col = _k_swizzle(lds_row, k_load_col_base)
                        lds_idx = k_base + lds_row * K_STRIDE + swz_col
                        vec = load_global_f16xN(k_ptr, g_idx)
                        if const_expr(ENABLE_K_PAIRPACK_B128):
                            base_col = _k_pairpack_swizzle(lds_row, k_load_col_base)
                            lds_idx_pair = k_base + lds_row * K_STRIDE + base_col
                            vector.store(_permute_k_pairpack_v16(vec), lds_kv, [lds_idx_pair])
                        else:
                            vector.store(vec, lds_kv, [lds_idx])
                        scf.YieldOp([])
                else:
                    g_idx = kv_global_idx(row_idx, _k_global_col(chunk_col_base))
                    lds_row = k_load_row_in_batch + row_offset
                    swz_col = _k_swizzle(lds_row, k_load_col_base)
                    lds_idx = k_base + lds_row * K_STRIDE + swz_col
                    vec = load_global_f16xN(k_ptr, g_idx)
                    if const_expr(ENABLE_K_PAIRPACK_B128):
                        base_col = _k_pairpack_swizzle(lds_row, k_load_col_base)
                        lds_idx_pair = k_base + lds_row * K_STRIDE + base_col
                        vector.store(_permute_k_pairpack_v16(vec), lds_kv, [lds_idx_pair])
                    else:
                        vector.store(vec, lds_kv, [lds_idx])

        def coop_load_k_global(tile_start, chunk_col_base=0):
            """Issue global loads for K, return vectors (non-blocking)."""
            vecs = []
            for batch in range_constexpr(NUM_BATCHES_K):
                row_offset = batch * K_ROWS_PER_BATCH_LOAD
                row_idx = tile_start + k_load_row_in_batch + row_offset
                if const_expr(K_NEEDS_GUARD):
                    row_valid = arith.cmpi(
                        arith.CmpIPredicate.ult,
                        k_load_row_in_batch,
                        arith.index(BLOCK_N),
                    )
                    row_idx = arith.select(row_valid, row_idx, tile_start)
                g_idx = kv_global_idx(row_idx, _k_global_col(chunk_col_base))
                raw = load_global_f16xN(k_ptr, g_idx)
                if const_expr(K_NEEDS_GUARD):
                    raw = arith.select(
                        row_valid,
                        raw,
                        arith.constant_vector(0.0, vxf16_type),
                    )
                vecs.append(raw)
            return vecs

        def coop_store_k_lds(vecs, buf_id=0):
            """Write previously-loaded K vectors to LDS."""
            k_base = k_buf_base(buf_id)
            for batch in range_constexpr(NUM_BATCHES_K):
                row_offset = batch * K_ROWS_PER_BATCH_LOAD
                lds_row = k_load_row_in_batch + row_offset
                swz_col = _k_swizzle(lds_row, k_load_col_base)
                lds_idx = k_base + lds_row * K_STRIDE + swz_col
                if const_expr(ENABLE_K_PAIRPACK_B128):
                    base_col = _k_pairpack_swizzle(lds_row, k_load_col_base)
                    lds_idx_pair = k_base + lds_row * K_STRIDE + base_col
                    vector.store(_permute_k_pairpack_v16(vecs[batch]), lds_kv, [lds_idx_pair])
                else:
                    vector.store(vecs[batch], lds_kv, [lds_idx])

        # ---- CK-like gfx942 K load: buffer_load_dword ... lds ----
        # CK's local fp16 FMHA kernel uses dword-to-LDS buffer loads on gfx942:
        # this avoids staging K through VGPRs and removes the paired ds_write
        # instructions from the K path.  The layout is the existing K_PAD=0 XOR
        # layout, so the GEMM1 LDS read helpers are unchanged.
        if const_expr(ENABLE_K_DWORD_DMA):
            k_rsrc_dword = buffer_ops.create_buffer_resource(K, max_size=True)
            k_lds_base_idx_dword = _memref.extract_aligned_pointer_as_index(lds_kv)
            K_DMA_BYTES = 4
            K_DMA_BATCH_BYTES = BLOCK_SIZE * K_DMA_BYTES
            K_DMA_TILE_BYTES = BLOCK_N * HEAD_DIM * 2
            NUM_K_DWORD_DMA = K_DMA_TILE_BYTES // K_DMA_BATCH_BYTES
            LANES_PER_K_DWORD_ROW = HEAD_DIM * 2 // K_DMA_BYTES
            ROWS_PER_K_DWORD_BATCH = K_DMA_BATCH_BYTES // (HEAD_DIM * 2)

            def coop_dma_k_dword(tile_start, buf_id=0):
                k_lds_byte_base = k_lds_base_idx_dword
                for d in range_constexpr(NUM_K_DWORD_DMA):
                    lds_addr = (
                        k_lds_byte_base
                        + wave_id * arith.index(WARP_SIZE * K_DMA_BYTES)
                        + arith.index(d * K_DMA_BATCH_BYTES)
                    )
                    row_in_tile = (
                        tid // LANES_PER_K_DWORD_ROW
                        + arith.index(d * ROWS_PER_K_DWORD_BATCH)
                    )
                    swiz_col_f16 = (tid % LANES_PER_K_DWORD_ROW) * (K_DMA_BYTES // 2)
                    if const_expr(HEAD_DIM == 64):
                        xor_mask = (row_in_tile & arith.index(0x3)) << arith.index(4)
                    else:
                        xor_mask = (row_in_tile & arith.index(0x7)) << arith.index(4)
                    unsw_col_f16 = swiz_col_f16 ^ xor_mask
                    col_byte = unsw_col_f16 * 2
                    global_row = batch_idx * seq_len_v + tile_start + row_in_tile
                    global_byte = (
                        global_row * arith.index(KV_STRIDE_TOKEN * 2)
                        + kv_head_idx * arith.index(HEAD_DIM * 2)
                        + col_byte
                    )
                    voffset = arith.index_cast(T.i32, global_byte)

                    # Keep the exact gfx942 ISA shape CK emits:
                    #   s_mov_b32 m0, <lds_base>
                    #   buffer_load_dword <voffset>, <rsrc>, 0 offen lds
                    # The raw ROCDL op crashed on gfx942 for this dword form;
                    # opaque inline asm gives the same m0/lds contract as CK.
                    lds_base_i32 = arith.index_cast(T.i32, lds_addr)
                    lds_base_sgpr = rocdl.readfirstlane(T.i32, lds_base_i32)
                    _llvm.inline_asm(
                        None,
                        [lds_base_sgpr, voffset, _raw(k_rsrc_dword)],
                        "s_mov_b32 m0, $0\n\tbuffer_load_dword $1, $2, 0 offen lds",
                        "s,v,s",
                        has_side_effects=True,
                    )

        # ---- Cooperative V load ----
        def _v_store_row_major(v_base, lds_row, vec):
            lds_idx = v_base + lds_row * V_STRIDE + load_col_base
            vector.store(vec, lds_kv, [lds_idx])

        _v1_type = T.vec(1, elem_type) if not USE_HW_TR else None
        _v2_type = T.vec(2, elem_type) if not USE_HW_TR else None
        _v4i32_type = T.vec(4, T.i32) if not USE_HW_TR else None
        _v1i32_type = T.vec(1, T.i32) if not USE_HW_TR else None
        _v8i32_type = T.vec(8, T.i32) if not USE_HW_TR else None

        def _v_perm_b32_inline(low_dw, high_dw, sel_dw):
            """v_perm_b32 vdst, src0(low), src1(high), src2(sel).
            Picks 4 bytes from {high<<32 | low} per sel byte indices.
            """
            return _llvm.inline_asm(
                ir.IntegerType.get_signless(32),
                [low_dw, high_dw, sel_dw],
                "v_perm_b32 $0, $1, $2, $3",
                "=v,v,v,v",
                has_side_effects=False,
            )

        def _ds_swizzle_xor16_inline(src):
            """ds_swizzle_b32 with offset:0x401F (BIT_MIX XOR-16) via inline asm.
            Explicit imm in the asm string ensures correct encoding (the rocdl
            wrapper passes offset as an SSA value which may not constant-fold).
            """
            return _llvm.inline_asm(
                ir.IntegerType.get_signless(32),
                [src],
                "ds_swizzle_b32 $0, $1 offset:0x401F\n\ts_waitcnt lgkmcnt(0)",
                "=v,v",
                has_side_effects=True,
            )

        def _ds_swizzle_xor8_inline(src):
            """ds_swizzle_b32 offset:0x201F (BIT_MIX XOR-8) via inline asm.
            For THREADS_PER_ROW_LOAD=8 (VEC_WIDTH=16): peer lane = own XOR 8.
            """
            return _llvm.inline_asm(
                ir.IntegerType.get_signless(32),
                [src],
                "ds_swizzle_b32 $0, $1 offset:0x201F\n\ts_waitcnt lgkmcnt(0)",
                "=v,v",
                has_side_effects=True,
            )

        def _ds_swizzle_xor4_inline(src):
            """ds_swizzle_b32 offset:0x101F (BIT_MIX XOR-4) via inline asm.
            For THREADS_PER_ROW_LOAD=4 (VEC_WIDTH=16, HEAD_DIM=64):
            peer lane = own XOR 4.
            """
            return _llvm.inline_asm(
                ir.IntegerType.get_signless(32),
                [src],
                "ds_swizzle_b32 $0, $1 offset:0x101F\n\ts_waitcnt lgkmcnt(0)",
                "=v,v",
                has_side_effects=True,
            )

        # Nowait variants: emit ds_swizzle without per-call s_waitcnt.
        # Caller must place a single `s_waitcnt lgkmcnt(0)` (via
        # `_swz_drain_lgkm0()` below) AFTER issuing all swizzles in the
        # batch but BEFORE consuming any peer result.
        # has_side_effects=True keeps LLVM from reordering the calls
        # past each other or past the trailing waitcnt.
        ENABLE_SWZ_DRAIN = (
            os.getenv("FLYDSL_FLASH_ATTN_FUNC_SWZ_DRAIN", "1") == "1"
        )

        def _swz_drain_lgkm0():
            """Emit a single `s_waitcnt lgkmcnt(0)` as inline asm so the
            barrier matches exactly what the per-call swizzle helpers used
            to bake in. Avoids ambiguity in s_waitcnt immediate encoding.
            """
            if const_expr(ENABLE_SWZ_DRAIN):
                _llvm.inline_asm(
                    None,
                    [],
                    "s_waitcnt lgkmcnt(0)",
                    "",
                    has_side_effects=True,
                )

        def _ds_swizzle_xor16_inline_nowait(src):
            return _llvm.inline_asm(
                ir.IntegerType.get_signless(32),
                [src],
                "ds_swizzle_b32 $0, $1 offset:0x401F",
                "=v,v",
                has_side_effects=ENABLE_SWZ_DRAIN,
            )

        def _ds_swizzle_xor8_inline_nowait(src):
            return _llvm.inline_asm(
                ir.IntegerType.get_signless(32),
                [src],
                "ds_swizzle_b32 $0, $1 offset:0x201F",
                "=v,v",
                has_side_effects=ENABLE_SWZ_DRAIN,
            )

        def _ds_swizzle_xor4_inline_nowait(src):
            return _llvm.inline_asm(
                ir.IntegerType.get_signless(32),
                [src],
                "ds_swizzle_b32 $0, $1 offset:0x101F",
                "=v,v",
                has_side_effects=ENABLE_SWZ_DRAIN,
            )

        def _v_store_transposed(v_base, lds_row, vec):
            for _e in range_constexpr(VEC_WIDTH):
                elem = vector.extract(vec, static_position=[_e], dynamic_position=[])
                vt_d = load_col_base + _e
                vt_n = _v_pairpack_n(lds_row) if ENABLE_V_PAIRPACK_B128 else lds_row
                vt_idx = v_base + vt_d * VT_STRIDE + vt_n
                v1 = vector.from_elements(_v1_type, [elem])
                vector.store(v1, lds_kv, [vt_idx])

        def _v_store_transposed_perm(v_base, lds_row, vec):
            """v14: cross-lane ds_swizzle (XOR-N) + intra-lane v_perm_b32 2x2 transpose.

            Auto-picks XOR mask matching THREADS_PER_ROW_LOAD:
              - VEC_WIDTH=8  → THREADS_PER_ROW_LOAD=16, peer at lane XOR 16, ds_swizzle 0x401F
              - VEC_WIDTH=16 → THREADS_PER_ROW_LOAD=8,  peer at lane XOR 8,  ds_swizzle 0x201F

            Within each row-pair (R, R+1), the lower-half lane (own=R) holds own data,
            and the upper-half lane (peer=R+1) holds peer data. ds_swizzle pairs them.
            v_perm_b32 does intra-lane 2x2 byte transpose to pack (own[c], peer[c])
            into one i32 per V^T col. Even-row lanes emit ds_write_b32; odd-row lanes
            are silent. Halves the LDS write instruction count vs scalar transpose.

            NOTE: gfx942 lacks permlanex16 (RDNA-only). ds_swizzle is the
            CDNA-compatible substitute. ds_swizzle routes through the LDS
            unit instruction queue but does not contend on LDS banks.
            """
            assert VEC_WIDTH in (8, 16), f"v14 only handles VEC_WIDTH 8/16 (got {VEC_WIDTH})"
            num_dwords = VEC_WIDTH // 2  # bf16 → dword pack ratio
            _vN_i32_type = T.vec(num_dwords, T.i32)
            own_vNi32 = vector.bitcast(_vN_i32_type, vec)

            if const_expr(THREADS_PER_ROW_LOAD == 16):
                _swz_inline = _ds_swizzle_xor16_inline_nowait
            elif const_expr(THREADS_PER_ROW_LOAD == 8):
                _swz_inline = _ds_swizzle_xor8_inline_nowait
            elif const_expr(THREADS_PER_ROW_LOAD == 4):
                _swz_inline = _ds_swizzle_xor4_inline_nowait
            else:
                raise NotImplementedError(
                    f"THREADS_PER_ROW_LOAD={THREADS_PER_ROW_LOAD} not supported")

            # Issue all `num_dwords` ds_swizzle_b32 ops back-to-back without
            # per-call s_waitcnt, then drain ONCE before consumers (v_perm /
            # vector.from_elements / ds_write below). The original per-call
            # `lgkmcnt(0)` baked into the swizzle inline-ASM forced N pipeline
            # drains for N swizzles; this batches them into 1.
            peer_dwords = []
            for k in range_constexpr(num_dwords):
                own_dw = vector.extract(
                    own_vNi32, static_position=[k], dynamic_position=[])
                peer_dw = _swz_inline(own_dw)
                peer_dwords.append(peer_dw)
            _swz_drain_lgkm0()

            row_lo_bit = arith.AndIOp(
                arith.index_cast(T.i32, lds_row),
                arith.constant(1, type=T.i32)).result
            is_even = arith.cmpi(
                arith.CmpIPredicate.eq, row_lo_bit,
                arith.constant(0, type=T.i32))

            # Reconstruct peer vec for vector.extract pack.
            peer_vNi32 = vector.from_elements(_vN_i32_type, peer_dwords)
            peer_vec = vector.bitcast(vxf16_type, peer_vNi32)

            _if_writer = scf.IfOp(is_even)
            with ir.InsertionPoint(_if_writer.then_block):
                # Pack via explicit vector.from_elements instead of v_perm_b32.
                # Each pair = (own[col], peer[col]) packed as v2bf16 → 1 b32 store.
                for _e in range_constexpr(VEC_WIDTH):
                    own_elem = vector.extract(
                        vec, static_position=[_e], dynamic_position=[])
                    peer_elem = vector.extract(
                        peer_vec, static_position=[_e], dynamic_position=[])
                    pair = vector.from_elements(
                        _v2_type, [own_elem, peer_elem])
                    vt_d = load_col_base + _e
                    vt_n = _v_pairpack_n(lds_row) if ENABLE_V_PAIRPACK_B128 else lds_row
                    vt_idx = v_base + vt_d * VT_STRIDE + vt_n
                    vector.store(pair, lds_kv, [vt_idx])
                scf.YieldOp([])

        def _v_store_transposed_paired(v_base, lds_row, vec):
            # Pair-write: ds_bpermute swap with lane (tid XOR 8) so each
            # even-row lane has data for 2 adjacent V rows. Then ds_write_b32
            # the (own, peer) pair at V^T[c, n..n+1] (contiguous in n).
            # Halves the LDS write count vs the scalar variant.
            tid_i32 = arith.index_cast(T.i32, tid)
            peer_lane_i32 = arith.XOrIOp(
                tid_i32, arith.constant(8, type=T.i32)).result
            peer_byte_i32 = arith.MulIOp(
                peer_lane_i32, arith.constant(4, type=T.i32)).result

            own_v8i32 = vector.bitcast(_v8i32_type, vec)
            peer_dwords = []
            for k in range(8):
                own_dw = vector.extract(
                    own_v8i32, static_position=[k], dynamic_position=[])
                peer_dw = rocdl.ds_bpermute(T.i32, peer_byte_i32, own_dw)
                peer_dwords.append(peer_dw)
            peer_v8i32 = vector.from_elements(_v8i32_type, peer_dwords)
            peer_vec = vector.bitcast(vxf16_type, peer_v8i32)

            # Predicate: only even-row lanes (lds_row even) emit writes.
            # The odd-row lane's data is already covered as the peer.
            row_lo_bit = arith.AndIOp(
                arith.index_cast(T.i32, lds_row),
                arith.constant(1, type=T.i32)).result
            is_even = arith.cmpi(
                arith.CmpIPredicate.eq, row_lo_bit,
                arith.constant(0, type=T.i32))
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
                    vt_n = _v_pairpack_n(lds_row) if ENABLE_V_PAIRPACK_B128 else lds_row
                    vt_idx = v_base + vt_d * VT_STRIDE + vt_n
                    vector.store(pair, lds_kv, [vt_idx])
                scf.YieldOp([])

        def _v_store_transposed_swz_b16_debug(v_base, lds_row, vec):
            """DEBUG: ds_swizzle gets peer, then 16 separate b16 writes
            (own at lds_row, peer at lds_row+1). Verifies cross-lane logic
            independently of v_perm/bitcast pack path.
            """
            assert VEC_WIDTH == 8
            own_v4i32 = vector.bitcast(_v4i32_type, vec)
            peer_dwords = []
            for k in range_constexpr(4):
                own_dw = vector.extract(
                    own_v4i32, static_position=[k], dynamic_position=[])
                peer_dw = _ds_swizzle_xor16_inline(own_dw)
                peer_dwords.append(peer_dw)
            peer_v4i32 = vector.from_elements(_v4i32_type, peer_dwords)
            peer_vec = vector.bitcast(vxf16_type, peer_v4i32)

            row_lo_bit = arith.AndIOp(
                arith.index_cast(T.i32, lds_row),
                arith.constant(1, type=T.i32)).result
            is_even = arith.cmpi(
                arith.CmpIPredicate.eq, row_lo_bit,
                arith.constant(0, type=T.i32))
            _if_writer = scf.IfOp(is_even)
            with ir.InsertionPoint(_if_writer.then_block):
                for _e in range_constexpr(VEC_WIDTH):
                    own_elem = vector.extract(
                        vec, static_position=[_e], dynamic_position=[])
                    peer_elem = vector.extract(
                        peer_vec, static_position=[_e], dynamic_position=[])
                    vt_d = load_col_base + _e
                    vt_n = _v_pairpack_n(lds_row) if ENABLE_V_PAIRPACK_B128 else lds_row
                    vt_idx_own = v_base + vt_d * VT_STRIDE + vt_n
                    vt_idx_peer = vt_idx_own + arith.index(1)
                    v1_own = vector.from_elements(_v1_type, [own_elem])
                    v1_peer = vector.from_elements(_v1_type, [peer_elem])
                    vector.store(v1_own, lds_kv, [vt_idx_own])
                    vector.store(v1_peer, lds_kv, [vt_idx_peer])
                scf.YieldOp([])

        # v14 default: ds_swizzle XOR-N + paired ds_write_b32 transpose.
        # Halves LDS write count vs scalar; ~22-24% kernel speedup on MI308X.
        # Set FLYDSL_FLASH_ATTN_FUNC_V_TRANSPOSE_PERM=0 to fall back to scalar.
        DISABLE_V_TRANSPOSE_PERM = (
            os.getenv("FLYDSL_FLASH_ATTN_FUNC_V_TRANSPOSE_PERM", "1") == "0"
        )
        ENABLE_V_BPERM_PAIRED = (
            os.getenv("FLYDSL_FLASH_ATTN_FUNC_V_BPERM_PAIRED", "0") == "1"
        )
        ENABLE_V_TRANSPOSE_SWZ_DEBUG = (
            os.getenv("FLYDSL_FLASH_ATTN_FUNC_V_TRANSPOSE_SWZ_DEBUG", "0") == "1"
        )
        if const_expr(USE_HW_TR):
            _v_store_to_lds = _v_store_row_major
        elif const_expr(DISABLE_V_TRANSPOSE_PERM and ENABLE_V_TRANSPOSE_SWZ_DEBUG):
            _v_store_to_lds = _v_store_transposed_swz_b16_debug
        elif const_expr(DISABLE_V_TRANSPOSE_PERM and ENABLE_V_BPERM_PAIRED):
            _v_store_to_lds = _v_store_transposed_paired
        elif const_expr(DISABLE_V_TRANSPOSE_PERM):
            _v_store_to_lds = _v_store_transposed
        else:
            _v_store_to_lds = _v_store_transposed_perm

        def coop_load_v(tile_start, buf_id=0):
            v_base = v_buf_base(buf_id)
            for batch in range_constexpr(NUM_BATCHES_KV):
                row_offset = batch * ROWS_PER_BATCH_LOAD
                row_idx = tile_start + load_row_in_batch + row_offset
                if const_expr(KV_NEEDS_GUARD):
                    row_valid = arith.cmpi(
                        arith.CmpIPredicate.ult,
                        load_row_in_batch,
                        arith.index(BLOCK_N),
                    )
                    _if_v = scf.IfOp(row_valid)
                    with ir.InsertionPoint(_if_v.then_block):
                        g_idx = kv_global_idx(row_idx, load_col_base)
                        lds_row = load_row_in_batch + row_offset
                        vec = load_global_f16xN(v_ptr, g_idx)
                        _v_store_to_lds(v_base, lds_row, vec)
                        scf.YieldOp([])
                else:
                    g_idx = kv_global_idx(row_idx, load_col_base)
                    lds_row = load_row_in_batch + row_offset
                    vec = load_global_f16xN(v_ptr, g_idx)
                    _v_store_to_lds(v_base, lds_row, vec)

        def coop_load_v_global(tile_start):
            """Issue global loads for V, return vectors (non-blocking)."""
            vecs = []
            for batch in range_constexpr(NUM_BATCHES_KV):
                row_offset = batch * ROWS_PER_BATCH_LOAD
                row_idx = tile_start + load_row_in_batch + row_offset
                if const_expr(KV_NEEDS_GUARD):
                    row_valid = arith.cmpi(
                        arith.CmpIPredicate.ult,
                        load_row_in_batch,
                        arith.index(BLOCK_N),
                    )
                    row_idx = arith.select(row_valid, row_idx, tile_start)
                g_idx = kv_global_idx(row_idx, load_col_base)
                raw = load_global_f16xN(v_ptr, g_idx)
                if const_expr(KV_NEEDS_GUARD):
                    raw = arith.select(
                        row_valid,
                        raw,
                        arith.constant_vector(0.0, vxf16_type),
                    )
                vecs.append(raw)
            return vecs

        def coop_store_v_lds(vecs, buf_id=0):
            """Write previously-loaded V vectors to LDS."""
            v_base = v_buf_base(buf_id)
            for batch in range_constexpr(NUM_BATCHES_KV):
                row_offset = batch * ROWS_PER_BATCH_LOAD
                if const_expr(KV_NEEDS_GUARD):
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
        if const_expr(ENABLE_DMA):
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
                    if const_expr(HEAD_DIM == 64):
                        xor_mask = (row_in_tile & arith.index(0x3)) << arith.index(4)
                    else:
                        xor_mask = (row_in_tile & arith.index(0x7)) << arith.index(4)
                    unsw_col_f16 = swiz_col_f16 ^ xor_mask
                    col_byte = unsw_col_f16 * 2
                    global_row = (batch_idx * seq_len_v + tile_start
                                  + row_in_tile)
                    global_byte = (global_row * arith.index(KV_STRIDE_TOKEN * 2)
                                   + kv_head_idx * arith.index(HEAD_DIM * 2)
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
        if const_expr(ENABLE_DMA):
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
                    global_byte = (global_row * arith.index(KV_STRIDE_TOKEN * 2)
                                   + kv_head_idx * arith.index(HEAD_DIM * 2)
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

        def reduction_peer_ds(v_f32):
            if const_expr(REDUCE_MODE != "xor"):
                v_i32 = arith.ArithValue(v_f32).bitcast(T.i32)
                peer_i32 = rocdl.ds_bpermute(T.i32, lane_xor_32_byte, v_i32)
                return arith.ArithValue(peer_i32).bitcast(compute_type)

            return arith.ArithValue(v_f32).shuffle_xor(shuf_32_i32, width_i32)

        def reduction_peer_ds_lgkm(v_f32):
            if const_expr(REDUCE_MODE != "xor"):
                v_i32 = arith.ArithValue(v_f32).bitcast(T.i32)
                peer_i32 = _llvm.inline_asm(
                    ir.IntegerType.get_signless(32),
                    [lane_xor_32_byte, v_i32],
                    "ds_bpermute_b32 $0, $1, $2",
                    "=v,v,v",
                    has_side_effects=True,
                )
                _llvm.inline_asm(
                    None,
                    [],
                    "s_waitcnt lgkmcnt(0)",
                    "",
                    has_side_effects=True,
                )
                return arith.ArithValue(peer_i32).bitcast(compute_type)

            return arith.ArithValue(v_f32).shuffle_xor(shuf_32_i32, width_i32)

        def reduction_peer_max(v_f32):
            if const_expr(REDUCE_MODE in ("ds_bpermute_lgkm", "ds_bpermute_lgkm_max")):
                return reduction_peer_ds_lgkm(v_f32)
            return reduction_peer_ds(v_f32)

        def reduction_peer_sum(v_f32):
            if const_expr(REDUCE_MODE in ("ds_bpermute_lgkm", "ds_bpermute_lgkm_sum")):
                return reduction_peer_ds_lgkm(v_f32)
            return reduction_peer_ds(v_f32)

        # ---- KV loop upper bound ----
        # Non-causal with mask: iterate over full seq_len.
        kv_upper = seq_len_v

        # Loop-carried: [m_old, l_old, o_acc_chunks..., (buf_id if DMA dbuf)]
        _use_dma_dbuf = ENABLE_DMA and not ENABLE_PREFETCH_3BUF
        init_args = [c_neg_inf, c_zero_f]
        for _ in range_constexpr(D_CHUNKS):
            init_args.append(c_zero_v16f32)
        if const_expr(_use_dma_dbuf):
            init_args.append(arith.index(0))
            coop_dma_k(arith.index(0), buf_id=0)
        if const_expr(ENABLE_K_INTERBLOCK):
            _k_ib_init = coop_load_k_global(arith.index(0))
            init_args.extend(_k_ib_init)

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
            if const_expr(ENABLE_K_INTERBLOCK):
                _k_ib_offset = 2 + D_CHUNKS + (1 if _use_dma_dbuf else 0)
                _k_interblock_vecs = [
                    inner_iter_args[_k_ib_offset + i]
                    for i in range(NUM_BATCHES_KV)
                ]
            preload_k_count = (
                NUM_PREFETCH_K if NUM_PREFETCH_K < N_SUBTILES else N_SUBTILES
            )

            if const_expr(ENABLE_PREFETCH_3BUF):
                for pre_k in range_constexpr(preload_k_count):
                    pre_k_slot = CK_LDS_SEQ[pre_k % len(CK_LDS_SEQ)] % NUM_PREFETCH_K
                    pre_k_start = kv_block_start + pre_k * BLOCK_N
                    if const_expr(ENABLE_DMA):
                        coop_dma_k(pre_k_start, pre_k_slot)
                    else:
                        coop_load_k(pre_k_start, pre_k_slot)
                if const_expr(ENABLE_DMA):
                    rocdl.s_waitcnt(0)
                else:
                    rocdl.sched_group_barrier(rocdl.mask_vmem_rd, 1, 0)
                gpu.barrier()

            for kv_sub in range_constexpr(N_SUBTILES):
                kv_start = kv_block_start + kv_sub * BLOCK_N

                if const_expr(ENABLE_PREFETCH_3BUF):
                    k_slot = CK_LDS_SEQ[kv_sub % len(CK_LDS_SEQ)] % NUM_PREFETCH_K
                elif const_expr(_use_dma_dbuf):
                    if const_expr(kv_sub % 2 == 0):
                        _k_buf_id = _cur_buf_id
                    else:
                        _k_buf_id = arith.index(1) - _cur_buf_id
                    rocdl.s_waitcnt(0)
                    gpu.barrier()
                    _next_k_buf_id = arith.index(1) - _k_buf_id
                    if const_expr(kv_sub + 1 < N_SUBTILES):
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
                elif const_expr(ENABLE_K_INTERBLOCK):
                    k_slot = 0
                    _waitcnt_vm_n(0)
                    coop_store_k_lds(_k_interblock_vecs, k_slot)
                    gpu.barrier()
                elif const_expr(ENABLE_K_DWORD_DMA):
                    k_slot = 0
                    coop_dma_k_dword(kv_start, k_slot)
                    _waitcnt_vm_n(0)
                    gpu.barrier()
                elif const_expr(ENABLE_K_CHUNK32):
                    k_slot = 0
                else:
                    k_slot = 0
                    coop_load_k(kv_start, k_slot)
                    gpu.barrier()
                if const_expr(not _use_dma_dbuf):
                    k_base = k_buf_base(k_slot)

                if const_expr(not USE_HW_TR or (not ENABLE_DMA and not ENABLE_PREFETCH_3BUF)):
                    _v_vecs_prefetch = coop_load_v_global(kv_start)

                if const_expr(ENABLE_DMA and not ENABLE_PREFETCH_3BUF):
                    coop_dma_v(kv_start, 0)
                    rocdl.sched_barrier(0)

                # Pre-load mask bits before GEMM1 to overlap with MFMA pipeline
                if const_expr(HAS_MASK):
                    _mask_row_base = (batch_idx * mask_stride_b_idx
                                      + q_row_safe * mask_stride_s_idx)
                    _kv_word_lo = kv_start // arith.index(32)
                    _kv_word_hi = _kv_word_lo + arith.index(1)
                    _mask_bits_lo = _gep_load_u32(
                        mask_ptr, _mask_row_base + _kv_word_lo)
                    _mask_bits_hi = _gep_load_u32(
                        mask_ptr, _mask_row_base + _kv_word_hi)

                s_acc_lo = c_zero_v16f32
                s_acc_hi = c_zero_v16f32
                # ==== GEMM1: bulk-read all K packs, then pipeline MFMAs ====
                k_hi_offset = K_SUB_N * K_STRIDE

                def _run_qk_mfma(k_steps_this, q_step_base):
                    if const_expr(ENABLE_K_PAIRPACK_B128):
                        def _k_pair_idx_lo(pair_idx):
                            col = _pairpack_half_col(pair_idx)
                            col = _k_pairpack_swizzle(lane_mod_32, col)
                            return k_base + lane_mod_32 * K_STRIDE + col

                        def _k_pair_idx_hi(pair_idx):
                            col = _pairpack_half_col(pair_idx)
                            col = _k_pairpack_swizzle(lane_mod_32, col)
                            return (k_base + k_hi_offset
                                    + lane_mod_32 * K_STRIDE + col)

                        cur_pair_lo = vector.load_op(
                            v8f16_type, lds_kv, [_k_pair_idx_lo(0)])
                        cur_pair_hi = vector.load_op(
                            v8f16_type, lds_kv, [_k_pair_idx_hi(0)])
                        rocdl.sched_barrier(0)
                        _s_acc_lo = s_acc_lo
                        _s_acc_hi = s_acc_hi
                        for pair_idx in range_constexpr(k_steps_this // 2):
                            if const_expr(pair_idx + 1 < k_steps_this // 2):
                                nxt_pair_lo = vector.load_op(
                                    v8f16_type, lds_kv, [_k_pair_idx_lo(pair_idx + 1)])
                                nxt_pair_hi = vector.load_op(
                                    v8f16_type, lds_kv, [_k_pair_idx_hi(pair_idx + 1)])
                            k0_lo, k1_lo = _split_v8_to_v4_pair(cur_pair_lo)
                            k0_hi, k1_hi = _split_v8_to_v4_pair(cur_pair_hi)
                            q_step = q_step_base + pair_idx * 2
                            _s_acc_lo = mfma_acc(
                                k0_lo, q_b_packs[q_step], _s_acc_lo)
                            _s_acc_hi = mfma_acc(
                                k0_hi, q_b_packs[q_step], _s_acc_hi)
                            _s_acc_lo = mfma_acc(
                                k1_lo, q_b_packs[q_step + 1], _s_acc_lo)
                            _s_acc_hi = mfma_acc(
                                k1_hi, q_b_packs[q_step + 1], _s_acc_hi)
                            if const_expr(pair_idx + 1 < k_steps_this // 2):
                                cur_pair_lo = nxt_pair_lo
                                cur_pair_hi = nxt_pair_hi
                        return _s_acc_lo, _s_acc_hi

                    # K_PAD>0 path uses padded stride (no XOR); K_PAD==0 uses XOR swizzle.
                    if const_expr(K_PAD > 0 or DISABLE_K_SWIZZLE):
                        def _k_idx_lo(ks):
                            col = arith.index(ks * K_STEP_QK) + lane_div_32 * MFMA_LANE_K
                            return k_base + lane_mod_32 * K_STRIDE + col

                        def _k_idx_hi(ks):
                            col = arith.index(ks * K_STEP_QK) + lane_div_32 * MFMA_LANE_K
                            return (k_base + k_hi_offset
                                    + lane_mod_32 * K_STRIDE + col)
                    else:
                        # XOR swizzle is valid only when the logical K row is wide enough.
                        k_swz_mask = (lane_mod_32 & arith.index(0x7)) << arith.index(4)

                        def _k_idx_lo(ks):
                            col = arith.index(ks * K_STEP_QK) + lane_div_32 * MFMA_LANE_K
                            return k_base + lane_mod_32 * K_STRIDE + (col ^ k_swz_mask)

                        def _k_idx_hi(ks):
                            col = arith.index(ks * K_STEP_QK) + lane_div_32 * MFMA_LANE_K
                            return (k_base + k_hi_offset
                                    + lane_mod_32 * K_STRIDE + (col ^ k_swz_mask))

                    _QK_PREFETCH_DEPTH = QK_PREFETCH_DEPTH
                    if const_expr(k_steps_this < _QK_PREFETCH_DEPTH):
                        _QK_PREFETCH_DEPTH = k_steps_this
                    k_packs_lo = [None] * k_steps_this
                    k_packs_hi = [None] * k_steps_this
                    for p in range_constexpr(_QK_PREFETCH_DEPTH):
                        k_packs_lo[p] = vector.load_op(
                            mfma_pack_type, lds_kv, [_k_idx_lo(p)])
                        k_packs_hi[p] = vector.load_op(
                            mfma_pack_type, lds_kv, [_k_idx_hi(p)])

                    rocdl.sched_barrier(0)
                    _s_acc_lo = s_acc_lo
                    _s_acc_hi = s_acc_hi
                    for ks in range_constexpr(k_steps_this):
                        q_step = q_step_base + ks
                        _s_acc_lo = mfma_acc(
                            k_packs_lo[ks], q_b_packs[q_step], _s_acc_lo)
                        _s_acc_hi = mfma_acc(
                            k_packs_hi[ks], q_b_packs[q_step], _s_acc_hi)
                        if const_expr(ks + _QK_PREFETCH_DEPTH < k_steps_this):
                            k_packs_lo[ks + _QK_PREFETCH_DEPTH] = vector.load_op(
                                mfma_pack_type, lds_kv,
                                [_k_idx_lo(ks + _QK_PREFETCH_DEPTH)])
                            k_packs_hi[ks + _QK_PREFETCH_DEPTH] = vector.load_op(
                                mfma_pack_type, lds_kv,
                                [_k_idx_hi(ks + _QK_PREFETCH_DEPTH)])
                    return _s_acc_lo, _s_acc_hi

                if const_expr(ENABLE_K_CHUNK32):
                    K_CHUNK_STEPS_QK = 32 // K_STEP_QK
                    for k_chunk in range_constexpr(2):
                        if const_expr(k_chunk > 0):
                            _llvm.inline_asm(
                                None,
                                [],
                                "s_waitcnt lgkmcnt(0)",
                                "",
                                has_side_effects=True,
                            )
                        coop_load_k(kv_start, k_slot, k_chunk * 32)
                        gpu.barrier()
                        s_acc_lo, s_acc_hi = _run_qk_mfma(
                            K_CHUNK_STEPS_QK, k_chunk * K_CHUNK_STEPS_QK)
                else:
                    s_acc_lo, s_acc_hi = _run_qk_mfma(K_STEPS_QK, 0)

                # ==== Online softmax over 64 KV positions ====
                s_raw_lo = []
                s_raw_hi = []
                for r in range_constexpr(16):
                    s_raw_lo.append(vector.extract(
                        s_acc_lo, static_position=[r], dynamic_position=[]))
                    s_raw_hi.append(vector.extract(
                        s_acc_hi, static_position=[r], dynamic_position=[]))

                # ==== Apply free-form attention mask (bit-packed) ====
                if const_expr(HAS_MASK):
                    rocdl.sched_barrier(0)
                if const_expr(HAS_MASK):
                    mask_bits_lo = _mask_bits_lo
                    mask_bits_hi = _mask_bits_hi

                    c_zero_i32 = arith.constant(0, type=T.i32)
                    c_mask_penalty = arith.constant(-1.0e6, type=T.f32)
                    c_zero_f32_scalar = arith.constant(0.0, type=T.f32)
                    _lane_bit_base = arith.index_cast(
                        T.i32, lane_div_32 * arith.index(4))
                    _mask_lo_ps = arith.ShRUIOp(mask_bits_lo, _lane_bit_base).result
                    _mask_hi_ps = arith.ShRUIOp(mask_bits_hi, _lane_bit_base).result

                    for grp in range_constexpr(4):
                        for sub in range_constexpr(4):
                            r = grp * 4 + sub
                            _bit_mask = arith.constant(1 << (grp * 8 + sub), type=T.i32)
                            is_attend_lo = arith.cmpi(
                                arith.CmpIPredicate.ne,
                                arith.AndIOp(_mask_lo_ps, _bit_mask).result,
                                c_zero_i32)
                            m_val_lo = arith.select(
                                is_attend_lo, c_zero_f32_scalar, c_mask_penalty)
                            s_raw_lo[r] = arith.AddFOp(
                                s_raw_lo[r], m_val_lo,
                                fastmath=fm_fast).result
                            is_attend_hi = arith.cmpi(
                                arith.CmpIPredicate.ne,
                                arith.AndIOp(_mask_hi_ps, _bit_mask).result,
                                c_zero_i32)
                            m_val_hi = arith.select(
                                is_attend_hi, c_zero_f32_scalar, c_mask_penalty)
                            s_raw_hi[r] = arith.AddFOp(
                                s_raw_hi[r], m_val_hi,
                                fastmath=fm_fast).result

                if const_expr(HAS_MASK):
                    rocdl.sched_barrier(0)
                elif const_expr(ENABLE_NOMASK_SOFTMAX_BARRIER):
                    rocdl.sched_barrier(NOMASK_SOFTMAX_BARRIER_MASK)
                _max_fm = {"fastmath": fm_fast}
                local_max = s_raw_lo[0]
                for r in range_constexpr(15):
                    local_max = arith.MaxNumFOp(local_max, s_raw_lo[r + 1], **_max_fm).result
                for r in range_constexpr(16):
                    local_max = arith.MaxNumFOp(local_max, s_raw_hi[r], **_max_fm).result
                peer_max = reduction_peer_max(local_max)
                row_max = arith.MaxNumFOp(local_max, peer_max, **_max_fm).result
                m_new_raw = arith.MaxNumFOp(m_running, row_max, **_max_fm).result

                diff_m_raw = arith.SubFOp(m_running, m_new_raw, fastmath=fm_fast).result
                diff_m_scaled = arith.MulFOp(diff_m_raw, c_sm_scale_log2e, fastmath=fm_fast).result
                # rocdl.exp2 lowers to single v_exp_f32; arith/math.exp2 expands
                # to v_ldexp+v_cmp+v_cndmask (see SageAttention recipe: -38% VALU).
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

                peer_sum = reduction_peer_sum(local_sum)
                tile_sum = arith.AddFOp(local_sum, peer_sum, fastmath=fm_fast).result
                l_corr = arith.MulFOp(corr, l_running, fastmath=fm_fast).result
                l_new = arith.AddFOp(l_corr, tile_sum, fastmath=fm_fast).result

                # ==== Rescale O accumulators (v_pk_mul_f32: 16 → 8 VALU per v16) ====
                if const_expr(not USE_HW_TR):
                    if const_expr(EARLY_RESCALE_ALL):
                        for dc_r in range_constexpr(D_CHUNKS):
                            o_accs[dc_r] = _pk_mul_f32_v16(
                                o_accs[dc_r], corr)
                    else:
                        o_accs[0] = _pk_mul_f32_v16(o_accs[0], corr)
                else:
                    for dc in range_constexpr(D_CHUNKS):
                        o_accs[dc] = _pk_mul_f32_v16(o_accs[dc], corr)

                if const_expr(ENABLE_PREFETCH_3BUF and (kv_sub + preload_k_count) < N_SUBTILES):
                    next_k_sub = kv_sub + preload_k_count
                    next_k_start = kv_block_start + next_k_sub * BLOCK_N
                    next_k_slot = (
                        CK_LDS_SEQ[next_k_sub % len(CK_LDS_SEQ)] % NUM_PREFETCH_K
                    )
                    if const_expr(ENABLE_DMA):
                        coop_dma_k(next_k_start, next_k_slot)
                    else:
                        coop_load_k(next_k_start, next_k_slot)

                if const_expr(ENABLE_PREFETCH_3BUF):
                    v_slot = CK_LDS_SEQ[kv_sub % len(CK_LDS_SEQ)] % NUM_PREFETCH_V
                    v_base = v_buf_base(v_slot)
                    coop_load_v(kv_start, v_slot)
                    rocdl.sched_group_barrier(rocdl.mask_dswr, 1, 0)
                    gpu.barrier()
                elif const_expr(ENABLE_DMA):
                    v_base = v_buf_base(0)
                    rocdl.s_waitcnt(0)
                    gpu.barrier()
                else:
                    v_slot = 0
                    v_base = v_buf_base(v_slot)
                    _waitcnt_vm_n(0)
                    if const_expr(ENABLE_KV_LDS_OVERLAY):
                        _llvm.inline_asm(
                            None,
                            [],
                            "s_waitcnt lgkmcnt(0)",
                            "",
                            has_side_effects=True,
                        )
                        gpu.barrier()
                    coop_store_v_lds(_v_vecs_prefetch, v_slot)
                    if const_expr(ENABLE_K_INTERBLOCK):
                        _next_kv_start = kv_block_start + arith.index(BLOCK_N_OUT)
                        _next_kv_safe = arith.MinSIOp(
                            _next_kv_start,
                            kv_upper - arith.index(BLOCK_N)).result
                        _next_kv_safe = arith.MaxSIOp(
                            _next_kv_safe, arith.index(0)).result
                        _k_interblock_next = coop_load_k_global(_next_kv_safe)
                    rocdl.sched_barrier(0)
                    gpu.barrier()

                # ==== Build P packs for lo and hi halves ====
                if const_expr(dtype_str == "bf16" and not USE_K16):
                    p_packs_lo = []
                    p_packs_hi = []
                    for pks in range_constexpr(PV_K_STEPS):
                        p_base = pks * 4
                        p_packs_lo.append(bf16_trunc_pack_v4(
                            p_vals_lo[p_base:p_base+4]))
                        p_packs_hi.append(bf16_trunc_pack_v4(
                            p_vals_hi[p_base:p_base+4]))
                elif const_expr(dtype_str == "bf16" and USE_K16):
                    p_packs_lo = []
                    p_packs_hi = []
                    for pks in range_constexpr(PV_K_STEPS):
                        p_base = pks * 8
                        p_packs_lo.append(bf16_trunc_pack_v8(
                            p_vals_lo[p_base:p_base+8]))
                        p_packs_hi.append(bf16_trunc_pack_v8(
                            p_vals_hi[p_base:p_base+8]))
                else:
                    if const_expr(FP16_PKRTZ_PACK and not USE_K16):
                        p_packs_lo = []
                        p_packs_hi = []
                        for pks in range_constexpr(PV_K_STEPS):
                            p_base = pks * 4
                            p_packs_lo.append(fp16_pkrtz_pack_v4(
                                p_vals_lo[p_base:p_base+4]))
                            p_packs_hi.append(fp16_pkrtz_pack_v4(
                                p_vals_hi[p_base:p_base+4]))
                    else:
                        p_f16_lo = []
                        p_f16_hi = []
                        for r in range_constexpr(16):
                            p_f16_lo.append(arith.trunc_f(elem_type, p_vals_lo[r]))
                            p_f16_hi.append(arith.trunc_f(elem_type, p_vals_hi[r]))

                        if const_expr(USE_K16):
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

                # Build flat (dc, pks) schedule for interleaved GEMM2.
                _steps = [(dc, pks)
                          for dc in range(D_CHUNKS)
                          for pks in range(PV_K_STEPS)]
                TOTAL_PV = len(_steps)

                def _read_v_pack(step_idx):
                    dc, pks = _steps[step_idx]
                    if const_expr(USE_HW_TR):
                        d_col = (arith.index(dc * D_CHUNK)
                                 + tr_col_half * 16 + tr_col_sub * 4)
                        k_row = (arith.index(pks * PV_K_STEP)
                                 + lane_div_32 * 4 + tr_k_group)
                        _d_col_eff = _v_swizzle(k_row, d_col) if ENABLE_DMA else d_col
                        lds_lo = v_base + k_row * V_STRIDE + _d_col_eff
                        lds_hi = lds_lo + arith.index(K_SUB_N * V_STRIDE)
                        if const_expr(USE_K16):
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

                # Sage V12-V17: pre-load ALL V values into SSA, then pure MFMA
                # inner loop. Lets the compiler globally schedule the dense
                # MFMA block (no intervening LDS reads constraining the
                # pipeline). VGPR pressure stays under 3-wave limit because we
                # were already AccumVGPR-bound at 2 waves/SIMD.
                v_los = [None] * TOTAL_PV
                v_his = [None] * TOTAL_PV
                if const_expr(ENABLE_V_PAIRPACK_B128):
                    for dc in range_constexpr(D_CHUNKS):
                        d_pos = arith.index(dc * D_CHUNK) + lane_mod_32
                        for pair_idx in range_constexpr(PV_K_STEPS // 2):
                            n_base = _pairpack_half_col(pair_idx)
                            v_lo_idx = v_base + d_pos * VT_STRIDE + n_base
                            v_hi_idx = v_lo_idx + arith.index(K_SUB_N)
                            vlo8 = vector.load_op(v8f16_type, lds_kv, [v_lo_idx])
                            vhi8 = vector.load_op(v8f16_type, lds_kv, [v_hi_idx])
                            vlo0, vlo1 = _split_v8_to_v4_pair(vlo8)
                            vhi0, vhi1 = _split_v8_to_v4_pair(vhi8)
                            si0 = dc * PV_K_STEPS + pair_idx * 2
                            v_los[si0] = vlo0
                            v_his[si0] = vhi0
                            v_los[si0 + 1] = vlo1
                            v_his[si0 + 1] = vhi1
                else:
                    for si in range_constexpr(TOTAL_PV):
                        v_los[si], v_his[si] = _read_v_pack(si)

                # All-upfront rescale of o_accs[1..D_CHUNKS-1] before the pure
                # MFMA loop (interleaved deferred rescale regressed; see v8).
                if const_expr(not USE_HW_TR and not EARLY_RESCALE_ALL):
                    for dc_r in range_constexpr(D_CHUNKS - 1):
                        o_accs[dc_r + 1] = _pk_mul_f32_v16(
                            o_accs[dc_r + 1], corr)

                # ==== GEMM2: O += V^T_lo @ P_lo + V^T_hi @ P_hi (pure MFMA) ====
                if const_expr(PV_MFMA_HINT > 0):
                    rocdl.sched_mfma(PV_MFMA_HINT)
                for si in range_constexpr(TOTAL_PV):
                    dc, pks = _steps[si]
                    o_accs[dc] = mfma_acc(
                        v_los[si], p_packs_lo[pks], o_accs[dc])
                    o_accs[dc] = mfma_acc(
                        v_his[si], p_packs_hi[pks], o_accs[dc])

                m_running = m_new_raw
                l_running = l_new

            _yield_args = [m_running, l_running] + o_accs
            if const_expr(_use_dma_dbuf):
                if const_expr(N_SUBTILES % 2 == 1):
                    _yield_args.append(arith.index(1) - _cur_buf_id)
                else:
                    _yield_args.append(_cur_buf_id)
            if const_expr(ENABLE_K_INTERBLOCK):
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
            # MFMA f32_32x32x8 output layout (per lane, 16 vals):
            #   col = lane_div_32 * 4 + (r // 4) * 8 + (r % 4)
            # Each group of 4 consecutive r ∈ [4g, 4g+4) maps to 4 contiguous
            # cols (col_base + 0..3) ⇒ pack into v4bf16 and emit one
            # buffer_store_dwordx2 instead of 16 scalar bf16 stores.
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

    @flyc.jit
    def launch_flash_attn_func(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        O: fx.Tensor,
        Mask: fx.Tensor,
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

        launcher = flash_attn_func_kernel(Q, K, V, O, Mask, mask_stride_b, mask_stride_s, seq_len)

        if waves_per_eu is not None:
            _wpe = int(waves_per_eu)
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
    _fmha_compile_hints = {
        "fast_fp_math": fast_fp_math,
        "unsafe_fp_math": unsafe_fp_math,
        "llvm_options": {
            "enable-post-misched": True,
            "lsr-drop-solution": True,
            "amdgpu-early-inline-all": True,
        },
    }

    def _launch(*args, **kwargs):
        with CompilationContext.compile_hints(_fmha_compile_hints):
            return launch_flash_attn_func(*args, **kwargs)

    def _compile(Q, K, V, O, Mask, mask_stride_b, mask_stride_s, batch_size, seq_len, stream=None):
        with CompilationContext.compile_hints(_fmha_compile_hints):
            return flyc.compile(
                launch_flash_attn_func, Q, K, V, O, Mask, mask_stride_b, mask_stride_s,
                batch_size, seq_len, fx.Stream(stream))

    _launch.compile = _compile

    return _launch


build_flash_attn_func_module = build_flash_attn_func_module_primary
