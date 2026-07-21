# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors

"""MoE GEMM stage1/stage2 kernel implementations (FlyDSL MFMA FP8).

TUNED FOR: AMD MI308X (CDNA3 / gfx942), FP8 PTPC Fused MoE checkpoint.

Scope:
- E=512, topk=10, model_dim=4096, inter_dim=256
- tokens=1/16/32/64/128/256/512
- FP8 GEMM inputs, BF16 stage/output activations, F32 scales, I32 metadata

This file is archived from omoExplore proj007 task66. It intentionally preserves
the task64 promoted source state plus task65 documentation-only negative probe
boundary. The checkpoint is not full-gate complete: stage1 token512 is the only
row passing the 5 percent gate at task66.

Related docs:
- docs/amd/cdna3/mi308x/ref-docs/flydsl/cdna3-fused-moe-fp8-ptpc-pause-checkpoint.md
- docs/amd/cdna3/mi308x/pitfalls/flydsl/fused-moe-fp8-ptpc-pitfalls.md

This module intentionally contains the **kernel builder code** for:
- `moe_gemm1` (stage1)
- `moe_gemm2` (stage2)

It is extracted from `tests/kernels/test_moe_gemm.py` so that:
- `kernels/` holds the implementation
- `tests/` holds correctness/perf harnesses
"""

import logging
import os
import functools
from contextlib import contextmanager

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith
from flydsl.expr import gpu, buffer_ops, vector, rocdl
from flydsl.expr import range_constexpr, const_expr
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

try:
    from flydsl.runtime.device import (
        supports_bf16_global_atomics,
        bf16_global_atomics_arch_description,
    )
except ImportError:
    # Backward compatibility for runtime.device versions that only expose get_rocm_arch.
    def supports_bf16_global_atomics(arch: str) -> bool:
        return str(arch).startswith(("gfx94", "gfx95", "gfx12"))

    def bf16_global_atomics_arch_description() -> str:
        return "gfx94+/gfx95+/gfx12+"


from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm, scf, memref
from flydsl.expr.typing import T


from .mfma_preshuffle_pipeline import (
    buffer_copy_gmem16_dwordx4,
    lds_store_4b_xor16,
    lds_store_8b_xor16,
    lds_store_16b_xor16,
    make_preshuffle_b_layout,
    load_b_pack_k32,
    load_b_packs_k64,
    load_b_packs_k64_strided,
    load_b_raw_w4a16,
    unpack_b_w4a16,
    load_b_raw_w4a16_groupwise,
    extract_bf16_scale,
    tile_chunk_coord_i32,
    swizzle_xor16,
    crd2idx,
)
from .mfma_epilogues import c_shuffle_epilog, default_epilog, mfma_epilog


@contextmanager
def _if_then(if_op):
    """Compat helper for SCF IfOp then-region across old/new Python APIs."""
    with ir.InsertionPoint(if_op.then_block):
        try:
            yield if_op.then_block
        finally:
            blk = if_op.then_block
            if (not blk.operations) or not isinstance(blk.operations[-1], scf.YieldOp):
                scf.YieldOp([])


@contextmanager
def _if_else(if_op):
    """Compat helper for SCF IfOp else-region across old/new Python APIs."""
    if getattr(if_op, "else_block", None) is None:
        raise RuntimeError("IfOp has no else block")
    with ir.InsertionPoint(if_op.else_block):
        try:
            yield if_op.else_block
        finally:
            blk = if_op.else_block
            if (not blk.operations) or not isinstance(blk.operations[-1], scf.YieldOp):
                scf.YieldOp([])


def _inline_barrier(vmcnt=63, lgkmcnt=63):
    """Emit a targeted waitcnt plus s_barrier without LLVM's conservative barrier wait."""
    parts = []
    if vmcnt < 63 or lgkmcnt < 63:
        waits = []
        if vmcnt < 63:
            waits.append(f"vmcnt({vmcnt})")
        if lgkmcnt < 63:
            waits.append(f"lgkmcnt({lgkmcnt})")
        parts.append("s_waitcnt " + " ".join(waits))
    parts.append("s_barrier")
    llvm.InlineAsmOp(
        res=None,
        operands_=[],
        asm_string="\n".join(parts),
        constraints="",
        has_side_effects=True,
        is_align_stack=False,
    )


@functools.lru_cache(maxsize=1024)
def compile_moe_gemm1(
    *,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    # NOTE: aiter swap passes these for API symmetry; stage1 uses dynamic memrefs so they are ignored.
    doweight_stage1: bool,
    in_dtype: str = "fp8",
    group_size: int = -1,
    out_dtype: str = "f16",
    use_cshuffle_epilog: bool | None = None,
    scale_is_bf16: bool = False,
    k_batch: int = 1,
    b_nt: int = 0,
    waves_per_eu: int = 0,
    assume_valid_grid: bool = False,
):
    """Compile stage1 kernel (`moe_gemm1`) and return the compiled executable.

    in_dtype:
      - "fp8": X/W are fp8
      - "fp16": X/W are fp16
      - "bf16": X/W are bf16
      - "int8": X/W are int8 (X is [tokens, K])
      - "int8smooth": X/W are int8, but X is pre-expanded to [tokens*topk, K] with per-(token,slot)
        quant scales (used to emulate MoE smoothquant behavior where each (token,slot)->expert route can
        have a distinct input scaling before quantization).
      - "int4": W4A8 path: X is int8, W is packed int4 (2 values per byte) unpacked to int8 in-kernel
      - "int4_bf16": W4A16 path: X is bf16, W is packed int4 unpacked to bf16 in-kernel
    scale_is_bf16: When True, groupwise scales are bf16 (halves scale bandwidth).
    k_batch: Split-K factor. When >1, K is partitioned across k_batch CTAs that
      atomically accumulate gate/up partials. Caller must pre-zero output.
    """

    gpu_arch = get_hip_arch()
    allocator = SmemAllocator(None, arch=gpu_arch)
    _state = {}  # legacy; kept until stage2/reduction are migrated

    _valid_dtypes = ("fp8", "fp16", "bf16", "int8", "int8smooth", "int4", "int4_bf16")
    if in_dtype not in _valid_dtypes:
        raise ValueError(f"in_dtype must be one of {_valid_dtypes}, got {in_dtype!r}")
    is_int4_bf16 = (
        in_dtype == "int4_bf16"
    )  # W4A16: bf16 activations, packed int4 weights
    is_f16 = in_dtype == "fp16"
    is_bf16 = is_int4_bf16 or in_dtype == "bf16"
    is_f16_or_bf16 = is_f16 or is_bf16
    needs_scale_w = (not is_f16_or_bf16) or is_int4_bf16
    elem_bytes = 2 if is_f16_or_bf16 else 1
    if out_dtype not in ("f16", "bf16"):
        raise ValueError(f"out_dtype must be 'f16' or 'bf16', got {out_dtype!r}")

    # NOTE: don't materialize MLIR types outside an active MLIR Context.
    def out_mlir():
        return (lambda ty: ty() if callable(ty) else ty)(
            T.f16 if out_dtype == "f16" else T.bf16
        )

    tile_k_bytes = int(tile_k) * int(elem_bytes)
    # K64-byte micro-step: always 64 bytes per `ku`. For fp16 this is 32 elements.
    if (tile_k_bytes % 64) != 0:
        raise ValueError(
            f"tile_k_bytes must be divisible by 64, got tile_k_bytes={tile_k_bytes} "
            f"(tile_k={tile_k}, elem_bytes={elem_bytes})"
        )
    is_int4 = in_dtype == "int4"
    # INT4 here means W4A8: X is int8, W is packed int4 and unpacked to int8 in-kernel.
    is_int8 = (in_dtype == "int8") or is_int4
    x_is_token_slot = in_dtype == "int8smooth"
    # "int8smooth" still uses int8 MFMA, but X/scale_x are provided per (token,slot).
    is_int8 = is_int8 or x_is_token_slot

    # w_is_int4: True for any variant where weights are packed int4.
    w_is_int4 = is_int4 or is_int4_bf16

    # Group-wise scale support for W4A16
    # NOTE: Only group_size=32 is supported due to int4 preshuffle layout constraints.
    use_groupwise_scale = w_is_int4 and group_size > 0
    if use_groupwise_scale and group_size != 32:
        raise ValueError(
            f"FlyDSL groupwise scale only supports group_size=32, got {group_size}. "
            f"This is due to int4 preshuffle layout constraints. "
            f"Please use Triton kernel for other group sizes."
        )
    is_int4_bf16_groupwise = is_int4_bf16 and use_groupwise_scale
    num_groups = model_dim // group_size if use_groupwise_scale else 1
    _scale_is_bf16 = scale_is_bf16 and use_groupwise_scale
    experts * (2 * inter_dim) * num_groups
    # For groupwise scale, weight scale is applied per-group in the K loop,
    # so epilogue can skip weight scale multiplication (uses 1.0 for sw).

    _is_gfx950 = "gfx95" in get_hip_arch()
    _has_cvt_off_f32_i4 = hasattr(rocdl, "cvt_off_f32_i4")
    use_gfx950_cvt = is_int4_bf16 and _is_gfx950 and _has_cvt_off_f32_i4

    # Split-K validation
    _is_splitk = k_batch > 1
    if _is_splitk:
        _k_per_batch = model_dim // k_batch
        assert (
            model_dim % k_batch == 0
        ), f"model_dim={model_dim} not divisible by k_batch={k_batch}"
        assert (
            _k_per_batch % tile_k == 0
        ), f"K_per_batch={_k_per_batch} not divisible by tile_k={tile_k}"
        # The ping-pong K-loop requires an even number of K tiles (>=4).
        _k_tiles = _k_per_batch // tile_k
        assert _k_tiles >= 4 and _k_tiles % 2 == 0, (
            f"K_per_batch/tile_k={_k_tiles} must be even and >=4 for the ping-pong pipeline. "
            f"Try a different k_batch (model_dim={model_dim}, tile_k={tile_k})."
        )
    else:
        _k_per_batch = model_dim

    mfma_i32_k32 = None
    if is_int8:
        mfma_i32_k32 = getattr(rocdl, "mfma_i32_16x16x32i8", None) or getattr(
            rocdl, "mfma_i32_16x16x32_i8", None
        )
        if mfma_i32_k32 is None:
            raise AttributeError(
                "INT8 K32 MFMA op not found: expected `rocdl.mfma_i32_16x16x32i8` "
                "(or `rocdl.mfma_i32_16x16x32_i8`)."
            )

    mfma_f32_bf16_k16 = None
    if is_bf16:
        mfma_f32_bf16_k16 = getattr(rocdl, "mfma_f32_16x16x16bf16_1k", None) or getattr(
            rocdl, "mfma_f32_16x16x16_bf16_1k", None
        )
        if mfma_f32_bf16_k16 is None:
            raise AttributeError(
                "BF16 K16 MFMA op not found: expected `rocdl.mfma_f32_16x16x16bf16_1k` "
                "(or `rocdl.mfma_f32_16x16x16_bf16_1k`)."
            )

    # gfx950: use 16x16x32 MFMA for f16/bf16 (K=32 per MFMA, vs K=16 on gfx942).
    # Check if K=32 MFMA supports the (result_type, operands_list) calling convention.
    _has_k32_mfma_compat = False
    if _is_gfx950 and (is_f16 or is_bf16):
        import inspect

        _k32_fn = (
            rocdl.mfma_f32_16x16x32_bf16 if is_bf16 else rocdl.mfma_f32_16x16x32_f16
        )
        try:
            _k32_sig = inspect.signature(_k32_fn)
            _k32_params = list(_k32_sig.parameters.keys())
            # Compatible if second param is "operands" (list-based API)
            _has_k32_mfma_compat = (
                len(_k32_params) >= 2 and _k32_params[1] == "operands"
            )
        except (ValueError, TypeError):
            _has_k32_mfma_compat = False
    _use_mfma_k32 = _is_gfx950 and (is_f16 or is_bf16) and _has_k32_mfma_compat

    ir.ShapedType.get_dynamic_size()
    # W is packed int4 for W4A8/W4A16/W4A_FP8: 2 values per byte.
    (
        (experts * (2 * inter_dim) * model_dim) // 2
        if w_is_int4
        else (experts * (2 * inter_dim) * model_dim)
    )

    if (
        is_bf16
        and int(tile_m) == 16
        and int(tile_n) == 32
        and int(tile_k) in (128, 256)
    ):
        default_total_threads = "128"
    else:
        default_total_threads = "256"
    total_threads = int(os.environ.get("AITER_FLYDSL_MOE1_THREADS", default_total_threads))
    if total_threads not in (128, 256):
        raise ValueError(
            f"AITER_FLYDSL_MOE1_THREADS must be 128 or 256, got {total_threads}"
        )
    num_waves_static = total_threads // 64
    if int(tile_n) // num_waves_static < 16:
        raise ValueError(
            "tile_n per wave must cover at least one MFMA N fragment: "
            f"tile_n={tile_n}, total_threads={total_threads}, "
            f"num_waves={num_waves_static}"
        )
    bytes_x_per_tile = int(tile_m) * int(tile_k) * int(elem_bytes)
    if bytes_x_per_tile % total_threads != 0:
        raise ValueError(
            "tile_m*tile_k*elem_bytes must be divisible by "
            f"{total_threads}: tile_m={tile_m}, tile_k={tile_k}, elem_bytes={elem_bytes}"
        )
    bytes_per_thread_x = bytes_x_per_tile // total_threads
    # Keep MoE stage1 X gmem->LDS pipeline consistent with the optimized GEMM kernel:
    # split into <=16B pieces and use direct buffer_load for smaller widths.
    # (Compute the split lens inside the kernel so the code matches GEMM structure.)

    # LDS128 mode (same idea as test_preshuffle_gemm.py):
    # - LDS stride == tile_k (no extra padding) + XOR16 swizzle
    # - Use ds_{read,write}_b128 (16B) and extract 8B halves for MFMA steps
    _ck_lds128 = os.environ.get("FLYDSL_CK_LDS128", "1") in (
        "1",
        "true",
        "True",
        "YES",
        "yes",
    )
    pad_k = 0 if _ck_lds128 else 8
    lds_stride = tile_k + pad_k
    if use_cshuffle_epilog is None:
        use_cshuffle_epilog = os.environ.get("FLYDSL_MOE_STAGE1_CSHUFFLE", "1") in (
            "1",
            "true",
            "True",
            "YES",
            "yes",
        )
    use_cshuffle_epilog = bool(use_cshuffle_epilog)
    epilog_tag = "cshuffle" if use_cshuffle_epilog else "direct"
    if (
        is_bf16
        and not use_cshuffle_epilog
        and int(tile_m) == 16
        and int(tile_n) in (32, 64)
        and int(tile_k) in (128, 256)
        and total_threads in (128, 256)
    ):
        default_sched_variant = "nosched"
    elif (
        is_bf16
        and use_cshuffle_epilog
        and int(tile_m) == 16
        and int(tile_n) == 256
        and int(tile_k) == 128
        and total_threads == 256
    ):
        default_sched_variant = "split"
    else:
        default_sched_variant = "gfx942"
    sched_variant = (
        os.environ.get("AITER_FLYDSL_MOE1_SCHED", default_sched_variant)
        .strip()
        .lower()
    )
    if sched_variant not in {
        "gfx942",
        "nosched",
        "vmem2",
        "vmem2early",
        "vmem3early",
        "vmem4early",
        "split",
        "split3",
    }:
        raise ValueError(
            "AITER_FLYDSL_MOE1_SCHED must be one of "
            "{'gfx942', 'nosched', 'vmem2', 'vmem2early', 'vmem3early', 'vmem4early', "
            "'split', 'split3'}, "
            f"got {sched_variant!r}"
        )
    sched_loop_vmem = {
        "gfx942": 1,
        "nosched": 1,
        "vmem2": 2,
        "vmem2early": 2,
        "vmem3early": 3,
        "vmem4early": 4,
        "split": 1,
        "split3": 3,
    }[sched_variant]
    sched_early_vmem = (
        sched_loop_vmem
        if sched_variant.endswith("early") or sched_variant.startswith("split")
        else 1
    )
    sched_split_mfma = sched_variant.startswith("split")
    sched_disable = sched_variant == "nosched"
    sched_setprio = os.environ.get("AITER_FLYDSL_MOE1_PRIO", "0").strip() in (
        "1",
        "true",
        "True",
        "YES",
        "yes",
    )
    # Disabled for now: raw_ptr_buffer_load_lds on this dynamic BF16 X resource
    # hits an LLVM operand expansion failure on gfx942.
    use_x_dma_req = False and os.environ.get("AITER_FLYDSL_MOE1_X_DMA", "0").strip() in (
        "1",
        "true",
        "True",
        "YES",
        "yes",
    )
    use_tid_lds_req = os.environ.get("AITER_FLYDSL_MOE1_TID_LDS", "0").strip() in (
        "1",
        "true",
        "True",
        "YES",
        "yes",
    )
    fast_silu_req = os.environ.get("AITER_FLYDSL_MOE1_FAST_SILU", "0").strip() in (
        "1",
        "true",
        "True",
        "YES",
        "yes",
    )
    prefetch_a_req = os.environ.get("AITER_FLYDSL_MOE1_PREFETCH_A", "0").strip() in (
        "1",
        "true",
        "True",
        "YES",
        "yes",
    )
    tiny_row0_x_req = os.environ.get(
        "AITER_FLYDSL_MOE1_TINY_ROW0_X", "0"
    ).strip() in (
        "1",
        "true",
        "True",
        "YES",
        "yes",
    )
    stage1_row_limit_supported = (is_f16_or_bf16 or in_dtype == "fp8") and not x_is_token_slot
    tiny_row0_x_req = tiny_row0_x_req and stage1_row_limit_supported
    row_limit_x_req = int(os.environ.get("AITER_FLYDSL_MOE1_ROW_LIMIT_X", "0"))
    row_limit_x_req = (
        row_limit_x_req
        if stage1_row_limit_supported and row_limit_x_req > 0
        else 0
    )
    row_limit_epilog_req = int(os.environ.get("AITER_FLYDSL_MOE1_ROW_LIMIT_EPILOG", "0"))
    row_limit_epilog_req = (
        row_limit_epilog_req
        if stage1_row_limit_supported and not use_cshuffle_epilog and row_limit_epilog_req > 0
        else 0
    )
    prefetch_epi_tid_req = (
        os.environ.get("AITER_FLYDSL_MOE1_PREFETCH_EPI_TID", "0").strip()
        in ("1", "true", "True", "YES", "yes")
    )
    prefetch_epi_tid_req = (
        prefetch_epi_tid_req
        and stage1_row_limit_supported
        and not use_cshuffle_epilog
        and row_limit_epilog_req in (1, 4)
    )
    use_a_direct_req = os.environ.get("AITER_FLYDSL_MOE1_A_DIRECT", "0").strip() in (
        "1",
        "true",
        "True",
        "YES",
        "yes",
    )
    use_a_direct_req = (
        use_a_direct_req
        and is_f16_or_bf16
        and int(tile_m) == 16
        and not _is_splitk
    )
    if use_a_direct_req:
        prefetch_a_req = False
    cshuffle_evec_req = int(os.environ.get("AITER_FLYDSL_MOE1_CSHUFFLE_EVEC", "0"))
    default_cshuffle_nlane = (
        "16"
        if (
            is_bf16
            and use_cshuffle_epilog
            and int(tile_m) == 16
            and int(tile_n) == 256
            and int(tile_k) == 128
            and total_threads == 256
        )
        else "32"
    )
    cshuffle_nlane_req = int(
        os.environ.get("AITER_FLYDSL_MOE1_CSHUFFLE_NLANE", default_cshuffle_nlane)
    )
    if cshuffle_nlane_req <= 0 or total_threads % cshuffle_nlane_req != 0:
        raise ValueError(
            "AITER_FLYDSL_MOE1_CSHUFFLE_NLANE must be a positive divisor of "
            f"{total_threads}, got {cshuffle_nlane_req}"
        )
    fast_barrier_req = os.environ.get("AITER_FLYDSL_MOE1_FAST_BARRIER", "0").strip() in (
        "1",
        "true",
        "True",
        "YES",
        "yes",
    )
    dswr_advance_req = int(os.environ.get("AITER_FLYDSL_MOE1_DSWR_ADVANCE", "2"))
    out_cache_modifier_req = int(os.environ.get("AITER_FLYDSL_MOE1_OUT_NT", "0"))
    # IMPORTANT: module name participates in FlyDSL's compile cache key.
    # Keep an explicit ABI tag so signature changes can't accidentally reuse an old binary.
    _gs_tag = f"_g{group_size}" if use_groupwise_scale else ""
    scale_tag = "_sbf16" if _scale_is_bf16 else ""
    _split_k_tag = f"_splitk{k_batch}" if _is_splitk else ""
    module_name = (
        f"mfma_moe1_{in_dtype}_{out_dtype}_{epilog_tag}"
        f"_t{tile_m}x{tile_n}x{tile_k}"
        f"{_gs_tag}{scale_tag}{_split_k_tag}"
        f"_bnt{b_nt}"
        f"_thr{total_threads}"
        f"_wpe{int(waves_per_eu)}"
        f"_sch{sched_variant}"
        f"_prio{int(sched_setprio)}"
        f"_xdma{int(use_x_dma_req)}"
        f"_tidlds{int(use_tid_lds_req)}"
        f"_fsilu{int(fast_silu_req)}"
        f"_apf{int(prefetch_a_req)}"
        f"_tr0x{int(tiny_row0_x_req)}"
        f"_rlx{int(row_limit_x_req)}"
        f"_rle{int(row_limit_epilog_req)}"
        f"_epitid{int(prefetch_epi_tid_req)}"
        f"_adir{int(use_a_direct_req)}"
        f"_cse{int(cshuffle_evec_req)}"
        f"_csn{int(cshuffle_nlane_req)}"
        f"_fbar{int(fast_barrier_req)}"
        f"_dswa{int(dswr_advance_req)}"
        f"_outnt{int(out_cache_modifier_req)}"
        f"_avg{int(assume_valid_grid)}"
        f"_abi23"  # stage1 CShuffle output dtype fix
    ).replace("-", "_")
    _cache_tag = (
        module_name,
        model_dim,
        inter_dim,
        experts,
        topk,
        tile_m,
        tile_n,
        tile_k,
        doweight_stage1,
        in_dtype,
        group_size,
        out_dtype,
        use_cshuffle_epilog,
        scale_is_bf16,
        k_batch,
        b_nt,
        waves_per_eu,
        assume_valid_grid,
        total_threads,
        sched_variant,
        sched_setprio,
        use_x_dma_req,
        use_tid_lds_req,
        fast_silu_req,
        prefetch_a_req,
        tiny_row0_x_req,
        row_limit_x_req,
        row_limit_epilog_req,
        prefetch_epi_tid_req,
        use_a_direct_req,
        cshuffle_evec_req,
        cshuffle_nlane_req,
        fast_barrier_req,
        dswr_advance_req,
        out_cache_modifier_req,
    )

    # -- LDS sizing (pure Python; no MLIR Context needed) ---------------------
    # Reuse the same LDS bytes for both:
    # - ping-pong X tiles (2 * tile_m * lds_stride bytes)
    # - optional epilogue CShuffle tile (tile_m * tile_n f16 -> 2 * tile_m * tile_n bytes)
    _use_cshuffle_epilog = bool(use_cshuffle_epilog)
    # Split-K requires CShuffle epilogue (atomic adds via store_pair callback)
    if _is_splitk:
        _use_cshuffle_epilog = True
    # bf16 split-K: use bf16 atomics (halves bandwidth, gfx950 has buffer_atomic_pk_add_bf16).
    # Other dtypes keep f32 for precision.
    _splitk_use_bf16 = _is_splitk and is_bf16
    _cshuffle_elem_bytes = 2 if (not _is_splitk or _splitk_use_bf16) else 4
    lds_x_bytes = 2 * int(tile_m) * int(lds_stride) * int(elem_bytes)
    lds_out_bytes = (
        _cshuffle_elem_bytes * int(tile_m) * int(tile_n) if _use_cshuffle_epilog else 0
    )
    lds_total_bytes = max(lds_x_bytes, lds_out_bytes)
    lds_total_elems = lds_total_bytes if elem_bytes == 1 else (lds_total_bytes // 2)

    lds_alloc_bytes = int(lds_total_elems) * int(elem_bytes)
    lds_alloc_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_alloc_offset + lds_alloc_bytes
    lds_tid_offset = 0
    if use_tid_lds_req:
        lds_tid_offset = allocator._align(allocator.ptr, 16)
        allocator.ptr = lds_tid_offset + int(tile_m) * 4
    if True:

        @flyc.kernel(known_block_size=[total_threads, 1, 1])
        def moe_gemm1(
            arg_out: fx.Tensor,
            arg_x: fx.Tensor,
            arg_w: fx.Tensor,
            arg_scale_x: fx.Tensor,
            arg_scale_w: fx.Tensor,
            arg_sorted_token_ids: fx.Tensor,
            arg_expert_ids: fx.Tensor,
            arg_sorted_weights: fx.Tensor,
            arg_max_token_ids: fx.Tensor,
            i32_tokens_in: fx.Int32,
            i32_inter_in: fx.Int32,
            i32_k_in: fx.Int32,
            i32_size_expert_ids_in: fx.Int32,
        ):
            tokens_in = arith.index_cast(T.index, i32_tokens_in)
            inter_in = arith.index_cast(T.index, i32_inter_in)
            k_in = arith.index_cast(T.index, i32_k_in)
            size_expert_ids_in = arith.index_cast(T.index, i32_size_expert_ids_in)
            # i32 versions for layout construction (fly.make_shape requires i32/i64)
            tokens_i32_v = i32_tokens_in
            k_i32_v = i32_k_in
            x_elem = (
                T.bf16
                if is_bf16
                else (T.f16 if is_f16 else (T.i8 if is_int8 else T.f8))
            )
            # For int4/int4_bf16, weights are stored as packed bytes (i8) and unpacked in-kernel.
            w_elem = (
                T.i8
                if w_is_int4
                else (
                    T.bf16
                    if is_bf16
                    else (T.f16 if is_f16 else (T.i8 if is_int8 else T.f8))
                )
            )
            scale_dtype = T.bf16 if _scale_is_bf16 else T.f32
            vec16_elems = 16 if elem_bytes == 1 else 8
            vec8_elems = 8 if elem_bytes == 1 else 4
            vec8_x = T.vec(vec8_elems, x_elem)
            vec16_x = T.vec(vec16_elems, x_elem)

            def silu(x):
                if const_expr(fast_silu_req):
                    # sigmoid(x) ~= 0.5 + 0.25*x - x^3/48, so
                    # silu(x)=x*sigmoid(x) ~= 0.5*x + 0.25*x^2 - x^4/48.
                    # This stays within the BF16 task harness tolerance while avoiding
                    # exp/rcp latency in the output epilogue.
                    x2 = x * x
                    x4 = x2 * x2
                    return (x * 0.5) + (x2 * 0.25) - (x4 * 0.020833333333333332)
                # device fast path:
                #   emu = exp(-x)  ~= exp2(log2e * (-x))  -> v_exp_f32
                #   sig = rcp(1 + emu)                   -> v_rcp_f32
                #   y = x * sig
                #
                # Using llvm.amdgcn intrinsics prevents lowering to the div_scale/div_fixup
                # sequences that introduce extra compares/cndmasks.
                t = x * (-1.4426950408889634)  # -log2(e)
                emu = rocdl.exp2(T.f32, t)
                den = 1.0 + emu
                sig = rocdl.rcp(T.f32, den)
                return x * sig

            acc_init = (
                arith.constant_vector(0, T.i32x4)
                if is_int8
                else arith.constant_vector(0.0, T.f32x4)
            )
            zero_f32_acc = (
                arith.constant_vector(0.0, T.f32x4) if is_int4_bf16_groupwise else None
            )

            # Layouts (use i32 values; fly.make_shape requires i32/i64, not index)
            fx.make_layout((tokens_i32_v, k_i32_v), stride=(k_i32_v, 1))

            # B preshuffle layout: match GEMM test helper exactly.
            c_n_total = arith.index(experts * (2 * inter_dim))
            # For packed int4 (W4A8/W4A16/W4A_FP8), kpack_bytes=8.
            kpack_bytes = 8 if w_is_int4 else 16
            w_elem_bytes = 1 if w_is_int4 else elem_bytes
            b_layout = make_preshuffle_b_layout(
                arith,
                c_n=c_n_total,
                c_k=k_in,
                kpack_bytes=kpack_bytes,
                elem_bytes=w_elem_bytes,
            )
            layout_b = b_layout.layout_b
            (k_in * arith.index(int(elem_bytes))) // fx.Index(64)

            shape_lds = fx.make_shape(tile_m, tile_k)
            stride_lds = fx.make_stride(lds_stride, 1)
            layout_lds = fx.make_layout(shape_lds, stride_lds)

            tx = gpu.thread_id("x")

            def stage_barrier():
                if const_expr(use_a_direct):
                    return
                if const_expr(fast_barrier_req):
                    _inline_barrier(vmcnt=63, lgkmcnt=0)
                else:
                    gpu.barrier()

            # Default matches Aiter launch mapping (NSwizzle==false):
            # - blockIdx.x -> N dimension (tile along inter_dim)
            # - blockIdx.y -> expert-block id / M dimension (tile along sorted M)
            by = gpu.block_id("x")  # tile along inter_dim
            bx = gpu.block_id("y")  # tile along sorted M

            if const_expr(_is_splitk):
                bz = gpu.block_id("z")  # K-batch id
                k_base_idx = bz * arith.index(_k_per_batch)
            else:
                k_base_idx = arith.index(0)

            # Block validity: compute as early as possible so invalid blocks skip all buffer-resource
            # setup, LDS pointer math, and gmem prefetch work.
            bx_m = bx * fx.Index(tile_m)
            if const_expr(assume_valid_grid):
                blk_valid = arith.cmpi(
                    arith.CmpIPredicate.eq, fx.Index(0), fx.Index(0)
                )
            else:
                maxids_rsrc = buffer_ops.create_buffer_resource(
                    arg_max_token_ids,
                    max_size=False,
                    num_records_bytes=fx.Index(4),
                )
                max_token_id_i32 = buffer_ops.buffer_load(
                    maxids_rsrc, fx.Index(0), vec_width=1, dtype=T.i32
                )
                bx_m_i32 = arith.index_cast(T.i32, bx_m)
                blk_valid = arith.cmpi(
                    arith.CmpIPredicate.ult, bx_m_i32, max_token_id_i32
                )
            # Common constants/atoms (hoisted): keep IR small like GEMM.
            # XOR16 swizzle parameter (in bytes; constant, power-of-two in our configs).
            k_blocks16 = arith.index(tile_k_bytes // 16)
            layout_tx_wave_lane = fx.make_layout((num_waves_static, 64), stride=(64, 1))
            layout_lane16 = fx.make_layout((4, 16), stride=(16, 1))

            # Everything below is gated by `blk_valid` to avoid doing buffer-resource setup and
            # gmem work for padding blocks.
            _if_blk = scf.IfOp(blk_valid)
            with _if_then(_if_blk):
                base_ptr = allocator.get_base()
                lds_x_ptr = SmemPtr(
                    base_ptr,
                    lds_alloc_offset,
                    (
                        T.bf16
                        if is_bf16
                        else (T.f16 if is_f16 else (T.i8 if is_int8 else T.f8))
                    ),
                    shape=(lds_total_elems,),
                )
                lds_x = lds_x_ptr.get()
                # Alias LDS bytes for optional CShuffle epilogue. The epilogue
                # element type follows the output dtype, not the input dtype.
                # bf16 split-K uses bf16 (2B); other split-K uses f32 (4B);
                # normal uses f16/bf16 (2B).
                _lds_out_elem_type = (
                    T.f32
                    if (_is_splitk and not _splitk_use_bf16)
                    else (T.bf16 if out_dtype == "bf16" else T.f16)
                )
                lds_out = (
                    SmemPtr(
                        base_ptr,
                        lds_x_ptr.byte_offset,
                        _lds_out_elem_type,
                        shape=(tile_m * tile_n,),
                    ).get()
                    if _use_cshuffle_epilog
                    else None
                )
                lds_tid = (
                    SmemPtr(base_ptr, lds_tid_offset, T.i32, shape=(tile_m,)).get()
                    if use_tid_lds_req
                    else None
                )
                # Buffer resources: for dynamic memrefs, provide `num_records_bytes` explicitly so
                # hardware OOB behavior is stable (otherwise it falls back to a large max size).
                c_topk = fx.Index(topk)

                # X: [tokens, k] bytes = tokens*k*elem_bytes
                x_rows = tokens_in * (c_topk if x_is_token_slot else fx.Index(1))
                x_nbytes_idx = x_rows * k_in * arith.index(int(elem_bytes))
                x_nbytes_i32 = arith.index_cast(T.i32, x_nbytes_idx)
                x_rsrc = buffer_ops.create_buffer_resource(
                    arg_x, max_size=False, num_records_bytes=x_nbytes_i32
                )

                w_rsrc = buffer_ops.create_buffer_resource(arg_w, max_size=False)

                # OUT: normal=[tokens, topk, inter] f16/bf16,
                #      split-K=[tokens*topk, 2*inter] f32 (or bf16 for bf16 split-K)
                out_elem_bytes = 4 if (_is_splitk and not _splitk_use_bf16) else 2
                if const_expr(_is_splitk):
                    out_nbytes_idx = (
                        tokens_in * c_topk * inter_in * fx.Index(2 * out_elem_bytes)
                    )
                else:
                    out_nbytes_idx = (
                        tokens_in * c_topk * inter_in * fx.Index(out_elem_bytes)
                    )
                out_rsrc = buffer_ops.create_buffer_resource(
                    arg_out, max_size=False, num_records_bytes=out_nbytes_idx
                )
                # scale_x: fp16/bf16 path ignores (implicit scale=1.0); int4_bf16 also uses 1.0.
                if const_expr(is_f16_or_bf16):
                    sx_rsrc = None
                else:
                    sx_rows = tokens_in * (c_topk if x_is_token_slot else fx.Index(1))
                    sx_nbytes_idx = sx_rows * fx.Index(4)
                    sx_rsrc = buffer_ops.create_buffer_resource(
                        arg_scale_x, max_size=False, num_records_bytes=sx_nbytes_idx
                    )
                # scale_w: fp16/bf16 (non-int4) path ignores; int4_bf16 needs dequant scale.
                if const_expr(not needs_scale_w):
                    sw_rsrc = None
                else:
                    sw_rsrc = buffer_ops.create_buffer_resource(
                        arg_scale_w, max_size=False
                    )

                sorted_rsrc = buffer_ops.create_buffer_resource(
                    arg_sorted_token_ids, max_size=False
                )
                sorted_w_rsrc = buffer_ops.create_buffer_resource(
                    arg_sorted_weights, max_size=False
                )

                if const_expr(use_tid_lds_req):
                    tid_in_range = arith.cmpi(
                        arith.CmpIPredicate.ult, tx, fx.Index(tile_m)
                    )
                    _if_tid = scf.IfOp(tid_in_range)
                    with _if_then(_if_tid):
                        tid_row = bx_m + tx
                        tid_val = buffer_ops.buffer_load(
                            sorted_rsrc, tid_row, vec_width=1, dtype=T.i32
                        )
                        tid_vec = vector.from_elements(T.vec(1, T.i32), [tid_val])
                        vector.store(tid_vec, lds_tid, [tx], alignment=4)
                    gpu.barrier()

                # expert ids: [blocks] i32 -> bytes = size_expert_ids_in*4
                expert_rsrc = buffer_ops.create_buffer_resource(
                    arg_expert_ids,
                    max_size=False,
                    num_records_bytes=(size_expert_ids_in * fx.Index(4)),
                )

                # Expert id for this M tile (keep address math in `index`)
                expert_i32 = buffer_ops.buffer_load(
                    expert_rsrc, bx, vec_width=1, dtype=T.i32
                )
                expert_idx = arith.index_cast(T.index, expert_i32)
                inter2_idx = arith.index(2 * inter_dim)
                expert_off_idx = expert_idx * inter2_idx  # index

                # ---- X gmem->reg prefetch (match preshuffle GEMM mapping) ----
                # Prefer 16B buffer-load (dwordx4). If the per-thread byte count isn't divisible by
                # 16, fall back to 8B (dwordx2) or 4B (dword) loads.
                if const_expr(is_f16_or_bf16):
                    if const_expr(bytes_per_thread_x % 16 == 0):
                        x_load_bytes = 16
                    elif const_expr(bytes_per_thread_x % 8 == 0):
                        x_load_bytes = 8
                    else:
                        raise ValueError(
                            f"[fp16/bf16] bytes_per_thread_x ({bytes_per_thread_x}) must be divisible by 8"
                        )
                else:
                    if const_expr(bytes_per_thread_x % 16 == 0):
                        x_load_bytes = 16
                    elif const_expr(bytes_per_thread_x % 8 == 0):
                        x_load_bytes = 8
                    elif const_expr(bytes_per_thread_x % 4 == 0):
                        x_load_bytes = 4
                    else:
                        raise ValueError(
                            f"bytes_per_thread_x ({bytes_per_thread_x}) must be divisible by 4 to use the dword-indexed load mapping."
                        )
                num_x_loads = bytes_per_thread_x // x_load_bytes
                chunk_i32 = x_load_bytes // 4  # dwords per chunk (1/2/4)
                use_x_dma = use_x_dma_req and is_f16_or_bf16 and x_load_bytes == 16
                use_a_direct = use_a_direct_req and x_load_bytes == 16

                c_k_div4 = (k_in * arith.index(int(elem_bytes))) // fx.Index(4)
                c_k_div4_i32 = arith.index_cast(T.i32, c_k_div4)
                fx.make_layout((tokens_i32_v, c_k_div4_i32), stride=(c_k_div4_i32, 1))
                tile_k_dwords = (int(tile_k) * int(elem_bytes)) // 4
                layout_x_tile_div4 = fx.make_layout(
                    (tile_m, tile_k_dwords), stride=(tile_k_dwords, 1)
                )
                c_chunk_i32 = fx.Index(chunk_i32)
                tx_i32_base = tx * c_chunk_i32
                mask24 = fx.Int32(0xFFFFFF)
                tokens_i32 = arith.index_cast(T.i32, tokens_in)
                topk_i32 = fx.Int32(topk)

                def x_tile_chunk_coord_i32(i: int):
                    return tile_chunk_coord_i32(
                        arith,
                        tx_i32_base=tx_i32_base,
                        i=i,
                        total_threads=total_threads,
                        layout_tile_div4=layout_x_tile_div4,
                        chunk_i32=chunk_i32,
                    )

                # decode token once (per thread's M-slice) and build a base row offset.
                x_row_base_div4 = []
                x_col_local_i32 = []
                x_row_local = []
                x_row_valid = []
                for i in range_constexpr(num_x_loads):
                    row_local, col_local_i32 = x_tile_chunk_coord_i32(i)
                    x_row_local.append(row_local)
                    x_col_local_i32.append(col_local_i32)

                    if const_expr(tiny_row0_x_req):
                        x_row_base_div4.append(fx.Index(0))
                        x_row_valid.append(
                            arith.cmpi(
                                arith.CmpIPredicate.eq, row_local, fx.Index(0)
                            )
                        )
                    else:
                        sorted_row_i = bx_m + row_local
                        # NOTE: rows beyond `num_valid_ids` can contain garbage (within the allocated
                        # buffer). That's OK as long as we never use an out-of-range token id to index X.
                        fused_i = (
                            memref.load(lds_tid, [row_local])
                            if use_tid_lds_req
                            else buffer_ops.buffer_load(
                                sorted_rsrc, sorted_row_i, vec_width=1, dtype=T.i32
                            )
                        )
                        t_raw = fused_i & mask24
                        # NOTE: aiter moe_sorting uses sentinel token_id == tokens for padding.
                        # Do NOT rely on buffer OOB semantics for X loads; explicitly mask to a safe row.
                        t_valid_i32 = arith.cmpi(arith.CmpIPredicate.ult, t_raw, tokens_i32)
                        if const_expr(row_limit_x_req > 0):
                            row_in_limit = arith.cmpi(
                                arith.CmpIPredicate.ult,
                                row_local,
                                fx.Index(row_limit_x_req),
                            )
                            t_valid_i32 = arith.andi(t_valid_i32, row_in_limit)
                        x_row_valid.append(t_valid_i32)
                        if const_expr(x_is_token_slot):
                            s_raw = fused_i >> 24
                            # X is indexed by token-slot in **slot-major** order:
                            #   row_ts = slot * tokens + token
                            # This matches CK's moe_smoothquant output layout.
                            row_ts_i32 = s_raw * tokens_i32 + t_raw
                            row_ts_idx = arith.index_cast(T.index, row_ts_i32)
                            # Apply bounds check to token-slot index
                            row_ts_safe = t_valid_i32.select(row_ts_idx, fx.Index(0))
                            x_row_base_div4.append(row_ts_safe * c_k_div4)
                        else:
                            t_idx = arith.index_cast(T.index, t_raw)
                            t_safe = t_valid_i32.select(t_idx, fx.Index(0))
                            x_row_base_div4.append(t_safe * c_k_div4)

                vec4_x = T.vec(4, x_elem)

                def load_x(idx_i32):
                    """Load `x_load_bytes` bytes from X (gmem) into regs.

                    For 16B, keep the fast dwordx4 path. For 8B/4B, use byte offsets.
                    idx_i32 is in dword units; convert to element index for _buffer_load_vec.
                    """
                    if const_expr(x_load_bytes == 16):
                        idx_elem = (
                            idx_i32 if elem_bytes == 1 else (idx_i32 * fx.Index(2))
                        )
                        return buffer_copy_gmem16_dwordx4(
                            buffer_ops,
                            vector,
                            elem_type=x_elem,
                            idx_i32=idx_elem,
                            rsrc=x_rsrc,
                            vec_elems=vec16_elems,
                            elem_bytes=elem_bytes,
                        )
                    # For 8B/4B, load raw i32 dwords directly.
                    if const_expr(x_load_bytes == 8):
                        return buffer_ops.buffer_load(
                            x_rsrc, idx_i32, vec_width=2, dtype=T.i32
                        )
                    return buffer_ops.buffer_load(
                        x_rsrc, idx_i32, vec_width=1, dtype=T.i32
                    )

                def load_x_tile(base_k):
                    """Prefetch the per-thread X tile portion (gmem -> regs) for a given K base (in elements)."""
                    base_k_div4 = (base_k * arith.index(int(elem_bytes))) // fx.Index(4)
                    parts = []
                    for i in range_constexpr(num_x_loads):
                        idx_i32 = x_row_base_div4[i] + base_k_div4 + x_col_local_i32[i]
                        if const_expr(tiny_row0_x_req or row_limit_x_req > 0):
                            if const_expr(x_load_bytes == 16):
                                _if_x = scf.IfOp(
                                    x_row_valid[i], results_=[T.i32x4], has_else=True
                                )
                                with _if_then(_if_x):
                                    scf.YieldOp([vector.bitcast(T.i32x4, load_x(idx_i32))])
                                with _if_else(_if_x):
                                    scf.YieldOp([arith.constant_vector(0, T.i32x4)])
                                parts.append(_if_x.results[0])
                            elif const_expr(x_load_bytes == 8):
                                _if_x = scf.IfOp(
                                    x_row_valid[i], results_=[T.i32x2], has_else=True
                                )
                                with _if_then(_if_x):
                                    scf.YieldOp([load_x(idx_i32)])
                                with _if_else(_if_x):
                                    scf.YieldOp([arith.constant_vector(0, T.i32x2)])
                                parts.append(_if_x.results[0])
                            else:
                                _if_x = scf.IfOp(
                                    x_row_valid[i], results_=[T.i32], has_else=True
                                )
                                with _if_then(_if_x):
                                    scf.YieldOp([load_x(idx_i32)])
                                with _if_else(_if_x):
                                    scf.YieldOp([arith.constant(0, type=T.i32)])
                                parts.append(_if_x.results[0])
                        else:
                            x_vec = load_x(idx_i32)
                            if const_expr(x_load_bytes == 16):
                                parts.append(vector.bitcast(T.i32x4, x_vec))
                            elif const_expr(x_load_bytes == 8):
                                parts.append(x_vec)
                            else:
                                parts.append(x_vec)
                    return parts

                # tx -> wave/lane (GEMM-style decomposition).
                coord_wl = fx.idx2crd(tx, layout_tx_wave_lane)
                wave_id = fx.get(coord_wl, 0)
                lane_id = fx.get(coord_wl, 1)
                coord_l16 = fx.idx2crd(lane_id, layout_lane16)
                lane_div_16 = fx.get(coord_l16, 0)
                lane_mod_16 = fx.get(coord_l16, 1)

                # Match GEMM naming/pattern: row in LDS is lane_mod_16, and col base is lane_div_16 * a_kpack_elems.
                # A-side kpack is always 16 bytes (activation elements); B-side kpack_bytes
                # may differ (e.g. 8 for int4 weights), but that only affects B preshuffle.
                row_a_lds = lane_mod_16
                a_kpack_elems = 16 // elem_bytes
                col_offset_base = lane_div_16 * arith.index(int(a_kpack_elems))
                col_offset_base_bytes = (
                    col_offset_base
                    if elem_bytes == 1
                    else (col_offset_base * arith.index(int(elem_bytes)))
                )
                a_direct_row_base_div4 = None
                if const_expr(use_a_direct):
                    # Direct-A mode duplicates the tiny A stream across waves to avoid
                    # the hot-loop LDS ping-pong and its barriers for bm16 kernels.
                    sorted_row_a = bx_m + row_a_lds
                    fused_a = (
                        memref.load(lds_tid, [row_a_lds])
                        if use_tid_lds_req
                        else buffer_ops.buffer_load(
                            sorted_rsrc, sorted_row_a, vec_width=1, dtype=T.i32
                        )
                    )
                    t_a_raw = fused_a & mask24
                    t_a_valid = arith.cmpi(
                        arith.CmpIPredicate.ult, t_a_raw, tokens_i32
                    )
                    if const_expr(x_is_token_slot):
                        s_a_raw = fused_a >> 24
                        row_ts_a_i32 = s_a_raw * tokens_i32 + t_a_raw
                        row_ts_a_idx = arith.index_cast(T.index, row_ts_a_i32)
                        row_ts_a_safe = t_a_valid.select(row_ts_a_idx, fx.Index(0))
                        a_direct_row_base_div4 = row_ts_a_safe * c_k_div4
                    else:
                        t_a_idx = arith.index_cast(T.index, t_a_raw)
                        t_a_safe = t_a_valid.select(t_a_idx, fx.Index(0))
                        a_direct_row_base_div4 = t_a_safe * c_k_div4

                # Dynamic N tiling within block (same as existing kernels)
                by_n = by * fx.Index(tile_n)
                num_waves = num_waves_static
                n_per_wave = tile_n // num_waves
                num_acc_n = n_per_wave // 16
                c_n_per_wave = fx.Index(n_per_wave)
                wave_mod_n = wave_id % fx.Index(num_waves)
                n_tile_base = wave_mod_n * c_n_per_wave

                # Precompute n_blk/n_intra for gate and up rows (GEMM-style: idx2crd/get)
                n_intra_gate = []
                n_blk_gate = []
                n_blk_gate_local = []
                n_intra_up = []
                n_blk_up = []
                n_blk_up_local = []
                col_g_list = []
                inter_idx = fx.Index(inter_dim)
                c_n_total // fx.Index(16)
                c_n0_static = experts * (2 * inter_dim) // 16
                layout_n_blk_intra = fx.make_layout((c_n0_static, 16), stride=(16, 1))
                for ni in range_constexpr(num_acc_n):
                    offset = arith.index(ni * 16)
                    col_g = by_n + n_tile_base
                    col_g = col_g + offset
                    col_g = col_g + lane_mod_16
                    col_g_list.append(col_g)

                    row_gate = expert_off_idx + col_g
                    row_up = row_gate + inter_idx
                    row_up_local = col_g + inter_idx

                    n_blk_gate_local.append(col_g // fx.Index(16))
                    n_blk_up_local.append(row_up_local // fx.Index(16))

                    coord_gate = fx.idx2crd(row_gate, layout_n_blk_intra)
                    n_blk_gate.append(fx.get(coord_gate, 0))
                    n_intra_gate.append(fx.get(coord_gate, 1))

                    coord_up = fx.idx2crd(row_up, layout_n_blk_intra)
                    n_blk_up.append(fx.get(coord_up, 0))
                    n_intra_up.append(fx.get(coord_up, 1))

                m_repeat = tile_m // 16
                k_unroll = tile_k_bytes // 64  # K64-byte micro-step (2x MFMA)

                # --- B Load Logic (K64) - shared layout with preshuffle GEMM ---
                _b_stride_nlane = kpack_bytes // int(w_elem_bytes)
                _b_stride_klane = 16 * _b_stride_nlane
                _b_stride_k0 = 4 * _b_stride_klane
                _b_stride_n0 = (
                    (int(model_dim) * int(w_elem_bytes)) // 64
                ) * _b_stride_k0
                _expert_b_stride = ((2 * int(inter_dim)) // 16) * _b_stride_n0
                expert_b_base = expert_idx * arith.index(_expert_b_stride)

                def load_b_pack(
                    base_k, ki_step, ni, blk_list, intra_list, cache_modifier
                ):
                    return load_b_pack_k32(
                        buffer_ops,
                        arith,
                        vector,
                        arg_b=arg_w,
                        b_rsrc=w_rsrc,
                        layout_b=layout_b,
                        base_k=base_k,
                        ki_step=ki_step,
                        n_blk=blk_list[ni],
                        n_intra=intra_list[ni],
                        lane_div_16=lane_div_16,  # 0..3
                        elem_type=w_elem,
                        kpack_bytes=kpack_bytes,
                        elem_bytes=w_elem_bytes,
                        unpack_int4=is_int4,
                        cache_modifier=cache_modifier,
                    )

                def load_b_tile(base_k, blk_list, intra_list):
                    """Prefetch the entire per-thread B tile (gmem -> regs) for a given K base.

                    Returns a list of length `k_unroll`, where each entry is a tuple:
                      (packs_half0[ni], packs_half1[ni])  for the K64 micro-step.
                    For groupwise variants, each entry also includes per-group scales:
                      (packs0[ni], packs1[ni], scales0[ni], scales1[ni])
                    """
                    if const_expr(is_int4_bf16_groupwise):
                        # W4A16 groupwise: load raw packed32 + scale; defer dequant to compute_tile.
                        raw_data = []
                        for ku in range_constexpr(k_unroll):
                            raw_ku = []
                            for ni in range_constexpr(num_acc_n):
                                packed32, scale_val = load_b_raw_w4a16_groupwise(
                                    buffer_ops,
                                    arith,
                                    vector,
                                    arg_b=arg_w,
                                    b_rsrc=w_rsrc,
                                    layout_b=layout_b,
                                    base_k=base_k,
                                    ku=ku,
                                    n_blk=blk_list[ni],
                                    n_intra=intra_list[ni],
                                    lane_div_16=lane_div_16,
                                    elem_type=w_elem,
                                    scale_rsrc=sw_rsrc,
                                    expert_offset=expert_off_idx,
                                    num_groups=num_groups,
                                    group_size=group_size,
                                    n_per_expert=2 * inter_dim,
                                    kpack_bytes=kpack_bytes,
                                    scale_dtype=scale_dtype,
                                )
                                raw_ku.append((packed32, scale_val))
                            raw_data.append(raw_ku)
                        return raw_data
                    elif const_expr(is_int4_bf16):
                        # W4A16 per-row: load raw packed32; defer dequant to compute_tile.
                        raw_data = []
                        for ku in range_constexpr(k_unroll):
                            raw_ku = []
                            for ni in range_constexpr(num_acc_n):
                                raw = load_b_raw_w4a16(
                                    buffer_ops,
                                    arith,
                                    vector,
                                    arg_b=arg_w,
                                    b_rsrc=w_rsrc,
                                    layout_b=layout_b,
                                    base_k=base_k,
                                    ku=ku,
                                    n_blk=blk_list[ni],
                                    n_intra=intra_list[ni],
                                    lane_div_16=lane_div_16,
                                    elem_type=w_elem,
                                    kpack_bytes=kpack_bytes,
                                )
                                raw_ku.append(raw)
                            raw_data.append(raw_ku)
                        return raw_data
                    else:
                        # fp8/int8/bf16/fp16: K32 halves may map to distinct packed fragments;
                        # dense bf16/fp16 uses the static-stride path to include expert_b_base.
                        b_tile = []
                        for ku in range_constexpr(k_unroll):
                            packs0 = []
                            packs1 = []
                            for ni in range_constexpr(num_acc_n):
                                if const_expr(is_int4):
                                    ki0 = (ku * 2) + 0
                                    ki1 = (ku * 2) + 1
                                    b0 = load_b_pack(
                                        base_k,
                                        ki0,
                                        ni,
                                        blk_list,
                                        intra_list,
                                        b_nt,
                                    )
                                    b1 = load_b_pack(
                                        base_k,
                                        ki1,
                                        ni,
                                        blk_list,
                                        intra_list,
                                        b_nt,
                                    )
                                else:
                                    b0, b1 = load_b_packs_k64_strided(
                                        buffer_ops,
                                        arith,
                                        vector,
                                        b_rsrc=w_rsrc,
                                        base_k=base_k,
                                        ku=ku,
                                        n_blk=blk_list[ni],
                                        n_intra=intra_list[ni],
                                        lane_div_16=lane_div_16,
                                        elem_type=w_elem,
                                        stride_n0=_b_stride_n0,
                                        stride_k0=_b_stride_k0,
                                        stride_klane=_b_stride_klane,
                                        stride_nlane=_b_stride_nlane,
                                        base_offset=expert_b_base,
                                        kpack_bytes=kpack_bytes,
                                        elem_bytes=w_elem_bytes,
                                        cache_modifier=b_nt,
                                    )
                                packs0.append(b0)
                                packs1.append(b1)
                            b_tile.append((packs0, packs1))
                        return b_tile

                acc_gate = [acc_init] * (num_acc_n * m_repeat)
                acc_up = [acc_init] * (num_acc_n * m_repeat)

                # ---- Pipeline helpers: store X tile to LDS with ping-pong base ----
                def store_x_tile_to_lds(vec_x_in_parts, lds_base):
                    for i in range_constexpr(num_x_loads):
                        row_local = x_row_local[i]
                        col_local_i32 = x_col_local_i32[i]
                        if const_expr(x_load_bytes == 16):
                            lds_store_16b_xor16(
                                arith,
                                vector,
                                lds_memref=lds_x,
                                vec16_ty=vec16_x,
                                layout_lds=layout_lds,
                                row_local=row_local,
                                col_local_i32=col_local_i32,
                                tx_c4=fx.Index(4),
                                k_blocks16=k_blocks16,
                                lds_base=lds_base,
                                vec_part_i32x4=vec_x_in_parts[i],
                                elem_bytes=elem_bytes,
                            )
                        elif const_expr(x_load_bytes == 8):
                            lds_store_8b_xor16(
                                arith,
                                vector,
                                lds_memref=lds_x,
                                vec8_ty=vec8_x,
                                layout_lds=layout_lds,
                                row_local=row_local,
                                col_local_i32=col_local_i32,
                                tx_c4=fx.Index(4),
                                k_blocks16=k_blocks16,
                                lds_base=lds_base,
                                vec_part_i32x2=vec_x_in_parts[i],
                                elem_bytes=elem_bytes,
                            )
                        else:
                            lds_store_4b_xor16(
                                arith,
                                vector,
                                lds_memref=lds_x,
                                vec4_ty=vec4_x,
                                layout_lds=layout_lds,
                                row_local=row_local,
                                col_local_i32=col_local_i32,
                                tx_c4=fx.Index(4),
                                k_blocks16=k_blocks16,
                                lds_base=lds_base,
                                vec_part_i32x1=vec_x_in_parts[i],
                                elem_bytes=elem_bytes,
                            )

                def dma_x_tile_to_lds(base_k, lds_base_bytes: int):
                    """Copy X directly to LDS with buffer_load_lds.

                    The LDS address is linear while the global column is XOR-swizzled.
                    Since XOR is its own inverse, the later MFMA LDS read at
                    swizzle(row, col) observes the original unswizzled X[col].
                    """
                    c4_idx = arith.index(4)
                    base_k_div4 = (
                        base_k * arith.index(int(elem_bytes))
                    ) // fx.Index(4)
                    dma_bytes = 16
                    wave_size = 64
                    lds_ptr = None
                    for i in range_constexpr(num_x_loads):
                        row_local_i = x_row_local[i]
                        col_local_i32_i = x_col_local_i32[i]
                        col_local_sw = swizzle_xor16(
                            row_local_i,
                            col_local_i32_i * c4_idx,
                            k_blocks16,
                        )
                        row_k_dw = x_row_base_div4[i] + base_k_div4
                        global_byte_idx = row_k_dw * c4_idx + col_local_sw
                        global_offset = arith.index_cast(T.i32, global_byte_idx)

                        if const_expr(i == 0):
                            lds_byte_base = (
                                memref.extract_aligned_pointer_as_index(lds_x)
                                + arith.index(int(lds_base_bytes))
                            )
                            warp_offset = rocdl.readfirstlane(
                                T.i64,
                                arith.index_cast(
                                    T.i64,
                                    wave_id * arith.index(wave_size * dma_bytes),
                                ),
                            )
                            lds_ptr_base = buffer_ops.create_llvm_ptr(
                                arith.index_cast(T.i64, lds_byte_base),
                                address_space=3,
                            )
                            lds_ptr = buffer_ops.get_element_ptr(
                                lds_ptr_base, warp_offset
                            )
                        else:
                            lds_ptr = buffer_ops.get_element_ptr(
                                lds_ptr,
                                static_byte_offset=total_threads * dma_bytes,
                            )
                        rocdl.raw_ptr_buffer_load_lds(
                            x_rsrc,
                            lds_ptr,
                            arith.constant(dma_bytes, type=T.i32),
                            global_offset,
                            arith.constant(0, type=T.i32),
                            arith.constant(0, type=T.i32),
                            arith.constant(0, type=T.i32),
                        )

                # --- A LDS load helper for K64 (load 16B once, extract 2x i64 halves) ---
                def lds_load_packs_k64(curr_row_a_lds, col_base_bytes, lds_base):
                    col_base_swz_bytes = swizzle_xor16(
                        curr_row_a_lds, col_base_bytes, k_blocks16
                    )
                    col_base_swz = (
                        col_base_swz_bytes
                        if elem_bytes == 1
                        else (col_base_swz_bytes // arith.index(int(elem_bytes)))
                    )
                    idx_a16 = crd2idx((curr_row_a_lds, col_base_swz), layout_lds)
                    idx_a16 = idx_a16 + lds_base
                    loaded_a16 = vector.load_op(vec16_x, lds_x, [idx_a16])
                    a_i64x2 = vector.bitcast(T.i64x2, loaded_a16)
                    a0 = vector.extract(
                        a_i64x2, static_position=[0], dynamic_position=[]
                    )
                    a1 = vector.extract(
                        a_i64x2, static_position=[1], dynamic_position=[]
                    )
                    return a0, a1

                def gmem_load_a_packs_k64(base_k, col_base_bytes):
                    base_k_div4 = (
                        base_k * arith.index(int(elem_bytes))
                    ) // fx.Index(4)
                    col_base_div4 = col_base_bytes // fx.Index(4)
                    idx_i32 = a_direct_row_base_div4 + base_k_div4 + col_base_div4
                    loaded_a16 = load_x(idx_i32)
                    a_i64x2 = vector.bitcast(T.i64x2, loaded_a16)
                    a0 = vector.extract(
                        a_i64x2, static_position=[0], dynamic_position=[]
                    )
                    a1 = vector.extract(
                        a_i64x2, static_position=[1], dynamic_position=[]
                    )
                    return a0, a1

                def prefetch_a_tile_from_lds(lds_base):
                    a_regs = []
                    for ku in range_constexpr(k_unroll):
                        ki64 = arith.index(ku * 64)
                        col_base = col_offset_base_bytes + ki64
                        for mi in range_constexpr(m_repeat):
                            mi_val = arith.index(mi * 16)
                            curr_row_a_lds = row_a_lds + mi_val
                            a_regs.append(
                                lds_load_packs_k64(curr_row_a_lds, col_base, lds_base)
                            )
                    return a_regs

                def compute_tile(
                    acc_gate_in,
                    acc_up_in,
                    b_gate_tile_in,
                    b_up_tile_in,
                    lds_base,
                    *,
                    base_k_for_a=None,
                    prefetch_epilogue: bool = False,
                    a0_prefetch=None,
                    a_tile_regs=None,
                ):
                    gate_list = list(acc_gate_in)
                    up_list = list(acc_up_in)
                    mfma_res_ty = T.i32x4 if is_int8 else T.f32x4
                    if const_expr(_use_mfma_k32):
                        mfma_fn = (
                            rocdl.mfma_f32_16x16x32_f16
                            if is_f16
                            else rocdl.mfma_f32_16x16x32_bf16
                        )
                    else:
                        mfma_fn = (
                            mfma_i32_k32
                            if is_int8
                            else (
                                mfma_f32_bf16_k16
                                if is_bf16
                                else (
                                    rocdl.mfma_f32_16x16x16f16
                                    if is_f16
                                    else rocdl.mfma_f32_16x16x32_fp8_fp8
                                )
                            )
                        )

                    # Optional: prefetch epilogue scales while we are about to run the last MFMA tile,
                    # matching the preshuffle GEMM pattern of overlapping scale loads with MFMA.
                    epilogue_pf = None
                    if const_expr(prefetch_epilogue and not use_groupwise_scale):
                        expert_off_pf = expert_off_idx
                        sw_gate_pf = []
                        sw_up_pf = []
                        for ni in range_constexpr(num_acc_n):
                            col_g = col_g_list[ni]
                            row_gate_idx = expert_off_pf + col_g
                            row_up_idx = row_gate_idx + inter_idx
                            sw_gate_pf.append(
                                fx.Float32(1.0)
                                if not needs_scale_w
                                else buffer_ops.buffer_load(
                                    sw_rsrc, row_gate_idx, vec_width=1, dtype=T.f32
                                )
                            )
                            sw_up_pf.append(
                                fx.Float32(1.0)
                                if not needs_scale_w
                                else buffer_ops.buffer_load(
                                    sw_rsrc, row_up_idx, vec_width=1, dtype=T.f32
                                )
                            )
                        epilogue_tid_pf = None
                        if const_expr(prefetch_epi_tid_req):
                            epilogue_tid_pf = []
                            lane0_for_rows = arith.cmpi(
                                arith.CmpIPredicate.eq, lane_div_16, fx.Index(0)
                            )
                            for ii_pf in range_constexpr(4):
                                if const_expr(ii_pf < row_limit_epilog_req):
                                    row_pf = bx_m + fx.Index(ii_pf)
                                    _if_tid_pf = scf.IfOp(
                                        lane0_for_rows,
                                        results_=[T.i32],
                                        has_else=True,
                                    )
                                    with _if_then(_if_tid_pf):
                                        scf.YieldOp(
                                            [
                                                buffer_ops.buffer_load(
                                                    sorted_rsrc,
                                                    row_pf,
                                                    vec_width=1,
                                                    dtype=T.i32,
                                                )
                                            ]
                                        )
                                    with _if_else(_if_tid_pf):
                                        scf.YieldOp([arith.constant(0, type=T.i32)])
                                    epilogue_tid_pf.append(_if_tid_pf.results[0])
                                else:
                                    epilogue_tid_pf.append(
                                        arith.constant(0, type=T.i32)
                                    )
                        epilogue_pf = (sw_gate_pf, sw_up_pf, epilogue_tid_pf)

                    def _i64_to_v4f16(x_i64):
                        v1 = vector.from_elements(T.vec(1, T.i64), [x_i64])
                        return vector.bitcast(T.f16x4, v1)

                    def _i64_to_v4i16(x_i64):
                        v1 = vector.from_elements(T.vec(1, T.i64), [x_i64])
                        return vector.bitcast(T.i16x4, v1)

                    def _i64x2_to_v8f16(lo, hi):
                        v2 = vector.from_elements(T.i64x2, [lo, hi])
                        return vector.bitcast(T.f16x8, v2)

                    def _i64x2_to_v8bf16(lo, hi):
                        v2 = vector.from_elements(T.i64x2, [lo, hi])
                        return vector.bitcast(T.bf16x8, v2)

                    def mfma_k64(acc_in, a0, a1, b0, b1):
                        if const_expr(_use_mfma_k32):
                            # gfx950: single 16x16x32 MFMA consuming all 128 bits (K=32 f16/bf16)
                            if const_expr(is_f16):
                                av = _i64x2_to_v8f16(a0, a1)
                                bv = _i64x2_to_v8f16(b0, b1)
                            else:
                                av = _i64x2_to_v8bf16(a0, a1)
                                bv = _i64x2_to_v8bf16(b0, b1)
                            return mfma_fn(mfma_res_ty, [av, bv, acc_in, 0, 0, 0])
                        if const_expr(is_f16):
                            a0v = _i64_to_v4f16(a0)
                            a1v = _i64_to_v4f16(a1)
                            b0v = _i64_to_v4f16(b0)
                            b1v = _i64_to_v4f16(b1)
                            acc_mid = mfma_fn(mfma_res_ty, [a0v, b0v, acc_in, 0, 0, 0])
                            return mfma_fn(mfma_res_ty, [a1v, b1v, acc_mid, 0, 0, 0])
                        if const_expr(is_bf16):
                            a0v = _i64_to_v4i16(a0)
                            a1v = _i64_to_v4i16(a1)
                            b0v = _i64_to_v4i16(b0)
                            b1v = _i64_to_v4i16(b1)
                            acc_mid = mfma_fn(mfma_res_ty, [a0v, b0v, acc_in, 0, 0, 0])
                            return mfma_fn(mfma_res_ty, [a1v, b1v, acc_mid, 0, 0, 0])
                        acc_mid = mfma_fn(mfma_res_ty, [a0, b0, acc_in, 0, 0, 0])
                        return mfma_fn(mfma_res_ty, [a1, b1, acc_mid, 0, 0, 0])

                    def get_a_pair_for_mfma(ku: int, mi: int, curr_row_a_lds, col_base):
                        if const_expr(use_a_direct):
                            return gmem_load_a_packs_k64(base_k_for_a, col_base)
                        if a_tile_regs is not None:
                            return a_tile_regs[ku * m_repeat + mi]
                        if (
                            a0_prefetch is not None
                            and ku == 0
                            and mi == 0
                        ):
                            return a0_prefetch
                        return lds_load_packs_k64(curr_row_a_lds, col_base, lds_base)

                    def _acc_scaled_f32(f32_acc_vec, f32_partial_vec, scale_val):
                        """MFMA f32 partial -> scale -> add to f32 accumulator via math.fma on vector."""
                        from flydsl._mlir.dialects._math_ops_gen import fma as _math_fma

                        _uw = arith._to_raw
                        scale_vec = _uw(vector.broadcast(T.f32x4, scale_val))
                        return arith.ArithValue(
                            _math_fma(scale_vec, _uw(f32_partial_vec), _uw(f32_acc_vec))
                        )

                    if const_expr(is_int4_bf16 or is_int4_bf16_groupwise):
                        if sched_setprio:
                            rocdl.s_setprio(1)
                        # W4A16: deferred dequant -- unpack int4->bf16 right before MFMA
                        # to minimize VGPR lifetime of dequantized bf16 values.
                        _pending_gate_up = None
                        for ku in range_constexpr(k_unroll):
                            b_gate_raw = b_gate_tile_in[ku]
                            b_up_raw = b_up_tile_in[ku]
                            ki64 = arith.index(ku * 64)
                            col_base = col_offset_base_bytes + ki64

                            for mi in range_constexpr(m_repeat):
                                mi_val = arith.index(mi * 16)
                                curr_row_a_lds = row_a_lds + mi_val

                                a0, a1 = get_a_pair_for_mfma(
                                    ku, mi, curr_row_a_lds, col_base
                                )

                                for ni in range_constexpr(num_acc_n):
                                    acc_idx = mi * num_acc_n + ni
                                    if const_expr(is_int4_bf16_groupwise):
                                        packed_g, sc_g = b_gate_raw[ni]
                                        packed_u, sc_u = b_up_raw[ni]
                                        if const_expr(_scale_is_bf16):
                                            sc_g = extract_bf16_scale(arith, sc_g, ku)
                                            sc_u = extract_bf16_scale(arith, sc_u, ku)
                                    else:
                                        packed_g, sc_g = b_gate_raw[ni], None
                                        packed_u, sc_u = b_up_raw[ni], None
                                    if const_expr(
                                        is_int4_bf16_groupwise and use_gfx950_cvt
                                    ):
                                        # Defer group scale to post-MFMA FMA with pipeline:
                                        # Issue current MFMA, then apply FMA for previous iteration's result.
                                        bg0, bg1 = unpack_b_w4a16(
                                            packed_g,
                                            arith,
                                            vector,
                                            scale_val=None,
                                            use_gfx950_cvt=True,
                                            defer_scale16=True,
                                        )
                                        tmp_g = mfma_k64(zero_f32_acc, a0, a1, bg0, bg1)
                                        bu0, bu1 = unpack_b_w4a16(
                                            packed_u,
                                            arith,
                                            vector,
                                            scale_val=None,
                                            use_gfx950_cvt=True,
                                            defer_scale16=True,
                                        )
                                        tmp_u = mfma_k64(zero_f32_acc, a0, a1, bu0, bu1)
                                        # Apply FMA for previous pending result (MFMA already completed).
                                        if _pending_gate_up is not None:
                                            p_idx, p_g, p_u, p_sc_g, p_sc_u = (
                                                _pending_gate_up
                                            )
                                            gate_list[p_idx] = _acc_scaled_f32(
                                                gate_list[p_idx], p_g, p_sc_g
                                            )
                                            up_list[p_idx] = _acc_scaled_f32(
                                                up_list[p_idx], p_u, p_sc_u
                                            )
                                        _pending_gate_up = (
                                            acc_idx,
                                            tmp_g,
                                            tmp_u,
                                            sc_g,
                                            sc_u,
                                        )
                                    else:
                                        bg0, bg1 = unpack_b_w4a16(
                                            packed_g,
                                            arith,
                                            vector,
                                            scale_val=sc_g,
                                            use_gfx950_cvt=use_gfx950_cvt,
                                            defer_scale16=use_gfx950_cvt,
                                        )
                                        gate_list[acc_idx] = mfma_k64(
                                            gate_list[acc_idx], a0, a1, bg0, bg1
                                        )
                                        bu0, bu1 = unpack_b_w4a16(
                                            packed_u,
                                            arith,
                                            vector,
                                            scale_val=sc_u,
                                            use_gfx950_cvt=use_gfx950_cvt,
                                            defer_scale16=use_gfx950_cvt,
                                        )
                                        up_list[acc_idx] = mfma_k64(
                                            up_list[acc_idx], a0, a1, bu0, bu1
                                        )
                        # Drain last pending FMA.
                        if _pending_gate_up is not None:
                            p_idx, p_g, p_u, p_sc_g, p_sc_u = _pending_gate_up
                            gate_list[p_idx] = _acc_scaled_f32(
                                gate_list[p_idx], p_g, p_sc_g
                            )
                            up_list[p_idx] = _acc_scaled_f32(
                                up_list[p_idx], p_u, p_sc_u
                            )
                        if sched_setprio:
                            rocdl.s_setprio(0)
                    else:
                        if sched_setprio:
                            rocdl.s_setprio(1)
                        for ku in range_constexpr(k_unroll):
                            b_gate_packs0, b_gate_packs1 = b_gate_tile_in[ku]
                            b_up_packs0, b_up_packs1 = b_up_tile_in[ku]
                            ki64 = arith.index(ku * 64)
                            col_base = col_offset_base_bytes + ki64

                            for mi in range_constexpr(m_repeat):
                                mi_val = arith.index(mi * 16)
                                curr_row_a_lds = row_a_lds + mi_val

                                a0, a1 = get_a_pair_for_mfma(
                                    ku, mi, curr_row_a_lds, col_base
                                )

                                for ni in range_constexpr(num_acc_n):
                                    acc_idx = mi * num_acc_n + ni
                                    gate_list[acc_idx] = mfma_k64(
                                        gate_list[acc_idx],
                                        a0,
                                        a1,
                                        b_gate_packs0[ni],
                                        b_gate_packs1[ni],
                                    )
                                    up_list[acc_idx] = mfma_k64(
                                        up_list[acc_idx],
                                        a0,
                                        a1,
                                        b_up_packs0[ni],
                                        b_up_packs1[ni],
                                    )
                        if sched_setprio:
                            rocdl.s_setprio(0)
                    return gate_list, up_list, epilogue_pf

                # ---------------- 2-stage pipeline (ping-pong LDS + B tile prefetch) ----------------
                lds_tile_elems = arith.index(tile_m * lds_stride)
                lds_base_cur = fx.Index(0)
                lds_base_nxt = lds_tile_elems
                lds_base_cur_bytes = 0
                lds_base_nxt_bytes = int(tile_m) * int(lds_stride) * int(elem_bytes)

                # Optional scheduler hints (copied from tuned GEMM); can be disabled via env.
                if const_expr(not sched_disable):
                    rocdl.sched_barrier(0)

                def hot_loop_scheduler():
                    if const_expr(sched_disable):
                        return
                    rocdl.sched_barrier(0)
                    mfma_group = num_acc_n if sched_split_mfma else num_acc_n * 2
                    # K64 micro-step: 2x K32 MFMA per gemm.
                    mfma_total = (
                        (k_unroll * 4) * m_repeat * num_acc_n
                        if sched_split_mfma
                        else (k_unroll * 2) * m_repeat * mfma_group
                    )
                    mfma_per_iter = 2 * mfma_group
                    sche_iters = (
                        0 if mfma_per_iter == 0 else (mfma_total // mfma_per_iter)
                    )

                    if const_expr(use_a_direct):
                        # No X LDS traffic exists in direct-A mode. Bias the scheduler
                        # toward VMEM and MFMA only; DS read/write hints otherwise make
                        # the compiler schedule for instructions that are not present.
                        rocdl.sched_vmem(sched_early_vmem + 1)
                        rocdl.sched_mfma(1)
                        rocdl.sched_vmem(sched_early_vmem + 1)
                        rocdl.sched_mfma(1)
                        for _ in range_constexpr(sche_iters):
                            rocdl.sched_vmem(sched_loop_vmem + 1)
                            rocdl.sched_mfma(mfma_group)
                            rocdl.sched_vmem(sched_loop_vmem + 1)
                            rocdl.sched_mfma(mfma_group)
                        rocdl.sched_barrier(0)
                        return

                    rocdl.sched_dsrd(2)
                    rocdl.sched_mfma(1)
                    if const_expr(tile_m == 16):
                        rocdl.sched_vmem(sched_early_vmem)
                    rocdl.sched_mfma(1)
                    if const_expr(tile_m == 16):
                        rocdl.sched_vmem(sched_early_vmem)
                    if const_expr(num_acc_n < 4):
                        rocdl.sched_dsrd(1)
                        rocdl.sched_mfma(1)
                        if const_expr(tile_m == 16):
                            rocdl.sched_vmem(sched_early_vmem)
                        rocdl.sched_dsrd(1)
                        rocdl.sched_mfma(1)
                        if const_expr(tile_m == 16):
                            rocdl.sched_vmem(sched_early_vmem)
                        rocdl.sched_mfma(1)

                    # DS-write hints near the end: match total X LDS-store micro-ops per thread.
                    dswr_tail = num_x_loads
                    dstr_advance = dswr_advance_req
                    if const_expr(dswr_tail > sche_iters):
                        dswr_tail = sche_iters
                    dswr_start = max(sche_iters - dswr_tail - dstr_advance, 0)
                    for sche_i in range_constexpr(sche_iters):
                        rocdl.sched_vmem(sched_loop_vmem)
                        rocdl.sched_mfma(mfma_group)
                        rocdl.sched_dsrd(1)
                        rocdl.sched_mfma(mfma_group)
                        if const_expr(sche_i >= dswr_start - 1):
                            rocdl.sched_dswr(1)
                    rocdl.sched_barrier(0)

                # Prologue: prefetch tile0, store to LDS(cur), sync.
                b_n_blk_gate = (
                    n_blk_gate
                    if (is_int4 or is_int4_bf16 or is_int4_bf16_groupwise)
                    else n_blk_gate_local
                )
                b_n_blk_up = (
                    n_blk_up
                    if (is_int4 or is_int4_bf16 or is_int4_bf16_groupwise)
                    else n_blk_up_local
                )

                c_tile_k = arith.index(tile_k)
                total_tiles = int(_k_per_batch) // int(tile_k)
                k0 = k_base_idx
                if const_expr(use_a_direct):
                    pass
                elif const_expr(use_x_dma):
                    dma_x_tile_to_lds(k0, lds_base_cur_bytes)
                else:
                    x_regs0 = load_x_tile(k0)
                b_gate_cur = load_b_tile(k0, b_n_blk_gate, n_intra_gate)
                b_up_cur = load_b_tile(k0, b_n_blk_up, n_intra_up)
                if const_expr(not use_a_direct):
                    if const_expr(not use_x_dma):
                        store_x_tile_to_lds(x_regs0, lds_base_cur)
                    if const_expr(use_x_dma):
                        rocdl.s_waitcnt(0)
                stage_barrier()

                # Loop-carried ping/pong state.
                lds_base_pong = lds_base_cur  # current/compute
                lds_base_ping = lds_base_nxt  # next/load+store

                # Cross-tile A0 LDS prefetch (default-on): prefetch the first A-pack (K64) for the
                # tile we are about to compute from LDS, to overlap with upcoming VMEM.
                a0_prefetch_pong = (
                    (fx.Int64(0).ir_value(), fx.Int64(0).ir_value())
                    if use_a_direct
                    else lds_load_packs_k64(
                        row_a_lds, col_offset_base_bytes, lds_base_pong
                    )
                )

                # Ping-pong main loop (2 tiles per iteration), leaving 2 tail tiles.
                # Uses scf.for with loop-carried accumulators, B-tile prefetch, and A0 LDS prefetch.
                arith.index(tile_k * 2)
                pair_iters = max((total_tiles - 2) // 2, 0)

                # B-tile data layout per k_unroll entry (3 variants):
                #
                # 1) int4 + groupwise scale (is_int4_bf16_groupwise):
                #    [(packed_w4, scale), (packed_w4, scale), ...]   per ni
                #    Each ni has a (packed_weights, groupwise_scale) pair.
                #    Flattened as: [packed_0..N, scale_0..N]  -> 2 * num_acc_n values
                #
                # 2) int4_bf16 without groupwise scale (int4_bf16_single_field):
                #    [raw_i64, raw_i64, ...]   per ni
                #    Single packed i64 per ni, already contains both weight halves.
                #    Flattened as: [raw_0..N]  -> 1 * num_acc_n values
                #
                # 3) fp8/int8/bf16/fp16 (default -- two register packs per ku):
                #    (packs_even_list, packs_odd_list)
                #    Two lists of num_acc_n regs for even/odd MFMA operands.
                #    Flattened as: [even_0..N, odd_0..N]  -> 2 * num_acc_n values
                #
                int4_bf16_single_field = is_int4_bf16 and not is_int4_bf16_groupwise
                _fields_per_ku = 1 if int4_bf16_single_field else 2
                _vals_per_b_tile = k_unroll * _fields_per_ku * num_acc_n

                def _flatten_b_tile(b_tile):
                    """Flatten B tile to a 1-D list for scf.for loop-carried state."""
                    flat = []
                    for ku_entry in b_tile:
                        if is_int4_bf16_groupwise:
                            # [(packed, scale), ...] -> [packed_0..N, scale_0..N]
                            flat.extend(t[0] for t in ku_entry)
                            flat.extend(t[1] for t in ku_entry)
                        elif int4_bf16_single_field:
                            # [raw_i64, ...] -> [raw_0..N]
                            flat.extend(ku_entry)
                        else:
                            # (packs_even, packs_odd) -> [even_0..N, odd_0..N]
                            flat.extend(ku_entry[0])
                            flat.extend(ku_entry[1])
                    return flat

                def _unflatten_b_tile(vals):
                    """Reconstruct B tile from flattened scf.for loop-carried state."""
                    b_tile, idx = [], 0
                    for _ in range_constexpr(k_unroll):
                        if is_int4_bf16_groupwise:
                            packed = list(vals[idx : idx + num_acc_n])
                            idx += num_acc_n
                            scales = list(vals[idx : idx + num_acc_n])
                            idx += num_acc_n
                            b_tile.append(
                                [
                                    (packed[ni], scales[ni])
                                    for ni in range_constexpr(num_acc_n)
                                ]
                            )
                        elif int4_bf16_single_field:
                            b_tile.append(list(vals[idx : idx + num_acc_n]))
                            idx += num_acc_n
                        else:
                            packs_even = list(vals[idx : idx + num_acc_n])
                            idx += num_acc_n
                            packs_odd = list(vals[idx : idx + num_acc_n])
                            idx += num_acc_n
                            b_tile.append((packs_even, packs_odd))
                    return b_tile

                _n_acc = m_repeat * num_acc_n
                _p_bg = 2 * _n_acc
                _p_bu = _p_bg + _vals_per_b_tile
                _p_a0 = _p_bu + _vals_per_b_tile
                init_state = (
                    list(acc_gate)
                    + list(acc_up)
                    + _flatten_b_tile(b_gate_cur)
                    + _flatten_b_tile(b_up_cur)
                    + list(a0_prefetch_pong)
                )

                for pair_iv, state in range(0, pair_iters, 1, init=init_state):
                    _ag = list(state[:_n_acc])
                    _au = list(state[_n_acc:_p_bg])
                    _bg = _unflatten_b_tile(list(state[_p_bg:_p_bu]))
                    _bu = _unflatten_b_tile(list(state[_p_bu:_p_a0]))
                    _a0pf = (state[_p_a0], state[_p_a0 + 1])

                    k_iv = k_base_idx + pair_iv * (c_tile_k + c_tile_k)
                    _a_tile_pong = (
                        prefetch_a_tile_from_lds(lds_base_pong)
                        if prefetch_a_req
                        else None
                    )

                    # ---- stage 0: prefetch+store ping, compute pong ----
                    next_k1 = k_iv + c_tile_k
                    if const_expr(use_a_direct):
                        pass
                    elif const_expr(use_x_dma):
                        dma_x_tile_to_lds(next_k1, lds_base_nxt_bytes)
                    else:
                        x_regs_ping = load_x_tile(next_k1)
                    _bg_ping = load_b_tile(next_k1, b_n_blk_gate, n_intra_gate)
                    _bu_ping = load_b_tile(next_k1, b_n_blk_up, n_intra_up)

                    _ag, _au, _ = compute_tile(
                        _ag,
                        _au,
                        _bg,
                        _bu,
                        lds_base_pong,
                        base_k_for_a=k_iv,
                        a0_prefetch=_a0pf,
                        a_tile_regs=_a_tile_pong,
                    )
                    if const_expr(not use_a_direct and not use_x_dma):
                        store_x_tile_to_lds(x_regs_ping, lds_base_ping)
                    hot_loop_scheduler()
                    if const_expr(use_x_dma and not use_a_direct):
                        rocdl.s_waitcnt(0)
                    stage_barrier()

                    _a0pf_ping = (
                        (fx.Int64(0).ir_value(), fx.Int64(0).ir_value())
                        if use_a_direct
                        else lds_load_packs_k64(
                            row_a_lds, col_offset_base_bytes, lds_base_ping
                        )
                    )
                    _a_tile_ping = (
                        prefetch_a_tile_from_lds(lds_base_ping)
                        if prefetch_a_req
                        else None
                    )

                    # ---- stage 1: prefetch+store pong, compute ping ----
                    next_k2 = k_iv + c_tile_k + c_tile_k
                    if const_expr(use_a_direct):
                        pass
                    elif const_expr(use_x_dma):
                        dma_x_tile_to_lds(next_k2, lds_base_cur_bytes)
                    else:
                        x_regs_pong = load_x_tile(next_k2)
                    _bg_next = load_b_tile(next_k2, b_n_blk_gate, n_intra_gate)
                    _bu_next = load_b_tile(next_k2, b_n_blk_up, n_intra_up)

                    _ag, _au, _ = compute_tile(
                        _ag,
                        _au,
                        _bg_ping,
                        _bu_ping,
                        lds_base_ping,
                        base_k_for_a=next_k1,
                        a0_prefetch=_a0pf_ping,
                        a_tile_regs=_a_tile_ping,
                    )
                    if const_expr(not use_a_direct and not use_x_dma):
                        store_x_tile_to_lds(x_regs_pong, lds_base_pong)
                    hot_loop_scheduler()
                    if const_expr(use_x_dma and not use_a_direct):
                        rocdl.s_waitcnt(0)
                    stage_barrier()

                    _a0pf_new = (
                        (fx.Int64(0).ir_value(), fx.Int64(0).ir_value())
                        if use_a_direct
                        else lds_load_packs_k64(
                            row_a_lds, col_offset_base_bytes, lds_base_pong
                        )
                    )

                    yield_vals = (
                        list(_ag)
                        + list(_au)
                        + _flatten_b_tile(_bg_next)
                        + _flatten_b_tile(_bu_next)
                        + list(_a0pf_new)
                    )
                    loop_results = yield yield_vals

                # After scf.for: extract final state from yielded results.
                SmemPtr._view_cache = None
                if pair_iters > 0:
                    acc_gate = list(loop_results[:_n_acc])
                    acc_up = list(loop_results[_n_acc:_p_bg])
                    b_gate_cur = _unflatten_b_tile(list(loop_results[_p_bg:_p_bu]))
                    b_up_cur = _unflatten_b_tile(list(loop_results[_p_bu:_p_a0]))
                    a0_prefetch_pong = (loop_results[_p_a0], loop_results[_p_a0 + 1])
                k_tail0 = k_base_idx + arith.index(_k_per_batch - (2 * tile_k))
                k_tail1 = k_base_idx + arith.index(_k_per_batch - tile_k)
                if const_expr(use_a_direct):
                    pass
                elif const_expr(use_x_dma):
                    dma_x_tile_to_lds(k_tail1, lds_base_nxt_bytes)
                else:
                    x_regs_ping = load_x_tile(k_tail1)
                b_gate_ping = load_b_tile(k_tail1, b_n_blk_gate, n_intra_gate)
                b_up_ping = load_b_tile(k_tail1, b_n_blk_up, n_intra_up)

                acc_gate, acc_up, _ = compute_tile(
                    acc_gate,
                    acc_up,
                    b_gate_cur,
                    b_up_cur,
                    lds_base_pong,
                    base_k_for_a=k_tail0,
                    a0_prefetch=a0_prefetch_pong,
                    a_tile_regs=(
                        prefetch_a_tile_from_lds(lds_base_pong)
                        if prefetch_a_req
                        else None
                    ),
                )
                a0_prefetch_pong = None
                if const_expr(not use_a_direct and not use_x_dma):
                    store_x_tile_to_lds(x_regs_ping, lds_base_ping)
                hot_loop_scheduler()
                if const_expr(use_x_dma and not use_a_direct):
                    rocdl.s_waitcnt(0)
                stage_barrier()

                # Cross-tile prefetch for the final ping tile.
                a0_prefetch_ping = (
                    (fx.Int64(0).ir_value(), fx.Int64(0).ir_value())
                    if use_a_direct
                    else lds_load_packs_k64(
                        row_a_lds, col_offset_base_bytes, lds_base_ping
                    )
                )
                a_tile_ping = (
                    prefetch_a_tile_from_lds(lds_base_ping)
                    if prefetch_a_req
                    else None
                )

                # Epilogue: compute last tile with epilogue scale prefetch to overlap loads with MFMA.
                acc_gate, acc_up, epilogue_pf = compute_tile(
                    acc_gate,
                    acc_up,
                    b_gate_ping,
                    b_up_ping,
                    lds_base_ping,
                    base_k_for_a=k_tail1,
                    prefetch_epilogue=True,
                    a0_prefetch=a0_prefetch_ping,
                    a_tile_regs=a_tile_ping,
                )

                # Store epilogue to out[t, slot, inter]
                expert_off = expert_off_idx
                tokens_i32_v = tokens_i32
                topk_i32_v = topk_i32
                inter_i32_v = fx.Int32(inter_dim)
                mask24_i32 = fx.Int32(0xFFFFFF)

                if const_expr(use_groupwise_scale):
                    sw_gate_vals = [arith.constant(1.0, type=T.f32)] * num_acc_n
                    sw_up_vals = [arith.constant(1.0, type=T.f32)] * num_acc_n
                    epilogue_tid_vals = None
                elif const_expr(epilogue_pf is not None):
                    sw_gate_vals, sw_up_vals, epilogue_tid_vals = epilogue_pf
                else:
                    epilogue_tid_vals = None
                    sw_gate_vals = []
                    sw_up_vals = []
                    for ni in range_constexpr(num_acc_n):
                        col_g = col_g_list[ni]
                        row_gate_idx = expert_off + col_g
                        row_up_idx = row_gate_idx + inter_idx
                        sw_gate_vals.append(
                            fx.Float32(1.0)
                            if not needs_scale_w
                            else buffer_ops.buffer_load(
                                sw_rsrc, row_gate_idx, vec_width=1, dtype=T.f32
                            )
                        )
                        sw_up_vals.append(
                            fx.Float32(1.0)
                            if not needs_scale_w
                            else buffer_ops.buffer_load(
                                sw_rsrc, row_up_idx, vec_width=1, dtype=T.f32
                            )
                        )

                # When defer_scale16 was used, the x16 correction for v_cvt_off_f32_i4
                # was omitted from the hot loop.  Fold it into the epilogue scale.
                if const_expr(use_gfx950_cvt):
                    _c16 = fx.Float32(16.0)
                    sw_gate_vals = [v * _c16 for v in sw_gate_vals]
                    sw_up_vals = [v * _c16 for v in sw_up_vals]

                # Epilogue hoists to keep IR + Python build time small:
                col_i32_list = []
                for ni in range_constexpr(num_acc_n):
                    col_i32_list.append(arith.index_cast(T.i32, col_g_list[ni]))

                lane_div_16 * fx.Index(4)
                inter_i32_local = inter_i32_v

                # Uses EVec=4 (buffer store "x4" of fp16 elements).
                use_cshuffle_epilog_flag = _use_cshuffle_epilog
                stage1_cshuffle_nlane = cshuffle_nlane_req
                stage1_cshuffle_mlane = total_threads // stage1_cshuffle_nlane
                if tile_m % stage1_cshuffle_mlane != 0:
                    raise ValueError(
                        "AITER_FLYDSL_MOE1_CSHUFFLE_NLANE gives CShuffleMLane that "
                        f"does not divide tile_m: nlane={stage1_cshuffle_nlane}, "
                        f"mlane={stage1_cshuffle_mlane}, tile_m={tile_m}"
                    )
                stage1_cshuffle_evec = (
                    2
                    if (
                        is_bf16
                        and int(tile_m) == 16
                        and int(tile_n) == 256
                        and int(tile_k) == 128
                    )
                    else (4 if tile_n % (stage1_cshuffle_nlane * 4) == 0 else 2)
                )
                if const_expr(
                    cshuffle_evec_req in (2, 4, 8)
                    and tile_n % (stage1_cshuffle_nlane * cshuffle_evec_req) == 0
                ):
                    stage1_cshuffle_evec = cshuffle_evec_req
                stage1_out_elem = T.bf16 if out_dtype == "bf16" else T.f16

                # --- Split-K epilogue: two-pass gate/up with atomic fadd ---
                # bf16 split-K uses bf16 atomics; other dtypes use f32 atomics.
                if const_expr(_is_splitk):
                    if const_expr(lds_out is None):
                        raise RuntimeError(
                            "Split-K epilogue requires lds_out (CShuffle)"
                        )

                    _has_buffer_atomic_bf16_s1 = str(gpu_arch).startswith(
                        ("gfx95", "gfx12")
                    )
                    _needs_global_atomic_bf16_s1 = (
                        _splitk_use_bf16 and not _has_buffer_atomic_bf16_s1
                    )

                    out_base_idx = buffer_ops.extract_base_index(arg_out)
                    _split_k_out_row_stride = (
                        inter_dim * 2 * out_elem_bytes
                    )  # bytes per row
                    _split_k_e_vec = 2  # vec2 for atomic fadd (f32 or bf16)

                    # Mutable slot: 0 for gate pass, inter_dim for up pass
                    _split_k_n_offset = [0]

                    # Mutable slots for two-pass gate/up selection
                    _split_k_acc = [acc_gate]
                    _split_k_sw_vals = [sw_gate_vals]

                    _splitk_lds_elem = T.bf16 if _splitk_use_bf16 else T.f32
                    _splitk_lds_align = 2 if _splitk_use_bf16 else 4

                    def write_row_to_lds_splitk(
                        *,
                        mi: int,
                        ii: int,
                        row_in_tile,
                        row,
                        row_base_lds,
                        col_base_local,
                        num_acc_n: int,
                        lds_out,
                    ):
                        """Write scaled partial sums to LDS (no silu, no doweight)."""
                        _acc = _split_k_acc[0]
                        _sw = _split_k_sw_vals[0]
                        # Load per-row scale_x (sx) -- same logic as normal epilogue.
                        fused2 = buffer_ops.buffer_load(
                            sorted_rsrc, row, vec_width=1, dtype=T.i32
                        )
                        t2 = fused2 & mask24_i32
                        t_valid = arith.cmpi(arith.CmpIPredicate.ult, t2, tokens_i32_v)
                        if const_expr(x_is_token_slot):
                            s2 = fused2 >> 24
                            ts2 = s2 * tokens_i32_v + t2
                            sx = (
                                fx.Float32(1.0)
                                if is_f16_or_bf16
                                else arith.select(
                                    t_valid,
                                    buffer_ops.buffer_load(
                                        sx_rsrc, ts2, vec_width=1, dtype=T.f32
                                    ),
                                    fx.Float32(0.0),
                                )
                            )
                        else:
                            sx = (
                                fx.Float32(1.0)
                                if is_f16_or_bf16
                                else arith.select(
                                    t_valid,
                                    buffer_ops.buffer_load(
                                        sx_rsrc, t2, vec_width=1, dtype=T.f32
                                    ),
                                    fx.Float32(0.0),
                                )
                            )
                        for ni in range_constexpr(num_acc_n):
                            col_local = col_base_local + (ni * 16)
                            acc_idx = mi * num_acc_n + ni
                            v = vector.extract(
                                _acc[acc_idx], static_position=[ii], dynamic_position=[]
                            )
                            if is_int8:
                                v = arith.sitofp(T.f32, v)
                            v = v * sx * _sw[ni]
                            if _splitk_use_bf16:
                                v = arith.trunc_f(T.bf16, v)
                            lds_idx = row_base_lds + col_local
                            v1 = vector.from_elements(T.vec(1, _splitk_lds_elem), [v])
                            vector.store(
                                v1, lds_out, [lds_idx], alignment=_splitk_lds_align
                            )

                    def precompute_row_splitk(*, row_local, row):
                        fused2 = buffer_ops.buffer_load(
                            sorted_rsrc, row, vec_width=1, dtype=T.i32
                        )
                        t2 = fused2 & mask24_i32
                        s2 = fused2 >> 24
                        t_ok = arith.cmpi(arith.CmpIPredicate.ult, t2, tokens_i32_v)
                        t_idx = arith.index_cast(T.index, t2)
                        s_idx = arith.index_cast(T.index, s2)
                        ts_idx = t_idx * arith.index(topk) + s_idx
                        if const_expr(
                            _splitk_use_bf16 and not _needs_global_atomic_bf16_s1
                        ):
                            # For buffer atomics: compute relative byte offset from buffer base
                            row_byte_off = ts_idx * arith.index(_split_k_out_row_stride)
                            return (row_byte_off, t_ok)
                        else:
                            # For global atomics: compute absolute address
                            row_byte_base = out_base_idx + ts_idx * arith.index(
                                _split_k_out_row_stride
                            )
                            return (row_byte_base, t_ok)

                    _splitk_zero_i32 = [fx.Int32(0) if _splitk_use_bf16 else None]

                    def store_pair_splitk(
                        *, row_local, row, row_ctx, col_pair0, col_g0, frag
                    ):
                        row_byte_ctx = row_ctx
                        col_idx = col_g0 + arith.index(_split_k_n_offset[0])
                        byte_off_col = col_idx * arith.index(out_elem_bytes)
                        if const_expr(_splitk_use_bf16):
                            _z = _splitk_zero_i32[0]
                            if const_expr(_needs_global_atomic_bf16_s1):
                                # gfx942: global atomicrmw fadd for bf16
                                ptr_addr_idx = row_byte_ctx + byte_off_col
                                out_ptr = buffer_ops.create_llvm_ptr(
                                    ptr_addr_idx, address_space=1
                                )
                                out_ptr_v = (
                                    out_ptr._value
                                    if hasattr(out_ptr, "_value")
                                    else out_ptr
                                )
                                frag_v = (
                                    frag._value if hasattr(frag, "_value") else frag
                                )
                                llvm.AtomicRMWOp(
                                    llvm.AtomicBinOp.fadd,
                                    out_ptr_v,
                                    frag_v,
                                    llvm.AtomicOrdering.monotonic,
                                    syncscope="agent",
                                    alignment=_split_k_e_vec * out_elem_bytes,
                                )
                            else:
                                # gfx950+: buffer_atomic_pk_add_bf16
                                byte_off_i32 = arith.index_cast(
                                    T.i32, row_byte_ctx + byte_off_col
                                )
                                rocdl.raw_ptr_buffer_atomic_fadd(
                                    frag,
                                    out_rsrc,
                                    byte_off_i32,
                                    _z,
                                    _z,
                                )
                        else:
                            # f32 atomic: global atomicrmw fadd
                            ptr_addr_idx = row_byte_ctx + byte_off_col
                            out_ptr = buffer_ops.create_llvm_ptr(
                                ptr_addr_idx, address_space=1
                            )
                            out_ptr_v = (
                                out_ptr._value
                                if hasattr(out_ptr, "_value")
                                else out_ptr
                            )
                            frag_v = frag._value if hasattr(frag, "_value") else frag
                            llvm.AtomicRMWOp(
                                llvm.AtomicBinOp.fadd,
                                out_ptr_v,
                                frag_v,
                                llvm.AtomicOrdering.monotonic,
                                syncscope="agent",
                                alignment=_split_k_e_vec * out_elem_bytes,
                            )

                    _cshuffle_nlane_splitk = min(32, tile_n // _split_k_e_vec)
                    _splitk_frag_elem = (
                        ir.BF16Type.get() if _splitk_use_bf16 else ir.F32Type.get()
                    )

                    # Pass 1: gate (offset=0)
                    _split_k_acc[0] = acc_gate
                    _split_k_sw_vals[0] = sw_gate_vals
                    _split_k_n_offset[0] = 0
                    c_shuffle_epilog(
                        arith=arith,
                        vector=vector,
                        gpu=gpu,
                        scf=scf,
                        range_constexpr=range_constexpr,
                        tile_m=tile_m,
                        tile_n=tile_n,
                        e_vec=_split_k_e_vec,
                        cshuffle_nlane=_cshuffle_nlane_splitk,
                        block_size=total_threads,
                        m_repeat=m_repeat,
                        num_acc_n=num_acc_n,
                        tx=tx,
                        lane_div_16=lane_div_16,
                        lane_mod_16=lane_mod_16,
                        bx_m=bx_m,
                        by_n=by_n,
                        n_tile_base=n_tile_base,
                        lds_out=lds_out,
                        frag_elem_type=_splitk_frag_elem,
                        write_row_to_lds=write_row_to_lds_splitk,
                        precompute_row=precompute_row_splitk,
                        store_pair=store_pair_splitk,
                    )

                    gpu.barrier()

                    # Pass 2: up (offset=inter_dim)
                    _split_k_acc[0] = acc_up
                    _split_k_sw_vals[0] = sw_up_vals
                    _split_k_n_offset[0] = inter_dim
                    c_shuffle_epilog(
                        arith=arith,
                        vector=vector,
                        gpu=gpu,
                        scf=scf,
                        range_constexpr=range_constexpr,
                        tile_m=tile_m,
                        tile_n=tile_n,
                        e_vec=_split_k_e_vec,
                        cshuffle_nlane=_cshuffle_nlane_splitk,
                        block_size=total_threads,
                        m_repeat=m_repeat,
                        num_acc_n=num_acc_n,
                        tx=tx,
                        lane_div_16=lane_div_16,
                        lane_mod_16=lane_mod_16,
                        bx_m=bx_m,
                        by_n=by_n,
                        n_tile_base=n_tile_base,
                        lds_out=lds_out,
                        frag_elem_type=_splitk_frag_elem,
                        write_row_to_lds=write_row_to_lds_splitk,
                        precompute_row=precompute_row_splitk,
                        store_pair=store_pair_splitk,
                    )
                    return

                if const_expr(use_cshuffle_epilog_flag):
                    if const_expr(lds_out is None):
                        raise RuntimeError(
                            "CShuffle epilogue enabled but lds_out is not allocated/aliased."
                        )

                    def write_row_to_lds(
                        *,
                        mi: int,
                        ii: int,
                        row_in_tile,
                        row,
                        row_base_lds,
                        col_base_local,
                        num_acc_n: int,
                        lds_out,
                    ):
                        # `row` is the sorted-row index (bx_m + row_in_tile).
                        fused2 = (
                            memref.load(lds_tid, [row_in_tile])
                            if use_tid_lds_req
                            else buffer_ops.buffer_load(
                                sorted_rsrc, row, vec_width=1, dtype=T.i32
                            )
                        )
                        t2 = fused2 & mask24_i32
                        s2 = fused2 >> 24
                        # aiter moe_sorting uses sentinel token_id == tokens for padding.
                        # Do NOT rely on buffer OOB semantics for scale loads; explicitly mask.
                        t_valid = arith.cmpi(arith.CmpIPredicate.ult, t2, tokens_i32_v)
                        if const_expr(x_is_token_slot):
                            # slot-major: slot*tokens + token
                            ts2 = s2 * tokens_i32_v + t2
                            sx = (
                                fx.Float32(1.0)
                                if is_f16_or_bf16
                                else arith.select(
                                    t_valid,
                                    buffer_ops.buffer_load(
                                        sx_rsrc, ts2, vec_width=1, dtype=T.f32
                                    ),
                                    fx.Float32(0.0),
                                )
                            )
                        else:
                            sx = (
                                fx.Float32(1.0)
                                if is_f16_or_bf16
                                else arith.select(
                                    t_valid,
                                    buffer_ops.buffer_load(
                                        sx_rsrc, t2, vec_width=1, dtype=T.f32
                                    ),
                                    fx.Float32(0.0),
                                )
                            )

                        # Sorted weight aligned with `row` (matches aiter moe_sorting output).
                        if const_expr(doweight_stage1):
                            tw = buffer_ops.buffer_load(
                                sorted_w_rsrc, row, vec_width=1, dtype=T.f32
                            )

                        for ni in range_constexpr(num_acc_n):
                            col_local = col_base_local + (ni * 16)
                            sw_gate = sw_gate_vals[ni]
                            sw_up = sw_up_vals[ni]

                            acc_idx = mi * num_acc_n + ni
                            vg = vector.extract(
                                acc_gate[acc_idx],
                                static_position=[ii],
                                dynamic_position=[],
                            )
                            vu = vector.extract(
                                acc_up[acc_idx],
                                static_position=[ii],
                                dynamic_position=[],
                            )

                            if const_expr(is_int8):
                                vg = arith.sitofp(T.f32, vg)
                                vu = arith.sitofp(T.f32, vu)
                            vg = vg * sx * sw_gate
                            vu = vu * sx * sw_up

                            y = silu(vg) * vu
                            if const_expr(doweight_stage1):
                                y = y * tw
                            y_out = arith.trunc_f(stage1_out_elem, y)

                            lds_idx = row_base_lds + col_local
                            v1 = vector.from_elements(T.vec(1, stage1_out_elem), [y_out])
                            vector.store(v1, lds_out, [lds_idx], alignment=2)

                    def precompute_row(*, row_local, row):
                        fused2 = (
                            memref.load(lds_tid, [row_local])
                            if use_tid_lds_req
                            else buffer_ops.buffer_load(
                                sorted_rsrc, row, vec_width=1, dtype=T.i32
                            )
                        )
                        t2 = fused2 & mask24_i32
                        s2 = fused2 >> 24
                        t_valid = arith.cmpi(
                            arith.CmpIPredicate.ult, t2, tokens_i32_v
                        )
                        return (t2 * topk_i32_v + s2) * inter_i32_local, t_valid

                    def store_pair(*, row_local, row, row_ctx, col_pair0, col_g0, frag):
                        idx0 = row_ctx
                        col_i32 = arith.index_cast(T.i32, col_g0)
                        idx_out = idx0 + col_i32
                        # Vectorized fp16/bf16 store (EVec=2/4/8). Row validity
                        # is handled once by c_shuffle_epilog's row predicate.
                        buffer_ops.buffer_store(
                            frag,
                            out_rsrc,
                            idx_out,
                            cache_modifier=out_cache_modifier_req,
                        )

                    mfma_epilog(
                        use_cshuffle=True,
                        arith=arith,
                        vector=vector,
                        gpu=gpu,
                        scf=scf,
                        range_constexpr=range_constexpr,
                        tile_m=tile_m,
                        tile_n=tile_n,
                        e_vec=stage1_cshuffle_evec,
                        cshuffle_nlane=stage1_cshuffle_nlane,
                        block_size=total_threads,
                        m_repeat=m_repeat,
                        num_acc_n=num_acc_n,
                        tx=tx,
                        lane_div_16=lane_div_16,
                        lane_mod_16=lane_mod_16,
                        bx_m=bx_m,
                        by_n=by_n,
                        n_tile_base=n_tile_base,
                        lds_out=lds_out,
                        write_row_to_lds=write_row_to_lds,
                        precompute_row=precompute_row,
                        store_pair=store_pair,
                        frag_elem_type=stage1_out_elem,
                    )
                    return

                def _stage1_store_row(*, mi: int, ii: int, row_in_tile, row):
                    # `row` is the sorted-row index (bx_m + row_in_tile).
                    # Block-level early-exit already guards `bx_m` range.
                    # Here we rely on buffer OOB semantics for any tail rows.
                    fused2 = (
                        epilogue_tid_vals[ii]
                        if prefetch_epi_tid_req and epilogue_tid_vals is not None
                        else (
                            memref.load(lds_tid, [row_in_tile])
                            if use_tid_lds_req
                            else buffer_ops.buffer_load(
                                sorted_rsrc, row, vec_width=1, dtype=T.i32
                            )
                        )
                    )
                    t2_raw = fused2 & mask24_i32
                    s2_raw = fused2 >> 24
                    t2 = t2_raw
                    s2 = s2_raw
                    t_valid = arith.cmpi(arith.CmpIPredicate.ult, t2, tokens_i32_v)

                    # Do NOT rely on buffer OOB semantics for scale loads; explicitly mask.
                    if const_expr(x_is_token_slot):
                        # slot-major: slot*tokens + token
                        ts2 = s2 * tokens_i32_v + t2
                        sx0 = (
                            fx.Float32(1.0)
                            if is_f16_or_bf16
                            else arith.select(
                                t_valid,
                                buffer_ops.buffer_load(
                                    sx_rsrc, ts2, vec_width=1, dtype=T.f32
                                ),
                                fx.Float32(0.0),
                            )
                        )
                    else:
                        sx0 = (
                            fx.Float32(1.0)
                            if is_f16_or_bf16
                            else arith.select(
                                t_valid,
                                buffer_ops.buffer_load(
                                    sx_rsrc, t2, vec_width=1, dtype=T.f32
                                ),
                                fx.Float32(0.0),
                            )
                        )
                    sx = sx0
                    arith.constant(0.0, type=out_mlir())

                    # out linear index base = ((t*topk + s)*inter_dim) (invariant across ni)
                    idx0 = (t2 * topk_i32_v + s2) * inter_i32_local

                    # Sorted weight aligned with `row` (matches aiter moe_sorting output).
                    if const_expr(doweight_stage1):
                        tw = buffer_ops.buffer_load(
                            sorted_w_rsrc, row, vec_width=1, dtype=T.f32
                        )

                    def _store_valid_row():
                        for ni in range_constexpr(num_acc_n):
                            col_i32 = col_i32_list[ni]
                            sw_gate = sw_gate_vals[ni]
                            sw_up = sw_up_vals[ni]

                            acc_idx = mi * num_acc_n + ni
                            vg = vector.extract(
                                acc_gate[acc_idx],
                                static_position=[ii],
                                dynamic_position=[],
                            )
                            vu = vector.extract(
                                acc_up[acc_idx],
                                static_position=[ii],
                                dynamic_position=[],
                            )

                            if const_expr(is_int8):
                                vg = arith.sitofp(T.f32, vg)
                                vu = arith.sitofp(T.f32, vu)
                            vg = vg * sx * sw_gate
                            vu = vu * sx * sw_up

                            y = silu(vg) * vu
                            if const_expr(doweight_stage1):
                                y = y * tw
                            y = arith.trunc_f(out_mlir(), y)
                            idx_out0 = idx0 + col_i32
                            buffer_ops.buffer_store(
                                y,
                                out_rsrc,
                                idx_out0,
                                cache_modifier=out_cache_modifier_req,
                            )

                    _if_valid = scf.IfOp(t_valid)
                    with _if_then(_if_valid):
                        _store_valid_row()

                def _stage1_store_row_maybe_limited(
                    *, mi: int, ii: int, row_in_tile, row
                ):
                    if const_expr(row_limit_epilog_req > 0):
                        row_in_limit = arith.cmpi(
                            arith.CmpIPredicate.ult,
                            row_in_tile,
                            fx.Index(row_limit_epilog_req),
                        )
                        _if_row_limit = scf.IfOp(row_in_limit)
                        with _if_then(_if_row_limit):
                            _stage1_store_row(
                                mi=mi, ii=ii, row_in_tile=row_in_tile, row=row
                            )
                    else:
                        _stage1_store_row(
                            mi=mi, ii=ii, row_in_tile=row_in_tile, row=row
                        )

                mfma_epilog(
                    use_cshuffle=False,
                    arith=arith,
                    range_constexpr=range_constexpr,
                    m_repeat=m_repeat,
                    lane_div_16=lane_div_16,
                    bx_m=bx_m,
                    body_row=_stage1_store_row_maybe_limited,
                )

    # -- Host launcher (flyc.jit + .launch) --------------------------------
    @flyc.jit
    def launch_moe_gemm1(
        arg_out: fx.Tensor,
        arg_x: fx.Tensor,
        arg_w: fx.Tensor,
        arg_scale_x: fx.Tensor,
        arg_scale_w: fx.Tensor,
        arg_sorted_token_ids: fx.Tensor,
        arg_expert_ids: fx.Tensor,
        arg_sorted_weights: fx.Tensor,
        arg_max_token_ids: fx.Tensor,
        i32_tokens_in: fx.Int32,
        i32_inter_in: fx.Int32,
        i32_k_in: fx.Int32,
        i32_size_expert_ids_in: fx.Int32,
        stream: fx.Stream,
    ):
        _ = _cache_tag
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        if const_expr(waves_per_eu > 0):
            for op in ctx.gpu_module_body.operations:
                if const_expr(
                    hasattr(op, "attributes") and op.OPERATION_NAME == "gpu.func"
                ):
                    op.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(
                        T.i32, int(waves_per_eu)
                    )

        inter_in = arith.index_cast(T.index, i32_inter_in)
        size_expert_ids_in = arith.index_cast(T.index, i32_size_expert_ids_in)
        gx = inter_in // fx.Index(tile_n)
        gy = size_expert_ids_in

        moe_gemm1(
            arg_out,
            arg_x,
            arg_w,
            arg_scale_x,
            arg_scale_w,
            arg_sorted_token_ids,
            arg_expert_ids,
            arg_sorted_weights,
            arg_max_token_ids,
            i32_tokens_in,
            i32_inter_in,
            i32_k_in,
            i32_size_expert_ids_in,
        ).launch(
            grid=(gx, gy, k_batch),
            block=(total_threads, 1, 1),
            stream=stream,
        )

    return launch_moe_gemm1


@functools.lru_cache(maxsize=1024)
def compile_moe_gemm2(
    *,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    doweight_stage2: bool,
    in_dtype: str = "fp8",
    group_size: int = -1,
    out_dtype: str = "f16",
    use_cshuffle_epilog: bool | None = None,
    accumulate: bool = True,
    scale_is_bf16: bool = False,
    b_nt: int = 0,
    xcd_swizzle: int = 0,
):
    """Compile stage2 kernel (`moe_gemm2`) and return the compiled executable.

    in_dtype:
      - "fp8": A2/W are fp8
      - "fp16": A2/W are fp16
      - "bf16": A2/W are bf16
      - "int8": A2/W are int8
      - "int4": W4A8 path: A2 is int8, W is packed int4 unpacked to int8 in-kernel
      - "int4_bf16": W4A16 path: A2 is bf16, W is packed int4 unpacked to bf16 in-kernel
    scale_is_bf16: When True, groupwise scales are bf16 (halves scale bandwidth).

    Stage2 output supports:
      - out_dtype="f16": fp16 half2 atomics (fast, can overflow to +/-inf for bf16 workloads)
      - out_dtype="f32": fp32 scalar atomics (slower, but avoids fp16 atomic overflow)

    `use_cshuffle_epilog` controls whether we use the LDS CShuffle epilogue before
    global atomics (recommended for performance).
    """
    gpu_arch = get_hip_arch()
    allocator = SmemAllocator(None, arch=gpu_arch)
    _state = {}

    _valid_dtypes = ("fp8", "fp16", "bf16", "int8", "int8smooth", "int4", "int4_bf16")
    if in_dtype not in _valid_dtypes:
        raise ValueError(f"in_dtype must be one of {_valid_dtypes}, got {in_dtype!r}")
    is_int4_bf16 = (
        in_dtype == "int4_bf16"
    )  # W4A16: bf16 activations, packed int4 weights
    is_f16 = in_dtype == "fp16"
    is_bf16 = is_int4_bf16 or in_dtype == "bf16"
    is_f16_or_bf16 = is_f16 or is_bf16
    needs_scale_w = (not is_f16_or_bf16) or is_int4_bf16
    elem_bytes = 2 if is_f16_or_bf16 else 1
    out_s = str(out_dtype).strip().lower()
    if out_s not in ("f16", "fp16", "half", "bf16", "bfloat16", "f32", "fp32", "float"):
        raise ValueError(
            f"out_dtype must be 'f16', 'bf16', or 'f32', got {out_dtype!r}"
        )
    out_is_f32 = out_s in ("f32", "fp32", "float")
    out_is_bf16 = out_s in ("bf16", "bfloat16")
    if (not bool(accumulate)) and out_is_f32:
        raise ValueError(
            "compile_moe_gemm2(accumulate=False) only supports out_dtype in {'f16','bf16'}"
        )
    is_int4 = in_dtype == "int4"
    # w_is_int4: True for any variant where weights are packed int4.
    w_is_int4 = is_int4 or is_int4_bf16
    # INT4 here means W4A8: A2 is int8, W is packed int4 and unpacked to int8 in-kernel.
    is_int8 = (in_dtype in ("int8", "int8smooth")) or is_int4

    # Group-wise scale support for W4A16
    use_groupwise_scale = w_is_int4 and group_size > 0
    if use_groupwise_scale and group_size != 32:
        raise ValueError(
            f"FlyDSL groupwise scale only supports group_size=32, got {group_size}. "
            f"This is due to int4 preshuffle layout constraints. "
            f"Please use Triton kernel for other group sizes."
        )
    is_int4_bf16_groupwise = is_int4_bf16 and use_groupwise_scale
    # Stage2 K dimension is inter_dim (weight shape: [E, model_dim, inter_dim])
    num_groups = inter_dim // group_size if use_groupwise_scale else 1
    _scale_is_bf16 = scale_is_bf16 and use_groupwise_scale
    experts * model_dim * num_groups

    _is_gfx950 = "gfx95" in get_hip_arch()
    _has_cvt_off_f32_i4 = hasattr(rocdl, "cvt_off_f32_i4")
    use_gfx950_cvt = is_int4_bf16 and _is_gfx950 and _has_cvt_off_f32_i4

    mfma_i32_k32 = None
    if is_int8:
        mfma_i32_k32 = getattr(rocdl, "mfma_i32_16x16x32i8", None) or getattr(
            rocdl, "mfma_i32_16x16x32_i8", None
        )
        if mfma_i32_k32 is None:
            raise AttributeError(
                "INT8 K32 MFMA op not found: expected `rocdl.mfma_i32_16x16x32i8` "
                "(or `rocdl.mfma_i32_16x16x32_i8`)."
            )

    mfma_f32_bf16_k16 = None
    if is_bf16:
        mfma_f32_bf16_k16 = getattr(rocdl, "mfma_f32_16x16x16bf16_1k", None) or getattr(
            rocdl, "mfma_f32_16x16x16_bf16_1k", None
        )
        if mfma_f32_bf16_k16 is None:
            raise AttributeError(
                "BF16 K16 MFMA op not found: expected `rocdl.mfma_f32_16x16x16bf16_1k` "
                "(or `rocdl.mfma_f32_16x16x16_bf16_1k`)."
            )

    # gfx950: use 16x16x32 MFMA for f16/bf16 (K=32 per MFMA, vs K=16 on gfx942).
    # Check if K=32 MFMA supports the (result_type, operands_list) calling convention.
    _has_k32_mfma_compat = False
    if _is_gfx950 and (is_f16 or is_bf16):
        import inspect

        _k32_fn = (
            rocdl.mfma_f32_16x16x32_bf16 if is_bf16 else rocdl.mfma_f32_16x16x32_f16
        )
        try:
            _k32_sig = inspect.signature(_k32_fn)
            _k32_params = list(_k32_sig.parameters.keys())
            # Compatible if second param is "operands" (list-based API)
            _has_k32_mfma_compat = (
                len(_k32_params) >= 2 and _k32_params[1] == "operands"
            )
        except (ValueError, TypeError):
            _has_k32_mfma_compat = False
    _use_mfma_k32 = _is_gfx950 and (is_f16 or is_bf16) and _has_k32_mfma_compat

    ir.ShapedType.get_dynamic_size()
    # W is packed int4 for W4A8/W4A16/W4A_FP8: 2 values per byte.
    (
        (experts * model_dim * inter_dim) // 2
        if w_is_int4
        else (experts * model_dim * inter_dim)
    )

    total_threads = int(os.environ.get("AITER_FLYDSL_MOE2_THREADS", "256"))
    if total_threads not in (128, 256, 512):
        raise ValueError(
            f"AITER_FLYDSL_MOE2_THREADS must be 128, 256, or 512, got {total_threads}"
        )
    num_waves_static = total_threads // 64
    tile_k_bytes = int(tile_k) * int(elem_bytes)
    if (tile_k_bytes % 64) != 0:
        raise ValueError(
            f"tile_k_bytes must be divisible by 64, got tile_k_bytes={tile_k_bytes} "
            f"(tile_k={tile_k}, elem_bytes={elem_bytes})"
        )
    bytes_x_per_tile = int(tile_m) * int(tile_k) * int(elem_bytes)
    if bytes_x_per_tile % total_threads != 0:
        raise ValueError(
            "tile_m*tile_k*elem_bytes must be divisible by "
            f"{total_threads}: tile_m={tile_m}, tile_k={tile_k}, elem_bytes={elem_bytes}"
        )
    bytes_per_thread_x = bytes_x_per_tile // total_threads

    _ck_lds128 = os.environ.get("FLYDSL_CK_LDS128", "1") in (
        "1",
        "true",
        "True",
        "YES",
        "yes",
    )
    pad_k = 0 if _ck_lds128 else 8
    lds_stride = tile_k + pad_k
    # gfx950+ has buffer_atomic_pk_add_bf16 -> bf16 can use buffer atomics (same as f16).
    # gfx942 only has global_atomic_pk_add_bf16 -> must use global atomics with raw pointer.
    _has_buffer_atomic_bf16 = str(gpu_arch).startswith(("gfx95", "gfx12"))
    _needs_global_atomic_bf16 = out_is_bf16 and not _has_buffer_atomic_bf16
    if out_is_bf16:
        if not supports_bf16_global_atomics(gpu_arch):
            raise ValueError(
                f"out_dtype='bf16' requires bf16 global atomics ({bf16_global_atomics_arch_description()}), got arch={gpu_arch!r}"
            )

    if out_is_f32:
        # Match origin/dev_a16w4: f32 output uses scalar atomics and does NOT use the CShuffle epilogue.
        _use_cshuffle_epilog = (
            False if use_cshuffle_epilog is None else bool(use_cshuffle_epilog)
        )
        if _use_cshuffle_epilog:
            raise ValueError(
                "out_dtype='f32' does not support CShuffle epilogue (set use_cshuffle_epilog=False)."
            )
    else:
        if use_cshuffle_epilog is None:
            _use_cshuffle_epilog = os.environ.get(
                "FLYDSL_MOE_STAGE2_CSHUFFLE", "1"
            ) in (
                "1",
                "true",
                "True",
                "YES",
                "yes",
            )
        else:
            _use_cshuffle_epilog = bool(use_cshuffle_epilog)
        if not _use_cshuffle_epilog:
            raise ValueError(
                "stage2 f16 output currently requires CShuffle epilogue (FLYDSL_MOE_STAGE2_CSHUFFLE=1)."
            )

    # NOTE: Keep this as a callable so we don't require an MLIR Context at Python-time.
    def out_elem():
        ty = T.f32 if out_is_f32 else (T.bf16 if out_is_bf16 else T.f16)
        return ty() if callable(ty) else ty

    epilog_tag = "cshuffle"
    # IMPORTANT: include tiling in the module name to avoid accidentally reusing a compiled
    # binary for a different (tile_m, tile_n, tile_k) configuration.
    # See stage1 note: include ABI tag to prevent binary reuse across signature changes.
    # IMPORTANT: module name participates in FlyDSL's compile cache key.
    # Dynamic-shape variant: safe to reuse across (tokens/sorted_size/size_expert_ids) at runtime.
    # Keep a distinct ABI tag so the compile cache never mixes with historical signatures.
    _sched_enabled = os.environ.get("AITER_FLYDSL_MOE2_SCHED", "0") in (
        "1",
        "true",
        "True",
        "YES",
        "yes",
    )
    _sched_vmem_req = int(os.environ.get("AITER_FLYDSL_MOE2_SCHED_VMEM", "1"))
    _sched_early_vmem_req = int(
        os.environ.get("AITER_FLYDSL_MOE2_SCHED_EARLY_VMEM", str(_sched_vmem_req))
    )
    _sched_dswr_advance_req = int(
        os.environ.get("AITER_FLYDSL_MOE2_DSWR_ADVANCE", "0")
    )
    if _sched_vmem_req < 0 or _sched_early_vmem_req < 0 or _sched_dswr_advance_req < 0:
        raise ValueError(
            "AITER_FLYDSL_MOE2_SCHED_VMEM, AITER_FLYDSL_MOE2_SCHED_EARLY_VMEM, "
            "and AITER_FLYDSL_MOE2_DSWR_ADVANCE must be non-negative"
        )
    _dense_b_k64_single_load = os.environ.get("AITER_FLYDSL_MOE2_BK64", "0") in (
        "1",
        "true",
        "True",
        "YES",
        "yes",
    )
    _dense_b_k64_layout_load = os.environ.get("AITER_FLYDSL_MOE2_BK64_LAYOUT", "0") in (
        "1",
        "true",
        "True",
        "YES",
        "yes",
    )
    if _dense_b_k64_layout_load:
        _dense_b_k64_single_load = True
    _cshuffle_nlane_req = int(os.environ.get("AITER_FLYDSL_MOE2_CSHUFFLE_NLANE", "32"))
    if _cshuffle_nlane_req <= 0 or total_threads % _cshuffle_nlane_req != 0:
        raise ValueError(
            "AITER_FLYDSL_MOE2_CSHUFFLE_NLANE must divide total_threads "
            f"{total_threads}, got {_cshuffle_nlane_req}"
        )
    waves_per_eu = int(os.environ.get("AITER_FLYDSL_MOE2_WPE", "0"))
    if waves_per_eu < 0:
        raise ValueError(f"AITER_FLYDSL_MOE2_WPE must be >= 0, got {waves_per_eu}")
    _fast_barrier_req = os.environ.get("AITER_FLYDSL_MOE2_FAST_BARRIER", "0").strip() in (
        "1",
        "true",
        "True",
        "YES",
        "yes",
    )
    _grid_m_fast_req = os.environ.get("AITER_FLYDSL_MOE2_GRID_M_FAST", "0").strip() in (
        "1",
        "true",
        "True",
        "YES",
        "yes",
    )
    if _grid_m_fast_req and xcd_swizzle > 0:
        raise ValueError("AITER_FLYDSL_MOE2_GRID_M_FAST does not support xcd_swizzle")
    _use_a_direct_req = os.environ.get("AITER_FLYDSL_MOE2_A_DIRECT", "0").strip() in (
        "1",
        "true",
        "True",
        "YES",
        "yes",
    )
    _use_a_direct_req = _use_a_direct_req and is_f16_or_bf16
    _late_b_req = os.environ.get("AITER_FLYDSL_MOE2_LATE_B", "0").strip() in (
        "1",
        "true",
        "True",
        "YES",
        "yes",
    )
    _stream_b_req = os.environ.get("AITER_FLYDSL_MOE2_STREAM_B", "0").strip() in (
        "1",
        "true",
        "True",
        "YES",
        "yes",
    )
    _stream_b_req = _stream_b_req and is_f16_or_bf16 and not use_groupwise_scale
    _row16_skip_req = os.environ.get("AITER_FLYDSL_MOE2_ROW16_SKIP", "0").strip() in (
        "1",
        "true",
        "True",
        "YES",
        "yes",
    )
    stage2_row_skip_supported = is_f16_or_bf16 or in_dtype == "fp8"
    _row16_skip_req = _row16_skip_req and stage2_row_skip_supported and int(tile_m) in (32, 48)
    _row32_skip_req = os.environ.get("AITER_FLYDSL_MOE2_ROW32_SKIP", "0").strip() in (
        "1",
        "true",
        "True",
        "YES",
        "yes",
    )
    _row32_skip_req = _row32_skip_req and stage2_row_skip_supported and int(tile_m) == 48
    _row32_skip_outer_req = os.environ.get(
        "AITER_FLYDSL_MOE2_ROW32_SKIP_OUTER", "0"
    ).strip() in (
        "1",
        "true",
        "True",
        "YES",
        "yes",
    )
    _row32_skip_outer_req = (
        _row32_skip_outer_req
        and stage2_row_skip_supported
        and int(tile_m) == 48
        and not _stream_b_req
    )
    if _row32_skip_outer_req:
        _row32_skip_req = True
    _epilog_row_skip_req = os.environ.get(
        "AITER_FLYDSL_MOE2_EPILOG_ROW_SKIP", "0"
    ).strip() in (
        "1",
        "true",
        "True",
        "YES",
        "yes",
    )
    _epilog_row_skip_req = _epilog_row_skip_req and stage2_row_skip_supported
    _rowctx_bcast_req = os.environ.get(
        "AITER_FLYDSL_MOE2_ROWCTX_BCAST", "0"
    ).strip() in (
        "1",
        "true",
        "True",
        "YES",
        "yes",
    )
    _rowctx_bcast_req = _rowctx_bcast_req and stage2_row_skip_supported
    _rowctx_base_req = os.environ.get(
        "AITER_FLYDSL_MOE2_ROWCTX_BASE", "0"
    ).strip() in (
        "1",
        "true",
        "True",
        "YES",
        "yes",
    )
    _rowctx_base_req = _rowctx_base_req and stage2_row_skip_supported
    _skip_even_mask_req = os.environ.get(
        "AITER_FLYDSL_MOE2_SKIP_EVEN_MASK", "0"
    ).strip() in (
        "1",
        "true",
        "True",
        "YES",
        "yes",
    )
    _skip_even_mask_req = _skip_even_mask_req and stage2_row_skip_supported
    _atomic_aux_req = int(os.environ.get("AITER_FLYDSL_MOE2_ATOMIC_AUX", "0"))
    if _atomic_aux_req < 0:
        raise ValueError(
            f"AITER_FLYDSL_MOE2_ATOMIC_AUX must be >= 0, got {_atomic_aux_req}"
        )
    _gs_tag = f"_g{group_size}" if use_groupwise_scale else ""
    scale_tag = "_sbf16" if _scale_is_bf16 else ""
    _xcd_tag = f"_xcd{xcd_swizzle}" if xcd_swizzle > 0 else ""
    module_name = (
        f"mfma_moe2_{in_dtype}_{out_s}_{epilog_tag}"
        f"_t{tile_m}x{tile_n}x{tile_k}"
        f"{_gs_tag}{scale_tag}"
        f"_bnt{b_nt}"
        f"_thr{total_threads}"
        f"_sched{int(_sched_enabled)}"
        f"_svm{int(_sched_vmem_req)}"
        f"_sevm{int(_sched_early_vmem_req)}"
        f"_dswa{int(_sched_dswr_advance_req)}"
        f"_bk64{int(_dense_b_k64_single_load)}"
        f"_bk64l{int(_dense_b_k64_layout_load)}"
        f"_csn{int(_cshuffle_nlane_req)}"
        f"_wpe{int(waves_per_eu)}"
        f"_fbar{int(_fast_barrier_req)}"
        f"_gmf{int(_grid_m_fast_req)}"
        f"_adir{int(_use_a_direct_req)}"
        f"_lateb{int(_late_b_req)}"
        f"_strb{int(_stream_b_req)}"
        f"_r16s{int(_row16_skip_req)}"
        f"_r32s{int(_row32_skip_req)}"
        f"_r32o{int(_row32_skip_outer_req)}"
        f"_ers{int(_epilog_row_skip_req)}"
        f"_rcb{int(_rowctx_bcast_req)}"
        f"_rbase{int(_rowctx_base_req)}"
        f"_noemask{int(_skip_even_mask_req)}"
        f"_aatom{int(_atomic_aux_req)}"
        f"{_xcd_tag}"
        f"_abi29"  # stage2 FP8 epilogue-row context support and cache-policy knobs
    ).replace("-", "_")
    _cache_tag = (
        module_name,
        model_dim,
        inter_dim,
        experts,
        topk,
        tile_m,
        tile_n,
        tile_k,
        doweight_stage2,
        in_dtype,
        group_size,
        out_dtype,
        use_cshuffle_epilog,
        accumulate,
        scale_is_bf16,
        b_nt,
        total_threads,
        _sched_enabled,
        _sched_vmem_req,
        _sched_early_vmem_req,
        _sched_dswr_advance_req,
        _dense_b_k64_single_load,
        _dense_b_k64_layout_load,
        _cshuffle_nlane_req,
        waves_per_eu,
        _fast_barrier_req,
        _grid_m_fast_req,
        _use_a_direct_req,
        _late_b_req,
        _stream_b_req,
        _row16_skip_req,
        _row32_skip_req,
        _row32_skip_outer_req,
        _epilog_row_skip_req,
        _rowctx_bcast_req,
        _rowctx_base_req,
        _skip_even_mask_req,
        _atomic_aux_req,
        xcd_swizzle,
    )

    # -- CShuffle epilogue e_vec (pure Python; must be computed before @flyc.kernel
    # because the AST rewriter intercepts `if` statements inside kernel bodies and
    # turns them into closure dispatches, which breaks variable reassignment) ----
    _cshuffle_nlane = _cshuffle_nlane_req
    if bool(accumulate):
        _e_vec = 2
    else:
        _e_vec = 8 if int(tile_n) % (_cshuffle_nlane * 8) == 0 else 2
        _cshuffle_stride = _cshuffle_nlane * _e_vec
        if int(tile_n) % _cshuffle_stride != 0:
            raise ValueError(
                f"tile_n={tile_n} must be divisible by {_cshuffle_stride} when accumulate=False"
            )

    # -- LDS sizing (pure Python; no MLIR Context needed) ---------------------
    lds_x_bytes = 2 * int(tile_m) * int(lds_stride) * int(elem_bytes)
    lds_out_bytes = (
        2 * int(tile_m) * int(tile_n) if _use_cshuffle_epilog else 0
    )  # f16 bytes
    lds_total_bytes = max(lds_x_bytes, lds_out_bytes)
    lds_total_elems = lds_total_bytes if elem_bytes == 1 else (lds_total_bytes // 2)

    lds_alloc_bytes = int(lds_total_elems) * int(elem_bytes)
    lds_alloc_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_alloc_offset + lds_alloc_bytes

    if True:

        @flyc.kernel(
            name=f"moe_gemm2_m{tile_m}_n{tile_n}_k{tile_k}",
            known_block_size=[total_threads, 1, 1],
        )
        def moe_gemm2(
            arg_out: fx.Tensor,
            arg_x: fx.Tensor,
            arg_w: fx.Tensor,
            arg_scale_x: fx.Tensor,
            arg_scale_w: fx.Tensor,
            arg_sorted_token_ids: fx.Tensor,
            arg_expert_ids: fx.Tensor,
            arg_sorted_weights: fx.Tensor,
            arg_num_valid_ids: fx.Tensor,
            i32_tokens_in: fx.Int32,
            i32_n_in: fx.Int32,
            i32_k_in: fx.Int32,
            i32_size_expert_ids_in: fx.Int32,
        ):
            tokens_in = arith.index_cast(T.index, i32_tokens_in)
            n_in = arith.index_cast(T.index, i32_n_in)
            k_in = arith.index_cast(T.index, i32_k_in)
            size_expert_ids_in = arith.index_cast(T.index, i32_size_expert_ids_in)
            # i32 versions for layout construction (fly.make_shape requires i32/i64)
            k_i32_v = i32_k_in
            x_elem = (
                T.bf16
                if is_bf16
                else (T.f16 if is_f16 else (T.i8 if is_int8 else T.f8))
            )
            # For int4/int4_bf16, weights are stored as packed bytes (i8) and unpacked in-kernel.
            w_elem = (
                T.i8
                if w_is_int4
                else (
                    T.bf16
                    if is_bf16
                    else (T.f16 if is_f16 else (T.i8 if is_int8 else T.f8))
                )
            )
            scale_dtype = T.bf16 if _scale_is_bf16 else T.f32
            vec16_elems = 16 if elem_bytes == 1 else 8
            vec8_elems = 8 if elem_bytes == 1 else 4
            vec4_elems = 4 if elem_bytes == 1 else 2
            vec8_x = T.vec(vec8_elems, x_elem)
            vec16_x = T.vec(vec16_elems, x_elem)

            acc_init = (
                arith.constant_vector(0, T.i32x4)
                if is_int8
                else arith.constant_vector(0.0, T.f32x4)
            )
            zero_f32_acc = (
                arith.constant_vector(0.0, T.f32x4) if is_int4_bf16_groupwise else None
            )

            # A2 layout (flatten token-slot -> M; use i32 for fly.make_shape).
            topk_idx = fx.Index(topk)
            m_in = tokens_in * topk_idx
            m_i32_v = arith.index_cast(T.i32, m_in)
            fx.make_layout((m_i32_v, k_i32_v), stride=(k_i32_v, 1))

            # B preshuffle layout: [experts*model_dim, inter_dim]
            c_n_total = arith.index(experts * model_dim)
            # For packed int4 (W4A8/W4A16/W4A_FP8), kpack_bytes=8.
            kpack_bytes = 8 if w_is_int4 else 16
            w_elem_bytes = 1 if w_is_int4 else elem_bytes
            b_layout = make_preshuffle_b_layout(
                arith,
                c_n=c_n_total,
                c_k=k_in,
                kpack_bytes=kpack_bytes,
                elem_bytes=w_elem_bytes,
            )
            layout_b = b_layout.layout_b
            (k_in * arith.index(int(elem_bytes))) // fx.Index(64)

            shape_lds = fx.make_shape(tile_m, tile_k)
            stride_lds = fx.make_stride(lds_stride, 1)
            layout_lds = fx.make_layout(shape_lds, stride_lds)

            tx = gpu.thread_id("x")
            # Default maps blockIdx.x -> N and blockIdx.y -> sorted M (AITER/CK order).
            # M-fast order flips the launch grid so adjacent CTAs walk sorted M blocks
            # for the same N tile, improving same-expert B-tile cache reuse.
            if const_expr(_grid_m_fast_req):
                bx = gpu.block_id("x")  # tile along sorted M
                by = gpu.block_id("y")  # tile along model_dim
            else:
                by = gpu.block_id("x")  # tile along model_dim
                bx = gpu.block_id("y")  # tile along sorted M

            if const_expr(xcd_swizzle > 0):
                _NUM_XCDS_S2 = 8
                _c1_sw = arith.constant(1, index=True)
                _c_tn_sw = arith.constant(tile_n, index=True)
                _gx = (n_in + _c_tn_sw - _c1_sw) / _c_tn_sw
                _gy = size_expert_ids_in

                _linear_id = bx * _gx + by
                _num_wgs = _gx * _gy

                _c_xcds = arith.constant(_NUM_XCDS_S2, index=True)
                _wgs_per_xcd = _num_wgs / _c_xcds
                _wgid = (_linear_id % _c_xcds) * _wgs_per_xcd + (_linear_id / _c_xcds)

                _c_wgm = arith.constant(xcd_swizzle, index=True)
                _num_wgid_in_group = _c_wgm * _gx
                _group_id = _wgid / _num_wgid_in_group
                _first_pid_m = _group_id * _c_wgm
                _remaining_m = _gy - _first_pid_m
                _cmp_m = arith.cmpi(arith.CmpIPredicate.ult, _remaining_m, _c_wgm)
                _group_size_m = arith.select(_cmp_m, _remaining_m, _c_wgm)

                _wgid_in_group = _wgid % _num_wgid_in_group
                bx = _first_pid_m + (_wgid_in_group % _group_size_m)
                by = _wgid_in_group / _group_size_m

            # XOR16 swizzle parameter (in bytes; constant, power-of-two in our configs).
            k_blocks16 = arith.index(tile_k_bytes // 16)
            layout_tx_wave_lane = fx.make_layout(
                (num_waves_static, 64), stride=(64, 1)
            )
            layout_lane16 = fx.make_layout((4, 16), stride=(16, 1))
            fx.make_layout((tile_m, tile_k), stride=(tile_k, 1))

            base_ptr = allocator.get_base()
            lds_x_ptr = SmemPtr(
                base_ptr,
                lds_alloc_offset,
                (
                    T.bf16
                    if is_bf16
                    else (T.f16 if is_f16 else (T.i8 if is_int8 else T.f8))
                ),
                shape=(lds_total_elems,),
            )
            lds_x = lds_x_ptr.get()
            # Alias the same underlying LDS bytes as f16/bf16 for epilogue shuffle.
            lds_out = (
                SmemPtr(
                    base_ptr,
                    lds_x_ptr.byte_offset,
                    (T.bf16 if out_is_bf16 else T.f16),
                    shape=(tile_m * tile_n,),
                ).get()
                if _use_cshuffle_epilog
                else None
            )

            # Buffer resources.
            # For dynamic memrefs, `max_size=False` cannot infer the logical size from the memref *type*,
            # so we should pass `num_records_bytes` explicitly for stable hardware OOB behavior.
            c_topk = fx.Index(topk)

            # X(A2): [tokens*topk, inter_dim] bytes = tokens*topk*k*elem_bytes
            x_nbytes_idx = (tokens_in * c_topk) * k_in * arith.index(int(elem_bytes))
            x_rsrc = buffer_ops.create_buffer_resource(
                arg_x, max_size=False, num_records_bytes=x_nbytes_idx
            )

            w_rsrc = buffer_ops.create_buffer_resource(arg_w, max_size=False)

            # OUT: [tokens, model_dim] -> clamp to descriptor max (i32 bytes) to avoid overflow on huge tokens.
            out_elem_bytes = 4 if out_is_f32 else 2
            out_nbytes_idx = tokens_in * n_in * fx.Index(out_elem_bytes)
            if const_expr(not bool(accumulate)):
                out_nbytes_idx = (
                    tokens_in * fx.Index(topk) * n_in * fx.Index(out_elem_bytes)
                )
            out_rsrc = buffer_ops.create_buffer_resource(
                arg_out, max_size=False, num_records_bytes=out_nbytes_idx
            )
            # scale_x: fp16/bf16 path ignores (implicit scale=1.0); int4_bf16 also uses 1.0.
            if const_expr(is_f16_or_bf16):
                sx_rsrc = None
            else:
                # scale_x (A2 scale): [tokens*topk] f32 -> bytes = tokens*topk*4
                sx_nbytes_idx = (tokens_in * c_topk) * fx.Index(4)
                sx_rsrc = buffer_ops.create_buffer_resource(
                    arg_scale_x, max_size=False, num_records_bytes=sx_nbytes_idx
                )
            # scale_w: fp16/bf16 (non-int4) path ignores; int4_bf16 needs dequant scale.
            if const_expr(not needs_scale_w):
                sw_rsrc = None
            else:
                # scale_w: [experts*model_dim] f32 (static shape in practice)
                sw_rsrc = buffer_ops.create_buffer_resource(arg_scale_w, max_size=False)

            # sorted_token_ids / sorted_weights: [blocks*tile_m] (CK-style padded length)
            sorted_nbytes_idx = size_expert_ids_in * fx.Index(tile_m) * fx.Index(4)
            sorted_rsrc = buffer_ops.create_buffer_resource(
                arg_sorted_token_ids,
                max_size=False,
                num_records_bytes=sorted_nbytes_idx,
            )
            sorted_w_rsrc = buffer_ops.create_buffer_resource(
                arg_sorted_weights, max_size=False, num_records_bytes=sorted_nbytes_idx
            )

            # expert ids: [blocks] i32 -> bytes = size_expert_ids_in*4
            eid_nbytes_idx = size_expert_ids_in * fx.Index(4)
            expert_rsrc = buffer_ops.create_buffer_resource(
                arg_expert_ids, max_size=False, num_records_bytes=eid_nbytes_idx
            )
            bx_m = bx * fx.Index(tile_m)

            # Early-exit guard (as in 2ce65fb): some routing paths can produce extra/garbage
            # expert blocks beyond `num_valid_ids`. Skip those blocks entirely to avoid OOB.
            numids_rsrc = buffer_ops.create_buffer_resource(
                arg_num_valid_ids,
                max_size=False,
                num_records_bytes=fx.Index(4),
            )
            num_valid_i32 = buffer_ops.buffer_load(
                numids_rsrc, fx.Index(0), vec_width=1, dtype=T.i32
            )
            bx_m_i32 = arith.index_cast(T.i32, bx_m)
            blk_valid = arith.cmpi(arith.CmpIPredicate.ult, bx_m_i32, num_valid_i32)

            def _moe_gemm2_then_body():
                # Expert id for this M tile.
                expert_i32 = buffer_ops.buffer_load(
                    expert_rsrc, bx, vec_width=1, dtype=T.i32
                )
                expert_idx = arith.index_cast(T.index, expert_i32)
                n_idx = fx.Index(model_dim)
                expert_off_idx = expert_idx * n_idx  # index

                # ---- X gmem->reg prefetch (match preshuffle GEMM mapping) ----
                # Prefer 16B buffer-load (dwordx4). If the per-thread byte count isn't divisible by
                # 16, fall back to 8B (dwordx2) or 4B (dword) loads. For fp16/bf16 we require 16B.
                if const_expr(is_f16_or_bf16):
                    if const_expr(bytes_per_thread_x % 16 == 0):
                        x_load_bytes = 16
                    elif const_expr(bytes_per_thread_x % 8 == 0):
                        x_load_bytes = 8
                    elif const_expr(bytes_per_thread_x % 4 == 0):
                        x_load_bytes = 4
                    else:
                        raise ValueError(
                            f"[fp16] bytes_per_thread_x ({bytes_per_thread_x}) must be divisible by 4"
                        )
                else:
                    if const_expr(bytes_per_thread_x % 16 == 0):
                        x_load_bytes = 16
                    elif const_expr(bytes_per_thread_x % 8 == 0):
                        x_load_bytes = 8
                    elif const_expr(bytes_per_thread_x % 4 == 0):
                        x_load_bytes = 4
                    else:
                        raise ValueError(
                            f"bytes_per_thread_x ({bytes_per_thread_x}) must be divisible by 4 to use the dword-indexed load mapping."
                        )
                num_x_loads = bytes_per_thread_x // x_load_bytes
                chunk_i32 = x_load_bytes // 4  # dwords per chunk (1/2/4)
                use_a_direct = _use_a_direct_req and x_load_bytes == 16

                c_k_div4 = (k_in * arith.index(int(elem_bytes))) // fx.Index(4)
                c_k_div4_i32 = arith.index_cast(T.i32, c_k_div4)
                fx.make_layout((m_i32_v, c_k_div4_i32), stride=(c_k_div4_i32, 1))
                tile_k_dwords = (int(tile_k) * int(elem_bytes)) // 4
                layout_x_tile_div4 = fx.make_layout(
                    (tile_m, tile_k_dwords), stride=(tile_k_dwords, 1)
                )
                c_chunk_i32 = fx.Index(chunk_i32)
                tx_i32_base = tx * c_chunk_i32

                topk_i32 = fx.Int32(topk)
                mask24 = fx.Int32(0xFFFFFF)
                # Sentinel clamp uses `tokens` as the upper bound: t_valid = (t < tokens).
                tokens_i32 = arith.index_cast(T.i32, tokens_in)

                def x_tile_chunk_coord_i32(i: int):
                    return tile_chunk_coord_i32(
                        arith,
                        tx_i32_base=tx_i32_base,
                        i=i,
                        total_threads=total_threads,
                        layout_tile_div4=layout_x_tile_div4,
                        chunk_i32=chunk_i32,
                    )

                vec4_x = T.vec(vec4_elems, x_elem)

                def load_x(idx_i32):
                    if const_expr(x_load_bytes == 16):
                        idx_elem = (
                            idx_i32 if elem_bytes == 1 else (idx_i32 * fx.Index(2))
                        )
                        return buffer_copy_gmem16_dwordx4(
                            buffer_ops,
                            vector,
                            elem_type=x_elem,
                            idx_i32=idx_elem,
                            rsrc=x_rsrc,
                            vec_elems=vec16_elems,
                            elem_bytes=elem_bytes,
                        )
                    if const_expr(x_load_bytes == 8):
                        return buffer_ops.buffer_load(
                            x_rsrc, idx_i32, vec_width=2, dtype=T.i32
                        )
                    return buffer_ops.buffer_load(
                        x_rsrc, idx_i32, vec_width=1, dtype=T.i32
                    )

                # decode routed token once (per thread's M-slice) and build a base offset.
                x_row_base_div4 = []
                x_col_local_i32 = []
                x_row_local = []
                for i in range_constexpr(num_x_loads):
                    row_local, col_local_i32 = x_tile_chunk_coord_i32(i)
                    x_row_local.append(row_local)
                    x_col_local_i32.append(col_local_i32)

                    sorted_row_i = bx_m + row_local
                    fused_i = buffer_ops.buffer_load(
                        sorted_rsrc, sorted_row_i, vec_width=1, dtype=T.i32
                    )
                    t_i32 = fused_i & mask24
                    s_i32 = fused_i >> 24
                    # aiter moe_sorting uses sentinel token_id == tokens for padding.
                    # Do NOT rely on buffer OOB semantics for A2/scale loads; explicitly mask.
                    t_valid = arith.cmpi(arith.CmpIPredicate.ult, t_i32, tokens_i32)
                    s_valid = arith.cmpi(arith.CmpIPredicate.ult, s_i32, topk_i32)
                    ts_valid = t_valid & s_valid
                    t_safe = ts_valid.select(t_i32, fx.Int32(0))
                    s_safe = ts_valid.select(s_i32, fx.Int32(0))
                    row_ts_i32 = t_safe * topk_i32 + s_safe
                    row_ts_idx = arith.index_cast(T.index, row_ts_i32)
                    # Base row offset in dword units: row_ts_idx * (k_in/4)
                    x_row_base_div4.append(row_ts_idx * c_k_div4)

                def load_x_tile(base_k):
                    base_k_div4 = (base_k * arith.index(int(elem_bytes))) // fx.Index(4)
                    parts = []
                    for i in range_constexpr(num_x_loads):
                        idx_i32 = x_row_base_div4[i] + base_k_div4 + x_col_local_i32[i]
                        x_vec = load_x(idx_i32)
                        if const_expr(x_load_bytes == 16):
                            parts.append(vector.bitcast(T.i32x4, x_vec))
                        elif const_expr(x_load_bytes == 8):
                            parts.append(vector.bitcast(T.vec(2, T.i32), x_vec))
                        else:
                            parts.append(vector.from_elements(T.vec(1, T.i32), [x_vec]))
                    return parts

                # tx -> wave/lane (GEMM-style decomposition).
                coord_wl = fx.idx2crd(tx, layout_tx_wave_lane)
                wave_id = fx.get(coord_wl, 0)
                lane_id = fx.get(coord_wl, 1)
                coord_l16 = fx.idx2crd(lane_id, layout_lane16)
                lane_div_16 = fx.get(coord_l16, 0)
                lane_mod_16 = fx.get(coord_l16, 1)

                row_a_lds = lane_mod_16
                # A-side kpack is always 16 bytes; kpack_bytes is B-side (may be 8 for int4).
                a_kpack_elems = 16 // elem_bytes
                col_offset_base = lane_div_16 * arith.index(int(a_kpack_elems))
                col_offset_base_bytes = (
                    col_offset_base
                    if elem_bytes == 1
                    else (col_offset_base * arith.index(int(elem_bytes)))
                )
                a_direct_row_base_div4 = []
                if const_expr(use_a_direct):
                    for mi in range_constexpr(tile_m // 16):
                        row_local_a = row_a_lds + arith.index(mi * 16)
                        sorted_row_a = bx_m + row_local_a
                        fused_a = buffer_ops.buffer_load(
                            sorted_rsrc, sorted_row_a, vec_width=1, dtype=T.i32
                        )
                        t_a = fused_a & mask24
                        s_a = fused_a >> 24
                        t_ok_a = arith.cmpi(arith.CmpIPredicate.ult, t_a, tokens_i32)
                        s_ok_a = arith.cmpi(arith.CmpIPredicate.ult, s_a, topk_i32)
                        ts_ok_a = t_ok_a & s_ok_a
                        t_safe_a = ts_ok_a.select(t_a, fx.Int32(0))
                        s_safe_a = ts_ok_a.select(s_a, fx.Int32(0))
                        row_ts_a = t_safe_a * topk_i32 + s_safe_a
                        row_ts_a_idx = arith.index_cast(T.index, row_ts_a)
                        a_direct_row_base_div4.append(row_ts_a_idx * c_k_div4)

                # Dynamic N tiling within block.
                by_n = by * fx.Index(tile_n)
                num_waves = num_waves_static
                n_per_wave = tile_n // num_waves
                num_acc_n = n_per_wave // 16
                c_n_per_wave = fx.Index(n_per_wave)
                wave_mod_n = wave_id % fx.Index(num_waves_static)
                n_tile_base = wave_mod_n * c_n_per_wave

                # Precompute (n_blk, n_intra) for B, and col indices for output.
                n_intra_list = []
                n_blk_list = []
                col_g_list = []
                c_n_total // fx.Index(16)
                c_n0_static = experts * model_dim // 16
                layout_n_blk_intra = fx.make_layout((c_n0_static, 16), stride=(16, 1))
                for ni in range_constexpr(num_acc_n):
                    offset = arith.index(ni * 16)
                    col_g = by_n + n_tile_base + offset + lane_mod_16
                    col_g_list.append(col_g)

                    row_w = expert_off_idx + col_g
                    coord_w = fx.idx2crd(row_w, layout_n_blk_intra)
                    n_blk_list.append(fx.get(coord_w, 0))
                    n_intra_list.append(fx.get(coord_w, 1))

                m_repeat = tile_m // 16
                k_unroll = tile_k_bytes // 64  # K64-byte micro-step (2x MFMA)
                upper_m16_valid = None
                if const_expr(_row16_skip_req):
                    fused_upper = buffer_ops.buffer_load(
                        sorted_rsrc, bx_m + arith.index(16), vec_width=1, dtype=T.i32
                    )
                    t_upper = fused_upper & mask24
                    s_upper = fused_upper >> 24
                    t_upper_ok = arith.cmpi(
                        arith.CmpIPredicate.ult, t_upper, tokens_i32
                    )
                    s_upper_ok = arith.cmpi(
                        arith.CmpIPredicate.ult, s_upper, topk_i32
                    )
                    upper_m16_valid = t_upper_ok & s_upper_ok
                upper_m32_valid = None
                if const_expr(_row32_skip_req):
                    fused_upper32 = buffer_ops.buffer_load(
                        sorted_rsrc, bx_m + arith.index(32), vec_width=1, dtype=T.i32
                    )
                    t_upper32 = fused_upper32 & mask24
                    s_upper32 = fused_upper32 >> 24
                    t_upper32_ok = arith.cmpi(
                        arith.CmpIPredicate.ult, t_upper32, tokens_i32
                    )
                    s_upper32_ok = arith.cmpi(
                        arith.CmpIPredicate.ult, s_upper32, topk_i32
                    )
                    upper_m32_valid = t_upper32_ok & s_upper32_ok

                # --- B Load Logic (K64) ---
                _b_stride_nlane = kpack_bytes // int(w_elem_bytes)
                _b_stride_klane = 16 * _b_stride_nlane
                _b_stride_k0 = 4 * _b_stride_klane
                _b_stride_n0 = (
                    (int(inter_dim) * int(w_elem_bytes)) // 64
                ) * _b_stride_k0

                def load_b_pack(base_k, ki_step, ni):
                    return load_b_pack_k32(
                        buffer_ops,
                        arith,
                        vector,
                        arg_b=arg_w,
                        b_rsrc=w_rsrc,
                        layout_b=layout_b,
                        base_k=base_k,
                        ki_step=ki_step,
                        n_blk=n_blk_list[ni],
                        n_intra=n_intra_list[ni],
                        lane_div_16=lane_div_16,  # 0..3
                        elem_type=w_elem,
                        kpack_bytes=kpack_bytes,
                        elem_bytes=w_elem_bytes,
                        unpack_int4=is_int4,
                        cache_modifier=b_nt,
                    )

                def load_b_tile(base_k):
                    """Prefetch the entire per-thread B tile (gmem -> regs) for a given K base.

                    Returns a list of length `k_unroll`, where each entry is a tuple:
                      (packs_half0[ni], packs_half1[ni])  for the K64 micro-step.
                    For groupwise variants, each entry also includes per-group scales:
                      (packs0[ni], packs1[ni], scales0[ni], scales1[ni])
                    """
                    if const_expr(is_int4_bf16_groupwise):
                        # W4A16 groupwise: load raw packed32 + scale; defer dequant to compute_tile.
                        raw_data = []
                        for ku in range_constexpr(k_unroll):
                            raw_ku = []
                            for ni in range_constexpr(num_acc_n):
                                packed32, scale_val = load_b_raw_w4a16_groupwise(
                                    buffer_ops,
                                    arith,
                                    vector,
                                    arg_b=arg_w,
                                    b_rsrc=w_rsrc,
                                    layout_b=layout_b,
                                    base_k=base_k,
                                    ku=ku,
                                    n_blk=n_blk_list[ni],
                                    n_intra=n_intra_list[ni],
                                    lane_div_16=lane_div_16,
                                    elem_type=w_elem,
                                    scale_rsrc=sw_rsrc,
                                    expert_offset=expert_off_idx,
                                    num_groups=num_groups,
                                    group_size=group_size,
                                    n_per_expert=model_dim,
                                    kpack_bytes=kpack_bytes,
                                    scale_dtype=scale_dtype,
                                )
                                raw_ku.append((packed32, scale_val))
                            raw_data.append(raw_ku)
                        return raw_data
                    elif const_expr(is_int4_bf16):
                        # W4A16 per-row: load raw packed32; defer dequant to compute_tile.
                        raw_data = []
                        for ku in range_constexpr(k_unroll):
                            raw_ku = []
                            for ni in range_constexpr(num_acc_n):
                                raw = load_b_raw_w4a16(
                                    buffer_ops,
                                    arith,
                                    vector,
                                    arg_b=arg_w,
                                    b_rsrc=w_rsrc,
                                    layout_b=layout_b,
                                    base_k=base_k,
                                    ku=ku,
                                    n_blk=n_blk_list[ni],
                                    n_intra=n_intra_list[ni],
                                    lane_div_16=lane_div_16,
                                    elem_type=w_elem,
                                    kpack_bytes=kpack_bytes,
                                )
                                raw_ku.append(raw)
                            raw_data.append(raw_ku)
                        return raw_data
                    else:
                        # fp8/int8/bf16/fp16: load one full K64 B pack and split it
                        # into the two i64 MFMA operands. The old K32 path loaded the
                        # same 16B pack twice for dense bf16/fp16.
                        b_tile = []
                        for ku in range_constexpr(k_unroll):
                            packs0 = []
                            packs1 = []
                            for ni in range_constexpr(num_acc_n):
                                if const_expr(is_int4 or not _dense_b_k64_single_load):
                                    ki0 = (ku * 2) + 0
                                    ki1 = (ku * 2) + 1
                                    b0 = load_b_pack(base_k, ki0, ni)
                                    b1 = load_b_pack(base_k, ki1, ni)
                                elif const_expr(_dense_b_k64_layout_load):
                                    b0, b1 = load_b_packs_k64(
                                        buffer_ops,
                                        arith,
                                        vector,
                                        b_rsrc=w_rsrc,
                                        layout_b=layout_b,
                                        base_k=base_k,
                                        ku=ku,
                                        n_blk=n_blk_list[ni],
                                        n_intra=n_intra_list[ni],
                                        lane_div_16=lane_div_16,
                                        elem_type=w_elem,
                                        kpack_bytes=kpack_bytes,
                                        elem_bytes=w_elem_bytes,
                                        cache_modifier=b_nt,
                                    )
                                else:
                                    b0, b1 = load_b_packs_k64_strided(
                                        buffer_ops,
                                        arith,
                                        vector,
                                        b_rsrc=w_rsrc,
                                        base_k=base_k,
                                        ku=ku,
                                        n_blk=n_blk_list[ni],
                                        n_intra=n_intra_list[ni],
                                        lane_div_16=lane_div_16,
                                        elem_type=w_elem,
                                        stride_n0=_b_stride_n0,
                                        stride_k0=_b_stride_k0,
                                        stride_klane=_b_stride_klane,
                                        stride_nlane=_b_stride_nlane,
                                        kpack_bytes=kpack_bytes,
                                        elem_bytes=w_elem_bytes,
                                        cache_modifier=b_nt,
                                    )
                                packs0.append(b0)
                                packs1.append(b1)
                            b_tile.append((packs0, packs1))
                        return b_tile

                # ---- Pipeline helpers: store X tile to LDS with ping-pong base ----
                def store_x_tile_to_lds(vec_x_in_parts, lds_base):
                    for i in range_constexpr(num_x_loads):
                        row_local = x_row_local[i]
                        col_local_i32 = x_col_local_i32[i]
                        if const_expr(x_load_bytes == 16):
                            lds_store_16b_xor16(
                                arith,
                                vector,
                                lds_memref=lds_x,
                                vec16_ty=vec16_x,
                                layout_lds=layout_lds,
                                row_local=row_local,
                                col_local_i32=col_local_i32,
                                tx_c4=fx.Index(4),
                                k_blocks16=k_blocks16,
                                lds_base=lds_base,
                                vec_part_i32x4=vec_x_in_parts[i],
                                elem_bytes=elem_bytes,
                            )
                        elif const_expr(x_load_bytes == 8):
                            lds_store_8b_xor16(
                                arith,
                                vector,
                                lds_memref=lds_x,
                                vec8_ty=vec8_x,
                                layout_lds=layout_lds,
                                row_local=row_local,
                                col_local_i32=col_local_i32,
                                tx_c4=fx.Index(4),
                                k_blocks16=k_blocks16,
                                lds_base=lds_base,
                                vec_part_i32x2=vec_x_in_parts[i],
                                elem_bytes=elem_bytes,
                            )
                        else:
                            lds_store_4b_xor16(
                                arith,
                                vector,
                                lds_memref=lds_x,
                                vec4_ty=vec4_x,
                                layout_lds=layout_lds,
                                row_local=row_local,
                                col_local_i32=col_local_i32,
                                tx_c4=fx.Index(4),
                                k_blocks16=k_blocks16,
                                lds_base=lds_base,
                                vec_part_i32x1=vec_x_in_parts[i],
                                elem_bytes=elem_bytes,
                            )

                # --- A LDS load helper for K64 (load 16B once, extract 2x i64 halves) ---
                def lds_load_packs_k64(curr_row_a_lds, col_base_bytes, lds_base):
                    col_base_swz_bytes = swizzle_xor16(
                        curr_row_a_lds, col_base_bytes, k_blocks16
                    )
                    col_base_swz = (
                        col_base_swz_bytes
                        if elem_bytes == 1
                        else (col_base_swz_bytes // arith.index(int(elem_bytes)))
                    )
                    idx_a16 = crd2idx((curr_row_a_lds, col_base_swz), layout_lds)
                    idx_a16 = idx_a16 + lds_base
                    loaded_a16 = vector.load_op(vec16_x, lds_x, [idx_a16])
                    a_i64x2 = vector.bitcast(T.i64x2, loaded_a16)
                    a0 = vector.extract(
                        a_i64x2, static_position=[0], dynamic_position=[]
                    )
                    a1 = vector.extract(
                        a_i64x2, static_position=[1], dynamic_position=[]
                    )
                    return a0, a1

                def gmem_load_a_packs_k64(base_k, col_base_bytes, mi: int):
                    base_k_div4 = (
                        base_k * arith.index(int(elem_bytes))
                    ) // fx.Index(4)
                    col_base_div4 = col_base_bytes // fx.Index(4)
                    idx_i32 = (
                        a_direct_row_base_div4[mi] + base_k_div4 + col_base_div4
                    )
                    loaded_a16 = load_x(idx_i32)
                    a_i64x2 = vector.bitcast(T.i64x2, loaded_a16)
                    a0 = vector.extract(
                        a_i64x2, static_position=[0], dynamic_position=[]
                    )
                    a1 = vector.extract(
                        a_i64x2, static_position=[1], dynamic_position=[]
                    )
                    return a0, a1

                def compute_tile(
                    acc_in,
                    b_tile_in,
                    lds_base,
                    *,
                    base_k_for_a=None,
                    base_k_for_b=None,
                    prefetch_epilogue: bool = False,
                    a0_prefetch=None,
                ):
                    acc_list = list(acc_in)
                    mfma_res_ty = T.i32x4 if is_int8 else T.f32x4
                    if const_expr(_use_mfma_k32):
                        mfma_fn = (
                            rocdl.mfma_f32_16x16x32_f16
                            if is_f16
                            else rocdl.mfma_f32_16x16x32_bf16
                        )
                    else:
                        mfma_fn = (
                            mfma_i32_k32
                            if is_int8
                            else (
                                mfma_f32_bf16_k16
                                if is_bf16
                                else (
                                    rocdl.mfma_f32_16x16x16f16
                                    if is_f16
                                    else rocdl.mfma_f32_16x16x32_fp8_fp8
                                )
                            )
                        )

                    epilogue_pf = None
                    if const_expr(prefetch_epilogue and not use_groupwise_scale):
                        expert_off_pf = expert_off_idx
                        sw_pf = []
                        for ni in range_constexpr(num_acc_n):
                            col_g = col_g_list[ni]
                            row_w_idx = expert_off_pf + col_g
                            sw_pf.append(
                                fx.Float32(1.0)
                                if not needs_scale_w
                                else buffer_ops.buffer_load(
                                    sw_rsrc, row_w_idx, vec_width=1, dtype=T.f32
                                )
                            )
                        # Also prefetch per-row routed/topk weights (sorted_weights) when enabled.
                        tw_pf = None
                        if const_expr(doweight_stage2):
                            tw_pf = []
                            lane_div_16_mul4_pf = lane_div_16 * fx.Index(4)
                            ii_idx_list_pf = [fx.Index(ii) for ii in range(4)]
                            for mi in range_constexpr(m_repeat):
                                mi_base_pf = arith.index(mi * 16)
                                for ii in range_constexpr(4):
                                    row_off_pf = (
                                        lane_div_16_mul4_pf + ii_idx_list_pf[ii]
                                    )
                                    row_in_tile_pf = mi_base_pf + row_off_pf
                                    sorted_row_pf = bx_m + row_in_tile_pf
                                    tw_pf.append(
                                        buffer_ops.buffer_load(
                                            sorted_w_rsrc,
                                            sorted_row_pf,
                                            vec_width=1,
                                            dtype=T.f32,
                                        )
                                    )
                        epilogue_pf = (sw_pf, tw_pf)

                    def _i64_to_v4f16(x_i64):
                        v1 = vector.from_elements(T.vec(1, T.i64), [x_i64])
                        return vector.bitcast(T.f16x4, v1)

                    def _i64_to_v4i16(x_i64):
                        v1 = vector.from_elements(T.vec(1, T.i64), [x_i64])
                        return vector.bitcast(T.i16x4, v1)

                    def _i64x2_to_v8f16(lo, hi):
                        v2 = vector.from_elements(T.i64x2, [lo, hi])
                        return vector.bitcast(T.f16x8, v2)

                    def _i64x2_to_v8bf16(lo, hi):
                        v2 = vector.from_elements(T.i64x2, [lo, hi])
                        return vector.bitcast(T.bf16x8, v2)

                    def mfma_k64(acc0, a0, a1, b0, b1):
                        if const_expr(_use_mfma_k32):
                            # gfx950: single 16x16x32 MFMA consuming all 128 bits (K=32 f16/bf16)
                            if const_expr(is_f16):
                                av = _i64x2_to_v8f16(a0, a1)
                                bv = _i64x2_to_v8f16(b0, b1)
                            else:
                                av = _i64x2_to_v8bf16(a0, a1)
                                bv = _i64x2_to_v8bf16(b0, b1)
                            return mfma_fn(mfma_res_ty, [av, bv, acc0, 0, 0, 0])
                        if const_expr(is_f16):
                            a0v = _i64_to_v4f16(a0)
                            a1v = _i64_to_v4f16(a1)
                            b0v = _i64_to_v4f16(b0)
                            b1v = _i64_to_v4f16(b1)
                            acc1 = mfma_fn(mfma_res_ty, [a0v, b0v, acc0, 0, 0, 0])
                            return mfma_fn(mfma_res_ty, [a1v, b1v, acc1, 0, 0, 0])
                        if const_expr(is_bf16):
                            a0v = _i64_to_v4i16(a0)
                            a1v = _i64_to_v4i16(a1)
                            b0v = _i64_to_v4i16(b0)
                            b1v = _i64_to_v4i16(b1)
                            acc1 = mfma_fn(mfma_res_ty, [a0v, b0v, acc0, 0, 0, 0])
                            return mfma_fn(mfma_res_ty, [a1v, b1v, acc1, 0, 0, 0])
                        acc1 = mfma_fn(mfma_res_ty, [a0, b0, acc0, 0, 0, 0])
                        return mfma_fn(mfma_res_ty, [a1, b1, acc1, 0, 0, 0])

                    def _acc_scaled_f32(f32_acc_vec, f32_partial_vec, scale_val):
                        """MFMA f32 partial -> scale -> add to f32 accumulator via math.fma on vector."""
                        from flydsl._mlir.dialects._math_ops_gen import fma as _math_fma

                        _uw = arith._to_raw
                        scale_vec = _uw(vector.broadcast(T.f32x4, scale_val))
                        return arith.ArithValue(
                            _math_fma(scale_vec, _uw(f32_partial_vec), _uw(f32_acc_vec))
                        )

                    if const_expr(is_int4_bf16 or is_int4_bf16_groupwise):
                        # W4A16: deferred dequant -- unpack int4->bf16 right before MFMA
                        # to minimize VGPR lifetime of dequantized bf16 values.
                        _pending_acc = None
                        for ku in range_constexpr(k_unroll):
                            b_raw = b_tile_in[ku]
                            ki64 = arith.index(ku * 64)
                            col_base = col_offset_base_bytes + ki64

                            for mi in range_constexpr(m_repeat):
                                mi_val = arith.index(mi * 16)
                                curr_row_a_lds = row_a_lds + mi_val

                                if const_expr(
                                    use_a_direct
                                ):
                                    a0, a1 = gmem_load_a_packs_k64(
                                        base_k_for_a, col_base, mi
                                    )
                                elif const_expr(
                                    (a0_prefetch is not None)
                                    and (ku == 0)
                                    and (mi == 0)
                                ):
                                    a0, a1 = a0_prefetch
                                else:
                                    a0, a1 = lds_load_packs_k64(
                                        curr_row_a_lds, col_base, lds_base
                                    )

                                for ni in range_constexpr(num_acc_n):
                                    acc_idx = mi * num_acc_n + ni
                                    if const_expr(is_int4_bf16_groupwise):
                                        packed, sc = b_raw[ni]
                                        if const_expr(_scale_is_bf16):
                                            sc = extract_bf16_scale(arith, sc, ku)
                                    else:
                                        packed, sc = b_raw[ni], None
                                    if const_expr(
                                        is_int4_bf16_groupwise and use_gfx950_cvt
                                    ):
                                        b0, b1 = unpack_b_w4a16(
                                            packed,
                                            arith,
                                            vector,
                                            scale_val=None,
                                            use_gfx950_cvt=True,
                                            defer_scale16=True,
                                        )
                                        tmp = mfma_k64(zero_f32_acc, a0, a1, b0, b1)
                                        if _pending_acc is not None:
                                            p_idx, p_tmp, p_sc = _pending_acc
                                            acc_list[p_idx] = _acc_scaled_f32(
                                                acc_list[p_idx], p_tmp, p_sc
                                            )
                                        _pending_acc = (acc_idx, tmp, sc)
                                    else:
                                        b0, b1 = unpack_b_w4a16(
                                            packed,
                                            arith,
                                            vector,
                                            scale_val=sc,
                                            use_gfx950_cvt=use_gfx950_cvt,
                                            defer_scale16=use_gfx950_cvt,
                                        )
                                        acc_list[acc_idx] = mfma_k64(
                                            acc_list[acc_idx], a0, a1, b0, b1
                                        )
                        # Drain last pending FMA.
                        if _pending_acc is not None:
                            p_idx, p_tmp, p_sc = _pending_acc
                            acc_list[p_idx] = _acc_scaled_f32(
                                acc_list[p_idx], p_tmp, p_sc
                            )
                    else:
                        for ku in range_constexpr(k_unroll):
                            if const_expr(_stream_b_req):
                                b_packs0 = []
                                b_packs1 = []
                                for ni in range_constexpr(num_acc_n):
                                    if const_expr(_dense_b_k64_layout_load):
                                        b0, b1 = load_b_packs_k64(
                                            buffer_ops,
                                            arith,
                                            vector,
                                            b_rsrc=w_rsrc,
                                            layout_b=layout_b,
                                            base_k=base_k_for_b,
                                            ku=ku,
                                            n_blk=n_blk_list[ni],
                                            n_intra=n_intra_list[ni],
                                            lane_div_16=lane_div_16,
                                            elem_type=w_elem,
                                            kpack_bytes=kpack_bytes,
                                            elem_bytes=w_elem_bytes,
                                            cache_modifier=b_nt,
                                        )
                                    elif const_expr(_dense_b_k64_single_load):
                                        b0, b1 = load_b_packs_k64_strided(
                                            buffer_ops,
                                            arith,
                                            vector,
                                            b_rsrc=w_rsrc,
                                            base_k=base_k_for_b,
                                            ku=ku,
                                            n_blk=n_blk_list[ni],
                                            n_intra=n_intra_list[ni],
                                            lane_div_16=lane_div_16,
                                            elem_type=w_elem,
                                            stride_n0=_b_stride_n0,
                                            stride_k0=_b_stride_k0,
                                            stride_klane=_b_stride_klane,
                                            stride_nlane=_b_stride_nlane,
                                            kpack_bytes=kpack_bytes,
                                            elem_bytes=w_elem_bytes,
                                            cache_modifier=b_nt,
                                        )
                                    else:
                                        ki0 = (ku * 2) + 0
                                        ki1 = (ku * 2) + 1
                                        b0 = load_b_pack(base_k_for_b, ki0, ni)
                                        b1 = load_b_pack(base_k_for_b, ki1, ni)
                                    b_packs0.append(b0)
                                    b_packs1.append(b1)
                            else:
                                b_packs0, b_packs1 = b_tile_in[ku]
                            ki64 = arith.index(ku * 64)
                            col_base = col_offset_base_bytes + ki64

                            for mi in range_constexpr(m_repeat):
                                def _compute_one_mi():
                                    mi_val = arith.index(mi * 16)
                                    curr_row_a_lds = row_a_lds + mi_val

                                    if const_expr(use_a_direct):
                                        a0, a1 = gmem_load_a_packs_k64(
                                            base_k_for_a, col_base, mi
                                        )
                                    elif const_expr(
                                        (a0_prefetch is not None)
                                        and (ku == 0)
                                        and (mi == 0)
                                    ):
                                        a0, a1 = a0_prefetch
                                    else:
                                        a0, a1 = lds_load_packs_k64(
                                            curr_row_a_lds, col_base, lds_base
                                        )

                                    mi_results = []
                                    for ni in range_constexpr(num_acc_n):
                                        acc_idx = mi * num_acc_n + ni
                                        mi_results.append(
                                            mfma_k64(
                                                acc_list[acc_idx],
                                                a0,
                                                a1,
                                                b_packs0[ni],
                                                b_packs1[ni],
                                            )
                                        )
                                    return mi_results

                                if const_expr(_row32_skip_outer_req and mi == 2):
                                    pass
                                elif const_expr(_row16_skip_req and mi == 1):
                                    _if_mi = scf.IfOp(
                                        upper_m16_valid,
                                        [mfma_res_ty] * num_acc_n,
                                        has_else=True,
                                    )
                                    with ir.InsertionPoint(_if_mi.then_block):
                                        scf.YieldOp(_compute_one_mi())
                                    with ir.InsertionPoint(_if_mi.else_block):
                                        scf.YieldOp(
                                            [
                                                acc_list[mi * num_acc_n + ni]
                                                for ni in range_constexpr(num_acc_n)
                                            ]
                                        )
                                    for ni in range_constexpr(num_acc_n):
                                        acc_list[mi * num_acc_n + ni] = _if_mi.results[
                                            ni
                                        ]
                                elif const_expr(
                                    _row32_skip_req
                                    and not _row32_skip_outer_req
                                    and mi == 2
                                ):
                                    _if_mi = scf.IfOp(
                                        upper_m32_valid,
                                        [mfma_res_ty] * num_acc_n,
                                        has_else=True,
                                    )
                                    with ir.InsertionPoint(_if_mi.then_block):
                                        scf.YieldOp(_compute_one_mi())
                                    with ir.InsertionPoint(_if_mi.else_block):
                                        scf.YieldOp(
                                            [
                                                acc_list[mi * num_acc_n + ni]
                                                for ni in range_constexpr(num_acc_n)
                                            ]
                                        )
                                    for ni in range_constexpr(num_acc_n):
                                        acc_list[mi * num_acc_n + ni] = _if_mi.results[
                                            ni
                                        ]
                                else:
                                    mi_results = _compute_one_mi()
                                    for ni in range_constexpr(num_acc_n):
                                        acc_list[mi * num_acc_n + ni] = mi_results[ni]
                        if const_expr(_row32_skip_outer_req):
                            _if_mi2 = scf.IfOp(
                                upper_m32_valid,
                                [mfma_res_ty] * num_acc_n,
                                has_else=True,
                            )
                            with ir.InsertionPoint(_if_mi2.then_block):
                                mi2_acc = [
                                    acc_list[2 * num_acc_n + ni]
                                    for ni in range_constexpr(num_acc_n)
                                ]
                                for ku2 in range_constexpr(k_unroll):
                                    b_packs0, b_packs1 = b_tile_in[ku2]
                                    ki64 = arith.index(ku2 * 64)
                                    col_base = col_offset_base_bytes + ki64
                                    curr_row_a_lds = row_a_lds + arith.index(32)

                                    if const_expr(use_a_direct):
                                        a0, a1 = gmem_load_a_packs_k64(
                                            base_k_for_a, col_base, 2
                                        )
                                    else:
                                        a0, a1 = lds_load_packs_k64(
                                            curr_row_a_lds, col_base, lds_base
                                        )

                                    mi2_results = []
                                    for ni in range_constexpr(num_acc_n):
                                        mi2_results.append(
                                            mfma_k64(
                                                mi2_acc[ni],
                                                a0,
                                                a1,
                                                b_packs0[ni],
                                                b_packs1[ni],
                                            )
                                        )
                                    mi2_acc = mi2_results
                                scf.YieldOp(mi2_acc)
                            with ir.InsertionPoint(_if_mi2.else_block):
                                scf.YieldOp(
                                    [
                                        acc_list[2 * num_acc_n + ni]
                                        for ni in range_constexpr(num_acc_n)
                                    ]
                                )
                            for ni in range_constexpr(num_acc_n):
                                acc_list[2 * num_acc_n + ni] = _if_mi2.results[ni]
                    return acc_list, epilogue_pf

                # ---------------- 2-stage pipeline (ping-pong LDS + B tile prefetch) ----------------
                lds_tile_elems = arith.index(tile_m * lds_stride)
                lds_base_cur = fx.Index(0)
                lds_base_nxt = lds_tile_elems

                rocdl.sched_barrier(0)

                # def hot_loop_scheduler():
                #     mfma_group = num_acc_n
                #     # K64 micro-step: 2x K32 MFMA per accumulator update.
                #     mfma_total = (k_unroll * 2) * m_repeat * mfma_group
                #     mfma_per_iter = 2 * mfma_group
                #     sche_iters = 0 if mfma_per_iter == 0 else (mfma_total // mfma_per_iter)
                #     rocdl.sched_dsrd(2)
                #     rocdl.sched_mfma(1)
                #     rocdl.sched_mfma(1)
                #     if num_acc_n < 4:
                #         rocdl.sched_dsrd(1)
                #         rocdl.sched_mfma(1)
                #         rocdl.sched_dsrd(1)
                #         rocdl.sched_mfma(1)
                #         rocdl.sched_vmem(1)
                #         rocdl.sched_mfma(1)
                #         rocdl.sched_vmem(1)
                #         rocdl.sched_mfma(2)
                #         rocdl.sched_dsrd(1)
                #         rocdl.sched_mfma(2)
                #         rocdl.sched_vmem(1)

                #     dswr_tail = num_x_loads
                #     if dswr_tail > sche_iters:
                #         dswr_tail = sche_iters
                #     dswr_start = sche_iters - dswr_tail
                #     for sche_i in range_constexpr(sche_iters):
                #         rocdl.sched_mfma(mfma_group // 2)
                #         rocdl.sched_dsrd(1)
                #         rocdl.sched_mfma(mfma_group // 2)
                #         rocdl.sched_vmem(1)
                #         rocdl.sched_mfma(mfma_group)
                #         if sche_i >= dswr_start - 1:
                #             rocdl.sched_dswr(1)
                #     rocdl.sched_barrier(0)

                def hot_loop_scheduler():
                    if const_expr(not _sched_enabled):
                        rocdl.sched_barrier(0)
                        return
                    # - MFMA group size per "slot": num_acc_n
                    # - Total MFMA per tile: (2*K32 per K64) * k_unroll * m_repeat * num_acc_n
                    # - We emit (mfma_group + dsrd + mfma_group) per scheduler iteration.
                    mfma_group = num_acc_n
                    mfma_total = (k_unroll * 2) * m_repeat * mfma_group
                    mfma_per_iter = 2 * mfma_group
                    sche_iters = (
                        0 if mfma_per_iter == 0 else (mfma_total // mfma_per_iter)
                    )

                    rocdl.sched_dsrd(2)
                    rocdl.sched_mfma(1)
                    if const_expr(tile_m == 16 and _sched_early_vmem_req > 0):
                        rocdl.sched_vmem(_sched_early_vmem_req)
                    rocdl.sched_mfma(1)
                    if const_expr(tile_m == 16 and _sched_early_vmem_req > 0):
                        rocdl.sched_vmem(_sched_early_vmem_req)
                    if const_expr(num_acc_n < 4):
                        rocdl.sched_dsrd(1)
                        rocdl.sched_mfma(1)
                        if const_expr(tile_m == 16 and _sched_early_vmem_req > 0):
                            rocdl.sched_vmem(_sched_early_vmem_req)
                        rocdl.sched_dsrd(1)
                        rocdl.sched_mfma(1)
                        if const_expr(tile_m == 16 and _sched_early_vmem_req > 0):
                            rocdl.sched_vmem(_sched_early_vmem_req)
                        rocdl.sched_mfma(1)

                    # DS-write hints near the end: match total A LDS-store micro-ops per thread.
                    dswr_tail = num_x_loads
                    if const_expr(dswr_tail > sche_iters):
                        dswr_tail = sche_iters
                    dswr_start = max(
                        sche_iters - dswr_tail - _sched_dswr_advance_req, 0
                    )

                    for sche_i in range_constexpr(sche_iters):
                        if const_expr(_sched_vmem_req > 0):
                            rocdl.sched_vmem(_sched_vmem_req)
                        rocdl.sched_mfma(mfma_group)
                        rocdl.sched_dsrd(1)
                        rocdl.sched_mfma(mfma_group)
                        if const_expr(sche_i >= dswr_start - 1):
                            rocdl.sched_dswr(1)

                    rocdl.sched_barrier(0)

                def stage_barrier():
                    # Only LDS writes need to complete before this CTA barrier. B-tile
                    # VMEM prefetches are consumed later by ordinary data dependencies.
                    if const_expr(use_a_direct):
                        return
                    if const_expr(_fast_barrier_req):
                        _inline_barrier(vmcnt=63, lgkmcnt=0)
                    else:
                        gpu.barrier()

                # Prologue.
                k0 = fx.Index(0)
                if const_expr(not use_a_direct):
                    x_regs0 = load_x_tile(k0)
                b_cur = [] if const_expr(_stream_b_req) else load_b_tile(k0)
                if const_expr(not use_a_direct):
                    store_x_tile_to_lds(x_regs0, lds_base_cur)
                stage_barrier()

                acc = [acc_init] * (num_acc_n * m_repeat)
                lds_base_pong = lds_base_cur
                lds_base_ping = lds_base_nxt

                # Cross-tile A0 LDS prefetch (default-on): prefetch the first A-pack (K64) for the
                # tile we are about to compute from LDS, to overlap with upcoming VMEM.
                a0_prefetch_pong = (
                    (fx.Int64(0).ir_value(), fx.Int64(0).ir_value())
                    if use_a_direct
                    else lds_load_packs_k64(
                        row_a_lds, col_offset_base_bytes, lds_base_pong
                    )
                )

                # Main loop: process K tiles in 2-tile ping-pong steps.
                #
                # IMPORTANT: for odd number of K tiles, leave **1** tail tile; for even, leave **2**.
                # Otherwise the 2-tile tail below would double-count the last tile when num_tiles is odd
                # (e.g. inter_dim=192, tile_k=64 -> 3 tiles).
                num_k_tiles_py = int(inter_dim) // int(tile_k)
                odd_k_tiles = (num_k_tiles_py % 2) == 1
                tail_tiles = 1 if odd_k_tiles else 2
                k_main2_py = (num_k_tiles_py - tail_tiles) * int(tile_k)
                if const_expr(k_main2_py < 0):
                    k_main2_py = 0

                arith.index(tile_k * 2)
                c_tile_k_s2 = arith.index(tile_k)
                pair_iters = k_main2_py // (int(tile_k) * 2)

                # B-tile data layout per k_unroll entry (3 variants):
                #   See gemm1 _flatten_b_tile for full layout documentation.
                int4_bf16_single_field = is_int4_bf16 and not is_int4_bf16_groupwise
                _fields_per_ku = 1 if int4_bf16_single_field else 2
                _vals_per_b_tile = (
                    0
                    if _stream_b_req
                    else k_unroll * _fields_per_ku * num_acc_n
                )
                _n_acc = m_repeat * num_acc_n
                _p_b = _n_acc
                _p_a0 = _p_b + _vals_per_b_tile

                def _flatten_b_tile(b_tile):
                    """Flatten B tile to a 1-D list for scf.for loop-carried state."""
                    if const_expr(_stream_b_req):
                        return []
                    flat = []
                    for ku_entry in b_tile:
                        if is_int4_bf16_groupwise:
                            flat.extend(t[0] for t in ku_entry)
                            flat.extend(t[1] for t in ku_entry)
                        elif int4_bf16_single_field:
                            flat.extend(ku_entry)
                        else:
                            flat.extend(ku_entry[0])
                            flat.extend(ku_entry[1])
                    return flat

                def _unflatten_b_tile(vals):
                    """Reconstruct B tile from flattened scf.for loop-carried state."""
                    if const_expr(_stream_b_req):
                        return []
                    b_tile, idx = [], 0
                    for _ in range_constexpr(k_unroll):
                        if is_int4_bf16_groupwise:
                            packed = list(vals[idx : idx + num_acc_n])
                            idx += num_acc_n
                            scales = list(vals[idx : idx + num_acc_n])
                            idx += num_acc_n
                            b_tile.append(
                                [
                                    (packed[ni], scales[ni])
                                    for ni in range_constexpr(num_acc_n)
                                ]
                            )
                        elif int4_bf16_single_field:
                            b_tile.append(list(vals[idx : idx + num_acc_n]))
                            idx += num_acc_n
                        else:
                            packs_even = list(vals[idx : idx + num_acc_n])
                            idx += num_acc_n
                            packs_odd = list(vals[idx : idx + num_acc_n])
                            idx += num_acc_n
                            b_tile.append((packs_even, packs_odd))
                    return b_tile

                init_state = list(acc) + _flatten_b_tile(b_cur) + list(a0_prefetch_pong)

                for pair_iv, state in range(0, pair_iters, 1, init=init_state):
                    _ac = list(state[:_n_acc])
                    _bc = _unflatten_b_tile(list(state[_p_b:_p_a0]))
                    _a0 = (state[_p_a0], state[_p_a0 + 1])

                    k_iv = pair_iv * (c_tile_k_s2 + c_tile_k_s2)

                    next_k1 = k_iv + c_tile_k_s2
                    if const_expr(not use_a_direct):
                        x_regs_ping = load_x_tile(next_k1)
                    if const_expr(_stream_b_req):
                        _bp = []
                    elif const_expr(not _late_b_req):
                        _bp = load_b_tile(next_k1)

                    _ac, _ = compute_tile(
                        _ac,
                        _bc,
                        lds_base_pong,
                        base_k_for_a=k_iv,
                        base_k_for_b=k_iv,
                        a0_prefetch=_a0,
                    )
                    if const_expr(_stream_b_req):
                        pass
                    elif const_expr(_late_b_req):
                        _bp = load_b_tile(next_k1)
                    if const_expr(not use_a_direct):
                        store_x_tile_to_lds(x_regs_ping, lds_base_ping)
                    hot_loop_scheduler()
                    stage_barrier()

                    _a0p = (
                        (fx.Int64(0).ir_value(), fx.Int64(0).ir_value())
                        if use_a_direct
                        else lds_load_packs_k64(
                            row_a_lds, col_offset_base_bytes, lds_base_ping
                        )
                    )

                    next_k2 = k_iv + c_tile_k_s2 + c_tile_k_s2
                    if const_expr(not use_a_direct):
                        x_regs_pong = load_x_tile(next_k2)
                    if const_expr(_stream_b_req):
                        _bn = []
                    elif const_expr(not _late_b_req):
                        _bn = load_b_tile(next_k2)

                    _ac, _ = compute_tile(
                        _ac,
                        _bp,
                        lds_base_ping,
                        base_k_for_a=next_k1,
                        base_k_for_b=next_k1,
                        a0_prefetch=_a0p,
                    )
                    if const_expr(_stream_b_req):
                        pass
                    elif const_expr(_late_b_req):
                        _bn = load_b_tile(next_k2)
                    if const_expr(not use_a_direct):
                        store_x_tile_to_lds(x_regs_pong, lds_base_pong)
                    hot_loop_scheduler()
                    stage_barrier()

                    _a0n = (
                        (fx.Int64(0).ir_value(), fx.Int64(0).ir_value())
                        if use_a_direct
                        else lds_load_packs_k64(
                            row_a_lds, col_offset_base_bytes, lds_base_pong
                        )
                    )

                    loop_results = yield list(_ac) + _flatten_b_tile(_bn) + list(_a0n)

                SmemPtr._view_cache = None
                if pair_iters > 0:
                    acc = list(loop_results[:_n_acc])
                    b_cur = _unflatten_b_tile(list(loop_results[_p_b:_p_a0]))
                    a0_prefetch_pong = (loop_results[_p_a0], loop_results[_p_a0 + 1])

                if const_expr(odd_k_tiles):
                    # Tail: single remaining tile (already in `b_cur` / `lds_base_pong`).
                    k_tail0 = k_in - tile_k
                    acc, epilogue_pf = compute_tile(
                        acc,
                        b_cur,
                        lds_base_pong,
                        base_k_for_a=k_tail0,
                        base_k_for_b=k_tail0,
                        prefetch_epilogue=True,
                        a0_prefetch=a0_prefetch_pong,
                    )
                else:
                    k_tail1 = k_in - tile_k
                    k_tail0 = k_in - (tile_k + tile_k)
                    if const_expr(not use_a_direct):
                        x_regs_ping = load_x_tile(k_tail1)
                    if const_expr(_stream_b_req):
                        b_ping = []
                    elif const_expr(not _late_b_req):
                        b_ping = load_b_tile(k_tail1)

                    acc, _ = compute_tile(
                        acc,
                        b_cur,
                        lds_base_pong,
                        base_k_for_a=k_tail0,
                        base_k_for_b=k_tail0,
                        a0_prefetch=a0_prefetch_pong,
                    )
                    if const_expr(_stream_b_req):
                        pass
                    elif const_expr(_late_b_req):
                        b_ping = load_b_tile(k_tail1)
                    if const_expr(not use_a_direct):
                        store_x_tile_to_lds(x_regs_ping, lds_base_ping)
                    hot_loop_scheduler()
                    stage_barrier()

                    a0_prefetch_ping = (
                        (fx.Int64(0).ir_value(), fx.Int64(0).ir_value())
                        if use_a_direct
                        else lds_load_packs_k64(
                            row_a_lds, col_offset_base_bytes, lds_base_ping
                        )
                    )
                    acc, epilogue_pf = compute_tile(
                        acc,
                        b_ping,
                        lds_base_ping,
                        base_k_for_a=k_tail1,
                        base_k_for_b=k_tail1,
                        prefetch_epilogue=True,
                        a0_prefetch=a0_prefetch_ping,
                    )

                # ---------------- Epilogue: LDS CShuffle + atomic half2 (x2) ----------------
                # Reuse the shared helper so GEMM / MoE kernels share the exact same CShuffle skeleton.
                expert_off = expert_off_idx
                mask24_i32 = fx.Int32(0xFFFFFF)
                model_i32 = fx.Int32(model_dim)
                topk_i32_v = topk_i32

                zero_i32 = fx.Int32(0)
                atomic_aux_i32 = fx.Int32(_atomic_aux_req)
                c2_i32 = fx.Int32(2)  # 2B element size for f16/bf16
                mask_even_i32 = fx.Int32(
                    0xFFFFFFFE
                )  # align element index to even for half2 atomics

                e_vec = _e_vec

                def atomic_add_f16x2(val_f16x2, byte_off_i32):
                    rocdl.raw_ptr_buffer_atomic_fadd(
                        val_f16x2,
                        out_rsrc,
                        byte_off_i32,
                        zero_i32,
                        atomic_aux_i32,
                    )

                sw_pf = None
                tw_pf = None
                if const_expr(epilogue_pf is not None):
                    sw_pf, tw_pf = epilogue_pf

                # Weight scales for the N tile (col_g depends on lane/wave/by but not on (t,s)).
                if const_expr(use_groupwise_scale):
                    # Groupwise: weight scale already applied per-group in K-loop.
                    sw_vals = [arith.constant(1.0, type=T.f32)] * num_acc_n
                elif const_expr(sw_pf is not None):
                    sw_vals = sw_pf
                else:
                    sw_vals = []
                    for ni in range_constexpr(num_acc_n):
                        col_g = col_g_list[ni]
                        row_w_idx = expert_off + col_g
                        sw_vals.append(
                            fx.Float32(1.0)
                            if not needs_scale_w
                            else buffer_ops.buffer_load(
                                sw_rsrc, row_w_idx, vec_width=1, dtype=T.f32
                            )
                        )

                # When defer_scale16 was used, the x16 correction for v_cvt_off_f32_i4
                # was omitted from the hot loop.  Fold it into the epilogue scale.
                if const_expr(use_gfx950_cvt):
                    _c16 = fx.Float32(16.0)
                    sw_vals = [v * _c16 for v in sw_vals]

                if const_expr(out_is_f32):
                    # origin/dev_a16w4: f32 output uses scalar f32 atomics and skips CShuffle/LDS.
                    c4_i32 = fx.Int32(4)

                    def atomic_add_f32(val_f32, byte_off_i32):
                        rocdl.raw_ptr_buffer_atomic_fadd(
                            val_f32,
                            out_rsrc,
                            byte_off_i32,
                            zero_i32,
                            atomic_aux_i32,
                        )

                    def _stage2_row_atomic(*, mi: int, ii: int, row_in_tile, row):
                        fused2 = buffer_ops.buffer_load(
                            sorted_rsrc, row, vec_width=1, dtype=T.i32
                        )
                        t2 = fused2 & mask24_i32
                        s2 = fused2 >> 24

                        # Mask sentinel (token_id==tokens, slot==topk) to avoid OOB scale_x loads.
                        # For invalid rows, force sx=0 so they contribute exactly 0 to output.
                        t_ok = arith.cmpi(arith.CmpIPredicate.ult, t2, tokens_i32)
                        s_ok = arith.cmpi(arith.CmpIPredicate.ult, s2, topk_i32_v)
                        ts_ok = t_ok & s_ok
                        t2_safe = ts_ok.select(t2, fx.Int32(0))
                        s2_safe = ts_ok.select(s2, fx.Int32(0))
                        ts2 = t2_safe * topk_i32_v + s2_safe
                        sx = (
                            arith.select(ts_ok, fx.Float32(1.0), fx.Float32(0.0))
                            if is_f16_or_bf16
                            else arith.select(
                                ts_ok,
                                buffer_ops.buffer_load(
                                    sx_rsrc, ts2, vec_width=1, dtype=T.f32
                                ),
                                fx.Float32(0.0),
                            )
                        )

                        if const_expr(doweight_stage2):
                            tw_idx = (mi * 4) + ii
                            if const_expr(tw_pf is not None):
                                tw = ts_ok.select(tw_pf[tw_idx], fx.Float32(0.0))
                            else:
                                tw = arith.select(
                                    ts_ok,
                                    buffer_ops.buffer_load(
                                        sorted_w_rsrc, row, vec_width=1, dtype=T.f32
                                    ),
                                    fx.Float32(0.0),
                                )

                        idx0 = (
                            t2_safe * model_i32
                        )  # i32 element index base (safe for sentinel rows)

                        for ni in range_constexpr(num_acc_n):
                            col_g = col_g_list[ni]
                            sw = sw_vals[ni]
                            acc_idx = mi * num_acc_n + ni
                            v = vector.extract(
                                acc[acc_idx], static_position=[ii], dynamic_position=[]
                            )
                            if const_expr(is_int8):
                                v = arith.sitofp(T.f32, v)
                            v = v * sx * sw
                            if const_expr(doweight_stage2):
                                v = v * tw
                            col_i32 = arith.index_cast(T.i32, col_g)
                            idx_elem = idx0 + col_i32
                            byte_off = idx_elem * c4_i32
                            atomic_add_f32(v, byte_off)

                    default_epilog(
                        arith=arith,
                        range_constexpr=range_constexpr,
                        m_repeat=m_repeat,
                        lane_div_16=lane_div_16,
                        bx_m=bx_m,
                        body_row=_stage2_row_atomic,
                    )
                else:
                    if const_expr(lds_out is None):
                        raise RuntimeError(
                            "FLYDSL_MOE_STAGE2_CSHUFFLE=1 but lds_out is not allocated/aliased."
                        )

                    # For bf16 global atomics (gfx942 only), precompute the output base address.
                    # gfx950+ has buffer_atomic_pk_add_bf16, so bf16 uses buffer atomics there.
                    out_base_idx = None
                    if const_expr(_needs_global_atomic_bf16):
                        out_base_idx = buffer_ops.extract_base_index(arg_out)

                    def write_row_to_lds(
                        *,
                        mi: int,
                        ii: int,
                        row_in_tile,
                        row,
                        row_base_lds,
                        col_base_local,
                        num_acc_n: int,
                        lds_out,
                    ):
                        row_valid_write = None
                        if const_expr(_epilog_row_skip_req):
                            fused_w = buffer_ops.buffer_load(
                                sorted_rsrc, row, vec_width=1, dtype=T.i32
                            )
                            row_i32_w = arith.index_cast(T.i32, row)
                            row_valid0_w = arith.cmpi(
                                arith.CmpIPredicate.ult, row_i32_w, num_valid_i32
                            )
                            t_w = fused_w & mask24_i32
                            s_w = fused_w >> 24
                            t_ok_w = arith.cmpi(
                                arith.CmpIPredicate.ult, t_w, tokens_i32
                            )
                            s_ok_w = arith.cmpi(
                                arith.CmpIPredicate.ult, s_w, topk_i32_v
                            )
                            row_valid_write = row_valid0_w & t_ok_w & s_ok_w

                        def _write_valid_row_to_lds():
                            if const_expr(is_f16_or_bf16):
                                # BF16/F16 stage2 has implicit activation scale=1.0.
                                sx = fx.Float32(1.0)
                            else:
                                fused2 = buffer_ops.buffer_load(
                                    sorted_rsrc, row, vec_width=1, dtype=T.i32
                                )
                                t2 = fused2 & mask24_i32
                                s2 = fused2 >> 24
                                # Explicitly mask sentinel token/slot to avoid OOB scale_x loads.
                                t_ok = arith.cmpi(
                                    arith.CmpIPredicate.ult, t2, tokens_i32
                                )
                                s_ok = arith.cmpi(
                                    arith.CmpIPredicate.ult, s2, topk_i32_v
                                )
                                ts_ok = t_ok & s_ok
                                t2_safe = ts_ok.select(t2, fx.Int32(0))
                                s2_safe = ts_ok.select(s2, fx.Int32(0))
                                ts2 = t2_safe * topk_i32_v + s2_safe
                                sx = arith.select(
                                    ts_ok,
                                    buffer_ops.buffer_load(
                                        sx_rsrc, ts2, vec_width=1, dtype=T.f32
                                    ),
                                    fx.Float32(0.0),
                                )

                            if const_expr(doweight_stage2):
                                tw_idx = (mi * 4) + ii
                                if const_expr(tw_pf is not None):
                                    tw = tw_pf[tw_idx]
                                else:
                                    tw = buffer_ops.buffer_load(
                                        sorted_w_rsrc, row, vec_width=1, dtype=T.f32
                                    )

                            for ni in range_constexpr(num_acc_n):
                                col_local = col_base_local + (ni * 16)
                                sw = sw_vals[ni]
                                acc_idx = mi * num_acc_n + ni
                                v = vector.extract(
                                    acc[acc_idx],
                                    static_position=[ii],
                                    dynamic_position=[],
                                )
                                if const_expr(is_int8):
                                    v = arith.sitofp(T.f32, v)
                                v = v * sx * sw
                                if const_expr(doweight_stage2):
                                    v = v * tw
                                v_out = arith.trunc_f(out_elem(), v)

                                lds_idx = row_base_lds + col_local
                                vec1_out = T.vec(1, out_elem())
                                v1 = vector.from_elements(vec1_out, [v_out])
                                vector.store(v1, lds_out, [lds_idx], alignment=2)

                        if const_expr(_epilog_row_skip_req):
                            _if_write = scf.IfOp(row_valid_write)
                            with _if_then(_if_write):
                                _write_valid_row_to_lds()
                        else:
                            _write_valid_row_to_lds()

                    def precompute_row(*, row_local, row):
                        # Precompute row context for cshuffle stores.
                        # Return (fused_i32, row_valid_i1) so the epilogue can skip the entire row
                        # for invalid tail rows (CK-style), avoiding per-store branching.
                        fused2 = buffer_ops.buffer_load(
                            sorted_rsrc, row, vec_width=1, dtype=T.i32
                        )
                        row_i32 = arith.index_cast(T.i32, row)
                        row_valid0 = arith.cmpi(
                            arith.CmpIPredicate.ult, row_i32, num_valid_i32
                        )
                        t = fused2 & mask24_i32
                        s = fused2 >> 24
                        t_ok = arith.cmpi(arith.CmpIPredicate.ult, t, tokens_i32)
                        s_ok = arith.cmpi(arith.CmpIPredicate.ult, s, topk_i32_v)
                        row_valid = row_valid0 & t_ok & s_ok
                        if const_expr(_rowctx_base_req):
                            idx0 = t * model_i32
                            if const_expr(not bool(accumulate)):
                                ts = t * topk_i32_v + s
                                idx0 = ts * model_i32
                            return (idx0, row_valid)
                        return (fused2, row_valid)

                    def store_pair(*, row_local, row, row_ctx, col_pair0, col_g0, frag):
                        if const_expr(_rowctx_base_req):
                            idx0 = row_ctx
                        else:
                            fused = row_ctx
                            t = fused & mask24_i32
                            s = fused >> 24
                            idx0 = t * model_i32
                            if const_expr(not bool(accumulate)):
                                ts = t * topk_i32_v + s
                                idx0 = ts * model_i32
                        col_i32 = arith.index_cast(T.i32, col_g0)
                        idx_elem = idx0 + col_i32
                        idx_elem_even = (
                            idx_elem
                            if const_expr(_skip_even_mask_req)
                            else (idx_elem & mask_even_i32)
                        )
                        if const_expr(_needs_global_atomic_bf16):
                            # gfx942: no buffer_atomic_pk_add_bf16, use global atomicrmw fadd
                            if const_expr(bool(accumulate)):
                                byte_off = idx_elem_even * c2_i32
                                byte_off_idx = arith.index_cast(T.index, byte_off)
                                ptr_addr_idx = out_base_idx + byte_off_idx
                                out_ptr = buffer_ops.create_llvm_ptr(
                                    ptr_addr_idx, address_space=1
                                )
                                out_ptr_v = (
                                    out_ptr._value
                                    if const_expr(hasattr(out_ptr, "_value"))
                                    else out_ptr
                                )
                                frag_v = (
                                    frag._value if hasattr(frag, "_value") else frag
                                )
                                llvm.AtomicRMWOp(
                                    llvm.AtomicBinOp.fadd,
                                    out_ptr_v,
                                    frag_v,
                                    llvm.AtomicOrdering.monotonic,
                                    syncscope="agent",
                                    alignment=4,
                                )
                            else:
                                buffer_ops.buffer_store(frag, out_rsrc, idx_elem_even)
                        else:
                            # f16, or bf16 on gfx950+ (has buffer_atomic_pk_add_bf16)
                            byte_off = idx_elem_even * c2_i32
                            if const_expr(bool(accumulate)):
                                atomic_add_f16x2(frag, byte_off)
                            else:
                                buffer_ops.buffer_store(frag, out_rsrc, idx_elem_even)

                    c_shuffle_epilog(
                        arith=arith,
                        vector=vector,
                        gpu=gpu,
                        scf=scf,
                        range_constexpr=range_constexpr,
                        tile_m=tile_m,
                        tile_n=tile_n,
                        e_vec=e_vec,
                        cshuffle_nlane=_cshuffle_nlane,
                        block_size=total_threads,
                        m_repeat=m_repeat,
                        num_acc_n=num_acc_n,
                        tx=tx,
                        lane_div_16=lane_div_16,
                        lane_mod_16=lane_mod_16,
                        bx_m=bx_m,
                        by_n=by_n,
                        n_tile_base=n_tile_base,
                        lds_out=lds_out,
                        frag_elem_type=(T.bf16 if out_is_bf16 else T.f16),
                        write_row_to_lds=write_row_to_lds,
                        precompute_row=precompute_row,
                        precompute_broadcast_i32=_rowctx_bcast_req,
                        store_pair=store_pair,
                    )

            _if_blk = scf.IfOp(blk_valid)
            with _if_then(_if_blk):
                _moe_gemm2_then_body()

    # -- Host launcher (flyc.jit + .launch) --------------------------------
    @flyc.jit
    def launch_moe_gemm2(
        arg_out: fx.Tensor,
        arg_x: fx.Tensor,
        arg_w: fx.Tensor,
        arg_scale_x: fx.Tensor,
        arg_scale_w: fx.Tensor,
        arg_sorted_token_ids: fx.Tensor,
        arg_expert_ids: fx.Tensor,
        arg_sorted_weights: fx.Tensor,
        arg_num_valid_ids: fx.Tensor,
        i32_tokens_in: fx.Int32,
        i32_n_in: fx.Int32,
        i32_k_in: fx.Int32,
        i32_size_expert_ids_in: fx.Int32,
        stream: fx.Stream,
    ):
        _ = _cache_tag
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        if const_expr(waves_per_eu > 0):
            for op in ctx.gpu_module_body.operations:
                if const_expr(hasattr(op, "attributes") and op.OPERATION_NAME == "gpu.func"):
                    op.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(
                        T.i32, int(waves_per_eu)
                    )

        n_in = arith.index_cast(T.index, i32_n_in)
        size_expert_ids_in = arith.index_cast(T.index, i32_size_expert_ids_in)
        gx = n_in // fx.Index(tile_n)
        gy = size_expert_ids_in

        launch_grid = (gy, gx, 1) if _grid_m_fast_req else (gx, gy, 1)

        moe_gemm2(
            arg_out,
            arg_x,
            arg_w,
            arg_scale_x,
            arg_scale_w,
            arg_sorted_token_ids,
            arg_expert_ids,
            arg_sorted_weights,
            arg_num_valid_ids,
            i32_tokens_in,
            i32_n_in,
            i32_k_in,
            i32_size_expert_ids_in,
        ).launch(
            grid=launch_grid,
            block=(total_threads, 1, 1),
            stream=stream,
        )

    return launch_moe_gemm2


# MoE Reduction Kernel (reduce sum over topk dimension)
@functools.lru_cache(maxsize=1024)
def compile_moe_reduction(
    *,
    topk: int,
    model_dim: int,
    dtype_str: str = "f16",
    use_mask: bool = False,
):
    """Compile a reduction kernel that sums over the topk dimension.

    Input:  X [tokens, topk, model_dim]
            valid_mask [tokens, topk] (optional, if use_mask=True)
    Output: Y [tokens, model_dim]

    This kernel performs: Y[t, d] = sum(X[t, :, d]) for all t, d.
    When use_mask=True, only sums slots where valid_mask[t,k]=1.
    Used in conjunction with compile_moe_gemm2(accumulate=False) to avoid atomic contention.
    """
    get_hip_arch()
    ir.ShapedType.get_dynamic_size()

    # Kernel Config
    BLOCK_SIZE = 256
    VEC_WIDTH = 8

    if dtype_str == "f32":
        elem_type_tag = "f32"
    elif dtype_str == "f16":
        elem_type_tag = "f16"
    elif dtype_str == "bf16":
        elem_type_tag = "bf16"
    else:
        raise ValueError(f"Unsupported dtype: {dtype_str}")

    def compute_type():
        return T.f32

    def i32_type():
        return T.i32

    def i8_type():
        return T.i8

    def elem_type():
        ty = (
            T.f32
            if elem_type_tag == "f32"
            else (T.f16 if elem_type_tag == "f16" else T.bf16)
        )
        return ty() if callable(ty) else ty

    if True:

        @flyc.kernel
        def moe_reduction_kernel(
            X: fx.Tensor,
            Y: fx.Tensor,
            valid_mask: fx.Tensor,
            i32_m_tokens: fx.Int32,
        ):
            m_tokens = fx.Index(i32_m_tokens)
            c_topk = fx.Index(topk)
            c_model_dim = fx.Index(model_dim)
            mask_nbytes_idx = m_tokens * c_topk
            elem_bits = 32 if dtype_str == "f32" else 16
            copy_vec_width = 128 // elem_bits  # 8 for f16/bf16, 4 for f32
            n_sub = VEC_WIDTH // copy_vec_width  # 1 for f16/bf16, 2 for f32
            # Buffer-backed tensors via layout API (all dtypes)
            X_buf = fx.rocdl.make_buffer_tensor(X)
            Y_buf = fx.rocdl.make_buffer_tensor(Y)
            # Scalar buffer resources for tail path and mask
            x_rsrc = buffer_ops.create_buffer_resource(X, max_size=True)
            y_rsrc = buffer_ops.create_buffer_resource(Y, max_size=True)
            mask_rsrc = buffer_ops.create_buffer_resource(
                valid_mask, max_size=False, num_records_bytes=mask_nbytes_idx
            )

            token_idx = gpu.block_id("x")
            tile_idx = gpu.block_id("y")
            tid = gpu.thread_id("x")

            # Guard: token in range (Index is unsigned -> auto ult)
            tok_ok = token_idx < m_tokens
            _if_tok = scf.IfOp(tok_ok)
            with _if_then(_if_tok):
                tile_cols = BLOCK_SIZE * VEC_WIDTH
                c_tile_cols = fx.Index(tile_cols)
                c_vecw = fx.Index(VEC_WIDTH)

                col_base = tile_idx * c_tile_cols + tid * c_vecw

                # Guard: any work in bounds (Index < -> ult)
                col_ok = col_base < c_model_dim
                _if_col = scf.IfOp(col_ok)
                with _if_then(_if_col):
                    # Fast path: full vector in-bounds (Index <= -> ule)
                    end_ok = col_base + c_vecw <= c_model_dim
                    _if_full = scf.IfOp(end_ok, has_else=True)
                    with _if_then(_if_full):
                        # -- Vector path via layout API (all dtypes) --
                        # fx.copy auto-iterates when atom width < VEC_WIDTH
                        # (e.g. f32: BufferCopy128b handles 4, fx.copy issues 2 calls for 8)
                        copy_atom = fx.make_copy_atom(
                            fx.rocdl.BufferCopy128b(), elem_bits
                        )
                        vec_type_c = T.vec(copy_vec_width, compute_type())
                        vec_type_e = T.vec(copy_vec_width, elem_type())

                        acc_vecs = [
                            vector.broadcast(vec_type_c, fx.Float32(0.0).ir_value())
                            for _ in range(n_sub)
                        ]
                        reg_ty = fx.MemRefType.get(
                            elem_type(),
                            fx.LayoutType.get(copy_vec_width, 1),
                            fx.AddressSpace.Register,
                        )
                        reg_lay = fx.make_layout(copy_vec_width, 1)

                        tok_i32 = fx.Int32(token_idx)
                        tile_i32 = fx.Int32(tile_idx)
                        tid_i32 = fx.Int32(tid)

                        for k in range_constexpr(topk):
                            # X[token, k, :] -> tile -> thread's VEC_WIDTH slice
                            x_row = X_buf[tok_i32, fx.Int32(k), None]
                            x_tiled = fx.logical_divide(
                                x_row, fx.make_layout(tile_cols, 1)
                            )
                            x_div = fx.logical_divide(
                                x_tiled[None, tile_i32], fx.make_layout(VEC_WIDTH, 1)
                            )
                            x_thread = x_div[None, tid_i32]

                            if const_expr(use_mask):
                                m_idx_i32 = fx.Int32(token_idx * c_topk + fx.Index(k))
                                mv = buffer_ops.buffer_load(
                                    mask_rsrc, m_idx_i32, vec_width=1, dtype=i8_type()
                                )
                                mv_ok = mv != fx.Int8(0)

                            if const_expr(n_sub > 1):
                                x_inner = fx.logical_divide(
                                    x_thread, fx.make_layout(copy_vec_width, 1)
                                )
                            for si in range_constexpr(n_sub):
                                src = (
                                    x_inner[None, fx.Int32(si)]
                                    if n_sub > 1
                                    else x_thread
                                )
                                r = fx.memref_alloca(reg_ty, reg_lay)
                                fx.copy_atom_call(copy_atom, src, r)
                                vec_e = fx.memref_load_vec(r)

                                if const_expr(use_mask):
                                    zero_e = vector.broadcast(
                                        vec_type_e,
                                        arith.constant(0.0, type=elem_type()),
                                    )
                                    vec_e = mv_ok.select(vec_e, zero_e)

                                if const_expr(elem_bits < 32):
                                    vec_c = vec_e.extf(vec_type_c)
                                else:
                                    vec_c = vec_e
                                acc_vecs[si] = acc_vecs[si] + vec_c

                        # -- Store results --
                        if const_expr(n_sub > 1):
                            y_row = Y_buf[tok_i32, None]
                            y_tiled = fx.logical_divide(
                                y_row, fx.make_layout(tile_cols, 1)
                            )
                            y_div = fx.logical_divide(
                                y_tiled[None, tile_i32], fx.make_layout(VEC_WIDTH, 1)
                            )
                            y_inner = fx.logical_divide(
                                y_div[None, tid_i32], fx.make_layout(copy_vec_width, 1)
                            )

                        for si in range_constexpr(n_sub):
                            out_vec = acc_vecs[si]
                            if const_expr(elem_bits < 32):
                                out_vec = out_vec.truncf(vec_type_e)

                            if const_expr(n_sub > 1):
                                dst = y_inner[None, fx.Int32(si)]
                            else:
                                y_row = Y_buf[tok_i32, None]
                                y_tiled = fx.logical_divide(
                                    y_row, fx.make_layout(tile_cols, 1)
                                )
                                y_div = fx.logical_divide(
                                    y_tiled[None, tile_i32],
                                    fx.make_layout(VEC_WIDTH, 1),
                                )
                                dst = y_div[None, tid_i32]

                            r_out = fx.memref_alloca(reg_ty, reg_lay)
                            fx.memref_store_vec(out_vec, r_out)
                            fx.copy_atom_call(copy_atom, r_out, dst)

                    with _if_else(_if_full):
                        # Tail path: scalar load/store per lane.
                        for lane in range_constexpr(VEC_WIDTH):
                            col = col_base + fx.Index(lane)
                            lane_ok = col < c_model_dim
                            _if_lane = scf.IfOp(lane_ok)
                            with _if_then(_if_lane):
                                a = arith.constant(0.0, type=compute_type())
                                token_base = token_idx * c_topk
                                for k in range_constexpr(topk):
                                    k_idx = fx.Index(k)
                                    x_idx_i32 = fx.Int32(
                                        (token_base + k_idx) * c_model_dim + col
                                    )
                                    if const_expr(use_mask):
                                        m_idx_i32 = fx.Int32(token_base + k_idx)
                                        mv = buffer_ops.buffer_load(
                                            mask_rsrc,
                                            m_idx_i32,
                                            vec_width=1,
                                            dtype=i8_type(),
                                        )
                                        v = (mv != fx.Int8(0)).select(
                                            buffer_ops.buffer_load(
                                                x_rsrc,
                                                x_idx_i32,
                                                vec_width=1,
                                                dtype=elem_type(),
                                            ),
                                            arith.constant(0.0, type=elem_type()),
                                        )
                                    else:
                                        v = buffer_ops.buffer_load(
                                            x_rsrc,
                                            x_idx_i32,
                                            vec_width=1,
                                            dtype=elem_type(),
                                        )
                                    if const_expr(dtype_str in ("f16", "bf16")):
                                        v = v.extf(compute_type())
                                    a = a + v

                                out = a
                                if const_expr(dtype_str in ("f16", "bf16")):
                                    out = out.truncf(elem_type())
                                y_idx_i32 = fx.Int32(token_idx * c_model_dim + col)
                                buffer_ops.buffer_store(out, y_rsrc, y_idx_i32)

    # -- Host launcher (flyc.jit + .launch) --------------------------------
    tile_size = BLOCK_SIZE * VEC_WIDTH
    gy_static = (model_dim + tile_size - 1) // tile_size

    @flyc.jit
    def launch_moe_reduction(
        X: fx.Tensor,
        Y: fx.Tensor,
        valid_mask: fx.Tensor,
        i32_m_tokens: fx.Int32,
        stream: fx.Stream,
    ):
        gx = fx.Index(i32_m_tokens)
        moe_reduction_kernel(X, Y, valid_mask, i32_m_tokens).launch(
            grid=(gx, gy_static, 1),
            block=(BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    return launch_moe_reduction


# MoE GEMM2 Execution Modes
class MoeGemm2Mode:
    """Execution mode for MoE GEMM2."""

    ATOMIC = "atomic"  # Use atomic accumulation (default)
    REDUCE = "reduce"  # Use non-atomic write + reduce kernel


class _MoeGemm2ReduceWrapper:
    """Wrapper combining GEMM2 (no atomics) with reduction kernel.

    This wrapper handles the intermediate buffer allocation and orchestrates
    the two-phase computation:
    1. GEMM2 outputs to [tokens*topk, model_dim] without atomics
    2. Reduce sums over topk to produce [tokens, model_dim]
    """

    def __init__(
        self,
        gemm2_exe,
        reduce_exe,
        topk: int,
        model_dim: int,
        out_dtype_str: str = "f16",
        use_mask: bool = False,
        zero_intermediate: bool = True,
    ):
        self._gemm2_exe = gemm2_exe
        self._reduce_exe = reduce_exe
        self._topk = topk
        self._model_dim = model_dim
        self._out_dtype_str = out_dtype_str
        self._use_mask = use_mask
        self._zero_intermediate = zero_intermediate

    def _get_torch_dtype(self):
        """Convert dtype string to torch dtype."""
        import torch

        dtype_map = {
            "f16": torch.float16,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "f32": torch.float32,
        }
        return dtype_map.get(self._out_dtype_str, torch.float16)

    def __call__(
        self,
        arg_out,
        arg_x,
        arg_w,
        arg_scale_x,
        arg_scale_w,
        arg_sorted_token_ids,
        arg_expert_ids,
        arg_sorted_weights,
        arg_num_valid_ids,
        tokens_in,
        n_in,
        k_in,
        size_expert_ids_in,
        valid_mask=None,
        stream=None,
    ):
        """Execute GEMM2 + reduce.

        Args match moe_gemm2 kernel signature (see compile_moe_gemm2).
        """
        import torch

        if stream is None:
            stream = torch.cuda.current_stream()
        intermediate = torch.empty(
            tokens_in * self._topk,
            self._model_dim,
            device=arg_out.device,
            dtype=self._get_torch_dtype(),
        )
        if self._zero_intermediate and not self._use_mask:
            intermediate.zero_()
        # Phase 1: GEMM2 (no atomics) -> [tokens*topk, model_dim]
        self._gemm2_exe(
            intermediate.view(-1),
            arg_x,
            arg_w,
            arg_scale_x,
            arg_scale_w,
            arg_sorted_token_ids,
            arg_expert_ids,
            arg_sorted_weights,
            arg_num_valid_ids,
            tokens_in,
            n_in,
            k_in,
            size_expert_ids_in,
            stream,
        )
        # Phase 2: Reduce over topk -> [tokens, model_dim]
        X = intermediate.view(tokens_in, self._topk, self._model_dim)
        Y = arg_out.view(tokens_in, self._model_dim)
        if not self._use_mask:
            if valid_mask is not None:
                logging.warning(
                    "valid_mask provided but use_mask=False; ignoring valid_mask"
                )
            valid_mask = torch.empty(
                (0, self._topk), device=arg_out.device, dtype=torch.uint8
            )
        self._reduce_exe(X, Y, valid_mask, tokens_in, stream)

    @property
    def mode(self) -> str:
        """Return the execution mode."""
        return MoeGemm2Mode.REDUCE


def compile_moe_gemm2_ex(
    *,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    doweight_stage2: bool,
    in_dtype: str = "fp8",
    group_size: int = -1,
    out_dtype: str = "f16",
    use_cshuffle_epilog: bool | None = None,
    # Extended parameters for mode control
    mode: str = MoeGemm2Mode.ATOMIC,
    valid_mask=None,
    zero_intermediate: bool = True,
    scale_is_bf16: bool = False,
):
    """Compile MoE GEMM2 kernel with optional reduction.

    This is the extended interface that supports explicit mode control.

    Args:
        mode: Execution mode selection:
            - "atomic": Use atomic accumulation (original behavior)
            - "reduce": Use non-atomic write + reduce kernel

        zero_intermediate: If all output slots are valid,
            set False to increase performance

    Returns:
        Compiled executable (either wrapped or raw depending on mode).
    """
    # Compile based on mode
    if mode == MoeGemm2Mode.REDUCE:
        # Determine if we need masked reduction
        use_mask = valid_mask is not None

        # Compile GEMM2 with accumulate=False
        gemm2_exe = compile_moe_gemm2(
            model_dim=model_dim,
            inter_dim=inter_dim,
            experts=experts,
            topk=topk,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            doweight_stage2=doweight_stage2,
            in_dtype=in_dtype,
            group_size=group_size,
            out_dtype=out_dtype,
            use_cshuffle_epilog=use_cshuffle_epilog,
            accumulate=False,
            scale_is_bf16=scale_is_bf16,
        )
        # Compile reduction kernel with masking support
        out_s = str(out_dtype).strip().lower()
        if out_s in ("f16", "fp16", "half"):
            dtype_str = "f16"
        elif out_s in ("bf16", "bfloat16"):
            dtype_str = "bf16"
        else:
            dtype_str = "f32"
        reduce_exe = compile_moe_reduction(
            topk=topk,
            model_dim=model_dim,
            dtype_str=dtype_str,
            use_mask=use_mask,
        )
        return _MoeGemm2ReduceWrapper(
            gemm2_exe=gemm2_exe,
            reduce_exe=reduce_exe,
            topk=topk,
            model_dim=model_dim,
            out_dtype_str=dtype_str,
            use_mask=use_mask,
            zero_intermediate=zero_intermediate,
        )
    else:
        # Compile GEMM2 with accumulate=True (atomic mode)
        return compile_moe_gemm2(
            model_dim=model_dim,
            inter_dim=inter_dim,
            experts=experts,
            topk=topk,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            doweight_stage2=doweight_stage2,
            in_dtype=in_dtype,
            group_size=group_size,
            out_dtype=out_dtype,
            use_cshuffle_epilog=use_cshuffle_epilog,
            accumulate=True,
        )
