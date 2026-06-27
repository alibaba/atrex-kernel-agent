"""SM120 fused FA-epilogue + NVFP4 quant kernel (CuTeDSL).

⚠ TUNED FOR sm_120 (NVIDIA RTX PRO 5000 Blackwell-Geforce). NOT a generic-arch
   baseline. Differences from a hypothetical generic version:
   - SM80-era mma.sync reuse (no UMMA / wgmma)
   - 99 KB SMEM cap (not 163 KB Ampere or 228 KB Blackwell-datacenter)
   - CUTE_DSL_ARCH must be "sm_120a"
   - Cluster cutlass-DSL is 4.4.2 (no 4.5+ private-name API)
   - NVFP4 SF byte stream MUST go through st.global / cp.async, NEVER TMA
     (PipelineTmaAsync tx_count rejects non-tensor copies)
   - L1/TEX hit-rate ncu counter on sm_120 is ld.shared-polluted; verify
     cache mode via sm__inst_executed_pipe_tma vs smsp__inst_executed_op_global_ld

This kernel is V_final from the optimization journey
docs/ref-docs/nvidia/cutedsl/sm120/sm120-fused-fa-epilogue-nvfp4-bf16-optimization.md.

Path-1 epilogue: x = attn_out * sigmoid(gate) -> NVFP4 (e2m1 packed + e4m3
swizzled SF, group_size=16, swizzled-128x4 layout). Replaces (gate-mul +
sigmoid + standalone scaled_fp4_quant) two-kernel chain with one CuTeDSL kernel.

Performance: 88.05 us cuda.Event / 103.58 us ncu = 91.9% memcpy ceiling on
M=6144, D=4096 (sm_120 RTX PRO 5000). 6.5x fused vs (sigmoid_mul + standalone)
at canonical shape. See README.md for multi-shape matrix.

V0 = V1 = V2 within 1.4% (LDG / cp.async / TMA all hit memory wall). True V3
fusion (FA fwd + sigmoid+gate + nvfp4 in single kernel) is in
docs/ref-docs/nvidia/cutedsl/sm120/v3-fa-fusion-deferred-plan.md, blocked on
cluster cutlass-DSL >= 4.5.
"""

# IMPORTANT: do NOT use `from __future__ import annotations` here — cute 4.4.2's
# @cute.struct decorator iterates class annotations as REAL type objects (not
# strings), so PEP 563 lazy annotations would break SharedStorage definition.

import os
# CRITICAL: must be set BEFORE `import cutlass` so the DSL targets sm_120a
os.environ.setdefault("CUTE_DSL_ARCH", "sm_120a")

import math
import subprocess
import sys


# ----------------------------- Bootstrap real CuTeDSL -----------------------
#
# On the OLD cluster (sz6wd8l56pnf), system has *two* `cutlass` packages:
#   1) ``cutlass==0.1.0`` — unrelated ML lib pip-installed at
#      /usr/local/lib/python3.10/dist-packages/cutlass/__init__.py
#   2) ``nvidia_cutlass_dsl_libs_base==4.4.2`` — real CuTeDSL via .pth file
#      pointing at .../nvidia_cutlass_dsl/python_packages/cutlass/
# When both are visible, Python picks #1 first and `cutlass.cute` cannot be
# imported. We must remove #1 before importing cutlass. No-op on the new
# cluster (no bogus ML lib) and on machines without the path entirely.
def _bootstrap_cutedsl():
    bogus = "/usr/local/lib/python3.10/dist-packages/cutlass/__init__.py"
    if not os.path.exists(bogus):
        return
    try:
        with open(bogus, "r") as f:
            head = f.read(2000)
    except Exception:
        return
    is_bogus = (
        "CutlassClassifier" in head
        or "CutlassLogisticCV" in head
        or 'version = "0.1.0"' in head
    )
    if not is_bogus:
        return

    r = subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y", "cutlass"],
        capture_output=True, text=True, timeout=120,
    )
    if r.returncode != 0 or os.path.exists(bogus):
        import shutil
        import glob
        shutil.rmtree(
            "/usr/local/lib/python3.10/dist-packages/cutlass",
            ignore_errors=True,
        )
        for d in glob.glob(
            "/usr/local/lib/python3.10/dist-packages/cutlass-0.1.0.dist-info"
        ):
            shutil.rmtree(d, ignore_errors=True)

    for k in list(sys.modules):
        if k == "cutlass" or k.startswith("cutlass."):
            del sys.modules[k]
    import importlib
    importlib.invalidate_caches()


_bootstrap_cutedsl()


import torch
import torch.nn.functional as F

import cutlass
import cutlass.cute as cute
import cutlass.utils
import cutlass.cute.nvgpu.cpasync as cpasync
import cutlass.pipeline as pipeline
from cutlass import Float32, Int32, Uint8, Uint32
from cutlass.cute.runtime import from_dlpack
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

from cute_helpers import (
    NVFP4_SF_VEC_SIZE,
    st_global_u64,
    get_ptr_as_int64,
    get_smem_ptr_as_int32,
    ld_shared_v4_u32,
    rcp_approx_ftz,
    cvt_f32_to_e4m3,
    nvfp4_compute_output_scale,
    bfloat2_max_abs_8,
    bfloat2_hmax_reduce_to_f32,
    bfloat2x8_to_e2m1x16_packed,
    bfloat2_sigmoid_mul,
    compute_sf_index_swizzled_128x4_gpu,
)


# ============================================================================
# V1-TMA constants (must be MODULE-LEVEL Python int literals; cute 4.4.2
# rejects Constexpr-typed shape values inside make_layout / make_tiled_tma_atom)
# ============================================================================

# Tile dimensions — match flashinfer NVFP4QuantizeTMAKernel
_TMA_ROW_TILE              = 16
_TMA_COL_TILE              = 64                          # per-consumer-warp col stripe
_TMA_NUM_CONSUMER_WARPS    = 8
_TMA_NUM_STAGES            = 2                            # V1 starts at 2 (96 KB at 3 stages too tight)
_TMA_COLS_PER_STAGE        = _TMA_NUM_CONSUMER_WARPS * _TMA_COL_TILE   # 512
_PRODUCER_WARP_ID          = 0
_TOTAL_WARPS_PER_CTA       = _TMA_NUM_CONSUMER_WARPS + 1  # 9
_THREADS_PER_CTA           = 32 * _TOTAL_WARPS_PER_CTA    # 288
_BUFFER_ALIGN_BYTES        = 1024

# Consumer thread layout (matches CUDA TmaKernelTraitsTwoBytes)
_THREADS_PER_ROW_TMA       = 4                            # 4 threads per row, 8 rows / 4 = 8 row-iters / warp halved
_ROWS_PER_WARP_TMA         = 8                            # 32 lanes / 4 thr-per-row
_ELTS_PER_THREAD_TMA       = NVFP4_SF_VEC_SIZE            # 16

_ELEMS_PER_STAGE           = _TMA_ROW_TILE * _TMA_COLS_PER_STAGE   # 8192 = 16 KB / buffer / stage
_TOTAL_SMEM_ELEMS          = _ELEMS_PER_STAGE * _TMA_NUM_STAGES    # 16384 / buffer

# Path-1 K (Qwen3.5-35B-A3B): 4096 = num_sf_blocks_per_row * NVFP4_SF_VEC_SIZE
_PATH1_K_TOTAL             = 4096
_PATH1_NUM_COL_CHUNKS      = _PATH1_K_TOTAL // _TMA_COLS_PER_STAGE   # 8

ROW_TILE_SIZE = _TMA_ROW_TILE      # used by host wrapper to round padded_M


# ============================================================================
# SharedStorage struct factory — cute 4.4.2 evaluates `cute.struct.MemRange[T, N]`
# at class-body time only when the class is defined inside a function whose body
# runs at call time (matches flashinfer NVFP4QuantizeTMAKernel style: SharedStorage
# is defined inside __call__, not at module level). We pre-build it once via a
# module-level factory function and stash on a module-level attribute.
# ============================================================================

def _make_v1_tma_shared_storage_class():
    @cute.struct
    class _V1TmaSharedStorage:
        load_full_mbar:  cute.struct.MemRange[cutlass.Int64, _TMA_NUM_STAGES]
        load_empty_mbar: cute.struct.MemRange[cutlass.Int64, _TMA_NUM_STAGES]
        attn_smem: cute.struct.Align[
            cute.struct.MemRange[cutlass.BFloat16, _TOTAL_SMEM_ELEMS],
            _BUFFER_ALIGN_BYTES,
        ]
        gate_smem: cute.struct.Align[
            cute.struct.MemRange[cutlass.BFloat16, _TOTAL_SMEM_ELEMS],
            _BUFFER_ALIGN_BYTES,
        ]
    return _V1TmaSharedStorage


_V1TmaSharedStorage = _make_v1_tma_shared_storage_class()


# ============================================================================
# CuTeDSL fused kernel (sm_120a) — TMA G2S + warp-specialized 1 producer + 8 consumer
# ============================================================================

@cute.kernel
def fused_sigmoid_mul_nvfp4_kernel(
    tma_atom_attn: cute.CopyAtom,
    gAttn_tma: cute.Tensor,
    tma_atom_gate: cute.CopyAtom,
    gGate_tma: cute.Tensor,
    smem_outer_staged: cute.Layout,
    smem_swizzle: cute.Swizzle,
    smem_layout_flat: cute.Layout,
    tx_count_total: cutlass.Constexpr[int],
    mOutput: cute.Tensor,
    mScales: cute.Tensor,
    M: Int32,
    padded_M: Int32,
    mGlobalScale: cute.Tensor,
    num_sf_blocks_per_row: cutlass.Constexpr[int],
    padded_sf_cols: cutlass.Constexpr[int],
    num_col_chunks: cutlass.Constexpr[int],
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    grid_dim_x, _, _ = cute.arch.grid_dim()

    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane_idx = tidx % Int32(32)

    global_scale = Float32(mGlobalScale[Int32(0)])

    # ---- SMEM allocation (mbar before data, data 1024-B aligned) ----
    # _V1TmaSharedStorage referenced via module-level lexical scope.
    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(_V1TmaSharedStorage)

    load_mbar_ptr = storage.load_full_mbar.data_ptr()

    # Swizzled views for TMA destinations + flat views for consumer ld.shared
    sAttn_staged = storage.attn_smem.get_tensor(smem_outer_staged, swizzle=smem_swizzle)
    sGate_staged = storage.gate_smem.get_tensor(smem_outer_staged, swizzle=smem_swizzle)
    sAttn_flat   = storage.attn_smem.get_tensor(smem_layout_flat)
    sGate_flat   = storage.gate_smem.get_tensor(smem_layout_flat)

    # ---- Pipeline setup (single mbar carries BOTH attn + gate per stage) ----
    load_pipeline = pipeline.PipelineTmaAsync.create(
        barrier_storage=load_mbar_ptr,
        num_stages=_TMA_NUM_STAGES,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, _TMA_NUM_CONSUMER_WARPS),
        tx_count=tx_count_total,
        cta_layout_vmnk=cute.tiled_divide(cute.make_layout((1, 1, 1, 1)), (1,)),
        defer_sync=True,
    )
    pipeline_init_arrive(cluster_shape_mn=(1, 1), is_relaxed=True)
    pipeline_init_wait(cluster_shape_mn=(1, 1))

    # ---- TMA partitions for attn + gate (separate gSrc_tiled per buffer) ----
    gAttn_tiled = cute.local_tile(
        gAttn_tma,
        (_TMA_ROW_TILE, _TMA_NUM_CONSUMER_WARPS, _TMA_COL_TILE),
        (None, None, None),
    )
    gGate_tiled = cute.local_tile(
        gGate_tma,
        (_TMA_ROW_TILE, _TMA_NUM_CONSUMER_WARPS, _TMA_COL_TILE),
        (None, None, None),
    )
    tA_attn_s, tA_attn_g = cpasync.tma_partition(
        tma_atom_attn, 0, cute.make_layout(1),
        cute.group_modes(sAttn_staged, 0, 3),
        cute.group_modes(gAttn_tiled, 0, 3),
    )
    tA_gate_s, tA_gate_g = cpasync.tma_partition(
        tma_atom_gate, 0, cute.make_layout(1),
        cute.group_modes(sGate_staged, 0, 3),
        cute.group_modes(gGate_tiled, 0, 3),
    )

    num_row_tiles = cute.ceil_div(padded_M, _TMA_ROW_TILE)

    # Consumer thread indexing
    col_idx_local = lane_idx % Int32(_THREADS_PER_ROW_TMA)
    row_idx_local = lane_idx // Int32(_THREADS_PER_ROW_TMA)

    # ============ Producer warp (warp 0) ============
    if warp_idx == Int32(_PRODUCER_WARP_ID):
        prod_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, _TMA_NUM_STAGES,
        )
        row_tile_idx = bidx
        while row_tile_idx < num_row_tiles:
            col_chunk = Int32(0)
            while col_chunk < Int32(num_col_chunks):
                load_pipeline.producer_acquire(prod_state)
                bar = load_pipeline.producer_get_barrier(prod_state)
                # Issue BOTH TMA loads onto the SAME barrier; tx_count covers both.
                cute.copy(
                    tma_atom_attn,
                    tA_attn_g[(None, row_tile_idx, col_chunk, 0)],
                    tA_attn_s[(None, prod_state.index)],
                    tma_bar_ptr=bar,
                )
                cute.copy(
                    tma_atom_gate,
                    tA_gate_g[(None, row_tile_idx, col_chunk, 0)],
                    tA_gate_s[(None, prod_state.index)],
                    tma_bar_ptr=bar,
                )
                prod_state.advance()
                col_chunk = col_chunk + Int32(1)
            row_tile_idx = row_tile_idx + grid_dim_x
        load_pipeline.producer_tail(prod_state)

    # ============ Consumer warps (warps 1-8) ============
    if warp_idx > Int32(0):
        cons_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, _TMA_NUM_STAGES,
        )
        consumer_warp_idx = warp_idx - Int32(1)

        # SMEM 3D layout per stage: [rows=16, warps=8, cols=64], stride [64, 1024, 1]
        # warp tile base = warp * (16*64) = warp * 1024
        warp_tile_elems = _TMA_ROW_TILE * _TMA_COL_TILE      # 1024
        warp_tile_base = consumer_warp_idx * Int32(warp_tile_elems)
        f4_base = col_idx_local * Int32(2)                   # float4 base 0/2/4/6

        # Global SF column offset for this thread's slice (16 elts = 1 SF block)
        base_col_in_stage = (
            consumer_warp_idx * Int32(_TMA_COL_TILE)
            + col_idx_local * Int32(_ELTS_PER_THREAD_TMA)
        )

        row_tile_idx = bidx
        while row_tile_idx < num_row_tiles:
            base_row = row_tile_idx * Int32(_TMA_ROW_TILE)

            col_chunk = Int32(0)
            while col_chunk < Int32(num_col_chunks):
                load_pipeline.consumer_wait(cons_state)
                stage = cons_state.index
                stage_base = stage * Int32(_ELEMS_PER_STAGE)

                # ---- Read SMEM with SWIZZLE_128B addressing ----
                # Per-row XOR: float4_idx ^= row & 7
                # Physical elem offset in warp tile for (row, float4 f):
                #   row * 64 + (f ^ (row & 7)) * 8

                # Row iteration 0: row = row_idx_local (0..7)
                r0_xor = row_idx_local & Int32(7)
                r0_f4_0 = f4_base ^ r0_xor
                r0_f4_1 = (f4_base + Int32(1)) ^ r0_xor
                r0_row_base = (
                    stage_base + warp_tile_base
                    + row_idx_local * Int32(_TMA_COL_TILE)
                )
                attn_addr_0_r0 = get_smem_ptr_as_int32(
                    sAttn_flat, r0_row_base + r0_f4_0 * Int32(8))
                attn_addr_1_r0 = get_smem_ptr_as_int32(
                    sAttn_flat, r0_row_base + r0_f4_1 * Int32(8))
                a0, a1, a2, a3 = ld_shared_v4_u32(attn_addr_0_r0)
                a4, a5, a6, a7 = ld_shared_v4_u32(attn_addr_1_r0)
                gate_addr_0_r0 = get_smem_ptr_as_int32(
                    sGate_flat, r0_row_base + r0_f4_0 * Int32(8))
                gate_addr_1_r0 = get_smem_ptr_as_int32(
                    sGate_flat, r0_row_base + r0_f4_1 * Int32(8))
                g0, g1, g2, g3 = ld_shared_v4_u32(gate_addr_0_r0)
                g4, g5, g6, g7 = ld_shared_v4_u32(gate_addr_1_r0)

                # Row iteration 1: row = row_idx_local + 8 (8..15)
                r1_row = row_idx_local + Int32(_ROWS_PER_WARP_TMA)
                r1_xor = r1_row & Int32(7)
                r1_f4_0 = f4_base ^ r1_xor
                r1_f4_1 = (f4_base + Int32(1)) ^ r1_xor
                r1_row_base = (
                    stage_base + warp_tile_base
                    + r1_row * Int32(_TMA_COL_TILE)
                )
                attn_addr_0_r1 = get_smem_ptr_as_int32(
                    sAttn_flat, r1_row_base + r1_f4_0 * Int32(8))
                attn_addr_1_r1 = get_smem_ptr_as_int32(
                    sAttn_flat, r1_row_base + r1_f4_1 * Int32(8))
                ra0, ra1, ra2, ra3 = ld_shared_v4_u32(attn_addr_0_r1)
                ra4, ra5, ra6, ra7 = ld_shared_v4_u32(attn_addr_1_r1)
                gate_addr_0_r1 = get_smem_ptr_as_int32(
                    sGate_flat, r1_row_base + r1_f4_0 * Int32(8))
                gate_addr_1_r1 = get_smem_ptr_as_int32(
                    sGate_flat, r1_row_base + r1_f4_1 * Int32(8))
                rg0, rg1, rg2, rg3 = ld_shared_v4_u32(gate_addr_0_r1)
                rg4, rg5, rg6, rg7 = ld_shared_v4_u32(gate_addr_1_r1)

                # ---- Compute SF column for this thread, both row iters ----
                global_col_base = col_chunk * Int32(_TMA_COLS_PER_STAGE)
                sf_col = (global_col_base + base_col_in_stage) // Int32(NVFP4_SF_VEC_SIZE)

                # ---- Quantize row 0 ----
                global_row_0 = base_row + row_idx_local
                _quantize_block_inline(
                    a0, g0, a1, g1, a2, g2, a3, g3,
                    a4, g4, a5, g5, a6, g6, a7, g7,
                    global_row_0, sf_col, global_scale,
                    M, padded_M, padded_sf_cols,
                    mOutput, mScales,
                )

                # ---- Quantize row 1 ----
                global_row_1 = base_row + row_idx_local + Int32(_ROWS_PER_WARP_TMA)
                _quantize_block_inline(
                    ra0, rg0, ra1, rg1, ra2, rg2, ra3, rg3,
                    ra4, rg4, ra5, rg5, ra6, rg6, ra7, rg7,
                    global_row_1, sf_col, global_scale,
                    M, padded_M, padded_sf_cols,
                    mOutput, mScales,
                )

                load_pipeline.consumer_release(cons_state)
                cons_state.advance()
                col_chunk = col_chunk + Int32(1)

            # ---- Zero padding SF columns (swizzled-128x4 layout) ----
            # Only the first _TMA_ROW_TILE consumer threads cooperate on this
            # (one thread per row). Reuse warp_idx-1 + lane_idx as flat tid.
            consumer_tid = (warp_idx - Int32(1)) * Int32(32) + lane_idx
            if consumer_tid < Int32(_TMA_ROW_TILE):
                pad_row_idx = base_row + consumer_tid
                if pad_row_idx < padded_M:
                    padding_sf = Int32(num_sf_blocks_per_row)
                    while padding_sf < Int32(padded_sf_cols):
                        sf_offset = compute_sf_index_swizzled_128x4_gpu(
                            pad_row_idx, padding_sf, Int32(padded_sf_cols),
                        )
                        mScales[sf_offset] = Uint8(0)
                        padding_sf = padding_sf + Int32(1)

            row_tile_idx = row_tile_idx + grid_dim_x


@cute.jit
def _quantize_block_inline(
    a0: Uint32, g0: Uint32, a1: Uint32, g1: Uint32,
    a2: Uint32, g2: Uint32, a3: Uint32, g3: Uint32,
    a4: Uint32, g4: Uint32, a5: Uint32, g5: Uint32,
    a6: Uint32, g6: Uint32, a7: Uint32, g7: Uint32,
    global_row: Int32,
    sf_col: Int32,
    global_scale: Float32,
    M: Int32,
    padded_M: Int32,
    padded_sf_cols: Int32,
    mOutput: cute.Tensor,
    mScales: cute.Tensor,
):
    """Fused sigmoid_mul + NVFP4 quantization for one 16-elt SF block.

    Mirrors V0's epilogue body (bfloat2_sigmoid_mul x8, amax, e4m3 cvt,
    e2m1 packing, st_global_u64 + scalar SF byte) but is now called from
    consumer warps with operands sourced from SMEM.
    """
    if global_row < padded_M:
        is_padding_row = global_row >= M
        if is_padding_row:
            sf_offset = compute_sf_index_swizzled_128x4_gpu(
                global_row, sf_col, padded_sf_cols,
            )
            mScales[sf_offset] = Uint8(0)
        else:
            x0 = bfloat2_sigmoid_mul(a0, g0)
            x1 = bfloat2_sigmoid_mul(a1, g1)
            x2 = bfloat2_sigmoid_mul(a2, g2)
            x3 = bfloat2_sigmoid_mul(a3, g3)
            x4 = bfloat2_sigmoid_mul(a4, g4)
            x5 = bfloat2_sigmoid_mul(a5, g5)
            x6 = bfloat2_sigmoid_mul(a6, g6)
            x7 = bfloat2_sigmoid_mul(a7, g7)

            block_max_h2 = bfloat2_max_abs_8(x0, x1, x2, x3, x4, x5, x6, x7)
            block_max = bfloat2_hmax_reduce_to_f32(block_max_h2)

            fp4_max_rcp = rcp_approx_ftz(Float32(6.0))
            scale_float = global_scale * block_max * fp4_max_rcp
            scale_fp8_u32 = cvt_f32_to_e4m3(scale_float)
            scale_fp8 = Uint8(scale_fp8_u32 & Uint32(0xFF))

            output_scale = nvfp4_compute_output_scale(scale_fp8_u32, global_scale)
            packed64 = bfloat2x8_to_e2m1x16_packed(
                x0, x1, x2, x3, x4, x5, x6, x7, output_scale,
            )

            sf_offset = compute_sf_index_swizzled_128x4_gpu(
                global_row, sf_col, padded_sf_cols,
            )
            mScales[sf_offset] = scale_fp8

            row_output = mOutput[global_row, None]
            out_base = sf_col * Int32(NVFP4_SF_VEC_SIZE // 2)
            out_ptr = get_ptr_as_int64(row_output, out_base)
            st_global_u64(out_ptr, packed64)


@cute.jit
def fused_sigmoid_mul_nvfp4_launch(
    mAttnOut: cute.Tensor,
    mGate: cute.Tensor,
    mOutput: cute.Tensor,
    mScales: cute.Tensor,
    M: Int32,
    padded_M: Int32,
    num_blocks: Int32,
    mGlobalScale: cute.Tensor,
    num_sf_blocks_per_row: cutlass.Constexpr[int],
    padded_sf_cols: cutlass.Constexpr[int],
    num_col_chunks: cutlass.Constexpr[int],
):
    # 3D global tensor view: [padded_M, K/64, 64] so each warp's 64-col
    # contiguous stripe is the innermost dim, matching the TMA descriptor.
    gAttn = cute.make_tensor(
        mAttnOut.iterator,
        cute.make_layout(
            (M, _PATH1_K_TOTAL // _TMA_COL_TILE, _TMA_COL_TILE),
            stride=(_PATH1_K_TOTAL, _TMA_COL_TILE, 1),
        ),
    )
    gGate = cute.make_tensor(
        mGate.iterator,
        cute.make_layout(
            (M, _PATH1_K_TOTAL // _TMA_COL_TILE, _TMA_COL_TILE),
            stride=(_PATH1_K_TOTAL, _TMA_COL_TILE, 1),
        ),
    )

    # SMEM single-stage layout: [rows=16, warps=8, cols_per_warp=64] bf16
    # SWIZZLE_128B = make_swizzle(3, 4, 3) for 2-byte dtype
    smem_swizzle = cute.make_swizzle(3, 4, 3)
    smem_outer_single = cute.make_layout(
        (_TMA_ROW_TILE, _TMA_NUM_CONSUMER_WARPS, _TMA_COL_TILE),
        stride=(_TMA_COL_TILE, _TMA_ROW_TILE * _TMA_COL_TILE, 1),
    )
    smem_single_composed = cute.make_composed_layout(smem_swizzle, 0, smem_outer_single)
    cta_tiler = (_TMA_ROW_TILE, _TMA_NUM_CONSUMER_WARPS, _TMA_COL_TILE)

    tma_atom_attn, tma_tensor_attn = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(),
        gAttn,
        smem_single_composed,
        cta_tiler,
    )
    tma_atom_gate, tma_tensor_gate = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(),
        gGate,
        smem_single_composed,
        cta_tiler,
    )

    # Staged outer layout (no swizzle here — passed separately to get_tensor)
    smem_outer_staged = cute.make_layout(
        (_TMA_ROW_TILE, _TMA_NUM_CONSUMER_WARPS, _TMA_COL_TILE, _TMA_NUM_STAGES),
        stride=(
            _TMA_COL_TILE,
            _TMA_ROW_TILE * _TMA_COL_TILE,
            1,
            _ELEMS_PER_STAGE,
        ),
    )
    smem_layout_flat = cute.make_layout((_TOTAL_SMEM_ELEMS,))

    num_tma_load_bytes_per_buffer = cute.size_in_bytes(cutlass.BFloat16, smem_outer_single)
    # CRITICAL: tx_count covers BOTH attn + gate bytes — they share the same mbar.
    tx_count_total = num_tma_load_bytes_per_buffer * 2

    fused_sigmoid_mul_nvfp4_kernel(
        tma_atom_attn, tma_tensor_attn,
        tma_atom_gate, tma_tensor_gate,
        smem_outer_staged, smem_swizzle, smem_layout_flat,
        tx_count_total,
        mOutput, mScales,
        M, padded_M, mGlobalScale,
        num_sf_blocks_per_row, padded_sf_cols,
        num_col_chunks,
    ).launch(
        grid=(num_blocks, 1, 1),
        block=(_THREADS_PER_CTA, 1, 1),
        cluster=(1, 1, 1),
    )


# ============================================================================
# Compile cache + Python launch wrapper
# ============================================================================

_compiled_cache: dict = {}


def _round_up(x: int, d: int) -> int:
    return ((x + d - 1) // d) * d


def fused_sigmoid_mul_nvfp4(
    attn_out: torch.Tensor,
    gate: torch.Tensor,
    sf_scale: torch.Tensor,
):
    """Host-side wrapper.

    Args:
        attn_out: (M, D) bf16 cuda
        gate:     (M, D) bf16 cuda
        sf_scale: scalar fp32 == ``input_global_scale_inv`` (vllm naming)

    Returns:
        x_fp4 : (M, D//2) uint8
        x_bs  : (padded_M, padded_sf_cols) uint8 swizzled-128x4 layout
    """
    assert attn_out.is_cuda and gate.is_cuda
    assert attn_out.shape == gate.shape
    assert attn_out.dtype == torch.bfloat16
    M, D = attn_out.shape
    K = D
    assert K == _PATH1_K_TOTAL, (
        f"V1-TMA launcher specialised for K={_PATH1_K_TOTAL} "
        f"(Path-1 Qwen3.5-35B-A3B); got K={K}"
    )
    assert K % NVFP4_SF_VEC_SIZE == 0
    assert K % _TMA_COLS_PER_STAGE == 0, (
        f"K ({K}) must be multiple of TMA_COLS_PER_STAGE ({_TMA_COLS_PER_STAGE})"
    )

    num_sf_blocks_per_row = K // NVFP4_SF_VEC_SIZE
    padded_sf_cols = ((num_sf_blocks_per_row + 3) // 4) * 4
    padded_M = _round_up(M, ROW_TILE_SIZE)
    num_col_chunks = K // _TMA_COLS_PER_STAGE

    num_sm = torch.cuda.get_device_properties(attn_out.device).multi_processor_count
    # Persistent grid: NUM_SMs (110 on Pro5000), grid-stride row-tile loop.
    num_blocks = num_sm

    x_fp4 = torch.empty((M, D // 2), dtype=torch.uint8, device=attn_out.device)
    x_bs_u8 = torch.empty(
        (padded_M * padded_sf_cols,), dtype=torch.uint8, device=attn_out.device,
    )

    sf_scale_buf = (
        sf_scale.to(torch.float32).reshape(1).contiguous().to(attn_out.device)
    )

    mAttn = from_dlpack(attn_out.contiguous(), assumed_align=16)
    mGate = from_dlpack(gate.contiguous(), assumed_align=16)
    mOut  = from_dlpack(x_fp4, assumed_align=16)
    mScl  = from_dlpack(x_bs_u8, assumed_align=16)
    mGS   = from_dlpack(sf_scale_buf, assumed_align=4)

    cache_key = (K, num_sf_blocks_per_row, padded_sf_cols, num_col_chunks)
    compiled = _compiled_cache.get(cache_key)
    if compiled is None:
        compiled = cute.compile(
            fused_sigmoid_mul_nvfp4_launch,
            mAttn, mGate, mOut, mScl,
            Int32(M), Int32(padded_M), Int32(num_blocks),
            mGS,
            num_sf_blocks_per_row, padded_sf_cols, num_col_chunks,
        )
        _compiled_cache[cache_key] = compiled

    # compiled(...) only takes the DYNAMIC args (Constexpr params are baked
    # in at cute.compile time and dropped from the runtime signature).
    compiled(
        mAttn, mGate, mOut, mScl,
        Int32(M), Int32(padded_M), Int32(num_blocks),
        mGS,
    )

    x_bs = x_bs_u8.view(padded_M, padded_sf_cols)
    return x_fp4, x_bs


# ============================================================================
# attention forward
#
# V0  baseline:        torch SDPA causal (~1742 us on path-1, M=6144).
# V3 (hybrid, current): vllm.vllm_flash_attn.flash_attn_varlen_func (~99 us
#                       est. on remote sm_120, fast cuda path).
#
# V3 (hybrid) replaces only this function — `fused_sigmoid_mul_nvfp4`
# (V2-TMA) is unchanged. Together end-to-end ~99 + 103 = ~200 us, vs
# SDPA + V0 ~1845 us = ~9× wall-clock speedup at the path1_forward boundary.
#
# True V3 fusion (FA + sigmoid·gate + nvfp4 quant in single kernel)
# deferred — blocked on remote nvidia_cutlass_dsl 4.4.2 vs flash_attn cute
# expecting cutlass 4.5+ (see commit `4fa44ed` archive). Hybrid keeps
# attn_out 50 MB DRAM round-trip; real fusion would eliminate it.
# ============================================================================

def _flash_attention_sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                           causal: bool = True) -> torch.Tensor:
    """SDPA fallback (V0 baseline) — slow but always available."""
    N, H_q, D_h = q.shape
    _, H_kv, _ = k.shape
    assert H_q % H_kv == 0
    repeat = H_q // H_kv

    q_bhsd = q.transpose(0, 1).unsqueeze(0)
    k_bhsd = k.transpose(0, 1).unsqueeze(0)
    v_bhsd = v.transpose(0, 1).unsqueeze(0)
    if repeat > 1:
        k_bhsd = k_bhsd.repeat_interleave(repeat, dim=1)
        v_bhsd = v_bhsd.repeat_interleave(repeat, dim=1)

    out = F.scaled_dot_product_attention(
        q_bhsd, k_bhsd, v_bhsd,
        is_causal=causal,
        scale=1.0 / math.sqrt(D_h),
    )
    out = out.squeeze(0).transpose(0, 1).contiguous()
    return out.reshape(N, H_q * D_h)


_HAS_VLLM_FLASH_ATTN = None  # tri-state: None=not-tried, True/False=cached


def _try_vllm_flash_attn():
    """Lazy probe + cache for vllm.vllm_flash_attn availability."""
    global _HAS_VLLM_FLASH_ATTN
    if _HAS_VLLM_FLASH_ATTN is None:
        try:
            from vllm.vllm_flash_attn import flash_attn_varlen_func  # noqa: F401
            _HAS_VLLM_FLASH_ATTN = True
        except Exception:
            _HAS_VLLM_FLASH_ATTN = False
    return _HAS_VLLM_FLASH_ATTN


def _flash_attention_vllm(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                           causal: bool = True) -> torch.Tensor:
    """V3 hybrid producer: vllm.vllm_flash_attn.flash_attn_varlen_func.

    Single varlen sequence (whole N treated as one seq, matches Path-1 spec
    which is one full-attn batch of N=6144 tokens).
    """
    from vllm.vllm_flash_attn import flash_attn_varlen_func

    N, H_q, D_h = q.shape
    _, H_kv, _ = k.shape
    assert H_q % H_kv == 0  # vllm flash-attn handles GQA natively (no expand)

    cu_seqlens = torch.tensor([0, N], dtype=torch.int32, device=q.device)
    out = flash_attn_varlen_func(
        q, k, v,
        max_seqlen_q=N, cu_seqlens_q=cu_seqlens,
        max_seqlen_k=N, cu_seqlens_k=cu_seqlens,
        causal=causal,
        softmax_scale=1.0 / math.sqrt(D_h),
    )
    # out shape: (N, H_q, D_h); flatten H_q*D_h to match downstream contract
    return out.contiguous().reshape(N, H_q * D_h)


def flash_attention_bf16(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                          causal: bool = True) -> torch.Tensor:
    """Attention forward with GQA. (N, H_q, D_h) bf16 -> (N, H_q*D_h) bf16.

    Hybrid V3: prefers vllm.vllm_flash_attn (fast path, ~99 us on sm_120 RTX
    PRO 5000), falls back to SDPA (~1742 us, baseline) if vllm not installed.
    """
    if _try_vllm_flash_attn():
        return _flash_attention_vllm(q, k, v, causal=causal)
    return _flash_attention_sdpa(q, k, v, causal=causal)


# ============================================================================
# Top-level Path-1 forward
# ============================================================================

def path1_forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                   gate: torch.Tensor, sf_scale: torch.Tensor):
    attn_out = flash_attention_bf16(q, k, v, causal=True)
    x_fp4, x_bs = fused_sigmoid_mul_nvfp4(attn_out, gate, sf_scale)
    return attn_out, x_fp4, x_bs
