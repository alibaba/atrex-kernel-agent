# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# gpu-wiki archive note:
# Diagnostic omoExplore task39 fork of FlashInfer SM120 dense b12x CuTe DSL.
# It localizes scalar SF LDS / OMMA.SF bank-conflict behavior and records
# shallow variants that did not become deployable. Do not use as an upstream
# FlashInfer replacement without fresh correctness and nsys gates.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# This file is ported from the CUTLASS dense block-scaled GEMM example
# and adapted for the current Blackwell GeForce target.
#
# Ported from the b12x kernel library to FlashInfer.
#
# Task39 note:
# This is a local research fork of FlashInfer's SM120 dense b12x CuTe DSL
# kernel.  The initial revision is intentionally behavior-equivalent to the
# upstream file so the benchmark can first validate correctness/performance
# parity before making structural mainloop changes for Qwen3.5 gate/up prefill.

from typing import Callable, List, Optional, Tuple, Type, Union

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm120_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
import cutlass.utils.hopper_helpers as sm90_utils
import functools
import torch
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.nvgpu.warp.mma import Field as WarpField

from flashinfer.cute_dsl.utils import (
    cutlass_to_torch_dtype,
    get_cutlass_dtype,
    get_max_active_clusters,
    get_num_sm,
    make_ptr,
    sm120_make_smem_layout_sfa,
    sm120_make_smem_layout_sfb,
)
from flashinfer.cute_dsl.fp4_common import get_smem_ptr_as_int32


def current_cuda_stream():
    """Return current CUDA stream as a CUDA driver stream handle."""
    import cuda.bindings.driver as cuda_driver

    return cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


@dsl_user_op
def task39_ld_shared_v4_u32_sideeffect(
    smem_addr: cutlass.Int32, *, loc=None, ip=None
):
    """Side-effecting ld.shared.v4.u32 so ptxas cannot scalarize gap loads."""

    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
        [cutlass.Int32(smem_addr).ir_value(loc=loc, ip=ip)],
        "ld.shared.v4.u32 {$0,$1,$2,$3}, [$4];",
        "=r,=r,=r,=r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    v0 = llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)
    v1 = llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)
    v2 = llvm.extractvalue(T.i32(), result, [2], loc=loc, ip=ip)
    v3 = llvm.extractvalue(T.i32(), result, [3], loc=loc, ip=ip)
    return cutlass.Uint32(v0), cutlass.Uint32(v1), cutlass.Uint32(v2), cutlass.Uint32(v3)


@dsl_user_op
def task39_make_packed_sf_gmem_layout(
    sf_m: cutlass.Int64,
    sf_k: cutlass.Int64,
    l: cutlass.Constexpr,
    sf_vec_size: int,
    *,
    loc=None,
    ip=None,
) -> cute.Layout:
    """Packed-block SF GMEM layout: [MN/128, SF_K/4, 32, 4, 4, L]."""

    packed_block_elems = 32 * 4 * 4
    return cute.make_layout(
        ((32, 4, sf_m), (sf_vec_size, 4, sf_k), l),
        stride=(
            (16, 4, sf_k * packed_block_elems),
            (0, 1, packed_block_elems),
            sf_m * sf_k * packed_block_elems,
        ),
    )


@dsl_user_op
def task39_sm120_make_smem_layout_sfa_padded(
    tiled_mma: cute.TiledMma,
    tile_shape_mnk: cute.Tile,
    sf_vec_size: int,
    num_stages: int,
    mn_stride_pad: int,
    *,
    loc=None,
    ip=None,
) -> cute.Layout:
    """Task39 SF-A SMEM layout with a padded 128x4 SF block stride.

    The upstream SM120 b12x layout uses ``mn_basic_block_stride=(16, 4)`` for
    the 32x4 scale-factor chunk.  NCU source attribution for the Task39
    gate/up shape shows all excessive shared wavefronts on scalar 32-bit SF
    ``LDS`` instructions, not on A/B ``LDSM``.  This experimental layout keeps
    logical coordinates and TMA semantics unchanged while padding the
    scale-factor row stride to break the repeated 2-way bank pattern.
    """

    assert sf_vec_size == 16 or sf_vec_size == 32, "sf_vec_size must be 16 or 32"

    blk_mn = 128
    blk_sf = 4
    mma_nsf = tiled_mma.shape_mnk[2] // sf_vec_size

    mn_basic_block_shape = (32, 4)
    mn_basic_block_stride = (16 + mn_stride_pad, 4)
    raw_padded_blk_elems = (mn_basic_block_shape[0] - 1) * mn_basic_block_stride[0] + (
        mn_basic_block_shape[1] - 1
    ) * mn_basic_block_stride[1] + 1
    padded_blk_elems = _ceil_div(raw_padded_blk_elems, 16) * 16
    k_basic_block_shape = (sf_vec_size, mma_nsf)
    k_basic_block_stride = (0, 1)

    assert tile_shape_mnk[0] % (blk_mn // 2) == 0, (
        "tile_shape_mnk[0] must be divisible by 64"
    )

    sfa_tile_m = max(blk_mn, _ceil_div(tile_shape_mnk[0], blk_mn) * blk_mn)

    sSFA_shapeM = (mn_basic_block_shape, sfa_tile_m // blk_mn)
    sSF_strideM = (mn_basic_block_stride, padded_blk_elems)

    assert tile_shape_mnk[2] % (blk_sf * mma_nsf) == 0, (
        "tile_shape_mnk[2] must be divisible by blk_sf * mma_nsf"
    )
    assert tile_shape_mnk[2] % (sf_vec_size * blk_sf) == 0, (
        "tile_shape_mnk[2] must be divisible by sf_vec_size * blk_sf"
    )
    assert blk_sf % mma_nsf == 0, "blk_sf must be divisible by mma_nsf"

    sSFA_shapeK = (
        k_basic_block_shape,
        blk_sf // mma_nsf,
        tile_shape_mnk[2] // sf_vec_size // blk_sf,
    )
    sSF_strideK = (
        k_basic_block_stride,
        mma_nsf,
        sfa_tile_m // blk_mn * padded_blk_elems,
    )

    smem_layout = cute.make_layout(
        (sSFA_shapeM, sSFA_shapeK),
        stride=(sSF_strideM, sSF_strideK),
    )

    return cute.append(
        smem_layout,
        cute.make_layout(
            num_stages,
            stride=cute.cosize(cute.filter_zeros(smem_layout)),
        ),
    )


@dsl_user_op
def task39_sm120_make_smem_layout_sfb_padded(
    tiled_mma: cute.TiledMma,
    tile_shape_mnk: cute.Tile,
    sf_vec_size: int,
    num_stages: int,
    mn_stride_pad: int,
    *,
    loc=None,
    ip=None,
) -> cute.Layout:
    """Task39 SF-B SMEM layout with a padded 128x4 SF block stride."""

    assert sf_vec_size == 16 or sf_vec_size == 32, "sf_vec_size must be 16 or 32"

    blk_mn = 128
    blk_sf = 4
    mma_nsf = tiled_mma.shape_mnk[2] // sf_vec_size

    mn_basic_block_shape = (32, 4)
    mn_basic_block_stride = (16 + mn_stride_pad, 4)
    raw_padded_blk_elems = (mn_basic_block_shape[0] - 1) * mn_basic_block_stride[0] + (
        mn_basic_block_shape[1] - 1
    ) * mn_basic_block_stride[1] + 1
    padded_blk_elems = _ceil_div(raw_padded_blk_elems, 16) * 16
    k_basic_block_shape = (sf_vec_size, mma_nsf)
    k_basic_block_stride = (0, 1)

    assert tile_shape_mnk[1] % (blk_mn // 2) == 0, (
        "tile_shape_mnk[1] must be divisible by 64"
    )
    assert tile_shape_mnk[2] % sf_vec_size == 0, (
        "tile_shape_mnk[2] must be divisible by sf_vec_size"
    )

    sfb_tile_n = max(blk_mn, _ceil_div(tile_shape_mnk[1], blk_mn) * blk_mn)

    sSFB_shapeN = (mn_basic_block_shape, sfb_tile_n // blk_mn)
    sSF_strideN = (mn_basic_block_stride, padded_blk_elems)

    assert tile_shape_mnk[2] % (blk_sf * mma_nsf) == 0, (
        "tile_shape_mnk[2] must be divisible by blk_sf * mma_nsf"
    )
    assert tile_shape_mnk[2] % (sf_vec_size * blk_sf) == 0, (
        "tile_shape_mnk[2] must be divisible by sf_vec_size * blk_sf"
    )
    assert blk_sf % mma_nsf == 0, "blk_sf must be divisible by mma_nsf"

    sSFB_shapeK = (
        k_basic_block_shape,
        blk_sf // mma_nsf,
        tile_shape_mnk[2] // sf_vec_size // blk_sf,
    )
    sSF_strideK = (
        k_basic_block_stride,
        mma_nsf,
        sfb_tile_n // blk_mn * padded_blk_elems,
    )

    smem_layout = cute.make_layout(
        (sSFB_shapeN, sSFB_shapeK),
        stride=(sSF_strideN, sSF_strideK),
    )

    return cute.append(
        smem_layout,
        cute.make_layout(
            num_stages,
            stride=cute.cosize(cute.filter_zeros(smem_layout)),
        ),
    )


@dsl_user_op
def task39_sm120_make_smem_layout_sfa_bankmajor(
    tiled_mma: cute.TiledMma,
    tile_shape_mnk: cute.Tile,
    sf_vec_size: int,
    num_stages: int,
    *,
    loc=None,
    ip=None,
) -> cute.Layout:
    """SFA SMEM layout with the 32x4 SF block stored bank-major.

    The footprint remains exactly one 512-element 128x4 scale block.  Only the
    affine order inside the 32x4 row-group block changes from stride ``(16, 4)``
    to ``(4, 128)`` so neighboring row groups rotate by one 32-bit bank.
    """

    assert sf_vec_size == 16 or sf_vec_size == 32, "sf_vec_size must be 16 or 32"

    blk_mn = 128
    blk_sf = 4
    blk_elems = blk_mn * blk_sf
    mma_nsf = tiled_mma.shape_mnk[2] // sf_vec_size

    mn_basic_block_shape = (32, 4)
    mn_basic_block_stride = (4, 128)
    k_basic_block_shape = (sf_vec_size, mma_nsf)
    k_basic_block_stride = (0, 1)

    assert tile_shape_mnk[0] % blk_mn == 0, (
        "tile_shape_mnk[0] must be divisible by blk_mn"
    )

    sSFA_shapeM = (mn_basic_block_shape, tile_shape_mnk[0] // blk_mn)
    sSF_strideM = (mn_basic_block_stride, blk_elems)

    assert tile_shape_mnk[2] % (blk_sf * mma_nsf) == 0, (
        "tile_shape_mnk[2] must be divisible by blk_sf * mma_nsf"
    )

    sSFA_shapeK = (
        k_basic_block_shape,
        blk_sf // mma_nsf,
        tile_shape_mnk[2] // sf_vec_size // blk_sf,
    )
    sSF_strideK = (
        k_basic_block_stride,
        mma_nsf,
        tile_shape_mnk[0] // blk_mn * blk_elems,
    )

    smem_layout = cute.make_layout(
        (sSFA_shapeM, sSFA_shapeK),
        stride=(sSF_strideM, sSF_strideK),
    )

    return cute.append(
        smem_layout,
        cute.make_layout(
            num_stages,
            stride=cute.cosize(cute.filter_zeros(smem_layout)),
        ),
    )


@dsl_user_op
def task39_sm120_make_smem_layout_sfb_bankmajor(
    tiled_mma: cute.TiledMma,
    tile_shape_mnk: cute.Tile,
    sf_vec_size: int,
    num_stages: int,
    *,
    loc=None,
    ip=None,
) -> cute.Layout:
    """SFB SMEM layout with the 32x4 SF block stored bank-major."""

    assert sf_vec_size == 16 or sf_vec_size == 32, "sf_vec_size must be 16 or 32"

    blk_mn = 128
    blk_sf = 4
    blk_elems = blk_mn * blk_sf
    mma_nsf = tiled_mma.shape_mnk[2] // sf_vec_size

    mn_basic_block_shape = (32, 4)
    mn_basic_block_stride = (4, 128)
    k_basic_block_shape = (sf_vec_size, mma_nsf)
    k_basic_block_stride = (0, 1)

    assert tile_shape_mnk[1] % blk_mn == 0, (
        "tile_shape_mnk[1] must be divisible by blk_mn"
    )
    assert tile_shape_mnk[2] % sf_vec_size == 0, (
        "tile_shape_mnk[2] must be divisible by sf_vec_size"
    )

    sSFB_shapeN = (mn_basic_block_shape, tile_shape_mnk[1] // blk_mn)
    sSF_strideN = (mn_basic_block_stride, blk_elems)

    assert tile_shape_mnk[2] % (blk_sf * mma_nsf) == 0, (
        "tile_shape_mnk[2] must be divisible by blk_sf * mma_nsf"
    )

    sSFB_shapeK = (
        k_basic_block_shape,
        blk_sf // mma_nsf,
        tile_shape_mnk[2] // sf_vec_size // blk_sf,
    )
    sSF_strideK = (
        k_basic_block_stride,
        mma_nsf,
        tile_shape_mnk[1] // blk_mn * blk_elems,
    )

    smem_layout = cute.make_layout(
        (sSFB_shapeN, sSFB_shapeK),
        stride=(sSF_strideN, sSF_strideK),
    )

    return cute.append(
        smem_layout,
        cute.make_layout(
            num_stages,
            stride=cute.cosize(cute.filter_zeros(smem_layout)),
        ),
    )


@dsl_user_op
def task39_sm120_make_smem_layout_sfa_fragmajor(
    tiled_mma: cute.TiledMma,
    tile_shape_mnk: cute.Tile,
    sf_vec_size: int,
    num_stages: int,
    *,
    loc=None,
    ip=None,
) -> cute.Layout:
    """SFA SMEM layout matching the inline/vector scale-load fragment order.

    The stock layout stores one 128x4 scale block as [32 row groups, 4 row
    lanes, 4 k scales].  The scalar LDS path then hits a deterministic 2-way
    bank pattern.  This layout stores the same logical tile as
    [16 scale lanes, 8 m-atoms, 4 k scales], matching the task23 inline OMMA
    layout and the standalone CUDA fragment repack path.
    """

    assert sf_vec_size == 16 or sf_vec_size == 32, "sf_vec_size must be 16 or 32"

    blk_mn = 128
    blk_sf = 4
    blk_elems = blk_mn * blk_sf
    mma_nsf = tiled_mma.shape_mnk[2] // sf_vec_size

    mn_basic_block_shape = (16, 8)
    mn_basic_block_stride = (32, 4)
    k_basic_block_shape = (sf_vec_size, mma_nsf)
    k_basic_block_stride = (0, 1)

    assert tile_shape_mnk[0] % (blk_mn // 2) == 0, (
        "tile_shape_mnk[0] must be divisible by 64"
    )

    sfa_tile_m = max(blk_mn, _ceil_div(tile_shape_mnk[0], blk_mn) * blk_mn)

    sSFA_shapeM = (mn_basic_block_shape, sfa_tile_m // blk_mn)
    sSF_strideM = (mn_basic_block_stride, blk_elems)

    assert tile_shape_mnk[2] % (blk_sf * mma_nsf) == 0, (
        "tile_shape_mnk[2] must be divisible by blk_sf * mma_nsf"
    )
    assert tile_shape_mnk[2] % (sf_vec_size * blk_sf) == 0, (
        "tile_shape_mnk[2] must be divisible by sf_vec_size * blk_sf"
    )
    assert blk_sf % mma_nsf == 0, "blk_sf must be divisible by mma_nsf"

    sSFA_shapeK = (
        k_basic_block_shape,
        blk_sf // mma_nsf,
        tile_shape_mnk[2] // sf_vec_size // blk_sf,
    )
    sSF_strideK = (
        k_basic_block_stride,
        mma_nsf,
        sfa_tile_m // blk_mn * blk_elems,
    )

    smem_layout = cute.make_layout(
        (sSFA_shapeM, sSFA_shapeK),
        stride=(sSF_strideM, sSF_strideK),
    )

    return cute.append(
        smem_layout,
        cute.make_layout(
            num_stages,
            stride=cute.cosize(cute.filter_zeros(smem_layout)),
        ),
    )


@dsl_user_op
def task39_sm120_make_smem_layout_sfb_fragmajor(
    tiled_mma: cute.TiledMma,
    tile_shape_mnk: cute.Tile,
    sf_vec_size: int,
    num_stages: int,
    *,
    loc=None,
    ip=None,
) -> cute.Layout:
    """SFB SMEM layout matching the inline/vector scale-load fragment order."""

    assert sf_vec_size == 16 or sf_vec_size == 32, "sf_vec_size must be 16 or 32"

    blk_mn = 128
    blk_sf = 4
    blk_elems = blk_mn * blk_sf
    mma_nsf = tiled_mma.shape_mnk[2] // sf_vec_size

    mn_basic_block_shape = (8, 16)
    mn_basic_block_stride = (64, 4)
    k_basic_block_shape = (sf_vec_size, mma_nsf)
    k_basic_block_stride = (0, 1)

    assert tile_shape_mnk[1] % (blk_mn // 2) == 0, (
        "tile_shape_mnk[1] must be divisible by 64"
    )
    assert tile_shape_mnk[2] % sf_vec_size == 0, (
        "tile_shape_mnk[2] must be divisible by sf_vec_size"
    )

    sfb_tile_n = max(blk_mn, _ceil_div(tile_shape_mnk[1], blk_mn) * blk_mn)

    sSFB_shapeN = (mn_basic_block_shape, sfb_tile_n // blk_mn)
    sSF_strideN = (mn_basic_block_stride, blk_elems)

    assert tile_shape_mnk[2] % (blk_sf * mma_nsf) == 0, (
        "tile_shape_mnk[2] must be divisible by blk_sf * mma_nsf"
    )
    assert tile_shape_mnk[2] % (sf_vec_size * blk_sf) == 0, (
        "tile_shape_mnk[2] must be divisible by sf_vec_size * blk_sf"
    )
    assert blk_sf % mma_nsf == 0, "blk_sf must be divisible by mma_nsf"

    sSFB_shapeK = (
        k_basic_block_shape,
        blk_sf // mma_nsf,
        tile_shape_mnk[2] // sf_vec_size // blk_sf,
    )
    sSF_strideK = (
        k_basic_block_stride,
        mma_nsf,
        sfb_tile_n // blk_mn * blk_elems,
    )

    smem_layout = cute.make_layout(
        (sSFB_shapeN, sSFB_shapeK),
        stride=(sSF_strideN, sSF_strideK),
    )

    return cute.append(
        smem_layout,
        cute.make_layout(
            num_stages,
            stride=cute.cosize(cute.filter_zeros(smem_layout)),
        ),
    )


class DenseGemmKernel:
    """Implements batched matrix multiplication (C = A x SFA x B x SFB) for
    Blackwell GeForce architecture using warp-level MMA.

    Key architectural differences from the tcgen05 donor path:
    - No TMEM, no tcgen05, no 2-CTA instructions, no multi-cluster
    - Warp-level MMA: MmaMXF4NVF4Op atom m16n8k64, atom_layout=(4,2,1)
    - 256 MMA threads + 32 DMA = 288 total threads
    - PipelineTmaAsync (not PipelineTmaUmma)
    - Manual atom unroll workaround for CuTe DSL compiler SF address space bug
    - Cluster shape always (1,1,1)

    Notes:
        - Supported combinations:
            * NVF4 only: A/B: Float4E2M1FN, SF: Float8E4M3FN, sf_vec_size: 16
            (MXF4 / sf_vec_size=32 is not supported — the CUTLASS DSL
            MmaMXF4NVF4Op hardcodes sf_vec_size=16 in its constructor.)
        - Tile shape constraints:
            * tile_m must be divisible by 64
            * tile_n must be divisible by 64
            * tile_k = sf_vec_size * 8 = 128
    """

    def __init__(
        self,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        use_prefetch: bool = False,
        enable_pdl: bool = True,
    ):
        self.acc_dtype = cutlass.Float32
        self.sf_vec_size = sf_vec_size
        # K = sf_vec_size * 8 for FP4 (each FP4 element is 0.5 bytes, sf_vec_size
        # elements per scale factor, and we want 4 MMA k-tiles per stage)
        tile_k = sf_vec_size * 8  # 128 for sf_vec_size=16
        self.tile_shape_mnk = (mma_tiler_mn[0], mma_tiler_mn[1], tile_k)
        self.sfa_tile_shape_mk = (max(128, mma_tiler_mn[0]), tile_k)
        self.sfa_tiles_per_block = self.sfa_tile_shape_mk[0] // mma_tiler_mn[0]
        self.sfb_tile_shape_nk = (max(128, mma_tiler_mn[1]), tile_k)
        self.sfb_tiles_per_block = self.sfb_tile_shape_nk[0] // mma_tiler_mn[1]
        self.cluster_shape_mnk = (1, 1, 1)  # Always (1,1,1) on the current target
        self.epi_tile = (mma_tiler_mn[0], mma_tiler_mn[1])
        self.use_prefetch = use_prefetch
        self.enable_pdl = enable_pdl
        self.use_unfiltered_sf_copy = False
        self.use_fixed_single_sf_tile = False
        self.assume_alpha_one_identity_epilogue = False
        self.use_padded_sf_smem_layout = False
        self.sf_smem_mn_stride_pad = 0
        self.use_bankmajor_sf_smem_layout = False
        self.use_fragmajor_sf_smem_layout = False
        self.use_swizzled_sf_smem_layout = False
        self.sf_smem_swizzle_bits = 0
        self.copy_sf_before_ab = False
        self.swap_sfa_tv_layout = False
        self.swap_sfb_tv_layout = False
        self.use_vectorized_sf_copy_u32 = False
        self.sf_copy_vector_bits = 0
        self.sf_num_bits_per_copy = 0
        self.use_manual_sfa_gap2_v4_copy = False
        self.use_manual_sfb_v4_copy = False
        self.use_manual_sfb_v4_fixed_copy = False
        self.manual_sfb_v4_fixed_perm = 0
        self.swap_sfb_atom_layout = False
        self.swap_sfb_fragment_grouping = False
        self.wait_after_last_mma = False
        self.preload_all_k_blocks = False
        self.preload_wait_before_last_mma = False
        self.preload_sf_k_blocks = False
        self.pretry_wait_after_last_mma = False
        self.hoist_sfa_set = False
        self.use_packed_sf_tma_layout = False

        self.tiled_mma = None
        self.occupancy = 1
        self.num_mma_warps = 8
        self.tma_load_warp_id = self.num_mma_warps
        self.num_threads_per_warp = 32
        self.threads_per_cta = (
            self.num_mma_warps + 1  # 1 warp for DMA
        ) * self.num_threads_per_warp

        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_120")

        self.ab_stage = None
        self.epi_stage = None
        self.a_smem_layout_staged = None
        self.b_smem_layout_staged = None
        self.epi_smem_layout_staged = None

        self.buffer_align_bytes = 1024

        self.mma_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=self.num_mma_warps * self.num_threads_per_warp,
        )
        self.epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=self.num_mma_warps * self.num_threads_per_warp,
        )
        self.load_register_requirement = 40
        self.mma_register_requirement = 232

    def _setup_attributes(self):
        mma_op = cute.nvgpu.warp.MmaMXF4NVF4Op(
            self.a_dtype,
            self.acc_dtype,
            self.sf_dtype,
        )
        atom_shape = (4, 2, 1)
        atom_layout = cute.make_layout(atom_shape)
        permutation_mnk = sm120_utils.get_permutation_mnk(
            self.tile_shape_mnk, self.sf_vec_size, False
        )
        self.tiled_mma = cute.make_tiled_mma(
            mma_op,
            atom_layout,
            permutation_mnk=permutation_mnk,
        )
        # Bare atom for manual unroll workaround (avoids hasAuxTensor address space bug)
        self.mma_atom = cute.make_mma_atom(mma_op)
        # Compute atom loop bounds from tile shape and atom/layout shape
        # MMA atom: m16, n8, k64; atom_layout: (4,2,1) -> group: m64, n16, k64
        mma_m, mma_n, mma_k = 16, 8, 64
        self.num_m_tiles = self.tile_shape_mnk[0] // (mma_m * atom_shape[0])
        self.num_n_tiles = self.tile_shape_mnk[1] // (mma_n * atom_shape[1])
        self.num_k_blocks = self.tile_shape_mnk[2] // mma_k

        self.cta_layout_mnk = cute.make_layout(self.cluster_shape_mnk)

        # Compute the smem size of SFA/SFB
        if self.use_padded_sf_smem_layout:
            sfa_smem_layout_per_stage = task39_sm120_make_smem_layout_sfa_padded(
                self.tiled_mma,
                self.tile_shape_mnk,
                self.sf_vec_size,
                1,
                self.sf_smem_mn_stride_pad,
            )
            sfb_smem_layout_per_stage = task39_sm120_make_smem_layout_sfb_padded(
                self.tiled_mma,
                self.tile_shape_mnk,
                self.sf_vec_size,
                1,
                self.sf_smem_mn_stride_pad,
            )
        elif self.use_bankmajor_sf_smem_layout:
            sfa_smem_layout_per_stage = task39_sm120_make_smem_layout_sfa_bankmajor(
                self.tiled_mma,
                self.tile_shape_mnk,
                self.sf_vec_size,
                1,
            )
            sfb_smem_layout_per_stage = task39_sm120_make_smem_layout_sfb_bankmajor(
                self.tiled_mma,
                self.tile_shape_mnk,
                self.sf_vec_size,
                1,
            )
        elif self.use_fragmajor_sf_smem_layout:
            sfa_smem_layout_per_stage = task39_sm120_make_smem_layout_sfa_fragmajor(
                self.tiled_mma,
                self.tile_shape_mnk,
                self.sf_vec_size,
                1,
            )
            sfb_smem_layout_per_stage = task39_sm120_make_smem_layout_sfb_fragmajor(
                self.tiled_mma,
                self.tile_shape_mnk,
                self.sf_vec_size,
                1,
            )
        else:
            # For swizzle-only variants, stage count is the same as the
            # upstream physical footprint, so compute stages on the base
            # layout and wrap the staged layout later.
            sfa_smem_layout_per_stage = sm120_make_smem_layout_sfa(
                self.tiled_mma,
                self.tile_shape_mnk,
                self.sf_vec_size,
                1,
            )
            sfb_smem_layout_per_stage = sm120_make_smem_layout_sfb(
                self.tiled_mma,
                self.tile_shape_mnk,
                self.sf_vec_size,
                1,
            )

        # Compute stage before compute smem layout
        self.ab_stage, self.epi_stage = self._compute_stages(
            self.tile_shape_mnk,
            self.a_dtype,
            self.b_dtype,
            self.sf_dtype,
            sfa_smem_layout_per_stage,
            sfb_smem_layout_per_stage,
            self.epi_tile,
            self.c_dtype,
            self.smem_capacity,
            self.occupancy,
        )

        assert self.epi_stage > 0, (
            "epi_stage <= 0, not enough shared memory. This configuration will be skipped."
        )

        (
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.epi_smem_layout_staged,
        ) = self._make_smem_layouts(
            self.tile_shape_mnk,
            self.epi_tile,
            self.a_dtype,
            self.a_layout,
            self.b_dtype,
            self.b_layout,
            self.ab_stage,
            self.c_dtype,
            self.c_layout,
            self.epi_stage,
            self.sf_vec_size,
            self.tiled_mma,
            self.use_padded_sf_smem_layout,
            self.sf_smem_mn_stride_pad,
            self.use_bankmajor_sf_smem_layout,
            self.use_fragmajor_sf_smem_layout,
            self.use_swizzled_sf_smem_layout,
            self.sf_smem_swizzle_bits,
        )

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        sfa: cute.Tensor,
        sfb: cute.Tensor,
        c: cute.Tensor,
        alpha: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
        epilogue_op: cutlass.Constexpr = lambda x: x,
    ):
        """Execute the GEMM operation.

        Args:
            a: Input tensor A
            b: Input tensor B
            sfa: Scale factor tensor for A
            sfb: Scale factor tensor for B
            c: Output tensor C
            alpha: Alpha scaling factor tensor, shape (1,), float32
            max_active_clusters: Max active clusters
            stream: CUDA stream
            epilogue_op: Elementwise epilogue function
        """
        # Setup static attributes
        self.a_dtype = a.element_type
        self.b_dtype = b.element_type
        self.c_dtype = c.element_type
        self.sf_dtype = sfa.element_type

        self.a_layout = utils.LayoutEnum.from_tensor(a)
        self.b_layout = utils.LayoutEnum.from_tensor(b)
        self.c_layout = utils.LayoutEnum.from_tensor(c)

        if cutlass.const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(f"Type mismatch: {self.a_dtype} != {self.b_dtype}")

        self._setup_attributes()

        # Setup sfa/sfb tensor by filling A/B tensor to scale factor atom layout.
        # The packed-SF experiment passes an already-packed GMEM layout from the
        # wrapper so TMA sees [MN/128, SF_K/4, 32, 4, 4] contiguous blocks.
        if cutlass.const_expr(self.use_packed_sf_tma_layout):
            sfa_tensor = sfa
            sfb_tensor = sfb
        else:
            self.sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(
                a.shape, self.sf_vec_size
            )
            sfa_tensor = cute.make_tensor(sfa.iterator, self.sfa_layout)

            self.sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(
                b.shape, self.sf_vec_size
            )
            sfb_tensor = cute.make_tensor(sfb.iterator, self.sfb_layout)

        tma_atom_a, tma_tensor_a = self._make_tma_atoms_and_tensors(
            a,
            self.a_smem_layout_staged,
            (self.tile_shape_mnk[0], self.tile_shape_mnk[2]),
            1,
        )
        tma_atom_b, tma_tensor_b = self._make_tma_atoms_and_tensors(
            b,
            self.b_smem_layout_staged,
            (self.tile_shape_mnk[1], self.tile_shape_mnk[2]),
            1,
        )
        tma_atom_sfa, tma_tensor_sfa = self._make_tma_atoms_and_tensors(
            sfa_tensor,
            self.sfa_smem_layout_staged,
            self.sfa_tile_shape_mk,
            1,
            internal_type=cutlass.Int16,
        )
        tma_atom_sfb, tma_tensor_sfb = self._make_tma_atoms_and_tensors(
            sfb_tensor,
            self.sfb_smem_layout_staged,
            self.sfb_tile_shape_nk,
            1,
            internal_type=cutlass.Int16,
        )
        tma_atom_c, tma_tensor_c = self._make_tma_store_atoms_and_tensors(
            c,
            self.epi_smem_layout_staged,
            self.epi_tile,
        )

        tile_sched_params, grid = self._compute_grid(
            c,
            self.tile_shape_mnk,
            max_active_clusters,
        )

        @cute.struct
        class SharedStorage:
            mainloop_pipeline_array_ptr: cute.struct.MemRange[
                cutlass.Int64, self.ab_stage * 2
            ]
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    self.a_dtype, cute.cosize(self.a_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    self.b_dtype, cute.cosize(self.b_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sSFA: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(self.sfa_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sSFB: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(self.sfb_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sC: cute.struct.Align[
                cute.struct.MemRange[
                    self.c_dtype, cute.cosize(self.epi_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        self.kernel(
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_sfa,
            tma_tensor_sfa,
            tma_atom_sfb,
            tma_tensor_sfb,
            tma_atom_c,
            tma_tensor_c,
            self.tiled_mma,
            self.mma_atom,
            self.cta_layout_mnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.epi_smem_layout_staged,
            tile_sched_params,
            epilogue_op,
            alpha,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=[1, 1, 1],
            stream=stream,
        )
        return

    def _partition_fragment_SFA(
        self,
        sfa_tensor: cute.Tensor,
        thr_mma: cute.ThrMma,
        tidx: int,
    ):
        thrfrg_sfa_layout = self._thrfrg_SFA(sfa_tensor.layout, thr_mma)
        thr_tensor = cute.make_tensor(sfa_tensor.iterator, thrfrg_sfa_layout)
        thr_vmnk = thr_mma.thr_layout_vmnk.get_flat_coord(tidx)
        thr_vmk = (thr_vmnk[0], (thr_vmnk[1], thr_vmnk[3]))
        partitioned_sfa = thr_tensor[thr_vmk, (None, None)]
        partitioned_sfa = cute.group_modes(cute.flatten(partitioned_sfa), 0, 2)
        return cute.make_fragment_like(partitioned_sfa)

    def _partition_fragment_SFB(
        self,
        sfb_tensor: cute.Tensor,
        thr_mma: cute.ThrMma,
        tidx: int,
    ):
        thrfrg_sfb_layout = self._thrfrg_SFB(sfb_tensor.layout, thr_mma)
        thr_tensor = cute.make_tensor(sfb_tensor.iterator, thrfrg_sfb_layout)
        thr_vmnk = thr_mma.thr_layout_vmnk.get_flat_coord(tidx)
        thr_vnk = (thr_vmnk[0], (thr_vmnk[2], thr_vmnk[3]))
        partitioned_sfb = thr_tensor[thr_vnk, (None, None)]
        if cutlass.const_expr(self.swap_sfb_fragment_grouping):
            partitioned_sfb = cute.group_modes(cute.flatten(partitioned_sfb), 1, 3)
            partitioned_sfb = cute.group_modes(partitioned_sfb, 0, 2)
        else:
            partitioned_sfb = cute.group_modes(cute.flatten(partitioned_sfb), 0, 2)
            partitioned_sfb = cute.group_modes(partitioned_sfb, 1, 3)
        return cute.make_fragment_like(partitioned_sfb)

    def _thrfrg_SFA(self, sfa_tensor, tiled_mma: cute.TiledMma):
        assert cute.rank(sfa_tensor) >= 2

        atom_shape_mnk = tiled_mma.shape_mnk
        atom_sfa_layout = cute.make_layout(
            shape=((2, 2, 8), 64), stride=((8, 0, 1), 16)
        )
        permutation_mnk = tiled_mma.permutation_mnk
        thr_layout_vmnk = tiled_mma.thr_layout_vmnk

        # Reorder the tensor for TiledAtom
        t_tile = (permutation_mnk[0], permutation_mnk[2])
        t_tensor = cute.logical_divide(sfa_tensor, t_tile)

        # Tile the tensor for the Atom
        a_tile = (
            cute.make_layout((atom_shape_mnk[0])),
            cute.make_layout((atom_shape_mnk[2])),
        )
        a_tensor = cute.zipped_divide(t_tensor, a_tile)

        # Transform the Atom mode from (M,K) to (Thr,Val)
        tv_tensor = cute.composition(a_tensor, (atom_sfa_layout, None))

        # Tile the tensor for the Thread
        thr_tile = (
            None,
            (
                cute.make_layout(cute.size(thr_layout_vmnk[1])),
                cute.make_layout(cute.size(thr_layout_vmnk[3])),
            ),
        )
        thr_tensor = cute.zipped_divide(tv_tensor, thr_tile)
        return thr_tensor

    def _thrfrg_SFB(self, sfb_tensor, tiled_mma: cute.TiledMma):
        assert cute.rank(sfb_tensor) >= 2

        atom_shape_mnk = tiled_mma.shape_mnk
        if cutlass.const_expr(self.swap_sfb_atom_layout):
            atom_sfb_layout = cute.make_layout(
                shape=((4, 8), 64), stride=((1, 0), 8)
            )
        else:
            atom_sfb_layout = cute.make_layout(
                shape=((4, 8), 64), stride=((0, 1), 8)
            )
        permutation_mnk = tiled_mma.permutation_mnk
        thr_layout_vmnk = tiled_mma.thr_layout_vmnk

        # Reorder the tensor for TiledAtom
        t_tile = (permutation_mnk[1], permutation_mnk[2])
        t_tensor = cute.logical_divide(sfb_tensor, t_tile)

        # Tile the tensor for the Atom
        a_tile = (
            cute.make_layout((atom_shape_mnk[1])),
            cute.make_layout((atom_shape_mnk[2])),
        )
        a_tensor = cute.zipped_divide(t_tensor, a_tile)

        # Transform the Atom mode from (N,K) to (Thr,Val)
        tv_tensor = cute.composition(a_tensor, (atom_sfb_layout, None))

        # Tile the tensor for the Thread
        thr_tile = (
            None,
            (
                cute.make_layout(cute.size(thr_layout_vmnk[2])),
                cute.make_layout(cute.size(thr_layout_vmnk[3])),
            ),
        )
        thr_tensor = cute.zipped_divide(tv_tensor, thr_tile)
        return thr_tensor

    def _get_layoutSFA_TV(self, tiled_mma: cute.TiledMma):
        if tiled_mma.permutation_mnk is not None:
            perm_m = tiled_mma.permutation_mnk[0]
            perm_k = tiled_mma.permutation_mnk[2]
            tile_m = cute.size(perm_m)
            tile_k = cute.size(perm_k)
        else:
            tile_shape_mnk = tiled_mma.shape_mnk * tiled_mma.thr_layout_vmnk
            tile_m = cute.size(tile_shape_mnk[0])
            tile_k = cute.size(tile_shape_mnk[2])

        ref_A = cute.make_layout((tile_m, tile_k))
        thr_layout_vmnk = tiled_mma.thr_layout_vmnk

        sfa_tv_stride = (0, 1) if self.swap_sfa_tv_layout else (1, 0)
        atile = (
            None,
            (
                cute.make_layout(
                    shape=(
                        cute.size(thr_layout_vmnk[1]),
                        cute.size(thr_layout_vmnk[2]),
                    ),
                    stride=sfa_tv_stride,
                ),
                None,
            ),
        )

        thridx_2_thrid = cute.right_inverse(thr_layout_vmnk)
        thrfrg_sfa = self._thrfrg_SFA(ref_A, tiled_mma)
        layout_tv_1 = cute.composition(thrfrg_sfa, (atile, None))
        layout_tv = cute.composition(layout_tv_1, (thridx_2_thrid, None))
        return layout_tv

    def _get_layoutSFB_TV(self, tiled_mma: cute.TiledMma):
        if tiled_mma.permutation_mnk is not None:
            perm_n_layout = tiled_mma.permutation_mnk[1]
            perm_k = tiled_mma.permutation_mnk[2]
            tile_n = cute.size(perm_n_layout)
            tile_k = cute.size(perm_k)
        else:
            tile_shape_mnk = tiled_mma.shape_mnk * tiled_mma.thr_layout_vmnk
            tile_n = cute.size(tile_shape_mnk[1])
            tile_k = cute.size(tile_shape_mnk[2])

        ref_B = cute.make_layout((tile_n, tile_k))
        thr_layout_vmnk = tiled_mma.thr_layout_vmnk

        sfb_tv_stride = (1, 0) if self.swap_sfb_tv_layout else (0, 1)
        atile = (
            None,
            (
                cute.make_layout(
                    shape=(
                        cute.size(thr_layout_vmnk[1]),
                        cute.size(thr_layout_vmnk[2]),
                    ),
                    stride=sfb_tv_stride,
                ),
                None,
            ),
        )

        thridx_2_thrid = cute.right_inverse(thr_layout_vmnk)
        thrfrg_sfb = self._thrfrg_SFB(ref_B, tiled_mma)
        layout_tv = cute.composition(thrfrg_sfb, (atile, None))
        layout_tv = cute.composition(layout_tv, (thridx_2_thrid, None))
        return layout_tv

    def _copy_sfa_smem_to_fragment(
        self,
        smem_tiled_copy_SFA,
        src: cute.Tensor,
        dst: cute.Tensor,
    ):
        if cutlass.const_expr(self.use_manual_sfa_gap2_v4_copy):
            src_flat = cute.flatten(src)
            dst_flat = cute.flatten(dst)
            base_addr = get_smem_ptr_as_int32(src_flat, cutlass.Int32(0))
            v0, _, v2, _ = task39_ld_shared_v4_u32_sideeffect(base_addr)
            dst_flat[0] = cutlass.Int32(v0)
            dst_flat[1] = cutlass.Int32(v2)
        else:
            cute.copy(smem_tiled_copy_SFA, src, dst)

    def _copy_sfb_smem_to_fragment(
        self,
        smem_tiled_copy_SFB,
        src: cute.Tensor,
        dst: cute.Tensor,
    ):
        if cutlass.const_expr(self.use_manual_sfb_v4_fixed_copy):
            src_flat = cute.flatten(src)
            dst_flat = cute.flatten(dst)
            base_addr_0 = get_smem_ptr_as_int32(src_flat, cutlass.Int32(0))
            base_addr_4 = get_smem_ptr_as_int32(src_flat, cutlass.Int32(4))
            e0, e1, e2, e3 = task39_ld_shared_v4_u32_sideeffect(base_addr_0)
            e4, e5, e6, e7 = task39_ld_shared_v4_u32_sideeffect(base_addr_4)
            if cutlass.const_expr(self.manual_sfb_v4_fixed_perm >= 4):
                base_addr_8 = get_smem_ptr_as_int32(src_flat, cutlass.Int32(8))
                base_addr_12 = get_smem_ptr_as_int32(src_flat, cutlass.Int32(12))
                e8, e9, e10, e11 = task39_ld_shared_v4_u32_sideeffect(base_addr_8)
                e12, e13, e14, e15 = task39_ld_shared_v4_u32_sideeffect(
                    base_addr_12
                )
                if cutlass.const_expr(self.manual_sfb_v4_fixed_perm == 5):
                    dst_flat[0] = cutlass.Int32(e0)
                    dst_flat[1] = cutlass.Int32(e4)
                    dst_flat[2] = cutlass.Int32(e8)
                    dst_flat[3] = cutlass.Int32(e12)
                    dst_flat[4] = cutlass.Int32(e2)
                    dst_flat[5] = cutlass.Int32(e6)
                    dst_flat[6] = cutlass.Int32(e10)
                    dst_flat[7] = cutlass.Int32(e14)
                elif cutlass.const_expr(self.manual_sfb_v4_fixed_perm == 6):
                    dst_flat[0] = cutlass.Int32(e0)
                    dst_flat[1] = cutlass.Int32(e8)
                    dst_flat[2] = cutlass.Int32(e2)
                    dst_flat[3] = cutlass.Int32(e10)
                    dst_flat[4] = cutlass.Int32(e4)
                    dst_flat[5] = cutlass.Int32(e12)
                    dst_flat[6] = cutlass.Int32(e6)
                    dst_flat[7] = cutlass.Int32(e14)
                else:
                    dst_flat[0] = cutlass.Int32(e0)
                    dst_flat[1] = cutlass.Int32(e2)
                    dst_flat[2] = cutlass.Int32(e4)
                    dst_flat[3] = cutlass.Int32(e6)
                    dst_flat[4] = cutlass.Int32(e8)
                    dst_flat[5] = cutlass.Int32(e10)
                    dst_flat[6] = cutlass.Int32(e12)
                    dst_flat[7] = cutlass.Int32(e14)
            elif cutlass.const_expr(self.manual_sfb_v4_fixed_perm == 1):
                dst_flat[0] = cutlass.Int32(e0)
                dst_flat[1] = cutlass.Int32(e2)
                dst_flat[2] = cutlass.Int32(e4)
                dst_flat[3] = cutlass.Int32(e6)
                dst_flat[4] = cutlass.Int32(e1)
                dst_flat[5] = cutlass.Int32(e3)
                dst_flat[6] = cutlass.Int32(e5)
                dst_flat[7] = cutlass.Int32(e7)
            elif cutlass.const_expr(self.manual_sfb_v4_fixed_perm == 2):
                dst_flat[0] = cutlass.Int32(e0)
                dst_flat[1] = cutlass.Int32(e4)
                dst_flat[2] = cutlass.Int32(e1)
                dst_flat[3] = cutlass.Int32(e5)
                dst_flat[4] = cutlass.Int32(e2)
                dst_flat[5] = cutlass.Int32(e6)
                dst_flat[6] = cutlass.Int32(e3)
                dst_flat[7] = cutlass.Int32(e7)
            elif cutlass.const_expr(self.manual_sfb_v4_fixed_perm == 3):
                dst_flat[0] = cutlass.Int32(e0)
                dst_flat[1] = cutlass.Int32(e4)
                dst_flat[2] = cutlass.Int32(e2)
                dst_flat[3] = cutlass.Int32(e6)
                dst_flat[4] = cutlass.Int32(e1)
                dst_flat[5] = cutlass.Int32(e5)
                dst_flat[6] = cutlass.Int32(e3)
                dst_flat[7] = cutlass.Int32(e7)
            else:
                dst_flat[0] = cutlass.Int32(e0)
                dst_flat[1] = cutlass.Int32(e1)
                dst_flat[2] = cutlass.Int32(e2)
                dst_flat[3] = cutlass.Int32(e3)
                dst_flat[4] = cutlass.Int32(e4)
                dst_flat[5] = cutlass.Int32(e5)
                dst_flat[6] = cutlass.Int32(e6)
                dst_flat[7] = cutlass.Int32(e7)
        elif cutlass.const_expr(self.use_manual_sfb_v4_copy):
            src_flat = cute.flatten(src)
            dst_flat = cute.flatten(dst)
            base_addr_0 = get_smem_ptr_as_int32(src_flat, cutlass.Int32(0))
            base_addr_2 = get_smem_ptr_as_int32(src_flat, cutlass.Int32(2))
            base_addr_4 = get_smem_ptr_as_int32(src_flat, cutlass.Int32(4))
            base_addr_6 = get_smem_ptr_as_int32(src_flat, cutlass.Int32(6))
            v0, _, v1, _ = task39_ld_shared_v4_u32_sideeffect(base_addr_0)
            v2, _, v3, _ = task39_ld_shared_v4_u32_sideeffect(base_addr_2)
            v4, _, v5, _ = task39_ld_shared_v4_u32_sideeffect(base_addr_4)
            v6, _, v7, _ = task39_ld_shared_v4_u32_sideeffect(base_addr_6)
            dst_flat[0] = cutlass.Int32(v0)
            dst_flat[1] = cutlass.Int32(v2)
            dst_flat[2] = cutlass.Int32(v4)
            dst_flat[3] = cutlass.Int32(v6)
            dst_flat[4] = cutlass.Int32(v1)
            dst_flat[5] = cutlass.Int32(v3)
            dst_flat[6] = cutlass.Int32(v5)
            dst_flat[7] = cutlass.Int32(v7)
        else:
            cute.copy(smem_tiled_copy_SFB, src, dst)

    # GPU device kernel
    @cute.kernel
    def kernel(
        self,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_sfa: cute.CopyAtom,
        mSFA_mkl: cute.Tensor,
        tma_atom_sfb: cute.CopyAtom,
        mSFB_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        tiled_mma: cute.TiledMma,
        mma_atom: cute.MmaAtom,
        cta_layout_mnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout],
        sfb_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout],
        epi_smem_layout_staged: cute.ComposedLayout,
        tile_sched_params: utils.PersistentTileSchedulerParams,
        epilogue_op: cutlass.Constexpr,
        alpha: cute.Tensor,
    ):
        # Keep alpha in FP32 for precision
        alpha_value = alpha[0].to(cutlass.Float32)

        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        # Prefetch TMA descriptors
        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_sfa)
            cpasync.prefetch_descriptor(tma_atom_sfb)
            cpasync.prefetch_descriptor(tma_atom_c)

        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        cluster_coord_mnk = cta_layout_mnk.get_flat_coord(cta_rank_in_cluster)

        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, 0))
        sfa_smem_layout = cute.slice_(sfa_smem_layout_staged, (None, None, 0))
        sfb_smem_layout = cute.slice_(sfb_smem_layout_staged, (None, None, 0))
        tma_copy_bytes = (
            cute.size_in_bytes(self.a_dtype, a_smem_layout)
            + cute.size_in_bytes(self.b_dtype, b_smem_layout)
            + cute.size_in_bytes(self.sf_dtype, sfa_smem_layout)
            + cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)
        )

        # Allocate shared memory
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # Pipeline setup
        mainloop_pipeline_array_ptr = storage.mainloop_pipeline_array_ptr.data_ptr()
        mainloop_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread
        )
        mainloop_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, self.num_mma_warps
        )

        cta_layout_vmnk = cute.make_layout((1, *cta_layout_mnk.shape))
        mainloop_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=self.ab_stage,
            producer_group=mainloop_pipeline_producer_group,
            consumer_group=mainloop_pipeline_consumer_group,
            tx_count=tma_copy_bytes,
            barrier_storage=mainloop_pipeline_array_ptr,
            cta_layout_vmnk=cta_layout_vmnk,
        )

        if cute.size(self.cluster_shape_mnk) > 1:
            cute.arch.cluster_arrive_relaxed()

        # Generate smem tensors
        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )
        sC = storage.sC.get_tensor(
            epi_smem_layout_staged.outer, swizzle=epi_smem_layout_staged.inner
        )
        if cutlass.const_expr(self.use_swizzled_sf_smem_layout):
            sSFA = storage.sSFA.get_tensor(
                sfa_smem_layout_staged.outer,
                swizzle=sfa_smem_layout_staged.inner,
            )
            sSFB = storage.sSFB.get_tensor(
                sfb_smem_layout_staged.outer,
                swizzle=sfb_smem_layout_staged.inner,
            )
        else:
            sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
            sSFB = storage.sSFB.get_tensor(sfb_smem_layout_staged)

        # Local_tile partition global tensors
        gA_mkl = cute.local_tile(
            mA_mkl,
            cute.slice_(self.tile_shape_mnk, (None, 0, None)),
            (None, None, None),
        )
        gB_nkl = cute.local_tile(
            mB_nkl,
            cute.slice_(self.tile_shape_mnk, (0, None, None)),
            (None, None, None),
        )
        gSFA_mkl = cute.local_tile(
            mSFA_mkl,
            self.sfa_tile_shape_mk,
            (None, None, None),
        )
        gSFB_nkl = cute.local_tile(
            mSFB_nkl,
            self.sfb_tile_shape_nk,
            (None, None, None),
        )
        gC_mnl = cute.local_tile(
            mC_mnl,
            cute.slice_(self.tile_shape_mnk, (None, None, 0)),
            (None, None, None),
        )

        # Partition for TiledMMA
        thr_mma = tiled_mma.get_slice(tidx)

        # TMA partitions for A
        a_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (0, None, 0)).shape)
        a_cta_crd = cluster_coord_mnk[1]
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            a_cta_crd,
            a_cta_layout,
            cute.group_modes(sA, 0, 2),
            cute.group_modes(gA_mkl, 0, 2),
        )

        # TMA partitions for B
        b_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (None, 0, 0)).shape)
        b_cta_crd = cluster_coord_mnk[0]
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            b_cta_crd,
            b_cta_layout,
            cute.group_modes(sB, 0, 2),
            cute.group_modes(gB_nkl, 0, 2),
        )

        # TMA partitions for SFA
        tAsSFA, tAgSFA = cpasync.tma_partition(
            tma_atom_sfa,
            a_cta_crd,
            a_cta_layout,
            cute.group_modes(sSFA, 0, 2),
            cute.group_modes(gSFA_mkl, 0, 2),
        )
        if cutlass.const_expr(not self.use_unfiltered_sf_copy):
            tAsSFA = cute.filter_zeros(tAsSFA)
            tAgSFA = cute.filter_zeros(tAgSFA)

        # TMA partitions for SFB
        tBsSFB, tBgSFB = cpasync.tma_partition(
            tma_atom_sfb,
            b_cta_crd,
            b_cta_layout,
            cute.group_modes(sSFB, 0, 2),
            cute.group_modes(gSFB_nkl, 0, 2),
        )
        if cutlass.const_expr(not self.use_unfiltered_sf_copy):
            tBsSFB = cute.filter_zeros(tBsSFB)
            tBgSFB = cute.filter_zeros(tBgSFB)

        # Make fragments
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)

        tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
        tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
        tCrSFA_full = self._partition_fragment_SFA(sSFA[None, None, 0], thr_mma, tidx)
        tCrSFB_full = self._partition_fragment_SFB(sSFB[None, None, 0], thr_mma, tidx)

        tCgC = thr_mma.partition_C(gC_mnl)
        acc_shape = tCgC.shape[:3]
        accumulators = cute.make_rmem_tensor(acc_shape, self.acc_dtype)

        # Cluster/thread sync
        if cute.size(self.cluster_shape_mnk) > 1:
            cute.arch.cluster_wait()
        else:
            cute.arch.sync_threads()

        k_tile_cnt = cute.size(gA_mkl, mode=[3])

        # Tile scheduler
        tile_sched = utils.StaticPersistentTileScheduler.create(
            tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
        )
        work_tile = tile_sched.initial_work_tile_info()

        # Pipeline states
        mainloop_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.ab_stage
        )
        mainloop_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.ab_stage
        )

        # MMA warp group
        if warp_idx < self.num_mma_warps:
            cute.arch.setmaxregister_increase(self.mma_register_requirement)

            num_k_blocks = cute.size(tCrA, mode=[2])

            # Copy atoms for SMEM->RMEM
            atom_copy_ldmatrix_A = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(self.a_layout.is_m_major_a(), 4),
                self.a_dtype,
            )
            atom_copy_ldmatrix_B = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(self.b_layout.is_n_major_b(), 4),
                self.b_dtype,
            )
            smem_tiled_copy_A = cute.make_tiled_copy_A(atom_copy_ldmatrix_A, tiled_mma)
            smem_tiled_copy_B = cute.make_tiled_copy_B(atom_copy_ldmatrix_B, tiled_mma)

            sf_copy_dtype = (
                cutlass.Int128
                if cutlass.const_expr(self.sf_copy_vector_bits == 128)
                else (
                    cutlass.Int64
                    if cutlass.const_expr(self.sf_copy_vector_bits == 64)
                    else (
                        cutlass.Int32
                        if cutlass.const_expr(
                            self.use_vectorized_sf_copy_u32
                            or self.sf_copy_vector_bits == 32
                        )
                        else self.sf_dtype
                    )
                )
            )
            if cutlass.const_expr(self.sf_num_bits_per_copy != 0):
                atom_copy_ldmatrix_SF = cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(),
                    sf_copy_dtype,
                    num_bits_per_copy=self.sf_num_bits_per_copy,
                )
            else:
                atom_copy_ldmatrix_SF = cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(),
                    sf_copy_dtype,
                )
            smem_tiled_copy_SFA = cute.make_tiled_copy(
                atom_copy_ldmatrix_SF,
                self._get_layoutSFA_TV(tiled_mma),
                (
                    cute.size(tiled_mma.permutation_mnk[0]),
                    cute.size(tiled_mma.permutation_mnk[2]),
                ),
            )
            smem_tiled_copy_SFB = cute.make_tiled_copy(
                atom_copy_ldmatrix_SF,
                self._get_layoutSFB_TV(tiled_mma),
                (
                    cute.size(tiled_mma.permutation_mnk[1]),
                    cute.size(tiled_mma.permutation_mnk[2]),
                ),
            )

            thr_copy_ldmatrix_A = smem_tiled_copy_A.get_slice(tidx)
            thr_copy_ldmatrix_B = smem_tiled_copy_B.get_slice(tidx)
            tCsA_copy_view = thr_copy_ldmatrix_A.partition_S(sA)
            tCrA_copy_view = thr_copy_ldmatrix_A.retile(tCrA)
            tCsB_copy_view = thr_copy_ldmatrix_B.partition_S(sB)
            tCrB_copy_view = thr_copy_ldmatrix_B.retile(tCrB)

            thr_copy_ldmatrix_SFA = smem_tiled_copy_SFA.get_slice(tidx)
            thr_copy_ldmatrix_SFB = smem_tiled_copy_SFB.get_slice(tidx)
            tCsSFA_copy_view_full = thr_copy_ldmatrix_SFA.partition_S(sSFA)
            tCrSFA_copy_view_full = thr_copy_ldmatrix_SFA.retile(tCrSFA_full)
            tCsSFB_copy_view_full = thr_copy_ldmatrix_SFB.partition_S(sSFB)
            tCrSFB_copy_view_full = thr_copy_ldmatrix_SFB.retile(tCrSFB_full)
            if cutlass.const_expr(
                self.use_vectorized_sf_copy_u32 or self.sf_copy_vector_bits != 0
            ):
                tCsSFA_copy_view_full = cute.recast_tensor(
                    tCsSFA_copy_view_full, sf_copy_dtype
                )
                tCrSFA_copy_view_full = cute.recast_tensor(
                    tCrSFA_copy_view_full, sf_copy_dtype
                )
                tCsSFB_copy_view_full = cute.recast_tensor(
                    tCsSFB_copy_view_full, sf_copy_dtype
                )
                tCrSFB_copy_view_full = cute.recast_tensor(
                    tCrSFB_copy_view_full, sf_copy_dtype
                )

            while work_tile.is_valid_tile:
                tile_coord_mnl = work_tile.tile_idx
                gC_mnl_slice = gC_mnl[(None, None, *tile_coord_mnl)]
                if cutlass.const_expr(self.use_fixed_single_sf_tile):
                    tCsSFA_tile_copy_view = tCsSFA_copy_view_full
                    tCrSFA_tile = tCrSFA_full
                    tCrSFA_tile_copy_view = tCrSFA_copy_view_full
                    tCsSFB_tile_copy_view = tCsSFB_copy_view_full
                    tCrSFB_tile = tCrSFB_full
                    tCrSFB_tile_copy_view = tCrSFB_copy_view_full
                else:
                    sfa_tile_offset = tile_coord_mnl[0] % self.sfa_tiles_per_block
                    sfb_tile_offset = tile_coord_mnl[1] % self.sfb_tiles_per_block
                    if cutlass.const_expr(self.sfa_tiles_per_block > 1):
                        sSFA_tile = cute.local_tile(
                            sSFA,
                            cute.slice_(self.tile_shape_mnk, (None, 0, None)),
                            (sfa_tile_offset, 0, None),
                        )
                        tCsSFA_tile_copy_view = thr_copy_ldmatrix_SFA.partition_S(sSFA_tile)
                        tCrSFA_tile = self._partition_fragment_SFA(
                            sSFA_tile[None, None, 0], thr_mma, tidx
                        )
                        tCrSFA_tile_copy_view = thr_copy_ldmatrix_SFA.retile(tCrSFA_tile)
                        if cutlass.const_expr(
                            self.use_vectorized_sf_copy_u32
                            or self.sf_copy_vector_bits != 0
                        ):
                            tCsSFA_tile_copy_view = cute.recast_tensor(
                                tCsSFA_tile_copy_view, sf_copy_dtype
                            )
                            tCrSFA_tile_copy_view = cute.recast_tensor(
                                tCrSFA_tile_copy_view, sf_copy_dtype
                            )
                    else:
                        tCsSFA_tile_copy_view = tCsSFA_copy_view_full
                        tCrSFA_tile = tCrSFA_full
                        tCrSFA_tile_copy_view = tCrSFA_copy_view_full
                    if cutlass.const_expr(self.sfb_tiles_per_block > 1):
                        sSFB_tile = cute.local_tile(
                            sSFB,
                            cute.slice_(self.tile_shape_mnk, (0, None, None)),
                            (sfb_tile_offset, 0, None),
                        )
                        tCsSFB_tile_copy_view = thr_copy_ldmatrix_SFB.partition_S(sSFB_tile)
                        tCrSFB_tile = self._partition_fragment_SFB(
                            sSFB_tile[None, None, 0], thr_mma, tidx
                        )
                        tCrSFB_tile_copy_view = thr_copy_ldmatrix_SFB.retile(tCrSFB_tile)
                        if cutlass.const_expr(
                            self.use_vectorized_sf_copy_u32
                            or self.sf_copy_vector_bits != 0
                        ):
                            tCsSFB_tile_copy_view = cute.recast_tensor(
                                tCsSFB_tile_copy_view, sf_copy_dtype
                            )
                            tCrSFB_tile_copy_view = cute.recast_tensor(
                                tCrSFB_tile_copy_view, sf_copy_dtype
                            )
                    else:
                        tCsSFB_tile_copy_view = tCsSFB_copy_view_full
                        tCrSFB_tile = tCrSFB_full
                        tCrSFB_tile_copy_view = tCrSFB_copy_view_full
                accumulators.fill(0.0)

                # Pipelined MAINLOOP
                mainloop_consumer_state.reset_count()

                peek_ab_full_status = cutlass.Boolean(1)
                if mainloop_consumer_state.count < k_tile_cnt:
                    peek_ab_full_status = mainloop_pipeline.consumer_try_wait(
                        mainloop_consumer_state
                    )

                mainloop_pipeline.consumer_wait(
                    mainloop_consumer_state, peek_ab_full_status
                )
                tCsA_p = tCsA_copy_view[None, None, None, mainloop_consumer_state.index]
                tCsB_p = tCsB_copy_view[None, None, None, mainloop_consumer_state.index]
                tCsSFA_p = tCsSFA_tile_copy_view[
                    None, None, None, mainloop_consumer_state.index
                ]
                tCsSFB_p = tCsSFB_tile_copy_view[
                    None, None, None, mainloop_consumer_state.index
                ]
                if cutlass.const_expr(self.use_unfiltered_sf_copy):
                    tCsSFA_p_filtered = tCsSFA_p
                    tCsSFB_p_filtered = tCsSFB_p
                    tCrSFA_copy_view_filtered = tCrSFA_tile_copy_view
                    tCrSFB_copy_view_filtered = tCrSFB_tile_copy_view
                else:
                    tCsSFA_p_filtered = cute.filter_zeros(tCsSFA_p)
                    tCsSFB_p_filtered = cute.filter_zeros(tCsSFB_p)
                    tCrSFA_copy_view_filtered = cute.filter_zeros(tCrSFA_tile_copy_view)
                    tCrSFB_copy_view_filtered = cute.filter_zeros(tCrSFB_tile_copy_view)

                if cutlass.const_expr(self.copy_sf_before_ab):
                    self._copy_sfa_smem_to_fragment(
                        smem_tiled_copy_SFA,
                        tCsSFA_p_filtered[None, None, 0],
                        tCrSFA_copy_view_filtered[None, None, 0],
                    )
                    self._copy_sfb_smem_to_fragment(
                        smem_tiled_copy_SFB,
                        tCsSFB_p_filtered[None, None, 0],
                        tCrSFB_copy_view_filtered[None, None, 0],
                    )
                cute.copy(
                    smem_tiled_copy_A,
                    tCsA_p[None, None, 0],
                    tCrA_copy_view[None, None, 0],
                )
                cute.copy(
                    smem_tiled_copy_B,
                    tCsB_p[None, None, 0],
                    tCrB_copy_view[None, None, 0],
                )
                if cutlass.const_expr(not self.copy_sf_before_ab):
                    self._copy_sfa_smem_to_fragment(
                        smem_tiled_copy_SFA,
                        tCsSFA_p_filtered[None, None, 0],
                        tCrSFA_copy_view_filtered[None, None, 0],
                    )
                    self._copy_sfb_smem_to_fragment(
                        smem_tiled_copy_SFB,
                        tCsSFB_p_filtered[None, None, 0],
                        tCrSFB_copy_view_filtered[None, None, 0],
                    )
                if cutlass.const_expr(self.preload_all_k_blocks):
                    if cutlass.const_expr(num_k_blocks > 1):
                        if cutlass.const_expr(self.copy_sf_before_ab):
                            self._copy_sfa_smem_to_fragment(
                                smem_tiled_copy_SFA,
                                tCsSFA_p_filtered[None, None, 1],
                                tCrSFA_copy_view_filtered[None, None, 1],
                            )
                            self._copy_sfb_smem_to_fragment(
                                smem_tiled_copy_SFB,
                                tCsSFB_p_filtered[None, None, 1],
                                tCrSFB_copy_view_filtered[None, None, 1],
                            )
                if cutlass.const_expr(self.preload_sf_k_blocks):
                    if cutlass.const_expr(num_k_blocks > 1):
                        self._copy_sfa_smem_to_fragment(
                            smem_tiled_copy_SFA,
                            tCsSFA_p_filtered[None, None, 1],
                            tCrSFA_copy_view_filtered[None, None, 1],
                        )
                        self._copy_sfb_smem_to_fragment(
                            smem_tiled_copy_SFB,
                            tCsSFB_p_filtered[None, None, 1],
                            tCrSFB_copy_view_filtered[None, None, 1],
                        )
                        cute.copy(
                            smem_tiled_copy_A,
                            tCsA_p[None, None, 1],
                            tCrA_copy_view[None, None, 1],
                        )
                        cute.copy(
                            smem_tiled_copy_B,
                            tCsB_p[None, None, 1],
                            tCrB_copy_view[None, None, 1],
                        )
                        if cutlass.const_expr(not self.copy_sf_before_ab):
                            self._copy_sfa_smem_to_fragment(
                                smem_tiled_copy_SFA,
                                tCsSFA_p_filtered[None, None, 1],
                                tCrSFA_copy_view_filtered[None, None, 1],
                            )
                            self._copy_sfb_smem_to_fragment(
                                smem_tiled_copy_SFB,
                                tCsSFB_p_filtered[None, None, 1],
                                tCrSFB_copy_view_filtered[None, None, 1],
                            )

                for _k_tile in range(0, k_tile_cnt - 1, 1, unroll=2):  # type: ignore[call-overload]
                    for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                        k_block_next = (
                            0 if k_block_idx + 1 == num_k_blocks else k_block_idx + 1
                        )

                        if cutlass.const_expr(self.preload_all_k_blocks):
                            if k_block_idx == 0:
                                mainloop_pipeline.consumer_release(
                                    mainloop_consumer_state
                                )
                                mainloop_consumer_state.advance()
                                if cutlass.const_expr(
                                    self.preload_wait_before_last_mma
                                ):
                                    peek_ab_full_status = cutlass.Boolean(1)
                                    peek_ab_full_status = (
                                        mainloop_pipeline.consumer_try_wait(
                                            mainloop_consumer_state
                                        )
                                    )

                        if cutlass.const_expr(not self.preload_all_k_blocks):
                            if k_block_idx == num_k_blocks - 1:
                                mainloop_pipeline.consumer_release(
                                    mainloop_consumer_state
                                )
                                mainloop_consumer_state.advance()

                        if k_block_idx == num_k_blocks - 1:
                            if cutlass.const_expr(self.preload_all_k_blocks):
                                pass
                            elif cutlass.const_expr(self.wait_after_last_mma):
                                pass
                            elif cutlass.const_expr(self.pretry_wait_after_last_mma):
                                peek_ab_full_status = cutlass.Boolean(1)
                                peek_ab_full_status = mainloop_pipeline.consumer_try_wait(
                                    mainloop_consumer_state
                                )

                                tCsA_p = tCsA_copy_view[
                                    None, None, None, mainloop_consumer_state.index
                                ]
                                tCsB_p = tCsB_copy_view[
                                    None, None, None, mainloop_consumer_state.index
                                ]
                                tCsSFA_p = tCsSFA_tile_copy_view[
                                    None, None, None, mainloop_consumer_state.index
                                ]
                                tCsSFB_p = tCsSFB_tile_copy_view[
                                    None, None, None, mainloop_consumer_state.index
                                ]
                            else:
                                peek_ab_full_status = cutlass.Boolean(1)
                                peek_ab_full_status = mainloop_pipeline.consumer_try_wait(
                                    mainloop_consumer_state
                                )

                                tCsA_p = tCsA_copy_view[
                                    None, None, None, mainloop_consumer_state.index
                                ]
                                tCsB_p = tCsB_copy_view[
                                    None, None, None, mainloop_consumer_state.index
                                ]
                                tCsSFA_p = tCsSFA_tile_copy_view[
                                    None, None, None, mainloop_consumer_state.index
                                ]
                                tCsSFB_p = tCsSFB_tile_copy_view[
                                    None, None, None, mainloop_consumer_state.index
                                ]
                                mainloop_pipeline.consumer_wait(
                                    mainloop_consumer_state, peek_ab_full_status
                                )

                        # Manual atom unroll: avoids hasAuxTensor address space bug
                        for _mt in range(self.num_m_tiles):
                            if cutlass.const_expr(self.hoist_sfa_set):
                                mma_atom.set(
                                    WarpField.SFA,
                                    tCrSFA_tile[None, _mt, k_block_idx].iterator,
                                )
                            for _nt in range(self.num_n_tiles):
                                if cutlass.const_expr(not self.hoist_sfa_set):
                                    mma_atom.set(
                                        WarpField.SFA,
                                        tCrSFA_tile[None, _mt, k_block_idx].iterator,
                                    )
                                mma_atom.set(
                                    WarpField.SFB,
                                    tCrSFB_tile[None, _nt, k_block_idx].iterator,
                                )
                                cute.gemm(
                                    mma_atom,
                                    accumulators[None, _mt, _nt],
                                    tCrA[None, _mt, k_block_idx],
                                    tCrB[None, _nt, k_block_idx],
                                    accumulators[None, _mt, _nt],
                                )
                        if cutlass.const_expr(
                            self.preload_all_k_blocks
                            and self.preload_wait_before_last_mma
                        ):
                            if k_block_idx == 0:
                                tCsA_p = tCsA_copy_view[
                                    None, None, None, mainloop_consumer_state.index
                                ]
                                tCsB_p = tCsB_copy_view[
                                    None, None, None, mainloop_consumer_state.index
                                ]
                                tCsSFA_p = tCsSFA_tile_copy_view[
                                    None, None, None, mainloop_consumer_state.index
                                ]
                                tCsSFB_p = tCsSFB_tile_copy_view[
                                    None, None, None, mainloop_consumer_state.index
                                ]
                                mainloop_pipeline.consumer_wait(
                                    mainloop_consumer_state, peek_ab_full_status
                                )
                        if cutlass.const_expr(self.pretry_wait_after_last_mma):
                            if k_block_idx == num_k_blocks - 1:
                                mainloop_pipeline.consumer_wait(
                                    mainloop_consumer_state, peek_ab_full_status
                                )
                        if cutlass.const_expr(
                            self.wait_after_last_mma
                            or (
                                self.preload_all_k_blocks
                                and not self.preload_wait_before_last_mma
                            )
                        ):
                            if k_block_idx == num_k_blocks - 1:
                                peek_ab_full_status = cutlass.Boolean(1)
                                peek_ab_full_status = mainloop_pipeline.consumer_try_wait(
                                    mainloop_consumer_state
                                )

                                tCsA_p = tCsA_copy_view[
                                    None, None, None, mainloop_consumer_state.index
                                ]
                                tCsB_p = tCsB_copy_view[
                                    None, None, None, mainloop_consumer_state.index
                                ]
                                tCsSFA_p = tCsSFA_tile_copy_view[
                                    None, None, None, mainloop_consumer_state.index
                                ]
                                tCsSFB_p = tCsSFB_tile_copy_view[
                                    None, None, None, mainloop_consumer_state.index
                                ]
                                mainloop_pipeline.consumer_wait(
                                    mainloop_consumer_state, peek_ab_full_status
                                )
                        if cutlass.const_expr(self.use_unfiltered_sf_copy):
                            tCsSFA_p_filtered = tCsSFA_p
                            tCsSFB_p_filtered = tCsSFB_p
                            tCrSFA_copy_view_filtered = tCrSFA_tile_copy_view
                            tCrSFB_copy_view_filtered = tCrSFB_tile_copy_view
                        else:
                            tCsSFA_p_filtered = cute.filter_zeros(tCsSFA_p)
                            tCsSFB_p_filtered = cute.filter_zeros(tCsSFB_p)
                            tCrSFA_copy_view_filtered = cute.filter_zeros(
                                tCrSFA_tile_copy_view
                            )
                            tCrSFB_copy_view_filtered = cute.filter_zeros(
                                tCrSFB_tile_copy_view
                            )
                        should_copy_k_block_next = cutlass.Boolean(1)
                        if cutlass.const_expr(self.preload_all_k_blocks):
                            if k_block_next > 0:
                                should_copy_k_block_next = cutlass.Boolean(0)
                        should_copy_sf_k_block_next = cutlass.Boolean(1)
                        if cutlass.const_expr(self.preload_sf_k_blocks):
                            if k_block_next > 0:
                                should_copy_sf_k_block_next = cutlass.Boolean(0)

                        if cutlass.const_expr(self.copy_sf_before_ab):
                            if should_copy_k_block_next:
                                if should_copy_sf_k_block_next:
                                    self._copy_sfa_smem_to_fragment(
                                        smem_tiled_copy_SFA,
                                        tCsSFA_p_filtered[None, None, k_block_next],
                                        tCrSFA_copy_view_filtered[
                                            None, None, k_block_next
                                        ],
                                    )
                                    self._copy_sfb_smem_to_fragment(
                                        smem_tiled_copy_SFB,
                                        tCsSFB_p_filtered[None, None, k_block_next],
                                        tCrSFB_copy_view_filtered[
                                            None, None, k_block_next
                                        ],
                                    )
                        if should_copy_k_block_next:
                            cute.copy(
                                smem_tiled_copy_A,
                                tCsA_p[None, None, k_block_next],
                                tCrA_copy_view[None, None, k_block_next],
                            )
                            cute.copy(
                                smem_tiled_copy_B,
                                tCsB_p[None, None, k_block_next],
                                tCrB_copy_view[None, None, k_block_next],
                            )
                        if cutlass.const_expr(not self.copy_sf_before_ab):
                            if should_copy_k_block_next:
                                if should_copy_sf_k_block_next:
                                    self._copy_sfa_smem_to_fragment(
                                        smem_tiled_copy_SFA,
                                        tCsSFA_p_filtered[None, None, k_block_next],
                                        tCrSFA_copy_view_filtered[
                                            None, None, k_block_next
                                        ],
                                    )
                                    self._copy_sfb_smem_to_fragment(
                                        smem_tiled_copy_SFB,
                                        tCsSFB_p_filtered[None, None, k_block_next],
                                        tCrSFB_copy_view_filtered[
                                            None, None, k_block_next
                                        ],
                                    )
                        if cutlass.const_expr(self.preload_all_k_blocks):
                            if k_block_idx == num_k_blocks - 1:
                                if cutlass.const_expr(num_k_blocks > 1):
                                    if cutlass.const_expr(self.copy_sf_before_ab):
                                        self._copy_sfa_smem_to_fragment(
                                            smem_tiled_copy_SFA,
                                            tCsSFA_p_filtered[None, None, 1],
                                            tCrSFA_copy_view_filtered[None, None, 1],
                                        )
                                        self._copy_sfb_smem_to_fragment(
                                            smem_tiled_copy_SFB,
                                            tCsSFB_p_filtered[None, None, 1],
                                            tCrSFB_copy_view_filtered[None, None, 1],
                                        )
                        if cutlass.const_expr(self.preload_sf_k_blocks):
                            if k_block_idx == num_k_blocks - 1:
                                if cutlass.const_expr(num_k_blocks > 1):
                                    self._copy_sfa_smem_to_fragment(
                                        smem_tiled_copy_SFA,
                                        tCsSFA_p_filtered[None, None, 1],
                                        tCrSFA_copy_view_filtered[None, None, 1],
                                    )
                                    self._copy_sfb_smem_to_fragment(
                                        smem_tiled_copy_SFB,
                                        tCsSFB_p_filtered[None, None, 1],
                                        tCrSFB_copy_view_filtered[None, None, 1],
                                    )
                                    cute.copy(
                                        smem_tiled_copy_A,
                                        tCsA_p[None, None, 1],
                                        tCrA_copy_view[None, None, 1],
                                    )
                                    cute.copy(
                                        smem_tiled_copy_B,
                                        tCsB_p[None, None, 1],
                                        tCrB_copy_view[None, None, 1],
                                    )
                                    if cutlass.const_expr(not self.copy_sf_before_ab):
                                        self._copy_sfa_smem_to_fragment(
                                            smem_tiled_copy_SFA,
                                            tCsSFA_p_filtered[None, None, 1],
                                            tCrSFA_copy_view_filtered[None, None, 1],
                                        )
                                        self._copy_sfb_smem_to_fragment(
                                            smem_tiled_copy_SFB,
                                            tCsSFB_p_filtered[None, None, 1],
                                            tCrSFB_copy_view_filtered[None, None, 1],
                                        )

                # Hoist out last k_tile
                for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                    k_block_next = (
                        0 if k_block_idx + 1 == num_k_blocks else k_block_idx + 1
                    )

                    if cutlass.const_expr(self.preload_all_k_blocks):
                        if k_block_idx == 0:
                            mainloop_pipeline.consumer_release(mainloop_consumer_state)
                            mainloop_consumer_state.advance()

                    if cutlass.const_expr(not self.preload_all_k_blocks):
                        if k_block_idx == num_k_blocks - 1:
                            mainloop_pipeline.consumer_release(mainloop_consumer_state)
                            mainloop_consumer_state.advance()

                    should_copy_k_block_next = cutlass.Boolean(1)
                    if cutlass.const_expr(self.preload_all_k_blocks):
                        if k_block_next > 0:
                            should_copy_k_block_next = cutlass.Boolean(0)
                    should_copy_sf_k_block_next = cutlass.Boolean(1)
                    if cutlass.const_expr(self.preload_sf_k_blocks):
                        if k_block_next > 0:
                            should_copy_sf_k_block_next = cutlass.Boolean(0)

                    if k_block_next > 0 and should_copy_k_block_next:
                        if cutlass.const_expr(self.use_unfiltered_sf_copy):
                            tCsSFA_p_filtered = tCsSFA_p
                            tCsSFB_p_filtered = tCsSFB_p
                            tCrSFA_copy_view_filtered = tCrSFA_tile_copy_view
                            tCrSFB_copy_view_filtered = tCrSFB_tile_copy_view
                        else:
                            tCsSFA_p_filtered = cute.filter_zeros(tCsSFA_p)
                            tCsSFB_p_filtered = cute.filter_zeros(tCsSFB_p)
                            tCrSFA_copy_view_filtered = cute.filter_zeros(
                                tCrSFA_tile_copy_view
                            )
                            tCrSFB_copy_view_filtered = cute.filter_zeros(
                                tCrSFB_tile_copy_view
                            )
                        if cutlass.const_expr(self.copy_sf_before_ab):
                            if should_copy_sf_k_block_next:
                                self._copy_sfa_smem_to_fragment(
                                    smem_tiled_copy_SFA,
                                    tCsSFA_p_filtered[None, None, k_block_next],
                                    tCrSFA_copy_view_filtered[
                                        None, None, k_block_next
                                    ],
                                )
                                self._copy_sfb_smem_to_fragment(
                                    smem_tiled_copy_SFB,
                                    tCsSFB_p_filtered[None, None, k_block_next],
                                    tCrSFB_copy_view_filtered[
                                        None, None, k_block_next
                                    ],
                                )
                        cute.copy(
                            smem_tiled_copy_A,
                            tCsA_p[None, None, k_block_next],
                            tCrA_copy_view[None, None, k_block_next],
                        )
                        cute.copy(
                            smem_tiled_copy_B,
                            tCsB_p[None, None, k_block_next],
                            tCrB_copy_view[None, None, k_block_next],
                        )
                        if cutlass.const_expr(not self.copy_sf_before_ab):
                            if should_copy_sf_k_block_next:
                                self._copy_sfa_smem_to_fragment(
                                    smem_tiled_copy_SFA,
                                    tCsSFA_p_filtered[None, None, k_block_next],
                                    tCrSFA_copy_view_filtered[
                                        None, None, k_block_next
                                    ],
                                )
                                self._copy_sfb_smem_to_fragment(
                                    smem_tiled_copy_SFB,
                                    tCsSFB_p_filtered[None, None, k_block_next],
                                    tCrSFB_copy_view_filtered[
                                        None, None, k_block_next
                                    ],
                                )
                    # Manual atom unroll: avoids hasAuxTensor address space bug
                    for _mt in range(self.num_m_tiles):
                        if cutlass.const_expr(self.hoist_sfa_set):
                            mma_atom.set(
                                WarpField.SFA,
                                tCrSFA_tile[None, _mt, k_block_idx].iterator,
                            )
                        for _nt in range(self.num_n_tiles):
                            if cutlass.const_expr(not self.hoist_sfa_set):
                                mma_atom.set(
                                    WarpField.SFA,
                                    tCrSFA_tile[None, _mt, k_block_idx].iterator,
                                )
                            mma_atom.set(
                                WarpField.SFB,
                                tCrSFB_tile[None, _nt, k_block_idx].iterator,
                            )
                            cute.gemm(
                                mma_atom,
                                accumulators[None, _mt, _nt],
                                tCrA[None, _mt, k_block_idx],
                                tCrB[None, _nt, k_block_idx],
                                accumulators[None, _mt, _nt],
                            )

                # EPILOGUE
                _is_m_major = self.c_layout.is_m_major_c()
                if cutlass.const_expr(self.c_dtype.width == 16):
                    copy_atom_r2s = cute.make_copy_atom(
                        cute.nvgpu.warp.StMatrix8x8x16bOp(_is_m_major, 2),
                        self.c_dtype,
                    )
                else:
                    copy_atom_r2s = cute.make_copy_atom(
                        cute.nvgpu.CopyUniversalOp(),
                        self.c_dtype,
                    )

                copy_atom_C = cute.make_copy_atom(
                    cute.nvgpu.warp.StMatrix8x8x16bOp(
                        self.c_layout.is_m_major_c(),
                        2,
                    ),
                    self.c_dtype,
                )

                tiled_copy_C_Atom = cute.make_tiled_copy_C_atom(copy_atom_C, tiled_mma)

                tiled_copy_r2s = cute.make_tiled_copy_S(
                    copy_atom_r2s,
                    tiled_copy_C_Atom,
                )

                thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
                tRS_sD = thr_copy_r2s.partition_D(sC)
                tRS_rAcc = tiled_copy_r2s.retile(accumulators)

                rD_shape = cute.shape(thr_copy_r2s.partition_S(sC))
                tRS_rD_layout = cute.make_layout(rD_shape[:3])
                tRS_rD = cute.make_rmem_tensor(tRS_rD_layout.shape, self.acc_dtype)

                sepi_for_tma_partition = cute.group_modes(sC, 0, 2)
                tcgc_for_tma_partition = cute.zipped_divide(gC_mnl_slice, self.epi_tile)

                bSG_sD, bSG_gD = cpasync.tma_partition(
                    tma_atom_c,
                    0,
                    cute.make_layout(1),
                    sepi_for_tma_partition,
                    tcgc_for_tma_partition,
                )

                epi_rest_m = bSG_gD.shape[1][0]
                epi_rest_n = bSG_gD.shape[1][1]
                epi_tile_m = self.epi_tile[0]
                epi_tile_n = self.epi_tile[1]
                mma_tile_m = self.tile_shape_mnk[0] // cute.size(tRS_rAcc, mode=[1])
                mma_tile_n = self.tile_shape_mnk[1] // cute.size(tRS_rAcc, mode=[2])
                has_multi_epi_store = cutlass.const_expr(
                    not (self.epi_stage == 1 and epi_rest_m == 1 and epi_rest_n == 1)
                )
                tma_store_producer_group = pipeline.CooperativeGroup(
                    pipeline.Agent.Thread,
                    self.num_mma_warps * self.num_threads_per_warp,
                )
                tma_store_pipeline = pipeline.PipelineTmaStore.create(
                    num_stages=self.epi_stage,
                    producer_group=tma_store_producer_group,
                )

                for epi_m in cutlass.range_constexpr(epi_rest_m):
                    for epi_n in cutlass.range_constexpr(epi_rest_n):
                        MmaMPerEpiM = epi_tile_m // mma_tile_m
                        MmaNPerEpiN = epi_tile_n // mma_tile_n
                        for mma_n_in_epi in cutlass.range_constexpr(MmaNPerEpiN):
                            for mma_m_in_epi in cutlass.range_constexpr(MmaMPerEpiM):
                                mma_n = (epi_n * MmaNPerEpiN) + mma_n_in_epi
                                mma_m = (epi_m * MmaMPerEpiM) + mma_m_in_epi
                                tRS_rD_slice = tRS_rD[
                                    (None, mma_m_in_epi, mma_n_in_epi)
                                ]
                                tRS_rAcc_slice = tRS_rAcc[(None, mma_m, mma_n)]
                                for elem_idx in cutlass.range_constexpr(
                                    cute.size(tRS_rD_slice)
                                ):
                                    tRS_rD_slice[elem_idx] = tRS_rAcc_slice[elem_idx]

                        # Type conversion with alpha scaling
                        tRS_rD_out = cute.make_rmem_tensor(
                            tRS_rD_layout.shape, self.c_dtype
                        )
                        acc_vec = tRS_rD.load()
                        if cutlass.const_expr(self.assume_alpha_one_identity_epilogue):
                            acc_vec = acc_vec.to(self.c_dtype)
                        else:
                            # Multiply alpha in FP32 before converting to c_dtype
                            # to avoid overflow when c_dtype is FP16
                            acc_vec = epilogue_op(
                                (alpha_value * acc_vec).to(self.c_dtype)
                            )
                        tRS_rD_out.store(acc_vec)

                        # Register to shared memory
                        epi_buffer = (epi_m * epi_rest_n + epi_n) % cute.size(
                            tRS_sD, mode=[3]
                        )
                        if has_multi_epi_store:
                            self.epilog_sync_barrier.arrive_and_wait()
                        cute.copy(
                            tiled_copy_r2s,
                            tRS_rD_out,
                            tRS_sD[(None, None, None, epi_buffer)],
                        )
                        cute.arch.fence_proxy(
                            "async.shared",
                            space="cta",
                        )
                        self.epilog_sync_barrier.arrive_and_wait()

                        # Copy from shared memory to global memory
                        gmem_coord = (epi_m, epi_n)
                        if warp_idx == 0:
                            cute.copy(
                                tma_atom_c,
                                bSG_sD[(None, epi_buffer)],
                                bSG_gD[(None, gmem_coord)],
                            )
                            if has_multi_epi_store:
                                tma_store_pipeline.producer_commit()
                                tma_store_pipeline.producer_acquire()

                # Advance to the next work tile
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
                if has_multi_epi_store:
                    tma_store_pipeline.producer_tail()

        # DMA warp group
        elif warp_idx == self.tma_load_warp_id:
            cute.arch.setmaxregister_decrease(self.load_register_requirement)

            while work_tile.is_valid_tile:
                tile_coord_mnl = work_tile.tile_idx
                tAgA_mkl = tAgA[(None, tile_coord_mnl[0], None, tile_coord_mnl[2])]
                tBgB_nkl = tBgB[(None, tile_coord_mnl[1], None, tile_coord_mnl[2])]
                sfa_tile_coord_m = tile_coord_mnl[0] // self.sfa_tiles_per_block
                tAgSFA_mkl = tAgSFA[(None, sfa_tile_coord_m, None, tile_coord_mnl[2])]
                sfb_tile_coord_n = tile_coord_mnl[1] // self.sfb_tiles_per_block
                tBgSFB_nkl = tBgSFB[(None, sfb_tile_coord_n, None, tile_coord_mnl[2])]

                mainloop_producer_state.reset_count()

                for _k_tile in range(0, k_tile_cnt, 1, unroll=2):  # type: ignore[call-overload]
                    mainloop_pipeline.producer_acquire(mainloop_producer_state)

                    tAgA_k = tAgA_mkl[(None, mainloop_producer_state.count)]
                    tAsA_pipe = tAsA[(None, mainloop_producer_state.index)]

                    tBgB_k = tBgB_nkl[(None, mainloop_producer_state.count)]
                    tBsB_pipe = tBsB[(None, mainloop_producer_state.index)]

                    tAgSFA_k = tAgSFA_mkl[(None, mainloop_producer_state.count)]
                    tAsSFA_pipe = tAsSFA[(None, mainloop_producer_state.index)]

                    tBgSFB_k = tBgSFB_nkl[(None, mainloop_producer_state.count)]
                    tBsSFB_pipe = tBsSFB[(None, mainloop_producer_state.index)]

                    cute.copy(
                        tma_atom_a,
                        tAgA_k,
                        tAsA_pipe,
                        tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                            mainloop_producer_state
                        ),
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB_k,
                        tBsB_pipe,
                        tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                            mainloop_producer_state
                        ),
                    )
                    cute.copy(
                        tma_atom_sfa,
                        tAgSFA_k,
                        tAsSFA_pipe,
                        tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                            mainloop_producer_state
                        ),
                    )
                    cute.copy(
                        tma_atom_sfb,
                        tBgSFB_k,
                        tBsSFB_pipe,
                        tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                            mainloop_producer_state
                        ),
                    )
                    mainloop_pipeline.producer_commit(mainloop_producer_state)
                    mainloop_producer_state.advance()

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            mainloop_pipeline.producer_tail(mainloop_producer_state)
        return

    @staticmethod
    def _compute_stages(
        tile_shape_mnk: tuple,
        a_dtype,
        b_dtype,
        sf_dtype,
        sfa_smem_layout,
        sfb_smem_layout,
        epi_tile: tuple,
        c_dtype,
        smem_capacity: int,
        occupancy: int,
    ) -> tuple:
        epi_stage_max = (tile_shape_mnk[1] // epi_tile[1]) * (
            tile_shape_mnk[0] // epi_tile[0]
        )
        epi_stage = min(epi_stage_max, 4)
        c_bytes_per_stage = cute.size(epi_tile) * c_dtype.width // 8
        epi_bytes = c_bytes_per_stage * epi_stage

        a_shape = cute.slice_(tile_shape_mnk, (None, 0, None))
        b_shape = cute.slice_(tile_shape_mnk, (0, None, None))
        ab_bytes_per_stage = (
            cute.size(a_shape) * a_dtype.width // 8
            + cute.size(b_shape) * b_dtype.width // 8
        )
        sf_bytes_per_stage = (
            cute.size(cute.filter_zeros(sfa_smem_layout).shape) * sf_dtype.width // 8
            + cute.size(cute.filter_zeros(sfb_smem_layout).shape) * sf_dtype.width // 8
        )
        mbar_helpers_bytes = 1024

        ab_stage = (
            (smem_capacity - occupancy * 1024) // occupancy
            - mbar_helpers_bytes
            - epi_bytes
        ) // (ab_bytes_per_stage + sf_bytes_per_stage)
        ab_stage = max(1, min(ab_stage, 4))
        return ab_stage, epi_stage

    @staticmethod
    def _make_smem_layouts(
        tile_shape_mnk: tuple,
        epi_tile: tuple,
        a_dtype,
        a_layout,
        b_dtype,
        b_layout,
        ab_stage: int,
        c_dtype,
        c_layout,
        epi_stage: int,
        sf_vec_size: int,
        tiled_mma,
        use_padded_sf_smem_layout: bool = False,
        sf_smem_mn_stride_pad: int = 0,
        use_bankmajor_sf_smem_layout: bool = False,
        use_fragmajor_sf_smem_layout: bool = False,
        use_swizzled_sf_smem_layout: bool = False,
        sf_smem_swizzle_bits: int = 0,
    ) -> tuple:
        a_smem_shape = cute.slice_(tile_shape_mnk, (None, 0, None))

        a_is_k_major = a_layout.is_k_major_a()
        b_is_k_major = b_layout.is_k_major_b()
        a_major_mode_size = tile_shape_mnk[2 if a_is_k_major else 0]

        a_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(
                a_layout,
                a_dtype,
                a_major_mode_size,
            ),
            a_dtype,
        )
        a_smem_layout_staged = cute.tile_to_shape(
            a_smem_layout_atom,
            cute.append(a_smem_shape, ab_stage),
            order=(0, 1, 2) if a_is_k_major else (1, 0, 2),
        )

        b_smem_shape = cute.slice_(tile_shape_mnk, (0, None, None))
        b_major_mode_size = tile_shape_mnk[2 if b_is_k_major else 1]
        b_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(
                b_layout,
                b_dtype,
                b_major_mode_size,
            ),
            b_dtype,
        )
        b_smem_layout_staged = cute.tile_to_shape(
            b_smem_layout_atom,
            cute.append(b_smem_shape, ab_stage),
            order=(0, 1, 2) if b_is_k_major else (1, 0, 2),
        )

        if use_padded_sf_smem_layout:
            sfa_smem_layout_staged = task39_sm120_make_smem_layout_sfa_padded(
                tiled_mma,
                tile_shape_mnk,
                sf_vec_size,
                ab_stage,
                sf_smem_mn_stride_pad,
            )
            sfb_smem_layout_staged = task39_sm120_make_smem_layout_sfb_padded(
                tiled_mma,
                tile_shape_mnk,
                sf_vec_size,
                ab_stage,
                sf_smem_mn_stride_pad,
            )
        elif use_bankmajor_sf_smem_layout:
            sfa_smem_layout_staged = task39_sm120_make_smem_layout_sfa_bankmajor(
                tiled_mma,
                tile_shape_mnk,
                sf_vec_size,
                ab_stage,
            )
            sfb_smem_layout_staged = task39_sm120_make_smem_layout_sfb_bankmajor(
                tiled_mma,
                tile_shape_mnk,
                sf_vec_size,
                ab_stage,
            )
        elif use_fragmajor_sf_smem_layout:
            sfa_smem_layout_staged = task39_sm120_make_smem_layout_sfa_fragmajor(
                tiled_mma,
                tile_shape_mnk,
                sf_vec_size,
                ab_stage,
            )
            sfb_smem_layout_staged = task39_sm120_make_smem_layout_sfb_fragmajor(
                tiled_mma,
                tile_shape_mnk,
                sf_vec_size,
                ab_stage,
            )
        else:
            sfa_smem_layout_staged = sm120_make_smem_layout_sfa(
                tiled_mma,
                tile_shape_mnk,
                sf_vec_size,
                ab_stage,
            )
            sfb_smem_layout_staged = sm120_make_smem_layout_sfb(
                tiled_mma,
                tile_shape_mnk,
                sf_vec_size,
                ab_stage,
            )
            if use_swizzled_sf_smem_layout:
                sfa_smem_layout_staged = cute.make_composed_layout(
                    cute.make_swizzle(sf_smem_swizzle_bits, 4, 3),
                    0,
                    sfa_smem_layout_staged,
                )
                sfb_smem_layout_staged = cute.make_composed_layout(
                    cute.make_swizzle(sf_smem_swizzle_bits, 4, 3),
                    0,
                    sfb_smem_layout_staged,
                )

        c_smem_shape = epi_tile
        c_major_mode_size = epi_tile[1] if c_layout.is_n_major_c() else epi_tile[0]
        c_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(
                c_layout,
                c_dtype,
                c_major_mode_size,
            ),
            c_dtype,
        )
        epi_smem_layout_staged = cute.tile_to_shape(
            c_smem_layout_atom,
            cute.append(c_smem_shape, epi_stage),
            order=(1, 0, 2) if c_layout.is_m_major_c() else (0, 1, 2),
        )

        return (
            a_smem_layout_staged,
            b_smem_layout_staged,
            sfa_smem_layout_staged,
            sfb_smem_layout_staged,
            epi_smem_layout_staged,
        )

    @staticmethod
    def _compute_grid(
        c,
        tile_shape_mnk: tuple,
        max_active_clusters,
    ) -> tuple:
        c_shape = cute.slice_(tile_shape_mnk, (None, None, 0))
        gc = cute.zipped_divide(c, tiler=c_shape)
        num_ctas_mnl = gc[(0, (None, None, None))].shape
        cluster_shape_mnl = (1, 1, 1)
        tile_sched_params = utils.PersistentTileSchedulerParams(
            num_ctas_mnl, cluster_shape_mnl
        )
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(
            tile_sched_params, max_active_clusters
        )
        return tile_sched_params, grid

    @staticmethod
    def _make_tma_store_atoms_and_tensors(
        tensor_c,
        epi_smem_layout_staged,
        epi_tile: tuple,
    ) -> tuple:
        epi_smem_layout = cute.slice_(epi_smem_layout_staged, (None, None, 0))
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            tensor_c,
            epi_smem_layout,
            epi_tile,
        )
        return tma_atom_c, tma_tensor_c

    @staticmethod
    def _make_tma_atoms_and_tensors(
        tensor,
        smem_layout_staged,
        smem_tile: tuple,
        mcast_dim: int,
        internal_type=None,
    ) -> tuple:
        op = (
            cpasync.CopyBulkTensorTileG2SOp()
            if mcast_dim == 1
            else cpasync.CopyBulkTensorTileG2SMulticastOp()
        )
        smem_layout = cute.slice_(smem_layout_staged, (None, None, 0))
        tma_atom, tma_tensor = cpasync.make_tiled_tma_atom(
            op,
            tensor,
            smem_layout,
            smem_tile,
            num_multicast=mcast_dim,
            internal_type=internal_type,
        )
        return tma_atom, tma_tensor

    @staticmethod
    def can_implement(
        ab_dtype,
        sf_dtype,
        sf_vec_size: int,
        c_dtype,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        m: int,
        n: int,
        k: int,
        l: int,
        a_major: str,
        b_major: str,
        c_major: str,
    ) -> bool:
        # The current target only supports cluster (1,1)
        if cluster_shape_mn != (1, 1):
            return False
        # Tile M must be divisible by 128; tile N follows 64-column warpgroup
        # quanta, while the SF paths round narrow tiles up to full 128-element
        # scale-factor blocks.
        if mma_tiler_mn[0] % 64 != 0 or mma_tiler_mn[1] % 64 != 0:
            return False
        # The current target only supports FP4 (MmaMXF4NVF4Op)
        if ab_dtype != cutlass.Float4E2M1FN:
            return False
        # SM120 warp-level MmaMXF4NVF4Op only supports sf_vec_size=16
        # (CUTLASS DSL hardcodes sf_vec_size=16 in the MMA atom constructor)
        if sf_vec_size != 16:
            return False
        if sf_dtype != cutlass.Float8E4M3FN:
            return False
        # Only 16-bit output types supported for now
        if c_dtype not in (cutlass.Float16, cutlass.BFloat16):
            return False
        # A must be K-major, B must be K-major
        if a_major != "k" or b_major != "k":
            return False
        # Alignment: K must be divisible by tile_k
        tile_k = sf_vec_size * 8
        if k % tile_k != 0:
            return False
        # Reject tiles that cannot fit even one pipeline stage in SM120
        # shared memory.  A+B are FP4 (0.5 bytes/element), SF blocks are
        # rounded up to 128-element granularity, epilogue is 16-bit output.
        sfa_tile_m = max(128, ((mma_tiler_mn[0] + 127) // 128) * 128)
        sfb_tile_n = max(128, ((mma_tiler_mn[1] + 127) // 128) * 128)
        ab_bytes = (mma_tiler_mn[0] * tile_k + mma_tiler_mn[1] * tile_k) // 2
        # SF: 128 * 4 elements per SF block, 1 byte each
        sf_bytes = (sfa_tile_m // 128) * 4 * 128 + (sfb_tile_n // 128) * 4 * 128
        epi_bytes = mma_tiler_mn[0] * mma_tiler_mn[1] * 2  # 16-bit output
        mbar_bytes = 1024
        smem_capacity = utils.get_smem_capacity_in_bytes("sm_120")
        if ab_bytes + sf_bytes + epi_bytes + mbar_bytes > smem_capacity:
            return False
        return True

    # ------------------------------------------------------------------
    # wrapper: compile-time entry point matching the SM100 interface
    # for FlashInfer's _compile_block_scaled_gemm
    # ------------------------------------------------------------------
    @cute.jit
    def wrapper(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        sf_m: cutlass.Int64,
        sf_n: cutlass.Int64,
        sf_k: cutlass.Int64,
        l: cutlass.Constexpr,
        a_sf_ptr: cute.Pointer,
        b_sf_ptr: cute.Pointer,
        alpha_tensor: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        current_stream,
        swap_ab: cutlass.Constexpr = False,
        epilogue_op: cutlass.Constexpr = lambda x: x,
    ):
        """Wrapper matching the SM100 compile interface."""
        m = cute.size(mA, mode=[0])
        k_raw = cute.size(mA, mode=[1])
        n = cute.size(mB, mode=[0])

        if cutlass.const_expr(
            mA.element_type == cutlass.Uint8 and mB.element_type == cutlass.Uint8
        ):
            k = k_raw * 2
            a_ptr = cute.recast_ptr(mA.iterator, dtype=cutlass.Float4E2M1FN)
            b_ptr = cute.recast_ptr(mB.iterator, dtype=cutlass.Float4E2M1FN)
        elif cutlass.const_expr(mA.element_type != mB.element_type):
            raise TypeError("Unsupported mixed input dtypes for block-scaled GEMM.")
        else:
            k = k_raw
            a_ptr = mA.iterator
            b_ptr = mB.iterator

        a_tensor = cute.make_tensor(
            a_ptr,
            layout=cute.make_ordered_layout((m, k, l), order=(1, 0, 2)),
        )
        b_tensor = cute.make_tensor(
            b_ptr,
            layout=cute.make_ordered_layout((n, k, l), order=(1, 0, 2)),
        )
        if cutlass.const_expr(swap_ab):
            c_tensor = cute.make_tensor(
                mC.iterator,
                layout=cute.make_ordered_layout((m, n, l), order=(0, 1, 2)),
            )
        else:
            c_tensor = cute.make_tensor(
                mC.iterator,
                layout=cute.make_ordered_layout((m, n, l), order=(1, 0, 2)),
            )
        if cutlass.const_expr(self.use_packed_sf_tma_layout):
            sfa_tensor = cute.make_tensor(
                a_sf_ptr,
                layout=task39_make_packed_sf_gmem_layout(
                    sf_m, sf_k, l, self.sf_vec_size
                ),
            )
            sfb_tensor = cute.make_tensor(
                b_sf_ptr,
                layout=task39_make_packed_sf_gmem_layout(
                    sf_n, sf_k, l, self.sf_vec_size
                ),
            )
        else:
            sfa_tensor = cute.make_tensor(
                a_sf_ptr,
                layout=cute.make_ordered_layout(
                    (32, 4, sf_m, 4, sf_k, l),
                    order=(2, 1, 4, 0, 3, 5),
                ),
            )
            sfb_tensor = cute.make_tensor(
                b_sf_ptr,
                layout=cute.make_ordered_layout(
                    (32, 4, sf_n, 4, sf_k, l),
                    order=(2, 1, 4, 0, 3, 5),
                ),
            )

        self(
            a_tensor,
            b_tensor,
            sfa_tensor,
            sfb_tensor,
            c_tensor,
            alpha_tensor,
            max_active_clusters,
            current_stream,
            epilogue_op,
        )


# Alias for FlashInfer integration
Sm120BlockScaledDenseGemmKernel = DenseGemmKernel

# Explicit task39-local alias used by the fork runner.  Keeping a distinct name
# makes benchmark logs and future structural diffs unambiguous while preserving
# the no-op baseline behavior in this first fork revision.
Task39Sm120GateUpDenseGemmKernel = DenseGemmKernel


class Task39Sm120GateUpDirectGridDenseGemmKernel(DenseGemmKernel):
    """Task39 scheduler experiment: launch one CTA per output tile.

    Upstream b12x uses CUTLASS' static persistent tile scheduler, which launches
    at most one wave of persistent CTAs and lets each CTA loop over multiple
    output tiles.  Large gate/up prefill has many tiles
    (`ceil(M/128) * ceil(34816/128)`), so this variant keeps the same b12x
    TMA/MMA mainloop but returns a full problem grid.  The existing scheduler
    then advances once and exits, making this a narrow structural fork point
    with exact output expected.
    """

    @staticmethod
    def _compute_grid(
        c,
        tile_shape_mnk: tuple,
        max_active_clusters,
    ) -> tuple:
        c_shape = cute.slice_(tile_shape_mnk, (None, None, 0))
        gc = cute.zipped_divide(c, tiler=c_shape)
        num_ctas_mnl = gc[(0, (None, None, None))].shape
        cluster_shape_mnl = (1, 1, 1)
        tile_sched_params = utils.PersistentTileSchedulerParams(
            num_ctas_mnl, cluster_shape_mnl
        )
        grid = (1, 1, cute.size(num_ctas_mnl))
        return tile_sched_params, grid


class Task39Sm120GateUpRasterNDenseGemmKernel(DenseGemmKernel):
    """Task39 scheduler experiment: keep persistence but rasterize along N."""

    @staticmethod
    def _compute_grid(
        c,
        tile_shape_mnk: tuple,
        max_active_clusters,
    ) -> tuple:
        c_shape = cute.slice_(tile_shape_mnk, (None, None, 0))
        gc = cute.zipped_divide(c, tiler=c_shape)
        num_ctas_mnl = gc[(0, (None, None, None))].shape
        cluster_shape_mnl = (1, 1, 1)
        tile_sched_params = utils.PersistentTileSchedulerParams(
            num_ctas_mnl, cluster_shape_mnl, 1, False
        )
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(
            tile_sched_params, max_active_clusters
        )
        return tile_sched_params, grid


class Task39Sm120GateUpUnfilteredSfDenseGemmKernel(DenseGemmKernel):
    """Task39 mainloop experiment: skip generic SF filter_zeros calls.

    This targets the fixed gate/up path where the accepted tile is 128x128 and
    the scale-factor tiles are exactly one tile per CTA in both M and N.  The
    rest of the b12x mainloop is unchanged.
    """

    def __init__(
        self,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        use_prefetch: bool = False,
        enable_pdl: bool = True,
    ):
        super().__init__(
            sf_vec_size,
            mma_tiler_mn,
            cluster_shape_mn,
            use_prefetch,
            enable_pdl,
        )
        self.use_unfiltered_sf_copy = True


class Task39Sm120GateUpFixedSfTileDenseGemmKernel(DenseGemmKernel):
    """Task39 mainloop experiment: specialize to one SF tile per CTA.

    For the fixed `128x128` gate/up tile, both `sfa_tiles_per_block` and
    `sfb_tiles_per_block` are one.  This preserves the upstream filtered SF copy
    semantics while removing the per-work-tile modulo/branch path.
    """

    def __init__(
        self,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        use_prefetch: bool = False,
        enable_pdl: bool = True,
    ):
        super().__init__(
            sf_vec_size,
            mma_tiler_mn,
            cluster_shape_mn,
            use_prefetch,
            enable_pdl,
        )
        self.use_fixed_single_sf_tile = True


class Task39Sm120GateUpAlphaOneEpilogueDenseGemmKernel(DenseGemmKernel):
    """Task39 epilogue experiment for alpha=1 identity output conversion."""

    def __init__(
        self,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        use_prefetch: bool = False,
        enable_pdl: bool = True,
    ):
        super().__init__(
            sf_vec_size,
            mma_tiler_mn,
            cluster_shape_mn,
            use_prefetch,
            enable_pdl,
        )
        self.assume_alpha_one_identity_epilogue = True


class Task39Sm120GateUpAlphaOneFixedSfDenseGemmKernel(DenseGemmKernel):
    """Task39 combined experiment: alpha=1 epilogue plus fixed SF tile."""

    def __init__(
        self,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        use_prefetch: bool = False,
        enable_pdl: bool = True,
    ):
        super().__init__(
            sf_vec_size,
            mma_tiler_mn,
            cluster_shape_mn,
            use_prefetch,
            enable_pdl,
        )
        self.assume_alpha_one_identity_epilogue = True
        self.use_fixed_single_sf_tile = True


class Task39Sm120GateUpAlphaOneSfPad1DenseGemmKernel(DenseGemmKernel):
    """Task39 SF-bank experiment: alpha=1 epilogue plus padded SF SMEM.

    This leaves A/B TMA layout, A/B ldmatrix consume, and the persistent
    scheduler unchanged.  Only SFA/SFB shared-memory staging uses a one-byte
    pad in the 32x4 scale-factor block row stride.
    """

    def __init__(
        self,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        use_prefetch: bool = False,
        enable_pdl: bool = True,
    ):
        super().__init__(
            sf_vec_size,
            mma_tiler_mn,
            cluster_shape_mn,
            use_prefetch,
            enable_pdl,
        )
        self.assume_alpha_one_identity_epilogue = True
        self.use_padded_sf_smem_layout = True
        self.sf_smem_mn_stride_pad = 1


class Task39Sm120GateUpAlphaOneSfPad4DenseGemmKernel(DenseGemmKernel):
    """Task39 SF-bank experiment with 4-byte aligned SF row padding."""

    def __init__(
        self,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        use_prefetch: bool = False,
        enable_pdl: bool = True,
    ):
        super().__init__(
            sf_vec_size,
            mma_tiler_mn,
            cluster_shape_mn,
            use_prefetch,
            enable_pdl,
        )
        self.assume_alpha_one_identity_epilogue = True
        self.use_padded_sf_smem_layout = True
        self.sf_smem_mn_stride_pad = 4


class Task39Sm120GateUpAlphaOneSfPad16DenseGemmKernel(DenseGemmKernel):
    """Task39 SF-bank experiment with 16-byte aligned SF row padding."""

    def __init__(
        self,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        use_prefetch: bool = False,
        enable_pdl: bool = True,
    ):
        super().__init__(
            sf_vec_size,
            mma_tiler_mn,
            cluster_shape_mn,
            use_prefetch,
            enable_pdl,
        )
        self.assume_alpha_one_identity_epilogue = True
        self.use_padded_sf_smem_layout = True
        self.sf_smem_mn_stride_pad = 16


class Task39Sm120GateUpAlphaOneSfSwizzle128DenseGemmKernel(DenseGemmKernel):
    """Task39 SF-bank experiment using a composed 128B SF SMEM swizzle."""

    def __init__(
        self,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        use_prefetch: bool = False,
        enable_pdl: bool = True,
    ):
        super().__init__(
            sf_vec_size,
            mma_tiler_mn,
            cluster_shape_mn,
            use_prefetch,
            enable_pdl,
        )
        self.assume_alpha_one_identity_epilogue = True
        self.use_swizzled_sf_smem_layout = True
        self.sf_smem_swizzle_bits = 3


class Task39Sm120GateUpAlphaOneSfSwizzle32DenseGemmKernel(DenseGemmKernel):
    """Task39 SF-bank experiment using a lighter composed SF SMEM swizzle."""

    def __init__(
        self,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        use_prefetch: bool = False,
        enable_pdl: bool = True,
    ):
        super().__init__(
            sf_vec_size,
            mma_tiler_mn,
            cluster_shape_mn,
            use_prefetch,
            enable_pdl,
        )
        self.assume_alpha_one_identity_epilogue = True
        self.use_swizzled_sf_smem_layout = True
        self.sf_smem_swizzle_bits = 1


class Task39Sm120GateUpAlphaOneSfSwizzle64DenseGemmKernel(DenseGemmKernel):
    """Task39 SF-bank experiment using a medium composed SF SMEM swizzle."""

    def __init__(
        self,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        use_prefetch: bool = False,
        enable_pdl: bool = True,
    ):
        super().__init__(
            sf_vec_size,
            mma_tiler_mn,
            cluster_shape_mn,
            use_prefetch,
            enable_pdl,
        )
        self.assume_alpha_one_identity_epilogue = True
        self.use_swizzled_sf_smem_layout = True
        self.sf_smem_swizzle_bits = 2


class Task39Sm120GateUpAlphaOneSfCopyFirstDenseGemmKernel(DenseGemmKernel):
    """Task39 copy-schedule experiment: issue SF SMEM copies before A/B."""

    def __init__(
        self,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        use_prefetch: bool = False,
        enable_pdl: bool = True,
    ):
        super().__init__(
            sf_vec_size,
            mma_tiler_mn,
            cluster_shape_mn,
            use_prefetch,
            enable_pdl,
        )
        self.assume_alpha_one_identity_epilogue = True
        self.copy_sf_before_ab = True


class Task39Sm120GateUpAlphaOneSfCopyU32DenseGemmKernel(DenseGemmKernel):
    """Task39 SF-bank experiment: recast SF copy views to 32-bit words."""

    def __init__(
        self,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        use_prefetch: bool = False,
        enable_pdl: bool = True,
    ):
        super().__init__(
            sf_vec_size,
            mma_tiler_mn,
            cluster_shape_mn,
            use_prefetch,
            enable_pdl,
        )
        self.assume_alpha_one_identity_epilogue = True
        self.use_vectorized_sf_copy_u32 = True


class Task39Sm120GateUpAlphaOneSfCopyU32HoistSfaDenseGemmKernel(DenseGemmKernel):
    """Task39 mainloop experiment: hoist invariant SFA atom setup out of N loop."""

    def __init__(
        self,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        use_prefetch: bool = False,
        enable_pdl: bool = True,
    ):
        super().__init__(
            sf_vec_size,
            mma_tiler_mn,
            cluster_shape_mn,
            use_prefetch,
            enable_pdl,
        )
        self.assume_alpha_one_identity_epilogue = True
        self.use_vectorized_sf_copy_u32 = True
        self.hoist_sfa_set = True


class Task39Sm120GateUpAlphaOneSfCopyU64HoistSfaDenseGemmKernel(DenseGemmKernel):
    """Task39 SF-copy experiment: 64-bit SF copies plus SFA setup hoist."""

    def __init__(
        self,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        use_prefetch: bool = False,
        enable_pdl: bool = True,
    ):
        super().__init__(
            sf_vec_size,
            mma_tiler_mn,
            cluster_shape_mn,
            use_prefetch,
            enable_pdl,
        )
        self.assume_alpha_one_identity_epilogue = True
        self.sf_copy_vector_bits = 64
        self.hoist_sfa_set = True


class Task39Sm120GateUpAlphaOneSfCopyU128HoistSfaDenseGemmKernel(DenseGemmKernel):
    """Task39 SF-copy experiment: 128-bit SF copies plus SFA setup hoist."""

    def __init__(
        self,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        use_prefetch: bool = False,
        enable_pdl: bool = True,
    ):
        super().__init__(
            sf_vec_size,
            mma_tiler_mn,
            cluster_shape_mn,
            use_prefetch,
            enable_pdl,
        )
        self.assume_alpha_one_identity_epilogue = True
        self.sf_copy_vector_bits = 128
        self.hoist_sfa_set = True


class Task39Sm120GateUpAlphaOneSfCopyU32Bits64HoistSfaDenseGemmKernel(
    DenseGemmKernel
):
    """Task39 SF-copy experiment: force 64-bit S2R copy atoms for SF."""

    def __init__(
        self,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        use_prefetch: bool = False,
        enable_pdl: bool = True,
    ):
        super().__init__(
            sf_vec_size,
            mma_tiler_mn,
            cluster_shape_mn,
            use_prefetch,
            enable_pdl,
        )
        self.assume_alpha_one_identity_epilogue = True
        self.use_vectorized_sf_copy_u32 = True
        self.sf_num_bits_per_copy = 64
        self.hoist_sfa_set = True


class Task39Sm120GateUpAlphaOneSfCopyU32Bits128HoistSfaDenseGemmKernel(
    DenseGemmKernel
):
    """Task39 SF-copy experiment: force 128-bit S2R copy atoms for SF."""

    def __init__(
        self,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        use_prefetch: bool = False,
        enable_pdl: bool = True,
    ):
        super().__init__(
            sf_vec_size,
            mma_tiler_mn,
            cluster_shape_mn,
            use_prefetch,
            enable_pdl,
        )
        self.assume_alpha_one_identity_epilogue = True
        self.use_vectorized_sf_copy_u32 = True
        self.sf_num_bits_per_copy = 128
        self.hoist_sfa_set = True


class Task39Sm120GateUpAlphaOneSfCopyU32ManualSfaV4HoistSfaDenseGemmKernel(
    DenseGemmKernel
):
    """Task39 SF-copy experiment: manual SFA gap-2 load via ld.shared.v4.u32."""

    def __init__(
        self,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        use_prefetch: bool = False,
        enable_pdl: bool = True,
    ):
        super().__init__(
            sf_vec_size,
            mma_tiler_mn,
            cluster_shape_mn,
            use_prefetch,
            enable_pdl,
        )
        self.assume_alpha_one_identity_epilogue = True
        self.use_vectorized_sf_copy_u32 = True
        self.use_manual_sfa_gap2_v4_copy = True
        self.hoist_sfa_set = True


class Task39Sm120GateUpAlphaOneSfCopyU32ManualSfbV4HoistSfaDenseGemmKernel(
    DenseGemmKernel
):
    """Task39 SF-copy experiment: manual SFB contiguous load via ld.shared.v4.u32."""

    def __init__(
        self,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        use_prefetch: bool = False,
        enable_pdl: bool = True,
    ):
        super().__init__(
            sf_vec_size,
            mma_tiler_mn,
            cluster_shape_mn,
            use_prefetch,
            enable_pdl,
        )
        self.assume_alpha_one_identity_epilogue = True
        self.use_vectorized_sf_copy_u32 = True
        self.use_manual_sfb_v4_copy = True
        self.hoist_sfa_set = True


class Task39Sm120GateUpAlphaOneSfCopyU32ManualSfabV4HoistSfaDenseGemmKernel(
    DenseGemmKernel
):
    """Task39 SF-copy experiment: manual SFA and SFB vector shared loads."""

    def __init__(
        self,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        use_prefetch: bool = False,
        enable_pdl: bool = True,
    ):
        super().__init__(
            sf_vec_size,
            mma_tiler_mn,
            cluster_shape_mn,
            use_prefetch,
            enable_pdl,
        )
        self.assume_alpha_one_identity_epilogue = True
        self.use_vectorized_sf_copy_u32 = True
        self.use_manual_sfa_gap2_v4_copy = True
        self.use_manual_sfb_v4_copy = True
        self.hoist_sfa_set = True


class Task39Sm120GateUpAlphaOneSfCopyU32ManualSfbV4FixedDenseGemmKernel(
    DenseGemmKernel
):
    """Task39 SF-copy experiment: fixed contiguous SFB vector load."""

    def __init__(
        self,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        use_prefetch: bool = False,
        enable_pdl: bool = True,
    ):
        super().__init__(
            sf_vec_size,
            mma_tiler_mn,
            cluster_shape_mn,
            use_prefetch,
            enable_pdl,
        )
        self.assume_alpha_one_identity_epilogue = True
        self.use_vectorized_sf_copy_u32 = True
        self.use_manual_sfb_v4_fixed_copy = True
        self.hoist_sfa_set = True


class Task39Sm120GateUpAlphaOneSfCopyU32ManualSfbV4FixedPerm02461357DenseGemmKernel(
    Task39Sm120GateUpAlphaOneSfCopyU32ManualSfbV4FixedDenseGemmKernel
):
    """Task39 SF-copy experiment: fixed SFB vector load with 02461357 perm."""

    def __init__(
        self,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        use_prefetch: bool = False,
        enable_pdl: bool = True,
    ):
        super().__init__(
            sf_vec_size,
            mma_tiler_mn,
            cluster_shape_mn,
            use_prefetch,
            enable_pdl,
        )
        self.manual_sfb_v4_fixed_perm = 1


class Task39Sm120GateUpAlphaOneSfCopyU32ManualSfbV4FixedPerm04152637DenseGemmKernel(
    Task39Sm120GateUpAlphaOneSfCopyU32ManualSfbV4FixedDenseGemmKernel
):
    """Task39 SF-copy experiment: fixed SFB vector load with 04152637 perm."""

    def __init__(
        self,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        use_prefetch: bool = False,
        enable_pdl: bool = True,
    ):
        super().__init__(
            sf_vec_size,
            mma_tiler_mn,
            cluster_shape_mn,
            use_prefetch,
            enable_pdl,
        )
        self.manual_sfb_v4_fixed_perm = 2


class Task39Sm120GateUpAlphaOneSfCopyU32ManualSfbV4FixedPerm04261537DenseGemmKernel(
    Task39Sm120GateUpAlphaOneSfCopyU32ManualSfbV4FixedDenseGemmKernel
):
    """Task39 SF-copy experiment: fixed SFB vector load with 04261537 perm."""

    def __init__(
        self,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        use_prefetch: bool = False,
        enable_pdl: bool = True,
    ):
        super().__init__(
            sf_vec_size,
            mma_tiler_mn,
            cluster_shape_mn,
            use_prefetch,
            enable_pdl,
        )
        self.manual_sfb_v4_fixed_perm = 3


class Task39Sm120GateUpAlphaOneSfCopyU32ManualSfbV4FixedEvenDenseGemmKernel(
    Task39Sm120GateUpAlphaOneSfCopyU32ManualSfbV4FixedDenseGemmKernel
):
    """Task39 SF-copy experiment: SFB vector load from even source lanes."""

    def __init__(
        self,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        use_prefetch: bool = False,
        enable_pdl: bool = True,
    ):
        super().__init__(
            sf_vec_size,
            mma_tiler_mn,
            cluster_shape_mn,
            use_prefetch,
            enable_pdl,
        )
        self.manual_sfb_v4_fixed_perm = 4


class Task39Sm120GateUpAlphaOneSfCopyU32ManualSfbV4FixedEven048c26aeDenseGemmKernel(
    Task39Sm120GateUpAlphaOneSfCopyU32ManualSfbV4FixedDenseGemmKernel
):
    """Task39 SF-copy experiment: even SFB load with 048c26ae perm."""

    def __init__(
        self,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        use_prefetch: bool = False,
        enable_pdl: bool = True,
    ):
        super().__init__(
            sf_vec_size,
            mma_tiler_mn,
            cluster_shape_mn,
            use_prefetch,
            enable_pdl,
        )
        self.manual_sfb_v4_fixed_perm = 5


class Task39Sm120GateUpAlphaOneSfCopyU32ManualSfbV4FixedEven082a4c6eDenseGemmKernel(
    Task39Sm120GateUpAlphaOneSfCopyU32ManualSfbV4FixedDenseGemmKernel
):
    """Task39 SF-copy experiment: even SFB load with 082a4c6e perm."""

    def __init__(
        self,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        use_prefetch: bool = False,
        enable_pdl: bool = True,
    ):
        super().__init__(
            sf_vec_size,
            mma_tiler_mn,
            cluster_shape_mn,
            use_prefetch,
            enable_pdl,
        )
        self.manual_sfb_v4_fixed_perm = 6


class Task39Sm120GateUpAlphaOneSfCopyU32HoistSfaUnfilteredDenseGemmKernel(
    DenseGemmKernel
):
    """Task39 scoreboard experiment: skip SF filters on fixed full tiles."""

    def __init__(
        self,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        use_prefetch: bool = False,
        enable_pdl: bool = True,
    ):
        super().__init__(
            sf_vec_size,
            mma_tiler_mn,
            cluster_shape_mn,
            use_prefetch,
            enable_pdl,
        )
        self.assume_alpha_one_identity_epilogue = True
        self.use_vectorized_sf_copy_u32 = True
        self.hoist_sfa_set = True
        self.use_unfiltered_sf_copy = True


class Task39Sm120GateUpAlphaOneSfCopyU32HoistSfaFixedSfDenseGemmKernel(
    DenseGemmKernel
):
    """Task39 scoreboard experiment: specialize SF tile selection for 128x128."""

    def __init__(
        self,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        use_prefetch: bool = False,
        enable_pdl: bool = True,
    ):
        super().__init__(
            sf_vec_size,
            mma_tiler_mn,
            cluster_shape_mn,
            use_prefetch,
            enable_pdl,
        )
        self.assume_alpha_one_identity_epilogue = True
        self.use_vectorized_sf_copy_u32 = True
        self.hoist_sfa_set = True
        self.use_fixed_single_sf_tile = True


class Task39Sm120GateUpAlphaOneSfCopyU32HoistSfaFixedSfUnfilteredDenseGemmKernel(
    DenseGemmKernel
):
    """Task39 scoreboard experiment: fixed SF tile plus unfiltered SF copies."""

    def __init__(
        self,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        use_prefetch: bool = False,
        enable_pdl: bool = True,
    ):
        super().__init__(
            sf_vec_size,
            mma_tiler_mn,
            cluster_shape_mn,
            use_prefetch,
            enable_pdl,
        )
        self.assume_alpha_one_identity_epilogue = True
        self.use_vectorized_sf_copy_u32 = True
        self.hoist_sfa_set = True
        self.use_fixed_single_sf_tile = True
        self.use_unfiltered_sf_copy = True


class Task39Sm120GateUpAlphaOneSfCopyU32HoistSfaPackedSfTmaDenseGemmKernel(
    DenseGemmKernel
):
    """Task39 experiment: packed-block scale GMEM layout loaded via TMA."""

    def __init__(
        self,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        use_prefetch: bool = False,
        enable_pdl: bool = True,
    ):
        super().__init__(
            sf_vec_size,
            mma_tiler_mn,
            cluster_shape_mn,
            use_prefetch,
            enable_pdl,
        )
        self.assume_alpha_one_identity_epilogue = True
        self.use_vectorized_sf_copy_u32 = True
        self.hoist_sfa_set = True
        self.use_packed_sf_tma_layout = True


class Task39Sm120GateUpAlphaOneSfCopyU32HoistSfaBankMajorSfDenseGemmKernel(
    DenseGemmKernel
):
    """Task39 experiment: bank-major affine SF SMEM layout."""

    def __init__(
        self,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        use_prefetch: bool = False,
        enable_pdl: bool = True,
    ):
        super().__init__(
            sf_vec_size,
            mma_tiler_mn,
            cluster_shape_mn,
            use_prefetch,
            enable_pdl,
        )
        self.assume_alpha_one_identity_epilogue = True
        self.use_vectorized_sf_copy_u32 = True
        self.hoist_sfa_set = True
        self.use_bankmajor_sf_smem_layout = True


class Task39Sm120GateUpAlphaOneHoistSfaBankMajorSfDenseGemmKernel(DenseGemmKernel):
    """Task39 experiment: bank-major SF SMEM layout without u32 recast."""

    def __init__(
        self,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        use_prefetch: bool = False,
        enable_pdl: bool = True,
    ):
        super().__init__(
            sf_vec_size,
            mma_tiler_mn,
            cluster_shape_mn,
            use_prefetch,
            enable_pdl,
        )
        self.assume_alpha_one_identity_epilogue = True
        self.hoist_sfa_set = True
        self.use_bankmajor_sf_smem_layout = True


class Task39Sm120GateUpAlphaOneSfCopyU32HoistSfaFragMajorSfDenseGemmKernel(
    DenseGemmKernel
):
    """Task39 experiment: fragment-major SF SMEM layout for scalar LDS banks."""

    def __init__(
        self,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        use_prefetch: bool = False,
        enable_pdl: bool = True,
    ):
        super().__init__(
            sf_vec_size,
            mma_tiler_mn,
            cluster_shape_mn,
            use_prefetch,
            enable_pdl,
        )
        self.assume_alpha_one_identity_epilogue = True
        self.use_vectorized_sf_copy_u32 = True
        self.hoist_sfa_set = True
        self.use_fragmajor_sf_smem_layout = True


class Task39Sm120GateUpAlphaOneSfCopyU32WaitLateDenseGemmKernel(DenseGemmKernel):
    """Task39 mainloop experiment: move next-stage wait after last MMA issue."""

    def __init__(
        self,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        use_prefetch: bool = False,
        enable_pdl: bool = True,
    ):
        super().__init__(
            sf_vec_size,
            mma_tiler_mn,
            cluster_shape_mn,
            use_prefetch,
            enable_pdl,
        )
        self.assume_alpha_one_identity_epilogue = True
        self.use_vectorized_sf_copy_u32 = True
        self.wait_after_last_mma = True


class Task39Sm120GateUpAlphaOneSfCopyU32PreloadKDenseGemmKernel(DenseGemmKernel):
    """Task39 mainloop experiment: preload both 64K blocks before MMA issue."""

    def __init__(
        self,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        use_prefetch: bool = False,
        enable_pdl: bool = True,
    ):
        super().__init__(
            sf_vec_size,
            mma_tiler_mn,
            cluster_shape_mn,
            use_prefetch,
            enable_pdl,
        )
        self.assume_alpha_one_identity_epilogue = True
        self.use_vectorized_sf_copy_u32 = True
        self.wait_after_last_mma = True
        self.preload_all_k_blocks = True


class Task39Sm120GateUpAlphaOneSfCopyU32PreloadKOverlapDenseGemmKernel(
    DenseGemmKernel
):
    """Task39 mainloop experiment: preload k-blocks but wait before final block."""

    def __init__(
        self,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        use_prefetch: bool = False,
        enable_pdl: bool = True,
    ):
        super().__init__(
            sf_vec_size,
            mma_tiler_mn,
            cluster_shape_mn,
            use_prefetch,
            enable_pdl,
        )
        self.assume_alpha_one_identity_epilogue = True
        self.use_vectorized_sf_copy_u32 = True
        self.preload_all_k_blocks = True
        self.preload_wait_before_last_mma = True


class Task39Sm120GateUpAlphaOneSfCopyU32PreloadSfDenseGemmKernel(DenseGemmKernel):
    """Task39 mainloop experiment: preload only next SF fragment."""

    def __init__(
        self,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        use_prefetch: bool = False,
        enable_pdl: bool = True,
    ):
        super().__init__(
            sf_vec_size,
            mma_tiler_mn,
            cluster_shape_mn,
            use_prefetch,
            enable_pdl,
        )
        self.assume_alpha_one_identity_epilogue = True
        self.use_vectorized_sf_copy_u32 = True
        self.preload_sf_k_blocks = True


class Task39Sm120GateUpAlphaOneSfCopyU32HoistSfaPreloadSfDenseGemmKernel(
    DenseGemmKernel
):
    """Task39 mainloop experiment: hoist SFA setup and preload next SF fragment."""

    def __init__(
        self,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        use_prefetch: bool = False,
        enable_pdl: bool = True,
    ):
        super().__init__(
            sf_vec_size,
            mma_tiler_mn,
            cluster_shape_mn,
            use_prefetch,
            enable_pdl,
        )
        self.assume_alpha_one_identity_epilogue = True
        self.use_vectorized_sf_copy_u32 = True
        self.hoist_sfa_set = True
        self.preload_sf_k_blocks = True


class Task39Sm120GateUpAlphaOneSfCopyU32PreloadSfPretryDenseGemmKernel(
    DenseGemmKernel
):
    """Task39 mainloop experiment: pretry next-stage wait under final k-block MMA."""

    def __init__(
        self,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        use_prefetch: bool = False,
        enable_pdl: bool = True,
    ):
        super().__init__(
            sf_vec_size,
            mma_tiler_mn,
            cluster_shape_mn,
            use_prefetch,
            enable_pdl,
        )
        self.assume_alpha_one_identity_epilogue = True
        self.use_vectorized_sf_copy_u32 = True
        self.preload_sf_k_blocks = True
        self.pretry_wait_after_last_mma = True


class Task39Sm120GateUpAlphaOneFixedSfCopyU32DenseGemmKernel(DenseGemmKernel):
    """Task39 combined experiment: fixed SF tile plus 32-bit SF copies."""

    def __init__(
        self,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        use_prefetch: bool = False,
        enable_pdl: bool = True,
    ):
        super().__init__(
            sf_vec_size,
            mma_tiler_mn,
            cluster_shape_mn,
            use_prefetch,
            enable_pdl,
        )
        self.assume_alpha_one_identity_epilogue = True
        self.use_fixed_single_sf_tile = True
        self.use_vectorized_sf_copy_u32 = True


class Task39Sm120GateUpAlphaOneSfTvSwapDenseGemmKernel(DenseGemmKernel):
    """Task39 SF copy experiment: swap SFA/SFB copy TV layouts together."""

    def __init__(
        self,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        use_prefetch: bool = False,
        enable_pdl: bool = True,
    ):
        super().__init__(
            sf_vec_size,
            mma_tiler_mn,
            cluster_shape_mn,
            use_prefetch,
            enable_pdl,
        )
        self.assume_alpha_one_identity_epilogue = True
        self.swap_sfa_tv_layout = True
        self.swap_sfb_tv_layout = True


class Task39Sm120GateUpAlphaOneSfaTvSwapDenseGemmKernel(DenseGemmKernel):
    """Task39 SF copy experiment: swap only SFA copy TV layout."""

    def __init__(
        self,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        use_prefetch: bool = False,
        enable_pdl: bool = True,
    ):
        super().__init__(
            sf_vec_size,
            mma_tiler_mn,
            cluster_shape_mn,
            use_prefetch,
            enable_pdl,
        )
        self.assume_alpha_one_identity_epilogue = True
        self.swap_sfa_tv_layout = True


class Task39Sm120GateUpAlphaOneSfbTvSwapDenseGemmKernel(DenseGemmKernel):
    """Task39 SF copy experiment: swap only SFB copy TV layout."""

    def __init__(
        self,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        use_prefetch: bool = False,
        enable_pdl: bool = True,
    ):
        super().__init__(
            sf_vec_size,
            mma_tiler_mn,
            cluster_shape_mn,
            use_prefetch,
            enable_pdl,
        )
        self.assume_alpha_one_identity_epilogue = True
        self.swap_sfb_tv_layout = True


class Task39Sm120GateUpAlphaOneSfbTvSwapAtomDenseGemmKernel(DenseGemmKernel):
    """Task39 SFB copy experiment: TV swap paired with SFB atom-layout swap."""

    def __init__(
        self,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        use_prefetch: bool = False,
        enable_pdl: bool = True,
    ):
        super().__init__(
            sf_vec_size,
            mma_tiler_mn,
            cluster_shape_mn,
            use_prefetch,
            enable_pdl,
        )
        self.assume_alpha_one_identity_epilogue = True
        self.swap_sfb_tv_layout = True
        self.swap_sfb_atom_layout = True


class Task39Sm120GateUpAlphaOneSfbTvSwapGroupDenseGemmKernel(DenseGemmKernel):
    """Task39 SFB copy experiment: TV swap paired with fragment regrouping."""

    def __init__(
        self,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        use_prefetch: bool = False,
        enable_pdl: bool = True,
    ):
        super().__init__(
            sf_vec_size,
            mma_tiler_mn,
            cluster_shape_mn,
            use_prefetch,
            enable_pdl,
        )
        self.assume_alpha_one_identity_epilogue = True
        self.swap_sfb_tv_layout = True
        self.swap_sfb_fragment_grouping = True


class Task39Sm120GateUpOccupancy2DenseGemmKernel(DenseGemmKernel):
    """Task39 mainloop experiment: compute stages with occupancy=2."""

    def __init__(
        self,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        use_prefetch: bool = False,
        enable_pdl: bool = True,
    ):
        super().__init__(
            sf_vec_size,
            mma_tiler_mn,
            cluster_shape_mn,
            use_prefetch,
            enable_pdl,
        )
        self.occupancy = 2


class _DenseGemmLaunch:
    def __init__(
        self,
        m: int,
        n: int,
        k: int,
        l: int,
        a_major: str,
        b_major: str,
        c_major: str,
        ab_dtype: torch.dtype,
        sf_dtype: torch.dtype,
        c_dtype: torch.dtype,
        alpha_dtype: torch.dtype,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        sm_count: int,
        sm_version: str,
    ):
        self._m = m
        self._n = n
        self._k = k
        self._l = l
        self._a_major = a_major
        self._b_major = b_major
        self._c_major = c_major
        self._ab_dtype = ab_dtype
        self._sf_dtype = sf_dtype
        self._c_dtype = c_dtype
        self._alpha_dtype = alpha_dtype
        self._sf_vec_size = sf_vec_size
        self._mma_tiler_mn = mma_tiler_mn
        self._cluster_shape_mn = cluster_shape_mn

        if sm_version != "sm_120":
            raise ValueError(
                f"dense_gemm launch only supports sm_120, got {sm_version}"
            )

        if not DenseGemmKernel.can_implement(
            ab_dtype,
            sf_dtype,
            sf_vec_size,
            c_dtype,
            mma_tiler_mn,
            cluster_shape_mn,
            m,
            n,
            k,
            l,
            a_major,
            b_major,
            c_major,
        ):
            raise TypeError(
                "dense_gemm launch is unsupported with "
                f"{ab_dtype}, {sf_dtype}, {sf_vec_size}, {c_dtype}, "
                f"{mma_tiler_mn}, {cluster_shape_mn}, {m}, {n}, {k}, {l}, "
                f"{a_major}, {b_major}, {c_major}"
            )

        self._max_active_clusters = min(
            get_max_active_clusters(
                self._cluster_shape_mn[0] * self._cluster_shape_mn[1]
            ),
            sm_count,
        )

    @cute.jit
    def __call__(
        self,
        a_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        sfa_ptr: cute.Pointer,
        sfb_ptr: cute.Pointer,
        c_ptr: cute.Pointer,
        alpha_ptr: cute.Pointer,
        current_stream: cuda.CUstream,
    ):
        a_tensor = cute.make_tensor(
            a_ptr,
            layout=cute.make_ordered_layout(
                (self._m, self._k, self._l),
                order=(0, 1, 2) if self._a_major == "m" else (1, 0, 2),
            ),
        )
        b_tensor = cute.make_tensor(
            b_ptr,
            layout=cute.make_ordered_layout(
                (self._n, self._k, self._l),
                order=(0, 1, 2) if self._b_major == "n" else (1, 0, 2),
            ),
        )
        c_tensor = cute.make_tensor(
            c_ptr,
            layout=cute.make_ordered_layout(
                (self._m, self._n, self._l),
                order=(0, 1, 2) if self._c_major == "m" else (1, 0, 2),
            ),
        )
        alpha_tensor = cute.make_tensor(
            alpha_ptr,
            layout=cute.make_ordered_layout((1,), order=(0,)),
        )
        sfa_tensor = cute.make_tensor(sfa_ptr, layout=cute.make_layout((1,)))
        sfb_tensor = cute.make_tensor(sfb_ptr, layout=cute.make_layout((1,)))

        DenseGemmKernel(
            sf_vec_size=self._sf_vec_size,
            mma_tiler_mn=self._mma_tiler_mn,
            cluster_shape_mn=self._cluster_shape_mn,
        )(
            a_tensor,
            b_tensor,
            sfa_tensor,
            sfb_tensor,
            c_tensor,
            alpha_tensor,
            self._max_active_clusters,
            current_stream,
        )


@functools.cache
def _get_compiled_dense_gemm(
    m: int,
    n: int,
    k: int,
    l: int,
    a_major: str,
    b_major: str,
    c_major: str,
    ab_dtype: Type[cutlass.Numeric],
    sf_dtype: Type[cutlass.Numeric],
    c_dtype: Type[cutlass.Numeric],
    alpha_dtype: Type[cutlass.Numeric],
    sf_vec_size: int,
    mma_tiler_mn: Tuple[int, int],
    cluster_shape_mn: Tuple[int, int],
    sm_count: int,
    sm_version: str,
) -> Callable:
    def _make_runtime_pointers(
        input_tensors: Optional[List[torch.Tensor]],
    ) -> List[cute.Pointer]:
        if input_tensors is None:
            (
                a_data_ptr,
                b_data_ptr,
                sfa_data_ptr,
                sfb_data_ptr,
                c_data_ptr,
                alpha_data_ptr,
            ) = [16 for _ in range(6)]
        else:
            (
                a_tensor_gpu,
                b_tensor_gpu,
                sfa_tensor_gpu,
                sfb_tensor_gpu,
                c_tensor_gpu,
                alpha_tensor_gpu,
            ) = input_tensors
            (
                a_data_ptr,
                b_data_ptr,
                sfa_data_ptr,
                sfb_data_ptr,
                c_data_ptr,
                alpha_data_ptr,
            ) = (
                a_tensor_gpu.data_ptr(),
                b_tensor_gpu.data_ptr(),
                sfa_tensor_gpu.data_ptr(),
                sfb_tensor_gpu.data_ptr(),
                c_tensor_gpu.data_ptr(),
                alpha_tensor_gpu.data_ptr(),
            )

        return [
            make_ptr(ab_dtype, a_data_ptr, cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(ab_dtype, b_data_ptr, cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(sf_dtype, sfa_data_ptr, cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(sf_dtype, sfb_data_ptr, cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(c_dtype, c_data_ptr, cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(
                alpha_dtype, alpha_data_ptr, cute.AddressSpace.gmem, assumed_align=16
            ),
        ]

    compiled_kernel = cute.compile(
        _DenseGemmLaunch(
            m=m,
            n=n,
            k=k,
            l=l,
            a_major=a_major,
            b_major=b_major,
            c_major=c_major,
            ab_dtype=ab_dtype,
            sf_dtype=sf_dtype,
            c_dtype=c_dtype,
            alpha_dtype=alpha_dtype,
            sf_vec_size=sf_vec_size,
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
            sm_count=sm_count,
            sm_version=sm_version,
        ),
        *_make_runtime_pointers(None),
        current_cuda_stream(),
    )

    def tensor_api(
        a_tensor_gpu: torch.Tensor,
        b_tensor_gpu: torch.Tensor,
        sfa_tensor_gpu: torch.Tensor,
        sfb_tensor_gpu: torch.Tensor,
        c_tensor_gpu: Optional[torch.Tensor] = None,
        alpha_tensor_gpu: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if c_tensor_gpu is None:
            c_tensor_gpu = torch.empty(
                (m, n, l),
                dtype=cutlass_to_torch_dtype(c_dtype),
                device=a_tensor_gpu.device,
            )
        if alpha_tensor_gpu is None:
            alpha_tensor_gpu = torch.ones(
                (1,),
                dtype=torch.float32,
                device=a_tensor_gpu.device,
            )

        nonlocal compiled_kernel
        compiled_kernel(
            *_make_runtime_pointers(
                [
                    a_tensor_gpu,
                    b_tensor_gpu,
                    sfa_tensor_gpu,
                    sfb_tensor_gpu,
                    c_tensor_gpu,
                    alpha_tensor_gpu,
                ]
            ),
            current_cuda_stream(),
        )
        return c_tensor_gpu

    return tensor_api


def _select_default_mma_tiler_mn(m: int, n: int, sm_count: int) -> Tuple[int, int]:
    coarse_tile = (128, 128)
    coarse_tiles = ((m + coarse_tile[0] - 1) // coarse_tile[0]) * (
        (n + coarse_tile[1] - 1) // coarse_tile[1]
    )
    if m <= 128 and coarse_tiles < max(1, sm_count // 2):
        if n > 1536:
            return (64, 128)
        medium_tile = (128, 64)
        medium_tiles = ((m + medium_tile[0] - 1) // medium_tile[0]) * (
            (n + medium_tile[1] - 1) // medium_tile[1]
        )
        if medium_tiles < max(1, sm_count // 2):
            return (64, 64)
        return (128, 64)
    return (128, 128)


def dense_gemm(
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: Optional[torch.Tensor] = None,
    *,
    ab_dtype: str,
    sf_dtype: str,
    c_dtype: str,
    sf_vec_size: int,
    sm_count: Optional[int] = None,
    mma_tiler_mn: Optional[Tuple[int, int]] = None,
    cluster_shape_mn: Tuple[int, int] = (1, 1),
    alpha: Optional[torch.Tensor] = None,
    alpha_dtype: Optional[str] = None,
) -> torch.Tensor:
    """Execute dense block-scaled GEMM for one expert-major batch stack."""
    a_torch, sfa_torch = lhs
    b_torch, sfb_torch = rhs

    m, k, l = a_torch.shape
    n, _, _ = b_torch.shape
    if ab_dtype == "float4_e2m1fn":
        k *= 2

    if sm_count is None:
        sm_count = get_num_sm(a_torch.device)
    if mma_tiler_mn is None:
        mma_tiler_mn = _select_default_mma_tiler_mn(m, n, sm_count)
    if alpha_dtype is None:
        alpha_dtype = "float32" if alpha is None else str(alpha.dtype).split(".")[-1]

    return _get_compiled_dense_gemm(
        m=m,
        n=n,
        k=k,
        l=l,
        a_major="k",
        b_major="k",
        c_major="n",
        ab_dtype=get_cutlass_dtype(ab_dtype),
        sf_dtype=get_cutlass_dtype(sf_dtype),
        c_dtype=get_cutlass_dtype(c_dtype),
        alpha_dtype=get_cutlass_dtype(alpha_dtype),
        sf_vec_size=sf_vec_size,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        sm_count=sm_count,
        sm_version="sm_120",
    )(
        a_tensor_gpu=a_torch,
        b_tensor_gpu=b_torch,
        sfa_tensor_gpu=sfa_torch,
        sfb_tensor_gpu=sfb_torch,
        c_tensor_gpu=out,
        alpha_tensor_gpu=alpha,
    )
