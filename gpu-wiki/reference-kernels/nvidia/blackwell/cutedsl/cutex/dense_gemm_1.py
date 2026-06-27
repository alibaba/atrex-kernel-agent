# SPDX-FileCopyrightText: Copyright (c) 2024 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
Low-level mbarrier API CuTeDSL Dense GEMM:

| Parameter              | Value         |
|------------------------|---------------|
| MMA Instruction Shape  | (128, 256, 16)|
| MMA Tiler             | (128, 256, 64)|
| Threads per CTA        | 128           |
| Pipeline Stages        | 1 (AB), 1 (acc)|

This version uses low-level mbarrier API instead of Pipeline abstractions.
"""

import argparse
from typing import Tuple

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl.cutlass import if_generate

"""
The first tutorial GEMM demonstrating a simple kernel implementation in CuTeDSL

This dense GEMM kernel is implemented in just over 200 lines of code.
With large tile sizes, it can achieve very high performance on 8k×8k×8k problem sizes.
It can serve as a starting point to help users quickly experiment
with optimizations for challenges that may arise with other problem sizes.

To run this example:
.. code-block:: bash

    python examples/blackwell/tutorial_gemm/fp16_gemm_0.py  \
      --mnk 8192,8192,8192

Constraints for this example:
* The problem size of m and n must be divisible by the tile size m & n (128, 256)
"""

# Why K=64 (instruction K=16)?
# - Memory bandwidth & alignment: GPU memory/L1/L2 cache lines are typically 128 bytes.
# - For FP16/BF16 (2 bytes/element), 128 bytes = 64 elements.
# - This ensures efficient vectorized transfers via TMA (Tensor Memory Accelerator).

io_dtype = cutlass.Float16
acc_dtype = cutlass.Float32
mma_inst_shape_mnk = (128, 256, 16)
mma_tiler_mnk = (128, 256, 64)
threads_per_cta = 128

# Pipeline stage configuration - minimal pipelining (single stage)
ab_stages = 1


@cute.struct
class SharedStorage:
    ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, ab_stages * 2]
    acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 1]
    tmem_holding_buf: cutlass.Int32


@cute.kernel
def kernel(
    tiled_mma: cute.TiledMma,
    tma_atom_a: cute.CopyAtom,
    mA_mkl: cute.Tensor,
    tma_atom_b: cute.CopyAtom,
    mB_nkl: cute.Tensor,
    mC_mnl: cute.Tensor,
    a_smem_layout: cute.ComposedLayout,
    b_smem_layout: cute.ComposedLayout,
):
    # Current thread/warp/block coordinates
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    bidx, bidy, _ = cute.arch.block_idx()
    mma_coord_mnk = (bidx, bidy, None)

    #
    # 1. Prepare args
    #

    # Allocate SMEM
    # smem.allocate() vs smem.allocate_tensor():
    #   - smem.allocate(SharedStorage): Allocates a struct in SMEM for barrier pointers (MemRange)
    #   - smem.allocate_tensor(): Allocates a tensor in SMEM with specific layout and swizzle
    #     * layout=a_smem_layout.outer: The outer layout (tiled shape)
    #     * swizzle=a_smem_layout.inner: The inner swizzle pattern for bank conflict avoidance
    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)
    # sA: SMEM tensor with shape ((128,16),1,4,1) = (mma_atom, tma_rest_m, tma_rest_k, buffer_stages)
    # My Notation: ((128,16),1,4,1) = mma_atom, *res_tma_rest, buffer_stages
    #     smem_desc is one descriptor for the whole block (128, 16)
    sA = smem.allocate_tensor(
        element_type=io_dtype,
        layout=a_smem_layout.outer,
        byte_alignment=128,
        swizzle=a_smem_layout.inner,
    )
    # sB: SMEM tensor with shape ((256,16),1,4,1) = (mma_atom, tma_rest_n, tma_rest_k, buffer_stages)
    # My Notation: ((256,16),1,4,1) = mma_atom, *res_tma_rest, buffer_stages
    #     smem_desc is one descriptor for the whole block (256, 16)
    sB = smem.allocate_tensor(
        element_type=io_dtype,
        layout=b_smem_layout.outer,
        byte_alignment=128,
        swizzle=b_smem_layout.inner,
    )

    # Allocate all TMEM columns
    # utils.TmemAllocator parameters:
    #   - alloc_result_dst_smem_ptr: SMEM pointer holding base address of allocated tensor memory
    #   - barrier_for_retrieve: NamedBarrier for synchronizing TMEM pointer retrieval
    #   - allocator_warp_id: Warp ID of allocator warp (default: 0)
    #   - is_two_cta: Whether to coordinate two CTAs (default: False)
    #   - arch: GPU architecture (default: "sm_100")
    tmem_alloc_barrier = pipeline.NamedBarrier(
        barrier_id=1,
        num_threads=threads_per_cta,
    )
    tmem = utils.TmemAllocator(
        storage.tmem_holding_buf,
        barrier_for_retrieve=tmem_alloc_barrier,
    )
    num_tmem_cols = 512
    tmem.allocate(num_tmem_cols)

    # Prefetch tma descriptor
    if warp_idx == 0:
        cpasync.prefetch_descriptor(tma_atom_a)
        cpasync.prefetch_descriptor(tma_atom_b)

    # Pipeline configuration
    num_tma_copy_bytes = cute.size_in_bytes(
        io_dtype, cute.select(a_smem_layout, mode=[0, 1, 2])
    ) + cute.size_in_bytes(io_dtype, cute.select(b_smem_layout, mode=[0, 1, 2]))

    # Get mbarrier pointers directly (replacing PipelineTmaUmma)
    ab_mbar_full = storage.ab_mbar_ptr.data_ptr()  # index 0
    ab_mbar_empty = storage.ab_mbar_ptr.data_ptr() + 1  # index 1

    # Get accumulator mbarrier pointer
    acc_mbar_ptr = storage.acc_mbar_ptr.data_ptr()

    # Initialize accumulator mbarrier - warp 0 initializes with 1 arrival expected
    # if_generate(warp_idx == 0, lambda: cute.arch.mbarrier_init(acc_mbar_ptr, 1))
    if warp_idx == 0:
        # Initialize mbarriers
        cute.arch.mbarrier_init(ab_mbar_full, 1)
        cute.arch.mbarrier_init(ab_mbar_empty, 1)
        cute.arch.mbarrier_init(acc_mbar_ptr, 1)

    # Ensure mbarrier init is visible and sync all threads
    cute.arch.mbarrier_init_fence()
    cute.arch.sync_threads()

    # Partition tensors for MMA and make fragments
    # gA: (tma_tile_m, tma_tile_k, RestM, RestK)
    gA = cute.local_tile(
        mA_mkl,
        cute.slice_(mma_tiler_mnk, (None, 0, None)),
        (None, None),
    )
    # gB: (tma_tile_n, tma_tile_k, RestN, RestK)
    gB = cute.local_tile(
        mB_nkl,
        cute.slice_(mma_tiler_mnk, (0, None, None)),
        (None, None),
    )
    # gC: (tma_tile_m, tma_tile_n, RestM, RestN)
    gC = cute.local_tile(
        mC_mnl,
        cute.slice_(mma_tiler_mnk, (None, None, 0)),
        (None, None),
    )
    thr_mma = tiled_mma.get_slice(0)
    # (mma_atom, tma_rest_m, tma_rest_k, RestM, RestK)
    tCgA = thr_mma.partition_A(gA)
    # (mma_atom, tma_rest_n, tma_rest_k, RestN, RestK)
    tCgB = thr_mma.partition_B(gB)
    # (mma_atom, tma_rest_m, tma_rest_n, RestM, RestN)
    tCgC = thr_mma.partition_C(gC)
    # tCrA: MMA fragment for A (smem_desc) stored in shared memory
    #        shape (mma_atom, tma_rest_m, tma_rest_k, buffer_stages)
    # My Notation: (mma_atom, tma_rest_m, tma_rest_k, buffer_stages)
    tCrA = tiled_mma.make_fragment_A(sA)
    # tCrB: MMA fragment for B (smem_desc) stored in shared memory
    #        shape (mma_atom, tma_rest_n, tma_rest_k, buffer_stages)
    # My Notation: (mma_atom, tma_rest_n, tma_rest_k, buffer_stages)
    tCrB = tiled_mma.make_fragment_B(sB)
    # (mma_atom, tma_rest_m, tma_rest_n)
    acc_shape = tiled_mma.partition_shape_C(mma_tiler_mnk[:2])
    # tCtAcc: MMA accumulator fragment, shape ((128,256),1,1) = (mma_atom, tma_rest_m, tma_rest_n)
    # My Notation: (mma_atom, tma_rest_m, tma_rest_n)
    tCtAcc = tiled_mma.make_fragment_C(acc_shape)
    # https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html
    # CU_TENSOR_MAP_SWIZZLE_128B* require the bounding box inner dimension to be <= 128.
    # Partition tensors for TMA; This requires the tensors partitioned for MMA
    # tAsA: TMA descriptor for SMEM A, shape ((tma_atom, num_tma_atom), stages)
    #        describes how each TMA atom is replicated per stage
    # tAgA: GMEM tensor for A used by TMA loads, shape ((tma_atom, num_tma_atom), RestM, RestK)
    #        # num_tma_atom corresponds to the number of MMA instructions per TMA atom
    # My Notation: ((tma_atom, num_tma_atom), RestM, RestK)
    # def tma_partition(atom, cta_coord, cta_layout, smem_tensor, gmem_tensor) -> (smem_desc, gmem_desc)
    # NOTE: tma_partition requires input tensors folded in shape (Each_Iter, Num_Iters)
    tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
        tma_atom_a,  # atom: TMA Copy Atom
        0,  # cta_coord: CTA coordinate
        cute.make_layout(1),  # cta_layout: CTA layout
        cute.group_modes(sA, 0, 3),  # smem_tensor: SMEM tensor grouped for A
        # cute.group_modes(tCgA, 0, 3),  # gmem_tensor: GMEM tensor grouped for A
        cute.group_modes(gA, 0, 2),  # gmem_tensor: GMEM tensor grouped for A
    )
    # tBsB: SMEM descriptor for B, shape ((tma_atom, num_tma_atom), stages)
    # My Notation: ((tma_atom, num_tma_atom), stages)
    # tBgB: GMEM tensor for B used by TMA loads, shape ((tma_atom, num_tma_atom), RestN, RestK)
    # My Notation: ((tma_atom, num_tma_atom), RestN, RestK)
    tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
        tma_atom_b,  # atom: TMA Copy Atom
        0,  # cta_coord: CTA coordinate
        cute.make_layout(1),  # cta_layout: CTA layout
        cute.group_modes(
            sB[(None, None), None, None, None], 0, 4
        ),  # smem_tensor: SMEM tensor grouped for B
        cute.group_modes(
            tCgB[(None, None), None, None, None, None], 0, 4
        ),  # gmem_tensor: GMEM tensor grouped for B
    )

    # CTA-wide sync before retrieving the pointer to the start of the allocated TMEM
    # Only warp 0 does the allocation so we need to sync before retrieving the TMEM start address
    tmem.wait_for_alloc()
    tmem_ptr = tmem.retrieve_ptr(acc_dtype)
    # Swap the pointer in tCtAcc
    tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc.layout)

    subtile_cnt = 4
    # epi_tiler: ((epi_tile_m, epi_tile_n)) = subtile for epilogue (each thread loads 1/4 of MMA_N columns)
    #   - epi_tile_m = MMA_atom M (full width)
    #   - epi_tile_n = 1/4 × MMA_atom N
    epi_tiler = (
        cute.size(tCtAcc, mode=[0, 0]),
        cute.size(tCtAcc, mode=[0, 1]) // subtile_cnt,
    )
    # epi_tiler = (128, 64)
    # tCtAcc: ((128, 256),1,1) = (mma_atom, tma_rest_m, tma_rest_n)
    # epi_tiler: (epi_tile_m, epi_tile_n)
    # tEPItAcc: (epi_tile_m, epi_tile_n, mma_rest_m, mma_rest_n, tma_rest_m, tma_rest_n)
    tEPItAcc = cute.flat_divide(tCtAcc[(None, None), None, None], epi_tiler)
    tEPIgC = cute.flat_divide(tCgC[(None, None), None, None, None, None], epi_tiler)
    # tEPIgC: (epi_tile_m, epi_tile_n, mma_rest_m, mma_rest_n, tma_rest_m, tma_rest_n, RestM, RestN)

    # TMEM copy atom: loads 32x32 blocks with x64 repetition (64 elements per instruction)
    # Every thread loads 64 columns per iteration (32 elements * 2 for x64 repetition)
    tmem_atom = cute.make_copy_atom(
        # tcgen05.Ld16x64bOp(tcgen05.Repetition.x16),
        tcgen05.Ld32x32bOp(tcgen05.Repetition.x64),
        cutlass.Float32,
    )
    # tmem_tiled_copy: creates tiled copy with the TMEM atom
    tmem_tiled_copy = tcgen05.make_tmem_copy(
        tmem_atom, tEPItAcc[None, None, 0, 0, 0, 0]
    )
    # Get thread slice of the tiled copy
    tmem_thr_copy = tmem_tiled_copy.get_slice(tidx)

    # tTRtC: (((64,32),1),1,((1,4),1,1)) = (TmemCpy, NumTmemCpy, NumTiles)
    # tTRtC: (((64,32),1),1,1,1,4,1,1) = (TmemCpy, NumTmemCpy, NumTiles)
    # My Notation: (tmem_atom, epi_tile_m, epi_tile_n, mma_rest_m, mma_rest_n, tma_rest_m, tma_rest_n)
    tTRtC = tmem_thr_copy.partition_S(tEPItAcc)
    # (((64,32),1),1,1,(1,4,1,1))
    tTRtC = cute.group_modes(tTRtC, 3, cute.rank(tTRtC))

    # Just to get shape (adds RestM/RestN)
    # tTRgC: (((64,32),1),1,1,1,4,1,1,RestM,RestN)
    # My Notation: (tmem_atom, epi_tile_m, epi_tile_n, mma_rest_m, mma_rest_n, tma_rest_m, tma_rest_n, RestM, RestN)
    tTRgC = tmem_thr_copy.partition_D(tEPIgC)
    # tCrAcc: ((64,1),1,1) = register tensor for accumulator (acc_dtype = Float32)
    # My Notation: ((64,1),1,1) = (tmem_atom, epi_tile_m, epi_tile_n)
    tCrAcc = cute.make_rmem_tensor(
        tTRgC[None, None, None, 0, 0, 0, 0, 0, 0].shape, acc_dtype
    )
    # My Notation: ((64,1),1,1) = (tmem_atom, epi_tile_m, epi_tile_n)
    tCrC = cute.make_rmem_tensor(
        tTRgC[None, None, None, 0, 0, 0, 0, 0, 0].shape, io_dtype
    )

    #
    # 2. Main loop
    #
    # Number of TMA tiles along the K axis (RestK)
    num_k_tiles = cute.size(gA, mode=[3])
    phase = 1  # Toggle between 0 and 1

    if warp_idx == 0:
        # Simple loop without prefetch (single stage pipelining)
        for k_tile_idx in range(num_k_tiles):
            # TMA Load: wait for empty buffer
            cute.arch.mbarrier_wait(ab_mbar_empty, phase)

            # TMA loads
            cute.copy(
                tma_atom_a,
                tAgA[(None, mma_coord_mnk[0], k_tile_idx)],  # Notation: tma_atom
                tAsA[(None, 0)],  # Notation: tma_atom, stage = 1
                tma_bar_ptr=ab_mbar_full,
            )
            cute.copy(
                tma_atom_b,
                tBgB[(None, mma_coord_mnk[1], k_tile_idx)],
                tBsB[(None, 0)],
                tma_bar_ptr=ab_mbar_full,
            )

            # TMA Load arrives on full: elect_one + expect_tx
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(
                    ab_mbar_full, num_tma_copy_bytes
                )

            # TCGen05 MMA: wait for full buffer
            cute.arch.mbarrier_wait(ab_mbar_full, 1 - phase)

            # Execute one K-block worth of MMA instructions
            num_k_blocks = cute.size(tCrA, mode=[2])
            for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                k_block_coord = (
                    None,
                    None,
                    k_block_idx,
                    0,
                )  # mma_atom, tma_rest_m/n, tma_rest_k, P
                cute.gemm(
                    tiled_mma,
                    tCtAcc,  # mma_atom, tma_rest_m/n
                    tCrA[k_block_coord],  # mma_atom, tma_rest_m
                    tCrB[k_block_coord],  # mma_atom, tma_rest_n
                    tCtAcc,
                )
                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

            # TCGen05 MMA arrives on empty: elect_one + tcgen05.commit
            with cute.arch.elect_one():
                cute.nvgpu.tcgen05.commit(ab_mbar_empty)

            # Toggle phase
            phase = 1 - phase

        # Signal MMA done - use tcgen05.commit like PipelineUmmaAsync
        with cute.arch.elect_one():
            cute.nvgpu.tcgen05.commit(acc_mbar_ptr)

    #

    # 3. Epilogue
    #

    # Release TMEM allocation lock
    tmem.relinquish_alloc_permit()

    # Wait for MMA to complete (warp 0 signals via mbarrier)
    cute.arch.mbarrier_wait(acc_mbar_ptr, phase=0)

    # Epilogue: TMEM -> Register -> Global Memory
    # Sub-tiling: iterate over 4 subtiles (subtile_cnt = 4) for better ILP
    #   - tTRtC has 4 tiles in mode[2] (N dimension: 64 * 4 = 256 total columns)
    # Loop over each subtile:
    #   1. Copy from TMEM (tTRtC) to register (tCrAcc) - Float32
    #   2. Convert to output dtype and store to register (tCrC) - Float16
    #   3. Copy from register (tCrC) to global memory (tTRgC) - Float16

    # grouped shape: (tmem_atom, epi_tile_m, epi_tile_n, (mma_rest_m, mma_rest_n, tma_rest_m, tma_rest_n, RestM, RestN)) = (tmem_atom, epi_tile_m, epi_tile_n, other_rests)
    tTRgC_tile = tTRgC[None, None, None, None, None, None, None, mma_coord_mnk[0], mma_coord_mnk[1]]
    tTRgC_tile = cute.group_modes(tTRgC_tile, 3, cute.rank(tTRgC_tile))
    for i in cutlass.range(cute.size(tTRtC, mode=[3])):  # mma_rest_m * mma_rest_n tiles
        cute.copy(
            tmem_tiled_copy, tTRtC[None, None, None, i], tCrAcc
        )  # TMEM -> Reg (Float32)
        tCrC.store(tCrAcc.load().to(io_dtype))  # Convert to Float16
        cute.autovec_copy(tCrC, tTRgC_tile[None, None, None, i])  # Reg -> Global (Float16)

    # Deallocate TMEM
    pipeline.sync(barrier_id=1)
    tmem.free(tmem_ptr)


@cute.jit
def host_function(a: cute.Tensor, b: cute.Tensor, c: cute.Tensor, stream):
    # Construct tiled MMA
    op = tcgen05.MmaF16BF16Op(
        io_dtype,
        acc_dtype,
        mma_inst_shape_mnk,
        tcgen05.CtaGroup.ONE,
        tcgen05.OperandSource.SMEM,
        tcgen05.OperandMajorMode.K,
        tcgen05.OperandMajorMode.K,
    )
    tiled_mma = cute.make_tiled_mma(op)

    # Construct SMEM layouts for A and B
    # make_smem_layout_a(tiled_mma, mma_tiler_mnk, a_dtype, num_stages)
    #   ├─> tiled_mma.partition_shape_A()  # Get partitioned shape (M, N, K partitions per CTA)
    #   │   # Returns: ((M_part, K_part), M_tile, K_tile)
    #   │
    #   ├─> get_smem_layout_atom_ab()  # Heuristic selection
    #   │   │   # Based on major_mode (K or MN), dtype width, and major_mode_size_bits
    #   │   └─> Returns: SmemLayoutAtomKind (K_SW128, K_SW64, K_SW32, K_INTER, MN_SW128, MN_SW64, etc.)
    #   │
    #   ├─> make_smem_layout_atom()  # Create actual layout atom
    #   │   │   # Uses core.make_composed_layout + core.make_swizzle
    #   │   │   # Swizzle patterns: INTER(128b), SW32(256b), SW64(512b), SW128(1024b)
    #   │   └─> Returns: ComposedLayout with swizzle applied
    #   │
    #   └─> tile_to_mma_shape()  # Tile and stage the layout
    #       └─> cute.tile_to_shape(cute.append(shape, num_stages), order=...)
    #       # Final shape: ((M_atom, K_atom), MMA_M_tiles, MMA_K_tiles, num_stages)
    #       # e.g., ((128,16), 1, 4, 1) for M=128, K=64, stages=1
    #       # cute.select(mode=[0,1,2]) removes trailing stage dim when stages=1
    a_smem_layout = sm100_utils.make_smem_layout_a(
        tiled_mma,
        mma_tiler_mnk,
        a.element_type,
        ab_stages,
    )
    b_smem_layout = sm100_utils.make_smem_layout_b(
        tiled_mma,
        mma_tiler_mnk,
        b.element_type,
        ab_stages,
    )
    a_smem_layout_one_stage = cute.select(a_smem_layout, mode=[0, 1, 2])
    b_smem_layout_one_stage = cute.select(b_smem_layout, mode=[0, 1, 2])

    # Construct TMA load atoms
    op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
    a_tma_atom, a_tma_tensor = cute.nvgpu.make_tiled_tma_atom_A(
        op,
        a,
        a_smem_layout_one_stage,
        mma_tiler_mnk,
        tiled_mma,
    )
    b_tma_atom, b_tma_tensor = cute.nvgpu.make_tiled_tma_atom_B(
        op,
        b,
        b_smem_layout_one_stage,
        mma_tiler_mnk,
        tiled_mma,
    )

    # Pretty prints kernel attributes useful for debugging
    # print(f"a            = {cute.pretty_str(a)}")
    # print(f"b            = {cute.pretty_str(b)}")
    # print(f"c            = {cute.pretty_str(c)}")
    # print(f"tiled_mma    = {cute.pretty_str(tiled_mma)}")
    # print(f"a_tma_atom   = {cute.pretty_str(a_tma_atom)}")
    # print(f"b_tma_atom   = {cute.pretty_str(b_tma_atom)}")
    # print(f"a_tma_tensor = {cute.pretty_str(a_tma_tensor)}")
    # print(f"b_tma_tensor = {cute.pretty_str(b_tma_tensor)}")

    # Launch the kernel
    grid_shape = cute.ceil_div((*c.layout.shape, 1), mma_tiler_mnk[:2])
    kernel(
        tiled_mma,
        a_tma_atom,
        a_tma_tensor,
        b_tma_atom,
        b_tma_tensor,
        c,
        a_smem_layout,
        b_smem_layout,
    ).launch(
        grid=grid_shape,
        block=(threads_per_cta, 1, 1),
    )


def run_dense_gemm(
    mnk: Tuple[int, int, int],
    tolerance: float,
    warmup_iterations=10,
    iterations=100,
    skip_ref_check=False,
):
    global torch, cutlass_torch
    import torch
    import cutlass.torch as cutlass_torch
    import cuda.bindings.driver as cuda
    import cutlass.cute.testing as testing

    print("===================================================================")
    print("Running Blackwell fp16 GEMM example 1 with:")
    print(f"  mnk:       {mnk}")
    print(f"  tolerance: {tolerance}")
    print("===================================================================")
    print()

    m, n, k = mnk
    l = 1
    torch.manual_seed(1111)
    ab_dtype = cutlass.Float16
    c_dtype = cutlass.Float16
    a_major = "k"
    b_major = "k"
    c_major = "n"

    def make_tensors(mn, k, dtype):
        shape = (mn, k)
        return (
            torch.empty(*shape, dtype=torch.int32)
            .random_(-2, 2)
            .to(dtype=dtype, device="cuda")
        )

    def create_tensors(l, m, n, k, a_major, b_major, c_major, ab_dtype, c_dtype):
        import torch
        import cutlass.torch as cutlass_torch

        torch.manual_seed(1111)

        a_torch_cpu = make_tensors(m, k, cutlass_torch.dtype(io_dtype))
        b_torch_cpu = make_tensors(n, k, cutlass_torch.dtype(io_dtype))
        c_torch_cpu = make_tensors(m, n, cutlass_torch.dtype(io_dtype))

        a_tensor, _ = cutlass_torch.cute_tensor_like(
            a_torch_cpu, ab_dtype, is_dynamic_layout=True, assumed_align=16
        )
        b_tensor, _ = cutlass_torch.cute_tensor_like(
            b_torch_cpu, ab_dtype, is_dynamic_layout=True, assumed_align=16
        )
        c_tensor, c_torch_gpu = cutlass_torch.cute_tensor_like(
            c_torch_cpu, c_dtype, is_dynamic_layout=True, assumed_align=16
        )

        return (
            a_tensor,
            b_tensor,
            c_tensor,
            a_torch_cpu,
            b_torch_cpu,
            c_torch_cpu,
            c_torch_gpu,
        )

    def generate_tensors():
        import cutlass.torch as cutlass_torch

        a_tensor, _ = cutlass_torch.cute_tensor_like(
            a_torch_cpu, ab_dtype, is_dynamic_layout=True, assumed_align=16
        )
        b_tensor, _ = cutlass_torch.cute_tensor_like(
            b_torch_cpu, ab_dtype, is_dynamic_layout=True, assumed_align=16
        )
        c_tensor, _ = cutlass_torch.cute_tensor_like(
            c_torch_cpu, c_dtype, is_dynamic_layout=True, assumed_align=16
        )
        return testing.JitArguments(a_tensor, b_tensor, c_tensor, current_stream)

    a_tensor, b_tensor, c_tensor, a_torch_cpu, b_torch_cpu, c_torch_cpu, c_torch_gpu = (
        create_tensors(l, m, n, k, a_major, b_major, c_major, ab_dtype, c_dtype)
    )

    torch_stream = torch.cuda.current_stream()
    current_stream = cuda.CUstream(torch_stream.cuda_stream)

    compiled_gemm = cute.compile(
        host_function, a_tensor, b_tensor, c_tensor, current_stream
    )

    def compare(a_torch_cpu, b_torch_cpu, c_torch_gpu, c_dtype, tolerance):
        import torch
        import cutlass.torch as cutlass_torch

        kernel_result = c_torch_gpu.cpu()

        ref = torch.einsum(
            "mk,nk->mn",
            a_torch_cpu.to(dtype=torch.float32),
            b_torch_cpu.to(dtype=torch.float32),
        )

        _, ref_torch_gpu = cutlass_torch.cute_tensor_like(
            ref, c_dtype, is_dynamic_layout=True, assumed_align=16
        )
        ref_result = ref_torch_gpu.cpu()

        torch.testing.assert_close(
            kernel_result, ref_result, atol=tolerance, rtol=1e-05
        )

    if not skip_ref_check:
        compiled_gemm(a_tensor, b_tensor, c_tensor, current_stream)
        compare(a_torch_cpu, b_torch_cpu, c_torch_gpu, c_dtype, tolerance)

    workspace_count = 1
    exec_time = testing.benchmark(
        compiled_gemm,
        workspace_generator=generate_tensors,
        workspace_count=workspace_count,
        stream=current_stream,
        warmup_iterations=warmup_iterations,
        iterations=iterations,
    )

    return exec_time


if __name__ == "__main__":

    def parse_comma_separated_ints(s: str):
        try:
            return [int(x.strip()) for x in s.split(",")]
        except ValueError:
            raise argparse.ArgumentTypeError(
                "Invalid format. Expected comma-separated integers."
            )

    from cuda.bindings import driver as cu_driver

    cu_driver.cuInit(0)
    err, device_count = cu_driver.cuDeviceGetCount()
    if err != cu_driver.CUresult.CUDA_SUCCESS or device_count < 1:
        raise RuntimeError("A GPU is required to run this example")

    parser = argparse.ArgumentParser(description="Blackwell fp16 GEMM example 0")
    parser.add_argument(
        "--mnk",
        type=parse_comma_separated_ints,
        default=[8192, 8192, 8192],
        help="MNK dimensions (comma-separated)",
    )
    parser.add_argument(
        "--tolerance", type=float, default=1e-01, help="Tolerance for validation"
    )
    args = parser.parse_args()
    if len(args.mnk) != 3:
        parser.error("--mnk must contain exactly 3 values")
    if args.mnk[0] % mma_tiler_mnk[0] != 0 or args.mnk[1] % mma_tiler_mnk[1] != 0:
        parser.error("m n must be divisible by mma_tiler_mn")

    run_dense_gemm(
        args.mnk,
        args.tolerance,
    )
    print("PASS")
