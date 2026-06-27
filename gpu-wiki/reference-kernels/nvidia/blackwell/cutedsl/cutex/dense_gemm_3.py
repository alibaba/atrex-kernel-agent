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
CuTeDSL Dense GEMM with Cluster Support:

| Parameter              | Value         |
|------------------------|---------------|
| MMA Instruction Shape  | (128, 256, 16)|
| MMA Tiler             | (128, 256, 64)|
| Threads per CTA        | 128           |
| Pipeline Stages        | 4 (AB), 1 (acc)|
| Cluster Shape          | (2, 1) - default |

Step 1: Added cluster support for parallel CTA execution.
"""

import argparse
from typing import Tuple

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.pipeline as pipeline

# [CLUSTER] Import pipeline_init functions for cluster support
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.runtime import from_dlpack

import cuda.bindings.driver as cuda
import cutlass.cute.testing as testing

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

io_dtype = cutlass.Float16
acc_dtype = cutlass.Float32
mma_inst_shape_mnk = (128, 256, 16)
mma_tiler_mnk = (128, 256, 64)
threads_per_cta = 128

# Pipeline stage configuration
ab_stages = 4
acc_stage = 1

# Cluster configuration (Step 1: Add cluster support)
cluster_shape_mn = (2, 1)


@cute.struct
class SharedStorage:
    ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, ab_stages * 2]
    acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, acc_stage * 2]
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
    # [CLUSTER] Cluster parameters for TMA multicast
    cluster_layout_vmnk: cute.Layout,
    num_mcast_ctas_a: int,
    num_mcast_ctas_b: int,
    is_a_mcast: cutlass.Constexpr,
    is_b_mcast: cutlass.Constexpr,
    num_tma_producer: cutlass.Constexpr,
):
    # Current thread/warp/block coordinates
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    bidx, bidy, _ = cute.arch.block_idx()
    mma_coord_mnk = (bidx, bidy, None)

    # [CLUSTER] Get block's cluster coordinates
    cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
    block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
        cta_rank_in_cluster
    ) # column-major cta id within cluster

    #
    # 1. Prepare args
    #

    # Allocate SMEM
    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)
    sA = smem.allocate_tensor(
        element_type=io_dtype,
        layout=a_smem_layout.outer,
        byte_alignment=128,
        swizzle=a_smem_layout.inner,
    )
    sB = smem.allocate_tensor(
        element_type=io_dtype,
        layout=b_smem_layout.outer,
        byte_alignment=128,
        swizzle=b_smem_layout.inner,
    )

    # Allocate all TMEM columns
    tmem_alloc_barrier = pipeline.NamedBarrier(
        barrier_id=1,
        num_threads=threads_per_cta,
    )
    tmem = utils.TmemAllocator(
        storage.tmem_holding_buf,
        barrier_for_retrieve=tmem_alloc_barrier,
    )

    # Prefetch tma descriptor
    if warp_idx == 0:
        cpasync.prefetch_descriptor(tma_atom_a)
        cpasync.prefetch_descriptor(tma_atom_b)

    # Pipeline configuration
    num_tma_copy_bytes = cute.size_in_bytes(
        io_dtype, cute.select(a_smem_layout, mode=[0, 1, 2])
    ) + cute.size_in_bytes(io_dtype, cute.select(b_smem_layout, mode=[0, 1, 2]))
    # [CLUSTER] Create pipeline with cluster layout and multicast producer count
    ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
        num_stages=ab_stages,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_tma_producer # number of participants
        ),
        tx_count=num_tma_copy_bytes,
        barrier_storage=storage.ab_mbar_ptr.data_ptr(),
        cta_layout_vmnk=cluster_layout_vmnk,
    ).make_participants()
    acc_producer, acc_consumer = pipeline.PipelineUmmaAsync.create(
        num_stages=acc_stage,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            threads_per_cta,
        ),
        barrier_storage=storage.acc_mbar_ptr.data_ptr(),
    ).make_participants()

    # [CLUSTER] Initialize pipeline arrive for cluster, barrier.cluster.arrive.relaxed
    pipeline_init_arrive(cluster_shape_mn=cluster_layout_vmnk, is_relaxed=True)
    # [CLUSTER] Wait for pipeline to complete before tmem alloc
    pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)

    num_tmem_cols = 512
    tmem.allocate(num_tmem_cols)

    # Partition tensors for MMA and make fragments
    # (bM, bK, RestK)
    gA = cute.local_tile(mA_mkl, mma_tiler_mnk, mma_coord_mnk, proj=(1, None, 1))
    # (bN, bK, RestK)
    gB = cute.local_tile(mB_nkl, mma_tiler_mnk, mma_coord_mnk, proj=(None, 1, 1))
    # (bM, bN)
    gC = cute.local_tile(mC_mnl, mma_tiler_mnk, mma_coord_mnk, proj=(1, 1, None))
    thr_mma = tiled_mma.get_slice(0)
    # (MMA, MMA_M, MMA_K)
    tCgA = thr_mma.partition_A(gA)
    # (MMA, MMA_N, MMA_K)
    tCgB = thr_mma.partition_B(gB)
    # (MMA, MMA_M, MMA_N)
    tCgC = thr_mma.partition_C(gC)
    # (MMA, MMA_M, MMA_K)
    tCrA = tiled_mma.make_fragment_A(sA)
    # (MMA, MMA_N, MMA_K)
    tCrB = tiled_mma.make_fragment_B(sB)
    # (MMA, MMA_M, MMA_N)
    acc_shape = tiled_mma.partition_shape_C(mma_tiler_mnk[:2])
    # (MMA, MMA_M, MMA_N)
    tCtAcc = tiled_mma.make_fragment_C(acc_shape)
    # Partition tensors for TMA; This requires the tensors partitioned for MMA
    # [CLUSTER] Create CTA layouts from cluster_layout_vmnk
    a_cta_layout = cute.make_layout(
        cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape # dim N
    )
    b_cta_layout = cute.make_layout(
        cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
    )
    tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
        tma_atom_a,
        block_in_cluster_coord_vmnk[2], # cta coord on N
        a_cta_layout, # cta layout on dim N
        cute.group_modes(sA, 0, 3),
        cute.group_modes(tCgA, 0, 3),
    )
    tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
        tma_atom_b,
        block_in_cluster_coord_vmnk[1],
        b_cta_layout,
        cute.group_modes(sB, 0, 3),
        cute.group_modes(tCgB, 0, 3),
    )

    # CTA-wide sync before retrieving the pointer to the start of the allocated TMEM
    # Only warp 0 does the allocation so we need to sync before retrieving the TMEM start address
    tmem.wait_for_alloc()
    tmem_ptr = tmem.retrieve_ptr(acc_dtype)
    # Swap the pointer in tCtAcc
    tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc.layout)

    subtile_cnt = 4
    # (EpiTile)
    epi_tiler = (
        (cute.size(tCtAcc, mode=[0, 0]), cute.size(tCtAcc, mode=[0, 1]) // subtile_cnt),
    )
    # (EpiTile, NumTiles)
    tCtAcc_epi = cute.zipped_divide(tCtAcc, epi_tiler)
    # (EpiTile, NumTiles)
    gC_epi = cute.zipped_divide(tCgC, epi_tiler)

    # Every thread loads 64 x fp32
    tmem_atom = cute.make_copy_atom(
        tcgen05.Ld32x32bOp(tcgen05.Repetition.x64),
        cutlass.Float32,
    )
    tmem_tiled_copy = tcgen05.make_tmem_copy(tmem_atom, tCtAcc_epi[None, 0])
    tmem_thr_copy = tmem_tiled_copy.get_slice(tidx)

    # (TmemCpy,NumTmemCpy,NumTiles)
    tDtC = tmem_thr_copy.partition_S(tCtAcc_epi)
    # (TmemCpy,NumTmemCpy,NumTiles)
    tDgC = tmem_thr_copy.partition_D(gC_epi)

    # (TmemCpy,NumTmemCpy)
    tCrAcc = cute.make_rmem_tensor(tDgC[None, None, 0].shape, acc_dtype)
    # (TmemCpy,NumTmemCpy)
    tCrC = cute.make_rmem_tensor(tDgC[None, None, 0].shape, io_dtype)

    #
    # 2. Main loop
    #

    # [CLUSTER] Create multicast masks, 1. tma: whom to mcast 2. mma: whom to arrive
    a_full_mcast_mask = None
    b_full_mcast_mask = None
    if cutlass.const_expr(is_a_mcast or is_b_mcast):
        a_full_mcast_mask = cpasync.create_tma_multicast_mask(
            cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
        )
        b_full_mcast_mask = cpasync.create_tma_multicast_mask(
            cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
        )

    num_k_tiles = cute.size(gA, mode=[2])
    if warp_idx == 0:
        # Wait for a empty accumulator buffer
        acc_empty = acc_producer.acquire_and_advance()
        for k_tile_idx in cutlass.range(num_k_tiles, prefetch_stages=ab_stages - 2):
            # Issue TMA loads
            ab_empty = ab_producer.acquire_and_advance()
            cute.copy(
                tma_atom_a,
                tAgA[(None, ab_empty.count)],
                tAsA[(None, ab_empty.index)],
                tma_bar_ptr=ab_empty.barrier,
                mcast_mask=a_full_mcast_mask, # mcast
            )
            cute.copy(
                tma_atom_b,
                tBgB[(None, ab_empty.count)],
                tBsB[(None, ab_empty.index)],
                tma_bar_ptr=ab_empty.barrier,
                mcast_mask=b_full_mcast_mask, # mcast
            )

            # Execute one K-block worth of MMA instructions
            ab_full = ab_consumer.wait_and_advance()
            num_k_blocks = cute.size(tCrA, mode=[2])
            for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                k_block_coord = (None, None, k_block_idx, ab_full.index)
                cute.gemm(
                    tiled_mma,
                    tCtAcc,
                    tCrA[k_block_coord],
                    tCrB[k_block_coord],
                    tCtAcc,
                )
                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

            # Signal that the A/B buffers have been consumed and are ready for the next load
            # [CLUSTER] internally call umma_arrive_multicast
            # PipelineTmaUmma._compute_mcast_arrival_mask
            # PipelineTmaUmma._is_leader_cta
            ab_full.release()

        # Signal that the accumulator is fully computed
        acc_empty.commit()

    #
    # 3. Epilogue
    #

    # Release TMEM allocation lock
    tmem.relinquish_alloc_permit()

    # Wait for the accumulator buffer to be full
    acc_full = acc_consumer.wait_and_advance()

    # TMEM -> RMEM -> GEMM
    # Sub-tiling for better instruction-level parallelism
    for i in cutlass.range(cute.size(tDtC, mode=[2])):
        cute.copy(tmem_tiled_copy, tDtC[None, None, i], tCrAcc)
        tCrC.store(tCrAcc.load().to(io_dtype))
        cute.autovec_copy(tCrC, tDgC[None, None, i])
    acc_full.release()

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

    # [CLUSTER] Compute cluster layout for TMA
    cluster_layout_vmnk = cute.tiled_divide(
        cute.make_layout((*cluster_shape_mn, 1)),
        (tiled_mma.thr_id.shape,),
    )

    # [CLUSTER] Compute multicast parameters
    num_mcast_ctas_a = cute.size(cluster_layout_vmnk.shape[2]) # A multicast on dimension N
    num_mcast_ctas_b = cute.size(cluster_layout_vmnk.shape[1]) # B multicast on dimension M
    is_a_mcast = num_mcast_ctas_a > 1
    is_b_mcast = num_mcast_ctas_b > 1 # if only 2 cta, just B multicast

    # [CLUSTER] num_tma_producer for pipeline, 2 participants on dimension M
    num_tma_producer = num_mcast_ctas_a + num_mcast_ctas_b - 1

    print(f"cluster_layout_vmnk shape: {cluster_layout_vmnk.shape}")
    print(f"num_mcast_ctas_a: {num_mcast_ctas_a}, num_mcast_ctas_b: {num_mcast_ctas_b}")
    print(f"is_a_mcast: {is_a_mcast}, is_b_mcast: {is_b_mcast}")

    # [CLUSTER] Construct TMA load atoms with cluster support, mainly useful for pair-UMMA
    a_op = sm100_utils.cluster_shape_to_tma_atom_A(cluster_shape_mn, tiled_mma.thr_id)
    a_tma_atom, a_tma_tensor = cute.nvgpu.make_tiled_tma_atom_A(
        a_op,
        a,
        a_smem_layout_one_stage,
        mma_tiler_mnk,
        tiled_mma,
        cluster_layout_vmnk.shape,
    )
    b_op = sm100_utils.cluster_shape_to_tma_atom_B(cluster_shape_mn, tiled_mma.thr_id)
    b_tma_atom, b_tma_tensor = cute.nvgpu.make_tiled_tma_atom_B(
        b_op,
        b,
        b_smem_layout_one_stage,
        mma_tiler_mnk,
        tiled_mma,
        cluster_layout_vmnk.shape,
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

    # Launch the kernel with cluster support
    cluster_shape_mnl = (*cluster_shape_mn, 1)
    grid_shape = cute.round_up(
        cute.ceil_div((*c.layout.shape, 1), mma_tiler_mnk[:2]),
        cluster_shape_mnl,
    )
    # [CLUSTER] Launch kernel with cluster parameters
    kernel(
        tiled_mma,
        a_tma_atom,
        a_tma_tensor,
        b_tma_atom,
        b_tma_tensor,
        c,
        a_smem_layout,
        b_smem_layout,
        cluster_layout_vmnk,
        num_mcast_ctas_a,
        num_mcast_ctas_b,
        is_a_mcast,
        is_b_mcast,
        num_tma_producer,
    ).launch(
        grid=grid_shape,
        block=(threads_per_cta, 1, 1),
        cluster=cluster_shape_mnl,
        stream=stream,
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

    print("===================================================================")
    print("Running Blackwell fp16 GEMM example 2 with:")
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

    ## Make K-major tensors (torch tensors are row-major)
    def make_tensors(mn, k, dtype):
        shape = (mn, k)
        return (
            torch.empty(*shape, dtype=torch.int32)
            .random_(-2, 2)
            .to(dtype=dtype, device="cuda")
        )

    # a = make_tensors(m, k, cutlass_torch.dtype(io_dtype))
    # b = make_tensors(n, k, cutlass_torch.dtype(io_dtype))
    # c = make_tensors(m, n, cutlass_torch.dtype(io_dtype))
    # a_tensor = (
    #    from_dlpack(a, assumed_align=32)
    #    .mark_layout_dynamic(leading_dim=1)
    #    .mark_compact_shape_dynamic(mode=1, divisibility=k)
    # )
    # b_tensor = (
    #    from_dlpack(b, assumed_align=32)
    #    .mark_layout_dynamic(leading_dim=1)
    #    .mark_compact_shape_dynamic(mode=1, divisibility=k)
    # )
    # c_tensor = (
    #    from_dlpack(c, assumed_align=32)
    #    .mark_layout_dynamic(leading_dim=1)
    #    .mark_compact_shape_dynamic(mode=1, divisibility=n)
    # )

    def create_tensors(l, m, n, k, a_major, b_major, c_major, ab_dtype, c_dtype):
        import torch
        import cutlass.torch as cutlass_torch

        torch.manual_seed(1111)

        # a_torch_cpu = cutlass_torch.matrix(l, m, k, a_major == "m", ab_dtype) # no prepend l
        # b_torch_cpu = cutlass_torch.matrix(l, n, k, b_major == "n", ab_dtype)
        # c_torch_cpu = cutlass_torch.matrix(l, m, n, c_major == "m", c_dtype)
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

    # Get current CUDA stream from PyTorch
    torch_stream = torch.cuda.current_stream()
    # Get the raw stream pointer as a CUstream
    current_stream = cuda.CUstream(torch_stream.cuda_stream)
    # Entry point to the host JIT function
    compiled_gemm = cute.compile(
        host_function, a_tensor, b_tensor, c_tensor, current_stream
    )

    # host_function(a_tensor, b_tensor, c_tensor, no_cache=True)
    ## Compute reference result and verify
    # ref = (torch.einsum("mk,nk->mn", a.to(torch.float32), b.to(torch.float32))).cpu()

    # torch.testing.assert_close(
    #    c.cpu(), ref.to(cutlass_torch.dtype(io_dtype)), atol=tolerance, rtol=1e-05
    # )

    def compare(a_torch_cpu, b_torch_cpu, c_torch_gpu, c_dtype, tolerance):
        import torch
        import cutlass.torch as cutlass_torch

        # Copy gpu result back
        kernel_result = c_torch_gpu.cpu()

        # Compute reference result
        ref = torch.einsum(
            "mk,nk->mn",
            a_torch_cpu.to(dtype=torch.float32),
            b_torch_cpu.to(dtype=torch.float32),
        )

        # Convert ref to c_dtype
        _, ref_torch_gpu = cutlass_torch.cute_tensor_like(
            ref, c_dtype, is_dynamic_layout=True, assumed_align=16
        )
        ref_result = ref_torch_gpu.cpu()

        # Assert close results
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

    return exec_time  # Return execution time in microseconds


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

    parser = argparse.ArgumentParser(description="Blackwell fp16 GEMM example 2")
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
