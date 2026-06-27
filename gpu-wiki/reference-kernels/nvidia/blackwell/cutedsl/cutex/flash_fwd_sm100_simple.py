# Supported features:
# - BF16 & FP16 dtype
# - noncausal attention only (is_causal=False hardcoded)
# - MHA, GQA, MQA
# - hdim 128 (Q/V) only.
# - sliding window
# - split-kv
# Unsupported features that will be added later:
# - page size != 128
# - additional head dimensions
# Based on the cutlass example and cute-dsl example:
# https://github.com/NVIDIA/cutlass/tree/main/examples/77_blackwell_fmha
# https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/blackwell/fmha.py

import enum
import math
from typing import Type, Tuple, Callable, Optional, Literal
from functools import partial

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Boolean, const_expr
from cutlass.cute.nvgpu import cpasync
import cutlass.cute.nvgpu.tcgen05 as tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils_basic
from cutlass import pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.base_dsl.arch import Arch
from cutlass.cutlass_dsl import BaseDSL

from quack import copy_utils, layout_utils

from flash_attn.cute.cute_dsl_utils import assume_tensor_aligned
import flash_attn.cute.pipeline as pipeline_custom
from flash_attn.cute.softmax import SoftmaxSm100
from flash_attn.cute.seqlen_info import SeqlenInfoQK
from flash_attn.cute.block_info import BlockInfo
from flash_attn.cute import mma_sm100_desc as sm100_desc
from flash_attn.cute import blackwell_helpers as sm100_utils
from quack.cute_dsl_utils import ParamsBase
from flash_attn.cute.tile_scheduler import (
    TileSchedulerArguments,
    StaticPersistentTileScheduler,
    SingleTileLPTScheduler,
)
from cutlass.cute.nvgpu.tcgen05 import make_smem_layout_atom


class NamedBarrierFwd(enum.IntEnum):
    Epilogue = enum.auto()  # starts from 1 as barrier 0 is reserved for sync_threads()
    TmemPtr = enum.auto()
    SoftmaxStatsW0 = enum.auto()
    SoftmaxStatsW1 = enum.auto()
    SoftmaxStatsW2 = enum.auto()
    SoftmaxStatsW3 = enum.auto()
    SoftmaxStatsW4 = enum.auto()
    SoftmaxStatsW5 = enum.auto()
    SoftmaxStatsW6 = enum.auto()
    SoftmaxStatsW7 = enum.auto()


#     WarpSchedulerWG1 = enum.auto()
#     WarpSchedulerWG2 = enum.auto()


class FlashAttentionForwardSm100Simple:
    def __init__(
        self,
        # dtype: Type[cutlass.Numeric],
        head_dim: int,
        head_dim_v: Optional[int] = None,
    ):
        print(
            "USING SIMPLE_18 (is_causal=False, is_local=False, is_split_kv=False, pack_gqa=False, q_subtile_factor=None, score_mod=None, mask_mod=None, has_aux_tensors=False, paged_kv_non_tma=False, is_varlen_q=False, use_2cta_instrs=True, is_persistent=True, m_block_size=128, n_block_size=128, q_stage=2, qhead_per_kvhead=1)"
        )
        # 128(head_dim)
        # 128(head_dim_v)
        # 1(qhead_per_kvhead)
        # is_causal=False (hardcoded)
        # False(is_local)
        # False(is_split_kv)
        # False(pack_gqa)
        # None(q_subtile_factor)
        # None(score_mod)
        # None(mask_mod)
        # False(has_aux_tensors)
        # False(is_varlen_q)
        # True(use_2cta_instrs)
        # True(is_persistent)
        # 128(m_block_size)
        # 128(n_block_size)
        # 2(q_stage)
        # 1(qhead_per_kvhead)

        self.use_tma_KV = True  # paged_kv_non_tma=False hardcoded in simple_9
        # use_2cta_instrs hardcoded to True in simple_11
        # is_persistent hardcoded to True in simple_12
        # m_block_size hardcoded to 128 in simple_13
        # n_block_size hardcoded to 128 in simple_14
        # q_stage hardcoded to 2 in simple_15
        # qhead_per_kvhead hardcoded to 1 in simple_16
        # self.dtype = dtype
        assert head_dim == 128, "simple_17 assumes head_dim=128"
        head_dim_v = head_dim if head_dim_v is None else head_dim_v
        assert head_dim_v == 128, "simple_17 assumes head_dim_v=128"
        self.head_dim_padded = 128
        self.head_dim_v_padded = 128
        self.m_block_size = 128  # hardcoded in simple_13
        self.n_block_size = 128  # hardcoded in simple_14
        self.q_stage = 2  # hardcoded in simple_15
        assert self.q_stage in [1, 2]
        # If split_P_arrive, the softmax warps write some columns of P first, signal to the MMA warp
        # to being the P @ V MMA, then write the rest of P and signal again. This allows some overlap
        # between compute the last couple columns of P and the P @ V MMA.
        self.split_P_arrive = self.n_block_size // 4 * 3
        self.split_P_arrive = int(self.split_P_arrive / 32) * 32  # multiple of 32
        #self.split_P_arrive = 0
        assert self.split_P_arrive % 32 == 0
        assert self.split_P_arrive < self.n_block_size
        self.arch = BaseDSL._get_dsl().get_arch_enum()
        assert self.arch >= Arch.sm_100 and self.arch <= Arch.sm_110f, (
            "Only SM 10.x and 11.x are supported"
        )

        self.cta_group_size = 2
        # cta_tiler M includes only 1 CTA, the scheduler will take into account the cluster shape
        self.cta_tiler = (
            self.q_stage * self.m_block_size,
            self.n_block_size,
            self.head_dim_padded,
        )
        # With 2CTA, the MMA tiler M covers both CTAs, so it's cta_group_size * m_block_size.
        # Each CTA owns m_block_size rows; the 2CTA MMA instruction spans both.
        self.mma_tiler_qk = (
            self.cta_group_size * self.m_block_size,
            self.n_block_size,
            self.head_dim_padded,
        )
        self.mma_tiler_pv = (
            self.cta_group_size * self.m_block_size,
            self.head_dim_v_padded,
            self.n_block_size,
        )
        self.qk_acc_dtype = Float32
        self.pv_acc_dtype = Float32
        self.cluster_shape_mn = (2, 1)
        self.use_2cta_instrs = True  # hardcoded in simple_11
        self.is_persistent = True  # hardcoded in simple_12
        # is_causal, is_local, is_split_kv, pack_gqa, q_subtile_factor, score_mod, mask_mod, has_aux_tensors, paged_kv_non_tma, is_varlen_q, use_2cta_instrs hardcoded in simple_11
        self.use_correction_warps_for_epi = False  # is_varlen_q=False
        self.qhead_per_kvhead = 1  # hardcoded in simple_16 (MHA)
        self.vec_size: cutlass.Constexpr = 2  # score_mod=None, has_aux_tensors=False
        # Does S1 need to wait for S0 to finish
        is_sm103 = self.arch >= Arch.sm_103 and self.arch <= Arch.sm_103f
        self.enable_ex2_emu = not is_sm103
        #self.enable_ex2_emu = False
        self.s0_s1_barrier = False
        self.overlap_sO_sQ = False

        self.softmax0_warp_ids = (0, 1, 2, 3)
        self.softmax1_warp_ids = (4, 5, 6, 7)
        self.correction_warp_ids = (8, 9, 10, 11)
        self.mma_warp_id = 12
        self.epilogue_warp_ids = (13,)
        self.load_warp_ids = (14,)
        self.empty_warp_ids = (15,)
        self.tmem_alloc_cols = cute.arch.get_max_tmem_alloc_cols("sm_100")

        self.threads_per_cta = cute.arch.WARP_SIZE * len(
            (
                *self.softmax0_warp_ids,
                *self.softmax1_warp_ids,
                *self.correction_warp_ids,
                self.mma_warp_id,
                *self.load_warp_ids,
                *self.epilogue_warp_ids,
                *self.empty_warp_ids,
            )
        )

        if self.q_stage == 1:
            self.empty_warp_ids = self.empty_warp_ids + self.softmax1_warp_ids
            self.softmax1_warp_ids = ()

        self.tmem_s_offset = [0, self.n_block_size]  # e.g., s0:0-128, s1:128-256 # add by n_block_size
        self.tmem_o_offset = [
            self.tmem_s_offset[-1] + self.n_block_size + i * self.head_dim_v_padded
            for i in range(self.q_stage)
        ]  # e.g., o0: 256-384, o1: 384-512 # add by head_dim_v_padded
        self.tmem_total = self.tmem_o_offset[-1] + self.head_dim_v_padded
        assert self.tmem_total <= self.tmem_alloc_cols
        self.tmem_s_to_p_offset = self.n_block_size // 2 # 64, fp32 to fp16
        self.tmem_p_offset = [
            self.tmem_s_offset[i] + self.tmem_s_to_p_offset for i in range(2)
        ]  # p0: 64-128, p1: 192-256

        # vec buffer for row_max & row_sum
        self.tmem_vec_offset = self.tmem_s_offset

        self.num_regs_softmax = 192
        self.num_regs_correction = 80
        self.num_regs_other = 48

        self.buffer_align_bytes = 1024

    def _setup_attributes(self):
        """Set up configurations and parameters for the FMHA kernel operation.

        This method initializes and configures various attributes required for the
        execution of the fused multi-head attention kernel, mainly about the pipeline stages:

        - Sets up staging parameters for Q, K, V inputs and accumulator data
        - Configures pipeline stages for softmax, correction, and epilogue operations
        """

        smem_size_q = (
            self.q_stage
            * self.m_block_size
            * self.head_dim_padded
            * self.q_dtype.width
            // 8
        )
        smem_size_o = (
            self.q_stage
            * self.m_block_size
            * self.head_dim_v_padded
            * self.o_dtype.width
            // 8
        )
        smem_size_q_o = (
            smem_size_q + smem_size_o
            if not self.overlap_sO_sQ
            else max(smem_size_q, smem_size_o)
        )
        smem_size_k_per_stage = (
            self.n_block_size * self.head_dim_padded * self.k_dtype.width // 8
        )
        smem_size_v_per_stage = (
            self.n_block_size * self.head_dim_v_padded * self.v_dtype.width // 8
        )
        smem_size_kv_per_stage = (
            max(smem_size_k_per_stage, smem_size_v_per_stage) // self.cta_group_size
        )
        kv_stage = (224 * 1024 - smem_size_q_o) // smem_size_kv_per_stage
        self.kv_stage = kv_stage
        # print("kv_stage", self.kv_stage)
        self.s_stage = 2
        assert self.s_stage >= self.q_stage

    # mQ: (0,0,0,0) o (8192,128,16,4):(1@1,1@0,1@2,1@3)
    # mK: (0,0,0,0) o (8192,128,16,4):(1@1,1@0,1@2,1@3)
    # mV: (0,0,0,0) o (128,8192,16,4):(1@0,1@1,1@2,1@3)
    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,  # (b, s_q, h, d) or (total_q, h, d) if there is cu_seqlens_q
        mK: cute.Tensor,  # (b_k, s_k, h_k, d) or (total_k, h_k, d) if there is cu_seqlens_k
        mV: cute.Tensor,  # (b_k, s_k, h_k, dv) or (total_k, h_k, dv) if there is cu_seqlens_k
        mO: cute.Tensor,  # (b, s_q, h, dv) or (total_q, h, dv) if there is cu_seqlens_q
        mLSE: Optional[cute.Tensor],
        softmax_scale: Float32,
        stream: cuda.CUstream,
        window_size_left: Int32 | int | None = None,
        window_size_right: Int32 | int | None = None,
        learnable_sink: Optional[cute.Tensor] = None,
    ):
        """Execute the Fused Multi-Head Attention operation on the provided tensors.

        This method prepares the input tensors for processing, validates their shapes and types,
        configures the computation parameters, and launches the CUDA kernel.

        The method handles:
        1. Tensor layout transformations for specific memory access patterns
        2. Validation of tensor shapes and data types
        3. Initialization of hardware-specific parameters and memory layouts
        4. Configuration of TMA (Tensor Memory Access) operations
        5. Grid and work scheduling computation
        6. Kernel launch with appropriate parameters
        """
        # setup static attributes before smem/grid/tma computation
        print("Local called")
        self.q_dtype = mQ.element_type
        self.k_dtype = mK.element_type
        self.v_dtype = mV.element_type
        self.o_dtype = mO.element_type
        mQ, mK, mV, mO = [assume_tensor_aligned(t) for t in (mQ, mK, mV, mO)]
        Q_layout_transpose = [1, 3, 2, 0]
        mQ = cute.make_tensor(
            mQ.iterator, cute.select(mQ.layout, mode=Q_layout_transpose)
        )
        # (s_k, d, h_k, b_k) or (total_k, d, h_k) if there's cu_seqlens_k
        KV_layout_transpose = [1, 3, 2, 0]
        mK, mV = [
            cute.make_tensor(
                t.iterator, cute.select(t.layout, mode=KV_layout_transpose)
            )
            for t in (mK, mV)
        ]
        O_layout_transpose = [1, 3, 2, 0]
        LSE_layout_transpose = [2, 1, 0]
        num_splits = Int32(1)
        mO = cute.make_tensor(
            mO.iterator, cute.select(mO.layout, mode=O_layout_transpose)
        )
        mLSE = (
            cute.make_tensor(
                mLSE.iterator, cute.select(mLSE.layout, mode=LSE_layout_transpose)
            )
            if const_expr(mLSE is not None)
            else None
        )
        # (s, d, h, b) -> (d, s, h, b)
        V_layout_transpose = [1, 0, 2, 3]
        mV = cute.make_tensor(
            mV.iterator, cute.select(mV.layout, mode=V_layout_transpose)
        )

        # check type consistency
        if const_expr(self.q_dtype != self.k_dtype):
            raise TypeError(f"Type mismatch: {self.q_dtype} != {self.k_dtype}")
        if const_expr(self.q_dtype != self.v_dtype):
            raise TypeError(f"Type mismatch: {self.q_dtype} != {self.v_dtype}")
        self._setup_attributes()
        # simple_16 runtime assumption keeps epilogue on the universal-copy path
        self.use_tma_O = False
        # This can be tuned
        # This is currently very ad-hoc, we should tune it systematically
        self.ex2_emu_freq = 0
        # self.ex2_emu_start_frg = 1 if self.is_causal else 0
        self.ex2_emu_start_frg = 1
        self.ex2_emu_freq = 12
        # if const_expr(self.enable_ex2_emu):
        #    self.ex2_emu_freq = 16
        #    if const_expr(self.head_dim_padded == 128 and True):
        #        self.ex2_emu_freq = 12

        cta_group = tcgen05.CtaGroup.TWO
        q_major_mode = tcgen05.OperandMajorMode.K
        k_major_mode = tcgen05.OperandMajorMode.K
        v_major_mode = tcgen05.OperandMajorMode.MN
        self.o_layout = cutlass.utils.LayoutEnum.from_tensor(mO)
        # the intermediate tensor p is from tmem & mK-major
        p_source = tcgen05.OperandSource.TMEM
        p_major_mode = tcgen05.OperandMajorMode.K
        tiled_mma_qk = sm100_utils_basic.make_trivial_tiled_mma(
            self.q_dtype,
            q_major_mode,
            k_major_mode,
            self.qk_acc_dtype,
            cta_group,
            self.mma_tiler_qk[:2],
        )
        tiled_mma_pv = sm100_utils_basic.make_trivial_tiled_mma(
            self.v_dtype,
            p_major_mode,
            v_major_mode,
            self.pv_acc_dtype,
            cta_group,
            self.mma_tiler_pv[:2],
            p_source,
        )

        self.cluster_shape_mnk = (*self.cluster_shape_mn, 1)
        cta_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk), (tiled_mma_qk.thr_id.shape,)
        )

        # epi_tile is per-CTA (not full 2CTA) since each CTA writes its own O portion
        self.epi_tile = (self.m_block_size, self.head_dim_v_padded)

        sQ_layout = sm100_utils_basic.make_smem_layout_a(
            tiled_mma_qk, self.mma_tiler_qk, self.q_dtype, self.q_stage
        )
        sK_layout = sm100_utils_basic.make_smem_layout_b(
            tiled_mma_qk, self.mma_tiler_qk, self.k_dtype, self.kv_stage
        )
        tP_layout = sm100_utils_basic.make_smem_layout_a(
            tiled_mma_pv, self.mma_tiler_pv, self.q_dtype, self.s_stage
        )
        sV_layout = sm100_utils_basic.make_smem_layout_b(
            tiled_mma_pv, self.mma_tiler_pv, self.v_dtype, self.kv_stage
        )
        sO_layout = sm100_utils_basic.make_smem_layout_epi(
            self.o_dtype, self.o_layout, self.epi_tile, self.q_stage
        )
        print(tiled_mma_qk)
        print(f"sQ_layout: {sQ_layout}")
        print(f"sK_layout: {sK_layout}")
        print(f"sV_layout: {sV_layout}")
        print(f"sO_layout: {sO_layout}")
        self.tma_copy_bytes = {
            name: cute.size_in_bytes(
                mX.element_type, cute.select(layout, mode=[0, 1, 2])
            )
            for name, mX, layout in [
                ("Q", mQ, sQ_layout),
                ("K", mK, sK_layout),
                ("V", mV, sV_layout),
            ]
        }
        for name in ("Q", "K", "V"):
            self.tma_copy_bytes[name] *= self.cta_group_size
        print(f"self.tma_copy_bytes: {self.tma_copy_bytes}")

        # TMA load for Q
        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(cta_group)
        tma_store_op = cpasync.CopyBulkTensorTileS2GOp()

        tma_atom_Q, mQ = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            mQ,
            cute.select(sQ_layout, mode=[0, 1, 2]),
            self.mma_tiler_qk,
            tiled_mma_qk,
            cta_layout_vmnk.shape,
        )

        # TMA load for K
        tma_atom_K, mK = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            mK,
            cute.select(sK_layout, mode=[0, 1, 2]),
            self.mma_tiler_qk,
            tiled_mma_qk,
            cta_layout_vmnk.shape,
        )
        # TMA load for V
        tma_atom_V, mV = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            mV,
            cute.select(sV_layout, mode=[0, 1, 2]),
            self.mma_tiler_pv,
            tiled_mma_pv,
            cta_layout_vmnk.shape,
        )

        self.num_epilogue_threads = cute.arch.WARP_SIZE * len(self.epilogue_warp_ids)
        universal_copy_bits = 128
        async_copy_elems = universal_copy_bits // self.o_dtype.width
        atom_universal_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.o_dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        tO_shape_dim_1 = sO_layout.outer.shape[1][0] // async_copy_elems
        tO_layout = cute.make_ordered_layout(
            (self.num_epilogue_threads // tO_shape_dim_1, tO_shape_dim_1),
            order=(1, 0),
        )
        # So that we don't have to check if we overshoot kBlockM when we store O
        assert self.m_block_size % tO_layout.shape[0] == 0
        vO_layout = cute.make_layout((1, async_copy_elems))
        gmem_tiled_copy_O = cute.make_tiled_copy_tv(
            atom_universal_copy, tO_layout, vO_layout
        )

        # is_local is always False, is_persistent=True hardcoded, use StaticPersistentTileScheduler
        TileScheduler = StaticPersistentTileScheduler
        tile_sched_args = TileSchedulerArguments(
            # num_block: Number of Q blocks = ceil(seqlen_q / m_block_size)
            # mQ.shape[0] = seqlen_q, cta_tiler[0] = m_block_size (e.g., 128)
            cute.ceil_div(cute.size(mQ.shape[0]), self.cta_tiler[0]),
            # num_head: Number of KV heads
            # mQ.shape[2] = num_heads (H dimension)
            cute.size(mQ.shape[2]),
            # num_batch: Number of batches
            # mQ.shape[3] = batch dimension (B)
            cute.size(mQ.shape[3]),
            # num_splits: Number of KV splits for split-KV (1 = no split)
            num_splits,
            # seqlen_k: KV sequence length
            # mK.shape[0] = seqlen_k (N dimension)
            cute.size(mK.shape[0]),
            # headdim: Query/Key head dimension (D)
            # mQ.shape[1] = head_dim
            mQ.shape[1],
            # headdim_v: Value head dimension (D_v)
            # mV.shape[0] = head_dim_v (different from Sm90 due to V transpose)
            mV.shape[
                0
            ],  # Note that this is different from Sm90 since we transpose mV in Sm100
            # total_q: Total Q tokens = seqlen_q * num_batch (for non-varlen)
            total_q=cute.size(mQ.shape[0]) * cute.size(mQ.shape[3]),
            # tile_shape_mn: CTA tile shape (M, N) for attention
            tile_shape_mn=self.cta_tiler[:2],
            # qhead_per_kvhead_packgqa: Q heads per KV head (1 for MHA/simple attention)
            qhead_per_kvhead_packgqa=1,
            # element_size: Bytes per element (FP16/BF16 = 2)
            element_size=self.k_dtype.width // 8,
            # is_persistent: Use persistent kernel (fixed CTAs, multiple tiles each)
            is_persistent=True,  # hardcoded in simple_18
            # lpt: Use LPT scheduling for L2 cache optimization
            lpt=False,  # is_local hardcoded to False
            # is_split_kv: Split KV for long sequences
            is_split_kv=False,
            # cluster_shape_mn: CTAs per cluster for Blackwell
            cluster_shape_mn=self.cluster_shape_mn,
        )
        tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
        self.tile_scheduler_cls = TileScheduler
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)

        sO_size = cute.cosize(sO_layout) if const_expr(not self.overlap_sO_sQ) else 0
        sQ_size = (
            cute.cosize(sQ_layout)
            if const_expr(not self.overlap_sO_sQ)
            else cutlass.max(
                cute.cosize(sQ_layout),
                cute.cosize(sO_layout) * self.o_dtype.width // self.q_dtype.width,
            )
        )

        @cute.struct
        class SharedStorage:
            # m_barriers for pipelines
            mbar_load_Q: cute.struct.MemRange[Int64, self.q_stage * 2]
            mbar_load_KV: cute.struct.MemRange[Int64, self.kv_stage * 2]
            mbar_S_full_P_full_O_rescaled: cute.struct.MemRange[Int64, self.q_stage * 2]
            mbar_P_full_lastsplit: cute.struct.MemRange[Int64, self.q_stage * 2]
            mbar_O_full: cute.struct.MemRange[Int64, self.q_stage * 2]
            mbar_softmax_stats: cute.struct.MemRange[Int64, self.q_stage * 2]
            # mbar_softmax_stats: cute.struct.MemRange[Int64, self.q_stage * 4 * 2]
            mbar_O_epi: cute.struct.MemRange[Int64, self.q_stage * 2]
            mbar_s0_s1_sequence: cute.struct.MemRange[Int64, 2 * 2]
            # Tmem dealloc cluster barrier
            tmem_dealloc_mbar_ptr: Int64
            # Tmem holding buffer
            tmem_holding_buf: Int32
            # Smem tensors
            # store row max and row sum
            sScale: cute.struct.MemRange[Float32, self.q_stage * self.m_block_size * 2]
            sO: cute.struct.Align[
                cute.struct.MemRange[self.o_dtype, sO_size], self.buffer_align_bytes
            ]
            sQ: cute.struct.Align[
                cute.struct.MemRange[self.q_dtype, sQ_size], self.buffer_align_bytes
            ]
            sK: cute.struct.Align[
                cute.struct.MemRange[self.k_dtype, cute.cosize(sK_layout)],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        LOG2_E = math.log2(math.e)
        # score_mod is always None in simple_6
        softmax_scale_log2 = softmax_scale * LOG2_E
        softmax_scale = None

        assert window_size_left is None and window_size_right is None, (
            "simple_18 does not support window masks"
        )
        assert learnable_sink is None, "simple_18 does not support learnable sink"

        fastdiv_mods = (None, None)

        head_divmod = None

        # Launch the kernel synchronously
        self.kernel(
            mQ,
            mK,
            mV,
            mO,
            mLSE,
            tma_atom_Q,
            tma_atom_K,
            tma_atom_V,
            softmax_scale_log2,
            softmax_scale,
            sQ_layout,
            sK_layout,
            tP_layout,
            sV_layout,
            sO_layout,
            gmem_tiled_copy_O,
            tiled_mma_qk,
            tiled_mma_pv,
            tile_sched_params,
            num_splits,
            fastdiv_mods,
            head_divmod,
        ).launch(
            grid=grid_dim,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk
            if cute.size(self.cluster_shape_mnk) > 1
            else None,
            stream=stream,
            min_blocks_per_mp=1,
        )

    #  GPU device kernel
    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,  # (s_q, d, h, b) or (total_q, d, h) if there is cu_seqlens_q
        mK: cute.Tensor,  # (s_k, d, h_k, b_k) or (total_k, d, h_k) if there is cu_seqlens_k
        mV: cute.Tensor,  # (d, s_k, h_k, b_k) or (d, total_k, h_k) if there is cu_seqlens_k
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: Optional[cute.CopyAtom],
        tma_atom_V: Optional[cute.CopyAtom],
        softmax_scale_log2: Float32,
        softmax_scale: Float32 | None,
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        tP_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        gmem_tiled_copy_O: cute.TiledCopy,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tile_sched_params: ParamsBase,
        num_splits: Int32,
        fastdiv_mods=(None, None),
        head_divmod=None,
    ):
        """The device kernel implementation of the Fused Multi-Head Attention.

        This kernel coordinates multiple specialized warps to perform different phases of the FMHA computation:
        1. Load warp: Loads Q, K, V data from global memory to shared memory using TMA
        2. MMA warp: Performs matrix multiplications (Q*K^T and P*V)
        3. Softmax warps: Compute softmax normalization on attention scores
        4. Correction warps: Apply adjustments to intermediate results
        5. Epilogue warp: Handles final output transformation and storage

        The kernel implements a complex pipeline with overlapping computation and memory operations,
        using tensor memory access (TMA) for efficient data loading, warp specialization for different
        computation phases, and optional attention masking.
        """

        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, bidz = cute.arch.block_idx()

        # Prefetch tma descriptor
        if warp_idx == 0:
            for tma_atom in (tma_atom_Q, tma_atom_K, tma_atom_V):
                if const_expr(tma_atom is not None):
                    cpasync.prefetch_descriptor(tma_atom)

        cta_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk), (tiled_mma_qk.thr_id.shape,)
        )
        # Setup cta/thread coordinates
        bidx, _, _ = cute.arch.block_idx()
        if const_expr(cute.size(tiled_mma_qk.thr_id.shape) == 1):
            mma_tile_coord_v = 0
        else:
            mma_tile_coord_v = bidx % cute.size(tiled_mma_qk.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0

        # Alloc
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierFwd.TmemPtr),
            num_threads=cute.arch.WARP_SIZE
            * len(
                (
                    self.mma_warp_id,
                    *self.softmax0_warp_ids,
                    *self.softmax1_warp_ids,
                    *self.correction_warp_ids,
                )
            ),
        )
        # Tensor memory dealloc barrier init
        tmem = cutlass.utils.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.mma_warp_id,
            is_two_cta=True,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr,
        )

        ThreadCooperativeGroup = partial(
            pipeline.CooperativeGroup, pipeline.Agent.Thread
        )
        mma_warp = ThreadCooperativeGroup(len([self.mma_warp_id]))
        load_warps = ThreadCooperativeGroup(len(self.load_warp_ids))
        tma_warp = ThreadCooperativeGroup(1)
        softmax_warps = ThreadCooperativeGroup(len(self.softmax0_warp_ids))
        softmax_threads = ThreadCooperativeGroup(
            cute.arch.WARP_SIZE * len(self.softmax0_warp_ids)
        )
        # softmax_threads = ThreadCooperativeGroup(cute.arch.WARP_SIZE)
        correction_threads = ThreadCooperativeGroup(
            cute.arch.WARP_SIZE * len(self.correction_warp_ids)
        )
        # correction_threads = ThreadCooperativeGroup(cute.arch.WARP_SIZE)
        softmax_correction_threads = ThreadCooperativeGroup(
            cute.arch.WARP_SIZE * len(self.softmax0_warp_ids + self.correction_warp_ids)
        )
        epilogue_threads = ThreadCooperativeGroup(
            cute.arch.WARP_SIZE * len(self.epilogue_warp_ids)
        )
        # For UMMA-bridging pipelines: the non-MMA side spans both CTAs in the cluster,
        # so the thread count must include warps from both CTAs.
        softmax_warps_cluster = ThreadCooperativeGroup(
            len(self.softmax0_warp_ids) * self.cta_group_size
        )
        correction_threads_cluster = ThreadCooperativeGroup(
            cute.arch.WARP_SIZE * len(self.correction_warp_ids) * self.cta_group_size
        )
        softmax_correction_threads_cluster = ThreadCooperativeGroup(
            cute.arch.WARP_SIZE
            * len(self.softmax0_warp_ids + self.correction_warp_ids)
            * self.cta_group_size
        )
        pipeline_q = pipeline_custom.PipelineTmaUmma.create(
            barrier_storage=storage.mbar_load_Q.data_ptr(),
            num_stages=self.q_stage,
            producer_group=tma_warp,
            consumer_group=mma_warp,
            tx_count=self.tma_copy_bytes["Q"],
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        pipeline_kv = pipeline_custom.PipelineTmaUmma.create(
            barrier_storage=storage.mbar_load_KV.data_ptr(),
            num_stages=self.kv_stage,
            producer_group=tma_warp,
            consumer_group=mma_warp,
            tx_count=self.tma_copy_bytes["K"],
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        # This pipeline is not the typical producer-consumer pipeline. The "producer" mma warp
        # uses it to signal that S is ready, and the softmax threads wait for S to be ready.
        # When softmax threads write P to tmem and the correction threads have rescaled O, they
        # signal as "consumer". The mma warp then waits for that signal to do the P @ V gemm.
        pipeline_s_p_o = pipeline_custom.PipelineUmmaAsync.create(
            barrier_storage=storage.mbar_S_full_P_full_O_rescaled.data_ptr(),
            num_stages=self.q_stage,
            producer_group=mma_warp,
            consumer_group=softmax_correction_threads_cluster,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        pipeline_p_lastsplit = pipeline_custom.PipelineAsyncUmma.create(
            barrier_storage=storage.mbar_P_full_lastsplit.data_ptr(),
            num_stages=self.q_stage,
            producer_group=softmax_warps_cluster,
            consumer_group=mma_warp,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        # MMA warp uses this to signal to the correction warps that O is ready.
        pipeline_o_acc = pipeline_custom.PipelineUmmaAsync.create(
            barrier_storage=storage.mbar_O_full.data_ptr(),
            num_stages=self.q_stage,
            producer_group=mma_warp,
            consumer_group=correction_threads_cluster,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        pipeline_sm_stats = pipeline_custom.PipelineAsync.create(
            barrier_storage=storage.mbar_softmax_stats.data_ptr(),
            num_stages=self.q_stage,
            producer_group=softmax_threads,
            consumer_group=correction_threads,
            defer_sync=True,
        )
        # Should put the NamedBarrier inside the pipeline class so we'll just have pipeline_sm_stats
        sm_stats_barrier = pipeline_custom.NamedBarrier(
            barrier_id=int(NamedBarrierFwd.SoftmaxStatsW0),
            num_threads=cute.arch.WARP_SIZE * 2,
        )
        pipeline_o_epi = pipeline_custom.PipelineAsync.create(
            barrier_storage=storage.mbar_O_epi.data_ptr(),
            num_stages=self.q_stage,
            producer_group=correction_threads,
            consumer_group=epilogue_threads,
            defer_sync=True,
        )

        # Cluster arrive after barrier init
        pipeline_init_arrive(cluster_shape_mn=cta_layout_vmnk, is_relaxed=True)

        #  Generate smem tensor Q/K/V/O
        # (MMA, MMA_Q, MMA_D, PIPE)
        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        # (MMA, MMA_K, MMA_D, PIPE)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        # (MMA, MMA_K, MMA_D, PIPE)
        # Strip swizzle info to reuse smem
        sV = cute.make_tensor(
            cute.recast_ptr(sK.iterator, sV_layout.inner), sV_layout.outer
        )
        if const_expr(not self.overlap_sO_sQ):
            sO = storage.sO.get_tensor(sO_layout.outer, swizzle=sO_layout.inner)
        else:
            sO = cute.make_tensor(
                cute.recast_ptr(sQ.iterator, sO_layout.inner, self.o_dtype),
                sO_layout.outer,
            )

        sScale = storage.sScale.get_tensor(
            cute.make_layout(self.q_stage * self.m_block_size * 2)
        )

        thr_mma_qk = tiled_mma_qk.get_slice(mma_tile_coord_v)
        thr_mma_pv = tiled_mma_pv.get_slice(mma_tile_coord_v)

        qk_acc_shape = thr_mma_qk.partition_shape_C(self.mma_tiler_qk[:2])
        # This is a fake tensor, by right we need to retrieve tmem_ptr. But we know that we always
        # request 512 columns of tmem, so we know that it starts at 0.
        tStS = thr_mma_qk.make_fragment_C(cute.append(qk_acc_shape, self.s_stage))
        pv_acc_shape = thr_mma_pv.partition_shape_C(self.mma_tiler_pv[:2])
        tOtO = thr_mma_pv.make_fragment_C(cute.append(pv_acc_shape, self.q_stage))
        tOtO = cute.make_tensor(tOtO.iterator + self.tmem_o_offset[0], tOtO.layout)
        tP = cute.make_tensor(tStS.iterator, tP_layout.outer)
        tOrP = thr_mma_pv.make_fragment_A(tP)[None, None, None, 0]
        # Need to multiply by width ratio bc tP is in v_dtype but tmem offsets are in FP32
        tP_width_ratio = Float32.width // self.v_dtype.width
        # Need to adjust the stage stride manually since the two stages aren't contiguous in tmem
        tP_stage_stride = (
            self.tmem_p_offset[1] - self.tmem_p_offset[0]
        ) * tP_width_ratio
        tOrP = cute.make_tensor(
            tOrP.iterator + self.tmem_p_offset[0] * tP_width_ratio,
            cute.append(
                tOrP.layout,
                cute.make_layout((self.s_stage,), stride=(tP_stage_stride,)),
            ),
        )

        block_info = BlockInfo(
            # This is cta_tiler, not mma_tiler_qk, since we move by block by (2 * mma_tiler[0], mma_tiler[1])
            self.cta_tiler[0],
            self.cta_tiler[1],
            False,  # is_causal - hardcoded to False
            False,  # is_local - hardcoded to False
            False,
            qhead_per_kvhead_packgqa=1,
        )
        SeqlenInfoCls = partial(
            SeqlenInfoQK.create,
            seqlen_q_static=mQ.shape[0],
            seqlen_k_static=mK.shape[0],
        )
        TileSchedulerCls = partial(self.tile_scheduler_cls.create, tile_sched_params)

        # Cluster wait before tensor memory alloc
        pipeline_init_wait(cluster_shape_mn=cta_layout_vmnk)

        # ///////////////////////////////////////////////////////////////////////////////
        #  EMPTY
        # ///////////////////////////////////////////////////////////////////////////////
        for i in cutlass.range_constexpr(len(self.empty_warp_ids)):
            if warp_idx == self.empty_warp_ids[i]:
                cute.arch.setmaxregister_decrease(self.num_regs_other)

        # ///////////////////////////////////////////////////////////////////////////////
        #  LOAD
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx >= self.load_warp_ids[0] and warp_idx <= self.load_warp_ids[-1]:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            self.load(
                thr_mma_qk,
                thr_mma_pv,
                mQ,
                mK,
                mV,
                sQ,
                sK,
                sV,
                tma_atom_Q,
                tma_atom_K,
                tma_atom_V,
                pipeline_q,
                pipeline_kv,
                block_info,
                num_splits,
                SeqlenInfoCls,
                TileSchedulerCls,
            )

        # ///////////////////////////////////////////////////////////////////////////////
        #  MMA
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            # Alloc tensor memory buffer
            tmem.allocate(cute.arch.get_max_tmem_alloc_cols("sm_100"))
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.qk_acc_dtype)
            self.mma(
                tiled_mma_qk,
                tiled_mma_pv,
                sQ,
                sK,
                sV,
                tStS,
                tOtO,
                tOrP,
                pipeline_q,
                pipeline_kv,
                pipeline_s_p_o,
                pipeline_p_lastsplit,
                pipeline_o_acc,
                is_leader_cta,
                block_info,
                num_splits,
                SeqlenInfoCls,
                TileSchedulerCls,
            )
            # Dealloc the tensor memory buffer
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)

        # ///////////////////////////////////////////////////////////////////////////////
        #  Epilogue
        # ///////////////////////////////////////////////////////////////////////////////
        if (
            warp_idx >= self.epilogue_warp_ids[0]
            and warp_idx <= self.epilogue_warp_ids[-1]
        ):
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            self.epilogue_s2g(
                mO,
                sO,
                gmem_tiled_copy_O,
                pipeline_o_epi,
                block_info,
                num_splits,
                SeqlenInfoCls,
                TileSchedulerCls,
                mma_tile_coord_v,
            )

        # ///////////////////////////////////////////////////////////////////////////////
        #  Softmax
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx <= self.softmax1_warp_ids[-1]:
            # increase register after decreasing
            cute.arch.setmaxregister_increase(self.num_regs_softmax)
            # sync with mma warp before retrieving tmem ptr
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.qk_acc_dtype)
            softmax_loop = partial(
                self.softmax_loop,
                softmax_scale_log2=softmax_scale_log2,
                softmax_scale=softmax_scale,
                thr_mma_qk=thr_mma_qk,
                sScale=sScale,
                mLSE=mLSE,
                pipeline_s_p_o=pipeline_s_p_o,
                pipeline_p_lastsplit=pipeline_p_lastsplit,
                pipeline_sm_stats=pipeline_sm_stats,
                sm_stats_barrier=sm_stats_barrier,
                block_info=block_info,
                num_splits=num_splits,
                SeqlenInfoCls=SeqlenInfoCls,
                TileSchedulerCls=TileSchedulerCls,
                head_divmod=head_divmod,
            )

            stage = Int32(0 if warp_idx < self.softmax1_warp_ids[0] else 1)
            softmax_loop(stage=stage, tStS=tStS)

        # ///////////////////////////////////////////////////////////////////////////////
        #  Correction
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx >= self.correction_warp_ids[0] and warp_idx < self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_correction)
            # sync with mma warp before retrieving tmem ptr
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.qk_acc_dtype)
            self.correction_loop(
                thr_mma_qk,
                thr_mma_pv,
                tStS,
                tOtO,
                sScale,
                mO,
                mLSE,
                sO,
                pipeline_s_p_o,
                pipeline_o_acc,
                pipeline_sm_stats,
                sm_stats_barrier,
                pipeline_o_epi,
                gmem_tiled_copy_O,
                softmax_scale_log2,
                block_info,
                num_splits,
                SeqlenInfoCls,
                TileSchedulerCls,
            )

        return

    @cute.jit
    def load(
        self,
        thr_mma_qk: cute.core.ThrMma,
        thr_mma_pv: cute.core.ThrMma,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: Optional[cute.CopyAtom],
        tma_atom_V: Optional[cute.CopyAtom],
        pipeline_q: pipeline.PipelineAsync,
        pipeline_kv: pipeline.PipelineAsync,
        block_info: BlockInfo,
        num_splits: Int32,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
    ):
        num_load_threads = len(self.load_warp_ids) * cute.arch.WARP_SIZE
        tidx = cute.arch.thread_idx()[0] % num_load_threads
        bidx, bidy, bidz = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        q_producer_phase = Int32(1)
        kv_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.kv_stage
        )
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            mQ_cur = mQ[(None, None, None, batch_idx)][None, None, head_idx]
            # (128*2*2, 128)
            tiler_gQ = ((self.mma_tiler_qk[0] * self.q_stage), self.head_dim_padded)
            gQ = cute.local_tile(mQ_cur, tiler_gQ, (m_block, 0))  # (128 * 2 * 2, 128)
            gQ = cute.flat_divide(gQ, (self.mma_tiler_qk[0], ))
            gQ = cute.make_tensor(gQ.iterator, cute.select(gQ.layout, mode= [0, 2, 1]))
            #gQ = layout_utils.select(
            #    cute.flat_divide(gQ, (self.mma_tiler_qk[0],)), mode=[0, 2, 1]
            #)  # (128 * 2, 128, 2) (tma_tile_m, tma_tile_k, q_stage)

            head_idx_kv = head_idx // self.qhead_per_kvhead
            mK_cur, mV_cur = [t[None, None, head_idx_kv, batch_idx] for t in (mK, mV)]
            gK = cute.local_tile(
                mK_cur, cute.select(self.mma_tiler_qk, mode=[1, 2]), (None, 0)
            ) # (tma_tile_n, tma_tile_k, RestN)
            gV = cute.local_tile(
                mV_cur, cute.select(self.mma_tiler_pv, mode=[1, 2]), (0, None)
            ) # (tma_tile_k, tma_tile_n, RestN)
            # gQ: (256,128,2) (bM, bN, q_stage)
            # ((128,16),1,8,2), here thr_mma help divide the 256 to 128
            # ((mma_tile_m, mma_tile_k), bRestM, bRestK, q_stage)
            tSgQ = thr_mma_qk.partition_A(gQ)
            # ((64,16),1,8,64)
            # ((mma_tile_n, mma_tile_k), bRestN, bRestK, RestN)
            tSgK = thr_mma_qk.partition_B(gK)
            # ((64,16),1,8,64) but different strides
            # ((mma_tile_d, mma_tile_n), bRestD, bRestN, RestN)
            # ((mma_tile_n, mma_tile_k), bRestN, bRestK, RestN)
            tOgV = thr_mma_pv.partition_B(gV)
            # helper: 1. determine group modes 2. src -> dst, indices of stages
            load_Q_fn, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_Q, 0, cute.make_layout(1), tSgQ, sQ
            )

            tKsK, tKgK = cpasync.tma_partition(
                tma_atom_K,
                0,  # no multicast
                cute.make_layout(1),
                cute.group_modes(sK, 0, 3),
                cute.group_modes(tSgK, 0, 3),
            )
            tVsV, tVgV = cpasync.tma_partition(
                tma_atom_V,
                0,  # no multicast
                cute.make_layout(1),
                cute.group_modes(sV, 0, 3),
                cute.group_modes(tOgV, 0, 3),
            )

            load_Q = partial(
                self.load_Q, load_Q_fn, pipeline_q=pipeline_q, phase=q_producer_phase
            )
            load_K = partial(
                self.load_KV,
                tma_atom_K,
                tKgK,
                tKsK,
                sK,
                pipeline_kv=pipeline_kv,
                K_or_V="K",
            )
            load_V = partial(
                self.load_KV,
                tma_atom_V,
                tVgV,
                tVsV,
                sV,
                pipeline_kv=pipeline_kv,
                K_or_V="V",
            )

            # what current m_block should attend to
            n_block_min, n_block_max = block_info.get_n_block_min_max(
                seqlen, m_block, split_idx, num_splits
            )

            load_K(
                block=n_block_max - 1,
                producer_state=kv_producer_state,
            )  # K0
            if (
                const_expr(len(self.load_warp_ids) == 1)
                or warp_idx == self.load_warp_ids[0]
            ):
                # load_Q(block=0, stage=0)  # Q0
                pipeline_q.producer_acquire_w_index_phase(0, q_producer_phase)
                # pipeline_q.sync_object_empty.wait(0, q_producer_phase)
                tma_bar_ptr = pipeline_q.sync_object_full.get_barrier(0)
                # tma_bar_ptr = pipeline_kv.producer_get_barrier(kv_producer_state)
                load_Q_fn(src_idx=0, dst_idx=0, tma_bar_ptr=tma_bar_ptr)
            kv_producer_state.advance()
            if (
                const_expr(len(self.load_warp_ids) == 1)
                or warp_idx == self.load_warp_ids[0]
            ):
                # load_Q(block=1, stage=1)  # Q1
                pipeline_q.producer_acquire_w_index_phase(1, q_producer_phase)
                tma_bar_ptr = pipeline_q.sync_object_full.get_barrier(1)
                load_Q_fn(src_idx=1, dst_idx=1, tma_bar_ptr=tma_bar_ptr)
            q_producer_phase ^= 1
            load_V(
                block=n_block_max - 1,
                producer_state=kv_producer_state,
            )  # V0
            kv_producer_state.advance()
            for i in cutlass.range(n_block_max - 1 - n_block_min, unroll=1):
                n_block = n_block_max - 2 - i
                # if cute.arch.thread_idx()[0] % 32 == 0: cute.printf("n_block = {}", n_block)
                load_K(
                    block=n_block,
                    producer_state=kv_producer_state,
                )  # Ki
                kv_producer_state.advance()
                load_V(
                    block=n_block,
                    producer_state=kv_producer_state,
                )  # Vi
                kv_producer_state.advance()

            tile_scheduler.prefetch_next_work()
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()
            # End of persistent scheduler loop

        pipeline_kv.producer_tail(kv_producer_state)
        # This is equivalent to pipeline_q.producer_tail
        if (
            const_expr(len(self.load_warp_ids) == 1)
            or warp_idx == self.load_warp_ids[0]
        ):
            pipeline_q.producer_acquire_w_index_phase(
                self.q_stage - 1, q_producer_phase
            )

    @cute.jit
    def mma(
        self,
        tiled_mma_qk: cute.core.ThrMma,
        tiled_mma_pv: cute.core.ThrMma,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        tStS: cute.Tensor,
        tOtO: cute.Tensor,
        tOrP: cute.Tensor,
        pipeline_q: pipeline.PipelineAsync,
        pipeline_kv: pipeline.PipelineAsync,
        pipeline_s_p_o: pipeline.PipelineAsync,
        pipeline_p_lastsplit: pipeline.PipelineAsync,
        pipeline_o_acc: pipeline.PipelineAsync,
        is_leader_cta: Boolean,
        block_info: BlockInfo,
        num_splits: Int32,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
    ):
        tSrQ = tiled_mma_qk.make_fragment_A(sQ)
        tSrK = tiled_mma_qk.make_fragment_B(sK)
        tOrV = tiled_mma_pv.make_fragment_B(sV)
        tSrQs = (tSrQ[None, None, None, 0], tSrQ[None, None, None, 1])

        qk_mma_op, pv_mma_op = tiled_mma_qk.op, tiled_mma_pv.op
        qk_mma_idesc, pv_mma_idesc = (
            sm100_desc.mma_op_to_idesc(qk_mma_op),
            sm100_desc.mma_op_to_idesc(pv_mma_op),
        )
        q_smem_base = sm100_desc.smem_desc_base_from_tensor(sQ, sm100_desc.Major.K)
        k_smem_base = sm100_desc.smem_desc_base_from_tensor(sK, sm100_desc.Major.K)
        v_smem_base = sm100_desc.smem_desc_base_from_tensor(sV, sm100_desc.Major.MN)
        q_smem_start = [
            sm100_desc.make_smem_desc_start_addr(sQ[None, None, None, stage].iterator)
            for stage in range(self.q_stage)
        ]

        sm100_utils.declare_ptx_smem_desc(
            q_smem_start[self.q_stage - 1],
            q_smem_base,
            tSrQ[None, None, None, 0].layout,
            var_name_prefix="fa_fwd_q_smem_desc",
        )
        sm100_utils.declare_ptx_idesc(qk_mma_op, var_name="fa_fwd_qk_mma_idesc")
        sm100_utils.declare_ptx_idesc(pv_mma_op, var_name="fa_fwd_pv_mma_idesc")

        sQ_stage_stride = (sQ.layout.stride[-1] * sQ.element_type.width // 8) >> 4
        gemm_Si = [
            partial(
                # sm100_utils.gemm_ptx_precomputed,
                # self.tmem_s_offset[stage],
                # smem_desc_start_a=q_smem_start[stage],
                # idesc=qk_mma_idesc,
                # smem_desc_base_a=q_smem_base,
                # smem_desc_base_b=k_smem_base,
                # tCrA_layout=tSrQ[None, None, None, 0].layout,
                sm100_utils.gemm_ptx_precomputed_varname,
                self.tmem_s_offset[stage], # tS
                # idesc=qk_mma_idesc,
                smem_desc_base_b=k_smem_base, # sB
                tCrB_layout=tSrK[None, None, None, 0].layout, # for calculating stride
                smem_var_name_prefix=f"fa_fwd_q_smem_desc",
                idesc_var_name=f"fa_fwd_qk_mma_idesc",
                # q_stage 0 smem declaration (-stride because we decalred for q_stage 1
                # and when q_stage 1, we add it back
                # but make sure you have already desclared 1. The declared of 2 stages is sequential
                smem_offset=-sQ_stage_stride if stage == 0 else sQ_stage_stride, # stride for sA
                zero_init=True,
                cta_group=self.cta_group_size,
            )
            for stage in range(self.q_stage)
        ]
        # gemm_Si = [
        #     partial(
        #         sm100_utils.gemm,
        #         tiled_mma_qk,
        #         tStS[None, None, None, stage],
        #         tCrA=tSrQ[None, None, None, stage],
        #         zero_init=True,
        #     )
        #     for stage in range(self.q_stage)
        # ]
        gemm_Pi = [
            partial(
                # sm100_utils.gemm_ptx_precomputed,
                sm100_utils.gemm_ptx_partial,
                pv_mma_op,
                self.tmem_o_offset[stage], # sO, acc_tmem addr
                tOrP[None, None, None, stage], # tOrP, tCrA
                sA=None, # Optional, only if A in smem
                split_arrive=self.split_P_arrive, # stop at 96th
                # smem_desc_start_a=tOrP[None, None, None, stage].iterator.toint(),
                # smem_desc_start_a=self.tmem_p_offset[stage],
                # idesc=pv_mma_idesc,
                # smem_desc_base_a=None,
                # smem_desc_base_b=v_smem_base,
                # tCrA_layout=tOrP[None, None, None, 0].layout,
                # tCrB_layout=tOrV[None, None, None, 0].layout
                cta_group=self.cta_group_size,
            )
            for stage in range(self.q_stage)
        ]
        # gemm_Pi = [
        #     partial(
        #         sm100_utils.gemm, tOtO[None, None, None, stage], tCrA=tOrP[None, None, None, stage]
        #     )
        #     for stage in range(self.q_stage)
        # ]

        mma_q_consumer_phase = Int32(0)
        mma_kv_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.kv_stage
        )
        P_full_O_rescaled_phase = Int32(0)

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)

            block_iter_count = Int32(0)

            n_block_min, n_block_max = block_info.get_n_block_min_max(
                seqlen, m_block, split_idx, num_splits
            )
            block_iter_count = n_block_max - n_block_min

            if is_leader_cta:
                for stage in cutlass.range_constexpr(self.q_stage):
                    # GEMM_QK00 (Q0 * K0 -> S0) or GEMM_QK01 (Q1 * K0 -> S1)
                    # 1. wait for Q0 / Q1
                    pipeline_q.consumer_wait_w_index_phase(stage, mma_q_consumer_phase)
                    # 2. wait for K0
                    if const_expr(stage == 0):
                        pipeline_kv.consumer_wait(mma_kv_consumer_state)
                    Ki_index, Ki_phase = (
                        mma_kv_consumer_state.index,
                        mma_kv_consumer_state.phase,
                    )
                    tSrKi = tSrK[None, None, None, Ki_index]
                    # We don't need to acquire empty S0 / S1.
                    # For the first iteration, we don't need to wait as we're guaranteed S0 / S1
                    # are empty. For subsequent iterations, the wait happened at the end
                    # of the while loop.
                    # 3. gemm
                    # sm100_utils.gemm(tiled_mma_qk, tStS[None, None, None, stage], tSrQ[None, None, None, stage], tSrKi, zero_init=True)
                    sK_cur = sK[None, None, None, Ki_index]
                    # gemm_Si[stage](tCrB=tSrKi, sB=sK_cur)
                    gemm_Si[stage]( # A_smem_desc precomputed
                        smem_desc_start_b=sm100_desc.make_smem_desc_start_addr(
                            sK_cur.iterator
                        )
                    )
                    # gemm_Si[stage](tCrB=tSrKi)
                    # 4. release S0 / S1
                    pipeline_s_p_o.producer_commit_w_index(stage) # q_stages
                mma_q_consumer_phase ^= 1
                # 5. release K0
                pipeline_kv.consumer_release(mma_kv_consumer_state)
                mma_kv_consumer_state.advance()
                # End of GEMM (Q1 * K0 -> S1)
                # Note: Q0 & Q1 are still needed in the seqlen_kv loop
                # so we need to release them after the seqlen_kv loop

                # O hasn't been accumulated yet, its first MMA calculation doesn't need to accumulate
                block_loop_count = block_iter_count - 1
                O_should_accumulate = False
                # O0n-1, S0n-2, O1n-1, S1n-2, O0n-2... O01, S00, O11, S10
                for i in cutlass.range(block_loop_count, unroll=1):
                    # GEMM_PV00 (P0 * V0 -> O0_partial), O0 needs to be accumulated in the seqlen_kv loop
                    # 1. wait for V0
                    pipeline_kv.consumer_wait(mma_kv_consumer_state)
                    mma_kv_release_state = mma_kv_consumer_state.clone()
                    Vi_index, Vi_phase = (
                        mma_kv_consumer_state.index,
                        mma_kv_consumer_state.phase,
                    )
                    tOrVi = tOrV[None, None, None, Vi_index] # smem V
                    for stage in cutlass.range_constexpr(self.q_stage):
                        # 2. acquire corrected O0/O1_partial and P0 / P1
                        # For the first iteration in this work tile, waiting for O0/O1_partial
                        # means that the correction warps has finished reading tO during
                        # the last iteration of the previous work tile.
                        pipeline_s_p_o.producer_acquire_w_index_phase( # P is ready
                            stage, P_full_O_rescaled_phase
                        )
                        # 3. gemm
                        # sm100_utils.gemm(tiled_mma_pv, tOtO0, tOrP0, tOrVi, zero_init=True)
                        # gemm_Pi[stage](tCrB=tOrVi, sB=sV[None, None, None, Vi_index], zero_init=not O_should_accumulate)
                        sV_cur = sV[None, None, None, Vi_index] # smem V
                        gemm_Pi[stage](
                            # tCrA = tOrP, acc = tmem_o_offset
                            tCrB=tOrVi,
                            sB=sV_cur,
                            # smem_desc_start_b=sm100_desc.make_smem_desc_start_addr(sV_cur.iterator),
                            zero_init=not O_should_accumulate,
                            mbar_ptr=pipeline_p_lastsplit.sync_object_full.get_barrier(
                                stage
                            )
                            if self.split_P_arrive > 0 # wait in the 3/4 of mma
                            else None,
                            mbar_phase=P_full_O_rescaled_phase, # another pipe with same phase
                        )
                        # Don't need to signal O_full to the correction warps since the
                        # correction warps wait for the softmax warps anyway. By the time the softmax
                        # warps finished, S_i for the next iteration must have been done, so O_i-1
                        # must have been done as well.
                        # pipeline_o_acc.producer_commit_w_index(stage)
                        # 4. release V(i-1)
                        if const_expr(stage == self.q_stage - 1):
                            pipeline_kv.consumer_release(mma_kv_release_state)
                            mma_kv_release_state.advance()
                        # End of GEMM_PV00 (P0 * V0 -> O0_partial)

                        # GEMM_QK0i (Q0 * Ki -> S0)
                        # 1. wait for Ki
                        if const_expr(stage == 0):
                            mma_kv_consumer_state.advance()
                            pipeline_kv.consumer_wait(mma_kv_consumer_state)
                        Ki_index, Ki_phase = (
                            mma_kv_consumer_state.index,
                            mma_kv_consumer_state.phase,
                        )
                        # 2. gemm
                        # Don't need to wait for the softmax warp to have finished reading the previous
                        # Si, since this gemm is scheduled after the PV gemm, which guaranteed that Si
                        # has been read and Pi has been written.
                        # sm100_utils.gemm(tiled_mma_qk, tStS[None, None, None, stage], tSrQ[None, None, None, stage], tSrK[None, None, None, Ki_index], zero_init=True)
                        sK_cur = sK[None, None, None, Ki_index]
                        # gemm_Si[stage](tCrB=tSrK[None, None, None, Ki_index], sB=sK_cur)
                        gemm_Si[stage](
                            smem_desc_start_b=sm100_desc.make_smem_desc_start_addr(
                                sK_cur.iterator
                            )
                        )
                        # gemm_Si[stage](tCrB=tSrK[None, None, None, Ki_index])
                        # 3. release S0 / S1
                        pipeline_s_p_o.producer_commit_w_index(stage)
                        # End of GEMM_QK0i (Q0 * Ki -> S0)
                    # 4. release Ki
                    pipeline_kv.consumer_release(mma_kv_consumer_state)
                    mma_kv_consumer_state.advance()
                    P_full_O_rescaled_phase ^= 1
                    O_should_accumulate = True
                # End of seqlen_kv loop

                # release Q0 & Q1
                for stage in cutlass.range(self.q_stage):
                    pipeline_q.consumer_release_w_index(stage)

                # The last O00 and O10
                # GEMM_PV00 (P0 * V0 -> O0_partial), O0 needs to be accumulated in the seqlen_kv loop
                # 1. wait for V0
                pipeline_kv.consumer_wait(mma_kv_consumer_state)
                Vi_index, Vi_phase = (
                    mma_kv_consumer_state.index,
                    mma_kv_consumer_state.phase,
                )
                tOrVi = tOrV[None, None, None, Vi_index]
                for stage in cutlass.range_constexpr(self.q_stage):
                    # 2. acquire corrected Oi_partial and Pi
                    pipeline_s_p_o.producer_acquire_w_index_phase(
                        stage, P_full_O_rescaled_phase
                    )
                    # 3. gemm
                    # sm100_utils.gemm(tiled_mma_pv, tOtO0, tOrP0, tOrVi, zero_init=True)
                    # gemm_Pi[stage](tCrB=tOrVi, sB=sV[None, None, None, Vi_index], zero_init=not O_should_accumulate)
                    sV_cur = sV[None, None, None, Vi_index]
                    gemm_Pi[stage](
                        tCrB=tOrVi,
                        sB=sV_cur,
                        # smem_desc_start_b=sm100_desc.make_smem_desc_start_addr(sV_cur.iterator),
                        zero_init=not O_should_accumulate,
                        mbar_ptr=pipeline_p_lastsplit.sync_object_full.get_barrier(
                            stage
                        )
                        if self.split_P_arrive > 0 # wait in the 3/4 of mma
                        else None,
                        mbar_phase=P_full_O_rescaled_phase,
                    )
                    # 4. release accumulated O0_partial
                    # We do need O_full here since for the last tile, by the time the softmax warp
                    # has signaled to the correction warps, the softmax warp has just finished
                    # computing the row sum of the current tile. It does not guarantee that the 1st
                    # tile of the next work tile has been computed yet.
                    pipeline_o_acc.producer_commit_w_index(stage)
                    # End of GEMM_PV00 (P0 * V0 -> O0_partial)
                P_full_O_rescaled_phase ^= 1
                # 5. release Vi_end
                pipeline_kv.consumer_release(mma_kv_consumer_state)
                mma_kv_consumer_state.advance()
                # End of GEMM_PV1(i_end) (P1 * Vi_end -> O1)

            # Advance to next tile
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()
        # End of persistent scheduler loop

        # We don't need pipeline_s_p_o.producer_tail() since there's no dangling mbarrier at the end
        # pipeline_s_p_o.producer_acquire_w_index_phase(self.q_stage - 1, P_full_O_rescaled_phase)
        # We don't need pipeline_o_acc.producer_tail() since we don't call
        # pipeline_o_acc.producer_acquire() inside the loop.

    # for both softmax0 and softmax1 warp group
    @cute.jit
    def softmax_loop(
        self,
        stage: int | Int32,
        softmax_scale_log2: Float32,
        softmax_scale: Float32,
        thr_mma_qk: cute.core.ThrMma,
        tStS: cute.Tensor,  # ((TILE_M, TILE_N), 1, 1, q_stage)
        sScale: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        pipeline_s_p_o: pipeline.PipelineAsync,
        pipeline_p_lastsplit: pipeline.PipelineAsync,
        pipeline_sm_stats: pipeline.PipelineAsync,
        sm_stats_barrier: pipeline.NamedBarrier,
        block_info: BlockInfo,
        num_splits: Int32,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        head_divmod=None,
    ):
        """Compute softmax on attention scores from QK matrix multiplication.

        This method handles the softmax computation for either the first or second half of the
        attention matrix, depending on the 'stage' parameter. It calculates row-wise maximum
        and sum values needed for stable softmax computation, applies optional masking, and
        transforms raw attention scores into probability distributions.

        The implementation uses specialized memory access patterns and efficient math operations
        for computing exp(x) using exp2 functions. It also coordinates pipeline
        synchronization between MMA, correction, and sequence processing stages.
        """
        tidx = cute.arch.thread_idx()[0] % (
            cute.arch.WARP_SIZE
            # * (len(self.softmax0_warp_ids) if stage == 0 else len(self.softmax1_warp_ids)
            * (len(self.softmax0_warp_ids))
        )
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
        bidx, bidy, bidz = cute.arch.block_idx()

        cta_qk_tiler = (
            self.mma_tiler_qk[0] // thr_mma_qk.thr_id.shape,
            self.mma_tiler_qk[1],
        )
        tSAcc = tStS[(None, None), 0, 0, stage]  # (128, 128)
        #tStScale = cute.slice_(tSAcc, (None, 0))
        #tStScale = cute.local_tile(tSAcc, (self.m_block_size, 1), (0, 0)) # (128, 1)
        tStScale = cute.composition(tSAcc, cute.make_layout((self.m_block_size, 1)))
        tScS = thr_mma_qk.partition_C(cute.make_identity_tensor(self.mma_tiler_qk[:2])) # (256, 128) -> (128, 128)
        tScS = tScS[(None, None), 0, 0]  # (128, 128)
        tScScale = cute.composition(tScS, cute.make_layout((self.m_block_size, 1)))
        #tScS: (0,0) o (128,128):(1@0,1@1)
        #tSAcc: raw_ptr(0x0000000000000000: f32, tmem, align<1>) o (128,128):(65536,1)
        #tStScale: raw_ptr(0x0000000000000080: f32, tmem, align<1>) o (128,1):(65536,0)

        tilePlikeFP32 = self.mma_tiler_qk[1] // Float32.width * self.v_dtype.width # 128 // 4 * 2
        tStP_layout = cute.composition(
            tSAcc.layout, cute.make_layout((self.m_block_size, tilePlikeFP32))
        ) # (128, 128) -> (128, 64)
        tStP = cute.make_tensor(tSAcc.iterator + self.tmem_s_to_p_offset, tStP_layout)

        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), self.qk_acc_dtype # FP32
        )
        thr_tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, tSAcc).get_slice(tidx)
        tStS_t2r = thr_tmem_load.partition_S(tSAcc)  # (((32,32),1),1,4)

        tmem_store_scale_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(1)), Float32
        )
        thr_tmem_store_scale = tcgen05.make_tmem_copy(
            tmem_store_scale_atom, tStScale
        ).get_slice(tidx)
        tStScale_r2t = thr_tmem_store_scale.partition_D(tStScale) # ((32, 1), 1, 1)

        tmem_store_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(16)), Float32 # half because fp32->fp16
        )
        thr_tmem_store = tcgen05.make_tmem_copy(tmem_store_atom, tStP).get_slice(tidx)
        tStP_r2t = thr_tmem_store.partition_D(tStP)  # (((16,32),1),1,4)

        mma_si_consumer_phase = Int32(0)
        sm_stats_producer_phase = Int32(1)
        s0_s1_sequence_phase = Int32(1 if stage == 0 else 0)

        # self.warp_scheduler_barrier_init()

        warp_idx_in_wg = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            n_block_min, n_block_max = block_info.get_n_block_min_max(
                seqlen, m_block, split_idx, num_splits
            )

            softmax = SoftmaxSm100.create(
                softmax_scale_log2,
                rescale_threshold=8.0 if const_expr(self.q_dtype.width == 16) else 0.0,
                softmax_scale=softmax_scale,
            )
            softmax.reset()

            softmax_step = partial(
                self.softmax_step,
                softmax=softmax,
                thr_mma_qk=thr_mma_qk,
                pipeline_s_p_o=pipeline_s_p_o,
                pipeline_p_lastsplit=pipeline_p_lastsplit,
                pipeline_sm_stats=pipeline_sm_stats,
                sm_stats_barrier=sm_stats_barrier,
                thr_tmem_load=thr_tmem_load,
                thr_tmem_store=thr_tmem_store,
                thr_tmem_store_scale=thr_tmem_store_scale,
                tStS_t2r=tStS_t2r,
                tStScale_r2t=tStScale_r2t,
                tStP_r2t=tStP_r2t,
                sScale=sScale,
                stage=stage,
                batch_idx=batch_idx,
                head_idx=head_idx,
                m_block=(self.q_stage * m_block + stage) * self.cta_group_size,
                seqlen=seqlen,
                head_divmod=head_divmod,
            )

            pipeline_sm_stats.producer_acquire_w_index_phase(
                stage, sm_stats_producer_phase
            )
            sm_stats_producer_phase ^= 1

            (
                mma_si_consumer_phase,
                sm_stats_producer_phase,
                s0_s1_sequence_phase,
            ) = softmax_step(
                mma_si_consumer_phase,
                sm_stats_producer_phase,
                s0_s1_sequence_phase,
                n_block_max - 1,
                is_first=True,
            )
            # if tidx == 0 or bidx == 0:
            #    cute.printf("[SOFTMAX] stage {} {} {}",
            #    stage, softmax.row_sum[0],
            #    softmax.row_max[0])
            n_block_max -= 1
            # is_local is always False, so skip local mask handling
            # The remaining iterations have no masking (but may still need mask_mod)
            n_block_min_before_local_mask = (
                block_info.get_n_block_min_before_local_mask(
                    seqlen, m_block, n_block_min
                )
            )
            for n_tile in cutlass.range(
                n_block_max - n_block_min_before_local_mask, unroll=1
            ):
                n_block = n_block_max - n_tile - 1
                # mask_mod is always None in simple_7
                (
                    mma_si_consumer_phase,
                    sm_stats_producer_phase,
                    s0_s1_sequence_phase,
                ) = softmax_step(
                    mma_si_consumer_phase,
                    sm_stats_producer_phase,
                    s0_s1_sequence_phase,
                    n_block,
                )
            # is_local is always False, so skip local mask handling
            # Dense path always writes scale / signals
            # 2 x q_stage x m_block_size
            sScale[tidx + stage * self.m_block_size] = softmax.row_sum[0]
            if const_expr(mLSE is not None):
                sScale[
                    tidx + stage * self.m_block_size + self.q_stage * self.m_block_size
                ] = softmax.row_max[0]
            # pipeline_sm_stats.producer_commit_w_index(stage)
            sm_stats_barrier.arrive_w_index(index=stage * 4 + warp_idx)

            # # Write LSE to gmem
            # if const_expr(mLSE is not None):
            #     acc_O_mn_row_is_zero_or_nan = softmax.row_sum[0] == 0.0 or softmax.row_sum[0] != softmax.row_sum[0]
            #     scale = (
            #         cute.arch.rcp_approx(softmax.row_sum[0] if not acc_O_mn_row_is_zero_or_nan else 1.0)
            #     )
            #     LN2 = math.log(2.0)
            #     lse = (
            #         (softmax.row_max[0] * softmax.scale_log2 + cute.math.log2(softmax.row_sum[0], fastmath=True)) * LN2
            #         if not acc_O_mn_row_is_zero_or_nan else -Float32.inf
            #     )
            #     if const_expr(not seqlen.has_cu_seqlens_q):
            #         mLSE_cur = mLSE[None, head_idx, batch_idx]
            #     else:
            #         mLSE_cur = cute.domain_offset((seqlen.offset_q,), mLSE[None, head_idx])
            #     gLSE = cute.local_tile(mLSE_cur, (self.m_block_size,), (m_block * 2 + stage,))
            #     if tidx < seqlen.seqlen_q - (m_block * 2 + stage) * self.m_block_size:
            #         gLSE[tidx] = lse

            # Advance to next tile
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()
        # End of persistent scheduler loop

        # This is equivalent to pipeline_sm_stats.producer_tail
        pipeline_sm_stats.producer_acquire_w_index_phase(stage, sm_stats_producer_phase)
        # This is equivalent to pipeline_s0_s1.producer_tail

    @cute.jit
    def softmax_step(
        self,
        mma_si_consumer_phase: Int32,
        sm_stats_producer_phase: Int32,
        s0_s1_sequence_phase: Int32,
        n_block: Int32,
        softmax: SoftmaxSm100,
        thr_mma_qk: cute.core.ThrMma,
        pipeline_s_p_o: pipeline.PipelineAsync,
        pipeline_p_lastsplit: pipeline.PipelineAsync,
        pipeline_sm_stats: pipeline.PipelineAsync,
        sm_stats_barrier: pipeline.NamedBarrier,
        thr_tmem_load: cute.CopyAtom,
        thr_tmem_store: cute.CopyAtom,
        thr_tmem_store_scale: cute.CopyAtom,
        tStS_t2r: cute.Tensor,
        tStScale_r2t: cute.Tensor,
        tStP_r2t: cute.Tensor,
        sScale: cute.Tensor,
        stage: int | Int32,
        batch_idx: Int32,
        head_idx: Int32,
        m_block: Int32,
        seqlen,
        head_divmod=None,
        mask_fn: Optional[Callable] = None,
        is_first: bool = False,
    ) -> Tuple[cute.Int32, cute.Int32, cute.Int32]:
        """Perform a single step of the softmax computation on a block of attention scores.

        This method processes one block of the attention matrix, computing numerically stable
        softmax by first finding the row maximum, subtracting it from all elements, applying
        exponential function, and then normalizing by the sum of exponentials. It also handles
        optional masking of attention scores.

        The method involves several key operations:
        1. Loading attention scores from tensor memory
        2. Applying optional masking based on position
        3. Computing row-wise maximum values for numerical stability
        4. Transforming scores using exp2(x*scale - max*scale)
        5. Computing row sums for normalization
        6. Coordinating pipeline synchronization between different processing stages
        """
        tidx = cute.arch.thread_idx()[0] % (
            cute.arch.WARP_SIZE
            # * (len(self.softmax0_warp_ids) if stage == 0 else len(self.softmax1_warp_ids)
            * (len(self.softmax0_warp_ids))
        )
        bidx, bidy, bidz = cute.arch.block_idx()

        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
        tilePlikeFP32 = self.mma_tiler_qk[1] // Float32.width * self.v_dtype.width
        tScS = thr_mma_qk.partition_C(cute.make_identity_tensor(self.mma_tiler_qk[:2]))
        tScS = tScS[(None, None), 0, 0]  # (128, 128)
        # tScScale = cute.composition(tScS, cute.make_layout((self.m_block_size, 1)))
        cta_qk_tiler = (
            self.mma_tiler_qk[0] // thr_mma_qk.thr_id.shape,
            self.mma_tiler_qk[1],
        )
        tScS_shape = cta_qk_tiler  # (128, 128)
        tScP_shape = (tScS_shape[0], tilePlikeFP32)  # (128, 64)

        # Wait for Si
        pipeline_s_p_o.consumer_wait_w_index_phase(stage, mma_si_consumer_phase)
        tSrS_t2r = cute.make_fragment(
            thr_tmem_load.partition_D(tScS).shape, self.qk_acc_dtype
        )
        # (((32,32),1),1,4) -> ((32,1),1,4)
        cute.copy(thr_tmem_load, tStS_t2r, tSrS_t2r)
        # score_mod is always None in simple_6, skip apply_score_mod

        if const_expr(mask_fn is not None):
            mask_fn(tSrS_t2r, n_block=n_block)
        row_max, acc_scale = softmax.update_row_max(tSrS_t2r.load(), is_first)

        if const_expr(not is_first):
            # tSrScale_r2t = cute.make_fragment(thr_tmem_store_scale.partition_S(tScScale).shape, Float32)
            # tSrScale_r2t[0] = acc_scale
            # cute.copy(thr_tmem_store_scale, tSrScale_r2t, tStScale_r2t)
            # cute.arch.fence_view_async_tmem_store()
            thread_idx = thr_tmem_load.thr_idx
            sScale[thread_idx + stage * self.m_block_size] = acc_scale
            # if thread_idx == 0: cute.printf("softmax acc_scale stage %d: %f, row_max = %f\n", stage, acc_scale, row_max)
        # Notify correction wg that row_max is ready
        # pipeline_sm_stats.producer_commit_w_index(stage)
        sm_stats_barrier.arrive_w_index(index=stage * 4 + warp_idx)

        # if thread_idx == 0 and stage == 0: cute.print_tensor(tSrS_t2r)
        softmax.scale_subtract_rowmax(tSrS_t2r, row_max)
        tSrP_r2t_f32 = cute.make_fragment(
            thr_tmem_store.partition_S(cute.make_identity_tensor(tScP_shape)).shape,
            Float32,
        ) # ((16, 1), 1, 4)
        tSrP_r2t = cute.make_tensor(
            cute.recast_ptr(tSrP_r2t_f32.iterator, dtype=self.q_dtype), tSrS_t2r.layout
        ) # ((32, 1), 1, 4)
        # softmax.scale_apply_exp2_convert(tSrS_t2r, row_max, tSrP_r2t)
        softmax.apply_exp2_convert(
            tSrS_t2r, # ((32, 1), 1, 4)
            tSrP_r2t, # ((32, 1), 1, 4)
            ex2_emu_freq=self.ex2_emu_freq if const_expr(mask_fn is None) else 0,
            ex2_emu_start_frg=self.ex2_emu_start_frg,
        )
        # print(tSrP_r2t_f32, tStP_r2t)
        # cute.copy(thr_tmem_store, tSrP_r2t_f32, tStP_r2t)
        for i in cutlass.range_constexpr(cute.size(tStP_r2t.shape[2])):
            cute.copy(
                thr_tmem_store, tSrP_r2t_f32[None, None, i], tStP_r2t[None, None, i]
            )
            if const_expr(self.split_P_arrive > 0):
                split_P_arrive_idx = (
                    cute.size(tStP_r2t.shape[2])
                    * self.split_P_arrive
                    // self.n_block_size
                )
                if const_expr(i + 1 == split_P_arrive_idx):
                    # Notify mma warp that the 1st half of P is ready
                    cute.arch.fence_view_async_tmem_store()
                    pipeline_s_p_o.consumer_release_w_index(stage)
        # Notify mma warp that the 2nd half of P is ready
        cute.arch.fence_view_async_tmem_store()
        if const_expr(self.split_P_arrive > 0):
            cute.arch.sync_warp()
            with cute.arch.elect_one():
                pipeline_p_lastsplit.producer_commit_w_index(stage)
        else:
            pipeline_s_p_o.consumer_release_w_index(stage)
        pipeline_sm_stats.producer_acquire_w_index_phase(stage, sm_stats_producer_phase)
        softmax.update_row_sum(tSrS_t2r.load(), acc_scale, is_first)
        # acc_scale = cute.math.exp2(acc_scale_, fastmath=True)
        return (
            mma_si_consumer_phase ^ 1,
            sm_stats_producer_phase ^ 1,
            s0_s1_sequence_phase ^ 1,
        )

    @cute.jit
    def correction_loop(
        self,
        thr_mma_qk: cute.core.ThrMma,
        thr_mma_pv: cute.core.ThrMma,
        tStS: cute.Tensor,
        tOtO: cute.Tensor,
        sScale: cute.Tensor,
        mO: cute.Tensor,
        mLSE: cute.Tensor,
        sO: cute.Tensor,
        pipeline_s_p_o: pipeline.PipelineAsync,
        pipeline_o_acc: pipeline.PipelineAsync,
        pipeline_sm_stats: pipeline.PipelineAsync,
        sm_stats_barrier: pipeline.NamedBarrier,
        pipeline_o_epi: pipeline.PipelineAsync,
        gmem_tiled_copy_O: cute.TiledCopy,
        softmax_scale_log2: Float32,
        block_info: BlockInfo,
        num_splits: Int32,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
    ):
        tidx = cute.arch.thread_idx()[0] % (
            cute.arch.WARP_SIZE * len(self.correction_warp_ids)
        )
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
        bidx, bidy, bidz = cute.arch.block_idx()
        mma_tile_coord_v = thr_mma_qk.thr_idx

        tScS = thr_mma_qk.partition_C(cute.make_identity_tensor(self.mma_tiler_qk[:2]))
        tStScale_layout = cute.composition(
            tStS.layout, cute.make_layout((self.m_block_size, 1))
        )
        tStScales = tuple(
            cute.make_tensor(
                tStS.iterator + self.tmem_vec_offset[stage], tStScale_layout
            )
            for stage in range(self.q_stage)
        )
        tScScale = cute.composition(tScS, cute.make_layout((self.m_block_size, 1)))
        tmem_load_v_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(1)), self.qk_acc_dtype
        )
        thr_tmem_load_vec = tcgen05.make_tmem_copy(
            tmem_load_v_atom, tStScales[0]
        ).get_slice(tidx)

        tStScales_t2r = [
            thr_tmem_load_vec.partition_S(tStScales[stage])
            for stage in range(self.q_stage)
        ]
        tSrScale_t2r_shape = thr_tmem_load_vec.partition_D(tScScale).shape

        # First iter: no correction is required
        # Notify mma warp that O has been rescaled
        for stage in cutlass.range(self.q_stage):
            pipeline_s_p_o.consumer_release_w_index(stage)

        sm_stats_consumer_phase = Int32(0)
        o_corr_consumer_phase = Int32(0)
        corr_epi_producer_phase = Int32(1)

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            n_block_min, n_block_max = block_info.get_n_block_min_max(
                seqlen, m_block, split_idx, num_splits
            )

            mO_cur = mO[(None, None, None, batch_idx)][None, None, head_idx] # [M, 128]
            tiler_gO = ((self.mma_tiler_pv[0] * self.q_stage), self.head_dim_v_padded)
            gO = cute.local_tile(mO_cur, tiler_gO, (m_block, 0))  # (128 * 2 * 2, 128)
            gO = layout_utils.select(
                cute.flat_divide(gO, (self.mma_tiler_pv[0],)), mode=[0, 2, 1]
            )  # (256, 128, 2)
            gO = cute.flat_divide(gO, (self.mma_tiler_pv[0] // self.cta_group_size,))[
                None, mma_tile_coord_v, None, None
            ] # (128, 128, 2) (tmem_m, tmem_h, q_stages)

            # Default LSE to -inf for invalid split_idx tiles
            stats = [
                (
                    0.0,
                    -Float32.inf if const_expr(mLSE is not None) else None,
                    True,
                )
            ] * self.q_stage

            total_block_count = n_block_max - n_block_min
            has_work = True

            if has_work:
                # Ignore first signal from softmax as no correction is required
                # pipeline_sm_stats.consumer_wait_w_index_phase(0, sm_stats_consumer_phase)
                sm_stats_barrier.arrive_and_wait_w_index(index=0 * 4 + warp_idx)
                pipeline_sm_stats.consumer_release_w_index(0)
                # pipeline_sm_stats.consumer_wait_w_index_phase(1, sm_stats_consumer_phase)
                sm_stats_barrier.arrive_and_wait_w_index(index=1 * 4 + warp_idx)
                sm_stats_consumer_phase ^= 1

                tSrScale_t2r = cute.make_fragment(tSrScale_t2r_shape, Float32)
                for i in cutlass.range(total_block_count - 1, unroll=1):
                    for stage in cutlass.range_constexpr(self.q_stage):
                        # wait for S0 / S1
                        # pipeline_sm_stats.consumer_wait_w_index_phase(stage, sm_stats_consumer_phase)
                        sm_stats_barrier.arrive_and_wait_w_index(
                            index=stage * 4 + warp_idx
                        )
                        # cute.copy(tiled_tmem_load_vec, tStScales_t2r[stage], tSrScale_t2r)
                        # cute.arch.fence_view_async_tmem_load()
                        # scale = tSrScale_t2r[0]
                        scale = sScale[tidx + stage * self.m_block_size]
                        # vote.sync.ballot.b32 d, {!}a, membermask;
                        # rescale as long as one thread needs rescale
                        should_rescale = cute.arch.vote_ballot_sync(scale < 1.0) != 0
                        #should_rescale = True
                        # Don't need O_full anymore, since by the time softmax has signaled the correction
                        # warps, S_i must have been done, so O_i-1 must have been done as well.
                        # pipeline_o_acc.consumer_wait_w_index_phase(stage, o_corr_consumer_phase)
                        if should_rescale:
                            self.correction_rescale(
                                thr_mma_pv, tOtO[None, None, None, stage], tidx, scale
                            )
                        # Notify mma warp that O has been rescaled
                        pipeline_s_p_o.consumer_release_w_index(stage)
                        pipeline_sm_stats.consumer_release_w_index(
                            self.q_stage - 1 - stage
                        )
                    sm_stats_consumer_phase ^= 1
                    # o_corr_consumer_phase ^= 1
                pipeline_sm_stats.consumer_release_w_index(1)
                # End of seqlen_corr_loop_steps

                # Even in the case of self.overlap_sO_sQ, we can write to stage 0 of sO without
                # additional sync because the MMA in the top half must have been done.
                # Similarly we can write to stage 1 of sO without additional sync.
                for stage in cutlass.range_constexpr(self.q_stage):
                    # pipeline_sm_stats.consumer_wait_w_index_phase(stage, sm_stats_consumer_phase)
                    sm_stats_barrier.arrive_and_wait_w_index(index=stage * 4 + warp_idx)
                    # cute.copy(tiled_tmem_load_vec, tStScales_t2r[stage], tSrScale_t2r)
                    # cute.arch.fence_view_async_tmem_load()
                    # scale = tSrScale_t2r[0]
                    row_sum = sScale[tidx + stage * self.m_block_size]
                    if const_expr(mLSE is not None):
                        row_max = sScale[
                            tidx
                            + stage * self.m_block_size
                            + self.q_stage * self.m_block_size
                        ]
                    else:
                        row_max = None
                    pipeline_sm_stats.consumer_release_w_index(stage)
                    acc_O_mn_row_is_zero_or_nan = row_sum == 0.0 or row_sum != row_sum
                    stats[stage] = (row_sum, row_max, acc_O_mn_row_is_zero_or_nan)
                    # rcp.approx.ftz.f32 d, a
                    scale = cute.arch.rcp_approx(
                        row_sum if not acc_O_mn_row_is_zero_or_nan else 1.0
                    )
                    # Wait for the last O to be ready from the MMA warp
                    pipeline_o_acc.consumer_wait_w_index_phase(
                        stage, o_corr_consumer_phase
                    )
                    pipeline_o_epi.producer_acquire_w_index_phase(
                        stage, corr_epi_producer_phase
                    )
                    self.correction_epilogue(
                        thr_mma_pv,
                        tOtO[None, None, None, stage],
                        tidx,
                        stage,
                        m_block,
                        seqlen.seqlen_q,
                        scale,
                        sO[None, None, stage],
                    )
                    # Signal for the next work tile that O buffers in tmem are already read, so
                    # mma warp can write to them
                    pipeline_s_p_o.consumer_release_w_index(stage)
                    pipeline_o_epi.producer_commit_w_index(stage)
                    # if tidx == 0: cute.printf("Correction final scale for stage %d: %f\n", stage, scale)

                o_corr_consumer_phase ^= 1
                sm_stats_consumer_phase ^= 1
                corr_epi_producer_phase ^= 1
            if const_expr(mLSE is not None):
                # if const_expr(not seqlen.has_cu_seqlens_q):
                mLSE_cur = mLSE[None, head_idx, batch_idx]
                # else:
                #    offset = seqlen.offset_q
                #    mLSE_cur = cute.domain_offset((offset,), mLSE[None, head_idx])
                for stage in cutlass.range_constexpr(self.q_stage):
                    m_tile_idx = (
                        m_block * self.q_stage + stage
                    ) * self.cta_group_size + mma_tile_coord_v
                    gLSE = cute.local_tile(
                        mLSE_cur, (self.m_block_size,), (m_tile_idx,)
                    )
                    row_sum, row_max, acc_O_mn_row_is_zero_or_nan = stats[stage]
                    # if tidx == 0 and stage <= 1:
                    #     cute.printf("row_sum = {}, row_max = {}, acc_O_mn_row_is_zero_or_nan = {}\n", row_sum, row_max, acc_O_mn_row_is_zero_or_nan)
                    LN2 = math.log(2.0)
                    lse = (
                        (
                            row_max * softmax_scale_log2
                            + cute.math.log2(row_sum, fastmath=True)
                        )
                        * LN2
                        if not acc_O_mn_row_is_zero_or_nan
                        else -Float32.inf
                    )
                    seqlen_q = seqlen.seqlen_q
                    if tidx < seqlen_q - m_tile_idx * self.m_block_size:
                        gLSE[tidx] = lse

            # Advance to next tile
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()
        # End of persistent scheduler loop

        # This is equivalent to pipeline_o_epi.consumer_tail() for the correction warps
        pipeline_o_epi.producer_acquire_w_index_phase(
            self.q_stage - 1, corr_epi_producer_phase
        )

    @cute.jit
    def correction_rescale(
        self,
        thr_mma: cute.core.ThrMma,
        tOtO: cute.Tensor,
        tidx: Int32,
        scale: Float32,
    ):
        """Rescale intermediate attention results based on softmax normalization factor.

        This method performs a crucial correction step in the attention computation pipeline.
        When processing attention in blocks, the softmax normalization factors may change
        as new blocks are processed. This method rescales previously computed partial
        output values to account for updated normalization factors.

        The implementation uses efficient tensor memory operations to:
        1. Load existing partial attention output from tensor memory
        2. Apply the scaling factor to all elements
        3. Store the rescaled results back to tensor memory
        """
        tidx = cute.arch.thread_idx()[0] % (
            cute.arch.WARP_SIZE * len(self.correction_warp_ids)
        )
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
        bidx, bidy, bidz = cute.arch.block_idx()

        # (128, 128)
        tOcO = thr_mma.partition_C(cute.make_identity_tensor(self.mma_tiler_pv[:2]))
        corr_tile_size = 16  # tuneable parameter
        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(corr_tile_size)),
            self.pv_acc_dtype,
        )
        tmem_store_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(corr_tile_size)),
            self.pv_acc_dtype,
        )
        tOtO_i = cute.composition(
            tOtO, cute.make_layout((self.m_block_size, corr_tile_size))
        ) # (128, 16)
        tOcO_i = cute.composition(
            tOcO, cute.make_layout((self.m_block_size, corr_tile_size))
        ) # (128, 16)
        thr_tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, tOtO_i).get_slice(tidx)
        thr_tmem_store = tcgen05.make_tmem_copy(tmem_store_atom, tOtO_i).get_slice(tidx)
        tOtO_t2r = thr_tmem_load.partition_S(tOtO_i) # (((16,32),1),1,1)
        tOrO_t2r_shape = thr_tmem_load.partition_D(tOcO_i).shape # ((16, 1), 1, 1)
        tOtO_r2t = thr_tmem_store.partition_D(tOtO_i) # (((16,32),1),1,1)

        frg_count = self.head_dim_v_padded // corr_tile_size # 128 // 16 = 8
        # (((16, 1), 1, 1), 8)
        tOrO_frg = cute.make_fragment((tOrO_t2r_shape, frg_count), self.pv_acc_dtype)
        for i in cutlass.range_constexpr(frg_count):
            tOrO_frg = cute.make_fragment(tOrO_t2r_shape, self.pv_acc_dtype)
            tOtO_t2r_i = cute.make_tensor(
                tOtO_t2r.iterator + i * corr_tile_size, tOtO_t2r.layout
            )
            cute.copy(thr_tmem_load, tOtO_t2r_i, tOrO_frg)
            for j in cutlass.range(0, cute.size(tOrO_frg), 2, unroll_full=True):
                tOrO_frg[j], tOrO_frg[j + 1] = cute.arch.mul_packed_f32x2(
                    (tOrO_frg[j], tOrO_frg[j + 1]), (scale, scale)
                )
            tOtO_r2t_i = cute.make_tensor(
                tOtO_r2t.iterator + i * corr_tile_size, tOtO_r2t.layout
            )
            cute.copy(thr_tmem_store, tOrO_frg, tOtO_r2t_i)
        cute.arch.fence_view_async_tmem_store()

    @cute.jit
    def correction_epilogue(
        self,
        thr_mma: cute.core.ThrMma,
        tOtO: cute.Tensor,
        tidx: Int32,
        stage: Int32,
        m_block: Int32,
        seqlen_q: Int32,
        scale: Float32,
        sO: cute.Tensor,
        *legacy_epilogue_args,
    ):
        """Apply the final scaling in shared memory before gmem stores.

        `legacy_epilogue_args` absorbs the older `(mO_cur, gO, gmem_tiled_copy_O)`
        parameters for compatibility with upstream helpers.

        The method performs:
        1. Loading of accumulated attention results from tensor memory
        2. Application of the final output scaling factor
        3. Type conversion if necessary (typically from higher precision accumulator to output precision)
        4. Reorganization of data for optimal memory access patterns

        :param thr_mma: Thread MMA operation for the computation
        :type thr_mma: cute.core.ThrMma
        :param tOtO: Tensor containing accumulated attention output
        :type tOtO: cute.Tensor
        :param scale: Final scaling factor to apply to the output
        :type scale: Float32
        :param sO: Shared memory tensor for the final output
        :type sO: cute.Tensor
        """

        bidx, bidy, bidz = cute.arch.block_idx()
        corr_tile_size = 8 * 32 // self.o_dtype.width # 16
        # Use CTA 0 mapping for smem partitioning since sO is per-CTA sized
        # ((8, 16), (64, 2), (1, 2)) -> ((128, (64, 2)), 1, 1)
        tOsO = thr_mma.get_slice(0).partition_C(sO)
        # (256, 128) -> (128, 128)
        tOcO = thr_mma.partition_C(cute.make_identity_tensor(self.mma_tiler_pv[:2]))

        # ((128, 128), 1, 1) -> ((128, 16), 8)
        tOtO_i = cute.logical_divide(
            tOtO, cute.make_layout((self.m_block_size, corr_tile_size))
        ) # ((128, 16), 8)
        tOcO_i = cute.logical_divide(
            tOcO, cute.make_layout((self.m_block_size, corr_tile_size))
        ) # ((128, 16), 8)
        tOsO_i = cute.logical_divide(
            tOsO, cute.make_layout((self.m_block_size, corr_tile_size))
        ) # ((128, 16), (4, 2))
        epi_subtile = (self.epi_tile[0], corr_tile_size)
        tmem_copy_atom = sm100_utils_basic.get_tmem_load_op(
            self.mma_tiler_pv,
            self.o_layout,
            self.o_dtype,
            self.pv_acc_dtype,
            epi_subtile,
            use_2cta_instrs=True,
        )
        tiled_tmem_load = tcgen05.make_tmem_copy(
            tmem_copy_atom, tOtO_i[(None, None), 0]
        )
        thr_tmem_load = tiled_tmem_load.get_slice(tidx)
        smem_copy_atom = sm100_utils_basic.get_smem_store_op(
            self.o_layout, self.o_dtype, self.pv_acc_dtype, tiled_tmem_load
        )
        tiled_smem_store = cute.make_tiled_copy_D(smem_copy_atom, tiled_tmem_load)
        # (128, 16, 8) -> (((16,32),1),1,1,8)
        tOtO_t2r = thr_tmem_load.partition_S(tOtO_i[(None, None), None])
        # ((128, 16, (4, 2)) -> ((16, 1), 1, 1, (4, 2))
        tOsO_s2r = copy_utils.partition_D_position_independent(
            thr_tmem_load, tOsO_i[(None, None), None]
        ) # (((8,2),1),1,1,((2,2),2))
        tOcO_t2r = thr_tmem_load.partition_D(tOcO_i[(None, None), None]) # ((16,1),1,1,8)
        for i in cutlass.range(
            self.head_dim_v_padded // corr_tile_size, unroll_full=True
        ):
            tOtO_t2r_i = tOtO_t2r[None, 0, 0, i]
            tOsO_r2s_i = tOsO_s2r[None, 0, 0, i]
            tOrO_frg = cute.make_fragment(
                tOcO_t2r[None, 0, 0, i].shape, self.pv_acc_dtype
            )
            cute.copy(tiled_tmem_load, tOtO_t2r_i, tOrO_frg)
            for j in cutlass.range(0, cute.size(tOrO_frg), 2, unroll_full=True):
                tOrO_frg[j], tOrO_frg[j + 1] = cute.arch.mul_packed_f32x2(
                    (tOrO_frg[j], tOrO_frg[j + 1]), (scale, scale)
                )
            copy_utils.cvt_copy(tiled_smem_store, tOrO_frg, tOsO_r2s_i)
        cute.arch.fence_view_async_shared()

    @cute.jit
    def _store_O_to_gmem(
        self,
        sO_stage: cute.Tensor,
        gO: cute.Tensor,
        gmem_tiled_copy_O: cute.TiledCopy,
        tidx: Int32,
        seqlen_q: Int32,
        m_tile_idx: Int32,
    ):
        """Copy a single stage of O from smem to gmem via registers."""
        gmem_thr_copy_O = gmem_tiled_copy_O.get_slice(tidx)
        tOsO = gmem_thr_copy_O.partition_S(sO_stage)
        cO = cute.make_identity_tensor((self.m_block_size, self.head_dim_v_padded))
        tOgO = gmem_thr_copy_O.partition_D(gO)
        tOcO = gmem_thr_copy_O.partition_S(cO)
        t0OcO = gmem_tiled_copy_O.get_slice(0).partition_S(cO)
        # load acc O from smem to rmem for wider vectorization
        tOrO = cute.make_fragment_like(tOsO, self.o_dtype)
        cute.autovec_copy(tOsO, tOrO)
        # copy acc O from rmem to gmem
        for rest_m in cutlass.range_constexpr(cute.size(tOrO.shape[1])):
            if (
                t0OcO[0, rest_m, 0][0]
                < seqlen_q - m_tile_idx * self.m_block_size - tOcO[0][0]
            ):
                cute.copy(
                    gmem_tiled_copy_O,
                    tOrO[None, rest_m, None],
                    tOgO[None, rest_m, None],
                )

    @cute.jit
    def epilogue_s2g(
        self,
        mO: cute.Tensor,
        sO: cute.Tensor,
        gmem_tiled_copy_O: cute.TiledCopy,
        pipeline_o_epi: pipeline.PipelineAsync,
        block_info: BlockInfo,
        num_splits: int,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        mma_tile_coord_v: Int32 = 0,
    ):
        tidx = cute.arch.thread_idx()[0] % (
            cute.arch.WARP_SIZE * len(self.epilogue_warp_ids)
        )
        bidx, bidy, bidz = cute.arch.block_idx()
        epi_consumer_phase = Int32(0)
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            n_block_min, n_block_max = block_info.get_n_block_min_max(
                seqlen, m_block, split_idx, num_splits
            )

            mO_cur = mO[(None, None, None, batch_idx)][None, None, head_idx]
            tiler_gO = (
                (self.mma_tiler_pv[0] * self.q_stage),
                self.head_dim_v_padded,
            ) # (256 * 2, 128)
            gO = cute.local_tile(mO_cur, tiler_gO, (m_block, 0))  # (128 * 2 * 2, 128)
            gO = layout_utils.select(
                cute.flat_divide(gO, (self.mma_tiler_pv[0],)), mode=[0, 2, 1]
            ) # (128 * 2, 128, 2)
            gO = cute.flat_divide(gO, (self.mma_tiler_pv[0] // self.cta_group_size,))[
                None, mma_tile_coord_v, None, None
            ] # (128, 128, 2)

            for stage in cutlass.range_constexpr(self.q_stage):
                # wait from corr, issue copy on smem
                # 1. wait for O0 / O1 final
                pipeline_o_epi.consumer_wait_w_index_phase(stage, epi_consumer_phase)
                # 2. copy O0 / O1 to gmem
                m_tile_idx = (
                    m_block * self.q_stage + stage
                ) * self.cta_group_size + mma_tile_coord_v
                self._store_O_to_gmem(
                    sO[None, None, stage],
                    gO[None, None, stage],
                    gmem_tiled_copy_O,
                    tidx,
                    seqlen.seqlen_q,
                    m_tile_idx,
                )
                pipeline_o_epi.consumer_release_w_index(stage)

            epi_consumer_phase ^= 1

            # Advance to next tile
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    def load_Q(
        self,
        load_Q_fn: Callable,
        pipeline_q: pipeline.PipelineAsync,
        block: Int32,
        stage: int,
        phase: Int32,
    ):
        pipeline_q.producer_acquire_w_index_phase(stage, phase)
        load_Q_fn(
            src_idx=block,
            dst_idx=stage,
            tma_bar_ptr=pipeline_q.sync_object_full.get_barrier(stage),
        )

    @cute.jit
    def load_KV(
        self,
        tma_atom: cute.CopyAtom,
        tXgX: cute.Tensor,
        tXsX: cute.Tensor,
        sX: cute.Tensor,
        block: Int32,
        pipeline_kv: pipeline.PipelineAsync,
        producer_state: pipeline.PipelineState,
        K_or_V: Literal["K", "V"],
        extra_tx_count: Optional[Int32] = None,
    ):
        assert K_or_V in ("K", "V")
        stage, phase = producer_state.index, producer_state.phase
        extra_tx_count_kv = self.tma_copy_bytes[K_or_V] - self.tma_copy_bytes["K"]
        extra_tx_count = extra_tx_count_kv + (
            extra_tx_count if extra_tx_count is not None else 0
        )
        pipeline_kv.producer_acquire(producer_state, extra_tx_count=extra_tx_count)

        tXsX_cur = tXsX[None, stage]
        tXgX_cur = tXgX[None, block]
        cute.copy(
            tma_atom,
            tXgX_cur,
            tXsX_cur,
            tma_bar_ptr=pipeline_kv.producer_get_barrier(producer_state),
        )

    # @cute.jit
    # def warp_scheduler_barrier_init(self):
    #     warp_group_idx = utils.canonical_warp_group_idx(sync=False)
    #     if warp_group_idx == 0:
    #         cute.arch.barrier_arrive(
    #             barrier_id=int(NamedBarrierFwd.WarpSchedulerWG1), number_of_threads=2 * 128,
    #         )

    # def warp_scheduler_barrier_sync(self):
    #     cute.arch.barrier(
    #         barrier_id=int(NamedBarrierFwd.WarpSchedulerWG1) + utils.canonical_warp_group_idx(sync=False),
    #         number_of_threads=2 * 128
    #     )

    # def warp_scheduler_barrier_arrive(self):
    #     cur_wg = utils.canonical_warp_group_idx(sync=False)
    #     next_wg = 1 - cur_wg
    #     cute.arch.barrier_arrive(
    #         barrier_id=int(NamedBarrierFwd.WarpSchedulerWG1) + next_wg, number_of_threads=2 * 128,
    #     )
