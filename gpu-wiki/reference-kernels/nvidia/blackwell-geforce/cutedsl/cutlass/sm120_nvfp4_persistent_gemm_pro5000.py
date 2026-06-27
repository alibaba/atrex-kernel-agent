"""SM120 NVFP4 fully-optimized persistent GEMM — TUNED FOR RTX PRO 5000 (sm_120, 110 SMs).

⚠ This is the SCALE-UP / persistent / multi-stage version that follows the CUTLASS
sm_120 recipe. It builds on the m16n8k64 inline-PTX atom shown in
`sm120_nvfp4_inline_ptx_gemm.py` (atom-level demo / correctness reference).

Final perf (4096³): 581 TFLOPS = 71% of CUTLASS C++ (808 T) on RTX PRO 5000.
Required pack helper: `sm120_nvfp4_pack_helpers.pack_sf_per_block(sf, ATOM_M, ATOMS_M)`
(the original `pack_sf_per_atom` was 8× bloated; compressed pack frees the smem
headroom needed for BLOCK_K=128 + STAGES=4).

Tuning vs the atom demo:
  * Persistent kernel (NUM_CTAS = 110 = SM count) with row-major tile rasterization
  * BLOCK_M=128 BLOCK_N=128 BLOCK_K=128, K_BLOCK_MAX=2 inner MMA iters per K-tile
  * 1 producer warp + 8 consumer warps (4 in M × 2 in N), warp-specialized
  * STAGES=4 TMA pipeline (PipelineTmaAsync); separate sA0/sA1 sub-buffers per k_block
  * SF stored in COMPRESSED layout (8× smaller than the original CUTLASS-style
    bloated (128, 4)-per-atom layout); 1 cp.async per warp covers all atoms
  * SF cp.async prefetched 1 K-tile ahead (decoupled SF stage tracking)
  * `cute.recast_tensor` MUST be inside the dynamic K-loop (hoisting silently breaks)

Required cross-references:
  * Optimization journey + perf table + ncu signatures:
      docs/ref-docs/nvidia/cutedsl/sm120/sm120-nvfp4-persistent-gemm-pro5000-optimization.md
  * Pitfalls (recast hoisting, pack bloat, cute-DSL bulk+pipeline broken):
      docs/pitfalls/nvidia/cutedsl/nvfp4-gemm-pitfalls.md
  * Atom-level inline-PTX demo (`mma.sync.aligned.kind::mxf4nvf4...`) + register packing:
      reference-kernels/nvidia/blackwell-geforce/cutedsl/cutlass/sm120_nvfp4_inline_ptx_gemm.py
"""
import os
os.environ.setdefault("CUTE_DSL_ARCH", "sm_120a")

import cutlass
import cutlass.cute as cute
import cutlass.utils
import cutlass.pipeline as pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.nvgpu.cpasync import CopyG2SOp, LoadCacheMode
from cutlass.cute.nvgpu.warp import MmaMXF4NVF4Op
from cutlass.cute.nvgpu.warp.copy import LdMatrix8x8x16bOp

AB_DTYPE = cutlass.Float4E2M1FN
SF_DTYPE = cutlass.Float8E4M3FN
ACC_DTYPE = cutlass.Float32

ATOM_M, ATOM_N, ATOM_K = 16, 8, 64
SFA_PER_ATOM = ATOM_M  # 16 unique SF entries per atom (compressed)
SFB_PER_ATOM = ATOM_N  # 8 unique SF entries per atom
CWARPS_M = 4
CWARPS_N = 2
ATOMS_M_PER_WARP = 2
ATOMS_N_PER_WARP = 8
ATOMS_M = CWARPS_M * ATOMS_M_PER_WARP    # 8
ATOMS_N = CWARPS_N * ATOMS_N_PER_WARP    # 16
BLOCK_M = ATOMS_M * ATOM_M               # 128
BLOCK_N = ATOMS_N * ATOM_N               # 128
K_BLOCK_MAX = 2
BLOCK_K = K_BLOCK_MAX * ATOM_K           # 128
SF_VEC_SIZE = 16
PRODUCER_WARPS = 1
CONSUMER_WARPS = CWARPS_M * CWARPS_N     # 8
WARPS = PRODUCER_WARPS + CONSUMER_WARPS  # 9
THREADS = 32 * WARPS                     # 288
STAGES = 4                                # compressed SF gives lots of headroom
NUM_CTAS = 110


def _ir(v, loc=None, ip=None):
    return v.ir_value(loc=loc, ip=ip) if hasattr(v, "ir_value") else v
def _ext(v, i, ty, *, loc=None, ip=None):
    return llvm.extractelement(_ir(v, loc, ip), _ir(cutlass.Int32(i), loc, ip), loc=loc, ip=ip)


@dsl_user_op
def _mma_inline(a_regs, b_regs, c_regs, sfa, sfb, *, loc=None, ip=None):
    a = [_ext(a_regs, i, T.i32(), loc=loc, ip=ip) for i in range(4)]
    b = [_ext(b_regs, i, T.i32(), loc=loc, ip=ip) for i in range(2)]
    c = [_ext(c_regs, i, T.f32(), loc=loc, ip=ip) for i in range(4)]
    rst = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32()]*4),
        [*a, *b, *c,
         _ir(sfa, loc, ip), _ir(cutlass.Int16(0), loc, ip), _ir(cutlass.Int16(0), loc, ip),
         _ir(sfb, loc, ip), _ir(cutlass.Int16(0), loc, ip), _ir(cutlass.Int16(0), loc, ip)],
        """mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3
           {$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9},
           {$10, $11, $12, $13}, {$14}, {$15, $16}, {$17}, {$18, $19};""",
        "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f,r,h,h,r,h,h",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )
    out = cute.make_rmem_tensor((4,), ACC_DTYPE)
    for i in range(4):
        out[i] = cutlass.Float32(llvm.extractvalue(T.f32(), rst, [i]))
    return out.load()


_BYTES_A = 2 * BLOCK_M * (ATOM_K // 2)
_BYTES_B = 2 * BLOCK_N * (ATOM_K // 2)
_TX_BYTES = _BYTES_A + _BYTES_B


def _build_v43(M, N, K):
    K_HALF = K // 2
    SF_TILES_K = K // ATOM_K
    K_TILES = K // BLOCK_K
    GRID_X = (N + BLOCK_N - 1) // BLOCK_N
    GRID_Y = (M + BLOCK_M - 1) // BLOCK_M
    TOTAL_TILES = GRID_X * GRID_Y
    TILES_PER_CTA = (TOTAL_TILES + NUM_CTAS - 1) // NUM_CTAS

    @cute.kernel
    def gemm_v43_kernel(
        gA_SF, gB_SF, alpha, gO,
        tma_atom_a, tma_tensor_a,
        tma_atom_b, tma_tensor_b,
        tiled_copy_sfa, tiled_copy_sfb,
        smem_tiled_copy_A, smem_tiled_copy_B,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        cta_id, _, _ = cute.arch.block_idx()

        warp_id = tidx // cutlass.Int32(32)
        lane = tidx % cutlass.Int32(32)
        is_producer = warp_id == cutlass.Int32(0)
        is_consumer = warp_id >= cutlass.Int32(1)
        cwid = warp_id - cutlass.Int32(1)
        cwid_m = cwid % cutlass.Int32(CWARPS_M)
        cwid_n = cwid // cutlass.Int32(CWARPS_M)

        smem = cutlass.utils.SmemAllocator()
        # A,B: 2 sub-buffers per k_block (separated for ldmatrix correctness)
        sA0_layout = cute.make_layout((STAGES, ATOMS_M, ATOM_M, ATOM_K // 2),
                                       stride=(BLOCK_M * (ATOM_K // 2),
                                               ATOM_M * (ATOM_K // 2),
                                               ATOM_K // 2, 1))
        sB0_layout = cute.make_layout((STAGES, ATOMS_N, ATOM_N, ATOM_K // 2),
                                       stride=(BLOCK_N * (ATOM_K // 2),
                                               ATOM_N * (ATOM_K // 2),
                                               ATOM_K // 2, 1))
        # Compressed SF: per atom (atom_dim, 4) bytes. Innermost stride = 1 byte.
        # SFA: ATOMS_M atoms × SFA_PER_ATOM × 4 = 8 × 16 × 4 = 512 bytes per (k_block, stage)
        # SFB: ATOMS_N atoms × SFB_PER_ATOM × 4 = 16 × 8 × 4 = 512 bytes per (k_block, stage)
        sSFA_layout = cute.make_layout((STAGES, K_BLOCK_MAX, ATOMS_M, SFA_PER_ATOM, 4),
                                       stride=(K_BLOCK_MAX * ATOMS_M * SFA_PER_ATOM * 4,
                                               ATOMS_M * SFA_PER_ATOM * 4,
                                               SFA_PER_ATOM * 4,
                                               4, 1))
        sSFB_layout = cute.make_layout((STAGES, K_BLOCK_MAX, ATOMS_N, SFB_PER_ATOM, 4),
                                       stride=(K_BLOCK_MAX * ATOMS_N * SFB_PER_ATOM * 4,
                                               ATOMS_N * SFB_PER_ATOM * 4,
                                               SFB_PER_ATOM * 4,
                                               4, 1))
        sA0 = smem.allocate_tensor(cutlass.Uint8, sA0_layout, 128)
        sA1 = smem.allocate_tensor(cutlass.Uint8, sA0_layout, 128)
        sB0 = smem.allocate_tensor(cutlass.Uint8, sB0_layout, 128)
        sB1 = smem.allocate_tensor(cutlass.Uint8, sB0_layout, 128)
        sSFA = smem.allocate_tensor(cutlass.Uint8, sSFA_layout, 128)
        sSFB = smem.allocate_tensor(cutlass.Uint8, sSFB_layout, 128)
        mbar_storage = smem.allocate_array(cutlass.Int64, STAGES * 2)

        producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, CONSUMER_WARPS)
        cta_layout_vmnk = cute.make_layout((1, 1, 1, 1))
        p = pipeline.PipelineTmaAsync.create(
            barrier_storage=mbar_storage,
            num_stages=STAGES,
            producer_group=producer_group,
            consumer_group=consumer_group,
            tx_count=_TX_BYTES,
            cta_layout_vmnk=cta_layout_vmnk,
        )
        pipeline_init_arrive(cluster_shape_mn=(1, 1))
        pipeline_init_wait(cluster_shape_mn=(1, 1))

        sA0_for_tma = cute.make_tensor(sA0.iterator,
                                        cute.make_layout((BLOCK_M, ATOM_K // 2, STAGES),
                                                         stride=(ATOM_K // 2, 1,
                                                                 BLOCK_M * (ATOM_K // 2))))
        sA1_for_tma = cute.make_tensor(sA1.iterator,
                                        cute.make_layout((BLOCK_M, ATOM_K // 2, STAGES),
                                                         stride=(ATOM_K // 2, 1,
                                                                 BLOCK_M * (ATOM_K // 2))))
        sB0_for_tma = cute.make_tensor(sB0.iterator,
                                        cute.make_layout((BLOCK_N, ATOM_K // 2, STAGES),
                                                         stride=(ATOM_K // 2, 1,
                                                                 BLOCK_N * (ATOM_K // 2))))
        sB1_for_tma = cute.make_tensor(sB1.iterator,
                                        cute.make_layout((BLOCK_N, ATOM_K // 2, STAGES),
                                                         stride=(ATOM_K // 2, 1,
                                                                 BLOCK_N * (ATOM_K // 2))))
        cta_layout_1 = cute.make_layout(1)
        sA0_grouped = cute.group_modes(sA0_for_tma, 0, 2)
        sA1_grouped = cute.group_modes(sA1_for_tma, 0, 2)
        sB0_grouped = cute.group_modes(sB0_for_tma, 0, 2)
        sB1_grouped = cute.group_modes(sB1_for_tma, 0, 2)

        gA_full_tiled = cute.local_tile(tma_tensor_a, (BLOCK_M, ATOM_K // 2), (None, None))
        gB_full_tiled = cute.local_tile(tma_tensor_b, (BLOCK_N, ATOM_K // 2), (None, None))

        # Compressed SF gmem tiles: per (mb, kt) chunk holds ALL atoms (M-side or N-side)
        # SFA: ATOMS_M × SFA_PER_ATOM = 128 rows × 4 bytes = 512 bytes per (mb, kt)
        # SFB: ATOMS_N × SFB_PER_ATOM = 128 rows × 4 bytes = 512 bytes per (bx, kt)
        gSFA_tiled = cute.local_tile(gA_SF, (ATOMS_M * SFA_PER_ATOM, 4), (None, 0))
        gSFB_tiled = cute.local_tile(gB_SF, (ATOMS_N * SFB_PER_ATOM, 4), (None, 0))

        op = MmaMXF4NVF4Op(AB_DTYPE, ACC_DTYPE, SF_DTYPE)
        tiled_mma = cute.make_tiled_mma(op)
        thr_mma = tiled_mma.get_slice(lane)
        thr_lmc_A = smem_tiled_copy_A.get_slice(lane)
        thr_lmc_B = smem_tiled_copy_B.get_slice(lane)

        sA_atom_proto = cute.recast_tensor(sA0[0, 0, None, None], AB_DTYPE)
        sB_atom_proto = cute.recast_tensor(sB0[0, 0, None, None], AB_DTYPE)
        tCsA_proto = thr_mma.partition_A(sA_atom_proto)
        tCsB_proto = thr_mma.partition_B(sB_atom_proto)
        tCrA = [tiled_mma.make_fragment_A(tCsA_proto) for _ in range(ATOMS_M_PER_WARP)]
        tCrB = [tiled_mma.make_fragment_B(tCsB_proto) for _ in range(ATOMS_N_PER_WARP)]
        tCrA_view = [smem_tiled_copy_A.retile(t) for t in tCrA]
        tCrB_view = [smem_tiled_copy_B.retile(t) for t in tCrB]

        gO_proto_sub = cute.make_tensor(gO.iterator,
                                        cute.make_layout((ATOM_M, ATOM_N), stride=(N, 1)))
        tCgO_proto = thr_mma.partition_C(gO_proto_sub)
        acc = [[tiled_mma.make_fragment_C(tCgO_proto)
                for _ in range(ATOMS_N_PER_WARP)] for _ in range(ATOMS_M_PER_WARP)]

        prod_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, STAGES)
        cons_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, STAGES)

        thr_copy_sfa = tiled_copy_sfa.get_slice(lane)
        thr_copy_sfb = tiled_copy_sfb.get_slice(lane)

        # Compressed SF read: thread `lane` reads byte at row=sfa_logical (NO *4)
        sfa_logical = (lane // cutlass.Int32(4)) + (lane % cutlass.Int32(2)) * cutlass.Int32(8)
        sfb_logical = lane // cutlass.Int32(4)

        for tile_off in cutlass.range(TILES_PER_CTA, unroll=1):
            tile_idx = cta_id * cutlass.Int32(TILES_PER_CTA) + tile_off
            if tile_idx < cutlass.Int32(TOTAL_TILES):
                by = tile_idx // cutlass.Int32(GRID_X)
                bx = tile_idx % cutlass.Int32(GRID_X)
                m0 = by * cutlass.Int32(BLOCK_M)
                n0 = bx * cutlass.Int32(BLOCK_N)

                gA_tiled_t = gA_full_tiled[None, None, by, None]
                gB_tiled_t = gB_full_tiled[None, None, bx, None]
                gA_grouped = cute.group_modes(gA_tiled_t, 0, 2)
                gB_grouped = cute.group_modes(gB_tiled_t, 0, 2)
                tA0sA, tAgA_full = cpasync.tma_partition(tma_atom_a, 0, cta_layout_1,
                                                          sA0_grouped, gA_grouped)
                tA1sA, _         = cpasync.tma_partition(tma_atom_a, 0, cta_layout_1,
                                                          sA1_grouped, gA_grouped)
                tB0sB, tBgB_full = cpasync.tma_partition(tma_atom_b, 0, cta_layout_1,
                                                          sB0_grouped, gB_grouped)
                tB1sB, _         = cpasync.tma_partition(tma_atom_b, 0, cta_layout_1,
                                                          sB1_grouped, gB_grouped)

                gO_subs = [[
                    cute.make_tensor(
                        gO.iterator
                        + (m0 + (cwid_m * cutlass.Int32(ATOMS_M_PER_WARP) + cutlass.Int32(mi_l))
                               * cutlass.Int32(ATOM_M)) * cutlass.Int32(N)
                        + (n0 + (cwid_n * cutlass.Int32(ATOMS_N_PER_WARP) + cutlass.Int32(ni_l))
                               * cutlass.Int32(ATOM_N)),
                        cute.make_layout((ATOM_M, ATOM_N), stride=(N, 1)),
                    ) for ni_l in range(ATOMS_N_PER_WARP)
                ] for mi_l in range(ATOMS_M_PER_WARP)]

                for mi_l in cutlass.range_constexpr(ATOMS_M_PER_WARP):
                    for ni_l in cutlass.range_constexpr(ATOMS_N_PER_WARP):
                        acc[mi_l][ni_l].fill(0.0)

                if is_producer:
                    for kt in cutlass.range(K_TILES, unroll=1):
                        p.producer_acquire(prod_state)
                        bar = p.producer_get_barrier(prod_state)
                        st = prod_state.index
                        kt_g0 = kt * cutlass.Int32(K_BLOCK_MAX)
                        kt_g1 = kt_g0 + cutlass.Int32(1)
                        cute.copy(tma_atom_a, tAgA_full[(None, kt_g0)], tA0sA[(None, st)], tma_bar_ptr=bar)
                        cute.copy(tma_atom_b, tBgB_full[(None, kt_g0)], tB0sB[(None, st)], tma_bar_ptr=bar)
                        cute.copy(tma_atom_a, tAgA_full[(None, kt_g1)], tA1sA[(None, st)], tma_bar_ptr=bar)
                        cute.copy(tma_atom_b, tBgB_full[(None, kt_g1)], tB1sB[(None, st)], tma_bar_ptr=bar)
                        p.producer_commit(prod_state)
                        prod_state.advance()

                if is_consumer:
                    # SF prologue: K-tile 0's two k_blocks into stage 0
                    # 1 cp.async per (warp, k_block) per (SFA, SFB) = 4 cp.async per warp per K-tile
                    # gSFA_tiled[None, None, idx] selects the idx-th (128, 4)-byte chunk = all atoms for one (mb, kt)
                    for kb_pre in cutlass.range_constexpr(K_BLOCK_MAX):
                        sfa_chunk_idx0 = by * cutlass.Int32(SF_TILES_K) + cutlass.Int32(kb_pre)
                        sSFA_chunk = cute.make_tensor(sSFA[0, kb_pre, None, None, None].iterator,
                                                      cute.make_layout((ATOMS_M * SFA_PER_ATOM, 4),
                                                                       stride=(4, 1)))
                        cute.copy(tiled_copy_sfa,
                                  thr_copy_sfa.partition_S(gSFA_tiled[None, None, sfa_chunk_idx0]),
                                  thr_copy_sfa.partition_D(sSFA_chunk))
                        sfb_chunk_idx0 = bx * cutlass.Int32(SF_TILES_K) + cutlass.Int32(kb_pre)
                        sSFB_chunk = cute.make_tensor(sSFB[0, kb_pre, None, None, None].iterator,
                                                      cute.make_layout((ATOMS_N * SFB_PER_ATOM, 4),
                                                                       stride=(4, 1)))
                        cute.copy(tiled_copy_sfb,
                                  thr_copy_sfb.partition_S(gSFB_tiled[None, None, sfb_chunk_idx0]),
                                  thr_copy_sfb.partition_D(sSFB_chunk))
                    cute.arch.cp_async_commit_group()

                    for kt in cutlass.range(K_TILES, unroll=1):
                        sf_st_curr = kt % cutlass.Int32(STAGES)
                        sf_st_pref = (kt + cutlass.Int32(1)) % cutlass.Int32(STAGES)
                        kt_pref = cutlass.min(kt + cutlass.Int32(1),
                                              cutlass.Int32(K_TILES - 1))
                        for kb_pre in cutlass.range_constexpr(K_BLOCK_MAX):
                            sfa_chunk_idx_p = by * cutlass.Int32(SF_TILES_K) + kt_pref * cutlass.Int32(K_BLOCK_MAX) + cutlass.Int32(kb_pre)
                            sSFA_chunk_p = cute.make_tensor(sSFA[sf_st_pref, kb_pre, None, None, None].iterator,
                                                            cute.make_layout((ATOMS_M * SFA_PER_ATOM, 4),
                                                                             stride=(4, 1)))
                            cute.copy(tiled_copy_sfa,
                                      thr_copy_sfa.partition_S(gSFA_tiled[None, None, sfa_chunk_idx_p]),
                                      thr_copy_sfa.partition_D(sSFA_chunk_p))
                            sfb_chunk_idx_p = bx * cutlass.Int32(SF_TILES_K) + kt_pref * cutlass.Int32(K_BLOCK_MAX) + cutlass.Int32(kb_pre)
                            sSFB_chunk_p = cute.make_tensor(sSFB[sf_st_pref, kb_pre, None, None, None].iterator,
                                                            cute.make_layout((ATOMS_N * SFB_PER_ATOM, 4),
                                                                             stride=(4, 1)))
                            cute.copy(tiled_copy_sfb,
                                      thr_copy_sfb.partition_S(gSFB_tiled[None, None, sfb_chunk_idx_p]),
                                      thr_copy_sfb.partition_D(sSFB_chunk_p))
                        cute.arch.cp_async_commit_group()
                        cute.arch.cp_async_wait_group(1)

                        p.consumer_wait(cons_state)
                        cst = cons_state.index

                        for k_block in cutlass.range_constexpr(K_BLOCK_MAX):
                            sA_kb = sA0 if k_block == 0 else sA1
                            sB_kb = sB0 if k_block == 0 else sB1

                            for mi_l in cutlass.range_constexpr(ATOMS_M_PER_WARP):
                                mi_g = cwid_m * cutlass.Int32(ATOMS_M_PER_WARP) + cutlass.Int32(mi_l)
                                sA_atom_fp4 = cute.recast_tensor(sA_kb[cst, mi_g, None, None], AB_DTYPE)
                                cute.copy(smem_tiled_copy_A,
                                          thr_lmc_A.partition_S(sA_atom_fp4),
                                          tCrA_view[mi_l])
                            for ni_l in cutlass.range_constexpr(ATOMS_N_PER_WARP):
                                ni_g = cwid_n * cutlass.Int32(ATOMS_N_PER_WARP) + cutlass.Int32(ni_l)
                                sB_atom_fp4 = cute.recast_tensor(sB_kb[cst, ni_g, None, None], AB_DTYPE)
                                cute.copy(smem_tiled_copy_B,
                                          thr_lmc_B.partition_S(sB_atom_fp4),
                                          tCrB_view[ni_l])

                            tCrA_u32 = [cute.recast_tensor(t, cutlass.Uint32) for t in tCrA]
                            tCrB_u32 = [cute.recast_tensor(t, cutlass.Uint32) for t in tCrB]

                            sSFA_u32 = cute.recast_tensor(sSFA, cutlass.Uint32)
                            sSFB_u32 = cute.recast_tensor(sSFB, cutlass.Uint32)

                            sfb_regs = [cutlass.Uint32(0)] * ATOMS_N_PER_WARP
                            for ni_l in cutlass.range_constexpr(ATOMS_N_PER_WARP):
                                ni_g = cwid_n * cutlass.Int32(ATOMS_N_PER_WARP) + cutlass.Int32(ni_l)
                                sfb_regs[ni_l] = sSFB_u32[sf_st_curr, k_block, ni_g, sfb_logical, 0]

                            for mi_l in cutlass.range_constexpr(ATOMS_M_PER_WARP):
                                mi_g = cwid_m * cutlass.Int32(ATOMS_M_PER_WARP) + cutlass.Int32(mi_l)
                                sfa_reg = sSFA_u32[sf_st_curr, k_block, mi_g, sfa_logical, 0]
                                for ni_l in cutlass.range_constexpr(ATOMS_N_PER_WARP):
                                    acc_vec = cute.make_rmem_tensor((4,), ACC_DTYPE)
                                    for i in cutlass.range_constexpr(4):
                                        acc_vec[i] = acc[mi_l][ni_l][i]
                                    acc_vec.store(_mma_inline(
                                        tCrA_u32[mi_l].load(), tCrB_u32[ni_l].load(),
                                        acc_vec.load(),
                                        cutlass.Int32(sfa_reg), cutlass.Int32(sfb_regs[ni_l])))
                                    for i in cutlass.range_constexpr(4):
                                        acc[mi_l][ni_l][i] = acc_vec[i]

                        p.consumer_release(cons_state)
                        cons_state.advance()

                    cute.arch.cp_async_wait_group(0)
                    cute.arch.barrier(barrier_id=4, number_of_threads=CONSUMER_WARPS * 32)

                    av = alpha[0]
                    g = lane // cutlass.Int32(4)
                    c = lane % cutlass.Int32(4)
                    two_c = c * cutlass.Int32(2)
                    g8 = g + cutlass.Int32(8)
                    two_c1 = two_c + cutlass.Int32(1)
                    for mi_l in cutlass.range_constexpr(ATOMS_M_PER_WARP):
                        for ni_l in cutlass.range_constexpr(ATOMS_N_PER_WARP):
                            gO_subs[mi_l][ni_l][g, two_c]   = acc[mi_l][ni_l][0] * av
                            gO_subs[mi_l][ni_l][g, two_c1]  = acc[mi_l][ni_l][1] * av
                            gO_subs[mi_l][ni_l][g8, two_c]  = acc[mi_l][ni_l][2] * av
                            gO_subs[mi_l][ni_l][g8, two_c1] = acc[mi_l][ni_l][3] * av

    return gemm_v43_kernel


def _build_launcher_v43(M, N, K):
    kern = _build_v43(M, N, K)

    @cute.jit
    def launcher(a_q, b_q, a_sf, b_sf, alpha, out, stream):
        K_HALF = K // 2
        SF_TILES_K = K // ATOM_K

        gA = cute.make_tensor(a_q.iterator,
                              cute.make_layout((M, K_HALF), stride=(K_HALF, 1)))
        gB = cute.make_tensor(b_q.iterator,
                              cute.make_layout((N, K_HALF), stride=(K_HALF, 1)))
        # Block-grouped compressed SF gmem (per CTA M-block × SF_TILES_K × ATOMS × atom_dim × 4):
        # SFA: (M//BLOCK_M, SF_TILES_K, ATOMS_M, SFA_PER_ATOM, 4) flattened.
        # Per (mbo, kt) chunk = ATOMS_M × SFA_PER_ATOM × 4 = 512 bytes contiguous → fits 1 cp.async.
        gSFA = cute.make_tensor(a_sf.iterator,
                                cute.make_layout(((M // BLOCK_M) * SF_TILES_K * ATOMS_M * SFA_PER_ATOM, 4),
                                                 stride=(4, 1)))
        gSFB = cute.make_tensor(b_sf.iterator,
                                cute.make_layout(((N // BLOCK_N) * SF_TILES_K * ATOMS_N * SFB_PER_ATOM, 4),
                                                 stride=(4, 1)))

        sA_one = cute.make_layout((BLOCK_M, ATOM_K // 2), stride=(ATOM_K // 2, 1))
        sB_one = cute.make_layout((BLOCK_N, ATOM_K // 2), stride=(ATOM_K // 2, 1))
        tma_atom_a, tma_tensor_a = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), gA, sA_one, (BLOCK_M, ATOM_K // 2),
        )
        tma_atom_b, tma_tensor_b = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), gB, sB_one, (BLOCK_N, ATOM_K // 2),
        )

        # cp.async atoms for compressed SF: ALL atoms × atom_dim × 4 = 512 bytes per call.
        # 128-bit per thread (cp.async constraint), 32 threads × 16 bytes = 512 bytes total.
        cp_atom_sf = cute.make_copy_atom(
            CopyG2SOp(cache_mode=LoadCacheMode.GLOBAL),
            cutlass.Uint8, num_bits_per_copy=128,
        )
        thr_layout_sf = cute.make_layout((32, 1), stride=(1, 1))
        val_layout_sf = cute.make_layout((4, 4), stride=(4, 1))
        tiled_copy_sfa = cute.make_tiled_copy_tv(cp_atom_sf, thr_layout_sf, val_layout_sf)
        tiled_copy_sfb = cute.make_tiled_copy_tv(cp_atom_sf, thr_layout_sf, val_layout_sf)

        ldm_a_op = LdMatrix8x8x16bOp(num_matrices=4)
        ldm_b_op = LdMatrix8x8x16bOp(num_matrices=2)
        ldm_atom_A = cute.make_copy_atom(ldm_a_op, AB_DTYPE)
        ldm_atom_B = cute.make_copy_atom(ldm_b_op, AB_DTYPE)
        op = MmaMXF4NVF4Op(AB_DTYPE, ACC_DTYPE, SF_DTYPE)
        tiled_mma = cute.make_tiled_mma(op)
        smem_tiled_copy_A = cute.make_tiled_copy_A(ldm_atom_A, tiled_mma)
        smem_tiled_copy_B = cute.make_tiled_copy_B(ldm_atom_B, tiled_mma)

        smem_bytes = (
            2 * STAGES * BLOCK_M * (ATOM_K // 2)
            + 2 * STAGES * BLOCK_N * (ATOM_K // 2)
            + STAGES * K_BLOCK_MAX * ATOMS_M * SFA_PER_ATOM * 4
            + STAGES * K_BLOCK_MAX * ATOMS_N * SFB_PER_ATOM * 4
            + STAGES * 2 * 8
            + 1024
        )
        kern(
            gSFA, gSFB, alpha, out,
            tma_atom_a, tma_tensor_a,
            tma_atom_b, tma_tensor_b,
            tiled_copy_sfa, tiled_copy_sfb,
            smem_tiled_copy_A, smem_tiled_copy_B,
        ).launch(
            grid=(NUM_CTAS, 1, 1),
            block=(THREADS, 1, 1),
            smem=smem_bytes,
            stream=stream,
        )
    return launcher


_COMPILED = {}

def launch_gemm_v43(a_q, b_q, a_sf, b_sf, alpha, out, M, N, K, stream):
    key = (int(M), int(N), int(K))
    if key not in _COMPILED:
        launcher = _build_launcher_v43(M, N, K)
        _COMPILED[key] = cute.compile(launcher, a_q, b_q, a_sf, b_sf, alpha, out, stream)
    _COMPILED[key](a_q, b_q, a_sf, b_sf, alpha, out, stream)
