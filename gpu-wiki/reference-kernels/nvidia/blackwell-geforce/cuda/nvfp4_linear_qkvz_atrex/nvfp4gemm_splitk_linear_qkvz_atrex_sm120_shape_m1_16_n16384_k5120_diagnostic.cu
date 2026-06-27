// gpu-wiki archive note:
// TUNED FOR RTX PRO 5000 / SM120 diagnostic linear_qkvz shapes
// M=1..16,N=16384,K=5120. This source is the omoExplore task41 ATREX Split-K
// CUDA snapshot. It contains env-gated A-staging and CTA-3D paths, but the
// project conclusion was blocked: default build 0/16 wins, best env-gated
// A-staging 7/16 only, and CTE 0/16. Use as a structural-ceiling reference.
//
// Split-K NVFP4 decode GEMM/GEMV kernel for SM120a (Blackwell).
// Splits K dimension into S segments for better SM occupancy on small-N shapes.
// Two-phase: splitk kernel produces f32 partials -> reduce kernel sums + writes bf16.
//
// Primary target: C2 shape (N=5120, K=8704) where base v3 only gets 1.45 CTA/SM.
// With S=4: 640 CTAs -> 5.8 CTA/SM, CG GPU kernel +18.6% vs cutlass.
//
// Ported from omoExplore proj_001 task_20 gemm_v3_splitk.
// task_33 extends the original M=1 decode path to power-of-two M.
// task_38 adds a CTA-3D TMA fast path: fused intra-CTA split-K with a
// CTA-level 3D TMA B load, preserving the public pybind API. The latest
// M=1..16 CTA-3D TMA implementation is kept in this file before the two-stage
// split-K fallback implementation.

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

#include "cute/arch/cluster_sm90.hpp"
#include "cute/arch/copy_sm90_desc.hpp"
#include "cute/arch/copy_sm90_tma.hpp"

#include "nvfp4gemm.h"

#define K_CHUNK_SK 64

static constexpr int CTA3D_TMA_S = 8;
static constexpr int CTA3D_TMA_TILE_N = 8;
static constexpr int C2_TMA_M = 1;
static constexpr int C2_TMA_N = 5120;
static constexpr int C2_TMA_K = 17408;

__device__ __forceinline__ long sf_128x4_offset(int row, int k_chunk, int K_chunks)
{
    const int row_block = row >> 7;
    const int row_local = row & 127;
    return (long)(row_block * K_chunks + k_chunk) * 512
        + (row_local & 31) * 16
        + (row_local >> 5) * 4;
}

#ifndef USE_TMA_B
#define USE_TMA_B 1
#endif

#define K_CHUNK 64
#ifndef B_SWIZZLE_MODE
#define B_SWIZZLE_MODE 0
#endif
#ifndef ZERO_A3_ONLY
#define ZERO_A3_ONLY 1
#endif
#ifndef BCAST_SFA
#define BCAST_SFA 0
#endif
#ifndef B_REG_PIPE
#define B_REG_PIPE 0
#endif
#ifndef B_L2_PREFETCH
#define B_L2_PREFETCH 0
#endif
#ifndef B_L2_PREFETCH_T0_ONLY
#define B_L2_PREFETCH_T0_ONLY 0
#endif
#ifndef B_L2_PREFETCH_KCHUNKS
#define B_L2_PREFETCH_KCHUNKS 4
#endif
#ifndef SKIP_POST_MMA_SYNC
#define SKIP_POST_MMA_SYNC 0
#endif
#ifndef B_WORDROT_FACTOR
#define B_WORDROT_FACTOR 2
#endif
#ifndef PERSISTENT_GRID_CTAS
#define PERSISTENT_GRID_CTAS 0
#endif
#ifndef FIXED_C2_S8_TN8
#define FIXED_C2_S8_TN8 0
#endif
#ifndef RECUR_ADDR_C2
#define RECUR_ADDR_C2 0
#endif
#ifndef TMA_B_PIPE_STAGES
#define TMA_B_PIPE_STAGES 1
#endif
#ifndef TMA_B_CTA_3D
#define TMA_B_CTA_3D 1
#endif
#ifndef MTILE_N_GROUPS
#define MTILE_N_GROUPS 3
#endif
#ifndef MTILE_TMA_STAGES
#define MTILE_TMA_STAGES 3
#endif
#ifndef MTILE_CONSUMER_WAIT_BARRIER
#define MTILE_CONSUMER_WAIT_BARRIER 1
#endif
#ifndef MTILE_POST_K_SYNC
#define MTILE_POST_K_SYNC 1
#endif
#ifndef MTILE_NWARP_N_GROUPS
#define MTILE_NWARP_N_GROUPS 2
#endif
#ifndef MTILE_NWARP_MIN_M
#define MTILE_NWARP_MIN_M 17
#endif
#ifndef MTILE_HYBRID_SPLITS
#define MTILE_HYBRID_SPLITS 2
#endif
#ifndef MTILE_HYBRID_MIN_M
#define MTILE_HYBRID_MIN_M 1
#endif
#ifndef MTILE_HYBRID_MIN_N
#define MTILE_HYBRID_MIN_N 20000
#endif
#ifndef MTILE_HYBRID_PAIR_REDUCE
#define MTILE_HYBRID_PAIR_REDUCE 0
#endif
#ifndef MTILE_HYBRID_WARPS_N
#define MTILE_HYBRID_WARPS_N 0
#endif
#ifndef MTILE_SFA_PAIR_BCAST
#define MTILE_SFA_PAIR_BCAST 0
#endif
#ifndef MTILE_HYBRID_TMA_STAGES
#define MTILE_HYBRID_TMA_STAGES 2
#endif
// task_41 m1_specialize_probe_20260601: route M==1 to the mtile16 CTA-3D TMA
// kernel only when N is small enough to benefit. Cross-shape M=1 probe (GPU 4,
// single-variable vs byte-identical base) showed mtile16 WINS for N<=16384
// (linear_qkvz -9.45%, mlp_down 5120x17408 -4.29%, o_proj 5120x6144 -7.96%,
// 14336x5120 -3.03%) but REGRESSES for large N (mlp_gate_up/C1 N=34816 +43%,
// lm_head N=152064 +12%): base's fine-grained grid (N/8) already saturates the
// GPU there, so mtile16's A-staging + wide-tile setup becomes net overhead.
// M>=2 always uses mtile16 (this gate only affects M==1). Empirical crossover
// lies in (16384, 34816]; 16384 is the conservative max N proven to win.
#ifndef CTA3D_M1_MTILE_MAX_N
#define CTA3D_M1_MTILE_MAX_N 16384
#endif
// task_41 §4.2 cross-tile pipelining (design_cte_20260529.md).
// Default 0: byte-equivalent off, must pass §6.4.1.a resource-usage diff.
// Set 1 at build time to enable the persistent grid-stride `_cte` kernels.
#ifndef MTILE_CROSS_TILE_PIPE
#define MTILE_CROSS_TILE_PIPE 0
#endif
// Compile-time guard for the design_cte_20260529.md §3 smem-budget table.
// SPLIT_K is runtime (`S`); the dispatcher must runtime-assert S==8 before
// launching `_cte` kernels; the budget assumption is documented here.
// Only enforced when MTILE_CROSS_TILE_PIPE is enabled — default builds
// (NG=3/ST=3) continue to compile because the `_cte` budget does not apply.
#if MTILE_CROSS_TILE_PIPE
static_assert(MTILE_N_GROUPS == 4 && MTILE_TMA_STAGES == 2 && K_CHUNK == 64,
              "_cte smem budget table assumes NG=4 ST=2 K_CHUNK=64 (and "
              "runtime S=8); re-derive §3 of design_cte_20260529.md before "
              "changing these.");
#endif
#if USE_TMA_B && (B_SWIZZLE_MODE != 0)
#error "USE_TMA_B writes a row-major shared B tile; build it with B_SWIZZLE_MODE=0."
#endif
#if USE_TMA_B && B_REG_PIPE
#error "USE_TMA_B replaces the B global-load path and cannot be combined with B_REG_PIPE."
#endif
#if !USE_TMA_B && (TMA_B_PIPE_STAGES != 1)
#error "TMA_B_PIPE_STAGES is only meaningful with USE_TMA_B."
#endif
#if USE_TMA_B && (TMA_B_PIPE_STAGES != 1) && (TMA_B_PIPE_STAGES != 2)
#error "Only TMA_B_PIPE_STAGES=1 or 2 is supported."
#endif
#if TMA_B_CTA_3D && !USE_TMA_B
#error "TMA_B_CTA_3D requires USE_TMA_B."
#endif
#if TMA_B_CTA_3D && (TMA_B_PIPE_STAGES != 1)
#error "TMA_B_CTA_3D currently supports only single-stage TMA."
#endif
#if TMA_B_CTA_3D && (MTILE_TMA_STAGES < 2)
#error "The small-M CTA-3D TMA path expects at least two TMA stages."
#endif

#if FIXED_C2_S8_TN8
static constexpr int C2_FIXED_M = 1;
static constexpr int C2_FIXED_N = 5120;
static constexpr int C2_FIXED_K = 17408;
static constexpr int C2_FIXED_K_HALF = C2_FIXED_K / 2;
static constexpr int C2_FIXED_K_CHUNKS = C2_FIXED_K / 64;
static constexpr int C2_FIXED_S = 8;
static constexpr int C2_FIXED_TILE_N = 8;
static constexpr int C2_FIXED_K_SPLIT = C2_FIXED_K / C2_FIXED_S;
static constexpr int C2_FIXED_N_TILES = C2_FIXED_N / C2_FIXED_TILE_N;
#endif

__device__ __forceinline__ long cta3d_sf_128x4_offset(int row, int k_chunk, int K_chunks)
{
    const int row_block = row >> 7;
    const int row_local = row & 127;
    return (long)(row_block * K_chunks + k_chunk) * 512
        + (row_local & 31) * 16
        + (row_local >> 5) * 4;
}

__device__ __forceinline__ uint32_t load_mtile_sfa(
    const uint8_t* __restrict__ SF_A,
    int M, int kc, int K_chunks, int lane_id)
{
    const int t0 = lane_id & 3;
    const int t1 = lane_id >> 2;
    const int sfa_row = (M > 8)
        ? ((lane_id & 1) * 8 + t1)
        : (((lane_id & 1) == 0) ? t1 : 16 + t1);
#if MTILE_SFA_PAIR_BCAST
    uint32_t sfa = 0x38383838u;
    if ((t0 & 2) == 0 && sfa_row < M) {
        sfa = *reinterpret_cast<const uint32_t*>(
            SF_A + cta3d_sf_128x4_offset(sfa_row, kc, K_chunks));
    }
    return __shfl_sync(0xffffffffu, sfa, lane_id & ~2);
#else
    return (sfa_row < M)
        ? *reinterpret_cast<const uint32_t*>(
            SF_A + cta3d_sf_128x4_offset(sfa_row, kc, K_chunks))
        : 0x38383838u;
#endif
}

__device__ __forceinline__ uint32_t load_m16_sfa(
    const uint8_t* __restrict__ SF_A,
    int kc, int K_chunks, int lane_id)
{
    const int t0 = lane_id & 3;
    const int t1 = lane_id >> 2;
    const int sfa_row = (lane_id & 1) * 8 + t1;
#if MTILE_SFA_PAIR_BCAST
    uint32_t sfa = 0;
    if ((t0 & 2) == 0) {
        sfa = *reinterpret_cast<const uint32_t*>(
            SF_A + cta3d_sf_128x4_offset(sfa_row, kc, K_chunks));
    }
    return __shfl_sync(0xffffffffu, sfa, lane_id & ~2);
#else
    return *reinterpret_cast<const uint32_t*>(
        SF_A + cta3d_sf_128x4_offset(sfa_row, kc, K_chunks));
#endif
}

__device__ __forceinline__ void cta3d_tma_shared_fence()
{
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
#endif
}

// Async global->shared copy of a 16-byte chunk (cp.async.cg). Bypasses L1 on
// the way in, lands directly in shared memory, and is tracked by cp.async
// commit groups so it does not occupy the long-scoreboard the way a plain
// global load + smem store would.
__device__ __forceinline__ void cp_async_cg_16(void* smem_dst, const void* gmem_src)
{
    unsigned smem_int =
        static_cast<unsigned>(__cvta_generic_to_shared(smem_dst));
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        :: "r"(smem_int), "l"(gmem_src));
}

__device__ __forceinline__ void cp_async_commit()
{
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

template <int N>
__device__ __forceinline__ void cp_async_wait_group()
{
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory");
}

__device__ __forceinline__ int cta3d_b_smem_offset(int row, int logical_byte)
{
#if B_SWIZZLE_MODE == 1
    // 16-byte chunk XOR. Keeps vectorized shared stores and uses no extra smem.
    const int chunk = logical_byte >> 4;
    const int intra = logical_byte & 15;
    const int phys_chunk = chunk ^ (row & 3);
    return row * 64 + phys_chunk * 16 + intra;
#elif B_SWIZZLE_MODE == 2 || B_SWIZZLE_MODE == 4 || B_SWIZZLE_MODE == 5
    // 4-byte word rotation. More aggressive bank spreading, but scalarizes
    // shared stores in mode 2. Mode 4 keeps the same read layout but uses
    // mixed 16B/8B stores to reduce store instruction count. Mode 5 uses a
    // uniform two-8B store sequence to avoid warp-divergent store branches.
    const int word = logical_byte >> 2;
    const int intra = logical_byte & 3;
    const int phys_word = (word + (row & 7) * B_WORDROT_FACTOR) & 15;
    return row * 64 + phys_word * 4 + intra;
#elif B_SWIZZLE_MODE == 3
    // 16-byte chunk rotation using row pairs. This keeps vectorized shared
    // stores like row-major layout, while rows 0/2/4/6 and 1/3/5/7 map a
    // logical chunk to all four physical chunks for conflict-free shared loads.
    const int chunk = logical_byte >> 4;
    const int intra = logical_byte & 15;
    const int phys_chunk = (chunk + (row >> 1)) & 3;
    return row * 64 + phys_chunk * 16 + intra;
#else
    return row * 64 + logical_byte;
#endif
}

__device__ __forceinline__ void prefetch_l2_bswizzle(const void* ptr)
{
    asm volatile("prefetch.global.L2 [%0];\n" :: "l"(ptr));
}

__device__ __forceinline__ void mma_mxf4nvf4_k64_task38(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    uint32_t sfa, uint32_t sfb)
{
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X."
        "m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0, %1, %2, %3},"
        "{%4, %5, %6, %7},"
        "{%8, %9},"
        "{%0, %1, %2, %3},"
        "{%10},"
        "{%11, %12},"
        "{%13},"
        "{%14, %15};\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "r"(sfa), "h"((uint16_t)0), "h"((uint16_t)0),
          "r"(sfb), "h"((uint16_t)0), "h"((uint16_t)0));
}

template <int TILE_N_T>
__global__ void decode_gemv_nvfp4_splitk_kernel_fused(
    int M, int N, int K, int S, int K_split,
    const uint8_t* __restrict__ A_packed,
    const uint8_t* __restrict__ B_packed,
    const uint8_t* __restrict__ SF_A,
    const uint8_t* __restrict__ SF_B,
    const float* __restrict__ alpha_ptr,
    __nv_bfloat16* __restrict__ D
#if USE_TMA_B
    , CUTE_GRID_CONSTANT CUtensorMap const b_tma_desc
#endif
    )
{
#if FIXED_C2_S8_TN8
    constexpr int num_warps = C2_FIXED_S;
    const int m = 0;
#else
    const int groups_per_tile = TILE_N_T / 8;
    const int num_warps = groups_per_tile * S;
    const int m = blockIdx.y;
    if (m >= M) return;
#endif

    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    if (warp_id >= num_warps) return;

#if FIXED_C2_S8_TN8
    const int split_id = warp_id;
    constexpr int n_group = 0;
#else
    const int split_id = warp_id % S;
    const int n_group = warp_id / S;
#endif
    const int t0 = lane_id & 3;
    const int t1 = lane_id >> 2;

#if FIXED_C2_S8_TN8
    const int k_start = split_id * C2_FIXED_K_SPLIT;
    constexpr int K_chunks = C2_FIXED_K_CHUNKS;
#else
    const int k_start = split_id * K_split;
    const int K_chunks = K / 64;
    const int total_n_tiles = (N + TILE_N_T - 1) / TILE_N_T;
#endif

    extern __shared__ uint8_t smem[];
#if USE_TMA_B
    const int smem_barrier_off = 0;
#if TMA_B_CTA_3D
    const int smem_barrier_bytes = ((int)sizeof(uint64_t) + 15) & ~15;
#else
    const int smem_barrier_bytes =
        (num_warps * TMA_B_PIPE_STAGES * (int)sizeof(uint64_t) + 15) & ~15;
#endif
    uint64_t* tma_B_barriers =
        reinterpret_cast<uint64_t*>(smem + smem_barrier_off);
    const int smem_B_off = (smem_barrier_off + smem_barrier_bytes + 127) & ~127;
#else
    const int smem_B_off = 0;
#endif
    const int smem_B_bytes = num_warps * TMA_B_PIPE_STAGES * 8 * 64;
    const int partial_off = (smem_B_off + smem_B_bytes + 15) & ~15;
    uint8_t* smem_B = smem + smem_B_off;
    float* partials = reinterpret_cast<float*>(smem + partial_off);

#if USE_TMA_B
#if TMA_B_CTA_3D
    if (threadIdx.x == 0) {
        cute::initialize_barrier(tma_B_barriers[0], 1);
    }
#else
    if (threadIdx.x < num_warps * TMA_B_PIPE_STAGES) {
        cute::initialize_barrier(tma_B_barriers[threadIdx.x], 1);
    }
#endif
    __syncthreads();
    if (warp_id == 0 && cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&b_tma_desc);
    }
    __syncthreads();
#if TMA_B_CTA_3D
    int tma_B_phase = 0;
#else
#if TMA_B_PIPE_STAGES == 2
    int tma_B_phase[2] = {0, 0};
#else
    int tma_B_phase = 0;
#endif
#endif
#endif
#if FIXED_C2_S8_TN8
    const uint8_t* A_start = A_packed;
#else
    const uint8_t* A_start = A_packed + (long)m * (K / 2);
#endif

#if FIXED_C2_S8_TN8
    const int cta_n = blockIdx.x * C2_FIXED_TILE_N;
#else
    for (int tile_idx = blockIdx.x; tile_idx < total_n_tiles; tile_idx += gridDim.x) {
    const int cta_n = tile_idx * TILE_N_T;
#endif
    const int abs_n_sf = cta_n + n_group * 8 + t1;
    float d0 = 0.f, d1 = 0.f;
    float d2 = 0.f, d3 = 0.f;
    uint8_t* warp_B = smem_B + warp_id * 8 * 64;
    const int abs_n = cta_n + n_group * 8 + t1;
#if USE_TMA_B && !B_L2_PREFETCH && !RECUR_ADDR_C2
    (void)abs_n;
#endif

#if B_REG_PIPE
    uint4 packed_cur = make_uint4(0, 0, 0, 0);
    const bool valid_n = abs_n < N;
    if (valid_n) {
        const uint8_t* src = B_packed
            + (long)abs_n * (K / 2)
            + k_start / 2
            + t0 * 16;
        packed_cur = *reinterpret_cast<const uint4*>(src);
    }
#endif

#if RECUR_ADDR_C2 && !B_REG_PIPE && !B_L2_PREFETCH
    const int sf_kc_start = k_start >> 6;
    const uint8_t* a_iter = A_start + k_start / 2;
    const uint8_t* b_iter = B_packed
        + (long)abs_n * (K / 2)
        + k_start / 2
        + t0 * 16;
    const uint8_t* sfa_iter = SF_A + cta3d_sf_128x4_offset(m, sf_kc_start, K_chunks);
    const uint8_t* sfb_iter = SF_B + cta3d_sf_128x4_offset(abs_n_sf, sf_kc_start, K_chunks);
#endif

#if USE_TMA_B && TMA_B_CTA_3D
#elif USE_TMA_B && TMA_B_PIPE_STAGES == 2
    if (cute::elect_one_sync()) {
        cute::set_barrier_transaction_bytes(tma_B_barriers[warp_id * 2], 8 * 64);
        cute::SM90_TMA_LOAD_2D::copy(
            &b_tma_desc,
            &tma_B_barriers[warp_id * 2],
            static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL),
            warp_B,
            k_start / 2,
            cta_n + n_group * 8);
    }
#endif

#if FIXED_C2_S8_TN8
    for (int k_local = 0; k_local < C2_FIXED_K_SPLIT; k_local += K_CHUNK * 2) {
#else
    for (int k_local = 0; k_local < K_split; k_local += K_CHUNK * 2) {
#endif
        const int k_abs = k_start + k_local;
        uint8_t* warp_B_read = warp_B;

#if B_L2_PREFETCH
        const int pf_k_local = k_local + K_CHUNK * B_L2_PREFETCH_KCHUNKS;
        if (pf_k_local < K_split && abs_n < N
            && (!B_L2_PREFETCH_T0_ONLY || t0 == 0)) {
            const uint8_t* pf_src = B_packed
                + (long)abs_n * (K / 2)
                + (k_start + pf_k_local) / 2
                + t0 * 16;
            prefetch_l2_bswizzle(pf_src);
        }
#endif

#if USE_TMA_B
#if TMA_B_CTA_3D
        if (threadIdx.x == 0) {
            cute::set_barrier_transaction_bytes(tma_B_barriers[0], S * 8 * 64);
            cute::SM90_TMA_LOAD_3D::copy(
                &b_tma_desc,
                &tma_B_barriers[0],
                static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL),
                smem_B,
                k_local / 2,
                cta_n,
                0);
            cute::wait_barrier(tma_B_barriers[0], tma_B_phase);
        }
        __syncthreads();
        tma_B_phase ^= 1;
#elif TMA_B_PIPE_STAGES == 2
        const int tma_stage = (k_local / (K_CHUNK * 2)) & 1;
        uint64_t* tma_barrier = tma_B_barriers + warp_id * 2 + tma_stage;
        if (cute::elect_one_sync()) {
            cute::wait_barrier(*tma_barrier, tma_B_phase[tma_stage]);
        }
        __syncwarp();
        tma_B_phase[tma_stage] ^= 1;
        warp_B_read = smem_B + (tma_stage * num_warps + warp_id) * 8 * 64;

        const int next_k_local = k_local + K_CHUNK * 2;
        if (next_k_local < K_split) {
            const int next_stage = tma_stage ^ 1;
            uint8_t* next_warp_B =
                smem_B + (next_stage * num_warps + warp_id) * 8 * 64;
            uint64_t* next_barrier = tma_B_barriers + warp_id * 2 + next_stage;
            if (cute::elect_one_sync()) {
                cute::set_barrier_transaction_bytes(*next_barrier, 8 * 64);
                cute::SM90_TMA_LOAD_2D::copy(
                    &b_tma_desc,
                    next_barrier,
                    static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL),
                    next_warp_B,
                    (k_start + next_k_local) / 2,
                    cta_n + n_group * 8);
            }
        }
#else
        if (cute::elect_one_sync()) {
            cute::set_barrier_transaction_bytes(tma_B_barriers[warp_id], 8 * 64);
            cute::SM90_TMA_LOAD_2D::copy(
                &b_tma_desc,
                &tma_B_barriers[warp_id],
                static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL),
                warp_B,
                k_abs / 2,
                cta_n + n_group * 8);
            cute::wait_barrier(tma_B_barriers[warp_id], tma_B_phase);
        }
        __syncwarp();
        tma_B_phase ^= 1;
#endif
#else
#if B_REG_PIPE
        uint4 packed = packed_cur;
#else
#if FIXED_C2_S8_TN8
        {
            const uint8_t* src = B_packed
                + (long)abs_n * C2_FIXED_K_HALF
                + k_abs / 2
                + t0 * 16;
            uint4 packed = *reinterpret_cast<const uint4*>(src);
#else
        if (abs_n < N) {
#if RECUR_ADDR_C2 && !B_REG_PIPE && !B_L2_PREFETCH
            const uint8_t* src = b_iter;
#else
            const uint8_t* src = B_packed
                + (long)abs_n * (K / 2)
                + k_abs / 2
                + t0 * 16;
#endif
            uint4 packed = *reinterpret_cast<const uint4*>(src);
#endif
#endif
#if B_SWIZZLE_MODE == 2
            *reinterpret_cast<uint32_t*>(warp_B + cta3d_b_smem_offset(t1, t0 * 16 + 0))
                = packed.x;
            *reinterpret_cast<uint32_t*>(warp_B + cta3d_b_smem_offset(t1, t0 * 16 + 4))
                = packed.y;
            *reinterpret_cast<uint32_t*>(warp_B + cta3d_b_smem_offset(t1, t0 * 16 + 8))
                = packed.z;
            *reinterpret_cast<uint32_t*>(warp_B + cta3d_b_smem_offset(t1, t0 * 16 + 12))
                = packed.w;
#elif B_SWIZZLE_MODE == 4
            if ((t1 & 1) == 0) {
                *reinterpret_cast<uint4*>(warp_B + cta3d_b_smem_offset(t1, t0 * 16))
                    = packed;
            } else {
                const uint2 lo = make_uint2(packed.x, packed.y);
                const uint2 hi = make_uint2(packed.z, packed.w);
                *reinterpret_cast<uint2*>(warp_B + cta3d_b_smem_offset(t1, t0 * 16))
                    = lo;
                *reinterpret_cast<uint2*>(warp_B + cta3d_b_smem_offset(t1, t0 * 16 + 8))
                    = hi;
            }
#elif B_SWIZZLE_MODE == 5
            const uint2 lo = make_uint2(packed.x, packed.y);
            const uint2 hi = make_uint2(packed.z, packed.w);
            *reinterpret_cast<uint2*>(warp_B + cta3d_b_smem_offset(t1, t0 * 16))
                = lo;
            *reinterpret_cast<uint2*>(warp_B + cta3d_b_smem_offset(t1, t0 * 16 + 8))
                = hi;
#else
            *reinterpret_cast<uint4*>(warp_B + cta3d_b_smem_offset(t1, t0 * 16))
                = packed;
#endif
#if !B_REG_PIPE
        }
#endif
        __syncwarp();
#endif

#if B_REG_PIPE
        if (valid_n && k_local + K_CHUNK * 2 < K_split) {
            const uint8_t* src_next = B_packed
                + (long)abs_n * (K / 2)
                + (k_abs + K_CHUNK * 2) / 2
                + t0 * 16;
            packed_cur = *reinterpret_cast<const uint4*>(src_next);
        }
#endif

        // MMA 0
        {
            const int kc = k_abs >> 6;
#if RECUR_ADDR_C2 && !B_REG_PIPE && !B_L2_PREFETCH
            const uint8_t* ap = a_iter;
#else
            const uint8_t* ap = A_start + k_start / 2 + k_local / 2;
#endif
            uint32_t a0 = *reinterpret_cast<const uint32_t*>(ap + t0 * 4);
            uint32_t a1 = a0;
            uint32_t a2 = *reinterpret_cast<const uint32_t*>(ap + t0 * 4 + 16);
#if ZERO_A3_ONLY
            uint32_t a3 = 0;
#else
            uint32_t a3 = a2;
#endif

            uint32_t b0 = *reinterpret_cast<const uint32_t*>(
                warp_B_read + cta3d_b_smem_offset(t1, t0 * 4));
            uint32_t b1 = *reinterpret_cast<const uint32_t*>(
                warp_B_read + cta3d_b_smem_offset(t1, t0 * 4 + 16));

#if BCAST_SFA
            uint32_t sfa = 0;
            if (lane_id == 0) {
                sfa = *reinterpret_cast<const uint32_t*>(
                    SF_A + cta3d_sf_128x4_offset(m, kc, K_chunks));
            }
            sfa = __shfl_sync(0xffffffffu, sfa, 0);
#else
#if RECUR_ADDR_C2 && !B_REG_PIPE && !B_L2_PREFETCH
            uint32_t sfa = *reinterpret_cast<const uint32_t*>(sfa_iter);
#else
            uint32_t sfa = *reinterpret_cast<const uint32_t*>(
                SF_A + cta3d_sf_128x4_offset(m, kc, K_chunks));
#endif
#endif
#if FIXED_C2_S8_TN8
            uint32_t sfb = *reinterpret_cast<const uint32_t*>(
                SF_B + cta3d_sf_128x4_offset(abs_n_sf, kc, K_chunks));
#elif RECUR_ADDR_C2 && !B_REG_PIPE && !B_L2_PREFETCH
            uint32_t sfb = *reinterpret_cast<const uint32_t*>(sfb_iter);
#else
            uint32_t sfb = (abs_n_sf < N)
                ? *reinterpret_cast<const uint32_t*>(
                    SF_B + cta3d_sf_128x4_offset(abs_n_sf, kc, K_chunks))
                : 0x38383838u;
#endif

            asm volatile(
                "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X."
                "m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                "{%0, %1, %2, %3},"
                "{%4, %5, %6, %7},"
                "{%8, %9},"
                "{%0, %1, %2, %3},"
                "{%10},"
                "{%11, %12},"
                "{%13},"
                "{%14, %15};\n"
                : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                  "r"(b0), "r"(b1),
                  "r"(sfa), "h"((uint16_t)0), "h"((uint16_t)0),
                  "r"(sfb), "h"((uint16_t)0), "h"((uint16_t)0)
            );
        }

        // MMA 1
        {
            const int kc = (k_abs >> 6) + 1;
#if RECUR_ADDR_C2 && !B_REG_PIPE && !B_L2_PREFETCH
            const uint8_t* ap = a_iter + 32;
#else
            const uint8_t* ap = A_start + k_start / 2 + k_local / 2 + 32;
#endif
            uint32_t a0 = *reinterpret_cast<const uint32_t*>(ap + t0 * 4);
            uint32_t a1 = a0;
            uint32_t a2 = *reinterpret_cast<const uint32_t*>(ap + t0 * 4 + 16);
#if ZERO_A3_ONLY
            uint32_t a3 = 0;
#else
            uint32_t a3 = a2;
#endif

            uint32_t b0 = *reinterpret_cast<const uint32_t*>(
                warp_B_read + cta3d_b_smem_offset(t1, t0 * 4 + 32));
            uint32_t b1 = *reinterpret_cast<const uint32_t*>(
                warp_B_read + cta3d_b_smem_offset(t1, t0 * 4 + 48));

#if BCAST_SFA
            uint32_t sfa = 0;
            if (lane_id == 0) {
                sfa = *reinterpret_cast<const uint32_t*>(
                    SF_A + cta3d_sf_128x4_offset(m, kc, K_chunks));
            }
            sfa = __shfl_sync(0xffffffffu, sfa, 0);
#else
#if RECUR_ADDR_C2 && !B_REG_PIPE && !B_L2_PREFETCH
            uint32_t sfa = *reinterpret_cast<const uint32_t*>(sfa_iter + 512);
#else
            uint32_t sfa = *reinterpret_cast<const uint32_t*>(
                SF_A + cta3d_sf_128x4_offset(m, kc, K_chunks));
#endif
#endif
#if FIXED_C2_S8_TN8
            uint32_t sfb = *reinterpret_cast<const uint32_t*>(
                SF_B + cta3d_sf_128x4_offset(abs_n_sf, kc, K_chunks));
#elif RECUR_ADDR_C2 && !B_REG_PIPE && !B_L2_PREFETCH
            uint32_t sfb = *reinterpret_cast<const uint32_t*>(sfb_iter + 512);
#else
            uint32_t sfb = (abs_n_sf < N)
                ? *reinterpret_cast<const uint32_t*>(
                    SF_B + cta3d_sf_128x4_offset(abs_n_sf, kc, K_chunks))
                : 0x38383838u;
#endif

            asm volatile(
                "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X."
                "m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                "{%0, %1, %2, %3},"
                "{%4, %5, %6, %7},"
                "{%8, %9},"
                "{%0, %1, %2, %3},"
                "{%10},"
                "{%11, %12},"
                "{%13},"
                "{%14, %15};\n"
                : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                  "r"(b0), "r"(b1),
                  "r"(sfa), "h"((uint16_t)0), "h"((uint16_t)0),
                  "r"(sfb), "h"((uint16_t)0), "h"((uint16_t)0)
            );
        }
#if !SKIP_POST_MMA_SYNC
        __syncwarp();
#endif
#if USE_TMA_B && TMA_B_CTA_3D
        // CTA-3D TMA reuses one shared B tile for all split warps. A CTA-wide
        // barrier is required before thread 0 can issue the next TMA copy into
        // the same smem region; warp-local sync is not enough here.
        __syncthreads();
#endif
#if RECUR_ADDR_C2 && !B_REG_PIPE && !B_L2_PREFETCH
        a_iter += K_CHUNK;
        b_iter += K_CHUNK;
        sfa_iter += 1024;
        sfb_iter += 1024;
#endif
    }

    if (t1 == 0) {
        const int local_n0 = n_group * 8 + t0 * 2;
        const int local_n1 = local_n0 + 1;
        float* ps = partials + split_id * TILE_N_T;
        ps[local_n0] = d0;
        ps[local_n1] = d1;
    }
    __syncthreads();

#if FIXED_C2_S8_TN8
    for (int local_n = threadIdx.x; local_n < C2_FIXED_TILE_N; local_n += blockDim.x) {
        const int abs_n_out = cta_n + local_n;
        float sum = 0.f;
        for (int s = 0; s < C2_FIXED_S; ++s) {
            sum += partials[s * C2_FIXED_TILE_N + local_n];
        }
        D[abs_n_out] = __float2bfloat16(sum * (*alpha_ptr));
    }
#else
    for (int local_n = threadIdx.x; local_n < TILE_N_T; local_n += blockDim.x) {
        const int abs_n = cta_n + local_n;
        if (abs_n < N) {
            float sum = 0.f;
            for (int s = 0; s < S; ++s) {
                sum += partials[s * TILE_N_T + local_n];
            }
            D[(long)m * N + abs_n] = __float2bfloat16(sum * (*alpha_ptr));
        }
    }
#endif
    __syncthreads();
#if !FIXED_C2_S8_TN8
    }
#endif
}

#if USE_TMA_B && TMA_B_CTA_3D
// True-Mtile CTA-3D TMA path for small batched decode.
//
// The original CTA-3D kernel keeps the M=1 decomposition and launches one
// CTA grid row per M row. That repeats the same B TMA load for M=2..16. This
// kernel instead lets the native m16n8k64 MMA produce up to 16 output rows in
// one CTA while reusing the single 3D B TMA tile across all rows.
__global__ void decode_gemv_nvfp4_splitk_kernel_fused_mtile16_tma8(
    int M, int N, int K, int S, int K_split,
    const uint8_t* __restrict__ A_packed,
    const uint8_t* __restrict__ B_packed,
    const uint8_t* __restrict__ SF_A,
    const uint8_t* __restrict__ SF_B,
    const float* __restrict__ alpha_ptr,
    __nv_bfloat16* __restrict__ D,
    CUTE_GRID_CONSTANT CUtensorMap const b_tma_desc)
{
    (void)B_packed;
    if (M <= 1 || M > 16) return;

    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    if (warp_id >= S) return;

    const int split_id = warp_id;
    const int t0 = lane_id & 3;
    const int t1 = lane_id >> 2;
    const int k_start = split_id * K_split;
    const int K_chunks = K / 64;
    const int cta_n = blockIdx.x * 8;

    extern __shared__ uint8_t smem[];
    const int smem_barrier_off = 0;
    const int smem_barrier_bytes = ((int)sizeof(uint64_t) + 15) & ~15;
    uint64_t* tma_B_barriers =
        reinterpret_cast<uint64_t*>(smem + smem_barrier_off);
    const int smem_B_off = (smem_barrier_off + smem_barrier_bytes + 127) & ~127;
    const int smem_B_bytes = S * 8 * 64;
    const int partial_off = (smem_B_off + smem_B_bytes + 15) & ~15;
    uint8_t* smem_B = smem + smem_B_off;
    float* partials = reinterpret_cast<float*>(smem + partial_off);
    uint8_t* warp_B = smem_B + split_id * 8 * 64;

    if (threadIdx.x == 0) {
        cute::initialize_barrier(tma_B_barriers[0], 1);
    }
    __syncthreads();
    if (warp_id == 0 && cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&b_tma_desc);
    }
    __syncthreads();

    int tma_B_phase = 0;
    float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
    float unused0 = 0.f, unused1 = 0.f;

    for (int k_local = 0; k_local < K_split; k_local += K_CHUNK * 2) {
        const int k_abs = k_start + k_local;
        if (threadIdx.x == 0) {
            cute::set_barrier_transaction_bytes(tma_B_barriers[0], S * 8 * 64);
            cute::SM90_TMA_LOAD_3D::copy(
                &b_tma_desc,
                &tma_B_barriers[0],
                static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL),
                smem_B,
                k_local / 2,
                cta_n,
                0);
            cute::wait_barrier(tma_B_barriers[0], tma_B_phase);
        }
#if MTILE_POST_K_SYNC
        __syncthreads();
#endif
        tma_B_phase ^= 1;

        // MMA 0, rows 0..7. Keep the upper half of the m16 operand neutral:
        // task35 proves full m16 correctness through the two-phase path, but
        // this CTA-3D/TMA variant showed row corruption when the real upper
        // A/SFA half was present in the same instruction group. Splitting the
        // M tile into two M8 halves preserves B reuse while keeping the proven
        // M<=8 operand form.
        // this CTA-3D/TMA variant showed row corruption when the real upper
        // A/SFA half was present in the same instruction group. Splitting the
        // M tile into two M8 halves preserves B reuse while keeping the proven
        // M<=8 operand form.
        {
            const int kc = k_abs >> 6;
            const int m0 = t1;
            const uint8_t* ap0 = A_packed + (long)m0 * (K / 2) + k_abs / 2;
            uint32_t a0 = (m0 < M) ? *reinterpret_cast<const uint32_t*>(ap0 + t0 * 4) : 0;
            uint32_t a1 = 0;
            uint32_t a2 = (m0 < M) ? *reinterpret_cast<const uint32_t*>(ap0 + t0 * 4 + 16) : 0;
            uint32_t a3 = 0;

            uint32_t b0 = *reinterpret_cast<const uint32_t*>(
                warp_B + cta3d_b_smem_offset(t1, t0 * 4));
            uint32_t b1 = *reinterpret_cast<const uint32_t*>(
                warp_B + cta3d_b_smem_offset(t1, t0 * 4 + 16));

            const int sfa_row = ((lane_id & 1) == 0) ? t1 : 16 + t1;
            uint32_t sfa = (sfa_row < M)
                ? *reinterpret_cast<const uint32_t*>(
                    SF_A + cta3d_sf_128x4_offset(sfa_row, kc, K_chunks))
                : 0x38383838u;
            const int abs_n_sf = cta_n + t1;
            uint32_t sfb = (abs_n_sf < N)
                ? *reinterpret_cast<const uint32_t*>(
                    SF_B + cta3d_sf_128x4_offset(abs_n_sf, kc, K_chunks))
                : 0x38383838u;

            asm volatile(
                "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X."
                "m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                "{%0, %1, %2, %3},"
                "{%4, %5, %6, %7},"
                "{%8, %9},"
                "{%0, %1, %2, %3},"
                "{%10},"
                "{%11, %12},"
                "{%13},"
                "{%14, %15};\n"
                : "+f"(d0), "+f"(d1), "+f"(unused0), "+f"(unused1)
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                  "r"(b0), "r"(b1),
                  "r"(sfa), "h"((uint16_t)0), "h"((uint16_t)0),
                  "r"(sfb), "h"((uint16_t)0), "h"((uint16_t)0)
            );
        }

        if (M > 8) {
            const int kc = k_abs >> 6;
            const int m0 = t1 + 8;
            const uint8_t* ap0 = A_packed + (long)m0 * (K / 2) + k_abs / 2;
            uint32_t a0 = (m0 < M) ? *reinterpret_cast<const uint32_t*>(ap0 + t0 * 4) : 0;
            uint32_t a1 = 0;
            uint32_t a2 = (m0 < M) ? *reinterpret_cast<const uint32_t*>(ap0 + t0 * 4 + 16) : 0;
            uint32_t a3 = 0;

            uint32_t b0 = *reinterpret_cast<const uint32_t*>(
                warp_B + cta3d_b_smem_offset(t1, t0 * 4));
            uint32_t b1 = *reinterpret_cast<const uint32_t*>(
                warp_B + cta3d_b_smem_offset(t1, t0 * 4 + 16));

            const int sfa_row = ((lane_id & 1) == 0) ? (t1 + 8) : 16 + t1;
            uint32_t sfa = (sfa_row < M)
                ? *reinterpret_cast<const uint32_t*>(
                    SF_A + cta3d_sf_128x4_offset(sfa_row, kc, K_chunks))
                : 0x38383838u;
            const int abs_n_sf = cta_n + t1;
            uint32_t sfb = (abs_n_sf < N)
                ? *reinterpret_cast<const uint32_t*>(
                    SF_B + cta3d_sf_128x4_offset(abs_n_sf, kc, K_chunks))
                : 0x38383838u;

            asm volatile(
                "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X."
                "m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                "{%0, %1, %2, %3},"
                "{%4, %5, %6, %7},"
                "{%8, %9},"
                "{%0, %1, %2, %3},"
                "{%10},"
                "{%11, %12},"
                "{%13},"
                "{%14, %15};\n"
                : "+f"(d2), "+f"(d3), "+f"(unused0), "+f"(unused1)
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                  "r"(b0), "r"(b1),
                  "r"(sfa), "h"((uint16_t)0), "h"((uint16_t)0),
                  "r"(sfb), "h"((uint16_t)0), "h"((uint16_t)0)
            );
        }

        // MMA 1, rows 0..7
        {
            const int kc = (k_abs >> 6) + 1;
            const int m0 = t1;
            const uint8_t* ap0 = A_packed + (long)m0 * (K / 2) + k_abs / 2 + 32;
            uint32_t a0 = (m0 < M) ? *reinterpret_cast<const uint32_t*>(ap0 + t0 * 4) : 0;
            uint32_t a1 = 0;
            uint32_t a2 = (m0 < M) ? *reinterpret_cast<const uint32_t*>(ap0 + t0 * 4 + 16) : 0;
            uint32_t a3 = 0;

            uint32_t b0 = *reinterpret_cast<const uint32_t*>(
                warp_B + cta3d_b_smem_offset(t1, t0 * 4 + 32));
            uint32_t b1 = *reinterpret_cast<const uint32_t*>(
                warp_B + cta3d_b_smem_offset(t1, t0 * 4 + 48));

            const int sfa_row = ((lane_id & 1) == 0) ? t1 : 16 + t1;
            uint32_t sfa = (sfa_row < M)
                ? *reinterpret_cast<const uint32_t*>(
                    SF_A + cta3d_sf_128x4_offset(sfa_row, kc, K_chunks))
                : 0x38383838u;
            const int abs_n_sf = cta_n + t1;
            uint32_t sfb = (abs_n_sf < N)
                ? *reinterpret_cast<const uint32_t*>(
                    SF_B + cta3d_sf_128x4_offset(abs_n_sf, kc, K_chunks))
                : 0x38383838u;

            asm volatile(
                "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X."
                "m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                "{%0, %1, %2, %3},"
                "{%4, %5, %6, %7},"
                "{%8, %9},"
                "{%0, %1, %2, %3},"
                "{%10},"
                "{%11, %12},"
                "{%13},"
                "{%14, %15};\n"
                : "+f"(d0), "+f"(d1), "+f"(unused0), "+f"(unused1)
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                  "r"(b0), "r"(b1),
                  "r"(sfa), "h"((uint16_t)0), "h"((uint16_t)0),
                  "r"(sfb), "h"((uint16_t)0), "h"((uint16_t)0)
            );
        }

        if (M > 8) {
            const int kc = (k_abs >> 6) + 1;
            const int m0 = t1 + 8;
            const uint8_t* ap0 = A_packed + (long)m0 * (K / 2) + k_abs / 2 + 32;
            uint32_t a0 = (m0 < M) ? *reinterpret_cast<const uint32_t*>(ap0 + t0 * 4) : 0;
            uint32_t a1 = 0;
            uint32_t a2 = (m0 < M) ? *reinterpret_cast<const uint32_t*>(ap0 + t0 * 4 + 16) : 0;
            uint32_t a3 = 0;

            uint32_t b0 = *reinterpret_cast<const uint32_t*>(
                warp_B + cta3d_b_smem_offset(t1, t0 * 4 + 32));
            uint32_t b1 = *reinterpret_cast<const uint32_t*>(
                warp_B + cta3d_b_smem_offset(t1, t0 * 4 + 48));

            const int sfa_row = ((lane_id & 1) == 0) ? (t1 + 8) : 16 + t1;
            uint32_t sfa = (sfa_row < M)
                ? *reinterpret_cast<const uint32_t*>(
                    SF_A + cta3d_sf_128x4_offset(sfa_row, kc, K_chunks))
                : 0x38383838u;
            const int abs_n_sf = cta_n + t1;
            uint32_t sfb = (abs_n_sf < N)
                ? *reinterpret_cast<const uint32_t*>(
                    SF_B + cta3d_sf_128x4_offset(abs_n_sf, kc, K_chunks))
                : 0x38383838u;

            asm volatile(
                "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X."
                "m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                "{%0, %1, %2, %3},"
                "{%4, %5, %6, %7},"
                "{%8, %9},"
                "{%0, %1, %2, %3},"
                "{%10},"
                "{%11, %12},"
                "{%13},"
                "{%14, %15};\n"
                : "+f"(d2), "+f"(d3), "+f"(unused0), "+f"(unused1)
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                  "r"(b0), "r"(b1),
                  "r"(sfa), "h"((uint16_t)0), "h"((uint16_t)0),
                  "r"(sfb), "h"((uint16_t)0), "h"((uint16_t)0)
            );
        }

#if MTILE_POST_K_SYNC
        __syncthreads();
#endif
    }

    const int local_n0 = (lane_id & 3) * 2;
    const int local_n1 = local_n0 + 1;
    const int m0 = t1;
    const int m1 = t1 + 8;
    float* ps = partials + split_id * 16 * 8;
    if (local_n0 < 8) {
        if (m0 < M) {
            ps[m0 * 8 + local_n0] = d0;
            ps[m0 * 8 + local_n1] = d1;
        }
        if (m1 < M) {
            ps[m1 * 8 + local_n0] = d2;
            ps[m1 * 8 + local_n1] = d3;
        }
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < 16 * 8; idx += blockDim.x) {
        const int local_m = idx / 8;
        const int local_n = idx - local_m * 8;
        const int abs_n = cta_n + local_n;
        if (local_m < M && abs_n < N) {
            float sum = 0.f;
            for (int s = 0; s < S; ++s) {
                sum += partials[(s * 16 + local_m) * 8 + local_n];
            }
            D[(long)local_m * N + abs_n] = __float2bfloat16(sum * (*alpha_ptr));
        }
    }
}

__global__ void decode_gemv_nvfp4_splitk_kernel_fused_mtile16_tma8_n4(
    int M, int N, int K, int S, int K_split,
    const uint8_t* __restrict__ A_packed,
    const uint8_t* __restrict__ B_packed,
    const uint8_t* __restrict__ SF_A,
    const uint8_t* __restrict__ SF_B,
    const float* __restrict__ alpha_ptr,
    __nv_bfloat16* __restrict__ D,
    CUTE_GRID_CONSTANT CUtensorMap const b_tma_desc)
{
    (void)B_packed;
    if (M < 1 || M > 16) return;

    constexpr int N_GROUPS = MTILE_N_GROUPS;
    constexpr int CTA_N = N_GROUPS * 8;
    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    if (warp_id >= S) return;

    const int split_id = warp_id;
    const int t0 = lane_id & 3;
    const int t1 = lane_id >> 2;
    const int k_start = split_id * K_split;
    const int K_chunks = K / 64;
    const int cta_n = blockIdx.x * CTA_N;

    extern __shared__ uint8_t smem[];
    const int smem_barrier_off = 0;
    const int smem_barrier_bytes =
        (MTILE_TMA_STAGES * (int)sizeof(uint64_t) + 15) & ~15;
    uint64_t* tma_B_barriers =
        reinterpret_cast<uint64_t*>(smem + smem_barrier_off);
    const int smem_B_off = (smem_barrier_off + smem_barrier_bytes + 127) & ~127;
    const int smem_B_stage_bytes = N_GROUPS * S * 8 * 64;
    const int smem_B_bytes = MTILE_TMA_STAGES * smem_B_stage_bytes;
    const int partial_off = (smem_B_off + smem_B_bytes + 15) & ~15;
    uint8_t* smem_B = smem + smem_B_off;
    float* partials = reinterpret_cast<float*>(smem + partial_off);

    // A-staging (cp.async double-buffer): each warp pulls its per-iteration A
    // tile (M rows * 64 bytes) into shared memory ahead of the MMA via
    // cp.async, so the in-loop A reads hit smem instead of stalling on
    // long-scoreboard global loads. The buffer is aliased onto the same region
    // as `partials` (disjoint lifetimes: A is consumed inside the K-loop,
    // partials are written only after it) so it costs no extra occupancy.
    const int A_TILE_BYTES = M * 64;                 // per-warp per-stage tile
    const int A_db_off = partial_off;                // alias over partials
    uint8_t* smemA_db = smem + A_db_off;
    uint8_t* smemA_warp_base = smemA_db + (long)split_id * (2 * A_TILE_BYTES);

    if (threadIdx.x < MTILE_TMA_STAGES) {
        cute::initialize_barrier(tma_B_barriers[threadIdx.x], 1);
    }
    __syncthreads();
    if (warp_id == 0 && cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&b_tma_desc);
    }
    __syncthreads();

    float lo0[N_GROUPS], lo1[N_GROUPS], hi0[N_GROUPS], hi1[N_GROUPS];
#pragma unroll
    for (int ng = 0; ng < N_GROUPS; ++ng) {
        lo0[ng] = 0.f;
        lo1[ng] = 0.f;
        hi0[ng] = 0.f;
        hi1[ng] = 0.f;
    }
    float unused0 = 0.f, unused1 = 0.f;

    if (threadIdx.x == 0) {
        cute::set_barrier_transaction_bytes(
            tma_B_barriers[0], smem_B_stage_bytes);
        cute::SM90_TMA_LOAD_3D::copy(
            &b_tma_desc,
            &tma_B_barriers[0],
            static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL),
            smem_B,
            0,
            cta_n,
            0);
    }

    // Prologue: kick off the cp.async for the first iteration's A tile.
    {
        uint8_t* a_dst = smemA_warp_base;  // buffer 0
        const long a_src_row0 = (long)k_start / 2;
#pragma unroll 1
        for (int idx = lane_id; idx < M * 4; idx += 32) {
            const int row = idx >> 2;
            const int chunk = idx & 3;
            cp_async_cg_16(
                a_dst + row * 64 + chunk * 16,
                A_packed + (long)row * (K / 2) + a_src_row0 + chunk * 16);
        }
        cp_async_commit();
    }

    for (int k_local = 0, iter = 0; k_local < K_split;
         k_local += K_CHUNK * 2, ++iter) {
        const int k_abs = k_start + k_local;
        const int stage = iter % MTILE_TMA_STAGES;
        const int phase = (iter / MTILE_TMA_STAGES) & 1;
        uint8_t* smemA_cur = smemA_warp_base + (iter & 1) * A_TILE_BYTES;
        uint8_t* smem_B_stage = smem_B + stage * smem_B_stage_bytes;
#if MTILE_CONSUMER_WAIT_BARRIER
        cute::wait_barrier(tma_B_barriers[stage], phase);
        cta3d_tma_shared_fence();
#else
        if (threadIdx.x == 0) {
            cute::wait_barrier(tma_B_barriers[stage], phase);
        }
        __syncthreads();
#endif

        const int next_k_local = k_local + K_CHUNK * 2;
#if !MTILE_POST_K_SYNC
        if (next_k_local < K_split && (iter + 1) >= MTILE_TMA_STAGES) {
            __syncthreads();
        }
#endif
        if (threadIdx.x == 0 && next_k_local < K_split) {
            const int next_stage = (iter + 1) % MTILE_TMA_STAGES;
            cute::set_barrier_transaction_bytes(
                tma_B_barriers[next_stage], smem_B_stage_bytes);
            cute::SM90_TMA_LOAD_3D::copy(
                &b_tma_desc,
                &tma_B_barriers[next_stage],
                static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL),
                smem_B + next_stage * smem_B_stage_bytes,
                next_k_local / 2,
                cta_n,
                0);
        }

        // Prefetch the next iteration's A tile into the alternate buffer, then
        // wait until only that prefetch remains in flight (=> the current
        // iteration's A tile, committed last round, is now resident).
        if (next_k_local < K_split) {
            uint8_t* a_next = smemA_warp_base + ((iter + 1) & 1) * A_TILE_BYTES;
            const long a_src_next = (long)(k_abs + K_CHUNK * 2) / 2;
#pragma unroll 1
            for (int idx = lane_id; idx < M * 4; idx += 32) {
                const int row = idx >> 2;
                const int chunk = idx & 3;
                cp_async_cg_16(
                    a_next + row * 64 + chunk * 16,
                    A_packed + (long)row * (K / 2) + a_src_next + chunk * 16);
            }
            cp_async_commit();
            cp_async_wait_group<1>();
        } else {
            cp_async_wait_group<0>();
        }
        __syncwarp();

        {
            const int kc = k_abs >> 6;
            const int m0 = t1;
            const int m1 = t1 + 8;
            const int safe_m1 = (m1 < M) ? m1 : 0;
            const uint8_t* ap0 = smemA_cur + m0 * 64;
            const uint8_t* ap1 = smemA_cur + safe_m1 * 64;
            const uint32_t a0 =
                (m0 < M) ? *reinterpret_cast<const uint32_t*>(ap0 + t0 * 4) : 0;
            const uint32_t a1 =
                (m1 < M) ? *reinterpret_cast<const uint32_t*>(ap1 + t0 * 4) : 0;
            const uint32_t a2 =
                (m0 < M) ? *reinterpret_cast<const uint32_t*>(ap0 + t0 * 4 + 16) : 0;
            const uint32_t a3 =
                (m1 < M) ? *reinterpret_cast<const uint32_t*>(ap1 + t0 * 4 + 16) : 0;
            const uint32_t sfa =
                load_mtile_sfa(SF_A, M, kc, K_chunks, lane_id);
#pragma unroll
            for (int ng = 0; ng < N_GROUPS; ++ng) {
                uint8_t* warp_B =
                    smem_B_stage + split_id * CTA_N * 64 + ng * 8 * 64;
                const uint32_t b0 = *reinterpret_cast<const uint32_t*>(
                    warp_B + cta3d_b_smem_offset(t1, t0 * 4));
                const uint32_t b1 = *reinterpret_cast<const uint32_t*>(
                    warp_B + cta3d_b_smem_offset(t1, t0 * 4 + 16));
                const int abs_n_sf = cta_n + ng * 8 + t1;
                const uint32_t sfb = (abs_n_sf < N)
                    ? *reinterpret_cast<const uint32_t*>(
                        SF_B + cta3d_sf_128x4_offset(abs_n_sf, kc, K_chunks))
                    : 0x38383838u;
                mma_mxf4nvf4_k64_task38(
                    lo0[ng], lo1[ng], hi0[ng], hi1[ng],
                    a0, a1, a2, a3, b0, b1, sfa, sfb);
            }
        }

        if (false) {
            const int kc = k_abs >> 6;
            const int m0 = t1 + 8;
            const uint8_t* ap0 = A_packed + (long)m0 * (K / 2) + k_abs / 2;
            const uint32_t a0 =
                (m0 < M) ? *reinterpret_cast<const uint32_t*>(ap0 + t0 * 4) : 0;
            const uint32_t a1 = 0;
            const uint32_t a2 =
                (m0 < M) ? *reinterpret_cast<const uint32_t*>(ap0 + t0 * 4 + 16) : 0;
            const uint32_t a3 = 0;
            const int sfa_row = ((lane_id & 1) == 0) ? (t1 + 8) : 16 + t1;
            const uint32_t sfa = (sfa_row < M)
                ? *reinterpret_cast<const uint32_t*>(
                    SF_A + cta3d_sf_128x4_offset(sfa_row, kc, K_chunks))
                : 0x38383838u;
#pragma unroll
            for (int ng = 0; ng < N_GROUPS; ++ng) {
                uint8_t* warp_B =
                    smem_B_stage + split_id * CTA_N * 64 + ng * 8 * 64;
                const uint32_t b0 = *reinterpret_cast<const uint32_t*>(
                    warp_B + cta3d_b_smem_offset(t1, t0 * 4));
                const uint32_t b1 = *reinterpret_cast<const uint32_t*>(
                    warp_B + cta3d_b_smem_offset(t1, t0 * 4 + 16));
                const int abs_n_sf = cta_n + ng * 8 + t1;
                const uint32_t sfb = (abs_n_sf < N)
                    ? *reinterpret_cast<const uint32_t*>(
                        SF_B + cta3d_sf_128x4_offset(abs_n_sf, kc, K_chunks))
                    : 0x38383838u;
                mma_mxf4nvf4_k64_task38(
                    hi0[ng], hi1[ng], unused0, unused1,
                    a0, a1, a2, a3, b0, b1, sfa, sfb);
            }
        }

        {
            const int kc = (k_abs >> 6) + 1;
            const int m0 = t1;
            const int m1 = t1 + 8;
            const int safe_m1 = (m1 < M) ? m1 : 0;
            const uint8_t* ap0 = smemA_cur + m0 * 64 + 32;
            const uint8_t* ap1 = smemA_cur + safe_m1 * 64 + 32;
            const uint32_t a0 =
                (m0 < M) ? *reinterpret_cast<const uint32_t*>(ap0 + t0 * 4) : 0;
            const uint32_t a1 =
                (m1 < M) ? *reinterpret_cast<const uint32_t*>(ap1 + t0 * 4) : 0;
            const uint32_t a2 =
                (m0 < M) ? *reinterpret_cast<const uint32_t*>(ap0 + t0 * 4 + 16) : 0;
            const uint32_t a3 =
                (m1 < M) ? *reinterpret_cast<const uint32_t*>(ap1 + t0 * 4 + 16) : 0;
            const uint32_t sfa =
                load_mtile_sfa(SF_A, M, kc, K_chunks, lane_id);
#pragma unroll
            for (int ng = 0; ng < N_GROUPS; ++ng) {
                uint8_t* warp_B =
                    smem_B_stage + split_id * CTA_N * 64 + ng * 8 * 64;
                const uint32_t b0 = *reinterpret_cast<const uint32_t*>(
                    warp_B + cta3d_b_smem_offset(t1, t0 * 4 + 32));
                const uint32_t b1 = *reinterpret_cast<const uint32_t*>(
                    warp_B + cta3d_b_smem_offset(t1, t0 * 4 + 48));
                const int abs_n_sf = cta_n + ng * 8 + t1;
                const uint32_t sfb = (abs_n_sf < N)
                    ? *reinterpret_cast<const uint32_t*>(
                        SF_B + cta3d_sf_128x4_offset(abs_n_sf, kc, K_chunks))
                    : 0x38383838u;
                mma_mxf4nvf4_k64_task38(
                    lo0[ng], lo1[ng], hi0[ng], hi1[ng],
                    a0, a1, a2, a3, b0, b1, sfa, sfb);
            }
        }

        if (false) {
            const int kc = (k_abs >> 6) + 1;
            const int m0 = t1 + 8;
            const uint8_t* ap0 = A_packed + (long)m0 * (K / 2) + k_abs / 2 + 32;
            const uint32_t a0 =
                (m0 < M) ? *reinterpret_cast<const uint32_t*>(ap0 + t0 * 4) : 0;
            const uint32_t a1 = 0;
            const uint32_t a2 =
                (m0 < M) ? *reinterpret_cast<const uint32_t*>(ap0 + t0 * 4 + 16) : 0;
            const uint32_t a3 = 0;
            const int sfa_row = ((lane_id & 1) == 0) ? (t1 + 8) : 16 + t1;
            const uint32_t sfa = (sfa_row < M)
                ? *reinterpret_cast<const uint32_t*>(
                    SF_A + cta3d_sf_128x4_offset(sfa_row, kc, K_chunks))
                : 0x38383838u;
#pragma unroll
            for (int ng = 0; ng < N_GROUPS; ++ng) {
                uint8_t* warp_B =
                    smem_B_stage + split_id * CTA_N * 64 + ng * 8 * 64;
                const uint32_t b0 = *reinterpret_cast<const uint32_t*>(
                    warp_B + cta3d_b_smem_offset(t1, t0 * 4 + 32));
                const uint32_t b1 = *reinterpret_cast<const uint32_t*>(
                    warp_B + cta3d_b_smem_offset(t1, t0 * 4 + 48));
                const int abs_n_sf = cta_n + ng * 8 + t1;
                const uint32_t sfb = (abs_n_sf < N)
                    ? *reinterpret_cast<const uint32_t*>(
                        SF_B + cta3d_sf_128x4_offset(abs_n_sf, kc, K_chunks))
                    : 0x38383838u;
                mma_mxf4nvf4_k64_task38(
                    hi0[ng], hi1[ng], unused0, unused1,
                    a0, a1, a2, a3, b0, b1, sfa, sfb);
            }
        }
#if MTILE_POST_K_SYNC
        __syncthreads();
#endif
    }

    const int pair_col = (lane_id & 3) * 2;
    const int m0 = t1;
    const int m1 = t1 + 8;
#pragma unroll
    for (int ng = 0; ng < N_GROUPS; ++ng) {
        float* ps = partials + split_id * M * CTA_N;
        const int local_n0 = ng * 8 + pair_col;
        const int local_n1 = local_n0 + 1;
        if (m0 < M) {
            *reinterpret_cast<float2*>(ps + m0 * CTA_N + local_n0) =
                make_float2(lo0[ng], lo1[ng]);
        }
        if (m1 < M) {
            *reinterpret_cast<float2*>(ps + m1 * CTA_N + local_n0) =
                make_float2(hi0[ng], hi1[ng]);
        }
    }
    __syncthreads();

    const float alpha = *alpha_ptr;
    for (int idx = threadIdx.x; idx < M * CTA_N; idx += blockDim.x) {
        const int local_m = idx / CTA_N;
        const int local_n = idx - local_m * CTA_N;
        const int abs_n = cta_n + local_n;
        if (abs_n < N) {
            float sum = 0.f;
            for (int s = 0; s < S; ++s) {
                sum += partials[(s * M + local_m) * CTA_N + local_n];
            }
            D[(long)local_m * N + abs_n] = __float2bfloat16(sum * alpha);
        }
    }
}

#if MTILE_CROSS_TILE_PIPE
// task_41 §4.2 cross-tile pipelining variant of `_mtile16_tma8_n4`.
// Same persistent grid-stride / arithmetic global phase / mbarrier
// init-once / cross-tile B-TMA at iter K_iters-1 / cross-tile A cp.async
// in epilogue fenced against the partials reads, as `_m16_tma8_ng_cte`,
// but with runtime M (must be 2..16 per the dispatcher gate). Smem layout,
// MMA, partials write, reduce, D-write are byte-identical to the non-CTE
// kernel above; only the tile loop, barrier init placement, and tile
// boundary differ.
__global__ void decode_gemv_nvfp4_splitk_kernel_fused_mtile16_tma8_n4_cte(
    int M, int N, int K, int S, int K_split, int num_tiles,
    const uint8_t* __restrict__ A_packed,
    const uint8_t* __restrict__ B_packed,
    const uint8_t* __restrict__ SF_A,
    const uint8_t* __restrict__ SF_B,
    const float* __restrict__ alpha_ptr,
    __nv_bfloat16* __restrict__ D,
    CUTE_GRID_CONSTANT CUtensorMap const b_tma_desc)
{
    (void)B_packed;
    if (M <= 1 || M > 16) return;

    constexpr int N_GROUPS = MTILE_N_GROUPS;
    constexpr int CTA_N = N_GROUPS * 8;
    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    if (warp_id >= S) return;

    const int split_id = warp_id;
    const int t0 = lane_id & 3;
    const int t1 = lane_id >> 2;
    const int k_start = split_id * K_split;
    const int K_chunks = K / 64;

    extern __shared__ uint8_t smem[];
    const int smem_barrier_off = 0;
    const int smem_barrier_bytes =
        (MTILE_TMA_STAGES * (int)sizeof(uint64_t) + 15) & ~15;
    uint64_t* tma_B_barriers =
        reinterpret_cast<uint64_t*>(smem + smem_barrier_off);
    const int smem_B_off = (smem_barrier_off + smem_barrier_bytes + 127) & ~127;
    const int smem_B_stage_bytes = N_GROUPS * S * 8 * 64;
    const int smem_B_bytes = MTILE_TMA_STAGES * smem_B_stage_bytes;
    const int partial_off = (smem_B_off + smem_B_bytes + 15) & ~15;
    uint8_t* smem_B = smem + smem_B_off;
    float* partials = reinterpret_cast<float*>(smem + partial_off);

    const int A_TILE_BYTES = M * 64;
    const int A_db_off = partial_off;  // alias over partials
    uint8_t* smemA_db = smem + A_db_off;
    uint8_t* smemA_warp_base = smemA_db + (long)split_id * (2 * A_TILE_BYTES);

    // mbarrier init-once + TMA descriptor prefetch OUTSIDE persistent loop
    // (round-4: per-tile re-init races with in-flight async-proxy TMA → UB).
    if (threadIdx.x < MTILE_TMA_STAGES) {
        cute::initialize_barrier(tma_B_barriers[threadIdx.x], 1);
    }
    __syncthreads();
    if (warp_id == 0 && cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&b_tma_desc);
    }
    __syncthreads();

    const int K_iters_per_tile = K_split / (K_CHUNK * 2);
    int global_iter = 0;
    float unused0 = 0.f, unused1 = 0.f;

    for (int tile_idx = blockIdx.x, tile_local = 0;
         tile_idx < num_tiles;
         tile_idx += (int)gridDim.x, ++tile_local) {
        const int cta_n = tile_idx * CTA_N;
        const bool first_tile_for_cta = (tile_local == 0);
        const bool last_tile_for_cta = (tile_idx + (int)gridDim.x >= num_tiles);

        if (first_tile_for_cta) {
            if (threadIdx.x == 0) {
                cute::set_barrier_transaction_bytes(
                    tma_B_barriers[0], smem_B_stage_bytes);
                cute::SM90_TMA_LOAD_3D::copy(
                    &b_tma_desc, &tma_B_barriers[0],
                    static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL),
                    smem_B, 0, cta_n, 0);
            }
            uint8_t* a_dst = smemA_warp_base;  // buffer 0 (global_iter==0)
            const long a_src_row0 = (long)k_start / 2;
#pragma unroll 1
            for (int idx = lane_id; idx < M * 4; idx += 32) {
                const int row = idx >> 2;
                const int chunk = idx & 3;
                cp_async_cg_16(
                    a_dst + row * 64 + chunk * 16,
                    A_packed + (long)row * (K / 2) + a_src_row0 + chunk * 16);
            }
            cp_async_commit();
        }

        // Per-tile accumulators.
        float lo0[N_GROUPS], lo1[N_GROUPS], hi0[N_GROUPS], hi1[N_GROUPS];
#pragma unroll
        for (int ng = 0; ng < N_GROUPS; ++ng) {
            lo0[ng] = 0.f; lo1[ng] = 0.f; hi0[ng] = 0.f; hi1[ng] = 0.f;
        }

        for (int k_local = 0, iter_in_tile = 0;
             k_local < K_split;
             k_local += K_CHUNK * 2, ++iter_in_tile, ++global_iter) {
            const int k_abs = k_start + k_local;
            const int stage = global_iter % MTILE_TMA_STAGES;
            const int phase = (global_iter / MTILE_TMA_STAGES) & 1;
            uint8_t* smemA_cur =
                smemA_warp_base + (global_iter & 1) * A_TILE_BYTES;
            uint8_t* smem_B_stage = smem_B + stage * smem_B_stage_bytes;
#if MTILE_CONSUMER_WAIT_BARRIER
            cute::wait_barrier(tma_B_barriers[stage], phase);
            cta3d_tma_shared_fence();
#else
            if (threadIdx.x == 0) {
                cute::wait_barrier(tma_B_barriers[stage], phase);
            }
            __syncthreads();
#endif

            const int next_k_local = k_local + K_CHUNK * 2;
            const bool is_last_iter_in_tile = (next_k_local >= K_split);
#if !MTILE_POST_K_SYNC
            if (next_k_local < K_split && (global_iter + 1) >= MTILE_TMA_STAGES) {
                __syncthreads();
            }
#endif
            // Within-tile next-iter B-TMA, OR cross-tile B-TMA at last K-iter
            // of a non-final tile. K_iters_per_tile==5 (odd) at K=5120/S=8 so
            // next-tile iter 0 lands on the opposite stage parity → no slot
            // conflict against the current tile's iter 4 stage.
            if (threadIdx.x == 0) {
                if (next_k_local < K_split) {
                    const int next_stage = (global_iter + 1) % MTILE_TMA_STAGES;
                    cute::set_barrier_transaction_bytes(
                        tma_B_barriers[next_stage], smem_B_stage_bytes);
                    cute::SM90_TMA_LOAD_3D::copy(
                        &b_tma_desc, &tma_B_barriers[next_stage],
                        static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL),
                        smem_B + next_stage * smem_B_stage_bytes,
                        next_k_local / 2, cta_n, 0);
                } else if (is_last_iter_in_tile && !last_tile_for_cta) {
                    const int next_tile_idx = tile_idx + (int)gridDim.x;
                    const int next_cta_n = next_tile_idx * CTA_N;
                    const int next_stage = (global_iter + 1) % MTILE_TMA_STAGES;
                    cute::set_barrier_transaction_bytes(
                        tma_B_barriers[next_stage], smem_B_stage_bytes);
                    cute::SM90_TMA_LOAD_3D::copy(
                        &b_tma_desc, &tma_B_barriers[next_stage],
                        static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL),
                        smem_B + next_stage * smem_B_stage_bytes,
                        0, next_cta_n, 0);
                }
            }

            // Within-tile next-iter A prefetch, OR drain at last K-iter.
            // Cross-tile A cp.async is deferred to the epilogue (after
            // __syncthreads on partials reads) to avoid the alias race.
            if (next_k_local < K_split) {
                uint8_t* a_next =
                    smemA_warp_base + ((global_iter + 1) & 1) * A_TILE_BYTES;
                const long a_src_next = (long)(k_abs + K_CHUNK * 2) / 2;
#pragma unroll 1
                for (int idx = lane_id; idx < M * 4; idx += 32) {
                    const int row = idx >> 2;
                    const int chunk = idx & 3;
                    cp_async_cg_16(
                        a_next + row * 64 + chunk * 16,
                        A_packed + (long)row * (K / 2) + a_src_next + chunk * 16);
                }
                cp_async_commit();
                cp_async_wait_group<1>();
            } else {
                cp_async_wait_group<0>();
            }
            __syncwarp();

            {
                const int kc = k_abs >> 6;
                const int m0 = t1;
                const int m1 = t1 + 8;
                const int safe_m1 = (m1 < M) ? m1 : 0;
                const uint8_t* ap0 = smemA_cur + m0 * 64;
                const uint8_t* ap1 = smemA_cur + safe_m1 * 64;
                const uint32_t a0 =
                    (m0 < M) ? *reinterpret_cast<const uint32_t*>(ap0 + t0 * 4) : 0;
                const uint32_t a1 =
                    (m1 < M) ? *reinterpret_cast<const uint32_t*>(ap1 + t0 * 4) : 0;
                const uint32_t a2 =
                    (m0 < M) ? *reinterpret_cast<const uint32_t*>(ap0 + t0 * 4 + 16) : 0;
                const uint32_t a3 =
                    (m1 < M) ? *reinterpret_cast<const uint32_t*>(ap1 + t0 * 4 + 16) : 0;
                const uint32_t sfa =
                    load_mtile_sfa(SF_A, M, kc, K_chunks, lane_id);
#pragma unroll
                for (int ng = 0; ng < N_GROUPS; ++ng) {
                    uint8_t* warp_B =
                        smem_B_stage + split_id * CTA_N * 64 + ng * 8 * 64;
                    const uint32_t b0 = *reinterpret_cast<const uint32_t*>(
                        warp_B + cta3d_b_smem_offset(t1, t0 * 4));
                    const uint32_t b1 = *reinterpret_cast<const uint32_t*>(
                        warp_B + cta3d_b_smem_offset(t1, t0 * 4 + 16));
                    const int abs_n_sf = cta_n + ng * 8 + t1;
                    const uint32_t sfb = (abs_n_sf < N)
                        ? *reinterpret_cast<const uint32_t*>(
                            SF_B + cta3d_sf_128x4_offset(abs_n_sf, kc, K_chunks))
                        : 0x38383838u;
                    mma_mxf4nvf4_k64_task38(
                        lo0[ng], lo1[ng], hi0[ng], hi1[ng],
                        a0, a1, a2, a3, b0, b1, sfa, sfb);
                }
            }

            {
                const int kc = (k_abs >> 6) + 1;
                const int m0 = t1;
                const int m1 = t1 + 8;
                const int safe_m1 = (m1 < M) ? m1 : 0;
                const uint8_t* ap0 = smemA_cur + m0 * 64 + 32;
                const uint8_t* ap1 = smemA_cur + safe_m1 * 64 + 32;
                const uint32_t a0 =
                    (m0 < M) ? *reinterpret_cast<const uint32_t*>(ap0 + t0 * 4) : 0;
                const uint32_t a1 =
                    (m1 < M) ? *reinterpret_cast<const uint32_t*>(ap1 + t0 * 4) : 0;
                const uint32_t a2 =
                    (m0 < M) ? *reinterpret_cast<const uint32_t*>(ap0 + t0 * 4 + 16) : 0;
                const uint32_t a3 =
                    (m1 < M) ? *reinterpret_cast<const uint32_t*>(ap1 + t0 * 4 + 16) : 0;
                const uint32_t sfa =
                    load_mtile_sfa(SF_A, M, kc, K_chunks, lane_id);
#pragma unroll
                for (int ng = 0; ng < N_GROUPS; ++ng) {
                    uint8_t* warp_B =
                        smem_B_stage + split_id * CTA_N * 64 + ng * 8 * 64;
                    const uint32_t b0 = *reinterpret_cast<const uint32_t*>(
                        warp_B + cta3d_b_smem_offset(t1, t0 * 4 + 32));
                    const uint32_t b1 = *reinterpret_cast<const uint32_t*>(
                        warp_B + cta3d_b_smem_offset(t1, t0 * 4 + 48));
                    const int abs_n_sf = cta_n + ng * 8 + t1;
                    const uint32_t sfb = (abs_n_sf < N)
                        ? *reinterpret_cast<const uint32_t*>(
                            SF_B + cta3d_sf_128x4_offset(abs_n_sf, kc, K_chunks))
                        : 0x38383838u;
                    mma_mxf4nvf4_k64_task38(
                        lo0[ng], lo1[ng], hi0[ng], hi1[ng],
                        a0, a1, a2, a3, b0, b1, sfa, sfb);
                }
            }
            (void)unused0; (void)unused1;
#if MTILE_POST_K_SYNC
            __syncthreads();
#endif
        }  // end K-loop

        // Per-tile epilogue.
        const int pair_col = (lane_id & 3) * 2;
        const int m0 = t1;
        const int m1 = t1 + 8;
#pragma unroll
        for (int ng = 0; ng < N_GROUPS; ++ng) {
            float* ps = partials + split_id * M * CTA_N;
            const int local_n0 = ng * 8 + pair_col;
            if (m0 < M) {
                *reinterpret_cast<float2*>(ps + m0 * CTA_N + local_n0) =
                    make_float2(lo0[ng], lo1[ng]);
            }
            if (m1 < M) {
                *reinterpret_cast<float2*>(ps + m1 * CTA_N + local_n0) =
                    make_float2(hi0[ng], hi1[ng]);
            }
        }
        __syncthreads();

        const float alpha = *alpha_ptr;
        for (int idx = threadIdx.x; idx < M * CTA_N; idx += blockDim.x) {
            const int local_m = idx / CTA_N;
            const int local_n = idx - local_m * CTA_N;
            const int abs_n = cta_n + local_n;
            if (abs_n < N) {
                float sum = 0.f;
                for (int s = 0; s < S; ++s) {
                    sum += partials[(s * M + local_m) * CTA_N + local_n];
                }
                D[(long)local_m * N + abs_n] = __float2bfloat16(sum * alpha);
            }
        }

        // Cross-tile A prefetch — fence partials reads first (alias safety).
        if (!last_tile_for_cta) {
            __syncthreads();
            uint8_t* a_dst =
                smemA_warp_base + (global_iter & 1) * A_TILE_BYTES;
            const long a_src_row0 = (long)k_start / 2;
#pragma unroll 1
            for (int idx = lane_id; idx < M * 4; idx += 32) {
                const int row = idx >> 2;
                const int chunk = idx & 3;
                cp_async_cg_16(
                    a_dst + row * 64 + chunk * 16,
                    A_packed + (long)row * (K / 2) + a_src_row0 + chunk * 16);
            }
            cp_async_commit();
        }
    }
}
#endif  // MTILE_CROSS_TILE_PIPE

__global__ void decode_gemv_nvfp4_splitk_kernel_fused_m16_tma8_ng(
    int N, int K, int S, int K_split,
    const uint8_t* __restrict__ A_packed,
    const uint8_t* __restrict__ SF_A,
    const uint8_t* __restrict__ SF_B,
    const float* __restrict__ alpha_ptr,
    __nv_bfloat16* __restrict__ D,
    CUTE_GRID_CONSTANT CUtensorMap const b_tma_desc)
{
    constexpr int N_GROUPS = MTILE_N_GROUPS;
    constexpr int CTA_N = N_GROUPS * 8;
    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    if (warp_id >= S) return;

    const int split_id = warp_id;
    const int t0 = lane_id & 3;
    const int t1 = lane_id >> 2;
    const int k_start = split_id * K_split;
    const int K_chunks = K / 64;
    const int cta_n = blockIdx.x * CTA_N;

    extern __shared__ uint8_t smem[];
    const int smem_barrier_bytes =
        (MTILE_TMA_STAGES * (int)sizeof(uint64_t) + 15) & ~15;
    uint64_t* tma_B_barriers = reinterpret_cast<uint64_t*>(smem);
    const int smem_B_off = (smem_barrier_bytes + 127) & ~127;
    const int smem_B_stage_bytes = N_GROUPS * S * 8 * 64;
    const int smem_B_bytes = MTILE_TMA_STAGES * smem_B_stage_bytes;
    const int partial_off = (smem_B_off + smem_B_bytes + 15) & ~15;
    uint8_t* smem_B = smem + smem_B_off;
    float* partials = reinterpret_cast<float*>(smem + partial_off);

    // A-staging (cp.async double-buffer): mirrors the n4 kernel. Each warp pulls
    // its per-iteration A tile (16 rows * 64 bytes) into shared memory ahead of
    // the MMA via cp.async, so the in-loop A reads hit smem instead of stalling
    // on long-scoreboard global loads. The buffer aliases the partials region
    // (disjoint lifetimes: A is consumed inside the K-loop, partials only after)
    // so it costs no extra occupancy. M is always 16 on this path.
    constexpr int M16_A_TILE_BYTES = 16 * 64;     // per-warp per-stage tile
    const int A_db_off = partial_off;             // alias over partials
    uint8_t* smemA_db = smem + A_db_off;
    uint8_t* smemA_warp_base =
        smemA_db + (long)split_id * (2 * M16_A_TILE_BYTES);

    if (threadIdx.x < MTILE_TMA_STAGES) {
        cute::initialize_barrier(tma_B_barriers[threadIdx.x], 1);
    }
    __syncthreads();
    if (warp_id == 0 && cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&b_tma_desc);
    }
    __syncthreads();

    float lo0[N_GROUPS], lo1[N_GROUPS], hi0[N_GROUPS], hi1[N_GROUPS];
#pragma unroll
    for (int ng = 0; ng < N_GROUPS; ++ng) {
        lo0[ng] = 0.f;
        lo1[ng] = 0.f;
        hi0[ng] = 0.f;
        hi1[ng] = 0.f;
    }

    if (threadIdx.x == 0) {
        cute::set_barrier_transaction_bytes(
            tma_B_barriers[0], smem_B_stage_bytes);
        cute::SM90_TMA_LOAD_3D::copy(
            &b_tma_desc,
            &tma_B_barriers[0],
            static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL),
            smem_B,
            0,
            cta_n,
            0);
    }

    // Prologue: kick off the cp.async for the first iteration's A tile (buffer 0).
    {
        uint8_t* a_dst = smemA_warp_base;  // buffer 0
        const long a_src_row0 = (long)k_start / 2;
#pragma unroll 1
        for (int idx = lane_id; idx < 16 * 4; idx += 32) {
            const int row = idx >> 2;
            const int chunk = idx & 3;
            cp_async_cg_16(
                a_dst + row * 64 + chunk * 16,
                A_packed + (long)row * (K / 2) + a_src_row0 + chunk * 16);
        }
        cp_async_commit();
    }

    for (int k_local = 0, iter = 0; k_local < K_split;
         k_local += K_CHUNK * 2, ++iter) {
        const int k_abs = k_start + k_local;
        const int stage = iter % MTILE_TMA_STAGES;
        const int phase = (iter / MTILE_TMA_STAGES) & 1;
        uint8_t* smemA_cur = smemA_warp_base + (iter & 1) * M16_A_TILE_BYTES;
        uint8_t* smem_B_stage = smem_B + stage * smem_B_stage_bytes;
#if MTILE_CONSUMER_WAIT_BARRIER
        cute::wait_barrier(tma_B_barriers[stage], phase);
        cta3d_tma_shared_fence();
#else
        if (threadIdx.x == 0) {
            cute::wait_barrier(tma_B_barriers[stage], phase);
        }
        __syncthreads();
#endif

        const int next_k_local = k_local + K_CHUNK * 2;
#if !MTILE_POST_K_SYNC
        if (next_k_local < K_split && (iter + 1) >= MTILE_TMA_STAGES) {
            __syncthreads();
        }
#endif
        if (threadIdx.x == 0 && next_k_local < K_split) {
            const int next_stage = (iter + 1) % MTILE_TMA_STAGES;
            cute::set_barrier_transaction_bytes(
                tma_B_barriers[next_stage], smem_B_stage_bytes);
            cute::SM90_TMA_LOAD_3D::copy(
                &b_tma_desc,
                &tma_B_barriers[next_stage],
                static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL),
                smem_B + next_stage * smem_B_stage_bytes,
                next_k_local / 2,
                cta_n,
                0);
        }

        // Prefetch the next iteration's A tile into the alternate buffer, then
        // wait until only that prefetch remains in flight (=> the current
        // iteration's A tile, committed last round, is now resident).
        if (next_k_local < K_split) {
            uint8_t* a_next =
                smemA_warp_base + ((iter + 1) & 1) * M16_A_TILE_BYTES;
            const long a_src_next = (long)(k_abs + K_CHUNK * 2) / 2;
#pragma unroll 1
            for (int idx = lane_id; idx < 16 * 4; idx += 32) {
                const int row = idx >> 2;
                const int chunk = idx & 3;
                cp_async_cg_16(
                    a_next + row * 64 + chunk * 16,
                    A_packed + (long)row * (K / 2) + a_src_next + chunk * 16);
            }
            cp_async_commit();
            cp_async_wait_group<1>();
        } else {
            cp_async_wait_group<0>();
        }
        __syncwarp();

        {
            const int kc = k_abs >> 6;
            const uint8_t* ap0 = smemA_cur + t1 * 64;
            const uint8_t* ap1 = smemA_cur + (t1 + 8) * 64;
            const uint32_t a0 = *reinterpret_cast<const uint32_t*>(ap0 + t0 * 4);
            const uint32_t a1 = *reinterpret_cast<const uint32_t*>(ap1 + t0 * 4);
            const uint32_t a2 = *reinterpret_cast<const uint32_t*>(ap0 + t0 * 4 + 16);
            const uint32_t a3 = *reinterpret_cast<const uint32_t*>(ap1 + t0 * 4 + 16);
            const int sfa_row = (lane_id & 1) * 8 + t1;
            const uint32_t sfa = *reinterpret_cast<const uint32_t*>(
                SF_A + cta3d_sf_128x4_offset(sfa_row, kc, K_chunks));
#pragma unroll
            for (int ng = 0; ng < N_GROUPS; ++ng) {
                uint8_t* warp_B =
                    smem_B_stage + split_id * CTA_N * 64 + ng * 8 * 64;
                const uint32_t b0 = *reinterpret_cast<const uint32_t*>(
                    warp_B + cta3d_b_smem_offset(t1, t0 * 4));
                const uint32_t b1 = *reinterpret_cast<const uint32_t*>(
                    warp_B + cta3d_b_smem_offset(t1, t0 * 4 + 16));
                const int abs_n_sf = cta_n + ng * 8 + t1;
                const uint32_t sfb = (abs_n_sf < N)
                    ? *reinterpret_cast<const uint32_t*>(
                        SF_B + cta3d_sf_128x4_offset(abs_n_sf, kc, K_chunks))
                    : 0x38383838u;
                mma_mxf4nvf4_k64_task38(
                    lo0[ng], lo1[ng], hi0[ng], hi1[ng],
                    a0, a1, a2, a3, b0, b1, sfa, sfb);
            }
        }

        {
            const int kc = (k_abs >> 6) + 1;
            const uint8_t* ap0 = smemA_cur + t1 * 64 + 32;
            const uint8_t* ap1 = smemA_cur + (t1 + 8) * 64 + 32;
            const uint32_t a0 = *reinterpret_cast<const uint32_t*>(ap0 + t0 * 4);
            const uint32_t a1 = *reinterpret_cast<const uint32_t*>(ap1 + t0 * 4);
            const uint32_t a2 = *reinterpret_cast<const uint32_t*>(ap0 + t0 * 4 + 16);
            const uint32_t a3 = *reinterpret_cast<const uint32_t*>(ap1 + t0 * 4 + 16);
            const int sfa_row = (lane_id & 1) * 8 + t1;
            const uint32_t sfa = *reinterpret_cast<const uint32_t*>(
                SF_A + cta3d_sf_128x4_offset(sfa_row, kc, K_chunks));
#pragma unroll
            for (int ng = 0; ng < N_GROUPS; ++ng) {
                uint8_t* warp_B =
                    smem_B_stage + split_id * CTA_N * 64 + ng * 8 * 64;
                const uint32_t b0 = *reinterpret_cast<const uint32_t*>(
                    warp_B + cta3d_b_smem_offset(t1, t0 * 4 + 32));
                const uint32_t b1 = *reinterpret_cast<const uint32_t*>(
                    warp_B + cta3d_b_smem_offset(t1, t0 * 4 + 48));
                const int abs_n_sf = cta_n + ng * 8 + t1;
                const uint32_t sfb = (abs_n_sf < N)
                    ? *reinterpret_cast<const uint32_t*>(
                        SF_B + cta3d_sf_128x4_offset(abs_n_sf, kc, K_chunks))
                    : 0x38383838u;
                mma_mxf4nvf4_k64_task38(
                    lo0[ng], lo1[ng], hi0[ng], hi1[ng],
                    a0, a1, a2, a3, b0, b1, sfa, sfb);
            }
        }
#if MTILE_POST_K_SYNC
        __syncthreads();
#endif
    }

    const int pair_col = (lane_id & 3) * 2;
    const int m0 = t1;
    const int m1 = t1 + 8;
#pragma unroll
    for (int ng = 0; ng < N_GROUPS; ++ng) {
        float* ps = partials + split_id * 16 * CTA_N;
        const int local_n0 = ng * 8 + pair_col;
        const int local_n1 = local_n0 + 1;
        ps[m0 * CTA_N + local_n0] = lo0[ng];
        ps[m0 * CTA_N + local_n1] = lo1[ng];
        ps[m1 * CTA_N + local_n0] = hi0[ng];
        ps[m1 * CTA_N + local_n1] = hi1[ng];
    }
    __syncthreads();

    const float alpha = *alpha_ptr;
    for (int idx = threadIdx.x; idx < 16 * CTA_N; idx += blockDim.x) {
        const int local_m = idx / CTA_N;
        const int local_n = idx - local_m * CTA_N;
        const int abs_n = cta_n + local_n;
        if (abs_n < N) {
            float sum = 0.f;
            if (S == 8) {
                sum =
                    partials[(0 * 16 + local_m) * CTA_N + local_n] +
                    partials[(1 * 16 + local_m) * CTA_N + local_n] +
                    partials[(2 * 16 + local_m) * CTA_N + local_n] +
                    partials[(3 * 16 + local_m) * CTA_N + local_n] +
                    partials[(4 * 16 + local_m) * CTA_N + local_n] +
                    partials[(5 * 16 + local_m) * CTA_N + local_n] +
                    partials[(6 * 16 + local_m) * CTA_N + local_n] +
                    partials[(7 * 16 + local_m) * CTA_N + local_n];
            } else {
                for (int s = 0; s < S; ++s) {
                    sum += partials[(s * 16 + local_m) * CTA_N + local_n];
                }
            }
            D[(long)local_m * N + abs_n] = __float2bfloat16(sum * alpha);
        }
    }
}

#if MTILE_CROSS_TILE_PIPE
// task_41 §4.2 cross-tile pipelining variant of `_m16_tma8_ng`.
// Persistent grid-stride over output N-tiles; arithmetic global phase
// (no dynamic bphase[stage] array — round-4 spill avoidance); mbarrier
// init-once before the persistent loop; cross-tile B-TMA issued at the
// last K-iter of the current tile to the opposite stage parity; cross-tile
// A cp.async issued in the epilogue AFTER an explicit __syncthreads that
// fences the partials reads (partials and smemA alias the same bytes).
// Smem layout, MMA, partials write/reduce/D-write are byte-identical to
// the non-CTE kernel above; only the launch frame and tile boundary differ.
__global__ void decode_gemv_nvfp4_splitk_kernel_fused_m16_tma8_ng_cte(
    int N, int K, int S, int K_split, int num_tiles,
    const uint8_t* __restrict__ A_packed,
    const uint8_t* __restrict__ SF_A,
    const uint8_t* __restrict__ SF_B,
    const float* __restrict__ alpha_ptr,
    __nv_bfloat16* __restrict__ D,
    CUTE_GRID_CONSTANT CUtensorMap const b_tma_desc)
{
    constexpr int N_GROUPS = MTILE_N_GROUPS;
    constexpr int CTA_N = N_GROUPS * 8;
    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    if (warp_id >= S) return;

    const int split_id = warp_id;
    const int t0 = lane_id & 3;
    const int t1 = lane_id >> 2;
    const int k_start = split_id * K_split;
    const int K_chunks = K / 64;

    extern __shared__ uint8_t smem[];
    const int smem_barrier_bytes =
        (MTILE_TMA_STAGES * (int)sizeof(uint64_t) + 15) & ~15;
    uint64_t* tma_B_barriers = reinterpret_cast<uint64_t*>(smem);
    const int smem_B_off = (smem_barrier_bytes + 127) & ~127;
    const int smem_B_stage_bytes = N_GROUPS * S * 8 * 64;
    const int smem_B_bytes = MTILE_TMA_STAGES * smem_B_stage_bytes;
    const int partial_off = (smem_B_off + smem_B_bytes + 15) & ~15;
    uint8_t* smem_B = smem + smem_B_off;
    float* partials = reinterpret_cast<float*>(smem + partial_off);

    constexpr int M16_A_TILE_BYTES = 16 * 64;
    const int A_db_off = partial_off;  // alias over partials (disjoint lifetimes)
    uint8_t* smemA_db = smem + A_db_off;
    uint8_t* smemA_warp_base =
        smemA_db + (long)split_id * (2 * M16_A_TILE_BYTES);

    // mbarrier init-once + TMA descriptor prefetch (OUTSIDE the persistent
    // loop — re-initializing per tile races with in-flight async-proxy TMA
    // completions, the round-4 deadlock root cause).
    if (threadIdx.x < MTILE_TMA_STAGES) {
        cute::initialize_barrier(tma_B_barriers[threadIdx.x], 1);
    }
    __syncthreads();
    if (warp_id == 0 && cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&b_tma_desc);
    }
    __syncthreads();

    const int K_iters_per_tile = K_split / (K_CHUNK * 2);  // 5 at K=5120 S=8
    int global_iter = 0;

    // PERSISTENT GRID-STRIDE over N-tiles.
    for (int tile_idx = blockIdx.x, tile_local = 0;
         tile_idx < num_tiles;
         tile_idx += (int)gridDim.x, ++tile_local) {
        const int cta_n = tile_idx * CTA_N;
        const bool first_tile_for_cta = (tile_local == 0);
        const bool last_tile_for_cta = (tile_idx + (int)gridDim.x >= num_tiles);

        // First tile only: prologue B-TMA (stage 0) + prologue A cp.async (buf 0).
        // Subsequent tiles inherit the prior tile's iter-4 cross-tile B-TMA
        // and the prior tile's epilogue cross-tile A cp.async.
        if (first_tile_for_cta) {
            if (threadIdx.x == 0) {
                cute::set_barrier_transaction_bytes(
                    tma_B_barriers[0], smem_B_stage_bytes);
                cute::SM90_TMA_LOAD_3D::copy(
                    &b_tma_desc, &tma_B_barriers[0],
                    static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL),
                    smem_B, 0, cta_n, 0);
            }
            uint8_t* a_dst = smemA_warp_base;  // buffer 0 (global_iter==0)
            const long a_src_row0 = (long)k_start / 2;
#pragma unroll 1
            for (int idx = lane_id; idx < 16 * 4; idx += 32) {
                const int row = idx >> 2;
                const int chunk = idx & 3;
                cp_async_cg_16(
                    a_dst + row * 64 + chunk * 16,
                    A_packed + (long)row * (K / 2) + a_src_row0 + chunk * 16);
            }
            cp_async_commit();
        }

        // Per-tile accumulators (live in registers; reset every tile).
        float lo0[N_GROUPS], lo1[N_GROUPS], hi0[N_GROUPS], hi1[N_GROUPS];
#pragma unroll
        for (int ng = 0; ng < N_GROUPS; ++ng) {
            lo0[ng] = 0.f; lo1[ng] = 0.f; hi0[ng] = 0.f; hi1[ng] = 0.f;
        }

        // Inner K-loop. global_iter is the persistent counter feeding the
        // arithmetic phase formula; iter_in_tile is only used to detect the
        // last K-iter for the cross-tile B-TMA issue.
        for (int k_local = 0, iter_in_tile = 0;
             k_local < K_split;
             k_local += K_CHUNK * 2, ++iter_in_tile, ++global_iter) {
            const int k_abs = k_start + k_local;
            const int stage = global_iter % MTILE_TMA_STAGES;
            const int phase = (global_iter / MTILE_TMA_STAGES) & 1;
            uint8_t* smemA_cur =
                smemA_warp_base + (global_iter & 1) * M16_A_TILE_BYTES;
            uint8_t* smem_B_stage = smem_B + stage * smem_B_stage_bytes;
#if MTILE_CONSUMER_WAIT_BARRIER
            cute::wait_barrier(tma_B_barriers[stage], phase);
            cta3d_tma_shared_fence();
#else
            if (threadIdx.x == 0) {
                cute::wait_barrier(tma_B_barriers[stage], phase);
            }
            __syncthreads();
#endif

            const int next_k_local = k_local + K_CHUNK * 2;
            const bool is_last_iter_in_tile = (next_k_local >= K_split);
#if !MTILE_POST_K_SYNC
            if (next_k_local < K_split && (global_iter + 1) >= MTILE_TMA_STAGES) {
                __syncthreads();
            }
#endif
            // Within-tile next-iter B-TMA, OR cross-tile B-TMA at iter 4.
            if (threadIdx.x == 0) {
                if (next_k_local < K_split) {
                    const int next_stage = (global_iter + 1) % MTILE_TMA_STAGES;
                    cute::set_barrier_transaction_bytes(
                        tma_B_barriers[next_stage], smem_B_stage_bytes);
                    cute::SM90_TMA_LOAD_3D::copy(
                        &b_tma_desc, &tma_B_barriers[next_stage],
                        static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL),
                        smem_B + next_stage * smem_B_stage_bytes,
                        next_k_local / 2, cta_n, 0);
                } else if (is_last_iter_in_tile && !last_tile_for_cta) {
                    // Cross-tile: next tile's iter 0 stage parity is opposite
                    // of current tile's iter 4 because K_iters_per_tile==5 is odd.
                    const int next_tile_idx = tile_idx + (int)gridDim.x;
                    const int next_cta_n = next_tile_idx * CTA_N;
                    const int next_stage = (global_iter + 1) % MTILE_TMA_STAGES;
                    cute::set_barrier_transaction_bytes(
                        tma_B_barriers[next_stage], smem_B_stage_bytes);
                    cute::SM90_TMA_LOAD_3D::copy(
                        &b_tma_desc, &tma_B_barriers[next_stage],
                        static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL),
                        smem_B + next_stage * smem_B_stage_bytes,
                        0, next_cta_n, 0);
                }
            }

            // Within-tile next-iter A cp.async, OR drain at the last iter
            // (cross-tile A cp.async is deferred to the epilogue so it can
            // safely fence against the partials reads via __syncthreads).
            if (next_k_local < K_split) {
                uint8_t* a_next =
                    smemA_warp_base + ((global_iter + 1) & 1) * M16_A_TILE_BYTES;
                const long a_src_next = (long)(k_abs + K_CHUNK * 2) / 2;
#pragma unroll 1
                for (int idx = lane_id; idx < 16 * 4; idx += 32) {
                    const int row = idx >> 2;
                    const int chunk = idx & 3;
                    cp_async_cg_16(
                        a_next + row * 64 + chunk * 16,
                        A_packed + (long)row * (K / 2) + a_src_next + chunk * 16);
                }
                cp_async_commit();
                cp_async_wait_group<1>();
            } else {
                cp_async_wait_group<0>();
            }
            __syncwarp();

            {
                const int kc = k_abs >> 6;
                const uint8_t* ap0 = smemA_cur + t1 * 64;
                const uint8_t* ap1 = smemA_cur + (t1 + 8) * 64;
                const uint32_t a0 = *reinterpret_cast<const uint32_t*>(ap0 + t0 * 4);
                const uint32_t a1 = *reinterpret_cast<const uint32_t*>(ap1 + t0 * 4);
                const uint32_t a2 = *reinterpret_cast<const uint32_t*>(ap0 + t0 * 4 + 16);
                const uint32_t a3 = *reinterpret_cast<const uint32_t*>(ap1 + t0 * 4 + 16);
                const int sfa_row = (lane_id & 1) * 8 + t1;
                const uint32_t sfa = *reinterpret_cast<const uint32_t*>(
                    SF_A + cta3d_sf_128x4_offset(sfa_row, kc, K_chunks));
#pragma unroll
                for (int ng = 0; ng < N_GROUPS; ++ng) {
                    uint8_t* warp_B =
                        smem_B_stage + split_id * CTA_N * 64 + ng * 8 * 64;
                    const uint32_t b0 = *reinterpret_cast<const uint32_t*>(
                        warp_B + cta3d_b_smem_offset(t1, t0 * 4));
                    const uint32_t b1 = *reinterpret_cast<const uint32_t*>(
                        warp_B + cta3d_b_smem_offset(t1, t0 * 4 + 16));
                    const int abs_n_sf = cta_n + ng * 8 + t1;
                    const uint32_t sfb = (abs_n_sf < N)
                        ? *reinterpret_cast<const uint32_t*>(
                            SF_B + cta3d_sf_128x4_offset(abs_n_sf, kc, K_chunks))
                        : 0x38383838u;
                    mma_mxf4nvf4_k64_task38(
                        lo0[ng], lo1[ng], hi0[ng], hi1[ng],
                        a0, a1, a2, a3, b0, b1, sfa, sfb);
                }
            }
            {
                const int kc = (k_abs >> 6) + 1;
                const uint8_t* ap0 = smemA_cur + t1 * 64 + 32;
                const uint8_t* ap1 = smemA_cur + (t1 + 8) * 64 + 32;
                const uint32_t a0 = *reinterpret_cast<const uint32_t*>(ap0 + t0 * 4);
                const uint32_t a1 = *reinterpret_cast<const uint32_t*>(ap1 + t0 * 4);
                const uint32_t a2 = *reinterpret_cast<const uint32_t*>(ap0 + t0 * 4 + 16);
                const uint32_t a3 = *reinterpret_cast<const uint32_t*>(ap1 + t0 * 4 + 16);
                const int sfa_row = (lane_id & 1) * 8 + t1;
                const uint32_t sfa = *reinterpret_cast<const uint32_t*>(
                    SF_A + cta3d_sf_128x4_offset(sfa_row, kc, K_chunks));
#pragma unroll
                for (int ng = 0; ng < N_GROUPS; ++ng) {
                    uint8_t* warp_B =
                        smem_B_stage + split_id * CTA_N * 64 + ng * 8 * 64;
                    const uint32_t b0 = *reinterpret_cast<const uint32_t*>(
                        warp_B + cta3d_b_smem_offset(t1, t0 * 4 + 32));
                    const uint32_t b1 = *reinterpret_cast<const uint32_t*>(
                        warp_B + cta3d_b_smem_offset(t1, t0 * 4 + 48));
                    const int abs_n_sf = cta_n + ng * 8 + t1;
                    const uint32_t sfb = (abs_n_sf < N)
                        ? *reinterpret_cast<const uint32_t*>(
                            SF_B + cta3d_sf_128x4_offset(abs_n_sf, kc, K_chunks))
                        : 0x38383838u;
                    mma_mxf4nvf4_k64_task38(
                        lo0[ng], lo1[ng], hi0[ng], hi1[ng],
                        a0, a1, a2, a3, b0, b1, sfa, sfb);
                }
            }
#if MTILE_POST_K_SYNC
            __syncthreads();
#endif
        }  // end K-loop

        // Per-tile epilogue: write partials, cross-warp reduce, write D.
        const int pair_col = (lane_id & 3) * 2;
        const int m0 = t1;
        const int m1 = t1 + 8;
#pragma unroll
        for (int ng = 0; ng < N_GROUPS; ++ng) {
            float* ps = partials + split_id * 16 * CTA_N;
            const int local_n0 = ng * 8 + pair_col;
            const int local_n1 = local_n0 + 1;
            ps[m0 * CTA_N + local_n0] = lo0[ng];
            ps[m0 * CTA_N + local_n1] = lo1[ng];
            ps[m1 * CTA_N + local_n0] = hi0[ng];
            ps[m1 * CTA_N + local_n1] = hi1[ng];
        }
        __syncthreads();

        const float alpha = *alpha_ptr;
        for (int idx = threadIdx.x; idx < 16 * CTA_N; idx += blockDim.x) {
            const int local_m = idx / CTA_N;
            const int local_n = idx - local_m * CTA_N;
            const int abs_n = cta_n + local_n;
            if (abs_n < N) {
                float sum = 0.f;
                if (S == 8) {
                    sum =
                        partials[(0 * 16 + local_m) * CTA_N + local_n] +
                        partials[(1 * 16 + local_m) * CTA_N + local_n] +
                        partials[(2 * 16 + local_m) * CTA_N + local_n] +
                        partials[(3 * 16 + local_m) * CTA_N + local_n] +
                        partials[(4 * 16 + local_m) * CTA_N + local_n] +
                        partials[(5 * 16 + local_m) * CTA_N + local_n] +
                        partials[(6 * 16 + local_m) * CTA_N + local_n] +
                        partials[(7 * 16 + local_m) * CTA_N + local_n];
                } else {
                    for (int s = 0; s < S; ++s) {
                        sum += partials[(s * 16 + local_m) * CTA_N + local_n];
                    }
                }
                D[(long)local_m * N + abs_n] = __float2bfloat16(sum * alpha);
            }
        }

        // Cross-tile A prefetch (smem-alias-safe): fence partials reads
        // before the cp.async overwrites the aliased region. global_iter has
        // already been incremented past the last K-iter of this tile, so
        // (global_iter & 1) is the buffer the next tile's iter 0 will read.
        if (!last_tile_for_cta) {
            __syncthreads();
            uint8_t* a_dst =
                smemA_warp_base + (global_iter & 1) * M16_A_TILE_BYTES;
            const long a_src_row0 = (long)k_start / 2;
#pragma unroll 1
            for (int idx = lane_id; idx < 16 * 4; idx += 32) {
                const int row = idx >> 2;
                const int chunk = idx & 3;
                cp_async_cg_16(
                    a_dst + row * 64 + chunk * 16,
                    A_packed + (long)row * (K / 2) + a_src_row0 + chunk * 16);
            }
            cp_async_commit();
        }
    }  // end persistent grid-stride loop
}
#endif  // MTILE_CROSS_TILE_PIPE

__global__ void decode_gemv_nvfp4_mtile16_nwarp_tma_kernel(
    int M, int N, int K,
    const uint8_t* __restrict__ A_packed,
    const uint8_t* __restrict__ SF_A,
    const uint8_t* __restrict__ SF_B,
    const float* __restrict__ alpha_ptr,
    __nv_bfloat16* __restrict__ D,
    CUTE_GRID_CONSTANT CUtensorMap const b_tma_desc)
{
    constexpr int WARPS_N = 8;
    constexpr int N_GROUPS_PER_WARP = MTILE_NWARP_N_GROUPS;
    constexpr int CTA_N = WARPS_N * N_GROUPS_PER_WARP * 8;
    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    if (warp_id >= WARPS_N) return;

    const int t0 = lane_id & 3;
    const int t1 = lane_id >> 2;
    const int K_chunks = K / 64;
    const int cta_n = blockIdx.x * CTA_N;

    extern __shared__ uint8_t smem[];
    const int smem_barrier_bytes =
        (MTILE_TMA_STAGES * (int)sizeof(uint64_t) + 15) & ~15;
    uint64_t* tma_B_barriers = reinterpret_cast<uint64_t*>(smem);
    const int smem_B_off = (smem_barrier_bytes + 127) & ~127;
    const int smem_B_stage_bytes = CTA_N * 64;
    uint8_t* smem_B = smem + smem_B_off;

    if (threadIdx.x < MTILE_TMA_STAGES) {
        cute::initialize_barrier(tma_B_barriers[threadIdx.x], 1);
    }
    __syncthreads();
    if (warp_id == 0 && cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&b_tma_desc);
    }
    __syncthreads();

    float lo0[N_GROUPS_PER_WARP], lo1[N_GROUPS_PER_WARP];
    float hi0[N_GROUPS_PER_WARP], hi1[N_GROUPS_PER_WARP];
#pragma unroll
    for (int ng = 0; ng < N_GROUPS_PER_WARP; ++ng) {
        lo0[ng] = 0.f;
        lo1[ng] = 0.f;
        hi0[ng] = 0.f;
        hi1[ng] = 0.f;
    }

    if (threadIdx.x == 0) {
        cute::set_barrier_transaction_bytes(
            tma_B_barriers[0], smem_B_stage_bytes);
        cute::SM90_TMA_LOAD_3D::copy(
            &b_tma_desc,
            &tma_B_barriers[0],
            static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL),
            smem_B,
            0,
            cta_n,
            0);
    }

    for (int k_abs = 0, iter = 0; k_abs < K; k_abs += K_CHUNK * 2, ++iter) {
        const int stage = iter % MTILE_TMA_STAGES;
        const int phase = (iter / MTILE_TMA_STAGES) & 1;
        uint8_t* smem_B_stage = smem_B + stage * smem_B_stage_bytes;
#if MTILE_CONSUMER_WAIT_BARRIER
        cute::wait_barrier(tma_B_barriers[stage], phase);
        cta3d_tma_shared_fence();
#else
        if (threadIdx.x == 0) {
            cute::wait_barrier(tma_B_barriers[stage], phase);
        }
        __syncthreads();
#endif

        const int next_k = k_abs + K_CHUNK * 2;
#if !MTILE_POST_K_SYNC
        if (next_k < K && (iter + 1) >= MTILE_TMA_STAGES) {
            __syncthreads();
        }
#endif
        if (threadIdx.x == 0 && next_k < K) {
            const int next_stage = (iter + 1) % MTILE_TMA_STAGES;
            cute::set_barrier_transaction_bytes(
                tma_B_barriers[next_stage], smem_B_stage_bytes);
            cute::SM90_TMA_LOAD_3D::copy(
                &b_tma_desc,
                &tma_B_barriers[next_stage],
                static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL),
                smem_B + next_stage * smem_B_stage_bytes,
                next_k / 2,
                cta_n,
                0);
        }

        {
            const int kc = k_abs >> 6;
            const int m0 = t1;
            const int m1 = t1 + 8;
            const int safe_m1 = (m1 < M) ? m1 : 0;
            const uint8_t* ap0 = A_packed + (long)m0 * (K / 2) + k_abs / 2;
            const uint8_t* ap1 = A_packed + (long)safe_m1 * (K / 2) + k_abs / 2;
            const uint32_t a0 =
                (m0 < M) ? *reinterpret_cast<const uint32_t*>(ap0 + t0 * 4) : 0;
            const uint32_t a1 =
                (m1 < M) ? *reinterpret_cast<const uint32_t*>(ap1 + t0 * 4) : 0;
            const uint32_t a2 =
                (m0 < M) ? *reinterpret_cast<const uint32_t*>(ap0 + t0 * 4 + 16) : 0;
            const uint32_t a3 =
                (m1 < M) ? *reinterpret_cast<const uint32_t*>(ap1 + t0 * 4 + 16) : 0;
            const uint32_t sfa =
                load_mtile_sfa(SF_A, M, kc, K_chunks, lane_id);
#pragma unroll
            for (int ng = 0; ng < N_GROUPS_PER_WARP; ++ng) {
                const int n_group = warp_id * N_GROUPS_PER_WARP + ng;
                uint8_t* warp_B = smem_B_stage + n_group * 8 * 64;
                const uint32_t b0 = *reinterpret_cast<const uint32_t*>(
                    warp_B + cta3d_b_smem_offset(t1, t0 * 4));
                const uint32_t b1 = *reinterpret_cast<const uint32_t*>(
                    warp_B + cta3d_b_smem_offset(t1, t0 * 4 + 16));
                const int abs_n_sf = cta_n + n_group * 8 + t1;
                const uint32_t sfb = (abs_n_sf < N)
                    ? *reinterpret_cast<const uint32_t*>(
                        SF_B + cta3d_sf_128x4_offset(abs_n_sf, kc, K_chunks))
                    : 0x38383838u;
                mma_mxf4nvf4_k64_task38(
                    lo0[ng], lo1[ng], hi0[ng], hi1[ng],
                    a0, a1, a2, a3, b0, b1, sfa, sfb);
            }
        }

        {
            const int kc = (k_abs >> 6) + 1;
            const int m0 = t1;
            const int m1 = t1 + 8;
            const int safe_m1 = (m1 < M) ? m1 : 0;
            const uint8_t* ap0 = A_packed + (long)m0 * (K / 2) + k_abs / 2 + 32;
            const uint8_t* ap1 = A_packed + (long)safe_m1 * (K / 2) + k_abs / 2 + 32;
            const uint32_t a0 =
                (m0 < M) ? *reinterpret_cast<const uint32_t*>(ap0 + t0 * 4) : 0;
            const uint32_t a1 =
                (m1 < M) ? *reinterpret_cast<const uint32_t*>(ap1 + t0 * 4) : 0;
            const uint32_t a2 =
                (m0 < M) ? *reinterpret_cast<const uint32_t*>(ap0 + t0 * 4 + 16) : 0;
            const uint32_t a3 =
                (m1 < M) ? *reinterpret_cast<const uint32_t*>(ap1 + t0 * 4 + 16) : 0;
            const int sfa_row = (M > 8)
                ? ((lane_id & 1) * 8 + t1)
                : (((lane_id & 1) == 0) ? t1 : 16 + t1);
            const uint32_t sfa = (sfa_row < M)
                ? *reinterpret_cast<const uint32_t*>(
                    SF_A + cta3d_sf_128x4_offset(sfa_row, kc, K_chunks))
                : 0x38383838u;
#pragma unroll
            for (int ng = 0; ng < N_GROUPS_PER_WARP; ++ng) {
                const int n_group = warp_id * N_GROUPS_PER_WARP + ng;
                uint8_t* warp_B = smem_B_stage + n_group * 8 * 64;
                const uint32_t b0 = *reinterpret_cast<const uint32_t*>(
                    warp_B + cta3d_b_smem_offset(t1, t0 * 4 + 32));
                const uint32_t b1 = *reinterpret_cast<const uint32_t*>(
                    warp_B + cta3d_b_smem_offset(t1, t0 * 4 + 48));
                const int abs_n_sf = cta_n + n_group * 8 + t1;
                const uint32_t sfb = (abs_n_sf < N)
                    ? *reinterpret_cast<const uint32_t*>(
                        SF_B + cta3d_sf_128x4_offset(abs_n_sf, kc, K_chunks))
                    : 0x38383838u;
                mma_mxf4nvf4_k64_task38(
                    lo0[ng], lo1[ng], hi0[ng], hi1[ng],
                    a0, a1, a2, a3, b0, b1, sfa, sfb);
            }
        }
#if MTILE_POST_K_SYNC
        __syncthreads();
#endif
    }

    const float alpha = *alpha_ptr;
    const int pair_col = (lane_id & 3) * 2;
    const int m0 = t1;
    const int m1 = t1 + 8;
#pragma unroll
    for (int ng = 0; ng < N_GROUPS_PER_WARP; ++ng) {
        const int n_group = warp_id * N_GROUPS_PER_WARP + ng;
        const int abs_n0 = cta_n + n_group * 8 + pair_col;
        const int abs_n1 = abs_n0 + 1;
        if (abs_n1 < N) {
            if (m0 < M) {
                D[(long)m0 * N + abs_n0] = __float2bfloat16(lo0[ng] * alpha);
                D[(long)m0 * N + abs_n1] = __float2bfloat16(lo1[ng] * alpha);
            }
            if (m1 < M) {
                D[(long)m1 * N + abs_n0] = __float2bfloat16(hi0[ng] * alpha);
                D[(long)m1 * N + abs_n1] = __float2bfloat16(hi1[ng] * alpha);
            }
        } else if (abs_n0 < N) {
            if (m0 < M) {
                D[(long)m0 * N + abs_n0] = __float2bfloat16(lo0[ng] * alpha);
            }
            if (m1 < M) {
                D[(long)m1 * N + abs_n0] = __float2bfloat16(hi0[ng] * alpha);
            }
        }
    }
}

template <int HYBRID_S, bool FULL_M16 = false>
__global__ void decode_gemv_nvfp4_mtile16_hybrid_tma_kernel(
    int M, int N, int K, int K_split,
    const uint8_t* __restrict__ A_packed,
    const uint8_t* __restrict__ SF_A,
    const uint8_t* __restrict__ SF_B,
    const float* __restrict__ alpha_ptr,
    __nv_bfloat16* __restrict__ D,
    CUTE_GRID_CONSTANT CUtensorMap const b_tma_desc)
{
    static_assert(HYBRID_S == 2 || HYBRID_S == 4,
                  "small-M hybrid TMA supports 2 or 4 in-CTA K splits");
    constexpr int WARPS_N =
        (MTILE_HYBRID_WARPS_N > 0) ? MTILE_HYBRID_WARPS_N : (8 / HYBRID_S);
    constexpr int N_GROUPS = MTILE_N_GROUPS;
    constexpr int CTA_N = WARPS_N * N_GROUPS * 8;
    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    if (warp_id >= HYBRID_S * WARPS_N) return;

    const int M_rows = FULL_M16 ? 16 : M;
    const int split_id = warp_id / WARPS_N;
    const int n_warp = warp_id - split_id * WARPS_N;
    const int t0 = lane_id & 3;
    const int t1 = lane_id >> 2;
    const int k_start = split_id * K_split;
    const int K_chunks = K / 64;
    const int cta_n = blockIdx.x * CTA_N;

    extern __shared__ uint8_t smem[];
    const int smem_barrier_bytes =
        (MTILE_HYBRID_TMA_STAGES * (int)sizeof(uint64_t) + 15) & ~15;
    uint64_t* tma_B_barriers = reinterpret_cast<uint64_t*>(smem);
    const int smem_B_off = (smem_barrier_bytes + 127) & ~127;
    const int smem_B_stage_bytes = HYBRID_S * CTA_N * 64;
    const int smem_B_bytes = MTILE_HYBRID_TMA_STAGES * smem_B_stage_bytes;
    const int partial_off = (smem_B_off + smem_B_bytes + 15) & ~15;
    uint8_t* smem_B = smem + smem_B_off;
    float* partials = reinterpret_cast<float*>(smem + partial_off);

    if (threadIdx.x < MTILE_HYBRID_TMA_STAGES) {
        cute::initialize_barrier(tma_B_barriers[threadIdx.x], 1);
    }
    __syncthreads();
    if (warp_id == 0 && cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&b_tma_desc);
    }
    __syncthreads();

    float lo0[N_GROUPS], lo1[N_GROUPS], hi0[N_GROUPS], hi1[N_GROUPS];
#pragma unroll
    for (int ng = 0; ng < N_GROUPS; ++ng) {
        lo0[ng] = 0.f;
        lo1[ng] = 0.f;
        hi0[ng] = 0.f;
        hi1[ng] = 0.f;
    }

    if (threadIdx.x == 0) {
        cute::set_barrier_transaction_bytes(
            tma_B_barriers[0], smem_B_stage_bytes);
        cute::SM90_TMA_LOAD_3D::copy(
            &b_tma_desc,
            &tma_B_barriers[0],
            static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL),
            smem_B,
            0,
            cta_n,
            0);
    }

    for (int k_local = 0, iter = 0; k_local < K_split;
         k_local += K_CHUNK * 2, ++iter) {
        const int k_abs = k_start + k_local;
        const int stage = iter % MTILE_HYBRID_TMA_STAGES;
        const int phase = (iter / MTILE_HYBRID_TMA_STAGES) & 1;
        uint8_t* smem_B_stage = smem_B + stage * smem_B_stage_bytes;
#if MTILE_CONSUMER_WAIT_BARRIER
        cute::wait_barrier(tma_B_barriers[stage], phase);
        cta3d_tma_shared_fence();
#else
        if (threadIdx.x == 0) {
            cute::wait_barrier(tma_B_barriers[stage], phase);
        }
        __syncthreads();
#endif

        const int next_k_local = k_local + K_CHUNK * 2;
#if !MTILE_POST_K_SYNC
        if (next_k_local < K_split && (iter + 1) >= MTILE_HYBRID_TMA_STAGES) {
            __syncthreads();
        }
#endif
        if (threadIdx.x == 0 && next_k_local < K_split) {
            const int next_stage = (iter + 1) % MTILE_HYBRID_TMA_STAGES;
            cute::set_barrier_transaction_bytes(
                tma_B_barriers[next_stage], smem_B_stage_bytes);
            cute::SM90_TMA_LOAD_3D::copy(
                &b_tma_desc,
                &tma_B_barriers[next_stage],
                static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL),
                smem_B + next_stage * smem_B_stage_bytes,
                next_k_local / 2,
                cta_n,
                0);
        }

        {
            const int kc = k_abs >> 6;
            const int m0 = t1;
            const int m1 = t1 + 8;
            const uint8_t* ap0 = A_packed + (long)m0 * (K / 2) + k_abs / 2;
            uint32_t a0;
            uint32_t a1;
            uint32_t a2;
            uint32_t a3;
            uint32_t sfa;
            if constexpr (FULL_M16) {
                const uint8_t* ap1 =
                    A_packed + (long)m1 * (K / 2) + k_abs / 2;
                a0 = *reinterpret_cast<const uint32_t*>(ap0 + t0 * 4);
                a1 = *reinterpret_cast<const uint32_t*>(ap1 + t0 * 4);
                a2 = *reinterpret_cast<const uint32_t*>(ap0 + t0 * 4 + 16);
                a3 = *reinterpret_cast<const uint32_t*>(ap1 + t0 * 4 + 16);
                sfa = load_m16_sfa(SF_A, kc, K_chunks, lane_id);
            } else {
                const int safe_m1 = (m1 < M) ? m1 : 0;
                const uint8_t* ap1 =
                    A_packed + (long)safe_m1 * (K / 2) + k_abs / 2;
                a0 = (m0 < M)
                    ? *reinterpret_cast<const uint32_t*>(ap0 + t0 * 4)
                    : 0;
                a1 = (m1 < M)
                    ? *reinterpret_cast<const uint32_t*>(ap1 + t0 * 4)
                    : 0;
                a2 = (m0 < M)
                    ? *reinterpret_cast<const uint32_t*>(ap0 + t0 * 4 + 16)
                    : 0;
                a3 = (m1 < M)
                    ? *reinterpret_cast<const uint32_t*>(ap1 + t0 * 4 + 16)
                    : 0;
                sfa = load_mtile_sfa(SF_A, M, kc, K_chunks, lane_id);
            }
#pragma unroll
            for (int ng = 0; ng < N_GROUPS; ++ng) {
                const int n_group = n_warp * N_GROUPS + ng;
                uint8_t* warp_B =
                    smem_B_stage + (split_id * CTA_N + n_group * 8) * 64;
                const uint32_t b0 = *reinterpret_cast<const uint32_t*>(
                    warp_B + cta3d_b_smem_offset(t1, t0 * 4));
                const uint32_t b1 = *reinterpret_cast<const uint32_t*>(
                    warp_B + cta3d_b_smem_offset(t1, t0 * 4 + 16));
                const int abs_n_sf = cta_n + n_group * 8 + t1;
                const uint32_t sfb = (abs_n_sf < N)
                    ? *reinterpret_cast<const uint32_t*>(
                        SF_B + cta3d_sf_128x4_offset(abs_n_sf, kc, K_chunks))
                    : 0x38383838u;
                mma_mxf4nvf4_k64_task38(
                    lo0[ng], lo1[ng], hi0[ng], hi1[ng],
                    a0, a1, a2, a3, b0, b1, sfa, sfb);
            }
        }

        {
            const int kc = (k_abs >> 6) + 1;
            const int m0 = t1;
            const int m1 = t1 + 8;
            const uint8_t* ap0 =
                A_packed + (long)m0 * (K / 2) + k_abs / 2 + 32;
            uint32_t a0;
            uint32_t a1;
            uint32_t a2;
            uint32_t a3;
            uint32_t sfa;
            if constexpr (FULL_M16) {
                const uint8_t* ap1 =
                    A_packed + (long)m1 * (K / 2) + k_abs / 2 + 32;
                a0 = *reinterpret_cast<const uint32_t*>(ap0 + t0 * 4);
                a1 = *reinterpret_cast<const uint32_t*>(ap1 + t0 * 4);
                a2 = *reinterpret_cast<const uint32_t*>(ap0 + t0 * 4 + 16);
                a3 = *reinterpret_cast<const uint32_t*>(ap1 + t0 * 4 + 16);
                sfa = load_m16_sfa(SF_A, kc, K_chunks, lane_id);
            } else {
                const int safe_m1 = (m1 < M) ? m1 : 0;
                const uint8_t* ap1 =
                    A_packed + (long)safe_m1 * (K / 2) + k_abs / 2 + 32;
                a0 = (m0 < M)
                    ? *reinterpret_cast<const uint32_t*>(ap0 + t0 * 4)
                    : 0;
                a1 = (m1 < M)
                    ? *reinterpret_cast<const uint32_t*>(ap1 + t0 * 4)
                    : 0;
                a2 = (m0 < M)
                    ? *reinterpret_cast<const uint32_t*>(ap0 + t0 * 4 + 16)
                    : 0;
                a3 = (m1 < M)
                    ? *reinterpret_cast<const uint32_t*>(ap1 + t0 * 4 + 16)
                    : 0;
                sfa = load_mtile_sfa(SF_A, M, kc, K_chunks, lane_id);
            }
#pragma unroll
            for (int ng = 0; ng < N_GROUPS; ++ng) {
                const int n_group = n_warp * N_GROUPS + ng;
                uint8_t* warp_B =
                    smem_B_stage + (split_id * CTA_N + n_group * 8) * 64;
                const uint32_t b0 = *reinterpret_cast<const uint32_t*>(
                    warp_B + cta3d_b_smem_offset(t1, t0 * 4 + 32));
                const uint32_t b1 = *reinterpret_cast<const uint32_t*>(
                    warp_B + cta3d_b_smem_offset(t1, t0 * 4 + 48));
                const int abs_n_sf = cta_n + n_group * 8 + t1;
                const uint32_t sfb = (abs_n_sf < N)
                    ? *reinterpret_cast<const uint32_t*>(
                        SF_B + cta3d_sf_128x4_offset(abs_n_sf, kc, K_chunks))
                    : 0x38383838u;
                mma_mxf4nvf4_k64_task38(
                    lo0[ng], lo1[ng], hi0[ng], hi1[ng],
                    a0, a1, a2, a3, b0, b1, sfa, sfb);
            }
        }
#if MTILE_POST_K_SYNC
        __syncthreads();
#endif
    }

    const int pair_col = (lane_id & 3) * 2;
    const int m0 = t1;
    const int m1 = t1 + 8;
#if MTILE_HYBRID_PAIR_REDUCE
    if constexpr (HYBRID_S == 2) {
        float* ps = partials;
        if (split_id == 0) {
#pragma unroll
            for (int ng = 0; ng < N_GROUPS; ++ng) {
                const int n_group = n_warp * N_GROUPS + ng;
                const int local_n0 = n_group * 8 + pair_col;
                if (m0 < M) {
                    *reinterpret_cast<float2*>(ps + m0 * CTA_N + local_n0) =
                        make_float2(lo0[ng], lo1[ng]);
                }
                if (m1 < M) {
                    *reinterpret_cast<float2*>(ps + m1 * CTA_N + local_n0) =
                        make_float2(hi0[ng], hi1[ng]);
                }
            }
        }
        __syncthreads();

        if (split_id == 1) {
            const float alpha = *alpha_ptr;
#pragma unroll
            for (int ng = 0; ng < N_GROUPS; ++ng) {
                const int n_group = n_warp * N_GROUPS + ng;
                const int local_n0 = n_group * 8 + pair_col;
                const int local_n1 = local_n0 + 1;
                const int abs_n0 = cta_n + local_n0;
                const int abs_n1 = abs_n0 + 1;
                if (abs_n0 < N) {
                    if (m0 < M) {
                        const float2 p =
                            *reinterpret_cast<const float2*>(
                                ps + m0 * CTA_N + local_n0);
                        D[(long)m0 * N + abs_n0] =
                            __float2bfloat16((p.x + lo0[ng]) * alpha);
                        if (abs_n1 < N) {
                            D[(long)m0 * N + abs_n1] =
                                __float2bfloat16((p.y + lo1[ng]) * alpha);
                        }
                    }
                    if (m1 < M) {
                        const float2 p =
                            *reinterpret_cast<const float2*>(
                                ps + m1 * CTA_N + local_n0);
                        D[(long)m1 * N + abs_n0] =
                            __float2bfloat16((p.x + hi0[ng]) * alpha);
                        if (abs_n1 < N) {
                            D[(long)m1 * N + abs_n1] =
                                __float2bfloat16((p.y + hi1[ng]) * alpha);
                        }
                    }
                }
            }
        }
        return;
    }
#endif

#pragma unroll
    for (int ng = 0; ng < N_GROUPS; ++ng) {
        const int n_group = n_warp * N_GROUPS + ng;
        float* ps = partials + split_id * M_rows * CTA_N;
        const int local_n0 = n_group * 8 + pair_col;
        const int local_n1 = local_n0 + 1;
        if constexpr (FULL_M16) {
            *reinterpret_cast<float2*>(ps + m0 * CTA_N + local_n0) =
                make_float2(lo0[ng], lo1[ng]);
            *reinterpret_cast<float2*>(ps + m1 * CTA_N + local_n0) =
                make_float2(hi0[ng], hi1[ng]);
        } else {
            if (m0 < M) {
                *reinterpret_cast<float2*>(ps + m0 * CTA_N + local_n0) =
                    make_float2(lo0[ng], lo1[ng]);
            }
            if (m1 < M) {
                *reinterpret_cast<float2*>(ps + m1 * CTA_N + local_n0) =
                    make_float2(hi0[ng], hi1[ng]);
            }
        }
    }
    __syncthreads();

    const float alpha = *alpha_ptr;
    for (int idx = threadIdx.x; idx < M_rows * CTA_N; idx += blockDim.x) {
        const int local_m = idx / CTA_N;
        const int local_n = idx - local_m * CTA_N;
        const int abs_n = cta_n + local_n;
        if (abs_n < N) {
            float sum = 0.f;
#pragma unroll
            for (int s = 0; s < HYBRID_S; ++s) {
                sum += partials[(s * M_rows + local_m) * CTA_N + local_n];
            }
            D[(long)local_m * N + abs_n] = __float2bfloat16(sum * alpha);
        }
    }
}

#endif

#if USE_TMA_B
#if FIXED_C2_S8_TN8
template __global__ void decode_gemv_nvfp4_splitk_kernel_fused<8>(int,int,int,int,int,const uint8_t*,const uint8_t*,const uint8_t*,const uint8_t*,const float*,__nv_bfloat16*,CUTE_GRID_CONSTANT CUtensorMap const);
#else
template __global__ void decode_gemv_nvfp4_splitk_kernel_fused<8>(int,int,int,int,int,const uint8_t*,const uint8_t*,const uint8_t*,const uint8_t*,const float*,__nv_bfloat16*,CUTE_GRID_CONSTANT CUtensorMap const);
template __global__ void decode_gemv_nvfp4_splitk_kernel_fused<16>(int,int,int,int,int,const uint8_t*,const uint8_t*,const uint8_t*,const uint8_t*,const float*,__nv_bfloat16*,CUTE_GRID_CONSTANT CUtensorMap const);
template __global__ void decode_gemv_nvfp4_splitk_kernel_fused<32>(int,int,int,int,int,const uint8_t*,const uint8_t*,const uint8_t*,const uint8_t*,const float*,__nv_bfloat16*,CUTE_GRID_CONSTANT CUtensorMap const);
template __global__ void decode_gemv_nvfp4_splitk_kernel_fused<64>(int,int,int,int,int,const uint8_t*,const uint8_t*,const uint8_t*,const uint8_t*,const float*,__nv_bfloat16*,CUTE_GRID_CONSTANT CUtensorMap const);
#endif
#else
#if FIXED_C2_S8_TN8
template __global__ void decode_gemv_nvfp4_splitk_kernel_fused<8>(int,int,int,int,int,const uint8_t*,const uint8_t*,const uint8_t*,const uint8_t*,const float*,__nv_bfloat16*);
#else
template __global__ void decode_gemv_nvfp4_splitk_kernel_fused<8>(int,int,int,int,int,const uint8_t*,const uint8_t*,const uint8_t*,const uint8_t*,const float*,__nv_bfloat16*);
template __global__ void decode_gemv_nvfp4_splitk_kernel_fused<16>(int,int,int,int,int,const uint8_t*,const uint8_t*,const uint8_t*,const uint8_t*,const float*,__nv_bfloat16*);
template __global__ void decode_gemv_nvfp4_splitk_kernel_fused<32>(int,int,int,int,int,const uint8_t*,const uint8_t*,const uint8_t*,const uint8_t*,const float*,__nv_bfloat16*);
template __global__ void decode_gemv_nvfp4_splitk_kernel_fused<64>(int,int,int,int,int,const uint8_t*,const uint8_t*,const uint8_t*,const uint8_t*,const float*,__nv_bfloat16*);
#endif
#endif

#if USE_TMA_B && TMA_B_CTA_3D
static void launch_splitk_fused_mtile16_tma8(int M, int N, int K, int S,
    const uint8_t* A, const uint8_t* B,
    const uint8_t* SF_A, const uint8_t* SF_B,
    const float* alpha, __nv_bfloat16* D,
    cudaStream_t stream,
    CUtensorMap b_tma_desc);

static bool make_b_tma_desc(
    CUtensorMap* desc, const unsigned char* B, int N, int K, int S,
    int box_n);
#endif

template <int TILE_N_T>
static void launch_splitk_fused(int M, int N, int K, int S,
    const uint8_t* A, const uint8_t* B,
    const uint8_t* SF_A, const uint8_t* SF_B,
    const float* alpha, __nv_bfloat16* D,
    cudaStream_t stream
#if USE_TMA_B
    , CUtensorMap b_tma_desc
#endif
    )
{
#if FIXED_C2_S8_TN8
    (void)M; (void)N; (void)K; (void)S; (void)TILE_N_T;
    constexpr int num_warps = C2_FIXED_S;
    constexpr int BS = num_warps * 32;
#if USE_TMA_B
#if TMA_B_CTA_3D
    constexpr int smem_barrier_bytes = ((int)sizeof(uint64_t) + 15) & ~15;
#else
    constexpr int smem_barrier_bytes =
        (num_warps * TMA_B_PIPE_STAGES * (int)sizeof(uint64_t) + 15) & ~15;
#endif
    constexpr int smem_B_off = (smem_barrier_bytes + 127) & ~127;
#else
    constexpr int smem_B_off = 0;
#endif
    constexpr int smem_B_bytes = num_warps * TMA_B_PIPE_STAGES * 8 * 64;
    constexpr int partial_off = (smem_B_off + smem_B_bytes + 15) & ~15;
    constexpr int smem = partial_off + C2_FIXED_S * C2_FIXED_TILE_N * (int)sizeof(float);
    dim3 grid(C2_FIXED_N_TILES, 1);
    decode_gemv_nvfp4_splitk_kernel_fused<8><<<grid, BS, smem, stream>>>(
        C2_FIXED_M, C2_FIXED_N, C2_FIXED_K, C2_FIXED_S, C2_FIXED_K_SPLIT,
        A, B, SF_A, SF_B, alpha, D
#if USE_TMA_B
        , b_tma_desc
#endif
        );
#else
    constexpr int GROUPS = TILE_N_T / 8;
    const int num_warps = GROUPS * S;
    const int BS = num_warps * 32;
    if (TILE_N_T % 8 != 0 || BS > 1024) {
        return;
    }
    const int K_split = K / S;
    const int logical_grid_n = (N + TILE_N_T - 1) / TILE_N_T;
#if USE_TMA_B && TMA_B_CTA_3D
    if (TILE_N_T == 8 && M <= 16 && (M > 1 || N <= CTA3D_M1_MTILE_MAX_N)) {
        launch_splitk_fused_mtile16_tma8(
            M, N, K, S, A, B, SF_A, SF_B, alpha, D, stream, b_tma_desc);
        return;
    }
#endif
#if PERSISTENT_GRID_CTAS > 0
    const int grid_n = (logical_grid_n < PERSISTENT_GRID_CTAS)
        ? logical_grid_n
        : PERSISTENT_GRID_CTAS;
#else
    const int grid_n = logical_grid_n;
#endif
#if USE_TMA_B
#if TMA_B_CTA_3D
    const int smem_barrier_bytes = ((int)sizeof(uint64_t) + 15) & ~15;
#else
    const int smem_barrier_bytes =
        (num_warps * TMA_B_PIPE_STAGES * (int)sizeof(uint64_t) + 15) & ~15;
#endif
    const int smem_B_off = (smem_barrier_bytes + 127) & ~127;
#else
    const int smem_B_off = 0;
#endif
    const int smem_B_bytes = num_warps * TMA_B_PIPE_STAGES * 8 * 64;
    const int partial_off = (smem_B_off + smem_B_bytes + 15) & ~15;
    const int smem = partial_off + S * TILE_N_T * (int)sizeof(float);

    dim3 grid(grid_n, M);
    decode_gemv_nvfp4_splitk_kernel_fused<TILE_N_T><<<grid, BS, smem, stream>>>(
        M, N, K, S, K_split, A, B, SF_A, SF_B, alpha, D
#if USE_TMA_B
        , b_tma_desc
#endif
        );
#endif
}

#if USE_TMA_B && TMA_B_CTA_3D
static void launch_splitk_fused_mtile16_tma8(int M, int N, int K, int S,
    const uint8_t* A, const uint8_t* B,
    const uint8_t* SF_A, const uint8_t* SF_B,
    const float* alpha, __nv_bfloat16* D,
    cudaStream_t stream,
    CUtensorMap b_tma_desc)
{
    if (M < 1 || M > 16 || S <= 0) {
        return;
    }
    const int K_split = K / S;
    const int num_warps = S;
    const int BS = num_warps * 32;
    if (BS > 1024) {
        return;
    }
    constexpr int N_GROUPS = MTILE_N_GROUPS;
    constexpr int CTA_N = N_GROUPS * 8;
    constexpr int NWARP_CTA_N = 8 * MTILE_NWARP_N_GROUPS * 8;
    constexpr int HYBRID_S = MTILE_HYBRID_SPLITS;
    constexpr int HYBRID_WARPS_N =
        (HYBRID_S == 2 || HYBRID_S == 4)
            ? ((MTILE_HYBRID_WARPS_N > 0)
                   ? MTILE_HYBRID_WARPS_N
                   : (8 / HYBRID_S))
            : 0;
    constexpr int HYBRID_CTA_N =
        (HYBRID_S == 2 || HYBRID_S == 4)
            ? (HYBRID_WARPS_N * MTILE_N_GROUPS * 8)
            : 0;
    const int smem_barrier_bytes =
        (MTILE_TMA_STAGES * (int)sizeof(uint64_t) + 15) & ~15;
    const int smem_B_off = (smem_barrier_bytes + 127) & ~127;
    const int smem_B_bytes = MTILE_TMA_STAGES * N_GROUPS * S * 8 * 64;
    const int partial_off = (smem_B_off + smem_B_bytes + 15) & ~15;
    const int partial_rows = (M == 16) ? 16 : M;
    const int partial_bytes = S * partial_rows * CTA_N * (int)sizeof(float);
    // A-staging cp.async double-buffer aliases the partials region; total smem
    // is the max of the two disjoint-lifetime regions placed at partial_off.
    const int A_db_bytes = S * 2 * M * 64;
    const int tail_bytes =
        (partial_bytes > A_db_bytes) ? partial_bytes : A_db_bytes;
    const int smem = partial_off + ((tail_bytes + 15) & ~15);
    const int grid_n = (N + CTA_N - 1) / CTA_N;

    if (M >= MTILE_NWARP_MIN_M && M <= 16 && K % (K_CHUNK * 2) == 0) {
        CUtensorMap b_tma_full_desc;
        if (!make_b_tma_desc(&b_tma_full_desc, B, N, K, 1, NWARP_CTA_N)) {
            return;
        }
        const int nwarp_smem_B_bytes = MTILE_TMA_STAGES * NWARP_CTA_N * 64;
        const int nwarp_smem = smem_B_off + nwarp_smem_B_bytes;
        if (nwarp_smem > 48 * 1024) {
            cudaFuncSetAttribute(
                decode_gemv_nvfp4_mtile16_nwarp_tma_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                nwarp_smem);
        }
        dim3 nwarp_grid((N + NWARP_CTA_N - 1) / NWARP_CTA_N, 1);
        decode_gemv_nvfp4_mtile16_nwarp_tma_kernel<<<
            nwarp_grid, 8 * 32, nwarp_smem, stream>>>(
            M, N, K, A, SF_A, SF_B, alpha, D, b_tma_full_desc);
        return;
    }

#if (MTILE_HYBRID_SPLITS == 2) || (MTILE_HYBRID_SPLITS == 4)
    if (M >= MTILE_HYBRID_MIN_M && M <= 16 && N >= MTILE_HYBRID_MIN_N &&
        K % (MTILE_HYBRID_SPLITS * K_CHUNK * 2) == 0) {
        CUtensorMap b_tma_hybrid_desc;
        if (!make_b_tma_desc(
                &b_tma_hybrid_desc, B, N, K, MTILE_HYBRID_SPLITS,
                HYBRID_CTA_N)) {
            return;
        }
        const int hybrid_smem_B_bytes =
            MTILE_HYBRID_TMA_STAGES * MTILE_HYBRID_SPLITS * HYBRID_CTA_N * 64;
        const int hybrid_partial_off =
            (smem_B_off + hybrid_smem_B_bytes + 15) & ~15;
        const int hybrid_partial_slices =
            (MTILE_HYBRID_SPLITS == 2 && MTILE_HYBRID_PAIR_REDUCE)
                ? 1
                : MTILE_HYBRID_SPLITS;
        const int hybrid_smem =
            hybrid_partial_off +
            hybrid_partial_slices * M * HYBRID_CTA_N * (int)sizeof(float);
        if (hybrid_smem > 48 * 1024) {
            cudaFuncSetAttribute(
                decode_gemv_nvfp4_mtile16_hybrid_tma_kernel<
                    MTILE_HYBRID_SPLITS>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                hybrid_smem);
        }
        dim3 hybrid_grid((N + HYBRID_CTA_N - 1) / HYBRID_CTA_N, 1);
        if (M == 16) {
            decode_gemv_nvfp4_mtile16_hybrid_tma_kernel<
                MTILE_HYBRID_SPLITS, true><<<
                hybrid_grid, MTILE_HYBRID_SPLITS * HYBRID_WARPS_N * 32,
                hybrid_smem, stream>>>(
                M, N, K, K / MTILE_HYBRID_SPLITS, A, SF_A, SF_B, alpha, D,
                b_tma_hybrid_desc);
        } else {
            decode_gemv_nvfp4_mtile16_hybrid_tma_kernel<
                MTILE_HYBRID_SPLITS, false><<<
                hybrid_grid, MTILE_HYBRID_SPLITS * HYBRID_WARPS_N * 32,
                hybrid_smem, stream>>>(
                M, N, K, K / MTILE_HYBRID_SPLITS, A, SF_A, SF_B, alpha, D,
                b_tma_hybrid_desc);
        }
        return;
    }
#endif

    if (smem > 48 * 1024) {
        cudaFuncSetAttribute(
            decode_gemv_nvfp4_splitk_kernel_fused_mtile16_tma8_n4,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem);
        cudaFuncSetAttribute(
            decode_gemv_nvfp4_splitk_kernel_fused_m16_tma8_ng,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem);
#if MTILE_CROSS_TILE_PIPE
        cudaFuncSetAttribute(
            decode_gemv_nvfp4_splitk_kernel_fused_mtile16_tma8_n4_cte,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem);
        cudaFuncSetAttribute(
            decode_gemv_nvfp4_splitk_kernel_fused_m16_tma8_ng_cte,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem);
#endif
    }
    dim3 grid(grid_n, 1);
#if MTILE_CROSS_TILE_PIPE
    if (S != 8) {
        printf("[task41_cte] MTILE_CROSS_TILE_PIPE=1 requires S==8, got S=%d\n", S);
        return;
    }
    int num_sms_cte = 0;
    int cte_dev = 0;
    cudaGetDevice(&cte_dev);
    cudaDeviceGetAttribute(&num_sms_cte, cudaDevAttrMultiProcessorCount, cte_dev);
    if (num_sms_cte <= 0) num_sms_cte = 110;
    const int persistent_cap = 2 * num_sms_cte;
    const int num_tiles_cte = grid_n;
    int persistent_grid = num_tiles_cte < persistent_cap ? num_tiles_cte : persistent_cap;
    dim3 cte_grid(persistent_grid, 1);
    if (M == 16) {
        decode_gemv_nvfp4_splitk_kernel_fused_m16_tma8_ng_cte<<<cte_grid, BS, smem, stream>>>(
            N, K, S, K_split, num_tiles_cte, A, SF_A, SF_B, alpha, D, b_tma_desc);
    } else {
        decode_gemv_nvfp4_splitk_kernel_fused_mtile16_tma8_n4_cte<<<cte_grid, BS, smem, stream>>>(
            M, N, K, S, K_split, num_tiles_cte, A, B, SF_A, SF_B, alpha, D, b_tma_desc);
    }
#else
    if (M == 16) {
        decode_gemv_nvfp4_splitk_kernel_fused_m16_tma8_ng<<<grid, BS, smem, stream>>>(
            N, K, S, K_split, A, SF_A, SF_B, alpha, D, b_tma_desc);
    } else {
        decode_gemv_nvfp4_splitk_kernel_fused_mtile16_tma8_n4<<<grid, BS, smem, stream>>>(
            M, N, K, S, K_split, A, B, SF_A, SF_B, alpha, D, b_tma_desc);
    }
#endif
}
#endif

#if USE_TMA_B
static bool make_b_tma_desc(
    CUtensorMap* desc, const unsigned char* B, int N, int K, int S,
    int box_n = 8)
{
    static bool driver_initialized = false;
    if (!driver_initialized) {
        CUresult init_status = cuInit(0);
        if (init_status != CUDA_SUCCESS) {
            printf("[task38_tma_b] cuInit failed: %d\n", (int)init_status);
            return false;
        }
        driver_initialized = true;
    }

#if TMA_B_CTA_3D
    const int K_split = K / S;
    const cuuint64_t global_dims[3] = {
        static_cast<cuuint64_t>(K_split / 2),
        static_cast<cuuint64_t>(N),
        static_cast<cuuint64_t>(S),
    };
    const cuuint64_t global_strides[2] = {
        static_cast<cuuint64_t>(K / 2),
        static_cast<cuuint64_t>(K_split / 2),
    };
    const cuuint32_t box_dims[3] = {
        64u, static_cast<cuuint32_t>(box_n), static_cast<cuuint32_t>(S)};
    const cuuint32_t element_strides[3] = {1u, 1u, 1u};

    CUresult status = cuTensorMapEncodeTiled(
        desc,
        CU_TENSOR_MAP_DATA_TYPE_UINT8,
        3,
        const_cast<unsigned char*>(B),
        global_dims,
        global_strides,
        box_dims,
        element_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
#else
    (void)box_n;
    (void)S;
    const cuuint64_t global_dims[2] = {
        static_cast<cuuint64_t>(K / 2),
        static_cast<cuuint64_t>(N),
    };
    const cuuint64_t global_strides[1] = {
        static_cast<cuuint64_t>(K / 2),
    };
    const cuuint32_t box_dims[2] = {64u, 8u};
    const cuuint32_t element_strides[2] = {1u, 1u};

    CUresult status = cuTensorMapEncodeTiled(
        desc,
        CU_TENSOR_MAP_DATA_TYPE_UINT8,
        2,
        const_cast<unsigned char*>(B),
        global_dims,
        global_strides,
        box_dims,
        element_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
#endif
    if (status != CUDA_SUCCESS) {
        printf("[task38_tma_b] cuTensorMapEncodeTiled failed: %d N=%d K=%d B=%p\n",
            (int)status, N, K, B);
        return false;
    }
    return true;
}
#endif

extern "C" void atrex_nvfp4_cta3d_tma_splitk(
    int M, int N, int K, int tile_n, int S,
    const unsigned char* A, const unsigned char* B,
    const unsigned char* SF_A, const unsigned char* SF_B,
    const float* alpha, unsigned char* D,
    float* workspace,
    unsigned long long stream_ptr)
{
    (void)workspace;
    auto d = reinterpret_cast<__nv_bfloat16*>(D);
    auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
#if USE_TMA_B
    CUtensorMap b_tma_desc;
    const int tma_box_n =
        (tile_n == 8 && M <= 16 && (M > 1 || N <= CTA3D_M1_MTILE_MAX_N))
            ? (MTILE_N_GROUPS * 8) : 8;
    if (!make_b_tma_desc(&b_tma_desc, B, N, K, S, tma_box_n)) {
        return;
    }
#endif
#if FIXED_C2_S8_TN8
    (void)workspace;
    if (M != C2_FIXED_M || N != C2_FIXED_N ||
        K != C2_FIXED_K || tile_n != C2_FIXED_TILE_N || S != C2_FIXED_S) {
        return;
    }
    launch_splitk_fused<8>(M,N,K,S,A,B,SF_A,SF_B,alpha,d,stream
#if USE_TMA_B
        , b_tma_desc
#endif
        );
#else
    switch (tile_n) {
        case   8: launch_splitk_fused<8> (M,N,K,S,A,B,SF_A,SF_B,alpha,d,stream
#if USE_TMA_B
            , b_tma_desc
#endif
            ); break;
        case  16: launch_splitk_fused<16>(M,N,K,S,A,B,SF_A,SF_B,alpha,d,stream
#if USE_TMA_B
            , b_tma_desc
#endif
            ); break;
        case  32: launch_splitk_fused<32>(M,N,K,S,A,B,SF_A,SF_B,alpha,d,stream
#if USE_TMA_B
            , b_tma_desc
#endif
            ); break;
        case  64: launch_splitk_fused<64>(M,N,K,S,A,B,SF_A,SF_B,alpha,d,stream
#if USE_TMA_B
            , b_tma_desc
#endif
            ); break;
        default:  launch_splitk_fused<16>(M,N,K,S,A,B,SF_A,SF_B,alpha,d,stream
#if USE_TMA_B
            , b_tma_desc
#endif
            ); break;
    }
#endif
}

extern "C" void atrex_nvfp4_cta3d_tma_splitk_auto(
    int M, int N, int K, int S,
    const unsigned char* A, const unsigned char* B,
    const unsigned char* SF_A, const unsigned char* SF_B,
    const float* alpha, unsigned char* D,
    float* workspace,
    unsigned long long stream_ptr)
{
#if FIXED_C2_S8_TN8
    int tile_n = C2_FIXED_TILE_N;
#else
    int tile_n = (S >= 8) ? 16 : 32;
#endif
    atrex_nvfp4_cta3d_tma_splitk(M, N, K, tile_n, S, A, B, SF_A, SF_B, alpha, D, workspace, stream_ptr);
}

// Phase 1: Split-K partial sum kernel
// Grid: (ceil(N/TILE_N), S, M)
template <int TILE_N_T>
__global__ void decode_gemv_nvfp4_splitk_kernel(
    int M, int N, int K, int K_split,
    const uint8_t* __restrict__ A_packed,
    const uint8_t* __restrict__ B_packed,
    const uint8_t* __restrict__ SF_A,
    const uint8_t* __restrict__ SF_B,
    float* __restrict__ workspace)
{
    constexpr int BLOCK_SIZE_T = TILE_N_T * 4;
    const int cta_n    = blockIdx.x * TILE_N_T;
    const int split_id = blockIdx.y;
    const int m        = blockIdx.z;
    const int warp_id  = threadIdx.x / 32;
    const int lane_id  = threadIdx.x % 32;
    const int t0 = lane_id & 3;
    const int t1 = lane_id >> 2;
    if (m >= M) return;

    const int k_start = split_id * K_split;

    extern __shared__ uint8_t smem[];
    const int smem_A_bytes = K_split / 2;
    const int smem_B_off   = (smem_A_bytes + 15) & ~15;
    uint8_t* smem_A = smem;
    uint8_t* smem_B = smem + smem_B_off;

    const uint8_t* A_start = A_packed + (long)m * (K / 2) + k_start / 2;
    for (int i = threadIdx.x; i < smem_A_bytes; i += BLOCK_SIZE_T)
        smem_A[i] = A_start[i];
    __syncthreads();

    float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;

    const int abs_n_sf  = cta_n + warp_id * 8 + t1;
    const int K_chunks  = K / 64;

    const int load_row   = threadIdx.x >> 2;
    const int load_group = threadIdx.x & 3;

    for (int k_local = 0; k_local < K_split; k_local += K_CHUNK_SK * 2) {
        const int k_abs = k_start + k_local;

        if (load_row < TILE_N_T) {
            const int abs_n = cta_n + load_row;
            if (abs_n < N) {
                const uint8_t* src = B_packed
                    + (long)abs_n * (K / 2)
                    + k_abs / 2
                    + load_group * 16;
                *reinterpret_cast<uint4*>(smem_B + load_row * 64 + load_group * 16)
                    = *reinterpret_cast<const uint4*>(src);
            }
        }
        __syncthreads();

        // MMA 0: first K-chunk
        {
            const int kc = k_abs >> 6;
            const uint8_t* ap = smem_A + k_local / 2;
            uint32_t a0 = *reinterpret_cast<const uint32_t*>(ap + t0 * 4);
            uint32_t a1 = a0;
            uint32_t a2 = *reinterpret_cast<const uint32_t*>(ap + t0 * 4 + 16);
            uint32_t a3 = a2;

            const uint8_t* br = smem_B + (warp_id * 8 + t1) * 64;
            uint32_t b0 = *reinterpret_cast<const uint32_t*>(br + t0 * 4);
            uint32_t b1 = *reinterpret_cast<const uint32_t*>(br + t0 * 4 + 16);

            uint32_t sfa = *reinterpret_cast<const uint32_t*>(
                SF_A + sf_128x4_offset(m, kc, K_chunks));
            uint32_t sfb = (abs_n_sf < N)
                ? *reinterpret_cast<const uint32_t*>(
                    SF_B + sf_128x4_offset(abs_n_sf, kc, K_chunks))
                : 0x38383838u;

            asm volatile(
                "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X."
                "m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                "{%0, %1, %2, %3},"
                "{%4, %5, %6, %7},"
                "{%8, %9},"
                "{%0, %1, %2, %3},"
                "{%10},"
                "{%11, %12},"
                "{%13},"
                "{%14, %15};\n"
                : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                  "r"(b0), "r"(b1),
                  "r"(sfa), "h"((uint16_t)0), "h"((uint16_t)0),
                  "r"(sfb), "h"((uint16_t)0), "h"((uint16_t)0)
            );
        }

        // MMA 1: second K-chunk
        {
            const int kc = (k_abs >> 6) + 1;
            const uint8_t* ap = smem_A + k_local / 2 + 32;
            uint32_t a0 = *reinterpret_cast<const uint32_t*>(ap + t0 * 4);
            uint32_t a1 = a0;
            uint32_t a2 = *reinterpret_cast<const uint32_t*>(ap + t0 * 4 + 16);
            uint32_t a3 = a2;

            const uint8_t* br = smem_B + (warp_id * 8 + t1) * 64 + 32;
            uint32_t b0 = *reinterpret_cast<const uint32_t*>(br + t0 * 4);
            uint32_t b1 = *reinterpret_cast<const uint32_t*>(br + t0 * 4 + 16);

            uint32_t sfa = *reinterpret_cast<const uint32_t*>(
                SF_A + sf_128x4_offset(m, kc, K_chunks));
            uint32_t sfb = (abs_n_sf < N)
                ? *reinterpret_cast<const uint32_t*>(
                    SF_B + sf_128x4_offset(abs_n_sf, kc, K_chunks))
                : 0x38383838u;

            asm volatile(
                "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X."
                "m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                "{%0, %1, %2, %3},"
                "{%4, %5, %6, %7},"
                "{%8, %9},"
                "{%0, %1, %2, %3},"
                "{%10},"
                "{%11, %12},"
                "{%13},"
                "{%14, %15};\n"
                : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                  "r"(b0), "r"(b1),
                  "r"(sfa), "h"((uint16_t)0), "h"((uint16_t)0),
                  "r"(sfb), "h"((uint16_t)0), "h"((uint16_t)0)
            );
        }

        __syncthreads();

    }

    if (t1 == 0) {
        int n0 = cta_n + warp_id * 8 + t0 * 2;
        int n1 = n0 + 1;
        float* ws = workspace + ((long)m * gridDim.y + split_id) * N;
        if (n0 < N) ws[n0] = d0;
        if (n1 < N) ws[n1] = d1;
    }
}

// Phase 2: Reduce S partial sums -> bf16 output
__global__ void reduce_splitk_kernel(
    int M, int N, int S,
    const float* __restrict__ workspace,
    const float* __restrict__ alpha_ptr,
    __nv_bfloat16* __restrict__ D)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = M * N;
    if (idx >= total) return;
    const int m = idx / N;
    const int n = idx - m * N;

    float sum = 0.f;
    for (int s = 0; s < S; s++)
        sum += workspace[((long)m * S + s) * N + n];

    D[(long)m * N + n] = __float2bfloat16(sum * (*alpha_ptr));
}

// Explicit template instantiations
template __global__ void decode_gemv_nvfp4_splitk_kernel<8>(int,int,int,int,const uint8_t*,const uint8_t*,const uint8_t*,const uint8_t*,float*);
template __global__ void decode_gemv_nvfp4_splitk_kernel<16>(int,int,int,int,const uint8_t*,const uint8_t*,const uint8_t*,const uint8_t*,float*);
template __global__ void decode_gemv_nvfp4_splitk_kernel<32>(int,int,int,int,const uint8_t*,const uint8_t*,const uint8_t*,const uint8_t*,float*);
template __global__ void decode_gemv_nvfp4_splitk_kernel<64>(int,int,int,int,const uint8_t*,const uint8_t*,const uint8_t*,const uint8_t*,float*);
template __global__ void decode_gemv_nvfp4_splitk_kernel<128>(int,int,int,int,const uint8_t*,const uint8_t*,const uint8_t*,const uint8_t*,float*);
template __global__ void decode_gemv_nvfp4_splitk_kernel<256>(int,int,int,int,const uint8_t*,const uint8_t*,const uint8_t*,const uint8_t*,float*);

// Launch helpers
template <int TILE_N_T>
static void launch_splitk(int M, int N, int K, int S,
    const uint8_t* A, const uint8_t* B,
    const uint8_t* SF_A, const uint8_t* SF_B,
    float* workspace, const float* alpha, __nv_bfloat16* D,
    cudaStream_t stream)
{
    constexpr int BS = TILE_N_T * 4;
    int K_split = K / S;
    int grid_n  = (N + TILE_N_T - 1) / TILE_N_T;
    int smem    = ((K_split / 2 + 15) & ~15) + TILE_N_T * 64;

    dim3 grid(grid_n, S, M);
    decode_gemv_nvfp4_splitk_kernel<TILE_N_T><<<grid, BS, smem, stream>>>(
        M, N, K, K_split, A, B, SF_A, SF_B, workspace);

    int reduce_bs = 256;
    int reduce_grid = (M * N + reduce_bs - 1) / reduce_bs;
    reduce_splitk_kernel<<<reduce_grid, reduce_bs, 0, stream>>>(
        M, N, S, workspace, alpha, D);
}

// PyTorch entry point
void nvfp4_gemv_splitk(
    const torch::Tensor& A_packed,
    const torch::Tensor& B_packed,
    const torch::Tensor& SF_A,
    const torch::Tensor& SF_B,
    const torch::Tensor& alpha,
    torch::Tensor& output,
    torch::Tensor& workspace,
    int tile_n,
    int split_k)
{
    int M = A_packed.dim() == 1 ? 1 : A_packed.size(0);
    int N = B_packed.size(0);
    int K = B_packed.size(1) * 2;

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    const auto* a   = A_packed.data_ptr<uint8_t>();
    const auto* b   = B_packed.data_ptr<uint8_t>();
    const auto* sfa = SF_A.data_ptr<uint8_t>();
    const auto* sfb = SF_B.data_ptr<uint8_t>();
    const auto* alp = alpha.data_ptr<float>();
    auto* ws = workspace.data_ptr<float>();
    auto* d  = reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>());

    if (tile_n <= 0) {
        tile_n = (M == C2_TMA_M && N == C2_TMA_N && K == C2_TMA_K
                  && split_k == CTA3D_TMA_S)
            ? CTA3D_TMA_TILE_N
            : ((N >= 2048) ? 32 : 16);
    }

    if (M >= 1 && M <= 16 && tile_n == CTA3D_TMA_TILE_N
        && split_k == CTA3D_TMA_S && (N % 8) == 0
        && (K % (split_k * 128)) == 0) {
        atrex_nvfp4_cta3d_tma_splitk(
            M, N, K, tile_n, split_k, a, b, sfa, sfb, alp,
            reinterpret_cast<unsigned char*>(d), ws,
            reinterpret_cast<unsigned long long>(stream));
        return;
    }

    switch (tile_n) {
        case   8: launch_splitk<8>  (M,N,K,split_k,a,b,sfa,sfb,ws,alp,d,stream); break;
        case  16: launch_splitk<16> (M,N,K,split_k,a,b,sfa,sfb,ws,alp,d,stream); break;
        case  32: launch_splitk<32> (M,N,K,split_k,a,b,sfa,sfb,ws,alp,d,stream); break;
        case  64: launch_splitk<64> (M,N,K,split_k,a,b,sfa,sfb,ws,alp,d,stream); break;
        case 128: launch_splitk<128>(M,N,K,split_k,a,b,sfa,sfb,ws,alp,d,stream); break;
        case 256: launch_splitk<256>(M,N,K,split_k,a,b,sfa,sfb,ws,alp,d,stream); break;
        default:  launch_splitk<32> (M,N,K,split_k,a,b,sfa,sfb,ws,alp,d,stream); break;
    }
}
