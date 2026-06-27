// gpu-wiki archive note:
// TUNED FOR RTX PRO 5000 / SM120, C2 decode shape M=1,N=5120,K=17408.
// This is the omoExplore task38 CTA-3D TMA source snapshot. It differs from
// gemm_v3_splitk_sm120.cu by fusing Split-K reduction inside one CTA and by
// encoding split as a third TMA dimension for B. Use it as a scoped reference,
// not as a generic NVFP4 GEMM baseline.
// Related docs:
//   docs/ref-docs/nvidia/cuda/sm120/sm120-nvfp4-decode-gemm-production-lessons.md
//   docs/pitfalls/nvidia/cuda/sm120-nvfp4-decode-gemm-production-pitfalls.md
//
// task_38 candidate: fused intra-CTA split-K NVFP4 GEMM with direct A loads
// and shared-memory B swizzle.
//
// The original split-K path launches one CTA per (N tile, K split), writes
// FP32 partials to global workspace, then launches a reduce kernel. This
// candidate keeps the same warp-level MMA work but places all S K-split warp
// groups for one N tile inside a single CTA. The CTA reduces partials in
// shared memory and writes the final BF16 output from the same kernel.
//
// Difference from gemm_v3_splitk_fused_cta_direct_a.cu: B still occupies
// 8 x 64 bytes per warp in shared memory, but logical B words are mapped to a
// swizzled physical offset to reduce shared-memory bank conflicts without the
// larger smem footprint that made bpad80 slower.
//
// Intended task38 shape: M=1, N=5120, K=17408. Useful configurations:
//   S=4, TILE_N=32 -> 16 warps / CTA
//   S=8, TILE_N=16 -> 16 warps / CTA
//   S=8, TILE_N=32 -> 32 warps / CTA

#ifndef USE_TMA_B
#define USE_TMA_B 0
#endif

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

#if USE_TMA_B
#include <cuda.h>
#include "cute/arch/cluster_sm90.hpp"
#include "cute/arch/copy_sm90_desc.hpp"
#include "cute/arch/copy_sm90_tma.hpp"
#endif

#define K_CHUNK 64
#ifndef B_SWIZZLE_MODE
#define B_SWIZZLE_MODE 1
#endif
#ifndef ZERO_A3_ONLY
#define ZERO_A3_ONLY 0
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
#define TMA_B_CTA_3D 0
#endif
#ifndef MTILE_N_GROUPS
#define MTILE_N_GROUPS 4
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

__device__ __forceinline__ long sf_128x4_offset(int row, int k_chunk, int K_chunks)
{
    const int row_block = row >> 7;
    const int row_local = row & 127;
    return (long)(row_block * K_chunks + k_chunk) * 512
        + (row_local & 31) * 16
        + (row_local >> 5) * 4;
}

__device__ __forceinline__ int b_smem_offset(int row, int logical_byte)
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
    const uint8_t* sfa_iter = SF_A + sf_128x4_offset(m, sf_kc_start, K_chunks);
    const uint8_t* sfb_iter = SF_B + sf_128x4_offset(abs_n_sf, sf_kc_start, K_chunks);
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
            *reinterpret_cast<uint32_t*>(warp_B + b_smem_offset(t1, t0 * 16 + 0))
                = packed.x;
            *reinterpret_cast<uint32_t*>(warp_B + b_smem_offset(t1, t0 * 16 + 4))
                = packed.y;
            *reinterpret_cast<uint32_t*>(warp_B + b_smem_offset(t1, t0 * 16 + 8))
                = packed.z;
            *reinterpret_cast<uint32_t*>(warp_B + b_smem_offset(t1, t0 * 16 + 12))
                = packed.w;
#elif B_SWIZZLE_MODE == 4
            if ((t1 & 1) == 0) {
                *reinterpret_cast<uint4*>(warp_B + b_smem_offset(t1, t0 * 16))
                    = packed;
            } else {
                const uint2 lo = make_uint2(packed.x, packed.y);
                const uint2 hi = make_uint2(packed.z, packed.w);
                *reinterpret_cast<uint2*>(warp_B + b_smem_offset(t1, t0 * 16))
                    = lo;
                *reinterpret_cast<uint2*>(warp_B + b_smem_offset(t1, t0 * 16 + 8))
                    = hi;
            }
#elif B_SWIZZLE_MODE == 5
            const uint2 lo = make_uint2(packed.x, packed.y);
            const uint2 hi = make_uint2(packed.z, packed.w);
            *reinterpret_cast<uint2*>(warp_B + b_smem_offset(t1, t0 * 16))
                = lo;
            *reinterpret_cast<uint2*>(warp_B + b_smem_offset(t1, t0 * 16 + 8))
                = hi;
#else
            *reinterpret_cast<uint4*>(warp_B + b_smem_offset(t1, t0 * 16))
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
                warp_B_read + b_smem_offset(t1, t0 * 4));
            uint32_t b1 = *reinterpret_cast<const uint32_t*>(
                warp_B_read + b_smem_offset(t1, t0 * 4 + 16));

#if BCAST_SFA
            uint32_t sfa = 0;
            if (lane_id == 0) {
                sfa = *reinterpret_cast<const uint32_t*>(
                    SF_A + sf_128x4_offset(m, kc, K_chunks));
            }
            sfa = __shfl_sync(0xffffffffu, sfa, 0);
#else
#if RECUR_ADDR_C2 && !B_REG_PIPE && !B_L2_PREFETCH
            uint32_t sfa = *reinterpret_cast<const uint32_t*>(sfa_iter);
#else
            uint32_t sfa = *reinterpret_cast<const uint32_t*>(
                SF_A + sf_128x4_offset(m, kc, K_chunks));
#endif
#endif
#if FIXED_C2_S8_TN8
            uint32_t sfb = *reinterpret_cast<const uint32_t*>(
                SF_B + sf_128x4_offset(abs_n_sf, kc, K_chunks));
#elif RECUR_ADDR_C2 && !B_REG_PIPE && !B_L2_PREFETCH
            uint32_t sfb = *reinterpret_cast<const uint32_t*>(sfb_iter);
#else
            uint32_t sfb = (abs_n_sf < N)
                ? *reinterpret_cast<const uint32_t*>(
                    SF_B + sf_128x4_offset(abs_n_sf, kc, K_chunks))
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
                warp_B_read + b_smem_offset(t1, t0 * 4 + 32));
            uint32_t b1 = *reinterpret_cast<const uint32_t*>(
                warp_B_read + b_smem_offset(t1, t0 * 4 + 48));

#if BCAST_SFA
            uint32_t sfa = 0;
            if (lane_id == 0) {
                sfa = *reinterpret_cast<const uint32_t*>(
                    SF_A + sf_128x4_offset(m, kc, K_chunks));
            }
            sfa = __shfl_sync(0xffffffffu, sfa, 0);
#else
#if RECUR_ADDR_C2 && !B_REG_PIPE && !B_L2_PREFETCH
            uint32_t sfa = *reinterpret_cast<const uint32_t*>(sfa_iter + 512);
#else
            uint32_t sfa = *reinterpret_cast<const uint32_t*>(
                SF_A + sf_128x4_offset(m, kc, K_chunks));
#endif
#endif
#if FIXED_C2_S8_TN8
            uint32_t sfb = *reinterpret_cast<const uint32_t*>(
                SF_B + sf_128x4_offset(abs_n_sf, kc, K_chunks));
#elif RECUR_ADDR_C2 && !B_REG_PIPE && !B_L2_PREFETCH
            uint32_t sfb = *reinterpret_cast<const uint32_t*>(sfb_iter + 512);
#else
            uint32_t sfb = (abs_n_sf < N)
                ? *reinterpret_cast<const uint32_t*>(
                    SF_B + sf_128x4_offset(abs_n_sf, kc, K_chunks))
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
        __syncthreads();
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
                warp_B + b_smem_offset(t1, t0 * 4));
            uint32_t b1 = *reinterpret_cast<const uint32_t*>(
                warp_B + b_smem_offset(t1, t0 * 4 + 16));

            const int sfa_row = ((lane_id & 1) == 0) ? t1 : 16 + t1;
            uint32_t sfa = (sfa_row < M)
                ? *reinterpret_cast<const uint32_t*>(
                    SF_A + sf_128x4_offset(sfa_row, kc, K_chunks))
                : 0x38383838u;
            const int abs_n_sf = cta_n + t1;
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
                warp_B + b_smem_offset(t1, t0 * 4));
            uint32_t b1 = *reinterpret_cast<const uint32_t*>(
                warp_B + b_smem_offset(t1, t0 * 4 + 16));

            const int sfa_row = ((lane_id & 1) == 0) ? (t1 + 8) : 16 + t1;
            uint32_t sfa = (sfa_row < M)
                ? *reinterpret_cast<const uint32_t*>(
                    SF_A + sf_128x4_offset(sfa_row, kc, K_chunks))
                : 0x38383838u;
            const int abs_n_sf = cta_n + t1;
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
                warp_B + b_smem_offset(t1, t0 * 4 + 32));
            uint32_t b1 = *reinterpret_cast<const uint32_t*>(
                warp_B + b_smem_offset(t1, t0 * 4 + 48));

            const int sfa_row = ((lane_id & 1) == 0) ? t1 : 16 + t1;
            uint32_t sfa = (sfa_row < M)
                ? *reinterpret_cast<const uint32_t*>(
                    SF_A + sf_128x4_offset(sfa_row, kc, K_chunks))
                : 0x38383838u;
            const int abs_n_sf = cta_n + t1;
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
                warp_B + b_smem_offset(t1, t0 * 4 + 32));
            uint32_t b1 = *reinterpret_cast<const uint32_t*>(
                warp_B + b_smem_offset(t1, t0 * 4 + 48));

            const int sfa_row = ((lane_id & 1) == 0) ? (t1 + 8) : 16 + t1;
            uint32_t sfa = (sfa_row < M)
                ? *reinterpret_cast<const uint32_t*>(
                    SF_A + sf_128x4_offset(sfa_row, kc, K_chunks))
                : 0x38383838u;
            const int abs_n_sf = cta_n + t1;
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
                : "+f"(d2), "+f"(d3), "+f"(unused0), "+f"(unused1)
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                  "r"(b0), "r"(b1),
                  "r"(sfa), "h"((uint16_t)0), "h"((uint16_t)0),
                  "r"(sfb), "h"((uint16_t)0), "h"((uint16_t)0)
            );
        }

        __syncthreads();
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
    const int cta_n = blockIdx.x * CTA_N;

    extern __shared__ uint8_t smem[];
    const int smem_barrier_off = 0;
    const int smem_barrier_bytes = ((int)sizeof(uint64_t) + 15) & ~15;
    uint64_t* tma_B_barriers =
        reinterpret_cast<uint64_t*>(smem + smem_barrier_off);
    const int smem_B_off = (smem_barrier_off + smem_barrier_bytes + 127) & ~127;
    const int smem_B_bytes = N_GROUPS * S * 8 * 64;
    const int partial_off = (smem_B_off + smem_B_bytes + 15) & ~15;
    uint8_t* smem_B = smem + smem_B_off;
    float* partials = reinterpret_cast<float*>(smem + partial_off);

    if (threadIdx.x == 0) {
        cute::initialize_barrier(tma_B_barriers[0], 1);
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
    int tma_B_phase = 0;

    for (int k_local = 0; k_local < K_split; k_local += K_CHUNK * 2) {
        const int k_abs = k_start + k_local;
        if (threadIdx.x == 0) {
            cute::set_barrier_transaction_bytes(
                tma_B_barriers[0], N_GROUPS * S * 8 * 64);
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
            const int sfa_row = (M > 8)
                ? ((lane_id & 1) * 8 + t1)
                : (((lane_id & 1) == 0) ? t1 : 16 + t1);
            const uint32_t sfa = (sfa_row < M)
                ? *reinterpret_cast<const uint32_t*>(
                    SF_A + sf_128x4_offset(sfa_row, kc, K_chunks))
                : 0x38383838u;
#pragma unroll
            for (int ng = 0; ng < N_GROUPS; ++ng) {
                uint8_t* warp_B = smem_B + split_id * CTA_N * 64 + ng * 8 * 64;
                const uint32_t b0 = *reinterpret_cast<const uint32_t*>(
                    warp_B + b_smem_offset(t1, t0 * 4));
                const uint32_t b1 = *reinterpret_cast<const uint32_t*>(
                    warp_B + b_smem_offset(t1, t0 * 4 + 16));
                const int abs_n_sf = cta_n + ng * 8 + t1;
                const uint32_t sfb = (abs_n_sf < N)
                    ? *reinterpret_cast<const uint32_t*>(
                        SF_B + sf_128x4_offset(abs_n_sf, kc, K_chunks))
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
                    SF_A + sf_128x4_offset(sfa_row, kc, K_chunks))
                : 0x38383838u;
#pragma unroll
            for (int ng = 0; ng < N_GROUPS; ++ng) {
                uint8_t* warp_B = smem_B + split_id * CTA_N * 64 + ng * 8 * 64;
                const uint32_t b0 = *reinterpret_cast<const uint32_t*>(
                    warp_B + b_smem_offset(t1, t0 * 4));
                const uint32_t b1 = *reinterpret_cast<const uint32_t*>(
                    warp_B + b_smem_offset(t1, t0 * 4 + 16));
                const int abs_n_sf = cta_n + ng * 8 + t1;
                const uint32_t sfb = (abs_n_sf < N)
                    ? *reinterpret_cast<const uint32_t*>(
                        SF_B + sf_128x4_offset(abs_n_sf, kc, K_chunks))
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
                    SF_A + sf_128x4_offset(sfa_row, kc, K_chunks))
                : 0x38383838u;
#pragma unroll
            for (int ng = 0; ng < N_GROUPS; ++ng) {
                uint8_t* warp_B = smem_B + split_id * CTA_N * 64 + ng * 8 * 64;
                const uint32_t b0 = *reinterpret_cast<const uint32_t*>(
                    warp_B + b_smem_offset(t1, t0 * 4 + 32));
                const uint32_t b1 = *reinterpret_cast<const uint32_t*>(
                    warp_B + b_smem_offset(t1, t0 * 4 + 48));
                const int abs_n_sf = cta_n + ng * 8 + t1;
                const uint32_t sfb = (abs_n_sf < N)
                    ? *reinterpret_cast<const uint32_t*>(
                        SF_B + sf_128x4_offset(abs_n_sf, kc, K_chunks))
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
                    SF_A + sf_128x4_offset(sfa_row, kc, K_chunks))
                : 0x38383838u;
#pragma unroll
            for (int ng = 0; ng < N_GROUPS; ++ng) {
                uint8_t* warp_B = smem_B + split_id * CTA_N * 64 + ng * 8 * 64;
                const uint32_t b0 = *reinterpret_cast<const uint32_t*>(
                    warp_B + b_smem_offset(t1, t0 * 4 + 32));
                const uint32_t b1 = *reinterpret_cast<const uint32_t*>(
                    warp_B + b_smem_offset(t1, t0 * 4 + 48));
                const int abs_n_sf = cta_n + ng * 8 + t1;
                const uint32_t sfb = (abs_n_sf < N)
                    ? *reinterpret_cast<const uint32_t*>(
                        SF_B + sf_128x4_offset(abs_n_sf, kc, K_chunks))
                    : 0x38383838u;
                mma_mxf4nvf4_k64_task38(
                    hi0[ng], hi1[ng], unused0, unused1,
                    a0, a1, a2, a3, b0, b1, sfa, sfb);
            }
        }
        __syncthreads();
    }

    const int pair_col = (lane_id & 3) * 2;
    const int m0 = t1;
    const int m1 = t1 + 8;
#pragma unroll
    for (int ng = 0; ng < N_GROUPS; ++ng) {
        float* ps = partials + split_id * 16 * CTA_N;
        const int local_n0 = ng * 8 + pair_col;
        const int local_n1 = local_n0 + 1;
        if (m0 < M) {
            ps[m0 * CTA_N + local_n0] = lo0[ng];
            ps[m0 * CTA_N + local_n1] = lo1[ng];
        }
        if (m1 < M) {
            ps[m1 * CTA_N + local_n0] = hi0[ng];
            ps[m1 * CTA_N + local_n1] = hi1[ng];
        }
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < 16 * CTA_N; idx += blockDim.x) {
        const int local_m = idx / CTA_N;
        const int local_n = idx - local_m * CTA_N;
        const int abs_n = cta_n + local_n;
        if (local_m < M && abs_n < N) {
            float sum = 0.f;
            for (int s = 0; s < S; ++s) {
                sum += partials[(s * 16 + local_m) * CTA_N + local_n];
            }
            D[(long)local_m * N + abs_n] = __float2bfloat16(sum * (*alpha_ptr));
        }
    }
}

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
    const int smem_barrier_bytes = ((int)sizeof(uint64_t) + 15) & ~15;
    uint64_t* tma_B_barriers = reinterpret_cast<uint64_t*>(smem);
    const int smem_B_off = (smem_barrier_bytes + 127) & ~127;
    const int smem_B_bytes = N_GROUPS * S * 8 * 64;
    const int partial_off = (smem_B_off + smem_B_bytes + 15) & ~15;
    uint8_t* smem_B = smem + smem_B_off;
    float* partials = reinterpret_cast<float*>(smem + partial_off);

    if (threadIdx.x == 0) {
        cute::initialize_barrier(tma_B_barriers[0], 1);
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
    int tma_B_phase = 0;

    for (int k_local = 0; k_local < K_split; k_local += K_CHUNK * 2) {
        const int k_abs = k_start + k_local;
        if (threadIdx.x == 0) {
            cute::set_barrier_transaction_bytes(
                tma_B_barriers[0], N_GROUPS * S * 8 * 64);
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

        {
            const int kc = k_abs >> 6;
            const uint8_t* ap0 = A_packed + (long)t1 * (K / 2) + k_abs / 2;
            const uint8_t* ap1 = A_packed + (long)(t1 + 8) * (K / 2) + k_abs / 2;
            const uint32_t a0 = *reinterpret_cast<const uint32_t*>(ap0 + t0 * 4);
            const uint32_t a1 = *reinterpret_cast<const uint32_t*>(ap1 + t0 * 4);
            const uint32_t a2 = *reinterpret_cast<const uint32_t*>(ap0 + t0 * 4 + 16);
            const uint32_t a3 = *reinterpret_cast<const uint32_t*>(ap1 + t0 * 4 + 16);
            const int sfa_row = (lane_id & 1) * 8 + t1;
            const uint32_t sfa = *reinterpret_cast<const uint32_t*>(
                SF_A + sf_128x4_offset(sfa_row, kc, K_chunks));
#pragma unroll
            for (int ng = 0; ng < N_GROUPS; ++ng) {
                uint8_t* warp_B = smem_B + split_id * CTA_N * 64 + ng * 8 * 64;
                const uint32_t b0 = *reinterpret_cast<const uint32_t*>(
                    warp_B + b_smem_offset(t1, t0 * 4));
                const uint32_t b1 = *reinterpret_cast<const uint32_t*>(
                    warp_B + b_smem_offset(t1, t0 * 4 + 16));
                const int abs_n_sf = cta_n + ng * 8 + t1;
                const uint32_t sfb = (abs_n_sf < N)
                    ? *reinterpret_cast<const uint32_t*>(
                        SF_B + sf_128x4_offset(abs_n_sf, kc, K_chunks))
                    : 0x38383838u;
                mma_mxf4nvf4_k64_task38(
                    lo0[ng], lo1[ng], hi0[ng], hi1[ng],
                    a0, a1, a2, a3, b0, b1, sfa, sfb);
            }
        }

        {
            const int kc = (k_abs >> 6) + 1;
            const uint8_t* ap0 =
                A_packed + (long)t1 * (K / 2) + k_abs / 2 + 32;
            const uint8_t* ap1 =
                A_packed + (long)(t1 + 8) * (K / 2) + k_abs / 2 + 32;
            const uint32_t a0 = *reinterpret_cast<const uint32_t*>(ap0 + t0 * 4);
            const uint32_t a1 = *reinterpret_cast<const uint32_t*>(ap1 + t0 * 4);
            const uint32_t a2 = *reinterpret_cast<const uint32_t*>(ap0 + t0 * 4 + 16);
            const uint32_t a3 = *reinterpret_cast<const uint32_t*>(ap1 + t0 * 4 + 16);
            const int sfa_row = (lane_id & 1) * 8 + t1;
            const uint32_t sfa = *reinterpret_cast<const uint32_t*>(
                SF_A + sf_128x4_offset(sfa_row, kc, K_chunks));
#pragma unroll
            for (int ng = 0; ng < N_GROUPS; ++ng) {
                uint8_t* warp_B = smem_B + split_id * CTA_N * 64 + ng * 8 * 64;
                const uint32_t b0 = *reinterpret_cast<const uint32_t*>(
                    warp_B + b_smem_offset(t1, t0 * 4 + 32));
                const uint32_t b1 = *reinterpret_cast<const uint32_t*>(
                    warp_B + b_smem_offset(t1, t0 * 4 + 48));
                const int abs_n_sf = cta_n + ng * 8 + t1;
                const uint32_t sfb = (abs_n_sf < N)
                    ? *reinterpret_cast<const uint32_t*>(
                        SF_B + sf_128x4_offset(abs_n_sf, kc, K_chunks))
                    : 0x38383838u;
                mma_mxf4nvf4_k64_task38(
                    lo0[ng], lo1[ng], hi0[ng], hi1[ng],
                    a0, a1, a2, a3, b0, b1, sfa, sfb);
            }
        }
        __syncthreads();
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

    for (int idx = threadIdx.x; idx < 16 * CTA_N; idx += blockDim.x) {
        const int local_m = idx / CTA_N;
        const int local_n = idx - local_m * CTA_N;
        const int abs_n = cta_n + local_n;
        if (abs_n < N) {
            float sum = 0.f;
            for (int s = 0; s < S; ++s) {
                sum += partials[(s * 16 + local_m) * CTA_N + local_n];
            }
            D[(long)local_m * N + abs_n] = __float2bfloat16(sum * (*alpha_ptr));
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
    if (TILE_N_T == 8 && M > 1 && M <= 16) {
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
    if (M <= 1 || M > 16 || S <= 0) {
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
    const int smem_barrier_bytes = ((int)sizeof(uint64_t) + 15) & ~15;
    const int smem_B_off = (smem_barrier_bytes + 127) & ~127;
    const int smem_B_bytes = N_GROUPS * S * 8 * 64;
    const int partial_off = (smem_B_off + smem_B_bytes + 15) & ~15;
    const int smem = partial_off + S * 16 * CTA_N * (int)sizeof(float);
    const int grid_n = (N + CTA_N - 1) / CTA_N;

    if (smem > 48 * 1024) {
        cudaFuncSetAttribute(
            decode_gemv_nvfp4_splitk_kernel_fused_mtile16_tma8_n4,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem);
        cudaFuncSetAttribute(
            decode_gemv_nvfp4_splitk_kernel_fused_m16_tma8_ng,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem);
    }
    dim3 grid(grid_n, 1);
    if (M == 16) {
        decode_gemv_nvfp4_splitk_kernel_fused_m16_tma8_ng<<<grid, BS, smem, stream>>>(
            N, K, S, K_split, A, SF_A, SF_B, alpha, D, b_tma_desc);
    } else {
        decode_gemv_nvfp4_splitk_kernel_fused_mtile16_tma8_n4<<<grid, BS, smem, stream>>>(
            M, N, K, S, K_split, A, B, SF_A, SF_B, alpha, D, b_tma_desc);
    }
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

extern "C" void kernel_v3_splitk(
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
        (tile_n == 8 && M > 1 && M <= 16) ? (MTILE_N_GROUPS * 8) : 8;
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

extern "C" void kernel_v3_splitk_auto(
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
    kernel_v3_splitk(M, N, K, tile_n, S, A, B, SF_A, SF_B, alpha, D, workspace, stream_ptr);
}
