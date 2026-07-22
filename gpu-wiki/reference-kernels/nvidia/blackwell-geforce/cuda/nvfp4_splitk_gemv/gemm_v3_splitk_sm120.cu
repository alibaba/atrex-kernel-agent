// TUNED FOR NVIDIA RTX PRO 5000 Blackwell-GeForce / sm_120.
//
// Kernel: NVFP4 decode GEMV Split-K for Qwen3.5 MLP C2-like shapes.
// Framework: CUDA C++ + inline PTX mma.sync.aligned.kind::mxf4nvf4.
//
// Tuning deltas vs a generic NVFP4 GEMM:
// - M=1 decode-only GEMV path, not a prefill or batched GEMM kernel.
// - Split K into S slices to increase CTA count for small-N / long-K shapes.
// - Phase 1 writes FP32 partial sums to workspace; Phase 2 reduces to BF16.
// - C1-like large-N shapes should remain on CUTLASS; this kernel targets C2.
// - Consumes CUTLASS-swizzled SF_B layout used by the archived vLLM dispatch.
//
// Related docs:
// - docs/nvidia/blackwell-geforce/ref-docs/cuda/sm120-nvfp4-split-k-gemv-bf16-optimization.md
// - docs/nvidia/blackwell-geforce/pitfalls/cuda/nvfp4-split-k-gemv-pitfalls.md
//
// Split-K version of v3 decode-optimized NVFP4 GEMV kernel for SM120a.
//
// Splits K dimension into S segments, giving Sx more CTAs for better SM occupancy.
// Two-phase: splitk kernel produces f32 partials, then reduce kernel sums and writes bf16.
//
// C2 (N=5120, K=8704) bottleneck: only 160 CTAs for 110 SMs (1.45 CTA/SM).
// With S=4: 640 CTAs, 5.8 CTA/SM.
//
// Requires: K % (S * 128) == 0  (K_split must be divisible by K_CHUNK*2=128).

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

#define K_CHUNK 64

// Phase 1: Split-K partial sum kernel
// Grid: (ceil(N/TILE_N), S)
template <int TILE_N_T>
__global__ void decode_gemv_nvfp4_splitk_kernel(
    int N, int K, int K_split,
    const uint8_t* __restrict__ A_packed,
    const uint8_t* __restrict__ B_packed,
    const uint8_t* __restrict__ SF_A,
    const uint8_t* __restrict__ SF_B,
    float* __restrict__ workspace)
{
    constexpr int BLOCK_SIZE_T = TILE_N_T * 4;
    const int cta_n    = blockIdx.x * TILE_N_T;
    const int split_id = blockIdx.y;
    const int warp_id  = threadIdx.x / 32;
    const int lane_id  = threadIdx.x % 32;
    const int t0 = lane_id & 3;
    const int t1 = lane_id >> 2;

    const int k_start = split_id * K_split;

    extern __shared__ uint8_t smem[];
    const int smem_A_bytes = K_split / 2;
    const int smem_B_off   = (smem_A_bytes + 15) & ~15;
    uint8_t* smem_A = smem;
    uint8_t* smem_B = smem + smem_B_off;

    const uint8_t* A_start = A_packed + k_start / 2;
    for (int i = threadIdx.x; i < smem_A_bytes; i += BLOCK_SIZE_T)
        smem_A[i] = A_start[i];
    __syncthreads();

    float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;

    const int abs_n_sf  = cta_n + warp_id * 8 + t1;
    const int n_block   = abs_n_sf / 128;
    const int n_local   = abs_n_sf % 128;
    const int K_chunks  = K / 64;
    const int sfb_inner = (n_local >> 2) * 16 + (n_local & 3) * 4;

    const int load_row   = threadIdx.x >> 2;
    const int load_group = threadIdx.x & 3;

    for (int k_local = 0; k_local < K_split; k_local += K_CHUNK * 2) {
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

        // MMA 0
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

            uint32_t sfa = *reinterpret_cast<const uint32_t*>(SF_A + kc * 512);
            uint32_t sfb = (abs_n_sf < N)
                ? *reinterpret_cast<const uint32_t*>(
                    SF_B + (n_block * K_chunks + kc) * 512 + sfb_inner)
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

        // MMA 1
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

            uint32_t sfa = *reinterpret_cast<const uint32_t*>(SF_A + kc * 512);
            uint32_t sfb = (abs_n_sf < N)
                ? *reinterpret_cast<const uint32_t*>(
                    SF_B + (n_block * K_chunks + kc) * 512 + sfb_inner)
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
        float* ws = workspace + (long)split_id * N;
        if (n0 < N) ws[n0] = d0;
        if (n1 < N) ws[n1] = d1;
    }
}

// Phase 2: Reduce S partial sums to bf16 output
__global__ void reduce_splitk_kernel(
    int N, int S,
    const float* __restrict__ workspace,
    const float* __restrict__ alpha_ptr,
    __nv_bfloat16* __restrict__ D)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    float sum = 0.f;
    for (int s = 0; s < S; s++)
        sum += workspace[(long)s * N + n];

    D[n] = __float2bfloat16(sum * (*alpha_ptr));
}

// Explicit template instantiations
template __global__ void decode_gemv_nvfp4_splitk_kernel<8>(int,int,int,const uint8_t*,const uint8_t*,const uint8_t*,const uint8_t*,float*);
template __global__ void decode_gemv_nvfp4_splitk_kernel<16>(int,int,int,const uint8_t*,const uint8_t*,const uint8_t*,const uint8_t*,float*);
template __global__ void decode_gemv_nvfp4_splitk_kernel<32>(int,int,int,const uint8_t*,const uint8_t*,const uint8_t*,const uint8_t*,float*);
template __global__ void decode_gemv_nvfp4_splitk_kernel<64>(int,int,int,const uint8_t*,const uint8_t*,const uint8_t*,const uint8_t*,float*);
template __global__ void decode_gemv_nvfp4_splitk_kernel<128>(int,int,int,const uint8_t*,const uint8_t*,const uint8_t*,const uint8_t*,float*);
template __global__ void decode_gemv_nvfp4_splitk_kernel<256>(int,int,int,const uint8_t*,const uint8_t*,const uint8_t*,const uint8_t*,float*);

// Launch helpers
template <int TILE_N_T>
static void launch_splitk(int N, int K, int S,
    const uint8_t* A, const uint8_t* B,
    const uint8_t* SF_A, const uint8_t* SF_B,
    float* workspace, const float* alpha, __nv_bfloat16* D,
    cudaStream_t stream)
{
    constexpr int BS = TILE_N_T * 4;
    int K_split = K / S;
    int grid_n  = (N + TILE_N_T - 1) / TILE_N_T;
    int smem    = ((K_split / 2 + 15) & ~15) + TILE_N_T * 64;

    dim3 grid(grid_n, S);
    decode_gemv_nvfp4_splitk_kernel<TILE_N_T><<<grid, BS, smem, stream>>>(
        N, K, K_split, A, B, SF_A, SF_B, workspace);

    int reduce_bs = 256;
    int reduce_grid = (N + reduce_bs - 1) / reduce_bs;
    reduce_splitk_kernel<<<reduce_grid, reduce_bs, 0, stream>>>(
        N, S, workspace, alpha, D);
}

// C entry points
extern "C" void kernel_v3_splitk(
    int M, int N, int K, int tile_n, int S,
    const unsigned char* A, const unsigned char* B,
    const unsigned char* SF_A, const unsigned char* SF_B,
    const float* alpha, unsigned char* D,
    float* workspace,
    unsigned long long stream_ptr)
{
    auto d = reinterpret_cast<__nv_bfloat16*>(D);
    auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    switch (tile_n) {
        case   8: launch_splitk<8>  (N,K,S,A,B,SF_A,SF_B,workspace,alpha,d,stream); break;
        case  16: launch_splitk<16> (N,K,S,A,B,SF_A,SF_B,workspace,alpha,d,stream); break;
        case  32: launch_splitk<32> (N,K,S,A,B,SF_A,SF_B,workspace,alpha,d,stream); break;
        case  64: launch_splitk<64> (N,K,S,A,B,SF_A,SF_B,workspace,alpha,d,stream); break;
        case 128: launch_splitk<128>(N,K,S,A,B,SF_A,SF_B,workspace,alpha,d,stream); break;
        case 256: launch_splitk<256>(N,K,S,A,B,SF_A,SF_B,workspace,alpha,d,stream); break;
        default:  launch_splitk<32> (N,K,S,A,B,SF_A,SF_B,workspace,alpha,d,stream); break;
    }
}

extern "C" void kernel_v3_splitk_auto(
    int M, int N, int K, int S,
    const unsigned char* A, const unsigned char* B,
    const unsigned char* SF_A, const unsigned char* SF_B,
    const float* alpha, unsigned char* D,
    float* workspace,
    unsigned long long stream_ptr)
{
    int tile_n = (N >= 2048) ? 32 : 16;
    kernel_v3_splitk(M, N, K, tile_n, S, A, B, SF_A, SF_B, alpha, D, workspace, stream_ptr);
}
