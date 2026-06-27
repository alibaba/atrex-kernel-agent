"""
V7: Adaptive single-CTA / 2-kernel multi-CTA with contention-free scatter.

Profile-driven optimization based on ncu evidence from V6 at T=6144:
- V6 scatter_kernel No Eligible Warps 90.28% → global atomicAdd contention
- Root cause: 8 CTAs x 1024 threads all doing global atomicAdd on 256 expert offsets

V7 fix: pre-compute per-CTA expert offsets in histogram kernel, scatter uses
smem-only atomics + a single per-CTA base offset add.

Algorithm:
  1. histogram_kernel_v7 (same as V6) + store per-CTA expert counts in
     global_buf[258 + bid * 256 + e].
  2. Last-arriving CTA additionally computes per-CTA cumulative offsets in
     parallel (256 threads x num_blocks CTAs).
  3. scatter_kernel_v7 loads per-CTA base offsets into smem, runs smem
     atomicAdd for local_rank, then global_rank = base + local_rank.

global_buf layout:
  [0..255]                            expert counts (then cumulative offsets)
  [256]                               arrival counter
  [257]                               num_tokens
  [258..258 + num_blocks * 256 - 1]   per-CTA cumulative base offsets
"""

import argparse
import torch
import vllm_stub  # noqa: F401  registers vllm._custom_ops shim before next import
from torch.utils.cpp_extension import load_inline
from vllm._custom_ops import get_cutlass_moe_mm_data, shuffle_rows

NUM_EXPERTS = 256
TOPK = 8

CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

static constexpr int NUM_EXPERTS = 256;

// ========== Single-CTA kernel (small/medium T, same as V3/V6) ==========
__launch_bounds__(1024)
__global__ void fused_moe_data_single_cta(
    const int32_t* __restrict__ topk_ids,
    int32_t* __restrict__ expert_offsets,
    int32_t* __restrict__ problem_sizes1,
    int32_t* __restrict__ problem_sizes2,
    int32_t* __restrict__ a_map,
    int32_t* __restrict__ c_map,
    int32_t* __restrict__ blockscale_offsets,
    const int topk_length,
    const int topk,
    const int n,
    const int k)
{
    __shared__ int32_t cnts[NUM_EXPERTS];

    const int tid = threadIdx.x;
    const int T = blockDim.x;

    for (int e = tid; e < NUM_EXPERTS; e += T)
        cnts[e] = 0;
    __syncthreads();

    for (int i = tid; i < topk_length; i += T) {
        int eid = topk_ids[i];
        if (eid >= 0)
            atomicAdd(&cnts[eid], 1);
    }
    __syncthreads();

    __shared__ int32_t num_tokens;
    if (tid == 0) {
        int tot = 0;
        int tot_round = 0;
        int n1 = 2 * n;
        expert_offsets[0] = 0;
        blockscale_offsets[0] = 0;

        for (int i = 0; i < NUM_EXPERTS; i++) {
            int count = cnts[i];
            problem_sizes1[i * 3]     = count;
            problem_sizes1[i * 3 + 1] = n1;
            problem_sizes1[i * 3 + 2] = k;
            problem_sizes2[i * 3]     = count;
            problem_sizes2[i * 3 + 1] = k;
            problem_sizes2[i * 3 + 2] = n;
            cnts[i] = tot;
            tot += count;
            expert_offsets[i + 1] = tot;
            tot_round += (count + 127) / 128 * 128;
            blockscale_offsets[i + 1] = tot_round;
        }
        num_tokens = tot;
    }
    __syncthreads();

    int ntok = num_tokens;
    for (int i = tid; i < topk_length; i += T) {
        int eid = topk_ids[i];
        if (eid == -1) {
            c_map[i] = ntok;
        } else {
            int rank = atomicAdd(&cnts[eid], 1);
            a_map[rank] = i / topk;
            c_map[i] = rank;
        }
    }
}

// ========== Kernel 1: V9 multi-CTA histogram + per-CTA offsets ==========
// V9 change: 4-way bank-replicated local_cnts to cut INTER-WARP smem-atomic
// contention. Each warp picks bank=warpid&3 and atomicAdd's into its private
// slice; a merge phase then reduces 4 banks into the original 1-D layout so
// the rest of the kernel (per-CTA store, global merge, last-CTA Phase A/B)
// is unchanged. V8 evidence (profiles/v8/SUMMARY.md) showed bank conflicts
// are inter-warp (32 warps hitting the same hot expert), not intra-warp,
// which is exactly what replication targets.
__launch_bounds__(1024)
__global__ void fused_moe_histogram_kernel_v7(
    const int32_t* __restrict__ topk_ids,
    int32_t* __restrict__ expert_offsets,
    int32_t* __restrict__ problem_sizes1,
    int32_t* __restrict__ problem_sizes2,
    int32_t* __restrict__ blockscale_offsets,
    int32_t* __restrict__ global_buf,
    const int topk_length,
    const int n,
    const int k,
    const int num_blocks)
{
    constexpr int NUM_BANKS = 4;
    __shared__ int32_t local_cnts_rep[NUM_BANKS][NUM_EXPERTS];  // 4 KB
    __shared__ int32_t local_cnts[NUM_EXPERTS];                 // 1 KB (merged + Phase A/B arena)
    __shared__ bool is_last;

    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int T = blockDim.x;
    const int warpid = tid >> 5;
    const int bank = warpid & (NUM_BANKS - 1);

    // Zero replicated histogram (NUM_BANKS * NUM_EXPERTS slots)
    for (int idx = tid; idx < NUM_BANKS * NUM_EXPERTS; idx += T) {
        int b = idx >> 8;            // / NUM_EXPERTS
        int e = idx & (NUM_EXPERTS - 1);
        local_cnts_rep[b][e] = 0;
    }
    __syncthreads();

    // Local histogram via smem atomics into per-warp-bank slice. Inter-warp
    // contention on hot experts drops by ~NUM_BANKS× because 32 warps now
    // spread across 4 banks (8 warps per bank, vs all 32 on one counter).
    int chunk_size = (topk_length + num_blocks - 1) / num_blocks;
    int chunk_start = bid * chunk_size;
    int chunk_end = min(chunk_start + chunk_size, topk_length);

    for (int i = chunk_start + tid; i < chunk_end; i += T) {
        int eid = topk_ids[i];
        if (eid >= 0)
            atomicAdd(&local_cnts_rep[bank][eid], 1);
    }
    __syncthreads();

    // Merge NUM_BANKS replicas into the 1-D local_cnts[e] used downstream.
    // 256 experts × NUM_BANKS adds; one thread per expert is sufficient.
    for (int e = tid; e < NUM_EXPERTS; e += T) {
        int s = 0;
        #pragma unroll
        for (int b = 0; b < NUM_BANKS; b++) s += local_cnts_rep[b][e];
        local_cnts[e] = s;
    }
    __syncthreads();

    // Store per-CTA counts directly (no atomic — each (bid, e) slot is unique)
    int32_t* cta_counts = global_buf + 258 + bid * NUM_EXPERTS;
    for (int e = tid; e < NUM_EXPERTS; e += T) {
        cta_counts[e] = local_cnts[e];
    }

    // Merge local histogram into global expert totals
    for (int e = tid; e < NUM_EXPERTS; e += T) {
        if (local_cnts[e] > 0)
            atomicAdd(&global_buf[e], local_cnts[e]);
    }
    __threadfence();

    // Last-block-arrives barrier
    if (tid == 0) {
        int arrived = atomicAdd(&global_buf[256], 1);
        is_last = (arrived == num_blocks - 1);
    }
    __syncthreads();

    if (is_last) {
        // Phase A: thread 0 does serial prefix sum + writes problem sizes /
        // expert_offsets / blockscale_offsets. Reuse local_cnts as the
        // shared "prefix[e]" buffer for phase B.
        if (tid == 0) {
            int tot = 0;
            int tot_round = 0;
            int n1 = 2 * n;
            expert_offsets[0] = 0;
            blockscale_offsets[0] = 0;

            for (int i = 0; i < NUM_EXPERTS; i++) {
                int count = global_buf[i];
                problem_sizes1[i * 3]     = count;
                problem_sizes1[i * 3 + 1] = n1;
                problem_sizes1[i * 3 + 2] = k;
                problem_sizes2[i * 3]     = count;
                problem_sizes2[i * 3 + 1] = k;
                problem_sizes2[i * 3 + 2] = n;
                local_cnts[i] = tot;        // prefix base for expert i (used by Phase B)
                tot += count;
                expert_offsets[i + 1] = tot;
                tot_round += (count + 127) / 128 * 128;
                blockscale_offsets[i + 1] = tot_round;
            }
            global_buf[257] = tot;
        }
        __syncthreads();

        // Phase B: 256 threads, one per expert, scan across num_blocks CTAs
        // turning per-CTA counts into cumulative per-CTA base offsets in-place.
        if (tid < NUM_EXPERTS) {
            int e = tid;
            int base = local_cnts[e];
            for (int b = 0; b < num_blocks; b++) {
                int slot = 258 + b * NUM_EXPERTS + e;
                int old = global_buf[slot];
                global_buf[slot] = base;
                base += old;
            }
        }
    }
}

// ========== Kernel 2: V7 multi-CTA scatter (contention-free) ==========
// global_buf is read-only here: each CTA loads its per-CTA base offsets into
// smem and uses smem atomicAdd for the local rank.
__launch_bounds__(1024)
__global__ void fused_moe_scatter_kernel_v7(
    const int32_t* __restrict__ topk_ids,
    int32_t* __restrict__ a_map,
    int32_t* __restrict__ c_map,
    const int32_t* __restrict__ global_buf,
    const int topk_length,
    const int topk,
    const int num_blocks)
{
    __shared__ int32_t cta_base_smem[NUM_EXPERTS];
    __shared__ int32_t local_cnts[NUM_EXPERTS];

    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int T = blockDim.x;
    const int ntok = global_buf[257];

    const int32_t* cta_base_g = global_buf + 258 + bid * NUM_EXPERTS;
    for (int e = tid; e < NUM_EXPERTS; e += T) {
        cta_base_smem[e] = cta_base_g[e];
        local_cnts[e]   = 0;
    }
    __syncthreads();

    int chunk_size = (topk_length + num_blocks - 1) / num_blocks;
    int chunk_start = bid * chunk_size;
    int chunk_end = min(chunk_start + chunk_size, topk_length);

    for (int i = chunk_start + tid; i < chunk_end; i += T) {
        int eid = topk_ids[i];
        if (eid == -1) {
            c_map[i] = ntok;
        } else {
            int local_rank = atomicAdd(&local_cnts[eid], 1);
            int global_rank = cta_base_smem[eid] + local_rank;
            a_map[global_rank] = i / topk;
            c_map[i] = global_rank;
        }
    }
}

void launch_fused_moe_data_v7(
    torch::Tensor topk_ids,
    torch::Tensor expert_offsets,
    torch::Tensor problem_sizes1,
    torch::Tensor problem_sizes2,
    torch::Tensor a_map,
    torch::Tensor c_map,
    int64_t num_experts,
    int64_t n,
    int64_t k,
    torch::Tensor blockscale_offsets,
    bool is_gated,
    int64_t num_threads,
    torch::Tensor global_buf,
    int64_t multi_cta_threshold,
    int64_t num_blocks)
{
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    int topk_length = topk_ids.numel();
    int topk = topk_ids.size(1);

    if (topk_length <= multi_cta_threshold) {
        fused_moe_data_single_cta<<<1, num_threads, 0, stream>>>(
            topk_ids.data_ptr<int32_t>(),
            expert_offsets.data_ptr<int32_t>(),
            problem_sizes1.data_ptr<int32_t>(),
            problem_sizes2.data_ptr<int32_t>(),
            a_map.data_ptr<int32_t>(),
            c_map.data_ptr<int32_t>(),
            blockscale_offsets.data_ptr<int32_t>(),
            topk_length, topk, (int)n, (int)k);
    } else {
        int nb = (int)num_blocks;

        // Only zero the first 258 ints. The per-CTA region [258..258+nb*256-1]
        // is fully overwritten by each CTA in fused_moe_histogram_kernel_v7.
        cudaMemsetAsync(global_buf.data_ptr<int32_t>(), 0,
                        258 * sizeof(int32_t), stream);

        fused_moe_histogram_kernel_v7<<<nb, num_threads, 0, stream>>>(
            topk_ids.data_ptr<int32_t>(),
            expert_offsets.data_ptr<int32_t>(),
            problem_sizes1.data_ptr<int32_t>(),
            problem_sizes2.data_ptr<int32_t>(),
            blockscale_offsets.data_ptr<int32_t>(),
            global_buf.data_ptr<int32_t>(),
            topk_length, (int)n, (int)k, nb);

        fused_moe_scatter_kernel_v7<<<nb, num_threads, 0, stream>>>(
            topk_ids.data_ptr<int32_t>(),
            a_map.data_ptr<int32_t>(),
            c_map.data_ptr<int32_t>(),
            global_buf.data_ptr<int32_t>(),
            topk_length, topk, nb);
    }
}
"""

CPP_SRC = r"""
#include <torch/extension.h>

void launch_fused_moe_data_v7(
    torch::Tensor topk_ids,
    torch::Tensor expert_offsets,
    torch::Tensor problem_sizes1,
    torch::Tensor problem_sizes2,
    torch::Tensor a_map,
    torch::Tensor c_map,
    int64_t num_experts,
    int64_t n,
    int64_t k,
    torch::Tensor blockscale_offsets,
    bool is_gated,
    int64_t num_threads,
    torch::Tensor global_buf,
    int64_t multi_cta_threshold,
    int64_t num_blocks);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fused_moe_data", &launch_fused_moe_data_v7);
}
"""


def build_ext():
    print("JIT compiling V7 CUDA kernel...")
    ext = load_inline(
        name="moe_data_v7",
        cpp_sources=CPP_SRC,
        cuda_sources=CUDA_SRC,
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False,
    )
    print("V7 compilation done.")
    return ext


def make_inputs(T, E=NUM_EXPERTS, K=TOPK, dev="cuda:0"):
    torch.manual_seed(0)
    router = torch.randn(T, E, device=dev, dtype=torch.float32)
    _, topk_ids = torch.topk(torch.softmax(router, dim=-1), K, dim=-1)
    return topk_ids.to(torch.int32).contiguous()


def make_outputs(E, topk_length, dev="cuda:0"):
    return (
        torch.empty(E + 1, dtype=torch.int32, device=dev),
        torch.empty(E + 1, dtype=torch.int32, device=dev),
        torch.empty(E, 3, dtype=torch.int32, device=dev),
        torch.empty(E, 3, dtype=torch.int32, device=dev),
        torch.empty(topk_length, dtype=torch.int32, device=dev),
        torch.empty(topk_length, dtype=torch.int32, device=dev),
    )


MULTI_CTA_THRESHOLD = 32768
NUM_BLOCKS = 8


def make_global_buf(num_blocks=NUM_BLOCKS, dev="cuda:0"):
    return torch.zeros(258 + num_blocks * NUM_EXPERTS,
                       dtype=torch.int32, device=dev)


def run_v7(ext, topk_ids, eo, ps1, ps2, am, cm, bso, n, k,
           global_buf, num_threads=1024, num_blocks=NUM_BLOCKS):
    ext.fused_moe_data(topk_ids, eo, ps1, ps2, am, cm,
                       NUM_EXPERTS, n, k, bso, True, num_threads,
                       global_buf, MULTI_CTA_THRESHOLD, num_blocks)


def run_vllm(topk_ids, eo, ps1, ps2, am, cm, bso, n, k):
    get_cutlass_moe_mm_data(
        topk_ids, eo, ps1, ps2, am, cm,
        NUM_EXPERTS, n, k, bso, is_gated=True,
    )


def _same_a_map_multiset_per_expert(am_ref, am_cand, eo_ref, eo_cand):
    for e in range(NUM_EXPERTS):
        s_ref, t_ref = eo_ref[e].item(), eo_ref[e + 1].item()
        s_cand, t_cand = eo_cand[e].item(), eo_cand[e + 1].item()
        if (t_ref - s_ref) != (t_cand - s_cand):
            return False
        if t_ref == s_ref:
            continue
        if not torch.equal(
            torch.sort(am_ref[s_ref:t_ref])[0],
            torch.sort(am_cand[s_cand:t_cand])[0],
        ):
            return False
    return True


def validate(ext, T, E=NUM_EXPERTS, K=TOPK, H=2048, I=512,
             num_threads=1024, num_blocks=NUM_BLOCKS):
    dev = "cuda:0"
    topk_ids = make_inputs(T, E, K, dev)
    tl = topk_ids.numel()
    global_buf = make_global_buf(num_blocks, dev)

    eo_ref, bso_ref, ps1_ref, ps2_ref, am_ref, cm_ref = make_outputs(E, tl, dev)
    run_vllm(topk_ids, eo_ref, ps1_ref, ps2_ref, am_ref, cm_ref, bso_ref, I, H)

    eo_v7, bso_v7, ps1_v7, ps2_v7, am_v7, cm_v7 = make_outputs(E, tl, dev)
    run_v7(ext, topk_ids, eo_v7, ps1_v7, ps2_v7, am_v7, cm_v7, bso_v7, I, H,
           global_buf, num_threads, num_blocks)
    torch.cuda.synchronize()

    shapes_ok = (
        torch.equal(eo_ref, eo_v7)
        and torch.equal(bso_ref, bso_v7)
        and torch.equal(ps1_ref, ps1_v7)
        and torch.equal(ps2_ref, ps2_v7)
    )
    if not shapes_ok:
        return False, "shape-mismatch"

    if torch.equal(am_ref, am_v7) and torch.equal(cm_ref, cm_v7):
        return True, "exact"

    if not _same_a_map_multiset_per_expert(am_ref, am_v7, eo_ref, eo_v7):
        return False, "expert-slice-mismatch"

    probe = torch.randn(T, H, device=dev, dtype=torch.bfloat16)
    ref_rt = shuffle_rows(shuffle_rows(probe, am_ref), cm_ref)
    v7_rt = shuffle_rows(shuffle_rows(probe, am_v7), cm_v7)
    if not torch.equal(ref_rt, v7_rt):
        return False, "roundtrip-mismatch"

    return True, "semantic"


def event_time(fn, warmup=50, iters=500):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) * 1000 / iters


def cudagraph_time(fn, warmup=50, iters=500):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        g.replay()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) * 1000 / iters


def bench(ext, T, E=NUM_EXPERTS, K=TOPK, H=2048, I=512,
          warmup=50, iters=500, num_threads=1024, num_blocks=NUM_BLOCKS):
    dev = "cuda:0"
    topk_ids = make_inputs(T, E, K, dev)
    tl = topk_ids.numel()
    eo, bso, ps1, ps2, am, cm = make_outputs(E, tl, dev)
    global_buf = make_global_buf(num_blocks, dev)

    fn_v7 = lambda: run_v7(ext, topk_ids, eo, ps1, ps2, am, cm, bso, I, H,
                           global_buf, num_threads, num_blocks)
    fn_vllm = lambda: run_vllm(topk_ids, eo, ps1, ps2, am, cm, bso, I, H)

    v7_ev = event_time(fn_v7, warmup, iters)
    vllm_ev = event_time(fn_vllm, warmup, iters)
    vllm_cg = cudagraph_time(fn_vllm, warmup, iters)
    v7_cg = cudagraph_time(fn_v7, warmup, iters)

    return vllm_ev, vllm_cg, v7_ev, v7_cg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokens", type=int, nargs="+",
        default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 6144],
    )
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--threads", type=int, default=1024)
    parser.add_argument("--num-blocks", type=int, default=NUM_BLOCKS)
    args = parser.parse_args()

    dev = "cuda:0"
    print(f"GPU: {torch.cuda.get_device_name(dev)}")
    print(f"V7: adaptive 1-CTA / 2-kernel multi-CTA (contention-free scatter), "
          f"{args.threads} threads, {args.num_blocks} blocks")
    print(f"    multi-CTA threshold: topk_length > {MULTI_CTA_THRESHOLD}")
    print()

    ext = build_ext()

    print("=== Validation vs vLLM ===")
    for T in [1, 2, 8, 64, 256, 1024, 4096, 6144]:
        ok, kind = validate(ext, T, num_threads=args.threads,
                            num_blocks=args.num_blocks)
        status = "PASS" if ok else "FAIL"
        print(f"  T={T:>5}: {status} ({kind})")
    print()

    if args.validate_only:
        return

    print("=== Performance: V7 vs vLLM ===")
    hdr = (
        f"{'T':>6} | {'vllm_ev(us)':>11} | {'vllm_cg(us)':>11} | "
        f"{'v7_ev(us)':>10} | {'v7_cg(us)':>10} | "
        f"{'v7cg/vllmcg':>11} | {'winner':>6}"
    )
    print(hdr)
    print("-" * len(hdr))

    for T in args.tokens:
        vllm_ev, vllm_cg, v7_ev, v7_cg = bench(
            ext, T, warmup=args.warmup, iters=args.iters,
            num_threads=args.threads, num_blocks=args.num_blocks,
        )
        ratio = v7_cg / vllm_cg if vllm_cg > 0 else 0
        winner = "V7" if v7_cg < vllm_cg else "vLLM"
        print(
            f"{T:>6} | {vllm_ev:>10.2f} | {vllm_cg:>10.2f} | "
            f"{v7_ev:>9.2f} | {v7_cg:>9.2f} | "
            f"{ratio:>10.3f}x | {winner:>6}"
        )


if __name__ == "__main__":
    main()
