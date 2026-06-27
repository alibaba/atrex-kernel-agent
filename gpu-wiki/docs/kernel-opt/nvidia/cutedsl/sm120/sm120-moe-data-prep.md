# SM120 MoE Data-Prep — Quick Reference

INT32 expert-routing data-prep (histogram + per-CTA prefix offsets + scatter)
on RTX PRO 5000 / 4000 Blackwell (sm_120). Speed-reference for what to apply
and what to AVOID. Beats vLLM CUDA-Graph at ALL T in [1..6144]; **0.706×** at
T=6144.

Full journey: [`../../../ref-docs/nvidia/cutedsl/sm120/sm120-moe-data-prep-optimization.md`](../../../../ref-docs/nvidia/cutedsl/sm120/sm120-moe-data-prep-optimization.md).
Final code: [`../../../../reference-kernels/nvidia/blackwell-geforce/cutedsl/moe_data_prep/`](../../../../../reference-kernels/nvidia/blackwell-geforce/cutedsl/moe_data_prep).
Cautionary patterns: [`../../../pitfalls/nvidia/cutedsl/sm120-moe-data-prep-pitfalls.md`](../../../../pitfalls/nvidia/cutedsl/sm120-moe-data-prep-pitfalls.md).

## When to apply this recipe

- Workload: MoE token-to-expert routing scatter with E=64..1024 experts, K=4..16
  topk, T=1024..16384 tokens
- Target: NVIDIA RTX PRO 5000 / 4000 Blackwell (sm_120 client) — also applicable
  in spirit to other multi-CTA scatter-style INT32 kernels on the same arch
- Integration: PyTorch + JIT CUDA C++ via `torch.utils.cpp_extension.load_inline`
  with CUDA-graph capture required
- Reference baseline: vLLM `get_cutlass_moe_mm_data` CUDA Graph

## Recipe (in this order)

### 1. CUDA C++, NOT CuTeDSL

CuTeDSL `cute.compile()` callable cannot be captured in CUDA graphs (graph is
empty), and CuTeDSL per-call overhead is ~16 µs which floors above vLLM's ~12 µs
CG baseline. Use `torch.utils.cpp_extension.load_inline` for JIT CUDA C++.

### 2. Adaptive single-CTA / multi-CTA dispatch

```cpp
if (topk_length <= 32768) {
    fused_moe_data_single_cta<<<1, 1024, 0, stream>>>(...);
} else {
    // 8 CTAs × 1024 threads, two kernels
    fused_moe_histogram_kernel<<<8, 1024, 0, stream>>>(...);
    fused_moe_scatter_kernel  <<<8, 1024, 0, stream>>>(...);
}
```

Rationale: single-CTA wins at small T (no cross-CTA sync overhead); multi-CTA
needed for large T to escape SM-busy 0.20% on a single SM.

### 3. V7 contention-free scatter (the V6→V7 jump, -14.2%)

**Don't** do `atomicAdd(&global_offsets[e], 1)` in scatter for cross-CTA rank —
8 CTAs × 1024 threads contending on 256 global counters → No Eligible Warps 90.28%.

**Do** pre-compute per-CTA per-expert base offsets in the histogram's last-CTA
phase, then scatter uses **smem-only** atomicAdd for local rank:

```cpp
// In histogram's last-arriving CTA:
//   global_buf[258 + bid*256 + e] = per-CTA cumulative base offset for expert e
//
// In scatter (each CTA):
__shared__ int cta_base_smem[NUM_EXPERTS];
__shared__ int local_cnts[NUM_EXPERTS];
// Phase 0: load cta_base[bid][e] from global_buf into smem
// Phase 1: for each token i in CTA's chunk:
int local_rank = atomicAdd(&local_cnts[eid], 1);   // smem atomic only
int global_rank = cta_base_smem[eid] + local_rank;
a_map[global_rank] = i / topk;
c_map[i] = global_rank;
```

`global_buf` layout: `[0..NUM_EXPERTS-1]` counts, `[NUM_EXPERTS]` arrival,
`[NUM_EXPERTS+1]` ntok, `[NUM_EXPERTS+2 .. NUM_EXPERTS+2+NB*NUM_EXPERTS-1]`
per-CTA base offsets.

Pattern source: aiter `moe_align_block_size.py` 4-stage contention-free approach.

### 4. V9-A bank-replicated histogram (architectural foundation, even when wall-time-flat)

**Don't** use a single `local_cnts[NUM_EXPERTS]` for the histogram-phase atomicAdd
when 32 warps × 32 lanes are hammering 256 banks — collisions when
`(e1 % 32) == (e2 % 32)` for hot experts cause inter-warp contention.

**Do** replicate 4-way: each warp picks bank=warpid&3, atomicAdds into its
private slice, then a merge phase reduces back to 1-D:

```cpp
constexpr int NUM_BANKS = 4;
__shared__ int local_cnts_rep[NUM_BANKS][NUM_EXPERTS];   // 4 KB
__shared__ int local_cnts[NUM_EXPERTS];                   // 1 KB
// Phase 1: per-warp bank-private atomicAdd
const int bank = (threadIdx.x >> 5) & (NUM_BANKS - 1);
atomicAdd(&local_cnts_rep[bank][eid], 1);
__syncthreads();
// Merge: 256 threads × 4 banks → local_cnts[e]
for (int e = tid; e < NUM_EXPERTS; e += blockDim.x) {
    int s = 0;
    #pragma unroll
    for (int b = 0; b < NUM_BANKS; b++) s += local_cnts_rep[b][e];
    local_cnts[e] = s;
}
__syncthreads();
```

Architectural gain: Eligible Warps/Sched +52%, Warp Cyc/Inst -10.6%. Wall time
~flat at this single change because the histogram is **barrier-bound** (~45% CPI
on `__syncthreads`), but the extra scheduling slack matters as a foundation.

## Anti-patterns (DO NOT apply, all REGRESSED)

| Anti-pattern | Why it regresses on this workload |
|---|---|
| `__match_any_sync` warp-aggregated smem atomicAdd | Bank conflicts are inter-warp not intra-warp at NUM_EXPERTS >> warp_size; expected intra-warp dup rate ≈ 1.94 per warp at E=256/K=8 — primitive cost (~16 cyc/iter) > savings. V8: +24.5% regression. |
| `cub::BlockRadixSort` over chunk_size > 1K keys for sort-then-flush a_map coalescing | 8-pass radix sort over 6144 keys = 5.8× instruction inflation, dwarfs the partial coalescing win. Plus c_map loses V7's source-stride coalescing. V10: +41.5% regression. |
| Warp-specialize barrier-bound histogram by exiting non-leader warps after Phase 1 | Achieved Occupancy collapses (64% → 21%); top stall just shifts from `Stall Barrier` to `Stall LongScoreboard` at comparable cost; lane-0-serial Phase A becomes fully exposed. V11-A: +25.1% regression. |
| Trust NCU "% est. speedup" as wall-time delta when 3+ stalls each > 30% CPI | Multi-bottleneck rebalances rather than reduces wall time. See [cross-arch meta-rule](../../../../ref-docs/nvidia/common/ncu-rule-est-speedup-meta-rules.md). |

## Expected gains vs vLLM CG (sm_120)

| T | vLLM CG (µs) | V9 (µs) | Ratio |
|---|---|---|---|
| 1..64 | 12.3..14.4 | 8.2 | 0.57..0.67× |
| 256..2048 | 15.7..20.5 | 10.3..18.5 | 0.65..0.90× |
| 4096 | 28.7 | 26.7 | 0.93× |
| **6144** | **34.85** | **24.61** | **0.706×** |

V9 wins at every T value. Stop condition (CuTeDSL kernel time < vLLM CG for all
T in [1..6144]) was met by the V6 multi-CTA split alone; V7+V9 are further
-29.4% improvement at T=6144.

## Related

- Final code & README: [`reference-kernels/nvidia/blackwell-geforce/cutedsl/moe_data_prep/`](../../../../../reference-kernels/nvidia/blackwell-geforce/cutedsl/moe_data_prep)
- Optimization journey (V0..V11): [`../../../ref-docs/nvidia/cutedsl/sm120/sm120-moe-data-prep-optimization.md`](../../../../ref-docs/nvidia/cutedsl/sm120/sm120-moe-data-prep-optimization.md)
- Pitfalls (4 traps): [`../../../pitfalls/nvidia/cutedsl/sm120-moe-data-prep-pitfalls.md`](../../../../pitfalls/nvidia/cutedsl/sm120-moe-data-prep-pitfalls.md)
- Cross-arch NCU rebalancing meta-rule: [`../../../ref-docs/nvidia/common/ncu-rule-est-speedup-meta-rules.md`](../../../../ref-docs/nvidia/common/ncu-rule-est-speedup-meta-rules.md)
- Sister sm_120 quick-references:
  - GDN decode cp.async cache mode: [`sm120-gdn-decode-cpasync-cache-mode.md`](sm120-gdn-decode-cpasync-cache-mode.md)
  - NVFP4 inline-PTX GEMM: [`sm120-nvfp4-inline-ptx-gemm.md`](../../../../ref-docs/nvidia/cutedsl/sm120/sm120-nvfp4-inline-ptx-gemm.md)
