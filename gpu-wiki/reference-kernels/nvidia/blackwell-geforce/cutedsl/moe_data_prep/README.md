# SM120 INT32 MoE Data-Prep Kernel (CUDA C++ via load_inline)

End-to-end JIT-compiled CUDA C++ kernel for MoE data preparation
(histogram + per-CTA prefix offsets + scatter), targeting RTX PRO 5000
Blackwell (sm_120) RTX PRO 4000 / RTX 50xx GeForce. CUDA-graph capture
compatible. INT32 throughout, no FP compute.

| File | What it is |
|------|------------|
| [`fused_moe_data_v9.py`](fused_moe_data_v9.py) | **Production version (V9 / 11 iterations)**. Adaptive single-CTA / 2-kernel multi-CTA. Histogram uses 4-way bank-replicated `local_cnts` to cut inter-warp smem-atomic contention. Scatter uses V7's contention-free per-CTA pre-computed offsets (smem-only atomics in scatter, zero global atomics in hot path). T=6144 CG = 24.61 µs = **0.706× of vLLM CG** (34.85 µs). All T in [1..6144] beat vLLM CG. |

## Workload contract

```
topk_ids [T, K]    int32   (E expert ids per token, -1 = invalid slot)
→
expert_offsets    [E+1]            int32   prefix sum of per-expert token counts
problem_sizes1/2  [E, 3]           int32   per-expert (m, n, k) for grouped GEMMs
blockscale_offsets[E+1]            int32   per-expert blockscale offsets
a_map             [topk_length]    int32   a_map[expert_offset[e]+r] = src token of r-th routed to e
c_map             [topk_length]    int32   c_map[i] = global rank of slot i (inverse of a_map)
```

For E=256, K=8, T up to 6144, topk_length up to 49152.

## Algorithm (V9 = V7 contention-free scatter + V9-A bank-replicated histogram)

Adaptive dispatch on `topk_length`:
- `topk_length ≤ 32768` → **single-CTA path** (1024 threads, smem atomicAdd histogram + smem atomicAdd scatter, ~1 KB smem)
- `topk_length > 32768` → **multi-CTA path** (8 CTAs × 1024 threads, two kernels)

Multi-CTA path:

1. **`fused_moe_histogram_kernel_v7`** (V9-A bank-replicated):
   - Phase 1: each warp atomicAdd's into `local_cnts_rep[warpid & 3][NUM_EXPERTS]`
     (4-way replication; 32 warps spread across 4 banks → 8-warp contention per bank
     instead of 32-warp on one counter)
   - Merge: 256 threads sum across 4 banks → `local_cnts[e]`
   - Per-CTA store: `global_buf[258 + bid*256 + e] = local_cnts[e]`
   - Global merge: atomicAdd `local_cnts[e]` into `global_buf[e]` (across all CTAs)
   - Last-CTA Phase A: serial prefix sum on `global_buf[0..255]` → `expert_offsets`
   - Last-CTA Phase B: per-CTA cumulative offsets `global_buf[258..258+8*256-1]`
     → `cta_base[bid][e]` (each CTA's start position for each expert in global a_map)

2. **`fused_moe_scatter_kernel_v7`** (V7 contention-free):
   - Load `cta_base[bid][e]` into smem
   - For each token i in CTA's chunk:
     - `eid = topk_ids[i]; local_rank = atomicAdd(&local_cnts[eid], 1)`
       (smem atomicAdd; no global atomic)
     - `a_map[cta_base[eid] + local_rank] = i / TOPK`
     - `c_map[i] = cta_base[eid] + local_rank`

## Why a separate sm_120 implementation

This is a CUDA C++ kernel built via `torch.utils.cpp_extension.load_inline`,
NOT a CuTeDSL kernel — CuTeDSL's `cute.compile()` callable cannot be captured
in CUDA graphs (graph capture sees an empty graph), and CuTeDSL per-call
overhead is ~16 µs which sets a hard floor above the vLLM CG baseline of
~12 µs. To beat vLLM CUDA-graph wall time at any T, a graph-compatible
CUDA C++ kernel is required.

## Performance (CUDA Graph, RTX PRO 5000, all T values)

| T | vLLM CG (µs) | V9 CG (µs) | V9 / vLLM |
|---|---|---|---|
| 1 | 12.33 | 8.21 | 0.666× |
| 64 | 14.36 | 8.21 | 0.572× |
| 256 | 16.40 | 10.26 | 0.625× |
| 1024 | 18.44 | 14.35 | 0.778× |
| 2048 | 22.54 | 18.46 | 0.819× |
| 4096 | 28.69 | 26.73 | 0.932× |
| **6144** | **34.85** | **24.61** | **0.706×** |

V0 (naive CuTeDSL single-CTA) at T=6144 was 202.96 µs = 5.83× SLOWER. The full V0..V9 journey is documented in the optimization journey doc.

## Documented null results (3 regressions, retained in git history)

V8 (warp-aggregated atomicAdd via `__match_any_sync`), V10 (cub::BlockRadixSort
scatter restructure), V11-A (warp-specialized histogram merge) all attacked
high-NCU-est. levers (27.52% / 48.85% / 45.27% respectively) and all REGRESSED
wall time by 24-41% due to multi-bottleneck rebalancing. See pitfalls doc.

## Hardware constraints honored

- sm_120 = client Blackwell (RTX PRO 5000/4000, RTX 50xx). NO TMEM, NO tcgen05,
  NO wgmma, NO clusters/DSMEM. CUDA C++ + smem atomics only.
- 110 SMs, up to 100 KB dynamic smem/SM, 32 banks × 4 B
- INT32 throughout, no FP compute
- CUDA graph capture compatible (`cudaMemsetAsync` for per-launch zero-init)

## See also

- Optimization journey: [`docs/nvidia/blackwell-geforce/ref-docs/cutedsl/sm120-moe-data-prep-optimization.md`](../../../../../docs/nvidia/blackwell-geforce/ref-docs/cutedsl/sm120-moe-data-prep-optimization.md)
- Pitfalls: [`docs/nvidia/blackwell-geforce/pitfalls/cutedsl/sm120-moe-data-prep-pitfalls.md`](../../../../../docs/nvidia/blackwell-geforce/pitfalls/cutedsl/sm120-moe-data-prep-pitfalls.md)
- Quick reference: [`docs/nvidia/blackwell-geforce/kernel-opt/cutedsl/sm120-moe-data-prep.md`](../../../../../docs/nvidia/blackwell-geforce/kernel-opt/cutedsl/sm120-moe-data-prep.md)
- Cross-arch lesson (NCU rebalancing): [`docs/nvidia/common/ref-docs/ncu-rule-est-speedup-meta-rules.md`](../../../../../docs/nvidia/common/ref-docs/ncu-rule-est-speedup-meta-rules.md)
