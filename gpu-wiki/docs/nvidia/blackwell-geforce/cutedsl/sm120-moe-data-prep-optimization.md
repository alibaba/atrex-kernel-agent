# SM120 INT32 MoE Data-Prep — Optimization Journey

End-to-end journey from naive single-CTA CuTeDSL (V0, **5.83× SLOWER** than vLLM

**Last updated**: 2026-06-30

CUDA Graph at T=6144) to V9 CUDA C++ (**0.706× of vLLM CG**, beats vLLM at all T
in [1..6144]). 11 iterations, 3 documented null results, on NVIDIA RTX PRO 5000
Blackwell (sm_120).

The kernel is `fused_moe_data` — a 3-output dispatcher producing `expert_offsets`,
`problem_sizes1/2`, `blockscale_offsets`, `a_map`, `c_map` from `topk_ids[T, K]`.
Routes tokens to E=256 experts for downstream grouped GEMMs. INT32 throughout, no FP.

## Target hardware

| Item | Value |
|---|---|
| GPU | NVIDIA RTX PRO 5000 72GB Blackwell |
| Compute Capability | sm_120 (CC 12.0) — *client* Blackwell |
| SMs | 110 |
| Smem / SM | 100 KB dynamic (48 KB static default; opt-in via `cudaFuncSetAttribute`) |
| Smem banks | 32 × 4 B |
| CUDA / Driver | 12.8 / 580.x |
| Host integration | `torch.utils.cpp_extension.load_inline` (CUDA C++ JIT) |

**Architectural constraint**: sm_120 is *client* Blackwell. **No `tcgen05` /
TMEM / wgmma / clusters / DSMEM**. Routes are Hopper-style **TMA + cp.async**
for memory + Ampere-style **warp ALU + warp shuffle** for compute. Do NOT copy
sm_100 patterns. (Same constraint as
[`sm120-gdn-decode-fp32state-bf16qkv-optimization.md`](sm120-gdn-decode-fp32state-bf16qkv-optimization.md)
and [`sm120-nvfp4-persistent-gemm-pro5000-optimization.md`](sm120-nvfp4-persistent-gemm-pro5000-optimization.md)
— same chip, same constraint.)

## Algorithm baseline

```
Input:  topk_ids [T, K] int32   (E expert ids per token; -1 = invalid slot)
Output: expert_offsets [E+1]            int32
        problem_sizes1/2 [E, 3]         int32
        blockscale_offsets [E+1]        int32
        a_map [topk_length]             int32   a_map[expert_offset[e]+r] = src_token of r-th routed to e
        c_map [topk_length]             int32   c_map[i] = global rank of slot i (inverse)

Roundtrip invariant (validated):  a_map[c_map[i]] == i / TOPK   for all valid i
```

For E=256, K=8, T up to 6144 → topk_length up to 49152. Reference: vLLM's
`get_cutlass_moe_mm_data` (CUDA Graph captured); shape contract verified in
project's `validate()` via `_same_a_map_multiset_per_expert` + `shuffle_rows`
roundtrip.

## Kernel resource footprint (final V9, multi-CTA path)

| Item | Value |
|---|---|
| Threads/CTA | 1024 |
| CTAs (multi-CTA) | 8 (when topk_length > 32768) |
| Single-CTA threshold | topk_length ≤ 32768 (1 CTA, 1024 threads) |
| Histogram smem | 5 KB (`local_cnts_rep[4][256]` 4 KB + `local_cnts[256]` 1 KB) |
| Scatter smem | 2 KB (`cta_base_smem[256]` 1 KB + `local_cnts[256]` 1 KB) |
| Registers/Thread | 41 (histogram) / 42 (scatter), no spill |
| Achieved Occupancy | 64.59% (histogram) / 59.89% (scatter) |
| CUDA Graph compatible | YES (both kernels) |

## Performance baseline (CUDA event timing, T=6144, µs)

| Reference | Time | Note |
|---|---|---|
| vLLM event | 40.95 | reference Python+CUDA implementation |
| vLLM CUDA Graph | 34.85 | the canonical baseline to beat |
| Naive CuTeDSL V0 | 202.96 | 5.83× SLOWER than vLLM CG |

## Optimization journey

Format: **vN — change → measured T=6144 CG → ncu evidence → notes**.

### V0 — Naive CuTeDSL single-CTA, 128 threads
Direct port from FLA-style reference. Single CTA, 128 threads, private counter
matrix `(128+1) × 256 × 2 = 66 KB` smem → forces low occupancy.
**Time T=6144 CG**: 202.96 µs (5.83× slower than vLLM 34.85 µs).
**Bottleneck**: Phase 2 serial reduction across 128 thread bins per expert
= O(128 × 256) smem ops; serial prefix sum by thread 0 = O(256) iterations;
single CTA = poor utilization for large T.

### V2 — Smem atomics, CuTeDSL, 256 threads
Replaced 66 KB private counter matrix with `[256]` smem array (1 KB) + `atomicAdd`.
Eliminated O(T*E) reduction phase entirely.
**Time**: 2× faster than V0 at large T.
**Issue**: CuTeDSL `cute.compile()` callable CANNOT be captured in CUDA Graph
(graph is empty). CuTeDSL per-call overhead is ~16 µs (event timing) — sets a
hard floor above vLLM CG's ~12 µs baseline.

### V3 — CUDA C++ kernel via `load_inline`, smem atomics, 256 threads
**Discovery**: to beat vLLM CG, MUST use CUDA C++ kernel (CuTeDSL CG-incompatible).
JIT compile via `torch.utils.cpp_extension.load_inline`. Same algorithm as V2.
**Time**: beats vLLM CG for T ≤ 2048; T=4096 0.999×; T=6144 1.06× (slightly slower).
**Issue**: single CTA on 1 of 110 SMs → SM Busy 0.20%, No Eligible Warps 76.68%,
L1/TEX 59.80% — single CTA underutilizes the GPU at large T.

### V3-1024t — Same + 1024 threads
Just bump thread count. **Time**: T=6144 1.059× — single-SM cap still bites.

### V4 — Multi-CTA with global atomicAdd
Split work across multiple CTAs, use `atomicAdd(&global_buf[e], ...)` for cross-CTA
merge. Last-block-arrives pattern (`atomicAdd(&arrival, 1)`) for prefix sum barrier.
**Time**: REGRESSION T=4096 1.07×. **Why**: global atomic overhead too high;
8 CTAs × 1024 threads contending on 256 global counters dominates.

### V5 — Vectorized loads (ABANDONED, never validated)
int4 vectorized loads (128-bit) for `topk_ids` in Phase 1 and Phase 3. Written but
never tested — user feedback: must use profile-driven optimization (ncu/rocprofv3
only, no `do_bench`/`cuda.Event` as optimization evidence).

### V6 — Adaptive 1-CTA / 2-kernel multi-CTA (PROFILE-DRIVEN)
Profile evidence: V3 at T=6144 ncu: SM Busy 0.20% (1/110 SMs), No Eligible 76.68%,
L1/TEX 59.80% — single-CTA underutilization + smem atomic contention on 1 SM.

**Fix**: split into 2 kernels for large T (`topk_length > 32768`):
- Kernel 1 (`fused_moe_histogram_kernel`): 8 CTAs × 1024 threads, local smem
  histogram + global merge + last-block prefix sum
- Kernel 2 (`fused_moe_scatter_kernel`): 8 CTAs × 1024 threads, **global atomicAdd
  for rank** in scatter
- Small T uses single-CTA (V3 approach), `topk_length ≤ 32768`

CUDA graph compatible (both paths).

**Time T=6144 CG**: **28.70 µs = 0.823× of vLLM CG** ✓ — beats vLLM CG at ALL T
in [1..6144] for the first time. Stop condition met by V6 alone.

**ncu evidence (T=6144)**:
- histogram_kernel: Duration 25.98 µs, Grid (8,1,1), L1/TEX 15.48%, SM Busy 0.28%
- scatter_kernel: Duration 17.25 µs, Grid (8,1,1), L1/TEX 40.81%, SM Busy 0.64%
- Total ncu: 43.23 µs vs V3 56.10 µs → 23% improvement
- L1/TEX pressure distributed: V3 59.80% on 1 SM → V6 15.5%/40.8% on 8 SMs

### V7 — Contention-free scatter via per-CTA pre-computed offsets (RETAINED)
Profile evidence: V6 scatter ncu shows **No Eligible 90.28%** = global atomicAdd
contention (8 CTAs × 1024 threads all `atomicAdd(&global_offsets[e], 1)` for rank).

**Fix**: pre-compute per-CTA per-expert base offsets in the histogram's last-CTA
phase (added 8×256 ints to `global_buf` layout at `[258 .. 258+8*256-1]`). Scatter
then uses **smem-only atomics** for local rank, then `global_rank = cta_base[eid]
+ local_rank`. Zero global atomic contention in scatter hot path.

**global_buf layout** (extended): `[0..255]` counts, `[256]` arrival counter,
`[257]` ntok, `[258..258+8*256-1]` per-CTA base offsets.

Pattern source: aiter `moe_align_block_size.py` 4-stage contention-free approach.

**Time T=6144 CG**: **24.62 µs = 0.706× of vLLM CG** (-14.2% vs V6 28.70 µs)
**ncu evidence (T=6144)**:
- histogram_v7: Duration 25.25 µs, No Elig 84.06%, top stall **Barrier 46.67%**
- scatter_v7:   Duration 11.42 µs, No Elig 80.51%, top stall LongScoreboard 33.59%
- Total ncu: 36.67 µs vs V6 43.23 µs → -15.2%

V7 achieves the practical wall-time floor; V8/V9/V10/V11-A all attempted to push
further by attacking individual NCU rules. Only V9-A succeeded as architectural
foundation; V8/V10/V11-A all regressed.

### V8 — `__match_any_sync` warp-aggregated scatter atomicAdd (REGRESSED, reverted)
**Profile target**: V7 scatter ncu shows `Bank Conflicts 27.52%` rule firing on
`local_cnts[eid]` smem atomicAdd. Apply textbook warp-aggregation:
`__match_any_sync` + leader `atomicAdd(&local_cnts[eid], __popc(peers))` + `__shfl_sync`
broadcast.

**Time T=6144 CG**: **30.65 µs vs V7 24.62 µs = +24.5% REGRESSION**
**Why**: bank conflicts are **inter-warp**, not intra-warp. Expected intra-warp
duplicate rate ≈ 1.94 per warp at E=256/K=8. Warp aggregation saves ~2 atomicAdds
per warp at a cost of ~16 cyc/iter for the warp primitives. Net negative. Top stall
shifted to **`ShortScoreboard MIO 51.0%`** (the warp primitives themselves are
MIO-bound on sm_120). Bank-conflict wavefronts UNCHANGED (V7 68.52% → V8 68.57%).
Branch eff 100→84%.

**Reverted as V8R** (commit 766b316). See pitfall #1 in
[`../../../pitfalls/nvidia/cutedsl/sm120-moe-data-prep-pitfalls.md`](pitfalls/sm120-moe-data-prep-pitfalls.md)
for full analysis.

### V9 — 4-way bank-replicated histogram `local_cnts` (RETAINED, neutral wall-time, foundation win)
**Profile target**: V7 histogram top stall `Stall Barrier 46.67%`. Hypothesis:
inter-warp atomic contention on `local_cnts[hot_eid]` (32 warps × ~32 lanes hammering
256 banks; collisions when `(e1 % 32) == (e2 % 32)` for hot experts).

**Fix**: replicate `local_cnts` 4-way → `local_cnts_rep[4][NUM_EXPERTS]` (4 KB smem).
Each warp atomicAdd's into bank `warpid & 3`. Reduces 32-warp contention to
8-warp-per-bank. Merge phase (256 threads × 4 banks) reduces back to 1-D
`local_cnts[e]` so downstream code (per-CTA store, global merge, last-CTA Phase A/B)
is byte-identical to V7. Scatter unchanged.

**Time T=6144 CG**: **24.61 µs vs V7 24.62 µs = ~flat** (NEUTRAL)
**ncu evidence (T=6144)**:
- histogram_v7 (V9): Duration 25.31 µs (~flat), Issued Inst +19.8%
- **Eligible Warps / Sched 0.31 → 0.47 (+51.6%)** ← inter-warp contention WAS real
- **Warp Cyc / Issued Inst 49.33 → 44.11 (-10.6%)** ← per-instruction stall down
- Top stall **STILL Barrier 45.27%** ← histogram is barrier-bound, not contention-bound
- Total ncu: 36.86 µs vs V7 36.67 µs (~flat)

**Architectural win** even though wall time is flat: more scheduling slack to absorb
future scatter optimization. Histogram is **barrier-bound, not contention-bound**;
fixing what's BETWEEN the barriers doesn't move wall time because threads finish
earlier but then wait at the next `__syncthreads` anyway. The barrier wait dominates
either way until the barriers themselves change.

### V10 — `cub::BlockRadixSort` scatter restructure (REGRESSED, reverted)
**Profile target**: V7/V9 scatter ncu shows `Uncoalesced Global Stores 48.85%` on
`a_map[]` writes — the largest single lever in the workload. Apply sort-then-flush:
`cub::BlockRadixSort<int32_t, 1024, 6, int32_t>::SortBlockedToStriped(keys, vals, 0, 8)`
over `(eid, packed{src_token, orig_pos})`, then per-expert coalesced bursts to
`a_map[base[e] + j]`.

c_map handling: pack `vals[j] = (src_token << 13) | orig_pos`; in the per-expert
flush phase write `a_map[base+j] = src_token` (coalesced) AND
`c_map[chunk_start + orig_pos] = base+j` (scattered) — both writes use the SAME
`base+j` to satisfy the roundtrip invariant.

**Time T=6144 CG**: **34.83 µs vs V9 24.61 µs = +41.5% REGRESSION**
**Why**: Issued Instructions EXPLODED 101,888 → **590,288 (5.8×)** — CUB 8-pass
radix sort over 6144 keys generates massive instruction count. Per-instruction stall
IMPROVED (37.4 → 17.6 cyc, lots of eligible warps from sort) but you can't out-IPC
a 6× larger instruction count. The a_map coalescing win materialized partially
(uncoalesced sectors 68.4% → 61.4%) but absolute uncoalesced sectors went UP (41,083
→ 42,996) because c_map now scattered offset the win. Smem 88 KB dynamic (CUB
TempStorage ~48 KB + sorted_vals/keys 48 KB; required `cudaFuncSetAttribute`
opt-in).

Note: a first V10 attempt before this version had a correctness bug (race-order
Phase 1.5 c_map vs sort-stable Phase 6 a_map → roundtrip mismatch at T=6144). Fix
required deriving c_map from the SAME sort-stable rank as a_map (drop Phase 1.5
c_map for valid tokens, recover from packed orig_pos in Phase 6).

**Reverted as V10R** (commit f16c4e8). See pitfall #2 in the pitfalls doc.

### V11-A — Warp-specialized histogram merge + Phase A/B (REGRESSED, reverted)
**Profile target**: V9 histogram `Stall Barrier 45.27%`. Warp-specialize: warp 0
owns Phase 2 (global merge), arrival counter, and last-CTA Phase A+B; warps 1-31
exit immediately after Phase 1's barrier. Replace 2 `__syncthreads` with `__syncwarp`.

**Time T=6144 CG**: **30.78 µs vs V9 24.61 µs = +25.1% REGRESSION**
**Why**: Achieved Occupancy COLLAPSED 64.59% → 20.77%. Warp Cyc/Inst dropped
44.11 → 25.12 (-43%, the fix worked at per-instruction level) and Issued
Instructions dropped 17.8%, but top stall shifted from `Stall Barrier 45.27%` to
**`Stall LongScoreboard (L1TEX) 48.42%`** at comparable cost. Removing the barriers
also removed the parallel warps that were hiding global-load latency. Lane-0-only
serial Phase A loop (256 iter × 4 global ops = ~1024 serialized) became fully exposed.

**Reverted as V11R** (commit 403f144 == b9baa83 byte-identical to V9). See
pitfall #3 in the pitfalls doc.

## Final perf vs baseline (CUDA Graph, all T)

| T | vLLM event (µs) | vLLM CG (µs) | V9 CG (µs) | V9 / vLLM CG |
|---|---|---|---|---|
| 1 | 16.40 | 12.35 | 8.21 | **0.666×** |
| 2 | 16.40 | 12.37 | 8.20 | 0.663× |
| 8 | 16.39 | 14.35 | 8.20 | 0.572× |
| 32 | 16.39 | 14.35 | 8.21 | 0.572× |
| 64 | 18.43 | 14.36 | 8.21 | 0.572× |
| 128 | 20.49 | 14.36 | 10.25 | 0.714× |
| 256 | 20.48 | 15.74 | 10.26 | 0.652× |
| 512 | 20.49 | 16.40 | 12.30 | 0.750× |
| 1024 | 22.54 | 18.44 | 14.35 | 0.778× |
| 2048 | 24.59 | 20.51 | 18.46 | 0.900× |
| 4096 | 32.79 | 28.69 | 26.73 | 0.932× |
| **6144** | **40.95** | **34.85** | **24.61** | **0.706×** |

**V9 wins at every T value vs vLLM CG**. Stop condition (CuTeDSL kernel time <
vLLM CG for all T in [1..6144]) was met by V6 alone; V7/V9 are further -14.2% / -29.4%
improvements at T=6144.

## Remaining bottlenecks (V9 NCU evidence, T=6144, untouched after V11-A revert)

| Stall / metric | Value | Lever est. | Status |
|---|---|---|---|
| Histogram `Stall Barrier` | 45.27% | NCU est. 45.27% | Untouched after V11-A revert; warp-specialization regresses (occupancy collapse). Multi-bottleneck — Barrier + LongScoreboard + atomic-merge each ~30-50% CPI. |
| Scatter `Uncoalesced a_map stores` | 68.4% sectors | NCU est. 48.85% | Untouched after V10 revert; CUB sort regresses (instruction inflation). Sort-based coalescing exhausted (CUB / bitmatrix / cumsum-α all face same instruction-count issue at this workload size). |
| `LaunchConfiguration` (8 vs 110 SMs) | est. 92.73% | Untouched. Persistent kernel scaling 16/32 CTAs would compound with both kernels' phases; high probability of replicating the V11-A occupancy-collapse pattern. |

## What would close the remaining gap (research only, NOT validated)

Three approaches were considered for V12 but ALL deferred / not attempted because
they share the same risk profile that bit V8/V10/V11-A:

1. **Persistent kernel scaling 16/32 CTAs** — addresses LaunchConfig 92.73% est.,
   but requires same warp-specialization that just bit V11-A. High probability of
   replicating the regression pattern.
2. **Split histogram kernel** (Phase 1 separate launch, Phase 2-3 separate launch)
   — each kernel can be occupancy-balanced for its phase, but loses launch overhead
   (2-3 µs at T=6144 scale).
3. **Holistic restructure** combining V9-A bank replication + bitmatrix scatter
   coalescing + asymmetric warp specialization — high engineering cost, multiple
   interacting changes prevent clean attribution.

V9 declared the **practical ceiling** after three-strike convergence (V8/V10/V11-A
all regressed). See the cross-arch meta-rule doc:
[`../../common/ncu-rule-est-speedup-meta-rules.md`](../../common/profiling/ncu-rule-est-speedup-meta-rules.md).

## Sustained recipe (do these, in this order)

For an INT32 expert-routing data-prep kernel on sm_120 client Blackwell:

1. **CUDA C++ via `load_inline`** — NOT CuTeDSL (CuTeDSL `cute.compile()`
   incompatible with CUDA graph capture; sets a 16 µs floor above vLLM's 12 µs CG)
2. **Adaptive single-CTA / multi-CTA dispatch** with threshold ~32K topk_length
   (single-CTA wins at small T; multi-CTA needed for T ≥ 4097)
3. **V6 multi-CTA shape**: 8 CTAs × 1024 threads, two kernels (histogram, scatter)
4. **V7 contention-free per-CTA offsets**: pre-compute per-CTA per-expert base
   offsets in histogram's last-CTA phase; scatter uses smem-only atomics for local
   rank. Zero global atomic contention in scatter hot path. Pattern source: aiter
   `moe_align_block_size.py`.
5. **V9-A bank-replicated `local_cnts`** in histogram: 4-way replication
   (`local_cnts_rep[4][NUM_EXPERTS]`, 4 KB smem). Cuts inter-warp contention
   architecturally (Eligible Warps/Sched +52%, Warp Cyc/Inst -10.6%) — sets up
   future improvements even though wall time is flat.
6. **DO NOT** apply heavyweight fixes (warp-aggregated atomics, library sort,
   warp-specialization that exits warps) without checking the multi-bottleneck
   regime first. See pitfall doc anti-pattern table.

## Related docs

- Pitfalls (4 traps): [`../../../pitfalls/nvidia/cutedsl/sm120-moe-data-prep-pitfalls.md`](pitfalls/sm120-moe-data-prep-pitfalls.md)
- Quick reference: [`../../../kernel-opt/nvidia/cutedsl/sm120/sm120-moe-data-prep.md`](sm120-moe-data-prep.md)
- Final code: [`../../../../reference-kernels/nvidia/blackwell-geforce/cutedsl/moe_data_prep/`](../../../../reference-kernels/nvidia/blackwell-geforce/cutedsl/moe_data_prep)
- Cross-arch meta-rule (NCU rebalancing): [`../../common/ncu-rule-est-speedup-meta-rules.md`](../../common/profiling/ncu-rule-est-speedup-meta-rules.md)
- Sister sm_120 journeys (same chip, different kernels):
  - GDN decode: [`sm120-gdn-decode-fp32state-bf16qkv-optimization.md`](sm120-gdn-decode-fp32state-bf16qkv-optimization.md)
  - NVFP4 GEMM: [`sm120-nvfp4-persistent-gemm-pro5000-optimization.md`](sm120-nvfp4-persistent-gemm-pro5000-optimization.md)
