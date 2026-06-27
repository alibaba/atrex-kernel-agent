# SM120 INT32 MoE Data-Prep — Pitfalls

Traps from optimizing the V9 final kernel (4 documented null results across
11 iterations on RTX PRO 5000 Blackwell). Companion to:

- Optimization journey: [`../../ref-docs/nvidia/cutedsl/sm120/sm120-moe-data-prep-optimization.md`](../../../ref-docs/nvidia/cutedsl/sm120/sm120-moe-data-prep-optimization.md)
- Final kernel: [`reference-kernels/nvidia/blackwell-geforce/cutedsl/moe_data_prep/`](../../../../reference-kernels/nvidia/blackwell-geforce/cutedsl/moe_data_prep)
- Cross-arch meta-rule: [`../../ref-docs/nvidia/common/ncu-rule-est-speedup-meta-rules.md`](../../../ref-docs/nvidia/common/ncu-rule-est-speedup-meta-rules.md)

The kernel is CUDA C++ via `torch.utils.cpp_extension.load_inline` (NOT
CuTeDSL) — see the optimization-journey doc for why CuTeDSL was abandoned
on the CG-capture front. The pitfalls below are CUDA / sm_120 / smem /
warp primitives, not CuTeDSL-specific.

---

## 1. `__match_any_sync` warp-aggregated smem atomicAdd regressed at low duplicate rates (V8, +24.5%)

**Trap**: V7 scatter has `atomicAdd(&local_cnts[eid], 1)` on a 256-int smem
counter. NCU shows `Bank Conflicts 27.52%` rule firing. Apply textbook
warp-aggregation: `__match_any_sync` finds same-eid lanes within a warp,
the lowest-index leader does ONE `atomicAdd(&local_cnts[eid], __popc(peers))`,
broadcasts the base via `__shfl_sync`, each lane gets exclusive prefix-popc
as offset.

**Result**:
- Bank-conflict wavefront ratio UNCHANGED (V7 68.52% → V8 68.57%)
- Top stall shifted to **`ShortScoreboard MIO 51.0%`** (the warp primitives
  themselves are MIO-bound on sm_120)
- Branch efficiency 100% → 84% (the `if (laneid == leader)` predicate)
- Per-token instruction count UP ~16 cyc/iter
- Wall time T=6144 CG: 24.62 → 30.65 µs (**+24.5% regression**)

**Why**: the bank-conflict source is **inter-warp**, not intra-warp. With
E=256 and K=8 expanded slots/token, the expected number of intra-warp
duplicate eids per warp ≈ `(32 lanes choose 2) / 256 ≈ 1.94`. Warp aggregation
only fuses intra-warp duplicates, so it saved ~2 atomicAdds per warp at a
cost of ~16 cycles of warp-primitives + branch divergence per iteration.
Net negative.

`match.sync.b32` on sm_120 client Blackwell is itself MIO-bound (no public
perf table for this primitive on this arch — Volta/Turing/Ampere data exists
but not sm_120). The primitive isn't free. PTX spec for `match.sync` is at
NVIDIA PTX ISA 9.2 reference manual section 9.7.13.10 (official: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html).

**Lesson**: Before applying warp-aggregation, **estimate the intra-warp
duplicate rate from your problem size**. Rule of thumb on sm_120: needs
duplicate rate **> ~4 per warp** to break even. For NUM_EXPERTS >> warp_size
workloads (E=256 here), the rate is too low — skip warp-aggregation entirely.

NCU "Bank Conflicts %" estimate **doesn't distinguish intra-warp vs inter-warp**
contention. That distinction has to come from understanding the access pattern.
The right fix for inter-warp contention is bank replication (V9-A succeeded
with that approach at the same lever).

## 2. Sorting over 6144+ keys — 5.8× instruction inflation dominates the coalescing win (V10, +41.5%)

**Trap**: V7 scatter has 68% uncoalesced wavefronts on a_map writes
(NCU est. 3.2× utilization penalty, the largest single lever in
the workload). Sort tokens by expert id with
`cub::DeviceRadixSort::SortPairs`,
then per-expert coalesced bursts to shared memory.

**Results**:
- a_map uncoalesced sectors percentage DROPPED 68.4% → 61.4% (lever was real,
  partial credit)
- Issued Instructions EXPLODED: 101,888 → **590,288** (5.8× inflation)
- Per-instruction stall IMPROVED (Warp Cyc/Inst 37.4 → 17.6 cyc; Eligible
  Warps/Sched 0.38 → 1.12) — sort produces lots of independent work to schedule
- BUT total uncoalesced **ABSOLUTE** sectors went UP: 41,083 → 42,996 (because
  c_map now scattered after sort offsets the a_map win)
- Wall time T=6144 CG: 24.61 → 34.83 µs (**+41.5% regression**)
- Smem: ~88 KB dynamic per CTA (CUB TempStorage ~48 KB + sorted_vals/keys 48 KB)
- Required dynamic smem opt-in via `cudaFuncSetAttribute`
- Branch eff 100% → 92% (radix sort generates divergent code)

**Why**: 8-bit radix sort over 6144 keys × 1024 threads × 6 items/thread =
~8 radix passes of (per-pass scan + scatter), plus its TempStorage smem
traffic. NCU instruction breakdown matched a textbook 8-pass radix sort.
IPC improvements can't out-run the instruction count when the kernel is
small (V7 scatter is 11.4 µs of compute):
- V9 scatter: 101,888 inst @ 0.76 IPC = ~134K cycles ≈ 11.5 µs wall
- V10 scatter: 590,288 inst @ 0.56 IPC = ~1054K cycles ≈ 24 µs wall (matches
  measured 26.4 µs)

The c_map regression (sort breaks V7's source-stride coalescing — c_map
writes now happen at scattered orig_pos addresses) cancels half the a_map
win at the sector level.

**Lesson**: For sort-then-flush patterns under inline CUDA via `torch.utils.cpp_extension.load_inline`:
- **Estimate sort instruction count vs target kernel duration BEFORE
  integrating**. Rule of thumb: CUB radix sort costs roughly
  50–80 inst per key per CTA. For
  ≤ 1024 keys/CTA the cost is acceptable; for **6000+ keys it dominates
  any small target kernel**. Library primitives optimized for general
  workloads pay overhead the smaller-N case can't amortize.
- When optimizing one of two related outputs (a_map AND c_map), **make
  sure the fix doesn't break the OTHER one's existing coalescing**. V7's
  c_map was already optimal — the sort was net-negative there. The
  invariant a_map[i] and c_map[i] must use the SAME global rank
  per token, which means restructuring how a_map is computed forces a
  matching restructure of how c_map is computed (the V10 first attempt
  failed validation precisely because Phase 1.5 race-order c_map and Phase 6
  sort-order a_map disagreed).
- No `cub::DeviceRadixSort` precedent existed in `torch.utils.cpp_extension.load_inline` before this
  session. Build flags needed: `-std=c++17 --expt-relaxed-constexpr
  --expt-extended-lambda -I/usr/local/cuda/include`
  (defensive — nvcc auto-includes CUB but header propagation is
  finicky).

## 3. Warp-specialization in barrier-bound histogram → occupancy collapse (V11-A, +25.1%)

**Trap**: V9 histogram top stall = `Stall Barrier 45.27%` across 3
`__syncthreads()` between Phase 1 (smem hist), Phase 2 (global merge),
Phase 3 (last-CTA prefix sum). Most threads idle during Phase 2 (256 atomic
merges = ~1 atomic per warp). Restructure: warp 0 owns Phase 2+3; warps 1-31
exit immediately after Phase 1's barrier. Replace inter-warp `__syncthreads`
with intra-warp `__syncwarp`.

**Result**:
- Warp Cyc / Issued Inst dropped 44.11 → 25.12 (**−43.0%**, barriers really
  removed — the fix worked at the per-instruction level)
- Issued Instructions dropped 17.8% (no per-warp redundant work after Phase 1)
- BUT Achieved Occupancy COLLAPSED: 64.59% → **20.77%** (catastrophic, −43.8 pts)
- Top stall shifted from `Stall Barrier 45.27%` to **`Stall LongScoreboard
  (L1TEX) 48.42%`** — same wall-time cost, different bucket
- Histogram Duration: 25.31 → 35.39 µs (+39.8%)
- Wall time T=6144 CG: 24.61 → 30.78 µs (**+25.1% regression**)

**Why**: After Phase 1's barrier, warps 1-31 fall through to kernel exit
while warp 0 continues. Hardware tracks **active warps per SM** based on
warps that have not yet exited. Once 31 of 32 warps exit, achieved
occupancy drops 4-fold (from 32 active warps per CTA to 1). Two compounding
effects:

1. **Memory latency hiding lost**: V9's 32 warps doing parallel atomicAdd
   on `local_cnts_rep` provided a deep pipeline of in-flight memory ops
   that hid the global-store / atomic-merge latency. With V11-A, warp 0
   is the only warp issuing global ops — nothing to hide its load-use
   stalls behind. Each global atomic now exposes its full ~250-cycle L2
   round-trip.
2. **Scheduler slot starvation (Little's Law)**: each SM has 4 schedulers
   × 12 warps = 48 warp slots. V9 had 32 warps × 1 block per SM = 8 active
   warps per scheduler. V11-A drops to 1 warp per scheduler-quartet during
   the post-Phase-1 region. Below 4 active warps per scheduler, latency
   hiding falls off a cliff.

Result: barrier savings (~10 µs per ncu estimate) were converted, almost
ton-for-ton, into LongScoreboard waits that now appear in serial because
there's no other warp work to schedule.

Additional effect: when `bid == last_block`, lane 0 alone executes the
256-iteration Phase A loop (4 global loads/stores per iteration ≈ 1024
serialized global ops through one lane). V9 hid this behind 32 parallel
warps; V11-A exposes it.

**Lesson**: When NCU's top stall is `Stall Barrier`, **don't reflexively
warp-specialize to remove the barrier**. First check whether the OTHER
warps are doing useful latency-hiding work during that wait. **The barriers
in V9 weren't pure waste** — they were serving as natural convergence points
where 32 warps cooperate on the next parallel phase, and the inter-barrier
work itself was hiding global-load latency.

Safer alternatives that DIDN'T burn this session (worth trying first in
similar workloads):
- Use **named `bar.sync N, threadcount`** with reduced participant count
  but keep all warps alive (lets warps 1-31 still issue global ops in
  background without blocking on this barrier)
- **Split the kernel** so Phase 1 vs Phase 2-3 each get their own grid
  (loses launch overhead but each kernel is occupancy-balanced for its
  phase)
- **Asymmetric warp specialization** (e.g. warps 0-7 do Phase 2-3, warps
  8-31 exit) — preserves enough warps for latency hiding while still
  skipping the merge for most warps. Requires tuning the split count.
---

## 4. Meta-rule: NCU "% est. speedup" rebalances rather than reduces wall time on multi-bottleneck kernels (V8 + V10 + V11-A, three-strike convergence)

**Trap**: NCU's per-rule `Estimated Speedup %` suggests fixing the rule's
root cause will save that fraction of kernel time. Three iterations attacked
three different high-est. levers in this kernel. **All three regressed
wall time** even though each fix DID move its targeted metric in the right
direction.

| Iter | Rule attacked | NCU est. | Targeted-metric outcome | Wall time delta |
|---|---|---|---|---|
| V8 | Smem bank conflicts (27.52%) | -27.52% | bank conflict wavefronts unchanged (lesson 1: wrong intra/inter axis) → primitive itself MIO-bound | **+24.5%** |
| V10 | Uncoalesced global stores (48.85%) | -48.85% | a_map uncoalesced 68% → 61% but absolute sectors UP; sort 5.8× inst inflation dominates | **+41.5%** |
| V11-A | Stall Barrier (45.27%) | -45.27% | Warp Cyc/Inst -43%; Achieved Occupancy 64% → 21%; LongScoreboard exposed | **+25.1%** |

In each case the targeted rule's % DID drop after the fix (the fix worked
at NCU's measurement level). But a different latency type took over at
comparable or higher cost.

**Why**: in a multi-bottleneck kernel where 3+ stalls each contribute 30%+
of CPI (this histogram had Barrier ~45%, LongScoreboard ~30%, atomic-merge
~20%), Amdahl's-style accounting on a single rule is misleading because
the other stalls aren't independent latencies — they're often **hidden by
the same parallelism the targeted rule looks "wasteful" against**.

NCU's rule estimates assume:
1. The targeted lever is THE bottleneck (true if other stalls < 20%)
2. The fix doesn't degrade other stalls (true if the fix is purely-additive,
   false if it removes parallelism, adds new instructions, collapses
   occupancy, or adds new sync points)

Both assumptions failed in V8/V10/V11-A. In V11-A specifically, fix #2 was
the killer: removing the barrier ALSO removed the parallelism that was
hiding the OTHER stalls.

**Lesson**:
- **Three-strike convergence as a stop signal**: if 3 independent attacks
  at 3 different NCU-cited levers all regress in the same shape (target
  metric improves architecturally, wall time degrades due to rebalancing),
  the kernel is at its practical ceiling for single-rule attacks. Either
  accept the ceiling or commit to a holistic restructure.
- When NCU shows multiple stalls each > 30%, **don't trust per-rule "% est."
  as a wall-time lever estimate**. Treat it as a **per-rule architectural
  ceiling** (the fix may improve that metric architecturally, but won't
  necessarily save wall time).
- The cheapest profile-driven optimizations (V9-A bank replication, ~5 KB
  smem reorganization, no new primitives) are often the ones that don't
  try to remove barriers or restructure phases — they just reduce contention
  within an existing structure. **These succeed more often than heavyweight
  restructures**, even when their NCU-estimate ceiling is smaller. V9-A
  delivered more architectural improvement (Eligible Warps/Sched +52%, Warp
  Cyc/Inst -10.6%) than V8/V10/V11-A combined, with zero regression.
- **Account for fix cost in pre-decision analysis**. NCU shows pre-fix
  metrics; you have to estimate post-fix instruction count + smem traffic +
  register pressure separately. V10's CUB introduced 5.8× issued instructions
  — predictable in advance from CUB's documented complexity, but not part of
  NCU's pre-fix picture.

This pattern generalizes beyond this workload — see the cross-arch meta-rule
doc at [`../../ref-docs/nvidia/common/ncu-rule-est-speedup-meta-rules.md`](../../../ref-docs/nvidia/common/ncu-rule-est-speedup-meta-rules.md).

---

## Anti-pattern quick-reference

| Don't | Do (instead) | Why |
|---|---|---|
| Apply `__match_any_sync` warp-agg without checking intra-warp dup rate | Bank replication (`local_cnts[NUM_BANKS][NUM_EXPERTS]`, warp picks bank=warpid&N) | Inter-warp contention is what bites at NUM_EXPERTS >> warp_size; warp-agg only handles intra-warp |
| Drop in `cub::BlockRadixSort` for sub-µs scatter kernels (>1K keys) | Hand-roll bitmatrix-popc (8 rounds for E=256) OR accept the lever | CUB instruction count dwarfs small target kernel; no library precedent in load_inline |
| Warp-specialize barrier-bound kernel by exiting non-leader warps early | Asymmetric specialization (keep enough warps for latency hiding) OR split kernel by phase | Active-warp count drives occupancy; exiting warps loses memory latency hiding |
| Trust NCU est. when 3+ stalls each > 30% CPI | Treat est. as architectural ceiling; account for fix cost | Multi-bottleneck rebalances rather than reduces wall time |
| Iterate past 3 single-rule regressions on the same kernel | Stop, declare practical ceiling | Three-strike convergence is strong evidence the cheap-fix space is exhausted |
