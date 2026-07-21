# NCU "Estimated Speedup" — When It's a Wall-Time Lever, When It Isn't

Cross-arch meta-rules for interpreting Nsight Compute's per-rule
`Estimated Speedup %` in profile-driven kernel optimization. Distilled from
multiple sessions where heavyweight single-rule attacks regressed wall time
even though each attack moved its targeted metric in the right direction.

## TL;DR

NCU's per-rule `Est. Speedup %` is an **architectural-ceiling estimate**,
NOT a wall-time delta predictor. It assumes:

1. The cited rule is THE dominant bottleneck
2. The fix is purely additive (doesn't introduce new instructions, smem traffic,
   register pressure, occupancy loss, or synchronization)

Both assumptions fail more often than the docs suggest. **In multi-bottleneck
kernels (3+ stalls each ~30%+ CPI), single-rule attacks tend to rebalance into
a different latency type rather than reduce wall time.**

## When the estimate IS reliable

All of:
- The kernel has ONE dominant stall > 60% of CPI
- The fix is **structural-additive** (e.g. enable cp.async, swap a divergent
  branch for a select, fix a stride to enable coalescing, add `assumed_align`
  hint, pad smem to break bank conflicts)
- Other stalls are < 15% (verify in the same NCU report)

When all three hold, NCU's est. is typically within ~30% of the actual wall-time
win.

Examples of fixes that tend to deliver close to NCU's est.:
- Smem padding to break intra-warp bank conflicts (free; no inst count change)
- Vectorization (16-byte loads when alignment permits)
- `assumed_align=16` to unlock cp.async cp_size=128b (single-line fix; ~50% wins
  on streaming workloads — see [`../cutedsl/sm120-gdn-decode-fp32state-bf16qkv-optimization.md`](../../blackwell-geforce/ref-docs/cutedsl/sm120-gdn-decode-fp32state-bf16qkv-optimization.md))
- Fixing a wrong cache-mode hint (`LoadCacheMode.GLOBAL` for streaming reads)

## When the estimate IS misleading

### Multi-bottleneck regime
- 3+ stalls each contribute 30%+ of CPI
- Fixing one rebalances into another (because the other was hidden by the same
  parallelism the targeted rule looks "wasteful" against)
- **Three-strike convergence pattern**: 3 independent attacks at 3 different
  rules all regress wall time despite each one's targeted metric improving

### Heavyweight fixes
Any of:
- Fix introduces new instructions (e.g. CUB sort = `O(N × log E)` inst,
  warp primitives like `__match_any_sync` / `__shfl_sync` = ~10-16 cyc/iter
  on sm_120)
- Fix collapses occupancy (early warp exit, larger smem footprint pushing
  past 1 CTA/SM)
- Fix introduces new sync points (named barriers, mbarrier hand-rolling,
  cooperative-groups sync)
- Fix removes parallelism that was hiding OTHER stalls

In these cases, the targeted metric's % may drop in the post-fix profile
(so the fix "worked" by NCU's own measure), but a different latency takes over
at comparable cost. Net wall time may be flat or worse.

### Cheap fixes that almost always work
These tend to deliver **50-80% of their NCU-estimated ceiling** because they
don't perturb the kernel's other stall categories:
- Bank conflict padding (free smem, no inst count change)
- Coalesced load alignment hints (`__restrict__`, `assumed_align`)
- Vectorization within already-coalesced loads
- **Replication-based contention reduction** (small smem cost, no new primitives)
- Cache-mode hints (`LoadCacheMode.GLOBAL` for streaming, `.STREAMING` for
  high-reuse misses)

## Three-strike convergence as a practical-ceiling signal

When 3 independent attacks at 3 different NCU-cited levers all regress wall
time in the same shape (target metric improves architecturally, wall time
degrades due to rebalancing), the kernel is at its **practical ceiling for
single-rule attacks**. Stop and either:

1. **Accept the ceiling and ship** — three nulls is strong evidence the cheap-fix
   space is exhausted. Extracting more requires holistic restructure.
2. **Commit to a holistic restructure** that addresses multiple stalls
   simultaneously (e.g. complete kernel rewrite changing both algorithm and
   smem layout). Much bigger engineering investment, much higher risk.

Continuing single-rule attacks past three strikes generally just adds more null
results to the journal.

## Documented case studies

### NVIDIA sm_120 — INT32 MoE data prep (2026-04, 11 iterations)

| Iter | Rule attacked | NCU est. | Targeted-metric outcome | Wall time delta |
|---|---|---|---|---|
| V8 | Smem bank conflicts (27.52%) | -27.52% | Bank conflicts unchanged (lesson: wrong intra/inter axis) → primitive itself MIO-bound | **+24.5%** |
| V10 | Uncoalesced global stores (48.85%) | -48.85% | a_map uncoalesced 68% → 61% but absolute sectors UP; sort 5.8× inst inflation dominates | **+41.5%** |
| V11-A | Stall Barrier (45.27%) | -45.27% | Warp Cyc/Inst -43%; Achieved Occupancy 64% → 21%; LongScoreboard exposed | **+25.1%** |

Standing best (V9, the cheap fix — 4-way bank-replicated histogram, ~5 KB smem,
no new primitives) survives. V9 architecturally improved Eligible Warps/Sched
+52% and Warp Cyc/Inst -10.6% with **zero regression**, even though wall time
was flat at the time it landed.

Full case study:
- Pitfalls: [`docs/nvidia/blackwell-geforce/pitfalls/cutedsl/sm120-moe-data-prep-pitfalls.md`](../../blackwell-geforce/pitfalls/cutedsl/sm120-moe-data-prep-pitfalls.md)
- Optimization journey: [`../cutedsl/sm120/sm120-moe-data-prep-optimization.md`](../../blackwell-geforce/ref-docs/cutedsl/sm120-moe-data-prep-optimization.md)
- Final code: [`reference-kernels/nvidia/blackwell-geforce/cutedsl/moe_data_prep/`](../../../../reference-kernels/nvidia/blackwell-geforce/cutedsl/moe_data_prep/)

(Add other case studies here as they accumulate from future sessions.)

## Practical workflow

Before committing engineering time on a heavyweight fix, run this checklist:

1. **Identify the target rule and its est. speedup**.
2. **Check OTHER stalls in the same NCU report** — are any > 30%? If yes, you're
   in multi-bottleneck regime; treat the est. as architectural ceiling, not
   wall-time delta.
3. **Estimate the cost of the FIX itself**:
   - Instructions added (count × per-inst stall)
   - Smem added (smem cap pressure, occupancy delta)
   - Sync points added (cross-warp barriers cost ~hundreds of cyc per warp at
     32-warp CTAs)
   - Register pressure increase (forces lower occupancy)
   - Active-warp delta (early exit / asymmetric specialization → latency hiding loss)

   NCU's pre-fix profile **doesn't show these**. Estimate them separately.
4. **If the FIX cost approaches the targeted savings, prefer a cheaper fix or
   skip**. Cheap fixes (replication, padding, alignment) succeed disproportionately
   often.
5. **If 3 independent fixes have already regressed the kernel** in the rebalancing
   pattern, **declare ceiling and stop iterating**.

## Why this matters

Profile-driven optimization sessions can burn 10+ iterations chasing NCU rules
that look high-leverage but rebalance instead of reduce. The pattern is hard
to spot in any single iteration — only after 3 in a row do you have enough
evidence that the kernel is multi-bottleneck and rule-by-rule attacks won't
work. The cost of one more iteration is high (compile + validate + profile +
revert ≈ 1 hour minimum), so calling the ceiling early matters.

V9-A in the case study above is the canonical "cheap fix that worked": ~5 KB
smem reorganization, no new primitives, zero regression, and it left the
architectural foundation strong enough that the team had something to ship.
The three subsequent heavyweight fixes (V8 / V10 / V11-A) added engineering
time but no wall-time gain and had to be reverted.

## Related docs

- NCU profiling fundamentals: [`ncu-profiling-guide.md`](ncu-profiling-guide.md)
- Measurement trust (Duration≠latency, noise floor, graph-capture pitfalls): [`ncu-measurement-discipline.md`](ncu-measurement-discipline.md)
- Bank conflict mitigation (cheap-fix family): [`smem-swizzling-bank-conflicts.md`](smem-swizzling-bank-conflicts.md)
- Hierarchical reduction patterns: [`hierarchical-reduction-memory-bound.md`](hierarchical-reduction-memory-bound.md)
- Warp specialization design (when it helps vs hurts): [`warp-specialization-design-principles.md`](warp-specialization-design-principles.md)
- Register pressure / warp occupancy interaction: [`register-pressure-warp-occupancy.md`](register-pressure-warp-occupancy.md)
