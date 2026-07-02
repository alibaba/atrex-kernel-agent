# sm_100 Triton sparse-decode / split-K pitfalls

Hardware: NVIDIA **B200 / Blackwell datacenter (`sm_100a`)**, 148 SMs, 126 MB L2

**Last updated**: 2026-06-30

Stack: CUDA 13.x, **Triton 3.6 / 3.7**, CUPTI-based latency harness (no CUDA
graphs, no input-pointer caching — see
[`../../../ref-docs/nvidia/common/ncu-measurement-discipline.md`](../../../common/profiling/ncu-measurement-discipline.md))
Distilled from MLSys26 FlashInfer-contest sessions on two DeepSeek Sparse
Attention operators: MLA-style sparse attention (`H=16`, `ckv=512`,
`topk=2048`) and an FP8 top-k indexer (`H=64`, `d=128`). The kernels are
SM-starved (grid often ≤ 8 CTAs) and heavily padded — a regime where the
textbook tuning order is wrong.

---

## 1. Split-K (flash-decoding) is the structural win on SM-starved grids

**Trap**: A decode/attention kernel with one CTA per token looks "fine" — it
compiles, it's correct, and per-CTA occupancy is high. You start tuning tile
sizes and cache hints.

**Result**: On a grid of ≤ 8 tokens, 1 CTA/token leaves **147/148 SMs idle**.
Splitting the TopK/KV axis across `NUM_SPLITS` programs + a combine pass gave a
**4× speedup** in one step, dwarfing every micro-tuning lever combined.

**Why**: Wall-clock on an SM-starved grid is set by how few SMs you light up,
not by per-CTA efficiency. Until the grid fills the machine, intra-CTA tuning is
rearranging deck chairs.

**Lesson**: Before any tile/cache tuning, compute `grid_CTAs / num_SMs`. If it's
< 1 wave, the first move is **split-K / flash-decoding** to manufacture
parallelism. `NUM_SPLITS` has a sweet spot (here 8 for `topk=2048`): too high
and the combine pass + per-split prologue overhead dominates. Re-open the
`NUM_SPLITS` axis after any structural rewrite — a "near HBM-floor" projection is
computed at a *fixed* split count; doubling splits halves per-CTA bytes and
moves the floor.

---

## 2. NaN landmine: `exp2(-inf - -inf)` in online softmax over maskable tiles

**Trap**: A local online-softmax over a *subset* of keys (one split, one tile)
works in testing. Full-kernel softmax over the whole TopK never tripped it.

**Result**: When a split is **fully masked** (all-padding), `m_new = -inf`, and
`exp2(m_prev - m_new) = exp2(-inf - -inf) = exp2(NaN) = NaN` silently poisons the
accumulator. Surfaces only on shapes where an entire split can be empty.

**Why**: Full-TopK softmax never hit it because "all 2048 entries padded" is
implausible — but per-split, an empty split is common on sparse/short inputs.

**Lesson**: Any online softmax over a possibly-empty subset must guard the max
before exponentiating:
`m_safe = tl.where(m_new == -inf, 0.0, m_new)` and
`alpha = tl.where(m_prev == -inf, 0.0, exp2(m_prev - m_safe))`. Splitting a
working full-reduction into partial reductions *introduces* this hazard — add the
guard at the same time.

---

## 3. A scalar `if` in the hot loop defeats `num_stages` prefetching

**Trap**: Guard the per-iteration body with `if <condition>:` for
workload-adaptive early skip. Looks like it saves work.

**Result**: **+25% regression** on always-valid workloads. When the body is
inside an `if`, Triton can't issue async K/V loads for iter `i+1` while computing
iter `i` — the branch breaks the `num_stages` software pipeline.

**Why**: `num_stages` pipelining requires the compiler to statically see the next
iteration's loads as unconditional. A data-dependent branch hides them.

**Lesson**: Keep compute **unconditional** in the hot loop. For adaptive
early-exit, precompute a dynamic loop bound *before* the loop
(`range(0, dyn_upper, STEP)`) so pipelining stays intact, or mask with
`tl.where` inside the tile ops. Note Triton's `for` AST rejects `break` outright,
so the dynamic-bound pattern is the only option.

---

## 4. `tl.static_range` unroll blows up past a small tile-bytes threshold

**Trap**: `tl.static_range(N)` unrolls the loop for ILP — sounds strictly better
than dynamic `range()`.

**Result**: `static_range(16)` loading a 32-KB tile/iter ran **5× slower** than
dynamic `range()` (register/icache blowup). But at the other extreme, switching a
*small-body* combine loop from `static_range(16)` to `range(16)` **regressed
+5-7%** — the unroll was buying real cross-iter ILP that `num_stages` (only ~1-2
iters ahead) couldn't.

**Why**: Unrolling helps when the per-iter body is small and serial (interleaves
iter `i` compute with iter `i+1` loads); it hurts when each iter loads multi-KB
tiles (the inlined N copies exhaust registers/icache).

**Lesson**: `static_range(N)` is a win when **`N × tile_bytes ≲ 8 KB`** per iter;
it regresses at ≥16 iters or ≥32-KB tiles. Don't switch static↔dynamic on
iteration count alone — decide on per-iter tile footprint.

---

## 5. `num_stages` is codegen-ignored on loop-free (single-dot) kernels

**Trap**: `num_stages` is the canonical latency-hiding knob, so autotune sweeps
it on every kernel.

**Result**: On a single-`tl.dot`, no-outer-loop kernel, `num_stages=3` vs default
was a dead tie (Δ ≈ 0). The attribute pipelines loads/compute *across loop
iterations*; with no loop there is nothing to overlap, so it is ignored.

**Why**: Blackwell's async MMA already hides the latency of a single dot; the
software pipeline needs iterations to stage against.

**Lesson**: Don't spend autotune budget on `num_stages` for loop-free kernels
(single matmul, elementwise). Reserve it for kernels with a real K-tile loop.

---

## 6. Cross-CTA atomic barrier works in Triton — with strict preconditions

**Trap**: Triton has no native cross-CTA barrier, so two serial kernels (split →
combine) seem to require two launches.

**Result**: An atomic-counter barrier fuses them into one launch and recovers
launch tax (~−11% on large shapes). Pattern: `atomic_add(+1, sem="release")` to
signal; spin `while cnt < N` reading the counter; the release/acquire pair
carries memory ordering of prior stores.

**Why & the sharp edges**:
- **Spin with `tl.load(volatile=True)`, not `atomic_add(0, sem="acquire")`** —
  `atom.add` value-0 is still an L2 RMW that serializes when many CTAs poll one
  line; `ld.volatile` is a contention-free L2 read (L2 is the coherence point on
  B200). Add `tl.debug_barrier()` after the spin so the compiler can't hoist
  combine loads above the wait.
- **Use a monotonic generation counter** (`target = gen × NUM_SPLITS`), not a
  reset-by-decrement counter — saves one RMW/CTA/call and is overflow-safe for
  realistic scopes.
- Only safe when **all CTAs fit concurrently on the SMs** — otherwise an
  un-launched CTA the spin waits on never runs ⇒ deadlock.

**Lesson**: The fused-launch barrier is a real lever on SM-starved grids, but the
spin primitive, the reset strategy, and the occupancy precondition all matter —
get any wrong and you regress or hang. Expect to recover only ~25-50% of a
profiler's "launch tax" estimate; the rest is unavoidable CTA-dispatch time.

---

## 7. Triton 3.6 CTA clustering is incompatible with atomic cross-CTA sync

**Trap**: Reach for `num_ctas=N` (thread-block clusters / DSMEM) to accelerate a
kernel that already uses the atomic barrier from #6.

**Result**: Hard compile failure — `PlanCTA.cpp: Assertion !tiled failed`. The
`TritonGPUPlanCTAPass` can't reconcile the CTA-tiling of mixed `atomic_add`
(barrier) + tile loads + softmax. Also, Triton 3.6 doesn't expose `cluster_dims`
at all — only `num_ctas` (driver hardcodes `clusterDim = (num_ctas,1,1)`).

**Why**: The cluster planner requires consistent tiling decisions across all
tensor ops; an atomic barrier op has no compatible tiling.

**Lesson**: On Triton 3.6, **clusters and hand-rolled atomic cross-CTA sync are
mutually exclusive**. Pick one. If you need cluster-scoped cooperation, that path
is Gluon `bw.mbarrier` (with its own tmem/shmem setup), not Triton atomics.

---

## 8. `cache_modifier` and `eviction_policy` are orthogonal levers with strict conditions

**Trap**: Treat `.cg` / eviction hints as global "make memory faster" switches.

**Result**: `.cg` on K loads **won** on an SM-starved split kernel (L1 can catch
re-touched lines) but **regressed** the same loads on a fuller small-T grid (L1 is
pure overhead when each CTA sees fresh K). `evict_first` on a re-read tensor cost
**+3-4%** (hint evicted lines the loop re-reads → HBM refetch). And on **stores**,
`.cg` + `evict_last` is a **ptxas error** ("`.evict_last` cannot be combined with
`.cg`") — store-side L2 eviction is HW-LRU only on Blackwell.

**Why**: `cache_modifier=".cg"` controls L1 bypass; `eviction_policy` is an L2
replacement hint — independent axes. Their value is entirely regime-dependent
(grid fullness, re-touch pattern, freed-capacity size).

**Lesson**: Apply cache/eviction hints **per-load, per-kernel**, never globally.
`evict_first` needs *(a)* the line is read exactly once over the kernel lifetime
**and** *(b)* the freed capacity is ≥ tens of KB. `evict_last` is the inverse
(multi-reader, worth pinning). On loads any combo is legal; on stores avoid
mixing `.cg` with an eviction hint.

---

## 9. `.item()` is a structural barrier (and CUPTI *does* see it); `tensor.shape` is free

**Trap**: Read a GPU-derived scalar (e.g. `num_valid.item()`) on the launch path
to pick a kernel variant. "It's a cheap reduction."

**Result**: A `.item()` before launch added **~60 µs** serial stall on a 12-µs
workload (~9× blowup) — it blocks the CPU until *all* prior stream work finishes,
then gates dispatch of the next kernel. Moving it *after* the launch still stalls
~30 µs (gates the following kernel). Unlike CUDA-graph overlap, this **is**
visible to a CUPTI harness because it serializes the queue.

**Why**: `.item()` is a device→host sync on the default stream, not just a copy —
the whole pipeline drains.

**Lesson**: Never put `.item()` on a hot launch path. **Host-known shapes are
free**: `max_num_pages = block_table.shape[1]` is a plain Python int (no sync),
so branch on *shape* predicates (`if max_num_pages == 1: fast_path()`) to
dispatch specialized kernels at ~0.3 µs cost. To dispatch on *data* properties
without a sync, keep the decision on-device (launch max splits + self-skip empty
work).

---

## 10. `tl.sort` scales badly past BLOCK_N=2048; radix-select beats it

**Trap**: Implement top-K in-kernel with `tl.sort` on packed `(score, index)`
keys.

**Result**: `tl.sort` is correct (via bit-monotonization) but **+349%** vs
`torch.topk` at BLOCK_N=8192. `torch.topk` on B200 uses radix-select (O(N));
Triton's `tl.sort` is bitonic/merge (O(N log²N)) with register blowup at large
tiles. A hand-rolled Triton **radix-select** (32× `tl.sum` greedy bit-threshold +
cumsum-scatter) instead **beat** `torch.topk` by −28% and fused the remap for
free.

**Why**: Sort networks pay log²N depth and explode register pressure past
BLOCK_N≈2048; radix-select is linear and reduces to tree sums.

**Lesson**: Don't try to beat `torch.topk` with `tl.sort` at BLOCK_N ≥ 4096.
`tl.sort` is only viable at small tiles (confident data: BLOCK_N=64). For large-K
in-kernel top-K, use radix-select. For reduction-heavy kernels (`tl.sum` /
`tl.cumsum` over BLOCK_N ≥ 2048), bump `num_warps=8` first (one case: −42%).

---

## 11. Verify a Triton intrinsic's compiled throughput before designing around it

**Trap**: Replace the radix bit-loop with `tl.histogram` (cost model predicted
~2.2 µs for both passes).

**Result**: **+327-443%** regression — actual ~35-40 µs/pass, off by ~30×. The
intrinsic does *not* lower to efficient SMEM-atomic accumulation on this
Triton/Blackwell combo; it materializes dense per-thread bucket-compare masks.

**Why**: Less-common intrinsics (`tl.histogram`, `tl.gather`, non-trivial
`tl.scatter`) have lowering that can be wildly worse than the obvious cost model.

**Lesson**: Before architecting a kernel around a Triton intrinsic, **benchmark
it in isolation** (standalone kernel, `torch.cuda.Event`). Preflight *correctness*
is not preflight *performance*.

---

## 12. `input_precision="ieee"` is a no-op for bf16×bf16 `tl.dot` on sm_100

**Trap**: Add `input_precision="ieee"` to a `tl.dot` to chase
accuracy/perf differences.

**Result**: Byte-identical output and Δ ≈ 0 latency. `input_precision` selects
among **FP32** MMA strategies (full-IEEE / TF32 / TF32x3); bf16 inputs bypass that
selection — there is one PTX instruction (`wgmma.bf16.bf16.f32`) on sm_100.

**Lesson**: `input_precision` is inert for bf16/fp16 MMAs — don't sweep it there.
It only matters when the dot's inputs are fp32. (Generalizes to: a sub-1%
directional result *without a mechanism* is noise, not signal — require a
mechanism story before keeping a marginal win.)

---

## "Use what / don't use what" cheatsheet

| Use | Don't use |
|---|---|
| `grid_CTAs / num_SMs` first → split-K when < 1 wave | Tile/cache tuning before the grid fills the SMs |
| Guard online softmax max for possibly-empty tiles | Split a full reduction without re-adding the NaN guard |
| Unconditional hot-loop body + dynamic loop bound | A scalar `if` in the hot loop (kills `num_stages`) |
| `static_range` only when `N×tile_bytes ≲ 8 KB` | Unroll large-tile / ≥16-iter loops |
| `num_stages` on real K-loops only | Sweep `num_stages` on single-dot kernels |
| `ld.volatile` spin + `debug_barrier` for atomic barrier | `atomic_add(0)` spin; clusters + atomics together |
| Per-load `.cg`/eviction hints with the two-condition rule | Global cache hints; `.cg`+`evict_last` on stores |
| Branch on `tensor.shape` (free) | `.item()` on the launch hot path |
| radix-select / `num_warps=8` for big reductions | `tl.sort` at BLOCK_N ≥ 4096; unbenchmarked intrinsics |

---

## Cross-references

- Measurement trust (what the CUPTI harness can/can't see, noise floor, A/B):
  [`../../../ref-docs/nvidia/common/ncu-measurement-discipline.md`](../../../common/profiling/ncu-measurement-discipline.md)
- `Est. Speedup %` is a ceiling, not a wall-time delta:
  [`../../../ref-docs/nvidia/common/ncu-rule-est-speedup-meta-rules.md`](../../../common/profiling/ncu-rule-est-speedup-meta-rules.md)
- Gluon-on-Blackwell counterpart traps (when porting these kernels off Triton):
  [`../gluon/README.md`](../../gluon/pitfalls/README.md)
- Blackwell kernel knowledge (split-K, flash-decode, sparse MLA):
  [`../../../kernel-opt/nvidia/common/blackwell/`](../../)


## Related

- [NVIDIA Nsight Compute (NCU) Profiling Guide](../../../common/profiling/ncu-profiling-guide.md)
- [AMD rocprofv3 Profiling Guide](../../../../amd/common/rocprofv3-profiling-guide.md)
- [Triton Embraces Tile IR: Beyond SIMT](../../../common/triton/triton-tile-ir-beyond-simt.md)
