# cute-DSL `cutlass.pipeline.PipelineTmaAsync` API notes for sm_120 (4.4.2)

This is a reference + a deferred-plan archive. Source-of-truth from spelunking `CUTLASS $CUTLASS_DIR/python/CuTeDSL/cutlass/pipeline/{helpers.py, sm90.py}` while planning a (D) "abandon warp-spec" experiment that was ultimately not executed in Stage 3 (V0/V1/V2 already proved the standalone fused-quant ceiling, see `wiki_drafts/stage3-closeout.md`). The findings are wiki-grade and apply to any future multi-warp TMA design on sm_120 cute-DSL 4.4.2.

---

## §1 API contract notes

### `CooperativeGroup(Agent.Thread, size: int)`

`CUTLASS $CUTLASS_DIR/python/CuTeDSL/cutlass/pipeline/helpers.py:46-78`:

```python
class CooperativeGroup:
    def __init__(self, agent: Agent, size: int = 1, ...):
        if agent is Agent.Thread:
            assert size > 0
        ...
        # Size indicates how many threads are participating in this CooperativeGroup
        self.size = size
```

- **`size = N threads`**, NOT N warps. `Agent.Thread` is the only currently-implemented enum (`ThreadBlock` and `ThreadBlockCluster` raise `NotImplementedError` in 4.4.2).
- **No thread identity is pinned at construction time.** `CooperativeGroup(Agent.Thread, 1)` says "1 thread participates", not "thread 0" or "warp 0 lane 0". Whichever thread actually calls `producer_acquire()` and `cute.copy(..., tma_bar_ptr=bar)` is the producer.

### `PipelineTmaAsync.create(barrier_storage=...)`

`CUTLASS $CUTLASS_DIR/python/CuTeDSL/cutlass/pipeline/sm90.py:434-516`:

- Each `create()` takes its own `barrier_storage: cute.Pointer` and writes its own `sync_object_full + sync_object_empty` mbarriers there (lines 479-484).
- The mbarrier slots are just bytes at `barrier_storage[0..2*num_stages*8)`. **Multiple instances with disjoint pointers are independent.**
- Cost per instance: `2 × num_stages × 8 B = 32 B for num_stages=2`. **16 instances = 512 B SMEM, negligible.**

### `is_signalling_thread` for empty-arrive

`CUTLASS $CUTLASS_DIR/python/CuTeDSL/cutlass/pipeline/sm90.py:404-406`:

```python
tidx = tidx % 32
is_signalling_thread = tidx < cute.size(cluster_shape_vmnk)
```

For single-CTA case (`cluster_shape_vmnk = (1,1,1,1)`, size=1), only **lane 0 of each warp** signals empty-arrive. So `consumer_release` is a no-op for non-lane-0 threads (`if_generate(self.is_signalling_thread, ...)` guard at lines 552-557). This is fine for correctness — only one thread per warp needs to signal.

### Warp-spec is convention, NOT API

There is **NO hard producer/consumer warp_idx distinction** in the API. The V2-TMA design pattern (`if warp_idx == 0` is producer, `if warp_idx > 0` are 8 consumers) is enforced by user-side code, not by `PipelineTmaAsync`. The pipeline class only knows about thread counts (`producer_group.size`, `consumer_group.size`).

---

## §2 Three feasible (D) "abandon warp-spec" designs

Background: V2-TMA hit `Achieved Occupancy 18.58 %` because warp-spec (1 producer + 8 consumer warps) limits active warps/sched to 2.26. V0 standalone had 10.28 active warps/sched. The 4× per-instruction-stall improvement TMA brings (42 → 10 cyc) was canceled by the 5× warp-loss. The (D) hypothesis: drop warp-spec, give all 16 warps equal status; keep TMA's low stall AND restore V0's occupancy.

### (D-1) Single PipelineTmaAsync, 16 warps all produce-and-consume — ⚠️ BROKEN

> **WARNING (2026-05-04)**: D-1 has been **empirically proven to deadlock** on SM120
> (RTX PRO 5000, cute 4.4.2). Three independent test kernels using this pattern all
> hung. The root cause is fundamental to how `PipelineTmaAsync` works — not a coding
> error. See analysis below and
> [`gdn-chunk-fwd-pitfalls.md`](../../pitfalls/cutedsl/gdn-chunk-fwd-pitfalls.md)
> traps #9-#10.

Original (broken) design:
```python
load_pipeline = PipelineTmaAsync.create(
    barrier_storage=load_mbar_ptr,
    num_stages=2,
    producer_group=CooperativeGroup(Agent.Thread, 1),       # any 1 thread issues TMA
    consumer_group=CooperativeGroup(Agent.Thread, 16*32),   # all 512 threads consume
    tx_count=tx_count_total,
    cta_layout_vmnk=cute.tiled_divide(cute.make_layout((1, 1, 1, 1)), (1,)),
    defer_sync=True,
)
prod_state = pipeline.make_pipeline_state(PipelineUserType.Producer, 2)
cons_state = pipeline.make_pipeline_state(PipelineUserType.Consumer, 2)
while col_chunk < num_col_chunks:
    if cute.arch.thread_idx()[0] == 0:                  # ← DEADLOCK: breaks elect_one
        load_pipeline.producer_acquire(prod_state)
        bar = load_pipeline.producer_get_barrier(prod_state)
        cute.copy(tma_atom_attn, ..., tma_bar_ptr=bar)
        cute.copy(tma_atom_gate, ..., tma_bar_ptr=bar)
        prod_state.advance()
    load_pipeline.consumer_wait(cons_state)             # all threads wait
    # ... read SMEM, compute, write fp4
    load_pipeline.consumer_release(cons_state)          # signalling thread arrives
    cons_state.advance()
    col_chunk += 1
```

**Why D-1 deadlocks** (two independent root causes):

1. **`if tidx == 0:` breaks `elect_one`**: `producer_acquire` and `cute.copy` for TMA
   internally use CuTeDSL's `elect_one` warp-level primitive (helpers.py:321-324),
   which requires ALL 32 threads of a warp to reach the call site together. Wrapping
   in `if tidx == 0:` creates intra-warp divergence — only 1 of 32 threads enters,
   hanging the warp-collective.

2. **Even with warp-level guard, N warps arriving breaks the barrier math**: If you
   "fix" the thread-guard to a warp-guard (`if warp_idx == 0:` — correctly letting all
   32 threads of warp 0 enter), D-1 still breaks when **all warps** call
   `producer_acquire`. Each warp's `elect_one` issues one `arrive_and_expect_tx`.
   With `producer_group.size = 1`, the full barrier's `arrive_count = 1`, but N warps
   each contribute one arrival → arrival count underflow. Additionally each arrival
   adds `tx_count` to expected bytes, but TMA only issues the data once → N× expected
   vs 1× actual → barrier never trips.

**The original caveats below were incorrect** — they assumed `if thread_idx == 0:` was
a valid guard. It is not, for the `elect_one` reason above.

**Conclusion**: D-1 is NOT a viable design pattern for PipelineTmaAsync. Use warp
specialization (§2.5 below) or D-2 (per-warp pipeline, untested).

### (D-2) 16 PipelineTmaAsync instances, one per warp

```python
mbar_slot_size = 2 * num_stages   # Int64 entries per warp (32 B per warp)
warp_idx = cute.arch.warp_idx()
warp_mbar_ptr = base_mbar_ptr + warp_idx * mbar_slot_size

load_pipeline = PipelineTmaAsync.create(
    barrier_storage=warp_mbar_ptr,                        # per-warp pointer
    num_stages=2,
    producer_group=CooperativeGroup(Agent.Thread, 1),     # 1 thread per warp issues
    consumer_group=CooperativeGroup(Agent.Thread, 32),    # warp's 32 threads consume
    tx_count=per_warp_tx_count,
    ...
)
# Each warp owns its own (gate_tile_slice, attn_tile_slice) and runs producer + consumer in lockstep.
```

**Advantage**: 16-stream pipeline — warps don't synchronize with each other, only within their own warp. Higher concurrency potential than (D-1).

**Caveats**:
- Each warp computes its own slice of (attn, gate) tile coordinates and issues its own TMA descriptors. Thread-0-of-warp issues `cute.copy`.
- 16 separate `tx_count` values, 16 separate SharedStorage mbar fields. SharedStorage struct gets uglier but is straightforward.
- **No cross-warp barrier needed** — warp k's stage k arrival is independent of warp k+1.

### (D-3) — RULED OUT

There is NO hard producer/consumer warp_idx distinction in the cute-DSL pipeline API. The V2-TMA convention (warp 0 = producer, warps 1-8 = consumer) is a user pattern, not an API requirement. So "you cannot do non-warp-spec on this API" is FALSE — both (D-1) and (D-2) are possible.

### §2.5 Validated working TMA pattern on SM120 (warp specialization)

Empirically verified on SM120 RTX PRO 5000 (cute 4.4.2): correctness PASS (max_err=0.0)
on a 32×128 bf16 TMA G2S copy test with 128 threads (4 warps).

**Key rules**:
1. **Warp-level guard, not thread-level**: `if warp_id == 0:` (all 32 threads enter) — NEVER `if tidx == 0:`
2. **Only producer warps call producer methods**: `producer_acquire`, `cute.copy`, `producer_commit`
3. **Only consumer warps call consumer methods**: `consumer_wait`, `consumer_release`
4. **ALL warps participate in computation** after `__syncthreads()` barrier

```python
PRODUCER_WARPS = 1
CONSUMER_WARPS = NUM_WARPS - PRODUCER_WARPS

warp_id = tidx // cutlass.Int32(32)
is_producer = warp_id == cutlass.Int32(0)
is_consumer = warp_id >= cutlass.Int32(1)

load_pipeline = pipeline.PipelineTmaAsync.create(
    barrier_storage=load_mbar,
    num_stages=1,
    producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, PRODUCER_WARPS),
    consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, CONSUMER_WARPS),
    tx_count=tx_count,
    cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
)
pipeline_init_arrive(cluster_shape_mn=(1, 1))
pipeline_init_wait(cluster_shape_mn=(1, 1))

prod_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
cons_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)

# --- producer (warp 0 only) ---
if is_producer:
    load_pipeline.producer_acquire(prod_state)
    bar = load_pipeline.producer_get_barrier(prod_state)
    cute.copy(tma_atom, tSgA[(None, tile_idx)], tSsA[(None, 0)], tma_bar_ptr=bar)
    load_pipeline.producer_commit(prod_state)       # NOOP for TMA, but call for API symmetry
    prod_state.advance()

# --- consumer (warps 1+ only) ---
if is_consumer:
    load_pipeline.consumer_wait(cons_state)

cute.arch.barrier()     # ALL warps sync — data is now in SMEM

# --- ALL warps participate in computation / SMEM reads ---
per_thr = total_elements // THREADS     # must be integer!
for i in cutlass.range_constexpr(per_thr):
    idx = i * cutlass.Int32(THREADS) + tidx
    # ... read sA[idx], compute, write output ...

if is_consumer:
    load_pipeline.consumer_release(cons_state)
    cons_state.advance()

cute.arch.barrier()
```
**Performance caveat**: This pattern works correctly but may regress vs cp.async for
small tiles (≤16KB) with serial dependency. Measured 55% regression on GDN chunk forward
(8KB tiles, 192 serial chunks). See
[`gdn-chunk-fwd-pitfalls.md`](../../pitfalls/cutedsl/gdn-chunk-fwd-pitfalls.md)
trap #12 for when TMA beats cp.async and when it doesn't.

Reference kernel (verified PASS):
`kernel_opt_gdn/test_tma_warp_spec.py`
in the working tree (not checked into reference-kernels).

---

## §3 When (D) actually matters

(D) was conceived as an exploratory probe to break the standalone-fused-quant 91.9% memcpy ceiling. **It was not executed in Stage 3** because:

1. V0/V1/V2 evidence (in `stage3-closeout.md` §3) showed that standalone fused-quant is at the memcpy wall regardless of architecture (3 variants within 1.4% of each other).
2. team-lead approved (B) attention fusion as the V3 deliverable, redirecting effort from "tactical (D) probe" to "structural attention fusion".
3. V3 attention fusion was then blocked on cutlass 4.4.2 vs 4.5+ (see `wiki_drafts/sm120-cutedsl-vendor-pitfalls.md`), and shipped as V3 hybrid (1.07× — see `stage3-closeout.md` §6/§7).

**(D) might matter in future stages if**:
- **Stage 4 needs to push the standalone fused-quant box further**, e.g. for a different shape where the memcpy ceiling is not yet hit, or to verify the (D-1) ~70% per-sched throughput math empirically.
- **Stage 4 reactivates the V3 true fusion plan** (`wiki_drafts/v3-fa-fusion-deferred-plan.md`) and chooses between warp-spec and non-warp-spec for the new fused FA epilogue. In that case (D-1) would be the starting point, with (D-2) as an upgrade if Eligible Warps Per Sched is still < 1.5 after (D-1).

**(D) does NOT matter if** the V3 attention fusion lands successfully — that path eliminates the 50 MB attn_out DRAM round-trip entirely, dwarfing any standalone-kernel optimization.

---

## §4 Recommendation if (D) probe is run

1. **Try (D-1) first** — smaller code delta from V2-TMA. Just delete the `if warp_idx == 0:` warp-spec wrapper, change `consumer_group` size from `8 × 32 = 256` to `16 × 32 = 512`, change consumer-side computation to cover the whole tile (each warp owns 2× the SF blocks compared to V2 which had 8 consumer warps). SharedStorage struct stays flat.
2. **Direct A/B test against V2-TMA**. Same TMA atom, same SWIZZLE_128B, same `tx_count`. Whatever delta you measure is purely "warp-spec vs no-warp-spec at same SMEM/CTA work amount".
3. **Only if (D-1) lands but doesn't break the ceiling** — try (D-2) for the per-warp pipeline concurrency.

**Estimated cost**: ~1 day patch from V2-TMA codebase (commit `71f84d8`).

---

## §5 Wiki value

This file's main value is the **API correctness reference** in §1 (precise line ranges in cute-DSL 4.4.2 source) and the **structural finding** that warp-spec is a user convention, not an API requirement. Future writers of multi-warp TMA kernels on sm_120 cute-DSL 4.4.2 (or any later version where the API doesn't change) can cite §1 directly. The (D-1)/(D-2)/(D-3) design analysis in §2 is preserved for whoever picks up that thread.
---

## related

- `wiki_drafts/stage3-closeout.md` — full Stage 3 narrative explaining why (D) was deferred
- `wiki_drafts/sm120-tma-warp-spec-pitfalls.md` — 5 cute 4.4.2 traps from V2-TMA implementation; will likely apply to (D) implementation too
- `wiki_drafts/sm120-ncu-l1-hit-rate-includes-shared.md` — the metric pitfall to remember when profiling (D)
- `wiki_drafts/v3-fa-fusion-deferred-plan.md` — the deferred plan that supersedes (D) once cutlass upgrades
- `wiki_drafts/sm120-flash-attn-vllm-no-fast-path.md` — why Stage 3 ended at 1.07× despite all this analysis
- Source: `CUTLASS $CUTLASS_DIR/python/CuTeDSL/cutlass/pipeline/helpers.py:25-105` (Agent enum + CooperativeGroup + PipelineOp + SyncObject + MbarrierArray)
- Source: `CUTLASS $CUTLASS_DIR/python/CuTeDSL/cutlass/pipeline/sm90.py:368-559` (PipelineTmaAsync class with create / producer_acquire / producer_commit / consumer_release)
