# GDN Chunk Forward Pitfalls (CuTeDSL, SM120)

Traps from optimizing `chunk_gated_delta_rule` forward on RTX PRO 5000 (sm_120)
with CuTeDSL. Accumulated over V0-V113, from 3.72 ms to 0.531-0.533 ms and
finally 1.51x faster than same-process FLA on the directional
`output_final_state=True` contract at `B=1,T=6144,H=16,HV=32,K=128,V=128`.

Related:
- Optimization report:
- Reference kernel:
- GDN decode pitfalls (complementary): [`gdn-decode-pitfalls.md`](gdn-decode-pitfalls.md)

---

## 1. cp.async bf16 val_layout must be (1,8) not (1,4) for 128-bit copies

**Trap**: You set `num_bits_per_copy=128` and pick `val_layout=(1,4)` because you're
used to fp32 where 4 elements = 128 bits. Seems right for bf16 too — 4 elements per
thread per copy.

**Result**: Compile-time error:
```
'cute.copy' op src ptr alignment (64 bits) does not meet requirement (128 bits)
of atom 'simt_async_copy<bf16, cache = global, 128 b>'
```
Error persists across all thr_layout permutations and SMEM alignment changes. Four
consecutive debugging attempts failed before identifying val_layout as root cause.

**Why**: `val_layout` tells CuTeDSL how many contiguous elements each thread handles
per copy invocation. For bf16 (16 bits), `val_layout=(1,4)` = 4 x 16 = 64 bits per
thread. CuTeDSL computes per-thread pointer alignment from val_layout, not from the
copy atom. 64 bits < 128-bit copy requirement = alignment error at compile time,
regardless of runtime address alignment.

Formula: `val_elements = num_bits_per_copy // element_bits`.
- bf16: 128 // 16 = **8** → `val_layout = (1, 8)`
- fp32: 128 // 32 = **4** → `val_layout = (1, 4)`

**Lesson**: Always compute val_layout from copy width and element size. For bf16
128-bit: `(1, 8)`. Working config for 32x128 bf16 with 128 threads:
```python
thr_layout = (32, 4), stride=(4, 1)   # 128 threads
val_layout = (1, 8), stride=(8, 1)    # 8 bf16 = 128 bits per copy
# tiles 32x32 per copy, K_DIM/32 = 4 copies per thread
```

Also see [gdn-decode-pitfalls.md](gdn-decode-pitfalls.md) trap #2 (`assumed_align=16`
for cp.async) — related but distinct: that trap is about the `from_dlpack` alignment
declaration, this one is about the TiledCopy val_layout shape.


## 2. SMEM row stride must be multiple of 8 bf16 for 128-bit cp.async alignment

**Trap**: You add +2 padding to SMEM stride for bank conflict avoidance (common pattern):
`KQ_STRIDE = K_DIM + 2 = 130`. Then enable cp.async 128-bit — should work because
global addresses are 16B-aligned.

**Result**: Silent misalignment or compile error. The cp.async destination address in
SMEM lands on a 4-byte boundary (130 x 2 = 260 bytes per row, not a multiple of 16).

**Why**: cp.async 128-bit requires the SMEM destination pointer to be 16-byte (128-bit)
aligned for every row. With stride=130 bf16 elements, row start = 130 x 2 = 260 bytes.
260 % 16 = 4 ≠ 0. The stride must be a multiple of 8 bf16 elements (= 16 bytes).

**Lesson**: `KQ_STRIDE = K_DIM + 8 = 136` (not +2). Rule: for N-bit cp.async with
M-bit elements, stride padding must be a multiple of `N / M` elements. For 128-bit
bf16: multiple of 8. This also provides bank conflict reduction (8 > 2 padding).


## 3. q_norm/k_norm need (B,H,T,K) layout for rank-2 slice alignment

**Trap**: You allocate `q_norm` as `(B,T,H,K)` (matching input tensor layout) and slice
with `mQnorm[(i_b, None, i_h, None)]` for cp.async. Seems natural — matches the
input tensor convention.

**Result**: The rank-2 slice `(T,K)` is NOT contiguous because H is sandwiched between
T and K in memory. CuTeDSL generates strided loads, defeating cp.async's contiguous
requirement.

**Why**: cp.async requires contiguous GMEM source. With `(B,T,H,K)`, the slice
`[i_b, :, i_h, :]` has stride `(H*K, 1)` in the T dimension — contiguous only if
H=1. With `(B,H,T,K)`, the slice `[i_b, i_h, :, :]` is physically contiguous
(stride (K, 1) in (T, K)).

**Lesson**: When using cp.async TiledCopy, the sliced dimensions must be innermost.
For rank-2 `(T, K)` slices indexed by `(i_b, i_h)`, use layout `(B, H, T, K)`.
K0 writes to `mKnorm[i_b, i_h, t0+r, c]`; K1 reads via
`gK_full = mKnorm[(i_b, i_h, None, None)]`.


## 4. cute.compile warm call drops Constexpr args — pass only tensors + Int32

**Trap**: You pass `cutlass.Constexpr[int]` values (T, H, HV, etc.) to the compiled
kernel's warm call, matching the cold compile signature.

**Result**: Silent garbage output or crash. The warm call ignores Constexpr args
entirely — they were baked into the compiled kernel during `cute.compile`. Extra
positional args shift tensor bindings.

**Why**: `cute.compile` captures all `Constexpr[...]` values at compile time and
eliminates them from the runtime signature. The warm call only accepts runtime
arguments: `cute.Tensor` (via `from_dlpack`) and `cutlass.Int32`.

**Lesson**: Cold compile: pass everything (tensors + Int32 + Constexpr).
Warm call: pass ONLY tensors + Int32, in the same order, omitting all Constexpr args.
```python
# Cold compile
compiled = cute.compile(launch_fn, mK, mQ, mO, cutlass.Int32(B), T, H, NT)
# Warm call — no T, H, NT
compiled(mK, mQ, mO, cutlass.Int32(B))
```


## 5. @cute.jit helper scoping has ZERO effect on register pressure

**Trap**: You factor accumulator-heavy code into `@cute.jit` helper functions, expecting
the compiler to limit accumulator live ranges to the helper's scope.

**Result**: No change in register count or spill traffic. Measured across three
independent attempts (V5a, V9, V12a of the broader DeltaNet optimization campaign).

**Why**: `@cute.jit` functions are always inlined before LLVM live-range analysis. The
compiler sees one flat function body regardless of source-level scoping. Python-level
function boundaries have no effect on register allocation.

**Lesson**: Don't waste time refactoring for register relief via helpers. The only
ways to reduce register pressure in CuTeDSL are: (a) reduce algorithmic live values,
(b) reduce tile sizes, (c) SMEM stride padding to reduce bank conflicts (which
reduces spill cycles, not register count).


## 6. v-outer/t-inner state update causes 64-register hoisting

**Trap**: You write the state update as v-outer, t-inner (natural for "accumulate
into state[v]"):
```python
for v in range(BV):           # outer
    for t in range(BT):       # inner
        state[v] += sK[t, tidx] * sExpDecay[t] * sV[t, v]
```
Reads nicely — inner loop accumulates into a single `state[v]`.

**Result**: ptxas hoists all BT values of `sK[t, tidx]` (32 regs) and `sExpDecay[t]`
(32 regs) out of the v-loop because they're v-independent. Total: 64 extra registers
hoisted, pushing the kernel to 255 reg/thr with massive spill traffic.

**Why**: `sK[t, tidx]` and `sExpDecay[t]` don't depend on `v`. With v-outer, ptxas
decides to preload all 32 t-indexed values before entering the v-loop, keeping them
alive across all BV iterations. With t-outer, only 2 values (`kd`, current `sExpDecay`)
are live per iteration.

**Lesson**: Swap to t-outer/v-inner:
```python
for v in range(BV):
    state[v] = phi * state[v]     # decay first
for t in range(BT):               # outer
    kd = sK[t, tidx] * sExpDecay[t]  # 2 regs
    for v in range(BV):           # inner — contiguous sV access
        state[v] += kd * sV[t, v]
```
Bonus: inner v-loop reads `sV[t_fixed, v]` contiguously (stride=2B) vs old pattern's
`sV[t, v_fixed]` (stride=68B).


## 7. High-pressure kernel + new long-lived iterator = reg-alloc collapse

**Trap**: Your kernel is at 255 reg/thr with >100M spill instructions. You add a
cp.async TiledCopy, swizzle layout, or other construct that creates a new long-lived
iterator. Should just be a few extra registers.

**Result**: cute.compile drops the 255 reg cap entirely and spill count jumps 6x.
Confirmed twice: V10 (swizzle iterator), V11 (first cp.async attempt). Both had to
be reverted.

**Why**: When the kernel is already at the register ceiling, any new long-lived
iterator (TiledCopy state, swizzle offset) forces ptxas to choose between the cap
and the iterator. It drops the cap, allocates freely, and spill traffic explodes
because the register file can't hold everything.

**Lesson**: Fix register pressure FIRST (via loop restructuring, tile size reduction,
or stride padding), THEN add iterators. In this project: V1 (t-outer + K2 fusion)
dropped regs from 255 to ~198, creating headroom for V3's cp.async TiledCopy.

**Exception**: Pure stride padding (SMEM `+8` instead of `+2`) does NOT create new
iterators and is safe even at high register pressure. V16' confirmed: pad reduced
reg/thr from 255 to 198 without triggering collapse.


## 8. Fusing K2 into K1 requires acc_qS persistence through Neumann solve

**Trap**: You try to fuse chunk_o (K2) into K1 by writing O_cross to GMEM after the
qS MMA, then reading it back after the Neumann solve for the final output combine.
Seems clean — separate the cross-chunk and intra-chunk paths.

**Result**: Extra GMEM round-trip costs ~0.23ms and nearly negates the fusion benefit
of eliminating the K2 kernel launch.

**Why**: acc_qS (8 fp32 regs) can simply stay alive in registers through the Neumann
solve stage. The Neumann solve uses its own accumulators (acc_vnew: 8 regs). Total
MMA register pressure during the overlapped section: acc_qS(8) + acc_vnew(8) + MMA
fragments = ~24 regs. Well within budget after loop restructuring.

**Lesson**: When fusing kernels, audit which intermediate values can stay in registers
vs need GMEM staging. In K1, acc_qS is only 8 fp32 regs and is consumed 2 stages
later — no reason to spill it. The gated epilogue `acc_qS[idx] * sExpGC[t] * scale`
combines naturally with the intra-chunk MMA result.

## 9. `if tidx == 0:` guard on TMA producer causes deadlock (elect_one)

**Trap**: You wrap TMA producer code (`producer_acquire`, `cute.copy`, `producer_commit`)
in `if tidx == 0:` — only thread 0 should issue the TMA DMA, and this is the standard
pattern for single-thread operations.

**Result**: Kernel hangs (deadlock). No GPU error, just infinite spin. Happens with both
raw `mbarrier_arrive_and_expect_tx` and `PipelineTmaAsync.producer_acquire`. Three
independent test kernels (raw mbarrier, PipelineTmaAsync minimal, PipelineTmaAsync
warp-spec) all deadlocked with this guard.

**Why**: CuTeDSL's `producer_acquire` and `cute.copy` for TMA internally use the
`elect_one` mechanism (helpers.py:321-324), which is a warp-level primitive that selects
one thread per warp. `elect_one` requires **ALL 32 threads of the warp** to reach the
call site together — it uses `__ballot_sync` or equivalent warp-collective. Wrapping in
`if tidx == 0:` causes only 1 of 32 threads to enter, creating intra-warp divergence
that hangs the warp-collective forever.

CuTeDSL documents this indirectly: algorithm.py:415-416 says "For Copy Atoms requiring
single-threaded execution, thread election is managed automatically by the copy
operation." Translation: don't do your own thread election — `elect_one` inside
`cute.copy` handles it.

**Lesson**: Always use warp-level guards for TMA producer code:
```python
warp_id = tidx // cutlass.Int32(32)
is_producer = warp_id == cutlass.Int32(0)
# NOT: if tidx == 0:
if is_producer:   # all 32 threads of warp 0 enter together
    load_pipeline.producer_acquire(prod_state)
    bar = load_pipeline.producer_get_barrier(prod_state)
    cute.copy(tma_atom, tSgA[(None, tile_idx)], tSsA[(None, 0)], tma_bar_ptr=bar)
    load_pipeline.producer_commit(prod_state)
    prod_state.advance()
```

Source: upstream CUTLASS CuTeDSL source, `cutlass/cute/algorithm.py:415-416` and `cutlass/pipeline/helpers.py:321-324`.


## 10. D-1 "all-warps-produce-and-consume" is broken with PipelineTmaAsync

**Trap**: You design a TMA kernel where all warps both produce and consume (the D-1
pattern from `sm120-pipeline-tma-async-api-notes.md`). Every warp calls
`producer_acquire` → `cute.copy` → `producer_commit`, then all warps call
`consumer_wait`. Seems elegant — no warp specialization needed, all warps stay busy.

**Result**: Deadlock. The full mbarrier never completes its wait phase.

**Why**: `producer_acquire` calls `sync_object_full.arrive()`, which maps to
`arrive_and_expect_tx(index, tx_count)` via `elect_one` (helpers.py:258-260). This
fires **once per warp** (elect_one picks one thread per warp). With N warps calling
`producer_acquire`:

1. **Arrival count underflow**: `arrive_count = producer_group.size` (helpers.py:172).
   If `producer_group = CooperativeGroup(Agent.Thread, 1)`, arrive_count = 1. But N
   warps each contribute one arrival → (N-1) excess arrivals → mbarrier phase flips
   prematurely or corrupts.
2. **N× tx_bytes**: Each warp's `arrive_and_expect_tx` adds `tx_count` to the expected
   bytes, but TMA only issues data once. Expected bytes = N × actual bytes → barrier
   never trips.

Even setting `producer_group.size = N` doesn't help: you'd need exactly N warps to
arrive, but `cute.copy` is also called N times, issuing N× TMA transactions to the
barrier.

**Lesson**: PipelineTmaAsync requires **warp specialization** — only the designated
producer warp(s) call `producer_acquire` / `cute.copy` / `producer_commit`. The D-1
pattern (all-warps-produce-and-consume) does NOT work. Use:
```python
producer_group = CooperativeGroup(Agent.Thread, PRODUCER_WARPS)   # e.g., 1
consumer_group = CooperativeGroup(Agent.Thread, CONSUMER_WARPS)   # e.g., NUM_WARPS - 1
```
Only threads in producer warps enter the producer code path. Only threads in consumer
warps call `consumer_wait` / `consumer_release`.

Source: upstream CUTLASS CuTeDSL source, `cutlass/pipeline/helpers.py:172,258-260` and `cutlass/pipeline/sm90.py:519-538`.


## 11. `producer_commit` is a NOOP for TMA — don't debug its internals

**Trap**: After calling `cute.copy(tma_atom, ..., tma_bar_ptr=bar)`, you call
`load_pipeline.producer_commit(prod_state)` and wonder whether it does the transaction
count update. When debugging deadlocks, you spend time trying to understand how
`producer_commit` interacts with the mbarrier.

**Result**: Wasted debugging time. `producer_commit` does nothing for TMA.

**Why**: sm90.py:541-545:
```python
def producer_commit(self, state, ...):
    # producer_commit is a NOOP for TMA async producer
    # The tma instructions directly update the barrier's
    # transaction count.
    pass
```
The TMA instruction (`cp.async.bulk.tensor`) directly injects bytes into the mbarrier
via the `tma_bar_ptr` parameter. There is no separate commit step. `producer_commit`
exists only for API symmetry with non-TMA producers.

**Lesson**: For TMA, the actual synchronization flow is:
1. `producer_acquire(state)` — waits for empty barrier, then arrives on full barrier
   (which triggers `arrive_and_expect_tx`)
2. `cute.copy(tma_atom, ..., tma_bar_ptr=bar)` — issues TMA DMA, bytes auto-arrive at bar
3. `producer_commit(state)` — **NOOP**
4. `consumer_wait(state)` — spins on full barrier until TMA bytes == expected bytes
5. `consumer_release(state)` — signals empty barrier

Source: upstream CUTLASS CuTeDSL source, `cutlass/pipeline/sm90.py:541-545`.


## 12. TMA + warp-spec on small tiles with serial dependency = performance regression

**Trap**: You replace cp.async with TMA for K/Q loads in the GDN chunk kernel, expecting
TMA's hardware DMA engine to be faster. TMA is the "modern" approach on Blackwell and
should outperform per-thread cp.async.

**Result**: 55% performance regression. P50 went from 1.1047ms (cp.async V1) to
1.7119ms (TMA V4). Correctness passes perfectly — purely a performance problem.

**Why**: Three compounding factors:

1. **Tiles too small for TMA overhead**: Each TMA copy is 8KB (BT=32 × K_DIM=128 ×
   2 bytes/bf16). TMA's DMA setup (descriptor preparation, mbarrier protocol,
   `arrive_and_expect_tx`) dominates at this size. cp.async's lightweight per-thread
   `cp.async.cg.shared.global [dst], [src], 16` has near-zero setup cost.

2. **No overlap opportunity**: The GDN kernel has 192-chunk serial dependency (state
   carries across chunks). You cannot double-buffer (load chunk N+1 while computing
   chunk N) because chunk N+1's computation depends on chunk N's state update. TMA's
   async DMA advantage is wasted when there's nothing to overlap with.

3. **Warp specialization overhead**: Dedicating warp 0 as a pure producer (only issues
   TMA) adds pipeline protocol overhead (mbarrier arrive/wait cycles) without benefit.
   In the cp.async version, all 128 threads cooperatively load K/Q with near-perfect
   utilization.

**Lesson**: TMA is NOT universally faster than cp.async. TMA wins when:
- Tiles are large (≥64KB, ideally ≥128KB)
- Double-buffering or multi-stage pipelining is possible
- Producer work can overlap with consumer computation

cp.async wins when:
- Tiles are small (≤16KB)
- Serial dependency prevents pipelining
- All threads need to stay busy (no warp to spare as producer)

Rule of thumb: if your tile fits in ≤2 L2 cache lines per thread, cp.async is likely
faster than TMA.


## 13. Consumer thread count must divide SMEM element count for copy-out

**Trap**: With warp specialization (1 producer + 3 consumer warps), you have consumer
warps (96 threads) copy data from SMEM to GMEM. The copy loop is:
```python
per_thr = (M * N) // CONSUMER_THREADS   # 4096 / 96 = 42.67 ← NOT integer!
```

**Result**: Incorrect output (max_err = 3.234375). Some SMEM elements are not copied,
others are double-copied.

**Why**: With 96 consumer threads and 4096 elements, each thread should copy 42.67
elements — not an integer. The for-loop skips the remainder, leaving elements uncopied.

**Lesson**: After the `consumer_wait` + `__syncthreads()` barrier, ALL threads (including
producer warp) should participate in SMEM reads. The TMA producer warp has finished its
work at this point — it's waiting at the barrier anyway:
```python
if is_consumer:
    load_pipeline.consumer_wait(cons_state)
cute.arch.barrier()      # <-- all warps sync here
# ALL 128 threads participate in computation and SMEM reads
per_thr = (M * N) // THREADS   # 4096 / 128 = 32 ← integer, correct
```

## 14. Direct global V for RHS is slower than cp.async staging

**Trap**: After removing the dead `v_new -> sV` store, original V is only used once
to build RHS. It is tempting to skip `mV -> sV` cp.async and read `gV_tile[row, v]`
directly inside the RHS R2S loop.

**Result**: Correct but slower. V30 repeat-100 regressed to `0.7200 ms` vs V29
`0.6861 ms`; ncu K1 regressed to `743.58 us` vs `706.40 us`.

**Why**: The CuTeDSL scalar `gV_tile[row, v_col]` access was not a vectorized
`CopyUniversalOp` load. It inserted scalar global loads on the RHS critical path
(`LDG.E.U16` rose) while removing only a small amount of shared-load work. cp.async
V staging remained better because it coalesces and overlaps the load before the RHS
dependency chain.

**Lesson**: "One-use" does not automatically mean "load directly from global."  In
this GDN K1, keep V cp.async staging unless the direct path is vectorized and ncu
proves lower long-scoreboard / global-sector cost.


## 15. `STS.U16 == 0` is not the end of shared-memory optimization

**Trap**: Once V28 converted scalar shared stores to R2S/STSM and `STS.U16` reached
zero, it looked like shared-store optimization was done.

**Result**: V29 still missed the FLA 1.3x target: repeat-100 P50 `0.6859 ms`, target
about `0.675 ms`. ncu still showed `LDS.U16=6,291,456`, `LDS.64=1,572,864`,
`STSM=2,752,512`, and `F2FP.BF16.F32.PACK_AB=5,898,240`.

**Why**: The remaining wall was no longer scalar stores. It was the K-decay scratch
materialization path: load `sK`, multiply by `exp_decay`, pack to bf16, store scratch
with STSM, then LDSM-load it back for state-update HMMA.

**Lesson**: After eliminating `STS.U16`, inspect `LDS.U16/LDS.64`, `F2FP`, `SHF`,
`STSM`, and barrier counts. The next win may be removing the scratch dataflow entirely,
not making the stores wider.


## 16. K-decay scratch is algebraically removable

**Trap**: Treat the state update as requiring a materialized K-decay scratch tile:
```text
state += (K[t,k] * exp_decay[t]) @ v_new[t,v]
```

**Result**: Even after R2S, the scratch path consumed millions of shared loads and
pack instructions. It kept K1 at `706.40 us` in V29.

**Why**: The decay multiply is separable over `t` and can be moved to the V operand:
```text
sum_t (K[t,k] * exp_decay[t]) * v_new[t,v]
==
sum_t K[t,k] * (exp_decay[t] * v_new[t,v])
```
V31 stores `v_new * exp_decay` into `sNK_A` after chunk-O and reuses existing `sK`
through a transposed LDSM view.

**Lesson**: For recurrences of the form `K^T @ diag(decay) @ V`, choose the side that
minimizes scratch. In this shape, scaling `v_new` eliminates K-decay scratch and
drops K1 from `706.40 us` to `607.33 us`.


## 17. Transposed `sK` view needs a transposed LDSM atom, not just a layout alias

**Trap**: Build a transposed `sK` tensor view with `stride=(1, K_DIM+8)` and feed it
through the existing non-transposed A-copy atom.

**Result**: CuTeDSL MLIR verification failed:
```text
'cute.copy' op src ptr alignment (16 bits) does not meet requirement (128 bits)
of atom '!cute_nvgpu.atom.ldsm<..., num_matrices = 4, n>'
```

**Why**: The layout alias exposed a 16-bit-aligned source pointer for the LDSM atom
that requires 128-bit alignment. The existing A-copy atom still expected the original
non-transposed access pattern.

**Lesson**: For a transposed shared-memory operand consumed by LDSM, create a dedicated
copy atom:
```python
smem_copy_atom_A_trans = cute.make_copy_atom(
    warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4),
    cutlass.BFloat16,
)
smem_tiled_copy_A_trans = cute.make_tiled_copy_A(smem_copy_atom_A_trans, tiled_mma)
```
Use this atom only for the transposed `sK` state-update MMAs.


## 18. Shared fp32 state-MMA can remove FFMA but still lose

**Trap**: Move the recurrent state into fp32 shared memory and use HMMA-like state
updates to remove scalar FFMA work.

**Result**: The probe was correct, and SASS did remove the old FFMA/SHF/PRMT wall,
but performance regressed to about `1.26 ms` or worse. Registers and shared-memory
traffic collapsed occupancy and scheduler eligibility.

**Why**: Persisting fp32 state in shared memory adds large shared stores/loads and
extra barriers. The old scalar compute wall disappears, but it is replaced by a
shared-memory dependency wall.

**Lesson**: Keep the recurrent state register-resident. Use shared memory only as a
bf16 MMA operand bridge, and make those bridges R2S/LDSM-friendly.


## 19. Wider V tiling and pair-V reuse hit the register cliff

**Trap**: Increase `BV` or have one CTA handle two V tiles to reuse K/Q/M/GQK work.
The arithmetic work per K/Q load looks better on paper.

**Result**: BV32 and pair-V probes regressed. Pair-V reduced some repeated work but
pushed registers near 198 per thread and halved occupancy; BV32 similarly increased
register pressure.

**Why**: The fused K1 already carries state fragments, MMA accumulators, and multiple
shared iterators. Doubling V-side state or output fragments increases long-lived
registers faster than it saves K/Q staging work.

**Lesson**: On this shape, `BV=16` is the stable point. Attack dataflow and scratch
materialization before widening V.


## 20. `CopyUniversalOp` output widening is constrained by accumulator layout

**Trap**: Replace scalar output stores with a 64-bit or 128-bit `CopyUniversalOp`
from the MMA accumulator fragment to GMEM and expect `STG.E.128`.

**Result**: Direct 128-bit and 64-bit output copy attempts failed MLIR verification
with static-stride constraints. A 32-bit direct copy compiled, was correct, and still
helped, but did not produce `STG.E.128`.

**Why**: The accumulator fragment's retiled view did not expose the static contiguous
inner dimension required by the wider universal copy. The GMEM tile is contiguous in
V, but the fragment layout is not the same as a simple row-major `(BT,BV)` tensor.

**Lesson**: In CuTeDSL, vector GMEM stores need both GMEM contiguity and fragment-view
contiguity. If wider copies fail, keep the legal 32-bit copy and look for larger
dataflow wins elsewhere.


## 21. Beating the fast path is not the same as beating FLA by 1.3x

**Trap**: Stop as soon as CuTeDSL beats the local fast path or Triton wrapper.

**Result**: V26-V29 beat the fast path but still missed the stronger FLA 1.3x target.
V29 was `0.6859 ms`, while the recorded target was `<=0.6753 ms`.

**Why**: The fast path and FLA varlen wrappers have different launch and specialization
costs. A kernel can beat one wrapper and still be above the explicit target line.

**Lesson**: For acceptance, run same-process FLA comparison and compute `FLA / current`.
V31 only became final after `0.6142 ms` vs FLA `0.8768 ms`, i.e. `1.4275x`.


## 22. Preprocess cache is not valid when the contract says recompute every execution

**Trap**: Cache K0/K_inv outputs because benchmark inputs are often repeated during
profiling. This makes the fast path look attractive: K1 dominates wall time, and
preprocess tensors are deterministic for a fixed input.

**Result**: Invalid acceptance path. The clarified target requires K0 preprocess
q/k normalization, K_inv beta-fold / Neumann intermediates, and K1 output/final_state
to run every execution. V113 meets the target without that shortcut:
`0.531-0.533 ms` vs FLA `0.804 ms`, about `1.51x`.

**Why**: K0 and K_inv are part of the GDN forward computation for the current input.
Caching them changes the measured operation from "run GDN" to "run K1 using stale
preprocessed state." That hides memory traffic and launch work that production must
pay whenever q/k/g/beta changes.

**Lesson**: Treat preprocess cache, CUDA graph/static replay, and FLA fallback as
separate system-level features, not kernel acceptance optimizations. For this target,
the only valid stop line is the full 3-kernel no-cache path.


## 23. Conditional Bx2/Bx4 copy-atom selection in one launcher breaks tail correctness

**Trap**: Put the fast non-tail B `LdMatrix num_matrices=2` path and the safer B4
tail path behind one conditional launcher. It looks cleaner and avoids maintaining
two K1 launch wrappers.

**Result**: T=50000 output correctness failed with rel_err from `1.5021e-02` to
`4.7210e-02`, even though the non-tail T=6144 path was correct.

**Why**: The two copy-atom shapes induce different static layouts and verifier-visible
assumptions. Hiding them behind a runtime condition is not equivalent to compiling
two statically separate kernels. Tail / `STATE_SPLIT=True` needs the conservative
B4 path.

**Lesson**: Specialize by launcher, not by runtime condition, when the CuTeDSL copy
atom changes shape. V113 accepts `_launch_kernel1_v31_final_state_bx2` for the
T=6144 non-tail path and keeps `_launch_kernel1_v31_final_state` for tail /
`STATE_SPLIT=True`.


## 24. Cache-policy and L2 access-window hints can be correct and still slower

**Trap**: After the core LDSM/R2S dataflow is fixed, try cache hints as a cheap
finishing move: `LoadCacheMode.STREAMING`, or CUDA L2 access-policy windows for
`neumann_m`, `gated_qk`, or both.

**Result**: The K1 `GLOBAL -> STREAMING` probe failed final output correctness
(`rel_err=1.662843e-02`). V121 L2 access-policy-window variants were correct but
slower: `m` = `0.5737 ms`, `gqk` = `0.5805 ms`, combined `mgqk` = `0.5759 ms`,
all behind V113's `0.531-0.533 ms`.

**Why**: K1's remaining cost is not a simple "keep this one tensor resident in L2"
problem. It has serial chunk dependency, LDSM-fed shared-memory staging, and
multiple short-lived intermediates. Access-window hints can perturb replacement
policy and add setup cost without improving the critical path.

**Lesson**: Cache hints require full correctness and same-process timing validation.
Keep `LoadCacheMode.GLOBAL` on the accepted path unless a new probe beats the full
no-cache V113 baseline and passes final-state/tail correctness.


## 25. NCU duration is not production latency when profiling replay inflates K1

**Trap**: Use NCU `gpu__time_duration` from a metric-heavy bandwidth pass as the
wall-clock latency and conclude K1 regressed.

**Result**: V122 NCU showed K1 duration `565.09 us`, while the same V113 one-call
nsys split showed K1 `383.456 us`. The NCU pass still produced useful memory
counters: K1 DRAM `152.45 MB`, DRAM BW `269.78 GB/s`, L2 bytes `1.62 GB`, L2 BW
`2.87 TB/s`, and L2 hit `91.32%`.

**Why**: NCU can replay or perturb launches to collect requested counters. The
duration paired with those counters is valid for counter bandwidth arithmetic, but
it is not a low-overhead production-latency measurement.

**Lesson**: Use nsys or the strict benchmark for latency. Use NCU for counter
breakdown, and label bandwidth as "NCU counter bytes divided by NCU duration."


## 26. Output-only TMA harnesses do not satisfy final-state GDN acceptance

**Trap**: Revisit an older TMA harness after V113 and compare its output path, hoping
TMA/warp-specialization can replace the accepted cp.async path.

**Result**: The existing TMA harness still failed before timing with
`CUDA_ERROR_MISALIGNED_ADDRESS`, and it remained output-only/no-final_state. Even
when earlier TMA megakernel probes were correct for output, they were much slower
than cp.async on the small serial GDN tiles.

**Why**: The final contract includes both output and final state. A TMA harness that
does not implement final_state cannot be an acceptance candidate, and the old
misaligned-address failure means it is not even a timing baseline.

**Lesson**: Do not use output-only TMA code as evidence for the current target.
Any future TMA path must first implement the full output + final_state contract,
then beat V113 with K0/K_inv/K1 recomputed every execution.

---

## Quick reference: do vs don't

| Do | Don't |
|----|-------|
| `val_layout = (1, num_bits_per_copy // element_bits)` | Hardcode `(1, 4)` for all dtypes |
| `KQ_STRIDE = K_DIM + 8` (multiple of 8 for bf16 128-bit) | `KQ_STRIDE = K_DIM + 2` with cp.async |
| `(B, H, T, K)` layout for cp.async source tensors | `(B, T, H, K)` — non-contiguous slice |
| Warm call: tensors + Int32 only | Pass Constexpr to warm call |
| t-outer/v-inner for v-independent terms | v-outer/t-inner with v-independent inner loads |
| Fix reg pressure before adding iterators | Add TiledCopy at 255 reg/thr |
| Keep small accumulators alive across stages | Round-trip through GMEM for 8 regs |
| SMEM stride padding (+8 bf16) for bank conflicts | @cute.jit helpers for register relief |
| `if warp_id == 0:` (warp-level) guard for TMA producer | `if tidx == 0:` (thread-level) — breaks elect_one |
| Warp specialization for PipelineTmaAsync | D-1 all-warps-produce-and-consume — arrival underflow |
| cp.async for small tiles (≤16KB) with serial dependency | TMA for small tiles — DMA setup overhead dominates |
| ALL threads participate in SMEM reads after barrier | Only consumer warps copy — non-integer division |
| Keep V cp.async staging unless direct V is vectorized and profiled | Scalar direct global V in RHS critical path |
| Move `exp_decay` to `v_new` to remove K-decay scratch | Materialize `K * exp_decay` scratch after `STS.U16` is already gone |
| Use a dedicated `transpose=True` LDSM atom for transposed `sK` | Feed transposed shared view through the normal LDSM A-copy atom |
| Keep recurrent state register-resident | Persist fp32 state in shared memory for state-MMA |
| Treat the strict same-process FLA speedup target as the stop line | Stop at "faster than fast path" |
| Recompute K0/K_inv/K1 for the no-cache target | Cache preprocess or replay stale intermediates |
| Split Bx2 non-tail and B4 tail launchers | Hide incompatible copy atoms behind one runtime condition |
| Use nsys/strict bench for latency and NCU for counters | Treat NCU metric duration as production wall time |
