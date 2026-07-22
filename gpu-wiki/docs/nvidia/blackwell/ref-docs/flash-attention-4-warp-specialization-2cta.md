# SVDQuant W4A4 on Blackwell: Warp Specialization, TMEM, and 2-CTA Persistent Kernel Using the FA4 Skeleton

A practical walkthrough of building a Blackwell NVFP4 kernel that borrows FlashAttention-4's synchronization skeleton — explicit per-warp pipeline state, warp specialization, and persistent tile scheduling — to achieve +198% TFLOPS from a single SMEM budget fix.

---

## 1. Introduction

The key insight: how to keep a Blackwell kernel alive in a complex pipeline synchronization state space without deadlocking — by borrowing FlashAttention-4's synchronization skeleton (explicit per-warp pipeline state + warp specialization + persistent tile scheduler), rather than writing a state machine from scratch.

Using the `gemm_w4a4` kernel as a case study: refactoring from a 1-CTA CUTLASS example direct-translation into an FA4-derived 2-CTA persistent kernel, and the single-line SMEM accounting bug hidden behind a "looks like it works" smoke test that was worth +198% TF.

The numbers are MFU (percentage of the chip's dense NVFP4 peak). The two sides are not same-generation chips: we run on B200 (SM_100, dense FP4 peak 10 PFLOPS); nunchaku's NVFP4 is gated on `__CUDA_ARCH__ >= 1200` with no SM_100 binary, only running on SM_120a/121a RTX PRO 6000 (4 PFLOPS peak) — two different Tensor Core ISAs, two toolchains, two Blackwell generations. MFU already divides out chip peak, but this table is not for judging which codebase is "better written" — it is merely an implementation quality reference point: "how fast can mature hand-written inline PTX run on its own target chip."

On the same B200, stripping LoRA and affine, running CUTLASS's `dense_blockscaled_gemm_persistent.py` at 2-CTA 256×256 yields 45%–63% MFU as the local ceiling — that is the remaining space actually worth chasing.

This operator is the compute-bound half of SVDQuant: NVFP4 scaled-MMA + small-rank LoRA residual + per-column affine. The math fits in one line; the implementation uses nearly every new primitive that SM_100/SM_103 provides over the previous generation.

Two kernel versions coexist in the repository. v1 (1-CTA, monolithic `@cute.kernel`, stock `cutlass.pipeline.PipelineState`) plateaus at ~27% MFU on production shapes; attempting to upgrade it to 2-CTA via `cta_group=TWO` yields essentially zero gain (28% vs 27%). v2_fa4 (FA4-derived warp specialization, three pipelines, 2-CTA persistent) is the shipping surface that produces the numbers above.

The single highest-ROI line change in the entire project: halving the per-CTA SMEM byte estimate for LoRA-up weight blocks in 2-CTA mode. The kernel solves an SMEM budget problem at trace time — "given this much shared memory per SM, how many K-blocks can the main K-loop keep in-flight? How many stages can the LoRA prefetch pipeline run?" The LoRA-up calculation had a hand-written arithmetic line that missed one thing: in 2-CTA mode, hardware already splits that tile between two CTAs in the cluster, so each CTA's actual on-chip allocation is only half of what the formula computed. The budget solver, given this 2× inflated number, silently cut the main K-loop's concurrent depth from 4 in-flight K-blocks to 2 to "make room" for shared memory that didn't actually exist.

Symptom: no symptom — trace passes, kernel runs, numerics are correct, it just "feels slow."

Fix: divide by the cluster's CTA group size one more time. Production shape wall-clock: 566 TF → 1685 TF (+198%), 4.2% → 16.9% MFU. Same launch config ncu A/B: Duration −31.2%, SM Throughput +11.99 pp, SM Active Cycles −36.3%.

## 2. Why This Operator, Why This Article

**Math:**

```
y = scaled_mma(act₄, wgt₄) · wcscale + bias + lora_act_in @ lora_up
```

Inputs are NVFP4 packed format (act, wgt: `[M, K/2]` uint8, each byte stores two E2M1 nibbles; ascales, wscales: `[K/16, *]` FP8-E4M3, one scale per 16 K-elements). `lora_act_in @ lora_up` is a small-rank R residual (production R ≤ 128, most commonly R=32). `wcscale` and `bias` are per-output-column.

No chained data flow, no softmax, no online correction: one main MMA, one LoRA MMA, one fused affine.

Two design constraints determine everything that follows:

**SM_100/SM_103 only.** Consumer Blackwell SM_120a/121a is covered by nunchaku; this repository exists to cover the data-center Blackwell line. Ampere–Hopper are also out of scope. Therefore the kernel can unconditionally assume tcgen05 scaled-MMA, TMEM, 2-CTA dense MMA, TMA bundling, etc.

**CuTe DSL Python, not CUDA C++.** Python DSL is NVIDIA's official authoring path on Blackwell, using the same `cutlass-dsl` package as upstream. Compared to CUDA C++ CuTe headers, template boilerplate is ~10× less. The actual kernel JITs on first call via MLIR → PTX. Trace-level checks work on any Linux box; real execution requires B200/B300.

**Editorial claim:** For teaching Blackwell primitives, this operator is better suited as a textbook example than FA4. FA4's online softmax and S→P→O chained data flow carry genuine cognitive tax — most of FA4's complexity is actually in attention itself, not in Blackwell. SVDQuant W4A4 strips that layer away: same warp-specialized main loop, same persistent tile scheduler, same tcgen05 accumulator, same TMA bundling, same 2-CTA split — but the math fits on one screen.

## 3. Blackwell Primitives Used by This Kernel

### 3.1 tcgen05.mma Scaled-MMA and NVFP4 Atom

NVFP4 is block-scaled FP4: two E2M1 nibbles packed into one byte as values, plus one FP8-E4M3 scale per 16 K-elements. Effective precision with block scale is approximately 7 bits. Blackwell's `tcgen05.mma.kind::mxf4nvf4.block_scale.scale_vec::4X` atom takes two packed operands plus two scale tensors simultaneously, outputting an FP32 accumulator landing in TMEM.

CuTe DSL exposes this through `make_blockscaled_trivial_tiled_mma(...)`.

Key facts:
- Only MXF4, NVFP4, and MXF8 scaled-MMA are exposed on Blackwell — INT4 scaled-MMA was cut at the ISA level
- Scale lives in TMEM (not SMEM): the kernel `cute.copy`s SMEM → TMEM once per K-block, then issues gemm
- The `tiled_mma.set(tcgen05.Field.SFA, ...)` and `.SFB` runtime entries accept scales

Usage pattern:

```python
tiled_mma.set(tcgen05.Field.SFA, tCtSFA[sf_kblock_coord].iterator)
tiled_mma.set(tcgen05.Field.SFB, tCtSFB[sf_kblock_coord].iterator)
cute.gemm(tiled_mma, tCtAcc, tCrA[kblock_coord], tCrB[kblock_coord], tCtAcc)
```

The first three lines are Python trace-time state modifications on the `tiled_mma` object — they take effect when the subsequent `cute.gemm` is captured in MLIR. The fourth line is the actual `umma.commit` issued on device.

**On "NVFP4" vs cuBLAS NVFP4 linear:** The full NVFP4 spec has two-level scaling — a per-tensor FP32 scale plus a per-16-K-element FP8-E4M3 block scale. The design choice here (following nunchaku) uses single-level only: block scale, with any per-tensor scaling folded offline into block scale (or into wcscale). cuBLAS's NVFP4 linear exposes both levels at runtime. Mathematically equivalent when per-tensor scale is pre-folded; the difference is what the spec brings to the runtime API, not achievable precision.

### 3.2 2-CTA Dense MMA via cta_group=TWO

`cluster_shape=(2, 1)` cluster where two CTAs cooperatively process a larger tile. The atom is constructed with `CtaGroup.TWO`, which inserts a size-2 V (volume) dimension into the MMA thread layout. Each CTA in the pair holds half of the cluster-level work, but every MMA issued by the leader CTA involves both CTAs participating.

Cluster layout factorization into (V, M, N, K):

```
cluster_shape_mn = (2, 1), CtaGroup.TWO:
  cluster_layout_vmnk.shape = ((2,), 1, 1, 1)
  rank=0 → flat coord (0, 0, 0, 0) ← leader CTA
  rank=1 → flat coord (1, 0, 0, 0) ← follower CTA
```

**SMEM benefit:** Under `CtaGroup.TWO`, the MMA atom's `partition_shape_A` halves A along M, and `partition_shape_B` halves B along N. Each CTA needs only half the operand SMEM of a 1-CTA atom — this is the "2xSM MMA: Shared Memory Optimization" from the Modular matmul-on-blackwell-part-3 article. CUTLASS's `dense_blockscaled_gemm_persistent.py` uses it, and v2_fa4's main path uses it for A and B as well.

### 3.3 TMEM — Addressable Accumulator Space

Before Blackwell, MMA accumulators lived in registers. On Blackwell, accumulators reside in Tensor Memory (TMEM) — an SM-local memory with its own allocator (`utils.TmemAllocator`), release barriers, and 512-column-wide layout.

Two direct consequences:

1. **TMEM is shared across threads within a CTA** (unlike registers). After the MMA warp issues `cute.gemm`, any warp group in the CTA can subsequently read TMEM cells via `tcgen05.tmem_load` for epilogue. This is why 4 epilogue warps can read the accumulator written by 1 MMA warp.

2. **Two MMA atoms can target the same TMEM range.** FA4's `blackwell_helpers.gemm_ptx_partial` accepts a raw `acc_tmem_addr: Int32`, not a `cute.Tensor`. With the TMEM address in hand, a second `ACCUMULATE=True` MMA can write to the same address; the second MMA reads exactly the TMEM cells written by the first.

SM_100 TMEM budget is at most 512 columns. NVFP4 block-scaled MMA at 256×128 tile requires 128 columns accumulator + 16 columns SFA + 32 columns SFB ≈ 176 columns. Doubling the accumulator (next-tile ping-pong `overlapping_accum`) fits at tile_n=128 but overflows at tile_n=256. Hence both CUTLASS reference and our kernel write `num_acc_stage = 1 if tile_n == 256 else 2`.

### 3.4 TMA, extra_tx_count Bundling, and is_leader_cta Gate

TMA copies are asynchronous, completed via mbarrier arrive signals. Each TMA adds its delivered byte count to the barrier's `expected_transactions` (tx_count); when accumulated to threshold, the barrier flips full and consumers proceed.

CuTe DSL's `pipeline.PipelineTmaUmma.create` packages this into a producer/consumer pattern. Two SM_100-specific knobs in the wrapper:

**extra_tx_count bundling:** Instead of one TMA per barrier, multiple TMAs' byte counts are added to the same barrier's tx_count — one barrier guards multiple TMAs. In the main K-loop, act + ascales + wgt + wscales (four TMAs) share one barrier, with `tx_count = num_tma_load_bytes`. This saves three mbarrier slots and one barrier wait per stage.

**is_leader_cta gate:** Under `CtaGroup.TWO`, only the leader in the cluster pair calls `arrive_and_expect_tx` (the other CTA's TMA participation is implicit in `tx_count × cta_group_size`). If the follower also arrives, the barrier double-counts and deadlocks. CuTe DSL automatically adds the gate in `PipelineTmaUmma.producer_acquire` based on `cluster_layout_vmnk`.

### 3.5 StaticPersistentTileScheduler

Pattern: launch `min(num_tiles, sm_count)` CTAs, each CTA walks tiles via `tile_idx += grid_dim()`, exits when out of bounds. Saves launch overhead, keeps warps hot across tiles, TMA pipeline carries state across tile boundaries.

Implementation is minimal (~30 lines). The hard part is not the scheduler — it is whether the rest of the kernel's state can survive across tile boundaries. That is the subject of subsequent sections.

### 3.6 Warp Specialization (Preview)

One warp does TMA load (load warp), one warp issues MMA (mma warp), four warps run epilogue (epilogue warps). Total: 6 warps × 32 threads = 192 threads per CTA. Each warp has its own pipeline state, advancing independently — there is no "kernel-global state." This is the structural pattern FA4 established for tcgen05 + TMEM kernels.

## 4. v1 — The Pre-FA4 Baseline

v1 is the baseline: main NVFP4 scaled-MMA + β-interleaved LoRA, 1-CTA only, monolithic `@cute.kernel`, stock `cutlass.pipeline.PipelineState`.

Origin: ported from `cutlass/examples/python/CuTeDSL/blackwell/dense_blockscaled_gemm_persistent.py`, stripping persistent TileScheduler, clusters > 1, TMA multicast, overlapping_accum, and tile_n ∈ {64, 192} SFB-shift workarounds. Two shape-adaptive 1-CTA tilers: small M uses (128, 128), otherwise (128, 256).

### 4.1 What v1 Did Well

- **Numerically clean.** {fp16, bf16} × {1-CTA} × R ∈ {32, 128, 256} smoke tests: 22/22 passing.
- **LoRA β-interleave math** (one MMA warp owns the issue stream, every `stride = K_atoms // R_atoms` main atoms interleaves one LoRA atom, both writing the same TMEM accumulator) is identical to v2_fa4.
- **Aliased tensor LoRA acc works.** Two MMA atoms (main NVFP4 + LoRA fp16) write to the same TMEM cell via two `cute.Tensor` objects with identical underlying addresses.
- **Shape-adaptive tiler.** Small M (M=256) uses (128, 128); large M uses (128, 256). The (128, 256) variant automatically gets `num_acc_stage = 1` since tile_n=256 overflows TMEM budget with `num_acc_stage = 2`.

### 4.2 Where v1 Hit the Wall — vs CUTLASS Same Hardware Same Shape

Same B200, same shapes, v1 vs CUTLASS's `dense_blockscaled_gemm_persistent.py` (only main NVFP4 MMA, no LoRA — strictly apples-to-apples):

| Shape (M, K, N) | CUTLASS 1-CTA 128×256 | CUTLASS 2-CTA 256×128 | CUTLASS 2-CTA 256×256 | ours 1-CTA | ours 2-CTA Phase 1 |
|---|---|---|---|---|---|
| 4352 × 3840 × 3072 | 38.5% | 42.0% | 45.4% | 13.1% | 11.8% |
| 4352 × 3840 × 15360 | 41.7% | 51.8% | 58.4% | 27.4% | 26.0% |
| 4352 × 15360 × 3840 | 41.0% | 59.0% | 63.4% | 26.5% | 29.6% |
| 4352 × 10240 × 3072 | 41.7% | 53.8% | 60.7% | 23.0% | 23.5% |

Two uncomfortable facts:
1. Same tile (128×256, 1-CTA): CUTLASS is ~14 pp faster. Persistent scheduler, multi-stage MMA/epilogue overlap, finer pipeline discipline — all absent in v1.
2. 2-CTA Phase 1 gained essentially nothing. Every shape lands within 1–3 pp of the 1-CTA column. CUTLASS's 2-CTA 256×128 (same FLOPs-per-atom as 1-CTA 128×256) lifts 10–18 pp on the same hardware.

### 4.3 Diagnosis

Attempting to elevate v1 to 2-CTA persistent causes the state space to expand along five dimensions simultaneously:

1. **Pipeline stage.** N A/B SMEM stages, each with an mbarrier, a phase bit, and an index.
2. **2-CTA pair barrier.** Under `cta_group=TWO`, each TMA barrier needs cluster awareness, `is_leader_cta` gating, and `tx_count` baking in `cta_group_size`.
3. **Persistent tile loop.** Tile boundaries do not drain the pipeline; state lives from tile N to tile N+1.
4. **LoRA β second MMA.** A second atom inserted into the main K-loop with its own LoRA prologue SMEM producer/consumer cycle.
5. **Epilogue correction chain.** Fused `× wcscale + bias` in epilogue, per-column factors with their own SMEM staging.

Stock `cutlass.pipeline.PipelineState` is implicit (state hidden inside `advance()` methods), branching (different code paths per phase), and single-dimensional (one PipelineState per pipeline, one pipeline per warp role). It cleanly handles dimension 1. It does not compose with dimensions 2–5.

## 5. Why FA4 — The Scaffolding We Adopted

FA4 (FlashAttention-4 forward on Blackwell) already solved the 5-dimensional state space problem for a different operator: attention. The solution: make pipeline state explicit and per-warp, lift the persistent tile loop to the kernel's outermost structure, and factor each Blackwell-specific footgun into a named primitive.

We did not take FA4's math — our operator has no online softmax, no S→P→O chained data flow, no Q/K/V splitting — but we adopted the scaffolding wholesale.

### 5.1 What We Took from FA4

- **Warp-specialized main loop.** Separate `load()`, `mma()`, `epilogue()` methods, each running on designated warp segments. Our MMA inherits FA4's "one warp runs two MMAs" pattern — but FA4 chains them (QK output feeds PV input via TMEM), while we make both MMAs accumulate to the same TMEM region (β-accumulation). Different math, same warp structure.
- **PipelineStateSimple.** One state object per warp, per pipeline. A single `_phase_index` counter: `index = phase_index % stages`, `phase = (phase_index // stages) & 1`. Pure divmod, no branching `advance()`.
- **PipelineTmaUmma with extra_tx_count + leader-CTA gate.** Both knobs are needed; the upstream wrapper exposes them, but the implicit PipelineState workflow cannot use them.
- **StaticPersistentTileScheduler.** 30 lines, plug-and-play.
- **gemm_ptx_partial.** Accepts raw `acc_tmem_addr: Int32`. Two MMAs can target the same TMEM range without `cute.Tensor` aliasing.

### 5.2 What We Did Not Take from FA4

- **Online softmax.** Unrelated to GEMM. Running max + rescale is an attention pattern, not a Blackwell pattern.
- **S→P→O chained data flow.** FA4's QK output (S) becomes P via softmax, then feeds PV (V matmul). This chain constrains how two MMAs use TMEM. Our β-interleave puts both MMAs into the same accumulator — no chaining, just accumulation.
- **Q/K/V splitting.** Attention has three tensor roles; we have one weight, one activation, and a LoRA pair.

Result: adapting the FA4 scaffolding to SVDQuant W4A4 actually strips away FA4's harder parts.

## 6. v2_fa4 — The Rewrite

The current production file is the FA4-derived rewrite's third real iteration: v0_fa4 (no LoRA, scaffolding only), v1_fa4 (v0_fa4 + single-stage LoRA), v2_fa4+C1 (v1_fa4 + 2-stage LoRA prologue + fused × wcscale + bias epilogue + LU SMEM fix). The shipping surface is v2_fa4+C1 post-LU-fix.

### 6.1 v0_fa4 — Scaffolding Without LoRA

The first commit of the FA4-derived branch: FA4 skeleton, no LoRA, no wcscale/bias. Purpose: validate the new state machine alone before re-inserting LoRA.

Numbers on production shape M=4352 K=3840 N=3072 fp16:

| | 1-CTA | 2-CTA |
|---|---|---|
| v0_fa4 | 7.7% | 7.6% |

Lower than v1's 27% — but expected. v0_fa4 is a partial-feature scaffolding; multi-stage pipeline is not tuned, `overlapping_accum` is not connected.

### 6.2 First Smoke: 9-Minute Hang

**Symptom:** Launch on Modal, `nvidia-smi` shows GPU busy, 9 minutes with no stdout, then container times out. No abort, no assert, no PTX error — clean hang.

**Root cause:** MMA warp's single-stage `pipeline_acc` producer phase initial value was `Int32(0)`. `pipeline_init_arrive` runs at kernel start and pre-arrives the empty mbarrier to parity 1. MMA warp calls `producer_acquire` with phase 0 — meaning "wait for barrier to flip to parity 0." But consumer (epilogue warp) has not run yet, barrier remains at parity 1, MMA warp waits forever.

**Fix:** Change `acc_producer_phase` initial value to `Int32(1)`.

**Lesson:** With explicit per-warp pipeline state, initial value invariants are your responsibility. No wrapper handles this for you. One bit wrong in initial phase → kernel silently hangs.

### 6.3 Re-adding LoRA — β-Interleave on Shared TMEM

The LoRA correction term `lora_act_in @ lora_up` is small (R ≤ 128). If run serially with the main MMA ("α" variant), wall-clock inflates ~50% in the worst production shape because tcgen05's async issue queue depth of 4–8 cannot be saturated by the few LoRA atoms.

The fix is β: interleave LoRA atoms into the main K-loop's issue stream, so the pipe never sees only LoRA.

The mechanism rests on three Blackwell facts:

1. **tcgen05 issue queue is sequential within each CTA.** Later-enqueued atoms see the effects of earlier atoms. A LoRA atom following main atom k can see main atom k's TMEM writes.
2. **Two atoms can target the same TMEM address.** Via `gemm_ptx_partial(acc_tmem_addr: Int32)`, both atoms write to the same FP32 accumulator cells.
3. **TV-layout match.** The main NVFP4 atom and LoRA fp16/bf16 atom slice per-CTA `cta_tile_shape_mnk` to per-thread register fragments. For β to work, thread t's "element i" under both atoms must land in the same TMEM cell. This is verified at trace time.

The interleave pattern: every `stride = K_atoms // R_atoms` main atoms, insert one LoRA atom. `r_next` and `next_lora_at` track which LoRA atom fires next.

**Critical MLIR trace detail:** `tiled_mma.set(tcgen05.Field.ACCUMULATE, ...)` is a Python trace-time object modification. Each `cute.gemm` call site captures field values at trace time; the setter is not re-executed at runtime. Therefore the K-tile loop must be Python fully-unrolled (`for k_tile in range(k_tile_cnt):`), not `cutlass.range(unroll=1)` — the latter traces the loop body once and reuses the first kblock's `ACCUMULATE=False` for every tile, wiping the accumulator at every tile boundary.

### 6.4 2-CTA LoRA Regression

After connecting LoRA with a single-stage prologue (v1_fa4 configuration), the 2-CTA path regressed:

| (M=4352 K=3840 N=3072 R=128 fp16) | v0_fa4 (no LoRA) | v1_fa4 (1-stage LoRA) |
|---|---|---|
| 2-CTA MFU | 7.6% | 6.0% |

Pathological: even bad LoRA should add TFLOPS, not subtract.

**Diagnosis:** LoRA SMEM (LA + LU) consumed budget. Single-stage LoRA prologue was large enough to make the budget solver (`_compute_stages`) trade away main K-loop's `num_ab_stage`. Fewer main-loop pipeline stages → fewer in-flight tcgen05 atoms → lower SM% → higher wall-clock.

### 6.5 C1 — 2-Stage LoRA Prologue

Raised `num_lora_stage` from 1 to 2. Two LA/LU buffers ping-pong. Cost: LoRA SMEM doubles. Benefit: prologue cost amortized over more main MMA iterations, budget solver returns some main-path stages.

Numbers (before LU SMEM fix, i.e., C1 contribution in isolation):

| Shape (M=4352, K, N, R) | v1_fa4 (pre-C1) 2-CTA | v2_fa4+C1 2-CTA | Δ |
|---|---|---|---|
| K=3840 N=3072 R=128 | 6.0% | 14.2% | +8.2 pp |
| K=3840 N=15360 R=128 | 15.2% | 18.6% | +3.4 pp |
| K=15360 N=3840 R=128 | 17.0% | 18.1% | +1.1 pp |
| K=10240 N=3072 R=32 | 11.6% | 26.1% | +14.5 pp |

C1 eliminated the "2-CTA LoRA is slower than 1-CTA" anomaly.

### 6.6 Fused × wcscale + bias Epilogue

The final addition over v1_fa4: folding per-output-column affine into the epilogue warp.

Math: `y[m, n] = acc[m, n] * wcscale[n] + bias[n]`

Epilogue warp path: TMEM → registers → mul-add → SMEM → GMEM via TMA store. SMEM cost is negligible (tile_n × c_dtype.width/8 = 256 or 512 bytes per buffer). This avoids a separate epilogue pass, saving one TMEM → SMEM → register round-trip, one TMA store, and one mbarrier set.

## 7. The Silent SMEM Budget Bug — LU ÷ cta_group_size

Single-line patch, +198% TF on production shapes. This is the section that makes this article worth writing.

### 7.1 The Hand-Written Formula

The setup attributes estimate how many SMEM bytes the LoRA prologue needs, so `_compute_stages` can subtract it from per-SM SMEM budget before deciding main K-loop stage count.

**Before fix:**

```python
la_bytes = mma_inst_shape_mn[0] * R * lora_ab_dtype.width // 8 // cta_group_size
lu_bytes = mma_inst_shape_mn[1] * R * lora_ab_dtype.width // 8  # ← bug
lora_smem_bytes = (la_bytes + lu_bytes) * num_lora_stage
```

LA (LoRA-down activation, dimension `[mma_tile_m, R]`) correctly divides by `cta_group_size` because the LoRA atom uses `partition_shape_A` to split along M (M-shard). LU (LoRA-up weight, dimension `[mma_tile_n, R]`) did not divide — the hand-written formula assumed each CTA holds the full `mma_tile_n × R` of LU SMEM.

### 7.2 Why This Is a Bug

Under `CtaGroup.TWO`, the 2-CTA dense MMA atom also splits B along N to the V partner (N-shard, via `partition_shape_B`). `make_smem_layout_b(tiled_mma_2cta, ...)` returns a per-CTA SMEM layout that is already half of `tile_n × tile_k`. So when LoRA's `make_smem_layout_b(...)` constructs the LU layout, it is already per-CTA half-sized. The hand-written estimate double-counted.

### 7.3 Why the Symptom Is "Nothing"

This is the dangerous part. Overestimated LoRA SMEM does not crash — it makes the budget solver pessimistic. The solver thinks LoRA SMEM consumes 16 KB more than actual, so it takes 16 KB from the main path, clamping `num_ab_stage` from 4 to 2. The kernel traces, runs, and produces correct numerics — but the main K-loop pipeline depth is halved. No assert fires, no shape mismatch, no allocation failure. Wall-clock is "slow but runs"; ncu says "low SM%, high long_scoreboard"; you spend a week tuning `num_lora_stage` and tile geometry with no improvement.

### 7.4 The Two-Minute Probe

`cute.cosize` works at trace time, returns `Int32`, giving a layout's actual SMEM cosize — exactly what the hand-written formula was trying to estimate. Insert a print in `_setup_attributes`:

```python
print("la_one =", cute.cosize(slice_(self.la_smem_layout_staged, (None, None, None, 0))))
print("lu_one =", cute.cosize(slice_(self.lu_smem_layout_staged, (None, None, None, 0))))
```

Captured output (production shape, R=128, fp16, 2-CTA):

```
[PROBE96] num_lora_stage=2 cta_group_size=2
[PROBE96] la_one cosize=16384 -> 32768 B (handwritten 32768 B, factor 1.000)
[PROBE96] lu_one cosize=8192 -> 16384 B (handwritten 32768 B, factor 0.500)
```

LA matches the hand-written value (factor 1.000). LU is exactly half (factor 0.500). Bug found in 120 seconds.

### 7.5 The Fix

One additional `// self.cta_group_size`:

```python
lora_smem_bytes = 0
if cutlass.const_expr(self.enable_lora):
    la_bytes = (self.mma_inst_shape_mn[0] * self.R
                * self.lora_ab_dtype.width // 8) // self.cta_group_size
    lu_bytes = (self.mma_inst_shape_mn[1] * self.R
                * self.lora_ab_dtype.width // 8) // self.cta_group_size
    lora_smem_bytes = (la_bytes + lu_bytes) * self.num_lora_stage
```

### 7.6 Production Shape Before/After

Same benchmark, fp16, 2-CTA, M=4352 K=3840 N=3072 R=128:

| Metric | Pre-fix | Post-fix | Δ |
|--------|---------|----------|---|
| TFLOPS | 566 | 1685 | +198% |
| MFU (B200 10 PFLOPS NVFP4) | 4.2% | 16.9% | +12.7 pp |

ncu A/B (same Verda B200, HEAD^ vs HEAD):

| Metric | Pre-LU-fix | Post-LU-fix | Δ |
|--------|-----------|-------------|---|
| Duration | 46.69 µs | 32.13 µs | −31.2% |
| Compute (SM) % | 41.63 | 53.62 | +11.99 pp |
| Memory % | 25.58 | 38.91 | +13.33 pp |
| SM Active Cycles | 72,433 | 46,126 | −36.3% |
| Memory Throughput | 386 GB/s | 561 GB/s | +45% |

Reading: same launch shape, same occupancy, but `num_ab_stage` raised to 4 fills the SM-side pipeline → SM% +12 pp, SM Active Cycles −36%. L1/TEX and L2 throughput rise proportionally because TMA producer now has more in-flight buffers to fill — not "saving bandwidth," but "bandwidth distributed more uniformly across kernel wall-clock."

### 7.7 Generalization — Teaching Content

The bug is specific (`lu_bytes` overestimated by 2×). The pattern is universal: any hand-written SMEM budget formula that feeds a stage solver, where the corresponding operand SMEM comes from `make_smem_layout_{a,b}(tiled_mma_2cta, ...)`, must divide along the sharded axis by `cta_group_size`. A is M-split (`partition_shape_A` halves along M), B is N-split (`partition_shape_B` halves along N). Under 2-CTA, both per-CTA halve — just along different axes.

Why hand-written budgets exist: `_compute_stages` needs byte estimates before operand SMEM is actually allocated (layout depends on stage count, stage count depends on budget — circular dependency). Hand-written formulas break the circularity but easily get `cta_group` sharding wrong on non-primary operands.

More robust alternative: build the layout first, read back `cute.cosize`, use the read-back value as budget input. More code, but consistent with hardware truth.

## 8. Reading ncu with Blackwell Kernel Author Eyes

### 8.1 Counter Access — Modal Blocked, Verda Open

Modal (fast iteration host) sets `NVreg_RestrictProfilingToAdminUsers=1` at the kernel module level. `torch.profiler(activities=[CUDA])` (CUPTI Activity) works, giving per-kernel wall-clock. Anything requiring perf counters fails — ncu, `nsys --gpu-metrics-device`, nvml counter queries all blocked.

Verda (deep trace host) has unrestricted counters. Workflow: iterate on Modal with wall-clock + activity trace; when a delta cannot be explained, move that specific kernel to Verda for ncu.

### 8.2 hmma Is the NVFP4 Tensor Pipe

tcgen05 UTCQMMA runs on the hmma sub-pipeline in the ncu metric tree. There is no separate `qmma_*` counter. The metric you want:

```
sm__pipe_tensor_subpipe_hmma_cycles_active.avg.pct_of_peak_sustained_active
```

Covers HMMA + UTCHMMA + UTCQMMA + UTCOMMA together. For FLOPS breakdown by accumulator dtype:

```
sm__ops_path_tensor_op_utcqmma_src_fp4_fp6_fp8_dst_fp32
sm__ops_path_tensor_op_utcqmma_src_fp4_fp6_fp8_dst_fp16
sm__ops_path_tensor_op_utcomma_src_fp4_dst_fp32
```

UTCQMMA appears under "HMMA Pipe" in the SOL "Compute (SM) Pipe Utilization" panel.

### 8.3 SOL Breakdown for 2-CTA UMMA Kernels

Reading methodology:

- **SM throughput %** — How busy SM pipelines are on average. Compute-bound NVFP4 GEMM should be high; if not, check hmma% to determine if tensor pipe specifically is busy.
- **hmma sub-pipeline %** — How busy the NVFP4 tensor pipe is. The key number. CUTLASS reference on production shapes: ~60%; v0_fa4 (no LoRA): 60.5% (matches); v2_fa4+C1 (with LoRA): 34.9% (LoRA prologue drag).
- **warp cycles / issued inst** — Average cycles per issued instruction (inverse of IPC). Rising value means more stalls per instruction; cross-reference with long_scoreboard to attribute.
- **long_scoreboard cyc (L1TEX)** — Average warp wait cycles on SMEM loads. The dominant stall source in LoRA-on configurations.

**Pattern to learn:** low hmma% and high long_scoreboard are not the same problem. The former says "tensor pipe idle," the latter says "warp has nothing to issue." Both can be true simultaneously; fixes differ.

## 9. Calibration — Where This Kernel Actually Stands

### 9.1 Honest Ceiling — CUTLASS NVFP4 on Same B200

CUTLASS's `dense_blockscaled_gemm_persistent.py` (pure main NVFP4 MMA, no LoRA, no wcscale, no bias) on the K-heavy production shape (M=4352 K=15360 N=3840):

| Variant | MFU |
|---------|-----|
| CUTLASS 1-CTA 128×256 | 41.0% |
| CUTLASS 2-CTA 256×128 | 59.0% |
| CUTLASS 2-CTA 256×256 | 63.4% |
| v2_fa4+C1+LU-fix, fp16 2-CTA | 27.3% |

Two takeaways:
1. The honest ceiling for NVFP4 on this hardware is ~60% MFU, not casually quoted 30–40%.
2. We are ~35 pp below that ceiling while doing more work (LoRA β-interleave + wcscale + bias epilogue + LA/LU prologue TMA). The remaining 35 pp represents future optimization opportunities.

### 9.2 Implementation Quality Reference — nunchaku on RTX PRO 6000

nunchaku's NVFP4 is gated on SM_120a/121a with no SM_100 binary. We run it on RTX PRO 6000 Blackwell Server Edition (SM_120a) as an implementation quality reference (not a ceiling). Hardware peaks differ 2.5× (B200 10 PFLOPS vs PRO 6000 4 PFLOPS), so MFU comparisons must stay within the same column.

| Shape (M, K, N, R) | ours fp16 (B200) | nunchaku fp16 (PRO 6000) | Δ pp |
|---|---|---|---|
| 4352 × 3840 × 3072 × R=128 | 16.9 | 16.2 | +0.7 |
| 4352 × 3840 × 15360 × R=128 | 26.5 | 19.5 | +7.0 |
| 4352 × 15360 × 3840 × R=128 | 27.3 | 25.0 | +2.3 |
| 4352 × 10240 × 3072 × R=32 | 26.4 | 21.4 | +5.0 |

fp16: 4/4 shapes leading. bf16: 2/4 leading, 1/4 within noise (−0.4 pp), 1/4 still trailing by 3.2 pp (K=15360, N=3840).

The −3.2 pp gap on bf16 traces to nunchaku's hand-written inline PTX with separate fp16/bf16 tuned paths (different register packing/accumulator precision) vs our single tcgen05 atom + `ab_dtype` substitution through the same MLIR lowering.

## 10. Remaining Levers

Ranked by current estimated ROI:

1. **bf16 register tuning.** The shape where nunchaku still leads (−3.2 pp) represents DSL MLIR lowering ceiling on bf16. Next step: bf16 LoRA atom via inline PTX, or more aggressive scheduler hints. Limited gain, ~3 pp.

2. **Wave quantization.** Production shapes land on non-integer "waves per SM" — tile geometry micro-tuning can recover 1–2 percentage points.

3. **num_lora_stage=3 is dead.** Post-LU-fix testing shows it is slower: the budget solver's cost of buying this LoRA stage tier is surrendering two main `num_ab_stage` tiers. The main K-loop loses more than LoRA prologue gains.

4. **Closing the gap to CUTLASS 2-CTA 256×256 (~60% MFU) — the remaining ~35 pp.** Two FA4-class optimizations not yet ported:
   - `overlapping_accum` at tile_n=128: `num_acc_stage=2`, two acc TMEM buffers ping-pong, hiding epilogue latency behind next tile's MMA. Only available at tile_n=128 (tile_n=256 overflows TMEM budget).
   - Tile 256×256: larger MMA per tile, fewer tile-boundary stalls, less epilogue-launch overhead per FLOP. Mutually exclusive with `overlapping_accum` under current TMEM budget.

5. **Out of scope:** Next-layer NVFP4 quantize epilogue — requires framework-layer integration.

## 11. Code Locations

- `cute_kernels/gemm_w4a4/kernel.py` — v1, pre-FA4 reference. 1-CTA, monolithic `@cute.kernel`, stock `cutlass.pipeline.PipelineState`.
- `cute_kernels/gemm_w4a4/kernel_v0_fa4.py` — FA4 scaffolding, no LoRA. Frozen as v0/v1 reference.
- `cute_kernels/gemm_w4a4/kernel_v2_fa4.py` — Production. Main NVFP4 + shared-TMEM β-interleaved LoRA + fused × wcscale + bias epilogue + LU SMEM fix.
- `cute_kernels/gemm_w4a4/_pipeline_simple.py` — 82-line copy of FA4's PipelineStateSimple.

## References

- Dao et al., "FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling," 2026
- NVIDIA CUTLASS, `dense_blockscaled_gemm_persistent.py`
- Modular, "matmul-on-blackwell-part-3" (2xSM MMA: Shared Memory Optimization)
- NVIDIA, "Parallel Thread Execution (PTX ISA)"
