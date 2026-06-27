# CuteDSL GDN Decode (fp32 state, bf16 q/k/v) on sm_120 — Optimization Journey

End-to-end journey of bringing a CuteDSL implementation of FLA's
`fused_recurrent_gated_delta_rule_fwd` from naive (1.97× slower than FLA Triton at B=8)
to wall-clock parity (1.00×) on NVIDIA RTX PRO 5000 Blackwell. 18 versions iterated.

## Target hardware

| Item | Value |
|---|---|
| GPU | NVIDIA RTX PRO 5000 Blackwell (also RTX PRO 4000 Blackwell, RTX 50xx GeForce) |
| Compute Capability | sm_120 (CC 12.0) |
| SMs | 110 |
| HBM | 48 GB (GDDR7, 384-bit, ~28 Gbps) |
| Driver / CUDA | 580.105 / 13.0 |
| D2D memcpy ceiling (≥256MB tile, measured) | **1032 GB/s** |
| ncu Max Bandwidth ceiling (vendor spec) | ~1186 GB/s |
| OS | Ubuntu 22.04 root container |

**Architectural constraint** (don't overlook): sm_120 is *client* Blackwell. It has
**no `tcgen05` / TMEM / wgmma**. Routes are Hopper-style **TMA + cp.async** for
memory + Ampere-style **warp ALU + warp shuffle** for compute. Do NOT copy sm_100
patterns.

## Algorithm baseline (one CTA = one (n, hv, v_chunk))

```
state ← h0[n, hv, :, v_chunk]              # 4KB fp32 tile per CTA (BK=128, BV=8)
load q[n,h,:], k[n,h,:], v[n,hv,v_chunk], g[n,hv], beta[n,hv]
q,k ← L2-norm(q,k); q ← q * scale
state ← state * exp(g)                     # scalar gate
hk     ← state^T @ k                       # GEMV  (BV)
v_new  ← beta * (v - hk)                   # delta-rule remove
state += k ⊗ v_new                         # rank-1 update
o      ← state^T @ q                       # GEMV  (BV)
store o, store state→ht
```

T=1 → no chunk loop. Per CTA work is small but state load/store dominates
bytes. Grid = `(NV=V/BV, B*HV)`, e.g. `(32, 1024)` for B=64, HV=16.

## Kernel resource footprint (final V13)

| Item | Value |
|---|---|
| Block | 64 threads (2 warps × 32) |
| Registers / Thread | 40 (no spill) |
| Dynamic SMEM / Block | 4096 B (h0 staging) |
| Achieved Occupancy | 79.23% |
| Per-thread state | `[4 K rows × 4 V cols] = 16 fp32 = 64 B` |
| Per-thread mem ops | 4× 16-byte cp.async + 4× 16-byte STG |

## Optimization journey

Format: **vN — change → measured B=64 → ncu Max BW → notes**.

### V1 — mirror Triton (1 warp, BV=8, no SMEM staging)
Direct translation of FLA's Triton kernel: 1 warp/CTA, BK=128, BV=8, scalar
loads, warp shuffle reduce. Per-thread state `[4 K, 8 V] = 32 fp32`.
**Time**: 396 μs. **Max BW**: 91% (ncu reports L2 throughput, not actual DRAM).
**Issue**: ncu shows L1 hit 86%, suggesting **register spill to local memory**
(80 reg/thread, conservative codegen).

### V2 — `cute.autovec_copy` for q/k/v/h0/ht
Replaced element-wise loads with `cute.autovec_copy`. **Time**: 395 μs (no Δ).
**Lesson**: compiler was already vec-loading; the bottleneck wasn't load count.

### V3 — register reuse (inline intermediates, single `r_buf`)
Combined `r_hk` / `r_vnew` / `r_o` into one buffer. **Time**: 395 μs (no Δ).
**Lesson**: spill not from declared frags but from compiler's allocation choice.

### V4 — TMA G2S with `tma_partition` (WIP)
Rank mismatch in `cute.copy(tma_atom, ...)`: `local_tile` returned rank-4 vs
SMEM rank-2. Did not finish.

### V5 — 4-warp K-distributed (1 K row per thread)
128 thr/CTA, each thread holds `[1 K × 8 V] = 8 fp32 state`.
Per-thread regs **dropped to 40**. Achieved Occupancy **38% → 96%**.
**Time**: 409 μs (slightly worse than V3).
**Why no win**: hk and o reductions now need cross-warp via SMEM (3× per CTA).
SMEM round-trips ate the occupancy gains.

### V6 — 4-warp V-disjoint (each warp owns 2 V cols, no cross-warp comm)
Per-thread state `[4 K × 2 V] = 8 fp32`. Reductions stay warp-internal.
**Time**: 389 μs (-1.5% vs V3). **Max BW**: 96% (still L2-saturated).
**Crucial finding from ncu**: L1 hit 81% but DRAM throughput only 34%.
**This is L2 false-saturation** caused by default `ld.ca` cache mode → L1 fill →
L2 traffic doubled.

### V7 — V6 + autovec on state load (vec=2 fp32)
Each thread tile is `[4K, 2V]` → only 8-byte vec, not 16-byte.
**Time**: 408 μs (regressed slightly).

### V8 — 2-warp V_PER_WARP=4 (vec=4 fp32 = 16 bytes)
Per-thread `[4K, 4V] = 16 fp32 state`. **Time**: 389 μs.
Same as V6; vec width alone doesn't help — still ld.ca cache.

### V9 — cp.async + `LoadCacheMode.GLOBAL` (BV=8 layout, vec=8 attempt)
Tried 64-bit cp.async — failed: `cp_size only supports (128)` on this CuTeDSL build.

### V10 — TMA G2S + manual mbarrier
Hand-coded `mbarrier_init + mbarrier_arrive_and_expect_tx + mbarrier_wait`.
**Hung in infinite loop**. Killed after 30+ min. mbarrier protocol is far less
forgiving than it looks; use `pipeline.PipelineTmaAsync` wrapper instead.

### V11 — `cute.arch.load(cop='cg')` element-wise on state
**L1 hit dropped from 81% → 36%** (cg actually skipped L1).
**Time**: 637 μs (much worse). Per-element loads lose vectorization; cg savings
were dwarfed by instruction count blow-up.

### V12 — V11 + vec dtype via `llvm.extractelement`
`cute.arch.load(ptr, vec4_f32_t, cop='cg')` worked, but extracting elements
needs `ir.Value` index, not Python int. MLIR API friction. Did not finish.

### **V13 — cp.async G2S `LoadCacheMode.GLOBAL` + `assumed_align=16` + V_PER_WARP=4** ✅

Key insight: **V8's 16B vec attempt failed only because PyTorch tensor pointer
alignment defaults to 4 bytes** in `from_dlpack`. PyTorch CUDA allocations are
actually 256-byte aligned; declaring `assumed_align=16` unlocks the path.

```python
# 1. Tell CuTeDSL the gmem ptr is 16-byte aligned
mH0 = from_dlpack(h0.contiguous(), assumed_align=16)
mHt = from_dlpack(ht, assumed_align=16)

# 2. cp.async with cache_mode=GLOBAL bypasses L1
cp_atom = cute.make_copy_atom(
    cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
    cutlass.Float32, num_bits_per_copy=128,
)

# 3. Thread layout that gives each thread 4 contiguous V cols (16-byte vec)
thr_layout = cute.make_layout((32, 2), stride=(2, 1))   # 64 threads
val_layout = cute.make_layout((4, 4), stride=(4, 1))    # 4K × 4V per thread

# 4. Compute pipeline: cp.async commits, then q/k/v + L2norm, then wait+barrier
cute.copy(tiled_copy, thr_gH0, thr_sH0)
cute.arch.cp_async_commit_group()
# (latency-hide window: q/k/v autovec_copy, L2 norm warp shuffle)
cute.arch.cp_async_wait_group(0)
cute.arch.barrier()
state[ki, vi] = sH0[k_off + ki, v_smem_base + vi]   # cheap SMEM→register
```

**Time @ B=64: 246 μs** (V6: 389 μs → 1.58× speedup; FLA Triton: 247 μs → tied).

### V14 — V13 + S2G TiledCopy for ht (WIP)
`cpasync.CopyOp` is abstract; need a concrete subclass. Did not finish.

### V15 — V13 + `cute.arch.store(cop='cs')` vec4 ht store
`cute.arch.vector.from_elements` for packing, `.value` (NOT `.ir_value`) for
extracting raw `ir.Value` from Numeric.
**Time**: 246 μs (no Δ). **Max BW**: 87.65%. cg variant: same.
The 4 fp32 elements per thread are already vectorized by autovec; cs hint
doesn't move ncu numbers because the remaining L1 traffic is q/k/v, not ht.

### V16 — BV=16 (half grid launches)
Doubled BV per CTA, halving grid count.
**Time**: 245 μs. **Max BW**: 87.24% (slightly **worse**).
**Why**: larger tile increased intra-CTA L1 reuse (L1 hit 33% → 53%), which
*reduces* DRAM throughput attribution.

### V17 — `cute.arch.warp_reduction_sum` HW intrinsic
Replaced 5-step XOR butterfly with single SHFL.IDX tree reduction.
**Time**: 246 μs. **Max BW**: 87.70%. Perf identical — Triton compiler already
emits efficient reductions either way.

### V18 — cg cache hint on q/k/v vec loads (WIP)
`cute.arch.vector.extract` needs `dynamic_position=[]` AND `static_position=[i]`.
Got past one error; hit `Illegal instruction` at runtime. The remaining L1 hits
in V13 are tiny (q/k/v ≈ 1MB total at B=64); even if eliminated, can't push
past ~88% Max BW because of how ncu attributes L1-served bytes.

## Final perf vs baseline

| B | V1 (1.97× FLA) | **V13** | FLA Triton | V13/FLA | V13 throughput | % memcpy ceiling |
|---|----------------|---------|------------|---------|----------------|------------------|
| 1   | 56  | 55  | 29  | 1.89× | (Python overhead bound) | — |
| 8   | 59  | 54  | 30  | 1.81× | (Python overhead bound) | — |
| 32  | 204 | 122 | 120 | **1.02×** | 1041 GB/s | 100.9% |
| 64  | 396 | **246** | 247 | **1.00×** | 1040 GB/s | **100.8%** |
| 128 | 788 | 491 | 494 | **0.99×** | 1042 GB/s | 101.0% |

(B=1, B=8 are CuteDSL Python launch overhead bound, not kernel bound. Move
launch to C++/CUDA Graph for those.)

## Remaining bottlenecks (PMC evidence on V13, B=64)

| ncu Section / Metric | Value | Interpretation |
|---|---|---|
| Speed-of-Light: Memory Throughput | 87.68% | DRAM near saturation |
| Memory Workload: L1/TEX Hit Rate | 33.6% | residual q/k/v + ht via L1 |
| Memory Workload: L2 Hit Rate | 59.8% | (down from V6's 87% — cache pollution gone) |
| Warp State: MIO Stall % | 36% | bound by SMEM ops + special math |
| Warp State: Long Scoreboard Stall | low | TMA/cp.async hides DRAM latency well |
| Compute SM Throughput | 27.6% | confirms memory-bound |
| Achieved Occupancy | 79.23% | 4 blocks/SM × 64 thr × 2 warps |

The 12.3pp gap from 100% Max BW is **not actually addressable**: the metric
itself penalizes any L1 hit (those bytes don't count toward DRAM). FLA Triton
gets 91.87% only because it hits L1 just 8%. To exceed 90% by ncu's measure,
every load must be `cs/cg` cached and zero L1 reuse — which CuteDSL's high-level
API (autovec, q/k/v small loads) doesn't expose without per-element MLIR
inline (tried in V18, hit API friction).

## What would close the remaining gap

1. **Move B=1/8 launch to C++** — kernel is fine (47% Max BW already), it's
   ~24μs Python `from_dlpack` × N + JIT dispatch overhead.
2. **Full TMA bulk transfer for h0+ht** — instruction count drops to 1 per
   tile. Requires `pipeline.PipelineTmaAsync` (V10's manual mbarrier hangs).
3. **Inline PTX for q/k/v loads via `cute.dsl_user_op`** — emit
   `ld.global.cg.v4.f32` directly, skipping the `cute.arch.vector.extract`
   API friction. Likely +1–3pp ncu Max BW.

None of these change wall-clock vs FLA, since FLA is also at the same memory
ceiling. They only move the ncu reading.

## Sustained recipe (do these, in this order, for any sm_120 BW-bound kernel)

1. **Diagnose with ncu first**. If L2 throughput is 95%+ but DRAM is <50% and
   L1 hit rate is high (>50%), you have **L2 false-saturation** from default
   `ld.ca` cache mode. The ceiling looks higher than it is.
2. **Set `assumed_align=16` on `from_dlpack(...)`** for any tensor you'll
   touch with cp.async — PyTorch CUDA tensors are 256-byte aligned but the
   default ptr alignment hint is conservative.
3. **Use `cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL)`** as
   the load atom for any large state load — bypass L1 to free L2.
4. **Layout per-thread V slice as 16 contiguous bytes** (4 fp32 / 8 bf16) to
   match `num_bits_per_copy=128`. Use `thr_layout=(K_thr, V_thr)` and
   `val_layout=(K_per_t, V_per_t)` where `V_per_t × dtype_bytes = 16`.
5. **Multi-warp design: V-disjoint, NOT K-distributed**. Each warp owns its
   own V cols → all reductions warp-internal → no SMEM cross-warp barriers.
   K-distribution (V5) needs SMEM reductions and loses the occupancy gain.
6. **Stage state via SMEM**: cp.async load → SMEM → per-thread register read.
   Don't load gmem→register directly element-wise (V11 lesson).
7. **Latency-hide cp.async with q/k/v compute**:
   `cp_async_commit_group()` → load q/k/v + L2 norm + gate prep →
   `cp_async_wait_group(0); barrier();` → consume SMEM.
8. **Don't iterate larger BV chasing throughput** (V16 lesson) — it
   *increases* intra-CTA L1 reuse, which *reduces* DRAM attribution.

## Related docs

- **Pitfalls** (the 9+ traps from this 18-version journey):
  [`docs/pitfalls/nvidia/cutedsl/gdn-decode-pitfalls.md`](../../../../pitfalls/nvidia/cutedsl/gdn-decode-pitfalls.md)
- **Final kernel + reference**:
  [`reference-kernels/nvidia/blackwell-geforce/cutedsl/gdn_decode/`](../../../../../reference-kernels/nvidia/blackwell-geforce/cutedsl/gdn_decode/)
- **Same algorithm, different chip + dtype** (Hopper bf16 state):
  [`reference-kernels/nvidia/hopper/cutedsl/flashinfer/gdn_decode_*.py`](../../../../../reference-kernels/nvidia/hopper/cutedsl/flashinfer/)
- **Latency-bound recurrence framework** (T>1 chunked variants):
  [`docs/ref-docs/nvidia/gluon/sm90/linear_attention.md`](../../gluon/sm90/linear_attention.md)
- **CuTeDSL programming model** prerequisites:
  [`docs/ref-docs/nvidia/cutedsl/cutedsl-programming-model.md`](../cutedsl-programming-model.md),
  [`docs/ref-docs/nvidia/cutedsl/cutedsl-pipeline-patterns.md`](../cutedsl-pipeline-patterns.md)
- **sm_120 NVFP4 GEMM journey** (sister project, also sm_120 + CuTeDSL):
  [`docs/ref-docs/nvidia/cutedsl/sm120/sm120-nvfp4-inline-ptx-gemm.md`](../sm120/sm120-nvfp4-inline-ptx-gemm.md)
