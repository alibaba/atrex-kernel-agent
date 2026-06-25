# CuTeDSL GDN Decode on sm_120 — Pitfalls

Traps encountered while iterating 18 versions of `fused_recurrent_gated_delta_rule_fwd`
(GDN decode, fp32 state, bf16 q/k/v, T=1) on RTX PRO 5000 Blackwell.
Companion to:

- Optimization journey:
- Final kernel:

---

## 1. Treating sm_120 as if it were sm_100 (the biggest framing trap)

**Trap**: "Blackwell is Blackwell, just port the SM100 wgmma/tcgen05 GEMM."
**Result**:
- Compiler errors looking for `tcgen05_mma`, `make_tmem_load_atom`, descriptor-based MMA
  operands that don't exist on sm_120
- For NVFP4 / blockscaled paths, generated PTX uses TMEM accumulators that crash on launch
**Why**: sm_120 is **client Blackwell** (RTX PRO 5000/4000, RTX 50xx GeForce). It has:
  - **No tensor memory (TMEM)**
  - **No tcgen05 instructions**
  - **No async warp-group MMA (wgmma)**
  Memory routes are Hopper-style (TMA + cp.async); compute is Ampere-style
  (warp shuffle + warp MMA `mma.sync.aligned.kind::mxf4nvf4...m16n8k64`).
**Lesson**: **Hopper TMA + Ampere warp ALU**. This is also documented in
the sister project
— same chip, same constraint.

---

## 2. cp.async cp_size=128b silently fails alignment check

**Trap**: Set up cp.async with `num_bits_per_copy=128` (16-byte vec), correctly aligned
per-thread V slice, run it.
**Result**: ICE IR verification:
```
'cute.copy' op src ptr alignment (32 bits) does not meet requirement (128 bits)
of atom '!cute_nvgpu.atom.simt_async_copy<f32, cache = global, 128 b>'
```
**Why**: `from_dlpack(tensor)` defaults to a conservative pointer alignment
(matches the dtype: 4-byte for fp32). PyTorch CUDA tensors are actually 256-byte
aligned in practice, but the DSL doesn't know that.
**Lesson**: When you'll touch a tensor with vec cp.async, **always** declare:
```python
mH0 = from_dlpack(h0.contiguous(), assumed_align=16)
mHt = from_dlpack(ht,                assumed_align=16)
```
This single line is what unlocked the 1.58× speedup from V8 → V13.

---

## 3. cp.async with smaller `num_bits_per_copy` is NOT a fallback

**Trap**: When 128b alignment fails, try `num_bits_per_copy=64` (8B vec) instead.
**Result**:
```
'cute.copy' op '!cute_nvgpu.atom.simt_async_copy<f32, cache = global, 64 b>'
'cp_size' only supports (128), but got 64.
```
**Why**: cp.async with `LoadCacheMode.GLOBAL` only supports the 128-bit (16-byte)
form. There is no 8-byte cp.async.cg variant.
**Lesson**: Either fix the alignment (Pitfall 2) or change the layout so each
thread's V slice is exactly 4 fp32 / 8 bf16 contiguous. Don't try smaller vec.## 4. `cute.arch.load(cop=..., level1_eviction_priority=...)` rejected by MLIR verifier

**Trap**: Compose cache hint and L1 eviction priority for maximum control:
```python
cute.arch.load(ptr, fp32, cop='cg', level1_eviction_priority='evict_first')
```
**Result**:
```
'nvvm.load.ext' op load_cache_modifier and eviction priority are not allowed together
```
**Why**: PTX semantics — `.cg` cache modifier already implies a specific eviction
behavior; you can't compose it with an explicit `.L1::evict_first` hint.
**Lesson**: Pick **one**. For state-load streaming, prefer `cop='cg'`.
For accidental-reuse data, prefer `level1_eviction_priority='evict_first'` alone.

---

## 5. `cpasync.CopyOp()` is abstract — can't instantiate directly

**Trap**: To set up a generic SMEM→GMEM TiledCopy for ht store with cache hint:
```python
cp_atom_ht = cute.make_copy_atom(cpasync.CopyOp(), cutlass.Float32, num_bits_per_copy=128)
```
**Result**:
```
TypeError: Can't instantiate abstract class CopyOp with abstract method _make_trait
```
**Why**: `CopyOp` is the abstract base. Concrete subclasses are
`CopyG2SOp`, `CopyBulkG2SOp`, `CopyBulkS2GOp`, `CopyDsmemStoreOp`,
`CopyReduceBulkTensorTileS2GOp`, etc. There is **no concrete generic SIMT
S2G atom** with cache hint exposed.
**Lesson**: For S2G with cache hint, either use `CopyBulkS2GOp` (TMA bulk,
needs descriptor + mbarrier) or fall back to `cute.arch.store(ptr, val, cop='cs')`
manually. Don't grep for an "Op" class that doesn't exist.

---

## 6. Multi-warp CTAs that distribute the K dim require cross-warp SMEM reduce → loses occupancy gains

**Trap**: 1 warp/CTA seems "wasteful" — go to 4 warps and distribute K (1 K-row
per thread). State per thread drops 4× (32 → 8 fp32). Occupancy 38% → 96%.
**Result**: Wall-clock 409 μs vs the 1-warp baseline's 395 μs — **slightly worse**.
**Why**: The cross-K reduction for `hk = state^T @ k` and `o = state^T @ q`
spans 4 warps. Each requires:
```
warp-internal reduce → SMEM write (lane 0) → barrier → cross-warp accumulate → barrier → broadcast
```
Two such reductions per CTA + one for L2 norm = **6 barriers** + ~32 SMEM r/w.
That cost equals the occupancy gain.
**Lesson**: For BV-bound recurrence kernels, prefer **V-disjoint** multi-warp
(V6 design): each warp owns its own V cols, all reductions stay warp-internal.
Same per-thread state size, but no SMEM cross-warp comm.

---

## 7. High L1 hit rate is a *bad* sign for streaming workloads (the diagnostic that flips intuition)

**Trap**: ncu shows L1 hit rate 81% — celebrate, "great cache reuse!"
**Result**: Memory throughput stuck at 411 GB/s; L2 throughput 96% saturated;
DRAM throughput only 34%. Kernel is 1.58× slower than FLA Triton despite
"good cache hits".
**Why**: Default LDG `.ca` cache mode pulls every load through L1. For
streaming workloads (T=1 decode, no inter-iteration reuse), the L1 just gets
filled and re-evicted. The "hit" is really **redundant LDGs serviced by
in-flight L1 lines from the same kernel**, doubling L2 traffic and saturating
the L2 throughput long before DRAM. The kernel looks "BW-bound" but on the
wrong unit.
**Lesson**: For decode/streaming kernels, **L1 hit rate >50% in ncu is a
diagnostic of L2 false-saturation**, not of beneficial reuse. Switch to
`cpasync.LoadCacheMode.GLOBAL` (skip L1 → DRAM directly). Compare:| Version | L1 hit | L2 throughput | DRAM throughput | Wall-clock |
|---|---|---|---|---|
| V6 (direct load) | 81% | 96% (saturated) | 34% | 389 μs |
| V13 (via cp.async copy) | 33.6% | 81% | 87.7% | **246 μs** |

---

## 8. Manually setting up TMA mbarrier hangs the kernel

**Trap**: Skip the high-level `tma_load` wrapper, write your own:
```python
mbar = smem.allocate_array(cutlass.Int64, num_elems=1)
if tidx == 0:
    cute.arch.mbarrier_init(mbar, 1)
    cute.arch.mbarrier_init_fence()
cute.arch.barrier()
if tidx == 0:
    cute.arch.mbarrier_arrive_and_expect_tx(mbar, BK*BV*4)
    cute.copy(tma_atom, src, dst, tma_bar_ptr=mbar)
cute.arch.mbarrier_wait(mbar, 0)
```
**Result**: Kernel never returns. Job killed after 30+ minutes.
**Why**: The mbarrier protocol has subtle requirements: arrival count, phase
semantics, fence visibility, transaction count semantics. The `cuTeXDsl`
helpers in `tma_load` exist precisely because this is so easy to get
wrong (the dense_gemm sm_120 reference uses `tma_load`).
**Lesson**: Don't hand-roll mbarrier for TMA. Use:
```python
mainloop_pipeline = pipeline.PipelineTmaAsync.create(
    num_stages=N, producer_group=..., consumer_group=...,
    tx_count=tma_copy_bytes, barrier_storage=...,
    cta_layout_vmnk=...)
```
For single-stage prefetch (our use case), even `cp.async` is simpler than full
TMA — the wins are comparable for tiles ≤ 32KB.

---

## 9. `get_value()` requires `raw_value`, NOT `value`

**Trap**: To pack 4 fp32 from a register fragment into a vec4 for cs store:
```python
elems = [state[ki, vi].ir_value for vi in range(4)]
vec = cute.arch.vector.from_elements(vec4_f32_t, elems)
```
**Result**:
```
AssertionError: isinstance(arg, _cext.ir.Value)
```
**Why**: Numeric proxies have **both** `value` and `raw_value` attributes, but
only `raw_value` returns the raw `Value` object that MLIR ops accept.
`value` returns a wrapper.
**Lesson**: When passing extracted scalars into MLIR vector ops:
```python
elems = [state[ki, vi].value for vi in range(4)]   # ✓
```

---

## 10. `get_value()` needs both `coord` AND `component`

**Trap**: Extract element `i` from a vector value:
```python
v_i = cute.arch.vector.extract(v_vec, static_position=[i])
```
**Result**:
```
TypeError: extract() missing 1 required positional argument: 'dynamic_position'
```
After fixing that:
```python
v_i = cute.arch.vector.extract(v_vec, dynamic_position=[], static_position=[i])
```
**Result**: `Immediate out of range` at runtime.
**Why**: The static_position list semantics aren't stable across CuTeDSL
versions; `lang.vector.static_position` looks valid syntactically but generates
malformed PTX in this build.
**Lesson**: Avoid `lang.vector.static_position` for now. If you need to unpack a loaded vec
back into per-element compute, easier paths:
1. Issue per-element loads (slow — V11 was 2.6× slower)
2. Use cp.async + SMEM staging (V13's chosen path)
3. Wait for a stable `lang.vector.dynamic_position` API in newer CuTeDSL## 11. Increasing BV per CTA can *decrease* ncu Max BW %

**Trap**: After V13 hits 87.7% Max BW, double BV (8 → 16) to amortize
fixed per-CTA overhead and (you assume) push closer to peak.
**Result**: Max BW drops to 87.24% (slightly worse), wall-clock unchanged.
L1 hit rate jumps from 33% → 53%.
**Why**: A bigger tile means more threads in the same CTA share L1 lines for
boundary loads. More L1 reuse = fewer DRAM bytes attributed = lower "DRAM
throughput %". The kernel hasn't actually slowed down; the metric just changed
its denominator.
**Lesson**: For ncu **DRAM Throughput %** specifically, larger tiles can be a
regression. To push the *metric*, push toward zero L1 hit (every load `cg`).
To push *wall-clock*, follow the regular roofline rules.

---

## 12. `cute.recast_tensor` only accepts Numeric dtypes (not vector types)

**Trap**: View a `[K, V]` fp32 fragment as a `[K, V/4]` vec4_f32 fragment
to do bulk vec stores:
```python
vec4_f32_t = ir.VectorType.get([4], cutlass.Float32.mlir_type)
state_vec = cute.recast_tensor(state, vec4_f32_t)
```
**Result**:
```
TypeError: dtype must be a type of Numeric, but got vector<4xf32>
```
**Why**: `recast_tensor` is for changing element dtype (e.g. fp32 ↔ bf16),
not for re-wrapping a region as a vector view.
**Lesson**: To get vector-typed elements from a fragment, build the vector
explicitly per-row using `cute.arch.vector.from_elements` (and Pitfall 9
about `.value`).

---

## "Use this, not that" cheat sheet

| Goal | Use | Don't use |
|------|-----|-----------|
| Bypass L1 on big state load | `cpasync.CopyG2SOp(cache_mode=LoadCacheMode.GLOBAL)` | default LDG `.ca` |
| 16B vec gmem path | `assumed_align=16` + `num_bits_per_copy=128` | scalar `cute.arch.load(cop='cg')` |
| Multi-warp on V dim | each warp owns disjoint V cols (V-disjoint) | each warp owns disjoint K rows (K-distributed) |
| TMA load with mbarrier | `pipeline.PipelineTmaAsync` | hand-rolled `mbarrier_init`+`mbarrier_wait` |
| Extract MLIR Value from Numeric | `.value` | `.ir_value` |
| Generic SIMT S2G atom | doesn't exist as `CopyOp()` | trying `cute.make_copy_atom(cpasync.CopyOp(), ...)` |
| sm_120 MMA | warp `mma.sync.aligned…m16n8k64` | `tcgen05_mma` / wgmma / TMEM |
| Diagnose L2 saturation | "L2 throughput 95%+ AND DRAM <50% AND L1 hit >50%" | trusting "Max BW 95%" alone |
| Larger tile to amortize overhead | careful — may hurt ncu DRAM % | assuming bigger BV is always better |
