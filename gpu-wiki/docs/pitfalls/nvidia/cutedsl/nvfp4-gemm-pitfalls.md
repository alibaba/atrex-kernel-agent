# cute-DSL NVFP4 GEMM pitfalls (sm_120, RTX PRO 5000)

Specific to building a fully-optimized persistent NVFP4 GEMM in cute-DSL 4.4.2. The first
two pitfalls cost ~2 days of debugging; the third is a hard cute-DSL limit you should
just route around.

Companion to:
-  (full perf journey)
-  (final kernel)
-  (single-atom MMA inline-PTX baseline)

---

## 1. Recast views MUST be inside the dynamic loop

**Trap**: View construction is "static" at the Python layer, so you assume hoisting it
above the K-loop is a free optimization:
```python
sSFA_u32 = cute.recast_tensor(sSFA, cutlass.Uint32)   # before for kt in ...
for kt in cutlass.range(K_TILES, unroll=1):
    ... sSFA_u32[stage, kb, atom, lane, 0] ...
```

**Result**: kernel runs, single-tile correctness is clean (rel_err 1e-7), but at multi-tile
shapes you get **non-deterministic** rel_err in the 1e-3..1e-2 range (different value
every run). v38 specifically: `rel_err = 7.5e-4`, then `2.3e-4`, then `5.0e-3` over
three identical runs.

**Why**: Inside an enclosing `cutlass.range(...)` dynamic loop, the IR pipeline carries
iterator state per-iteration. A hoisted recast view captures that state at construction
time and the lowered loads end up reading from a stale iteration's view. The Python
identity is the same; the MLIR IR isn't.

**Lesson**: Re-issue `cute.recast_tensor` inside the dynamic loop body where you use it.
Same warning applies to any cute view-construction call (`make_tensor` from an iterator,
new strided view) that wraps an smem tensor with stride changes inside the kernel. Treat
the recast like a per-iteration `volatile` view, not like a Python alias.

---

## 2. CUTLASS-style SF layout is 8× bloated by default

**Trap**: You read CUTLASS's reference `(128, 4)`-bytes-per-atom SF layout, copy it
faithfully, and budget your smem accordingly. Then BLOCK_K=128 doesn't fit.

**Result**: Every stage of SF takes 512 bytes per atom, of which only **64 bytes carry
data** — every 4th row holds a 4-byte SF entry (the lane-mapping `sfa_phys = sfa_logical * 4`),
the other 3 rows are zero padding. With STAGES * (ATOMS_M + ATOMS_N) atoms per CTA, that's
~21 KB of zeros taking up smem you need for BLOCK_K=128 + STAGES≥3.

**Why**: The (128, 4) layout matches a CUTLASS/MMA broadcast pattern where consecutive
groups of 4 rows correspond to one logical SF value (so a single ldmatrix-style copy
slices `sfa_phys = sfa_logical * 4`). When you instead read the SF as a single
`ld.shared.u32`, only one of the four rows is touched.

**Lesson**: Use a tight pack `(M_blocks_outer, SF_TILES_K, atoms_per_block, atom_dim, 4)`
with read index `sSFA_u32[stage, kb, atom, sfa_logical, 0]` (NO `* 4`). 8× smem
reduction, frees BLOCK_K=128 + STAGES=4 to fit in 100 KB. See
`pack_sf_per_block` in .
This single change accounts for **+37%** end-to-end perf (v38 → v43).## 3. cute-DSL 4.4.2 `cp.async.bulk` / `cp.async` + `PipelineTmaAsync` mbar = broken

**Trap**: CUTLASS C++ puts SF onto the SAME stage barrier as TMA(A,B). You try to do the
same in cute-DSL via three reasonable paths:
- (a) `cute.arch.cp_async_mbarrier_arrive_noinc(bar)` after issuing cp.async — register
  cp.async bytes onto an existing pipeline barrier.
- (b) TMA-for-SF: view the compressed 512-byte SF chunk as `(32, 16)`-bytes (16-byte
  innermost satisfies TMA's minimum) and issue a TMA load via
  `make_tiled_tma_atom(CopyBulkTensorTileG2SOp(), ...)`.
- (c) `cute.copy(make_copy_atom(CopyBulkG2SOp(), Uint8, num_bits_per_copy=4096), ...,
  mbar_ptr=bar)` — direct cp.async.bulk (1D, single-thread issue) signaling the same
  TMA bar.

**Result**:
- (a) → CUDA `LaunchFailed` / `IllegalInstruction` at runtime, regardless of whether
  `arrive_noinc` is called before or after `producer_commit`.
- (b) → compiles, runtime `IllegalInstruction` on the small `(32, 16)` TMA tile.
- (c) → compiles, runs to completion, **`rel_err ≈ 1.0`** (garbage). Adding
  `cp_async_bulk_wait_group(0)` sync doesn't help. Restricting issue to `lane == 0`
  causes hang because the producer `producer_commit` then fires before `lane == 0`'s wait
  returns.

**Why**: cute-DSL 4.4.2's `PipelineTmaAsync` uses an internal `tx_count` set at
`pipeline.create()` time and exclusively expects TMA bytes (cp.async.bulk.tensor) to
arrive on the bar. The mbarrier byte-counting machinery doesn't compose with non-tensor
copies (regular cp.async or 1D cp.async.bulk) routed through the public API. The inline-PTX
escape (`smem_ptr_to_uint` to convert the bar Pointer into the 32-bit shared address PTX
needs) isn't exposed in cute.arch in this version.

**Lesson**: For sm_120 in cute-DSL 4.4.2, **keep SF on a separate cp.async pipeline**
(the v43 pattern: per-warp cp.async for SF + own commit/wait/drain barrier). It costs
some L2 bandwidth (consumer warps redundantly load SF) but it's correct and gets to 71%
of CUTLASS. The "right" CUTLASS fix requires bypassing `PipelineTmaAsync` entirely and
building mbar control (`mbarrier_init` / `arrive_and_expect_tx` / `try_wait`) from inline
PTX — not a small detour.

---

## 4. Splitting BLOCK_K into a 5D smem layout (K_BLOCK as middle dim) breaks ldmatrix

**Trap**: Natural extension of v43: instead of two separate sub-buffers `sA0, sA1` for
the two k_blocks, make a single 5D smem `(STAGES, ATOMS_M, ATOM_M, K_BLOCK_MAX,
ATOM_K//2)` and slice as `sA[s, atom, _, k_block, _]`.

**Result**: Compiles and runs, single-tile (K_TILES=1) correct, multi-tile produces
`rel_err ≈ 5e-3`. The slice `[_, _, k_block, _]` has the SAME shape (16, 32) and SAME
strides (64, 1) as the proto for any `k_block`, but ldmatrix on the strided view (gaps
between rows because the OTHER k_block sits between rows of the same atom) silently reads
the wrong bytes for `k_block=1`.

**Why**: `ldmatrix.x4` of an 8×8 16-bit matrix interprets the per-thread address pattern
based on the *physical* smem layout, not the abstract slice. With our 64-byte row stride
(for the multi-K-block layout), the second k_block's data isn't at the address ldmatrix
expects within its 16-byte-per-thread access window.**Lesson**: For BLOCK_K split into multiple k_blocks, use **separate smem allocations
per k_block** (e.g., sA0, sA1) so that within each, the ldmatrix sees the v38 / v43
single-K-block layout it was designed for. TMA does 2 calls per outer K-tile (one per
k_block), which is fine.

---

## 5. Within-K-tile manual register K-pipelining doesn't help (and breaks correctness)

**Trap**: CUTLASS's mainloop has the textbook `copy_kblock(k+1) ; gemm_kblock(k)`
register K-pipeline. You replicate it in cute-DSL with double-buffered fragments
`tCrA[buf][mi_l]` and a manually unrolled inner loop.

**Result**: Same perf as v43 (no measurable gain, the compiler was already overlapping the
independent ldmatrix with mma.sync). Worse: at 4096³ correctness drops to `rel_err 3.8e-4`
due to register pressure pushing some fragments to spills, breaking lane-by-lane MMA
input alignment.

**Why**: cute-DSL lowers your code into LLVM-IR and lets ptxas instruction-schedule. With
2 register buffers per (mi, ni) you go from ~88 to ~112 regs/thread for fragments alone,
plus 64 for accumulators — close to the 168 reg/thread budget cute is using, and the IR
has no scheduling hint to actually overlap. ptxas already emits an interleaved schedule
when the loads and MMAs are independent in the IR.

**Lesson**: In cute-DSL, don't bother with manual register K-pipelining for K_BLOCK_MAX=2.
Trust the compiler's instruction scheduler. Manual unrolling with double-buffered
fragments only helps if you can also coordinate stage transitions and TMA waits, which
the cute pipeline layer doesn't make ergonomic.

---

## Quick "use this / not that" cheat sheet

| Topic | Use | Don't use |
|---|---|---|
| SF gmem layout | `pack_sf_per_block(sf, atom_dim, atoms_per_block)` (compressed) | original `pack_sf_per_atom` (8× bloated) |
| SF read | single `ld.shared.u32` per atom (recast SF smem to Uint32) | 4 byte loads + shifts + ORs |
| `cute.recast_tensor` | Inside the dynamic K-loop body | Hoisted above the loop (silent corruption) |
| BLOCK_K=128 smem | Two sub-buffers (sA0/sA1) per k_block | Single 5D smem with K_BLOCK as middle dim |
| Pipeline depth | STAGES=4 (drift tolerance ≥2) | STAGES=2 (forces per-K-tile barrier, eats gain) |
| SF transport | per-warp cp.async + own commit/wait | `cp.async.bulk` + `PipelineTmaAsync` mbar (broken in 4.4.2) |
| K-pipelining | Trust compiler instruction scheduler | Manual double-buffered fragments (no gain, breaks correctness) |
| Atom warp count | 1 producer + 8 consumer (4×2 warp grid) | 1 producer + 4 consumer (forces 2× MMA/warp, register pressure) |
| Persistent grid | NUM_CTAS = SM count (110 on Pro5000) | grid sized by output tile count (no L2 reuse across tiles) |
