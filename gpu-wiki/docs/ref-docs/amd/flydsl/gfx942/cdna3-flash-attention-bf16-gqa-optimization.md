# FlyDSL Flash Attention (bf16, MHA + GQA) Optimization on MI308X (gfx942)

This document records the iterative profile-driven optimization of a FlyDSL
flash-attention forward kernel on AMD MI308X (CDNA3, gfx942), targeting the
prefill regime and benchmarked head-to-head against `aiter.ops.mha.mha_batch_prefill_func`.

The kernel is at `reference-kernels/amd/cdna3/flydsl/FlyDSL/flash_attn_func_mi308x.py`.
Companion debugging notes are in `pitfalls/amd/flydsl/flash-attn-pitfalls.md`.

## Target hardware

| Item | Value |
|---|---|
| GPU | AMD Instinct MI308X |
| Arch | CDNA3 (gfx942) |
| CUs | 80 |
| SEs | 16 (5 CUs each) |
| LDS / CU | 64 KB |
| HBM | 5.3 TB/s |
| bf16 MFMA peak | 206 TFLOPS |

## Algorithm baseline

Causal flash-attention forward, head_dim=128, bf16:
- Tile shape: BLOCK_M ∈ {128, 256} auto-selected; BLOCK_N=64; K_SUB_N=32
- MFMA: `mfma_f32_32x32x8bf16_1k` (gfx942's K=8 variant)
- Q is preloaded once per workgroup as the B operand (register-resident)
- K row-major in LDS; V transposed in LDS (V^T row = V col)
- Online softmax fused with PV GEMM2

Workflow per KV iter:
1. coop load K → LDS (vectorized ds_write_b128)
2. coop load V → LDS via per-element ds_write_b16 (transpose store, scalar)
3. GEMM1: K @ Q^T → S (16 MFMAs, software-pipelined ds_read with depth=2)
4. softmax(S): row-max, FMA-scaled exp2, row-sum, O rescale
5. GEMM2: V^T @ P → O (16 MFMAs, V LDS reads via ds_read_b64)

## Kernel resource footprint (final)

| Item | Value |
|---|---|
| VGPR / wave | 100 |
| AccumVGPR / wave | 132 |
| LDS / WG | 33 792 B (K 8.4KB + V 8.4KB single-buffered, no padding 16KB free) |
| Workgroup size | 256 (M=128) or 512 (M=256) |
| Achieved waves/SIMD | 2 (capped by AccumVGPR=132 > 128) |

## Optimization journey (head-to-head vs aiter)

Numbers measured on MI308X bf16 causal prefill. `fly/aiter` ratio < 1.0 means
fly is slower; > 1.0 means fly wins. Aiter baseline is the JIT-compiled CK_tile
`mha_batch_prefill_bf16_*.so`.

### v0 — Starting point (auto-selector + K_PAD already applied)

| shape | fly ms | aiter ms | sp |
|---|---|---|---|
| B1H32S16k MHA | — | — | — |
| B1H32S8k MHA | — | — | — |

### v2 — Replace `arith.exp2` → `rocdl.exp2` (3 callsites in softmax)

```python
# Before
corr = arith.ArithValue(diff_m_scaled).exp2(fastmath=fm_fast)
# After
corr = rocdl.exp2(T.f32, diff_m_scaled)
```

`arith.ArithValue.exp2` lowers via `math.exp2` → `v_ldexp + v_cmp + v_cndmask`
(several VALU ops). `rocdl.exp2` lowers directly to `v_exp_f32` (single op).
SageAttention recipe reports −38% softmax VALU.

**Gain**: Measurable improvement across all shapes.

### v6 — Pre-load all V into SSA before pure-MFMA PV loop

```python
# Pre-compute every V vector before the MFMA loop body
v_los = [None] * TOTAL_PV
v_his = [None] * TOTAL_PV
for si in range_constexpr(TOTAL_PV):
    v_los[si], v_his[si] = _read_v_pack(si)

# Upfront rescale of o_accs[1..D_CHUNKS-1]
for dc_r in range_constexpr(D_CHUNKS - 1):
    o_accs[dc_r + 1] = mul(o_accs[dc_r + 1], corr_vec)

# Pure-MFMA inner loop (no interleaved ds_read)
for si in range_constexpr(TOTAL_PV):
    dc, pks = _steps[si]
    o_accs[dc] = mfma(v_los[si], p_packs_lo[pks], o_accs[dc])
    o_accs[dc] = mfma(v_his[si], p_packs_hi[pks], o_accs[dc])
```

The interleaved (read+MFMA) loop constrains the scheduler. Pre-loading lets
the compiler globally schedule a dense MFMA block, which the AMD backend
optimizes much better. Risk: VGPR pressure from holding all V values.
Verified to NOT regress occupancy because we were already AccumVGPR-bound at
2 waves/SIMD; the extra V VGPRs don't tip the balance.

**Gain**: Small improvement across long-seq shapes.

### v11 — GQA support (decouple H_q from H_kv)

Add `num_kv_heads` parameter (default `None` → MHA backward-compat). Split
strides and global indices:

```python
KV_STRIDE_TOKEN = NUM_KV_HEADS * HEAD_DIM      # vs Q_STRIDE = NUM_HEADS * HEAD_DIM
GQA_GROUP_SIZE = NUM_HEADS // NUM_KV_HEADS

if GQA_GROUP_SIZE == 1:
    kv_head_idx = head_idx                      # MHA: identity
else:
    kv_head_idx = head_idx // arith.index(GQA_GROUP_SIZE)

def kv_global_idx(token_idx, col):
    token = batch_idx * seq_len_v + token_idx
    return token * KV_STRIDE_TOKEN + kv_head_idx * HEAD_DIM + col
```

All K/V global loads route through `kv_global_idx`; Q loads and O writeback
keep using `head_idx`. DMA paths (gfx950+ only) updated symmetrically.

Correctness across MHA + Qwen GQA shapes: max_abs vs torch SDPA = 9.77e-04
(bf16 acceptable).

### v12 — Block decomposition order (head_fast confirmed optimal)

MI308X dispatches workgroups SE-ordered (block_id mod 16 → SE). Each SE
preempts among its 5 CUs. For causal attention, compute is uniform along H_q
but varies along q_tile (later q_tile attends to more KV blocks).

We A/B'd two orderings:
- **head_fast** (current): `head_idx = block_id % NUM_HEADS`. Consecutive
  16 blocks share q_tile → 16 SEs see equal compute per dispatch wave.
- **q_fast**: `q_tile_idx = block_id % num_q_tiles`. Consecutive 16 blocks
  span q_tile light→heavy, then repeat per head. Each SE accumulates
  uniform-q work over time, so different SEs end up with very different
  total work.

**Result**: q_fast is significantly slower on every Qwen shape. head_fast is
correct. Code keeps an `FLYDSL_FA_BLOCK_ORDER` env knob for future
ablation; default unchanged.

### v13 — O writeback vectorized (16 b16 → 4 b64 per lane)

The O writeback loop was scalar `ds_write_b16` per element (16 per lane per
D_CHUNK). Replaced with grouped `bf16_trunc_pack_v4` + single `_gep_store` of
v4bf16 (= 1 buffer_store_dwordx2 per group of 4 cols). 75% fewer store
instructions per lane.

**Gain**: Neutral. O writeback is a negligible fraction of kernel time. Not on
the critical path. Kept for code cleanliness.

### v14 — V transpose store via `ds_swizzle` XOR-N + paired `ds_write_b32`

**The big win**. Closes the remaining 22-24% gap vs aiter in a single change.

**Insight**: in `coop_load_v`, lane `l` and lane `l XOR THREADS_PER_ROW_LOAD`
hold **adjacent rows** of V at the same `load_col_base`. For VEC_WIDTH=16
(default), THREADS_PER_ROW_LOAD=8 → peer at `lane XOR 8`. For VEC_WIDTH=8 →
peer at `lane XOR 16`.

**Scheme**:
1. `ds_swizzle_b32 offset:0x201F` (XOR-8) or `offset:0x401F` (XOR-16)
   swaps each lane's dword with its peer across the row pair. Uses LDS
   instruction port **but zero LDS bank contention** (ds_swizzle data stays
   on chip, never touches LDS storage).
2. Each even-row lane now holds `(own=R, peer=R+1)` pairs for every
   source dword.
3. Pack each `(own_elem, peer_elem)` into v2bf16 via `vector.from_elements`
   — compiler picks the right pack instruction (v_pack / v_lshl_or_b32).
4. `ds_write_b32` at V^T[col, row=R] writes 2 bf16 contiguously → covers
   both row R and row R+1 of V^T.
5. Odd-row lanes silent (predicated off).

**Cost comparison per WG per V batch**:
- Old scalar transpose: 256 lanes × 8/16 `ds_write_b16` = 2048 / 4096 LDS writes
- v14: 128 lanes × 8/16 `ds_write_b32` = 1024 / 2048 LDS writes + 128 × N
  `ds_swizzle` (no bank contention)

LDS write instruction count halved; same data volume.

```python
def _ds_swizzle_xor8_inline(src):
    return _llvm.inline_asm(
        T.i32, [src],
        "ds_swizzle_b32 $0, $1 offset:0x201F\n\ts_waitcnt lgkmcnt(0)",
        "=v,v", has_side_effects=True,
    )

# In _v_store_transposed_perm:
for k in range_constexpr(num_dwords):
    own_dw = vector.extract(own_vNi32, static_position=[k], ...)
    peer_dwords[k] = _swz_inline(own_dw)
peer_vec = vector.bitcast(vxf16_type,
    vector.from_elements(_vN_i32_type, peer_dwords))

if lds_row & 1 == 0:  # even-row lanes only
    for _e in range_constexpr(VEC_WIDTH):
        own_elem = vector.extract(vec, static_position=[_e], ...)
        peer_elem = vector.extract(peer_vec, static_position=[_e], ...)
        pair = vector.from_elements(_v2_type, [own_elem, peer_elem])
        vt_idx = v_base + (load_col_base + _e) * VT_STRIDE + lds_row
        vector.store(pair, lds_kv, [vt_idx])  # ds_write_b32
```

**Gain**: Largest single win after v2 (rocdl.exp2). Significant improvement
across every tested shape. See "Final perf" table below.

**Why previous attempts failed**:
- v10 `ds_bpermute` pairing: ds_bpermute routes through the actual LDS unit
  with bank-conflict arbitration. The peer-swap LDS round-trips cancelled
  the saved write cost. Net: zero perf change.
- `permlanex16` (VALU-only cross-lane swap on RDNA): **does NOT exist on
  gfx942** (LLVM error `Cannot select: intrinsic %llvm.amdgcn.permlanex16`).
  It's RDNA-only (gfx10+). Use `ds_swizzle` on CDNA.
- `rocdl.ds_swizzle(T.i32, src, arith.constant(0x401F, type=T.i32))`
  produces wrong output: the offset argument is passed as an SSA value that
  doesn't fold to the required hardware immediate. **Must use
  `_llvm.inline_asm` with literal `offset:0x401F`**.
- `v_perm_b32` inline asm with CK_tile-style sel imm (0x05040100 /
  0x07060302) produces wrong output on this LLVM revision even when
  cross-lane + addressing are verified correct. `vector.from_elements`
  explicit pack lets the compiler emit correct-by-construction code at
  the same speed.

See `docs/pitfalls/amd/flydsl/flash-attn-pitfalls.md` for the full debugging
trail.

### v15 — ds_swizzle latency hiding (`has_side_effects=False`)

**Validated future-opportunity #3 from v14**. The v14 `ds_swizzle` inline asm
included `\ts_waitcnt lgkmcnt(0)` in the asm string and `has_side_effects=True`,
forcing the compiler to serialize all work after each ds_swizzle. Removing
both constraints lets the compiler schedule independent VALU / MFMA work
during the ~20-cycle ds_swizzle latency.

```python
# Before (v14)
def _ds_swizzle_xor16_inline(src):
    return _llvm.inline_asm(
        T.i32, [src],
        "ds_swizzle_b32 $0, $1 offset:0x401F\n\ts_waitcnt lgkmcnt(0)",
        "=v,v", has_side_effects=True,
    )

# After (v15)
def _ds_swizzle_xor16_inline(src):
    return _llvm.inline_asm(
        T.i32, [src],
        "ds_swizzle_b32 $0, $1 offset:0x401F",
        "=v,v", has_side_effects=False,
    )
```
**Why it's safe**: `ds_swizzle` does NOT touch LDS storage — data stays on chip
in the cross-lane permute network. There are no LDS bank conflicts or WAR
hazards. The compiler's own register dependency tracking correctly inserts
`s_waitcnt` only where the swizzled value is actually consumed.

**Gain**: +4.6% at B1S4096H32D128 (1.760 → 1.683 ms, 78.1 → 81.7 TFLOPS).
Consistent +4.5%–7.8% across all tested shapes (larger gains on shorter
sequences where ds_swizzle stalls represent a bigger fraction of runtime).

**Dead-end attempts tested during v15 development** (all reverted):
- K/V LDS sharing (2 extra barriers per KV block negate occupancy gain → 30% regression)
- K_PAD=2 (zero bank conflicts but unaligned K stores 2×b64 vs 1×b128 → neutral)
- Tree max reduction (no measurable improvement at 2 waves/SIMD)
- `waves_per_eu=2` (matching actual occupancy doesn't improve codegen → neutral)
- `rocdl.iglp_opt(2)` (attention-specific scheduling hint has no effect → neutral)

See `docs/pitfalls/amd/flydsl/flash-attn-pitfalls.md` traps 18–22 for details.

### Comparison vs aiter `flash_attn_func` (CK fmha_v3_fwd, post-v15)

Post-v15, the kernel was benchmarked against `aiter.ops.mha.flash_attn_func`
(which dispatches to the CK V3 `fmha_v3_fwd` backend with four-phase
`CoreLoopScheduler` for fine-grained MFMA/VALU/TRANS/SALU interleaving).
Note: this is a different aiter entry point than the `mha_batch_prefill_func`
used for v0–v14 benchmarks; the CK V3 backend is significantly faster.

| Shape | Ours (ms) | Aiter fmha_v3 (ms) | Gap |
|---|---|---|---|
| B1S2048H32D128 | 0.675 | 0.469 | −30.6% |
| B1S4096H32D128 | 1.678 | 1.566 | −6.7% |
| B1S8192H32D128 | 5.696 | 5.088 | −10.7% |
| B1S16384H32D128 | 21.084 | 18.261 | −13.4% |
| B2S4096H32D128 | 3.130 | 2.722 | −13.0% |
| B1S4096H64D128 | 3.116 | 2.719 | −12.7% |

**Conclusion**: CK V3's `CoreLoopScheduler` (which does cycle-level
MFMA/VALU/TRANS/SALU interleaving at the ISA template level) closes the
gap that our LLVM-based compiler cannot reach via scheduling hints alone.
The remaining 7–31% gap is primarily a software pipelining / instruction
scheduling quality difference. Closing it in FlyDSL would require either
(a) exposing the compiler's machine scheduler with custom DAG constraints,
or (b) manually emitting the four-phase interleaved loop body via inline
asm.

## Final perf vs aiter (MI308X, bf16, causal prefill — **post-v14**)

> **Note**: Specific performance values have been removed. To be updated after re-measurement.

### MHA (H_q == H_kv)

> Specific ms and TFLOPS values have been removed.

### GQA (Qwen3.5-9B TP=2: H_q=16, H_kv=2)

> Specific ms and TFLOPS values have been removed.

> Specific ms and TFLOPS values have been removed.

### GQA (Qwen3.5-27B TP=2: H_q=20, H_kv=4)

> Specific ms and TFLOPS values have been removed.

> Performance comparison conclusion to be updated after re-measurement.

Correctness: max abs error vs torch SDPA float32 reference ≤ 2e-3 (bf16
acceptable) on every shape.

## Remaining bottlenecks (pre-v14, PMC-confirmed, B2H8S2k MHA)

Historical PMC (v12, before V transpose vectorization) showed
`SQ_LDS_BANK_CONFLICT ≈ SQ_WAIT_INST_LDS` ⇒ LDS bank conflicts
dominated. Original diagnosis said this fundamentally required `v_perm_b32`
exposure in FlyDSL, which was believed blocking.
**~~Correction~~ post-v14**: V transpose DID get vectorized via
`ds_swizzle` (cross-lane swap, zero bank contention) + `vector.from_elements`
pack. No `v_perm_b32` or `permlanex16` needed. The significant perf gain
everywhere is consistent with roughly half the LDS write instructions and
no added bank conflicts.

### v16 — Inter-block K prefetch: hide K global load latency at KV block boundary

**Insight**: In the standard KV loop, K cooperative load starts AFTER the
previous iteration's barrier. The global load latency (~200–400 cycles) sits
on the critical path. But the K data for the NEXT iteration is independent of
the current iteration's GEMM1/softmax — it only depends on addresses.

**Scheme**: Issue `coop_load_k_global()` (buffer_load_dwordx4 into VGPRs) for
the next KV block BEFORE the barrier, while the current GEMM1 + softmax are
still running. After the barrier, the K data is already in VGPRs and can be
stored to LDS immediately via `coop_store_k_lds()` without waiting.

```python
# At end of KV iteration N (after GEMM1, before barrier):
k_interblock_vecs = coop_load_k_global(kv_block_start + BLOCK_N)  # next iter's K

gpu.barrier()

# Start of iteration N+1 (K data already in registers):
_waitcnt_vm_n(0)
coop_store_k_lds(k_interblock_vecs, buf_id)
```

Requires `_waitcnt_vm_n(0)` after the barrier to ensure the VMEM loads have
completed before the LDS store.

**Gain**: +2.8% across shapes. The K global load now overlaps with softmax
compute, removing it from the critical path.

### v17 — Split exp2/sum softmax pipeline: break serial dependency chain

**Insight**: The standard online softmax has a serial chain:
`fma(s, scale, -m_new)` → `exp2()` → `add(sum)`. Each exp2 depends on the
previous fma, and each sum depends on the previous exp2. With 32 elements
per lane (16 lo + 16 hi), this creates a 96-instruction serial chain that
the MFMA pipeline cannot overlap.

**Scheme**: Split into two passes:
1. **exp2 pass**: compute all 32 `exp2(fma(s, scale, -m_new))` values first.
   The fma→exp2 dependencies are independent per element, so the compiler
   can interleave multiple fma→exp2 pairs.
2. **sum pass**: accumulate the 32 exp2 results into `l_new`.

```python
# Pass 1: all exp2 (independent per element)
p_vals_lo = []
for r in range_constexpr(16):
    d = math_dialect.fma(s_raw_lo[r], c_sm_scale_log2e, neg_sm_new)
    p_vals_lo.append(rocdl.exp2(T.f32, d))
p_vals_hi = []
for r in range_constexpr(16):
    d = math_dialect.fma(s_raw_hi[r], c_sm_scale_log2e, neg_sm_new)
    p_vals_hi.append(rocdl.exp2(T.f32, d))

# Pass 2: sum (serial but short)
l_sum = c_zero_f
for r in range_constexpr(16):
    l_sum = arith.AddFOp(l_sum, p_vals_lo[r], fastmath=fm_fast).result
for r in range_constexpr(16):
    l_sum = arith.AddFOp(l_sum, p_vals_hi[r], fastmath=fm_fast).result
```

**Gain**: +3.1%. The exp2 pass has enough ILP for the compiler to fill
MFMA pipeline bubbles with exp2/fma pairs.

### v18 — v_perm_b32 bf16 pack for P softmax output

Replace the manual bit-shift bf16 truncation pack (AND + SHR + OR per pair)
with `v_perm_b32` via `rocdl.v_perm_b32_i32()`. The perm instruction
extracts arbitrary bytes from two 32-bit sources in a single cycle:

```python
def bf16_trunc_pack_v4(f32_vals):
    sel = arith.constant(0x07060302, type=T.i32)
    a = arith.bitcast(T.i32, f32_vals[0])
    b = arith.bitcast(T.i32, f32_vals[1])
    lo = rocdl.v_perm_b32_i32(b, a, sel)
    c = arith.bitcast(T.i32, f32_vals[2])
    d = arith.bitcast(T.i32, f32_vals[3])
    hi = rocdl.v_perm_b32_i32(d, c, sel)
    return vector.from_elements(v2i32_type, [lo, hi])
```

**Gain**: Cleaner codegen (fewer VALU instructions for P packing). Measurable
only in aggregate with v16+v17.

### Post-v18 scheduling A/B test results

Post-v18 (with K interblock prefetch + split softmax + v_perm pack), the
kernel was extensively A/B tested against every known scheduling technique.

All reference-kernel-inspired scheduling and structural optimizations
proved neutral or negative:

| Technique | Result | Why |
|---|---|---|
| PV pipeline (depth 1/2/4) | −1.3% | Adds VGPRs (214→220), opposite of intent |
| K XOR swizzle (K_PAD=0) | −16% | Extra VALU (AND/SHL/XOR per K read) not hidden in pipeline bubbles |
| Persistent kernel | −3.1% | Launch overhead savings < loop bookkeeping cost at this grid size |
| P-pack reorder (before V store) | −0.7% | Moves P-pack VALU into LDS-heavy phase |
| sched_group_barrier MFMA/DSRD interleave | −0.7% | Over-constrains the scheduler vs softer sched_mfma/sched_dsrd hints |
| Remove all scheduling hints | neutral | Default scheduling is roughly as good as our hints |
| post-misched=0 | neutral | Already enabled in production config |
| XSUB pipeline (2-subtile interleave) | neutral | Added complexity doesn't improve instruction mix |

**Conclusion**: All scheduling optimizations are neutral or negative.
The kernel structure is already optimal in critical paths (V LDS layout,
softmax pipeline, loop carry variables). Our O store is actually better
(vectorized v4bf16 vs scalar). Further gains require CK V3-style ISA-level
instruction scheduling.

Post-v18 performance (MI308X, bf16, causal):

| Shape | P50 (ms) | TFLOPS |
|---|---|---|
| B1S4096H32D128 | 1.573 | 87.4 |
| B1S8192H32D128 | 5.344 | 102.9 |
| B1S16384H32D128 | 20.047 | 109.7 |
| B2S4096H32D128 | 2.935 | 93.6 |
| B2S8192H32D128 | 5.260 | 104.4 |

**Peak**: 109.7 TFLOPS at B1S16384 (53.3% of 206T peak bf16).

## What would close the remaining gap (post-v18)

1. **Q-head batching within one workgroup** for GQA (one WG processes
   GROUP_SIZE Q-heads sharing one K/V load) → biggest remaining GQA
   opportunity on small-S shapes.
2. ~~**Persistent kernel**~~ — **Tested in v18 cycle, −3.1%**. Launch
   overhead savings < loop bookkeeping cost at this grid size.
3. ~~**ds_swizzle latency hiding**~~ — **Done in v15** (+4.6%).
4. **v_cvt_pk_bf16_f32 (gfx950 only)** — not available on MI308X (gfx942).
   Would be a win on MI355X (sibling recipe).
5. **CK V3–style four-phase instruction scheduling** — the remaining
   performance gap is dominated by instruction scheduling quality.
   Extensive A/B testing in v18 confirmed: all scheduling hint
   permutations (sched_group_barrier, sched_mfma/dsrd removal, iglp_opt)
   are neutral or negative. CK V3 uses a `CoreLoopScheduler` that does
   cycle-level MFMA/VALU/TRANS/SALU interleaving at the ISA template
   level. Bridging this in FlyDSL would require manually emitting the
   four-phase interleaved loop body via inline asm.
6. **DPP-based softmax row-reduce** — replace LDS-mediated reductions
   with `v_mov_b32 dpp` butterfly pattern. At 2 waves/SIMD this may be
   invisible (see pitfall #20), but worth retesting at higher occupancy.
7. **AccumVGPR reduction to ≤128** — would unlock 3+ waves/SIMD. Current
   gap is 214−168=46 VGPRs, too large to bridge without fundamentally
   changing the tile shape or D_CHUNKS decomposition.

## Sustained recipe (do these, in this order)

1. Single-buffered K + V at minimum LDS (no prefetch3, no DMA on gfx942).
2. K_PAD = 4 elements (264 B/row = 33 dwords, gcd(33,32)=1 ⇒
   conflict-free read; no XOR swizzle needed). See pitfalls doc for the
   full bank-stride math.
3. VT_STRIDE = BLOCK_N + 2 = 66 elements (132 B/row = 33 dwords ⇒
   conflict-free V read). DO NOT use BLOCK_N + 8 from the FP8
   SageAttention recipe — see pitfalls.
4. `rocdl.exp2(T.f32, x)` everywhere in softmax (NOT `arith.ArithValue.exp2`).
5. Pre-load all V into SSA before a pure-MFMA PV loop.
6. Auto BLOCK_M=256 for `B * S * H_q ≥ 32 768`, else BLOCK_M=128.
7. Block decomposition: head_idx fast, q_tile slow (already standard, but
   verify if you fork — q_fast is 30–76% slower).
8. **V transpose store via `ds_swizzle` XOR-N + paired `ds_write_b32`**.
   The XOR mask auto-picks from `THREADS_PER_ROW_LOAD`: `offset:0x201F`
   (XOR-8) for VEC_WIDTH=16 / default, `offset:0x401F` (XOR-16) for
   VEC_WIDTH=8. Use `_llvm.inline_asm` with literal offset (rocdl wrapper
   does NOT fold the i32 arg to the required hardware immediate). Pack
   `(own_elem, peer_elem)` via `vector.from_elements(_v2_type, ...)` —
   NOT via `v_perm_b32` inline asm (byte-sel semantics diverge from
   CK_tile's documented pattern on current LLVM).
9. LLVM flags: `enable-post-misched=True`, `lsr-drop-solution=True`,
   `amdgpu-early-inline-all=True`. (`misched-postra-direction=2` is
   rejected by current LLVM build.)
10. **ds_swizzle latency hiding**: set `has_side_effects=False` in
    `_llvm.inline_asm` for ds_swizzle, and remove the embedded
    `s_waitcnt lgkmcnt(0)` from the asm string. ds_swizzle uses the
    cross-lane permute network, not LDS storage, so it has no WAR
    hazards with surrounding LDS ops. The compiler's register dependency
    tracking inserts waits only where the swizzled value is consumed.
    +4.6% validated on MI308X.
11. **Inter-block K prefetch**: issue `coop_load_k_global()` for the next
    KV block BEFORE the barrier, while current GEMM1 + softmax are
    running. After barrier, K data is already in VGPRs — store to LDS
    immediately via `coop_store_k_lds()` with `_waitcnt_vm_n(0)`.
    Hides K global load latency (~200–400 cycles). +2.8% validated.
12. **Split exp2/sum softmax pipeline**: separate the softmax into two
    passes — (1) compute all 32 `exp2(fma(s, scale, -m_new))` values
    (independent per element, high ILP), then (2) sum the results. Breaks
    the serial fma→exp2→add chain that creates a 96-instruction
    dependency. +3.1% validated.
13. **v_perm_b32 bf16 pack**: use `rocdl.v_perm_b32_i32()` with sel
    `0x07060302` to extract upper-16-bit bf16 from two f32 values in a
    single cycle, replacing manual AND/SHR/OR bit manipulation. Cleaner
    codegen, fewer VALU instructions for P softmax output packing.

## Related docs

- Companion pitfalls: `pitfalls/amd/flydsl/flash-attn-pitfalls.md`
- FP8 sibling: `cdna3-sage-attention-flydsl-optimization.md`
- LDS bank conflicts: `kernel-opt/amd/common/lds-bank-conflict-optimization.md`
- Reference impl: `reference-kernels/amd/cdna3/flydsl/FlyDSL/flash_attn_func_mi308x.py`
