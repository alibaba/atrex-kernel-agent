# Pitfalls: FlyDSL Flash Attention on MI308X (gfx942)

Applicability: backend: flydsl; hardware: amd; topic: pitfalls

This document collects the non-obvious traps encountered while optimizing
the FlyDSL flash-attention forward kernel on MI308X. Companion: the
optimization journey doc at
`docs/amd/cdna3/mi308x/ref-docs/flydsl/cdna3-flash-attention-bf16-gqa-optimization.md`.

Also see: [attention-backward-dkdv-pitfalls.md](attention-backward-dkdv-pitfalls.md)
for backward-specific traps (Two-Pass regression, block_n=64 silent row drop,
cooperative LDS transpose, multi-wave regression).

The general theme: **bank-conflict math depends on element width and write
pattern, not just stride values copied from another recipe.**

> **See also**: `flash-attn-d256-pitfalls.md` for head_dim=256 specific pitfalls on MI355X (CDNA4/gfx950).

---

## 1. VT_STRIDE = BLOCK_N + 8 from the SageAttention recipe regresses bf16 by 5–12%

**Trap**: SageAttention's V27 recipe famously gained +78% by changing
`VT_STRIDE` from 66 to 72 (= BLOCK_N + 8). The doc says "ensure 8-byte
alignment for `ds_read_b64`." A naive port to bf16 says: my read is also
`ds_read_b64`, my BLOCK_N is also 64, so VT_STRIDE = 72 must be right.

**Result of the port**: every shape regressed 5–12%.

**Why**: the recipe is for **FP8** (1 byte/elem). Stride is in BYTES of
LDS rows, and bank conflicts are computed in 4-byte dwords:

| dtype | VT_STRIDE elements | Bytes/row | Dwords/row | gcd(stride, 32 banks) | Conflict |
|---|---|---|---|---|---|
| FP8 | 66 | 66 | 16.5 (not integer) | n/a | misaligned ds_read_b64 |
| FP8 | **72** | **72** | **18** | **gcd(18,32)=2** | **2-way (acceptable)** |
| bf16 | 66 | 132 | 33 | **gcd(33,32)=1** | **0 (conflict-free)** ✓ |
| bf16 | 72 | 144 | 36 | gcd(36,32)=4 | 4-way conflict |

For bf16 the existing VT_STRIDE = 66 (i.e. BLOCK_N+2) is **already
conflict-free** because 33 dwords is coprime with 32 banks; bumping to 72
deliberately introduces a 4-way conflict on every ds_read_b64.

**Lesson**: when porting a stride from a recipe in another dtype, redo the
bank-conflict math with `gcd(byte_stride / 4, 32)`. The optimal stride
depends on element width.

---

## 2. K_PAD = 4 looks 16-byte-misaligned but is the right choice anyway

**Trap**: The K LDS layout uses `K_STRIDE = HEAD_DIM + K_PAD = 132`
elements, i.e., 264 bytes/row. Naively 264 % 16 = 8 ⇒ NOT 16-byte aligned,
which violates `ds_write_b128`'s alignment requirement. Sweep says K_PAD=8
is the next 16-byte-aligned candidate and benches identically. Tempting to
"fix" K_PAD to 8.

**Result of changing to K_PAD=8**: identical perf at best, slight
regression at worst (264 B → 272 B, 33 → 34 dwords stride; gcd(33,32)=1
beats gcd(34,32)=2).

**Why**: `ds_write_b128` does NOT require row-base alignment to 16 B;
it only needs the per-thread address to be 16 B aligned. Per-thread
addresses for the cooperative store are always at multiples of
`load_col_base * 2` bytes within a row, which IS 16-aligned for
VEC_WIDTH=16 (32 B per thread). Row-to-row stride being non-16-aligned
just means rows start at different alignment classes within a 16-byte
window — the per-instruction alignment is fine.

**Lesson**: 16-byte alignment for `ds_write_b128` applies to the address
issued by each lane, not to the row-base of every LDS row. Row strides
should be picked to MAXIMIZE bank diversity (gcd(dword_stride, 32) = 1),
not to satisfy a row-base alignment constraint that doesn't exist.

---

## 3. `arith.ArithValue.exp2(fastmath=fast)` lowers to `v_ldexp + v_cmp + v_cndmask`, not `v_exp_f32`

**Trap**: FlyDSL's `arith.ArithValue` exposes an `.exp2(fastmath=...)`
method. The natural assumption is that `fastmath=arith.FastMathFlags.fast`
selects the fast (single `v_exp_f32`) lowering.

**Reality**: `.exp2()` calls `math.exp2` which expands to a multi-op
sequence even with fast-math. SageAttention measured −38% softmax VALU by
switching to direct `rocdl.exp2(T.f32, x)`.

**Lesson**: always use `rocdl.exp2(T.f32, x)` for hot-path exp2. The
high-level wrapper is for portability, not throughput.

---

## 4. `sched_mfma(4)` regresses the interleaved-loadV PV loop but helps the pure-MFMA variant

**Trap**: `rocdl.sched_mfma(N)` is documented as a hint to interleave
ds_read with MFMA. Adding it to the PV inner loop "should" help.

**Result, before v6**: +1% regression on every shape.
**Result, after v6 (pre-load V, pure-MFMA loop)**: no measurable change
(loop is already pure MFMA, hint is harmless but redundant).

**Why**: in the "load + MFMA interleaved" loop, the hint conflicts with
the explicit ds_reads inside the loop body, forcing the scheduler into a
suboptimal arrangement. In the "all-V-pre-loaded + pure MFMA" loop, the
hint is decoration — the body is already what `sched_mfma(N)` aims for.

**Lesson**: scheduling hints assume a specific loop shape. Verify the
shape before adding them. SageAttention's V27 sched_mfma(4) win was
specifically on its pre-loaded-V structure.

---

## 5. Deferred per-dc rescale interleaved with MFMA regresses vs all-upfront

**Trap**: Sage V9 says "delay O rescaling — `o_accs[1..3]` rescale at PV
sub-tile 0 setup". Sounds like the rescale should be tucked inside the
MFMA loop at the (dc==k, pks==0) boundary so the MULs overlap with
in-flight MFMAs.

**Result of porting**: regressed back to v0 numbers.

**Why**: with the v6 pre-load-V structure, the MUL is no longer naturally
adjacent to a free VALU slot. Putting the conditional `if pks==0 and
dc>0:` inside the loop creates a per-iter branch that the compiler can't
eliminate cleanly. Doing all rescales BEFORE the MFMA loop lets them
hoist outside any in-flight MFMA dependency.

**Lesson**: "deferred rescale" is shape-dependent. The benefit is
overlapping the rescale VALU with prior MFMAs; if your loop already has
a long stretch of uninterrupted MFMAs and you don't have register
pressure, doing the rescales upfront is simpler and faster.

---

## 6. `ds_bpermute`-based V transpose vectorization gains nothing

**Trap**: V transpose store is the dominant LDS bank conflict source
(2.06 M conflicts ≈ 2.07 M LDS waits per kernel). CK_tile uses
`v_perm_b32` (intra-lane byte permute) to vectorize this. FlyDSL doesn't
expose `v_perm_b32`, but it does expose `ds_bpermute` (cross-lane gather
through LDS). One can swap data with lane (tid XOR 8) so each even-row
lane has data for 2 adjacent V rows, then write `ds_write_b32` instead of
two scalar `ds_write_b16`. Halves the writes — must be a win.

**Result**: zero perf change.

**Why**: `ds_bpermute` is **not free**. It uses the LDS hardware
internally for the cross-lane gather. 8 bpermute calls per lane (one per
dword to swap an entire v16bf16 vector) ≈ 8 LDS reads worth of
LDS-fabric traffic. That cancels the savings from cutting 8 ds_writes per
lane. Net LDS pressure is unchanged.

**Lesson**: `ds_bpermute` has hidden LDS-bandwidth cost that cancels
savings from packed writes. `v_perm_b32` (intra-lane) would work in
principle but sat in a "FlyDSL doesn't expose it" dead-end for months.

**⚠ SUPERSEDED by v14 (see pitfall #10–13 below)**: there IS a viable
path on gfx942. `ds_swizzle_b32` (not `ds_bpermute`) does cross-lane
permute THROUGH the LDS instruction port WITHOUT consuming LDS bank
cycles — gfx942's LDS unit has a dedicated permute data-path for it.
Combined with `vector.from_elements` explicit pack (bypassing
`v_perm_b32` entirely), V transpose DID vectorize for −22% to −24% gain.
Old conclusion "v_perm_b32 required" was wrong for gfx942.

---

## 7. `misched-postra-direction=2` LLVM flag is silently rejected on current LLVM

**Trap**: SageAttention V23 lists this as one of the LLVM hints that
helped. Adding it to the FlyDSL `compile_hints` looks safe.

**Reality**: current LLVM build prints `Cannot find option named '2'!`
on every dispatch. The flag is rejected; nothing breaks but nothing
helps. Adds noise to stderr that masks real warnings.

**Lesson**: compile-hint LLVM options drift between LLVM versions. Verify
the flag is recognized before assuming it's helping; check stderr for
the rejection message.

---

## 8. q_tile-fast block decomposition wrecks SE load balance on MI308X

**Trap**: a "natural" decomposition of `block_id` is q_tile-fast / head-slow:

```python
q_tile_idx = block_id % num_q_tiles
head_idx   = block_id // num_q_tiles % NUM_HEADS
```

This puts each head's full q_tile sequence on consecutive block_ids.

**Result**: 30–76% slower on every Qwen GQA shape than head-fast.

**Why**: MI308X has 16 SEs × 5 CUs and dispatches block_id `i` to SE
`i mod 16`. With q_tile-fast, SE 0 gets blocks `{0, 16, 32, ...}` =
all the same q_tile of different heads — but for Qwen H_q=16 with 8
q_tiles, SE 0 ends up with all `q_tile=0` (light) blocks, while SE 7
ends up with all `q_tile=7` (heavy) blocks. The pipeline waits for the
slowest SE.

The correct head-fast decomposition (`head_idx = block_id % NUM_HEADS`)
gives each SE a balanced mix: SE 0 gets head 0 across all q_tiles, SE 1
gets head 1 across all q_tiles, etc. Each SE accumulates the same total
work.

**Lesson**: on AMD CDNA3, block scheduling is SE-major then CU-preemptive
within each SE. Always put the **uniform-compute** dimension in the fast
position of `block_id`, the **non-uniform-compute** dimension (causal
q_tile, varying batch lengths) in the slow position.

---

## 9. Aiter source and ROCm 6.4.3 compatibility notes

These notes come from historical attempts to use AITER as a comparison
baseline in a restricted ROCm 6.4.3 environment. Prefer local snapshots or
consumer-provided reference material when available; no task-time network access
is required by this wiki page.

### 9a. Upstream source provenance

Reference source: AITER upstream was observed via the ROCm/aiter GitHub
repository, with `codeload.github.com` used as historical provenance for the
source snapshot when direct repository access was unavailable. Treat both
`aiter` and its `composable_kernel` submodule as local reference material once
snapshotted.

### 9b. ROCm 6.4 renamed `hipDeviceAttributePciChipId` → `hipDeviceAttributePciDeviceId`

aiter main branch's `csrc/include/aiter_hip_common.h` references
`hipDeviceAttributePciChipId`. ROCm 6.4 dropped this name; compile fails
with "use of undeclared identifier ... did you mean
hipDeviceAttributePciBusId?". Patch:

```cpp
// Before
HIP_CALL(hipDeviceGetAttribute(&id, hipDeviceAttributePciChipId, dev));
// After (ROCm 6.4)
HIP_CALL(hipDeviceGetAttribute(&id, hipDeviceAttributePciDeviceId, dev));
```

Also add `#include <unordered_map>` — aiter relies on a transitive
include that ROCm 6.4 dropped.

### 9c. torch 2.4 `_library.infer_schema` doesn't accept `torch.dtype` defaults

aiter's `quant.py` registers ops like `def per_token_quant_hip(... quant_dtype: torch.dtype = torch.int8, ...)`.
torch 2.4's schema inferrer raises
`ValueError: Parameter quant_dtype has an unsupported default value (we only support int, float, bool, None)`.

Monkey-patch `torch._library.infer_schema.infer_schema` (and the legacy
`torch._custom_op.impl.infer_schema`) to substitute `None` for any
non-(int,float,bool,None) default before dispatching to the original.
Also add a shim at `torch.library.infer_schema` since aiter probes
`hasattr(torch.library, "infer_schema")` (added in torch 2.5).

### 9d. aiter complains about flydsl version then disables CK ops, but mha_batch_prefill still works

`Unsupported flydsl version: expected 0.1.3.1, got 0.1.2.dev462. CK and
HIP ops are disabled. Triton ops remain available.` Looks fatal — but
`mha_batch_prefill_func` uses CK_tile codegen at JIT time, not the
flydsl-tagged ops, so it still compiles and runs correctly. Ignore the
warning.

### 9e. First call to `mha_batch_prefill_func` JIT-compiles for ~100 s

The CK_tile generator + hipcc takes ~100 s on the first call for one
specific (dtype, dropout, mask, ...) combination. Cached on disk — second
call within the same session uses the cached `.so`. Plan for a 100 s
warmup on the first dispatch.

---

## 10. `permlanex16` does not exist on gfx942 (CDNA3) — RDNA-only

**Trap**: Googling "cross-lane swap XOR 16" points at `permlanex16_b32`
(MLIR `rocdl.permlanex16`). LLVM doc says it's "AMDGPU", the MLIR op is
in upstream `ROCDLOps.td`, so it should work on any AMD GPU.

**Result**: LLVM codegen error at JIT time:
```
LLVM ERROR: Cannot select: intrinsic %llvm.amdgcn.permlanex16
```
Kernel fails to compile.

**Why**: `v_permlanex16_b32` is a **gfx10+ (RDNA) instruction**. CDNA1/2/3
(gfx908/gfx90a/gfx942) don't have it. The MLIR op is defined but only
lowers on the right subtarget. MLIR won't refuse the call at parse time;
the error surfaces only during ISel.

**Lesson**: for CDNA cross-lane permute within a wave, the available
options are `ds_bpermute` (uses LDS bank arbitration), `ds_swizzle` (uses
LDS permute fabric, no bank contention), `ds_permute`, and DPP modifiers
on ALU instructions. `permlanex16` / `permlane16_swap` / `permlane32_swap`
exist only on RDNA+ / gfx950+. On gfx942, **use `ds_swizzle` for
constant-offset cross-lane permute** — it's the cheapest path.

---

## 11. `rocdl.ds_swizzle(res_ty, src, imm_value)` doesn't emit the
hardware immediate offset — use `_llvm.inline_asm`

**Trap**: `rocdl.ds_swizzle(T.i32, src, arith.constant(0x401F, type=T.i32))`
— the MLIR op takes offset as `I32:$offset`, looks just like any other
value. The `arith.constant` should fold away.

**Result**: kernel runs without error but produces garbage — correctness
error ~0.48 on bf16 outputs scaled 0.1. Same output whether you pass
0x401F or 0x201F or any other value; the offset is effectively ignored.

**Why**: `ds_swizzle_b32` encodes the offset as part of the **instruction
word** — it's a hardware immediate, not an operand. The MLIR ODS wrapper
and LLVM IR intrinsic accept an `i32` SSA value, but the backend requires
a `ConstantInt` at ISel time; on this LLVM revision the constant doesn't
always propagate through the intrinsic lowering. The result is a swizzle
with whatever offset bits happened to land in the encoding — usually
wrong.

**Lesson**: for instructions whose operand encodes in the instruction
word (ds_swizzle offset, DPP ctrl, s_waitcnt imm, etc.) use
`_llvm.inline_asm` with the literal in the asm string:

```python
def _ds_swizzle_xor16_inline(src):
    return _llvm.inline_asm(
        ir.IntegerType.get_signless(32),
        [src],
        "ds_swizzle_b32 $0, $1 offset:0x401F\n\ts_waitcnt lgkmcnt(0)",
        "=v,v",
        has_side_effects=True,
    )
```

Guarantees the offset ends up in the instruction word.

---

## 12. `v_perm_b32` inline asm with CK_tile-style sel imm gives wrong bytes

**Trap**: CK_tile's `transpose_vectors.hpp` uses
`__builtin_amdgcn_perm(low, high, 0x05040100)` for 2×2 bf16 transpose and
it works in HIP. So in FlyDSL an inline asm
`"v_perm_b32 $0, $1, $2, $3"` with the same sel imm must work too.

**Result**: kernel runs, cross-lane swap and predicate are verified
correct (via scalar b16 debug variant), but the packed v_perm output is
wrong. Max abs error 0.477 (vs 9.77e-04 with the same cross-lane + scalar
pack). Perf is great (−22%) but numbers are meaningless.

**Why**: HIP's `__builtin_amdgcn_perm` and the raw `v_perm_b32` inline
asm differ in how the sel imm is interpreted on the current LLVM
revision. The HIP builtin maps sel bytes directly to source-byte
selectors; the inline asm encoding appears to route through a different
SDWA-ish path where the byte-order convention is the opposite. The
CK_tile-documented pattern (0x05040100 / 0x07060302) does NOT produce
`{low16_a, low16_b}` / `{high16_a, high16_b}` packs as expected.

**Lesson**: don't try to hand-write byte permutes via `v_perm_b32` inline
asm unless you're ready to dump disassembly and verify sel encoding.
**Use `vector.from_elements(_v2_type, [own_elem, peer_elem])` to pack two
bf16 lanes into a v2bf16** — the compiler picks the right instruction
(v_pack_b32_f16 / v_lshl_or_b32 / v_mov_b32 + dword writes) and emits
correct code at the same speed. Same pattern in FlyDSL for CDNA3, CuteDSL
for Hopper, and Triton: prefer high-level vector ops over hand-rolled
byte permutes unless you have a benchmarkable win from the lower level.

---

## 13. VEC_WIDTH=8 (FLYDSL_FLASH_ATTN_FUNC_ENABLE_LDS_VEC16=0) has a
pre-existing correctness bug; do not toggle it to aid cross-lane swap design

**Trap**: `VEC_WIDTH=16` (default) makes `THREADS_PER_ROW_LOAD=8`, which
puts the "adjacent-row peer" at `lane XOR 8` — awkward for `permlanex16`
(XOR-16). Switching to `VEC_WIDTH=8` gives `THREADS_PER_ROW_LOAD=16`, so
peer is at `lane XOR 16` — clean. So just flip the env var and iterate.

**Result**: baseline (NO v14 changes) at VEC_WIDTH=8 has max abs error
0.030 — way above bf16 noise (~1e-3). The whole kernel is subtly wrong at
this setting. Adding any v14 change stacks a second bug on top and
obscures debugging (err 0.477 reported for a while because both bugs
combined).

**Why**: not diagnosed yet. `VEC_WIDTH=8` changes K_STRIDE, LDS tile
sizes, load/store granularity — somewhere in those derived constants
there's an off-by-something that the VEC_WIDTH=16 path doesn't hit.

**Lesson**: **always verify the baseline first** when a config knob
promises to simplify your optimization. The env gate
`FLYDSL_FLASH_ATTN_FUNC_ENABLE_LDS_VEC16=0` is marked in comments as "for
legacy compatibility" but is effectively **broken for production**. v14
uses VEC_WIDTH=16 + `ds_swizzle` XOR-8 instead (offset:0x201F) to avoid
touching VEC_WIDTH at all.

---

## 14. `bf16_trunc_pack_v*` is not dtype-aware — silently wrecks fp16 output

**Trap**: To pack 4/8 f32 accumulator values into v4bf16 / v8bf16 for the
P softmax output (and v13's O writeback), it's tempting to AND off the low
16 bits of each f32 and OR-shift them into i32 dwords. bf16 IS the upper
16 bits of f32 (same exponent layout, same bias, just truncated mantissa)
so this is ~2 fewer ops per element than `arith.TruncFOp`. Helper named
`bf16_trunc_pack_v4` made this explicit.

The same helper got reused unconditionally for the `dtype_str="f16"` build.

**Result**: fp16 outputs are garbage (max abs err ≫ noise — the kernel
"runs" but values are arbitrary). Bug went unnoticed because the entire
optimization journey (v0 → v14, all perf benches, all correctness checks)
was run on bf16 only. fp16 build only flagged when a downstream user
exercised it.

**Why**: fp16 (IEEE binary16) has a 5-bit exponent and bias 15. f32 has
8-bit exponent and bias 127. The high 16 bits of f32 are NOT a valid fp16
representation — different exponent field width, different bias, completely
different bit layout. Bitwise truncation produces an fp16 with whatever
random bits happened to be in the top half of the f32 → essentially
random data.

**Lesson**: 16-bit float pack helpers must branch on dtype:

```python
_IS_BF16 = (dtype_str == "bf16")

def pack_v4(f32_vals):
    if _IS_BF16:
        # bit truncation: bf16 is upper 16 bits of f32
        return ...bitwise OR/AND/shift code...
    # fp16 path: real round-to-nearest f32 → f16
    elems = [arith.trunc_f(elem_type, v) for v in f32_vals]
    return vector.from_elements(v4f16_type, elems)
```

The fp16 path is ~2 inst/elem slower but still much faster than scalar
stores because the pack is single-instruction at the LLVM level. After fix:
fp16 max abs err drops from ~0.5 to ~1.2e-4 (proper fp16 noise level).

**Always test every supported dtype before shipping** — don't trust that
a "bf16 helper" will silently work for fp16 just because both are "16-bit
float".

---

## Quick reference: what to set vs what to avoid

| Setting | Use | Avoid |
|---|---|---|
| K_PAD (bf16) | 4 | 0, 8, 16 (regress) |
| VT_STRIDE (bf16) | BLOCK_N + 2 = 66 | BLOCK_N + 8 = 72 (FP8 recipe) |
| exp2 | `rocdl.exp2(T.f32, x)` | `arith.ArithValue.exp2(fastmath=...)` |
| PV loop | pre-load V into SSA, pure-MFMA inner loop | interleaved load+MFMA |
| sched_mfma(4) | inside pure-MFMA PV loop | inside interleaved loop |
| O rescale | upfront before PV loop | per-iter conditional inside loop |
| Block decomposition | head_idx fast, q_tile slow | q_tile fast, head_idx slow |
| LLVM flags | `enable-post-misched`, `lsr-drop-solution`, `amdgpu-early-inline-all` | `misched-postra-direction=2` (rejected) |
| Cross-lane swap on gfx942 | `ds_swizzle` w/ `_llvm.inline_asm` literal `offset:0x…` | `permlanex16` (RDNA-only), `rocdl.ds_swizzle(i32, src, ssa_offset)` (imm not folded), `ds_bpermute` (LDS bank contention) |
| V^T pack 2×bf16→i32 | `vector.from_elements(_v2_type, [a, b])` | `v_perm_b32` inline asm w/ CK sel imm (byte-order bug) |
| VEC_WIDTH | 16 (default) | 8 (LDS_VEC16=0 has correctness bug at baseline) |
| V transpose vectorization | ds_swizzle XOR-N + vector.from_elements + ds_write_b32 (v14) | scalar ds_write_b16 (pre-v14), ds_bpermute pairing (no net win) |
| f32 → 16-bit float pack | dtype-aware: `arith.trunc_f` for fp16, bit-truncate top 16b for bf16 | unconditional bit-truncation (fp16 → garbage) |
| Mask loading (f32 mask) | `global_load_dwordx4` (v4f32), exploit MFMA register → KV column mapping | scalar f32 loads (32 per subtile, ~6.4ms overhead) |
| Mask loading (binary mask) | Bit-packed u32 bitmask, 2× `buffer_load_dword` per subtile (32× BW reduction) | f32 mask when mask is binary {0, -1e6} |
| Mask penalty value | -1e6 (large finite negative) | -inf (causes NaN in online softmax when Q rows have zero attend positions) |
| Mask application | Additive: `score + select(attend, 0, -1e6)` | Replacement: `select(attend, score, -1e6)` (breaks correction factor by ~3.5×) |
| GEMM1 sched hint (with mask VALU) | `sched_barrier(0)` (unconstrained, lets mask VALU fill MFMA bubbles) | `sched_dsrd(2) + sched_mfma(2)` (prevents VALU interleave, −1.8%) |
| BLOCK_M (D=64) | 128 (amortize per-tile mask/softmax overhead) | 64 (−45%, per-tile overhead dominates) |
| head_dim=64 support | D_CHUNKS=2, mfma_32x32x8, native D=64 | Pad D=64→128 (2× compute waste) |
| Mask stride for 3D→4D broadcast | Pass 2D strides `(mask_stride_b, mask_stride_s)` from 4D view | Use 3D tensor strides directly (misses batch dimension) |
| ds_swizzle `has_side_effects` | `False` (let compiler schedule around latency, +4.6%) | `True` + embedded `s_waitcnt` (forces serialization) |
| K/V LDS sharing | Separate K/V LDS regions (avoid extra barriers) | Shared region with 2 extra `s_barrier` per iter (30% regression) |
| K_PAD (bf16, alignment) | 4 (264B/row, ds_write_b128 aligned) | 2 (260B/row, zero conflicts but 2× ds_write_b64) |
| `waves_per_eu` at 2 waves/SIMD actual | Keep at 3 (no effect from matching) | Changing to 2 (no benefit, no harm) |
| `iglp_opt` on fused attention | Don't use (neutral with existing sched hints) | `iglp_opt(2)` (conflicts with sched_mfma/sched_dsrd) |
| PV loop pipelining | Pre-load ALL V into SSA (recipe item 5) | Depth-N interleaved pipeline (+6 VGPRs, −1.3%) |
| K LDS swizzle | K_PAD=4 padding (no address math) | XOR swizzle K_PAD=0 (−16%, extra VALU not hidden in pipeline bubbles) |
| Persistent kernel (attention) | Standard dispatch (large grid) | Persistent loop (−3.1%, per-tile overhead > launch savings) |
| GEMM1 scheduling hint type | Soft hints: `sched_dsrd(2)` + `sched_mfma(2)` | Hard `sched_group_barrier` 2:2 interleave (−0.7%) |
| P softmax pack placement | After barrier + V store (hidden by MFMA bubbles) | Before V store (−0.7%, competes with V-store address math) |
| bf16 truncation pack | `v_perm_b32` via `rocdl.v_perm_b32_i32(b, a, 0x07060302)` | Manual AND/SHR/OR bit manipulation (more VALU) |
| Softmax exp2/sum | Split into 2 passes: all exp2 first, then sum (+3.1% ILP) | Interleaved fma→exp2→add serial chain |
| K global load timing | Inter-block prefetch: issue before barrier (+2.8%) | After barrier (K load latency on critical path) |
| No-mask CK comparison | Match dtype first; use BF16 CK only for BF16 CK95, and local CK ATT only as scheduling evidence | Compare BF16 FlyDSL against the historical FP16 CK/SDPA row |
| No-mask reduction wait | `ds_bpermute_lgkm_sum`: rowsum gets `s_waitcnt lgkmcnt(0)`, VMEM wait delayed | Full-drain `vmcnt(0) lgkmcnt(0)` immediately after first reduction |
| No-mask QK prefetch | Keep depth 2 until ATT proves otherwise | Copy CK `_QK_PREFETCH_DEPTH=3` mechanically |
| No-mask softmax barrier | Keep barrier enabled with mask `0` | Disable it, or copy CK-like masks `1` / `0x7f` without profile proof |
| No-mask rescale placement | Rescale only the required accumulator chunk before VMEM wait | `EARLY_RESCALE_ALL=1` as a blanket latency-hiding rule |

---

## 15. head_dim=64 padded to 128 wastes 2× compute and memory

**Trap**: The MI308X causal+GQA kernel (`flash_attn_func_mi308x.py`) requires
`head_dim % 128 == 0` (uses `D_CHUNKS=4` with `mfma_f32_32x32x16_bf16`).
When given D=64, the obvious fix is to pad K/V to D=128 and set `K_PAD`
accordingly. "It'll just waste a bit of LDS."

**Result**: 20.3 ms for B1024×H8×S316×D64 — **2× slower** than necessary.
The kernel computes 4 D-chunks of MFMA where only 2 contain real data.
Every VMEM load, LDS read, and MFMA instruction is doubled. VGPR pressure
also doubles, forcing occupancy down to 1 wave/SIMD.

**Why**: The MFMA tile `32x32x16` processes 16 columns per instruction.
With D=128, you need 4 chunks (D_CHUNKS=4). With D=64, you only need 2
chunks using `mfma_f32_32x32x8` — which processes 8 columns per instruction
and is the correct tile for D=64. The entire datapath (Q load, K load,
V load, MFMA, O store) scales linearly with D_CHUNKS.

**Lesson**: Never pad head_dim to the next power-of-two. Adjust `D_CHUNKS`
and MFMA tile to match actual head_dim. For D=64: `D_CHUNKS=2`,
`mfma_f32_32x32x8_bf16`. For D=128: `D_CHUNKS=4`, `mfma_f32_32x32x16_bf16`.

---

## 16. Scalar mask loads dominate runtime for free-mask attention

**Trap**: Adding free-mask support to Flash Attention seems straightforward:
inside the KV loop, after computing S = Q·K^T, load `mask[q_row, kv_col]`
for each MFMA output element and add it to the score. With MFMA 32×32
layout, each thread holds 16 lo + 16 hi score values → 32 scalar f32
loads per thread per subtile. "32 loads is fine, it's just a few extra
instructions."

**Result**: Mask loading takes ~6.4 ms out of 20.3 ms total — **31.5% of
runtime** is just loading mask values.

**Why**: 32 scalar `global_load_dword` per thread × 256 threads = 8192
global loads per subtile. Each is a separate instruction with its own
address computation, bounds check, and `s_waitcnt`. The instruction
count bloats the VALU pipeline.

**Fix**: Exploit the MFMA 32×32 register layout: registers r=4g, 4g+1,
4g+2, 4g+3 (for g=0..3) map to 4 CONSECUTIVE KV columns. Load these 4
mask elements in one `global_load_dwordx4` (128-bit wide load), then
`vector.extract` each element. This reduces 32 scalar loads → 8 vector
loads per subtile (4× fewer instructions). On our test shape this cut
the mask overhead from ~6.4 ms to ~0.25 ms.

**Lesson**: Always check the MFMA register-to-output-element mapping before
deciding on a load strategy. If consecutive registers map to consecutive
memory addresses, use wide vector loads (`dwordx4`, `dwordx2`).

---

## 17. 3D mask tensor stride ignores batch dimension for 4D broadcast

**Trap**: The attention mask is conceptually (B, 1, S, S) but PyTorch
`F.scaled_dot_product_attention` receives it as a 4D tensor where dim=1
is broadcast over heads. When passing strides to the kernel, you might
compute `mask_stride_s = mask.stride(2)` from the 3D physical storage
(B, S, S) and use `mask[batch * B_stride + q * stride_s + kv]` as the
index. This "works" when B=1 but produces garbage for B>1.

**Result**: Incorrect attention output for B>1. The kernel reads mask rows
from the wrong batch, producing rel_err > 0.5.

**Why**: The 4D view `(B, 1, S, S)` has strides `(S*S, 0, S, 1)` where
dim=1 stride is 0 (broadcast). The 3D physical tensor `(B, S, S)` has
strides `(S*S, S, 1)`. If you pass the 3D strides, the kernel uses
`stride(0)=S*S` for the batch and `stride(1)=S` for the query row —
which matches. But if you accidentally use the 4D view's `stride(1)=0`
as the batch stride, or mix up dimension indices, the batch offset is
wrong.

**Lesson**: When passing tensor strides as kernel arguments for broadcast
dimensions, explicitly extract `mask_stride_b = mask.stride(0)` and
`mask_stride_s = mask.stride(2)` from the 4D view. Never assume the
physical storage layout matches the logical view. Add an assertion:
`assert mask.shape[1] == 1, "mask must broadcast over heads"`.

See also: [cdna3-flash-attention-bf16-mask-optimization.md](../../ref-docs/flydsl/cdna3-flash-attention-bf16-mask-optimization.md)
for the full optimization journey.

---

## 18. K/V LDS sharing with extra barriers regresses 30%

**Trap**: K and V each occupy ~8.4 KB of LDS (33 KB total per WG at
K_PAD=4). The CU has 64 KB LDS, so only 1 WG fits. Sharing a single LDS
region for both K and V would halve the LDS footprint to ~17 KB, allowing
2 WGs per CU (4 waves/SIMD) and doubling occupancy. "That's an easy 2×
occupancy win."

**Result**: 30% slower. The kernel went from 1.683 ms to ~2.2 ms.

**Why**: Sharing K/V LDS requires 2 additional `s_barrier` synchronizations
per KV block iteration — one before loading V (to ensure K GEMM is
complete) and one before loading the next K (to ensure V is consumed). At
2 waves/SIMD, the MFMA pipeline is already kept busy enough that the extra
barriers stall the pipeline more than the occupancy gain helps. The
barriers insert ~200 cycles of dead time per KV iteration that cannot be
filled because the wavefront is stalled waiting for all waves in the
workgroup.

**Lesson**: Higher occupancy is NOT automatically better. Extra barriers
have a fixed cost per iteration. If the existing occupancy already fills
pipeline bubbles, the barriers are pure overhead. Measure before sharing
LDS regions. (**Trap 50 resolves this**: SHARE_KV_LDS works when combined
with `waves_per_eu=4` + only 1 barrier + `K_INTERBLOCK=0`.)

---

## 19. K_PAD=2 zero bank conflicts but unaligned stores → neutral

**Trap**: K_PAD=4 gives stride 132 elements = 66 dwords, gcd(66,32)=2 →
2-way bank conflicts. K_PAD=2 gives stride 130 elements = 65 dwords,
gcd(65,32)=1 → zero bank conflicts. "K_PAD=2 should be strictly better."

**Result**: Net neutral (1.684 ms vs 1.683 ms).

**Why**: K_PAD=4 row stride is 264 bytes (16-byte aligned), so cooperative
K stores use a single `ds_write_b128` (128-bit write). K_PAD=2 row stride
is 260 bytes (not 16-byte aligned at per-thread offsets), forcing the
store to split into 2× `ds_write_b64`. The doubled store instruction count
exactly offsets the bank conflict reduction. The net LDS traffic and
latency are unchanged.

**Lesson**: Bank conflict freedom is not the only factor in LDS stride
selection. The store alignment determines instruction width
(`ds_write_b128` vs `ds_write_b64`), which affects instruction count and
scheduling. Always check both read conflicts AND write alignment when
tuning K_PAD.

---

## 20. Tree max reduction no effect at low occupancy (2 waves/SIMD)

**Trap**: The softmax row-max is computed via a sequential LDS-based
reduction across lanes. Replacing it with a tree-based DPP reduction
(butterfly pattern: `dpp_reduce(XOR-1, XOR-2, XOR-4, …)`) should reduce
the reduction depth from O(N) to O(log₂N). "Tree reduce is
algorithmically better."

**Result**: No measurable improvement (within noise, <0.5%).

**Why**: At 2 waves/SIMD, the MFMA pipeline has enough bubbles to hide
the sequential reduction latency. The reduction instructions execute during
MFMA stall cycles that would otherwise be idle. Tree reduction eliminates
work that was already "free" — the critical path is the MFMA chain, not
the reduction.

**Lesson**: Micro-optimizations that reduce instruction count but don't
shorten the critical path are invisible at low occupancy. The MFMA
pipeline is the pacemaker at 2 waves/SIMD; ancillary work (softmax
reductions, address computations) fills bubbles. Only optimize operations
that are ON the critical path.

---

## 21. `waves_per_eu=2` matching actual occupancy doesn't improve codegen

**Trap**: The kernel runs at 2 waves/SIMD due to AccumVGPR pressure.
Setting `waves_per_eu=2` (instead of 3) tells the compiler the true
occupancy, so it should generate better register allocation and
scheduling. "Telling the compiler the truth should help."

**Result**: Neutral (1.686 ms vs 1.683 ms at waves_per_eu=3).

**Why**: On CDNA3, `waves_per_eu` primarily affects the compiler's
register allocation pressure heuristic. Going from 3→2 allows the
allocator to use MORE registers (since it doesn't try to leave room for
a third wave), but the kernel is already not register-spilling at
waves_per_eu=3. The extra register freedom doesn't change the generated
code because the register demand is already met. There's also no
scheduling benefit because the AMDGPU backend's scheduler doesn't
meaningfully adjust its heuristics based on this hint alone.

**Lesson**: `waves_per_eu` is a soft hint, not a hard constraint. If the
kernel is already at the occupancy limit set by a different resource (like
AccumVGPR), changing the hint to match reality has no effect. The compiler
already generates code for the actual resource pressure, not the hint.

---

## 22. `iglp_opt(2)` scheduling hint has no effect on flash attention

**Trap**: CK (Composable Kernel) uses `iglp_opt(1)` in GEMM kernels with
measurable gains. The variant `iglp_opt(2)` is documented as enabling
"interleave scheduling" for overlapping MFMA with memory ops. Adding
`rocdl.iglp_opt(2)` before the GEMM1 section should help the compiler
interleave KV loads with MFMA. "CK does it, so should we."

**Result**: Neutral (1.681 ms vs 1.683 ms, within noise).

**Why**: `iglp_opt` provides hints to the AMDGPU post-RA scheduler. But
the flash attention loop structure (alternating GEMM1 → softmax → GEMM2)
is very different from a standalone GEMM. The scheduler already does a
reasonable job with the existing `sched_mfma` / `sched_dsrd` hints (which
are more targeted). Adding `iglp_opt` on top of existing scheduling hints
results in conflicting guidance — the scheduler falls back to its default
behavior.

**Lesson**: `iglp_opt` is effective for GEMM-only loops where CK's
template structure matches the hint's assumptions. For complex fused
kernels (attention = GEMM + softmax + GEMM), use targeted `sched_mfma` /
`sched_dsrd` / `sched_group_barrier` instead. Don't stack multiple
scheduling hint systems.

See also: [cdna3-flash-attention-bf16-gqa-optimization.md](../../ref-docs/flydsl/cdna3-flash-attention-bf16-gqa-optimization.md)
v15 section for the full list of dead-end attempts.

---

## 23. PV pipeline (interleaved V reads within MFMA loop) increases VGPRs, not decreases

**Trap**: The pre-load-all-V approach (recipe item 5) keeps all V vectors
in SSA before the MFMA loop. This uses ~32 VGPRs for the V data. A "PV
pipeline" with depth-N double buffering should reduce this to N buffer
slots (e.g., depth-2 = 4 VGPRs for 2 lo + 2 hi), freeing ~28 VGPRs.
"Pipeline reduces register pressure — basic CS."

**Result**: VGPRs increased from 214 to 220 (6 more, not 28 fewer).
Performance regressed −1.3%. Tested at depth 1, 2, and 4 — all worse.

**Why**: Interleaving V reads inside the MFMA loop creates MORE live
values, not fewer. Each in-flight `ds_read` result must stay live until
its consumer MFMA executes, but the MFMA it feeds is N steps away (where
N = pipeline depth). Meanwhile, the current MFMA's V operands are ALSO
live. The net effect is `depth × 2` extra live vectors (lo + hi per slot)
on top of the current iteration's operands. The compiler cannot collapse
the pipeline stages because the MFMA latency forces the prefetched data
to stay live across multiple loop iterations.

In contrast, the pre-load-all approach lets the compiler see ALL V values
at once and freely schedule the entire block — no cross-iteration
dependencies force extra live ranges.

**Lesson**: Software pipelining reduces register pressure only when the
pipeline REPLACES a blocking wait. If the original code already has good
ILP (pre-loaded V + pure MFMA loop), adding a pipeline just adds
bookkeeping registers. Verify VGPR count before and after — don't assume
"pipeline = fewer registers."

---

## 25. Persistent kernel regresses −3.1% on flash attention

**Trap**: Persistent kernels (one WG processes multiple `(batch, q_tile)`
tiles serially) are a standard optimization for reducing kernel launch
overhead and improving occupancy utilization. CK uses this for large-batch
GEMM with measurable wins. "Should help attention too."

**Result**: −3.1% regression (87.4T → 84.7T at B1S4096).

**Why**: Flash attention's per-tile state (Q registers, running max/sum,
O accumulators) is expensive to save/restore between tiles. The persistent
loop adds: (1) tile-index computation per iteration, (2) Q re-load for
each new tile, (3) O writeback + accumulator reset between tiles. These
costs exceed the launch overhead savings because the grid is already
large enough (512 WGs for B1S4096H32) that launch overhead is amortized.

Persistent kernels help when: (a) the grid is very small (few WGs,
launch overhead dominates), or (b) inter-tile data sharing is possible
(e.g., K/V reuse across tiles in streaming attention). Neither applies
to standard causal prefill flash attention with BLOCK_M=256.

**Lesson**: Persistent kernels are not free — they add per-tile overhead.
Only use them when launch overhead is a significant fraction of runtime
(small grid) or when tiles can share data. For flash attention with a
full-sized grid, standard dispatch is faster.

---

## 26. P-pack reorder (before V store) regresses −0.7%

**Trap**: P softmax output packing (f32 → bf16 truncation for GEMM2) is
currently done AFTER the V LDS store + barrier. Moving P-pack BEFORE the
V store would overlap P-pack VALU with the V store's LDS write latency.
"Classic latency hiding."

**Result**: −0.7% regression. Consistent across shapes.

**Why**: The P-pack VALU (~32 instructions) is already well-hidden by the
MFMA pipeline in its current position (after barrier, before GEMM2). Moving
it earlier puts it in competition with V store address computation and the
barrier's synchronization overhead. The compiler ends up serializing the
P-pack with V-store address math instead of overlapping them, because both
use VALU and the machine scheduler cannot reorder across the barrier.

**Lesson**: "Move compute earlier to hide memory latency" is not always
correct when the compute is already hidden by a different mechanism (MFMA
pipeline bubbles). Moving it creates new serialization with address-math
VALU. Profile before reordering.

---

## 27. sched_group_barrier MFMA/DSRD interleave regresses −0.7%

**Trap**: `sched_group_barrier(mask_mfma, 2, 0)` + `sched_group_barrier(
mask_dsrd, 2, 0)` per GEMM1 loop iteration forces fine-grained 2:2
interleaving between MFMA and ds_read instructions. CK GEMM uses similar
patterns for optimal MFMA occupancy. "Should help GEMM1 too."

**Result**: −0.7% regression vs the softer `sched_dsrd(2) + sched_mfma(2)`.

**Why**: `sched_group_barrier` is a HARD constraint — the scheduler MUST
emit exactly 2 MFMAs then 2 ds_reads, repeating. `sched_dsrd` / `sched_mfma`
are soft hints — the scheduler can deviate when it finds a better ordering.
For flash attention's GEMM1 (which has subtile-level K-read patterns and
interleaved K lo/hi reads), the hard 2:2 constraint prevents the scheduler
from grouping related K reads together, adding pipeline bubbles.

**Lesson**: `sched_group_barrier` is designed for GEMM kernels with
regular, symmetric read/compute patterns. Flash attention's asymmetric
K-read pattern (lo + hi subtiles with different LDS offsets) benefits
more from soft scheduling hints. Prefer `sched_dsrd` + `sched_mfma` over
`sched_group_barrier` for attention kernels.

---

## 29. Bit-packed mask with -inf penalty produces NaN in online softmax

**Trap**: When packing a binary f32 mask ({0.0, -1e6}) into a u32 bitmask, the
natural choice for the masked-out penalty is `float("-inf")` — it's semantically
correct (exp(-inf) = 0, fully suppressing masked positions).

**Result**: rel_err = 0.50, output corrupted. 56.4% of Q rows in the test mask
(B=1024, H=8, S=316) have ZERO attend positions across all 316 KV positions. For
these rows, `m_running` stays at `-inf` (the initial value). When the next KV tile
also has all-masked positions, the online softmax correction factor computes:
`corr = exp2((m_old - m_new) * log2e) = exp2((-inf - (-inf)) * 0.18) = exp2(NaN) = NaN`.
This NaN propagates through all accumulator updates.

**Why**: Online softmax tracks a running max `m`. If the penalty is -inf and there
are no attend positions, `m_old = m_new = -inf`, and `m_old - m_new = NaN` (IEEE 754:
inf - inf = NaN). With -1e6 as penalty, `m_old - m_new ≈ 0` (both are -1e6), and
`exp2(0) = 1` — the correction factor is 1.0, which is correct (uniform attention
over masked-out values, matching reference SDPA behavior).

**Lesson**: Use -1e6 (or any large finite negative), NEVER -inf, as the mask penalty
in online softmax with bit-packed masks. The -inf→NaN path exists whenever any Q row
has zero attend positions across all KV tiles, which is common with sparse masks
(56.4% in our test case).

See also: [cdna3-flash-attention-bf16-mask-optimization.md](../../ref-docs/flydsl/cdna3-flash-attention-bf16-mask-optimization.md) V7.

---

## 30. `select(attend, score, -1e6)` breaks online softmax correction factor

**Trap**: To eliminate the AddF VALU instruction in mask application, replace the
additive approach `score + select(attend, 0.0, -1e6)` with the replacement approach
`select(attend, score, -1e6)`. Both produce the same masked value (-1e6) for
non-attend positions. "The softmax result should be the same."

**Result**: rel_err = 0.117 (threshold 0.02). Correctness failure.

**Why**: For fully-masked rows (all 316 KV positions masked), the additive approach
yields `max(QK_i + (-1e6))` which depends on the actual QK values (typical max ≈
-999990 for QK_i ≈ 10). The replacement approach yields `max(-1e6)` exactly. The
difference of ~10 in the max value propagates through the online softmax correction
factor: `corr = exp2((m_old - m_new) * log2e)`. With `log2e * scale = 0.18`, a
difference of 10 produces `exp2(10 * 0.18) = exp2(1.8) ≈ 3.5×` error in the
correction factor, which accumulates across KV tiles.

**Lesson**: In online softmax, the exact max value matters for numerical stability
of the correction factor. Always use the additive approach (`score + mask_val`) so
that the masked score retains its QK-dependent component. The replacement approach
(`select(attend, score, penalty)`) discards this component and changes the max value
by an amount that gets amplified exponentially through exp2.

See also: [cdna3-flash-attention-bf16-mask-optimization.md](../../ref-docs/flydsl/cdna3-flash-attention-bf16-mask-optimization.md) V7.

---

## 31. `sched_dsrd(2) + sched_mfma(2)` over-constrains compiler when mask VALU needs MFMA latency bubbles

**Trap**: The causal+GQA variant uses `sched_dsrd(2) + sched_mfma(2)` at the GEMM1
start to hint the compiler to interleave 2 ds_reads with 2 MFMAs. "This should also
work for the mask variant — the GEMM1 structure is the same."

**Result**: Replacing `sched_dsrd(2) + sched_mfma(2)` with `sched_barrier(0)` at
GEMM1 start gave +1.8% (49.8 → 50.6 TFLOPS) on the bit-packed mask variant.

**Why**: `sched_dsrd(N) + sched_mfma(N)` are soft scheduling hints that guide the
compiler to interleave N ds_reads with N MFMAs. This works well for the causal variant
where the only non-MFMA work is K/V LDS reads. But with bit-packed mask, there are
additional VALU instructions (AND/CMP/CNDMASK/AddF per element) that can productively
fill MFMA's 64-cycle pipeline latency bubbles. The dsrd/mfma hints prevent the compiler
from scheduling mask VALU during MFMA stalls, leaving cycles wasted. `sched_barrier(0)`
resets the scheduling state entirely, giving the compiler full freedom to interleave
mask VALU with MFMA.

**Lesson**: Scheduling hints that are optimal for one kernel variant may be harmful for
another. When adding new compute work (mask VALU) that can fill MFMA latency bubbles,
prefer `sched_barrier(0)` (unconstrained) over `sched_dsrd/sched_mfma` (semi-constrained).
The key insight is that MFMA has 64-cycle pipeline latency where other work can execute
for free — hints that prevent this interleaving leave performance on the table.

**With [trap #27 (sched_group_barrier regresses)](#27-schedgroupbarrier-mfmadsrd-interleave-regresses-07) conflict note**:
Trap #27 shows that `sched_group_barrier` (HARD constraint) is worse than
`sched_dsrd/sched_mfma` (SOFT hints). This trap shows that even soft hints can be
harmful when there is additional VALU work that should fill MFMA bubbles. The ordering
from most to least constrained: `sched_group_barrier` > `sched_dsrd/sched_mfma` >
`sched_barrier(0)`. Pick the least-constrained option that doesn't regress.

See also: [cdna3-flash-attention-bf16-mask-optimization.md](../../ref-docs/flydsl/cdna3-flash-attention-bf16-mask-optimization.md) V7.

---

## 32. BLOCK_M=64 doubles KV iterations — 28 TFLOPS regression on small D

**Trap**: Reducing BLOCK_M from 128 to 64 halves the Q tile size, potentially
improving occupancy (fewer VGPRs, more waves per CU). "Smaller tiles = higher
occupancy = better performance."

**Result**: 28 TFLOPS (vs 50.6 at BLOCK_M=128). ~45% regression.

**Why**: With BLOCK_M=64, each Q tile covers half as many Q rows, so each row must
iterate over the same number of KV tiles but the work per Q tile is halved. At D=64
with `mfma_f32_32x32x8`, the MFMA work per tile is already small (only 32 MFMA
instructions). Halving it to 16 MFMA makes the per-tile overhead (mask loading,
softmax, K/V LDS prefetch, barrier synchronization) proportionally much larger.
Additionally, the number of KV iterations per Q row stays the same (ceil(S/BLOCK_N)),
so mask/softmax overhead per row doubles.

**Lesson**: For attention kernels with D=64, BLOCK_M=128 is the minimum efficient
tile size. The per-tile fixed costs (mask, softmax, synchronization) must be amortized
over enough MFMA work. Only consider BLOCK_M=64 if VGPR pressure from a larger tile
causes spills or occupancy collapse — and verify with profiling.

See also: [cdna3-flash-attention-bf16-mask-optimization.md](../../ref-docs/flydsl/cdna3-flash-attention-bf16-mask-optimization.md) V7.

---

## 33. Per-instruction bank conflict ratio is misleading for critical-path analysis

**Trap**: rocprofv3 reports `100 * SQ_LDS_BANK_CONFLICT / SQ_INSTS_LDS` = 72.7%
on the V7 bit-packed mask kernel. This looks alarming and triggers an LDS layout
optimization pivot.

**Result**: After testing VT_STRIDE+8 (trap #1 revisited) and K_PAD=0 XOR swizzle,
the actual `LDSBankConflict` derived metric (% of GPU time stalled by bank
conflicts) was only **3.7%** — the bank conflicts are hidden by MFMA parallelism.
The LDS layout changes either regressed or failed correctness.

**Why**: `SQ_LDS_BANK_CONFLICT / SQ_INSTS_LDS` counts conflicts per *instruction*,
not per *cycle*. When ds_read instructions execute in parallel with MFMA (which
has 16-31 cycle latency), the conflicts add stall cycles that are fully absorbed
by MFMA drain. The per-instruction ratio inflates the apparent cost by ~20x.

**Lesson**: Gate LDS-pivot decisions on the rocprofv3 derived `LDSBankConflict`
metric (% GPU time stalled), NOT on the per-instruction ratio. A 70% per-INST
ratio with <5% GPU-time stall means bank conflicts are off the critical path.

---

## 34. CK constants don't transfer without CK access patterns

**Trap**: CK FMHA uses `VT_STRIDE = BLOCK_N + 8` and `_QK_PREFETCH_DEPTH = 3`.
Copying these into the FlyDSL kernel should close the gap.

**Result**: VT_STRIDE+8 caused 5x worse bank conflicts and +0.65% regression
(trap #1, confirmed on bit-packed mask shape). Prefetch depth 3 caused -0.23%
regression vs depth 2.

**Why**: CK's MFMA tile shape produces different ds_read access patterns than
FlyDSL's. CK's depth-3 wins because their kernel doesn't have mask VALU on the
critical path competing for the same VALU slots as extra prefetch scheduling.
Bank-conflict arithmetic (gcd with bank count) depends on the access pattern,
not just the stride constant. Constants that are optimal in one pipeline topology
can be counterproductive in another.

**Lesson**: Never copy stride/depth/padding constants from a reference kernel
without also copying the access pattern that makes those constants optimal.
Re-derive from first principles using the actual ds_read pattern of your kernel.

---

## 35. s_nop hazard padding from VCC serialization is free when hidden by MFMA drain

**Trap**: ISA analysis shows 15% of instructions are `s_nop` from VCC
write→read serialization in the mask bit extraction section (32 `v_cmp` + 32
`v_cndmask` pairs). Eliminating VCC via bit→fma mask transformation should
reclaim ~37 cycles per inner loop.

**Result**: The fma-form mask (bfe + cvt + fmamk + add, zero v_cmp/v_cndmask/
s_nop) was **-1.78% slower** despite removing all 37 s_nop cycles.

**Why**: The mask section immediately follows the GEMM2 PV MFMA chain. The 4
MFMA chains have 16-31 cycle drain latency, which already absorbs the 37 s_nop
cycles for free. The fma-form mask has 128 VALU ops (32 bfe + 32 cvt + 32 fmamk
+ 32 add) vs the original's 96 ops (32 cmp + 32 cndmask + 32 and), creating
more VALU pressure that competes with MFMA drain rather than hiding behind it.

**Lesson**: `s_nop` instructions are only a bottleneck if they are on the
critical path. When s_nops follow a long-latency MFMA chain, they execute during
the drain window and cost zero additional cycles. Verify with the rocprofv3
`MfmaUtil` metric — if MFMA utilization is low (e.g. 29.6%), the kernel is
stall-dominated and eliminating s_nops won't help because the stalls are
elsewhere.

---

## 36. Dtype-mismatched CK rows are not a BF16 CK95 baseline

**Trap**: A wiki row says CK/SDPA no-mask is `2.94 ms / 71.3 TFLOPS`, so it is
tempting to compute CK95 from that number and use it as the BF16 target.

**Result**: The optimized FlyDSL no-mask kernel in this round is BF16:
`3.479213 ms / 60.189822 TFLOPS`, rel_err `0.017906` vs fp32 and `0.018016`
vs bf16. The older CK/SDPA row was FP16, so it is not an apples-to-apples BF16
baseline. Local CK ATT captures were also kept as scheduling evidence only:

| Local CK capture | Time | Throughput |
|---|---:|---:|
| S=316 | 3.746 ms | 55.91 TFLOPS |
| S=320 | 3.656 ms | 58.73 TFLOPS |

**Why**: Dtype changes the valid comparison set. BF16 FlyDSL vs FP16 CK can be
useful context, but it cannot define BF16 CK95. A slower local CK capture is
still useful for inspecting instruction ordering, but it also cannot redefine
the acceptance line.

**Lesson**: Separate "ISA reference", "historical context", and "performance
target". For BF16 acceptance, use a BF16 CK/SDPA baseline or explicitly state
that no dtype-matched CK95 number is available.

See also:
[cdna3-flash-attention-bf16-nomask-isa-scheduling.md](../../ref-docs/flydsl/cdna3-flash-attention-bf16-nomask-isa-scheduling.md).

---

## 37. Full-draining VMEM after `ds_bpermute` wastes the no-mask rescale window

**Trap**: A softmax row reduction uses `ds_bpermute`, so the obvious safe wait is
`s_waitcnt vmcnt(0) lgkmcnt(0)` immediately after it. That guarantees the LDS
permute is complete and also drains any outstanding V global loads.

**Result**: Splitting the wait improved the no-mask D64 kernel:

| Variant | P50 | TFLOPS |
|---|---:|---:|
| Previous retained default | 3.503414 ms | 59.774040 |
| `ds_bpermute_lgkm_sum` final default | 3.479213 ms | 60.189822 |

The useful change was rowsum-only inline `ds_bpermute_b32` followed by
`s_waitcnt lgkmcnt(0)`. The later `vmcnt(0)` moved after eight independent
`v_pk_mul_f32` rescale instructions and just before `ds_swizzle_b32` consumed V.

**Why**: `lgkmcnt(0)` is sufficient to consume the cross-lane reduction result.
The V loads are independent at that point, so draining VMEM early removes a
latency-hiding window the kernel can fill with VALU rescale work.

**Lesson**: In no-mask FlashAttention, wait for the dependency you actually need.
Do not full-drain VMEM at a reduction boundary unless the next instruction consumes
the VMEM data.

See also:
[cdna3-flash-attention-bf16-nomask-isa-scheduling.md](../../ref-docs/flydsl/cdna3-flash-attention-bf16-nomask-isa-scheduling.md).

---

## 38. CK-like prefetch depth and barrier masks do not transfer mechanically to FlyDSL no-mask

**Trap**: CK FMHA uses deeper prefetch and carefully placed scheduler barriers.
Copying `_QK_PREFETCH_DEPTH=3` or CK-like softmax barrier masks into FlyDSL should
move the kernel closer to CK.

**Result**: On the no-mask D64 path these changes regressed:

| Change | Throughput |
|---|---:|
| Default retained before V6 | 59.774040 TFLOPS |
| `QK_PREFETCH_DEPTH=3` | 59.312539 TFLOPS |
| Softmax barrier mask `1` | ~57.65 TFLOPS |
| Softmax barrier mask `0x7f` | ~57.71 TFLOPS |
| `NOMASK_SOFTMAX_BARRIER=0` | 59.957511 TFLOPS |

Disabling the no-mask barrier was less bad than the CK-like masks, but it was
still slower than the final `ds_bpermute_lgkm_sum` schedule.

**Why**: CK's constants work with CK's access pattern and scheduler template.
FlyDSL's register allocation, wait placement, and QK/V staging differ enough that
the same knobs perturb the schedule without reproducing CK's latency hiding.

**Lesson**: Port scheduling constants only after ATT proves the surrounding
pipeline shape matches. If the access pattern differs, treat CK constants as
hypotheses, not defaults.

See also trap #34 and
[cdna3-flash-attention-bf16-nomask-isa-scheduling.md](../../ref-docs/flydsl/cdna3-flash-attention-bf16-nomask-isa-scheduling.md).

---

## 39. More early rescale VALU is not automatically better latency hiding

**Trap**: Once VMEM wait is delayed across a rescale window, expand the window:
rescale all output accumulator chunks early so more `v_pk_mul_f32` instructions
can overlap the outstanding V loads.

**Result**: `EARLY_RESCALE_ALL=1` measured `3.494454 ms / 59.927313 TFLOPS`,
slower than the final default `3.479213 ms / 60.189822 TFLOPS`.

**Why**: The extra rescale work increases VALU pressure and changes scheduling.
Only the required rescale work fit usefully into the delayed-VMEM window; adding
more independent VALU did not create more useful overlap.

**Lesson**: Latency hiding is constrained by the actual dependency window, not by
the number of independent VALU instructions you can manufacture. Add early rescale
only when ATT shows it lands in a stall window and the benchmark confirms the gain.

See also:
[cdna3-flash-attention-bf16-nomask-isa-scheduling.md](../../ref-docs/flydsl/cdna3-flash-attention-bf16-nomask-isa-scheduling.md).

---

## 46. K_INTERBLOCK + SHARE_KV_LDS race condition corrupts output silently

**Trap**: After enabling SHARE_KV_LDS (K and V share the same LDS slot),
everything looks correct in standalone tests. But when integrated into the
production framework (atrex) where `FLYDSL_FLASH_ATTN_FUNC_K_INTERBLOCK`
defaults to `1`, output error jumps from 1.4% to 5.3%.

**Result**: rel_err = 0.054 (threshold 0.02). No NaN, no crash — just
silently wrong results. The error is data-dependent and only shows up
at larger batch sizes.

**Why**: K interblock prefetch loads next iteration's K into LDS at the
end of the current KV loop body. With SHARE_KV_LDS, K and V use the
same LDS offset (base 0). The prefetched K_{n+1} overwrites V_n's LDS
slot while V_n is still being read by GEMM2 (or by threads that haven't
reached the barrier yet). The DMA-to-LDS write is asynchronous and races
with the V ds_read instructions.

**Lesson**: SHARE_KV_LDS requires `K_INTERBLOCK=0`. Before enabling LDS
sharing, audit ALL asynchronous writes to the shared region. The
`ENABLE_K_INTERBLOCK` flag does not appear in the `SHARE_KV_LDS`
condition, so the conflict is not caught at build time.

---

## 47. v_pk_add_f32 inline asm chained with pk_fma produces wrong results

**Trap**: After successfully using `v_pk_fma_f32` inline asm for packed
softmax scaling (16 ops → 8 ops), extend the same pattern to `v_pk_add_f32`
for the softmax sum accumulation.

**Result**: rel_err = 25.7% with pk_add alone. Combined pk_fma + pk_add
produces NaN. The error is systematic, not random — every element is wrong
by roughly the same factor.

**Why**: Root cause not fully determined. Likely a MLIR lowering issue with
chained v2f32 inline asm operations where the output of one packed op feeds
into the next. The compiler may fail to correctly allocate v2f32 register
pairs when multiple inline asm blocks with v2f32 constraints are chained.
The `v_pk_fma_f32` alone works because its result goes through scalar
`rocdl.exp2` before further use, breaking the v2f32 chain.

**Lesson**: Inline asm with packed v2f32 types is fragile in FlyDSL/MLIR.
Test each packed operation in isolation AND in combination. When chaining
fails, keep only the highest-value packed op (pk_fma for scaling saves more
than pk_add for summation).

---

## 48. Removing computation barriers regresses flash attention throughput

**Trap**: The kernel has `gpu.barrier()` calls before GEMM1, before mask
application, and before softmax max reduction. These look like conservative
synchronization that could be relaxed — removing them should reduce stall
cycles and improve throughput.

**Result**: 64.9 → 59.8 TFLOPS (-8%). Removing barriers consistently
regresses performance across multiple runs.

**Why**: Barriers serve a dual purpose: synchronization AND scheduling hints.
The AMDGPU compiler uses barrier placement to infer instruction ordering
constraints. Without barriers, the compiler may reorder MFMA instructions
away from their operands, breaking the tight MFMA-to-MFMA pipeline that
maximizes throughput. The barriers keep MFMAs back-to-back by preventing
the scheduler from inserting unrelated VALU between them.

**Lesson**: Do not remove barriers based on correctness analysis alone.
Profile before and after. In MFMA-heavy kernels, barriers often serve as
implicit scheduling fences that keep the MFMA pipeline saturated.

---

## 49. BLOCK_M=64 regresses occupancy 3→2 on MI308X flash attention

**Trap**: With BLOCK_M=128 at occupancy 3, try BLOCK_M=64 for better tail
handling (S=316 with BLOCK_M=64 gives 5 Q tiles instead of 3, better CU
utilization). Smaller tiles should also reduce register pressure.

**Result**: 50.2 TFLOPS (from 64.9). Occupancy dropped from 3 to 2
waves/SIMD. KV iterations doubled (10 iterations for S=320 with BLOCK_M=64
vs 5 with BLOCK_M=128).

**Why**: BLOCK_M=64 uses 128 threads (2 waves × 64 threads) instead of
256 threads (4 waves × 64 threads). With only 2 waves and head_dim=64,
each thread must hold more Q and accumulator state, pushing VGPRs above
the 128 limit. At 2 waves/SIMD, each KV iteration has half the compute
throughput but the same fixed overhead (LDS loads, mask loads, barriers).

**Lesson**: On CDNA3 with MFMA32, BLOCK_M=128 (4 waves, 256 threads) is
the sweet spot for flash attention. BLOCK_M=64 only makes sense if the
VGPR footprint can be kept under 256/thread (occupancy 2 with 2 waves
needs 256 VGPRs/wave from 512 total).

---

## 50. SHARE_KV_LDS needs waves_per_eu=4 to realize occupancy gain (resolves trap 18)

**Trap**: After enabling SHARE_KV_LDS, LDS drops from 17,152 to 8,704 bytes.
Theory says this allows 4 WGs per CU (8,704 × 4 = 34,816 < 64 KB). But
actual occupancy stays at 3.

**Result**: VGPRs increased from 123 → 130. At 130 VGPRs/wave, occupancy
is still limited to 3 waves/SIMD (130 × 4 = 520 > 512 VGPRs/SIMD). The
compiler used the freed LDS budget to expand register allocation.

**Why**: Without an explicit occupancy hint, the compiler optimizes for
instruction-level parallelism by using more registers. The `waves_per_eu`
attribute is the only way to tell the compiler "I want occupancy N, so
limit VGPRs to 512/N = 128". Setting `waves_per_eu=4` forces ≤128 VGPRs,
and combined with the reduced LDS, achieves occupancy=4.

**Resolution of trap 18**: Trap 18 concluded "K/V LDS sharing regresses 30%"
and "higher occupancy is NOT automatically better." This was correct at the
time (causal+GQA variant, no `waves_per_eu` control, 2 barriers). The V8-V10
result shows that K/V sharing DOES work when: (a) only 1 barrier after GEMM1
(not 2), (b) `waves_per_eu=4` controls VGPR allocation, (c) `K_INTERBLOCK=0`
prevents the LDS race condition. The key insight is that trap 18's barrier
overhead was caused by having too many barriers (2 per iteration) and not
controlling VGPRs — the LDS sharing itself was not the problem.

**Lesson**: LDS reduction is necessary but not sufficient for occupancy gains.
Always pair LDS optimization with `waves_per_eu=N` to prevent the compiler
from absorbing the savings into VGPRs. Verify the compiled binary's VGPR
count matches the target.

---

## 51. Double-masking in softmax exp2 produces NaN

**Trap**: The kernel's BFI mask extraction already applies a -1e6 penalty to
masked positions (bit=0). As extra safety, add a separate `HAS_MASK`
conditional branch in the exp2 computation that also checks the mask and
applies the penalty again.

**Result**: NaN output. With the double penalty, masked positions get
score = -2e6 instead of -1e6. When combined with l_final clamping
(`max(l_final, 1e-6)`) in the epilogue, the interaction between clamped
denominators and doubly-penalized numerators creates NaN through 0/0.

**Why**: The BFI mask path and the softmax exp2 path are supposed to be
complementary: BFI applies the mask, then exp2 sees the masked score and
produces exp2(-1e6 × log2e) ≈ 0. Adding a second mask check doubles the
penalty and breaks the numerical balance. The clamping was added to
prevent division by zero for fully-masked rows, but with double-masking,
it creates a different path to NaN.

**Lesson**: Apply the mask penalty exactly once. In the bit-packed mask
kernel, the BFI extraction handles all masking — do not add redundant
mask checks in the softmax path. If you need numerical safety for
fully-masked rows, handle it at the Q-row level (`q_in_bounds` guard),
not by clamping intermediate values.

---

## Mask+LSE quick reference

| Do | Don't |
|----|-------|
| Set `K_INTERBLOCK=0` with SHARE_KV_LDS | Leave K_INTERBLOCK at default "1" — silent corruption |
| Set `waves_per_eu=4` after LDS reduction | Assume compiler will target max occupancy on its own |
| Use 1 barrier after GEMM1 for K/V sharing | Add 2 barriers (before V load AND before next K) — trap 18 |
| Use v_pk_fma_f32 for softmax scaling only | Chain v_pk_add_f32 with pk_fma — wrong results |
| Keep barriers before GEMM/mask/softmax | Remove barriers for "optimization" — -8% regression |
| Apply BFI mask penalty once in extraction | Double-mask in both BFI and exp2 — NaN |
| Verify compiled VGPR count matches target | Trust LDS reduction alone for occupancy gain |
