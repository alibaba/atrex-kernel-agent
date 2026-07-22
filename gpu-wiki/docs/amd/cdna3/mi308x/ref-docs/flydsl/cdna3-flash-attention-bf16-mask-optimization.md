# FlyDSL Flash Attention bf16 with Free Mask on MI308X (gfx942)

Applicability: backend: flydsl; hardware: amd; topic: reference

## Target hardware

- **Chip**: AMD MI308X (CDNA3, gfx942)
- **Peak bf16 MFMA**: 206 TFLOPS
- **HBM BW**: 5.3 TB/s
- **CUs**: 80 (16 SEs × 5 CUs)
- **LDS**: 64 KB per CU
- **VGPR**: 512 per CU

## Algorithm baseline

Standard Flash Attention forward pass with **arbitrary (non-causal) attention mask**:
- Input: Q, K, V ∈ (B, H, S, D) bf16; Mask ∈ (B, 1, S, S) f32 (broadcast over heads)
- Online softmax with per-row running max and sum
- Two-stage GEMM: S = Q·K^T, then O = softmax(S + Mask) · V

Starting point: the MI308X causal+GQA variant (`flash_attn_func_mi308x.py`) which
achieves ~110 TFLOPS on large causal shapes. See companion doc:
[cdna3-flash-attention-bf16-gqa-optimization.md](cdna3-flash-attention-bf16-gqa-optimization.md).

### Test shape

| Tensor | Shape | dtype | Notes |
|--------|-------|-------|-------|
| Q, K, V | (1024, 8, 316, 64) | bf16 | BSHD, non-contiguous stride |
| Mask | (1024, 1, 316, 316) | f32 | Broadcast over heads, contiguous |
| O | (1024, 8, 316, 64) | bf16 | Output |
| scale | 0.125 (= 1/√64) | — | |
| Theoretical FLOPs | 0.2094 TFLOP | — | 4·B·H·S²·D |

## Kernel resource footprint (V7 final)

| Resource | V3 (f32 mask) | V7 (bit-packed) | Notes |
|----------|---------------|-----------------|-------|
| VGPR | 148 | ~156 | No spill (extra regs for pre-loaded mask bits) |
| SGPR | 62 | 62 | No spill |
| AGPR | 0 | 0 | Not used |
| LDS | 17,152 B (16.8 KB) | 17,152 B (16.8 KB) | Unchanged |
| Scratch | 0 | 0 | No stack spill |
| Workgroup | 256 threads (4 waves) | 256 threads (4 waves) | BLOCK_M=128 |
| Mask buffer | 419 MB (f32) | 13 MB (u32 bitmask) | 32× reduction |
| Mask VMEM/subtile | 8× buffer_load_dwordx4 | 2× buffer_load_dword | 4× fewer loads |

### ISA instruction breakdown (V3)

| Category | Count | % | Notes |
|----------|-------|---|-------|
| VALU | 500 | 49.1% | Softmax, mask, pack |
| SALU | 365 | 35.8% | **s_nop=272 (26.7%)** compiler pipeline bubbles |
| SYNC | 50 | 4.9% | s_waitcnt + s_barrier |
| LDS | 44 | 4.3% | ds_read/write |
| MFMA | 32 | 3.1% | 32×32×8 bf16 |
| VMEM | 28 | 2.7% | global_load/store |

## Optimization journey

### V0 — Initial baseline with D=128 padding (20.3 ms, 10.4 TFLOPS)

Patched the MI308X causal+GQA kernel to accept a free mask argument. The existing
kernel requires `head_dim % 128 == 0`, so D=64 was padded to D=128.

**Problems**:
- 2× wasted compute and memory bandwidth from D padding
- Mask loaded as 32 scalar f32 loads per MFMA subtile (~6.4 ms overhead)
- Occupancy = 1 wave/SIMD (VGPR pressure from D=128)

### V1 — Mask index hoisting (20.5 ms, no improvement)

Attempted to hoist `mask_row_base` computation out of the KV loop.

**Result**: No improvement. The FlyDSL/LLVM compiler's CSE pass already performs
this optimization automatically.

**Lesson**: Don't manually hoist expressions that the compiler will CSE; verify
with ISA inspection before investing optimization effort.

### V2 — Native head_dim=64 support (breakthrough)

Eliminated D=64→128 padding by adjusting kernel parameters:
- `D_CHUNKS = 2` (was 4 for D=128)
- MFMA tile: `mfma_f32_32x32x8_bf16` (was `32x32x16`)
- Halved VGPR pressure → improved occupancy potential

This was the single largest optimization, eliminating 2× compute waste.

### V3 — Vectorized mask loading + final tuning (5.25 ms, 39.9 TFLOPS) ✅

**Key optimizations**:
1. **v4f32 mask loading**: MFMA 32×32 output layout maps groups of 4 consecutive
   `r ∈ [4g, 4g+4)` to 4 contiguous KV columns. Exploiting this, each `_gep_load_v4f32`
   loads 4 consecutive mask elements in one `global_load_dwordx4` instruction.
   Reduces 32 scalar loads → 8 vector loads per subtile (4× fewer instructions).
2. **rocdl.exp2**: Single `v_exp_f32` instruction for softmax (vs multi-instruction
   expansion from `arith.exp2`).
3. **V transpose via ds_swizzle**: XOR-N cross-lane shuffle + `vector.from_elements`
   pack + paired `ds_write_b32` (inherited from causal variant v14).
4. **All-upfront O rescale**: Rescale all `o_accs` before pure-MFMA PV loop
   (interleaved deferred rescale regressed in earlier experiments).

### Grid=(B×H) experiment (5.74 ms, 9.3% regression — reverted)

Tried persistent-kernel style: Grid uses only batch×head dimensions, each CTA
loops over all Q tiles internally.

**Result**: 9.3% slower. With S=316 (only 2-3 Q tiles at BLOCK_M=128), the
persistent approach adds loop control overhead without improving CU utilization.

**Lesson**: Persistent kernels help when there are many tiles to amortize setup
cost. For small seq_len (< 1024), the standard Grid=(B×num_q_tiles×H) is better.

### V4 — Mask bounds-check removal (5.47 ms, neutral — reverted)

Attempted to remove redundant KV-dimension mask bounds checks (`lo_grp_ok`/`hi_grp_ok`
cmpi+AND+clamp). The `verify.py` host pads the mask buffer with -1e6 in OOB positions,
making direct loads memory-safe without bounds checks.

**Result**: Flat / -1.5% regression on min time (5.47 ms vs 5.39 ms). -24 VALU per
subtile theoretical, but the compiler had already hidden bounds-check VALU in s_nop
pipeline bubbles — removing them freed no compute slot.

**Lesson**: With 26.7% s_nop, removing VALU that the compiler hides
in pipeline bubbles yields no gain. Profile ISA scheduling before assuming VALU
reduction translates to wall-clock improvement.

### V5 — Tile-level early-exit on fully-masked tiles (FUNDAMENTAL BLOCKER — reverted)

Precomputed `tile_skip[B, num_q_tiles, num_kv_tiles]` on host to skip KV tiles where
all mask positions are -1e6. 12.5% of (128×64) tiles appeared fully masked in the
diagnostic. Used `scf.IfOp` to wrap the KV-loop body, yielding `[m, l, o_accs...]`
from both branches.

**Result**: NaN output. The premise was flawed: **56.4% of in-bounds Q rows have
ZERO attend positions across ALL 316 KV positions.** For these rows, max=-1e6 across
every KV position, softmax produces uniform-mean attention over V (1/N per position).
Skipping any KV tile for such a row breaks the softmax normalization (denominator
becomes wrong). Only 21/3072 q-tiles have every row containing at least one attend
position — safe-to-skip fraction drops from naive 13.2% to actual 0.013%.

**Lesson**: Tile-level skip in attention requires that EVERY Q row in the tile has at
least one attend position in the remaining KV tiles. With sparse masks where many Q
rows attend to nothing, the skip opportunity is near zero. The naive "tile fully masked"
diagnostic is misleading — it doesn't check the per-row safety condition.

### V6 — K inter-block prefetch (5.41 ms, neutral — reverted)

Split K cooperative load into async global fetch + LDS store. K for the next iteration
is loaded globally after GEMM1 (to overlap with softmax + GEMM2), stored to LDS at the
next iteration start. Used `_waitcnt_vm_n(NUM_BATCHES_KV)` to keep K loads outstanding.

**Result**: Flat (5.41 ms avg vs 5.40 ms). The FlyDSL compiler's default scheduling
already overlaps K global loads with the previous iteration's compute. Explicit
inter-block prefetch provides no additional benefit.

**Lesson**: The compiler already does a good job of overlapping global
loads with compute. Inter-block prefetch is beneficial when the compiler can't see
across iteration boundaries, which isn't the case here.

### V7 — Bit-packed binary mask (4.14 ms, 50.6 TFLOPS) ✅

The breakthrough optimization. Two key changes:

**1. Bit-packed mask format (32× bandwidth reduction)**:
- Host packs binary f32 mask ({0.0, -1e6}) into u32 bitmask: bit=1 → attend, bit=0 → masked
- Mask buffer: 419 MB (f32) → 13 MB (u32 bitmask)
- Kernel loads 2× `buffer_load_dword` per subtile (was 8× `buffer_load_dwordx4`)
- Bit extraction exploits MFMA 32×32 output layout:
  - `lane_div_32` (0 or 1) selects which 32-bit word to load (lo/hi half of 64 KV positions)
  - Pre-shift by `lane_div_32 * 4` makes remaining bit positions compile-time constants
  - `bit_pos = grp*8 + sub` where grp=0..3, sub=0..3 (16 positions per lane per word)
  - Per-element: AND + CMP + CNDMASK + AddF (4 VALU vs 1 VMEM + 1 AddF for f32)

**2. sched_barrier(0) scheduling hints**:
- Replaced rigid `sched_dsrd(2) + sched_mfma(2)` at GEMM1 start with `sched_barrier(0)`
  (unconstrained — lets compiler freely reorder)
- Added `sched_barrier(0)` before mask application (allows VALU to fill MFMA latency bubbles)
- Added `sched_barrier(0)` before softmax max-reduction
- Added `sched_barrier(0)` before V store barrier

**Performance progression**:
- V3 f32 mask: 5.39 ms / 38.8 TFLOPS
- V7 bit-pack only: 4.49 ms / 46.6 TFLOPS (+20%)
- + sched_barrier before mask: 4.23 ms / 49.5 TFLOPS
- + sched_barrier before softmax: 4.21 ms / 49.7 TFLOPS
- + sched_barrier at V store: 4.20 ms / 49.8 TFLOPS
- + replace sched_dsrd/mfma with sched_barrier: **4.14 ms / 50.6 TFLOPS** (+30.5% total)

**Why it works**: The f32 mask was the dominant bandwidth bottleneck — 419 MB mask data
was 55% of total data movement (Q+K+V+mask+O). Bit-packing eliminated this bottleneck.
The remaining VALU instructions (AND/CMP/CNDMASK/AddF per element) can overlap with
MFMA's 64-cycle pipeline latency when the scheduler is unconstrained (sched_barrier(0)),
whereas the rigid sched_dsrd/sched_mfma hints prevented this interleaving.

## Final perf vs baseline

| Implementation | Time (ms) | TFLOPS | Relative |
|----------------|-----------|--------|----------|
| **FlyDSL V7 (bit-packed mask)** | **4.14** | **50.6** | **1.00×** |
| FlyDSL V3 (f32 mask) | 5.39 | 38.8 | 0.77× |
| CK/SDPA (with mask) | 15.50 | 13.5 | 0.27× |
| CK/SDPA (no mask, FP16 historical row) | 2.94 | 71.3 | dtype-mismatched context |
| FlyDSL no mask | 3.88 | 54.0 | 1.07× |
| FlyDSL V0 (D=128 padded) | 20.30 | 10.4 | 0.21× |

- **vs CK/SDPA with mask: 3.74× faster** (CK elementwise bias path is very slow)
- **vs CK/SDPA no mask historical FP16 row: 71% performance** (dtype-mismatched
  context; do not use as BF16 CK95)
- **vs FlyDSL V3 f32 mask: +30.5% improvement** (5.39 ms → 4.14 ms)
- Correctness: rel_err = 0.0175 (< 0.02 bf16 threshold) ✅
- **No-mask follow-up**: see
  [cdna3-flash-attention-bf16-nomask-isa-scheduling.md](cdna3-flash-attention-bf16-nomask-isa-scheduling.md).
  That later run uses rebuilt FlyDSL and manual ATT-guided
  scheduling, so its BF16 no-mask result (`3.479213 ms / 60.189822 TFLOPS`)
  supersedes only the no-mask row above, not the bit-packed mask
  result. The CK/SDPA no-mask row above is FP16 historical context, not a BF16
  CK95 target.

## Remaining bottlenecks (ISA evidence)

1. **s_nop dominance (~26%)**: Compiler-inserted `s_nop` for pipeline hazard avoidance
   remains the largest single overhead. This is a FlyDSL/LLVM codegen limitation,
   not addressable at the kernel source level.

2. **VALU overhead from bit-packed mask**: Each mask position requires AND + CMP +
   CNDMASK + AddF (4 VALU). With 32 positions per subtile (16 per lane × 2 halves),
   this is 128 VALU instructions per subtile — still significant vs 32 MFMA instructions.

3. **Small problem size**: S=316 yields only ~5 KV tiles × 3 Q tiles × 8 heads
   = ~120 CTAs for 80 CUs (1.5 CTAs/CU average). Insufficient for full saturation.

4. **Instruction scheduling quality**: Further gains require CK V3–style ISA-level
   scheduling (`CoreLoopScheduler`) that controls exact instruction ordering.

## What would close the remaining gap

1. **Larger seq_len** (S ≥ 2048): More Q/KV tiles → better CU utilization.
2. **CK V3–style ISA scheduling**: `CoreLoopScheduler` template that controls
   exact instruction ordering without relying on LLVM heuristics.
3. **Multi-word bitmask load**: If S grows, load `buffer_load_dwordx4` for 4 consecutive
   u32 words (128 KV positions) — amortizes load instruction overhead.

## Sustained recipe (do these, in this order)

1. **Eliminate D padding**: Set `D_CHUNKS` and MFMA tile to match actual `head_dim`
   (D=64 → D_CHUNKS=2, mfma_32x32x8). Never pad to next power-of-two.
2. **Bit-pack binary masks**: If mask is binary ({0, -1e6} or {0, -inf→-1e6}), pack
   into u32 bitmask on host (32× bandwidth reduction). Use -1e6 as penalty, NOT -inf
   (see pitfall #29). Use additive application `score + select(attend, 0, -1e6)`,
   NOT replacement `select(attend, score, -1e6)` (see pitfall #30).
3. **MFMA lane-aware bit extraction**: Pre-shift by `lane_div_32*4`, use compile-time
   bitmask `1 << (grp*8 + sub)` per element. AND + CMP + CNDMASK + AddF.
4. **sched_barrier(0) at strategic points**: Before mask application, before softmax
   max-reduction, before V store barrier. Lets VALU fill MFMA latency bubbles.
   Do NOT use rigid sched_dsrd/sched_mfma hints (see pitfall #31).
5. **Use rocdl.exp2**: Always use `rocdl.exp2(T.f32, x)` for softmax.
6. **Pre-load V into SSA**: Load all V packs before the pure-MFMA PV loop.
7. **Upfront O rescale**: Rescale all accumulator chunks before MFMA, not interleaved.
8. **Profile with rocprofv3**: Verify ISA with `llvm-objdump` from HSACO.

## Related docs

- **No-mask ISA scheduling follow-up**: [cdna3-flash-attention-bf16-nomask-isa-scheduling.md](cdna3-flash-attention-bf16-nomask-isa-scheduling.md)
  — Same shape without attention mask; archived separately because it coexists with
  this bit-packed mask implementation and does not replace it.
- **Causal+GQA optimization journey**: [cdna3-flash-attention-bf16-gqa-optimization.md](cdna3-flash-attention-bf16-gqa-optimization.md)
  — Same hardware, different scenario (causal prefill, ~110 TFLOPS on large shapes).
  The V transpose ds_swizzle technique (v14) is shared between both variants.
- **Continued in V8-V10**: [cdna3-flash-attention-bf16-mask-lse-optimization.md](cdna3-flash-attention-bf16-mask-lse-optimization.md)
  — SHARE_KV_LDS + pk_fma + waves_per_eu=4 + LSE output. 50.6 → 71.8 TFLOPS (+42%).
- **Pitfalls**: [flash-attn-pitfalls.md](../../pitfalls/flydsl/flash-attn-pitfalls.md)
  — Traps 15-17 are mask-specific (f32 mask); traps 29-32 are bit-packed mask pitfalls;
  traps 46-51 are mask+LSE (V8-V10) pitfalls. Traps 1-14 apply to both causal and mask variants.
- **Reference kernel (mask+LSE, V10)**: [flash_attn_func_mask_mi308x.py](../../../../../../reference-kernels/amd/cdna3/flydsl/FlyDSL/flash_attn_func_mask_mi308x.py)
- **Reference kernel (mask, V7 backup)**: `flash_attn_func_mask_mi308x.py.v7.bak`
- **Reference kernel (causal+GQA)**: [flash_attn_func_mi308x.py](../../../../../../reference-kernels/amd/cdna3/flydsl/FlyDSL/flash_attn_func_mi308x.py)
- **Generic CDNA baseline**: [flash_attn_func.py](../../../../../../reference-kernels/amd/cdna/flydsl/FlyDSL/flash_attn_func.py)
- **Backward kernel (dK+dV) optimization**: [cdna3-attention-backward-dkdv-bf16-causal-mask-optimization.md](cdna3-attention-backward-dkdv-bf16-causal-mask-optimization.md)
  — Same hardware, backward pass. 46.94 TFLOPS / 3.77× vs PyTorch SDPA bwd.
