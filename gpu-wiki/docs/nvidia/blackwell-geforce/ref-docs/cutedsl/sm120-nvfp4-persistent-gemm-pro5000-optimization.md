# SM120 NVFP4 Persistent GEMM (NVFP4×NVFP4, fp32 accum) on RTX PRO 5000

## Target hardware

NVIDIA RTX PRO 5000 Blackwell-GeForce / sm_120, 110 SMs.
**No UMMA / tcgen05 / clusters / multicast** on sm_120 — CUTLASS C++ for sm_120 also
uses the same warp-level `mma.sync.aligned.kind::mxf4nvf4...` instruction we use here.
The 29% gap to CUTLASS is purely pipeline / SF layout / warp-specialization structure.

## Algorithm baseline

Standard GEMM `C = A @ B^T` where A, B are NVFP4 (Float4E2M1FN, 1 ue4m3 SF per 16 fp4
elements), accumulator is fp32. The atom is `m16n8k64` — see
[reference-kernels/.../sm120_nvfp4_inline_ptx_gemm.py](../../../../../reference-kernels/nvidia/blackwell-geforce/cutedsl/cutlass/sm120_nvfp4_inline_ptx_gemm.py)
for the inline-PTX MMA call and the register-packing convention.

## Kernel resource footprint (final = v43)

| Field | Value |
|---|---|
| Block | `(THREADS=288, 1, 1)` = 1 producer warp + 8 consumer warps |
| Grid | `(NUM_CTAS=110, 1, 1)` = 1 CTA per SM (persistent) |
| BLOCK_M × BLOCK_N × BLOCK_K | 128 × 128 × 128 |
| K_BLOCK_MAX | 2 (inner MMA iters per K-tile) |
| STAGES | 4 (TMA pipeline depth) |
| Atoms per warp | 2 in M × 8 in N = 16 m16n8k64 MMAs / warp / k_block |
| Smem | ~74 KB (sA0+sA1=32, sB0+sB1=32, SF=8, mbar+slack≈2) |
| Registers / thread | 168 |
| Per warp per K-tile cp.async (SF) | 4 |

## Optimization journey

### v15 — single-warp baseline
TMA-less, single warp per output tile, byte-by-byte SF reads. **66 TFLOPS at 4096³ (8% of CUTLASS).**
Sets up the inline-PTX m16n8k64 atom and the SF read pattern.

### v22–v29 — wider tiles + more warps
Grow BLOCK to 32×16, 64×32, then 128×128. 1 producer warp + warp-specialized consumers (1→4→8 consumer warps).
Each step ~+30%. **v29 BLOCK 128×128 = 290 T (36%).**

### v32 — persistent kernel
Switch to `NUM_CTAS=110` (= SM count) with a persistent outer loop over `(by, bx)` tiles inside each CTA. Pipeline state continues across tiles. **+11%, 333 T (41%).**

### v36 — vectorized SF u32 reads
The original SF read pattern was 4 byte-loads + 3 shifts + 3 ORs per scale-factor.
Recast SF smem to Uint32 → single `ld.shared.u32` per atom. **+3%, 344 T (43%).**

### v38 — SF cp.async prefetch (1 K-iter ahead)
Issue cp.async for K-iter `kt+1`'s SF before doing MMAs for `kt`. Drain leftover prefetch + cross-warp barrier at tile boundaries. **The `cute.recast_tensor` MUST be inside the K-loop or correctness breaks silently** (see pitfalls). **+22% over v32, 421 T (52%).**

### v40, v44 — non-wins (kept as cautionary)
- v40: cooperative SF loading (no warp redundancy) + per-K-iter cross-warp barrier — barrier cost > L2 savings, **-4%**.
- v44: manual within-K-tile register K-pipelining — compiler already overlaps independent loads with MMAs, manual unrolling adds register pressure and breaks correctness, **0% to negative**.

### v43 — the breakthrough: compressed SF + BLOCK_K=128 + STAGES=4
Discovered that the original `pack_sf_per_atom` was producing **8× bloated SF storage** (rows of (128, 4) layout where only every 4th row had data — 7/8 was zero padding). Wrote `pack_sf_per_block(sf, atom_dim, atoms_per_block)` that produces `(M_blocks_outer, SF_TILES_K, atoms_per_block, atom_dim, 4)` so all atoms for one CTA tile's K-block are contiguous in 512 bytes — **1 cp.async per warp covers all atoms** (vs 10 in v38). 5× less L2 traffic for SF.

With this freed smem (~21 KB per stage saved):
- BLOCK_K = 128 (vs 64): K_BLOCK_MAX = 2 inner MMA iters per K-tile, fewer K-tile boundary syncs
- STAGES = 4 (vs 3): drift tolerance ≥ 2 → no per-K-tile barrier needed
- Sub-buffered A,B (sA0/sA1 per k_block) so ldmatrix slice has clean stride

**+37% over v38, 581 TFLOPS at 4096³ (71% of CUTLASS).** Deterministic clean rel_err 6.2e-7.

### v45/v46/v47 — failed attempts at full CUTLASS pipeline parity
All tried to put SF onto the SAME stage barrier as TMA(A,B) — see
[docs/nvidia/blackwell-geforce/pitfalls/cutedsl/nvfp4-gemm-pitfalls.md](../../pitfalls/cutedsl/nvfp4-gemm-pitfalls.md).
cute-DSL 4.4.2's `PipelineTmaAsync` mbarrier byte-counting machinery doesn't compose with
non-TMA (cp.async / cp.async.bulk) copies through its high-level API.

## Final perf vs baseline

| Shape | v15 | v32 (persistent) | v38 (prefetch) | **v43** | CUTLASS C++ | v43 % CUTLASS |
|---|---|---|---|---|---|---|
| 1024 × 1024 × 2048 | 62 T | 170 T | 205 T | 292 T | — | — |
| 2048 × 1024 × 2048 | 70 T | 186 T | 230 T | 317 T | — | — |
| 2048 × 2048 × 2048 | 72 T | 252 T | 317 T | 434 T | — | — |
| **4096 × 4096 × 4096** | **66 T** | **338 T** | **426 T** | **581 T** | **808 T** | **71%** |

**8.7× total improvement** over the v15 baseline.

## Remaining bottlenecks (PMC evidence at 4096³)

| Metric | v32 | v38 | v43 |
|---|---|---|---|
| L2 throughput | 66% | 87% | **94% (saturated)** |
| Tensor pipe util | 29% | 39% | 56% |
| Long-scoreboard stall | 4.90 | 1.28 | 0.89 |
| Wait stall | — | 2.17 | 3.03 |
| Issue active | 18% | 27% | 21% |
| Warps eligible / cycle | 0.40 | 0.39 | 0.31 |

**v43 is L2-bound at 94%.** Tensor utilization climbed to 56% (from 29%). Wait stall went up
(more pipeline waits) — symptomatic of being L2-limited.

## What would close the remaining gap

The CUTLASS sm_120 cooperative-pingpong recipe (verified by reading
`include/cutlass/gemm/collective/sm120_blockscaled_mma_tma.hpp`) does two more things
we couldn't easily replicate:

1. **TMA for SF on the same producer barrier as A/B.** Eliminates the entire SF cp.async
   pipeline. CUTLASS uses uint16 element type to satisfy TMA's 16-byte innermost. We tried
   3 paths (v45 cute mbar wrapper, v46 TMA-for-SF, v47 cp.async.bulk), all hit cute-DSL
   4.4.2 plumbing bugs — see pitfalls.
2. **Pingpong: 2 math warpgroups alternating tiles.** While WG0 does epilogue, WG1
   mainloops the next tile. Hides per-tile setup. Requires CUTLASS-style sm90 pingpong
   host structure not directly exposed by `PipelineTmaAsync`.

Either path needs **bypassing `PipelineTmaAsync` and building mbar control entirely from
inline PTX** (mbarrier_init / arrive_and_expect_tx / try_wait), or waiting for a cute-DSL
release that exposes bulk-on-pipeline cleanly.

## Sustained recipe (do these, in this order)

1. **Compress SF aggressively before anything else.** The 8× bloat is hidden in the
   default CUTLASS-style (128, 4)-per-atom layout. Without compression, BLOCK_K=128 +
   STAGES=4 won't fit in 100 KB smem and you're stuck at v38-class perf.
2. **Persistent kernel = NUM_SMS.** Pro5000 is 110. +11% almost for free.
3. **Vectorize SF u32 reads.** Single `ld.shared.u32` per atom, not 4 byte loads. +3%.
4. **SF cp.async prefetch by 1 K-iter.** Decouple SF stage tracking from the TMA pipeline
   stage (use `kt%STAGES`, not `cons_state.index`) and drain + barrier at tile boundary.
5. **Always re-issue `cute.recast_tensor` inside the dynamic K-loop**, never above. See
   [pitfalls](../../pitfalls/cutedsl/nvfp4-gemm-pitfalls.md#1-recast-views-must-be-inside-the-dynamic-loop).
6. **BLOCK_K=128 with 2 sub-buffered smem (sA0/sA1).** Don't try a single 5D smem layout
   with K_BLOCK as a middle dim — ldmatrix breaks on the strided slice.
7. **STAGES=4** (drift tolerance ≥ 2, no per-K-iter barrier required).
8. Don't bother with: register K-pipelining (compiler already does it),
   STAGES > 4, depth-2 SF prefetch (correctness fragile), CUTLASS-style swizzle (no win
   on this kernel).

## Related docs

- Atom-level inline-PTX demo + register packing convention:
  [docs/nvidia/blackwell-geforce/ref-docs/cutedsl/sm120-nvfp4-inline-ptx-gemm.md](sm120-nvfp4-inline-ptx-gemm.md)
- Pitfalls (recast, pack, bulk+pipeline):
  [docs/nvidia/blackwell-geforce/pitfalls/cutedsl/nvfp4-gemm-pitfalls.md](../../pitfalls/cutedsl/nvfp4-gemm-pitfalls.md)
- Final kernel:
  [reference-kernels/.../sm120_nvfp4_persistent_gemm_pro5000.py](../../../../../reference-kernels/nvidia/blackwell-geforce/cutedsl/cutlass/sm120_nvfp4_persistent_gemm_pro5000.py)
- Pack helpers:
  [reference-kernels/.../sm120_nvfp4_pack_helpers.py](../../../../../reference-kernels/nvidia/blackwell-geforce/cutedsl/cutlass/sm120_nvfp4_pack_helpers.py)
- Same-arch journey for a different kernel (cp.async cache mode trick):
  [sm120-gdn-decode-fp32state-bf16qkv-optimization.md](sm120-gdn-decode-fp32state-bf16qkv-optimization.md)
- CUTLASS source studied for the recipe:
  `include/cutlass/gemm/collective/sm120_blockscaled_mma_tma.hpp`,
  `include/cutlass/gemm/collective/builders/sm120_blockscaled_mma_builder.inl`
- Block-scaled background:
  [cutlass-quantization-block-scaled.md](../../../common/ref-docs/cutedsl/cutlass-quantization-block-scaled.md)
- cute-DSL pipeline + inline-PTX docs:
  [cutedsl-pipeline-patterns.md](../../../common/ref-docs/cutedsl/cutedsl-pipeline-patterns.md),
  [cutedsl-inline-ptx-patterns.md](../../../common/ref-docs/cutedsl/cutedsl-inline-ptx-patterns.md)
