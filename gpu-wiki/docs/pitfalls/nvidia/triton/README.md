# Triton Pitfalls

| File | Kernel | Hardware | Trap count |
|------|--------|----------|-----------|
| [sm120-fused-rmsnorm-gated-pitfalls.md](sm120-fused-rmsnorm-gated-pitfalls.md) | vLLM `_deltanet_post` fused RMSNormGated + SiLU(z) gating | sm_120 (RTX PRO 5000 Blackwell-GeForce) | 5 |
| [sm100-sparse-decode-split-k-pitfalls.md](sm100-sparse-decode-split-k-pitfalls.md) | DSA sparse attention (`H=16` MLA) + FP8 top-k indexer — SM-starved split-K decode | sm_100 (B200 Blackwell datacenter) | 12 |

## Cross-Pitfall Summary (sm_100 Triton, SM-starved split-K decode)

- **Split-K / flash-decoding is the structural win on SM-starved grids** — 1 CTA
  per token leaves ~147/148 SMs idle; manufacture parallelism before any tile or
  cache tuning. Re-open `NUM_SPLITS` after every structural rewrite.
- **`exp2(-inf - -inf) = NaN`** poisons online softmax over a possibly-empty tile
  — guard the running max when you split a full reduction into partials.
- **A scalar `if` in the hot loop kills `num_stages` prefetch** (+25%); keep the
  body unconditional and use a precomputed dynamic loop bound.
- **`tl.static_range` only helps when `N×tile_bytes ≲ 8 KB`**; `num_stages` is a
  no-op on loop-free single-dot kernels.
- **Cross-CTA atomic barrier works** (`ld.volatile` spin + `debug_barrier`,
  monotonic counter) only when all CTAs fit on the SMs; **Triton 3.6 clusters and
  atomic sync are mutually exclusive** (`PlanCTA` assertion).
- **`cache_modifier` and `eviction_policy` are orthogonal, regime-dependent
  levers** — per-load not global; `.cg`+`evict_last` is illegal on stores.
- **`.item()` is a structural barrier (and CUPTI sees it)**; branch on
  `tensor.shape` (free) instead.
- **`tl.sort` blows up past BLOCK_N≈2048** — use radix-select; **verify any
  intrinsic's compiled throughput** (`tl.histogram` was ~30× off the cost model)
  before designing around it.
- **`input_precision="ieee"` is inert for bf16×bf16 dot** on sm_100.

## Cross-Pitfall Summary (sm_120 Triton)

- **`cache_modifier=".cg"` + LDG.128 hints can land correctly in PTX yet yield 0% performance** — micro-optimizations can only improve things "inside the program" and cannot rescue structural bottlenecks (occupancy / wave tail). When you see 0% gain, stop adding hints and pivot to investigating BLOCK_M and grid.
- **The theoretical derivation "per-thread regs > 255 ⇒ spill" is not trustworthy on Triton** — Triton folds and reuses registers aggressively. Any spill assumption must be verified by grepping PTX for `st.local`/`ld.local`.
- **Under the same shape, a `BLOCK_M` sweep can have a 27× range** — on sm_120 elementwise / norm kernels, BLOCK_M is the primary knob; a sweep costs one minute and can yield 4-27× gains.
- **"Exceeding 100% memcpy ceiling" is not a measurement error** — when the kernel R:W is not 1:1 (the D2D ceiling assumes 1:1), 122% is a genuine hardware advantage; don't try to "fix" it by optimizing a non-existent gap.
- **`num_stages` has virtually no effect on sm_120 memory-bound elementwise kernels** — don't treat it as a primary autotune knob; spend that budget on BLOCK_M instead.

## Cross-Platform Correlation

- Similar measurement traps (e.g., ncu L1-hit-rate includes
  `ld.shared`) on CuTeDSL paths for sm_120: [`docs/pitfalls/nvidia/cutedsl/`](../cutedsl/README.md)
