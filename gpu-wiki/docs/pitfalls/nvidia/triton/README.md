# Triton Pitfalls

| File | Kernel | Hardware | Trap count |
|------|--------|----------|-----------|
| sm120-fused-rmsnorm-gated-pitfalls | vLLM `_deltanet_post` fused RMSNormGated + SiLU(z) gating | sm_120 (RTX PRO 5000 Blackwell-GeForce) | 5 |

## Cross-Pitfall Summary (sm_120 Triton)

- **`cache_modifier=".cg"` + LDG.128 hints can land correctly in PTX yet yield 0% performance** — micro-optimizations can only improve things "inside the program" and cannot rescue structural bottlenecks (occupancy / wave tail). When you see 0% gain, stop adding hints and pivot to investigating BLOCK_M and grid.
- **The theoretical derivation "per-thread regs > 255 ⇒ spill" is not trustworthy on Triton** — Triton folds and reuses registers aggressively. Any spill assumption must be verified by grepping PTX for `st.local`/`ld.local`.
- **Under the same shape, a `BLOCK_M` sweep can have a 27× range** — on sm_120 elementwise / norm kernels, BLOCK_M is the primary knob; a sweep costs one minute and can yield 4-27× gains.
- **"Exceeding 100% memcpy ceiling" is not a measurement error** — when the kernel R:W is not 1:1 (the D2D ceiling assumes 1:1), 122% is a genuine hardware advantage; don't try to "fix" it by optimizing a non-existent gap.
- **`num_stages` has virtually no effect on sm_120 memory-bound elementwise kernels** — don't treat it as a primary autotune knob; spend that budget on BLOCK_M instead.

## Cross-Platform Correlation

- Similar measurement traps (e.g., ncu L1-hit-rate includes
  `ld.shared`) on CuTeDSL paths for sm_120: [`docs/pitfalls/nvidia/cutedsl/`](../cutedsl/README.md)
