# GPU Kernel Pitfalls

Non-obvious traps encountered while implementing or porting GPU kernels.
Each entry includes the trap, the symptom, the root cause, and the lesson.

Organized by GPU vendor → DSL/framework → kernel category.

## Index

### AMD CDNA3 (gfx942) — FlyDSL

- [flash-attn-pitfalls.md](amd/flydsl/flash-attn-pitfalls.md) — bf16 / GQA flash-attention forward
  on MI308X. Bank-conflict math by element width, scheduling-hint loop dependencies, aiter
  install gotchas on ROCm 6.4; **v14: `permlanex16` is RDNA-only (gfx942 uses `ds_swizzle`),
  `rocdl.ds_swizzle` SSA-offset arg doesn't fold to imm (use inline asm), `v_perm_b32` byte-sel
  encoding inverts expectations on current LLVM (use `vector.from_elements`), `VEC_WIDTH=8`
  baseline has latent correctness bug; `bf16_trunc_pack_v*` not dtype-aware (fp16 → garbage,
  must use `arith.trunc_f` for IEEE round-to-nearest)**. 14 traps total.
- [chunk-gdn-mi308x-wave-specialization-pitfalls.md](amd/flydsl/chunk-gdn-mi308x-wave-specialization-pitfalls.md) —
  Chunk-GDN wave-specialized megakernel on MI308X. FlashQLA/Hopper mechanism cannot be copied
  directly; MI308X requires same-boundary Triton baseline, rocprofv3-only performance evidence,
  BDV64 hot path plus BDV32 small-H path, and counter-validated LDS/barrier decisions. 6 traps total.
- [fused-moe-fp8-ptpc-pitfalls.md](amd/flydsl/fused-moe-fp8-ptpc-pitfalls.md) —
  FP8 PTPC Fused MoE on MI308X. Task66 is a pause checkpoint, not full completion,
  and atrex-open v2 is a separate full-pipeline parity archive; BF16/F16 references,
  target_us timing-source drift, skip-atomic/no-output stage2, `block_m=8`, full
  intermediate reduce, task65 rowctx/rowinfo-LDS probes, host-sync valid-block reads,
  M=1 stream-boundary removal, routing re-sort, and M=1 event-average-only decisions
  must not be reused as positive evidence. 11 traps total.

### NVIDIA Blackwell GeForce (sm_120) — CuTeDSL

- [nvidia/cutedsl/gdn-chunk-fwd-pitfalls.md](nvidia/cutedsl/gdn-chunk-fwd-pitfalls.md) —
  Gated DeltaNet chunk forward on Pro5000. cp.async bf16 layout/alignment,
  TMA deadlocks and small-tile regression, direct global V RHS scalar-LDG
  regression, K-decay scratch algebraic removal, transposed LDSM atom, and
  no-cache V113 final-state acceptance discipline. **26 traps**.
- [nvidia/cutedsl/gdn-decode-pitfalls.md](nvidia/cutedsl/gdn-decode-pitfalls.md) —
  Gated DeltaNet decode on Pro5000. `assumed_align=16` requirement for cp.async
  cp_size=128b, L1-hit ≠ good (L2 false-saturation diagnosis), `cpasync.CopyOp`
  is abstract, mbarrier hand-rolling hangs, K-distributed multi-warp loses
  occupancy gain to SMEM reduction, sm_120 is NOT sm_100 (no tcgen05/wgmma/TMEM),
  `cute.arch.vector.from_elements` needs `.value` not `.ir_value`. **12 traps**.

### NVIDIA Blackwell GeForce (sm_120) — Triton

- [nvidia/triton/sm120-fused-rmsnorm-gated-pitfalls.md](nvidia/triton/sm120-fused-rmsnorm-gated-pitfalls.md) —
  Fused RMSNormGated + SiLU(z) gating on RTX PRO 5000 (vLLM `_deltanet_post`).
  `cache_modifier=".cg"` + LDG.128 hints can land in PTX with **0 % perf gain**
  (the bottleneck was structural, not microarch); **per-thread-regs > 255 ⇒ spill
  reasoning is unreliable** on Triton (always grep PTX `st.local`/`ld.local`);
  `BLOCK_M` sweep can show **27× max/min spread** at the same shape; "% memcpy
  ceiling > 100 %" is real when R:W ≠ 1:1; `num_stages` is a near-no-op for
  sm_120 memory-bound elementwise. **5 traps**.

### NVIDIA Blackwell datacenter (sm_100, B200) — Triton

- [nvidia/triton/sm100-sparse-decode-split-k-pitfalls.md](nvidia/triton/sm100-sparse-decode-split-k-pitfalls.md) —
  DSA sparse attention (`H=16` MLA) + FP8 top-k indexer on B200, SM-starved
  split-K decode. **Split-K is the structural win** when 1 CTA/token idles
  ~147/148 SMs; `exp2(-inf - -inf)=NaN` guard for online softmax over empty
  tiles; a scalar `if` in the hot loop kills `num_stages` prefetch (+25%);
  `static_range` helps only at `N×tile_bytes ≲ 8 KB`; `num_stages` no-op on
  loop-free kernels; cross-CTA atomic barrier (`ld.volatile`+`debug_barrier`,
  monotonic counter) needs all CTAs resident; **Triton 3.6 clusters ⊥ atomic
  sync**; `cache_modifier`/`eviction_policy` are orthogonal regime-dependent
  levers (`.cg`+`evict_last` illegal on stores); `.item()` is a structural
  barrier CUPTI sees (branch on `tensor.shape` instead); `tl.sort` blows up past
  BLOCK_N≈2048 (use radix-select); verify intrinsic throughput before designing
  around it (`tl.histogram` ~30× off); `input_precision="ieee"` inert for
  bf16×bf16. **12 traps**.

### NVIDIA Blackwell datacenter (sm_100, B200) — Gluon

- [nvidia/gluon/sm100-blackwell-primitives-pitfalls.md](nvidia/gluon/sm100-blackwell-primitives-pitfalls.md) —
  Porting the DSA Triton kernels to Gluon on B200. **`gl.dot_fma` is ~60-80×
  slower than `tl.dot`** (software FMA, no tensor cores — use `bw.tcgen05_mma` +
  `bw.alloc_tmem`); `dot_fma` layout/dtype rules are unforgiving (`k_width=0`,
  no auto-upcast); **`bw.tcgen05_mma` is uncompetitive at small `H`** (`blockM=64`
  forces ≥75% phantom-row padding at `H=16`); `gl.barrier` listed but not
  callable; layouts are explicit (`warps_per_cta` sums to `num_warps`,
  `convert_layout` bridges reductions); **`translator_helpers` is bring-up, not a
  perf port** (net-slower, never reaches `tcgen05_mma`). **6 traps**.

### NVIDIA Blackwell GeForce (sm_120) — CUDA

- [nvidia/cuda/sm120-nvfp4-decode-gemm-production-pitfalls.md](nvidia/cuda/sm120-nvfp4-decode-gemm-production-pitfalls.md) —
  NVFP4 decode and prefill GEMM production pitfalls: Split-K shape scope,
  CUDA Graph nsys requirements, stale scale-factor layout, b12x comparator
  discipline, cold-cache residency, structural DRAM ceiling, and scalar SF
  shared-memory conflicts. **10 traps**.
- [nvidia/cuda/sm120-rmsnorm-mlp-pdl-pitfalls.md](nvidia/cuda/sm120-rmsnorm-mlp-pdl-pitfalls.md) —
  RMSNorm + input NVFP4 quant -> parent C1 handoff pitfalls: PDL is not
  tile-level fusion, early C1 needs a device-side wait, row-chunk pipelines
  multiply launch/scheduler overhead, and sign-flipping TTFT should not be
  promoted. **7 traps**.

## How to add a new entry

1. File path: `pitfalls/<vendor>/<framework>/<short-name>.md`
2. Each pitfall section: trap → symptom → reality → why → lesson.
3. Cross-link the optimization journey doc in `ref-docs/` and the
   reference impl in `reference-kernels/`.
