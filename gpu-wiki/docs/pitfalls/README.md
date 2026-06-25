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

- [nvidia/triton/](nvidia/triton/README.md) —
  Fused RMSNormGated + SiLU(z) gating on RTX PRO 5000 (vLLM `_deltanet_post`).
  `cache_modifier=".cg"` + LDG.128 hints can land in PTX with **0 % perf gain**
  (the bottleneck was structural, not microarch); **per-thread-regs > 255 ⇒ spill
  reasoning is unreliable** on Triton (always grep PTX `st.local`/`ld.local`);
  `BLOCK_M` sweep can show **27× max/min spread** at the same shape; "% memcpy
  ceiling > 100 %" is real when R:W ≠ 1:1; `num_stages` is a near-no-op for
  sm_120 memory-bound elementwise. **5 traps**.

## How to add a new entry

1. File path: `pitfalls/<vendor>/<framework>/<short-name>.md`
2. Each pitfall section: trap → symptom → reality → why → lesson.
3. Cross-link the optimization journey doc in `ref-docs/` and the
   reference impl in `reference-kernels/`.
