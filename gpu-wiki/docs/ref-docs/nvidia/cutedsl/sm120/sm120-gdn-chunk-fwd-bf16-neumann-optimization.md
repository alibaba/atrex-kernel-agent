# CuTeDSL Gated DeltaNet Chunk Forward (bf16, Precomputed Neumann) on SM120

This document supersedes the earlier 1.18 ms and V31 0.615 ms SM120 GDN
chunk-forward notes.  The old 1.18 ms reference kernel was preserved as
[`sm120_gdn_chunk_fwd_3k.pre-v31-1p18ms.bak`](../../../../../reference-kernels/nvidia/blackwell-geforce/cutedsl/gdn_chunk_fwd/sm120_gdn_chunk_fwd_3k.py);
the V31 production snapshot was preserved as
[`sm120_gdn_chunk_fwd_3k.v31-0p615ms.bak`](../../../../../reference-kernels/nvidia/blackwell-geforce/cutedsl/gdn_chunk_fwd/sm120_gdn_chunk_fwd_3k.py);
the current production reference is
[`sm120_gdn_chunk_fwd_3k.py`](../../../../../reference-kernels/nvidia/blackwell-geforce/cutedsl/gdn_chunk_fwd/sm120_gdn_chunk_fwd_3k.py).

## Target Hardware

| Item | Value |
|------|-------|
| GPU | NVIDIA RTX PRO 5000 Blackwell, `sm_120a` |
| Framework | CuTeDSL / CUTLASS Python DSL |
| Shape | `B=1, T=6144, H=16, HV=32, K=128, V=128` |
| Dtype | bf16 Q/K/V/output, fp32 accumulators and recurrent state |
| Chunking | `BT=32`, `BV=16`, `NUM_THREADS=128` |
| Reference | FLA `chunk_gated_delta_rule(..., cu_seqlens=[0,T], use_qk_l2norm_in_kernel=True)` |
| Archived comparison contract | Directional input, `output_final_state=True`, no preprocess cache |
| Correctness gate | `rel_err < 0.01` plus final-state check |

## Algorithm Baseline

The accepted implementation uses a 3-kernel no-cache pipeline.  K0, K_inv, and
K1 recompute every execution; preprocess caching, CUDA graph/static replay, and
FLA fallback are not part of the accepted path.

1. `K0`: L2-normalize Q/K and precompute `KK = K_norm @ K_norm^T`.
2. `K_inv`: build the per-chunk Neumann inverse and gated QK intermediates in
   parallel over chunks and HV heads.
3. `K1`: sequential chunk scan over the recurrent state, fused with chunk-O.

For each chunk, K1 performs:

```text
kS = K_norm @ state
qS = Q_norm @ state
RHS = beta * (V - exp_gc * kS)
v_new = M_neumann @ RHS
O = scale * (exp_gc * qS + gated_qk @ v_new)
state = exp_gc_last * state + K^T @ (exp_decay * v_new)
```

The final identity in the state update is the V31 change.  Earlier versions
materialized `K[t,k] * exp_decay[t]` into a scratch tile.  V31 instead stores
`v_new[t,v] * exp_decay[t]` into `sNK_A` and reuses the already staged `sK`
through a transposed LDSM A-copy atom.

V113 keeps that algebra and removes additional LDSM redundancy: K1 reuses the
`sS` B fragment for both `K*S` and `Q*S`, reuses the `sNK_A` B fragment across
four final-state update fragments, fuses the non-split `acc_qS *= exp_gc[row]`,
and uses a B `LdMatrix` `num_matrices=2` launch for the non-tail T=6144 path.
Tail / `STATE_SPLIT=True` keeps the safer B4 path.

## Kernel Resource Footprint (Final)

V113 launch stats were captured with NCU `LaunchStats` on the archived accepted
CuTeDSL path.

| Kernel | Grid blocks | Threads | Registers/thread | Dynamic smem/block | Waves/SM |
|--------|------------:|--------:|-----------------:|-------------------:|---------:|
| K0 `preprocess_kk` | 3072 | 128 | 143 | 17.41 KiB | 9.31 |
| K_inv `precompute_inv` | 6144 | 128 | 55 | 12.54 KiB | 7.98 |
| K1 `fused_chunk_h_v31_final_state` | 256 | 128 | 72 | 30.46 KiB | 0.78 |

K1 still uses 128-bit `cp.async` for K/Q/M/GQK/V staging, `LDSM.16.M88`
feeders, and R2S `STSM` stores for state/RHS/v_new/scaled-vnew shared bridges.

## Final Performance vs FLA

The final accepted V113 numbers are same-process CuTeDSL vs FLA measurements on
the archived directional final-state comparison contract. They are not using
preprocess cache or replay.

| Path | P50 ms | Speedup vs FLA |
|------|-------:|---------------:|
| V113 accepted no-cache CuTeDSL path | 0.5316 | 1.5128x |
| V113 stop-hook rerun | 0.5315 | 1.5138x |
| FLA same-process baseline | 0.8041-0.8046 | 1.0000x |
| Previous V31 CuTeDSL path | 0.6142-0.6152 | 1.42x |

Correctness at `T=6144`: `rel_err=9.174309e-03`,
`state_rel_err=0.003994480706751347`, PASS.  Tail directional regression also
passed: `5 passed, 16 warnings`.

V113 one-call nsys latency split:

| Kernel | nsys one-call time |
|--------|-------------------:|
| K0 `preprocess_kk` | 78.080 us |
| K_inv `precompute_inv` | 42.720 us |
| K1 `fused_chunk_h_v31_final_state` | 383.456 us |
| Total kernel time | 504.256 us |

V122 NCU counter-bandwidth pass:

| Kernel | NCU duration | DRAM total | DRAM BW | L2 bytes | L2 BW | L2 hit |
|--------|-------------:|-----------:|--------:|---------:|------:|-------:|
| K0 `preprocess_kk` | 81.09 us | 73.67 MB | 908.50 GB/s | 126.80 MB | 1.56 TB/s | 29.50% |
| K_inv `precompute_inv` | 68.19 us | 12.52 MB | 183.60 GB/s | 64.02 MB | 0.94 TB/s | 56.08% |
| K1 `fused_chunk_h_v31_final_state` | 565.09 us | 152.45 MB | 269.78 GB/s | 1.62 GB | 2.87 TB/s | 91.32% |

NCU duration is inflated relative to nsys for K1; use the NCU table for memory
counter breakdown, not production latency.  The final NCU evidence is that K1
is not raw-DRAM bound; it has high L2 reuse and still spends most wall time in
the serial chunk scan.

Historical V31 K1 ncu comparison against V29:

| Metric | V29 | V31 |
|--------|----:|----:|
| K1 duration | 706.40 us | 607.33 us |
| Issued instructions | 74,653,696 | 48,299,008 |
| Registers/thread | 128 | 110 |
| Dynamic smem | 32.64 KiB | 32.64 KiB |
| Achieved occupancy | 19.02% | 19.39% |
| `LDS.U16` | 6,291,456 | 0 |
| `LDS.64` | 1,572,864 | 0 |
| `LDSM.16.M88.4` | 8,650,752 | 6,684,672 |
| `STSM` | 2,752,512 | 1,376,256 |
| `F2FP.BF16.F32.PACK_AB` | 5,898,240 | 3,145,728 |
| `SHF` | 3,154,944 | 403,456 |
| `BAR.SYNC` | 1,376,256 | 1,179,648 |
| `STS.U16` | 0 | 0 |
| `LDGSTS.E.BYPASS.LTC128B.128` | 2,162,688 | 2,162,688 |

## Optimization Journey

### V0 — Scalar 3-kernel baseline

The first correct implementation split preprocessing, inverse construction, and
state scan.  K1 used scalar loads/stores and a separate chunk-O stage.  End-to-end
latency was about 3.72 ms, dominated by K1 and chunk-O global-memory round trips.

### V1 — t-outer/v-inner state update and chunk-O fusion

Changing state update from v-outer/t-inner to t-outer/v-inner avoided ptxas
hoisting all `sK[t]` and `exp_decay[t]` values across the V loop.  Inlining chunk-O
into K1 kept `acc_qS` in registers and removed large GMEM intermediates.  This
reduced latency to about 1.62 ms.

### V3 — bf16 cp.async alignment fix

128-bit cp.async for bf16 requires `val_layout=(1,8)`, not `(1,4)`, because 8 bf16
elements make 128 bits.  Combined with `KQ_STRIDE=K_DIM+8` and `(B,H,T,K)` normalized
layout, this brought the old production kernel to about 1.18 ms.

### V4/V5 — TMA and megakernel attempts rejected

TMA is available on SM120, but GDN chunk-forward tiles are small and the chunk scan
has a real serial dependency.  TMA + warp-specialization added mbarrier/protocol
overhead without overlap and regressed badly.  Megakernel variants also recomputed
or serialized work across V tiles.

### V14-V17 — cp.async staging attribution

Staging M/GQK together mattered; staging either one alone did not.  Staging V through
cp.async also helped compared with scalar shared stores.  This established that
K/Q/M/GQK/V should remain staged through shared memory for the fused K1, except for
carefully bounded probes.

### V19 — LDSM-fed MMA operands

Replacing scalar shared-to-register MMA feeders with `LdMatrix8x8x16bOp` produced
the first structural SASS shift: `LDSM` appeared, static scalar `LDS` sites dropped,
and K1 fell below the old 1 ms region.  Remaining cost moved to accumulator-to-shared
stores and state-update scratch materialization.

### V22-V24 — Register-resident state fragments and scratch scheduling

Keeping four `(BT,BV)` state fragments in registers converted the old scalar
state-update FFMA wall into HMMA work, but introduced heavy shared scratch traffic.
Using a dedicated scratch tile and removing one reuse barrier made this path the
new base for store-side cleanup.

### V25-V29 — R2S/STSM cleanup and direct output store

The next sequence converted scalar shared stores to R2S `STSM` stores:

- V25: state-fragment spill by `STSM`.
- V26: direct 32-bit output copy from accumulator to GMEM; 64/128-bit variants hit
  CuTeDSL layout verification limits.
- V27: K-decay scratch R2S, large win.
- V28: RHS and v_new R2S, eliminating `STS.U16`.
- V29: skipped a dead `v_new -> sV` store, small but correct.

V29 reached `0.6859 ms`, faster than the fast path but still above the FLA 1.3x
line.

### V30 — Direct global V RHS rejected

After V29, original V is only needed to form RHS, so one plausible probe was to skip
`mV -> sV` staging and read `gV_tile[row, v_col]` directly during RHS R2S.  It was
correct but slower: repeat-100 P50 regressed to `0.7200 ms`, K1 to `743.58 us`.

SASS showed why: direct V created scalar global loads on the RHS critical path
(`LDG.E.U16` rose), while the shared-load reduction was too small.  The lesson is
not "remove shared staging whenever data is one-use"; for this kernel, cp.async V
staging is still better than scalar direct global loads.

### V31 — Scaled-vnew state update accepted

The winning transformation moved the decay multiply from the K operand to the V
operand:

```text
sum_t (K[t,k] * exp_decay[t]) * v_new[t,v]
==
sum_t K[t,k] * (exp_decay[t] * v_new[t,v])
```

V31 stores `acc_vnew * sExpDecay[t]` into `sNK_A` after chunk-O, then performs four
state-update MMAs from transposed `sK` shared views into the four state fragments.
The normal LDSM A-copy atom failed MLIR verification for this transposed view
because the source pointer alignment was only 16 bits.  Adding a dedicated
`transpose=True` LDSM A-copy atom fixed it.

This removed K-decay scratch materialization and the scalar shared-load wall:
`LDS.U16/LDS.64 -> 0`, `STSM` halved, `F2FP` nearly halved, registers dropped to 110,
and end-to-end latency reached `0.614-0.615 ms`.

### V32-V82 — Comparison contract tightened to no-cache directional final-state

The next phase moved from the original FLA-varlen target to a stricter comparison
contract: directional inputs, `output_final_state=True`, same-process FLA compare,
and a hard no-preprocess-cache rule. Several shortcuts were rejected:

- Preprocess cache/static replay/CUDA graph replay: invalid because K0/Kinv must
  run every execution.
- Whole-flow warp-specialized/TMA fusion: correct variants were far slower
  because they repeated inverse work per V tile or added producer/consumer
  protocol cost to a serial recurrence.
- FlashInfer/current remote paths: either not applicable to the final-state
  contract or not a bounded patch to this CuTeDSL path.

The useful change in this phase was early-GQK / beta folding in K_inv plus keeping
the original 3-kernel layering.  The layering matches the vLLM/FLA-style split:
parallel preprocess, parallel inverse/intermediate construction, then the single
serial state scan.

### V83-V112 — Bounded Ampere/Ada-style cleanup

Public Ampere/Ada guidance mapped to local bounded mechanisms already present in
this kernel: `cp.async` cache mode, LDSM/STSM/R2S, output-store width, TMA/warp
specialization, and launch-policy variants.  Most probes were rejected because
they either broke tail correctness, increased scalar global loads, or only moved
latency between K0/Kinv/K1.

Important rejected probes:

- `cp.async` `GLOBAL -> STREAMING` on the final K1 path failed correctness in the
  final-state contract (`rel_err=1.662843e-02`), so the accepted path keeps
  `LoadCacheMode.GLOBAL`.
- Compact `sS` / `sNK_A` stride variants were correct or nearly correct but
  slower; preserving the aligned/padded layouts remained better.
- K0 `KK/QK` B-reuse was correct but slower (`0.5450 ms` vs the later accepted
  path near `0.532 ms`).

### V113 — Reuse-B LDSM and Bx2 launch accepted

V113 is the final accepted no-cache path.  It makes two local changes inside K1:

1. `_mma_kq_s_reuse_b_ldsm` loads the `sS` B fragment once per `kk`, then computes
   both `K*S` and `Q*S`.  The non-split `acc_qS *= exp_gc[row]` multiply is fused
   into this helper.
2. `_mma_state4_ldsm_reuse_b` loads the `sNK_A` B fragment once per `kk`, then
   updates the four final-state fragments.  The non-tail T=6144 launcher uses B
   `LdMatrix` `num_matrices=2`; the tail / `STATE_SPLIT=True` path keeps the B4
   safe launcher because a conditional Bx2/Bx4 copy-atom launch failed T=50000
   output correctness.

Rejected V113 probes are part of the recipe:

| Probe | Result | Decision |
|-------|-------:|----------|
| Cross-chunk Q/V/M/GQK prefetch | `rel_err=1.376146e-01`, `0.5894 ms` | incorrect |
| Cross-chunk M/V prefetch | `rel_err=4.472475e-02`, `0.5921 ms` | incorrect |
| K/Q B-reuse with A autovec | correct, `0.5980 ms` | slower |
| `sS` compact stride | `rel_err=1.261467e-02`, `0.7539 ms` | incorrect/slower |
| `sNK_A` compact stride | correct, `0.5799 ms` | slower |
| K0 `KK/QK` B-reuse | correct, `0.5450 ms` | slower |
| Conditional Bx2/Bx4 in one launch | T=50000 output rel_err `1.5021e-02` to `4.7210e-02` | split launch |

Strict repeated runs landed at `0.5328-0.5329 ms` vs FLA `0.8043-0.8047 ms`
(`1.5097-1.5105x`), and the later stop-hook rerun landed at `0.5315 ms` vs
`0.8046 ms` (`1.5138x`).

### V114-V122 — TMA, cache-policy, and NCU bandwidth follow-up

The post-acceptance recheck kept V113 unchanged:

- Existing TMA harness failed before timing with `CUDA_ERROR_MISALIGNED_ADDRESS`,
  remained output-only/no-final_state, and therefore was not an accepted path.
- L2 access-policy-window variants for `neumann_m`, `gated_qk`, and combined
  `M/GQK` were correct but slower: `0.5737 ms`, `0.5805 ms`, and `0.5759 ms`.
- The final V122 NCU bandwidth pass confirmed the current memory profile: K0 has
  the highest DRAM bandwidth (`908.50 GB/s`), while K1 has high L2 traffic and
  reuse (`1.62 GB`, `91.32%` L2 hit, `2.87 TB/s` L2 counter bandwidth).

## Remaining Bottlenecks

- K1 is still a serial chunk scan; inter-chunk parallelism is not available without
  changing the recurrence.
- K1 still stages K/Q/M/GQK/V through shared memory with `cp.async`; this is intentional
  because direct global RHS was slower.
- K1 launch stats show only `0.78` waves/SM for the accepted path.  This is a
  structural result of the 256-block V-tile grid and serial scan, not a simple
  register-only problem: the accepted K1 uses 72 registers/thread and 30.46 KiB
  dynamic shared memory per block.
- K1 NCU bandwidth is not close to the raw DRAM ceiling.  The useful evidence is
  high L2 traffic/reuse plus nsys-dominant K1 latency, pointing to dependency and
  L2/shared dataflow rather than HBM bandwidth.
- Wider GMEM output stores would be desirable, but CuTeDSL `CopyUniversalOp` from the
  accumulator fragment only compiled reliably at 32-bit width in this layout.

## What Would Close the Remaining Gap

1. A legal CuTeDSL layout for 64/128-bit accumulator-to-GMEM output copies.
2. A better state-fragment layout that keeps V31/V113 algebra while reducing the
   remaining `LDSM` count without breaking the tail/final-state path.
3. A real recurrence-level algorithm change or prefix-scan formulation; local
   warp-specialized/TMA fusion alone did not create inter-chunk parallelism.
4. A framework/compiler update that supports the desired fused FlashQLA/FlashInfer
   style without 4.4.x layout and pipeline limitations.

Do not retry broad TMA, full megakernel fusion, or scalar direct global V RHS for this
shape without new evidence; those were measured regressions.  Do not cache
preprocess outputs for this benchmark contract; K0 and K_inv are part of every
execution.

## Sustained Recipe

1. Keep the 3-kernel split: K0 normalize, K_inv precompute, K1 serial fused scan.
2. Use `(B,H,T,K)` q/k normalized layout so K1 rank-2 slices are contiguous.
3. Use 128-bit cp.async with bf16 `val_layout=(1,8)` and 8-element SMEM row padding.
4. Fuse chunk-O into K1 and keep `acc_qS` live in registers.
5. Use LDSM-fed MMA operand copies for all K1 HMMA stages.
6. Keep recurrent state as register fragments, not fp32 shared memory.
7. Replace scalar accumulator-to-shared stores with R2S `STSM`.
8. Store output directly from the accumulator fragment; do not route O through shared.
9. Move `exp_decay[t]` to the v-new side and reuse transposed `sK` for state update.
10. Reuse B fragments only where the same fragment feeds multiple MMAs in the same
    `kk`; V113's accepted cases are `sS` for K*S/Q*S and `sNK_A` for four
    final-state fragments.
11. Keep `LoadCacheMode.GLOBAL` on the accepted path unless a new cache-policy probe
    passes output and final-state correctness.
12. Validate every step with ncu/nsys/SASS; CUDA-event timing alone hid several wrong
    turns, and NCU duration itself can be inflated relative to production nsys time.

## Related Docs

- Reference kernel:
  [`reference-kernels/nvidia/blackwell-geforce/cutedsl/gdn_chunk_fwd/sm120_gdn_chunk_fwd_3k.py`](../../../../../reference-kernels/nvidia/blackwell-geforce/cutedsl/gdn_chunk_fwd/sm120_gdn_chunk_fwd_3k.py)
- Pitfalls:
  [`docs/pitfalls/nvidia/cutedsl/gdn-chunk-fwd-pitfalls.md`](../../../../pitfalls/nvidia/cutedsl/gdn-chunk-fwd-pitfalls.md)
- SM120 TMA API notes:
  [`sm120-pipeline-tma-async-api-notes.md`](sm120-pipeline-tma-async-api-notes.md)
- GDN decode, different T=1 algorithm:
  [`sm120-gdn-decode-fp32state-bf16qkv-optimization.md`](sm120-gdn-decode-fp32state-bf16qkv-optimization.md)
- AMD FlyDSL chunk-GDN wave-specialized megakernel:
  [`docs/ref-docs/amd/flydsl/gfx942/cdna3-chunk-gdn-mi308x-wave-specialized-megakernel-optimization.md`](../../../amd/flydsl/gfx942/cdna3-chunk-gdn-mi308x-wave-specialized-megakernel-optimization.md)
