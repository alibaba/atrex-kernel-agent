---
pattern: flydsl_wave_specialized_megakernel
framework: FlyDSL
arch: cdna3/gfx942
gpu: MI308X
operator: Chunk-GDN fused forward
profiling: rocprofv3
knowledge_use: warp-specialization megakernel porting playbook
---

Applicability: backend: flydsl; hardware: amd; topic: reference

# MI308 Chunk-GDN Wave-Specialized Megakernel Playbook

Date: 2026-05-09

This reference distills reusable experience from porting the FlashQLA Hopper warp-specialization megakernel to MI308/CDNA3/FlyDSL. The full per-version changelog is no longer retained; the focus is on:

- When to attempt megakernel fusion;
- How to adapt Hopper design concepts into wave-specialization achievable on CDNA3;
- Acceptance thresholds for profile, ISA, LDS, and shape sweep;
- Which optimizations are effective on MI308 and which appear reasonable but should be avoided.

The reference implementation is located at `reference-kernels/amd/cdna3/flydsl/FlyDSL/`, with entry point
`chunk_gdn_flydsl_operator.py`. The gpu-wiki reference does not depend on the RTP-LLM runtime;
it only runs the FlyDSL backend megakernel, and the caller must pass in precomputed `a` and `g_cumsum`.
See the companion pitfalls page at
[`chunk-gdn-mi308x-wave-specialization-pitfalls.md`](../../pitfalls/flydsl/chunk-gdn-mi308x-wave-specialization-pitfalls.md).

## Applicability Card

### When to Use

When a linear attention / SSM / recurrent-state kernel satisfies the following conditions simultaneously, this experience can be applied as a warp-specialization megakernel porting method:

| Condition | How to Determine |
|---|---|
| The multi-kernel backend has large intermediate tensors | e.g., `w/u/h/v_new` is written to GMEM and later read back by a subsequent kernel |
| The backend has a chunk loop and recurrent state | e.g., `h += K^T @ V'`, with state dependencies between chunks |
| A single chunk contains multiple GEMM / elementwise stages | Intermediate values can be passed via LDS/register |
| The current bottleneck is not purely launch overhead | `rocprofv3` shows backend kernel dispatch dominates the time |
| The target GPU has sufficient LDS but cannot directly adopt Hopper SMEM | MI308 has ~64KB LDS and requires strict layout/alias |

Unsuitable situations:

- Intermediate tensors are small, and fusion would only save on launch overhead;
- Grid blocks are far fewer than CUs, and tile shapes cannot be split smaller;
- LDS cannot accommodate the necessary cross-stage data;
- Correctness depends on complex varlen/cache store, but there is no reliable reference.

### Acceptance Criteria

Use the following completion criteria for similar optimization work:

1. **Performance conclusions require `rocprofv3`**: Do not use `do_bench`, manual timing, or `torch.cuda.Event` as final conclusions.
2. **Baseline must be on the same boundary**: Use the current tuned Triton/Gluon comparison baseline for production performance comparisons; PyTorch is only for correctness.
3. **Must report resources**: LDS, VGPR/accum VGPR, scratch, barriers, and key waitcnt.
4. **Must check shape/grid**: At minimum, cover the hot shape and small-H / low-grid shapes; do not optimize for only one model dimension.
5. **Must provide a rejected table**: Each failed optimization must be explained with rocprof/ISA/PMC evidence, not just "no gain".

At minimum, the acceptance review must include:

| Artifact | Required Content |
|---|---|
| correctness | FlyDSL vs tuned baseline, including output and final/cache state |
| rocprofv3 table | Baseline, megakernel, and speedup for each T/shape |
| kernel trace mapping | Which baseline dispatches are summed, megakernel kernel name |
| ISA/resource snapshot | LDS, VGPR, scratch, barriers, key LDS/VMEM instructions |
| decision log | Accepted/rejected optimizations and reasons |

## One-Page Summary

- The final pipeline is not a fully single kernel, but rather 3 kernels:

```text
chunk_local_cumsum
  -> chunk_gated_delta_rule_fwd_kkt_solve_kernel
  -> FlyDSL megakernel(recompute_w_u + fwd_h + fwd_o)
```

- The core benefit comes from eliminating the GMEM intermediate tensor round-trips in the backend `recompute_w_u + fwd_h + fwd_o`, especially per-chunk state like `h[B,NT,H,K,V]`.
- MI308 cannot directly adopt the Hopper TMA/mbarrier/WGMMA pipeline; what can be migrated is the data flow and role division, not the hardware mechanisms.
- The viable CDNA3 approach is: 512-thread workgroup, 8 wave64, compute/producer wave division, single-buffer LDS with lifetime alias, and ordinary `buffer_load/ds_write` and `s_barrier`.
- The `(8,32,128,128)` hot path requires `BLOCK_DV=64`; otherwise, Q/K/A/g/beta staging and some GEMM work are duplicated.
- `(2,8)` / `(8,16)` small-H shapes require a BDV32 fast path to increase V-axis grid parallelism; the hot path tile should not be unconditionally applied to all shapes.
- The accepted version must simultaneously satisfy correctness, large-T rocprofv3 sweep, ISA resources, and consistent PMC/counter direction.

## FlashQLA to MI308 Migration Model

### Hopper Solution Abstraction

FlashQLA's Hopper/SM90 megakernel fuses the latter half as:

```text
cumsum -> kkt_solve -> megakernel(recompute_w_u + fwd_h + fwd_o)
```

Typical roles on the Hopper side:

| Role | Main Work |
|---|---|
| consumer_S | Maintains recurrent state `h`, performs decay and `K^T @ V'` |
| consumer_V | Performs `U=K@h`, `W=V-exp2(g)*U`, `Vd=A@W`, `V'` |
| consumer_O | Performs `Q@h`, `Q@K^T`, `Pg@Vd`, forms `O` |
| producer/storer | Moves Q/K/V/A/g/beta via TMA/async copy, writes O/state |

This solution relies on WGMMA async, TMA, mbarrier, relatively generous SMEM, and warpgroup-level resource management.

### MI308 Adaptation Principles

| Hopper Mechanism | MI308/CDNA3 Substitute | Migration Conclusion |
|---|---|---|
| WGMMA async | MFMA sync | Waves block after issuing MFMA; full async overlap cannot be expected |
| TMA/async copy | Plain `buffer_load + ds_write` | Producer waves consume execution resources themselves |
| mbarrier | Workgroup-level `s_barrier` | Barrier placement must be verified via ISA |
| Large SMEM double-buffer | ~64KB LDS | Primary task is fitting LDS, then addressing pipeline |
| Warpgroup resource split | Unified VGPR allocation by compiler | Lightening producer does not necessarily reduce full kernel VGPR |

The successful MI308 solution is not a "translation of FlashQLA," but rather one that preserves the fusion boundary and state residency:

```text
8 waves = 4 compute waves + 4 producer waves
single-buffer LDS + lifetime alias
h_acc stays in fp32 fragments
producer stages Q/K^T/A/g/beta in barrier windows
O direct GMEM store, not O-through-LDS
```

## Algorithm Boundary

Chunk-GDN has 6 GEMMs per `BT=64` chunk:

| GEMM | Computation | Description |
|---|---|---|
| 1 | `U = K @ h` | inter-chunk contribution |
| 2 | `O0 = Q @ h` | inter-chunk output |
| 3 | `P = Q @ K^T` | intra-chunk score |
| 4 | `Vd = Ag @ W` | inverse applied to W |
| 5 | `O += Pg @ Vd` | intra-chunk output |
| 6 | `h += K^T @ V'` | recurrent update |

The fair comparison boundary for the FlyDSL megakernel is `a/g_cumsum` already generated by the first half; the megakernel only fuses:

```text
recompute_w_u -> fwd_h -> fwd_o
```

Therefore, the performance baseline must also start from the same boundary: Triton comparison baseline = `recompute_w_u_fwd`
+ `chunk_gated_delta_rule_fwd_h` + `chunk_fwd_o`. Do not use the PyTorch reference as the performance baseline.

## Final Implementation Structure

### File Responsibilities

| File | Purpose |
|---|---|
| `chunk_gdn_flydsl_operator.py` | Standalone wrapper, validates shapes, constructs varlen chunk offsets |
| `fused_fwd_mi308x_v2.py` | Shape-aware front door, selects fast/tail/direct-store path |
| `fused_fwd_mi308x_v2_fast.py` | BDV64 hot path, primary `(8,32,128,128)` |
| `fused_fwd_mi308x_v2_bdv32_fast.py` | BDV32 small-H path, serves `(2,8)` / `(8,16)` |
| `reference-kernels/amd/cdna/triton/chunk_gdn/` | Migrated Triton back-half comparison baseline |

### Hot Path Resources

| Item | Value |
|---|---|
| GPU | MI308X / gfx942 |
| workgroup | 512 threads = 8 wave64 |
| compute waves | wave 0-3 |
| producer waves | wave 4-7 |
| chunk size | `BT=64` |
| hot shape | `(Hg,H,K,V)=(8,32,128,128)` |
| hot tile | `BLOCK_DV=64` |
| small-H tile | `BLOCK_DV=32` |
| LDS | about 63KB, near 64KB limit |
| scratch | 0 |

LDS retains only cross-stage essential data: `Q/K^T/A/h/W/Vd/Vn/Pg/g/beta`. `lds_k_row` is removed;
GEMM1 reads K directly from GMEM; `K^T` remains in LDS because GEMM3/GEMM6 are more sensitive to layout.

### Wave Roles

| Wave | Role | Work |
|---|---|---|
| 0-3 | compute | Primary MFMA computation, maintains `h_acc`, produces O/final state |
| 4-7 | producer | Stages Q/K^T/A/g/beta, coordinates barrier window |

The producer's responsibility is to create limited overlap, not to emulate Hopper TMA. Every producer must verify in the ISA whether
`s_barrier` is still at the expected position; if the compiler moves or removes the barrier, correctness may be directly invalidated.

## Performance Results

### Standalone 397B-TP2 back-half

Configuration:

- shape: `(B,Hg,H,K,V)=(1,8,32,128,128)`;
- boundary: `a/g_cumsum` precomputed;
- tool: `rocprofv3 --kernel-trace -f csv`;
- method: warmup 2, target 5, taking the last 5 target iteration P50;
- Triton sum: `recompute_w_u_fwd_kernel` + `chunk_gated_delta_rule_fwd_kernel_h_blockdim64`
  + `chunk_fwd_o` internal `zeros_like` fill + `chunk_fwd_kernel_o`.

| T | Triton back-half P50 | FlyDSL megakernel P50 | Speedup |
|---:|---:|---:|---:|
| 4096 | 1048.282us | 609.561us | 1.720x |
| 16384 | 4113.662us | 2498.645us | 1.646x |
| 65536 | 16403.075us | 9974.661us | 1.644x |
| 200000 | 50805.507us | 30473.504us | 1.667x |

Correctness spot-check, T=4096, same inputs:
`o_cos=0.99998677`, `o_mean_abs=1.081581e-06`, `h_cos=0.99999225`,
`h_mean_abs=1.668401e-05`. This is not performance evidence, only used to confirm consistent computation boundaries.

### RTP Direct-Store Operator

In the RTP integration path, the FlyDSL direct-store also writes cache-state into the megakernel, removing the external
same-boundary comparison step `fwd_h + store_ssm_state_to_block_map`. On long sequence `(8,32,128,128)`, the RTP operator
level is approximately `1.43-1.47x` relative to current Triton comparison data, and approximately `2.14-2.23x` relative to
old `507e` Triton comparison data.

### Shape Generalization

Shape generalization covers 10 runtime shapes for Qwen3.5/Qwen3.6. Key rules:

- `(8,32,128,128)` retains the BDV64 hot path;
- `(2,8,128,128)` and `(8,16,128,128)` use the BDV32 small-H fast path;
- tiny suffix `input_len=1/17`, 0.8B/2B shapes are not within the scope of this round of optimization;
- future shapes must first pass correctness + rocprofv3 sweep before entering dispatch.

## Optimization Journey (Condensed)

| Stage | Decision | Result |
|---|---|---|
| V1 baseline | First get the CDNA3 FlyDSL megakernel running | Slower than Triton, but proved the fusion boundary is viable |
| LDS fit | Remove `lds_k_row`, strengthen lifetime alias | Fit into 64KB LDS first, then there is room for subsequent optimization |
| Producer correctness | Introduce side-effect barrier anchor | Fix the correctness issue of producer overwriting LDS across iterations |
| Gate staging | `g/beta` moved into LDS | Reduce redundant GMEM access, need to watch LDS pressure |
| `BLOCK_DV=64` | Hot path switched from BDV32 to BDV64 | Largest single-point gain, avoids repeated Q/K/A/g/beta staging |
| Scheduling overlap | O direct GMEM store position moved later | Better suited for MI308 than O-through-LDS |
| Post-V47 cleanup | Multiple stride/beta/layout cleanup attempts | Static resources seemed better but rocprof/PMC worsened, all rejected |
| SG-V5 | Small-H BDV32 fast path | Fixed `(2,8)` / `(8,16)` by increasing grid parallelism |

## Tuning Workflow

### 1. Establish a Fair Baseline

1. Clarify the fusion boundary, e.g., whether `a/g_cumsum` is included.
2. Baseline uses the current tuned Triton/Gluon comparison data, not PyTorch.
3. Create kernel trace mapping for multiple dispatches of the split baseline, and explain the sum rules.
4. Correctness can use PyTorch or tuned baseline, but performance must only use `rocprofv3`.

### 2. First Check Resource Feasibility

The first gate for MI308/CDNA3 is LDS:

| Check Item | Stop Condition |
|---|---|
| LDS | Must be less than 64KB, with some headroom |
| scratch | Must be 0 |
| VGPR/accum VGPR | Must not significantly increase due to small changes, affecting occupancy |
| barrier | ISA must preserve the expected `s_barrier` |
| bank conflict | `SQ_LDS_BANK_CONFLICT` must not worsen due to padding/stride changes |

### 3. Then Design Wave Roles

Do not pursue a full Hopper pipeline from the beginning. Recommended order:

1. First make the compute-only megakernel correct;
2. Add minimal producer staging;
3. For each cross-iteration producer action added, perform correctness + ISA barrier checks;
4. Then use rocprofv3 to see if there are actual overlap benefits.

### 4. Perform Shape/Grid Pre-check

`BLOCK_DV` is not always better when larger:

| Shape | Recommendation | Reason |
|---|---|---|
| `(8,32,128,128)` | BDV64 | Reduces redundant staging and redundant P/Ag work |
| `(8,16,128,128)` | BDV32 | BDV64 grid only has 32 CTAs, CU underfill |
| `(2,8,128,128)` | BDV32 | BDV64 grid only has 16 CTAs, fixed cost too high |When optimizing similar kernels, the reference workflow should first calculate grid blocks / CU. In low-grid scenarios, increasing parallel partitioning usually takes priority over ISA micro-tuning.

### 5. Accepting or Rejecting a Version

Accepting a version requires consistency across four types of evidence:

| Evidence | Description |
|---|---|
| correctness | output/final state/cache state all pass |
| rocprofv3 | multi-T/shape sweep shows improvement, not just a single point |
| ISA resource | no hidden regressions in LDS/VGPR/scratch/barrier |
| counters | LDS bank conflict, VMEM/TCP, waitcnt direction can explain the gains |

If static resource usage decreases but rocprofv3 performance degrades, prioritize rocprofv3 and counters.

## Rejected Patterns

| Attempt | Why It Failed | Design Rule |
|---|---|---|
| Directly porting Hopper double-buffer | LDS exceeds 64KB or bank conflict explodes | Fit LDS first, then consider specialization |
| Producer-side gate precompute | SFU decreases but LDS/VMEM pressure increases | Don't just count `v_exp_f32`; must examine global counters |
| O-through-LDS storer | Resembles FlashQLA, but increases LDS traffic and bank conflict | O direct store is more reasonable on MI308 |
| Overly aggressive next-chunk prefetch | Correction cost and dependency complexity eat up the gains | Recurrent kernels must model dependency cost |
| `STRIDE_*` resource cleanup | LDS/VGPR decrease, but bank conflict or VMEM/TCP worsen | Padding requires PMC validation, not intuition |
| beta direct GMEM | LDS decreases, but repeated VMEM increases | Staging vs. direct load must be decided by T sweep |
| Single BDV64 covering all shapes | Grid is too small for small H | Tile shape must be shape-aware |

## Relationship with Other AMD Documents

- For Tile/Grid decisions, refer to `common/roofline-analysis-methodology.md` and
  `common/small-matrix-cu-utilization.md`.
- For LDS/bank conflict analysis, refer to `common/lds-bank-conflict-optimization.md`.
- For scratch/VGPR analysis, refer to `common/scratch-elimination-vgpr-spill.md`.
- For CDNA4 Chunk-GDN Gluon experience, see `../../gluon/gfx950/chunk_gdn_lessons.md`;
  CDNA4 experience can inform diagnostics but cannot replace MI308 rocprofv3 conclusions.

## Path Index

| Content | Path |
|---|---|
| gpu-wiki FlyDSL reference | `reference-kernels/amd/cdna3/flydsl/FlyDSL/` |
| gpu-wiki Triton comparison baseline | `reference-kernels/amd/cdna/triton/chunk_gdn/` |
| MI308X pitfalls | `docs/amd/cdna3/mi308x/pitfalls/flydsl/chunk-gdn-mi308x-wave-specialization-pitfalls.md` |
| standalone profile driver | `kernel_opt_chunk_gdn_gpu_wiki/rocprof_397b_tp2/profile_chunk_gdn_397b_tp2.py` |
| standalone rocprof CSV | `kernel_opt_chunk_gdn_gpu_wiki/rocprof_397b_tp2/out_*_p50/` |
| RTP checkpoint | `RTP-LLM/github-opensource/optimization_checkpoint.md` |
| original workspace handoff | `wenhua_code/flydsl/chunk_gdn_flydsl_workspace/megakernel/shape_generalization_next_handoff.md` |

## Open Issues

- tiny suffix `input_len=1/17` still requires separate optimization and profiling;
- 0.8B/2B shapes are outside the scope of this round of strategy;
- `(16,48)/(16,64)` Triton long normal `input_len=200000` memory fault requires independent investigation;
- MI355/CDNA4 requires re-doing LDS, tile, CU, ISAoro, and rocprofv3 sweeps; MI308 performance conclusions cannot be reused.
