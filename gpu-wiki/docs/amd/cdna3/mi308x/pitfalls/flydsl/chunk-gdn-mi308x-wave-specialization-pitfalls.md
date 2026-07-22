# Pitfalls: FlyDSL Chunk-GDN Wave-Specialized Megakernel on MI308X

Applicability: backend: flydsl; hardware: amd; topic: pitfalls

Companion optimization report:
[`cdna3-chunk-gdn-mi308x-wave-specialized-megakernel-optimization.md`](../../ref-docs/flydsl/cdna3-chunk-gdn-mi308x-wave-specialized-megakernel-optimization.md)

Reference implementation:
[`reference-kernels/amd/cdna3/flydsl/FlyDSL/`](../../../../../../reference-kernels/amd/cdna3/flydsl/FlyDSL/)

This document only records the most error-prone lessons learned when porting FlashQLA's warp-specialization approach to MI308X/gfx942.
For pitfalls of the 5-kernel Chunk-GDN pipeline on MI355X/gfx950, see
[`chunk-gdn-pitfalls.md`](../../../../cdna4/pitfalls/flydsl/chunk-gdn-pitfalls.md).

---

## 1. Mechanically Copying Hopper Warp-Specialization to MI308X

**Trap**: Seeing that FlashQLA uses WGMMA, TMA, mbarrier, and large SMEM to implement a megakernel on Hopper,
directly replicating double-buffer + producer/consumer pipelines on MI308X.

**Result**: LDS exceeds 64KB or bank conflicts explode; even if it barely fits, the producer executes ordinary
`buffer_load + ds_write` without the offloading effect of Hopper TMA.

**Why**: On MI308X/CDNA3, MFMA is a synchronous instruction, `s_barrier` is workgroup-level synchronization, LDS capacity is about
64KB, and VGPR cannot be budgeted differently per warpgroup. What is transferable is FlashQLA's fusion boundaries and state
residency, not Hopper's hardware mechanisms.

**Lesson**: First get the dataflow fusion of `recompute_w_u + fwd_h + fwd_o` working, then redesign the implementation using MI308X's LDS
capacity, barriers, and producer waves.

---

## 2. Writing Chip-Specific Tuning Code as a Generic gfx942 Implementation

**Trap**: Because gfx942 covers multiple CDNA3 models, writing MI308X-specific tuned code as a generic gfx942 implementation,
or keeping historical filenames from non-target chips.

**Result**: Readers will misinterpret MI308X-specific conclusions (80 CUs, ~64KB LDS, 397B-TP2 profile) as
generic gfx942 architecture conclusions, and won't know whether the file can safely cover a generic CDNA baseline.

**Why**: gpu-wiki's archival rules require not overwriting generic architecture baselines. Hardware-specific tuning implementations should be placed under
arch-specific paths like `amd/cdna3/...`, with the target chip clearly indicated in the filename, docstring, and README.

**Lesson**: This implementation is uniformly labeled as MI308X-specific; filenames use `mi308x`, and performance conclusions are only stated as validated on MI308X/gfx942.

---

## 3. Using PyTorch or Wall-Clock Time as Performance Baseline

**Trap**: For the convenience of standalone operators, using PyTorch references, manual timing, `torch.cuda.Event`,
or `do_bench` to assess gains.

**Result**: The performance boundary is unfair, easily miscounting input construction, front half, synchronization, or benchmark overhead into
kernel gains.

**Why**: The fair boundary for this megakernel is the back half after
`a/g_cumsum` has been precomputed. Triton is only the same-boundary comparison
baseline here: `recompute_w_u_fwd + fwd_h + fwd_o`. Performance conclusions can only be read from
the kernel dispatch duration of `rocprofv3 --kernel-trace`.

**Lesson**: PyTorch is only used for correctness. The performance table must clearly list which dispatches from the rocprofv3 kernel trace are summed.

---

## 4. Using `BLOCK_DV=64` Unconditionally for All Shapes

**Trap**: On the hot path, `(8,32,128,128)` with BDV64 is significantly faster, so all Qwen3.5/Qwen3.6 runtime
shapes are routed to the same BDV64 kernel.

**Result**: The grid for `(2,8,128,128)` and `(8,16,128,128)` has only 16/32 CTAs, leaving MI308X's 80 CUs
underfilled, and long sequences may also lose to Triton.

**Why**: BDV64 reduces duplication in Q/K/A/g/beta staging and P/Ag work, but also reduces V-axis
parallelism. Small H shapes need more CTAs, not the largest tile.

**Lesson**: The hot `(8,32)` retains BDV64; small H `(2,8)` / `(8,16)` use a BDV32 fast path.
For every new shape, first calculate grid blocks / CU, then run correctness + rocprofv3 sweep.

---

## 5. Accepting Optimizations Based Only on Static LDS/VGPR Reduction

**Trap**: Seeing that changes like stride adjustments, beta direct load, or producer-side precomputation reduce LDS or
VGPR, and assuming they must be faster.

**Result**: Multiple post-V47 cleanup attempts had better static resource usage, but `SQ_LDS_BANK_CONFLICT`, VMEM/TCP, or wait
deteriorated, making large-T rocprofv3 slower.

**Why**: On MI308X, this kernel is primarily constrained by LDS traffic, bank conflicts, barriers, recurrent
dependencies, and the one-CTA-per-CU constraint. Static resources are only a necessary condition, not an acceptance criterion.

**Lesson**: An accepted version must simultaneously satisfy correctness, rocprofv3 sweep, ISA resource usage, and counter
consistency. When static resources decrease but rocprofv3 becomes slower, rocprofv3 and counters take precedence.

## 6. O-through-LDS Storer Looks More Like FlashQLA but Is Slower

**Trap**: In order to get closer to Hopper FlashQLA, the O output is also passed through the LDS storer, letting the producer/storer wave handle the write-out.

**Result**: LDS traffic and bank conflicts increase, making it worse than direct O GMEM store on MI308X.

**Why**: Hopper's storer relies on the TMA/warpgroup pipeline and finer-grained synchronization. The MI308X producer wave has no equivalent hardware offload, and O-through-LDS will squeeze the already constrained 64KB LDS and bank bandwidth.

**Lesson**: MI308X V47 chooses direct GMEM store, only moving the store position to a barrier window that better facilitates overlap.

---

## Quick Judgment Table

| Idea | MI308X Verdict |
|---|---|
| Replicate Hopper TMA/mbarrier pipeline | Don't; only migrate data flow and role concepts |
| Use PyTorch as performance baseline | Don't; only use same-boundary tuned Triton/Gluon comparison data + rocprofv3 |
| BDV64 covers all shapes | Don't; use BDV64 for hot H=32, BDV32 for small H |
| Static LDS/VGPR reduction is acceptable | Don't; must check rocprofv3 + ISA + counters |
| O-through-LDS looks more like FlashQLA | Not suitable for MI308X; direct GMEM store is better |
