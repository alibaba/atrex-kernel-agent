# SM120 NVFP4 Decode GEMM Production Pitfalls

Pitfalls distilled from omoExplore's SM120 NVFP4 decode and prefill GEMM work.
The entries are written as reusable knowledge, without experiment numbering.

Related:

- Optimization report:
  [sm120-nvfp4-decode-gemm-production-lessons.md](../../../ref-docs/nvidia/cuda/sm120/sm120-nvfp4-decode-gemm-production-lessons.md)
- CUDA Split-K reference:
  [reference-kernels/nvidia/blackwell-geforce/cuda/nvfp4_splitk_gemv/](../../../../reference-kernels/nvidia/blackwell-geforce/cuda/nvfp4_splitk_gemv/)

## 1. Treating Split-K as a universal decode route

**Trap**: Once C2-like decode wins with Split-K, route every `M=1` NVFP4 GEMV
through Split-K.

**Result**: C1-like large-N shapes already have enough N-direction CTAs. Extra
K-splitting only adds workspace writes, reduce work, and dependency edges.

**Why**: Split-K solves low `M*N` tile count, not low arithmetic intensity by
itself. When the original grid already fills the GPU, more CTAs are overhead.

**Lesson**: Dispatch by shape. Split-K is for small-N / long-K under-filled
decode. Keep C1, batched decode, and prefill on CUTLASS or FlashInfer unless a
fresh same-shape gate proves otherwise.

## 2. Using standalone timing as production proof

**Trap**: A kernel is much faster in a tight loop, so served throughput should
improve by the same ratio.

**Result**: A C2-only hybrid that looked favorable in standalone timing lost to
full b12x inside real served CUDA Graph decode.

**Why**: CUDA Graph replay removes host overhead but keeps GPU node execution,
workspace traffic, dependency ordering, and reduce kernels. The real cost is
the graph node total, not the isolated partial phase.

**Lesson**: Operator claims need CUDA Graph nsys from the actual vLLM decode
path. For two-stage Split-K, compare `partial + reduce`, not partial alone.

## 3. Keeping stale scale-factor layout conversion

**Trap**: Reuse the old CUTLASS-to-ATREX scale conversion after swapping the
NVFP4 GEMV binary.

**Result**: The accepted CUDA Split-K binary consumed CUTLASS-swizzled `SF_B`
directly, so the older conversion became wrong even though A/B packed data
layout looked unchanged.

**Why**: FP4 scale-factor layout is part of the kernel ABI. It can change
independently from payload layout.

**Lesson**: Every binary swap needs a layout sanity test and a recorded kernel
identity. Do not assume scale-layout compatibility from matching tensor shapes.

## 4. Wrapping all shapes in one new custom op

**Trap**: Put C1 fallback and C2 Split-K behind the same new
`torch.library.custom_op`, with C1 internally calling CUTLASS.

**Result**: Even when C1 math uses CUTLASS, the graph boundary and compilation
surface differ from stock vLLM, which can perturb prompt exactness and make E2E
debugging ambiguous.

**Why**: "Same backend inside" is not the same CUDA Graph or compiler boundary.

**Lesson**: Keep C1 on the original vLLM CUTLASS path. Intercept only the
specific C2-like shapes that need Split-K.

## 5. Ignoring b12x as the served comparator

**Trap**: Compare a custom Split-K route only against stock CUTLASS after
FlashInfer b12x becomes available.

**Result**: The custom route can beat stock CUTLASS and still lose to b12x in
served CUDA Graph decode.

**Why**: The production comparator is the fastest correctness-clean route in
the same environment, not the historical baseline.

**Lesson**: Re-baseline against current FlashInfer/CUTLASS autotune and b12x
under the same CUDA Graph methodology before claiming a new route is useful.

## 6. Forcing b12x tactics without runtime-M safety

**Trap**: A forced b12x tactic gives a small win for one short-M bucket, so
route all matching `N,K` gate/up shapes through it.

**Result**: Other M buckets regress, and vLLM compile mode may not safely
branch on exact symbolic runtime `M`.

**Why**: An N/K-only guard is too broad for a tactic whose benefit depends on
runtime M.

**Lesson**: If the router cannot express the exact safe predicate, do not
promote the tactic. Tiny per-layer operator savings are not worth a broad
served-risk surface.

## 7. Fighting a structural DRAM roofline with local knobs

**Trap**: Continue tuning split count, stage depth, prefetch flags, or N-tile
width after the comparator is already near DRAM roofline.

**Result**: A multi-wave split-K structure may improve some points but fail the
strict all-shape gate against a persistent, high-bandwidth comparator.

**Why**: Launch structure and memory-level parallelism set the ceiling. Local
knobs cannot overcome a structurally better persistent route once bandwidth is
saturated.

**Lesson**: When roofline headroom is gone, stop shallow tuning. Either change
the launch/dataflow structure or bank the result as diagnostic-only.

## 8. Deepening prefetch when shared memory is the residency limiter

**Trap**: Long scoreboard is high, so increase the TMA prefetch depth.

**Result**: Single-warp latency improves but resident blocks per SM drop,
reducing total in-flight DRAM requests and lowering bandwidth.

**Why**: In a cold weight-streaming GEMV, shared memory can be the binding
resource. More stages consume shared memory and reduce block residency.

**Lesson**: If the kernel is latency-bound with bandwidth headroom, first shrink
the weight tile or shared-memory footprint to raise resident blocks per SM.
Treat prefetch depth as secondary and verify it does not lower residency.

## 9. Treating scalar scale-factor LDS conflicts as a shallow layout issue

**Trap**: A b12x/CuTe fork shows shared-memory conflicts, so try shallow
copy-order, padding, swizzle, or TV-layout changes.

**Result**: Correctness-clean variants keep the same excessive scalar `LDS`
wavefront count, while padding/swizzle attempts either misalign, fail CuTe copy
layout checks, or break exactness.

**Why**: The conflict is in scalar 32-bit SFA/SFB `LDS` feeding `OMMA.SF`, not
primarily in A/B TMA, A/B `LDSM`, `LDS.128`, or epilogue stores.

**Lesson**: Do not integrate shallow SF-layout variants. A real fix needs deeper
SF staging that preserves the exact `OMMA.SF` fragment contract, or a different
fusion boundary.

## 10. Promoting operator-only cold-cache evidence

**Trap**: Cold-cache NCU shows a strong operator result, so call it a production
win.

**Result**: The result may be valuable but still says nothing about served TPOT,
prompt exactness, CUDA Graph replay, or contention with surrounding kernels.

**Why**: Cold-cache operator measurements isolate one mechanism. Production
serving includes graph replay, routing, neighboring kernels, and prompt output.

**Lesson**: Label cold-cache operator evidence as operator-only. Production
claims need the served gate.

## Quick Table

| Use | Avoid |
|---|---|
| Shape-scoped Split-K | Split-K as an all-shape replacement |
| CUDA Graph nsys for operator claims | tight-loop timing as final proof |
| Current b12x / CUTLASS comparator | historical baseline only |
| Exact scale-layout sanity per binary | stale scale conversion |
| CTA_N shrinkage when residency is binding | deeper prefetch that lowers residency |
| Fusion boundary when HBM traffic dominates | shallow tactic forcing with unsafe predicates |
