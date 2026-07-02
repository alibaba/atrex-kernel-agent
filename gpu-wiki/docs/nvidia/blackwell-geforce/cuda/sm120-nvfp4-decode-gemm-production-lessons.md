# SM120 NVFP4 Decode GEMM Production Lessons

This report summarizes the reusable SM120 NVFP4 decode and prefill GEMM

**Last updated**: 2026-06-30

knowledge distilled from the omoExplore wiki. It intentionally removes
experiment numbering and keeps only the stable engineering conclusions.

## Scope

- Hardware: NVIDIA RTX PRO 5000 / 4000 Blackwell-GeForce, `sm_120` /
  `compute_120a`.
- Workload family: NVFP4 MLP GEMM/GEMV with BF16 output in vLLM serving.
- Main instruction: `mma.sync.aligned.kind::mxf4nvf4...m16n8k64`.
- Framework surface: FlashInfer / CUTLASS NVFP4, vLLM CUDA Graph replay, and
  CUDA C++ inline-PTX reference kernels.

## Shape Taxonomy

| Shape family | Typical dimensions | Production lesson |
|---|---:|---|
| C1 decode | `M=1,N=17408,K=5120` | Keep stock CUTLASS / FlashInfer. N-direction CTA count is already enough. |
| C2 decode | `M=1,N=5120,K=8704` or `K=17408` | Split-K can help when the base grid under-fills SMs, but must beat b12x under CUDA Graph nsys, not only in standalone timing. |
| Gate/up decode | `M=1..8,N=34816,K=5120` | For cold weight streaming, CTA-N shrinkage can increase block residency and DRAM request count. |
| Prefill GEMM | large runtime M | Treat as a separate GEMM regime. Symbolic-M routing and small-M decode conclusions do not transfer automatically. |
| Linear projection decode | `M=1..16,N=16384,K=5120` | If the comparator is already near roofline, shallow split/stage/tactic tuning is not enough. |

## Split-K Dispatch Rule

Split-K is a parallelism fix for shapes with too few output CTAs. For C2-like
decode, `TILE_N=32` gives only `5120/32 = 160` CTAs, or about `1.45 CTA/SM`
on a 110-SM GPU. With `S=4`, the grid becomes 640 CTAs, enough to expose more
memory-level parallelism.

The durable dispatch rule is shape-based:

```text
Use Split-K only when:
  M == 1
  N is small enough to under-fill the GPU
  K/S satisfies the kernel alignment contract
  workspace lifetime is stable across CUDA Graph capture/replay

Use CUTLASS / FlashInfer otherwise:
  C1-like large-N decode
  batched decode
  prefill
  any shape whose base grid is already saturated
```

For the archived CUDA reference kernel, `K/S % 128 == 0` is a hard predicate
because each local loop handles two `m16n8k64` atoms.

## CUDA Graph Methodology

CUDA Graph replay removes Python dispatch, C++ wrapper overhead, CPU-side
kernel launch cost, and CPU-side descriptor creation. It does not remove GPU
kernel execution, TMA traffic, tile scheduling, workspace reads/writes, or
inter-kernel dependencies.

Therefore:

- A standalone or tight-loop win is a screening signal only.
- Operator performance claims require CUDA Graph nsys evidence from the actual
  vLLM decode path.
- Two-stage Split-K must report partial kernel time, reduce kernel time, and
  `partial + reduce` total time.
- Served claims require prompt exactness, semantic smoke, dispatch-hit logs,
  same-session paired A/B, and GPU idle snapshots.

This rule explains why an older C2-only Split-K hybrid looked good in a
standalone loop but lost to full FlashInfer b12x inside served CUDA Graph
decode: the real graph paid the partial kernel, reduce kernel, workspace
traffic, and dependency ordering.

## CTA-3D TMA Improvement

The best scoped C2-only follow-up changed the B-load structure rather than only
tuning split count or tile width. It encoded split as a third tensor-map
dimension:

```text
global_dims   = [K_split / 2, N, S]
global_strides = [K / 2, K_split / 2]
box           = [64 bytes, 8 N rows, S]
```

This turns one small B TMA issue per split into one CTA-level 3D TMA issue per
K block, while preserving `S=8,tile_n=8` compute parallelism. In CUDA Graph
nsys, this path measured about `43.8 us/call` against a same-run b12x C2
estimate around `50.3 us/call`, roughly `1.15x` on the C2 operator. Served
TP=1 C2-only E2E was slightly positive, but the claim remains scoped to that
route until TP=2 production evidence exists.

## Cold Weight-Streaming GEMV Residency Lever

For a latency-bound, cold-cache, weight-streaming GEMV with visible bandwidth
headroom, the useful lever is the number of resident blocks per SM, not deeper
per-block prefetching.

The observed sequence under fixed prefetch depth was:

| CTA_N | Resident blocks / SM | DRAM throughput |
|---:|---:|---:|
| 128 | 2 | 54.57% |
| 64 | 5 | 72.99% |
| 32 | 10 | 89.51% |

Shrinking `CTA_N` reduces per-block shared-memory footprint, increases
resident blocks, and raises the number of in-flight DRAM requests. Deepening
the TMA prefetch ring can reduce the latency seen by a single warp, but it also
uses more shared memory and can lower residency. In this regime, deeper
prefetching was an anti-lever.

The boundary is important: this lever works only when current bandwidth is far
below the competitor or roofline. If the comparator is already at the
structural DRAM bandwidth ceiling, local knobs cannot close the gap.

## Structural DRAM-Bandwidth Ceiling

For `M=1..16,N=16384,K=5120` NVFP4 decode GEMM, the local ATREX-style
multi-wave split-K route could not reliably beat a FlashInfer/CUTLASS b12x
route already near the memory roofline. A-staging improved some M points, but
the strict all-shape acceptance target was unmet. Naive cross-tile persistent
pipelining regressed because per-tile synchronization/drain cost outweighed
prefetch benefit.

The reusable lesson is:

- If the comparator is a persistent, single-wave, high-bandwidth route, shallow
  tuning inside a multi-wave split-K structure has a hard ceiling.
- A true rewrite must change launch structure or cross-tile dataflow. It should
  be justified only when the remaining E2E headroom is large enough.
- Median-of-N or paired statistics are required when per-M tactic variance is
  around the same size as the claimed win.

## Prefill and Fusion Boundary

For large-M prefill, forced b12x tactics on gate/up were too small and
inconsistent to justify production routing. Some short-M buckets saved only a
few microseconds per layer, while a medium-M bucket regressed. Because vLLM
compile mode could not safely exact-branch on symbolic runtime `M`, a broad
`N=34816,K=5120` route would incorrectly affect slower buckets.

The stronger prefill direction is a Dense MLP fusion boundary:

```text
gate_up GEMM -> SiLU*Mul -> NVFP4 payload/scale for down_proj
```

The value is not a different GEMM tactic; it is avoiding materialization and
later reread of the large BF16 `gate_up` intermediate. Any implementation must
preserve the optimized SM120 producer path and prove payload/scale correctness
before served TTFT.

## Reference Kernels

- CUDA Split-K decode GEMV:
  [reference-kernels/nvidia/blackwell-geforce/cuda/nvfp4_splitk_gemv/](../../../../reference-kernels/nvidia/blackwell-geforce/cuda/nvfp4_splitk_gemv/)
- Split-K dispatch wrapper:
  [cutlass_splitk_dispatch_sm120.py](../../../../reference-kernels/nvidia/blackwell-geforce/cuda/nvfp4_splitk_gemv/cutlass_splitk_dispatch_sm120.py)
- CUDA phase-1/phase-2 kernel:
  [gemm_v3_splitk_sm120.cu](../../../../reference-kernels/nvidia/blackwell-geforce/cuda/nvfp4_splitk_gemv/gemm_v3_splitk_sm120.cu)
- SM120 CuTeDSL persistent NVFP4 GEMM, useful as a structural comparator:
  [sm120_nvfp4_persistent_gemm_pro5000.py](../../../../reference-kernels/nvidia/blackwell-geforce/cutedsl/cutlass/sm120_nvfp4_persistent_gemm_pro5000.py)

## Sustained Checklist

1. Classify the shape before choosing a route.
2. Compute output CTA count and CTA/SM before adding Split-K.
3. Prefer CUTLASS / FlashInfer for already saturated shapes.
4. Keep workspace allocation outside the hot path and graph capture.
5. Record the scale-factor layout and binary identity for every kernel swap.
6. Use CUDA Graph nsys before making operator performance claims.
7. Use paired served E2E before making production claims.


## Related

- [SM120 CUDA NVFP4 Split-K GEMV (BF16 out) on RTX PRO 5000](sm120-nvfp4-split-k-gemv-bf16-optimization.md)
- [SM120 RMSNorm-MLP NVFP4 Fusion and PDL Handoff Report](sm120-rmsnorm-mlp-pdl-fusion-report.md)
- [CUTLASS GEMM Optimization Strategy](../../common/cutedsl/cutlass-gemm-optimization.md)
- [Community GEMM Optimization Practical Summary](../../../generic/gemm-optimization-guide.md)
- [Composable Kernel (CK) Architecture Overview](../../../amd/common/ck-architecture-overview.md)
