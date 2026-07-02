# SM120 CUDA NVFP4 Split-K GEMV (BF16 out) on RTX PRO 5000

> Path-of-record for the CUDA Split-K decode GEMV route used by the

**Last updated**: 2026-06-30

> Qwen3.5 NVFP4 MLP C2 shapes. This is a CUDA C++ / inline-PTX archive,
> not a CuTeDSL optimization.

---

## Target hardware

- GPU: NVIDIA RTX PRO 5000 Blackwell-GeForce / `sm_120`, 110 SMs.
- Kernel route: CUDA C++ plus inline PTX
  `mma.sync.aligned.kind::mxf4nvf4.m16n8k64`.
- Workload: decode-only MLP NVFP4 GEMV (`M=1`) with CUTLASS-swizzled block-scale
  inputs and BF16 output.

## Algorithm baseline

Baseline C2 decode GEMV has too few output tiles:

```text
base_cta = ceil(M / 1) * ceil(N / TILE_N)
TILE_N = 32
C2 TP=2: N=5120  -> 160 CTA  -> 1.45 CTA/SM on 110 SMs
```

Split-K adds K-dimensional parallelism:

```text
S = 4
K_split = K / S
grid = (ceil(N / TILE_N), S)
C2 TP=2: 160 CTA -> 640 CTA -> 5.8 CTA/SM
```

The implementation is intentionally two phase:

1. `decode_gemv_nvfp4_splitk_kernel<TILE_N>` accumulates each K slice into FP32
   workspace `(S, N)`.
2. `reduce_splitk_kernel` reduces S partial sums and writes BF16 output.

C1-like large-N shapes are not the target. For `N=17408,K=5120,TILE_N=32`,
the ordinary grid already has 544 CTAs; forcing `S=4` would create 2176 CTAs
and pay reduction/workspace overhead without solving an under-occupancy problem.

## Kernel resource footprint

| Field | Value |
|---|---|
| Main kernel | `decode_gemv_nvfp4_splitk_kernel<TILE_N>` |
| Reduce kernel | `reduce_splitk_kernel` |
| Default TILE_N | 32 |
| Threads / CTA | `TILE_N * 4 = 128` |
| K atom | `m16n8k64`; loop handles two K chunks per iteration |
| Split count | `S=4` in accepted integration |
| K alignment | `K_split % 128 == 0` |
| Workspace | `(S, N)` FP32, preallocated and reused |
| C2 TP=2 workspace | `4 * 5120 * 4 = 80 KB` |
| C2 TP=2 dynamic smem | `align16(K_split/2) + TILE_N*64 = 3136 B` |

Implementation:
[reference-kernels/nvidia/blackwell-geforce/cuda/nvfp4_splitk_gemv/](../../../../reference-kernels/nvidia/blackwell-geforce/cuda/nvfp4_splitk_gemv/)

## Optimization journey

### v0 — noSplitK decode GEMV

The v3 decode GEMV path was exact but C2 remained under-filled. C2 had only
160 CTAs for 110 SMs, so wall time was dominated by insufficient wave parallelism
rather than tensor-core compute.

### v1 — Split-K two-kernel implementation

Split K into `S` slices and write FP32 partial sums to workspace. Correctness was
checked against the v3/CUTLASS reference across C1/C2 and `S in {1,2,4}` where
alignment allowed. C2 `S=2` was skipped in the TP=2 shape because `K_split=4352`
is not divisible by 128 for that path; `S=4` is aligned.

Standalone C2 cold-cache numbers improved from about `73.8 us` to about
`35.8 us`; CUDA Graph GPU timing improved from about `41.9 us` to about
`26.8 us`.

### v2 — Dispatch only C2-like shapes

The durable dispatch is **C1 CUTLASS + C2 Split-K**, not full replacement:

- C2-like: `M=1`, `N <= 8192`, `K/S % 128 == 0` -> Split-K.
- C1-like / prefill / batched GEMM -> stock CUTLASS.

This avoids C1 overhead and keeps the vLLM graph structure identical for the
large-N route.

### v3 — Layout and prompt gates

The accepted binary consumes CUTLASS-swizzled `SF_B` directly. The known-good
`gemm_v3_splitk.so` md5 in the source project was
`bc710eae90003480df15a01d81a09856`. Old CUTLASS-to-ATREX scale conversion was
invalid for this binary and was removed from the current dispatch path.

Prompt correctness became a hard gate: low-entropy `gate_v2` exact prompts must
pass before open-ended semantic smoke or performance measurement.

## Final perf vs baseline

| Scope | Baseline | C1 CUTLASS + C2 Split-K | Delta | Notes |
|---|---:|---:|---:|---|
| TP=2 task_27 paired E2E | `16.1652 ms/token` | `15.8394 ms/token` | `-2.02%` | `p < 0.000001`, CG enabled |
| TP=2 task_32 accepted E2E | `16.720385 ms/token` | `15.866395 ms/token` | `-5.107%` | `gate_v2` exact + semantic smoke pass |
| TP=1 same-parameter served | `27.773214 ms/token` | `27.476403 ms/token` | `-1.069%` | fifth split-K set was noisy; weaker evidence |
| TP=1 operator C2 warm | `95.872 us` | `31.840 us` | `3.01x` | exact=true, cosine=1.0 |

The final decision is based on paired E2E and dispatch evidence, not standalone
cold-cache timing alone.

## Remaining bottlenecks

The archived evidence identifies Split-K as a parallelism fix: C2 has only
`160` output CTAs before splitting. Once Split-K reaches `640` CTAs, remaining
E2E gain is capped because C2 is only part of the decode token budget. Future
work should collect a fresh `ncu --launch-count 1` pair for the current binary
before making further kernel-level claims.

Two practical ceilings were observed:

- C1 should stay on CUTLASS; it already has enough output-tile parallelism.
- C2 Split-K improves the kernel substantially, but E2E gain converges around
  low single digits unless more of the decode stack is changed.

## What would close the remaining gap

- Use a library path with proven split-K / Stream-K scheduling if it can consume
  the same NVFP4 scale layout and preserve prompt exactness.
- Keep workspace persistent and CUDA-Graph-compatible; eliminating allocation
  jitter matters more than shaving a few instructions from the reduce kernel.
- Re-profile phase-1 vs phase-2 time. If reduction becomes visible, compare
  deterministic two-kernel reduce with a controlled atomic route.

## Sustained recipe

1. Prove grid underfill first: `base_cta / SM` must be low.
2. Choose `S` from CTA target, then reduce it by `K/S` alignment and workspace cost.
3. Implement two-stage FP32 partial + reduce first; only consider atomics after
   correctness and determinism are understood.
4. Gate dispatch to C2-like shapes; keep C1 and prefill on CUTLASS.
5. Validate scale-factor layout against the exact binary md5.
6. Run prompt exactness before performance.
7. Use same-session paired E2E with GPU idle snapshots and dispatch-hit logs.

## Related docs

- Code: [reference-kernels/nvidia/blackwell-geforce/cuda/nvfp4_splitk_gemv/](../../../../reference-kernels/nvidia/blackwell-geforce/cuda/nvfp4_splitk_gemv/)
- Pitfalls: [nvfp4-split-k-gemv-pitfalls.md](pitfalls/nvfp4-split-k-gemv-pitfalls.md)
- CUTLASS scheduling background: [cutlass-tile-scheduling.md](../../common/cutedsl/cutlass-tile-scheduling.md)
- NCU estimate meta-rule: [ncu-rule-est-speedup-meta-rules.md](../../common/profiling/ncu-rule-est-speedup-meta-rules.md)
