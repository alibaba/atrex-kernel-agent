# NVFP4 Split-K GEMV Pitfalls

Specific to the CUDA C++ NVFP4 decode GEMV Split-K route on `sm_120`.

Related:

- Optimization report:
  [sm120-nvfp4-split-k-gemv-bf16-optimization.md](../../ref-docs/cuda/sm120-nvfp4-split-k-gemv-bf16-optimization.md)
- Code:
  [reference-kernels/nvidia/blackwell-geforce/cuda/nvfp4_splitk_gemv/](../../../../../reference-kernels/nvidia/blackwell-geforce/cuda/nvfp4_splitk_gemv/)

## 1. Treating Split-K as an all-shape replacement

**Trap**: If Split-K helps C2, route every M=1 NVFP4 decode GEMV through it.

**Result**: C1-like large-N shapes already have enough output-tile parallelism.
For `N=17408,TILE_N=32`, the noSplitK grid has 544 CTAs. `S=4` expands that
to 2176 CTAs and adds workspace/reduce overhead.

**Why**: Split-K fixes under-occupancy from too few `M*N` tiles. If `M*N`
already fills the GPU, K-splitting mostly adds traffic and synchronization.

**Lesson**: Dispatch by shape. C2-like small-N / long-K shapes can use Split-K;
C1-like large-N and prefill should stay on CUTLASS.

## 2. Accepting cold standalone speedup as the final answer

**Trap**: Standalone C2 drops from about `73.8 us` to `35.8 us`, so the E2E
service win must be similarly large.

**Result**: Paired E2E settled lower: task_27 measured `-2.02%`, task_32 accepted
run measured `-5.107%`, and TP=1 same-parameter served evidence was `-1.069%`
with one noisy split-K set.

**Why**: CUDA Graph replay, dispatch overhead, C1 fallback, prompt correctness
gates, and the share of C2 in total TPOT all shrink the standalone delta.

**Lesson**: Use standalone only for diagnosis. Accept with same-session paired
E2E, dispatch-hit evidence, and GPU idle snapshots.

## 3. Ignoring K/S alignment

**Trap**: Sweep any `S` and pick the fastest.

**Result**: Some split counts are invalid or silently require a different kernel
contract. The archived kernel requires `K_split % 128 == 0` because the loop
handles two `m16n8k64` chunks per iteration.

**Why**: NVFP4 MMA, block-scale indexing, and the B tile load pattern all assume
aligned K chunks.

**Lesson**: Treat `K/S % 128 == 0` as a hard dispatch predicate for this kernel.

## 4. Reusing stale scale-factor layout conversion

**Trap**: Keep the old CUTLASS-to-ATREX scale conversion when swapping binaries.

**Result**: The accepted `gemm_v3_splitk.so` md5
`bc710eae90003480df15a01d81a09856` consumes CUTLASS-swizzled `SF_B` directly.
Old conversion logic is invalid for this binary.

**Why**: FP4 scale-factor layout is part of the ABI. It can change independently
from A/B packed data layout.

**Lesson**: Every binary change needs a layout sanity test and md5 record.

## 5. Allocating workspace in the hot path

**Trap**: Allocate `(S, N)` FP32 workspace inside every call for simplicity.

**Result**: Allocation jitter pollutes TPOT and can break CUDA Graph capture or
make replay results incomparable.

**Why**: Split-K adds a real workspace dependency. Its lifetime must be stable
across graph capture and replay.

**Lesson**: Cache workspace by `N` and device, and keep allocation outside the
measured hot path.

## 6. Wrapping C1 fallback inside a new custom op

**Trap**: Put both C1 and C2 behind one `torch.library.custom_op`, with C1
falling back to CUTLASS internally.

**Result**: The graph structure changes even when C1 computes with CUTLASS. In
autoregressive decode, tiny numerical or graph-boundary changes can amplify
into prompt mismatch.

**Why**: The "same kernel internally" is not the same graph as the stock vLLM
CUTLASS route. Compilation and fusion boundaries can differ.

**Lesson**: Keep C1 on the original CUTLASS path. Only intercept C2-like shapes.

## 7. Confusing Split-K with Stream-K

**Trap**: Stack Split-K on top of Stream-K or persistent tile scheduling because
all of them sound like scheduling improvements.

**Result**: They attack overlapping decomposition problems. Extra splitting can
turn into more reduction traffic and less predictable scheduling.

**Why**: Split-K and Stream-K both partition K work and require partial-result
coordination. They are usually alternatives, not additive knobs.

**Lesson**: Choose one work-decomposition strategy per GEMM, then validate with
profile and E2E.

## Quick table

| Use | Avoid |
|---|---|
| C2-like small-N / long-K decode GEMV | C1-like large-N GEMV |
| `K/S % 128 == 0` | arbitrary split counts |
| Preallocated `(S, N)` FP32 workspace | per-call workspace allocation |
| Same-session paired E2E | cold standalone as final proof |
| Layout sanity tied to binary md5 | stale scale conversion assumptions |
| C1 stock CUTLASS + C2 Split-K | custom-op wrapping every shape |
