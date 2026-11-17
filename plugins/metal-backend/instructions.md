# metal-backend plugin

A local Apple Silicon execution backend for AKA campaigns. It compiles a Metal
kernel candidate, checks correctness against the operator's own CPU reference,
and produces ABBA-alternated measurement evidence — entirely on the local Mac,
with no Agate gateway and no SSH GPU allocation.

## Why this exists

AKA's Execution layer assumes remotely provisioned NVIDIA/AMD/PPU GPUs. On a
Mac, the "GPU backend" is the machine itself: unified memory, `xcrun metal`
toolchain, and Metal command-buffer GPU timing for measurement. This plugin is
the minimal bridge: it follows AKA's evidence rules (ABBA alternation, medians,
no tool-side promotion) while owning a Metal-specific build-and-measure path.

## Tool: `metal-eval`

Input: an operator workspace that already contains

- a `.metal` source with both the baseline and candidate kernel entry points,
- an Objective-C++ bench driver that loads a metallib from
  `ONYX_METALLIB_PATH`, accepts `--variant <name>`, prints a JSON result with
  `gpu_median_ms` / `correct` / `max_relative_error`, and exits `2` on numeric
  failure (the onyx-infer `tools/metal_bench.mm` contract),
- the bench arguments (shapes, iterations) for the campaign's contract shape.

Output: a measurement record with per-block medians, geomean inputs, device
identity, correctness, and a `latency_reduction_pct`. The record binds the
exact bench arguments; it does not promote. Promotion stays with the
Supervisor/workflow, matching AKA's authority boundaries.

Example request:

```json
{
  "operator_root": "/path/to/onyx-infer",
  "kernel_source": "ops/metal/q4_gemm.metal",
  "bench_source": "tools/metal_bench.mm",
  "baseline_variant": "tile16",
  "candidate_variant": "mma",
  "bench_args": ["--m", "1024", "--n", "5376", "--k", "5376", "--group", "64", "--iterations", "10"],
  "blocks": 4,
  "iterations": 10,
  "output": "runs/metal-eval-record.json"
}
```

Standalone bring-up (no AKA runtime required):

```sh
python3 plugins/metal-backend/adapter.py --input request.json
```

## Environment

- `ATREX_METAL_BIN` — optional absolute path to a Metal toolchain `bin/`
  directory. Useful when `xcrun --find metal` resolves to a workspace-local
  toolchain (as in some benchmark harnesses) rather than the default Xcode one.
- `xcrun clang++` must be available for the bench driver build.

## Metal knowledge (validated lessons, also seeded into gpu-wiki)

These are ABBA-measured on an Apple M5 Max across the onyx-infer
`kernel-agent/q4-mma` episodes E1-E8 (final: tile16 26.9 ms → mma64 8.4 ms
on the linear shape, 107.2 → 33.2 ms on qkv, 3.2x; decode-linear 0.473 ms
beats MLX 0.726 ms; correctness green throughout):

1. **simdgroup MMA with single-round fp16 dequant** — stage activations as
   fp16, unpack q4 nibbles, apply the fp32 group scale once, round to fp16,
   and feed 8×8 half fragments to `simdgroup_multiply_accumulate` (fp32
   accumulator). Scalar-FMA Metal GEMMs run at single-digit percent of MMA
   peak. Fragments are collective: never gate them on a thread condition.
2. **`simdgroup_load` `transpose=true` is unusable on metal4 toolchains** —
   its stride semantics do not match the row-major intuition and return
   partially-populated fragments. Verify any fragment-layout assumption with a
   10-line collective probe kernel before building on it.
3. **Fewer staging iterations beat coalesced staging** — a 32-iteration
   coalesced A-staging loop ran ~30 ms while an 8-iteration non-coalesced one
   ran ~13.3 ms on the same kernel: serialized staging dominated, and unified
   memory L2 absorbed the strided reads. Choose staging topology by ABBA
   measurement, not by CUDA intuition.
4. **Per-shape dispatch flips at small m** — the MMA kernel wins 3.2x at
   m=1024 but loses to the 16x16 scalar tile at m=16, where the scalar tile
   itself already beats MLX. Dispatch by shape regime, never one kernel for
   all shapes.
5. **Small-m scalar kernels are compute-bound near 8 TFLOPS** — thread-per-
   column GEMV (even split-k) cannot approach bandwidth; decode shapes that
   still trail MLX need an MMA-based split-k small-M kernel.
6. **Two driver pitfalls**: committing a Metal command buffer twice hangs
   the process permanently; `thread_elements()` is read-unreliable on the
   metal4 toolchain, so register-side fragment dequant cannot be verified
   cheaply — stage through threadgroup memory instead.

## Upstream integration status

This branch is local-first. What is intentionally NOT yet wired into the
framework, and what wiring it will require:

- `--platform`/`--framework` discovery: `aka/task/hardware.py` gains an
  `apple` vendor (platform/arch token match on `APPLE|MAC|M\d`) and
  `APPLE_FRAMEWORKS = ("Metal",)` so `--framework Metal` validates.
- The Light workflow's Evaluation currently plans Agate-shaped measurement
  requests. `metal-eval` can serve as a drop-in measurement command for a
  local-only profile, but full Light integration (measurement identity,
  report sealing, budget accounting) is follow-up work and should be validated
  against AKA maintainers' expectations before merging.
