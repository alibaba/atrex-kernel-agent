# GPU Kernel Optimizer — Agent Constraints

This file defines hard behavioral constraints for the optimization workflow.
The full stage-by-stage workflow is defined in `orchestrator/prompts/iteration.md` (self-contained per session).

## Framework Guidance

- **The V0 baseline is a pure-PyTorch reference wrapper** (correct + directly submittable), NOT yet in any optimized DSL. Migrating the body of `run()` from PyTorch to the `--framework` DSL is the *suggested* first lever of the optimization loop — do it in an early iteration, and update `solution.json` `spec.languages`/`dependencies` in the same iteration so the harness benches the real kernel.
- The `--framework` value is a **recommended optimization direction**, not a hard constraint. Sessions MAY use a different DSL or mixed approaches if evidence shows a better performance path.
- Third-party helper libraries (e.g., utility libraries, math libraries) MAY be introduced freely to assist optimization.
- `triton` and `gluon` belong to the same framework family (`triton/gluon`). When either is specified, both are acceptable implementation targets.
- When Triton-level optimization plateaus, the orchestrator spawns a dedicated **convert-only session** (`orchestrator/prompts/convert.md` → `gpu-kernel-convert`) that lowers the kernel Triton→Gluon with NO optimization, gated on correctness alone; the following sessions then optimize the Gluon kernel (deeper levers). Do not hand-trigger the rewrite inside a normal optimization iteration.

## Benchmark Harness Integrity

- **No hacking the evaluation script for performance.** Do NOT modify, monkey-patch, subclass, shadow, or otherwise subvert `test_kernel.py` — nor any other file/module the evaluator loads (`sol-execbench`, `torch.cuda.Event`/`time` shims, RNG/seeding utilities, the timing loop, the comparison/tolerance check) — to make a slower kernel *look* faster or to make an incorrect result *pass*. Any speedup must come from a faster `run()` on **arbitrary** inputs — not from gaming the measurement.
- **test_kernel.py is immutable for performance measurement**: DO NOT modify `test_kernel.py` to change the benchmark harness (e.g., warmup count, repetition count, `return_mode`, timing method, input shapes, or any other benchmark parameter) in order to obtain better performance numbers.
- `test_kernel.py` defines the ground-truth benchmark methodology. Any change to it invalidates cross-version comparisons.
- If a measurement methodology issue is discovered (e.g., outlier inflation, incorrect return mode), report it in `memory/v<N>.json` under `pitfalls_and_fixes` and propose the fix — but DO NOT apply the fix to `test_kernel.py` within an optimization iteration.
- **Validate + bench ONLY via `python test_kernel.py`** — it runs the real `sol-execbench` evaluator over EVERY workload in `workload.jsonl` (the full ground-truth shape set) with each workload's own tolerance. Never hand-roll a correctness test, bench a single "representative" shape, or edit the harness. A PASS here == a directly submittable solution.
- **The optimization goal is to minimize the GEOMEAN of per-workload kernel latency** (`performance.latency_us`, recorded by the harness). Per-workload latency is kept in `performance.latency_us_by_shape` (keyed by workload `uuid`). A version is committable only if ALL workloads pass AND the geomean drops vs HEAD beyond noise.
- **The SOL ground-truth files are immutable**: never edit `definition.json`, `reference.py`, or `workload.jsonl`. Edit `kernel.py` (DPS `run()`; args = definition.inputs then definition.outputs); update `solution.json` only when languages/dependencies/entry_point change.

### Real-submission input model (don't overfit to the local bench)

The local `test_kernel.py` run is a **proxy** for the real evaluator. In the real scenario,
**every `run()` invocation receives freshly randomized inputs at freshly allocated addresses** —
the shapes/dtypes come from `workload.jsonl`, but values, RNG seed, and tensor pointers are
NOT fixed across calls. Concretely:

- **Do NOT cache input data.** Never recognize "I've seen these inputs before" and return a
  precomputed result, a recorded reference output, or any branch that depends on the *values*
  of the inputs (checksums, hashes, sentinel detection, "if input == X return Y"). `run()` must
  recompute from its arguments every call.
- **Do NOT cache pointers / addresses.** Never key a code path, a precompiled plan, a
  specialized kernel, or a cached workspace on `tensor.data_ptr()` / `tensor.storage().data_ptr()`
  / raw CUDA addresses — they change every invocation. If you build a runtime plan (autotuned
  tile config, cublasLt algo, JIT-compiled specialization), key it on **shape + dtype + layout
  + device** (stable invariants), never on pointer identity.
- **Do NOT cache outputs / scratch buffers tied to a specific address.** Reuse of a workspace
  tensor across calls is fine *if* you re-allocate (or re-validate) it per call based on shape;
  it is NOT fine to assume the buffer at address `0x...` from a previous call is still valid.
- **Do NOT amortize work across iterations of the timed loop.** Any setup that is only correct
  because the same inputs repeat (e.g., a one-time precompute on call #1 cached for calls #2..N)
  is a correctness bug, not an optimization.

If a shortcut only works because inputs/addresses are stable, it is invalid — drop it and
optimize the per-call work directly.

### Multi-seed robustness (mandatory before commit)

A single-seed PASS is NOT sufficient. Before committing any kernel change, run:

```bash
python test_kernel.py --version v<N> --multi-seed 5
```

This re-runs the evaluator under 5 additional random seeds and reports PASS only if ALL seeds
pass. If any seed fails correctness, the kernel is BROKEN — revert with `git reset --hard HEAD`
and try a different lever. See `iteration.md` Stage 3 step 4 for the full procedure.

### Cache-hack ZERO-TOLERANCE policy

The following patterns are cache hacks and any version that contains them MUST be reverted
**immediately** (`git reset --hard HEAD`) and recorded as a dead-end in `pitfalls_and_fixes`:

- **Output / answer caching.** Returning a stored reference output (memoized from a previous
  run / from the harness, hardcoded constants, recorded `out_*` tensor) instead of recomputing
  from the current inputs.
- **Input-value caching.** Recognising "I've seen these inputs before" (checksum / hash /
  sentinel detection / value-dependent branch / first-call precompute reused on calls #2..N)
  and short-circuiting `run()`.
- **Pointer / address caching.** Keying a code path, plan, autotune specialization, cublasLt
  algo, or scratch workspace on `tensor.data_ptr()` / `tensor.storage().data_ptr()` / raw CUDA
  pointers. Plans MAY be keyed on **shape + dtype + layout + device** only.
- **Eval-harness shadowing.** Importing / monkey-patching `test_kernel.py` / `sol-execbench` /
  `torch.cuda.Event` / time shims / RNG / the timing loop / the comparator from inside
  `kernel.py`, or detecting "am I being benchmarked" to take a faster branch.
- **CUDA-graph capture of fixed pointers.** Capturing a graph against the addresses seen at
  capture time and replaying it without re-binding parameters per call. If you use CUDA graphs,
  you MUST update the kernel node parameters from the *current* tensor `data_ptr`s on each `run()`.

If you find ANY of the above in the inherited `HEAD` kernel at the start of an iteration,
your **first action** is to revert it (`git reset --hard HEAD~` until HEAD is hack-free),
then proceed with a clean optimization. A hack-ridden kernel that "looks fast" is a
regression, not a starting point.

### Precision margin requirement (don't surf the tolerance line)

Per-workload tolerances are **safety margins, not optimization targets**. Because the real
evaluator reseeds inputs every call, a kernel that passes "by a hair" on one local seed can
easily fail on the next draw — that's a correctness bug, not a flake. **If any workload's
measured error sits close to its tolerance line, STOP and re-review the whole `kernel.py`
end-to-end before committing**, and verify the margin is stable across multiple fresh seeds
(`test_kernel.py --multi-seed 5`). A speedup that only works by shrinking the precision
margin is not real — revert and try a different lever.

### No multi-stream timing tricks

**Do NOT use multiple CUDA streams (concurrency) to reduce measured latency.** Launching
work on several streams so that independent kernels/op calls overlap in time is forbidden —
the evaluator times a single `run()` call, and multi-stream overlap hides work behind other
work rather than making the kernel itself faster, producing a misleading (non-representative)
latency. Keep all of `run()`'s work on the **default stream**; do not create/sync extra
`torch.cuda.Stream`s or `cudaStream`s to parallelize the computation. Legitimate
single-stream optimizations — kernel fusion, better tiling, vectorization, lower precision,
library primitives — are unaffected.

## Hardware Architecture Constraints

- **blackwell-geforce is NOT blackwell**: `blackwell-geforce` (sm120) and `blackwell` (sm100) are completely different architectures. Do NOT conflate them or assume they share the same optimization strategies.
- **sm103 ≈ sm100 ≠ sm120**: The sm103 hardware architecture is similar to sm100 (both belong to the Blackwell data-center family), but is completely different from sm120 (Blackwell GeForce / consumer). When searching for reference kernels or optimization knowledge for sm103, prefer sm100/blackwell sources — NEVER use sm120/blackwell-geforce sources as a substitute.

## Workflow References

- Optimization loop orchestrator: `orchestrator/optimize.py`
- Per-iteration session prompt (self-contained): `orchestrator/prompts/iteration.md`
- Baseline setup session: `orchestrator/prompts/setup.md`
- Triton→Gluon convert session: `orchestrator/prompts/convert.md`
- NVIDIA profiling skill (Stage 1): `.claude/skills/ncu-report-skill/SKILL.md`
- Plan generation (Stage 2): `/humanize:gen-plan` (plugin, loaded via `--plugin-dir`)
