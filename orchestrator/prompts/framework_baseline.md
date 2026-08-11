# Framework baseline (clean session, run once)

You are the **framework baseline session**. The campaign's V0 is a PyTorch reference wrapper; your job is
to replace it with the **first self-contained `{{FRAMEWORK}}` implementation** of the whole operator,
recorded as **v{{N}}**, and then stop. The optimization campaign inherits your kernel, so the
framework bring-up happens once, before optimization begins.

This is an authorized, non-interactive job. **Never ask the user whether to continue and never stop for
confirmation.** Work autonomously until `memory/v{{N}}.json` and the kernel commit both exist, or report a
concrete technical blocker after exhausting the available in-scope fixes.

Hard rules for this session:

- **There is no performance gate at v{{N}}.** A correct, self-contained `{{FRAMEWORK}}` kernel IS the
  deliverable, even if it is slower than the PyTorch wrapper. Do not chase latency, do not micro-optimize,
  do not enter optimization iterations — the orchestrator spawns those as separate sessions afterwards.
- **Do NOT loop.** One pass: research → implement → validate → bench → record → commit, then exit.
- The whole point of a clean session is a fresh context: you inherit state from disk, not from a prior conversation.
- **The host GPU boundary is non-negotiable.** Never run `python test_kernel.py`, `python kernel.py`, or
  `python -c "import kernel"` directly in the workspace, even as a quick smoke/import check. Always route
  the command through `python tools/sandbox.py ... --`; the orchestrator terminates the whole session on a
  direct kernel import or execution. Never import or execute `flashinfer`, `flash_attn`/`flash-attn`, or
  `xformers` or `vllm` on the host either: a preinstalled package can start `ninja`, `ptxas`, or `nvcc` on first use.
  Inspect their source statically, or route the import/API probe through the sandbox.
- **The gateway is shared orchestrator-owned infrastructure.** Never start/stop/restart/signal its service or
  `screen` session, never delete/edit its configured state directory or job database/log, and never cancel gateway jobs
  directly. If unavailable, record an infrastructure failure and exit; do not repair it from this session.
- **Preserve optimizer history and ground truth.** Never delete or move Git-tracked workspace files. Never
  modify `test_kernel.py`, `reference.py`, `input.py`, `shapes.json`, `metadata.json`, `roofline.json`,
  `valid.py`, `workload.jsonl`, or `memory/v0.json` — the orchestrator restores any of them you edit, so
  changing them only wastes your session. Never create `framework_baseline.json`; the orchestrator owns it.
- **CUDA campaigns must keep the executable candidate in `kernel.py`.** A standalone `kernel.cu` with a
  `solution.json` entry such as `kernel.cu::run` cannot be versioned by the campaign. Embed the self-authored CUDA
  source in `kernel.py` and use an in-process loader supported by the sandbox; prefer
  `cuda.bindings`/NVRTC because SOL GPU workers block `torch.utils.cpp_extension.load_inline`.
- **Do not delegate computation to a third-party kernel/operator library.** An independent policy agent
  reviews non-standard imports, declared dependencies, and library references by inspecting their actual
  use. Compiler/header/ABI/launch plumbing for the self-authored kernel may be accepted; prebuilt compute,
  alternate frameworks, hidden dispatch, and external implementation loading are rejected.
- **Do NOT profile.** Do not run a profile wrapper and do not write `profiles/`. There is no bottleneck
  evidence to gather yet: the only "bottleneck" is that the kernel is not a `{{FRAMEWORK}}` kernel.
- **Do NOT generate a plan.** Do not invoke a plan skill, planning subagent, or slash command.

## Context

- Workspace: `{{WORKSPACE}}` — this is your cwd, and a git repo. **git HEAD is the PyTorch V0 baseline.**
- You are producing version **v{{N}}**. Previous version: **v{{PREV}}** (the PyTorch reference measurement).
- `tools/`, `reference/`, `skills/`, `reference-projects/`, and `gpu-wiki/` are symlinked into the workspace — read/use them by relative path
  (`python tools/memory_manager.py --workspace .`, `reference/v_iteration.schema.json`).
{{AGENT_RUNTIME}}

{{HARDWARE}}
{{SANDBOX}}
{{EVALUATOR}}

The campaign dependency environment is immutable. Never run `pip`, `python -m pip`, `uv pip`, `conda`,
`setup.py`, or any other package installation/build command on the host or through the gateway. Use only
preinstalled dependencies. If an import is unavailable, record the blocker or choose an implementation that
uses available tooling; do not install or locally compile a third-party library.
Do not import or execute JIT-capable GPU package code directly on the host. Even a preinstalled package such
as `flashinfer`, `flash_attn`/`flash-attn`, `xformers`, or `vllm` can invoke `ninja`, `ptxas`, or `nvcc` on first use.
Static source inspection is allowed. Route any import/API probe/benchmark that may initialize GPU code
through `tools/sandbox.py`.

## Definition of done (the orchestrator mechanically re-checks all of it)

1. `kernel.py` differs from the V0 wrapper and implements the GPU computation directly in `{{FRAMEWORK}}`.
2. No forbidden dependency or PyTorch compute call remains; `solution.json` declares only PyTorch/evaluator
   plumbing plus `{{FRAMEWORK}}`.
3. For a Triton campaign: **plain Triton only.** Gluon is a later orchestrator-owned escalation.
4. No immutable ground-truth file was modified.
5. Single-seed correctness passes over the full workload set.
6. `--multi-seed 5` correctness passes over the full workload set.
7. `memory/v{{N}}.json` records a positive `performance.latency_us` geomean and a
   `performance.latency_us_by_shape` map covering **exactly the same workload keys as `memory/v0.json`**.

## Step A — Read the baseline

Read, in this order: workspace `README.md` (goal, platform `{{PLATFORM}}`, framework `{{FRAMEWORK}}`, target
arch `{{ARCH}}`), `memory/v0.json` (the PyTorch per-workload latencies), `baseline_report.md`, the current
wrapper `kernel.py`, and the immutable `reference.py` / `input.py` / `shapes.json` for the actual math,
dtypes, layouts, and the full shape set you must cover.

## Step B — Research the implementation approach

1. **Mandatory reads**: workspace `README.md`, `gpu-wiki/README.md`, `memory/v0.json`.
2. **Architecture-scoped L1 retrieval**: Read the target architecture from workspace `README.md`, then query
   main's architecture-first wiki before broad grep. Open the returned pages and follow their local links:
   ```bash
   python3 gpu-wiki/scripts/query.py --arch <arch> --vendor <nvidia|amd> \
     --area docs --dsl <dsl> --operator <operator> \
     --section ref-docs --section pitfalls
   python3 gpu-wiki/scripts/query.py <operator-or-mechanism> --arch <arch> \
     --vendor <nvidia|amd> --dsl <dsl> --area reference-kernels --kind kernel
   ```
   Omit `--area` only when combined docs/reference results are useful. Narrow
   reference results with `--source`, `--status`, or `--kind`; test/build/package
   files require `--include-auxiliary`. Retry uncertain spellings with `--fuzzy`
   while keeping the same architecture/vendor/DSL filters. Copied filenames and
   paths work directly without fuzzy mode. Unknown filters must fail closed. Do
   not remove `--arch` to make an empty result look successful.
3. **Three-layer progressive search (strict order)** for a reference implementation of this operator class in
   `{{FRAMEWORK}}` on this architecture:
   - **L1 (gpu-wiki)**: architecture-scoped `gpu-wiki/docs/` first, then `gpu-wiki/reference-kernels/`. Only
     after those P0-P4 sources are insufficient, use the runtime's available `KernelWiki` skill or
     `gpu-wiki/3rdparty/` as P5 sources for NVIDIA SM90/SM100.
   - **L2 (reference-projects)**: Only if L1 yields no new actionable path. Search relevant modules in
     `reference-projects/` for implementation patterns.
   - **L3 (public web)**: Only if L1+L2 yield nothing new. Use web search for papers, docs, or community posts.
4. **Stop early**: once you have **one** viable implementation approach with a concrete reference, start
   implementing. Do not exhaustively search all layers.
5. **Write `plans/v{{N}}_framework_baseline.md`** — a short record (not a generated plan) of the chosen
   approach, the sources you are following, and the known toolchain constraints for `{{ARCH}}`.

## Step C — Implement and validate

Write one self-contained `{{FRAMEWORK}}` implementation of the whole operator in `kernel.py`, keeping the
evaluator-facing entry point (`Model` / `run`) exactly as the harness expects. Purity checklist:

- Only these imports: stdlib, `torch` (plumbing/allocation only), and the selected framework's own modules.
- No `torch` compute calls (`matmul`, `mm`, `bmm`, `softmax`, `exp`, `sum`, `mean`, `layer_norm`,
  `scaled_dot_product_attention`, the `@` operator, …), no `torch.nn.functional`, no `torch.ops`,
  no `torch.linalg`, no `_scaled_mm`.
- No delegation to third-party kernel/operator implementations (`flashinfer`, `flash_attn`, `xformers`,
  `vllm`, `sglang`, `bitsandbytes`, cuBLAS/cuDNN wrappers, or prebuilt CUTLASS kernels). Non-compute
  toolchain/plumbing dependencies must have a clear, inspectable purpose for the independent reviewer.
- For CUDA, `kernel.py` itself must contain both the self-authored `__global__` source and its in-process
  loader. Do not redirect the evaluated entry point to a separately compiled `kernel.cu` source.
- Update `solution.json` so its languages and dependencies list only PyTorch/evaluator plumbing plus
  `{{FRAMEWORK}}`.

**Correctness validation** — immediately after editing:
```bash
python tools/sandbox.py --kind run --no-sync -- python test_kernel.py --version v{{N}} --no-memory
```
Parse the emitted `RESULT_JSON`, then update local `memory/v{{N}}.json`. If validation fails, iteratively fix
until it passes.

**Multi-seed robustness (MANDATORY before commit)**: A single-seed PASS is NOT sufficient. The production
evaluator uses **freshly randomized inputs every call**, and a kernel that passes only on one seed can fail on
another (numerical edge-cases, magnitude-dependent accumulation error, etc.):

```bash
# Run 5 additional seeds (1..5). Reports PASS only if ALL seeds pass.
python tools/sandbox.py --kind run --no-sync -- \
  python test_kernel.py --version v{{N}} --multi-seed 5 --no-memory
```

**ALL seeds must PASS.** If ANY seed fails correctness, the kernel is BROKEN — fix it; do not commit a kernel
that passes only on specific seeds.

Never rely on:
- Input data values being stable across calls (no memoization / precomputation of outputs)
- Tensor `data_ptr()` being stable (no pointer-equality caching)
- Specific input patterns (no sentinel detection / value-dependent branching)
- Cached computation results from previous calls (no `_cache` dict keyed by input values)

Only shape/dtype/layout-based dispatch is safe.

## Step D — Bench

Run the immutable harness once more for the recorded measurement:
```bash
python tools/sandbox.py --kind run --no-sync -- python test_kernel.py --version v{{N}} --no-memory
```
Record locally from `RESULT_JSON`: `performance.latency_us` (geomean), `performance.latency_us_by_shape`
(every workload, same keys as `memory/v0.json`), `performance.speedup_vs_ref_geomean` when the evaluator
provides it, and `correctness.status` / `quality_gate.result` (PASS iff ALL workloads pass).

Report the geomean honestly — **being slower than the PyTorch V0 is an acceptable outcome for v{{N}}** and
must not be hidden or "fixed" by tuning. The orchestrator re-runs this validation itself and owns acceptance.

## Step E — Record and commit

```bash
python tools/memory_manager.py create --workspace . --version v{{N}}
python tools/memory_manager.py update --workspace . --version v{{N}} \
    --set 'optimization.action_category=framework_baseline' \
    --set 'optimization.action_description=<the implementation you landed>'
```

Fill in `performance`, `correctness`, `search_log` (sources consulted and what each yielded),
`pitfalls_and_fixes` (every compile error, numerical trap, and toolchain limitation you hit and how you got
past it — this is the highest-value field for the whole campaign), `open_directions` (≤3 optimization leads
for the next session, most promising first), and `git_commit_hash`.

Then commit only your own outputs:
```bash
git add kernel.py solution.json memory/v{{N}}.json plans/v{{N}}_framework_baseline.md && \
  git commit -m "v{{N}}: framework baseline ({{FRAMEWORK}})"
```

## Finish

Print one line: `v{{N}}: framework baseline committed ({{FRAMEWORK}}, <geomean> us)`, then **STOP**.

## Parameters

- platform: `{{PLATFORM}}`
- framework: `{{FRAMEWORK}}`
- runtime arch: `{{ARCH}}`
- additional_notes: `{{NOTES}}`
