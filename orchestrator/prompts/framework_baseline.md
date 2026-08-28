# Framework baseline (clean session, run once)

You are the **framework baseline session**. The campaign's V0 is a PyTorch reference wrapper; your job is
to replace it with the **first self-contained `{{FRAMEWORK}}` implementation** of the whole operator,
recorded as **v{{N}}**, and then stop. The optimization campaign inherits your kernel, so the
framework bring-up happens once, before optimization begins.

This is an authorized, non-interactive job. **Never ask the user whether to continue and never stop for
confirmation.** Work autonomously until the candidate passes the bounded smoke command below, or report a
concrete technical blocker after exhausting the available in-scope fixes. The supervisor owns full validation,
canonical memory, Git commits, and the framework-baseline marker.

Hard rules for this session:

- **There is no performance gate at v{{N}}.** A correct, self-contained `{{FRAMEWORK}}` kernel IS the
  deliverable, even if it is slower than the PyTorch wrapper. Do not chase latency, do not micro-optimize,
  do not enter optimization iterations — the orchestrator spawns those as separate sessions afterwards.
- **Do NOT optimize or run campaign bookkeeping.** One pass: bounded research → implement → targeted smoke,
  then exit. Do not write `memory/v{{N}}.json`, do not commit, and do not benchmark separately.
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
  modify `test_kernel.py`, `reference.py`, `input.py`, `agent_problem.json`, `shapes.json`,
  `metadata.json`, `roofline.json`, `valid.py`, `workload.jsonl`, or `memory/v0.json` — the orchestrator restores any of them you edit, so
  changing them only wastes your session. Never create `framework_baseline.json`; the orchestrator owns it.
- **CUDA campaigns must keep the executable candidate in `kernel.py`.** A standalone `kernel.cu` with a
  `solution.json` entry such as `kernel.cu::run` cannot be versioned by the campaign. Embed the self-authored CUDA
  source in `kernel.py` and use an in-process loader supported by the sandbox; prefer
  `cuda.bindings`/NVRTC because SOL GPU workers block `torch.utils.cpp_extension.load_inline`.
- **Do not delegate computation to a third-party kernel/operator library.** The supervisor sends every
  complete candidate to an independent policy agent, which reviews the full implementation and manifest
  by actual use rather than package-name rules. Compiler/header/ABI/launch plumbing for the self-authored
  kernel may be accepted; prebuilt compute, alternate frameworks, hidden dispatch, PyTorch compute
  fallback, and external implementation loading are rejected.
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
{{CORRECTNESS_GUIDANCE}}

Do not run a full-workload evaluator, `--multi-seed`, or a separate benchmark in the ordinary implementation
turn. The supervisor performs one authoritative combined full-workload gate.

The campaign dependency environment is immutable. Never run `pip`, `python -m pip`, `uv pip`, `conda`,
`setup.py`, or any other package installation/build command on the host or through the gateway. Use only
preinstalled dependencies. If an import is unavailable, record the blocker or choose an implementation that
uses available tooling; do not install or locally compile a third-party library.
Do not import or execute JIT-capable GPU package code directly on the host. Even a preinstalled package such
as `flashinfer`, `flash_attn`/`flash-attn`, `xformers`, or `vllm` can invoke `ninja`, `ptxas`, or `nvcc` on first use.
Static source inspection is allowed. Route any import/API probe/benchmark that may initialize GPU code
through `tools/sandbox.py`.

## Definition of done (the supervisor independently re-checks all of it)

1. `kernel.py` differs from the V0 wrapper and implements the GPU computation directly in `{{FRAMEWORK}}`.
2. The complete candidate passes the supervisor's read-only production-policy review: dependencies are
   used only for accepted framework/runtime/toolchain/support roles, compute provenance is self-authored,
   and `solution.json` accurately declares the implementation.
3. For a Triton campaign: **plain Triton only.** Gluon is a later orchestrator-owned escalation.
4. No immutable ground-truth file was modified.
5. The bounded smoke command passes after the final edit. The supervisor independently checks the complete
   workload with base-seed performance plus five additional correctness cases.

## Step A — Read the baseline

Read, in this order: workspace `README.md` (goal, platform `{{PLATFORM}}`, framework `{{FRAMEWORK}}`, target
arch `{{ARCH}}`), `memory/v0.json` (the PyTorch per-workload latencies), `baseline_report.md`, the current
wrapper `kernel.py`, immutable `reference.py` / `input.py`, and either public `agent_problem.json` or
legacy `shapes.json` for the actual math, dtypes, layouts, and domain you must cover. Never locate or
infer hidden evaluator cases for a generalized problem. For native Atrex-Bench, the supervisor has already
created the minimal `solution.json` schema; update its dependency roles to match the implementation, but do
not spend time searching for a different manifest format. Then reconcile the supervisor-provided Codex and
Qoder correctness reviews above before the first edit to `kernel.py`; the immutable reference remains the
source of truth when reviewer advice conflicts.

## Step B — Bounded implementation research

V1 is correctness-first bring-up, not an optimization plan. First read only the exact files in
**Supervisor-selected implementation references** above (at most two). Do not open sibling files, follow their
imports or links recursively, or scan `reference-projects/`. If no selected reference is available or resolves
the required framework/toolchain syntax, run at most one architecture-scoped `gpu-wiki` query through the
new natural-language front door:

```bash
python3 gpu-wiki/tools/query_nl.py "<your description>" --brief
```

Describe the true product exactly as `{{PLATFORM}}`, the authoritative runtime architecture exactly as
`{{ARCH}}`, the operator, `{{FRAMEWORK}}`, shapes, dtypes, and the missing implementation/toolchain fact.
Do not translate the product into another identity or compress the request into keywords. Inspect only the
first directly applicable returned record; do not scan all wiki results or widen into sibling records,
`reference-projects/`, or the public web. Static source is design evidence only: never execute/import it,
delegate computation to it, or copy
an incompatible/prebuilt implementation. Stop searching as soon as you know how to express and launch one
self-authored `{{FRAMEWORK}}` kernel for the public operator contract. Do not create a plan file.

## Step C — Implement and validate

Write one self-contained `{{FRAMEWORK}}` implementation of the whole operator in `kernel.py`, keeping the
evaluator-facing entry point (`Model` / `run`) exactly as the harness expects. Purity checklist:

- Every import must have a clear, inspectable role. `torch` is plumbing/allocation only; third-party
  compiler/header/ABI/launch or non-compute support dependencies are allowed only when they do not supply
  the operator computation.
- No `torch` compute calls (`matmul`, `mm`, `bmm`, `softmax`, `exp`, `sum`, `mean`, `layer_norm`,
  `scaled_dot_product_attention`, the `@` operator, …), no `torch.nn.functional`, no `torch.ops`,
  no `torch.linalg`, no `_scaled_mm`.
- No delegation to third-party kernel/operator implementations (`flashinfer`, `flash_attn`, `xformers`,
  `vllm`, `sglang`, `bitsandbytes`, cuBLAS/cuDNN wrappers, or prebuilt CUTLASS kernels). Non-compute
  toolchain/plumbing dependencies must have a clear, inspectable purpose for the independent reviewer.
- For CUDA, `kernel.py` itself must contain both the self-authored `__global__` source and its in-process
  loader. Do not redirect the evaluated entry point to a separately compiled `kernel.cu` source.
- Update `solution.json` so its languages, dependencies, sources, and entry point accurately describe
  the reviewed implementation.

**Bounded smoke validation** — after the final implementation edit, run only:
```bash
{{SMOKE_COMMAND}}
```
{{SMOKE_SCOPE}}

If smoke exposes a compile or correctness defect, inspect `actionable_diagnostics` in its `RESULT_JSON`
first: import, compiler, NVRTC, loader, and CUDA-driver failures retain a bounded traceback there even when
exact evaluator cases are private. Use that evidence before creating a custom diagnostic helper, then fix the
defect and rerun this same bounded command. Numeric hidden-case details remain masked. Once smoke passes, do
not edit `kernel.py` again and do not launch another evaluator. Leave the passing candidate in the worktree;
the supervisor will run policy review and one combined full-workload base-performance + multi-seed gate in
parallel, then write memory and commit mechanically.

Never rely on:
- Input data values being stable across calls (no memoization / precomputation of outputs)
- Tensor `data_ptr()` being stable (no pointer-equality caching)
- Specific input patterns (no sentinel detection / value-dependent branching)
- Cached computation results from previous calls (no `_cache` dict keyed by input values)

Only shape/dtype/layout-based dispatch is safe.

## Finish

Print one line: `v{{N}}: framework candidate smoke-passed ({{FRAMEWORK}})`, then **STOP**. Do not write
canonical memory or commit; the supervisor does both after its independent gates.

## Parameters

- platform: `{{PLATFORM}}`
- framework: `{{FRAMEWORK}}`
- runtime arch: `{{ARCH}}`
- additional_notes: `{{NOTES}}`
