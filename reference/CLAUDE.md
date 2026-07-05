# GPU Kernel Optimizer — Agent Constraints

This file defines hard behavioral constraints for the optimization workflow.
For the full stage-by-stage workflow, read the skill files referenced below.

## Workflow Stage Delegation Rules

When executing `skills/gpu-kernel-profile-optimizer/SKILL.md`:

### Workflow Integrity Constraint (MANDATORY)

- The agent MUST strictly follow every step defined in the workflow, in the exact order specified.
- NO step may be skipped, abbreviated, or merged with another step under any circumstance.
- Each stage's entry conditions, execution steps, and exit criteria MUST be fully satisfied before proceeding.
- If a step appears unnecessary for the current context, the agent MUST still execute it and document why the result was trivial — skipping is NEVER permitted.

### Stage 2 (Evidence-Driven Search and Planning)

- The main agent MUST launch a research subagent for Stage 2.
- The subagent MUST follow `agents/gpu-kernel-research.md` exactly.
- The main agent SHALL NOT perform evidence search or write the plan directly.
- The main agent SHALL NOT call gpu-wiki search, read reference-projects, or web search for optimization knowledge — this is the subagent's job.

### Stage 4 (Performance, Correctness, and Quality Gate)

- The main agent MUST launch a subagent for Stage 4 validation.
- The main agent SHALL NOT run correctness tests, measure performance, or write the iteration report directly.
- The main agent SHALL NOT fabricate performance numbers or skip validation.

## Prohibited Main Agent Actions

- DO NOT perform search/plan writing that belongs to Stage 2.
- DO NOT run validation/measurement that belongs to Stage 4.
- DO NOT skip the subagent delegation by inlining these tasks.
- DO NOT proceed to Stage 3 without receiving the subagent's plan output.
- DO NOT proceed to Stage 5 without receiving the subagent's quality gate result.

## Framework Constraint

- **The V0 baseline is a pure-PyTorch reference wrapper** (correct + directly submittable), NOT yet the target framework. Migrating the body of `run()` from PyTorch to the `--framework` DSL is the expected first lever of the optimization loop — do it in an early iteration, and update `solution.json` `spec.languages`/`dependencies` in the same iteration so the harness benches the real kernel.
- Once the kernel is in the `--framework` DSL, optimization iterations MUST stay within it — switching to a *different* DSL (other than the PyTorch→target migration above) is NEVER permitted.
- Third-party helper libraries (e.g., utility libraries, math libraries) MAY be introduced to assist optimization, but the final kernel implementation MUST use the designated framework.
- `triton` and `gluon` belong to the same framework family (`triton/gluon`). When either is specified, both are acceptable implementation targets.
- When Triton-level optimization plateaus, the orchestrator spawns a dedicated **convert-only session** (`orchestrator/prompts/convert.md` → `gpu-kernel-convert`) that lowers the kernel Triton→Gluon with NO optimization, gated on correctness alone; the following sessions then optimize the Gluon kernel (deeper levers). Do not hand-trigger the rewrite inside a normal optimization iteration.

## Benchmark Harness Integrity

- **test_kernel.py is immutable for performance measurement**: DO NOT modify `test_kernel.py` to change the benchmark harness (e.g., warmup count, repetition count, `return_mode`, timing method, input shapes, or any other benchmark parameter) in order to obtain better performance numbers.
- `test_kernel.py` defines the ground-truth benchmark methodology. Any change to it invalidates cross-version comparisons.
- If a measurement methodology issue is discovered (e.g., outlier inflation, incorrect return mode), report it in `memory/v<N>.json` under `pitfalls_and_fixes` and propose the fix — but DO NOT apply the fix to `test_kernel.py` within an optimization iteration.
- **Validate + bench ONLY via `python test_kernel.py`** — it runs the real `sol-execbench` evaluator over EVERY workload in `workload.jsonl` (the full ground-truth shape set) with each workload's own tolerance. Never hand-roll a correctness test, bench a single "representative" shape, or edit the harness. A PASS here == a directly submittable solution.
- **The optimization goal is to minimize the GEOMEAN of per-workload kernel latency** (`performance.latency_us`, recorded by the harness). Per-workload latency is kept in `performance.latency_us_by_shape` (keyed by workload `uuid`). A version is committable only if ALL workloads pass AND the geomean drops vs HEAD beyond noise.
- **The SOL ground-truth files are immutable**: never edit `definition.json`, `reference.py`, or `workload.jsonl`. Edit `kernel.py` (DPS `run()`; args = definition.inputs then definition.outputs); update `solution.json` only when languages/dependencies/entry_point change.

## Hardware Architecture Constraints

- **blackwell-geforce is NOT blackwell**: `blackwell-geforce` (sm120) and `blackwell` (sm100) are completely different architectures. Do NOT conflate them or assume they share the same optimization strategies.
- **sm103 ≈ sm100 ≠ sm120**: The sm103 hardware architecture is similar to sm100 (both belong to the Blackwell data-center family), but is completely different from sm120 (Blackwell GeForce / consumer). When searching for reference kernels or optimization knowledge for sm103, prefer sm100/blackwell sources — NEVER use sm120/blackwell-geforce sources as a substitute.

## Skill References

- Full optimization workflow: `skills/gpu-kernel-profile-optimizer/SKILL.md`
- Research subagent contract: `agents/gpu-kernel-research.md`
