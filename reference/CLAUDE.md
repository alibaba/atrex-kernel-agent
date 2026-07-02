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

- Optimization iterations MUST stay within the framework specified by the `--framework` parameter — switching to a different framework is NEVER permitted.
- Third-party helper libraries (e.g., utility libraries, math libraries) MAY be introduced to assist optimization, but the final kernel implementation MUST use the designated framework.
- `triton` and `gluon` belong to the same framework family (`triton/gluon`). When either is specified, both are acceptable implementation targets.
- When Triton-level optimizations have converged (i.e., further Triton-only changes yield no significant performance improvement), the kernel SHOULD be rewritten in Gluon to unlock deeper optimization opportunities.

## Benchmark Harness Integrity

- **test_kernel.py is immutable for performance measurement**: DO NOT modify `test_kernel.py` to change the benchmark harness (e.g., warmup count, repetition count, `return_mode`, timing method, input shapes, or any other benchmark parameter) in order to obtain better performance numbers.
- `test_kernel.py` defines the ground-truth benchmark methodology. Any change to it invalidates cross-version comparisons.
- If a measurement methodology issue is discovered (e.g., outlier inflation, incorrect return mode), report it in `memory/v<N>.json` under `pitfalls_and_fixes` and propose the fix — but DO NOT apply the fix to `test_kernel.py` within an optimization iteration.
- **Bench EVERY shape defined in the workspace shape set** — never a single hand-picked "representative" shape. Record per-shape latency (`performance.latency_us_by_shape`), set `performance.latency_us` = their mean, and compute `performance.priority_ms` = mean over shapes of `max(0, latency_ms - SOL_ms)`. This is what the orchestrator ranks on; an under-benched (single-shape) record silently mis-ranks the whole layer.

## Hardware Architecture Constraints

- **blackwell-geforce is NOT blackwell**: `blackwell-geforce` (sm120) and `blackwell` (sm100) are completely different architectures. Do NOT conflate them or assume they share the same optimization strategies.
- **sm103 ≈ sm100 ≠ sm120**: The sm103 hardware architecture is similar to sm100 (both belong to the Blackwell data-center family), but is completely different from sm120 (Blackwell GeForce / consumer). When searching for reference kernels or optimization knowledge for sm103, prefer sm100/blackwell sources — NEVER use sm120/blackwell-geforce sources as a substitute.

## Skill References

- Full optimization workflow: `skills/gpu-kernel-profile-optimizer/SKILL.md`
- Research subagent contract: `agents/gpu-kernel-research.md`
