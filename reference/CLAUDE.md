# GPU Kernel Optimizer — Agent Constraints

This file defines hard behavioral constraints for the optimization workflow.
The full stage-by-stage workflow is defined in `orchestrator/prompts/iteration.md` (self-contained per session).

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

## Workflow References

- Optimization loop orchestrator: `orchestrator/optimize.py`
- Per-iteration session prompt (self-contained): `orchestrator/prompts/iteration.md`
- Baseline setup session: `orchestrator/prompts/setup.md`
- Triton→Gluon convert session: `orchestrator/prompts/convert.md`
- NVIDIA profiling skill (Stage 1): `.claude/skills/ncu-report-skill/SKILL.md`
- Plan generation (Stage 2): `/humanize:gen-plan` (plugin, loaded via `--plugin-dir`)
