# GPU Kernel Optimizer — Agent Constraints

## Hardware Spec Sourcing

- **No fabrication**: Every hardware spec value (compute throughput, HBM bandwidth, cache/SMEM/LDS capacity, register counts, SM/CU count, occupancy limits, or any microarchitecture limit) MUST be sourced from the local `<gpu-wiki>/` knowledge base.
- **Source format**: Record every cited spec as: `<metric>: <value> <unit> <- <gpu-wiki>/<relative-path>:<line-or-section>`.
- **Missing values**: If a spec cannot be found in gpu-wiki, write `<metric>: UNKNOWN (gpu-wiki not found)`, record the gap in `memory/v<N>.json` under `pitfalls_and_fixes`, and ask whether a placeholder is acceptable. Never fill gaps with "approximately", "should be", "usually", or "similar product".
- Full rules: `SKILL.md` §Mandatory Rule.

## Benchmark Harness Integrity

- **`test_kernel.py` is immutable**: do not modify it to change warmup count, repetition count, `return_mode`, timing method, input shapes, or any benchmark parameter. It defines the ground-truth benchmark methodology; any change invalidates cross-version comparisons.
- **SOL ground-truth files are immutable**: never edit `definition.json`, `reference.py`, or `workload.jsonl`.
- **Validate + bench ONLY via `python test_kernel.py`** — it runs the real `sol_execbench` evaluator over every workload in `workload.jsonl` with each workload's own tolerance. Never hand-roll a correctness test or bench a single shape.
- **Optimization objective**: minimize the GEOMEAN of per-workload kernel latency (`performance.latency_us`). A version is committable only if ALL workloads pass AND the geomean drops vs HEAD beyond noise.

## Masked Memory

When reading `memory/v*.json` files, check the `masked` field. Skip any file where `masked` is `true` — masked files are discarded memory and must not influence planning, search deduplication, or optimization decisions. The `masked` field defaults to `false` and can be set to `true` by the `gpu-kernel-partial-restart` agent or by the user manually.

## Memory Management

- **All iteration memory operations MUST use `tools/memory_manager.py`** (create, read, update, mask/unmask, summary).
- Never hand-edit `memory/v*.json` files directly — use `python tools/memory_manager.py update --workspace . --version v<N> --set '<key>=<value>'`.
- Never hand-fabricate iteration JSON structure — use `python tools/memory_manager.py create --workspace . --version v<N>` to initialize from the schema.
- Run `python tools/memory_manager.py --help` for full usage.

## Profiling

- **Profiling tools**: `./tools/profile_nvidia.sh` (NVIDIA) / `./tools/profile_kernel.sh` (AMD) are the only authorized profiling tools.
- **Timing tool**: `triton.testing.do_bench` is for latency measurement only (Stop Conditions evaluation, performance recording). It is NOT a profiler — it must not replace the authorized profiling tools for identifying bottlenecks.

## Evidence Format

All profiling conclusions must follow the chain: `evidence → inference → optimization action`. Evidence must come from profiling tool outputs (`summary.txt`, SASS/ASM analysis, source-level localization).

## Framework

- Once the kernel is implemented in the target DSL, optimization iterations MUST stay within it — switching to a *different* DSL is NEVER permitted.
- **`triton` and `gluon` belong to the same framework family** (`triton/gluon`). When either is specified, both are acceptable implementation targets. When Triton-level optimizations have converged (further Triton-only changes yield no significant improvement), the kernel SHOULD be rewritten in Gluon to unlock deeper optimization opportunities.
- Third-party helper libraries MAY be introduced, but the final kernel implementation MUST use the designated framework.
- **SOL-ExecBench workspaces** (where `definition.json` + `solution.json` are present): the initial `kernel.py` is a pure-PyTorch reference wrapper. Migrating `run()` from PyTorch to the target DSL is the expected first optimization lever — update `solution.json` `spec.languages`/`dependencies` in the same iteration.

## Hardware Architecture Pitfalls

- **`blackwell-geforce` (sm120) ≠ `blackwell` (sm100)**: completely different architectures. Do NOT conflate them or assume they share optimization strategies.
- **sm103 ≈ sm100 ≠ sm120**: sm103 is similar to sm100 (both Blackwell data-center). When searching for reference kernels or optimization knowledge for sm103, prefer sm100/blackwell sources — NEVER use sm120/blackwell-geforce sources as a substitute.

## Skill References

| Resource | Path |
|----------|------|
| Full optimization workflow (router) | `SKILL.md` |
| Profile-optimizer (one iteration) | `skills/gpu-kernel-profile-optimizer/SKILL.md` |
| Plan template | `reference/plan.md` |
| Iteration memory schema | `reference/v_iteration.schema.json` |
| Workspace README template | `reference/README.md` |
| Profile tool guide | `reference/profile_guide.md` |
| Memory management tool | `tools/memory_manager.py` |
| Baseline agent | `agents/gpu-kernel-baseline.md` |
| Profiler agent | `agents/gpu-kernel-profiler.md` |
| Research agent | `agents/gpu-kernel-research.md` |
| Optimizer agent | `agents/kernel-optimize.md` |
| Partial restart agent | `agents/gpu-kernel-partial-restart.md` |
