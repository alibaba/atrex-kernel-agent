---
name: gpu-kernel-optimizer
description: End-to-end GPU kernel implementation and optimization router. Use this skill to turn PyTorch logic into a high-performance kernel or to systematically optimize an existing kernel. It initializes the local knowledge base, identifies the current phase, and routes work to baseline implementation, bottleneck analysis, and profile-driven optimization sub-skills.
---

# GPU Kernel Optimizer Router

This skill owns the global constraints and selects the right sub-skill for each phase.

## Mandatory Rule: Hardware Specs Must Come from gpu-wiki

Any target-hardware spec value, including compute throughput, HBM bandwidth, cache/LDS/SMEM capacity and bandwidth, register counts, SM/CU count, warp or wavefront size, issue rate, occupancy limits, or any other microarchitecture limit, must be sourced from the local `<gpu-wiki>/` knowledge base before use. Unsourced values invalidate the run and must be discarded.

### Hard Rules

1. **No fabrication**: Do not use memory, estimates, model output, web snippets, or verbal user statements as hardware specs in `memory/v<N>.json`, `baseline_plan.md`, `baseline_report.md`, `plans/v<N>_plan.md`, Roofline analysis, target calculations, or any archived decision.
2. **Source every spec**: Record every cited spec in this exact style:
   ```text
   <metric>: <value> <unit> <- <gpu-wiki>/<relative-path>:<line-or-section>
   ```
   Example: `H20 HBM3 bandwidth: 4.0 TB/s <- <gpu-wiki>/docs/hardware/h20_spec.md:L23`.
3. **Ownership**: Authoritative hardware lookup, registration, and downstream decisions must happen in this workflow. Values not registered in the workspace `README.md` or current `plans/v<N>_plan.md` cannot be used as specs, performance targets, or Roofline inputs.
4. **Fixed lookup flow**:
   1. Read `<gpu-wiki>/README.md` first to find the hardware-spec index.
   2. Search target-platform spec documents under `<gpu-wiki>/docs/`, `<gpu-wiki>/ref-docs/`, and other indexed directories.
   3. Register sources in archived files using the required format.
   4. Use the value only after registration.
5. **Missing wiki values**: If a spec cannot be found, write `<metric>: UNKNOWN (gpu-wiki not found)`, record the gap in `memory/v<N>.json` under `pitfalls_and_fixes`, report it to the user, and ask whether a placeholder is acceptable. Do not fill gaps with wording such as "approximately", "should be", "usually", or "similar product".
6. **Auditable archive**: Any reviewer must be able to verify every spec from the local `<gpu-wiki>/` path in the archive. Non-verifiable archives are invalid.
7. **Profiling-driven optimization only**:
   - NVIDIA: `ncu` (wrapped by `./tools/profile_nvidia.sh`)
   - AMD: `./tools/profile_kernel.sh`
   - `triton.testing.do_bench` is the designated tool for end-to-end kernel latency measurement used in Stop Conditions evaluation and performance recording. This is a timing tool, not a profiler — it may determine whether the target is met, but must not replace `ncu` or `profile_kernel.sh` for identifying bottlenecks.
8. Step 0 computes the performance targets and writes them into `README.md` under `Stop Conditions`.
9. Optimization runs in a temporary workspace and every accepted iteration is committed with git.
10. **Masked memory**: When reading `memory/v*.json` files, check the `masked` field in each file. Skip any file where `masked` is `true`. Masked files are treated as discarded memory and must not influence planning, search deduplication, or optimization decisions. The `masked` field defaults to `false` and can be set to `true` by the `gpu-kernel-partial-restart` sub-skill or by the user manually.
11. **Self-written kernel / anti-cheat (no exceptions under default policy `require_self_written_kernel`)**: The operator's core compute must be a kernel YOU write, launched from the entry function (`run`), using Triton (`@triton.jit`), CuteDSL (`@cute.jit`/`@cute.kernel`), cuTile, FlyDSL, or inline CUDA. The following are cheating and are rejected by `tools/validate_solution.py` (the single, non-bypassable gate, also invoked by the generated `test_kernel.py` and by `tools/sol_adapter.py package`):
    - **Library delegation (C1)**: importing/calling `flashinfer`, `flash_attn`, `xformers`, `vllm`, or `aiter`; using `torch.nn.functional.scaled_dot_product_attention` as the compute path; or wrapping the benchmark's target library op. `torch` may be used ONLY for setup/allocation/reshape/indexing/launch glue (a lone `F.linear` projection inside an otherwise real fused kernel is a WARN that needs justification, not a free pass).
    - **Language-tag camouflage (C2)**: declaring a framework in `solution.json` that is not actually launched from `run`, or pasting decorated kernels (`@cute.kernel`, `@triton.jit`) that are never reached "for language classification". `sol_adapter.py package` labels `spec.languages` from the framework actually detected on the data path — do not hand-edit it.
    - **Input/shape-keyed memoization (C3)**: process-global caches or `lru_cache` keyed on input shape metadata that move per-call work (host sync, python loops, H2D copies) out of the timed region. SOL-ExecBench's allocator only varies `data_ptr`; defeating it with a shape key is a cheat.
    - **Timing-methodology gaming (C4)**: skipping output initialization to exploit the allocator's pre-zeroing / cupti GPU-span measurement. Your kernel must write all output bytes.
    - **Masked-error PASS (C5)**: correctness must use SOL's exact tolerance (`max_atol`/`max_rtol`/`required_matched_ratio`/`max_error_cap`/NaN-Inf/`allow_negative_inf`). Do not substitute a looser global `rel_err` check.
    - **Fabricated target / Ref=0.000-as-success (C6)**: never invent a performance target (e.g. `peak * 0.9`, "990T") or a leaderboard number. Targets must derive from a **measured** baseline latency (`T_b`) — for SOL-ExecBench that is a library baseline (FlashInfer/DeepGEMM/cuDNN/torch) measured through the harness, cross-checked against the leaderboard "Scoring Baseline" row. If no real baseline latency is measured, the result is `UNSCORED` — you may not claim a speedup. `SOL Score` needs the hidden per-workload `T_SOL`; report it only as a clearly-labelled roofline **estimate** (or `N/A`), never as an official number.

12. **SOL-ExecBench leaderboard metrics (mandatory reporting)**: for a SOL-ExecBench problem, EVERY performance result you report (baseline_report, each iteration, final summary) MUST state the four leaderboard metrics, computed by `tools/sol_metrics.py`:
    - **Latency** = median over workloads of per-workload median `T_k` (ms) — EXACT.
    - **Fast** = `count(T_k < T_b)/N` — EXACT vs the measured library baseline.
    - **Avg Speedup** = `mean(T_b / T_k)` — vs the Scoring Baseline `T_b`, NOT the naive reference — EXACT.
    - **SOL Score** = `mean 1/(1+(T_k-T_SOL)/(T_b-T_SOL))` — ESTIMATE from a roofline `T_SOL` (Step 0), or `N/A`.
    The performance TARGET is to match or beat the **top-3** leaderboard entries (fetch with `tools/fetch_leaderboard.py --kernel-id <id> --gpu <gpu>`), not merely the roofline peak. Record the top-3 and Scoring Baseline in `README.md` under `Stop Conditions`.

### Scope Without Exceptions

- Baseline performance evaluation against theoretical limits
- Roofline peak FLOPS and peak bandwidth axes
- Memory-bound decisions and target comparisons
- TFLOPS, bandwidth, and absolute target comparisons for every iteration
- Occupancy, register pressure, LDS capacity, and other microarchitecture limits
- Any performance number written into archive files

### Startup Checklist

Before entering any phase or sub-skill that uses hardware specs, confirm:

1. Which hardware specs are needed for this task? Have they been sourced from `<gpu-wiki>/` and registered with the required source format?
2. Does any archived spec value lack a gpu-wiki source? If yes, remove it or add the source immediately.

If any answer is no, complete the missing work before continuing.

## When to Use

Trigger this skill when the request asks to:

- Implement PyTorch logic as a high-performance GPU kernel
- Optimize an existing kernel
- Analyze kernel bottlenecks, Roofline behavior, or bandwidth utilization
- Move from baseline implementation to profile-driven optimization

## Startup: Create Workspace and Parse User Input

Create an isolated optimization workspace in the **current working directory**。

```bash
bash reference/workspace_init.sh <name> <kernel_demo_path>
```

This script creates the workspace directory structure (`memory/`, `plans/`, `profiles/`), copies the kernel demo as `kernel.py`, initializes git, and creates `.gitignore`. See `reference/workspace_init.sh` for details.

**SOL-ExecBench problems**: when the input is a SOL-ExecBench problem directory (containing `definition.json` + `workload.jsonl`), initialize with the adapter instead:

```bash
bash reference/workspace_init.sh --sol-execbench <problem_dir> [<name>]
```

This parses `definition.json`/`workload.jsonl` and generates `reference.py` (the exact reference semantics), a destination-passing-style `kernel.py` stub with the correct `run(*inputs, *outputs)` signature, a frozen `baseline.json`, an honest `aka_bench_config.json` (`benchmark_reference=true`), and a SOL-faithful `test_kernel.py`. `test_kernel.py` is the single promotion gate: it runs the static anti-cheat validator, the authoritative `sol-execbench` CLI (real tolerance + leaderboard timing), honest `T_b` measurement, and a best-effort dynamic anti-memo/output-coverage probe. Package the final kernel for submission with `python tools/sol_adapter.py package kernel_opt_<name>` (refuses to emit `solution.json` if the validator reports a hard FAIL).

After global constraints are confirmed and before writing the workspace `README.md`, parse configuration from the user prompt. Do not read the current directory `README.md` for configuration. All configuration must come from explicit user input or defaults.

Flow: **parse user input -> Step 0 (hardware specs + Roofline analysis) -> write workspace README.md**.

| Field | Description | Default |
|------|-------------|---------|
| `platform` | **Required**. Target hardware platform, such as H20, H100, MI308X, or MI355X. | Ask the user if missing. |
| `arch` | Hardware architecture, derived from platform when possible. | H20/H100/H200 -> Hopper; MI300X/MI308X -> CDNA3; MI355X -> CDNA4. |
| `framework` | **Required**. Programming language/framework, such as CuteDSL or FlyDSL. | Ask the user if missing. |
| `gpu_wiki_path` | Local gpu-wiki path. Do not ask the user to confirm it. | `~/aka_kernel_opt/gpu-wiki/` |
| `reference_project` | Local reference-project path. | `~/aka_kernel_opt/reference-projects/` |
| `kernel_demo` | **Required**. Initial kernel implementation file to optimize. | Ask the user if missing. |
| `additional_notes` | Extra constraints, known bottlenecks, preferred directions, and edge cases. | `none` |

### Parsing Rules

1. Extract fields from the user prompt, for example "optimize on H20" -> `platform: H20`, "write it in CuteDSL" -> `framework: CuteDSL`.
2. Derive `arch` from `platform` using the mapping above.
3. Validate only the `kernel_demo` path.
4. If `platform`, `framework`, or `kernel_demo` is missing, ask the user. Do not guess.

## Step 0: Hardware Spec Lookup and Theoretical Roofline Analysis

Run Step 0 immediately after parsing user input and before writing the workspace `README.md`.

Goal: use the target `platform` and `kernel_demo` to source hardware specs from gpu-wiki, perform theoretical Roofline analysis, compute absolute performance targets, and write them into `README.md` under `Stop Conditions`.

1. **Lookup hardware specs from gpu-wiki**: Read `<gpu-wiki>/README.md`, follow its indexes, and find exact target-platform specs such as peak TFLOPS, HBM bandwidth, L2/LDS capacity, and relevant bandwidths. Match the exact hardware; do not infer from similar products. Every spec must include a gpu-wiki source and be written into `README.md` under `Hardware Spec`.
2. **Analyze the kernel demo statically**:
   - Compute theoretical FLOPs and theoretical data movement in bytes.
   - Compute `Arithmetic Intensity = FLOPs / Bytes`, compare with the Roofline ridge point, and classify the kernel as compute-bound or memory-bound.
   - Compute absolute targets as `hardware peak * 90%`, preferring gpu-wiki measured maxima when available, otherwise using documented hardware specs.
   - Write the Roofline analysis to `README.md`, including sourced specs, calculation process, bound classification, and absolute targets such as `compute-bound target >= 185.4 TFLOPS` or `memory-bound target >= 3.87 TB/s`.
   - Copy the targets to `README.md` under `Stop Conditions`.
3. **SOL-ExecBench problems — leaderboard target + baseline (do this in addition to roofline)**:
   - **Fetch the leaderboard**: `python tools/fetch_leaderboard.py --kernel-id <id> --gpu <platform> --out kernel_opt_<name>/leaderboard.json`. Record the **top-3** entries and the **Scoring Baseline** (`T_b`, SOL Score 0.5) and **SOL Bound** (`T_SOL`, SOL Score 1.0) rows in `README.md` under `Stop Conditions`. The performance target is to **match/beat the top-3** on Latency + Avg Speedup, not just the roofline peak.
   - **Build a measured library baseline** in `kernel_opt_<name>/baseline/` (FlashInfer for attention, DeepGEMM/cuBLAS for GEMM, else cuDNN/torch) and run it through the harness to get per-workload `T_b`; confirm its aggregate ≈ the leaderboard "Scoring Baseline" row. This is the `T_b` for Fast / Avg Speedup (measuring stick only — banned in the real kernel).
   - **Emit per-workload roofline `T_SOL`** to `kernel_opt_<name>/tsol.json` (`{idx: ms}`, `T_SOL_i = max(FLOPs_i/peak_tc, bytes_i/BW)`) so `sol_metrics.py` can produce a labelled SOL Score estimate.
   - Every result thereafter reports the four metrics via `tools/sol_metrics.py` (Constraint 12).

Completion criteria:

- Hardware specs, Roofline analysis, and `Stop Conditions` are written into workspace `README.md`.
- For SOL-ExecBench problems: `leaderboard.json` fetched, top-3 + Scoring Baseline recorded, library baseline built (or a TODO noted), `tsol.json` emitted.
- Bound type is determined.
- Absolute targets are computed.

Then proceed directly to writing the workspace `README.md`.

## Write Workspace README.md

Before entering any sub-skill, write the initial session constraints into the workspace `README.md`. The workspace `README.md` stores static configuration parsed from user input, including Task Context and ISA Optimization Targets (previously in `memory.md`). `memory/` stores structured iteration data as per-version JSON files (`memory/v<N>.json`). Files with `masked: true` are skipped during reads. `README.md` and the unmasked `memory/v*.json` files are the source of truth for every later iteration.

Fill `README.md` using `./reference/README.md`. Unknown fields must be `TBD`.

```bash
python tools/memory_manager.py init --workspace kernel_opt_<name>
```

Use `tools/memory_manager.py` for all memory JSON operations (create, read, update, mask/unmask, summary). See `python tools/memory_manager.py --help` for full usage.

If constraints change during the session, such as shape changes or relaxed thresholds, update `README.md` immediately. Do not leave changes only in the conversation.

## Workflow

Run the following phases in order. A phase must pass before the next phase starts.

### Stage 1: Baseline Implementation

**Sub-skill**: [gpu-kernel-baseline](skills/gpu-kernel-baseline/SKILL.md)

Goal: understand the PyTorch logic, extract compute pattern, input/output shapes, dtype, dependencies, and accuracy requirements; learn the target framework API and hardware constraints from `<gpu-wiki>/README.md`; then implement correct `kernel.py` and `test_kernel.py`, validate correctness and baseline performance, and create the starting point for profile-driven optimization.

The main agent must launch a subagent for Stage 1. The main agent must not implement the baseline directly. The subagent must read and follow `gpu-kernel-baseline`, read the PyTorch logic, learn framework APIs via `<gpu-wiki>/README.md`, implement `kernel.py` and `test_kernel.py`, validate correctness and performance, write `baseline_report.md`, write `memory/v0.json`, and commit.

Subagent requirements:

- **Task type**: editing task.
- **Required inputs**: workspace path, `README.md`, `memory/` directory, PyTorch logic or `kernel_demo`, platform, framework, dtype, shapes, and correctness threshold.
- **Must do**: read `gpu-kernel-baseline`; read workspace `README.md`; implement `kernel.py` and `test_kernel.py` base on CuteDSL or FlyDSL; run correctness and baseline performance validation; write `baseline_report.md`; write `memory/v0.json`; commit with git.
- **Forbidden**: do not skip `<gpu-wiki>/README.md`; do not fabricate hardware specs; do not modify Stage 2 plans or profiles; do not commit if correctness fails.
- **Return**: paths for `kernel.py` which implemented by CuteDSL or FlyDSL, `test_kernel.py`, and `baseline_report.md`; maximum `rel_err`; baseline performance; git commit hash; unresolved issues.

Entry criteria:

- `README.md` exists and includes Step 0 hardware specs, Roofline analysis, and `Stop Conditions`.
- Platform, framework, dtype, shapes, and correctness threshold are clear.
- PyTorch logic, `kernel_demo`, or reference code path is clear.

Exit criteria:

- Stage 1 subagent returned results.
- `kernel.py` exists and must be implemented based on CuteDSL or FlyDSL.
- `test_kernel.py` passes and records max `rel_err` plus PASS/FAIL.
- Baseline performance is recorded in `memory/v0.json`.
- `baseline_report.md` exists.
- Git commit is complete.

Then the main agent takes over and must enter Stage 2 immediately. It is forbidden to stop, summarize final deliverables, or exit the workflow after Stage 1 unless Stage 2 has also completed or the user explicitly asks to stop.

### Stage 2: Profile-Driven Iterative Optimization

**Sub-skill**: [gpu-kernel-profile-optimizer](skills/gpu-kernel-profile-optimizer/SKILL.md)

**Helper skill**: [gpu-kernel-bottleneck-analysis](skills/gpu-kernel-bottleneck-analysis/SKILL.md)

Goal: use Step 0 Roofline conclusions and multiple profile -> code change -> validation loops to approach the performance limit.

Stage 2 researches, plans, searches gpu-wiki/reference projects, writes an optimization plan, profiles, modifies code, validates correctness, applies quality gates, commits, and writes `memory/v<N>.json`. It must continually compare against ISA optimization targets recorded in `README.md`.

Entry criteria: Stage 1 passed and `README.md` contains Step 0 Roofline analysis and `Stop Conditions`.

Exit condition: performance reaches the absolute target in `README.md` under `Stop Conditions`.

When the exit condition is met, stop optimization and summarize deliverables.

## Recommended Flows

- PyTorch logic only: Step 0 -> Stage 1 -> Stage 2
- Existing kernel with "why is it slow": Step 0 -> Stage 2
- Roofline analysis only: Step 0 only

## Shared Tools

All sub-skills share top-level `tools/`:

- `tools/compute_utilization.py`
- `tools/bench_bandwidth.py`
- `tools/measure_bandwidth_ceiling.py`
- `tools/measure_kernel_time.py`
- `tools/extract_asm.py`
- `tools/profile_kernel.sh`
- `tools/profile_nvidia.sh`
- `tools/classify_ncu.py`
- `tools/extract_nvidia_asm.py`
- `tools/memory_manager.py`
- `tools/validate_solution.py` — static anti-cheat validator (single rule source for the library-delegation / camouflage / memoization bans; exit 0=clean, 1=WARN, 2=FAIL).
- `tools/sol_adapter.py` — SOL-ExecBench `materialize`/`package` adapter (truthful `solution.json`, refuses cheats).
- `tools/sol_metrics.py` — computes the four SOL leaderboard metrics (Latency / Fast / Avg Speedup exact vs a measured baseline; SOL Score as a labelled roofline estimate).
- `tools/fetch_leaderboard.py` — fetches the public leaderboard (top-3 targets + Scoring Baseline `T_b` + SOL Bound `T_SOL`) for a kernel id + GPU.

## Shared References

- `reference/workspace_init.sh` — workspace initialization script used in Startup phase.
- `reference/README.md` — workspace `README.md` template.
- `reference/plan.md` — optimization plan template.
- `reference/v_iteration.schema.json` — iteration JSON schema.
- `reference/profile_guide.md` — consolidated profile tool usage guide (ncu for NVIDIA, rocprofv3/ATT/PMC for AMD), sourced from gpu-wiki. Covers commands, key metrics, evidence extraction, SASS/ASM analysis, and troubleshooting.
