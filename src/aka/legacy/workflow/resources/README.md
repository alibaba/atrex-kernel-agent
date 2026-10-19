# Campaign Workspace Reference

This selected Workflow resource describes the campaign contract for Agent sessions.
Use its Task Context fields with the injected setup prompt when writing the workspace README.

## Target

- logical platform: `<H20 / B200 / MI308X / MI355X / ...>` from `--platform`
- runtime architecture: `<sm_90 / sm_100 / gfx942 / gfx950 / ...>` probed through the sandbox or
  supplied by `--arch`
- framework: `<Triton / CuteDSL / Cuda / FlyDSL / ...>` selected per campaign
- optimization mode: `leaderboard` or `production`
- operator and dtype: derived from immutable files under `--op-dir`

## Execution Boundary

- Every compile, correctness, benchmark, signature collection, and profile command runs through
  `python -m aka.legacy.execution.cli` on `--sandbox-hardware`.
- Host-side GPU/JIT execution and dependency installation are forbidden.
- The logical platform and gateway hardware selector may use different names; runtime architecture
  probing is authoritative for vendor/framework dispatch.
- Optimizer memory, plans, source edits, episode journals, worktrees, and Git state remain local.

## Inputs and Immutable State

- SOL campaigns consume `reference.py`, `definition.json`, and `workload.jsonl`.
- Legacy native Atrex-Bench campaigns expose `reference.py`, `input.py`, and `shapes.json`.
- Generalized native Atrex-Bench campaigns expose `reference.py`, `input.py`, and
  `agent_problem.json`; exact shapes and evaluator metadata remain private and are injected only by
  the sandbox into the canonical `scripts/run_eval.py` evaluation. Each canonical memory record
  retains evaluator latency for every real opaque shape id without publishing its input parameters;
  profiling injects only the selected `PROFILE_SHAPE_ID` into the ephemeral remote job.
- The orchestrator installs an immutable workspace `test_kernel.py` adapter. Agent sessions must
  not replace it or edit evaluator/ground-truth files.
- V0 is a correctness-passing baseline. Production normally creates and pins a framework-native
  V1 before optimization episodes begin.

## State Ownership

- The Agent owns only its isolated episode worktree, private checkpoints, plan, profile evidence,
  journal, and terminal handoff.
- The supervisor owns canonical `memory/v<N>.json`, incumbent `HEAD`, ABBA verification, promotion,
  workload aggregation, budgets, and final packaging.
- A `candidate_ready` handoff is only a request for verification. It is never promotion authority.
- Accepted candidates are squash-promoted; rejected, pivoted, and blocked episodes advance
  canonical memory without changing the incumbent kernel.

## Knowledge and Tools

- Consult enabled knowledge tools using `.atrex_plugins/instructions.md` and
  `python -m aka.legacy.resources.cli list`. Follow the injected phase instructions. State the exact product,
  authoritative runtime architecture, operator, framework, measurements and unresolved question.
  Preserve returned source identifiers. With no suitable tool, use available reference sources.
- Search explicitly installed `reference-projects/<project>/` resources only when the enabled knowledge tools are insufficient. No reference projects are selected by default.
- Profiling resources below are available only for explicitly selected capabilities; confirm the installed Resources inventory before using them. Report a missing required capability.
- Use `profiling/nvidia/profile_nvidia.sh` and `profiling/nvidia/classify_ncu.py` for NVIDIA evidence.
- Use `profiling/amd/profile_kernel.sh` for AMD rocprofv3/ATT/PMC evidence.
- Use `profiling/ppu/profile_ppu.py` for T-Head PPU evidence; it is an `ncu`-compatible front end for
  `acu` and must be reachable as `ncu` on PATH. See `profiling/profile_guide.md` for activation
  and the mandatory performance-counter release step.
- Use `profiling/common/compute_utilization.py` and the measurement/extraction helpers for supporting
  calculations; use `python -m aka.legacy.workflow.memory_manager` only for workspace memory operations allowed by the
  current prompt.

## Mechanical Stop and Acceptance Rules

- Correctness must pass before performance can influence promotion.
- Ordinary episode promotion requires a strict same-allocation ABBA improvement and policy pass.
- Aggregate promotion requires the base seed, five additional correctness seeds, and the configured
  full-workload geomean improvement.
- Campaigns stop on canonical version budget, token budget, optional stall budget, target
  utilization, or a terminal repeated blocker.
- Hardware ceilings and optimization claims must have auditable sources; unknown values remain
  explicitly unknown rather than guessed.

## Task Context

- platform:
- runtime arch:
- sandbox hardware:
- framework:
- optimization mode:
- operator/dtype:
- workload source:
- correctness threshold:
- target utilization:
- additional constraints:

## ISA Optimization Targets

### AMD
- Global memory: increase the share of `buffer_load_dwordx4`; avoid heavy use of `buffer_load_dword` and `buffer_load_dwordx2`.
- LDS memory: increase the share of `ds_read_b128` and `ds_write_b128`.
- Registers: keep `vgpr_spill_count == 0`; keep `scratch_load` and `scratch_store` counts at 0.
- LDS conflicts: keep `SQ_LDS_BANK_CONFLICT` below the target threshold.
- Compute utilization: push `mfma_busy` / `valu_busy` toward the target threshold.
- Pipeline: keep the `memory dependency` share in warp stalls below the target threshold.
- Occupancy: keep active wavefronts per CU at the target threshold.
- Stall cycles: keep average `s_waitcnt` stall cycles below the target threshold.

### NVIDIA
- Memory throughput / SOL reaches the target threshold.
- L2 hit rate reaches the target threshold.
- Tensor Core utilization reaches the target threshold.
- Warp stall reason distribution remains below target limits.
- TMA / `cp.async.bulk` usage reaches the target share.
- Shared-memory bank conflict rate stays below the target threshold.
