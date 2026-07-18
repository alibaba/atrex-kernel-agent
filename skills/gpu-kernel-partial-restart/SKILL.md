---
name: gpu-kernel-partial-restart
description: Partial-restart workflow for GPU kernel optimization when no new optimization direction is available. The main agent masks half of the optimization memory, then launches a fresh subagent that reads README.md and unmasked memory to extract experience, takes the latest kernel.py as the starting point, and continues optimization through the gpu-kernel-profile-optimizer workflow.
---

# GPU Kernel Partial Restart

Use this skill only when the current `gpu-kernel-optimizer` session cannot find a new actionable optimization direction, but the target in `README.md` has not been met.

## Goal

Restart optimization from a partially fresh context. The main agent masks approximately half of the `memory/v*.json` files to break stale conclusions, then launches a **new subagent** that extracts experience from `README.md` and the remaining unmasked memory, uses the current `kernel.py` as its starting point, and continues the `gpu-kernel-profile-optimizer` workflow independently.

## Required Inputs

- The `kernel_opt_<name>` workspace created by `gpu-kernel-optimizer`.
- Existing `kernel_opt_<name>/memory/` directory with `v*.json` files.
- Existing `kernel_opt_<name>/README.md`.
- Existing `kernel_opt_<name>/kernel.py` (latest version).

## Workflow

### Phase 1: Main Agent Prepares the Restart

The main agent performs the following steps before launching the subagent.

1. **Preserve the target**
   - Read `README.md` first.
   - Keep platform, framework, hardware-spec sources, Roofline analysis, and Stop Conditions unchanged.
   - Do not edit target thresholds unless the user explicitly requests it.

2. **Review current memory**
   - List and read all iteration files using the memory manager:

     ```bash
     python tools/memory_manager.py list --workspace kernel_opt_<name>
     python tools/memory_manager.py read --workspace kernel_opt_<name> --unmasked-only
     ```

   - Identify recorded optimization experience, including attempted directions, failures, conclusions, constraints, lessons, and rejected ideas stored in each iteration's JSON.

3. **Randomly mask half of the optimization experience**
   - List all unmasked `memory/v*.json` files (excluding v0 baseline).
   - Randomly choose approximately half of these files and mask them:

     ```bash
     python tools/memory_manager.py mask --workspace kernel_opt_<name> --version v2 v4 v5
     ```

   - Preserve the latest successful iteration (the current best version) as unmasked.
   - Do not delete any JSON files; only set `masked: true`. Data can be unmasked later.
   - Record that a partial restart was performed by adding an entry in the latest unmasked `memory/v*.json` under `pitfalls_and_fixes` summarizing what was masked.

4. **Determine the next version number**
   - Find the highest existing version N across all `memory/v*.json` files (both masked and unmasked).
   - The subagent's first iteration will be `v<N+1>`.

### Phase 2: Launch Restart Subagent

The main agent must launch a subagent for the restart. The main agent must not continue optimization directly after masking.

Subagent requirements:

- **Task type**: editing task.
- **Required inputs**: workspace path, `README.md`, all unmasked `memory/v*.json` file paths, current `kernel.py` path, platform, framework, next version number `v<N+1>`, gpu-wiki path, Stop Conditions.
- **Must do**:
  1. Read `README.md` to get hardware specs, Roofline analysis, Stop Conditions, and ISA optimization targets.
  2. Read all unmasked `memory/v*.json` files to extract: what optimizations were tried, what worked, what failed, what lessons were learned, and what bottlenecks remain.
  3. Summarize the extracted experience as context for planning — treat records as historical hints, not as hard proof. Previous poor results may have come from implementation details or overfitting, not necessarily from invalid ideas.
  4. Take the current `kernel.py` as the starting point (do not revert to baseline).
  5. Enter `gpu-kernel-profile-optimizer` Stage 1 (Profile) on the current `kernel.py`, then continue through Stage 2–6 as normal.
  6. Start from version `v<N+1>` for iteration numbering.
  7. Search for optimization paths without assuming masked conclusions are still valid.
- **Forbidden**:
  - Do not revert `kernel.py` to baseline or any earlier version.
  - Do not create a new workspace; continue in `kernel_opt_<name>`.
  - Do not read masked `memory/v*.json` files as active data.
  - Do not modify `README.md` target thresholds or hardware specs.
  - Do not re-run Stage 1 baseline implementation.
- **Return**: latest `memory/v<N+1>.json` path, performance summary, optimization action taken, quality gate result, git commit hash, and whether Stop Conditions are met.

## Hard Constraints

- **No new workspace**: do not create `kernel_opt_<op_name>_iter<version>` or any other restart directory.
- **Subagent required**: the main agent must launch a subagent for the restart; the main agent must not continue optimization directly after masking.
- **No target drift**: preserve Stop Conditions and hardware-spec sources from `README.md`.
- **No file deletion**: do not delete `memory/v*.json` files; only set `masked: true` to discard memory. Mask only about half of optimization experience; keep essential state required to continue safely.
- **No masked field on v0**: do not mask `memory/v0.json` (the baseline) unless the user explicitly requests it.
- **Optimizer workspace only**: continue all work in `kernel_opt_<name>`, the workspace created by `gpu-kernel-optimizer`.
- **kernel.py is the starting point**: the subagent must optimize from the current `kernel.py`, not from baseline.

## Completion Criteria

The partial restart is complete only when:

- `README.md` has been read and target constraints are preserved.
- Approximately half of the `memory/v*.json` files have been masked (`masked: true`), with data preserved intact.
- A subagent has been launched that read unmasked memory, extracted experience, and entered the `gpu-kernel-profile-optimizer` workflow starting from the current `kernel.py`.
- The subagent returned results from at least one optimization iteration.
