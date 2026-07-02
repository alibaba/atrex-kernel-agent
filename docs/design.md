# Architecture Design

## Overview

Atrex Kernel Agent is an end-to-end Agent project for GPU kernel implementation, analysis, profiling, and iterative optimization. It turns PyTorch logic or an existing kernel into an auditable optimization workspace, then drives the workflow from hardware-spec lookup and Roofline analysis to baseline implementation, profile-driven optimization, partial restart, and final candidate packaging.

The project centers on the `gpu-kernel-optimizer` Skill. The top-level Skill acts as a router and policy owner, while sub-skills handle baseline implementation, bottleneck analysis, profile-driven optimization, final packaging, and partial restart.

## Design Goals

- **Profile-driven optimization**: Kernel changes must be guided by official profiler evidence instead of intuition or ad-hoc timers.
- **Traceable hardware assumptions**: Hardware specs must come from the local `gpu-wiki` knowledge base and must be archived with source references.
- **Auditable optimization history**: Plans, profiles, reports, structured memory, and Git commits preserve every accepted iteration.
- **Reproducible workspaces**: Each task runs in an isolated `/tmp/kernel_opt_<name>/` workspace.
- **Controlled iteration state**: Structured `memory/v<N>.json` files record performance, correctness, profile evidence, search logs, risks, and commit hashes.
- **Safe final packaging**: Evaluator-facing output is separated into a clean `generated_kernel.py` contract when needed.

## Project Structure

```text
.
├── README.md
├── SKILL.md                              # Top-level gpu-kernel-optimizer router and global constraints
├── install.sh                            # Installer / uninstaller for Skills, hooks, gpu-wiki, and references
├── atrex-workflow.png                    # Workflow diagram
├── docs/
│   ├── design.md                         # This design document
│   └── gpu-wiki-design.md                # gpu-wiki knowledge-base design
├── gpu-wiki/                             # Local GPU knowledge base and reference index
│   ├── README.md
│   ├── docs/                             # Hardware specs, kernel optimization docs, pitfalls, ref docs
│   └── reference-kernels/                # AMD, NVIDIA, and generic reference kernels
├── reference/
│   ├── README.md                         # Workspace README template
│   ├── memory.md                         # Legacy / human-readable memory template
│   ├── plan.md                           # Per-iteration plan template
│   ├── iteration_report.md               # Iteration report template
│   ├── profile_guide.md                  # Consolidated NVIDIA / AMD profiling guide
│   ├── v_iteration.schema.json           # Structured memory JSON schema
│   └── workspace_init.sh                 # Creates /tmp/kernel_opt_<name>/ workspace
├── skills/
│   ├── gpu-kernel-baseline/              # Stage 1 baseline implementation Agent
│   ├── gpu-kernel-profile-optimizer/     # Stage 2 profile-driven optimization Skill
│   ├── gpu-kernel-output-contract/       # Final generated_kernel.py packaging Skill
│   └── gpu-kernel-partial-restart/       # Masked-memory partial restart Agent
└── tools/
    ├── bench_bandwidth.py                # Bandwidth benchmark helper
    ├── compute_utilization.py            # TFLOPS / bandwidth utilization calculator
    ├── extract_asm.py                    # Assembly extraction helper
    ├── measure_bandwidth_ceiling.py      # Same-size bandwidth ceiling measurement
    ├── measure_kernel_time.py            # Kernel latency helper
    ├── memory_manager.py                 # Structured memory JSON manager
    ├── profile_kernel.sh                 # AMD rocprofv3 / ATT / PMC / ASM wrapper
    ├── profile_nvidia.sh                 # NVIDIA ncu wrapper (metrics + symptom classification)
    ├── classify_ncu.py                   # NCU metrics -> symptom diagnosis
    ├── extract_nvidia_asm.py             # NVIDIA SASS extraction / analysis
    ├── ncu_helpers/                      # Bundled ncu-report parsing helpers
    ├── input_att.yaml                    # ATT profiling configuration
    └── rocprof-trace-decoder/ # AMD ATT decoder plugin
```

## Core Components

### Top-Level Router: `SKILL.md`

`SKILL.md` owns the global workflow and constraints. It parses user input, initializes the workspace, performs Step 0 hardware lookup and Roofline analysis, writes workspace configuration, and routes execution to the appropriate sub-skill.

The router is responsible for:

- Enforcing hardware-spec sourcing from `gpu-wiki`.
- Creating `/tmp/kernel_opt_<name>/` workspaces.
- Running Step 0 before baseline or optimization.
- Choosing between baseline implementation, profile-driven optimization, bottleneck analysis, partial restart, and final packaging.
- Ensuring accepted iterations are committed with Git.
- Treating unmasked memory plus workspace `README.md` as the source of truth.

### Baseline Agent

Path: `agents/gpu-kernel-baseline.md`

The baseline Skill implements the first correct kernel version from PyTorch logic or a kernel demo. It learns the target framework through `gpu-wiki`, writes `kernel.py` and `test_kernel.py`, validates correctness, records baseline performance, writes `baseline_report.md`, creates `memory/v0.json`, and commits the baseline.

### Profile Optimizer Skill

Path: `skills/gpu-kernel-profile-optimizer/SKILL.md`

This Skill runs the main iterative optimization loop:

```text
Profile and evidence extraction
-> Query gpu-wiki / reference projects / web sources for relevant optimization knowledge
-> Evidence-driven planning
-> Single-category implementation
-> Correctness / performance / quality gate
-> memory/v<N>.json update and Git commit
-> Stop-condition check or next iteration
```

Each iteration must use official profiler evidence, change exactly one optimization category, validate correctness before performance conclusions, and record results in structured memory.

### Output Contract Skill

Path: `skills/gpu-kernel-output-contract/SKILL.md`

This Skill packages a validated implementation into `generated_kernel.py` when a hidden evaluator requires a clean final candidate. The final file must contain valid Python source only, define `class Model(nn.Module)`, preserve the reference contract, and exclude tests, benchmarks, debug prints, Markdown, external file reads, and `__main__` blocks.

### Partial Restart Agent

Path: `agents/gpu-kernel-partial-restart.md`

This Skill is used when no new actionable direction is available but Stop Conditions are not met. It masks about half of the previous optimization memories, preserves the latest successful iteration and baseline, and launches a fresh subagent from the current `kernel.py` and unmasked memory.

### Layer Decomposition Overlay (optional)

Path: `agents/gpu-kernel-decompose.md` (rules) + `orchestrator/optimize.py --layer` (driver) +
`orchestrator/prompts/decompose.md` / `recombine.md`.

By default the whole workflow operates on a **single fusion-bounded operator**, and this overlay is inert. It
activates **only** when the input is a **composite of more than one separable op** — a whole LLM layer, or a
smaller multi-op composite such as `rope+attention` or `attention+moe` — or the user explicitly asks to split.
A single operator (one heavy op plus its fusable epilogue) bypasses it entirely — the common case. When active it:

1. **Decomposes** the layer into fused-operator boundaries per the evidence-backed rules in
   `agents/gpu-kernel-decompose.md` (the GEMM epilogue is the universal fuse surface; split where a reduction
   crosses a parallelism boundary or the tiling regime changes; keep the flash attention core and MoE
   token-sort monolithic). It emits one basic fused kernel per boundary plus a `boundaries.json` manifest
   (per-boundary op type, shapes, SOL time from `atrex-bench/scripts/roofline.py`, and expected `%SOL` ceiling).
2. **Fans out** each boundary into its **own standard atrex workspace** — own `kernel.py`, git (HEAD = best),
   Step 0 roofline, baseline `v0`, and profile-driven optimization. A boundary *is* a single-operator campaign;
   the per-operator machinery is reused unchanged.
3. **Schedules a shared budget**: `optimize.py` owns one iteration budget (`--max-iters`) across all boundaries.
   Each round it advances the highest live-ROI boundary by one version
   (`priority = max(0, latency − SOL/ceiling) × decay(stall)`), so `Σ (per-boundary versions) == --max-iters`.
   **No boundary is ever dropped** — a plateaued one is decayed and re-enters contention later. SOL is the
   yardstick, never a stop gate (termination is plateau + the shared budget).
4. **Recombines** each boundary's git-HEAD kernel into the full-layer kernel and validates end-to-end.

gpu-wiki holds **no** partitioning content; it remains operator-optimization knowledge only. The single-operator
path (`Campaign` in `optimize.py`) is untouched by this overlay.

## End-to-End Workflow

### 1. Parse Input and Initialize Workspace

The workflow parses required fields from user input:

| Field | Required | Notes |
|------|----------|-------|
| `platform` | Yes | Hardware target, e.g. `H20`, `H100`, `MI308X`, `MI355X`. |
| `arch` | Derived | `H20/H100/H200 -> Hopper`, `MI300X/MI308X -> CDNA3`, `MI355X -> CDNA4`. |
| `framework` | Yes | Target framework, e.g. `CuteDSL` or `FlyDSL`. |
| `kernel_demo` | Yes | Initial kernel or PyTorch logic file. |
| `gpu_wiki_path` | Default | `/tmp/gpu-wiki/`. |
| `reference_project` | Default | `/tmp/reference-projects/`. |

Workspace initialization uses:

```bash
bash reference/workspace_init.sh <name> <kernel_demo_path>
```

The script creates:

```text
/tmp/kernel_opt_<name>/
├── kernel.py          # Copied from kernel_demo
├── .gitignore
├── memory/
├── plans/
└── profiles/
```

### 2. Step 0: Hardware Specs and Roofline Analysis

Before writing the workspace `README.md`, the workflow must:

1. Read `/tmp/gpu-wiki/README.md` and follow the indexed hardware-spec path.
2. Find exact target-platform specs. Similar-product inference is forbidden.
3. Record each spec in auditable source format:

```text
<metric>: <value> <unit> <- <gpu-wiki>/<relative-path>:<line-or-section>
```

4. Statically analyze the kernel demo:
   - theoretical FLOPs
   - theoretical bytes moved
   - arithmetic intensity
   - Roofline ridge point
   - compute-bound or memory-bound classification
5. Compute Stop Conditions as `hardware peak * 90%`, preferring measured maxima from `gpu-wiki` when available.
6. Write hardware specs, Roofline analysis, and Stop Conditions into workspace `README.md`.

### 3. Write Workspace README and Initialize Memory

The workspace `README.md` is created from `reference/README.md`. It stores static task configuration, hardware specs, Roofline results, Stop Conditions, and ISA optimization targets.

Structured iteration memory is initialized with:

```bash
python tools/memory_manager.py init --workspace /tmp/kernel_opt_<name>
```

From this point, the workspace `README.md` plus all unmasked `memory/v<N>.json` files are the source of truth.

### 4. Stage 1: Baseline Implementation

The main agent launches a subagent that follows `agents/gpu-kernel-baseline.md`.

The subagent must:

- Understand PyTorch semantics, shapes, dtype, layout, masks, and accuracy requirements.
- Learn target framework APIs through `/tmp/gpu-wiki/README.md`.
- Implement a correct baseline `kernel.py` and `test_kernel.py` using CuteDSL or FlyDSL.
- Validate correctness with timeout protection.
- Measure baseline performance and calculate TFLOPS / bandwidth utilization.
- Write `baseline_report.md`.
- Create and fill `memory/v0.json`.
- Commit the baseline to Git.

### 5. Stage 2: Profile-Driven Iterative Optimization

Each optimization iteration follows the profile optimizer Skill.

Important rules:

- Official profile evidence is required before code changes.
- NVIDIA evidence comes from `ncu`, wrapped by `tools/profile_nvidia.sh` (metrics parsing + symptom classification).
- AMD evidence comes from `tools/profile_kernel.sh`, collecting ATT, PMC, and ASM artifacts.
- After extracting bottleneck evidence, each iteration must query `gpu-wiki` first, then reference projects, then public web sources when needed, before writing the optimization plan.
- Each iteration changes exactly one optimization category so attribution is clear.
- Correctness must pass before performance conclusions or commits.
- Performance records must include latency, TFLOPS, bandwidth, and peak-utilization ratios.
- If a quality gate fails, the workflow reverts to the previous commit and records the failure.

### 6. Stop or Continue

The optimizer stops when Stop Conditions in workspace `README.md` are met. Otherwise, it continues with the next profile-driven iteration. If no new actionable path is available, it may enter the partial restart workflow.

## Workspace Artifacts

A typical optimization task produces:

```text
/tmp/kernel_opt_<name>/
├── README.md                 # Static configuration, sourced specs, Roofline analysis, Stop Conditions
├── kernel.py                 # Current kernel implementation
├── reference.py              # Optional runnable reference, when generated or provided
├── test_kernel.py            # Correctness and performance validation entry point
├── baseline_report.md        # Baseline implementation report
├── generated_kernel.py       # Final evaluator-ready candidate, when packaging is required
├── memory/
│   ├── v0.json               # Baseline iteration record
│   ├── v1.json               # Optimization iteration record
│   └── ...                   # Files with masked=true are ignored by active planning
├── plans/
│   ├── v0_plan.md
│   ├── v1_plan.md
│   └── ...
└── profiles/
    ├── v1/                   # Per-version ncu / rocprof / ATT / PMC / ASM artifacts
    └── ...
```

## Structured Memory Design

`tools/memory_manager.py` manages per-iteration JSON files in `memory/v<N>.json`.

Supported operations include:

- `init`: create the `memory/` directory.
- `create`: create a new iteration JSON file from the schema template.
- `read`: read a specific iteration or all unmasked iterations.
- `update`: update fields in a specific iteration JSON file.
- `mask`: set `masked: true` on specified iterations.
- `unmask`: set `masked: false` on specified iterations.
- `summary`: print a performance summary table.
- `latest`: print the latest unmasked iteration version.
- `list`: list all iteration files with masked status.

Common commands:

```bash
python tools/memory_manager.py init --workspace /tmp/kernel_opt_<name>
python tools/memory_manager.py create --workspace /tmp/kernel_opt_<name> --version v0
python tools/memory_manager.py read --workspace /tmp/kernel_opt_<name> --unmasked-only
python tools/memory_manager.py update --workspace /tmp/kernel_opt_<name> --version v1 --set 'correctness.status=PASS'
python tools/memory_manager.py mask --workspace /tmp/kernel_opt_<name> --version v2 v4
python tools/memory_manager.py summary --workspace /tmp/kernel_opt_<name>
```

The `masked` field allows the workflow to discard stale optimization memory without deleting data. Masked files must not influence active planning, search deduplication, or optimization decisions.

## Profiling Design

The workflow trusts official profile evidence only:

- NVIDIA: `ncu`, wrapped by `tools/profile_nvidia.sh` (metrics parsing + symptom classification)
- AMD: `tools/profile_kernel.sh`, which wraps `rocprofv3`, ATT, PMC, and ASM extraction

`reference/profile_guide.md` consolidates profiling commands, metric interpretation, evidence extraction, SASS/ASM analysis, and troubleshooting.

AMD profiling outputs are placed under each iteration profile directory:

```text
profiles/v<N>/
├── att/           # Instruction-level trace
├── pmc/           # Hardware counter results
└── kernel.s       # Assembly
```

Optimization decisions should be written as an evidence chain:

```text
evidence -> inference -> optimization action
```

Examples:

- `PMC shows high SQ_LDS_BANK_CONFLICT` -> `LDS bank conflicts are significant` -> `try a swizzled layout`
- `ASM shows many buffer_load_dword and few dwordx4` -> `global memory vectorization is insufficient` -> `adjust alignment and vector width`
- `ncu shows memory dependency dominates warp stalls` -> `latency hiding is insufficient` -> `try double buffering or software pipelining`

## Tooling

The `tools/` directory provides:

- `compute_utilization.py`: calculate TFLOPS, bandwidth, and peak-utilization percentages.
- `bench_bandwidth.py`: run bandwidth benchmarks.
- `measure_bandwidth_ceiling.py`: measure same-size bandwidth ceilings.
- `measure_kernel_time.py`: helper for kernel latency measurement.
- `extract_asm.py`: extract AMD assembly for analysis.
- `profile_kernel.sh`: AMD profiling wrapper for rocprofv3, ATT, PMC, and ASM.
- `profile_nvidia.sh`: NVIDIA profiling wrapper for ncu; parses metrics and classifies symptoms via `classify_ncu.py` and the bundled `ncu_helpers/`.
- `extract_nvidia_asm.py`: extract and analyze NVIDIA SASS (from `.ncu-rep`, cubin, Triton, or CuteDSL).
- `memory_manager.py`: manage structured iteration records.

## Critical Constraints

- Hardware specifications must not be guessed. They must come from `gpu-wiki` and include source references.
- Archived specs without `gpu-wiki` sources are invalid.
- Missing specs must be recorded as `UNKNOWN (gpu-wiki not found)` and escalated to the user.
- Optimization decisions require official profile evidence: `ncu` / `tools/profile_nvidia.sh` for NVIDIA or `tools/profile_kernel.sh` / `rocprofv3` for AMD.
- `do_bench`, `torch.cuda.Event`, and handwritten timers are timing helpers only; they cannot replace official profiler evidence for bottleneck decisions.
- Each optimization iteration must change only one optimization category.
- Kernel changes that fail correctness tests must not proceed to performance validation or commit.
- Every accepted baseline and optimization iteration must be committed with Git.
- `memory/v*.json` files with `masked: true` are discarded from active planning and must not influence future optimization decisions.
- Final `generated_kernel.py`, when required, must contain only evaluator-ready runtime code.

## Use Cases

- Generate high-performance GPU kernels from PyTorch reference implementations.
- Diagnose bottlenecks through Roofline analysis and official profiler evidence.
- Optimize kernels on NVIDIA Hopper or AMD CDNA platforms.
- Continue stalled optimization runs with partial restart and masked memory.
- Produce clean evaluator-ready candidates for hidden benchmark systems.
- Maintain an auditable, reproducible, and continuously iterative kernel optimization workflow.
