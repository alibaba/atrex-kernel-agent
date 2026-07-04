# Atrex Kernel Agent

AKA is an end-to-end Agent project for GPU kernel implementation, analysis, profiling, and iterative optimization. It helps an Agent turn PyTorch logic or an existing kernel into a high-performance GPU kernel through a structured, profile-driven workflow.

![Atrex architecture](atrex-architecture.png)

![Atrex optimization loop](atrex-optimization-loop.png)

## What It Does

- Creates a self-contained per-kernel run root (which is also the workspace) at `--prefix` (default `/tmp/aka-opt`).
- Looks up target hardware specs from the local `gpu-wiki` knowledge base.
- Runs Roofline analysis and sets auditable performance targets.
- Implements a correct baseline kernel before entering optimization.
- Runs the profile-driven optimization loop: profile with `ncu` or `rocprofv3`, extract bottleneck evidence, query `gpu-wiki` / reference projects / web sources for relevant optimization knowledge, write an evidence-based plan, apply one optimization category, validate correctness and performance, record memory, commit, then repeat until Stop Conditions are met.
- Records plans, profile artifacts, structured memory, reports, and Git commits for every accepted iteration.

For the full architecture and workflow design, see [`docs/design.md`](docs/design.md).

## Requirements

Installation requires:

- `bash`
- `git`
- `jq`
- A compatible coding runtime installed

Running optimization tasks also requires platform-specific profiling tools:

- NVIDIA: `ncu`, wrapped by `tools/profile_nvidia.sh`
- AMD: `rocprofv3`, wrapped by `tools/profile_kernel.sh`

## Installation

### 1. Internal Development Environment Setup (internal users only — optional)

Internal users should configure git `insteadOf` URL redirect rules so that submodules and
dependencies resolve against the internal network before running `git submodule update` below.
**External users can skip this step entirely.**

### 2. Pull reference-projects Submodule

```bash
git submodule update --init
```

Downloads all reference projects managed under `reference-projects/`.

### 3. (Optional) Run the Installer

The orchestrator **self-installs** its hooks/subagents on first run (see Quick Start A), so this
step is **optional**. Use it to set up a **dedicated working directory, one per kernel** (fully
isolated from every other kernel), to install the Codex target, or to uninstall:

```bash
bash install.sh --prefix /work/kernelA
```

`--prefix` is optional; it defaults to `/tmp/aka-opt` (or `$AKA_KERNEL_OPT_HOME`).

This lays down, under the chosen directory:

- `.claude/` (and/or `.codex/`) — the `gpu-kernel-optimizer` skill, subagents, and hooks
- `gpu-wiki/`, `reference-projects/` — symlinked knowledge bases
- `orchestrator/` — `optimize.py` + `prompts/` + `anchor_bench.py`, so the directory is a
  self-contained run root (no need to reference the source repo)

Options:

```bash
bash install.sh --prefix /work/kernelA   # Install into a per-kernel working directory
bash install.sh --hooks-only             # Install or update hooks only (leaves orchestrator alone)
bash install.sh --without-github         # Skip GitHub reference repos from gpu-wiki
bash install.sh --uninstall --prefix /work/kernelA  # Remove hooks + installed orchestrator/
```

The installer detects supported runtime home directories and prepares local hooks when available.
After installation, restart the coding runtime or open a new session so the hooks are loaded.

## Quick Start

There are two ways to drive an optimization; both use the same skill, workspace, and memory format.

### A. Orchestrator (recommended for unattended, budget-bounded runs)

`optimize.py` owns the outer loop: it spawns a fresh `claude` session per iteration over the same
git workspace, and stops on a hard budget or a utilization target. It is driven by a single
`--op-dir` (an atrex-bench native op dir containing `reference.py`, `shapes.json`, `roofline.json`,
`metadata.json`, `input.py`) plus `--platform` / `--framework`. **It is self-installing — one command
sets up a self-contained run root and runs there; no separate `install.sh` step is required:**

```bash
git submodule update --init          # one-time; also attempted best-effort by the orchestrator
python orchestrator/optimize.py --prefix /work/kernelA \
    --op-dir /path/to/atrex-bench/data/<set>/<op> \
    --platform B300 --framework CuteDSL \
    --max-iters 20 --token-budget 8000000 --target-util 90
```

`--prefix <path>` is the **run root**: a single directory that is both the install base and the
workspace. On startup the orchestrator natively performs the `install.sh --prefix <path>` flow —
copies the skill/agents/orchestrator into `<path>`, symlinks `gpu-wiki` / `reference-projects`,
installs the gpu-kernel-optimizer hooks + subagents into `<path>/.claude` — then `cd`s into `<path>`
and pins each nested session's `CLAUDE_CONFIG_DIR` there, so the workflow gates and by-name subagents
are always active. **The workspace *is* `<path>`**: `memory/`, `plans/`, `profiles/`, `kernel.py`,
and its git repo are created directly in it, and every agent operation runs with `<path>` as the
current directory. `<path>` is then self-contained — it works even if the source repo is moved or removed.

`--prefix` defaults to `/tmp/aka-opt`; use **one `--prefix` per kernel** to keep campaigns isolated
(the default is a single shared scratch root). Pass `--skip-bootstrap` to skip the self-install on a
rerun of an already-set-up run root. Add `--layer` to decompose a fused multi-op reference (e.g. a
whole LLM layer) into per-boundary sub-workspaces under `<path>`, on one shared budget, then recombine.

### B. Interactive skill

Ask the Agent directly, with at least `platform`, `framework`, and the kernel file:

```text
/gpu-kernel-optimizer Optimize /path/to/kernel_demo.py on MI308X with FlyDSL, dtype bf16, rel_err < 0.01.
```

The Agent will initialize a workspace, source hardware specs from `gpu-wiki`, write the workspace configuration, build a baseline, profile the kernel, and iterate until the configured Stop Conditions are met.

## Main Files

```text
.
├── SKILL.md                         # Top-level gpu-kernel-optimizer router manifest
├── install.sh                       # Installer / uninstaller
├── docs/                            # Detailed project design docs
├── reference/                       # Workspace, plan, memory, and profiling templates
├── skills/                          # Baseline, optimizer, restart, and output-contract modules
├── tools/                           # Profiling, utilization, memory, and measurement tools
└── gpu-wiki/                        # Local GPU knowledge base
```

## License

Licensed under the [Apache License 2.0](LICENSE).
