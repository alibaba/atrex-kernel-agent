# Atrex Kernel Agent

AKA is an end-to-end Agent project for GPU kernel implementation, analysis, profiling, and iterative optimization. It helps an Agent turn PyTorch logic or an existing kernel into a high-performance GPU kernel through a structured, profile-driven workflow.

![Atrex architecture](atrex-architecture.png)

![Atrex optimization loop](atrex-optimization-loop.png)

## What It Does

- Creates an isolated optimization workspace under `kernel_opt_<name>/`.
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

### 3. Run the Installer

```bash
bash install.sh --prefix "$(pwd)"
```

The install path is optional and defaults to `~/aka_kernel_opt`. Using the repository root as
the prefix installs a self-contained optimizer under:

```text
.codex/skills/gpu-kernel-optimizer/
.claude/skills/gpu-kernel-optimizer/
```

Common options:

```bash
bash install.sh --prefix ~/my_path # Install to a custom directory
bash install.sh --hooks-only        # Install or update hooks only
bash install.sh --without-github    # Skip GitHub-hosted reference repositories
bash install.sh --uninstall         # Remove hooks installed by this script
```

The installer detects supported runtime home directories and prepares local hooks when available.

After installation, restart the coding runtime or open a new session so the hooks are loaded.

## Quick Start

### 1. Install the Optimizer

From the repository root:

```bash
git submodule update --init
bash install.sh --prefix "$(pwd)"
export AKA_KERNEL_OPT_HOME="$(pwd)"
```

Restart the coding runtime or open a new session after installation so the installed hooks are
loaded.

### 2. Prepare an Operator Directory

`--op-dir` points to an atrex-bench operator directory. A typical operator looks like:

```text
/path/to/ops/mla_decode/
├── reference.py
├── input.py
├── shapes.json
├── roofline.json
└── metadata.json
```

| File | Purpose |
| --- | --- |
| `reference.py` | Required reference implementation and initial kernel source. |
| `input.py` | Input construction and operator-specific test data. |
| `shapes.json` | Complete benchmark shape set; every optimization round should test all shapes. |
| `roofline.json` | Per-shape theoretical work, SOL time, and utilization targets. |
| `metadata.json` | Production-performance metadata used for priority weighting when available. |

`reference.py` is required by the orchestrator. Include the remaining files for complete
correctness, benchmark, Roofline, and scheduling behavior.

### 3. Run Optimization

Run the orchestrator from a directory where you want optimization workspaces to be created:

```bash
mkdir -p "$HOME/kernel-optimization-runs"
cd "$HOME/kernel-optimization-runs"
```

With Codex:

```bash
python "$AKA_KERNEL_OPT_HOME/.codex/skills/gpu-kernel-optimizer/orchestrator/optimize.py" \
  --op-dir /path/to/ops/mla_decode \
  --platform H20 \
  --framework CuteDSL \
  --agent codex \
  --max-iters 20 \
  --token-budget 8000000 \
  --target-util 90
```

With Claude:

```bash
python "$AKA_KERNEL_OPT_HOME/.claude/skills/gpu-kernel-optimizer/orchestrator/optimize.py" \
  --op-dir /path/to/ops/mla_decode \
  --platform H20 \
  --framework CuteDSL \
  --agent claude \
  --max-iters 20 \
  --target-util 90
```

Claude remains the default provider when `--agent` is omitted. The main runtime parameters are:

- `platform`: target hardware platform, such as `H20` or `MI308X`.
- `framework`: target implementation framework, such as `CuteDSL` or `FlyDSL`.
- `agent`: clean-session provider, either `codex` or `claude`.
- `max-iters`: hard limit on optimization iterations.
- `token-budget`: optional token limit across all sessions; `0` disables this limit.
- `target-util`: utilization percentage that can stop the campaign after a validated commit.

Each iteration runs in a fresh agent session. Cross-session state is carried through Git,
`memory/vN.json`, `plans/`, and `profiles/` rather than conversation history.

### 4. Find the Result

For an operator directory named `mla_decode`, the orchestrator creates:

```text
$HOME/kernel-optimization-runs/kernel_opt_mla_decode/
├── kernel.py
├── test_kernel.py
├── README.md
├── memory/
├── plans/
└── profiles/
```

`kernel.py` at Git `HEAD` is always the best accepted implementation. Regressing or incorrect
iterations are reverted, while their evidence is retained in `memory/`, `plans/`, and `profiles/`.

```bash
cd "$HOME/kernel-optimization-runs/kernel_opt_mla_decode"
git log --oneline
```

## Main Files

```text
.
├── SKILL.md                         # Top-level gpu-kernel-optimizer router manifest
├── install.sh                       # Installer / uninstaller
├── docs/                            # Detailed project design docs
├── orchestrator/                    # Clean-session optimization runner and prompts
├── agents/                          # Profiling, research, baseline, and implementation playbooks
├── reference/                       # Workspace, plan, memory, and profiling templates
├── skills/                          # Baseline, optimizer, restart, and output-contract modules
├── tools/                           # Profiling, utilization, memory, and measurement tools
└── gpu-wiki/                        # Local GPU knowledge base
```

## License

Licensed under the [Apache License 2.0](LICENSE).
