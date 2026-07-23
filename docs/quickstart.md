# Quick Start

AKA ships two independent ways to run the same profile-driven workflow. Use the interactive Skill route when you want to drive optimization inside a coding session, or the orchestrated route when you want an unattended, budget-bounded run.

## Prerequisites

- `bash`
- `git`
- A compatible coding runtime installed
- NVIDIA profiling: `ncu`, wrapped by `tools/profile_nvidia.sh`
- AMD profiling: `rocprofv3`, wrapped by `tools/profile_kernel.sh`

Route-specific prerequisites:

- Route 1 requires `jq` for `install.sh`.
- Route 2 requires Python 3, `torch`, and the `claude` CLI available on `PATH`.

## 1. Clone the Repository

```bash
git clone https://github.com/alibaba/atrex-kernel-agent.git
cd atrex-kernel-agent
```

## 2A. Run the Interactive Skill Route

Install the `gpu-kernel-optimizer` Skill and hooks:

```bash
git submodule update --init
bash install.sh --prefix ~/aka_kernel_opt
```

Restart your coding runtime or open a new session so the hooks are loaded. Then change into the directory where you want the optimization workspace to be created and ask the Agent to optimize a kernel demo:

```text
/gpu-kernel-optimizer Optimize /path/to/kernel_demo.py on MI308X with FlyDSL, dtype bf16, rel_err < 0.01.
```

The Agent creates `kernel_opt_<name>/` in the current working directory, sources hardware specs from `gpu-wiki`, builds a baseline, profiles the kernel, and iterates until Stop Conditions are met.

## 2B. Run the Orchestrated Loop Route

Run a single-operator campaign directly against a SOL-ExecBench op directory containing `definition.json`, `reference.py`, and `workload.jsonl`:

```bash
python orchestrator/optimize.py \
    --op-dir /path/to/sol-execbench/op \
    --platform H20 --framework CuteDSL \
    --max-iters 20 --token-budget 8000000 --target-util 90
```

The orchestrator initializes its required submodules on first run, creates `kernel_opt_<name>/` under `--workspace` or the current directory, spawns fresh clean sessions per iteration, and finalizes a directly submittable SOL-ExecBench output after a passing run.

## 3. Inspect Outputs

Each optimization workspace records the full optimization trail:

- `kernel.py`: current best kernel at Git `HEAD`
- `memory/v<N>.json`: structured iteration records
- `plans/`: evidence-based optimization plans
- `profiles/`: profiler artifacts and extracted bottleneck evidence
- `submission.json`: SOL-ExecBench submission output, when using the orchestrated SOL route
