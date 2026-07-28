# Quick Start

AKA ships two independent ways to run the same profile-driven workflow. Use the interactive Skill route when you want to drive optimization inside a coding session, or the orchestrated route when you want an unattended, budget-bounded run.

## Prerequisites

- `bash`
- `git`
- A compatible coding runtime installed
- `agate` (`atrex-gateway-client`) configured with gateway URL and credentials
- NVIDIA profiling: `ncu`, wrapped by `tools/profile_nvidia.sh`
- AMD profiling: `rocprofv3`, wrapped by `tools/profile_kernel.sh`

Route-specific prerequisites:

- Route 1 requires `jq` for `install.sh`.
- Route 2 requires Python 3, `torch`, and either `claude` or `qodercli` available on `PATH`.

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
    --platform TARGET_GPU --sandbox-hardware REMOTE_GPU --framework CuteDSL \
    --agent-cli qodercli \
    --max-iters 20 --token-budget 8000000 --target-util 90
```

The orchestrator initializes its required submodules on first run, creates a flat
`kernel_opt_<name>_<framework>_<platform>/` workspace under `--workspace` or the current directory, and
spawns fresh clean sessions per iteration. GPU tests and profiles run through `tools/sandbox.py` on
`--sandbox-hardware`; `memory/` and Git stay local. It finalizes a directly submittable SOL-ExecBench output
after a passing run. Omit `--agent-cli` to use Claude, or pass `--agent-cli qodercli` after authenticating
with `qodercli status`.

Omit `--framework` to run every framework supported by the detected GPU concurrently:

```bash
python orchestrator/optimize.py \
    --op-dir /path/to/sol-execbench/op \
    --platform TARGET_GPU --sandbox-hardware REMOTE_GPU \
    --workspace /path/to/runs --max-iters 20
```

The runtime architecture is authoritative for vendor selection. NVIDIA dispatches Triton, CuteDSL, and
Cuda; AMD dispatches Triton and FlyDSL; unknown hardware dispatches Triton. Workspaces use flat names such
as `/path/to/runs/kernel_opt_<name>_triton_h20`, and `--max-iters`/`--token-budget` apply independently to
each framework campaign. Passing `--framework` selects one campaign but uses the same flat
`kernel_opt_<name>_<framework>_<platform>` naming convention.

The default `--optimization-mode leaderboard` retains the existing permissive workflow: third-party kernel
libraries and evidence-backed framework changes are allowed. Use production mode for a deployable,
framework-pure implementation:

```bash
python orchestrator/optimize.py \
    --op-dir /path/to/sol-execbench/op \
    --platform TARGET_GPU --sandbox-hardware REMOTE_GPU \
    --optimization-mode production --framework Triton \
    --workspace /path/to/runs --max-iters 20
```

Production mode may omit `--framework`; like leaderboard mode, it auto-dispatches all frameworks supported
by the detected hardware. Every child receives one explicit framework constraint. V0 remains a PyTorch
correctness baseline, while every accepted optimization commit must implement the GPU computation exclusively
in that child's framework and must not call or depend on third-party kernel/operator libraries. The orchestrator writes the policy into
the workspace, injects it into every clean session, mechanically reverts violating commits, and refuses to
package a non-compliant final candidate. A workspace cannot be resumed under a different mode/framework.

To use the same gateway interface on a local GPU, start the bundled community scheduler. It has no
third-party Python dependencies:

```bash
python tools/local_gateway.py serve \
  --host 127.0.0.1 --port 8000 \
  --state-dir .atrex-local-gateway
```

The default single worker executes jobs FIFO, so concurrent optimizer requests queue instead of contending
for the GPU. `agate dev`, `agate get/jobs/cancel`, long polling, environment discovery, and
`tools/sandbox.py` use the same HTTP shapes as atrex-gateway. See [local_gateway.md](local_gateway.md) for
the exact compatibility surface.

This is interface compatibility, not process isolation: submitted code runs directly as the server user.
Bind it to localhost and submit trusted code only. The worker inherits the server process's Python/toolchain
environment, so install `torch`, Triton, and any kernel DSL needed by the workload into that environment.

Then select the localhost endpoint and the server's `local` GPU alias:

```bash
python orchestrator/optimize.py \
    --op-dir /path/to/sol-execbench/op \
    --platform H20 --framework Triton \
    --sandbox-hardware local \
    --sandbox-url http://127.0.0.1:8000 \
    --max-iters 20
```

`--sandbox-url` and `--sandbox-profile` are mutually exclusive. The localhost mode changes only where
agate executes jobs; tests and profiles still go through `tools/sandbox.py`, while `memory/`, plans, edits,
and Git remain workspace-local. `--platform` and the gateway's hardware selector are not name-validated:
inventory data may be aliased or desensitized, so runtime architecture probing drives automatic framework
selection.

## 3. Inspect Outputs

Each optimization workspace records the full optimization trail:

- `kernel.py`: current best kernel at Git `HEAD`
- `memory/v<N>.json`: structured iteration records
- `plans/`: evidence-based optimization plans
- `profiles/`: profiler artifacts and extracted bottleneck evidence
- `submission.json`: SOL-ExecBench submission output, when using the orchestrated SOL route
