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

- `install.sh` automatically installs `jq` with an available package manager when needed.
- Route 2 requires Python 3, `torch`, and one of `claude`, `qodercli`, `codex`, or `pi` available on `PATH`.

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
with `qodercli status`. To use Codex, authenticate with `codex login status` and pass
`--agent-cli codex`:

```bash
python orchestrator/optimize.py \
    --op-dir /path/to/sol-execbench/op \
    --platform TARGET_GPU --sandbox-hardware REMOTE_GPU --framework Triton \
    --agent-cli codex --max-iters 20 --token-budget 8000000
```

Each Codex iteration uses `codex exec --json --ephemeral`. The orchestrator installs the required
optimization and Humanize skills into the campaign's repository-scoped `.agents/skills/` tree; it does
not modify `${CODEX_HOME}`. Optional Codex config overrides use a JSON object or an array of literal
`key=value` values:

```bash
export ATREX_CODEX_SESSION_SETTINGS='{"model":"gpt-5.6-sol","model_reasoning_effort":"xhigh"}'
```

These entries become repeatable `codex exec -c key=value` arguments. The default Codex reasoning effort
is `max`; a value supplied through `ATREX_CODEX_SESSION_SETTINGS` appears later and overrides it.

To use Pi, authenticate/configure a model with Pi first, then select it as the backend:

```bash
pi --list-models
export ATREX_PI_SESSION_SETTINGS='{"provider":"anthropic","model":"claude-opus"}'  # optional
python orchestrator/optimize.py \
    --op-dir /path/to/sol-execbench/op \
    --platform TARGET_GPU --sandbox-hardware REMOTE_GPU --framework Triton \
    --agent-cli pi --max-iters 20 --token-budget 8000000
```

Pi runs in JSON mode with one unique persisted session per optimization attempt. The orchestrator trusts
the generated campaign workspace for that run so Pi can load repository-scoped `.agents/skills`, while
leaving provider credentials in Pi's normal auth/config files. `ATREX_PI_SESSION_SETTINGS` accepts only
`provider` and `model`; API keys are never added to process arguments.

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

## 2C. Run Offline Teacher Distillation

Use this opt-in mode to generate a 1→10 optimization trajectory against an independently measured, same-framework expert implementation. It is not the default online optimization route.

```bash
python3 orchestrator/optimize.py \
    --campaign-mode teacher-distill \
    --op-dir /path/to/operator \
    --teacher-solution /path/to/teacher_solution \
    --platform H20 --arch sm_90 --framework CuteDSL \
    --no-workload-bucketing \
    --sandbox-hardware REMOTE_GPU \
    --agent-cli pi \
    --max-iters 30 --max-stall 5
```

The Teacher bundle must contain `kernel.py`, `solution.json`, and `provenance.json`; it must use the same framework and architecture and pass full correctness before Candidate setup. V0 remains the reference implementation, V1 is an Agent-authored naive framework kernel, and V2+ follows the normal profile-driven loop. Success requires both aggregate and per-shape parity followed by same-allocation A-B-B-A verification.

Teacher mode is `hidden-audited`, not security-grade isolation. It disables workload bucketing, layer decomposition, automatic framework dispatch, reference-project access, and public-web fallback. It writes evidence-backed drafts under private campaign state but never modifies canonical `gpu-wiki/`. See [teacher-distill.md](teacher-distill.md) for the complete contract and threat model.

## 3. Inspect Outputs

Each optimization workspace records the full optimization trail:

- `kernel.py`: current best kernel at Git `HEAD`
- `memory/v<N>.json`: structured iteration records
- `plans/`: evidence-based optimization plans
- `profiles/`: profiler artifacts and extracted bottleneck evidence
- `submission.json`: SOL-ExecBench submission output, when using the orchestrated SOL route
