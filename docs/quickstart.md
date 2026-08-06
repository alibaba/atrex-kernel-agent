# Quick Start

AKA exposes one supported execution path: the unattended, budget-bounded orchestrator in
`orchestrator/optimize.py`.

## Prerequisites

- `bash`
- `git`
- Python 3, `torch`, and `jq` on the coordinator host
- One coding runtime available on `PATH`: `claude`, `qodercli`, `codex`, or `pi`
- `agate` (`atrex-gateway-client`) configured with gateway URL and credentials
- The selected gateway environment must provide the workload's framework and GPU stack
- NVIDIA workers: `ncu`, wrapped by `tools/profile_nvidia.sh`
- AMD workers: `rocprofv3`, wrapped by `tools/profile_kernel.sh`

The orchestrator verifies required submodules and `jq` before starting. Missing required submodules
are initialized automatically; the large `reference-projects/` collection remains optional.

## 1. Clone the Repository

```bash
git clone https://github.com/alibaba/atrex-kernel-agent.git
cd atrex-kernel-agent
```

`--op-dir` supports two evaluator-owned layouts:

- SOL-ExecBench: `reference.py`, `definition.json`, and `workload.jsonl`.
- Native Atrex-Bench: `reference.py` and `shapes.json`, inside a checkout containing
  `scripts/run_eval.py` and `src/atrex_bench`; `input.py`, `metadata.json`, `roofline.json`, and
  `valid.py` are copied when present.

The orchestrator never treats operator inputs as editable candidate files.

## 2. Run the Orchestrated Loop

Run a single-operator campaign directly against a SOL-ExecBench op directory containing `definition.json`, `reference.py`, and `workload.jsonl`:

```bash
python orchestrator/optimize.py \
    --op-dir /path/to/sol-execbench/op \
    --platform TARGET_GPU --sandbox-hardware REMOTE_GPU --framework CuteDSL \
    --agent-cli qodercli \
    --max-iters 20 --token-budget 8000000 --target-util 90
```

The orchestrator initializes its required submodules on first run, creates a flat
leaderboard workspace named `kernel_opt_<name>_<framework>_<platform>/` under `--workspace` or
the current directory, and runs each canonical version as an isolated Long Horizon episode. One
episode may contain many related
profile/edit/validate cycles; its candidate is promoted only after independent same-allocation ABBA
verification. GPU evaluations and profiles run through `tools/sandbox.py` on `--sandbox-hardware`; `memory/`,
episode journals, worktrees, and Git stay local. It finalizes a directly submittable SOL-ExecBench output
after a passing run. Omit `--agent-cli` to use Claude, or pass `--agent-cli qodercli` after authenticating
with `qodercli status`. To use Codex, authenticate with `codex login status` and pass
`--agent-cli codex`:

```bash
python orchestrator/optimize.py \
    --op-dir /path/to/sol-execbench/op \
    --platform TARGET_GPU --sandbox-hardware REMOTE_GPU --framework Triton \
    --agent-cli codex --max-iters 20 --token-budget 8000000
```

Each Codex episode starts with `codex exec --json`; bounded handoff recovery resumes that same thread.
The orchestrator installs the required
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

Pi runs in JSON mode with one unique session per optimization episode. The orchestrator trusts
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
Cuda; AMD dispatches Triton and FlyDSL; unknown hardware dispatches Triton. Leaderboard workspaces use
flat names such as `/path/to/runs/kernel_opt_<name>_triton_h20`; production workspaces append
`_production`. `--max-iters` and `--token-budget` apply independently to each framework campaign.
Passing `--framework` selects one campaign but keeps the same mode-specific naming convention.

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
the workspace, injects it into every episode, rejects violating candidates, and refuses to
package a non-compliant final candidate. Production runs use a separate
`kernel_opt_<name>_<framework>_<platform>_production` workspace and cannot accidentally resume a
leaderboard campaign.

With the default `--framework-baseline=auto`, production inserts one dedicated framework bring-up
session after V0. It validates the base seed plus five additional seeds and pins the resulting V1
for all workload buckets. Use `--framework-baseline=always` to enable the same stage in leaderboard
mode, or `never` to seed optimization directly from V0.

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
agate executes jobs; evaluations and profiles still go through `tools/sandbox.py`, while `memory/`, plans, edits,
and Git remain workspace-local. `--platform` and the gateway's hardware selector are not name-validated:
inventory data may be aliased or desensitized, so runtime architecture probing drives automatic framework
selection.

## 3. Inspect Outputs

Each optimization workspace records the full optimization trail:

- `kernel.py`: current best kernel at Git `HEAD`
- `memory/v<N>.json`: canonical episode/version records
- `memory/long_horizon_e<NNNN>.json`: promoted-episode evidence
- `plans/`: evidence-based optimization plans
- `profiles/`: profiler artifacts and extracted bottleneck evidence
- `.atrex_long_horizon/`: restart state, journals, handoffs, telemetry, and archived attempts
- `dispatch_signatures.json`, `workload_buckets.json`, `aggregate_dispatch.json`, and
  `aggregation_state.json`: workload coordination provenance when bucketing is enabled
- `submission.json`: SOL-ExecBench submission output for SOL campaigns
