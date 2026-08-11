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
episode may contain many related profile/edit/validate cycles; its candidate is promoted only after
independent same-allocation ABBA verification. GPU evaluations and profiles run through
`tools/sandbox.py` on `--sandbox-hardware`; `memory/`, episode journals, worktrees, and Git stay
local. It finalizes a directly submittable SOL-ExecBench output after a passing run. `--platform` is
required and names the logical optimization target.

### Agent backends

Authenticate the selected coding runtime before starting a campaign:

```bash
claude auth status
qodercli status
codex login status
pi --list-models
```

Omit `--agent-cli` to use Claude. Provider-specific settings can be supplied through
`ATREX_CLAUDE_SESSION_SETTINGS`, `ATREX_QODER_SESSION_SETTINGS`,
`ATREX_CODEX_SESSION_SETTINGS`, or `ATREX_PI_SESSION_SETTINGS`;
`ATREX_SESSION_SETTINGS` remains the generic fallback.

To use Codex, pass `--agent-cli codex`:

```bash
python orchestrator/optimize.py \
    --op-dir /path/to/sol-execbench/op \
    --platform TARGET_GPU --sandbox-hardware REMOTE_GPU --framework Triton \
    --agent-cli codex --max-iters 20 --token-budget 8000000
```

Each Codex episode starts with `codex exec --json`; bounded handoff recovery resumes that same thread.
Its native rollout is read incrementally for token and marker accounting. Non-episode Codex
orchestrator phases use a fresh thread in an isolated temporary `CODEX_HOME` that links existing auth,
config, and skills; newly written rollout and state files stay there, and the directory is removed
after normalization or terminal-only fallback. The orchestrator uses `session_meta` only to recover
the exact workspace or thread when stdout omits it, verifies every available usage component against
`turn.completed.usage`, and records ledger or cleanup errors without failing the Agent run. If ledger
observation fails during an episode resume, consecutive cumulative stdout totals still provide a
non-duplicated invocation total while phase attribution is disabled. Optimization and Humanize skills
stay in the campaign-scoped `.agents/skills/` tree, so the user's global Codex installation is not
modified. Optional Codex config overrides use a JSON object or an array of literal `key=value` values:

```bash
export ATREX_CODEX_SESSION_SETTINGS='{"model":"gpt-5.6-sol","model_reasoning_effort":"xhigh"}'
```

These entries become repeatable `codex exec -c key=value` arguments. The default Codex reasoning effort
is `max`; a value supplied through `ATREX_CODEX_SESSION_SETTINGS` appears later and overrides it.

To use Pi, select it as the backend and optionally configure its provider and model:

```bash
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

### Multi-framework campaigns

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
Every campaign optimizes the complete workload set in one version line.

### Production mode

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
in that child's framework. Non-standard imports, declared dependencies, and library references are
reviewed by a separate read-only policy Agent: build/ABI/launch plumbing for a self-authored kernel may
be accepted, while prebuilt compute, alternate frameworks, hidden dispatch, and external implementation
loading are rejected. The orchestrator writes the policy into the workspace, injects it into every episode,
rejects violating candidates, and refuses to
package a non-compliant final candidate. Production runs use a separate
`kernel_opt_<name>_<framework>_<platform>_production` workspace and cannot accidentally resume a
leaderboard campaign.

With the default `--framework-baseline=auto`, production inserts one dedicated framework bring-up
session after V0. It validates the base seed plus five additional seeds and pins the resulting V1.
Use `--framework-baseline=always` to enable the same stage in leaderboard
mode, or `never` to seed optimization directly from V0. A production Triton campaign escalates to
Gluon after three consecutive stalls; once triggered, conversion retries until correctness and
performance parity pass, and later episodes remain in Gluon.

### Common options

```text
--max-iters N                    Hard cap on canonical versions/episodes
--token-budget N                 Hard token cap across episode turns (0 = no cap)
--agent-cli CLI                  claude (default), qodercli, codex, or pi
--optimization-mode MODE         leaderboard (default) or production
--framework DSL                  Explicit DSL; omit for automatic parallel dispatch
--framework-baseline MODE        auto (production only), always, or never
--framework-baseline-timeout S   Framework bring-up wall-clock budget (default: 10800)
--target-util PCT                Peak-utilization short-circuit (default: 90)
--iter-timeout S                 Wall-clock budget and worst-case handoff cadence for one episode
                                 (default: 5400)
--setup-timeout S                V0 setup session timeout (default: 7200)
--sandbox-hardware GPU           Gateway selector or alias
--sandbox-profile PROFILE        Optional pre/prod endpoint profile
--sandbox-url URL                Explicit endpoint URL
--sandbox-timeout S              Remote command timeout, at most 600 seconds
--workspace DIR                  Campaign parent directory (default: current directory)
--max-stall N                    Stop after N unpromoted episodes (0 = disabled)
--convert-after N                Triton stalls before mandatory Gluon conversion (default: 3)
--handoff-resumes N              Same-thread incomplete-handoff recovery turns (default: 2)
--verify-repeats N               ABBA repeat pairs (default: 2)
--verify-run-timeout S           Evaluator budget per ABBA run (default: 120)
--min-improvement-pct PCT        Strict ABBA gain required for promotion
--arch ARCH                      Override runtime architecture detection
```

Run `python orchestrator/optimize.py --help` for the complete current interface. Some Qoder models
report zero token usage in stream JSON; in that case `--token-budget` cannot be enforced, so
`--max-iters` remains the hard campaign bound.

Lowering `--iter-timeout` makes numbered canonical memory arrive more frequently, but it also pays
terminal-handoff and ABBA verification overhead more often. `memory/live.json` is independent of
that tradeoff and refreshes within an active episode.

### Local gateway

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

### Direct sandbox and profiling

The gateway transport can also be used directly for validation and profiling:

```bash
python tools/sandbox.py --hardware REMOTE_GPU --no-sync -- python test_kernel.py --no-memory
python tools/sandbox.py --hardware REMOTE_GPU --sync profiles/v1 -- \
  bash tools/profile_nvidia.sh kernel.py --output-dir profiles/v1 --source

# Same interface through the bundled local gateway
python tools/sandbox.py --hardware local --url http://127.0.0.1:8000 \
  --no-sync -- python test_kernel.py --no-memory
```

The gateway receives code and evaluator/profile inputs only. Optimization memory, plans, edits, and
Git state remain on the coordinator.

## 3. Inspect Outputs

Each optimization workspace records the full optimization trail:

- `kernel.py`: current best kernel at Git `HEAD`
- `memory/live.json`: ignored, non-canonical progress for the active Long Horizon episode
- `memory/v<N>.json`: canonical episode/version records
- `memory/long_horizon_e<NNNN>.json`: promoted-episode evidence
- `plans/`: evidence-based optimization plans
- `profiles/`: profiler artifacts and extracted bottleneck evidence
- `.atrex_long_horizon/`: restart state, journals, handoffs, telemetry, and archived attempts
- `submission.json`: SOL-ExecBench submission output for SOL campaigns
