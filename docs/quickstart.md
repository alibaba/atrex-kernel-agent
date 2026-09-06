# Quick Start

AKA exposes one supported execution path: the unattended, budget-bounded orchestrator in
`orchestrator/optimize.py`. For interactive use, the recommended launch method is to ask a coding
agent in this repository to translate the task into that command and start the campaign.

## Prerequisites

- `bash`
- `git`
- Python 3 and `torch` on the coordinator host
- One coding runtime available on `PATH`: `claude`, `qodercli`, `codex`, or `pi`
- A sandbox execution environment containing the workload's framework and GPU stack
- For SSH execution: OpenSSH `ssh` and `scp` on the coordinator; Bash, Python 3, `tar`, `base64`,
  Bubblewrap (`bwrap`), unprivileged user namespaces, and accessible GPU device nodes on the remote
  host. Physical NVIDIA assignment also requires `nvidia-smi`. Authentication must be
  non-interactive for detached recovery.
- NVIDIA workers: `ncu`, wrapped by `tools/profile_nvidia.sh`
- AMD workers: `rocprofv3`, wrapped by `tools/profile_kernel.sh`

The orchestrator verifies required submodules before starting and initializes missing ones
automatically; the large `reference-projects/` collection remains optional. On PPU hardware the
t-head projects in that collection are the only PPU-specific implementation references available, and
they clone over SSH (`git@github.com:t-head/...`), so initialize them with an SSH key that can reach
that org. `reference-projects/README.md` indexes every project by vendor, DSL, and operator.

The repository-native `gen-plan` skill freezes a concrete candidate proposal, then requests the
configured independent, read-only Codex and Qoder reviews against the same proposal and bounded
repository evidence. V1, fast episodes, and full episodes each have independent Codex and Qoder
switches. V1 and fast reviewers default off; full reviewers default on. A Codex- or Qoder-owned
episode performs an enabled matching review in the current session to avoid recursion. The campaign
probes a reviewer only when
it is first enabled for an episode mode, caches that decision under `.atrex_long_horizon/`, reuses it
after restarts, and never retries a reviewer that failed the probe. Reviews are non-persistent by
default; an optional campaign-private Codex reviewer thread may span episodes. Disabled and
unavailable reviewers are recorded explicitly without discarding available reviews. Enabled
external consultations always run with maximum reasoning effort, independently of the primary
episode's configured effort.

## 1. Clone the Repository

```bash
git clone https://github.com/alibaba/atrex-kernel-agent.git
cd atrex-kernel-agent
```

`--op-dir` supports two evaluator-owned layouts:

- SOL-ExecBench: `reference.py`, `definition.json`, and `workload.jsonl`.
- Native Atrex-Bench: `reference.py`, `input.py`, and detailed `shapes.json`, inside a checkout
  containing `scripts/run_eval.py` and `src/atrex_bench`. An optional `agent_problem.json` may provide
  the generalized public contract using schema `atrex.agent_problem.v1`.

Production native campaigns never expose detailed shapes to baseline or optimization sessions. If
`agent_problem.json` is supplied, AKA validates and copies it directly. Otherwise a separate clean AKA
preprocessing session using the configured `--agent-cli` at maximum reasoning effort reads
`reference.py`, `input.py`, and the evaluator-owned detailed shapes, derives the public
`agent_problem.json`, validates that its development cases do not duplicate evaluator cases, and
persists only that contract in the campaign workspace. Exact shapes and evaluator metadata are then
injected privately during sandbox evaluation. Canonical memory retains real per-shape latency under
opaque ids; set `PROFILE_SHAPE_ID` to one of those ids to profile that real shape privately.

Leaderboard mode always preserves legacy exact-shape behavior, even when the source operator also
contains `agent_problem.json`; sandbox private-shape injection and generalized result masking are
production-only. The orchestrator never treats operator inputs as editable candidate files. Start a
fresh workspace when resuming an older production campaign that exposed exact shapes.

For native Atrex-Bench and SOL operators, V0 does not launch a coding Agent. The supervisor commits
the verbatim reference wrapper, runs exactly one official full-workload base-seed evaluator, writes
README/memory/report programmatically, and records measurement metadata in a second commit whose
memory points to the stable source SHA. A setup Agent is retained only for derived legacy inputs.

## 2. Launch the Orchestrated Loop

### Start with a coding agent (recommended)

Open Claude Code, Codex, or Qoder in the repository and provide a concrete task prompt. For example:

```text
Use AKA's orchestrator/optimize.py to start one optimization task for atrex-bench/xx. Put the workspace under ~/aka-opt, set the platform to H20, use the local sandbox, use claude as the Agent CLI, set max-iters to 300, specify cuda as the framework, and run in production mode.
```

The coding agent should resolve the requested values into `orchestrator/optimize.py` arguments,
verify the local prerequisites, and launch that command. This prompt-driven path is a convenience
layer over the same orchestrator, not a separate optimization workflow.

### Run directly

Run a single-operator campaign directly against a SOL-ExecBench op directory containing `definition.json`, `reference.py`, and `workload.jsonl`:

```bash
python orchestrator/optimize.py \
    --op-dir /path/to/sol-execbench/op \
    --platform TARGET_GPU --sandbox-hardware REMOTE_GPU --framework CuteDSL \
    --agent-cli qodercli \
    --max-iters 20 --token-budget 8000000 --target-util 90
```

### Isolated OpenSSH GPU host

Use a dedicated, low-privilege OpenSSH account or an alias from `~/.ssh/config`. Authentication,
ports, jump hosts, and host-key policy remain OpenSSH's responsibility. The account needs permission
to run `bwrap` and access only the intended GPU devices; do not attach cloud credentials or shared
service secrets to it. Runtime trees outside `/usr` must be exposed explicitly as read-only binds:

```bash
python orchestrator/optimize.py \
    --op-dir /path/to/sol-execbench/op \
    --platform H20 --sandbox-hardware H20 --framework Triton \
    --sandbox-ssh user@gpu-host \
    --sandbox-ssh-gpu 0 \
    --sandbox-ssh-runtime-bind /opt/aka-venv \
    --sandbox-ssh-init 'source /opt/aka-venv/bin/activate' \
    --environment-poll-interval 60 \
    --workspace /path/to/runs --max-iters 20
```

`--sandbox-ssh-gpu` is required and selects one physical NVIDIA index. The runner resolves that index
to its GPU UUID, exposes only its device node plus common driver control nodes, and exports UUID-based
`CUDA_VISIBLE_DEVICES`. MIG-enabled GPUs and MIG/UUID selectors fail closed because their capability
nodes are not assigned yet. SSH mode also requires an explicit `--framework`; automatic parallel
framework dispatch is rejected, and multi-shape ABBA batches are serialized on the assigned card.

`--sandbox-ssh-runtime-bind REMOTE_PATH[=SANDBOX_PATH]` is repeatable. A single path preserves its
location; the `source=destination` form can mount it elsewhere. The bind is read-only. For example, a
venv below a hidden login home can be exposed at its original path with
`--sandbox-ssh-runtime-bind /home/gpu/aka/.venv`, or remapped when it is relocatable.
Broad system/home roots and credential directories are rejected on the source side. A source below
`/home` must be a conventional `.venv`/`venv` root or a direct child of a Conda `envs` directory.
Before each execution, the remote host resolves every source symlink; the resolved target must still
pass the same denylist and be a directory.

`--sandbox-ssh-init` defaults to empty and runs inside the isolated namespace. The default health
command checks PyTorch GPU allocation, arithmetic, synchronization, and device properties;
override it when the remote stack uses a different runtime. The optimizer also checks the selected
framework's installed tooling and, for SOL operators, the evaluator interpreter and dtype mapping.
These workspace-independent checks run before seeding (even with `--arch`), after failed commands,
and during recovery polling. They never import candidate code. Native operator contracts and
complete workload coverage still require real evaluation; preflight is not a replacement for it.
Avoid putting credentials in either shell command. `--sandbox-ssh` is mutually exclusive
with `--sandbox-url` and `--sandbox-profile`.

Each sandbox call uploads its explicit input allowlist to a new `/tmp/atrex-sandbox.*` directory,
runs the requested evaluator or profiler inside mandatory Bubblewrap PID/IPC/UTS/network namespaces,
downloads only requested `--sync` artifacts, and removes the remote directory. The namespace has no
network, host home, inherited environment, or writable host filesystem; it sees only minimal read-only
system paths, configured runtime binds, the assigned GPU device node, and its writable job directory. There is no
unisolated fallback. A portable Python watchdog enforces `--sandbox-timeout` even when GNU `timeout`
is absent.

When preflight or SSH transport fails (including scp upload/download timeout), or a failed GPU command is followed by a failed health probe, the sandbox
writes a private environment marker and returns temporary-failure status 75. The supervisor stops all
active Agent/framework process groups without treating the failure as a bad candidate. It then starts
`tools/monitor_optimize_tasks.py` detached. Recovery state is stored below
`<workspace>/.atrex_environment/<command-id>/`:

- `failure.json`: the current failure stage and bounded diagnostic;
- `cleanup-*.json`: remote workspaces that must be removed before restart;
- `restart.json`: exact argument-array and working-directory metadata, mode `0600`;
- `monitor.lock`, `monitor.pid`, and `monitor.log`: an OS advisory lock plus live poller status;
- `restart-child.lock`, `restart.pid`, `restart.primary.pid`, and `restart.log`: diagnostic wrapper,
  primary, and cleanup status during the supervised resume handoff; durable registry identities
  remain the process authority;
- `restart.ready` and `restart.ack`: the two-phase resume handshake; activation is accepted only
  after the primary observes its matching `active.json` and acknowledges it;
- `restart.exit.json` and `restart.complete.json`: the primary result and the later confirmation that
  its process group and registered sessions were cleaned up;
- `restarting.json` and `active.json`: initialization and ready-but-still-running ownership states;
- `stopped.json` and `stop-requests/*.json`: a persistent operator stop plus immutable concurrent
  requests that prevent an earlier resume from erasing a later stop;
- `restart-processes/<handoff-id>/*.json`: PID-reuse-safe wrapper, primary, and independent cleanup
  guardian identities for the optimizer and every controlled process session it starts;
- `recover.sh`: an idempotent manual way to clear `stopped.json` and start the same single-instance
  poller;
- `stop-recovery.sh`: the verified stop path for rollback.

The monitor probes every 60 seconds by default. One successful explicit GPU health check first drains
all `cleanup-*.json` work, then spawns the original optimizer argv in the original working directory,
and moves the failure marker through `restarting.json`. The campaign first publishes readiness; the
monitor changes the marker to `active.json`, and the primary must then acknowledge that exact handoff.
The monitor keeps supervising the active run until a durable exit and cleanup result arrives. A clean
zero exit archives recovery; a fully cleaned non-zero exit restores a retryable failure and returns to
health polling. An exit without cleanup completion waits only while a verified owner remains and then
fails closed for manual process verification. Cleanup or spawn failures likewise retain the marker.
If a monitor dies during `restarting.json`, a replacement monitor uses the child-owned advisory lock
and the persistent session-owner identities to adopt a live handoff or request that every registered
owner terminate its own process group before atomically restoring `failure.json`. Each wrapper, gated
primary, and separate-session cleanup guardian is registered before the actual command can start. The
guardian retains the inherited handoff lock through the completion commit and takes over same-group
cleanup if its wrapper dies. The wrapper reports primary status before cleaning same-group leftovers.
Protocol files are fsynced before atomic replacement, and critical directory-entry changes are
directory-fsynced. The handoff timeout starts from the explicit
`restart_handoff.started_at` value in the marker, never from a failure marker's older filesystem
timestamp. Resolved environment-only settings, including the polling interval, are replayed into the
child. PID files are diagnostic, removed by their matching owner, and never used as the lock or
process-identity authority.
The normal campaign resume path reuses its interrupted worktree and journal. Candidate compilation,
correctness, timeout (status 124), and even explicit status 255 do not trigger this path when the
independent health probe succeeds.

To roll back the SSH transport, run `STATE_DIR/stop-recovery.sh` (or
`python tools/monitor_optimize_tasks.py --state-dir STATE_DIR --stop`) and require a zero exit status
before changing transport. Stop first publishes an immutable request; the live monitor observes it
without signalling its diagnostic PID, or the stopper takes over through the advisory lock after the
monitor exits. It terminates every identity-verified recovery process group, restores a
durable failure marker when needed, leaves the persistent `stopped.json` tombstone in place, and
reports success only after no owned process remains. This includes an optimizer that has already
reached `active.json`. Resume waits for the lock whenever stop state is present and returns zero only
after clearing its locked snapshot; a later stop request always wins. Do not signal the diagnostic PID
from `monitor.pid` directly. Preserve the private recovery directory for
diagnosis, then relaunch the same command with `--sandbox-url` or `--sandbox-profile`. Candidate Git
state and canonical memory are transport-independent and require no rollback. Run `STATE_DIR/recover.sh`
to re-enable automatic recovery. To clear a recovered marker without restarting, run
`python tools/monitor_optimize_tasks.py --state-dir STATE_DIR --resume --once --no-restart` after
verifying any deferred remote cleanup.

### What happens after launch

1. **Resolve and isolate the campaign.** The orchestrator validates the operator, initializes
   required submodules, probes the runtime GPU architecture, and creates or resumes
   `kernel_opt_<name>_<framework>_<platform>/` below `--workspace` or the current directory.
2. **Prepare production inputs.** Native production campaigns validate a supplied
   `agent_problem.json` or derive one in a clean preprocessing session, then keep detailed evaluator
   shapes private.
3. **Establish V0.** The supervisor commits the evaluator-owned reference wrapper, runs one official
   full-workload base-seed evaluation, and records canonical `memory/v0.json` without launching a
   coding Agent.
4. **Establish V1 when enabled.** `--framework-baseline=auto` creates a self-contained
   framework-native V1 in production mode. When enabled, read-only reviewers provide bounded
   correctness guidance; the coding Agent implements and smoke-tests, while the supervisor owns full
   evaluation, policy review, memory, and the final commit.
5. **Run isolated optimization episodes.** Each episode owns one candidate direction in a private
   Git branch and worktree. By default, the first two episodes run five
   `plan -> implement -> evaluator` trials at maximum primary-Agent reasoning effort without
   profiling, multi-seed validation, or ABBA. Later episodes use the full
   profile/research/plan/edit/repair loop.
6. **Verify and promote.** Fast mode compares the fastest passing hash-matched trial with canonical
   incumbent memory. Full mode runs an independent incumbent/candidate ABBA comparison in one
   isolated GPU allocation. Production also applies its fail-closed policy review. Only a strict
   passing improvement is squash-promoted.
7. **Recover or finalize.** A restarted supervisor reopens the registered episode worktree with its
   intermediate state. The campaign stops on mechanical budgets or target utilization, summarizes
   canonical memory, and emits a directly consumable `submission.json` for SOL campaigns.
GPU evaluations and full-mode profiles run through `tools/sandbox.py` on `--sandbox-hardware`;
`memory/`, episode journals, worktrees, and Git stay local. `--platform` is required and names the
logical target.

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
non-duplicated invocation total while phase attribution is disabled. Optimization and
plan-generation skills stay in the campaign-scoped `.agents/skills/` tree, so the user's global
Codex installation is not modified. Optional Codex config overrides use a JSON object or an array of
literal `key=value` values:

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
in that child's framework. The supervisor sends every candidate to a separate read-only policy Agent for a
complete implementation and manifest review, without package-name allowlists: build/ABI/launch plumbing for
a self-authored kernel may be accepted, while prebuilt compute, alternate frameworks, PyTorch compute
fallbacks, hidden dispatch, and external implementation loading are rejected. The orchestrator writes the
policy into the workspace, injects it into every episode,
rejects violating candidates, and refuses to
package a non-compliant final candidate. Production runs use a separate
`kernel_opt_<name>_<framework>_<platform>_production` workspace and cannot accidentally resume a
leaderboard campaign.

With the default `--framework-baseline=auto`, production inserts one dedicated framework bring-up
session after V0. Native V1 receives a pre-seeded manifest and three latency-quantile smoke ids; the
supervisor first runs the enabled isolated Codex and Qoder correctness reviews over the bounded public
contract and immutable reference, concurrently when both are enabled. Reviewers nominate only from a
bounded local path catalog; the supervisor reconciles their choices and injects at most two exact reference
paths alongside the available reviews. V1 reads only that shortlist without recursively browsing siblings.
The reviews are cached for restart and never receive private shapes or write access to the candidate. The
coding Agent implements and smoke-tests only, without full evaluation, memory writing, or commits. The
supervisor then runs policy review in parallel with one combined full-workload evaluator that measures the
base seed and checks five additional seeds, writes memory, and pins V1. Use
`--framework-baseline=always` to enable the same stage in leaderboard mode, or `never` to seed
optimization directly from V0. A Triton campaign escalates to Gluon after three consecutive stalls
by default; once triggered, conversion retries until correctness and performance parity pass, and
later episodes remain in Gluon. This applies independently of leaderboard or production mode.

If the V1 coding Agent exits unexpectedly, the orchestrator takes a one-time local snapshot and starts
a read-only progress supervisor to write
`.atrex_long_horizon/framework_baseline/resume.json`. The progress supervisor tries the configured
Agent CLI, then Codex, then Qoder; it does not change the CLI used by the outer V1 implementation.
Rerunning the same command keeps the interrupted worktree and resumes V1 from this handoff.

### Common options

```text
--max-iters N                    Hard cap on canonical versions/episodes
--fast-episodes N                Fast post-baseline episodes (default: 2; 0 disables)
--token-budget N                 Hard token cap across episode turns (0 = no cap)
--agent-cli CLI                  claude (default), qodercli, codex, or pi
--long-reviewer-session REVIEWER Reuse one reviewer session across episodes (codex, qoder)
--v1-ask-codex / --no-v1-ask-codex                 Configure ask-codex for V1 (default: off)
--v1-ask-qoder / --no-v1-ask-qoder                 Configure ask-qoder for V1 (default: off)
--fast-episode-ask-codex / --no-fast-episode-ask-codex
                                                    Configure fast ask-codex (default: off)
--fast-episode-ask-qoder / --no-fast-episode-ask-qoder
                                                    Configure fast ask-qoder (default: off)
--full-episode-ask-codex / --no-full-episode-ask-codex
                                                    Configure full ask-codex (default: on)
--full-episode-ask-qoder / --no-full-episode-ask-qoder
                                                    Configure full ask-qoder (default: on)
--optimization-mode MODE         leaderboard (default) or production
--framework DSL                  Explicit DSL; omit for automatic parallel dispatch
--framework-baseline MODE        auto (production only), always, or never
--framework-baseline-timeout S   Framework bring-up wall-clock budget (default: 10800)
--target-util PCT                Peak-utilization short-circuit (default: 90)
--setup-timeout S                Legacy V0/problem-authoring session timeout (default: 7200)
--sandbox-hardware GPU           Sandbox hardware selector or alias
--sandbox-ssh [USER@]HOST        Direct OpenSSH GPU executor
--sandbox-ssh-gpu INDEX          Assigned physical NVIDIA GPU (required for SSH)
--sandbox-ssh-init COMMAND       Remote environment activation before jobs/probes
--sandbox-ssh-runtime-bind PATH  Read-only runtime path inside the SSH namespace (repeatable)
--sandbox-health-command COMMAND GPU health probe used for failure classification
--environment-poll-interval S    Recovery probe interval (default: 60)
--sandbox-timeout S              Remote command timeout, at most 600 seconds
--workspace DIR                  Campaign parent directory (default: current directory)
--max-stall N                    Stop after N unpromoted episodes (0 = disabled)
--convert-after N                Triton stalls before mandatory Gluon conversion (default: 3)
--handoff-resumes N              Same-thread incomplete-handoff recovery turns (default: 2)
--verify-repeats N               Full-mode ABBA repeat pairs (default: 2)
--verify-run-timeout S           Full-mode evaluator budget per ABBA run (default: 120)
--min-improvement-pct PCT        Strict gain required in fast or full verification
--arch ARCH                      Override runtime architecture detection
```

Run `python orchestrator/optimize.py --help` for the complete current interface. Some Qoder models
report zero token usage in stream JSON; in that case `--token-budget` cannot be enforced, so
`--max-iters` remains the hard campaign bound.

Optimization episodes have no wall-clock deadline: an episode runs until it publishes a terminal
handoff or its coding-agent process exits. `memory/live.json` exposes progress during a long active
episode, while canonical `memory/vN.json` is written only after the episode reaches a terminal state.
The supervisor validates that this numbered record is both parseable and committed at `HEAD` before
it advances campaign state, including failed, pivoted, blocked, and interrupted rounds.

### Direct sandbox and profiling

The sandbox boundary can also be used directly for validation and profiling:

```bash
python tools/sandbox.py --hardware REMOTE_GPU --no-sync -- python test_kernel.py --no-memory
python tools/sandbox.py --hardware REMOTE_GPU --sync profiles/v1 -- \
  bash tools/profile_nvidia.sh kernel.py --output-dir profiles/v1 --source
python tools/sandbox.py --hardware H20 --ssh user@gpu-host \
  --ssh-gpu 0 \
  --ssh-runtime-bind /opt/aka-venv --ssh-init 'source /opt/aka-venv/bin/activate' \
  --no-sync -- python test_kernel.py --no-memory
```

Only code and evaluator/profile inputs cross the sandbox boundary. Optimization memory, plans,
edits, and Git state remain on the coordinator.

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
