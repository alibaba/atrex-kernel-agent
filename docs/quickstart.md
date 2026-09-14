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
automatically. The Supervisor Wiki corpus is required; NCU Skill dependencies are initialized only
when selected. Agent resources are controlled by these optional flags:

```bash
# Add these to orchestrator/optimize.py; explicit selection replaces default optional Skills.
--agent-skill ncu-report-skill
# Other choices: autonomous-gpu-kernel-timeline, ppu-acu-joint-profile
--no-agent-skills             # mandatory gpu-measurement + runtime-records + KernelWiki only
--agent-reference-projects    # expose installed source references read-only
```

`gpu-measurement` (GPU measurement), `runtime-records`, and `KernelWiki` are always mounted in the Agent workspace,
including with `--no-agent-skills` or an explicit Skill selection. `autonomous-gpu-kernel-timeline`
is optional and enabled by default. `gpu-measurement` documents GPU operations through the existing
HTTP client. `runtime-records` documents historical result/source lookup, Journal operations, and
Episode reports, including request/response examples and their ID linkage. Prompts only state the
required behavior and point to these Skills. They add no execution layer or host-level Skill installation.
Reference projects and the entire `reference/` source
directory are hidden; Agent `tools/` contains only the self-contained `sandbox.py` HTTP client. Existing
managed symlinks are updated. A real, unmanaged resource directory causes an explicit migration
error rather than deleting its contents. On PPU, opting into reference projects initializes the
t-head references over SSH (`git@github.com:t-head/...`), requiring repository access.

Agent `memory/` exposes read-only canonical `vN.json` reports. Full promotion audits are private at
`<campaign-parent>/.atrex-supervisor-runtime/<campaign-key>/promotions/long_horizon_eNNNN.json`.
Existing committed audits are copied there automatically; their Git history is preserved but their
files are hidden in the Agent sandbox. No manual migration or additional Agent action is needed.

The CLI starts with `HOME=$PWD=/home/agent/workspace`. Full conversations and Provider usage,
including exported subagent transcripts, are captured automatically outside the Agent view:

```text
<campaign-parent>/.atrex-supervisor-runtime/<campaign-key>/workspaces/<scope>/sessions/
└── run-<uuid>/
    ├── conversation.jsonl
    ├── token-usage.json
    └── provider/                 # stdout, stderr, and native transcript deltas
```

These files update during execution, not just at Episode completion. Each retry has its own capture;
the usage file identifies its Episode/invocation through `context`. Missing Provider counts are
marked unavailable or partial; Qoder credits are not converted to tokens. For reconciliation rules,
see [Session capture](../long_horizon/README.md).

Auxiliary sessions have their own bounded workspace projection. A policy Reviewer receives
read-only `review_request.json` and `candidate/`, and returns only `dependency_review.json`.
Problem authoring receives its staged reference inputs and returns `agent_problem.json`;
baseline correctness and exit reviewers receive `context/` or `candidate/` plus crash metadata,
and return `correctness_review.md` or `resume.json`. These roles do not expose the Campaign's
private ledger, and their outputs are validated by the Supervisor before use.

Episode plans live in Runtime Direction records, and conclusions live in Experiments referencing
Gateway Records. Separate plan drafts, profile-analysis files, and phase markers are not required.
Legacy plan-generation and memory-writing tools are not installed for the Agent; Supervisor
correctness, performance, and policy checks remain independent of its final report.

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
memory points to the stable source SHA. There is no Setup Agent fallback: native inputs require
`reference.py`, `input.py`, and `shapes.json` within an Atrex-Bench checkout containing
`scripts/run_eval.py` and `src/atrex_bench`; SOL inputs require `reference.py`, `definition.json`, and
`workload.jsonl`. Incomplete inputs are rejected before launch. Production public-contract authoring
and Framework Baseline sessions remain separate from this mechanical V0 initialization.

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
5. **Run isolated optimization episodes.** Each Episode explores a bounded number of Directions,
   one at a time, in a private Git branch and worktree. Every Episode uses maximum primary-Agent
   reasoning effort and the same research/implementation/validation loop, with no fixed Trial count.
   Direction and Experiment updates go through scoped Runtime
   tools: the Supervisor immediately stores them, validates the explicitly cited Gateway Record IDs,
   and makes prior Episode history queryable without exposing the private Journal files.
   All operations use `python3 tools/sandbox.py --kind <operation>`; see the CLI reference below.
6. **Verify and promote.** The Supervisor obtains an incumbent/candidate ABBA comparison from Runtime,
   reusing a matching completed record or measuring in an isolated GPU allocation. Production also
   applies its fail-closed policy review. Only a strict
   passing improvement is squash-promoted.
7. **Recover or finalize.** A restarted supervisor reopens the registered episode worktree with its
   intermediate state. The campaign stops on mechanical budgets or target utilization, summarizes
   canonical memory, and emits a directly consumable `submission.json` for SOL campaigns.
GPU evaluations and profiles run through `tools/sandbox.py` on `--sandbox-hardware`;
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
Its native rollouts, including exported child rollouts, are copied and read incrementally for
accounting. In Supervisor-managed sandbox sessions, `CODEX_HOME` points to `~/.codex`, backed by
session-scoped persistent storage. Non-sandbox trusted invocations retain the temporary-Home ledger
fallback. Missing native accounting is explicitly reported rather than inventing per-response
counters. Selected
Skills stay in the campaign-scoped `.agents/skills/` tree, so the user's global
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
--token-budget N                 Hard token cap across episode turns (0 = no cap)
--agent-cli CLI                  claude (default), qodercli, codex, or pi
--long-reviewer-session REVIEWER Reuse one reviewer session across episodes (codex, qoder)
--v1-ask-codex / --no-v1-ask-codex                 Configure ask-codex for V1 (default: off)
--v1-ask-qoder / --no-v1-ask-qoder                 Configure ask-qoder for V1 (default: off)
--full-episode-ask-codex / --no-full-episode-ask-codex
                                                    Configure full ask-codex (default: on)
--full-episode-ask-qoder / --no-full-episode-ask-qoder
                                                    Configure full ask-qoder (default: on)
--optimization-mode MODE         leaderboard (default) or production
--framework DSL                  Explicit DSL; omit for automatic parallel dispatch
--framework-baseline MODE        auto (production only), always, or never
--framework-baseline-timeout S   Framework bring-up wall-clock budget (default: 10800)
--target-util PCT                Peak-utilization short-circuit (default: 90)
--problem-generation-timeout S   Public contract authoring session timeout (default: 1800)
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
--verify-repeats N               ABBA repeat pairs (default: 2)
--verify-run-timeout S           Evaluator budget per ABBA run (default: 120)
--min-improvement-pct PCT        Strict gain required in verification
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

### Sandbox and profiling inside an Agent Session

Within a running Campaign Agent Session, use the scoped HTTP facade for validation and profiling:

```bash
python3 tools/sandbox.py --kind run --hardware REMOTE_GPU --no-sync
python3 tools/sandbox.py --kind profile --hardware REMOTE_GPU --profile-source
python tools/sandbox.py --kind check --no-sync
python tools/sandbox.py --kind check --sanitize memcheck --no-sync
python tools/sandbox.py --kind profile --requirement 'package==1' \
  --deps-mode freeze_installed --no-sync
python tools/sandbox.py --kind check --requirement 'package==1' \
  --deps-mode no_deps --no-sync
python tools/sandbox.py --kind disassemble --format isa --no-sync
python tools/sandbox.py --kind env --env-gpu REMOTE_GPU --env-capabilities
python tools/sandbox.py --kind profile --profile-shape-id 0 \
  --kernel-name candidate_kernel --profile-source \
  --launch-skip 1 --launch-count 5 --no-sync
python tools/sandbox.py --kind run --mode correctness_only --no-sync
python tools/sandbox.py --kind run \
  --input-path scratch/custom-input.py \
  --shapes-path scratch/custom-shapes.json --no-sync
python tools/sandbox.py --kind run --mode full \
  --baseline-path scratch/baseline.py --comparison-repeats 2 --no-sync
python tools/sandbox.py --kind record-read \
  --record-id gateway-<timestamp>-<kernel-prefix>
python tools/sandbox.py --kind record-read \
  --record-id kernel-<timestamp>-<opaque-id> \
  --view gateway-records
python tools/sandbox.py --kind record-read \
  --record-id kernel-<timestamp>-<opaque-id> \
  --view source \
  --output-path scratch/restored-kernel.py
python tools/sandbox.py --hardware H20 --ssh user@gpu-host \
  --ssh-gpu 0 \
  --ssh-runtime-bind /opt/aka-venv --ssh-init 'source /opt/aka-venv/bin/activate' \
  --kind run --no-sync
```

Only code and evaluator/profile inputs cross the remote GPU boundary. Agents edit source locally;
Git metadata, commits, canonical memory, structured plans/analysis, and Gateway Records are
Supervisor-owned. Agent workspaces require Linux Bubblewrap and contain no `.git`; `auto` fails closed
when this isolation is unavailable. A candidate report supplies `selected_experiment_id` and leaves
the matching measured source in `kernel.py`; it does not supply Git commit IDs.

These commands fail closed outside a Supervisor-managed Agent Session because no Runtime
capability is present. Operators configure the endpoint on `orchestrator/optimize.py`; they do not
pass credentials or endpoint overrides through the Agent facade.

`--requirement SPEC` is repeatable for `profile`, `check`, and `disassemble`. `--deps-mode freeze_installed`
reuses compatible packages already present in the worker image, while `--deps-mode no_deps`
installs only the explicitly named requirements. Both policies are scoped to that Gateway job.

For `--kind run`, `--mode correctness_only` performs correctness without timing. A custom public
input constructor may be supplied with `--input-path`; a custom Shape JSON object may be supplied
with `--shapes-path`, independently or together. Paths must name regular UTF-8 files inside the
Agent workspace. The constructor must define `_make_inputs(**input_kwargs)`, Shape IDs must be
integer strings, and each Shape value must be an object. These calls are exploratory and cannot
support a claimed measured improvement without a standard full evaluator result.

Profile accepts an exact `--kernel-name` or `--kernel-regex`, source correlation, launch skip/count,
and one opaque evaluator `--profile-shape-id`. An Agent-requested `--baseline-path` comparison runs
the baseline and current `kernel.py` in alternating AB/BA order within the same allocation for each
Shape; it records evidence but does not replace the orchestrator's promotion decision.

### Agent-facing errors

Runtime validation failures return a compact error rather than a full schema or CLI usage dump:

```json
{
  "ok": false,
  "repairable": true,
  "error": {
    "code": "invalid_arguments",
    "message": "argument --launch-count: invalid int value: 'oops'",
    "next_action": "Correct the indicated input before retrying. Use python3 tools/sandbox.py --kind OPERATION --help for CLI options and the session instructions for JSON fields."
  }
}
```

`repairable` means the Agent can correct its input or Journal state, not that replaying the same
request is safe. JSON field mismatches include `missing_fields`, `unexpected_fields`, and the small
set of `allowed_fields`. Direction conflicts include the blocking ID; the advancement limit explains
how to resume an already-started direction or finish the Episode.

Missing/corrupt private Journal data and missing Supervisor dependencies are infrastructure blockers,
not Agent repair tasks. An unhandled Runtime failure returns an `error_id` correlated with the private
service log and warns that the operation's outcome is unknown. Verify existing records before replaying
mutations; never repair private files, install Supervisor dependencies, or change credentials.

Terminal Gateway job failures retain `error_class`, `reason`, a bounded `message`, and `next_action`.
Failed Dev jobs preserve this information both immediately and through `record-read`, including when
Agate has no `result` or only partial logs. Hidden case inputs and raw error `details` remain private.
This changes error reporting, not the configured Gateway retry policy. HTTP errors are printed to
stderr; Journal/Gateway result errors appear on stdout with a nonzero exit status. The CLI does not
automatically replay either.

### Unified Agent CLI

GPU calls, Wiki queries, record queries, Journal operations, and terminal reports share one entry point:

```bash
python3 tools/sandbox.py --kind run --no-sync
python3 tools/sandbox.py --kind wiki-query "Target B200, runtime sm_100. BF16 Triton RMSNorm reaches 75% DRAM peak. Which fusion strategies apply?" --brief
python3 tools/sandbox.py --kind wiki-query --file scratch/research-request.txt
python3 tools/sandbox.py --kind wiki-search --arch sm_100 --dsl triton --coverage
python3 tools/sandbox.py --kind wiki-hardware --product b200 --field peak_compute.bf16.dense
python3 tools/sandbox.py --kind record-read --record-id gateway-...
python3 tools/sandbox.py --kind record-read --record-id kernel-... --view gateway-records
python3 tools/sandbox.py --kind record-read --record-id kernel-... --view source --output-path scratch/old.py
python3 tools/sandbox.py --kind update-direction --request-file scratch/direction.json
python3 tools/sandbox.py --kind record-experiment --request-file scratch/experiment.json
python3 tools/sandbox.py --kind list-directions --output-path scratch/directions.json
python3 tools/sandbox.py --kind load-direction --record-id direction_...
python3 tools/sandbox.py --kind list-experiments --output-path scratch/experiments.json
python3 tools/sandbox.py --kind load-experiment --record-id experiment_...
python3 tools/sandbox.py --kind episode-report --request-file scratch/episode-report.json
```

`wiki-query` accepts natural language, `wiki-search` provides structured experience retrieval,
and `wiki-hardware` looks up exact hardware facts. Their query flags and result formats are
preserved; use `--kind <operation> --help` for details. Wiki queries use the same scoped Runtime
capability as other operations. The Supervisor owns store selection, credentials, and output
bounds; the Agent cannot override the store or retain the private query workspace. The selected
KernelWiki Skill documents these commands; there is no separate `gpu-wiki/` directory in the sandbox.

Natural-language query events persist under
`<campaign-parent>/.atrex-supervisor-runtime/<campaign-key>/wiki-profile/`
(`run.json` and `raw/query_events/<date>/*.json`). The Supervisor fixes that path and strips it
from the Agent environment. Events survive Episode worktree cleanup and Runtime restarts;
query results and `query_id` still return normally to the Agent. At Campaign completion,
collect the private Campaign's `trace-retention-manifest.json` relative to that private root,
in addition to the workspace manifest. No query-event copies are written back to the workspace.
Existing `.gpu_wiki_profile` files are left untouched and are not included as trusted private
evidence; new queries do not append to them.

Each Experiment cites the Gateway results used in its analysis:

```json
{
  "gateway_record_ids": ["gateway-...", "gateway-..."]
}
```

The Supervisor verifies that all cited Gateway records are visible in the current Campaign and
stores exactly those references. They may refer to different Kernels or operation types; no
before/after or Kernel identity fields are needed. Every Experiment, including `abandon_direction`,
needs at least one real Kernel-bound Gateway Record. Failed diagnostics may document blockers;
Env/Wiki responses or a transport error without a saved record are not evidence.
`record-read --record-id gateway-...` returns the
Kernel ID; use that ID with `--view source --output-path scratch/old.py` to retrieve the source.
IDs are placeholders here; use the actual IDs returned by Gateway calls or `record-read`. The complete
Direction, Experiment, and report formats are in the mandatory Agent Skill
[`runtime-records`](../orchestrator/agent_skills/runtime-records/SKILL.md), which the Episode prompt references.

Direction proposals optionally accept `relationship` (`retry`, `refinement`, `reimplementation`,
`correction`, `port`, `combination`), `derived_from_direction_ids`, `derived_from_experiment_ids`, and
`supersedes_direction_id`. Explain the link in the existing `rationale`. Parent lists allow up to 32
unique visible IDs each; an Experiment implies its owning Direction as a parent. A relation needs a
parent, a combination needs two distinct parents, and only correction may supersede a parent. The
Supervisor validates before writing, leaves parent lifecycle untouched, and rejects later ancestry
rewrites. Continue an unchanged unfinished Direction with its ID; propose a new one for a revised
hypothesis. Old Journals have no inferred ancestry.

`list-directions` and `load-direction` return declared ancestry, lifecycle `status` and an independent
`hypothesis_status` (`unresolved`, `supported`, `refuted`). Every closure (`complete`, `abandon`,
`block`, `defer`) explicitly selects 1–32 unique `supporting_experiment_ids` from the Direction's
visible history and declares its assessment. Supported/refuted requires a completed Gateway
observation in every selected Experiment; Runtime verifies bindings, not scientific truth.
`load-direction` also returns all `associated_experiment_ids`. Late Experiments may append to closed
Directions without changing their status or selected support. Restarting an unchanged closed Direction
resets the assessment to unresolved and clears selected support; old events remain immutable.
No in-progress Direction may remain at report handoff. Empty blocked/pivot reports are possible only
when no Direction needs closing. Without any actual evidence, closure is blocked; do not fabricate it.
Old unmeasured notes remain readable but cannot support a new closure. Missing historical
assessments mean unresolved; old automatically associated IDs are not explicit hypothesis support.
Follow referenced Experiment IDs with `load-experiment` to retrieve Gateway Record references.
Relationships are Agent interpretations, not causal proof of performance gains. This feature does
not add the main Runtime's Pool scheduler to AKA.

Evaluate responses and record reads omit `measurement_aggregation`; the three-call per-Shape median
policy for full Evaluate/ABBA and its full internal evidence are unchanged. All GPU job operations
(including correctness-only, Profile, Check, Disassemble, and Dev) reject exact completed duplicates
from this Episode or visible earlier Episodes with `duplicate_gateway_task` and the original
`gateway_record_id`. Use `record-read` to retrieve it. Dev compares the command and uploaded file
contents, not just kernel.py. This does not add repetitions to diagnostics or Dev, and does not
restrict environment queries or record reads. Journal writes return the new record ID, list
operations confirm the written file and item count, and load operations return the requested record.
Report validation failures are repairable: correct the request or Direction state and submit again.

## 3. Inspect Outputs

Each optimization workspace records the full optimization trail:

- `kernel.py`: current editable candidate; the Supervisor owns its committed version
- `memory/live.json`: Supervisor-side, ignored non-canonical progress; not mounted into the Agent
- `memory/v<N>.json`: canonical episode/version records
- `<private-campaign-root>/promotions/long_horizon_e<NNNN>.json`: full promotion audit, not Agent-visible

- `scratch/`: temporary requests and optional diagnostics, ignored by Git; empty at each new
  Episode, preserved when resuming the same interrupted Episode
- Runtime Journal: structured Directions/plans and Experiments/analysis, read through Runtime tools
- Gateway Records: exact Kernel snapshots and measurements, read through `--kind record-read`
- `.atrex_long_horizon/`: restart state, journals, handoffs, telemetry, and archived attempts
- `submission.json`: SOL-ExecBench submission output for SOL campaigns

`memory/v<N>.json` combines the terminal report, Journal, and authoritative outcome; it contains no
`git_commit_hash` or `candidate_commit`. Agents submit no Commit IDs. The Supervisor tracks source
provenance and recovery in Git and its internal state.
