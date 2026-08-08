# Architecture Design

## Overview

Atrex Kernel Agent is an orchestrated system for GPU kernel implementation, profiling, and
iterative optimization. The repository has one supported entry point: `orchestrator/optimize.py`.
It owns the optimization lifecycle and launches isolated Long Horizon episodes through Claude,
Qoder, Codex, or Pi.

Agent sessions propose and implement changes. The orchestrator remains authoritative for
budgets, state transitions, sandbox execution, correctness and performance gates, production
policy, rollback, aggregation, and final packaging.

Each canonical optimization version is one multi-experiment episode in a private Git worktree.
The internal `long_horizon/` engine supplies worktree isolation, journals, handoff recovery,
same-allocation ABBA verification, and squash promotion; it is not a second CLI.

## Design Goals

- **Mechanical control**: termination and acceptance are decided by code rather than Agent
  self-assessment.
- **Profile-driven optimization**: kernel changes must be supported by official profiler
  evidence.
- **Reproducible state**: Git HEAD is the incumbent kernel; structured memory and artifacts
  preserve the reasoning and measurements behind each attempt.
- **Execution isolation**: GPU work crosses `tools/sandbox.py`; campaign memory, plans, edits,
  and Git state remain local.
- **Evaluator integrity**: immutable ground truth and full-workload validation prevent harness
  edits or partial-shape wins from becoming accepted results.
- **Production provenance**: production mode mechanically enforces the selected framework and
  rejects third-party operator dependencies and PyTorch compute fallbacks.
- **Backend portability**: one Agent Runtime interface normalizes commands, events, usage, and
  process policy across supported coding CLIs.

## Project Structure

```text
.
├── orchestrator/
│   ├── optimize.py                    # CLI, Campaign, workload coordination
│   ├── agent_runtime/                 # Claude/Qoder/Codex/Pi adapters and process policy
│   ├── telemetry/                     # Phase timing and token telemetry
│   ├── aggregate_dispatch.py          # Static single-file bucket embedding
│   ├── optimization_policy.py         # leaderboard/production policy gates
│   └── prompts/                       # Setup, inspection, baseline, and episode prompts
├── long_horizon/                      # Episode worktrees, handoff protocol, ABBA verification
├── agents/                            # Baseline Agent definition injected into campaign workspaces
├── skills/                            # Baseline runtime skill for Codex/Pi
├── tools/
│   ├── sandbox.py                     # Gateway packaging and execution boundary
│   ├── local_gateway.py               # Trusted localhost FIFO scheduler
│   ├── memory_manager.py              # Structured iteration memory manager
│   └── profile_*.sh / analysis tools  # NVIDIA and AMD profiling helpers
├── reference/                         # Workspace init, evaluator adapters, schema, SOL packaging
├── gpu-wiki/                          # Hardware and optimization knowledge base
├── reference-projects/                # Optional source-search repositories
└── 3rdparty/                          # Humanize and profiler-analysis dependencies
```

The `skills/` and `agents/` directories are internal runtime assets. The orchestrator links or
installs them into generated campaign workspaces; they are not standalone repository entry
points.

### Authority boundaries

| Boundary | Owner | Durable result |
| --- | --- | --- |
| Campaign control | `orchestrator/optimize.py` | Workspace Git history, canonical memory, aggregate state |
| Episode exploration | `long_horizon/` plus one coding-agent session | Journal, handoff, archived attempt and telemetry |
| GPU execution | `tools/sandbox.py` plus gateway | Structured evaluator result and requested profile artifacts |
| Optimization knowledge | `gpu-wiki/`, then optional `reference-projects/` | Evidence references recorded by the episode |

The Agent may edit only its isolated candidate worktree. It cannot decide promotion, mutate the
incumbent directly, replace evaluator inputs, or use local host GPU execution. Conversely, the
supervisor does not generate optimization code: it validates, measures, records, promotes, and
aggregates exact committed sources.

## Supported Entry Point

```bash
python orchestrator/optimize.py \
  --op-dir /path/to/operator \
  --platform TARGET_GPU \
  --sandbox-hardware REMOTE_GPU \
  --framework Triton
```

The public path creates an isolated Git-worktree episode for each optimization version. A fresh
Agent thread may perform several related profile/research/edit/validate cycles. Claude and Codex
support bounded same-thread recovery when the terminal handoff is incomplete; canonical state
crosses episode boundaries through Git, structured memory, journals, plans, and profiles.

The main workspace name is deterministic. Leaderboard mode uses
`kernel_opt_<op>_<framework>_<platform>`; production mode appends `_production` so a strict
production campaign cannot silently resume permissive leaderboard history. Omitting `--framework`
launches one child process and one independent workspace for every framework supported by the
runtime-detected GPU vendor.

## Core Components

### Campaign lifecycle

`Campaign` in `orchestrator/optimize.py` is the single-operator state machine:

1. Materialize or resume a Git workspace and validate its committed V0.
2. In production mode by default, create and pin a self-contained framework-native V1.
3. Create one private branch/worktree and launch one multi-cycle episode per canonical version.
4. Validate its structured journal and `candidate_ready`, `pivot`, or `blocked` handoff, with
   bounded same-thread recovery for Claude and Codex.
5. Check protected paths, clean worktree state, exact candidate commit, and production policy.
6. Independently compare a valid candidate with the incumbent in one ABBA allocation.
7. Squash-promote only a strict correctness-passing improvement; otherwise commit only canonical
   failure/pivot/block evidence.
8. Stop on version budget, token budget, optional stall budget, target utilization, or a terminal
   repeated blocker.
9. Recheck production policy and package the final candidate.

`HEAD` is always the incumbent. A failed, regressing, or policy-violating candidate is not
allowed to replace it.

### Agent Runtime

`orchestrator/agent_runtime/` separates backend-specific command and event formats from campaign
control. Adapters expose a common request/result model containing:

- exit status and timeout state;
- normalized session identity;
- terminal token usage;
- per-event usage deltas and phase-marker receipts when supported;
- backend capability and observation-error metadata.

The process supervisor also protects the host execution boundary by rejecting dependency builds,
direct host GPU execution, profiler use outside the sandbox, and mutations of a shared localhost
gateway.

### Workspace runtime assets

`link_runtime()` exposes `tools/`, `reference/`, `skills/`, `reference-projects/`, and `gpu-wiki/`
inside each campaign workspace. It also prepares backend-specific project-local discovery trees:

- `.claude/` and `.qoder/` receive Agent definitions and knowledge skills;
- `.agents/skills/` receives repository-scoped Codex/Pi optimization skills;
- Humanize planning assets are hydrated locally without changing global user configuration.

### Sandbox and gateway

All correctness, benchmark, signature-collection, and profiling work crosses
`tools/sandbox.py`. The sandbox builds an explicit input allowlist, omits optimizer-only state,
submits a typed `run`/`profile` job when representable, and falls back to a self-contained `dev`
job for SOL, aggregate, or custom commands.

Execution may target an external atrex-gpu-gateway or `tools/local_gateway.py`. The localhost
gateway persists jobs in SQLite and consumes them FIFO with one worker by default. It is a
transport-compatible trusted-code executor, not a security boundary.

### Workload-aware optimization

SOL and native Atrex-Bench operators use workload coordination by default:

1. Collect evaluator-faithful runtime signatures in the sandbox.
2. Retain only explicit scalar values and tensor shape/stride/dtype/layout metadata.
3. Run the workload inspector in a data-minimized temporary workspace.
4. Mechanically validate exact, disjoint coverage and reject partitions that split identical
   runtime signatures.
5. Materialize one independent Git campaign per bucket and optimize buckets concurrently.
6. Hold aggregate edits during the first ten bucket versions.
7. Statically embed exact committed bucket blobs into one deterministic, self-contained
   `kernel.py`; no Agent generates dispatcher code.
8. Accept the aggregate only after full-workload single-seed and five-additional-seed correctness
   plus the configured strict geomean improvement over the incumbent.

The generated dispatcher performs exact structural signature matching and never synchronizes or
reads tensor contents to choose a bucket.

### Production policy

`optimization_mode=leaderboard` allows evidence-backed framework changes and compatible
third-party libraries. `optimization_mode=production` is fail-closed:

- the selected framework is a hard constraint;
- third-party kernel/operator dependencies are forbidden;
- PyTorch compute fallbacks and dynamic external-code loading are rejected;
- `kernel.py`, embedded aggregate sources, and `solution.json` are checked mechanically;
- violating episode candidates are rejected before promotion and recorded as failed memory.

Production Triton campaigns enter a mandatory Triton-to-Gluon episode after the configured stall
threshold. The episode receives an explicit conversion directive and TTGIR/conversion-sheet
workflow. Conversion remains latched until a committed Gluon candidate passes correctness and
performance-parity gates.

### Long Horizon episode engine

`Campaign.run()` in `orchestrator/optimize.py` invokes the internal Long Horizon engine. It creates
an isolated branch and Git worktree from the incumbent for each episode. The Agent records
structured experiments in a journal and publishes one terminal handoff: `candidate_ready`,
`pivot`, or `blocked`.

A candidate must leave a clean worktree, change `kernel.py`, preserve protected paths, satisfy
production policy, and pass an exact same-allocation ABBA schedule. Accepted candidates are
squash-promoted to the incumbent with canonical memory. Rejected and non-candidate episodes
advance memory history without changing the incumbent kernel. Active episode state supports
crash recovery. The internal engine has no public parser or module entry point; all settings are
provided by `orchestrator/optimize.py`.

## End-to-End Flow

### 1. Resolve the operator and runtime

`--op-dir` supplies all operator-specific ground truth. The orchestrator detects SOL or native
Atrex-Bench format, probes the runtime GPU architecture, resolves the framework set, initializes
required submodules, and creates a framework/hardware-suffixed workspace below `--workspace` or
the current directory.

### 2. Establish V0

SOL operators receive a mechanically seeded PyTorch wrapper and immutable evaluator inputs.
Native Atrex-Bench and derived inputs use a bounded setup session. V0 must have a passing complete
workload result, `memory/v0.json`, and a Git root commit.

### 3. Establish the framework baseline

Production mode runs a dedicated framework-baseline session by default
(`--framework-baseline=auto`). The orchestrator restores immutable inputs, checks framework
purity, validates the base seed plus five additional seeds, commits the result as V1, and pins its commit for later bucket
seeding. `always` enables this stage in leaderboard mode; `never` seeds buckets from V0.

### 4. Inspect and partition workloads

In production mode, when `workload.jsonl` or `shapes.json` is present, the coordinator collects structural
signatures, validates the Agent-produced bucket manifest, derives filtered operator inputs, and creates
bucket campaigns. `--no-workload-bucketing` disables this production-only partitioning. Leaderboard mode
always runs one unbucketed episode campaign over the complete workload set.

### 5. Explore one episode per version

Each version repeats a coherent evidence loop as many times as needed within its episode:

```text
profile -> research -> plan -> edit/compile/repair
        -> correctness -> benchmark -> journal/checkpoint -> repeat or handoff
```

GPU commands run remotely while plans, source edits, journals, and Git remain local. A
`candidate_ready` handoff is not authoritative: the supervisor validates protected paths, policy,
clean worktree state, and the exact candidate commit, then runs incumbent/candidate ABBA in one
gateway allocation. A rejected candidate, `pivot`, or `blocked` outcome advances canonical memory
without changing the incumbent. Active episode state is restart-safe.

For progress visibility, the supervisor creates ignored `memory/live.json` at episode start and the
journal command refreshes it after every decisive experiment. This live view is explicitly
non-canonical; a numbered `memory/v<N>.json` is written only after terminal handoff processing and
independent verification.

### 6. Aggregate and finalize

Bucket improvements are serialized through an aggregate lock. A deterministic dispatcher is
generated from exact committed Git blobs and accepted only after complete evaluator validation.
At termination, production mode rechecks policy and SOL campaigns emit a directly submittable
output.

## Workspace State

```text
kernel_opt_<name>_<framework>_<platform>[_production]/
├── kernel.py
├── test_kernel.py
├── README.md
├── memory/v<N>.json
├── memory/long_horizon_e<NNNN>.json  # Evidence for promoted episodes
├── plans/
├── profiles/
├── framework_baseline.json
├── dispatch_signatures.json
├── workload_buckets.json
├── aggregate_dispatch.json
├── aggregation_state.json
├── workload_buckets/                 # Independent bucket Git workspaces
└── .atrex_long_horizon/               # Episode state, journals, telemetry, verification
```

Not every campaign uses every artifact. Git plus unmasked `memory/v<N>.json` files are the durable
optimization history. `.atrex_long_horizon/`, temporary verification payloads, and bucket
workspaces are excluded from main-workspace commits; their recoverable local state remains on disk.

## Profiling and Telemetry

- NVIDIA profiling uses `tools/profile_nvidia.sh` and Nsight Compute.
- AMD profiling uses `tools/profile_kernel.sh`, rocprofv3, ATT, PMC, and assembly extraction.
- `tools/memory_manager.py` creates, reads, updates, masks, and summarizes iteration records.
- Episodes attribute wall time and token usage to profile, research, planning, implementation,
  correctness, benchmark, and recording phases when the backend emits complete markers and usage
  deltas.
- Missing or inconsistent observations are retained with explicit partial/unavailable measurement
  labels rather than fabricated values.

## Critical Constraints

- Hardware specifications must come from `gpu-wiki` with auditable source references.
- Official profiler evidence is required before optimization code changes.
- Ground-truth evaluator inputs are immutable.
- Correctness must pass before performance conclusions or promotion.
- Every accepted candidate must be represented by Git and structured memory.
- `masked: true` memory is excluded from active planning.
- Production candidates must be self-contained in their selected framework.
- Local gateway mode accepts trusted code only and should remain bound to loopback.
