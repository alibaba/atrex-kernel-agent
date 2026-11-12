---
name: atrex-aka-light
description: Use AKA Light to validate accepted operator inputs and to install, launch, resume and inspect supervised GPU-kernel optimization campaigns. Covers Native Atrex-Bench and SOL contracts, Fast/Performance/Ultra final-source policy, persistent Agent Sessions and Git, Runtime evaluation, submit-kernel review, pooled search, observability, diagnosis and evidence export. Use for Light, not the Legacy workflow or standalone Aone Sandbox lifecycle.
---

# AKA Light

AKA Light runs persistent GPU-kernel optimization Sessions and sends GPU work to Agate. The coordinator needs matching AKA packages, an Agent CLI and Agate access; it does not need the target GPU locally. Treat the selected checkout as authoritative and run `aka optimize --repo-root REPO -- --help` before constructing a non-trivial launch.

## Route the request

| Goal | Action |
|---|---|
| Install/update Light | Run `scripts/install_light.sh AKA_REPO_ROOT`; read [installation](references/installation.md) |
| Validate an operator | Check [problem inputs](references/problem-inputs.md) |
| Start/resume/stop | Follow [operations](references/operations.md) |
| Configure CLI or composition | Read [configuration](references/configuration.md) |
| Inspect a live campaign or export evidence | Read [observability](references/observability.md) |

Do not use this skill for the Legacy `workflow` Profile or standalone Aone Sandbox lifecycle.

## Supported problem inputs

Light accepts exactly two problem families:

| Family | Required operator files | Boundary |
|---|---|---|
| Native Atrex-Bench | `reference.py`, `input.py`, non-empty `shapes.json` | Operator must be inside a canonical checkout containing `scripts/run_eval.py` and `src/atrex_bench/` |
| SOL-ExecBench | `definition.json`, `workload.jsonl`, `reference.py` | Uses the SOL definition/workload contract |

An arbitrary `reference.py`, custom evaluator script, or setup command is not a Light problem. Convert it to a supported layout or intentionally use Legacy. The Native V0 is derived from `reference.py`; an adjacent candidate tree is not treated as the initial source.

## Install matching packages

```bash
python3 -m venv /path/to/aka-venv
source /path/to/aka-venv/bin/activate
bash ~/.codex/skills/atrex-aka-light/scripts/install_light.sh \
  /path/to/atrex-kernel-agent
```

The checkout later supplied through `--repo-root` must match the installed wheels. Package version strings alone are insufficient; rebuild after source changes.

## Understand the current workflow

1. Light prepares and full-workload measures deterministic V0.
2. Each trajectory receives one persistent workspace, Agent conversation and Agent-owned Git repository.
3. An Episode is only an audit segment created when the Agent stops; the next Episode inherits the same conversation, workspace and Git history.
4. The Agent may probe, edit, commit, Evaluate, Profile, Check, Disassemble, use Dev/Command, and manage optional Direction/Experiment records throughout the Session.
5. The Agent decides when an exact commit is ready to serve as a final implementation for the current mode. It may then call `submit-kernel`; this is not required after every Evaluate and requires a passing standard full Evaluate with complete per-Shape performance for the clean commit.
6. `submit-kernel` reviews the exact commit against the selected mode's final-source policy. It does not end the Session, run automatic terminal ABBA, compare against a promotion threshold, roll back source, or modify Agent Git.
7. Accepted submissions with eligible full-Evaluate evidence determine lineage winners, optional global winners and the accepted-submission Pareto frontier.

There is no Framework Baseline stage, `episode-report`, candidate sealing, terminal handoff, automatic terminal ABBA, or Supervisor promotion threshold in the current Light workflow. Explicit ABBA remains available as an Agent-requested measurement operation.

Public Kernel identities are full Agent Git commit IDs. Controller state remains private task/V0 authority and audit storage; it is not a candidate Git mirror. Peer Git is imported through read-only refs and never overwrites local HEAD.

## Choose mode and profile

`--profile` selects the workflow/Agent boundary; `--optimization-mode` selects final-source policy.

| Profile | Meaning |
|---|---|
| `light` | Current Light workflow with native Agent execution and capture |
| `light-sandbox` | Current Light workflow with Bubblewrap Agent isolation and capture |
| `workflow` | Legacy workflow, not Light |
| `observed` / `sandbox` | Legacy workflow with an alternate Agent boundary, not Light |

Bubblewrap profiles require Linux. On macOS use native `light` or run Linux through Lima. In Aone Sandbox, use the documented `/proc` shim when private `/proc` is unavailable.

| Mode | Final submitted source |
|---|---|
| `fast` | Agent-selected DSL; explicitly declared reproducible libraries may perform core computation |
| `performance` | Agent selects a DSL using expert guidance; final core computation must be self-authored |
| `ultra` | One DSL must be explicitly configured and is enforced; final core computation must be self-authored |

All modes require a passing standard full Evaluate with complete per-Shape performance before `submit-kernel` can accept a commit. No mode requires automatic terminal ABBA or a minimum speedup. `leaderboard` and `production` remain compatibility aliases for existing workspaces.

Evaluator Shapes are exposed by default. A hidden request is valid only for the supported Performance/Ultra generalized flow and only when Light actually creates/loads the hidden `agent_problem.json`; otherwise launch fails instead of silently claiming Shapes are hidden.

## Candidate layout

- Native Atrex-Bench defaults to `source_tree`: editable `kernel.py` plus optional editable `impl/`.
- SOL uses the configured single-file contract.
- Native CUDA without a Python loader may use the configured `cuda_manifest` layout.
- Layout is fixed for a campaign and must match on resume.

`impl/` is optional. Do not create or require `impl/__init__.py` unless the implementation actually needs it.

## Search and budgets

Current Light defaults to pooled search:

```json
{
  "epochs": 10,
  "trajectories": 3,
  "episodes_per_trajectory": 1,
  "max_parallel_episodes": 3,
  "synchronization": "synchronous",
  "role_specialization": false
}
```

Set `workflow.config.search` to an explicit empty mapping to select serial mode. In serial mode `--max-iters N` is the number of Episode audit segments. Pooled mode uses `epochs × trajectories × episodes_per_trajectory` and does not use `--max-iters` as its segment count.

Episode active time has no limit by default; set `workflow.config.episode_timeout_seconds` only when a limit is wanted. `--token-budget 0` means no token cap. Agate request, queue and per-Shape execution budgets are separate.

At synchronization boundaries, pooled search publishes eligible Direction/Experiment records, relevant Gateway Records, peer Agent Git refs, accepted-submission winners and the aggregated Direction queue. It does not merge scratch files or conversations and does not rewrite another trajectory's checkout.

## Launch

Launch options precede `--`; campaign options follow it:

```bash
aka optimize --repo-root /path/to/atrex-kernel-agent \
  --profile light-sandbox -- \
  --op-dir /path/to/atrex-bench/data/operator \
  --platform L20D \
  --framework Cuda \
  --agent-cli claude \
  --optimization-mode ultra \
  --workspace /path/to/campaign-root
```

Omit `--profile` for native Light. `light-sandbox` isolates the Agent on the coordinator; it does not provision an Aone Sandbox.

Resolve before launch:

- matching checkout and installed wheels;
- supported problem input;
- logical platform and any distinct Agate worker alias;
- explicit framework for Ultra, or intentional Agent-selected DSL for Fast/Performance;
- authenticated Agent and Agate configuration;
- mode, Shape visibility and candidate layout;
- durable workspace, search topology and budgets.

Do not print credentials. Preserve uncommitted user changes before syncing or installing code.

## Runtime evaluation defaults

For standard Evaluate and explicit ABBA, current Light batches at most 10 Shape IDs per outer Ray Data request, sends the complete operator bundle, selects the current batch with `shape_ids`, and sets `shape_shards=1`. It uses 20 warmups, 100 timing iterations, deterministic correctness cases, and best-effort `lock_clocks=true` without a fixed MHz. Aggregation follows the benchmark's declared performance objective after all batches are merged.

Use Runtime tool help for exact operation-specific Agate fields. Endpoint, credentials, source selection and other authority-bearing options remain Supervisor-owned.

## Resume and inspect

Resume with the same checkout behavior, workspace, profile composition, problem identity, mode, Shape visibility, framework and candidate layout. Do not clear a workspace merely because the process exited; inspect Session, Runtime and Agate state first.

Start inspection with:

```bash
python3 "$AKA_REPO/observability/light_runtime_status.py" "$CAMPAIGN" --json
python3 "$AKA_REPO/observability/light_submission_status.py" "$CAMPAIGN" --json
python3 "$AKA_REPO/observability/light_agate_status.py" "$CAMPAIGN" --json
python3 "$AKA_REPO/observability/light_gateway_jobs.py" "$CAMPAIGN" \
  --gateway-record-id gateway-0123456789abcdef0123456789abcdef --json
python3 "$AKA_REPO/observability/light_token_time.py" "$CAMPAIGN" --json
python3 "$AKA_REPO/observability/light_research_status.py" "$CAMPAIGN" --json
```

For performance authority, inspect `PROGRESS.md`, `frontier.json`, private `kernel_submissions`, MeasurementStore and immutable Gateway Records. Do not interpret old promotion-oriented fields or scripts as current winner authority. Export evidence by Gateway Record ID and source by Agent Git commit ID before destructive cleanup.

## Safety boundaries

- A read-only status request does not authorize restart, resubmission, cleanup or deletion.
- A timeout, queue failure, missing receipt or transport error is not a correctness failure.
- Stop through the owning Supervisor/workload and verify Agent, Runtime and children exit.
- Workspace or Sandbox deletion requires explicit authorization.
- Do not edit private Runtime, Journal, MeasurementStore, checkpoint, lock or Git state to force recovery.

## Read as needed

| Scenario | Reference |
|---|---|
| Installation, Bubblewrap, Lima or Aone `/proc` shim | [Installation](references/installation.md) |
| Native/SOL contracts, Shapes, hidden flow or layouts | [Problem inputs](references/problem-inputs.md) |
| CLI flags, defaults, patches and Legacy-only parser flags | [Configuration](references/configuration.md) |
| Launch, modes, V0, persistent Episodes, resume and cleanup | [Operations](references/operations.md) |
| Live state, Agate, tokens, winners and evidence export | [Observability](references/observability.md) |
