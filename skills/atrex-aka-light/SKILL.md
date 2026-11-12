---
name: atrex-aka-light
description: Use AKA Light to validate accepted operator inputs and to install, launch, resume and inspect supervised GPU-kernel optimization campaigns. Covers Native Atrex-Bench and SOL problem contracts, leaderboard versus production policy, candidate layouts, V0/framework-baseline/Episode lifecycle, Agate-backed evaluation, observability, failure diagnosis and exact source/evidence export. Use for Light, not the Legacy workflow or standalone Aone Sandbox lifecycle.
---

# atrex-aka-light

Applies to the 0.2.0 module distributions with the public Workflow request CLI.
Use a resource checkout matching the installed source revision. Agent tools use
the managed Unix socket; launch examples below use the current request contract.

AKA Light runs supervised GPU-kernel optimization campaigns and sends GPU work to Agate. The coordinator needs the AKA packages, an Agent CLI and Agate access; it does not need the target GPU locally. Use `aka optimize --repo-root REPO --help` for the exact options exposed by the installed revision.

## Choose a task

| Purpose | Command or action | Inputs and boundaries |
|---|---|---|
| Install or update Light | `bash /path/to/atrex-kernel-agent/skills/atrex-aka-light/scripts/install_light.sh AKA_REPO_ROOT` | Builds matching framework, legacy and Light wheels from one checkout; read [installation](references/installation.md) |
| Inspect launch options | `aka optimize --repo-root AKA_REPO_ROOT --help` | All launch and request options belong to the same parser |
| Start a campaign | `aka optimize --repo-root AKA_REPO_ROOT --problem OPERATOR --platform GPU ...` | Requires an operator, platform, framework or intentional auto-dispatch, Agent backend, mode and workspace |
| Resume a campaign | Use its recorded `--workflow-request` | Mode, visibility, candidate layout, framework, composition and workspace identity must match |
| Inspect status or performance | Run the matching `observability/light_*.py` script | Read-only; use JSON for calculations and tables for user summaries |
| Export exact source or evidence | `export_light_source.py` or `export_light_evidence.py` | Select by accepted Git revision, Kernel identity or Gateway Record identity |
| Stop a campaign | Stop it through its owning Supervisor or workload | Verify Agent, Runtime and child processes exit; do not use broad process-name kills |

## Choose a supported problem input

Light accepts exactly two problem families:

| Family | Required operator files | Additional boundary |
|---|---|---|
| Native Atrex-Bench | `reference.py`, `input.py`, non-empty `shapes.json` | The operator must live under a canonical checkout containing `scripts/run_eval.py` and `src/atrex_bench/` |
| SOL-ExecBench | `definition.json`, `workload.jsonl`, `reference.py` | Uses the SOL definition/workload contract rather than Native Shapes |

An arbitrary directory containing only `reference.py`, a custom evaluator script, or
a setup command is not a Light problem. Use the Legacy `workflow` Profile for an
intentional custom setup, or convert it to a supported layout. Read [problem input
contracts](references/problem-inputs.md) for file meanings, `shapes.json` and
`agent_problem.json` schemas, hidden-shape requirements, multi-GPU metadata, V0
derivation, candidate layouts, and unsupported cases. Use the [minimum Native
vector-add example](examples/native-atrex-bench-vector-add/README.md) as a concrete
three-file template.

## Installation

This skill and the `observability/` scripts are checkout resources. To install
this skill in an Agent's skill directory, copy the complete `skills/atrex-aka-light/`
directory, including references, scripts and examples. Commands here use the
original checkout paths. Set `AKA_REPO` to the matching checkout; references to
`docs/...` and `src/...` are relative to that checkout, even when this skill was
copied elsewhere. Read [installation](references/installation.md) for runtime dependencies.

Create an isolated Python environment, then install all three matching distributions from one AKA checkout:

```bash
python3 -m venv /path/to/aka-venv
source /path/to/aka-venv/bin/activate
bash /path/to/atrex-kernel-agent/skills/atrex-aka-light/scripts/install_light.sh \
  /path/to/atrex-kernel-agent
```

The checkout later passed through `--repo-root` must come from the same revision as the installed wheels. The packages commonly all report version `0.2.0`; rebuild and force-reinstall after source changes rather than comparing only version strings. See [installation and verification](references/installation.md) for interpreter selection, retained wheels, manual installation and upgrades.

## Choose an optimization mode

Optimization modes control which candidates may be accepted. They do not select native versus Bubblewrap execution, an Agate endpoint, the Agent backend or evaluator-shape visibility.

| Mode | Use it for | Candidate policy |
|---|---|---|
| `leaderboard` (default) | Finding the fastest correct benchmark implementation | The selected framework is a preferred direction. Evidence-backed mixed or alternate implementations and third-party helper/kernel libraries are allowed. The result need not be self-contained or deployable solely in that framework. |
| `production` | Producing a deployable, framework-constrained implementation | After the V0 reference exception, GPU computation must use the selected framework. Prebuilt compute, alternate DSLs, PyTorch compute fallback, hidden dispatch and external implementation loading are rejected. Source checks and an independent read-only policy review fail closed. |

Both modes still require evaluator correctness, full Evaluate evidence, exact candidate sealing, incumbent comparison, native ABBA evidence and the configured improvement threshold. Production adds policy gates; leaderboard does not remove measurement gates.

With `--framework-baseline=auto`, production normally creates or pins a framework-compliant V1 after V0. Leaderboard normally retains permissive V0; use `--framework-baseline=always` when a distinct V1 is desired. Source layout and CUDA-specific handling can alter the exact Supervisor decision, so inspect the recorded decision rather than inferring it only from the flag.

Evaluator shapes are exposed by default on current main in both modes. A supported production operator can request the generalized hidden-shape flow with `--no-expose-evaluator-shapes`. Mode and visibility are workspace identity; use a fresh workspace when intentionally changing either one.

## Choose a launch profile

`--profile` selects the Workflow/Agent composition. It is separate from `--optimization-mode`: a Profile chooses how AKA runs, while Leaderboard/Production chooses which candidates the selected Workflow may accept.

| Profile | Workflow and workspace | Agent boundary | When to use it |
|---|---|---|---|
| `light` (default; omit `--profile`) | Managed drafts, unified Light Episodes and managed Unix-socket tools and shared Experiment records | Native process, session capture enabled | Normal Light runs on macOS or Linux when cooperative same-UID isolation is sufficient |
| `light-sandbox` | Same complete Light Workflow and services as `light` | Bubblewrap filesystem/process isolation, session capture enabled | Light runs on Linux when the Agent should see only the mounted workspace/resources |
| `workflow` | Legacy Setup/Fast/Full/goal Workflow and Legacy workspace | Legacy Agent behavior | Explicit compatibility with the old Workflow; not an AKA Light campaign |
| `observed` | Legacy Workflow and workspace | Native Light Agent with session capture | Diagnose Legacy behavior while retaining the Legacy Workflow |
| `sandbox` | Legacy Workflow and workspace | Bubblewrap Light Agent, capture disabled | Isolate the Agent while retaining Legacy Workflow semantics |

Only `light` and `light-sandbox` select the managed Light Workspace, Tool Runtime, evidence store and unified Episodes. `observed` and `sandbox` replace only the Agent used by Legacy; they do not become Light campaigns.

Bubblewrap is supported on Linux. Selecting `light-sandbox` or `sandbox` on macOS fails explicitly; there is no silent native fallback. It is filesystem/process isolation, not a network allowlist or a separate OS account.

The CLI also accepts an explicit Profile JSON path for advanced composition work. Use a supported named Profile unless the user intentionally supplies and owns a custom composition. Continuations restore their frozen Profile and reject Profile overrides; do not switch an existing Legacy workspace into Light or reuse one live workspace across the two Workflows.

## Choose the platform and framework

`--platform` is the required logical target hardware token, for example `L20D`, `H20`, `B200`, `MI308X` or a PPU product token. It serves four purposes:

- identifies the intended target to the Agent, prompts and GPU Wiki;
- defaults the Agate worker token when `--hardware` is not supplied;
- participates in hardware-vendor detection when a runtime architecture probe is unavailable;
- participates in the child Campaign workspace name and resume identity.

Use the exact hardware token accepted by the configured Agate environment. If the logical product name and the remote worker alias differ, keep the logical value in `--platform` and pass the executor alias separately:

```bash
--platform L20D --hardware AGATE_WORKER_ALIAS
```

`AGATE_WORKER_ALIAS` is a placeholder, not a literal token. For the usual L20D configuration, omit the override and write only `--platform L20D`; AKA then submits `L20D` to Agate. Use `--hardware` only when the selected Agate environment names that worker differently.

`--platform` is not the CUDA/ROCm architecture. Runtime probing normally determines values such as `sm_89`, `sm_100` or `gfx942`; use `--arch` only when that probe fails or must be overridden deliberately.

`--framework` selects the implementation DSL for one child Campaign. For example:

```bash
--platform L20D --framework CuteDSL
```

In Leaderboard mode the framework is a preferred direction; in Production it becomes a hard candidate constraint. Framework availability still depends on the selected worker.

Omitting `--framework` auto-dispatches one independent child Campaign per framework supported by the detected vendor:

| Detected vendor | Auto-dispatched frameworks |
|---|---|
| NVIDIA | `Triton`, `CuteDSL`, `Cuda`, `TileLang` |
| AMD | `Triton`, `FlyDSL`, `TileLang` |
| PPU | `Triton`, `Cuda`, `TileLang` |
| Unknown | `Triton`, `TileLang` |

Runtime architecture is authoritative for NVIDIA/AMD vendor detection; the platform name is the fallback. PPU is identified from its product token because its reported architecture may resemble NVIDIA. Auto-dispatched children have separate workspaces, version limits and token budgets; in Production each child is bound to its assigned framework. Specify one `--framework` when the user wants exactly one Campaign.

## Set the iteration budget

`--max-iters 20` makes `v20` the highest canonical version that one serial child Campaign may create. It does **not** mean 20 successful optimizations, 20 model turns, or 20 evaluations. The default is `20`.

- V0 is recorded as `v0`.
- If Framework Baseline establishes V1, optimization Episodes use `v2` through `v20`, leaving at most 19 Episode slots.
- Without V1, optimization Episodes use `v1` through `v20`, leaving at most 20 Episode slots.
- Every terminal Episode consumes its version slot, whether it is promoted, rejected, blocked, pivoted, invalid, or recovered as interrupted.
- The limit is checked before admitting the next Episode. An admitted Episode finishes and records its outcome; a later resume continues from the recorded version rather than resetting the budget.
- Auto-dispatched framework children each receive their own limit. Optional pooled multi-trajectory search instead uses `epochs × trajectories × episodes_per_trajectory`; it does not use the serial `--max-iters` budget.

Token budget, Episode active-time limits and stall stopping are independent and may stop a Campaign before `v20`.

## Choose other controls

The most common optional controls are:

| Goal | Controls |
|---|---|
| Bound search | `--max-iters`, `--token-budget`, `--max-stall`, `--target-util`, `--convert-after` |
| Control V1 and recovery | `--framework-baseline`, `--framework-baseline-timeout`, `--handoff-resumes`, repeatable `--framework-baseline-reviewers codex` or `qodercli` |
| Set promotion evidence | `--verify-repeats`, `--verify-run-timeout`, `--min-improvement-pct` |
| Select Agate execution | `--hardware`, Execution profile configuration or its endpoint environment |
| Configure Agent/Runtime modules | launch-level `--patch` for Episode active time, Agent pause, Wiki, generated memory, research records, candidate layout or pooled search |

Read [configuration](references/configuration.md) before changing defaults. It explains public request fields, composition ownership and continuation identity; the matching CLI/dataclasses own exact field lists and defaults.

## Launch a campaign

Pass all launch and request options to the same parser, without a `--` separator:

```bash
aka optimize --repo-root /path/to/atrex-kernel-agent \
  --profile light-sandbox \
  --problem /path/to/atrex-bench/operator \
  --platform L20D \
  --framework CuteDSL \
  --agent-cli claude \
  --optimization-mode production \
  --workspace-base /path/to/campaign-root \
  --max-iters 20
```

Omit `--profile` for native Light. `--profile light-sandbox` is an Agent isolation boundary on the coordinator; it is unrelated to provisioning an Aone Sandbox. Workflow help lists request fields; safe Agate-native options belong to the Agent's installed `tools/sandbox.py` schema.

Resolve these inputs before launch:

- a complete AKA checkout matching the installed wheels;
- a [supported Native Atrex-Bench or SOL problem](references/problem-inputs.md);
- the logical platform, any distinct Agate worker alias, and endpoint profile;
- one framework, or intentional hardware-based auto-dispatch;
- an authenticated Agent backend and Agate credentials;
- optimization mode and evaluator-shape visibility;
- candidate layout, workspace and campaign budgets.

Do not print credentials. Inspect the checkout and preserve uncommitted user changes before syncing or installing code.

## Candidate layouts and baseline sequence

Candidate layouts describe the source the Agent edits after the problem is loaded;
they are not alternate `--problem` formats. Native V0 is derived from `reference.py`,
not from an adjacent user-provided `kernel.py` or `impl/` directory.

- Native Atrex-Bench tasks default to `source_tree`: editable `kernel.py` may contain the implementation or import editable helpers from `impl/`.
- `single_file` is an explicit Python-only layout when one editable `kernel.py` is sufficient.
- Native CUDA without a Python loader uses `cuda_manifest`; use the checkout's documented composition instead of inventing a manifest layout.
- Candidate layout is fixed for the campaign and must match on resume.

The normal sequence is V0 full-workload measurement, optional framework baseline V1, then optimization Episodes. `--framework-baseline` controls V1, not V0: `auto` follows mode/layout policy, `always` requests V1 and `never` starts Episodes from measured V0. A failed V0 stops before V1; current main does not skip V0 and go directly to Framework Baseline.

DSH is selected by `--agent-cli dsh`; configure its headless profile and provider
authentication before launch. It uses a private session Home, JSON events and
checkpoints; native child-attempt usage excludes seeded or resumed history.
Bubblewrap execution requires Linux and the matching DSH installation.

New requests expose Shapes by default. Historical serialized requests without
that field retain their original hidden-shape behavior. Immutable historical
candidate adapters retain their pinned source policy.

Each Episode edits a managed draft, evaluates an exact source, submits an evidence-backed report, seals the candidate and compares it with the incumbent. A passing or faster ordinary Evaluate does not itself promote a candidate; terminal verification and the promotion gate decide that.

## Resume, failures and cleanup

Resume with the same checkout behavior, workspace, composition, framework, candidate layout, optimization mode and visibility policy. Light can reconcile accepted Agate job IDs, completed Evaluate/ABBA batches, an interrupted Framework Baseline and an interrupted Episode from durable state.

- Do not clear a workspace merely because a process exited. Inspect Workflow, Session and Agate state first.
- A timeout, queue failure, missing evidence or protocol error is not proof that the candidate is incorrect.
- `waiting_agate` means the local Runtime call is waiting; correlate the saved Job ID before claiming the remote job is running.
- Do not edit private Supervisor state, Measurement Records, checkpoints, locks or Git metadata to force recovery.
- A read-only status request does not authorize restart, resubmission, cleanup or deletion.
- Workspace or Sandbox deletion requires explicit authorization; export needed evidence first.

Production finalization is fail-closed and will not package a retained source that violates production policy. Validated multi-GPU evaluator contracts may skip single-GPU source-policy inspection, but correctness and performance promotion gates remain active.

## Observe and export

Set the compatible AKA checkout and campaign workspace, then use the bundled read-only scripts:

```bash
export AKA_REPO=/path/to/atrex-kernel-agent
export CAMPAIGN=/path/to/light-campaign-workspace

python3 "$AKA_REPO/observability/light_runtime_status.py" "$CAMPAIGN" --json
python3 "$AKA_REPO/observability/light_performance.py" "$CAMPAIGN" --json
python3 "$AKA_REPO/observability/light_agate_status.py" "$CAMPAIGN" --json
python3 "$AKA_REPO/observability/light_token_time.py" "$CAMPAIGN" --json
```

Distinguish `accepted_best` from `best_passing_full_eval`: the fastest passing full Evaluate may not have been promoted. Distinguish Agent phase, candidate Workflow state and Agate Job state. Preserve unavailable values as unavailable rather than estimating them.

Export exact source and raw/Agent-visible evidence before destructive cleanup or deep comparison:

```bash
python3 "$AKA_REPO/observability/export_light_evidence.py" "$CAMPAIGN" \
  --gateway-record-id gateway-0123456789abcdef0123456789abcdef \
  --output /path/to/new-evidence-directory
```

The output path must not already exist. Read [observability and evidence export](references/observability.md) for the complete command map, state meanings, token analysis and source-only export.

## Read as needed

| Scenario | Reference |
|---|---|
| Prerequisites, wheel installation, verification or upgrade | [Installation](references/installation.md) |
| Accepted operator layouts, required files, Shapes, hidden contracts or initial source | [Problem inputs](references/problem-inputs.md) |
| CLI flags, defaults, composition patches or environment controls | [Configuration](references/configuration.md) |
| Launch syntax, mode exceptions, V0/V1, layouts, budgets, resume, stop or cleanup | [Operations](references/operations.md) |
| Runtime health, latency, correctness, promotion, Agate, tokens, tools, GPU Wiki, source or evidence | [Observability](references/observability.md) |

For non-trivial launches, prefer the selected checkout's `aka optimize --repo-root REPO --help` over remembered flags. For inspection, anchor conclusions to durable timestamps and report fields that are missing instead of guessing.
