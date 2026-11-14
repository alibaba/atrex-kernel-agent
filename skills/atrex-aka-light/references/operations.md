# AKA Light operations

Read this file when starting, configuring, stopping, resuming, or explaining a Light campaign.

## Preflight

Resolve these inputs before launching:

- AKA checkout and commit;
- [supported Native Atrex-Bench or SOL problem](problem-inputs.md);
- target Agate hardware/platform and endpoint profile;
- one target framework, or intentional auto-dispatch;
- Agent backend and its existing authentication/configuration;
- optimization mode and evaluator-shape visibility;
- candidate layout;
- workspace and campaign budgets.

Inspect `git status` before syncing code. Preserve user changes. Verify `aka`, Python, the selected Agent CLI, and Agate credentials without printing secrets.

## Command shape

All launch and request options use one parser; pass them without a `--` separator:

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

Omit `--profile` for native Light. `--profile light-sandbox` is the Bubblewrap Agent boundary on the coordinator; it is unrelated to provisioning an Aone Sandbox.

Run the checkout's help before constructing non-trivial commands:

```bash
aka optimize --repo-root /path/to/atrex-kernel-agent --help
```

Agate-native options belong to the Agent's `tools/sandbox.py` help/schema, not the Workflow launch parser. The installed client supplies safe operation-specific options; authority-bearing values stay Supervisor-owned.

Read [Configuration](configuration.md) for public request fields, Execution settings and composition patches.

## Launch profiles

`--profile` selects the Workflow, workspace implementation, Agent boundary and services. It is orthogonal to `--optimization-mode`, which selects the candidate-acceptance policy inside the chosen Workflow.

| Profile | Workflow | Agent | Capture | Boundary |
|---|---|---|---|---|
| `light` (default) | Managed drafts and unified Light Episodes | Native Light Agent | Enabled | Cooperative same-UID execution |
| `light-sandbox` | Managed drafts and unified Light Episodes | Bubblewrap Light Agent | Enabled | Filesystem/process isolation on Linux |
| `workflow` | Legacy Setup/Fast/Full/goal Workflow | Legacy Agent | Legacy behavior | Explicit old-Workflow compatibility |
| `observed` | Legacy Workflow | Native Light Agent | Enabled | Observation upgrade only; not Light Workflow |
| `sandbox` | Legacy Workflow | Bubblewrap Light Agent | Disabled | Agent isolation only; not Light Workflow |

Only `light` and `light-sandbox` install the managed Light Workspace, managed Unix-socket Tool Runtime, shared Experiment records, Journal service and Light acceptance/promotion path. `observed` and `sandbox` replace only the Agent provider inside Legacy.

Bubblewrap profiles require Linux and fail explicitly on macOS. Bubblewrap limits filesystem/process visibility; it is not a network sandbox or separate user account. Native `light` remains cooperative under the same OS user.

The CLI accepts a Profile JSON path in addition to the named Profiles. Treat custom compositions as advanced configuration: required Light services form a coherent set and should not be removed independently. Recorded continuations restore their frozen launch Profile and reject a new `--profile` or composition patch. For manual restarts, retain the original Profile. Never point Legacy and Light Workflows at the same live workspace; start a new workspace when changing Workflow families.

## Optimization modes

The optimization mode controls which candidate implementations may be accepted. It is independent of the launch profile (`light` or `light-sandbox`), Agate endpoint, Agent backend, and evaluator-shape visibility.

### `leaderboard`

`leaderboard` is the permissive default on current main. Choose it for benchmark exploration where the main objective is the fastest correct result:

- the requested framework is a preferred direction, not a hard purity boundary;
- evidence-backed mixed or alternate implementations are allowed;
- third-party helper and kernel libraries are allowed;
- evaluator correctness, full Evaluate evidence, exact source sealing, ABBA comparison, and the minimum-improvement promotion gate remain mandatory.

The result may be valid for the benchmark without being self-contained or deployable solely in the requested framework. With `--framework-baseline=auto`, leaderboard normally retains V0 instead of running the production-oriented V1 bring-up stage; use `always` to request that stage explicitly. Source-layout-specific behavior, including CUDA tasks, can still pin or establish an applicable baseline, so report the actual Supervisor decision rather than inferring it from the flag alone.

### `production`

Choose `production` when accepted and exported candidates must be deployable and conform to the selected framework:

- V0 may be the PyTorch reference wrapper, but optimized candidates must implement GPU computation in the selected framework;
- self-authored build, compiler, header, ABI, and launch support may be accepted;
- prebuilt compute kernels/operators, alternate DSLs, PyTorch compute fallback, hidden dispatch, and external implementation loading are rejected;
- deterministic source checks and an independent read-only full-candidate policy review are required;
- missing, malformed, or rejecting policy evidence blocks acceptance even when correctness and performance pass;
- ordinary correctness, full-workload evaluation, candidate sealing, ABBA comparison, and minimum-improvement promotion remain required;
- `--framework-baseline=auto` normally creates or pins a framework-compliant V1 after V0;
- final packaging fails closed if the retained candidate is not production-compliant.

Validated multi-GPU evaluator contracts may skip the single-GPU source-policy inspection because distributed implementations require different communication facilities. They still retain evaluator correctness, numerical checks when enabled, and performance promotion gates.

### Shared behavior and separate controls

Both modes measure V0 and promote only verified improvements. Production adds implementation-policy gates; it does not replace measurement. Conversely, leaderboard removes the production purity requirement; it does not remove correctness or promotion evidence.

Evaluator shapes are exposed by default on current main in both modes. `--no-expose-evaluator-shapes` requests the generalized hidden-shape flow for a supported production operator. Shape visibility and optimization mode are both workspace identity: do not resume a workspace with a different mode or visibility setting. Use a fresh workspace when intentionally changing either one.

Always verify these defaults in the selected checkout because they have changed across versions.

## V0 and framework baseline

The current Light sequence is:

1. prepare the selected initial source;
2. run and commit the full-workload V0 measurement;
3. optionally establish a framework baseline V1;
4. begin optimization Episodes.

`--framework-baseline` controls V1, not V0:

- `auto`: normally run V1 in production; current behavior may pin an already compliant source;
- `always`: enable the V1 stage in leaderboard mode too;
- `never`: optimize directly from measured V0.

Current main does not support skipping V0 and going directly to Framework Baseline. A failed V0 stops before V1. Confirm this invariant in `src/aka/workflow/light/orchestration/campaign.py` before describing a newer branch.

## Platform, framework and candidate layout

`--platform` is required and cannot be deduced from the operator directory. It is the logical target product used in Agent instructions, Wiki context, hardware fallback detection and workspace identity. By default it is also the hardware token sent to Agate. Use the exact token accepted by the selected environment.

When the logical product and remote executor alias differ, use `--hardware` without changing the logical platform:

```bash
--platform L20D --hardware AGATE_WORKER_ALIAS
```

The alias in this example is a placeholder. If Agate accepts `L20D`, omit `--hardware`; the CLI automatically uses the `--platform` value. Supply the override only when the configured Agate endpoint exposes a different scheduler token.

`--platform` is distinct from runtime architecture. AKA normally probes the actual worker and obtains an architecture such as `sm_89`, `sm_100` or `gfx942`. Runtime architecture is authoritative for NVIDIA/AMD vendor selection; the platform name is a fallback. PPU is recognized from product tokens beginning with `PPU` or `ZW` because its borrowed architecture token is ambiguous. Use `--arch` only for an intentional override or when probing fails.

Supplying one `--framework` runs one child Campaign in that DSL. In Production the assigned framework is a hard acceptance constraint; in Leaderboard it is a preferred direction. Supported auto-dispatch sets are:

| Detected vendor | Frameworks when `--framework` is omitted |
|---|---|
| NVIDIA | `Triton`, `CuteDSL`, `Cuda`, `TileLang` |
| AMD | `Triton`, `FlyDSL`, `TileLang` |
| PPU | `Triton`, `Cuda`, `TileLang` |
| Unknown | `Triton`, `TileLang` |

Omitting `--framework` launches these children independently and normally in parallel. Each child has a separate workspace, version limit and token budget. Framework availability still depends on the worker; choose one explicit framework when the requested experiment should create only one Campaign.

- Native Atrex-Bench tasks use `source_tree` by default. `kernel.py` is the editable evaluator entry point and may contain the full implementation or import helpers from `impl/`.
- Native CUDA without a Python loader requires the `cuda_manifest` candidate configuration; use the checkout's documented composition patch rather than inventing a layout.
- The chosen candidate layout is fixed for the task and must match on resume.

For optional modules or pooled search, read `docs/optional-light-modules.md` from the selected checkout. A composition patch replaces the provider's whole configuration; retain required fields such as Workflow startup and the intended Agent sandbox/capture settings.

## Budgets

On current main, ordinary optimization Episodes default to four hours of Agent active time; blocking Runtime Tool waits pause the Agent and do not consume that allowance. Framework Baseline has a separate eight-hour wall-clock default. Agate queue/execution timeouts remain independent.

`--max-iters N` is an inclusive canonical-version ceiling for each serial child Campaign. The default is `20`, so `--max-iters 20` means that the Campaign may record through `v20` and then stops admitting new Episodes.

It is not a count of accepted kernels, model turns, attempts, or Agate calls:

- V0 is `v0` and is always measured before optimization.
- If Framework Baseline creates or pins V1, optimization starts at `v2`; `--max-iters 20` therefore permits at most 19 optimization Episodes.
- If Framework Baseline is skipped, optimization starts at `v1`; the same limit permits at most 20 optimization Episodes.
- A terminal Episode records the next canonical version even when it is not promoted. Rejected, blocked, pivoted, invalid-handoff and recovered-interrupted outcomes therefore consume version slots too.
- The Supervisor checks the ceiling before admitting another Episode. An already admitted Episode reaches a durable terminal outcome before the next check.
- Resume reads the latest recorded version and continues with only the remaining slots.
- When `--framework` is omitted, each auto-dispatched framework child has its own `--max-iters` ceiling.

Optional pooled multi-trajectory search replaces this serial budget with the explicit product `epochs × trajectories × episodes_per_trajectory`. Do not describe `--max-iters 20` as 20 Episodes per trajectory in that mode.

`--token-budget`, Episode active-time limits, target utilization, mandatory conversion and `--max-stall` are separate stop conditions and can end a Campaign before the version ceiling. If mandatory Triton-to-Gluon conversion is still pending when the version ceiling is reached, Light reports that conversion failure instead of declaring ordinary budget completion.

Treat these as versioned defaults. Confirm the selected checkout's help and composition before changing them. State whether a reported duration is Agent active time, session wall time, Agate wait time, or campaign wall time.

## Start and verify

After launch, verify all of the following rather than treating process creation as success:

- the Supervisor process remains alive;
- the expected campaign workspace and scope(s) exist;
- V0 or Framework Baseline has produced a durable progress update;
- the Agent Session is active or a concrete Agate job is pending;
- no immediate configuration, credential, source-layout, or policy error is present.

Prefer a background service or declared workload for long runs; do not keep an interactive command slot occupied for hours when the environment provides a durable launcher.

## Resume

Resume with the serialized `--workflow-request` recorded for that campaign and its frozen composition. For example, `aka workflow --repo-root REPO --workflow-request "$RECORDED_REQUEST"`. Obtain the exact value from the owning launch receipt; do not reconstruct it from remembered flags. New request fields, a Profile override or new patches are rejected on continuation. Light reconciles accepted Agate job IDs, completed batches, an interrupted Framework Baseline, and an interrupted Episode from durable state.

Do not clear a workspace merely because a process exited. First inspect runtime, Agate, and acceptance status. If the user requests a clean rerun that preserves Bootstrap/V0, identify the exact supported reset boundary for that checkout; do not manually delete arbitrary state directories.

## Stop and cleanup

Stop the campaign through its owning Supervisor or workload mechanism, then verify Agent, Runtime, and child processes exited. Avoid broad process-name kills that could affect unrelated campaigns. Workspace or Sandbox deletion requires explicit user authorization; export evidence first when results may be needed.
