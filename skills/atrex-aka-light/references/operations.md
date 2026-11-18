# AKA Light operations

Read this file when starting, explaining, resuming or stopping a Light campaign.

## Preflight

### Required user input

Before every new launch, explain the mode choices and explicitly ask the user for:

- a safe way to make Agate `AGATE_AK` and `AGATE_SK` available to the Supervisor process;
- the canonical Atrex-Bench checkout root containing `scripts/run_eval.py` and `src/atrex_bench/`;
- the target GPU/platform token, for example `L20N` or `L20D`; never infer it from the problem,
  available Sandbox, queue state or an older Campaign.
- the Agent CLI: `claude` (Claude Code), `codex`, `qodercli`, `pi`, or `dsh`;
- the optimization mode: Fast, Performance or Ultra; when Ultra is selected, also require its DSL.
- whether to run the Agent with Bubblewrap Sandbox (`light-sandbox`) or without it (`light`); when
  Sandbox is selected, ask for `private` or `host` proc mode.
- whether to expose optional knowledge directories with `--agent-reference-paths`, or explicitly use
  none. Bubblewrap mounts them read-only; native Light cannot protect the original directories from
  modification, so tell the user to retain an independent copy.

These are required user decisions for every new Campaign. Pass the chosen `--agent-cli`,
`--optimization-mode`, and `--profile` explicitly; do not rely on parser or composition defaults.
When Bubblewrap is chosen, also materialize the user's proc-mode choice explicitly.

Reference directories appear as `references/<basename>/` and must have unique basenames. They are
optional engineering knowledge, not candidate source or evaluation evidence. Native Light creates
workspace links to the original directories; it does not copy them and cannot enforce read-only
access under the coordinator OS user.

For Atrex-Bench, offer exactly two acquisition paths: use an existing canonical checkout supplied
by the user, or clone `https://github.com/alibaba/atrex-bench` into a destination selected by the
user. Do not suggest `/path/to/atrex-bench` merely because it exists locally.

| Agent CLI | Selection meaning |
|---|---|
| `claude` | Claude Code; requires its existing authentication/configuration |
| `codex` | OpenAI Codex CLI; requires its existing authentication/configuration |
| `qodercli` | Qoder CLI; requires its existing authentication/configuration |
| `pi` | Pi CLI; requires its existing authentication/configuration |

Explain the modes as follows:

| Mode | What the user is choosing |
|---|---|
| Fast | Agent-selected DSL; declared reproducible libraries or prebuilt implementations may perform core computation |
| Performance | Agent-selected DSL; final core computation must be self-authored |
| Ultra | User-pinned DSL plus self-authored core computation; launch is invalid without the DSL |

Explain the Agent boundary separately from optimization mode:

| Choice | Meaning |
|---|---|
| With Sandbox (`light-sandbox`) | Linux Bubblewrap isolates the Agent from Supervisor-private filesystem/process state while retaining Light capture and persistent Session/Git; it does not create an Aone Sandbox |
| Without Sandbox (`light`) | Agent runs natively as the coordinator's current OS user; Light persistence/capture remain, but Bubblewrap isolation is absent |

`light-sandbox` requires Linux and Bubblewrap. Native macOS must use `light` or run
`light-sandbox` inside Lima. An Aone Sandbox is an outer execution environment; selecting
`light-sandbox` inside it still refers to the inner Agent boundary.

When Sandbox is selected, explain and require one proc-mode choice:

| Proc mode | Isolation and supported environment |
|---|---|
| `private` | New PID namespace and procfs; unrelated coordinator processes are hidden. Supported on ordinary Linux and Lima. |
| `host` | Shares the outer PID namespace and complete `/proc`; filesystem allowlist and dropped capabilities remain, but process visibility is not isolated. |

**Aone Sandbox supports only `host`.** Its container policy rejects the nested procfs mount used by
`private`; do not try `private`, compatibility shims, or projected `/proc` there. Apply the complete
host-proc agent composition patch described in [installation](installation.md), and use a dedicated
Aone Sandbox because the Agent can see every process in it. Without Agent Sandbox (`light`), proc
mode is not applicable and must not be requested.

Do not reuse these values from an older Campaign or silently select a GPU based on availability.
Do not silently select a nearby checkout. Do not ask
the user to paste AK/SK values into chat. Credentials must already be present in the launch
environment or come from an existing secure credential source; verify presence without printing
values. Ask separately for `--hardware` only when its Agate worker token differs from the
logical platform, and require the user to confirm both tokens.

Resolve before launch:

- AKA checkout and commit matching installed wheels;
- supported Native Atrex-Bench or SOL input;
- target platform, Agate route and optional distinct worker alias;
- explicit Agent CLI and its existing authentication;
- explicit framework for Ultra, or intentional Agent-selected DSL for Fast/Performance;
- Fast, Performance or Ultra final-source policy;
- Shape visibility and candidate layout;
- durable workspace, search topology and budgets.

Inspect `git status` before syncing code. Preserve user changes. Verify `aka`, Python, the Agent CLI and Agate access without printing secrets.

For a complete smoke example, including a minimal operator, checkout validation and launch
command, use [the Native Vector Add example](../examples/native-atrex-bench-vector-add/README.md).

## Launch command

The uppercase values below are required user selections, not defaults. Include
`--framework USER_SELECTED_ULTRA_DSL` only for Ultra.

```bash
aka optimize --repo-root /path/to/atrex-kernel-agent \
  --profile USER_SELECTED_PROFILE \
  --problem /path/to/atrex-bench/data/operator \
  --platform USER_SELECTED_GPU \
  --agent-cli USER_SELECTED_AGENT_CLI \
  --optimization-mode USER_SELECTED_MODE \
  --workspace-base /path/to/campaign-root
```

`--workspace-base` is the parent for named Campaign directories. Light derives
`kernel_opt_<problem>_<mode>_<dsl>_<gpu>` below it; use `--workspace-suffix` to distinguish
repeated runs. Each Campaign is non-Git; only `lineages/tNNNN/agent-workspace` is an Agent repository.
All launch and Campaign arguments use the same parser, with no `--` separator.

Run checkout-specific help first:

```bash
aka optimize --repo-root /path/to/atrex-kernel-agent --help
```

Before constructing an ordinary command, use one independent serial trajectory: carry a patch with
`workflow.config.search: {}` and ask the user for a positive `--max-iters N`. Do not choose the
value for them. Direction and Experiment annotations stay disabled and should not burden ordinary
launch intake. Expose pooled search or Research module settings only after an explicit user request.

After validation, launch in the background by default. Preserve the Campaign root, launcher log,
checkout commit, and durable process identity. In Aone Sandbox use `akfctl sandbox command submit`
and retain its command ID; do not emulate durability with `bash ... &`. For a local host use the
host's durable process supervisor or a detached launch script with stdout/stderr redirected to a
known bounded log. Run in the foreground only when the user explicitly asks for an interactive
launch.

Once accepted, observe the Campaign with the `atrex-aka-light-observability` skill. The first status
view is `light_lineage_status.py`; use `light_kernel_history.py`, `light_kernel_best.py`,
`light_runtime_calls.py`, `light_agate_calls.py`, and evidence tracing/export only as needed. The
launcher process is not sufficient evidence of lineage progress or GPU execution health.

## Profiles

| Profile | Workflow | Agent boundary |
|---|---|---|
| `light` | Current Light | Native process, capture enabled |
| `light-sandbox` | Current Light | Bubblewrap, capture enabled |

Bubblewrap requires Linux. On macOS the user must choose native Light or a Lima Linux VM. In Lima,
the user may explicitly select `private` or `host`. Aone Sandbox allows only explicit `host`, which
binds the Sandbox's complete `/proc` and shares its PID namespace; if the user does not accept that
boundary, use native `light`. Host mode retains filesystem isolation but is not process isolation.
Bubblewrap is not a network allowlist or another OS account.

## Modes

### Fast

The Agent chooses any evaluator-supported DSL or mixture. Explicitly declared, reproducible
third-party or prebuilt implementations may perform core computation. Fast skips independent final-source review; input packaging and full Evaluate requirements still apply.

### Performance

The Agent chooses a DSL using supplied expert guidance and measurements; recommendations are not a
hard gate. Final core computation must be self-authored. Framework runtimes and
compiler/header/ABI/launch plumbing remain allowed, while prebuilt core compute, PyTorch/ATen
fallback, hidden dispatch and external implementations are rejected.

### Ultra

Ultra applies the Performance provenance rules and additionally pins one explicit DSL. Launch fails
before workspace creation when `--framework` is missing or open DSL is requested.

All three modes require a passing full Evaluate with complete per-Shape performance for the exact
Agent Git commit. None adds automatic terminal ABBA or a minimum-improvement threshold.

Evaluator Shapes are exposed by default. `--no-expose-evaluator-shapes` is valid only for the
supported generalized Performance/Ultra flow. If Light cannot create/load the actual hidden
projection, it fails instead of silently exposing Shapes under a hidden label.

## Current lifecycle

The current workflow is:

```text
prepare initial source and common Agent Git root
→ persistent Agent Session + Agent Git
→ repeated probe/edit/commit/measure cycles
→ optional submit-kernel for exact evaluated commits
→ accepted-submission lineage/global winner selection
```

An Episode is an audit segment, not a candidate slot. When the Agent stops, the segment ends; the next segment resumes the same conversation, workspace and Agent Git. There is no Framework Baseline, `episode-report`, source sealing, terminal handoff, automatic terminal ABBA, candidate rollback or Supervisor promotion threshold.

`submit-kernel` accepts only a reachable clean Agent Git commit whose standard full Evaluate is passing and has complete per-Shape performance. Fast skips the independent final-source Reviewer; Performance and Ultra run it, with Ultra additionally enforcing the configured DSL. The first call creates immutable `v1` under a stable Submission ID. A repaired and fully evaluated commit can revise that identity as the next `vN`; its Reviewer resumes the prior Session while rechecking the complete candidate. Submission does not end the Session or change Agent Git, and a rejected later revision does not revoke an older accepted revision.

Explicit ABBA remains available when the Agent wants paired comparison evidence. It is not automatically invoked at Episode boundaries.

## Platform, framework and source layout

`--platform` is the logical product shown to the Agent and normally sent to Agate. If the worker alias differs:

```bash
--platform L20D --hardware AGATE_WORKER_ALIAS
```

`--platform` is not the compiler architecture. Runtime probing should determine values such as `sm_103a`, `sm_100a`, or `gfx942`; use `--arch` only for an intentional override or failed probe.

Fast and Performance create one open-DSL campaign. Ultra requires exactly one explicit
`--framework`; it does not auto-dispatch and rejects `--no-specify-dsl`.

Candidate layouts:

- Native default `source_tree`: editable `kernel.py` plus optional editable `impl/`;
- SOL configured `single_file`;
- native CUDA without a Python loader: configured `cuda_manifest`.

The layout is fixed for the campaign. `impl/` is optional and must not be created merely to satisfy scaffolding.

## Search and budgets

One independent serial trajectory is the default. Explicit advanced pooled search may use, for
example, 10 Epochs, 3 trajectories, 1 Episode per trajectory per Epoch, 3 parallel Episodes,
asynchronous publication and no role specialization. Each pooled trajectory owns one durable
conversation/workspace/Git lineage.

This is an implementation default, not an ordinary launch recommendation. Ordinary Skill launches
use one independent serial trajectory and require a user-selected positive `--max-iters`. Pooled
settings are advanced options shown only when the user explicitly requests multi-trajectory search.

Set `workflow.config.search` to `{}` for serial mode. Then `--max-iters N` counts Episode audit segments. In pooled mode, the schedule is `epochs × trajectories × episodes_per_trajectory`; `--max-iters` is not the pooled segment count.

Agent and Review time limits are unset by default. `--token-budget 0` means no token admission cap. Runtime requests do not pause the Agent by default; independent work may continue and request time counts toward any configured Agent time limit. `agent.config.pause_agent_during_requests=true` opts into suspension and excludes request waits from active time. Agate queue/request/per-Shape timeouts remain independent.

Every archived Direction and Experiment revision is written immediately to the central Campaign
Research Library. Each workspace receives an ordinary `research-library/` copy following the configured visibility mode; Bubblewrap overlays that copy read-only:
synchronous trajectories freeze peer revisions at the last Epoch watermark, while asynchronous
trajectories see the catalog head immediately. Only `owner_lineage` may revise an identity; peers
derive a new local record. At synchronization/publication points Light also publishes:

- the corresponding visible Direction and Experiment records with their actual review quality;
- relevant immutable Gateway Records;
- peer Agent Git refs;
- accepted-submission lineage/global winner facts;
- aggregated Direction queue.

It does not merge conversations or scratch files and never resets a trajectory to a peer winner. Inspect peer refs with normal Git and explicitly cherry-pick or derive work when useful.

## Start verification

Process creation is not success. Verify:

- Supervisor remains alive;
- expected campaign and trajectory scopes exist;
- V0 measurement or Agent Session has a durable update;
- Runtime/Agate requests have identifiable records;
- no immediate problem-contract, credential, source-layout or policy error occurred;
- `PROGRESS.md` and `frontier.json` reflect accepted submissions after `submit-kernel` succeeds.

Prefer a durable workload/service launcher for long campaigns.

## Resume

Resume with the same checkout behavior, workspace, profile/composition, problem snapshot, framework, source layout, mode, visibility and search identity. Light restores the persistent Agent Session/Git and reconciles saved Runtime/Agate evidence.

Do not clear a workspace merely because a process exited. Inspect the latest Session, Runtime audit, Gateway Records, Agate state and search checkpoint first. Do not edit private files or Git metadata to force a resume.

## Stop and cleanup

Stop through the owning Supervisor or workload mechanism, then verify Agent, Runtime and child processes exited. Avoid broad process-name kills. Deleting a workspace or Sandbox requires explicit authorization; export source/evidence first when it may be needed.
