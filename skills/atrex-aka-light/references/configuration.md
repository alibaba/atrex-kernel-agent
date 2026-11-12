# AKA Light configuration

Read this file when constructing or reviewing an AKA Light launch. It describes the public controls on the current upstream `main`; always confirm the selected checkout with `aka optimize --repo-root REPO -- --help` before using a newer revision.

## Two argument layers

`aka optimize` has an explicit `--` boundary:

```bash
aka optimize \
  --repo-root /path/to/atrex-kernel-agent \
  --profile light-sandbox \
  --patch /path/to/light-options.json \
  -- \
  --op-dir /path/to/operator \
  --platform L20D \
  --framework CuteDSL \
  --max-iters 20
```

Options before `--` select installed components. Options after `--` configure the optimization Campaign.

### Launch options before `--`

| Option | Default | Purpose |
|---|---:|---|
| `--repo-root PATH` | required | Checkout whose resources, profiles and packaged skills match the installed AKA wheels. It is not the Campaign workspace. |
| `--profile PROFILE` | `light` | Select `light`, `light-sandbox`, a Legacy-compatible profile, or an explicit Profile JSON. This chooses the Workflow and Agent boundary, not Leaderboard versus Production policy. |
| `--patch FILE` | none | Apply a composition patch. Repeat to apply multiple patches in order. Use this for module options such as Episode time, Wiki, generated memory or pooled search. |
| `--execution-profile PROFILE` | selected default | Advanced: select the separately composed Execution provider for prepared controller calls. This does not select the Light Workflow or change acceptance policy. |
| `--execution-patch FILE` | none | Advanced: patch that independent Execution selection. Repeatable. |

Do not add `--profile` or `--patch` when resuming through recorded continuation metadata; the frozen launch selection is restored and overrides are rejected.

## Campaign inputs, identity and policy

These options appear after `--`.

| Option | Default | Purpose and effect |
|---|---:|---|
| `--op-dir PATH` | required | A [supported Native Atrex-Bench or SOL operator directory](problem-inputs.md). AKA snapshots the problem contract and derives the initial candidate from it; candidate layouts are not alternate problem formats. |
| `--platform TOKEN` | required | Logical target product shown to the Agent and Wiki and, unless overridden, sent to Agate as the worker token. Examples include `L20D`, `H20`, `B200` and PPU product tokens. |
| `--framework DSL` | auto-dispatch | Run one child Campaign in the named DSL. If omitted, AKA starts one independent child per framework supported by the detected vendor. In Production it is an acceptance constraint; in Leaderboard it is a preferred direction. |
| `--[no-]specify-dsl` | enabled | Name and constrain the selected DSL in prompts and final Production acceptance. `--no-specify-dsl` lets one Agent choose any supported self-authored GPU DSL. Development probes may use other tools either way; the constraint applies to the submitted candidate. |
| `--optimization-mode MODE` | `leaderboard` | `leaderboard` permits evidence-backed mixed implementations; `production` requires a deployable implementation in the selected framework and adds fail-closed source review. |
| `--[no-]expose-evaluator-shapes` | exposed | Expose `shapes.json` and detailed evaluator metadata, or request the generalized hidden-shape Production flow. Visibility is part of workspace identity. |
| `--workspace PATH` | current directory | Base directory under which AKA creates the child `kernel_opt_*` workspace. Use an absolute path for durable launches. |
| `--notes TEXT` | `none` | Extra operator constraints or known bottlenecks injected into Agent instructions. Do not put credentials here. |
| `--arch ARCH` | auto-probed | Override runtime architecture such as `sm_89`, `sm_100` or `gfx942`. Use only when probing fails or an intentional override is required; it is not the same as `--platform`. |
| `--agent-reference-dir DIR...` | none | Repeatable read-only knowledge mounts at `references/<basename>/`. Requires a Bubblewrap Agent profile; basenames must be unique. |

Mode, shape visibility, framework, candidate layout, composition and problem snapshot identity must match on resume. Start a fresh workspace when intentionally changing them.

## Agent, Framework Baseline and review

| Option | Default | Purpose and effect |
|---|---:|---|
| `--agent-cli CLI` | `claude` | Coding provider used by Framework Baseline and optimization Episodes: `claude`, `qodercli`, `codex` or `pi`. Authentication comes from the provider's existing configuration. |
| `--framework-baseline MODE` | `auto` | Controls Episode 0: `auto` normally runs it in Production, `always` requests it in Leaderboard too, and `never` starts optimization Episodes from measured V0. It does not skip V0. Episode 0 uses the ordinary Runtime, Git, Research and `episode-report` contract, but does not require incumbent ABBA. |
| `--framework-baseline-timeout SECONDS` | `28800` | Provider wall-clock ceiling for the Episode 0 Agent session. Runtime waits pause Agent active time; this is independent of Agate request limits. |
| `--setup-timeout SECONDS` | `7200` | In current Light, the per-attempt timeout for automatically authoring `agent_problem.json` when hidden-shape Production lacks one; at most two attempts are made. It is not the V1 timeout. |
| `--handoff-resumes N` | `2` | Same-session repair turns allowed after an optimization Episode produces an incomplete terminal handoff. `0` disables these repairs. |
| `--[no-]v1-ask-codex` | off | Ask Codex for bounded, read-only correctness-first guidance during V1. It does not edit or approve the candidate. |
| `--[no-]v1-ask-qoder` | off | Ask Qoder for the same V1 guidance. Both reviewers may be enabled. |

Optimization Episode active time is configured through the Workflow composition patch, not a campaign CLI flag; see [Composition options](#composition-options).

## Search budgets and stop conditions

| Option | Default | Purpose and effect |
|---|---:|---|
| `--max-iters N` | `20` | Inclusive canonical-version ceiling. `20` permits versions through `v20`; it is not a count of successful promotions. See [Operations: Budgets](operations.md#budgets). |
| `--token-budget N` | `0` | Admission cap for reported optimization-Episode tokens; `0` disables it. In-flight work may overshoot, Bootstrap/auxiliary usage is excluded, and a provider that reports no tokens cannot be fully enforced. |
| `--target-util PCT` | `90` | Stop after a promoted candidate's recorded peak utilization reaches this percentage. Set `0` to disable. This is a post-promotion stop, not an acceptance substitute. |
| `--max-stall N` | `0` | Stop after `N` consecutive unpromoted Episodes; `0` disables it. Mandatory conversion and optional Goal Episode policy take precedence when applicable. |
| `--convert-after N` | `3` | For a Triton-family serial Campaign with source checks, require Triton-to-Gluon conversion after `N` consecutive stalls. Failed conversions remain latched until a Gluon candidate passes; `0` disables conversion. |

Pooled multi-trajectory search uses its own `epochs × trajectories × episodes_per_trajectory` schedule instead of the serial `--max-iters` budget. `--token-budget` remains a shared admission cap and already-running Episodes may overshoot it.

## Evaluation, ABBA and promotion

| Option | Default | Purpose and effect |
|---|---:|---|
| `--sandbox-hardware TOKEN` | `--platform` | Agate worker token. Set it when the logical product name shown to the Agent differs from the executor alias. |
| `--sandbox-profile {pre,prod}` | normal Agate config | Select an Agate endpoint profile. Mutually exclusive with `--sandbox-url` and `--sandbox-ssh`. |
| `--sandbox-url URL` | normal Agate config | Explicit Agate endpoint, including a local gateway. Mutually exclusive with endpoint profile and SSH transport. |
| `--sandbox-timeout SECONDS` | derived | Advanced enclosing Eval/ABBA execution allocation. By default AKA derives it from candidate, performance and ABBA run limits; queue grace is separate. An explicit value must contain the full ABBA schedule. |
| `--verify-repeats N` | `2` | Native ABBA measurements per side. Light requires an even value from 2 through 16; `2` represents one A/B/B/A block. |
| `--evaluation-step-timeout SECONDS` | `60` | Unified per-Shape candidate/correctness, performance, and individual A/B run budget. Explicit Runtime-tool values are honored; impossible enclosing schedules fail with the computed reason. |
| `--verify-shape-batch-size N` | `10` | Maximum Shape IDs in one outer Eval/ABBA batch. Light sends the complete reference bundle, selects the batch with native `shape_ids`, and forces `shape_shards=1` for that Ray Data request. |
| `--gpu-clock-mhz MHZ` | compatibility only | Accepted for old resume identities, but current Light sends only `lock_clocks=true`; Agate selects the realizable clock bin and no explicit MHz goes on the wire. |
| `--min-improvement-pct PCT` | `0` | Candidate GeoMean latency improvement must be strictly greater than this percentage to promote. Correctness and complete evidence remain mandatory. |

Current Light uses up to 10 Shapes per outer batch and up to 16 outer batches in parallel. Each request sets `shape_shards=1`, uses 20 warmups, 100 timing iterations and five deterministic correctness cases. Eval and terminal ABBA both request best-effort clock locking without an explicit MHz. The benchmark's declared performance objective is applied only after all outer batches are merged.

## SSH execution options

Use SSH only when GPU work must run in a mandatory remote Bubblewrap namespace instead of Agate.

| Option | Default | Purpose and effect |
|---|---:|---|
| `--sandbox-ssh [USER@]HOST` | off | Select OpenSSH transport. It is mutually exclusive with Agate URL/profile and requires an explicit `--framework`. |
| `--sandbox-ssh-init COMMAND` | environment or empty | Activation command run before each remote job and health probe. |
| `--sandbox-ssh-gpu INDEX` | environment | Physical NVIDIA GPU index, 0–31. Required for SSH; MIG/UUID selectors are not supported. |
| `--sandbox-ssh-runtime-bind REMOTE[=SANDBOX]` | environment or none | Repeatable read-only runtime directory mounted into the remote Bubblewrap namespace. |
| `--sandbox-health-command COMMAND` | built-in SSH probe | GPU readiness command used for failure classification and recovery. |
| `--environment-poll-interval SECONDS` | `60` | Delay between probes while a blocked SSH environment is recovering. |

SSH Light supports native Atrex-Bench Evaluate, ABBA, Profile, Check, Disassemble and Env for
`source_tree` and CUDA-manifest candidates. SOL remains `single_file` and has no native ABBA
fallback, so verify the accepted problem and candidate layout before starting a long campaign.

## Composition options

Composition patches are JSON files passed with launch-level `--patch` before `--`. A patch supplies the whole configuration for the named module; omitted fields receive that provider's defaults, not values from the selected Profile. In a Workflow patch used by `aka optimize`, retain `"startup": true` or the Campaign has no startup service.

```json
{
  "api_version": 1,
  "patches": [
    {
      "id": "workflow",
      "config": {
        "startup": true,
        "numerical_validation": false,
        "goal_episodes": false,
        "generate_memory": false,
        "output_limit_recovery_retries": 2
      }
    },
    {
      "id": "tool_runtime",
      "config": {
        "measurement_repetitions": 1,
        "pause_agent_during_requests": true,
        "wiki": true,
        "queue_timeout": 60
      }
    }
  ]
}
```

| Module | Configurable behavior |
|---|---|
| `agent` | `sandbox` (`none`/`bwrap`), session `capture`, provider `timeout_seconds` (default 28800), Bubblewrap executable, `proc_mode` (`private`/`projected`) and additional absolute read-only paths. The named `light-sandbox` Profile already selects the intended sandbox/capture combination. |
| `tool_runtime` | Measurement aggregation repetitions (`1` or `3`), whether to pause the Agent during authenticated Runtime requests (default true), GPU/Wiki request queue and request timeouts, Wiki enablement, and Wiki model/effort/timeouts/concurrency. These do not replace Agate's own execution limits. |
| `workflow` | Optional Episode active-time limit (default: no limit), optional supplemental numerical validation (default off), Goal Episodes, generated memory summaries, output-limit recovery count, pooled search topology, and advanced source-tree import. A numeric `episode_timeout_seconds` enables the limit. |
| `candidate` / `workspace` | Candidate filesystem (`source_tree`, `single_file`, or `cuda_manifest`) and whether custom reusable state is inherited across Episodes. Layout is fixed for the Campaign. |
| `light_journal` | `tool_modules`: choose any subset of `directions` and `experiments`; an empty list disables both. Evaluator facts, candidate reports and promotion evidence remain required regardless. |
| `wiki` | Knowledge root or a replacement provider. The Tool Runtime separately decides whether Wiki is enabled and how queries execute. |

For pooled search, `workflow.config.search` accepts:

| Field | Default | Meaning |
|---|---:|---|
| `epochs` | `10` | Number of search Epochs. |
| `trajectories` | `3` | Number of private trajectory workspaces. |
| `episodes_per_trajectory` | `1` | Sequential Episodes each trajectory runs during one Epoch. |
| `max_parallel_episodes` | `3` | Maximum simultaneously running trajectory Episodes; it cannot exceed `trajectories`. |
| `synchronization` | `synchronous` | `synchronous` publishes at an Epoch barrier; `asynchronous` publishes after terminal Episodes. |
| `role_specialization` | `false` | Add trajectory role prompts. Disabled by default so trajectories use the same optimization contract. |

Completed publications share verified Agent Git history/refs, terminal Gateway Records, reviewed Directions and Experiments, queue state, and measured Kernel facts. Live drafts, scratch and Conversation history are not merged. Peer research remains read-only until a trajectory explicitly absorbs it into local work.

## Related environment controls

| Variable | Default | Purpose |
|---|---:|---|
| `ATREX_SANDBOX_QUEUE_WAIT_GRACE` | `120` in Light | Additional remote queue-wait allowance. It is separate from execution time and may be explicitly overridden. |
| `ATREX_CODEX_SESSION_SETTINGS` | unset | JSON settings forwarded to Codex sessions; falls back to `ATREX_SESSION_SETTINGS`. |
| `ATREX_PI_SESSION_SETTINGS` | unset | JSON settings forwarded to Pi sessions; falls back to `ATREX_SESSION_SETTINGS`. |
| `ATREX_SANDBOX_SSH*` | unset | Environment equivalents for SSH target, init command, GPU and runtime binds. Explicit CLI values take precedence. |

Keep credentials in provider/Agate configuration or secret injection, not command lines, notes or composition JSON.

## Shared-parser options that do not configure Light

The optimization CLI is shared with Legacy. On current `main`, the following accepted flags are not copied into the Light `ManagedCampaign` and should not be used to claim a Light behavior change:

| Retained flag | Legacy purpose / Light behavior |
|---|---|
| `--fast-episodes`, `--fast-trials` | Configure Legacy fast Episode structure. Light uses unified Episodes; these flags do not configure it. |
| `--fast-episode-ask-*`, `--full-episode-ask-*` | Configure Legacy Episode reviewers. Light currently consumes only the V1 reviewer flags. |
| `--long-reviewer-session` | Reuses a Legacy reviewer session; it is not a Light Campaign control. |
| `--review-trace-mode` | Controls Legacy reviewer trace retention; it is not wired into Light. |
| `--gateway-candidate-recovery` | Legacy Framework Baseline candidate-repair policy; Light uses its own typed feedback and recovery path. |
| `--resume-framework-candidate` | Legacy retained-candidate adoption switch; Light restores V1 from its own durable state. |
| `--conversion-budget-action` | Legacy behavior at a conversion budget boundary. Light fails explicitly when mandatory conversion remains incomplete at its budget boundary. |

If a later `main` wires one of these settings into `ManagedCampaign`, update this reference from the implementation rather than inferring support from parser acceptance alone.

## Agate-native Runtime options

Inside a Light Episode, Runtime tools discover safe, non-authority Agate options from the installed `atrex-gateway-client` and expose them through the tool schema. Updating the client and restarting the Supervisor refreshes that catalog. Credentials, endpoints, uploads, source selection, timeouts, evaluator version and other authority-bearing options remain Supervisor-owned and are deliberately blocked from this dynamic pass-through.

Use the installed Runtime tool help/schema for the exact operation-specific fields. Do not assume an option supported by Agate `eval` is also valid for Profile, Check, Disassemble or Dev.
